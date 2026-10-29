#!/usr/bin/env python3
"""
Playing-time availability and the engine value's regular-chance factor.

The engine value (eur_intervals.engine_value_eur) rates a player on his
projection, which an unproven player inherits mostly from a prior: a 95'
cameo was valued like a starter (Rafinha Monteiro, EUR 218k, 2026-10). This
module scales the value by how likely a player with his playing time is to
become a regular, measured as 900'+ the next season:

  share     = his season minutes (engine mins_lineup, exact API minutes)
              / the minutes available to him: his club's league matches while
              he was there x the season's median match length
  sub_frac  = the share of his appearances that came off the bench
  p         = 1 / (1 + exp(-(a + b log(share + EPS) + c sub_frac)))
  factor    = min(p / p(ANCHOR_SHARE, 0), 1)    (1.0 for a starter from ~1,000';
              a bench-heavy rotation player keeps a small discount above
              that, e.g. 40% share with 80% of appearances off the bench ~0.91)

Reserve sides ('<club> II', league_config.is_reserve_team) are exempt:
few Liga 3 minutes there often mean first-team football we can't see.

"While he was there" = club windows from the Wyscout transfer records
(models/value/transfers.parquet, fetch_transfers.py): an arrival or a
departure dated inside the season (more than EDGE_DAYS from either end) cuts
the window; the window always covers every match he was in that club's
matchday squad; a window under MIN_WINDOW matches falls back to the full
season; a season's availability never exceeds one season (two clubs, or both
leagues in one year). No record at all: a January arrival is inferred from
the lineups (first squad after 1 Jan with 5+ team matches before); a departure
is never inferred (lineups can't tell "left" from "dropped").

Validated 2026-10-28 (Liga 3 2022/23 + 2024/25, non-reserve, cross-season):
log-loss 0.6426 -> 0.6393, AUC 0.633 -> 0.643 against the full-season
schedule; excusing red-card suspensions or injury-like gaps did not help and
is not done. Plan + evidence: ACP_Official/value_minutes_plan_2026-10-28.md.

COEF is pinned; refit at the yearly rollover with `--fit` (prints the
coefficients and the cross-season check) and regenerate the fee calibration
(eur_intervals.py) in the same commit.

  python models/value/availability.py         # writes availability.parquet (engine rebuild, Phase 2)
  python models/value/availability.py --fit   # refit COEF (rollover)
"""
import argparse
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DASH = HERE.parent.parent
GPA_DATA = DASH.parent.parent / 'GPA Model Project v2' / 'parquet_data'
sys.path.insert(0, str(DASH))
from league_config import CAMP_SEASON_IDS, SEASON_YEAR, is_reserve_team  # noqa: E402

OUT = HERE / 'availability.parquet'
ENGINE_PATH = DASH / 'models' / 'ratings' / 'player_engine.parquet'

# logit p(regular next season) = a + b log(share + EPS) + c sub_frac
COEF = (0.2759, 0.6181, -0.2755)  # --fit 2026-10-28: 927 rows, cross-season log-loss 0.6401, AUC 0.6415
SHARE_EPS = 0.002       # ~5'; immaterial above the engine's 90' floor (3.3% share)
ANCHOR_SHARE = 0.36     # ~1,000' of a typical 2,744' season: factor 1 at/above
REGULAR_NEXT = 900.0    # the outcome the curve is fitted to
EDGE_DAYS = 14          # a move this close to the season's ends is a summer move
MIN_WINDOW = 6          # fewer matches than this in a window: full season instead
FIT_YEARS = (2022, 2024)  # Liga 3 seasons whose next season has both leagues in the data


def regular_factor(share, sub_frac, coef=COEF):
    """The value multiplier in (0, 1]; None when the share is unknown."""
    if share is None or sub_frac is None or pd.isna(share) or pd.isna(sub_frac):
        return None
    a, b, c = coef
    p = lambda s, f: 1.0 / (1.0 + math.exp(-(a + b * math.log(s + SHARE_EPS) + c * f)))
    return min(p(min(float(share), 1.0), float(sub_frac)) / p(ANCHOR_SHARE, 0.0), 1.0)


# --- inputs ----------------------------------------------------------------------
def load_inputs(dash=DASH, gpa=GPA_DATA, engine_path=ENGINE_PATH):
    ms = pd.read_parquet(dash / 'matches_summary.parquet', columns=['matchId', 'seasonId', 'dateutc'])
    ms['d'] = pd.to_datetime(ms['dateutc'], errors='coerce')
    if getattr(ms['d'].dt, 'tz', None) is not None:
        ms['d'] = ms['d'].dt.tz_localize(None)
    ev = pd.concat([pd.read_parquet(gpa / f'{f}.parquet', columns=['matchId', 'seasonId', 'team.id', 'player.id'])
                    .drop_duplicates() for f in ('liga3_portugal_events', 'campeonato_portugal_events')])
    with open(dash / 'match_lineups.pkl', 'rb') as f:
        lineups = pickle.load(f)
    iv = pd.read_parquet(dash / 'models' / 'ratings' / 'player_match_intervals.parquet', columns=['matchId', 'u_out'])
    tr = pd.read_parquet(HERE / 'transfers.parquet') if (HERE / 'transfers.parquet').exists() else pd.DataFrame(
        columns=['playerId', 'startDate', 'fromTeamId', 'toTeamId'])
    eng = pd.read_parquet(engine_path, columns=['playerId', 'seasonId', 'team', 'mins_lineup'])
    return {'ms': ms, 'ev': ev, 'lineups': lineups, 'iv': iv, 'tr': tr, 'eng': eng}


def _squad(lineups):
    """{(playerId, matchId): 'S' start | 'B' came on | 'U' unused bench}."""
    pres = {}
    for m, teams in lineups.items():
        m = int(m)
        for t in teams.values():
            subs_in = {s.get('playerIn') for s in t.get('substitutions') or []}
            for p in t.get('lineup') or []:
                pres[(p['playerId'], m)] = 'S'
            for p in t.get('bench') or []:
                pres[(p['playerId'], m)] = 'B' if p['playerId'] in subs_in else 'U'
    return pres


# --- availability ----------------------------------------------------------------
def availability_table(inp):
    """One row per (playerId, yr): avail_mins, window_cut, n_clubs, starts,
    sub_apps, sub_frac. Club identity = events team.id (the lineup file's team
    NAMES are wrong in some matches)."""
    ms, lineups = inp['ms'], inp['lineups']
    ev = inp['ev'].dropna(subset=['team.id']).astype({'matchId': int, 'seasonId': int, 'team.id': int})
    date = dict(zip(ms['matchId'].astype(int), ms['d']))
    season_of = {**dict(zip(ms['matchId'].astype(int), ms['seasonId'].astype(int))),
                 **dict(zip(ev['matchId'], ev['seasonId']))}
    # eventless matches have lineups but no team identity: each season's
    # schedule is scaled by its played / with-events match ratio
    played = pd.Series([season_of.get(int(m)) for m in lineups]).dropna().astype(int).value_counts()
    coverage = (played / ev.groupby('seasonId')['matchId'].nunique()).dropna().to_dict()
    iv = inp['iv'].assign(seasonId=inp['iv']['matchId'].astype(int).map(season_of))
    mlen = iv.groupby('matchId').agg(L=('u_out', 'max'), s=('seasonId', 'first')).groupby('s')['L'].median().to_dict()

    tm = ev[['seasonId', 'team.id', 'matchId']].drop_duplicates()
    tm['d'] = tm['matchId'].map(date)
    tm = tm.dropna(subset=['d']).sort_values('d')
    sched = {k: list(zip(g['d'], g['matchId'])) for k, g in tm.groupby(['seasonId', 'team.id'])}
    bounds = tm.groupby('seasonId')['d'].agg(['min', 'max']).to_dict('index')
    clubs = (ev.dropna(subset=['player.id']).astype({'player.id': int})
             .groupby(['player.id', 'seasonId'])['team.id'].unique())
    pres = _squad(lineups)

    tr = inp['tr'].copy()
    tr['sd'] = pd.to_datetime(tr['startDate'], errors='coerce')
    tr = tr.dropna(subset=['sd', 'playerId'])
    tr_by = {int(p): g for p, g in tr.groupby('playerId')}

    def window(pid, sid, tid, seq):
        b = bounds[sid]
        lo, hi = b['min'] + pd.Timedelta(days=EDGE_DAYS), b['max'] - pd.Timedelta(days=EDGE_DAYS)
        if pid in tr_by:
            g = tr_by[pid]
            arr = g[(g['toTeamId'] == tid) & (g['sd'] > lo) & (g['sd'] < hi)]
            dep = g[(g['fromTeamId'] == tid) & (g['sd'] > lo) & (g['sd'] < hi)]
            start = arr['sd'].min() if len(arr) else None
            end = dep['sd'].max() if len(dep) else None
            if start is not None and end is not None and end <= start:   # left, came back: the later stint
                end = None
            return start, end
        seen = [i for i, (_, m) in enumerate(seq) if (pid, m) in pres]
        jan = pd.Timestamp(f'{SEASON_YEAR[sid] + 1}-01-01')
        if seen and seen[0] >= 5 and seq[seen[0]][0] >= jan:
            return jan, None
        return None, None

    rows = []
    for (pid, sid), tids in clubs.items():
        if sid not in bounds or sid not in SEASON_YEAR:
            continue
        cov, L = coverage.get(sid, 1.0), mlen.get(sid, 97.0)
        per = []
        for tid in tids:
            seq = sched.get((sid, tid), [])
            if not seq:
                continue
            start, end = window(pid, sid, tid, seq)
            seen_d = [d for d, m in seq if (pid, m) in pres]
            if seen_d:       # never cut away a match he was in this club's squad for
                if start is not None and seen_d[0] < start:
                    start = seen_d[0]
                if end is not None and seen_d[-1] >= end:
                    end = None if seen_d[-1] >= seq[-1][0] else seen_d[-1] + pd.Timedelta(days=1)
            win = [x for x in seq if (start is None or x[0] >= start) and (end is None or x[0] < end)]
            cut = start is not None or end is not None
            if cut and len(win) < MIN_WINDOW:
                win, cut = seq, False
            per.append({'season_n': len(seq), 'n': len(win), 'cut': cut})
        if not per:
            continue
        v0 = np.mean([c['season_n'] for c in per]) * cov * L
        cap = max(c['season_n'] for c in per) * cov * L
        if len(per) > 1 and not any(c['cut'] for c in per):
            v1 = v0          # two clubs, no dated move at either: one season, the mean schedule
        else:
            v1 = min(sum(c['n'] for c in per) * cov * L, cap)
        rows.append({'playerId': int(pid), 'yr': SEASON_YEAR[sid], 'v0': v0, 'v1': v1, 'cap': cap,
                     'cut': any(c['cut'] for c in per), 'n_clubs': len(per)})
    a = pd.DataFrame(rows)
    py = a.groupby(['playerId', 'yr']).agg(v0=('v0', 'mean'), v1=('v1', 'sum'), cap=('cap', 'max'),
                                           window_cut=('cut', 'any'), n_clubs=('n_clubs', 'sum')).reset_index()
    # both leagues in one year: still one season at most; no dated window: the mean
    py['avail_mins'] = np.where(py['window_cut'], np.minimum(py['v1'], py['cap']), py['v0'])

    # starts / sub appearances per player-year, over every match he was listed in
    sy = defaultdict(lambda: [0, 0])
    for (p, m), f in pres.items():
        s = season_of.get(m)
        if s in SEASON_YEAR and f in ('S', 'B'):
            sy[(p, SEASON_YEAR[s])][0 if f == 'S' else 1] += 1
    py['starts'] = [sy[(p, y)][0] for p, y in zip(py['playerId'], py['yr'])]
    py['sub_apps'] = [sy[(p, y)][1] for p, y in zip(py['playerId'], py['yr'])]
    apps = py['starts'] + py['sub_apps']
    py['sub_frac'] = np.where(apps > 0, py['sub_apps'] / apps.where(apps > 0, 1), 0.0)
    return py[['playerId', 'yr', 'avail_mins', 'window_cut', 'n_clubs', 'starts', 'sub_apps', 'sub_frac']]


def build(inp=None):
    """One row per engine (playerId, seasonId): the factor and its inputs."""
    inp = inp or load_inputs()
    eng = inp['eng'].copy()
    eng['playerId'] = pd.to_numeric(eng['playerId'], errors='coerce').astype('Int64')
    eng['seasonId'] = pd.to_numeric(eng['seasonId'], errors='coerce').astype('Int64')
    eng = eng.dropna(subset=['playerId', 'seasonId'])
    eng['yr'] = eng['seasonId'].map(lambda s: SEASON_YEAR.get(int(s)))
    eng = eng.dropna(subset=['yr'])
    eng['yr'] = eng['yr'].astype(int)
    mins_year = eng.groupby(['playerId', 'yr'])['mins_lineup'].sum().rename('mins_year')
    av = availability_table(inp).astype({'playerId': 'Int64'})
    out = (eng[['playerId', 'seasonId', 'yr', 'team']].join(mins_year, on=['playerId', 'yr'])
           .merge(av, on=['playerId', 'yr'], how='left'))
    out['window_cut'] = out['window_cut'].fillna(False).astype(bool)
    out['avail_share'] = (out['mins_year'] / out['avail_mins']).clip(upper=1.0)
    out['reserve_team'] = out['team'].map(is_reserve_team)
    out['regular_factor'] = [1.0 if rt else (regular_factor(s, f) if pd.notna(s) else 1.0)
                             for rt, s, f in zip(out['reserve_team'], out['avail_share'], out['sub_frac'])]
    return out.drop(columns=['team'])


_STAMP = {}


def stamp(path=OUT):
    """Short content hash of the committed factors ('' without the file).
    Part of app.py's percentiles-cache fingerprint: the cached frame carries
    'Engine Value EUR', so a factor change must invalidate it like an engine
    rebuild does. Cached per file mtime."""
    import hashlib
    p = Path(path)
    if not p.exists():
        return ''
    key = (str(p), p.stat().st_mtime_ns)
    if key not in _STAMP:
        a = pd.read_parquet(p, columns=['playerId', 'seasonId', 'regular_factor'])
        a = a.astype(str).sort_values(['playerId', 'seasonId'])
        _STAMP.clear()
        _STAMP[key] = hashlib.sha256(a.to_csv(index=False).encode()).hexdigest()[:12]
    return _STAMP[key]


def attach(engine_df, path=OUT):
    """engine_df + `regular_factor` (1.0 where there is none), joined on the
    RAW playerId and seasonId — call before any alias remap."""
    df = engine_df.copy()
    if 'regular_factor' in df.columns:
        df = df.drop(columns=['regular_factor'])
    if not Path(path).exists():
        df['regular_factor'] = 1.0
        return df
    av = pd.read_parquet(path, columns=['playerId', 'seasonId', 'regular_factor'])
    av['playerId'] = av['playerId'].astype('Int64')
    av['seasonId'] = av['seasonId'].astype('Int64')
    av = av.drop_duplicates(['playerId', 'seasonId'])
    key_p = pd.to_numeric(df['playerId'], errors='coerce').astype('Int64')
    key_s = pd.to_numeric(df['seasonId'], errors='coerce').astype('Int64')
    f = pd.DataFrame({'playerId': key_p, 'seasonId': key_s}).merge(av, on=['playerId', 'seasonId'], how='left')
    df['regular_factor'] = f['regular_factor'].fillna(1.0).to_numpy()
    return df


# --- refit (yearly rollover) -------------------------------------------------------
def fit(inp=None):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, roc_auc_score
    inp = inp or load_inputs()
    eng = inp['eng'].assign(yr=inp['eng']['seasonId'].map(lambda s: SEASON_YEAR.get(int(s))))
    league = inp['eng']['seasonId'].map(lambda s: 'CAMP' if int(s) in CAMP_SEASON_IDS else 'L3')
    per = eng.groupby(['playerId', 'yr'])['mins_lineup'].sum()
    w = eng[(league == 'L3') & eng['yr'].isin(FIT_YEARS) & ~eng['team'].map(is_reserve_team)].drop_duplicates(['playerId', 'yr'])
    w = w[['playerId', 'yr']].merge(availability_table(inp), on=['playerId', 'yr']).dropna(subset=['avail_mins'])
    sh = np.minimum([per[(p, y)] for p, y in zip(w['playerId'], w['yr'])] / w['avail_mins'], 1.0)
    X = np.c_[np.log(sh + SHARE_EPS), w['sub_frac']]
    y = np.array([int(per.get((p, yy + 1), 0.0) >= REGULAR_NEXT) for p, yy in zip(w['playerId'], w['yr'])])
    ll, auc = [], []
    for tr_y, te_y in (FIT_YEARS, FIT_YEARS[::-1]):
        a, b = (w['yr'] == tr_y).to_numpy(), (w['yr'] == te_y).to_numpy()
        m = LogisticRegression(C=1e6, max_iter=5000).fit(X[a], y[a])
        p = m.predict_proba(X[b])[:, 1]
        ll.append(log_loss(y[b], p)); auc.append(roc_auc_score(y[b], p))
    m = LogisticRegression(C=1e6, max_iter=5000).fit(X, y)
    coef = (round(float(m.intercept_[0]), 4), round(float(m.coef_[0][0]), 4), round(float(m.coef_[0][1]), 4))
    print(f'rows {len(w)}; cross-season log-loss {np.mean(ll):.4f}, AUC {np.mean(auc):.4f}')
    print(f'COEF = {coef}   (pinned: {COEF})')
    return coef


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--fit', action='store_true', help='refit COEF and print it (yearly rollover)')
    if ap.parse_args().fit:
        fit()
    else:
        out = build()
        out.to_parquet(OUT, index=False)
        print(f"availability: {len(out)} engine rows; window cut {int(out['window_cut'].sum())}; reserve {int(out['reserve_team'].sum())}; "
              f"factor < 1: {int((out['regular_factor'] < 1).sum())} (median {out.loc[out['regular_factor'] < 1, 'regular_factor'].median():.2f})")
