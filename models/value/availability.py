#!/usr/bin/env python3
"""
Playing-time availability and the engine value's regular-chance factor.

The engine value (eur_intervals.engine_value_eur) rates a player on his
projection, which an unproven player inherits mostly from a prior: a 95'
cameo was valued like a starter (Rafinha Monteiro, EUR 218k, 2026-10). This
module scales the value by how likely a player with his playing time is to
become a regular, measured as 900'+ the next season:

  share     = his CAREER playing time (career_playing_time, 2026-10-29):
              each season's minutes (engine mins_lineup, exact API minutes)
              / the minutes available to him (the summed lengths of his
              club's league matches while he was there), seasons weighted
              by time before the as-of date (half-life H_PLAYING) — one
              continuous sample, not one season (Lucas 2026-10-29)

Available minutes are counted match by match (2026-10-29; they were the
event-data match count x a coverage ratio x the season's median length):
  - the club's schedule = every played match in matches_summary (status
    'Played', never a fixture; team names mapped to the events' team.id per
    season; the names agreed with the events in all 5,873 matches checked),
    less a per-match season's matches with no per-match API minutes — the
    numerator can't count those (match_minutes.excluded_match_ids, the
    app's per-90 rule);
  - a match's length = Wyscout's per-match API minutes (the most any player
    got: what a full-match player is credited) — every Liga 3 season;
    else the events' clock (1H end + 2H end - 45, + any extra time; within
    0.6' of the API on 1,595 Liga 3 matches) — the Campeonato, whose
    per-match API minutes are gone; else the season's median (`len_api`
    records the API-exact fraction).
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
leagues in one year). No transfer record: no window — nothing is inferred
from the lineups (a January-arrival rule — first squad after 1 Jan with 5+
team matches before — was tested and REMOVED 2026-10-29: those players became
regulars 19% of the time against 35% predicted, and dropping it changed
nothing measurable; lineups can't tell "left" from "dropped" either). A record
from a club to itself (a renewal) is not a move and is ignored.

Validated 2026-10-28 (Liga 3 2022/23 + 2024/25, non-reserve, cross-season):
log-loss 0.6426 -> 0.6393, AUC 0.633 -> 0.643 against the full-season
schedule; excusing red-card suspensions or injury-like gaps did not help and
is not done. Audit 2026-10-29 on years never used to choose the model (L3
2021/22 + 2023/24): windows beat the schedule, log-loss -0.0047 (95% CI
-0.0084 to -0.0010); AUC +0.008 (CI includes 0) — a real but small gain.
Plan + evidence: ACP_Official/value_minutes_plan_2026-10-28.md.

Each player-year also carries the matchday-squad view, which is display-only
(the profile card's headline, descriptive; never in the factor):
  - squad_mins: the lengths of the matches he was in the squad for (XI,
    bench, or API minutes);
  - n_squad of n_club: the club matches while he was there;
  - squad_clubs: 'name:n_squad:n_club' per club, '|'-joined.
Squad-only as the value's denominator was tested on 2026-10-29 and predicts
worse (hold-out log-loss +0.017). Excusing long absences didn't hold up on
the hold-out years.

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
COEF = (0.3365, 0.7516, -0.1886)  # --fit 2026-10-29 (career share, H_PLAYING 1y, no reserve-only years): 927 rows, cross-season log-loss 0.6360, AUC 0.6516
SHARE_EPS = 0.002       # ~5'; immaterial above the engine's 90' floor (3.3% share)
ANCHOR_SHARE = 0.36     # ~1,000' of a typical 2,744' season: factor 1 at/above
REGULAR_NEXT = 900.0    # the outcome the curve is fitted to
EDGE_DAYS = 14          # a move this close to the season's ends is a summer move
MIN_WINDOW = 6          # fewer matches than this in a window: full season instead
FIT_YEARS = (2022, 2024)  # Liga 3 seasons whose next season has both leagues in the data
# AS-OF coefficients (2026-10-29, no look-ahead): a row of season-year y uses
# the fit on the FIT_YEARS whose outcome (the next season) had been played by
# the end of y — {first year it applies: coef}. Before any (y <= 2022) there
# is no fitted evidence, so the factor is 1.0 (no playing-time adjustment).
# Pinned like COEF: `--fit` prints both at the rollover.
COEF_ASOF = {2023: (0.3016, 0.6196, -0.5704),   # --fit years (2022,): 497 rows
             2025: COEF}                         # FIT_YEARS (2022, 2024)


def coef_for(yr):
    """The coefficients a row of season-year yr may use (None: no fit yet)."""
    keys = [k for k in COEF_ASOF if k <= int(yr)]
    return COEF_ASOF[max(keys)] if keys else None


def regular_factor(share, sub_frac, coef=COEF):
    """The value multiplier in (0, 1]; None when the share is unknown."""
    if share is None or sub_frac is None or pd.isna(share) or pd.isna(sub_frac):
        return None
    a, b, c = coef
    p = lambda s, f: 1.0 / (1.0 + math.exp(-(a + b * math.log(s + SHARE_EPS) + c * f)))
    return min(p(min(float(share), 1.0), float(sub_frac)) / p(ANCHOR_SHARE, 0.0), 1.0)


# --- inputs ----------------------------------------------------------------------
EVENT_FILES = ('liga3_portugal_events', 'campeonato_portugal_events')


def load_inputs(dash=DASH, gpa=GPA_DATA, engine_path=ENGINE_PATH):
    from match_minutes import excluded_match_ids
    ms = _played(pd.read_parquet(dash / 'matches_summary.parquet',
                                 columns=['matchId', 'seasonId', 'status', 'dateutc', 'homeTeamName', 'awayTeamName']))
    ms['d'] = pd.to_datetime(ms['dateutc'], errors='coerce')
    if getattr(ms['d'].dt, 'tz', None) is not None:
        ms['d'] = ms['d'].dt.tz_localize(None)
    ev = pd.concat([pd.read_parquet(gpa / f'{f}.parquet', columns=['matchId', 'seasonId', 'team.id', 'team.name', 'player.id'])
                    .drop_duplicates() for f in EVENT_FILES])
    with open(dash / 'match_lineups.pkl', 'rb') as f:
        lineups = pickle.load(f)
    tr = pd.read_parquet(HERE / 'transfers.parquet') if (HERE / 'transfers.parquet').exists() else pd.DataFrame(
        columns=['playerId', 'startDate', 'fromTeamId', 'toTeamId'])
    eng = pd.read_parquet(engine_path, columns=['playerId', 'seasonId', 'team', 'mins_lineup'])
    mpm = dash / 'match_player_minutes.parquet'
    mins = (pd.read_parquet(mpm, columns=['matchId', 'playerId', 'minutes']) if mpm.exists()
            else pd.DataFrame(columns=['matchId', 'playerId', 'minutes']))
    # per-match seasons' matches with no per-match minutes (Wyscout has no
    # stats, or not fetched yet): the numerator can't count them, so the
    # schedule doesn't either — the app's per-90 rule (match_minutes.py)
    excluded = excluded_match_ids(ms[['matchId', 'seasonId']], str(mpm))
    return {'ms': ms, 'ev': ev, 'lineups': lineups, 'tr': tr, 'eng': eng,
            'mins': mins[mins['playerId'] != 0], 'clock': _clock(gpa), 'exact': _exact_minutes(dash),
            'excluded': excluded}


def _played(ms):
    """matches_summary rows that were played — never a scheduled fixture."""
    return ms[ms['status'].eq('Played')].copy() if 'status' in ms.columns else ms


def _clock(gpa=GPA_DATA):
    """Per (matchId, period) the last event's clock in minutes (columns 1H, 2H
    and, for extra time, 1E / 2E)."""
    parts = []
    for f in EVENT_FILES:
        e = pd.read_parquet(gpa / f'{f}.parquet', columns=['matchId', 'matchPeriod', 'minute', 'second'])
        e = e[e['matchPeriod'].isin(['1H', '2H', '1E', '2E'])]
        parts.append((e['minute'] + e['second'] / 60.0).groupby([e['matchId'], e['matchPeriod']]).max())
    return pd.concat(parts).unstack()


def _exact_minutes(dash=DASH):
    """(playerId, seasonId, minutes) for EVERY player — the exact API minutes
    the engine's mins_lineup uses (build_player_engine.py): per-match sums
    where a season has per-match data, else the stored season totals. The
    rows the engine doesn't cover (goalkeepers, sub-floor players) take their
    numerator from here."""
    from match_minutes import load_cache, per_match_seasons
    from precompute_minutes import _season_total_rows, resolve_aliases
    ms = pd.read_parquet(dash / 'matches_summary.parquet', columns=['matchId', 'seasonId'])
    pms = per_match_seasons(dash / 'match_player_minutes.parquet', ms)
    mm = load_cache(dash / 'match_player_minutes.parquet')
    parts = []
    if len(mm):
        lm = resolve_aliases(mm).groupby(['seasonId', 'playerId'], as_index=False)['minutes'].sum()
        parts.append(lm[lm['seasonId'].isin(pms)])
    pkl = dash / 'player_minutes_and_positions.pkl'
    if pkl.exists():
        with open(pkl, 'rb') as f:
            api = pickle.load(f)
        parts += [_season_total_rows(int(s), df)[['playerId', 'totalMinutes']]
                  .rename(columns={'totalMinutes': 'minutes'}).assign(seasonId=int(s))
                  for s, df in api.items() if int(s) not in pms]
    if not parts:
        return pd.DataFrame(columns=['playerId', 'seasonId', 'minutes'])
    return pd.concat(parts, ignore_index=True)[['playerId', 'seasonId', 'minutes']].astype(int)


def _norm_name(name):
    """Team name for matching across sources: no accents, case or extra spaces."""
    import unicodedata
    s = unicodedata.normalize('NFKD', str(name))
    return ' '.join(''.join(c for c in s if not unicodedata.combining(c)).casefold().split())


def match_lengths(inp):
    """{matchId: (minutes, 'api' | 'clock' | 'median')} for every played match
    in matches_summary — see the module docstring for the order."""
    api = inp['mins'].groupby('matchId')['minutes'].max()
    api = api[api > 0]
    c = inp['clock']
    clock = pd.Series(dtype=float)
    if len(c) and {'1H', '2H'} <= set(c.columns):
        clock = c['1H'].clip(lower=45) + c['2H'].clip(lower=90) - 45
        for p, start in (('1E', 90), ('2E', 105)):
            if p in c.columns:
                clock = clock + (c[p].clip(lower=start + 15) - start).fillna(0.0)
        clock = clock.dropna()
    ms = _played(inp['ms'])
    out, by_season = {}, defaultdict(list)
    for m, s in zip(ms['matchId'].astype(int), ms['seasonId'].astype(int)):
        if m in api.index:
            out[m] = (float(api[m]), 'api')
        elif m in clock.index:
            out[m] = (float(clock[m]), 'clock')
        else:
            continue
        by_season[s].append(out[m][0])
    for m, s in zip(ms['matchId'].astype(int), ms['seasonId'].astype(int)):
        if m not in out:
            out[m] = (float(np.median(by_season[s])) if by_season[s] else 97.0, 'median')
    return out


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
    """One row per (playerId, yr): avail_mins, len_api (its API-exact
    fraction), window_cut, n_clubs, starts, sub_apps, sub_frac. Club identity
    = events team.id (the lineup file's team NAMES are wrong in some matches);
    a club's schedule = its played matches in matches_summary, less the
    per-match seasons' matches with no per-match minutes (inp['excluded'])."""
    ms, lineups = _played(inp['ms']), inp['lineups']
    ms = ms[~ms['matchId'].astype(int).isin(inp.get('excluded') or set())]
    ev = inp['ev'].dropna(subset=['team.id']).astype({'matchId': int, 'seasonId': int, 'team.id': int})
    season_of = {**dict(zip(ms['matchId'].astype(int), ms['seasonId'].astype(int))),
                 **dict(zip(ev['matchId'], ev['seasonId']))}
    length = match_lengths(inp)

    # matches_summary has team names, not ids: map them through the events
    # (one id per name per season), accents / case / spacing ignored
    # ('Pevidem' vs 'Pevidém', 'Mortágua' vs 'Mortágua '); a club with no
    # events at all stays unmapped
    named = ev.dropna(subset=['team.name'])
    name_id = dict(zip(zip(named['seasonId'], named['team.name'].map(_norm_name)), named['team.id']))
    tm = pd.concat([ms[['matchId', 'seasonId', 'd', side]].rename(columns={side: 'name'})
                    for side in ('homeTeamName', 'awayTeamName')])
    tm['team.id'] = [name_id.get((int(s), _norm_name(n))) for s, n in zip(tm['seasonId'], tm['name'])]
    tm = tm.dropna(subset=['d', 'team.id']).astype({'matchId': int, 'seasonId': int, 'team.id': int}).sort_values('d')
    sched = {k: list(zip(g['d'], g['matchId'])) for k, g in tm.groupby(['seasonId', 'team.id'])}
    bounds = tm.groupby('seasonId')['d'].agg(['min', 'max']).to_dict('index')
    clubs = (ev.dropna(subset=['player.id']).astype({'player.id': int})
             .groupby(['player.id', 'seasonId'])['team.id'].unique())
    pres = _squad(lineups)
    # The squad view (display only).
    # - In the matchday squad = named in the XI or on the bench, credited API
    #   minutes, or in the events (covers a match with no or empty lineup data).
    # - A match counts toward "of N" when its lineups are known or he was in it.
    # - Each squad match belongs to ONE club, the side he was on (events, else
    #   the lineup side), so a match between his two clubs isn't counted twice.
    mm = inp.get('mins', pd.DataFrame(columns=['matchId', 'playerId', 'minutes']))
    on = mm[pd.to_numeric(mm['minutes'], errors='coerce') > 0]
    evp = ev.dropna(subset=['player.id']).astype({'player.id': int})
    side = dict(zip(zip(evp['player.id'], evp['matchId']), evp['team.id']))
    for m, teams in lineups.items():
        for name, t in teams.items():
            tid = name_id.get((season_of.get(int(m)), _norm_name(name)))
            for p in ((t or {}).get('lineup') or []) + ((t or {}).get('bench') or []):
                side.setdefault((p['playerId'], int(m)), tid)
    in_squad = set(pres) | set(side) | set(zip(on['playerId'].astype(int), on['matchId'].astype(int)))
    known = {int(m) for m, teams in lineups.items()
             if any((t or {}).get('lineup') for t in (teams or {}).values())}
    team_name = dict(zip(zip(named['seasonId'], named['team.id']), named['team.name'].str.strip()))

    tr = inp['tr'].copy()
    tr['sd'] = pd.to_datetime(tr['startDate'], errors='coerce')
    tr = tr.dropna(subset=['sd', 'playerId'])
    # a club to itself is a renewal / re-registration, not a move: read as an
    # arrival it would cut the window at the renewal date
    same_club = tr['fromTeamId'].notna() & tr['toTeamId'].notna() & (tr['fromTeamId'] == tr['toTeamId'])
    tr = tr[~same_club.fillna(False).astype(bool)]
    tr_by = {int(p): g for p, g in tr.groupby('playerId')}

    def window(pid, sid, tid):
        """(start, end) of his time at the club this season from his dated
        moves; (None, None) = the whole season (no record, or no move inside it)."""
        if pid not in tr_by:
            return None, None
        b = bounds[sid]
        lo, hi = b['min'] + pd.Timedelta(days=EDGE_DAYS), b['max'] - pd.Timedelta(days=EDGE_DAYS)
        g = tr_by[pid]
        arr = g[(g['toTeamId'] == tid) & (g['sd'] > lo) & (g['sd'] < hi)]
        dep = g[(g['fromTeamId'] == tid) & (g['sd'] > lo) & (g['sd'] < hi)]
        start = arr['sd'].min() if len(arr) else None
        end = dep['sd'].max() if len(dep) else None
        if start is not None and end is not None and end <= start:   # left, came back: the later stint
            end = None
        return start, end

    rows = []
    for (pid, sid), tids in clubs.items():
        if sid not in bounds or sid not in SEASON_YEAR:
            continue
        per, claimed = [], set()
        for tid in tids:
            seq = sched.get((sid, tid), [])
            if not seq:
                continue
            start, end = window(pid, sid, tid)
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
            mins = lambda xs: (sum(length[m][0] for _, m in xs),
                               sum(length[m][0] for _, m in xs if length[m][1] == 'api'))
            (season_L, season_api), (win_L, win_api) = mins(seq), mins(win)
            squad = [m for _, m in seq if (pid, m) in in_squad and m not in claimed
                     and side.get((pid, m)) in (None, tid)]
            claimed.update(squad)
            sq = set(squad)
            sq_d = [d for d, m in seq if m in sq]
            per.append({'season_L': season_L, 'season_api': season_api,
                        'win_L': win_L, 'win_api': win_api, 'cut': cut,
                        'squad_L': sum(length[m][0] for m in squad), 'n_squad': len(squad),
                        # "of N": his window; for two undated clubs, first to last squad there
                        'n_win': len({m for _, m in win if m in known} | sq),
                        'n_span': len({m for d, m in seq if sq_d and sq_d[0] <= d <= sq_d[-1] and m in known} | sq),
                        'name': team_name.get((sid, tid), tid)})
        if not per:
            continue
        undated = len(per) > 1 and not any(c['cut'] for c in per)
        for c in per:
            c['n_club'] = c['n_span'] if undated else c['n_win']
            c['club'] = f"{c['name']}:{c['n_squad']}:{c['n_club']}"
        v0 = np.mean([c['season_L'] for c in per])
        api0 = np.mean([c['season_api'] for c in per])
        cap = max(c['season_L'] for c in per)
        if len(per) > 1 and not any(c['cut'] for c in per):
            v1, api1 = v0, api0      # two clubs, no dated move at either: one season, the mean schedule
        else:
            w = sum(c['win_L'] for c in per)
            v1 = min(w, cap)
            api1 = sum(c['win_api'] for c in per) * (v1 / w if w > 0 else 0.0)
        rows.append({'playerId': int(pid), 'yr': SEASON_YEAR[sid], 'v0': v0, 'api0': api0,
                     'v1': v1, 'api1': api1, 'cap': cap,
                     'cut': any(c['cut'] for c in per), 'n_clubs': len(per),
                     'squad_mins': sum(c['squad_L'] for c in per), 'n_squad': sum(c['n_squad'] for c in per),
                     'n_club': sum(c['n_club'] for c in per), 'squad_clubs': '|'.join(c['club'] for c in per)})
    a = pd.DataFrame(rows)
    py = a.groupby(['playerId', 'yr']).agg(v0=('v0', 'mean'), api0=('api0', 'mean'), v1=('v1', 'sum'),
                                           api1=('api1', 'sum'), cap=('cap', 'max'),
                                           window_cut=('cut', 'any'), n_clubs=('n_clubs', 'sum'),
                                           squad_mins=('squad_mins', 'sum'), n_squad=('n_squad', 'sum'),
                                           n_club=('n_club', 'sum'),
                                           squad_clubs=('squad_clubs', '|'.join)).reset_index()
    # both leagues in one year: still one season at most; no dated window: the mean
    py['avail_mins'] = np.where(py['window_cut'], np.minimum(py['v1'], py['cap']), py['v0'])
    py['len_api'] = np.clip(np.where(py['window_cut'], py['api1'] / py['v1'].where(py['v1'] > 0),
                                     py['api0'] / py['v0'].where(py['v0'] > 0)), 0.0, 1.0)

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
    return py[['playerId', 'yr', 'avail_mins', 'len_api', 'window_cut', 'n_clubs', 'starts', 'sub_apps', 'sub_frac',
               'squad_mins', 'n_squad', 'n_club', 'squad_clubs']]


H_PLAYING = 1.0   # years: the career share's half-life (= the form decay, career.H_FORM)


def career_playing_time(rows, inp):
    """Career playing time as of each row's season (Lucas 2026-10-29: predict
    from the whole career as one continuous sample). Each player-year is a
    block — his minutes, available minutes, starts and sub appearances — at
    its minute-weighted match date; blocks up to the row's year are weighted
    0.5 ** (years before the as-of time / H_PLAYING), the as-of time being the
    season's last played match (for the season in progress: the data date).
      career_share    = sum(w x mins_year) / sum(w x avail_mins)   (<= 1)
      career_sub_frac = sum(w x sub_apps) / sum(w x appearances)
    Tested 2026-10-29 against the season share (900'+ next season): better
    on both splits (log-loss -0.0036 cross-season, -0.0032 hold-out)."""
    from models.ratings import career as _career
    cen, smid = _career.block_centroids(inp['mins'], inp['ms'])
    ends = _career.season_end_times(inp['ms'], SEASON_YEAR)
    yc = defaultdict(list)
    for (p, sid), c in cen.items():
        if sid in SEASON_YEAR:
            yc[(int(p), SEASON_YEAR[sid])].append(c)
    ymid = defaultdict(list)
    for sid, c in smid.items():
        if sid in SEASON_YEAR:
            ymid[SEASON_YEAR[sid]].append(c)
    # a year spent only at reserve sides says nothing about his chance of
    # playing (why the factor exempts them), so it isn't a block of a later
    # row's career share either
    rt = (rows['reserve_team'].fillna(False).astype(bool) if 'reserve_team' in rows.columns
          else pd.Series(False, index=rows.index))
    res = rt.groupby([rows['playerId'], rows['yr']]).transform('all')
    blk = (rows[~res].dropna(subset=['avail_mins']).drop_duplicates(['playerId', 'yr'])
               [['playerId', 'yr', 'mins_year', 'avail_mins', 'starts', 'sub_apps']].copy())
    blk['c'] = [np.mean(yc[(int(p), int(y))]) if yc.get((int(p), int(y))) else
                (np.mean(ymid[int(y)]) if ymid.get(int(y)) else np.nan)
                for p, y in zip(blk['playerId'], blk['yr'])]
    blk['t'] = blk['yr'].map(ends)
    res = {}
    for pid, g in blk.groupby('playerId'):
        yr, c = g['yr'].to_numpy(), g['c'].to_numpy(float)
        mins, avail = g['mins_year'].to_numpy(float), g['avail_mins'].to_numpy(float)
        subs = g['sub_apps'].fillna(0).to_numpy(float)
        apps = subs + g['starts'].fillna(0).to_numpy(float)
        for y_i, t_i in zip(yr, g['t'].to_numpy(float)):
            use = (yr <= y_i) & np.isfinite(c) & (avail > 0)
            if not use.any() or not np.isfinite(t_i):
                continue
            w = 0.5 ** (np.clip(t_i - c[use], 0.0, None) / H_PLAYING)
            share = min(float((w * mins[use]).sum() / (w * avail[use]).sum()), 1.0)
            a_ = float((w * apps[use]).sum())
            res[(int(pid), int(y_i))] = (share, float((w * subs[use]).sum()) / a_ if a_ > 0 else 0.0)
    key = list(zip(rows['playerId'].astype(int), rows['yr'].astype(int)))
    return (pd.Series([res.get(k, (np.nan, np.nan))[0] for k in key], index=rows.index),
            pd.Series([res.get(k, (np.nan, np.nan))[1] for k in key], index=rows.index))


def build(inp=None):
    """One row per engine (playerId, seasonId) — the factor and its inputs —
    plus, with in_engine False and no factor, every other (playerId,
    seasonId) with exact API minutes (goalkeepers, sub-floor players): the
    profile's minutes-share card reads those; attach() joins engine rows only,
    so they never touch a value."""
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
    out['in_engine'] = True

    ex = inp['exact'].copy()
    ex['yr'] = ex['seasonId'].map(SEASON_YEAR)
    ex = ex.dropna(subset=['yr']).astype({'playerId': 'Int64', 'seasonId': 'Int64', 'yr': int})
    ex = ex[ex['minutes'] > 0]
    ex_year = ex.groupby(['playerId', 'yr'])['minutes'].sum().rename('mins_year')
    seen = pd.MultiIndex.from_frame(out[['playerId', 'seasonId']])
    rest = ex[~pd.MultiIndex.from_frame(ex[['playerId', 'seasonId']]).isin(seen)]
    rest = (rest[['playerId', 'seasonId', 'yr']].join(ex_year, on=['playerId', 'yr'])
            .merge(av, on=['playerId', 'yr'], how='inner'))
    ev = inp['ev'].dropna(subset=['player.id', 'team.name'])
    reserve = (ev.assign(r=ev['team.name'].map(is_reserve_team))
                 .groupby(['player.id', 'seasonId'])['r'].any())
    rest['window_cut'] = rest['window_cut'].astype(bool)
    rest['avail_share'] = (rest['mins_year'] / rest['avail_mins']).clip(upper=1.0)
    rest['reserve_team'] = [bool(reserve.get((int(p), int(s)), False))
                            for p, s in zip(rest['playerId'], rest['seasonId'])]
    rest['in_engine'] = False
    out = pd.concat([out.drop(columns=['team']), rest[out.columns.drop('team')]], ignore_index=True)
    # the factor runs on the CAREER playing time (2026-10-29); the season share
    # and sub_frac stay as the season's description
    out['career_share'], out['career_sub_frac'] = career_playing_time(
        out.sort_values('in_engine', ascending=False, kind='stable'), inp)
    def _factor(ie, rt, s, f, c):
        if not ie:
            return np.nan
        if rt or c is None or pd.isna(s):
            return 1.0
        return regular_factor(s, f, c)
    _args = list(zip(out['in_engine'], out['reserve_team'], out['career_share'], out['career_sub_frac'], out['yr']))
    # regular_factor: the row's AS-OF factor (its season value, the fee
    # pairing) — the coefficients known by the end of its year. regular_factor_now:
    # TODAY's value of a projection row (a lapsed player's last row too) — today's
    # coefficients on his playing time up to that row.
    out['regular_factor'] = [_factor(ie, rt, s_, f_, coef_for(y)) for ie, rt, s_, f_, y in _args]
    out['regular_factor_now'] = [_factor(ie, rt, s_, f_, COEF) for ie, rt, s_, f_, y in _args]
    return out


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
    df = engine_df.drop(columns=['regular_factor', 'regular_factor_now'], errors='ignore')
    if not Path(path).exists():
        df['regular_factor'] = df['regular_factor_now'] = 1.0
        return df
    _cols = ['playerId', 'seasonId', 'regular_factor']
    import pyarrow.parquet as _pq
    if 'regular_factor_now' in _pq.read_schema(path).names:
        _cols.append('regular_factor_now')
    av = pd.read_parquet(path, columns=_cols)
    if 'regular_factor_now' not in av.columns:          # a file from before 2026-10-29
        av['regular_factor_now'] = av['regular_factor']
    av['playerId'] = av['playerId'].astype('Int64')
    av['seasonId'] = av['seasonId'].astype('Int64')
    av = av.drop_duplicates(['playerId', 'seasonId'])
    key_p = pd.to_numeric(df['playerId'], errors='coerce').astype('Int64')
    key_s = pd.to_numeric(df['seasonId'], errors='coerce').astype('Int64')
    f = pd.DataFrame({'playerId': key_p, 'seasonId': key_s}).merge(av, on=['playerId', 'seasonId'], how='left')
    df['regular_factor'] = f['regular_factor'].fillna(1.0).to_numpy()
    df['regular_factor_now'] = f['regular_factor_now'].fillna(1.0).to_numpy()
    return df


def scoped_share(rows):
    """Share of the available minutes over one player's availability rows
    (already cut to the years wanted): minutes ÷ available minutes, each
    summed once per season year (a year's L3 + Camp rows and a mid-season
    mover's club rows all carry the same player-year availability). A year's
    available minutes are never below its minutes (a clock-derived length can
    run a few seconds short; the 100% cap of avail_share). `exact` = every
    match length from the API. The squad view (the profile card's headline,
    descriptive): `squad_share` = minutes ÷ the minutes of the matches he was
    in the matchday squad for, `n_squad` of `n_club` club matches, and per
    club [(name, n_squad, n_club)]; None where there is no squad data. None
    without data."""
    if rows is None or rows.empty or 'avail_mins' not in rows.columns:
        return None
    # most minutes first: an alias-merged player could carry two raw ids' rows in one year
    r = (rows.dropna(subset=['yr', 'mins_year', 'avail_mins'])
             .sort_values('mins_year', ascending=False, kind='stable').drop_duplicates('yr'))
    mins = float(r['mins_year'].sum())
    avail = float(np.maximum(r['avail_mins'], r['mins_year']).sum())
    if r.empty or avail <= 0:
        return None
    exact = bool('len_api' in r.columns and (r['len_api'] >= 0.999).all())
    out = {'share': min(mins / avail, 1.0), 'mins': mins, 'avail': avail,
           'years': sorted(int(y) for y in r['yr']), 'exact': exact,
           'squad_share': None, 'squad_avail': None, 'n_squad': 0, 'n_club': 0, 'clubs': []}
    if 'squad_mins' in r.columns:
        q = r[r['n_squad'].fillna(0) > 0]
        if len(q):
            squad_avail = float(np.maximum(q['squad_mins'], q['mins_year']).sum())
            out.update({'squad_share': min(float(q['mins_year'].sum()) / squad_avail, 1.0),
                        'squad_avail': squad_avail, 'squad_played': float(q['mins_year'].sum()),
                        'n_squad': int(r['n_squad'].fillna(0).sum()),
                        'n_club': int(r['n_club'].fillna(0).sum()),
                        'clubs': [(c.rsplit(':', 2)[0], int(c.rsplit(':', 2)[1]), int(c.rsplit(':', 2)[2]))
                                  for s in r['squad_clubs'].dropna() for c in str(s).split('|') if c]})
    return out


# --- refit (yearly rollover) -------------------------------------------------------
def fit(inp=None, years=FIT_YEARS):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, roc_auc_score
    inp = inp or load_inputs()
    FIT = tuple(years)
    eng = inp['eng'].assign(yr=inp['eng']['seasonId'].map(lambda s: SEASON_YEAR.get(int(s))))
    league = inp['eng']['seasonId'].map(lambda s: 'CAMP' if int(s) in CAMP_SEASON_IDS else 'L3')
    per = eng.groupby(['playerId', 'yr'])['mins_lineup'].sum()
    w = eng[(league == 'L3') & eng['yr'].isin(FIT) & ~eng['team'].map(is_reserve_team)].drop_duplicates(['playerId', 'yr'])
    b = build(inp)
    b = b[b['in_engine']].drop_duplicates(['playerId', 'yr'])[['playerId', 'yr', 'career_share', 'career_sub_frac']]
    w = w[['playerId', 'yr']].merge(b.astype({'playerId': 'int64'}), on=['playerId', 'yr']).dropna(subset=['career_share'])
    X = np.c_[np.log(w['career_share'] + SHARE_EPS), w['career_sub_frac']]
    y = np.array([int(per.get((p, yy + 1), 0.0) >= REGULAR_NEXT) for p, yy in zip(w['playerId'], w['yr'])])
    ll, auc = [], []
    for tr_y, te_y in ((FIT, FIT[::-1]) if len(FIT) == 2 else ()):
        a, b = (w['yr'] == tr_y).to_numpy(), (w['yr'] == te_y).to_numpy()
        m = LogisticRegression(C=1e6, max_iter=5000).fit(X[a], y[a])
        p = m.predict_proba(X[b])[:, 1]
        ll.append(log_loss(y[b], p)); auc.append(roc_auc_score(y[b], p))
    m = LogisticRegression(C=1e6, max_iter=5000).fit(X, y)
    coef = (round(float(m.intercept_[0]), 4), round(float(m.coef_[0][0]), 4), round(float(m.coef_[0][1]), 4))
    print(f'years {FIT}: rows {len(w)}' + (f'; cross-season log-loss {np.mean(ll):.4f}, AUC {np.mean(auc):.4f}' if ll else ''))
    print(f'COEF = {coef}   (pinned: {COEF})')
    return coef


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--fit', action='store_true', help='refit COEF and print it (yearly rollover)')
    if ap.parse_args().fit:
        _inp = load_inputs()
        fit(_inp)
        fit(_inp, FIT_YEARS[:1])      # COEF_ASOF's earlier entry
    else:
        out = build()
        out.to_parquet(OUT, index=False)
        e = out[out['in_engine']]
        print(f"availability: {len(e)} engine rows (+{len(out) - len(e)} display-only); window cut {int(e['window_cut'].sum())}; "
              f"reserve {int(e['reserve_team'].sum())}; factor < 1: {int((e['regular_factor'] < 1).sum())} "
              f"(median {e.loc[e['regular_factor'] < 1, 'regular_factor'].median():.2f}); "
              f"API-exact lengths {e['len_api'].mean():.0%}")
