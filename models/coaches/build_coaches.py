"""Build the coach tables behind the Coach Profile page and the profile's
'Coached by' section.

    python models/coaches/build_coaches.py            # incremental
    python models/coaches/build_coaches.py --no-api   # local sources only

Writes, next to this file:

  coach_match.parquet       matchId, teamId, coachId — who was on the bench.
                            Write-once per match: seeded from the local
                            wyscout_loader cache when it exists, the rest from
                            the Wyscout match endpoint (teamsData.<id>.coachId).
  coaches.parquet           coachId, name, birthDate, nationality (Wyscout
                            /coaches/{id}; a coach the API refuses keeps a
                            'Coach #id' label).
  coach_team_match.parquet  one row per team per match: result, xG, xPts,
                            pre-match ratings, style counts, game state,
                            substitutions, line-up changes.
  coach_player.parquet      one row per coach × team × player: exact minutes,
                            starts, OBV, and the player's OWN EARLIER record.

No look-ahead: every expectation on a row (pre-match ratings, home edge, the
team's level when the coach arrived, a player's baseline) is computed only
from matches dated before the match / spell it is compared with. The rating
step is a fixed constant (RATING_K), never tuned on the results it judges.
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from event_tags import TagIndex  # noqa: E402
from models.coaches.formations import EVENT_COLS as FORMATION_EVENT_COLS  # noqa: E402
from models.coaches.formations import formations_from_events  # noqa: E402

COACH_MATCH = HERE / 'coach_match.parquet'
COACHES = HERE / 'coaches.parquet'
TEAM_MATCH = HERE / 'coach_team_match.parquet'
COACH_PLAYER = HERE / 'coach_player.parquet'
NAME_OVERRIDES = HERE / 'coach_names.csv'

LOADER_DB = ROOT.parents[1] / 'wyscout_loader' / 'data' / 'wyscout.duckdb'
API_BASE = 'https://apirest.wyscout.com/v3'

# Rating step: each match moves a team's xG-difference rating by this share of
# the surprise (about a 14-match memory). Fixed in advance — not fitted.
RATING_K = 0.07
# A team needs this many rated matches in the competition before its level
# counts as a baseline for a newly arrived coach.
MIN_BASELINE_MATCHES = 10
# possession types that make a shot a set-piece shot (a throw-in or a deep free
# kick that merely starts a possession does not)
SET_PIECE_TYPES = ('set_piece_attack', 'corner', 'free_kick_cross', 'direct_free_kick', 'penalty')

EVENT_COLS = [
    'matchId', 'matchPeriod', 'minute', 'second', 'type.primary',
    'type.secondary', 'location.x', 'team.id', 'team.name',
    'opponentTeam.id', 'pass.accurate', 'pass.length', 'possession.id',
    'possession.types', 'possession.team.id', 'shot.isGoal', 'shot.xg',
]


# ───────────────────────── coach per team per match ─────────────────────────
def _session():
    import requests
    from requests.auth import HTTPBasicAuth
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / '.env')
    except Exception:
        pass
    user = os.environ.get('WYSCOUT_USER_NEW') or os.environ.get('WYSCOUT_USER')
    pwd = os.environ.get('WYSCOUT_PASS_NEW') or os.environ.get('WYSCOUT_PASS')
    if not user or not pwd:
        return None
    s = requests.Session()
    s.auth = HTTPBasicAuth(user, pwd)
    return s


def _api_get(session, path):
    import time
    for attempt in range(3):
        try:
            r = session.get(API_BASE + path, timeout=30)
        except Exception:
            time.sleep(2)
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429:
            time.sleep(2 + attempt * 3)
            continue
        return None
    return None


def _from_loader(match_ids):
    """Coach ids already cached by wyscout_loader (local machines only)."""
    if not LOADER_DB.exists():
        return pd.DataFrame(columns=['matchId', 'teamId', 'coachId'])
    try:
        import duckdb
        con = duckdb.connect(str(LOADER_DB), read_only=True)
        names = con.execute('describe match_meta').df().column_name.tolist()
        cols = [c for c in names if c.endswith('.coachId')]
        if not cols:
            return pd.DataFrame(columns=['matchId', 'teamId', 'coachId'])
        sel = ', '.join(f'"{c}"' for c in cols)
        df = con.execute(f'select matchId, {sel} from match_meta').df()
    except Exception as exc:  # the cache is optional
        print(f'  loader cache not readable ({exc})')
        return pd.DataFrame(columns=['matchId', 'teamId', 'coachId'])
    df = df[df.matchId.isin(match_ids)]
    long = df.melt('matchId', var_name='col', value_name='coachId').dropna(subset=['coachId'])
    long['teamId'] = long['col'].str.extract(r'teamsData\.(\d+)\.')[0].astype('int64')
    long['coachId'] = long['coachId'].astype('int64')
    return long[['matchId', 'teamId', 'coachId']]


def build_coach_match(match_ids, use_api=True):
    have = pd.read_parquet(COACH_MATCH) if COACH_MATCH.exists() else \
        pd.DataFrame(columns=['matchId', 'teamId', 'coachId'])
    missing = sorted(set(match_ids) - set(have.matchId))
    if missing:
        seed = _from_loader(set(missing))
        have = pd.concat([have, seed], ignore_index=True)
        missing = sorted(set(missing) - set(seed.matchId))
        print(f'  coach ids: {seed.matchId.nunique()} matches from the loader cache, '
              f'{len(missing)} to fetch')
    if missing and use_api:
        session = _session()
        if session is None:
            print('  no Wyscout credentials: skipping the API fetch')
        else:
            rows = []
            for i, mid in enumerate(missing):
                data = _api_get(session, f'/matches/{mid}')
                for tid, td in ((data or {}).get('teamsData') or {}).items():
                    rows.append({'matchId': mid, 'teamId': int(td.get('teamId', tid)),
                                 'coachId': int(td.get('coachId') or 0)})
                if (i + 1) % 50 == 0:
                    print(f'    {i + 1}/{len(missing)} matches')
            if rows:
                have = pd.concat([have, pd.DataFrame(rows)], ignore_index=True)
    have = have.drop_duplicates(['matchId', 'teamId']).astype('int64')
    have.sort_values(['matchId', 'teamId']).to_parquet(COACH_MATCH, index=False)
    return have


def build_coaches(coach_ids, use_api=True):
    have = pd.read_parquet(COACHES) if COACHES.exists() else pd.DataFrame(
        columns=['coachId', 'name', 'fullName', 'birthDate', 'nationality', 'resolved'])
    done = set(have.loc[have.resolved, 'coachId'])
    todo = sorted(set(coach_ids) - done - {0})
    session = _session() if (use_api and todo) else None
    rows = []
    if session is not None:
        for i, cid in enumerate(todo):
            d = _api_get(session, f'/coaches/{cid}')
            if d:
                full = ' '.join(x.strip() for x in (d.get('firstName') or '', d.get('lastName') or '') if x.strip())
                rows.append({'coachId': cid, 'name': (d.get('shortName') or full or f'Coach #{cid}').strip(),
                             'fullName': full, 'birthDate': d.get('birthDate'),
                             'nationality': (d.get('passportArea') or {}).get('name')
                             or (d.get('birthArea') or {}).get('name'),
                             'resolved': True})
            if (i + 1) % 50 == 0:
                print(f'    {i + 1}/{len(todo)} coaches')
    # coaches the endpoint refuses: the loader's squad cache names the current ones
    got = {r['coachId'] for r in rows}
    if LOADER_DB.exists() and set(todo) - got:
        try:
            import duckdb
            sq = duckdb.connect(str(LOADER_DB), read_only=True).execute(
                "select wyId, shortName, firstName, lastName, birthDate, \"passportArea.name\" as nat "
                "from team_squad where role = 'coach'").df()
            for r in sq[sq.wyId.isin(set(todo) - got)].drop_duplicates('wyId').itertuples():
                full = ' '.join(x.strip() for x in (r.firstName or '', r.lastName or '') if x.strip())
                rows.append({'coachId': int(r.wyId), 'name': (r.shortName or full).strip(), 'fullName': full,
                             'birthDate': r.birthDate, 'nationality': r.nat, 'resolved': True})
        except Exception as exc:
            print(f'  loader squad cache not readable ({exc})')
    new = pd.DataFrame(rows)
    out =pd.concat([have[~have.coachId.isin(set(new.get('coachId', [])))], new], ignore_index=True)
    for cid in sorted(set(coach_ids) - set(out.coachId) - {0}):
        out.loc[len(out)] = {'coachId': cid, 'name': f'Coach #{cid}', 'fullName': '',
                             'birthDate': None, 'nationality': None, 'resolved': False}
    out['coachId'] = out['coachId'].astype('int64')
    out['resolved'] = out['resolved'].astype(bool)
    # hand-kept names for coaches the API refuses (coachId,name[,nationality,birthDate])
    if NAME_OVERRIDES.exists():
        ov = pd.read_csv(NAME_OVERRIDES).dropna(subset=['coachId', 'name']).set_index('coachId')
        hit = out.coachId.isin(ov.index)
        out.loc[hit, 'name'] = out.loc[hit, 'coachId'].map(ov['name'])
        for col in ('nationality', 'birthDate'):
            if col in ov.columns:
                out.loc[hit, col] = out.loc[hit, 'coachId'].map(ov[col]).fillna(out.loc[hit, col])
    out.sort_values('coachId').to_parquet(COACHES, index=False)
    return out


# ───────────────────────────── match features ─────────────────────────────
def _goal_dist(xgs):
    """Exact distribution of goals from a list of shot xG values."""
    dist = np.array([1.0])
    for p in xgs:
        p = min(max(float(p), 0.0), 0.99)
        dist = np.convolve(dist, [1 - p, p])
    return dist


def expected_points(xg_for, xg_against):
    """xPts for the side whose shots are `xg_for` (each shot an independent
    chance; no model parameters, so nothing to leak)."""
    a, b = _goal_dist(xg_for), _goal_dist(xg_against)
    joint = np.outer(a, b)
    win = np.tril(joint, -1).sum()
    draw = np.trace(joint)
    return 3 * win + draw


def team_match_counts(ev):
    """Per (matchId, team.id) style counts from an events frame."""
    tags = TagIndex(ev['type.secondary'])
    tp = ev['type.primary']
    x = pd.to_numeric(ev['location.x'], errors='coerce')
    is_pass = tp == 'pass'
    is_def = (tp == 'interception') | (tp == 'clearance') | tags.has('defensive_duel') | tags.has('sliding_tackle')
    rec = tags.has('recovery')
    xg = pd.to_numeric(ev['shot.xg'], errors='coerce')
    ptypes = TagIndex(ev['possession.types'])
    sp = np.zeros(len(ev), dtype=bool)
    for t in SET_PIECE_TYPES:
        sp |= np.asarray(ptypes.has(t))
    sp |= (tp == 'penalty').to_numpy()
    own_poss = ev['possession.team.id'] == ev['team.id']
    f = pd.DataFrame({
        'matchId': ev['matchId'].to_numpy(), 'teamId': ev['team.id'].to_numpy(),
        'passes': is_pass, 'passes_acc': is_pass & ev['pass.accurate'].fillna(False).astype(bool),
        'passes_f3': is_pass & (x >= 66), 'long_passes': is_pass & tags.has('long_pass'),
        'fwd_passes': is_pass & tags.has('forward_pass'), 'prog_passes': is_pass & tags.has('progressive_pass'),
        'crosses': tags.has('cross'), 'box_touches': tags.has('touch_in_box'),
        'press_passes': is_pass & (x <= 60), 'press_def': is_def & (x >= 40),
        'def_actions': is_def, 'def_x_sum': x.where(is_def).fillna(0),
        'recoveries': rec, 'high_recoveries': rec & (x >= 66),
        'counterpress': tags.has('counterpressing_recovery'),
        'shots': xg.notna(), 'xg': xg.fillna(0), 'xg_set_piece': xg.fillna(0).where(sp, 0),
        'dribbles': tags.has('dribble'),
    })
    out = f.groupby(['matchId', 'teamId']).sum()
    poss = ev.loc[own_poss, ['matchId', 'team.id', 'possession.id']].drop_duplicates()
    out['possessions'] = poss.groupby(['matchId', 'team.id']).size().reindex(out.index).fillna(0)
    counter = np.asarray(ptypes.has('counterattack')) & own_poss.to_numpy()
    cp = ev.loc[counter, ['matchId', 'team.id', 'possession.id']].drop_duplicates()
    out['counterattacks'] = cp.groupby(['matchId', 'team.id']).size().reindex(out.index).fillna(0)
    return out.reset_index()


def _clock(period, minute):
    """Minutes on a single 0-90+ clock (first-half stoppage folded into 45)."""
    m = np.asarray(minute, dtype=float)
    first = np.asarray(period) == '1H'
    return np.where(first, np.minimum(m, 45.0), np.maximum(m, 45.0))


def game_state_rows(ev):
    """Minutes and xG split by game state, from the goals in the events.
    Own goals are not shot events, so a match whose event goals differ from the
    final score is flagged by the caller and left out of the game-state view."""
    ev = ev.assign(_t=_clock(ev['matchPeriod'], ev['minute'] + ev['second'].fillna(0) / 60.0))
    xg = pd.to_numeric(ev['shot.xg'], errors='coerce')
    shots = ev.loc[xg.notna(), ['matchId', 'team.id', 'opponentTeam.id', '_t', 'shot.isGoal',
                                'matchPeriod', 'minute', 'second']].assign(xg=xg[xg.notna()])
    shots['goal'] = shots['shot.isGoal'].fillna(False).astype(bool)
    end = ev.groupby('matchId')['_t'].max()
    rows = []
    # real order (a first-half stoppage goal comes before any second-half shot)
    for mid, g in shots.sort_values(['matchPeriod', 'minute', 'second']).groupby('matchId', sort=False):
        teams = pd.unique(g[['team.id', 'opponentTeam.id']].to_numpy().ravel())
        teams = [int(t) for t in teams if pd.notna(t)][:2]
        if len(teams) < 2:
            continue
        a, b = teams
        acc = {t: {'min_lead': 0.0, 'min_draw': 0.0, 'min_trail': 0.0,
                   'xgd_lead': 0.0, 'xgd_draw': 0.0, 'xgd_trail': 0.0,
                   'ever_led': False, 'ever_trailed': False, 'scored_first': False,
                   'ev_goals': 0} for t in teams}
        diff, last, first_goal = 0, 0.0, True   # diff from a's side
        def state(t):
            d = diff if t == a else -diff
            return 'lead' if d > 0 else 'trail' if d < 0 else 'draw'
        for _, s in g.iterrows():
            for t in teams:
                acc[t]['min_' + state(t)] += max(s['_t'] - last, 0)
            last = max(last, s['_t'])
            shooter = int(s['team.id'])
            other = b if shooter == a else a
            acc[shooter]['xgd_' + state(shooter)] += s['xg']
            acc[other]['xgd_' + state(other)] -= s['xg']
            if s['goal']:
                diff += 1 if shooter == a else -1
                acc[shooter]['ev_goals'] += 1
                if first_goal:
                    acc[shooter]['scored_first'] = True
                    first_goal = False
                for t in teams:
                    st = state(t)
                    acc[t]['ever_led'] |= st == 'lead'
                    acc[t]['ever_trailed'] |= st == 'trail'
        for t in teams:
            acc[t]['min_' + state(t)] += max(float(end.get(mid, 90.0)) - last, 0)
            rows.append({'matchId': mid, 'teamId': t, **acc[t]})
    return pd.DataFrame(rows)


def lineup_rows(lineups, name_to_id):
    rows = []
    for mid, teams in lineups.items():
        for tname, d in (teams or {}).items():
            tid = name_to_id.get(tname)
            if tid is None:
                continue
            xi = [int(p['playerId']) for p in d.get('lineup', []) if p.get('playerId')]
            subs = [s for s in d.get('substitutions', []) if s.get('minute') is not None]
            mins = sorted(int(s['minute']) for s in subs)
            rows.append({'matchId': int(mid), 'teamId': tid, 'xi': xi,
                         'subs_used': len(subs),
                         'first_sub_min': mins[0] if mins else np.nan,
                         'mean_sub_min': float(np.mean(mins)) if mins else np.nan,
                         'subs_before_60': sum(m < 60 for m in mins),
                         'sub_in_ids': [int(s['playerIn']) for s in subs if s.get('playerIn')],
                         'bench': [int(p['playerId']) for p in d.get('bench', []) if p.get('playerId')]})
    return pd.DataFrame(rows)


def verify_sheet_teams(lu):
    """Some line-up sheets are filed under the OPPONENT's name. A sheet
    belongs to the team its starters acted for in the events: re-label a
    sheet whose starters played for the other side, and drop a match's
    sheets when two of them claim the same team."""
    obv = pd.read_parquet(ROOT / 'obv_match_player.parquet', columns=['matchId', 'teamId', 'playerId'])
    obv = obv.drop_duplicates(['matchId', 'playerId']).rename(columns={'teamId': 'evTeam'})
    x = lu[['matchId', 'teamId', 'xi']].explode('xi').dropna().rename(columns={'xi': 'playerId'})
    x['playerId'] = x.playerId.astype('int64')
    x = x.merge(obv, on=['matchId', 'playerId'])
    votes = x.groupby(['matchId', 'teamId', 'evTeam']).size().rename('n').reset_index()
    top = votes.sort_values('n', ascending=False).drop_duplicates(['matchId', 'teamId'])
    top = top[top.n >= 6]
    lu = lu.merge(top[['matchId', 'teamId', 'evTeam']], on=['matchId', 'teamId'], how='left')
    moved = int((lu.evTeam.notna() & (lu.evTeam != lu.teamId)).sum())
    lu['teamId'] = lu.evTeam.fillna(lu.teamId).astype('int64')
    clash = lu.duplicated(['matchId', 'teamId'], keep=False)
    print(f'  line-up sheets: {moved} re-labelled to the team their starters played for, '
          f'{int(clash.sum())} dropped (two sheets for one team)')
    # a sheet that cannot be checked (no event rows for its starters) is not used
    unverified = lu.evTeam.isna()
    print(f'  line-up sheets: {int((unverified & ~clash).sum())} unverifiable (no event rows) left out')
    return lu[~clash & ~unverified].drop(columns=['evTeam'])


def add_ratings(tm):
    """Walk-forward xG-difference ratings (xGD per match vs an average side of
    the same competition). Adds, per row, the numbers known BEFORE the match
    day: r_pre / r_opp_pre / hfa_pre / n_rated_pre. Matches of one calendar day
    are rated together from the state at the end of the previous day — a
    15:00 result never informs a 17:00 expectation. A side new to a
    competition starts at that competition's current mean."""
    tm = tm.sort_values(['date', 'matchId']).reset_index(drop=True)
    rating, n_rated = {}, {}
    home_sum, home_n = {}, {}
    cols = {k: np.full(len(tm), np.nan) for k in ('r_pre', 'r_opp_pre', 'hfa_pre', 'n_rated_pre')}
    day = tm.date.dt.normalize()
    for _, day_idx in tm.groupby(day, sort=True).groups.items():
        updates = []
        for mid, idx in tm.loc[day_idx].groupby('matchId', sort=False).groups.items():
            idx = list(idx)
            if len(idx) != 2:
                continue
            rows = tm.loc[idx]
            comp = rows.competitionId.iloc[0]
            cur = [v for (c, _), v in rating.items() if c == comp]
            mean = float(np.mean(cur)) if cur else 0.0
            hfa = home_sum.get(comp, 0.0) / home_n[comp] if home_n.get(comp) else 0.0
            keys = [(comp, int(t)) for t in rows.teamId]
            pre = [rating.get(k, mean) for k in keys]
            for j, i in enumerate(idx):
                cols['r_pre'][i] = pre[j]
                cols['r_opp_pre'][i] = pre[1 - j]
                cols['hfa_pre'][i] = hfa
                cols['n_rated_pre'][i] = n_rated.get(keys[j], 0)
            if rows.xgd.isna().any():
                continue
            hj = 0 if rows.side.iloc[0] == 'home' else 1
            obs = float(rows.xgd.iloc[hj])
            updates.append((comp, keys[hj], keys[1 - hj], pre[hj], pre[1 - hj],
                            obs - (pre[hj] - pre[1 - hj] + hfa), obs - (pre[hj] - pre[1 - hj])))
        for comp, kh, ka, ph, pa, surprise, home_resid in updates:
            # a side with two matches on one day (never in practice) keeps both steps
            rating[kh] = rating.get(kh, ph) + RATING_K * surprise
            rating[ka] = rating.get(ka, pa) - RATING_K * surprise
            n_rated[kh] = n_rated.get(kh, 0) + 1
            n_rated[ka] = n_rated.get(ka, 0) + 1
            home_sum[comp] = home_sum.get(comp, 0.0) + home_resid
            home_n[comp] = home_n.get(comp, 0) + 1
    for k, v in cols.items():
        tm[k] = v
    return tm


Z_METRICS = ('possession', 'field_tilt', 'passes_per_poss', 'long_share', 'forward_share',
             'def_height', 'log_ppda')
MIN_CONTEXT_ROWS = 300


def match_style(tm):
    """Per-match values of the style metrics (NaN without events)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        f = pd.DataFrame({
            'possession': tm.passes / (tm.passes + tm.opp_passes),
            'field_tilt': tm.passes_f3 / (tm.passes_f3 + tm.opp_passes_f3),
            'passes_per_poss': tm.passes / tm.possessions,
            'long_share': tm.long_passes / tm.passes,
            'forward_share': tm.fwd_passes / tm.passes,
            'def_height': tm.def_x_sum / tm.def_actions,
            'log_ppda': np.log(tm.opp_press_passes / tm.press_def),
        }, index=tm.index)
    return f.replace([np.inf, -np.inf], np.nan).where(tm.has_events.astype(bool))


def add_context_adjusted(tm):
    """z_<metric>: the match's style value with the league-typical effect of
    opponent strength and venue removed, in units of the league's match-to-
    match spread (the idea of Hooker's 'Manager Bounce' study, Hudl 2025).
    The adjustment for a match is fitted ONLY on that competition's matches
    of earlier calendar months; until MIN_CONTEXT_ROWS exist it is blank.
    Scoreline effects are not removed."""
    tm = tm.copy()
    vals = match_style(tm)
    d = (tm.r_pre - tm.r_opp_pre).to_numpy()
    home = np.where(tm.side == 'home', 1.0, 0.0)
    month = tm.date.dt.to_period('M')
    z = {m: np.full(len(tm), np.nan) for m in Z_METRICS}
    pos = np.arange(len(tm))
    for comp in tm.competitionId.unique():
        in_comp = (tm.competitionId == comp).to_numpy()
        for mth in sorted(month[in_comp].unique()):
            cur = in_comp & (month == mth).to_numpy()
            prior = in_comp & (month < mth).to_numpy() & ~np.isnan(d)
            for m in Z_METRICS:
                y = vals[m].to_numpy()
                ok = prior & ~np.isnan(y)
                if ok.sum() < MIN_CONTEXT_ROWS:
                    continue
                X = np.column_stack([np.ones(ok.sum()), d[ok], home[ok]])
                beta, *_ = np.linalg.lstsq(X, y[ok], rcond=None)
                sd = float(np.std(y[ok] - X @ beta, ddof=3))
                if sd <= 0:
                    continue
                i = pos[cur & ~np.isnan(y) & ~np.isnan(d)]
                z[m][i] = (y[i] - (beta[0] + beta[1] * d[i] + beta[2] * home[i])) / sd
    for m in Z_METRICS:
        tm['z_' + m] = z[m]
    return tm


STATES = ('lead', 'draw', 'trail')


def add_state_expected(tm):
    """Expected game-state numbers for each match, given how strong the side
    and its opponent were going in and who was at home:

      exp_min_<state>   minutes a side of that strength typically spends in
                        the state (the three add up to the match's minutes)
      exp_xgd_<state>   xG difference it typically makes over the minutes the
                        side actually spent in the state

    Both come from a straight-line fit on rating gap and venue, fitted ONLY
    on that competition's clean-timeline matches of earlier calendar months
    (blank until MIN_CONTEXT_ROWS exist). The xG fit is per minute in the
    state, weighted by those minutes. Every side does better on xG when
    behind; this is what puts a number on 'better than usual when behind'."""
    tm = tm.copy()
    d = (tm.r_pre - tm.r_opp_pre).to_numpy(dtype=float)
    home = np.where(tm.side == 'home', 1.0, 0.0)
    month = tm.date.dt.to_period('M')
    ok_row = tm.state_ok.eq(True).to_numpy() & ~np.isnan(d)
    mins = {s: tm[f'min_{s}'].to_numpy(dtype=float) for s in STATES}
    xgd = {s: tm[f'xgd_{s}'].to_numpy(dtype=float) for s in STATES}
    total = sum(mins.values())
    out = {f'exp_{k}_{s}': np.full(len(tm), np.nan) for k in ('min', 'xgd') for s in STATES}
    X_all = np.column_stack([np.ones(len(tm)), np.nan_to_num(d), home])
    for comp in tm.competitionId.unique():
        in_comp = (tm.competitionId == comp).to_numpy()
        for mth in sorted(month[in_comp].unique()):
            prior = in_comp & (month < mth).to_numpy() & ok_row & (total > 0)
            if prior.sum() < MIN_CONTEXT_ROWS:
                continue
            cur = in_comp & (month == mth).to_numpy() & ok_row
            share = {}
            for s in STATES:
                beta, *_ = np.linalg.lstsq(X_all[prior], mins[s][prior] / total[prior], rcond=None)
                share[s] = np.clip(X_all[cur] @ beta, 0.0, None)
            norm = sum(share.values())
            for s in STATES:
                out[f'exp_min_{s}'][cur] = share[s] / norm * total[cur]
                use = prior & (mins[s] > 0)
                w = np.sqrt(mins[s][use])
                beta, *_ = np.linalg.lstsq(X_all[use] * w[:, None], xgd[s][use] / mins[s][use] * w, rcond=None)
                out[f'exp_xgd_{s}'][cur] = (X_all[cur] @ beta) * mins[s][cur]
    for k, v in out.items():
        tm[k] = v
    return tm


def add_spells(tm):
    """Number each coach's spells at a team (a spell ends when another coach
    takes a match) and stamp the team's level on arrival."""
    tm = tm.sort_values(['teamId', 'date', 'matchId']).reset_index(drop=True)
    known = tm.coachId > 0
    # a match with no coach on the sheet does not end a spell: A, ?, A is one
    k = tm[known]
    change = (k.teamId != k.teamId.shift()) | (k.coachId != k.coachId.shift())
    tm['spell'] = -1
    tm.loc[known, 'spell'] = change.cumsum()
    first = tm[known].groupby('spell').head(1).set_index('spell')
    # the level a coach inherited: the team's pre-match rating in his first
    # match, if the side had enough rated matches in that competition by then
    base = first['r_pre'].where(first['n_rated_pre'] >= MIN_BASELINE_MATCHES)
    tm['r_inherited'] = tm['spell'].map(base)
    comp_first = tm['spell'].map(first['competitionId'])
    tm.loc[tm.competitionId != comp_first, 'r_inherited'] = np.nan
    return tm


def player_team_minutes(tm):
    """Exact minutes per player per match with the team he played for. The
    team comes from the line-up sheet (starters + players brought on), which
    the scheduled refresh keeps current, or from the OBV export (the team he
    acted for in the events). One row per (matchId, playerId)."""
    mins = pd.read_parquet(ROOT / 'match_player_minutes.parquet',
                           columns=['matchId', 'playerId', 'minutes', 'started'])
    mins = mins[mins.minutes > 0].drop_duplicates(['matchId', 'playerId'])
    sheet = tm[['matchId', 'teamId']].assign(
        playerId=[list(a) + list(b) for a, b in zip(tm.xi, tm.sub_in_ids)]).explode('playerId').dropna()
    sheet['playerId'] = sheet.playerId.astype('int64')
    obv = pd.read_parquet(ROOT / 'obv_match_player.parquet',
                          columns=['matchId', 'teamId', 'playerId', 'obv'])
    # the team a player acted for in the events wins; the sheet fills players without an event
    team = pd.concat([obv[['matchId', 'teamId', 'playerId']], sheet]).drop_duplicates(['matchId', 'playerId'])
    pm = mins.merge(team, on=['matchId', 'playerId'], how='inner')
    pm = pm.merge(obv.drop_duplicates(['matchId', 'teamId', 'playerId']),
                  on=['matchId', 'teamId', 'playerId'], how='left')
    pm['has_obv'] = pm.obv.notna()
    pm['obv'] = pm.obv.fillna(0.0)
    return pm


AGE_CUTS = (21, 23)


def _birth_dates():
    with open(ROOT / 'player_details.pkl', 'rb') as fh:
        det = pd.DataFrame(pickle.load(fh))[['playerId', 'birthDate']]
    det['birthDate'] = pd.to_datetime(det.birthDate, errors='coerce')
    det = det.dropna().drop_duplicates('playerId')
    det['playerId'] = det.playerId.astype('int64')
    return det


def add_youth_minutes(tm):
    """Age columns per team-match, players with a known birth date only.
    Minutes are the exact ones; the MATCHDAY SQUAD is the line-up sheet
    (starters + bench), so 'how many young players did he have to pick from'
    sits next to 'how many minutes did he give them'."""
    det = _birth_dates()
    pm = player_team_minutes(tm)[['matchId', 'teamId', 'playerId', 'minutes']].merge(det, on='playerId')
    pm = pm.merge(tm[['matchId', 'teamId', 'date']], on=['matchId', 'teamId'])
    pm['age'] = (pm.date - pm.birthDate).dt.days / 365.25
    pm['age_min'] = pm.age * pm.minutes
    named = {'squad_minutes': ('minutes', 'sum'), 'age_minutes_sum': ('age_min', 'sum')}
    for c in AGE_CUTS:
        pm[f'u{c}'] = pm.minutes.where(pm.age < c, 0)
        named[f'u{c}_minutes'] = (f'u{c}', 'sum')
    out = tm.merge(pm.groupby(['matchId', 'teamId']).agg(**named).reset_index(),
                   on=['matchId', 'teamId'], how='left')

    sheet = pd.concat([
        tm[['matchId', 'teamId', 'date']].assign(playerId=tm.xi, starter=True).explode('playerId'),
        tm[['matchId', 'teamId', 'date']].assign(playerId=tm.bench, starter=False).explode('playerId'),
    ]).dropna(subset=['playerId'])
    sheet['playerId'] = sheet.playerId.astype('int64')
    sheet = sheet.drop_duplicates(['matchId', 'teamId', 'playerId']).merge(det, on='playerId')
    sheet['age'] = (sheet.date - sheet.birthDate).dt.days / 365.25
    sheet['xi_age'] = sheet.age.where(sheet.starter)
    named = {'sheet_n': ('age', 'size'), 'sheet_age_sum': ('age', 'sum'),
             'xi_n': ('starter', 'sum'), 'xi_age_sum': ('xi_age', 'sum')}
    for c in AGE_CUTS:
        sheet[f'u{c}_named'] = sheet.age < c
        sheet[f'u{c}_start'] = sheet.starter & (sheet.age < c)
        named[f'u{c}_named'] = (f'u{c}_named', 'sum')
        named[f'u{c}_starts'] = (f'u{c}_start', 'sum')
    return out.merge(sheet.groupby(['matchId', 'teamId']).agg(**named).reset_index(),
                     on=['matchId', 'teamId'], how='left')


def build_team_match(coach_match):
    ms = pd.read_parquet(ROOT / 'matches_summary.parquet')
    ms = ms[ms.status == 'Played'].copy()
    print('  reading events …')
    ev = pd.read_parquet(ROOT / 'raw_events.parquet', columns=EVENT_COLS)
    names = ev[['team.id', 'team.name']].drop_duplicates('team.name')
    name_to_id = dict(zip(names['team.name'], names['team.id'].astype(int)))
    id_to_name = {v: k for k, v in name_to_id.items()}

    sc = ms['score'].str.extract(r'(\d+)\s*-\s*(\d+)').astype(float)
    ms['hg'], ms['ag'] = sc[0], sc[1]
    base = []
    for side, opp, gf, ga in (('home', 'away', 'hg', 'ag'), ('away', 'home', 'ag', 'hg')):
        base.append(pd.DataFrame({
            'matchId': ms.matchId.astype('int64'), 'seasonId': ms.seasonId,
            'competitionId': ms.competitionId.astype('int64'), 'date': ms.dateutc,
            'side': side, 'team': ms[f'{side}TeamName'], 'opponent': ms[f'{opp}TeamName'],
            'gf': ms[gf], 'ga': ms[ga]}))
    tm = pd.concat(base, ignore_index=True)
    tm['teamId'] = tm.team.map(name_to_id)
    tm['oppId'] = tm.opponent.map(name_to_id)
    tm = tm.dropna(subset=['teamId', 'gf']).copy()
    tm['teamId'] = tm.teamId.astype('int64')
    tm['points'] = np.where(tm.gf > tm.ga, 3, np.where(tm.gf == tm.ga, 1, 0))

    print('  style counts …')
    counts = team_match_counts(ev)
    tm = tm.merge(counts, on=['matchId', 'teamId'], how='left')
    opp = counts.rename(columns={c: 'opp_' + c for c in counts.columns if c not in ('matchId', 'teamId')}) \
        .rename(columns={'teamId': 'oppId'})
    keep = ['matchId', 'oppId', 'opp_xg', 'opp_shots', 'opp_passes', 'opp_passes_f3',
            'opp_press_passes', 'opp_xg_set_piece', 'opp_box_touches']
    tm = tm.merge(opp[keep], on=['matchId', 'oppId'], how='left')
    tm['has_events'] = tm.passes.notna() & tm.opp_passes.notna()
    tm['xgd'] = (tm.xg - tm.opp_xg).where(tm.has_events)

    print('  expected points …')
    xg = pd.to_numeric(ev['shot.xg'], errors='coerce')
    sh = ev.loc[xg.notna(), ['matchId', 'team.id']].assign(xg=xg[xg.notna()])
    shot_lists = sh.groupby(['matchId', 'team.id'])['xg'].apply(list).to_dict()
    tm['xpts'] = [expected_points(shot_lists.get((m, t), []), shot_lists.get((m, o), []))
                  if he else np.nan
                  for m, t, o, he in zip(tm.matchId, tm.teamId, tm.oppId, tm.has_events)]

    print('  game states …')
    gs = game_state_rows(ev)
    tm = tm.merge(gs, on=['matchId', 'teamId'], how='left')
    both = tm.groupby('matchId')['ev_goals'].transform('sum')
    tm['state_ok'] = tm.has_events & (both.fillna(-1) == (tm.gf + tm.ga)) & (tm.ev_goals.fillna(-1) == tm.gf)
    # a goalless match has no shots that changed the state
    nil = tm.has_events & (tm.gf + tm.ga == 0)
    tm.loc[nil, 'state_ok'] = True

    # formations are OUR OWN read of touch locations (formations.py): no
    # provider formation or position label is read anywhere in this build
    print('  formations …')
    shapes = formations_from_events(pd.read_parquet(ROOT / 'raw_events.parquet', columns=FORMATION_EVENT_COLS))
    tm = tm.merge(shapes[['matchId', 'teamId', 'formation', 'formation_margin']],
                  on=['matchId', 'teamId'], how='left')
    del ev

    print('  line-ups …')
    with open(ROOT / 'match_lineups.pkl', 'rb') as fh:
        lu = lineup_rows(pickle.load(fh), name_to_id)
    lu = verify_sheet_teams(lu)
    tm = tm.merge(lu, on=['matchId', 'teamId'], how='left')

    tm = tm.merge(coach_match, on=['matchId', 'teamId'], how='left')
    tm['coachId'] = tm.coachId.fillna(0).astype('int64')
    tm = add_ratings(tm)
    tm = add_context_adjusted(tm)
    tm = add_state_expected(tm)
    tm = add_spells(tm)

    # line-up changes vs the team's previous match (any coach — the first
    # match of a spell is compared with the predecessor's last XI)
    tm = tm.sort_values(['teamId', 'date', 'matchId']).reset_index(drop=True)
    prev_xi = tm.groupby('teamId')['xi'].shift()
    prev_season = tm.groupby('teamId')['seasonId'].shift()
    def _changes(cur, prev):
        if not isinstance(cur, (list, np.ndarray)) or not isinstance(prev, (list, np.ndarray)) \
                or len(cur) != 11 or len(prev) != 11:
            return np.nan
        return float(len(set(cur) - set(prev)))
    tm['xi_changes'] = [_changes(c, p) if s == ps else np.nan
                        for c, p, s, ps in zip(tm.xi, prev_xi, tm.seasonId, prev_season)]
    tm['team'] = tm.teamId.map(id_to_name).fillna(tm.team)
    tm['xi'] = tm.xi.apply(lambda v: list(v) if isinstance(v, (list, np.ndarray)) else [])
    tm['sub_in_ids'] = tm.sub_in_ids.apply(lambda v: list(v) if isinstance(v, (list, np.ndarray)) else [])
    tm['bench'] = tm.bench.apply(lambda v: list(v) if isinstance(v, (list, np.ndarray)) else [])
    tm = add_youth_minutes(tm)
    tm.to_parquet(TEAM_MATCH, index=False)
    return tm


def build_coach_player(tm):
    """Exact minutes and on-ball value per coach × team × player, next to the
    player's own record in matches played BEFORE his first match under that
    coach (never later ones)."""
    pm = player_team_minutes(tm)
    pm = pm.merge(tm[['matchId', 'teamId', 'coachId', 'spell', 'date', 'team', 'points', 'xgd']],
                  on=['matchId', 'teamId'], how='inner')
    pm = pm.sort_values(['playerId', 'date', 'matchId'])
    # the baseline counts only matches with an OBV row, minutes and value alike
    pm['obv_minutes'] = pm.minutes.where(pm.has_obv, 0)
    pm['cum_min_before'] = pm.groupby('playerId')['obv_minutes'].cumsum() - pm.obv_minutes
    pm['cum_obv_before'] = pm.groupby('playerId')['obv'].cumsum() - pm.obv
    pm['cum_all_min_before'] = pm.groupby('playerId')['minutes'].cumsum() - pm.minutes
    known = pm[pm.coachId > 0].copy()
    # first and second half of his minutes under the coach (by date): does he
    # get better WHILE playing for him? Needs no earlier record.
    grp = known.groupby(['coachId', 'playerId'])['obv_minutes']
    early = (grp.cumsum() - known.obv_minutes) < grp.transform('sum') / 2
    known['early_minutes'] = known.obv_minutes.where(early, 0)
    known['early_obv'] = known.obv.where(early, 0)
    g = known.groupby(['coachId', 'teamId', 'playerId'])
    out = g.agg(team=('team', 'first'), first_date=('date', 'min'), last_date=('date', 'max'),
                matches=('matchId', 'nunique'), starts=('started', 'sum'),
                minutes=('minutes', 'sum'), obv=('obv', 'sum'),
                obv_minutes=('obv_minutes', 'sum'),
                early_minutes=('early_minutes', 'sum'), early_obv=('early_obv', 'sum'),
                points=('points', 'sum')).reset_index()
    # the player's record before his FIRST match under this coach at any club
    first = known.groupby(['coachId', 'playerId']).agg(
        prior_minutes=('cum_min_before', 'first'), prior_obv=('cum_obv_before', 'first'),
        prior_all_minutes=('cum_all_min_before', 'first'),
        first_under_coach=('date', 'min')).reset_index()
    out = out.merge(first, on=['coachId', 'playerId'], how='left')
    # age on the day of his first match under this coach (any club)
    out = out.merge(_birth_dates(), on='playerId', how='left')
    out['age_first'] = (out.first_under_coach - out.birthDate).dt.days / 365.25
    out = out.drop(columns=['birthDate'])
    # times named in the coach's matchday squad (started or on the bench)
    named = pd.concat([tm[['coachId', 'teamId']].assign(playerId=tm.xi).explode('playerId'),
                       tm[['coachId', 'teamId']].assign(playerId=tm.bench).explode('playerId')]).dropna()
    named['playerId'] = named.playerId.astype('int64')
    out = out.merge(named.groupby(['coachId', 'teamId', 'playerId']).size().rename('named').reset_index(),
                    on=['coachId', 'teamId', 'playerId'], how='left')
    # of those sheets, the ones he actually played in (same matches top and bottom)
    sheet_keys = pd.concat([tm[['matchId', 'teamId']].assign(playerId=tm.xi).explode('playerId'),
                            tm[['matchId', 'teamId']].assign(playerId=tm.bench).explode('playerId')]).dropna()
    sheet_keys['playerId'] = sheet_keys.playerId.astype('int64')
    played = known.merge(sheet_keys.drop_duplicates(), on=['matchId', 'teamId', 'playerId'])
    out = out.merge(played.groupby(['coachId', 'teamId', 'playerId']).size().rename('named_played').reset_index(),
                    on=['coachId', 'teamId', 'playerId'], how='left')
    out['named_played'] = out.named_played.fillna(0)
    # team matches the coach took between the player's first and last match
    # under him — the denominator for 'share of matches played'
    span = tm[tm.coachId > 0][['coachId', 'teamId', 'date']]
    m = out[['coachId', 'teamId', 'playerId', 'first_date', 'last_date']].merge(span, on=['coachId', 'teamId'])
    m = m[(m.date >= m.first_date) & (m.date <= m.last_date)]
    out = out.merge(m.groupby(['coachId', 'teamId', 'playerId']).size().rename('team_matches_in_span').reset_index(),
                    on=['coachId', 'teamId', 'playerId'], how='left')
    out.to_parquet(COACH_PLAYER, index=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-api', action='store_true')
    args = ap.parse_args()
    ms = pd.read_parquet(ROOT / 'matches_summary.parquet', columns=['matchId', 'status'])
    ids = set(ms.loc[ms.status == 'Played', 'matchId'].astype(int))
    print('coach per match …')
    cm = build_coach_match(ids, use_api=not args.no_api)
    cm = cm[cm.matchId.isin(ids)]
    print(f'  {cm.matchId.nunique()} of {len(ids)} matches, '
          f'{(cm.coachId > 0).mean():.1%} of team-matches with a coach, '
          f'{cm.loc[cm.coachId > 0, "coachId"].nunique()} coaches')
    print('coach details …')
    co = build_coaches(set(cm.coachId), use_api=not args.no_api)
    print(f'  {int(co.resolved.sum())} of {len(co)} named')
    print('team-match table …')
    tm = build_team_match(cm)
    print(f'  {len(tm)} team-matches, {tm.has_events.mean():.1%} with events, '
          f'{tm.state_ok.mean():.1%} with a clean goal timeline')
    print('coach × player …')
    cp = build_coach_player(tm)
    print(f'  {len(cp)} rows')


if __name__ == '__main__':
    main()
