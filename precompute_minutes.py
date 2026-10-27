"""
Build complete_player_minutes.pkl ({season_id: DataFrame}) from EXACT
Wyscout API minutes only — see match_minutes.py.

Only matches that have events in raw_events.parquet count: these minutes
are the per-90 denominator for event-derived stats.

RULE (Lucas, 2026-10-20): minutes are never estimated. Until 2026-10 this
file merged three sources (season API, lineup arithmetic, event-timestamp
estimates); the estimates produced e.g. Alisson Calegari 93' instead of the
API's 357'. Now every row is the sum of Wyscout's per-match
`minutesOnField`; a player without API minutes has no row.

PLAYER_ID_ALIASES are resolved here at MATCH level: a duplicate record and
its canonical record in the SAME match are one appearance (max, not sum);
different matches add up. So the rows carry canonical pids only.

Identity columns (name / team / position) are labels, not minutes: season
API row first, then the season's events, then player_details.
"""
import pickle

import pandas as pd

from league_config import PLAYER_ID_ALIASES, all_season_ids, competition_for_season
from match_minutes import load_cache, matches_without_stats, per_match_seasons

ALL_SEASON_IDS = all_season_ids()
OUT_COLUMNS = ['playerId', 'playerName', 'teamName', 'primaryPosition',
               'totalMinutes', 'competitionId']


def resolve_aliases(match_minutes):
    """Map duplicate pids to canonical and collapse same-match duplicates."""
    mm = match_minutes.copy()
    mm['playerId'] = mm['playerId'].map(lambda p: PLAYER_ID_ALIASES.get(int(p), int(p)))
    mm = mm.sort_values('minutes', ascending=False)
    return mm.drop_duplicates(['matchId', 'playerId'], keep='first')


def _position_minutes(mm):
    """Per (playerId, position code) minutes from the API's per-match
    position percentages."""
    rows = []
    for pid, mins, pos in zip(mm['playerId'], mm['minutes'], mm['positions']):
        for part in str(pos or '').split(','):
            if ':' not in part:
                continue
            code, pct = part.split(':', 1)
            try:
                rows.append((pid, code.upper(), mins * int(pct) / 100.0))
            except ValueError:
                continue
    return pd.DataFrame(rows, columns=['playerId', 'position', 'minutes'])


def _event_identity(season_id):
    """{pid: (name, team, position)} from the season's events (labels only)."""
    try:
        ev = pd.read_parquet('raw_events.parquet',
                             columns=['player.id', 'player.name', 'team.name', 'player.position'],
                             filters=[('seasonId', '==', season_id)])
    except Exception:
        return {}
    ev = ev.dropna(subset=['player.id'])
    if ev.empty:
        return {}
    ev['player.id'] = pd.to_numeric(ev['player.id'], errors='coerce')
    ev = ev.dropna(subset=['player.id'])
    ev['player.id'] = ev['player.id'].astype(int).map(lambda p: PLAYER_ID_ALIASES.get(p, p))

    def _mode(s):
        s = s.dropna().astype(str)
        return s.mode().iloc[0] if len(s) else None

    g = ev.groupby('player.id')
    return {pid: (n, t, p) for pid, n, t, p in zip(
        g.size().index, g['player.name'].agg(_mode), g['team.name'].agg(_mode),
        g['player.position'].agg(_mode))}


_NO_POSITION = ('', 'Unknown', 'nan', 'None', '0')


def combine_season_minutes(dfs):
    """One row per player over several seasons' minutes frames: minutes
    summed, name / team from the first frame he appears in (the callers pass
    the newest season first within a league; the both-leagues scope lists
    Liga 3 before the Campeonato), and primaryPosition the position he played
    MOST — each season row's minutes credited to that season's
    primaryPosition (itself his most-played position that season, exact API
    minutes), ties to the earlier frame. Until 2026-10 the position was
    'first' like the labels — the newest season's, however few minutes he
    had there: a career striker with 194' of LAMF this season was ranked
    with the wingers in All Seasons (232 of the 1,380 All-Seasons players
    with 500'+ sat in another template group)."""
    combined = pd.concat(dfs, ignore_index=True)
    out = combined.groupby('playerId').agg({
        'playerName': 'first', 'teamName': 'first',
        'primaryPosition': 'first', 'totalMinutes': 'sum',
    }).reset_index()
    pos = combined['primaryPosition']
    vote = combined.assign(
        _mins=pd.to_numeric(combined['totalMinutes'], errors='coerce').fillna(0),
        _order=range(len(combined)),
    )[pos.notna() & ~pos.astype(str).isin(_NO_POSITION)]
    best = (vote.groupby(['playerId', 'primaryPosition'])
                .agg(_mins=('_mins', 'sum'), _order=('_order', 'min')).reset_index()
                .sort_values(['_mins', '_order'], ascending=[False, True])
                .drop_duplicates('playerId').set_index('playerId')['primaryPosition'])
    # a player with no real position anywhere keeps the old label
    out['primaryPosition'] = out['playerId'].map(best).fillna(out['primaryPosition'])
    return out


def build_season(season_id, mm_season, api_season, details, lineup_teams):
    if mm_season.empty:
        return pd.DataFrame(columns=OUT_COLUMNS)
    totals = mm_season.groupby('playerId')['minutes'].sum()
    pos = _position_minutes(mm_season)
    top_pos = (pos.groupby(['playerId', 'position'])['minutes'].sum()
               .reset_index().sort_values('minutes', ascending=False)
               .drop_duplicates('playerId').set_index('playerId')['position'])

    api_id = {}
    if isinstance(api_season, pd.DataFrame) and 'playerId' in api_season.columns:
        a = api_season.copy()
        a['playerId'] = a['playerId'].astype(int).map(lambda p: PLAYER_ID_ALIASES.get(p, p))
        for r in a.itertuples():
            api_id.setdefault(r.playerId, (r.playerName, r.teamName, r.primaryPosition))
    ev_id = _event_identity(season_id)

    def _ok(v):
        return v is not None and str(v) not in ('', 'Unknown', 'nan', 'None')

    rows = []
    for pid, mins in totals.items():
        a = api_id.get(pid, (None, None, None))
        e = ev_id.get(pid, (None, None, None))
        d = details.get(pid, {})
        # Labels: the season's own events first (they follow mid-season
        # transfers; the stored season-API frame is no longer refreshed for
        # live seasons), then the API frame, then the bio. Position: the
        # API's per-match position shares first.
        name = next((v for v in (e[0], a[0], d.get('shortName')) if _ok(v)), 'Unknown')
        team = next((v for v in (e[1], a[1], lineup_teams.get(pid)) if _ok(v)), 'Unknown')
        position = next((v for v in (top_pos.get(pid), a[2], e[2]) if _ok(v)), 'Unknown')
        rows.append({'playerId': int(pid), 'playerName': name, 'teamName': team,
                     'primaryPosition': position, 'totalMinutes': int(mins),
                     'competitionId': competition_for_season(season_id)})
    return pd.DataFrame(rows, columns=OUT_COLUMNS)


def _season_total_rows(season_id, api_season):
    """Stored exact season-level API rows, aliases resolved (a duplicate and
    its canonical record are one stint: keep the larger)."""
    if not isinstance(api_season, pd.DataFrame) or api_season.empty:
        return pd.DataFrame(columns=OUT_COLUMNS)
    a = api_season.copy()
    a['playerId'] = a['playerId'].astype(int).map(lambda p: PLAYER_ID_ALIASES.get(p, p))
    a = a[pd.to_numeric(a['totalMinutes'], errors='coerce') > 0]
    a = a.sort_values('totalMinutes', ascending=False).drop_duplicates('playerId')
    a['competitionId'] = competition_for_season(season_id)
    a['totalMinutes'] = a['totalMinutes'].astype(int)
    return a[OUT_COLUMNS].reset_index(drop=True)


def _event_match_ids(season_id):
    """matchIds with events in raw_events.parquet for one season."""
    try:
        ev = pd.read_parquet('raw_events.parquet', columns=['matchId'],
                             filters=[('seasonId', '==', season_id)])
    except Exception:
        return set()
    return set(pd.to_numeric(ev['matchId'], errors='coerce').dropna().astype('int64').unique())


def _lineup_teams(season_match_ids):
    """{pid: team name} from match_lineups.pkl (labels only)."""
    try:
        with open('match_lineups.pkl', 'rb') as f:
            lineups = pickle.load(f)
    except Exception:
        return {}
    out = {}
    for mid in season_match_ids:
        for team, sides in (lineups.get(int(mid)) or {}).items():
            for key in ('lineup', 'bench'):
                for p in sides.get(key) or []:
                    pid = p.get('playerId') if isinstance(p, dict) else None
                    if pid:
                        out[PLAYER_ID_ALIASES.get(int(pid), int(pid))] = team
    return out


def main():
    mm = load_cache()
    if mm.empty:
        raise SystemExit('match_player_minutes.parquet is empty — run match_minutes.py first')
    mm = resolve_aliases(mm)
    with open('player_minutes_and_positions.pkl', 'rb') as f:
        api_minutes = pickle.load(f)
    with open('player_details.pkl', 'rb') as f:
        details = {p['playerId']: p for p in pickle.load(f) if isinstance(p, dict)}
    matches = pd.read_parquet('matches_summary.parquet', columns=['matchId', 'seasonId'])

    no_stats_all = matches_without_stats()
    pms = per_match_seasons(matches_summary=matches)
    complete = {}
    for sid in ALL_SEASON_IDS:
        if sid not in pms:
            # No per-match data for this season (Campeonato: Wyscout access
            # lost 2026-10): the exact SEASON totals stored while access
            # existed. Never estimated — players without one get no row.
            df = _season_total_rows(sid, api_minutes.get(sid))
            complete[sid] = df
            print(f'{sid}: {len(df)} players, {int(df["totalMinutes"].sum()) if len(df) else 0} min '
                  '(stored exact season totals — no per-match API data)')
            continue
        season_mids = matches.loc[matches['seasonId'] == sid, 'matchId'].dropna().astype(int)
        mm_s = mm[mm['seasonId'] == sid]
        no_stats = set(season_mids) & no_stats_all
        missing = set(season_mids) - set(mm_s['matchId']) - no_stats
        # Per-90 denominator = minutes from the matches the stats come from:
        # matches with no events in raw_events (57 eventless Camp 25/26
        # matches) carry no numerator, so their minutes are left out.
        ev_mids = _event_match_ids(sid)
        no_events = set(mm_s['matchId']) - ev_mids
        mm_s = mm_s[mm_s['matchId'].isin(ev_mids)]
        df = build_season(sid, mm_s, api_minutes.get(sid), details, _lineup_teams(season_mids))
        complete[sid] = df
        print(f'{sid}: {len(df)} players, {int(df["totalMinutes"].sum())} min '
              f'from {mm_s["matchId"].nunique()} matches'
              + (f' — {len(no_stats)} matches have NO Wyscout stats (those '
                 f'appearances carry no minutes)' if no_stats else '')
              + (f' — {len(no_events)} matches without events excluded' if no_events else '')
              + (f' — {len(missing)} played matches NOT fetched yet' if missing else ''))

    with open('complete_player_minutes.pkl', 'wb') as f:
        pickle.dump(complete, f)
    print(f'Saved complete_player_minutes.pkl with {len(complete)} seasons')

    all_dfs = [d for d in complete.values() if not d.empty]
    if all_dfs:
        all_time = combine_season_minutes(all_dfs)
        all_time.to_pickle('all_time_player_minutes.pkl')
        print(f'Saved all_time_player_minutes.pkl with {len(all_time)} players')


if __name__ == '__main__':
    main()
