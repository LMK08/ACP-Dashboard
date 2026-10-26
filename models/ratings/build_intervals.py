#!/usr/bin/env python3
"""True on-pitch intervals per (match, player) from Wyscout lineup data.

Source: match_lineups.pkl — per match × team: lineup (starters), bench,
substitutions [{minute, playerIn, playerOut}], and red-card minutes in
the lineup/bench card fields (Wyscout stores the MINUTE as a string in
'redCards'; '0' = none).

Full time: the longest API minutesOnField in the match (exact).

Timeline: cumulative match minutes (Wyscout sub minutes and event
`minute` are both cumulative), so intervals and events are directly
comparable without period offsets.

Output: models/ratings/player_match_intervals.parquet
  (matchId, teamName, teamId, playerId, u_in, u_out, how_in, how_out)

Run from the Dashboard dir: python models/ratings/build_intervals.py
"""
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_DASH = _HERE.parent.parent
_GPA_DATA = _DASH.parent.parent / 'GPA Model Project v2' / 'parquet_data'

print("[1/3] sources…", flush=True)
with open(_DASH / 'match_lineups.pkl', 'rb') as f:
    ml = pickle.load(f)
ev = pd.concat([pd.read_parquet(_GPA_DATA / f'{f}.parquet',
                  columns=['matchId', 'team.id', 'team.name', 'minute', 'second'])
                  for f in ['liga3_portugal_events', 'campeonato_portugal_events']],
                 ignore_index=True)
ev['u'] = (pd.to_numeric(ev['minute'], errors='coerce').fillna(0)
             + pd.to_numeric(ev['second'], errors='coerce').fillna(0) / 60)
# Full time = the longest Wyscout minutesOnField in the match (a starter who
# played it all; includes stoppage time) — exact, never the last event
# (Lucas 2026-10-20: minutes are never estimated). A match without API
# minutes has no intervals.
import sys
sys.path.insert(0, str(_DASH))
from match_minutes import load_cache  # noqa: E402
_mm = load_cache()
match_end = _mm.groupby('matchId')['minutes'].max().astype(float).to_dict()
# Exception (Lucas 2026-10-20): seasons with NO per-match API data at all
# (Campeonato — Wyscout access lost 2026-10) take full time from the match's
# last event, as before, so Camp RAPM survives. It only places the end of the
# final on-pitch segment; no player's minutes are derived from it.
_ms = pd.read_parquet(_DASH / 'matches_summary.parquet', columns=['matchId', 'seasonId']).dropna()
from match_minutes import per_match_seasons  # noqa: E402
_pm_seasons = per_match_seasons(_DASH / 'match_player_minutes.parquet', _ms)
_ev_end = ev.groupby('matchId')['u'].max()
for _mid, _sid in zip(_ms['matchId'].astype('int64'), _ms['seasonId'].astype('int64')):
    if _sid not in _pm_seasons and _mid in _ev_end.index:
        match_end[_mid] = max(float(_ev_end[_mid]), 90.0)
team_id = (ev[['matchId', 'team.name', 'team.id']].drop_duplicates()
              .set_index(['matchId', 'team.name'])['team.id'].to_dict())

print("[2/3] intervals…", flush=True)
rows = []
n_no_end = 0
for mid, teams in ml.items():
    end_u = match_end.get(mid)
    if end_u is None:
        n_no_end += 1
        continue
    end_u = float(end_u)
    for tname, td in (teams or {}).items():
        if not isinstance(td, dict):
            continue
        lineup = td.get('lineup') or []
        bench = td.get('bench') or []
        subs = td.get('substitutions') or []
        red_min = {}
        for p in list(lineup) + list(bench):
            rc = str(p.get('redCards', '0') or '0')
            try:
                rcv = float(rc)
            except ValueError:
                rcv = 0.0
            if rcv > 0:
                red_min[p['playerId']] = rcv
        in_t, how_in = {}, {}
        for p in lineup:
            in_t[p['playerId']] = 0.0
            how_in[p['playerId']] = 'start'
        out_t, how_out = {}, {}
        for s in subs:
            try:
                m = float(s['minute'])
            except (TypeError, ValueError):
                continue
            if s.get('playerOut') is not None:
                out_t[s['playerOut']] = m
                how_out[s['playerOut']] = 'sub_off'
            if s.get('playerIn') is not None:
                in_t[s['playerIn']] = m
                how_in[s['playerIn']] = 'sub_on'
        tid = team_id.get((mid, tname))
        for pid, t0 in in_t.items():
            t1 = out_t.get(pid, end_u)
            ho = how_out.get(pid, 'end')
            if pid in red_min and red_min[pid] < t1:
                t1 = red_min[pid]
                ho = 'red'
            if t1 > t0:
                rows.append({'matchId': mid, 'teamName': tname, 'teamId': tid,
                              'playerId': int(pid), 'u_in': t0, 'u_out': t1,
                              'how_in': how_in[pid], 'how_out': ho})

iv = pd.DataFrame(rows)
print(f"  {len(iv):,} intervals across {iv['matchId'].nunique():,} matches "
       f"({n_no_end} lineup matches have no Wyscout minutes — skipped)")
print(f"  teamId resolved by name: {iv['teamId'].notna().mean()*100:.1f}%")

# Fallback for name drift between lineups and events: assign the
# unresolved (match, teamName) groups the MAJORITY event-team of their
# own lineup players in that match.
ev_pt = pd.read_parquet(
    _GPA_DATA / 'liga3_portugal_events.parquet',
    columns=['matchId', 'player.id', 'team.id'])
ev_pt = pd.concat([ev_pt, pd.read_parquet(
    _GPA_DATA / 'campeonato_portugal_events.parquet',
    columns=['matchId', 'player.id', 'team.id'])], ignore_index=True)
ev_pt = ev_pt.dropna().drop_duplicates()
ev_pt['player.id'] = ev_pt['player.id'].astype(int)
pmap = ev_pt.set_index(['matchId', 'player.id'])['team.id'].to_dict()
unres = iv['teamId'].isna()
fixes = {}
for (mid, tname), gg in iv[unres].groupby(['matchId', 'teamName']):
    tids = [pmap.get((mid, p)) for p in gg['playerId']]
    tids = [t for t in tids if t is not None]
    if tids:
        fixes[(mid, tname)] = pd.Series(tids).mode().iloc[0]
# numeric, NaN for groups with no fix: a raw list mixing ids and None
# becomes an object array, which pandas 3 (the CI engine env) refuses to
# write into the float teamId column (rebuild failure 2026-10-26)
iv.loc[unres, 'teamId'] = pd.to_numeric(pd.Series(
    [fixes.get((m, t)) for m, t in zip(iv.loc[unres, 'matchId'], iv.loc[unres, 'teamName'])],
    dtype=object), errors='coerce').to_numpy(dtype=float)
print(f"  teamId after player-majority fallback: {iv['teamId'].notna().mean()*100:.1f}%")
# Still unresolved = a match whose lineup exists but whose events have not
# arrived yet (Wyscout lag, e.g. one MW7 match on 2026-10-26): no events, no
# xG segments, nothing for RAPM to use — and rapm_v4 casts teamId to int.
_nores = iv['teamId'].isna()
if _nores.any():
    print(f"  dropping {int(_nores.sum())} intervals in {iv.loc[_nores, 'matchId'].nunique()} "
          f"match(es) with no team id (no events yet)")
    iv = iv[~_nores].copy()
print(f"  how_out mix: {iv['how_out'].value_counts(normalize=True).round(3).to_dict()}")

print("[3/3] sanity vs event spans…", flush=True)
iv['mins'] = iv['u_out'] - iv['u_in']
per_match = iv.groupby(['matchId', 'teamName'])['playerId'].count()
print(f"  players per team-match: median {per_match.median():.0f} "
       f"(11 + subs expected: 14-16)")
print(f"  interval minutes: median {iv['mins'].median():.0f}, "
       f"starters {iv[iv['how_in']=='start']['mins'].median():.0f}, "
       f"subs {iv[iv['how_in']=='sub_on']['mins'].median():.0f}")
iv.to_parquet(_HERE / 'player_match_intervals.parquet')
print("saved player_match_intervals.parquet")
