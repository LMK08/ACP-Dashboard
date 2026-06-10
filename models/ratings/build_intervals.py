#!/usr/bin/env python3
"""True on-pitch intervals per (match, player) from Wyscout lineup data.

Source: match_lineups.pkl — per match × team: lineup (starters), bench,
substitutions [{minute, playerIn, playerOut}], and red-card minutes in
the lineup/bench card fields (Wyscout stores the MINUTE as a string in
'redCards'; '0' = none).

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
match_end = ev.groupby('matchId')['u'].max().to_dict()
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
    end_u = max(float(end_u), 90.0)
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
       f"({n_no_end} matches had no events)")
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
iv.loc[unres, 'teamId'] = [fixes.get((m, t)) for m, t in
                             zip(iv.loc[unres, 'matchId'], iv.loc[unres, 'teamName'])]
print(f"  teamId after player-majority fallback: {iv['teamId'].notna().mean()*100:.1f}%")
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
