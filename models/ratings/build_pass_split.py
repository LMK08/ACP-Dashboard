#!/usr/bin/env python3
"""Futi adoption #2: split Passing Value into CREATING vs LINKING.

Creating = open-play passes into dangerous areas (futi: "final third
passes, attacking phase passes"): tagged pass_to_final_third /
pass_to_penalty_area / cross / through_pass / shot_assist, or ending in
the final third (x >= 66.7).
Linking = all other open-play passes.

Set-piece deliveries are excluded (they live in the separate set-piece
score). Also writes player_teams.parquet (modal team per player-season)
for the team-switch stability test (futi adoption #4).

Outputs: models/ratings/pass_split.parquet, player_teams.parquet
Run from the Dashboard dir: python models/ratings/build_pass_split.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_DASH = _HERE.parent.parent
_GPA_DATA = _DASH.parent.parent / 'GPA Model Project v2' / 'parquet_data'

print("[1/3] join pass values with pass context…", flush=True)
av = pd.read_parquet(_GPA_DATA / 'events_with_action_values.parquet',
                       columns=['matchId', 'id', 'seasonId', 'player.id',
                                 'type.primary', 'action_value'])
av = av[(av['type.primary'] == 'pass') & av['action_value'].notna()
          & av['player.id'].notna()]
av['player.id'] = av['player.id'].astype(int)
raw = pd.concat([pd.read_parquet(_GPA_DATA / f'{f}.parquet',
    columns=['matchId', 'id', 'type.secondary', 'pass.endLocation.x',
              'team.name', 'player.id', 'seasonId'])
    for f in ['liga3_portugal_events', 'campeonato_portugal_events']],
    ignore_index=True)
sec = raw['type.secondary'].apply(
    lambda x: set(x) if isinstance(x, (list, np.ndarray)) else set())
SP = {'corner', 'free_kick', 'throw_in', 'goal_kick', 'penalty',
       'free_kick_cross'}
CRE = {'pass_to_final_third', 'pass_to_penalty_area', 'cross',
        'through_pass', 'shot_assist'}
raw['is_sp'] = sec.apply(lambda s: bool(s & SP))
raw['is_cre_tag'] = sec.apply(lambda s: bool(s & CRE))
m = av.merge(raw[['matchId', 'id', 'is_sp', 'is_cre_tag',
                    'pass.endLocation.x']], on=['matchId', 'id'], how='left')
m = m[m['is_sp'] != True]
m['creating'] = m['is_cre_tag'] | (m['pass.endLocation.x'] >= 66.7)
print(f"  open-play passes with value: {len(m):,} "
       f"({m['creating'].mean()*100:.0f}% creating)")

print("[2/3] aggregate + sanity…", flush=True)
m['cv'] = np.where(m['creating'], m['action_value'], 0.0)
m['lv'] = np.where(~m['creating'], m['action_value'], 0.0)
out = (m.groupby(['player.id', 'seasonId'])[['cv', 'lv']].sum()
         .reset_index())
out.columns = ['playerId', 'seasonId', 'Creating Value', 'Linking Value']
g = pd.read_parquet(_DASH / 'gpa_player_season_values.parquet',
                      columns=['playerId', 'seasonId', 'Passing Value'])
g['playerId'] = pd.to_numeric(g['playerId'], errors='coerce').astype('Int64')
g['seasonId'] = pd.to_numeric(g['seasonId'], errors='coerce').astype('Int64')
chk = out.merge(g, on=['playerId', 'seasonId'])
ratio = ((chk['Creating Value'] + chk['Linking Value']).sum()
           / chk['Passing Value'].sum())
print(f"  sum(Creating+Linking) / sum(Passing Value) = {ratio:.3f} "
       f"(approx 1.0 expected; tag-based SP exclusion differs slightly)")
out.to_parquet(_HERE / 'pass_split.parquet')

print("[3/3] player teams (for team-switch stability test)…", flush=True)
pt = (raw.dropna(subset=['player.id', 'team.name'])
         .assign(pid=lambda d: d['player.id'].astype(int))
         .groupby(['pid', 'seasonId'])['team.name']
         .agg(lambda x: x.mode().iloc[0]).reset_index())
pt.columns = ['playerId', 'seasonId', 'team']
pt.to_parquet(_HERE / 'player_teams.parquet')
print(f"  saved pass_split.parquet ({len(out):,}) + player_teams.parquet ({len(pt):,})")
