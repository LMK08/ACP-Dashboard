#!/usr/bin/env python3
"""Value-gate consistency check: do the ACP Rating's own components link
to team outcomes (the same test that excluded territorial metrics)?

Method: minutes-weighted team-season means of each rating component,
z-scored within (league, season), correlated with team xGF/xGA/xGD per
90 (from shot xG). Run from the Dashboard dir.

Results 2026-06-11 (220 team-seasons):
  component       xGF/90  xGA/90  xGD/90
  off_pct          +0.42   -0.03   +0.26   creation -> creation
  defr_pct         -0.01   -0.06   +0.04   STRUCTURALLY BLIND TEST: the
      expectation model nets out team context, so team means of
      above-expectation surpluses carry no team signal by construction.
  qual_pct         +0.19   -0.38   +0.35   duel quality -> concede less
  datt_pct         +0.38   -0.21   +0.34   ball security -> both ends
  rapm_pct         +0.46   -0.53   +0.61   PARTLY CIRCULAR (fit on xGD)
  setpiece_pct     -0.14   +0.14   -0.17   NEGATIVE: set-piece-reliant
      squads are weaker overall -> separation from rating vindicated
  acp_rating       +0.53   -0.37   +0.53   the rating passes its gate
"""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import pearsonr

_HERE = Path(__file__).resolve().parent
_DASH = _HERE.parent.parent
_GPA = _DASH.parent.parent / 'GPA Model Project v2' / 'parquet_data'
ev = pd.concat([pd.read_parquet(_GPA / f'{f}.parquet',
    columns=['matchId','seasonId','team.name','shot.xg','player.id'])
    for f in ['liga3_portugal_events','campeonato_portugal_events']], ignore_index=True)
sh = ev.dropna(subset=['shot.xg'])
xg = sh.groupby(['matchId','team.name'])['shot.xg'].sum().reset_index()
sn = ev.dropna(subset=['team.name']).groupby('matchId')['seasonId'].first()
rows=[]
for mid, g in xg.groupby('matchId'):
    if len(g)!=2: continue
    a,b = g.iloc[0], g.iloc[1]
    rows.append({'team':a['team.name'],'seasonId':sn[mid],'xgf':a['shot.xg'],'xga':b['shot.xg'],'m':1})
    rows.append({'team':b['team.name'],'seasonId':sn[mid],'xgf':b['shot.xg'],'xga':a['shot.xg'],'m':1})
tg = pd.DataFrame(rows).groupby(['team','seasonId']).sum().reset_index()
for c in ['xgf','xga']: tg[c+'90'] = tg[c]/tg['m']
tg['xgd90'] = tg['xgf90']-tg['xga90']
evp = ev.dropna(subset=['player.id','team.name']).copy()
evp['player.id']=evp['player.id'].astype(int)
ptm = evp.groupby(['player.id','seasonId'])['team.name'].agg(lambda x: x.mode().iloc[0]).reset_index()
ptm.columns=['playerId','seasonId','team']
o = pd.read_parquet(_HERE / 'acp_rating_per_player_season.parquet').merge(ptm, on=['playerId','seasonId'])
COMPS = ['off_pct','defr_pct','qual_pct','datt_pct','rapm_pct','setpiece_pct','acp_rating']
agg = o.groupby(['team','seasonId']).apply(
    lambda g: pd.Series({c: np.average(g[c], weights=g['mins_played']) for c in COMPS}
                          | {'lg': g['league'].iloc[0]}), include_groups=False).reset_index()
M = agg.merge(tg, on=['team','seasonId'])
for c in COMPS+['xgf90','xga90','xgd90']:
    M[c+'_z'] = M.groupby(['lg','seasonId'])[c].transform(lambda x:(x-x.mean())/x.std())
M = M.dropna()
print(f"team-seasons: {len(M)}")
print(f"{'component':<14}{'xGF/90':>8}{'xGA/90':>8}{'xGD/90':>8}")
for c in COMPS:
    print(f"{c:<14}{pearsonr(M[c+'_z'],M['xgf90_z'])[0]:>+8.2f}"
          f"{pearsonr(M[c+'_z'],M['xga90_z'])[0]:>+8.2f}"
          f"{pearsonr(M[c+'_z'],M['xgd90_z'])[0]:>+8.2f}")
