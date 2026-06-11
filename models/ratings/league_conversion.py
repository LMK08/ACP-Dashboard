#!/usr/bin/env python3
"""League conversion: how many ACP rating points is Campeonato worth
relative to Liga 3? Triangulated from independent anchors.

  A1 MOVERS    — same player, consecutive seasons, league switch. The
                  rating change beyond age expectation, averaged across
                  BOTH directions (cancels mean selection).
  A2 TEAMS     — same club appearing in both leagues (promotion/releg):
                  xGD/90 shift, converted to rating points via the
                  within-league slope of team xGD on team mean rating.
  A3 MARKET    — Transfermarkt values are one cross-league scale:
                  log(MV) ~ rating + age + league. The league
                  coefficient divided by the rating coefficient = the
                  market's view of the gap in rating points.
  (A4 duel-ladder mover residuals measured earlier: L3 > Camp by
   ~10-25 Glicko pts, direction-consistent — cited as corroboration.)

Output: league_conversion.json {delta_pts: Camp rating - delta = L3-
equivalent}, and acp_rating_abs / projection_abs columns appended to
the rating and projection parquets (L3 = reference scale).

Run from the Dashboard dir: python models/ratings/league_conversion.py
"""
from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_DASH = _HERE.parent.parent
_GPA_DATA = _DASH.parent.parent / 'GPA Model Project v2' / 'parquet_data'
SEASON_YR = {188221: 2021, 188222: 2022, 189147: 2023, 190090: 2024,
              191782: 2025, 190230: 2023, 191779: 2025}

o = pd.read_parquet(_HERE / 'acp_rating_per_player_season.parquet')
o['yr'] = o['seasonId'].map(SEASON_YR)

# age expectation (same curve construction as build_projection, condensed)
with open(_DASH / 'player_details.pkl', 'rb') as f:
    det = pd.DataFrame(pickle.load(f))
det['birthDate'] = pd.to_datetime(det['birthDate'], errors='coerce')
bd = (det.dropna(subset=['birthDate']).drop_duplicates('playerId')
         .set_index('playerId')['birthDate'])
o['age'] = o['yr'] + 0.5 - (o['playerId'].map(bd).dt.year
                              + o['playerId'].map(bd).dt.dayofyear / 365)
AGE_BINS = [(15, 20.5), (20.5, 23.5), (23.5, 26.5), (26.5, 29.5), (29.5, 42)]
PRIOR = {0: 1.2, 1: 0.6, 2: 0.0, 3: -0.6, 4: -1.4}


def bin_of(a):
    for i, (lo, hi) in enumerate(AGE_BINS):
        if lo <= a < hi:
            return i
    return 4

# within-league deltas only (league switches excluded -> no circularity)
dl = []
for pid, g in o.dropna(subset=['age']).sort_values('yr').groupby('playerId'):
    rr = g.to_dict('records')
    for a, b in zip(rr, rr[1:]):
        if b['yr'] - a['yr'] == 1 and a['league'] == b['league']:
            dl.append((bin_of(a['age']), b['acp_rating'] - a['acp_rating']))
DL = pd.DataFrame(dl, columns=['ab', 'd'])
AGE_D = {b: (DL[DL['ab'] == b]['d'].sum() + PRIOR[b] * 60)
            / (len(DL[DL['ab'] == b]) + 60) for b in range(5)}

print("=== A1: MOVERS (rating change beyond age expectation) ===")
mv = []
for pid, g in o.dropna(subset=['age']).sort_values('yr').groupby('playerId'):
    rr = g.to_dict('records')
    for a, b in zip(rr, rr[1:]):
        if b['yr'] - a['yr'] == 1 and a['league'] != b['league']:
            resid = (b['acp_rating'] - a['acp_rating']) - AGE_D[bin_of(a['age'])]
            mv.append({'dir': f"{a['league']}->{b['league']}", 'resid': resid,
                         'w': min(a['mins_played'], b['mins_played'])})
MV = pd.DataFrame(mv)
res = {}
for d, g in MV.groupby('dir'):
    m = np.average(g['resid'], weights=g['w'])
    se = g['resid'].std() / np.sqrt(len(g))
    res[d] = (m, se, len(g))
    print(f"  {d}: mean resid {m:+.2f} ±{se:.2f} (n={len(g)})")
# L3 stronger by delta => moving UP costs delta, moving DOWN gains delta
d_up = -res.get('CAMP->L3', (0, 9, 0))[0]
d_dn = res.get('L3->CAMP', (0, 9, 0))[0]
a1 = (d_up + d_dn) / 2
a1_se = np.sqrt(res['CAMP->L3'][1]**2 + res['L3->CAMP'][1]**2) / 2
print(f"  A1 delta = {a1:+.2f} ±{a1_se:.2f} rating pts (L3 minus Camp)")

print("\n=== A2: CROSS-LEAGUE TEAMS (xGD shift -> rating pts) ===")
ev = pd.concat([pd.read_parquet(_GPA_DATA / f'{f}.parquet',
    columns=['matchId', 'seasonId', 'team.name', 'shot.xg', 'player.id'])
    for f in ['liga3_portugal_events', 'campeonato_portugal_events']],
    ignore_index=True)
sh = ev.dropna(subset=['shot.xg'])
xg = sh.groupby(['matchId', 'team.name'])['shot.xg'].sum().reset_index()
sn = ev.dropna(subset=['team.name']).groupby('matchId')['seasonId'].first()
rows = []
for mid, g in xg.groupby('matchId'):
    if len(g) != 2:
        continue
    a, b = g.iloc[0], g.iloc[1]
    rows.append({'team': a['team.name'], 'seasonId': sn[mid],
                   'xgd': a['shot.xg'] - b['shot.xg'], 'm': 1})
    rows.append({'team': b['team.name'], 'seasonId': sn[mid],
                   'xgd': b['shot.xg'] - a['shot.xg'], 'm': 1})
tg = pd.DataFrame(rows).groupby(['team', 'seasonId']).sum().reset_index()
tg['xgd90'] = tg['xgd'] / tg['m']
tg['yr'] = tg['seasonId'].map(SEASON_YR)
tg['lg'] = np.where(tg['seasonId'].isin({190230, 191779}), 'CAMP', 'L3')
# slope: team xGD/90 per team-mean-rating point (within league-season)
evp = ev.dropna(subset=['player.id', 'team.name']).copy()
evp['player.id'] = evp['player.id'].astype(int)
ptm = (evp.groupby(['player.id', 'seasonId'])['team.name']
          .agg(lambda x: x.mode().iloc[0]).reset_index())
ptm.columns = ['playerId', 'seasonId', 'team']
ot = o.merge(ptm, on=['playerId', 'seasonId'])
tr = (ot.groupby(['team', 'seasonId'])
         .apply(lambda g: np.average(g['acp_rating'], weights=g['mins_played']),
                 include_groups=False).rename('mrate').reset_index())
TT = tg.merge(tr, on=['team', 'seasonId'])
TT['xgd_c'] = TT['xgd90'] - TT.groupby(['lg', 'seasonId'])['xgd90'].transform('mean')
TT['mr_c'] = TT['mrate'] - TT.groupby(['lg', 'seasonId'])['mrate'].transform('mean')
slope = (TT['xgd_c'] * TT['mr_c']).sum() / (TT['mr_c'] ** 2).sum()
print(f"  slope: {slope:.4f} xGD/90 per rating pt")
# same club consecutive seasons across leagues
tm_pairs = []
for team, g in tg.sort_values('yr').groupby('team'):
    rr = g.to_dict('records')
    for a, b in zip(rr, rr[1:]):
        if b['yr'] - a['yr'] == 1 and a['lg'] != b['lg']:
            # xGD vs the league it ENTERS: shift = xgd_b - xgd_a
            tm_pairs.append({'dir': f"{a['lg']}->{b['lg']}",
                               'shift': b['xgd90'] - a['xgd90']})
TP = pd.DataFrame(tm_pairs)
print(f"  cross-league club season-pairs: {len(TP)} "
       f"({TP.groupby('dir').size().to_dict() if len(TP) else {}})")
if len(TP) >= 6:
    s_up = -TP[TP['dir'] == 'CAMP->L3']['shift'].mean()
    s_dn = TP[TP['dir'] == 'L3->CAMP']['shift'].mean()
    a2_xgd = np.nanmean([s_up, s_dn])
    a2 = a2_xgd / slope
    a2_se = TP['shift'].std() / np.sqrt(len(TP)) / abs(slope)
    print(f"  A2 delta = {a2:+.2f} ±{a2_se:.2f} rating pts "
           f"(xGD shift {a2_xgd:+.2f} / slope)")
else:
    a2, a2_se = np.nan, np.nan
    print("  A2: too few club pairs — skipped")

print("\n=== A3: MARKET VALUES (Transfermarkt, one scale) ===")
val = pd.read_parquet(_DASH / 'valuations' / 'valuations.parquet')
val = val[val['source'] != 'reported_fee'].dropna(subset=['value_eur'])
val['as_of_date'] = pd.to_datetime(val['as_of_date'], errors='coerce')
val['yr'] = val['as_of_date'].dt.year - (val['as_of_date'].dt.month < 7)
pv = (val.groupby(['playerId', 'yr'])['value_eur'].max().reset_index())
M = o.merge(pv, on=['playerId', 'yr'], how='inner')
M = M[(M['value_eur'] > 0) & M['age'].notna()]
print(f"  player-seasons with TM value + rating: {len(M)} "
       f"({M.groupby('league').size().to_dict()})")
X = pd.DataFrame({'rating': M['acp_rating'], 'age': M['age'],
                    'age2': (M['age'] - 26) ** 2,
                    'camp': (M['league'] == 'CAMP').astype(float)})
X['const'] = 1.0
y = np.log(M['value_eur'])
beta, *_ = np.linalg.lstsq(X.values, y.values, rcond=None)
b = dict(zip(X.columns, beta))
a3 = -b['camp'] / b['rating'] if b['rating'] > 0 else np.nan
# crude SE via residual bootstrap-lite
resid = y.values - X.values @ beta
se_scale = np.sqrt(np.diag(np.linalg.inv(X.T.values @ X.values))
                     * (resid ** 2).mean())
se_map = dict(zip(X.columns, se_scale))
a3_se = abs(a3) * np.sqrt((se_map['camp'] / max(abs(b['camp']), 1e-9)) ** 2
                            + (se_map['rating'] / max(abs(b['rating']), 1e-9)) ** 2) \
          if a3 == a3 else np.nan
print(f"  log(MV): rating {b['rating']:+.4f}/pt, camp {b['camp']:+.3f}")
print(f"  A3 delta = {a3:+.2f} ±{a3_se:.2f} rating pts")

print("\n=== SYNTHESIS ===")
anchors = {'A1 movers': (a1, a1_se), 'A2 teams': (a2, a2_se),
             'A3 market': (a3, a3_se)}
ws, vs = [], []
for k, (v, se) in anchors.items():
    if v == v and se and se > 0:
        ws.append(1 / se ** 2)
        vs.append(v)
        print(f"  {k:<11} {v:+.2f} ±{se:.2f}")
delta = float(np.average(vs, weights=ws))
delta_se = float(1 / np.sqrt(sum(ws)))
print(f"  COMBINED: Liga 3 minus Campeonato = {delta:+.2f} ±{delta_se:.2f} "
       f"rating pts\n  (a Camp rating R ~ L3 rating R - {delta:.1f})")
json.dump({'reference': 'L3', 'delta_pts_L3_minus_CAMP': delta,
             'delta_se': delta_se,
             'anchors': {k: {'delta': (None if v != v else round(v, 3)),
                              'se': (None if se != se else round(se, 3))}
                          for k, (v, se) in anchors.items()}},
            open(_HERE / 'league_conversion.json', 'w'), indent=2)

# append absolute columns (L3 reference)
for fname, col in [('acp_rating_per_player_season.parquet', 'acp_rating'),
                     ('acp_projection.parquet', 'projection')]:
    df = pd.read_parquet(_HERE / fname)
    df[col + '_abs'] = df[col] - np.where(df['league'] == 'CAMP', delta, 0.0)
    df.to_parquet(_HERE / fname)
    print(f"  {fname}: wrote {col}_abs")
