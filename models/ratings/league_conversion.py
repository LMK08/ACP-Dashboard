#!/usr/bin/env python3
"""League conversion: how many ACP rating points is Campeonato worth
relative to Liga 3? Triangulated from independent anchors.

  A1 MOVERS    — same player, consecutive seasons, league switch,
                  benchmarked against a STAYER shrinkage model
                  (next_dev = r x cur_dev + age-bin intercepts) so
                  regression-to-the-mean from selection-on-extremes is
                  removed (Lucas's catch: movers up are selected HIGH,
                  movers down LOW — both inflate the naive gap; the
                  naive +2.23 was ~37% RTM artifact, corrected +1.40).
  A2 TEAMS     — same club appearing in both leagues (promotion/releg):
                  xGD/90 shift / within-league slope of team xGD on
                  team mean rating. Direction check; magnitude inflated
                  by squad turnover.
  (Transfermarkt anchor REMOVED — Lucas directive: TM values are
   near-nonexistent at L3/Camp level; ignore moving forward. Duel-
   ladder mover residuals corroborate direction: L3 > Camp.)

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
o['age'] = o['yr'] + 1.0 - (o['playerId'].map(bd).dt.year
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
        if (b['yr'] - a['yr'] == 1 and a['league'] == b['league']
                and a['mins_played'] >= 500 and b['mins_played'] >= 500):
            dl.append((bin_of(a['age']), b['acp_rating'] - a['acp_rating']))
DL = pd.DataFrame(dl, columns=['ab', 'd'])
AGE_D = {b: (DL[DL['ab'] == b]['d'].sum() + PRIOR[b] * 60)
            / (len(DL[DL['ab'] == b]) + 60) for b in range(5)}

print("=== A1: MOVERS (vs STAYER shrinkage model — RTM-corrected) ===")
# Regression-to-mean correction (Lucas): movers up are selected on HIGH
# observed ratings (lucky noise included) and would regress down anyway;
# movers down the reverse — both inflate the naive gap. Benchmark every
# mover against what an identical STAYER would do: fit on within-league
# pairs  next_dev = r * cur_dev + c[age_bin], dev = rating minus the
# league-season mean of rated players; mover residual = actual - that.
# mean over the historic >=500' cohort (v6.4 admits sub-floor rows)
o['_lgmean'] = (o['acp_rating'].where(o['mins_played'] >= 500)
                  .groupby([o['league'], o['seasonId']]).transform('mean'))
o['_dev'] = o['acp_rating'] - o['_lgmean']
stay, mv = [], []
for pid, g in o.dropna(subset=['age']).sort_values('yr').groupby('playerId'):
    rr = g.to_dict('records')
    for a, b in zip(rr, rr[1:]):
        # estimators pinned to the historic >=500' regime
        if (b['yr'] - a['yr'] != 1 or a['mins_played'] < 500
                or b['mins_played'] < 500):
            continue
        row = {'cur_dev': a['_dev'], 'nxt_dev': b['_dev'],
                 'ab': bin_of(a['age']),
                 'w': min(a['mins_played'], b['mins_played'])}
        if a['league'] == b['league']:
            stay.append(row)
        else:
            mv.append(row | {'dir': f"{a['league']}->{b['league']}"})
ST = pd.DataFrame(stay)
Xs = np.column_stack([ST['cur_dev'].values] +
                       [(ST['ab'] == k).astype(float).values for k in range(5)])
beta_s, *_ = np.linalg.lstsq(Xs, ST['nxt_dev'].values, rcond=None)
r_shrink = beta_s[0]
print(f"  stayer shrinkage r = {r_shrink:.2f} (n={len(ST)}; a +10 dev player "
       f"regresses to +{10*r_shrink:.1f} in a year)")
MV = pd.DataFrame(mv)
exp_dev = (MV['cur_dev'] * r_shrink
             + MV['ab'].map({k: beta_s[1 + k] for k in range(5)}))
MV['resid'] = MV['nxt_dev'] - exp_dev
print("  mover selection profile (Lucas's question — how selected are they?):")
for d, g in MV.groupby('dir'):
    print(f"    {d}: mean prior dev {np.average(g['cur_dev'], weights=g['w']):+.1f} pts"
           f" vs own-league average")
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

# A3 MARKET VALUES — REMOVED (Lucas directive 2026-06-11): Transfermarkt
# coverage at Liga 3 / Campeonato level is near-nonexistent (9 Camp
# player-seasons) — ignore TM values moving forward.

print("\n=== SYNTHESIS ===")
anchors = {'A1 movers': (a1, a1_se), 'A2 teams': (a2, a2_se)}
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
# RECRUITMENT conversion (Lucas 2026-06-11: Camp projections still too
# high): the both-directions average is right for DESCRIPTION, but the
# up-mover direction (+0.10) is contaminated by elite selection — those
# players held their level BECAUSE they were the chosen few. For
# projecting a typical Camp player into L3 terms (the recruitment
# counterfactual), use the L3->Camp direction estimate, which is free
# of up-selection (its own bias runs the other way, making it the
# conservative bound).
delta_recruit = res['L3->CAMP'][0]
print(f"  RECRUIT delta (down-mover bound): {delta_recruit:+.2f} "
       f"±{res['L3->CAMP'][1]:.2f} — applied to projections")
json.dump({'reference': 'L3',
             'delta_pts_L3_minus_CAMP_descriptive': delta,
             'delta_se': delta_se,
             'delta_pts_recruit': delta_recruit,
             'anchors': {k: {'delta': (None if v != v else round(v, 3)),
                              'se': (None if se != se else round(se, 3))}
                          for k, (v, se) in anchors.items()}},
            open(_HERE / 'league_conversion.json', 'w'), indent=2)

# absolute columns: descriptive delta for the rating, recruit delta for
# the projection (L3 reference)
for fname, col, dd in [('acp_rating_per_player_season.parquet', 'acp_rating', delta),
                          ('acp_projection.parquet', 'projection', delta_recruit)]:
    df = pd.read_parquet(_HERE / fname)
    df[col + '_abs'] = df[col] - np.where(df['league'] == 'CAMP', dd, 0.0)
    df.to_parquet(_HERE / fname)
    print(f"  {fname}: wrote {col}_abs (delta {dd:+.2f})")
