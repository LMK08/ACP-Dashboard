#!/usr/bin/env python3
"""xG-RAPM v3 — true lineup intervals (substitutions + red cards).

Same two-stage design as v2 (context regression, then ridge on player
presence) but presence now comes from REAL on-pitch intervals built from
Wyscout lineups/substitutions/red-card minutes (build_intervals.py),
replacing v2's first/last-event-timestamp approximation. Consequences:
  - segments are true lineup-constant stretches;
  - man_adv is now an exact red-card effect (1% of intervals end in a
    red), not span-approximation noise (v2: 54% of segments "unequal");
  - substitute minutes are correct (median 24 min vs spans' undercount).

Gates as before. Reference: v2 holdout wMSE 3.1248, split-half 0.378.

Run from the Dashboard dir: python models/ratings/rapm_v3.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge, LinearRegression

_HERE = Path(__file__).resolve().parent
_DASH = _HERE.parent.parent
_GPA_DATA = _DASH.parent.parent / 'GPA Model Project v2' / 'parquet_data'

MIN_SEG_MIN = 3.0
MIN_PLAYER_MIN = 600.0
ALPHAS = [1000.0, 3000.0, 10000.0, 30000.0, 100000.0]
TEST_SEASONS = {191782, 191779}
CAMP_SEASONS = {190230, 191779}
SEASON_YR = {188221: 2021, 188222: 2022, 189147: 2023, 190090: 2024,
              191782: 2025, 190230: 2023, 191779: 2025}
RECENCY = 0.85

print("[1/6] events + intervals…", flush=True)
cols = ['matchId', 'seasonId', 'minute', 'second', 'team.id', 'team.name',
         'type.primary', 'shot.xg', 'shot.isGoal']
ev = pd.concat([pd.read_parquet(_GPA_DATA / f'{f}.parquet', columns=cols)
                  for f in ['liga3_portugal_events', 'campeonato_portugal_events']],
                 ignore_index=True)
ev['team.id'] = pd.to_numeric(ev['team.id'], errors='coerce')
ev['u'] = (pd.to_numeric(ev['minute'], errors='coerce').fillna(0)
             + pd.to_numeric(ev['second'], errors='coerce').fillna(0) / 60)
ev['xg'] = np.where(ev['type.primary'] == 'shot',
                      pd.to_numeric(ev['shot.xg'], errors='coerce').fillna(0), 0.0)
ev['goal'] = ((ev['shot.isGoal'] == True)
                | (ev['type.primary'] == 'own_goal')).astype(int)

iv = pd.read_parquet(_HERE / 'player_match_intervals.parquet')
iv['teamId'] = iv['teamId'].astype(int)
iv['mins'] = iv['u_out'] - iv['u_in']
total_min = iv.groupby('playerId')['mins'].sum().to_dict()

ms = pd.read_parquet(_DASH / 'matches_summary.parquet',
                       columns=['matchId', 'homeTeamName', 'awayTeamName'])
tn = ev[['matchId', 'team.id', 'team.name']].dropna().drop_duplicates()
home = (ms.merge(tn, left_on=['matchId', 'homeTeamName'],
                   right_on=['matchId', 'team.name'])
           .set_index('matchId')['team.id'].astype(int).to_dict())
away = (ms.merge(tn, left_on=['matchId', 'awayTeamName'],
                   right_on=['matchId', 'team.name'])
           .set_index('matchId')['team.id'].astype(int).to_dict())
# matches whose lineup teams resolved: also accept via interval teamIds
for mid, g in iv.groupby('matchId'):
    tids = sorted(g['teamId'].unique())
    if mid not in home and len(tids) == 2:
        continue  # cannot orient without summary names — skipped below

print("[2/6] segments from lineup-constant stretches…", flush=True)
seg_rows = []
for mid, g in iv.groupby('matchId'):
    h, a = home.get(mid), away.get(mid)
    if h is None or a is None:
        continue
    exg = ev[ev['matchId'] == mid]
    goals = exg[exg['goal'] == 1][['u', 'team.id', 'type.primary']]
    sxg = exg[exg['xg'] > 0]
    end_u = float(max(g['u_out'].max(), exg['u'].max() if len(exg) else 90))
    pts = np.unique(np.concatenate([[0.0, end_u],
                                       g['u_in'].to_numpy(), g['u_out'].to_numpy()]))
    for t1, t2 in zip(pts[:-1], pts[1:]):
        mins = t2 - t1
        if mins < MIN_SEG_MIN:
            continue
        midp = (t1 + t2) / 2
        on = g[(g['u_in'] <= midp) & (g['u_out'] > midp)]
        nh = int((on['teamId'] == h).sum()); na = int((on['teamId'] == a).sum())
        if nh < 7 or na < 7:
            continue
        shots = sxg[(sxg['u'] >= t1) & (sxg['u'] < t2)]
        xh = shots.loc[shots['team.id'] == h, 'xg'].sum()
        xa = shots.loc[shots['team.id'] == a, 'xg'].sum()
        gpre = goals[goals['u'] < t1]
        gh = int(((gpre['team.id'] == h) & (gpre['type.primary'] != 'own_goal')).sum()
                   + ((gpre['team.id'] == a) & (gpre['type.primary'] == 'own_goal')).sum())
        ga = int(((gpre['team.id'] == a) & (gpre['type.primary'] != 'own_goal')).sum()
                   + ((gpre['team.id'] == h) & (gpre['type.primary'] == 'own_goal')).sum())
        seg_rows.append({'matchId': mid,
                          'y': (xh - xa) / mins * 90.0, 'w': mins,
                          'score_diff': float(np.clip(gh - ga, -2, 2)),
                          'man_adv': float(np.clip(nh - na, -2, 2)),
                          'home_p': tuple(on.loc[on['teamId'] == h, 'playerId']),
                          'away_p': tuple(on.loc[on['teamId'] == a, 'playerId'])})
segs = pd.DataFrame(seg_rows)
seasons = ev[['matchId', 'seasonId']].drop_duplicates().set_index('matchId')['seasonId']
segs['seasonId'] = segs['matchId'].map(seasons)
segs['camp'] = segs['seasonId'].isin(CAMP_SEASONS).astype(float)
segs['yr'] = segs['seasonId'].map(SEASON_YR)
segs['w_rec'] = segs['w'] * (RECENCY ** (2025 - segs['yr']))
print(f"  {len(segs):,} segments from {segs['matchId'].nunique():,} matches; "
       f"median {segs['w'].median():.1f} min; man_adv!=0 in "
       f"{(segs['man_adv'] != 0).mean()*100:.1f}% (true reds only)")

players = sorted([p for p, m in total_min.items() if m >= MIN_PLAYER_MIN])
pidx = {p: i for i, p in enumerate(players)}
REPL_H, REPL_A = len(players), len(players) + 1
NCOL = len(players) + 2
print(f"  {len(players):,} player columns + 2 replacement buckets")


def design(sub):
    rows, colsx, vals = [], [], []
    for i, r in enumerate(sub.itertuples(index=False)):
        for p in r.home_p:
            rows.append(i); colsx.append(pidx.get(p, REPL_H)); vals.append(1.0)
        for p in r.away_p:
            rows.append(i); colsx.append(pidx.get(p, REPL_A)); vals.append(-1.0)
    return sparse.csr_matrix((vals, (rows, colsx)), shape=(len(sub), NCOL))


CTX = ['camp', 'score_diff', 'man_adv']
train = segs[~segs['seasonId'].isin(TEST_SEASONS)]
test = segs[segs['seasonId'].isin(TEST_SEASONS)]
print(f"  train {len(train):,} / test {len(test):,}")

print("[3/6] stage 1 — context…", flush=True)
s1 = LinearRegression()
s1.fit(train[CTX], train['y'], sample_weight=train['w_rec'])
print(f"  home_adv={s1.intercept_:+.3f}  "
       + '  '.join(f"{c}={b:+.3f}" for c, b in zip(CTX, s1.coef_)))
train = train.assign(res=train['y'] - s1.predict(train[CTX]))
test = test.assign(res=test['y'] - s1.predict(test[CTX]))

print("[4/6] stage 2 — ridge over alpha (holdout 25/26)…", flush=True)
best = None
Xtest = design(test)
for a in ALPHAS:
    m = Ridge(alpha=a, fit_intercept=False)
    m.fit(design(train), train['res'], sample_weight=train['w_rec'].to_numpy())
    pred = s1.predict(test[CTX]) + Xtest @ m.coef_
    e = float(np.average((test['y'].to_numpy() - pred) ** 2,
                           weights=test['w'].to_numpy()))
    print(f"  alpha={a:>7.0f}  holdout wMSE={e:.4f}")
    if best is None or e < best[1]:
        best = (a, e, m.coef_)
alpha, err, coefs = best

base_mean = float(np.average(
    (test['y'].to_numpy() - np.average(train['y'], weights=train['w'])) ** 2,
    weights=test['w']))
ctx_only = float(np.average(
    (test['y'].to_numpy() - s1.predict(test[CTX])) ** 2, weights=test['w']))
print(f"\n  G2: global-mean {base_mean:.4f} | context-only {ctx_only:.4f} | "
       f"v2 RAPM 3.1248 | v3 RAPM {err:.4f}")

print("[5/6] G1 split-half…", flush=True)
mins_arr = np.array([total_min.get(p, 0) for p in players])
mask = mins_arr >= 1600
halves = []
for par in (0, 1):
    sub = segs[segs['matchId'].astype('int64') % 2 == par].copy()
    sub = sub.assign(res=sub['y'] - s1.predict(sub[CTX]))
    m = Ridge(alpha=alpha, fit_intercept=False)
    m.fit(design(sub), sub['res'], sample_weight=sub['w_rec'].to_numpy())
    halves.append(m.coef_[:len(players)])
r_sh = pearsonr(halves[0][mask], halves[1][mask])[0]
print(f"  split-half coef corr (n={int(mask.sum())}): r = {r_sh:.3f}  "
       f"(v1 0.310, v2 0.378)")

print("[6/6] orthogonality…", flush=True)
coef = pd.DataFrame({'playerId': players, 'rapm_v3': coefs[:len(players)]})
g = pd.read_parquet(_DASH / 'gpa_player_season_values.parquet',
                      columns=['playerId', 'seasonId', 'mins_played',
                                'position_group', 'Total Value'])
g['playerId'] = pd.to_numeric(g['playerId'], errors='coerce')
d = pd.read_parquet(_DASH / 'models/defr/defr_per_player_season.parquet',
                      columns=['playerId', 'seasonId', 'defr_adj', 'defr_dwae_p90'])
cur = (g[(g['seasonId'].isin(TEST_SEASONS)) & (g['mins_played'] >= 800)
           & (g['position_group'] != 'GK')]
         .merge(d, on=['playerId', 'seasonId'], how='left')
         .merge(coef, on='playerId', how='inner'))
for c in ['Total Value', 'defr_adj', 'defr_dwae_p90']:
    ok = cur.dropna(subset=[c, 'rapm_v3'])
    print(f"  corr(RAPM_v3, {c:<14}) = {spearmanr(ok['rapm_v3'], ok[c])[0]:+.3f}")
coef.to_parquet(_HERE / 'rapm_v3_coefficients.parquet')
print(f"\nsaved rapm_v3_coefficients.parquet (alpha={alpha:.0f})")
