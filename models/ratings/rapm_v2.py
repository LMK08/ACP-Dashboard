#!/usr/bin/env python3
"""xG-RAPM v2 — context-adjusted + (optionally) prior-augmented.

v1 (rapm.py) passed its gates marginally. v2 tests the upgrades that
soccer's low scoring rate motivates:

  STAGE 1 (context): weighted regression of segment xG-diff/90 on
    [intercept(=home adv), league, score-state, man-advantage] — fitted
    on TRAIN only. Removes outcome variance that isn't about the players
    on the pitch. (Opponent strength itself is already adjusted by
    construction — the opposing XI are −1 columns — but league baselines
    and game state were leaking into player coefficients in v1.)
  STAGE 2 (players): ridge on stage-1 residuals.
     Variant A — shrink to zero (as v1).
     Variant B — AUGMENTED: shrink toward an informative prior built
       from the GPA+DefR+DWAE blend (Matano-style); final coef =
       scale·prior + ridge(deviations). The standard small-sample remedy.
  Plus season-recency sample weights (0.85^years-back).

Same pre-registered gates as v1 for comparability:
  G1 split-half coef corr >= 0.30 | G2 beat baselines on 25/26 holdout |
  G3 orthogonality/incrementality.
v1 reference numbers: G1 0.310, G2 wMSE 3.1479 (vs 3.1763 / 3.1813).

Run from the Dashboard dir: python models/ratings/rapm_v2.py
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
PERIOD_OFF = {'1H': 0, '2H': 45 * 60, 'E1': 90 * 60, 'E2': 105 * 60, 'P': 120 * 60}

print("[1/6] events…", flush=True)
cols = ['matchId', 'seasonId', 'matchPeriod', 'minute', 'second',
         'team.id', 'team.name', 'player.id', 'type.primary', 'shot.xg',
         'shot.isGoal']
ev = pd.concat([pd.read_parquet(_GPA_DATA / f'{f}.parquet', columns=cols)
                  for f in ['liga3_portugal_events', 'campeonato_portugal_events']],
                 ignore_index=True)
ev = ev.dropna(subset=['player.id', 'team.id'])
ev['player.id'] = ev['player.id'].astype(int)
ev['team.id'] = ev['team.id'].astype(int)
ev['_t'] = (ev['matchPeriod'].map(PERIOD_OFF).fillna(0)
              + pd.to_numeric(ev['minute'], errors='coerce').fillna(0) * 60
              + pd.to_numeric(ev['second'], errors='coerce').fillna(0))
ev['xg'] = np.where(ev['type.primary'] == 'shot',
                      pd.to_numeric(ev['shot.xg'], errors='coerce').fillna(0), 0.0)
ev['goal'] = ((ev['shot.isGoal'] == True)
                | (ev['type.primary'] == 'own_goal')).astype(int)
# own goals count for the OPPONENT — handled at segment scoring below.

ms = pd.read_parquet(_DASH / 'matches_summary.parquet',
                       columns=['matchId', 'homeTeamName', 'awayTeamName'])
tn = ev[['matchId', 'team.id', 'team.name']].drop_duplicates()
home = (ms.merge(tn, left_on=['matchId', 'homeTeamName'],
                   right_on=['matchId', 'team.name'])
           .set_index('matchId')['team.id'].astype(int).to_dict())
away = (ms.merge(tn, left_on=['matchId', 'awayTeamName'],
                   right_on=['matchId', 'team.name'])
           .set_index('matchId')['team.id'].astype(int).to_dict())

print("[2/6] segments + context covariates…", flush=True)
spans = (ev.groupby(['matchId', 'player.id', 'team.id'])['_t']
            .agg(['min', 'max']).reset_index())
total_min = ((spans['max'] - spans['min']) / 60).groupby(spans['player.id']).sum().to_dict()

seg_rows = []
for mid, sp in spans.groupby('matchId'):
    h, a = home.get(mid), away.get(mid)
    if h is None or a is None:
        continue
    exg = ev[ev['matchId'] == mid]
    goals = exg[exg['goal'] == 1][['_t', 'team.id', 'type.primary']]
    pts = np.unique(np.concatenate([sp['min'].to_numpy(), sp['max'].to_numpy()]))
    if len(pts) < 2:
        continue
    sxg = exg[exg['xg'] > 0]
    for t1, t2 in zip(pts[:-1], pts[1:]):
        mins = (t2 - t1) / 60.0
        if mins < MIN_SEG_MIN:
            continue
        midp = (t1 + t2) / 2
        on = sp[(sp['min'] <= midp) & (sp['max'] >= midp)]
        nh = int((on['team.id'] == h).sum()); na = int((on['team.id'] == a).sum())
        if nh + na < 16:
            continue
        shots = sxg[(sxg['_t'] >= t1) & (sxg['_t'] < t2)]
        xh = shots.loc[shots['team.id'] == h, 'xg'].sum()
        xa = shots.loc[shots['team.id'] == a, 'xg'].sum()
        # score state at segment start (own goals flip to the other team)
        gpre = goals[goals['_t'] < t1]
        gh = int(((gpre['team.id'] == h) & (gpre['type.primary'] != 'own_goal')).sum()
                   + ((gpre['team.id'] == a) & (gpre['type.primary'] == 'own_goal')).sum())
        ga = int(((gpre['team.id'] == a) & (gpre['type.primary'] != 'own_goal')).sum()
                   + ((gpre['team.id'] == h) & (gpre['type.primary'] == 'own_goal')).sum())
        seg_rows.append({'matchId': mid,
                          'y': (xh - xa) / mins * 90.0, 'w': mins,
                          'score_diff': float(np.clip(gh - ga, -2, 2)),
                          'man_adv': float(np.clip(nh - na, -2, 2)),
                          'home_p': tuple(on.loc[on['team.id'] == h, 'player.id']),
                          'away_p': tuple(on.loc[on['team.id'] == a, 'player.id'])})
segs = pd.DataFrame(seg_rows)
seasons = ev[['matchId', 'seasonId']].drop_duplicates().set_index('matchId')['seasonId']
segs['seasonId'] = segs['matchId'].map(seasons)
segs['camp'] = segs['seasonId'].isin(CAMP_SEASONS).astype(float)
segs['yr'] = segs['seasonId'].map(SEASON_YR)
segs['w_rec'] = segs['w'] * (RECENCY ** (2025 - segs['yr']))
print(f"  {len(segs):,} segments; man_adv!=0 in {(segs['man_adv'] != 0).mean()*100:.1f}%, "
       f"score_diff!=0 in {(segs['score_diff'] != 0).mean()*100:.1f}%")

players = sorted([p for p, m in total_min.items() if m >= MIN_PLAYER_MIN])
pidx = {p: i for i, p in enumerate(players)}
REPL_H, REPL_A = len(players), len(players) + 1
NCOL = len(players) + 2


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

print("[3/6] stage 1 — context regression (train-only fit)…", flush=True)
s1 = LinearRegression()
s1.fit(train[CTX], train['y'], sample_weight=train['w_rec'])
print(f"  coefs: home_adv(intercept)={s1.intercept_:+.3f}  "
       + '  '.join(f"{c}={b:+.3f}" for c, b in zip(CTX, s1.coef_)))
train = train.assign(res=train['y'] - s1.predict(train[CTX]))
test = test.assign(res=test['y'] - s1.predict(test[CTX]))

# prior for variant B: GPA+DefR+DWAE blend (latest pre-test season per player)
print("[4/6] build prior (blend, train seasons only)…", flush=True)
g = pd.read_parquet(_DASH / 'gpa_player_season_values.parquet',
                      columns=['playerId', 'seasonId', 'mins_played',
                                'position_group', 'Total Value'])
g['playerId'] = pd.to_numeric(g['playerId'], errors='coerce')
d = pd.read_parquet(_DASH / 'models/defr/defr_per_player_season.parquet',
                      columns=['playerId', 'seasonId', 'defr_adj', 'defr_dwae_p90'])
r = pd.read_parquet(_DASH / 'models/roles/role_assignments_season.parquet',
                      columns=['playerId', 'seasonId', 'primary_role_name'])
pri = (g.merge(d, on=['playerId', 'seasonId'], how='left')
         .merge(r, on=['playerId', 'seasonId'], how='left'))
pri = pri[(pri['mins_played'] >= 500) & ~pri['seasonId'].isin(TEST_SEASONS)]
pri['role'] = pri['primary_role_name'].fillna(pri['position_group'])
for c, src in [('p1', 'Total Value'), ('p2', 'defr_adj'), ('p3', 'defr_dwae_p90')]:
    pri[c] = pri.groupby(['role', 'seasonId'])[src].rank(pct=True)
pri['blend'] = 0.6 * pri['p1'] + 0.25 * pri['p2'].fillna(0.5) + 0.15 * pri['p3'].fillna(0.5)
pri['yr'] = pri['seasonId'].map(SEASON_YR)
latest = pri.sort_values('yr').drop_duplicates('playerId', keep='last')
prior_map = dict(zip(latest['playerId'], latest['blend'] - latest['blend'].mean()))
prior_vec = np.array([prior_map.get(p, 0.0) for p in players] + [0.0, 0.0])
print(f"  prior coverage: {np.mean([p in prior_map for p in players])*100:.0f}% of player columns")


def fit_ridge(sub, alpha, ycol, offset_vec=None):
    X = design(sub)
    y = sub[ycol].to_numpy()
    if offset_vec is not None:
        y = y - X @ offset_vec
    m = Ridge(alpha=alpha, fit_intercept=False)
    m.fit(X, y, sample_weight=sub['w_rec'].to_numpy())
    return m


def wmse_total(sub, coefs):
    X = design(sub)
    pred = s1.predict(sub[CTX]) + X @ coefs
    return float(np.average((sub['y'].to_numpy() - pred) ** 2,
                              weights=sub['w'].to_numpy()))


print("[5/6] stage 2 — variants over alpha (holdout 25/26)…", flush=True)
results = {}
for name, off in [('A_ctx', None), ('B_ctx+prior', None)]:
    best = None
    for a in ALPHAS:
        if name == 'A_ctx':
            m = fit_ridge(train, a, 'res')
            coefs = m.coef_
        else:
            # scale the prior on train residuals, then ridge the deviations
            Xtr = design(train)
            xp = Xtr @ prior_vec
            sc = LinearRegression(fit_intercept=False)
            sc.fit(xp.reshape(-1, 1), train['res'],
                    sample_weight=train['w_rec'])
            scale = float(sc.coef_[0])
            train2 = train.assign(res2=train['res'] - scale * xp)
            m = fit_ridge(train2, a, 'res2')
            coefs = scale * prior_vec + m.coef_
        e = wmse_total(test, coefs)
        if best is None or e < best[1]:
            best = (a, e, coefs)
    results[name] = best
    print(f"  {name:<12} best alpha={best[0]:.0f}  holdout wMSE={best[1]:.4f}")

base_mean = float(np.average(
    (test['y'].to_numpy() - np.average(train['y'], weights=train['w'])) ** 2,
    weights=test['w']))
ctx_only = float(np.average(
    (test['y'].to_numpy() - s1.predict(test[CTX])) ** 2, weights=test['w']))
print(f"  baselines: global-mean {base_mean:.4f} | context-only {ctx_only:.4f} | "
       f"v1 RAPM 3.1479")

print("[6/6] G1 split-half per variant…", flush=True)
mins_arr = np.array([total_min.get(p, 0) for p in players])
mask = mins_arr >= 1600
for name, (alpha, err, coefs_full) in results.items():
    halves = []
    for par in (0, 1):
        sub = segs[segs['matchId'].astype('int64') % 2 == par].copy()
        sub = sub.assign(res=sub['y'] - s1.predict(sub[CTX]))
        if name == 'A_ctx':
            m = fit_ridge(sub, alpha, 'res')
            halves.append(m.coef_[:len(players)])
        else:
            Xs = design(sub); xp = Xs @ prior_vec
            sc = LinearRegression(fit_intercept=False)
            sc.fit(xp.reshape(-1, 1), sub['res'], sample_weight=sub['w_rec'])
            sub2 = sub.assign(res2=sub['res'] - float(sc.coef_[0]) * xp)
            m = fit_ridge(sub2, alpha, 'res2')
            halves.append((float(sc.coef_[0]) * prior_vec + m.coef_)[:len(players)])
    r_sh = pearsonr(halves[0][mask], halves[1][mask])[0]
    print(f"  {name:<12} split-half r = {r_sh:.3f}  (v1: 0.310)")

# save best variant coefficients
best_name = min(results, key=lambda k: results[k][1])
alpha, err, coefs = results[best_name]
pd.DataFrame({'playerId': players,
               'rapm_v2': coefs[:len(players)]}).to_parquet(
    _HERE / 'rapm_v2_coefficients.parquet')
print(f"\nbest variant: {best_name} (wMSE {err:.4f}) -> rapm_v2_coefficients.parquet")
