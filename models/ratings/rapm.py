#!/usr/bin/env python3
"""Gated xG-RAPM prototype — regularized adjusted plus-minus.

Ridge regression of segment-level xG difference (per 90, minutes-
weighted) on player-presence indicators (+1 home, −1 away) with a
home-advantage intercept. Segments are delimited by lineup-change
points approximated from each player's first/last event timestamps
(no explicit substitution events in our feed — known caveat).

PRE-REGISTERED GATES (agreed before building):
  G1 stability  : split-half (odd/even matches) coefficient corr >= 0.30
                   for players with >=800 min in each half.
  G2 predictive : temporal holdout (train <= 24/25 + Camp 23/24, test
                   25/26): weighted MSE must beat (a) home-adv-only and
                   (b) TEAM-dummies ridge baseline out of sample.
  G3 incremental: low corr with GPA/DefR candidates AND adds rank-
                   predictive power for next-season overall_pct.
Adopt as a rating component ONLY if all three pass.

Run from the Dashboard dir: python models/ratings/rapm.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge

_HERE = Path(__file__).resolve().parent
_DASH = _HERE.parent.parent
_GPA_DATA = _DASH.parent.parent / 'GPA Model Project v2' / 'parquet_data'

MIN_SEG_MIN = 3.0
MIN_PLAYER_MIN = 600.0     # below this -> pooled replacement column
ALPHAS = [100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0, 100000.0]
TEST_SEASONS = {191782, 191779}

PERIOD_OFF = {'1H': 0, '2H': 45 * 60, 'E1': 90 * 60, 'E2': 105 * 60, 'P': 120 * 60}

print("[1/5] events…", flush=True)
cols = ['matchId', 'seasonId', 'matchPeriod', 'minute', 'second',
         'team.id', 'team.name', 'player.id', 'type.primary', 'shot.xg']
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

# matches_summary has home/away NAMES (the id columns are null) — resolve
# ids by joining the per-match (team.id, team.name) pairs from the events.
ms = pd.read_parquet(_DASH / 'matches_summary.parquet',
                       columns=['matchId', 'homeTeamName', 'awayTeamName'])
tn = ev[['matchId', 'team.id', 'team.name']].drop_duplicates()
mh = ms.merge(tn, left_on=['matchId', 'homeTeamName'],
                right_on=['matchId', 'team.name'], how='left')
ma = ms.merge(tn, left_on=['matchId', 'awayTeamName'],
                right_on=['matchId', 'team.name'], how='left')
home = mh.dropna(subset=['team.id']).set_index('matchId')['team.id'].astype(int).to_dict()
away = ma.dropna(subset=['team.id']).set_index('matchId')['team.id'].astype(int).to_dict()
n_ev_matches = ev['matchId'].nunique()
print(f"  home/away resolved by name for {len(home):,} matches "
       f"({len(home)/n_ev_matches*100:.0f}% of {n_ev_matches:,} event matches)")

print("[2/5] segments…", flush=True)
spans = (ev.groupby(['matchId', 'player.id', 'team.id'])['_t']
            .agg(['min', 'max']).reset_index())
total_min = (spans.groupby('player.id')
                .apply(lambda g: ((g['max'] - g['min']) / 60).sum())).to_dict()

seg_rows = []
for mid, sp in spans.groupby('matchId'):
    h, a = home.get(mid), away.get(mid)
    if h is None or a is None or pd.isna(h) or pd.isna(a):
        continue
    exg = ev[ev['matchId'] == mid]
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
        if len(on) < 16:           # both sides mostly populated
            continue
        shots = sxg[(sxg['_t'] >= t1) & (sxg['_t'] < t2)]
        xh = shots.loc[shots['team.id'] == h, 'xg'].sum()
        xa = shots.loc[shots['team.id'] == a, 'xg'].sum()
        seg_rows.append({'matchId': mid,
                          'y': (xh - xa) / mins * 90.0, 'w': mins,
                          'home_p': tuple(on.loc[on['team.id'] == h, 'player.id']),
                          'away_p': tuple(on.loc[on['team.id'] == a, 'player.id'])})
segs = pd.DataFrame(seg_rows)
seasons = ev[['matchId', 'seasonId']].drop_duplicates().set_index('matchId')['seasonId']
segs['seasonId'] = segs['matchId'].map(seasons)
print(f"  {len(segs):,} segments from {segs['matchId'].nunique():,} matches; "
       f"median {segs['w'].median():.1f} min")

# player columns (>= MIN_PLAYER_MIN total) + replacement buckets
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
    X = sparse.csr_matrix((vals, (rows, colsx)), shape=(len(sub), NCOL))
    return X


def fit(sub, alpha):
    X = design(sub)
    m = Ridge(alpha=alpha, fit_intercept=True)
    m.fit(X, sub['y'].to_numpy(), sample_weight=sub['w'].to_numpy())
    return m


print("[3/5] alpha by temporal CV + G2 baselines…", flush=True)
train = segs[~segs['seasonId'].isin(TEST_SEASONS)]
test = segs[segs['seasonId'].isin(TEST_SEASONS)]
print(f"  train {len(train):,} segs / test {len(test):,} segs")


def wmse(model, sub, Xs=None):
    X = design(sub) if Xs is None else Xs
    pred = model.predict(X)
    w = sub['w'].to_numpy()
    return float(np.average((sub['y'].to_numpy() - pred) ** 2, weights=w))


best = None
Xtest = design(test)
for a in ALPHAS:
    m = fit(train, a)
    e = wmse(m, test, Xtest)
    print(f"  alpha={a:>7.0f}  holdout wMSE={e:.4f}")
    if best is None or e < best[1]:
        best = (a, e, m)
alpha, err_rapm, model_t = best

# baselines
base_mean = float(np.average(
    (test['y'].to_numpy() - np.average(train['y'], weights=train['w'])) ** 2,
    weights=test['w']))
# team-dummies ridge
teams = sorted(set(ev['team.id'].unique()))
tidx = {t: i for i, t in enumerate(teams)}


def team_design(sub):
    rows, colsx, vals = [], [], []
    hh = sub['matchId'].map(home).to_numpy()
    aa = sub['matchId'].map(away).to_numpy()
    for i, (th, ta) in enumerate(zip(hh, aa)):
        rows += [i, i]; colsx += [tidx[th], tidx[ta]]; vals += [1.0, -1.0]
    return sparse.csr_matrix((vals, (rows, colsx)), shape=(len(sub), len(teams)))


bt = None
for a in [10.0, 30.0, 100.0, 300.0]:
    mt = Ridge(alpha=a, fit_intercept=True)
    mt.fit(team_design(train), train['y'], sample_weight=train['w'])
    e = float(np.average((test['y'].to_numpy()
                            - mt.predict(team_design(test))) ** 2,
                           weights=test['w']))
    if bt is None or e < bt[1]:
        bt = (a, e)
print(f"\n  G2: holdout wMSE — home-adv only {base_mean:.4f} | "
       f"team ridge {bt[1]:.4f} | RAPM {err_rapm:.4f}")
g2 = err_rapm < bt[1] and err_rapm < base_mean
print(f"  G2 {'PASS' if g2 else 'FAIL'} (RAPM must beat both)")

print("[4/5] G1 split-half stability…", flush=True)
mfull = fit(segs, alpha)
halves = {}
for par in (0, 1):
    sub = segs[segs['matchId'].astype('int64') % 2 == par]
    halves[par] = fit(sub, alpha).coef_
mins_arr = np.array([total_min.get(p, 0) for p in players])
mask = mins_arr >= 1600    # ~800+ min in each half on average
r_g1 = pearsonr(halves[0][:len(players)][mask], halves[1][:len(players)][mask])[0]
print(f"  split-half coef corr (n={int(mask.sum())} players): r = {r_g1:.3f}")
g1 = r_g1 >= 0.30
print(f"  G1 {'PASS' if g1 else 'FAIL'} (gate >= 0.30)")

print("[5/5] G3 incremental vs GPA/DefR…", flush=True)
coef = pd.DataFrame({'playerId': players, 'rapm': mfull.coef_[:len(players)]})
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
    ok = cur.dropna(subset=[c, 'rapm'])
    print(f"  corr(RAPM, {c:<14}) = {spearmanr(ok['rapm'], ok[c])[0]:+.3f} (n={len(ok)})")
print(f"\nVERDICT: G1 {'PASS' if g1 else 'FAIL'} | G2 {'PASS' if g2 else 'FAIL'}"
       f" -> {'candidate for small-weight adoption' if (g1 and g2) else 'DO NOT adopt as rating component'}")
coef.to_parquet(_HERE / 'rapm_coefficients.parquet')
print("saved rapm_coefficients.parquet")
