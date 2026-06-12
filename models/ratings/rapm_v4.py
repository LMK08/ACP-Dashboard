#!/usr/bin/env python3
"""xG-RAPM v4 — per-(player, season), leak-free walk-forward (Lucas).

v3 produced ONE career coefficient per player, recency-anchored at 2025
and merged onto every season — so its YoY was ~0.97 by construction and
old seasons' ratings contained future information. v4 makes RAPM fluid:

  - one coefficient per (player, season): for season-year T, fit on all
    segments with yr <= T, sample-weighted by mins x DECAY^(T - yr).
    No future data ever enters a season's estimate.
  - Bayesian shrinkage comes from the ridge prior toward 0 (= league
    average): a player with a thin window shrinks hard, a full-sample
    player barely moves. DECAY and ALPHA chosen by WALK-FORWARD
    validation (fit through T, predict season T+1 segments), not by
    self-stability.

Segment construction identical to v3 (true lineup intervals).

Run from the Dashboard dir: python models/ratings/rapm_v4.py
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
ALPHAS = [1000.0, 3000.0, 10000.0, 30000.0]
DECAYS = [1.0, 0.85, 0.7, 0.5]
CAMP_SEASONS = {190230, 191779}
SEASON_YR = {188221: 2021, 188222: 2022, 189147: 2023, 190090: 2024,
              191782: 2025, 190230: 2023, 191779: 2025}
YRS = [2021, 2022, 2023, 2024, 2025]

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
print(f"  {len(segs):,} segments from {segs['matchId'].nunique():,} matches; "
       f"median {segs['w'].median():.1f} min")

players = sorted([p for p, m in total_min.items() if m >= MIN_PLAYER_MIN])
pidx = {p: i for i, p in enumerate(players)}
REPL_H, REPL_A = len(players), len(players) + 1
NCOL = len(players) + 2
print(f"  {len(players):,} player columns + 2 replacement buckets")

rows, colsx, vals = [], [], []
for i, r in enumerate(segs.itertuples(index=False)):
    for p in r.home_p:
        rows.append(i); colsx.append(pidx.get(p, REPL_H)); vals.append(1.0)
    for p in r.away_p:
        rows.append(i); colsx.append(pidx.get(p, REPL_A)); vals.append(-1.0)
X_ALL = sparse.csr_matrix((vals, (rows, colsx)), shape=(len(segs), NCOL))
CTX = ['camp', 'score_diff', 'man_adv']
YR_ARR = segs['yr'].to_numpy(float)


def fit_window(T, decay, alpha):
    """Fit context + ridge on segments with yr <= T, decayed toward T."""
    m = YR_ARR <= T
    sub = segs[m]
    wts = (sub['w'] * (decay ** (T - sub['yr']))).to_numpy()
    s1 = LinearRegression()
    s1.fit(sub[CTX], sub['y'], sample_weight=wts)
    res = sub['y'].to_numpy() - s1.predict(sub[CTX])
    rg = Ridge(alpha=alpha, fit_intercept=False)
    rg.fit(X_ALL[np.flatnonzero(m)], res, sample_weight=wts)
    return s1, rg.coef_


print("[3/6] walk-forward selection of (decay, alpha)…", flush=True)
best = None
for decay in DECAYS:
    for alpha in ALPHAS:
        se = wsum = 0.0
        for T in YRS[:-1]:
            nxt = YR_ARR == T + 1
            if not nxt.any():
                continue
            s1, coef = fit_window(T, decay, alpha)
            sub = segs[nxt]
            pred = s1.predict(sub[CTX]) + X_ALL[np.flatnonzero(nxt)] @ coef
            w = sub['w'].to_numpy()
            se += float(np.sum(w * (sub['y'].to_numpy() - pred) ** 2))
            wsum += float(np.sum(w))
        wmse = se / wsum
        print(f"  decay={decay:<5} alpha={alpha:>6.0f}  walk-forward wMSE={wmse:.4f}")
        if best is None or wmse < best[2]:
            best = (decay, alpha, wmse)
DECAY, ALPHA, WF = best
print(f"  chosen: decay={DECAY}, alpha={ALPHA:.0f} (wMSE {WF:.4f})")

print("[4/6] per-season coefficients (leak-free)…", flush=True)
iv['seasonId'] = iv['matchId'].map(seasons)
pmins = iv.groupby(['playerId', 'seasonId'])['mins'].sum().reset_index()
out_rows = []
for sid, T in SEASON_YR.items():
    pres = pmins[(pmins['seasonId'] == sid) & (pmins['mins'] > 0)]
    if pres.empty:
        continue
    _, coef = fit_window(T, DECAY, ALPHA)
    for p in pres['playerId']:
        if p in pidx:
            out_rows.append({'playerId': p, 'seasonId': sid,
                               'rapm_v4': float(coef[pidx[p]])})
out = pd.DataFrame(out_rows)
print(f"  {len(out):,} player-season coefficients across {out['seasonId'].nunique()} seasons")

print("[5/6] gates…", flush=True)
# split-half on the 2025 window (compare v3 career split-half 0.378)
mins_arr = np.array([total_min.get(p, 0) for p in players])
mask = mins_arr >= 1600
halves = []
for par in (0, 1):
    m = (YR_ARR <= 2025) & (segs['matchId'].astype('int64').to_numpy() % 2 == par)
    sub = segs[m]
    wts = (sub['w'] * (DECAY ** (2025 - sub['yr']))).to_numpy()
    s1 = LinearRegression(); s1.fit(sub[CTX], sub['y'], sample_weight=wts)
    res = sub['y'].to_numpy() - s1.predict(sub[CTX])
    rg = Ridge(alpha=ALPHA, fit_intercept=False)
    rg.fit(X_ALL[np.flatnonzero(m)], res, sample_weight=wts)
    halves.append(rg.coef_[:len(players)])
r_sh = pearsonr(halves[0][mask], halves[1][mask])[0]
print(f"  split-half (2025 window, n={int(mask.sum())}): r = {r_sh:.3f}  (v3 career: 0.378)")

# honest YoY of the per-season coefficient (v3 was 0.97 by construction)
t = out.copy()
t['yr'] = t['seasonId'].map(SEASON_YR)
P = []
for pid, gg in t.sort_values('yr').groupby('playerId'):
    rr = gg.to_dict('records')
    for a, b in zip(rr, rr[1:]):
        if b['yr'] - a['yr'] == 1:
            P.append((a['rapm_v4'], b['rapm_v4']))
P = pd.DataFrame(P)
print(f"  per-season RAPM YoY: r = {pearsonr(P[0], P[1])[0]:.3f} (n={len(P)}) — "
       f"honest persistence, no longer 0.97-by-construction")

print("[6/6] orthogonality (25/26)…", flush=True)
g = pd.read_parquet(_DASH / 'gpa_player_season_values.parquet',
                      columns=['playerId', 'seasonId', 'mins_played',
                                'position_group', 'Total Value'])
g['playerId'] = pd.to_numeric(g['playerId'], errors='coerce')
d = pd.read_parquet(_DASH / 'models/defr/defr_per_player_season.parquet',
                      columns=['playerId', 'seasonId', 'defr_adj', 'defr_dwae_p90'])
cur = (g[(g['seasonId'].isin({191782, 191779})) & (g['mins_played'] >= 800)
           & (g['position_group'] != 'GK')]
         .merge(d, on=['playerId', 'seasonId'], how='left')
         .merge(out, on=['playerId', 'seasonId'], how='inner'))
for c in ['Total Value', 'defr_adj', 'defr_dwae_p90']:
    ok = cur.dropna(subset=[c, 'rapm_v4'])
    print(f"  corr(RAPM_v4, {c:<14}) = {spearmanr(ok['rapm_v4'], ok[c])[0]:+.3f}")

out.to_parquet(_HERE / 'rapm_v4_per_season.parquet')
print(f"\nsaved rapm_v4_per_season.parquet (decay={DECAY}, alpha={ALPHA:.0f})")
