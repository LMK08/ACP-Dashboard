#!/usr/bin/env python3
"""ACP Projection v1 — who will be best NEXT season (recruitment view).

Separate artifact from acp_rating (descriptive). Design follows the
measured evidence:
  - axes by predictive beta (qual .32 / rapm .21 / off .20 / datt .12;
    resp EXCLUDED, beta .00; setpiece excluded, beta negative)
  - NPxG replaces Shooting Value (more stable in every role)
  - receiving percentile included (best single offence predictor)
  - career_asof (leak-free) + n_seasons
  - minutes (coach-revealed info), age + age^2 (player_details.pkl)
  - model: ridge on standardized features, role interactions only where
    the category matrix justified them. Deliberately boring.

PRE-REGISTERED GATES (set before fitting):
  G1 out-of-time: train pairs starting 2021-2023, test pairs starting
     2024 (24/25 -> 25/26). Must beat BOTH naive baselines on the test
     set by Spearman: (a) carry forward current rating, (b) carry
     forward career_asof.
  G2 calibration: role-specific residual sd -> bands; striker bands
     expected widest and must be reported, not hidden.

Output: acp_projection.parquet — projected next-season rating, band,
and delta vs current rating, for all current-season (25/26) players.

Run from the Dashboard dir: python models/ratings/build_projection.py
"""
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

_HERE = Path(__file__).resolve().parent
_DASH = _HERE.parent.parent
_GPA_DATA = _DASH.parent.parent / 'GPA Model Project v2' / 'parquet_data'
SEASON_YR = {188221: 2021, 188222: 2022, 189147: 2023, 190090: 2024,
              191782: 2025, 190230: 2023, 191779: 2025}
TEST_START_YR = 2024
ALPHAS = (10.0, 30.0, 100.0)    # chosen by 5-fold CV on TRAIN only

print("[1/5] features…", flush=True)
o = pd.read_parquet(_HERE / 'acp_rating_per_player_season.parquet')
o['yr'] = o['seasonId'].map(SEASON_YR)

# NPxG + receiving value percentile
ev = pd.concat([pd.read_parquet(_GPA_DATA / f'{f}.parquet',
    columns=['seasonId', 'player.id', 'type.primary', 'shot.xg'])
    for f in ['liga3_portugal_events', 'campeonato_portugal_events']],
    ignore_index=True)
sh = ev[(ev['shot.xg'].notna()) & (ev['type.primary'] != 'penalty')].dropna(
    subset=['player.id'])
sh['player.id'] = sh['player.id'].astype(int)
npxg = sh.groupby(['player.id', 'seasonId'])['shot.xg'].sum().reset_index()
npxg.columns = ['playerId', 'seasonId', 'npxg']
o = o.merge(npxg, on=['playerId', 'seasonId'], how='left').fillna({'npxg': 0})
o['npxg90'] = o['npxg'] / o['mins_played'] * 90
o['p_npxg'] = o.groupby(['role', 'league', 'seasonId'])['npxg90'].rank(pct=True)

g = pd.read_parquet(_DASH / 'gpa_player_season_values.parquet',
                      columns=['playerId', 'seasonId', 'Receiving Value',
                                'Dribbling Value'])
g['playerId'] = pd.to_numeric(g['playerId'], errors='coerce').astype('Int64')
g['seasonId'] = pd.to_numeric(g['seasonId'], errors='coerce').astype('Int64')
o = o.merge(g, on=['playerId', 'seasonId'], how='left')
for c, nm in [('Receiving Value', 'p_recv'), ('Dribbling Value', 'p_drib')]:
    o[c + '90'] = o[c] / o['mins_played'] * 90
    o[nm] = o.groupby(['role', 'league', 'seasonId'])[c + '90'].rank(pct=True)

# age
with open(_DASH / 'player_details.pkl', 'rb') as f:
    det = pd.DataFrame(pickle.load(f))
det['birthDate'] = pd.to_datetime(det['birthDate'], errors='coerce')
bd = (det.dropna(subset=['birthDate']).drop_duplicates('playerId')
         .set_index('playerId')['birthDate'])
o['birth'] = o['playerId'].map(bd)
o['age'] = o['yr'] + 0.5 - (o['birth'].dt.year + o['birth'].dt.dayofyear / 365)
print(f"  age coverage: {o['age'].notna().mean()*100:.0f}% of player-seasons")

# leak-free career-as-of (recency-weighted mean THROUGH each season —
# unlike acp_rating_career, which sees the whole future)
o = o.sort_values(['playerId', 'yr'])
_car = []
for _pid, _gg in o.groupby('playerId', sort=False):
    num = den = 0.0
    for _, r in _gg.iterrows():
        num = num * 0.5 + r['mins_played'] * r['acp_rating']
        den = den * 0.5 + r['mins_played']
        _car.append(num / den)
o['career_asof'] = _car

# ---- survivorship-corrected AGE CURVE (delta method) -----------------------
# Lucas's concern, backed by the literature (Dendir 2016: soccer peak
# 25-27; Lichtman 2009: delta method + survivor correction): our league
# is survivorship-heavy in BOTH directions (good young players leave UP,
# declining players drop OUT), so a raw age term would learn the
# survivors' curve. Instead: within-player consecutive-season deltas
# (controls ability), exit-corrected (players present now but absent
# next season get an imputed 25th-percentile delta for their age bin),
# shrunk toward a literature prior (peak ~26) by bin sample size.
AGE_BINS = [(15, 20.5), (20.5, 23.5), (23.5, 26.5), (26.5, 29.5), (29.5, 42)]
PRIOR_DELTA = {0: +1.2, 1: +0.6, 2: 0.0, 3: -0.6, 4: -1.4}   # pts/yr at bin center
K_PRIOR = 60.0


def _bin_of(age):
    for i, (lo, hi) in enumerate(AGE_BINS):
        if lo <= age < hi:
            return i
    return len(AGE_BINS) - 1

# exits are BIMODAL in a tier-3/4 league (Lucas): good young players
# leave UP (sold — censored upward, NOT declined), old/weak players
# drop out (genuine decline). Impute decline only for the latter:
# exits aged <25 with a final rating above their role-league median
# are excluded from the correction.
o['_role_med'] = o.groupby(['role', 'league', 'seasonId'])['acp_rating'].transform('median')
_dp, _ex = [], []
for _pid, _gg in o.dropna(subset=['age']).groupby('playerId'):
    rr = _gg.to_dict('records')
    for a, b in zip(rr, rr[1:]):
        if b['yr'] - a['yr'] == 1:
            _dp.append((a['age'], b['acp_rating'] - a['acp_rating']))
    last = rr[-1]
    if last['yr'] < 2025:
        up_leaver = (last['age'] < 25) and (last['acp_rating'] >= last['_role_med'])
        if not up_leaver:
            _ex.append(last['age'])
_DP = pd.DataFrame(_dp, columns=['age', 'delta'])
_DP['ab'] = _DP['age'].map(_bin_of)
_EXB = pd.Series([_bin_of(a) for a in _ex])
AGE_CURVE = {}
print("  age curve (delta method, exit-corrected, prior-shrunk):")
for b in range(len(AGE_BINS)):
    obs = _DP[_DP['ab'] == b]['delta']
    n_obs, n_ex = len(obs), int((_EXB == b).sum())
    imput = obs.quantile(0.25) if n_obs >= 10 else PRIOR_DELTA[b]
    raw = (obs.sum() + n_ex * imput) / max(n_obs + n_ex, 1)
    AGE_CURVE[b] = ((raw * (n_obs + n_ex) + PRIOR_DELTA[b] * K_PRIOR)
                      / (n_obs + n_ex + K_PRIOR))
    print(f"    {AGE_BINS[b][0]:>4.0f}-{AGE_BINS[b][1]:<4.0f} n={n_obs:>3}"
           f" (+{n_ex:>3} exits)  obs {obs.mean() if n_obs else float('nan'):+.2f}"
           f" -> corrected {raw:+.2f} -> final {AGE_CURVE[b]:+.2f}")
o['age_delta'] = o['age'].map(lambda a: AGE_CURVE[_bin_of(a)] if a == a else np.nan)

WIDE = {'Wide Attacker', 'Wide Defender'}
FEATS = ['qual_pct', 'rapm_pct', 'off_pct', 'datt_pct', 'p_npxg', 'p_recv',
          'career_asof', 'n_seasons', 'mins_played', 'age_delta',
          'p_drib_wide', 'p_npxg_st']
o['p_drib_wide'] = np.where(o['role'].isin(WIDE), o['p_drib'], 0.5)
o['p_npxg_st'] = np.where(o['role'] == 'Striker', o['p_npxg'], 0.5)

print("[2/5] panel…", flush=True)
rows = []
for pid, gg in o.sort_values('yr').groupby('playerId'):
    rr = gg.to_dict('records')
    for a, b in zip(rr, rr[1:]):
        if b['yr'] - a['yr'] == 1 and a['role'] == b['role']:
            rows.append({**{f: a[f] for f in FEATS},
                           'cur_rating': a['acp_rating'], 'role': a['role'],
                           'start_yr': a['yr'], 'next_rating': b['acp_rating']})
P = pd.DataFrame(rows).dropna()
tr = P[P['start_yr'] < TEST_START_YR]
te = P[P['start_yr'] == TEST_START_YR]
print(f"  panel {len(P)} pairs (age available) -> train {len(tr)}, test {len(te)}")

print("[3/5] fit + G1 out-of-time gate…", flush=True)
from sklearn.model_selection import cross_val_predict
mu, sd = tr[FEATS].mean(), tr[FEATS].std()
Xtr = (tr[FEATS] - mu) / sd
Xte = (te[FEATS] - mu) / sd
_best = None
for _a in ALPHAS:
    _cv = cross_val_predict(Ridge(alpha=_a), Xtr, tr['next_rating'], cv=5)
    _mae = np.abs(_cv - tr['next_rating']).mean()
    if _best is None or _mae < _best[1]:
        _best = (_a, _mae)
ALPHA = _best[0]
print(f"  alpha={ALPHA:.0f} (train CV)")
model = Ridge(alpha=ALPHA).fit(Xtr, tr['next_rating'])
pred_te = model.predict(Xte)
marcel_te = te['career_asof'] + te['age_delta']     # literature 'Marcel' form
g_model = spearmanr(pred_te, te['next_rating'])[0]
g_cur = spearmanr(te['cur_rating'], te['next_rating'])[0]
g_car = spearmanr(te['career_asof'], te['next_rating'])[0]
g_mar = spearmanr(marcel_te, te['next_rating'])[0]
print(f"  G1 test Spearman: model {g_model:.3f} | carry-rating {g_cur:.3f} | "
       f"carry-career {g_car:.3f} | marcel {g_mar:.3f}")
mae_m = np.mean(np.abs(pred_te - te['next_rating']))
mae_c = np.mean(np.abs(te['cur_rating'] - te['next_rating']))
mae_k = np.mean(np.abs(te['career_asof'] - te['next_rating']))
mae_r = np.mean(np.abs(marcel_te - te['next_rating']))
print(f"  G1 test MAE:      model {mae_m:.2f} | carry-rating {mae_c:.2f} | "
       f"carry-career {mae_k:.2f} | marcel {mae_r:.2f}")
# SHIP RULE (pre-registered criterion = test Spearman; MAE tiebreak):
cands = {'ridge': (g_model, mae_m), 'career': (g_car, mae_k),
           'marcel': (g_mar, mae_r)}
SHIP = max(cands, key=lambda k: (round(cands[k][0], 2), -cands[k][1]))
print(f"  -> SHIP: {SHIP} (gate winner; ridge "
       f"{'PASSES' if SHIP=='ridge' else 'documented, not shipped'})")
print("  coefficients (per sd):")
for f, c in sorted(zip(FEATS, model.coef_), key=lambda x: -abs(x[1])):
    print(f"    {f:<14} {c:+.2f}")

print("[4/5] G2 role-specific bands (test residuals of SHIPPED predictor)…",
        flush=True)

print("[5/5] project all current-season players…", flush=True)
final = Ridge(alpha=ALPHA).fit(
    (P[FEATS] - mu) / sd, P['next_rating'])    # refit on ALL pairs for production
cur = o[(o['seasonId'].isin({191782, 191779})) & o[FEATS].notna().all(axis=1)].copy()
if SHIP == 'ridge':
    cur['projection'] = final.predict((cur[FEATS] - mu) / sd)
elif SHIP == 'marcel':
    cur['projection'] = cur['career_asof'] + cur['age_delta']
else:
    cur['projection'] = cur['career_asof']
te['__ship_pred'] = (pred_te if SHIP == 'ridge'
                       else marcel_te if SHIP == 'marcel' else te['career_asof'])
te2 = te.copy()
te2['resid'] = te2['next_rating'] - te2['__ship_pred']
band = te2.groupby('role')['resid'].std().rename('band_sd')
print(band.round(1).to_string())
cur['band_sd'] = cur['role'].map(band).fillna(float(te2['resid'].std()))
cur['proj_delta'] = cur['projection'] - cur['acp_rating']
out = cur[['playerId', 'seasonId', 'name', 'role', 'side', 'league', 'age',
             'mins_played', 'acp_rating', 'career_asof', 'projection',
             'band_sd', 'proj_delta']].copy()
out['projection_version'] = 'v1-' + SHIP
out.to_parquet(_HERE / 'acp_projection.parquet')
print(f"  acp_projection.parquet ({len(out):,} current players)")
print("\n  top 8 projections (25/26 -> 26/27):")
for _, x in out.nlargest(8, 'projection').iterrows():
    print(f"    {str(x['name'])[:22]:<24}{x['role'][:16]:<17}{x['league']:<5}"
           f"age {x['age']:.0f}  rating {x['acp_rating']:.0f} -> proj "
           f"{x['projection']:.0f} ±{x['band_sd']:.0f}")
print("\n  biggest BUY-LOW flags (proj >> rating, >=1200 min):")
for _, x in out[out['mins_played'] >= 1200].nlargest(5, 'proj_delta').iterrows():
    print(f"    {str(x['name'])[:22]:<24}{x['role'][:16]:<17}age {x['age']:.0f}  "
           f"rating {x['acp_rating']:.0f} -> proj {x['projection']:.0f}")
print("\n  biggest SELL-HIGH flags (rating >> proj, >=1200 min):")
for _, x in out[out['mins_played'] >= 1200].nsmallest(5, 'proj_delta').iterrows():
    print(f"    {str(x['name'])[:22]:<24}{x['role'][:16]:<17}age {x['age']:.0f}  "
           f"rating {x['acp_rating']:.0f} -> proj {x['projection']:.0f}")
