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

# v6.0 units fix: 'X Value' cols are per-90 already — use RAW sums and
# compute per-90 exactly once (same bug as the rating builder had).
g = pd.read_parquet(_DASH / 'gpa_player_season_values.parquet',
                      columns=['playerId', 'seasonId', 'Receiving',
                                'Dribbling'])
g['playerId'] = pd.to_numeric(g['playerId'], errors='coerce').astype('Int64')
g['seasonId'] = pd.to_numeric(g['seasonId'], errors='coerce').astype('Int64')
o = o.merge(g, on=['playerId', 'seasonId'], how='left')
for c, nm in [('Receiving', 'p_recv'), ('Dribbling', 'p_drib')]:
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
_car, _e5, _e8, _e10 = [], [], [], []
for _pid, _gg in o.groupby('playerId', sort=False):
    num = den = d8 = d10 = 0.0
    for _, r in _gg.iterrows():
        num = num * 0.5 + r['mins_played'] * r['acp_rating']
        den = den * 0.5 + r['mins_played']
        d8 = d8 * 0.8 + r['mins_played']
        d10 = d10 + r['mins_played']
        _car.append(num / den)
        _e5.append(den); _e8.append(d8); _e10.append(d10)
o['career_asof'] = _car
# evidence-of-identity decays SLOWER than form (Lucas): three decay
# variants; the projection picks one by train CV below
o['eff_mins'] = _e5
o['eff8'] = _e8
o['eff10'] = _e10

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
# v4 age curve (Lucas): INDUSTRY-STANDARD shape, not trained on our
# panel. Our 24-26 bin measured -1.9/yr — a survivorship composition
# artifact (half of 24-26 player-seasons exit; the best get sold UP to
# Liga 2+, leaving plateauers), not football aging. Literature
# consensus (Dendir 2016 peak 25-27; delta-method studies peak ~26;
# ASA aging curves): improvement tapering to a peak at 26, accelerating
# decline after. Scaled to our 17-pt-SD rating (cumulative 26->31
# ~0.3 SD decline). The projection FITS a coefficient on this curve, so
# the SHAPE is industry consensus while our data sets the magnitude.
# Per-role curves from the literature. Peaks: forwards ~25 (Dendir
# 2016), FULLBACKS ~26 — NOT with CBs: wide defenders run on repeated
# high-intensity actions and age like wingers (ESPN/macro-football
# position curves), while CENTRE-BACKS peak ~27.5 and decline only
# gently before ~31 (guile ages well) -> later peak AND flatter slope.
ROLE_PEAK = {'Striker': 25.0, 'Wide Attacker': 25.0,
              'Advanced Midfielder': 26.0, 'Deep Midfielder': 26.0,
              'Wide Defender': 26.0, 'Central Defender': 27.5}
ROLE_DECLINE = {'Central Defender': 0.25}      # default 0.35/yr per year


def std_age_delta(age, role=None):
    """Expected rating change (pts/yr) — consensus shape, role params."""
    peak = ROLE_PEAK.get(role, 26.0)
    slope = ROLE_DECLINE.get(role, 0.35)
    if age < peak:
        return min(2.0, 2.0 * (peak - age) / 7.0)
    return max(-2.5, -slope * (age - peak))

# observed bin deltas printed for REFERENCE only (not used):
_dp = []
for _pid, _gg in o.dropna(subset=['age']).groupby('playerId'):
    rr = _gg.to_dict('records')
    for a, b in zip(rr, rr[1:]):
        if b['yr'] - a['yr'] == 1:
            _dp.append((a['age'], b['acp_rating'] - a['acp_rating']))
_DP = pd.DataFrame(_dp, columns=['age', 'delta'])
_DP['ab'] = _DP['age'].map(_bin_of)
print("  age curve = INDUSTRY STANDARD (peak 26); observed bins shown for reference:")
for b in range(len(AGE_BINS)):
    obs = _DP[_DP['ab'] == b]['delta']
    mid = (AGE_BINS[b][0] + min(AGE_BINS[b][1], 36)) / 2
    print(f"    {AGE_BINS[b][0]:>4.0f}-{AGE_BINS[b][1]:<4.0f} std(mid-role) {std_age_delta(mid):+.2f}/yr"
           f"   (our panel observed {obs.mean() if len(obs) else float('nan'):+.2f}, n={len(obs)} — survivor-biased)")
o['age_delta'] = o.apply(
    lambda r: std_age_delta(r['age'], r['role']) if r['age'] == r['age'] else np.nan,
    axis=1)

WIDE = {'Wide Attacker', 'Wide Defender'}
FEATS = ['qual_pct', 'rapm_pct', 'off_pct', 'datt_pct', 'p_npxg', 'p_recv',
          'career_asof', 'n_seasons', 'mins_played', 'eff_mins', 'eff8',
          'eff10', 'age_delta', 'p_drib_wide', 'p_npxg_st']
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
# REPLACEMENT-SHRINK candidate (Lucas hypothesis, CONFIRMED on train:
# implied convergence targets 44.7 / 47.6 / 51.7 for 500-900 / 900-1500 /
# 1500+ minutes — thin evidence converges to REPLACEMENT level, not the
# mean, because playing time is information: only 23% of low-minute
# players survive to a rated next season vs 37% of starters).
# Continuous form: next ~ c0 + c1*w + c2*career + c3*career*w,
# w = mins/(mins+900). Level pull toward replacement when w is small.
# grid-select evidence decay + K on TRAIN CV (test untouched)
from sklearn.model_selection import KFold
_best_w = None
for _ec, _K in [('eff_mins', 900), ('eff_mins', 1500), ('eff8', 1500),
                  ('eff8', 2500), ('eff10', 1500), ('eff10', 2500)]:
    _w = (tr[_ec] / (tr[_ec] + _K)).values
    _X = np.column_stack([np.ones(len(tr)), _w, tr['career_asof'].values,
                            tr['career_asof'].values * _w,
                            tr['age_delta'].values])
    _maes = []
    for _itr, _iva in KFold(5, shuffle=True, random_state=0).split(_X):
        _c, *_ = np.linalg.lstsq(_X[_itr], tr['next_rating'].values[_itr],
                                   rcond=None)
        _maes.append(np.abs(_X[_iva] @ _c
                              - tr['next_rating'].values[_iva]).mean())
    _m = float(np.mean(_maes))
    if _best_w is None or _m < _best_w[2]:
        _best_w = (_ec, _K, _m)
EVID_COL, EVID_K, _ = _best_w
print(f"  evidence config (train CV): {EVID_COL}, K={EVID_K}")
_wtr = (tr[EVID_COL] / (tr[EVID_COL] + EVID_K)).values
_Xr = np.column_stack([np.ones(len(tr)), _wtr, tr['career_asof'].values,
                         tr['career_asof'].values * _wtr])
_cr, *_ = np.linalg.lstsq(_Xr, tr['next_rating'].values, rcond=None)
_wte = (te[EVID_COL] / (te[EVID_COL] + EVID_K)).values
repl_te = (np.column_stack([np.ones(len(te)), _wte, te['career_asof'].values,
                              te['career_asof'].values * _wte]) @ _cr)
# FULL form: career + replacement-pull + age curve at FIXED c = 1.0
# (Lucas): the fitted dial (was 0.21) came from the same survivor-
# biased panel that distorted the curve's shape, and marcel (c=1) beat
# the fitted form on the held-out test. Literature trusted on magnitude
# as well as shape: regress (next - age_delta) on the replacement-pull
# terms, add the curve back at full volume.
_Xf = np.column_stack([np.ones(len(tr)), _wtr, tr['career_asof'].values,
                         tr['career_asof'].values * _wtr])
_cf, *_ = np.linalg.lstsq(_Xf, (tr['next_rating'].values
                                  - tr['age_delta'].values), rcond=None)
full_te = (np.column_stack([np.ones(len(te)), _wte, te['career_asof'].values,
                              te['career_asof'].values * _wte]) @ _cf
             + te['age_delta'].values)
print("  age-curve coefficient FIXED at c = 1.00 (literature volume)")
g_model = spearmanr(pred_te, te['next_rating'])[0]
g_cur = spearmanr(te['cur_rating'], te['next_rating'])[0]
g_car = spearmanr(te['career_asof'], te['next_rating'])[0]
g_mar = spearmanr(marcel_te, te['next_rating'])[0]
g_rep = spearmanr(repl_te, te['next_rating'])[0]
g_ful = spearmanr(full_te, te['next_rating'])[0]
print(f"  G1 test Spearman: model {g_model:.3f} | carry-rating {g_cur:.3f} | "
       f"carry-career {g_car:.3f} | marcel {g_mar:.3f} | replacement {g_rep:.3f} | "
       f"full {g_ful:.3f}")
mae_m = np.mean(np.abs(pred_te - te['next_rating']))
mae_c = np.mean(np.abs(te['cur_rating'] - te['next_rating']))
mae_k = np.mean(np.abs(te['career_asof'] - te['next_rating']))
mae_r = np.mean(np.abs(marcel_te - te['next_rating']))
mae_rp = np.mean(np.abs(repl_te - te['next_rating']))
mae_f = np.mean(np.abs(full_te - te['next_rating']))
print(f"  G1 test MAE:      model {mae_m:.2f} | carry-rating {mae_c:.2f} | "
       f"carry-career {mae_k:.2f} | marcel {mae_r:.2f} | replacement {mae_rp:.2f} | "
       f"full {mae_f:.2f}")
# SHIP RULE (pre-registered criterion = test Spearman; MAE tiebreak):
cands = {'ridge': (g_model, mae_m), 'career': (g_car, mae_k),
           'marcel': (g_mar, mae_r), 'replacement': (g_rep, mae_rp),
           'full': (g_ful, mae_f)}
_gate_winner = max(cands, key=lambda k: (round(cands[k][0], 2), -cands[k][1]))
# DOCUMENTED OVERRIDE (Lucas, 2026-06-12): ship the FULL form (career +
# replacement-pull + age) even though career carry edges it on test
# rank (0.574 vs 0.550, inside noise at n=128, SE~0.07). Reason: the
# gate is structurally BLIND to the players these adjustments treat —
# test pairs require a next rated season, so the aging decliner and the
# washed-out fringe player never produce test rows (77% of low-minute
# players don't survive; old exits dominate the age tails). Recruitment
# decisions are about exactly those cohorts. The full form ties on what
# the gate can see (rank, within noise; MAE better 9.63 vs 10.22) and
# is principled on what it can't. Gate stays printed every build; if
# career beats 'full' OUTSIDE noise on the 26/27 panel, revisit.
SHIP = 'full'
print(f"  gate winner: {_gate_winner}; SHIPPED: {SHIP} "
       f"(documented survivor-blindness override, see comment)")
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
elif SHIP == 'replacement':
    _wc = (cur[EVID_COL] / (cur[EVID_COL] + EVID_K)).values
    cur['projection'] = (np.column_stack([np.ones(len(cur)), _wc,
        cur['career_asof'].values, cur['career_asof'].values * _wc]) @ _cr)
elif SHIP == 'full':
    _wc = (cur[EVID_COL] / (cur[EVID_COL] + EVID_K)).values
    cur['projection'] = (np.column_stack([np.ones(len(cur)), _wc,
        cur['career_asof'].values, cur['career_asof'].values * _wc]) @ _cf
        + cur['age_delta'].values)
else:
    cur['projection'] = cur['career_asof']
te['__ship_pred'] = (pred_te if SHIP == 'ridge'
                       else marcel_te if SHIP == 'marcel'
                       else repl_te if SHIP == 'replacement'
                       else full_te if SHIP == 'full'
                       else te['career_asof'])
te2 = te.copy()
te2['resid'] = te2['next_rating'] - te2['__ship_pred']
band = te2.groupby('role')['resid'].std().rename('band_sd')
print(band.round(1).to_string())
cur['band_sd'] = cur['role'].map(band).fillna(float(te2['resid'].std()))
cur['proj_delta'] = cur['projection'] - cur['acp_rating']
cur['w_evidence'] = cur[EVID_COL] / (cur[EVID_COL] + EVID_K)
out = cur[['playerId', 'seasonId', 'name', 'role', 'side', 'league', 'age',
             'mins_played', 'acp_rating', 'career_asof', 'age_delta',
             'w_evidence', 'projection', 'band_sd', 'proj_delta']].copy()
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
