#!/usr/bin/env python3
"""ACP Projection v1 — who will be best NEXT season (recruitment view).

Separate artifact from acp_rating (descriptive). Design follows the
measured evidence:
  - axes by predictive beta (qual .32 / rapm .21 / off .20 / datt .12;
    resp EXCLUDED, beta .00; setpiece excluded, beta negative)
  - NPxG replaces Shooting Value (more stable in every role)
  - receiving percentile included (best single offence predictor)
  - career_asof (leak-free); n_seasons dropped in v7.0 (a whole-career count —
    it told a 2021 row how many seasons the player WOULD play)
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
import sys as _sys_lc
_sys_lc.path.insert(0, str(Path(__file__).resolve().parents[2]))
from league_config import SEASON_YEAR, CAMP_SEASON_IDS, IN_PROGRESS_SEASON_IDS  # noqa: E402
import json
import pickle
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

_HERE = Path(__file__).resolve().parent
_DASH = _HERE.parent.parent
_GPA_DATA = _DASH.parent.parent / 'GPA Model Project v2' / 'parquet_data'
SEASON_YR = SEASON_YEAR   # from league_config (every season, incl. the current one)
TEST_START_YR = None   # set from the panel below (the last completed pair year)
# v2.2 lookahead test hook (tests/test_projection_no_lookahead.py), as in
# build_rating: ACP_MAX_YEAR drops every season after that year from every
# input; ACP_ENGINE_OUT reads the rating from and writes the projections to
# that directory.
import os as _os
_MAX_YR = int(_os.environ['ACP_MAX_YEAR']) if _os.environ.get('ACP_MAX_YEAR') else None
_OUT = Path(_os.environ.get('ACP_ENGINE_OUT', str(_HERE)))


def _cut(frame, col='seasonId'):
    """frame truncated to seasons <= ACP_MAX_YEAR when it is set."""
    if _MAX_YR is None:
        return frame
    yr = pd.to_numeric(frame[col], errors='coerce').map(SEASON_YEAR)
    return frame[yr <= _MAX_YR].copy()
ALPHAS = (10.0, 30.0, 100.0)    # chosen by 5-fold CV on TRAIN only

print("[1/5] features…", flush=True)
o = _cut(pd.read_parquet(_OUT / 'acp_rating_per_player_season.parquet'))
# The projection is an off-season model: "current" = the last COMPLETED
# season, projected one season ahead. A season still in progress is rated
# for the dashboard but is neither a training pair nor a projection base
# (it would age players twice and treat them as lapsed — review 2026-10-21).
# Its rows DO get features (per-season percentiles and the leak-free career
# loop never look forward, so completed rows are unchanged): the AS-OF
# projections at the end use them; everything else runs on `o` = completed.
o = o.copy()
o['_live'] = o['seasonId'].isin(IN_PROGRESS_SEASON_IDS)
o['yr'] = o['seasonId'].map(SEASON_YR)

# NPxG + receiving value percentile
ev = _cut(pd.concat([pd.read_parquet(_GPA_DATA / f'{f}.parquet',
    columns=['seasonId', 'player.id', 'type.primary', 'shot.xg', 'team.name'])
    for f in ['liga3_portugal_events', 'campeonato_portugal_events']],
    ignore_index=True))
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
g = _cut(g)
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
# age at SEASON MIDPOINT (Jan 1 of the spanning winter) — the
# standard convention (Lucas caught season-START ages reading ~1yr
# young vs players' current age)
o['age'] = o['yr'] + 1.0 - (o['birth'].dt.year + o['birth'].dt.dayofyear / 365)
print(f"  age coverage: {o['age'].notna().mean()*100:.0f}% of player-seasons")

# leak-free career-as-of (recency-weighted mean THROUGH each season —
# unlike acp_rating_career, which sees the whole future)
o = o.sort_values(['playerId', 'yr'])
# v2 career loop: decay per SEASON STEP, not per row — same-year rows
# (winter movers with a row in each league) pool at full weight, and a
# missed season decays by the true year gap. Every row of a season
# carries the season's full pooled evidence.
_car, _e5, _e8, _e9, _e10 = [], [], [], [], []
for _pid, _gg in o.groupby('playerId', sort=False):
    num = den = d8 = d9 = d10 = 0.0
    prev_yr = None
    for _yr, _gy in _gg.groupby('yr', sort=True):
        gap = 1 if prev_yr is None else int(_yr - prev_yr)
        num *= 0.5 ** gap; den *= 0.5 ** gap; d8 *= 0.8 ** gap
        d9 *= 0.9 ** gap
        _m = float(_gy['mins_played'].sum())
        num += float((_gy['mins_played'] * _gy['acp_rating']).sum())
        den += _m
        d8 += _m
        d9 += _m
        d10 += _m
        _car.extend([num / den] * len(_gy))
        _e5.extend([den] * len(_gy)); _e8.extend([d8] * len(_gy))
        _e9.extend([d9] * len(_gy)); _e10.extend([d10] * len(_gy))
        prev_yr = _yr
o['career_asof'] = _car
# evidence-of-identity decays SLOWER than form (Lucas): three decay
# variants; the projection picks one by train CV below
o['eff_mins'] = _e5
o['eff8'] = _e8
o['eff9'] = _e9
o['eff10'] = _e10

# ---- the career as ONE continuous sample (models/ratings/career.py) ----------
# Lucas 2026-10-29: predict from all of a player's data as a continuous
# entity. The shipped form's two inputs — career_asof and the evidence (eff9)
# — are rebuilt from season BLOCKS placed at their minute-weighted match date
# and decayed by TIME before each row's as-of time (a completed season: its
# last match; the season in progress: the data date), with the old per-season
# rates made continuous (0.5/yr form, 0.9/yr evidence). Replay-tested
# 2026-10-29: as accurate as the season steps, ~20% smaller jump when a live
# value starts. eff_mins / eff8 / eff10 (the unshipped ridge's) stay stepped.
from models.ratings import career as _career  # noqa: E402
from models.ratings import asof as _asof  # noqa: E402
_ms_c = _cut(pd.read_parquet(_DASH / 'matches_summary.parquet', columns=['matchId', 'seasonId', 'dateutc', 'status']))
_mm_c = pd.read_parquet(_DASH / 'match_player_minutes.parquet', columns=['matchId', 'playerId', 'minutes'])
_mm_c = _mm_c[_mm_c['matchId'].isin(_ms_c['matchId'])]
_cen, _smid = _career.block_centroids(_mm_c, _ms_c)
_ends = _career.season_end_times(_ms_c, SEASON_YR)
_live_t = _ends.get(max(SEASON_YR[s] for s in IN_PROGRESS_SEASON_IDS)) if IN_PROGRESS_SEASON_IDS else None
_tref = o['yr'].map(_ends)
if _live_t is not None:
    _tref = _tref.where(~o['_live'], _live_t)
# ONE SCALE (v2.2, 2026-10-29): every block goes onto the Liga 3 scale
# (acp_rating_abs: its own season's descriptive Campeonato gap — so
# league_conversion runs BEFORE this script), is averaged, and the career is
# put back on the row's own league scale (+ the row's own gap). v2.1 averaged
# raw ratings across leagues: a Campeonato 50 counted as a Liga 3 50.
if 'acp_rating_abs' in o.columns:
    _blk = o[['playerId', 'seasonId', 'yr', 'mins_played']].assign(acp_rating=o['acp_rating_abs'])
    _own_gap = (o['acp_rating'] - o['acp_rating_abs']).fillna(0.0)
else:
    print("  !! acp_rating_abs missing (league_conversion not run first): career blocks on raw ratings")
    _blk, _own_gap = o[['playerId', 'seasonId', 'yr', 'mins_played', 'acp_rating']], 0.0
_cf = _career.career_features(_blk, _cen, _smid, _tref)
o['career_asof'] = _cf['career_asof'] + _own_gap
# the CAREER role (2026-10-29): the role a player is valued at is the one he
# has played most up to the row's as-of time, by time-weighted minutes
# (career.career_role) — not one season's (a 196' spell at full-back no longer
# makes a winger a full-back). value_role is what the engine value reads.
o['value_role'] = _career.career_role(o[['playerId', 'seasonId', 'yr', 'mins_played', 'role']],
                                      _cen, _smid, _tref)
o['eff9'] = _cf['evidence']
o['evidence'] = _cf['evidence']
# pooled within-season minutes (winter movers carry two rows/season) —
# the "playing time is information" signal for the full2 level pull
o['season_mins'] = o.groupby(['playerId', 'yr'])['mins_played'].transform('sum')

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
for _pid, _gg in o[~o['_live']].dropna(subset=['age']).groupby('playerId'):
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
# v6.3: age curve blended by role shares (exported by the rating). v2.2
# (2026-10-29): the CAREER shares — time-weighted over the blocks up to each
# row's as-of time, like career_asof — not one season's.
_shcols = [c for c in o.columns if c.startswith('sh_')]
_cshcols = ['c' + c for c in _shcols]
o[_cshcols] = _career.time_weighted(o[['playerId', 'seasonId', 'yr', 'mins_played']], o[_shcols],
                                    _cen, _smid, _tref).to_numpy()
def _blended_age_delta(r):
    if r['age'] != r['age']:
        return np.nan
    tot = sum(r[c] for c in _cshcols if r[c] == r[c])
    if tot < 0.5:
        return std_age_delta(r['age'], r['value_role'])
    return sum(r[c] * std_age_delta(r['age'], c[4:]) for c in _cshcols if r[c] == r[c]) / tot
o['age_delta'] = o.apply(_blended_age_delta, axis=1)

WIDE = {'Wide Attacker', 'Wide Defender'}
FEATS = ['qual_pct', 'rapm_pct', 'off_pct', 'datt_pct', 'p_npxg', 'p_recv',
          'career_asof', 'mins_played', 'eff_mins', 'eff8',
          'eff9', 'eff10', 'age_delta', 'p_drib_wide', 'p_npxg_st']
o['p_drib_wide'] = np.where(o['role'].isin(WIDE), o['p_drib'], 0.5)
o['p_npxg_st'] = np.where(o['role'] == 'Striker', o['p_npxg'], 0.5)
# every row (live season included) for the as-of projections; the panel, the
# fit and today's projections use completed seasons only, as before
o_all = o
o = o_all[~o_all['_live']].copy()

print("[2/5] panel…", flush=True)
rows = []
for pid, gg in o.sort_values('yr', kind='stable').groupby('playerId'):
    rr = gg.to_dict('records')
    for a, b in zip(rr, rr[1:]):
        # panel pinned to the historic >=500' regime — sub-floor rows
        # (v6.4 season pooling) are projected but train nothing
        if (b['yr'] - a['yr'] == 1 and a['role'] == b['role']
                and a['mins_played'] >= 500 and b['mins_played'] >= 500):
            rows.append({**{f: a[f] for f in FEATS},
                           'season_mins': a['season_mins'],
                           'cur_rating': a['acp_rating'], 'role': a['role'],
                           'start_yr': a['yr'], 'next_rating': b['acp_rating']})
P = pd.DataFrame(rows).dropna()
# the G1 test pairs start in the last completed pair year (2024 today: 24/25 ->
# 25/26) — derived so a replay or rollover never ends up with an empty test set
TEST_START_YR = int(P['start_yr'].max())
tr = P[P['start_yr'] < TEST_START_YR]
te = P[P['start_yr'] == TEST_START_YR]
if tr.empty:   # one pair-year only (a replay as of 2023/24): nothing is held out
    print("  !! one pair-year: G1 runs in-sample (train = test = every pair)")
    tr = te = P
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
                  ('eff8', 2500), ('eff9', 900), ('eff9', 1500),
                  ('eff9', 2500), ('eff10', 1500), ('eff10', 2500)]:
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
print(f"  evidence grid (train CV, reference): best {_best_w[0]}/{_best_w[1]} "
       f"MAE {_best_w[2]:.3f}")
# PINNED (Lucas, 2026-06-12): the grid is FLAT — spread ~0.03 MAE against
# fold noise ±1.3, and projections move <0.6 pts across every config —
# so accuracy cannot choose. Principle does: eff9 (0.9/yr decay) makes
# an idle year cost evidence (fixes the eff10 no-decay wart), and
# K=1500 trusts the career record EARLIER (w=0.5 at 1,500 eff-mins vs
# 2,500). "Identity decays slower than form (0.5), but it does decay."
# Grid stays printed every build; revisit at the 26/27 refit.
EVID_COL, EVID_K = 'eff9', 1500.0
print(f"  evidence config PINNED: {EVID_COL}, K={EVID_K:.0f} "
       f"(principle on a flat grid — trust the record more, earlier)")
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
# full2 (Lucas): TWO-TERM evidence. With eff10 as the only w, "thin
# evidence" means SHORT CAREER — mostly young improvers — so the fitted
# convergence target drifted to ~league mean. Career evidence keeps the
# career slope; CURRENT-SEASON minutes (pooled across competitions)
# carry the level pull — that is the fringe-player signal the original
# replacement target (44.7) was measured on.
_ws_tr = (tr['season_mins'] / (tr['season_mins'] + 900.0)).values
_ws_te = (te['season_mins'] / (te['season_mins'] + 900.0)).values
_Xf2 = np.column_stack([np.ones(len(tr)), _wtr, _ws_tr,
                          tr['career_asof'].values,
                          tr['career_asof'].values * _wtr])
_cf2, *_ = np.linalg.lstsq(_Xf2, (tr['next_rating'].values
                                    - tr['age_delta'].values), rcond=None)
full2_te = (np.column_stack([np.ones(len(te)), _wte, _ws_te,
                               te['career_asof'].values,
                               te['career_asof'].values * _wte]) @ _cf2
              + te['age_delta'].values)
print(f"  full  coefs: c0 {_cf[0]:+.1f}, w_career {_cf[1]:+.1f}, "
       f"career {_cf[2]:+.3f}, career*w {_cf[3]:+.3f}")
print(f"  full2 coefs: c0 {_cf2[0]:+.1f}, w_career {_cf2[1]:+.1f}, "
       f"w_season {_cf2[2]:+.1f}, career {_cf2[3]:+.3f}, career*w {_cf2[4]:+.3f}")
for _sm in (300, 900, 2000):
    _wsx = _sm / (_sm + 900.0)
    for _wcx in (0.2, 0.6):
        _fp = ((_cf2[0] + _cf2[1] * _wcx + _cf2[2] * _wsx)
                / (1 - (_cf2[3] + _cf2[4] * _wcx)))
        print(f"    full2 convergence @ season_mins={_sm}, w_career={_wcx}: {_fp:.1f}")
print("  age-curve coefficient FIXED at c = 1.00 (literature volume)")
g_model = spearmanr(pred_te, te['next_rating'])[0]
g_cur = spearmanr(te['cur_rating'], te['next_rating'])[0]
g_car = spearmanr(te['career_asof'], te['next_rating'])[0]
g_mar = spearmanr(marcel_te, te['next_rating'])[0]
g_rep = spearmanr(repl_te, te['next_rating'])[0]
g_ful = spearmanr(full_te, te['next_rating'])[0]
g_f2 = spearmanr(full2_te, te['next_rating'])[0]
print(f"  G1 test Spearman: model {g_model:.3f} | carry-rating {g_cur:.3f} | "
       f"carry-career {g_car:.3f} | marcel {g_mar:.3f} | replacement {g_rep:.3f} | "
       f"full {g_ful:.3f} | full2 {g_f2:.3f}")
mae_m = np.mean(np.abs(pred_te - te['next_rating']))
mae_c = np.mean(np.abs(te['cur_rating'] - te['next_rating']))
mae_k = np.mean(np.abs(te['career_asof'] - te['next_rating']))
mae_r = np.mean(np.abs(marcel_te - te['next_rating']))
mae_rp = np.mean(np.abs(repl_te - te['next_rating']))
mae_f = np.mean(np.abs(full_te - te['next_rating']))
mae_f2 = np.mean(np.abs(full2_te - te['next_rating']))
print(f"  G1 test MAE:      model {mae_m:.2f} | carry-rating {mae_c:.2f} | "
       f"carry-career {mae_k:.2f} | marcel {mae_r:.2f} | replacement {mae_rp:.2f} | "
       f"full {mae_f:.2f} | full2 {mae_f2:.2f}")
# SHIP RULE (pre-registered criterion = test Spearman; MAE tiebreak):
cands = {'ridge': (g_model, mae_m), 'career': (g_car, mae_k),
           'marcel': (g_mar, mae_r), 'replacement': (g_rep, mae_rp),
           'full': (g_ful, mae_f), 'full2': (g_f2, mae_f2)}
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
# the base = the latest COMPLETED season-year (was hard-coded {191782, 191779}
# / 2025 — the same seasons today; derived so a replay or a rollover can't miss)
CUR_YR = int(o['yr'].max())
cur = o[(o['yr'] == CUR_YR) & o[FEATS].notna().all(axis=1)].copy()
cur['seasons_ago'] = 0
# Non-current players (Lucas 2026-06-24): project EVERY player with an eligible
# (>=90') season off their MOST RECENT one — not just the one-season-lapsed.
# Generalizes the lapsed handling by the idle gap to 25/26 (seasons_ago):
#   - aged to their CURRENT (25/26) age, then ONE step to the 26/27 horizon
#     (the extra-age-year removal, generalized — no missed-year aging steps);
#   - evidence is NOT decayed for idle years (Lucas 2026-06-24: minimize the
#     effect staleness has on evidence — a stale player keeps the evidence
#     weight it had at its last season; within-record recency still applies);
#   - band widened sqrt(1+seasons_ago) further down — staler = wider (staleness
#     now surfaces only as wider uncertainty, not a lower evidence weight).
_stale = o[~o['playerId'].isin(set(cur['playerId']))].copy()
# most recent eligible season per player (ties broken by minutes)
_stale = _stale.sort_values(['yr', 'mins_played']).drop_duplicates('playerId', keep='last')
# players we STILL cannot project — one or more model inputs are missing
_missing = _stale[~_stale[FEATS].notna().all(axis=1)].copy()
if len(_missing):
    _missing['_miss'] = _missing.apply(
        lambda r: ','.join(c for c in FEATS if pd.isna(r[c])), axis=1)
    print(f"  !! {len(_missing)} >=90' players NOT projected (missing model inputs):")
    for _, _m in _missing.sort_values(['seasonId', 'playerId']).iterrows():
        print(f"     MISSING|{int(_m['playerId'])}|{int(_m['seasonId'])}|"
               f"{_m.get('role')}|{_m['mins_played']:.0f}|{_m['_miss']}")
_stale = _stale[_stale[FEATS].notna().all(axis=1)].copy()
_stale['seasons_ago'] = (CUR_YR - _stale['yr']).clip(lower=1).astype(int)
_stale['age'] = _stale['age'] + _stale['seasons_ago']            # forward to current (25/26) age
_stale['age_delta'] = _stale.apply(_blended_age_delta, axis=1)   # ONE step: current age -> 26/27
# idle-year evidence decay REMOVED (Lucas 2026-06-24, "minimize"): eff_mins /
# eff8 / eff9 stay at their last-season values — staleness no longer lowers the
# evidence weight (it still widens the band via seasons_ago below).
cur = pd.concat([cur, _stale], ignore_index=True)
print(f"  +{len(_stale)} non-current players projected off their latest season "
       f"(seasons_ago {int(_stale['seasons_ago'].min()) if len(_stale) else 0}"
       f"-{int(_stale['seasons_ago'].max()) if len(_stale) else 0})")
def _ship_predict(d, cf=None):
    """The SHIPPED predictor on any feature frame (today's projections and
    the as-of ones below share it). cf: the 'full' coefficients to use (the
    as-of fits below); default the train fit."""
    if SHIP == 'ridge':
        return final.predict((d[FEATS] - mu) / sd)
    if SHIP == 'marcel':
        return (d['career_asof'] + d['age_delta']).values
    _wc = (d[EVID_COL] / (d[EVID_COL] + EVID_K)).values
    if SHIP == 'replacement':
        return (np.column_stack([np.ones(len(d)), _wc,
            d['career_asof'].values, d['career_asof'].values * _wc]) @ _cr)
    if SHIP == 'full':
        return (np.column_stack([np.ones(len(d)), _wc,
            d['career_asof'].values, d['career_asof'].values * _wc])
            @ (_cf if cf is None else cf) + d['age_delta'].values)
    if SHIP == 'full2':
        _wsc = (d['season_mins'] / (d['season_mins'] + 900.0)).values
        return (np.column_stack([np.ones(len(d)), _wc, _wsc,
            d['career_asof'].values, d['career_asof'].values * _wc]) @ _cf2
            + d['age_delta'].values)
    return d['career_asof'].values


# ---- AS-OF FITS (v2.2, 2026-10-29: no look-ahead) --------------------------
# A projection made at the end of season-year y may only use the pairs that
# had ENDED by then (start_yr < y; asof.fit_by_year). v2.1 applied the train
# fit (pairs starting 2021-23) to every as-of row, so a 2021/22 row was
# projected with coefficients fitted on later seasons, its own pair included.
# Today's projection (end of CUR_YR) now uses EVERY completed pair — the
# whole sample at our disposal; the G1 gate above stays the train/test check.
# Fewer than FIT_MIN_PAIRS -> FULL_PRIOR, a hand-set design constant:
# projection = 50·(1 - w) + w·career + age_delta (evidence-weighted pull to
# the scale's centre, 50).
FIT_MIN_PAIRS = 100
FULL_PRIOR = np.array([50.0, -50.0, 0.0, 1.0])


def _fit_full(p):
    _w = (p[EVID_COL] / (p[EVID_COL] + EVID_K)).values
    _X = np.column_stack([np.ones(len(p)), _w, p['career_asof'].values,
                          p['career_asof'].values * _w])
    return np.linalg.lstsq(_X, p['next_rating'].values - p['age_delta'].values, rcond=None)[0]


if SHIP != 'full':
    raise SystemExit(f"as-of fits are implemented for SHIP='full' only (got {SHIP!r})")
CF_BY_YR = _asof.fit_by_year(P, sorted(set(o_all['yr']) | {CUR_YR + 1}), _fit_full,
                             FULL_PRIOR, FIT_MIN_PAIRS)
for _y, _c in CF_BY_YR.items():
    _n = int((P['start_yr'] < _y).sum())
    print(f"  as-of fit {_y}: {_n} pairs" + (" -> PRIOR" if _n < FIT_MIN_PAIRS else
          f" c0 {_c[0]:+.1f} w {_c[1]:+.1f} career {_c[2]:+.3f} career*w {_c[3]:+.3f}"))
cur['projection'] = _ship_predict(cur, CF_BY_YR[CUR_YR + 1])
te = te.copy()   # slice of P — write on a copy (silences SettingWithCopyWarning)
te['__ship_pred'] = (pred_te if SHIP == 'ridge'
                       else marcel_te if SHIP == 'marcel'
                       else repl_te if SHIP == 'replacement'
                       else full_te if SHIP == 'full'
                       else full2_te if SHIP == 'full2'
                       else te['career_asof'])
te2 = te.copy()
te2['resid'] = te2['next_rating'] - te2['__ship_pred']
band = te2.groupby('role')['resid'].std().rename('band_sd')
print(band.round(1).to_string())
cur['band_sd'] = cur['value_role'].map(band).fillna(float(te2['resid'].std()))
cur['band_sd'] = cur['band_sd'] * np.sqrt(1.0 + cur['seasons_ago'])
cur['proj_delta'] = cur['projection'] - cur['acp_rating']
cur['w_evidence'] = cur[EVID_COL] / (cur[EVID_COL] + EVID_K)
out = cur[['playerId', 'seasonId', 'name', 'role', 'value_role', 'side', 'league', 'age',
             'mins_played', 'acp_rating', 'career_asof', 'age_delta',
             'w_evidence', 'projection', 'band_sd', 'proj_delta',
             'seasons_ago']].copy()
out['projection_version'] = 'v2.2-' + SHIP
# _abs (Liga 3 scale; was written by league_conversion, which now runs first):
# a Campeonato row takes today's recruit gap, an as-of row (below) the gap
# measurable by the end of ITS year
_lc_path = _OUT / 'league_conversion.json'
_lc = json.load(open(_lc_path)) if _lc_path.exists() else None
if _lc is None:
    print("  !! league_conversion.json missing: projection_abs = projection")
REC_TODAY = float(_lc['delta_pts_recruit']) if _lc else 0.0
REC_BY_YR = {int(k): float(v) for k, v in (_lc or {}).get('delta_pts_recruit_by_year', {}).items()}
out['projection_abs'] = out['projection'] - np.where(out['league'] == 'CAMP', REC_TODAY, 0.0)
out.to_parquet(_OUT / 'acp_projection.parquet')
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

# ---- AS-OF projections (live-season value, 2026-10-29) ----------------------
# The shipped predictor on EVERY player-season: next season's level from
# everything known at the end of that season — for the season in progress,
# known at the DATA DATE (its rows hold the matches played so far; the career
# rating and evidence weigh them by their minutes). A completed season keeps
# its midpoint age + one step; a live row takes his age at the data date + one
# step, so nobody is aged twice. For the per-season / live engine value;
# `projection` above (today's forecast, boards and cards) is unchanged.
# Replay-tested on 2024/25: ACP_Official/live_value_plan_2026-10-29.md.
_ms = _cut(pd.read_parquet(_DASH / 'matches_summary.parquet', columns=['seasonId', 'dateutc', 'status']))
_live_d = pd.to_datetime(_ms.loc[_ms['seasonId'].isin(IN_PROGRESS_SEASON_IDS)
                                  & _ms['status'].eq('Played'), 'dateutc'], errors='coerce', utc=True)
AS_OF = _live_d.max().tz_convert(None).normalize() if _live_d.notna().any() else None
asof = o_all.copy()
_lv = asof['_live']
if AS_OF is not None and _lv.any():
    asof.loc[_lv, 'age'] = (AS_OF - asof.loc[_lv, 'birth']).dt.days / 365.25
    asof.loc[_lv, 'age_delta'] = asof.loc[_lv].apply(_blended_age_delta, axis=1)
asof = asof[asof[FEATS].notna().all(axis=1)].copy()
asof['projection_asof'] = np.nan
for _y, _idx in asof.groupby('yr').groups.items():
    asof.loc[_idx, 'projection_asof'] = _ship_predict(asof.loc[_idx], CF_BY_YR[_y])
asof['w_evidence'] = asof[EVID_COL] / (asof[EVID_COL] + EVID_K)
# band: in-sample residual sd of the pairs behind each year's fit (by role,
# >= 20 pairs, else all); none before any pair. v2.1 used the 24/25 test
# residuals for every year.
asof['band_sd'] = np.nan
for _y, _idx in asof.groupby('yr').groups.items():
    _p = P[P['start_yr'] < _y]
    if len(_p) < FIT_MIN_PAIRS:
        continue
    _r = _p['next_rating'] - _ship_predict(_p, CF_BY_YR[_y])
    _rb = _r.groupby(_p['role']).agg(['std', 'size'])
    _rb = _rb.loc[_rb['size'] >= 20, 'std']
    asof.loc[_idx, 'band_sd'] = asof.loc[_idx, 'value_role'].map(_rb).fillna(float(_r.std())).values
asof['is_live'] = asof['_live']
asof['as_of_date'] = np.where(asof['_live'], AS_OF.strftime('%Y-%m-%d') if AS_OF is not None else None, None)
asof_out = asof[['playerId', 'seasonId', 'yr', 'name', 'role', 'value_role', 'league', 'age', 'mins_played',
                 'acp_rating', 'career_asof', 'evidence', 'age_delta', 'w_evidence', 'projection_asof',
                 'band_sd', 'is_live', 'as_of_date']].copy()
asof_out['projection_version'] = 'v2.2-' + SHIP + '-asof'


def _rec_asof(y):
    """The recruit gap measurable by the end of season-year y: the latest
    year's gap up to y; none -> NaN (never a later year's gap)."""
    _ks = [k for k in REC_BY_YR if k <= y]
    return REC_BY_YR[max(_ks)] if _ks else np.nan


asof_out['projection_asof_abs'] = asof_out['projection_asof'] - np.where(
    asof_out['league'] == 'CAMP', asof_out['yr'].map(_rec_asof), 0.0)
asof_out.to_parquet(_OUT / 'acp_projection_asof.parquet')
print(f"\n  acp_projection_asof.parquet ({len(asof_out):,} player-seasons; "
      f"{int(asof_out['is_live'].sum())} live as of {AS_OF.date() if AS_OF is not None else '—'})")

# ---- AS-OF POOLS (2026-10-29: no look-ahead) --------------------------------
# A past season's value ranks in the projection pool AS IT STOOD at the end of
# that season-year, built the way today's is (acp_projection above): every
# player of that year projected off it, plus every player seen before,
# projected off his latest season with his age carried forward (the stale
# rule) — all on the fit known then (CF_BY_YR[y]). Read by eur_intervals
# (asof_pool) for season values and the fee pairing; today's values keep
# today's pool.
_pool_rows = []
_full = o[FEATS].notna().all(axis=1)
for _y in sorted(set(o.loc[_full, 'yr'])):
    _base = o[(o['yr'] == _y) & _full].copy()
    # today's stale rule: each other player's LATEST season up to y, dropped
    # when that season can't be projected (not his latest complete one)
    _st = o[(o['yr'] <= _y) & ~o['playerId'].isin(set(_base['playerId']))]
    _st = _st.sort_values(['yr', 'mins_played'], kind='stable').drop_duplicates('playerId', keep='last')
    _st = _st[_st[FEATS].notna().all(axis=1)].copy()
    if len(_st):
        _st['age'] = _st['age'] + (_y - _st['yr']).clip(lower=1)
        _st['age_delta'] = _st.apply(_blended_age_delta, axis=1)
    _py = pd.concat([_base, _st], ignore_index=True).dropna(subset=['age_delta'])
    _pr = _ship_predict(_py, CF_BY_YR.get(_y, FULL_PRIOR))
    _ks = [k for k in REC_BY_YR if k <= _y]
    _gy = REC_BY_YR[max(_ks)] if _ks else np.nan
    _pool_rows.append(pd.DataFrame({'yr': _y, 'playerId': _py['playerId'].to_numpy(),
                                    'seasonId': _py['seasonId'].to_numpy(),
                                    'projection_abs': _pr - np.where(_py['league'] == 'CAMP', _gy, 0.0)}))
_pools = pd.concat(_pool_rows, ignore_index=True).dropna(subset=['projection_abs'])
_pools.to_parquet(_OUT / 'acp_projection_pools.parquet')
print(f"  acp_projection_pools.parquet: " + ", ".join(
    f"{int(k)} n={int(v['n'])} median {v['m']:.1f}" for k, v in
    _pools.groupby('yr')['projection_abs'].agg(n='size', m='median').iterrows()))

# ---- SALE-DATE projections (fee pairing, 2026-10-29: no look-ahead) ---------
# A sale DURING a season (before its last match) meets the value from what
# had been played before it, like a live value on that date: his earlier
# seasons, plus the sale season's row when every match of it came before the
# sale (he left our leagues), decayed and aged to the sale date, on the fit
# of the pairs that had ended by then (CF_BY_YR[y - 1]) and the Campeonato
# gap measurable then. A sale-season row with matches after the sale (a move
# within our leagues) can't be split: his earlier seasons only. A summer
# sale needs none of this: its season's as-of row already ends before it.
_fees_path = _DASH / 'valuations' / 'reported_fees.csv'
_sale_rows = []
if _fees_path.exists():
    from league_config import PLAYER_ID_ALIASES as _ALIAS
    _fees = pd.read_csv(_fees_path, comment='#')
    _md = _career.match_dates(_ms_c)
    _pm = _mm_c[_mm_c['minutes'] > 0].merge(_ms_c[['matchId', 'seasonId']], on='matchId')
    _last = _pm.assign(t=_pm['matchId'].astype(int).map(_md)).groupby(['playerId', 'seasonId'])['t'].max()
    _evt = ev.dropna(subset=['player.id', 'team.name'])
    _PTEAMS = _evt.groupby([_evt['player.id'].astype(int), _evt['seasonId'].astype(int)])['team.name'].unique().to_dict()

    def _club_key(name):
        return ' '.join(str(name or '').casefold().split())
    for _f in _fees.itertuples(index=False):
        try:
            _pid, _sid = int(_f.playerId), int(_f.season_id)
        except (TypeError, ValueError):
            continue
        _sale = pd.to_datetime(_f.as_of_date, errors='coerce')
        _y = SEASON_YR.get(_sid)
        if pd.isna(_sale) or _y is None or _ends.get(_y) is None:
            continue
        _st = _career.to_years(_sale)
        if not _st < _ends[_y]:
            continue                                   # a summer sale
        # a player with two Wyscout ids (league_config.PLAYER_ID_ALIASES): every id's rows
        _ids = {_pid} | {a for a, b in _ALIAS.items() if b == _ALIAS.get(_pid, _pid)} | {_ALIAS.get(_pid, _pid)}
        _p0 = o_all[o_all['playerId'].isin(_ids)]
        _cen_p = {(_pid, int(s_)): _cen.get((int(i_), int(s_)), np.nan) for i_, s_ in zip(_p0['playerId'], _p0['seasonId'])}
        _p = _p0.assign(playerId=_pid)
        # the sale season's row counts when he played that season for the
        # SELLING club only (he left our leagues: every match came before the
        # move — the fee file's dates are conventions, e.g. 15 Jan), or when
        # its last match came before the sale date
        _lastp = {s: max((_last.get((i, s), -np.inf) for i in _ids), default=-np.inf) for s in _p['seasonId']}
        _clubs = {s: set().union(*[set(_PTEAMS.get((i, s), ())) for i in _ids]) for s in _p['seasonId']}
        _seller = _club_key(_f.from_team)
        _only_seller = {s: bool(_clubs[s]) and all(_club_key(c) == _seller for c in _clubs[s]) for s in _clubs}
        _before = [(yy < _y) or (yy == _y and (_only_seller[s] or -np.inf < _lastp[s] < _st))
                   for yy, s in zip(_p['yr'], _p['seasonId'])]
        _p = _p[_before]
        if _p.empty:
            continue
        _tr = pd.Series(_st, index=_p.index)
        if 'acp_rating_abs' in _p.columns:
            _b = _p[['playerId', 'seasonId', 'yr', 'mins_played']].assign(acp_rating=_p['acp_rating_abs'])
            _g = (_p['acp_rating'] - _p['acp_rating_abs']).fillna(0.0)
        else:
            _b, _g = _p[['playerId', 'seasonId', 'yr', 'mins_played', 'acp_rating']], 0.0
        _c = _career.career_features(_b, _cen_p, _smid, _tr)
        _r = _p.assign(career_asof=_c['career_asof'] + _g, eff9=_c['evidence'], evidence=_c['evidence'])
        _r['value_role'] = _career.career_role(_r[['playerId', 'seasonId', 'yr', 'mins_played', 'role']],
                                               _cen_p, _smid, _tr)
        _r[_cshcols] = _career.time_weighted(_r[['playerId', 'seasonId', 'yr', 'mins_played']], _r[_shcols],
                                             _cen_p, _smid, _tr).to_numpy()
        _r = _r.sort_values(['yr', 'mins_played'], kind='stable').iloc[[-1]].copy()
        _r['age'] = (_sale - _r['birth']).dt.days / 365.25
        _r['age_delta'] = _r.apply(_blended_age_delta, axis=1)
        if _r[['career_asof', 'eff9', 'age_delta']].isna().any(axis=None):
            continue
        _proj = float(_ship_predict(_r, CF_BY_YR.get(_y - 1, FULL_PRIOR))[0])
        _gap = 0.0
        if _r['league'].iloc[0] == 'CAMP':
            _ks = [k for k in REC_BY_YR if k <= _y - 1]
            _gap = REC_BY_YR[max(_ks)] if _ks else np.nan
        _sale_rows.append({'playerId': _pid, 'season_id': _sid, 'as_of_date': str(_f.as_of_date),
                           'base_seasonId': int(_r['seasonId'].iloc[0]),
                           'used_sale_season': bool(_r['yr'].iloc[0] == _y),
                           'projection_sale': _proj, 'projection_sale_abs': _proj - _gap,
                           'age': float(_r['age'].iloc[0]), 'value_role': _r['value_role'].iloc[0],
                           'w_evidence': float(_r['eff9'].iloc[0] / (_r['eff9'].iloc[0] + EVID_K)),
                           'evidence': float(_r['evidence'].iloc[0])})
_sales = pd.DataFrame(_sale_rows, columns=['playerId', 'season_id', 'as_of_date', 'base_seasonId',
                                           'used_sale_season', 'projection_sale', 'projection_sale_abs',
                                           'age', 'value_role', 'w_evidence', 'evidence'])
_sales.to_parquet(_OUT / 'acp_projection_sales.parquet')
print(f"  acp_projection_sales.parquet ({len(_sales)} mid-season sales; "
      f"{int(_sales['used_sale_season'].sum()) if len(_sales) else 0} with the sale season's pre-sale row)")
