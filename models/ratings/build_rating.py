#!/usr/bin/env python3
"""ACP Rating v5 — the production player rating.

v2 adds the DUEL component (Glicko ladders, models/duels/): a player's
trait ratings (aerial/takeon/stopper/shield/press, >=30 contests each)
percentiled within role x league and volume-weighted into one duel
score. Career-level trait merged on playerId, like RAPM. Weights
rebalanced; duel quality now enters via both DWAE (season-level,
static-expectation) and the ladders (career-level, opponent-adjusted).

Architecture settled by the GPA audit + ratings harness + gated RAPM:
  - GPA possession value is the on-ball ENGINE (use Total OFFENSIVE
    Value — the clean part; GPA's defensive 'Interrupting' is near-noise
    at the player level, YoY r~0.07, so it is NOT the defensive axis).
  - Defence enters OUTSIDE the engine via DefR (workload above role
    expectation) + DWAE (contested-duel quality) — both validated.
  - RAPM v3 is a small "intangibles" nudge (off-ball impact; passed its
    gates; near-orthogonal to GPA/DefR).
  - Everything is normalised within OBSERVED ROLE x LEAGUE (Futi:
    "scaled by tactical role") so a CB and a striker are comparable.
  - Single-season percentiles are noisy (harness), so the headline
    rating is minutes-shrunk and a recency-weighted CAREER rating is the
    trustworthy scouting number.
  - Bespoke template scores are NOT in the rating (demoted to role-fit).

Blend (role-percentiles, 0-1, weights sum to 1):
    0.50 offence (Total Offensive Value)
    0.20 DefR workload (defr_adj)
    0.15 DWAE quality (defr_dwae_p90)
    0.15 RAPM v3 intangibles
Weights are deliberately fixed + documented (v2 candidate: role-specific
weighting). Role-percentile means a striker's defensive axes compare him
to other strikers, so low-defending roles aren't penalised — only
players unusual FOR THEIR ROLE move on those axes.

Outputs models/ratings/acp_rating_per_player_season.parquet:
    acp_rating            single-season, minutes-shrunk, 0-100
    acp_rating_career     recency-weighted across seasons, 0-100
    off_pct/defr_pct/dwae_pct/rapm_pct  the components (0-1, role-fair)
    role, side, n_seasons

Run from the Dashboard dir: python models/ratings/build_rating.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

_HERE = Path(__file__).resolve().parent
_DASH = _HERE.parent.parent
SEASON_YR = {188221: 2021, 188222: 2022, 189147: 2023, 190090: 2024,
              191782: 2025, 190230: 2023, 191779: 2025}
CAMP = {190230, 191779}
# v3 ROLE-SPECIFIC weights over 5 axes (Lucas-confirmed design):
#   off  = GPA Total Offensive (role-pct)
#   resp = DefR — defensive RESPONSIBILITY ABOVE EXPECTATION (not raw
#           volume: presence-conditioned expected vs actual)
#   qual = defensive quality, MERGED dwae+defensive-ladders (r=0.43
#           overlap — two measurements of one skill, one weight)
#   datt = take-on/shield ladders. NOT redundant with off: within-role
#           r = -0.05..+0.11 — press resistance & 1v1 are skills the
#           possession-value engine doesn't capture. Small weight.
#   rapm = intangibles.
# Quality > responsibility in EVERY role (Lucas: doing it well beats
# doing lots of it).
# v4 (Lucas): qual ~3x resp everywhere; datt trimmed to 0.05 flat; rapm
# HELD at 0.10 — raising a career-level one-number trait mechanically
# inflates measured YoY (same value both seasons) without adding truth,
# and RAPM's own gates passed only marginally (split-half 0.31).
# off weights HELD at v3 levels (off is the noisiest season axis — the
# first v4 draft raised them and rating YoY fell 0.41->0.30; reverted);
# the weight freed from resp/datt goes to QUAL per Lucas.
# v5.2 post outcome-audit (commit 7f8ce5d): resp KEPT small (micro-
# anchored, YoY 0.41, orthogonal to qual — the team-level null is the
# test's blindness, not the metric's); qual UP (cleanest defensive
# outcome link, -0.38 xGA; ceiling acknowledged — it only measures
# contested defence); datt UP for ball-playing roles (links both ends
# +0.38 xGF/-0.21 xGA, measured non-overlap with off); rapm HELD 0.10
# (gate-earned; split-half 0.31 caps it).
ROLE_WEIGHTS = {
    'Striker':              {'off': 0.525, 'resp': 0.05,  'qual': 0.25,  'datt': 0.075, 'rapm': 0.10},
    'Wide Attacker':        {'off': 0.525, 'resp': 0.05,  'qual': 0.25,  'datt': 0.075, 'rapm': 0.10},
    'Advanced Midfielder':  {'off': 0.50,  'resp': 0.05,  'qual': 0.275, 'datt': 0.075, 'rapm': 0.10},
    'Deep Midfielder':      {'off': 0.45,  'resp': 0.075, 'qual': 0.30,  'datt': 0.075, 'rapm': 0.10},
    'Wide Defender':        {'off': 0.40,  'resp': 0.10,  'qual': 0.35,  'datt': 0.05,  'rapm': 0.10},
    'Central Defender':     {'off': 0.35,  'resp': 0.10,  'qual': 0.40,  'datt': 0.05,  'rapm': 0.10},
}
DEFAULT_W = {'off': 0.45, 'resp': 0.10, 'qual': 0.30, 'datt': 0.05, 'rapm': 0.10}
# v4 offence axis: reliability x relevance weighted GPA CATEGORY blend.
# Measured (2026-06): 53% of striker offence is Shooting Value at YoY
# 0.09 — finishing variance, not skill; receiving/dribbling/set-piece
# craft repeat far better. Category weight = value-share x max(YoY,.05),
# estimated per role on all seasons (meta-parameters, noted in-sample).
# v4.1: big-4 + ONE merged Dead-Ball category. Separately the four
# dead-ball cats are zero-inflated (percentile noise -> excluded in the
# first v4), but exclusion DELETED real skill: dead-ball delivery is
# 14-26% of wide/AM role offence (Joao Pais: 51%!) and the MOST
# repeatable skill measured (corners YoY 0.44-0.56). Merging the four
# into one category concentrates the non-zero mass.
GPA_RAW_CATS = ['Shooting Value', 'Passing Value', 'Receiving Value',
                 'Dribbling Value', 'Set Piece Value', 'Corner Value',
                 'Free Kick Value', 'Throw-In Value']
# v5.1 (Lucas): dead-ball SEPARATED from the overall rating — it is a
# specialist skill the club wants visible as its own score, not folded
# into the headline. Offence axis = big-4 open-play value only;
# setpiece_pct exported alongside.
GPA_CATS = ['Shooting Value', 'Passing Value', 'Receiving Value',
             'Dribbling Value']
SHRINK_K = 300.0          # residual blend shrink, renormalized to full season (v5.4)
CAREER_DECAY = 0.5        # recency weight = mins x 0.5^(seasons_back)
MIN_MINS = 500            # rating eligibility / percentile cohort floor

print("[1/4] assemble components…", flush=True)
g = pd.read_parquet(_DASH / 'gpa_player_season_values.parquet',
                      columns=['playerId', 'seasonId', 'name', 'position_group',
                                'mins_played', 'Total Offensive Value'] + GPA_RAW_CATS)
g['Dead-Ball Value'] = (g['Set Piece Value'] + g['Corner Value']
                          + g['Free Kick Value'] + g['Throw-In Value'])
g['playerId'] = pd.to_numeric(g['playerId'], errors='coerce').astype('Int64')
g['seasonId'] = pd.to_numeric(g['seasonId'], errors='coerce').astype('Int64')
d = pd.read_parquet(_DASH / 'models/defr/defr_per_player_season.parquet',
                      columns=['playerId', 'seasonId', 'defr_adj', 'defr_dwae_p90'])
r = pd.read_parquet(_DASH / 'models/roles/role_assignments_season.parquet',
                      columns=['playerId', 'seasonId', 'primary_role_name', 'side'])
rapm = pd.read_parquet(_HERE / 'rapm_v3_coefficients.parquet')   # one coef/player
duels = pd.read_parquet(_DASH / 'models/duels/duel_ratings.parquet')
duels = duels[(duels['playerId'] > 0) & (duels['n'] >= 30)]
duelw = duels.pivot_table(index='playerId', columns='trait',
                            values=['rating', 'n'], aggfunc='first')

df = (g.merge(d, on=['playerId', 'seasonId'], how='left')
        .merge(r, on=['playerId', 'seasonId'], how='left')
        .merge(rapm, on='playerId', how='left'))
TRAITS = ['aerial', 'takeon', 'stopper', 'shield', 'press']
for t in TRAITS:
    df[f'duel_{t}'] = df['playerId'].map(duelw[('rating', t)])
    df[f'dueln_{t}'] = df['playerId'].map(duelw[('n', t)])
df['role'] = df['primary_role_name'].fillna(df['position_group'])
df['league'] = np.where(df['seasonId'].isin(CAMP), 'CAMP', 'L3')
df['yr'] = df['seasonId'].map(SEASON_YR)
df = df[(df['mins_played'] >= MIN_MINS) & (df['position_group'] != 'GK')
          & df['role'].notna()].copy()
print(f"  {len(df):,} eligible player-seasons (>= {MIN_MINS} min, outfield)")


def role_pct(col):
    """Percentile within (role x league x season); NaN inputs -> 0.5
    (neutral) so a missing component doesn't punish or reward."""
    s = df.groupby(['role', 'league', 'seasonId'])[col].rank(pct=True)
    return s.fillna(0.5)


# --- v4 offence: category blend ------------------------------------------
from scipy.stats import pearsonr as _pr
for c in GPA_CATS:
    df[c + '90'] = df[c] / df['mins_played'] * 90
    df[c + '_pct'] = role_pct(c + '90')
# v5 offence: combine in VALUE space, not percentile space. GPA
# categories are all in goals, so they SUM — percentile-averaging is
# ordinal arithmetic that caps specialists (the root cause of the
# v4 share-weighting machinery, now deleted). Each category deviation
# from the cohort mean is shrunk by its per-role rank reliability
# (regression-to-mean: best estimate of true value = r x observed),
# then summed in goals/90 and percentiled ONCE for display.
_off_adj = pd.Series(0.0, index=df.index)
_lam_log = []
for role, sub in df.groupby('role'):
    for c in GPA_CATS:
        P = []
        for pid, gg in sub.sort_values('yr').groupby('playerId'):
            rr = gg[[c + '_pct', 'yr']].to_dict('records')
            for a, b in zip(rr, rr[1:]):
                if b['yr'] - a['yr'] == 1:
                    P.append((a[c + '_pct'], b[c + '_pct']))
        P = pd.DataFrame(P)
        lam = _pr(P[0], P[1])[0] if len(P) >= 30 else 0.15
        lam = float(np.clip(lam, 0.05, 1.0))
        dev = (sub[c + '90']
                 - sub.groupby(['league', 'seasonId'])[c + '90'].transform('mean'))
        _off_adj.loc[sub.index] += lam * dev
        _lam_log.append({'role': role, 'cat': c.replace(' Value', ''),
                           'lam': round(lam, 2),
                           'infl': lam * float(dev.std())})
df['off_blend'] = _off_adj
# standalone set-piece score (not in the rating)
df['Dead-Ball Value90'] = df['Dead-Ball Value'] / df['mins_played'] * 90
df['setpiece_pct'] = role_pct('Dead-Ball Value90')
df['off_pct'] = role_pct('off_blend')   # re-uniform within role x league x season
df['off_total_pct'] = role_pct('Total Offensive Value')   # kept for reference
df['defr_pct'] = role_pct('defr_adj')
df['dwae_pct'] = role_pct('defr_dwae_p90')
df['rapm_pct'] = role_pct('rapm_v3')
# duel composites split by side: defensive ladders feed QUALITY (merged
# with DWAE); take-on/shield are their own small axis
def duel_composite(traits):
    num = 0.0; den = 0.0
    for t in traits:
        pct_t = df.groupby(['role', 'league', 'seasonId'])[f'duel_{t}'].rank(pct=True)
        w_t = df[f'dueln_{t}'].fillna(0.0) * pct_t.notna()
        num = num + pct_t.fillna(0.5) * w_t
        den = den + w_t
    return np.where(den > 0, num / den.replace(0, 1), 0.5)

df['ddef_pct'] = duel_composite(['aerial', 'stopper', 'press'])
# composites are averages of percentiles -> their spread is compressed
# (an average of two uniforms cannot reach the tails), so RE-PERCENTILE
# them within role x league x season before weighting (v4.1 fix: the
# heaviest defensive axis could never say "elite").
df['datt_raw'] = duel_composite(['takeon', 'shield'])
df['datt_pct'] = role_pct('datt_raw')
df['qual_raw'] = (df['dwae_pct'] + df['ddef_pct']) / 2.0
df['qual_pct'] = role_pct('qual_raw')
df['duel_pct'] = df['ddef_pct']    # kept for backward compat in exports

_lt = pd.DataFrame(_lam_log)
_lt['infl_share'] = _lt['infl'] / _lt.groupby('role')['infl'].transform('sum')
print("  off-axis lambda (reliability shrink) and influence share:")
print(_lt.pivot(index='role', columns='cat', values='lam').to_string())
print((_lt.pivot(index='role', columns='cat', values='infl_share') * 100)
        .round(0).to_string())

# v5.3 per-component shrinkage: reliability scales with minutes
# DIFFERENTLY per axis (off YoY 0.09 at 500-900min vs 0.21 at 1500+;
# qual nearly flat 0.48->0.58 because its inputs are already shrunk at
# source: DWAE/defr EB-shrunk, ladders RD-shrunk, rapm ridge-shrunk).
# So: shrink the OFF axis individually (K=2000) where low-minute noise
# actually leaks in, and lighten the blanket blend shrink (900->300)
# to avoid double-shrinking the source-shrunk axes.
# v5.4 form KEPT after Lucas's reexamination of universal (v5.3)
# shrinkage. Fixed-target A/B (predicting face-value next-season
# rating with each input version): v5.4 inputs career 0.497 /
# replacement 0.507; universal inputs 0.499 / 0.482 — a tie on carry,
# v5.4 better for the shipped replacement form. Universal shrinkage's
# apparent YoY/predictive edge (0.56 vs 0.47) is a SMOOTHED-TARGET
# artifact: shrunk ratings are easier to predict because they are
# shrunk. Conclusion: full seasons at face value; prediction-side
# shrinkage lives in the projection only.
_anchor = 2500.0 / (2500.0 + 2000.0)
_s_off = np.minimum((df['mins_played'] / (df['mins_played'] + 2000.0)) / _anchor, 1.0)
df['off_pct'] = 0.5 + (df['off_pct'] - 0.5) * _s_off

print("[2/4] role-weighted blend + minutes shrink…", flush=True)
AXES = {'off': 'off_pct', 'resp': 'defr_pct', 'qual': 'qual_pct',
         'datt': 'datt_pct', 'rapm': 'rapm_pct'}
W = pd.DataFrame([ROLE_WEIGHTS.get(ro, DEFAULT_W) for ro in df['role']],
                   index=df.index)
raw = sum(W[a] * df[col] for a, col in AXES.items())        # 0-1
# shrink the player-vs-role deviation toward 0.5 by minutes reliability
_ab = 2500.0 / (2500.0 + SHRINK_K)
shrink = np.minimum((df['mins_played'] / (df['mins_played'] + SHRINK_K)) / _ab, 1.0)
df['acp_rating'] = (0.5 + (raw - 0.5) * shrink) * 100.0

print("[3/4] recency-weighted career rating…", flush=True)
car_rows = []
for pid, gg in df.sort_values('yr').groupby('playerId'):
    gg = gg.reset_index(drop=True)
    latest = gg['yr'].max()
    w = gg['mins_played'].to_numpy() * (CAREER_DECAY ** (latest - gg['yr'].to_numpy()))
    car = float((gg['acp_rating'] * w).sum() / w.sum()) if w.sum() > 0 else np.nan
    for i in range(len(gg)):
        h = gg.iloc[:i + 1]
        wa = h['mins_played'].to_numpy() * (CAREER_DECAY ** (gg.loc[i, 'yr'] - h['yr'].to_numpy()))
        car_rows.append({'playerId': pid, 'seasonId': gg.loc[i, 'seasonId'],
                           'acp_rating_career': car,   # full career (display; has lookahead on old rows)
                           'career_asof': float((h['acp_rating'] * wa).sum() / wa.sum()),
                           'n_seasons': int(gg['seasonId'].nunique())})
df = df.merge(pd.DataFrame(car_rows), on=['playerId', 'seasonId'], how='left')

out_cols = ['playerId', 'seasonId', 'name', 'role', 'side', 'league',
              'position_group', 'mins_played', 'off_pct', 'defr_pct',
              'dwae_pct', 'qual_pct', 'datt_pct', 'duel_pct', 'rapm_pct', 'setpiece_pct', 'acp_rating', 'acp_rating_career',
              'n_seasons']
out = df[out_cols].copy()
out['rating_version'] = 'v5.4'
out.to_parquet(_HERE / 'acp_rating_per_player_season.parquet')
print(f"  saved acp_rating_per_player_season.parquet ({len(out):,} rows)")

print("[4/4] validation…", flush=True)


def yoy(col):
    P = []
    for pid, gg in df.sort_values('yr').groupby('playerId'):
        rr = gg.dropna(subset=[col]).to_dict('records')
        for a, b in zip(rr, rr[1:]):
            if b['yr'] - a['yr'] == 1 and a['position_group'] == b['position_group']:
                P.append((a[col], b[col]))
    P = pd.DataFrame(P, columns=['a', 'b'])
    return pearsonr(P['a'], P['b'])[0], len(P)

r_s, n = yoy('acp_rating')
print(f"  acp_rating YoY (same pos): r = {r_s:.3f} (n={n})")
# component contribution check: corr of each component with the rating
cur = df[df['seasonId'].isin({191782, 191779})]
print("  component corr with rating (25/26):")
for c in ['off_pct', 'defr_pct', 'qual_pct', 'datt_pct', 'rapm_pct']:
    print(f"    {c:<10} {pearsonr(cur[c], cur['acp_rating'])[0]:+.2f}")
print(f"\n  Top 12 by career rating (>=2 seasons):")
top = (df.drop_duplicates('playerId')
         .query('n_seasons >= 2')
         .nlargest(12, 'acp_rating_career'))
for _, x in top.iterrows():
    print(f"    {str(x['name'])[:22]:<24} {x['role'][:18]:<19} {x['side']} "
           f"career={x['acp_rating_career']:.0f}  "
           f"(off {x['off_pct']*100:.0f} def {x['defr_pct']*100:.0f} "
           f"dwae {x['dwae_pct']*100:.0f} rapm {x['rapm_pct']*100:.0f})")
print(f"\n  Top 8 current-season (25/26) by acp_rating:")
for _, x in cur.nlargest(8, 'acp_rating').iterrows():
    print(f"    {str(x['name'])[:22]:<24} {x['role'][:18]:<19} {x['league']:<5} "
           f"rating={x['acp_rating']:.0f}")
