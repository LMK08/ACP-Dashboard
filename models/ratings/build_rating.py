#!/usr/bin/env python3
"""ACP Rating v4 — the production player rating.

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
ROLE_WEIGHTS = {
    'Striker':              {'off': 0.55,  'resp': 0.05,  'qual': 0.25,  'datt': 0.05, 'rapm': 0.10},
    'Wide Attacker':        {'off': 0.55,  'resp': 0.05,  'qual': 0.25,  'datt': 0.05, 'rapm': 0.10},
    'Advanced Midfielder':  {'off': 0.50,  'resp': 0.075, 'qual': 0.275, 'datt': 0.05, 'rapm': 0.10},
    'Deep Midfielder':      {'off': 0.45,  'resp': 0.10,  'qual': 0.30,  'datt': 0.05, 'rapm': 0.10},
    'Wide Defender':        {'off': 0.40,  'resp': 0.125, 'qual': 0.325, 'datt': 0.05, 'rapm': 0.10},
    'Central Defender':     {'off': 0.325, 'resp': 0.15,  'qual': 0.375, 'datt': 0.05, 'rapm': 0.10},
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
GPA_CATS = ['Shooting Value', 'Passing Value', 'Receiving Value',
             'Dribbling Value', 'Dead-Ball Value']
SHRINK_K = 900.0          # minutes shrink toward role-mean (0.5), as defr_adj
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
_off_blend = pd.Series(0.0, index=df.index)
for role, sub in df.groupby('role'):
    shares = np.array([sub[c + '90'].abs().mean() for c in GPA_CATS])
    shares = shares / max(shares.sum(), 1e-9)
    rels = []
    for c in GPA_CATS:
        P = []
        for pid, gg in sub.sort_values('yr').groupby('playerId'):
            rr = gg[[c + '_pct', 'yr']].to_dict('records')
            for a, b in zip(rr, rr[1:]):
                if b['yr'] - a['yr'] == 1:
                    P.append((a[c + '_pct'], b[c + '_pct']))
        P = pd.DataFrame(P)
        rels.append(_pr(P[0], P[1])[0] if len(P) >= 30 else 0.15)
    rels_c = np.clip(np.array(rels), 0.05, None)
    # v4.2: 50/50 blend of ROLE value-shares and PLAYER value-shares —
    # pure role shares dilute specialists (a 49%-dead-ball winger gets
    # the role's 14% weight on his best stable skill); pure player
    # shares are noisy. Shrink halfway.
    pv = sub[[c + '90' for c in GPA_CATS]].abs()
    pshare = pv.div(pv.sum(axis=1).replace(0, 1), axis=0)
    wrow = (0.5 * shares + 0.5 * pshare) * rels_c
    wrow = wrow.div(wrow.sum(axis=1).replace(0, 1), axis=0)
    _off_blend.loc[sub.index] = sum(
        wrow[c + '90'] * sub[c + '_pct'] for c in GPA_CATS)
df['off_blend'] = _off_blend
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

print("[2/4] role-weighted blend + minutes shrink…", flush=True)
AXES = {'off': 'off_pct', 'resp': 'defr_pct', 'qual': 'qual_pct',
         'datt': 'datt_pct', 'rapm': 'rapm_pct'}
W = pd.DataFrame([ROLE_WEIGHTS.get(ro, DEFAULT_W) for ro in df['role']],
                   index=df.index)
raw = sum(W[a] * df[col] for a, col in AXES.items())        # 0-1
# shrink the player-vs-role deviation toward 0.5 by minutes reliability
shrink = df['mins_played'] / (df['mins_played'] + SHRINK_K)
df['acp_rating'] = (0.5 + (raw - 0.5) * shrink) * 100.0

print("[3/4] recency-weighted career rating…", flush=True)
car_rows = []
for pid, gg in df.groupby('playerId'):
    latest = gg['yr'].max()
    w = gg['mins_played'].to_numpy() * (CAREER_DECAY ** (latest - gg['yr'].to_numpy()))
    if w.sum() <= 0:
        continue
    car_rows.append({'playerId': pid,
                      'acp_rating_career': float((gg['acp_rating'] * w).sum() / w.sum()),
                      'n_seasons': int(gg['seasonId'].nunique())})
df = df.merge(pd.DataFrame(car_rows), on='playerId', how='left')

out_cols = ['playerId', 'seasonId', 'name', 'role', 'side', 'league',
              'position_group', 'mins_played', 'off_pct', 'defr_pct',
              'dwae_pct', 'qual_pct', 'datt_pct', 'duel_pct', 'rapm_pct', 'acp_rating', 'acp_rating_career',
              'n_seasons']
out = df[out_cols].copy()
out['rating_version'] = 'v4.1'
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
