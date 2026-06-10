#!/usr/bin/env python3
"""ACP Rating v1 — the production player rating.

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
WEIGHTS = {'off_pct': 0.50, 'defr_pct': 0.20, 'dwae_pct': 0.15, 'rapm_pct': 0.15}
SHRINK_K = 900.0          # minutes shrink toward role-mean (0.5), as defr_adj
CAREER_DECAY = 0.5        # recency weight = mins x 0.5^(seasons_back)
MIN_MINS = 500            # rating eligibility / percentile cohort floor

print("[1/4] assemble components…", flush=True)
g = pd.read_parquet(_DASH / 'gpa_player_season_values.parquet',
                      columns=['playerId', 'seasonId', 'name', 'position_group',
                                'mins_played', 'Total Offensive Value'])
g['playerId'] = pd.to_numeric(g['playerId'], errors='coerce').astype('Int64')
g['seasonId'] = pd.to_numeric(g['seasonId'], errors='coerce').astype('Int64')
d = pd.read_parquet(_DASH / 'models/defr/defr_per_player_season.parquet',
                      columns=['playerId', 'seasonId', 'defr_adj', 'defr_dwae_p90'])
r = pd.read_parquet(_DASH / 'models/roles/role_assignments_season.parquet',
                      columns=['playerId', 'seasonId', 'primary_role_name', 'side'])
rapm = pd.read_parquet(_HERE / 'rapm_v3_coefficients.parquet')   # one coef/player

df = (g.merge(d, on=['playerId', 'seasonId'], how='left')
        .merge(r, on=['playerId', 'seasonId'], how='left')
        .merge(rapm, on='playerId', how='left'))
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


df['off_pct'] = role_pct('Total Offensive Value')
df['defr_pct'] = role_pct('defr_adj')
df['dwae_pct'] = role_pct('defr_dwae_p90')
df['rapm_pct'] = role_pct('rapm_v3')

print("[2/4] blend + minutes shrink…", flush=True)
raw = sum(WEIGHTS[c] * df[c] for c in WEIGHTS)              # 0-1
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
              'dwae_pct', 'rapm_pct', 'acp_rating', 'acp_rating_career',
              'n_seasons']
out = df[out_cols].copy()
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
for c in WEIGHTS:
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
