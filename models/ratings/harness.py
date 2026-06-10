#!/usr/bin/env python3
"""Rating validation harness — evaluate every player-rating candidate on
the same battery so 'which rating?' is an evidence question.

Candidates (per player-season, >=800 min, outfield):
  gpa_total90   raw GPA Total Value /90 (possession-value engine)
  gpa_rolepct   GPA Total Value percentile within OBSERVED ROLE x season
  defr_adj      DefR workload above role expectation (minutes-shrunk)
  dwae90        defensive wins above expectation /90 (quality)
  interrupt90   GPA Interrupting Value /90 (contrast — known-noise)
  blend_pct     0.60 gpa_rolepct + 0.25 defr_pct + 0.15 dwae_pct
                  (percentiles within role x season)
  bespoke       best-role template Score (only 25/26 L3 has a current-
                  version cache -> correlation structure only)

Battery:
  A split-half (odd/even matches) where per-match values exist
  B year-over-year stability (same position-group, consecutive seasons)
  C predictive: candidate(t) -> Spearman with target(t+1):
       T1 = next-season gpa_rolepct        (future on-ball performance)
       T2 = next-season (gpa_rolepct + defr_pct)/2  (future two-axis)
  D cross-candidate correlation structure (25/26)

Run from the Dashboard dir: python models/ratings/harness.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

_HERE = Path(__file__).resolve().parent
_DASH = _HERE.parent.parent
_GPA_REPO = _DASH.parent.parent / 'GPA Model Project v2'
SEASON_YR = {188221: 2021, 188222: 2022, 189147: 2023, 190090: 2024,
              191782: 2025, 190230: 2023, 191779: 2025}

print("[1/5] assemble candidates…", flush=True)
g = pd.read_parquet(_DASH / 'gpa_player_season_values.parquet')
g['playerId'] = pd.to_numeric(g['playerId'], errors='coerce')
d = pd.read_parquet(_DASH / 'models/defr/defr_per_player_season.parquet')
r = pd.read_parquet(_DASH / 'models/roles/role_assignments_season.parquet')

df = g[['playerId', 'seasonId', 'name', 'position_group', 'mins_played',
          'Total Value', 'Total Offensive Value', 'Interrupting Value']].copy()
df = df.merge(d[['playerId', 'seasonId', 'defr_adj', 'defr_dwae_p90']],
                on=['playerId', 'seasonId'], how='left')
df = df.merge(r[['playerId', 'seasonId', 'primary_role_name']],
                on=['playerId', 'seasonId'], how='left')
df['role'] = df['primary_role_name'].fillna(df['position_group'])
df = df[(df['mins_played'] >= 800) & (df['position_group'] != 'GK')].copy()
df['yr'] = df['seasonId'].map(SEASON_YR)

def role_pct(col):
    return df.groupby(['role', 'seasonId'])[col].rank(pct=True)

df['gpa_total90'] = df['Total Value']
df['interrupt90'] = df['Interrupting Value']
df['dwae90'] = df['defr_dwae_p90']
df['gpa_rolepct'] = role_pct('Total Value')
df['defr_pct'] = role_pct('defr_adj')
df['dwae_pct'] = role_pct('defr_dwae_p90')
df['blend_pct'] = (0.60 * df['gpa_rolepct'] + 0.25 * df['defr_pct']
                     + 0.15 * df['dwae_pct'])
CANDS = ['gpa_total90', 'gpa_rolepct', 'defr_adj', 'dwae90',
           'interrupt90', 'blend_pct']
print(f"  {len(df):,} qualifying player-seasons")

# ---- A. split-half for GPA total (per-match values from the GPA repo) ----
print("[2/5] split-half (GPA per-match)…", flush=True)
try:
    av = pd.read_parquet(_GPA_REPO / 'parquet_data/events_with_action_values.parquet',
                           columns=['matchId', 'seasonId', 'player.id',
                                     'action_value', 'pass.recipient.id',
                                     'receiver_action_value'])
    actor = (av.groupby(['player.id', 'seasonId', 'matchId'])['action_value']
                .sum().reset_index().rename(columns={'player.id': 'playerId',
                                                       'action_value': 'v'}))
    recv = (av[av['receiver_action_value'] != 0]
              .groupby(['pass.recipient.id', 'seasonId', 'matchId'])
              ['receiver_action_value'].sum().reset_index()
              .rename(columns={'pass.recipient.id': 'playerId',
                                'receiver_action_value': 'v'}))
    pm = (pd.concat([actor, recv]).groupby(['playerId', 'seasonId', 'matchId'])
            ['v'].sum().reset_index())
    pm['half'] = (pm['matchId'].astype('int64') % 2)
    h = pm.groupby(['playerId', 'seasonId', 'half'])['v'].agg(['sum', 'count']).reset_index()
    a = h[h['half'] == 0]; b = h[h['half'] == 1]
    m = a.merge(b, on=['playerId', 'seasonId'], suffixes=('_a', '_b'))
    m = m[(m['count_a'] >= 8) & (m['count_b'] >= 8)]
    m['pa'] = m['sum_a'] / m['count_a']; m['pb'] = m['sum_b'] / m['count_b']
    rsh = pearsonr(m['pa'], m['pb'])[0]
    print(f"  GPA total per-match split-half: half r={rsh:.3f} -> "
           f"full {2*rsh/(1+rsh):.3f} (n={len(m)})")
except Exception as e:
    print(f"  [skip] {e}")
print("  (measured elsewhere: defr_adj 0.85, dwae 0.86, interrupt90 0.09)")

# ---- B. YoY ----
print("[3/5] year-over-year stability…", flush=True)


def yoy(col):
    P = []
    for pid, gg in df.sort_values('yr').groupby('playerId'):
        rr = gg.dropna(subset=[col]).to_dict('records')
        for x, y in zip(rr, rr[1:]):
            if y['yr'] - x['yr'] == 1 and x['position_group'] == y['position_group']:
                P.append((x[col], y[col]))
    P = pd.DataFrame(P, columns=['a', 'b'])
    if len(P) < 30:
        return np.nan, len(P)
    return pearsonr(P['a'], P['b'])[0], len(P)


for c in CANDS:
    rr, n = yoy(c)
    print(f"  {c:<14} YoY r = {rr:.3f} (n={n})")

# ---- C. predictive ----
print("[4/5] predictive (candidate t -> target t+1, Spearman)…", flush=True)
df['overall_pct'] = (df['gpa_rolepct'] + df['defr_pct']) / 2
nxt = df[['playerId', 'yr', 'position_group', 'gpa_rolepct', 'overall_pct']].copy()
nxt['yr'] = nxt['yr'] - 1
j = df.merge(nxt, on=['playerId', 'yr', 'position_group'],
               suffixes=('', '_next'))
print(f"  consecutive same-position pairs: {len(j)}")
print(f"  {'candidate':<14} {'-> T1 gpa_rolepct':>18} {'-> T2 overall':>14}")
for c in CANDS:
    ok = j.dropna(subset=[c, 'gpa_rolepct_next', 'overall_pct_next'])
    t1 = spearmanr(ok[c], ok['gpa_rolepct_next'])[0]
    t2 = spearmanr(ok[c], ok['overall_pct_next'])[0]
    print(f"  {c:<14} {t1:>18.3f} {t2:>14.3f}")

# ---- D. correlation structure + bespoke (25/26 L3 only) ----
print("[5/5] correlation structure, 25/26…", flush=True)
cur = df[df['seasonId'].isin([191782, 191779])]
print(cur[CANDS].corr(method='spearman').round(2).to_string())
try:
    pct = pd.read_parquet(_DASH / 'stats_cache/player_percentiles_v13_191782.parquet')
    score_cols = [c for c in pct.columns if c.endswith('_Score')]
    pct['bespoke'] = pct[score_cols].max(axis=1)
    pct['playerId'] = pd.to_numeric(pct['playerId'], errors='coerce')
    jb = cur[cur['seasonId'] == 191782].merge(
        pct[['playerId', 'bespoke']], on='playerId', how='inner')
    print(f"\nbespoke best-role Score vs candidates (25/26 L3, n={len(jb)}):")
    for c in CANDS:
        ok = jb.dropna(subset=[c, 'bespoke'])
        print(f"  corr(bespoke, {c:<14}) = {spearmanr(ok['bespoke'], ok[c])[0]:+.3f}")
    print("  NOTE: bespoke YoY/predictive not measurable — only 25/26 L3 has a")
    print("  current-version stats cache; would need full per-season rebuilds.")
except Exception as e:
    print(f"  bespoke unavailable: {e}")
