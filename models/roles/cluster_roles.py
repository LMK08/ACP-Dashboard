#!/usr/bin/env python3
"""Player roles — Phase 1: learn the role taxonomy + per-match assignment.

Pipeline:
  1. Season-level signatures (>=600 min) -> NMF on the 80-dim two-channel
     maps (non-negative 'zones of operation') + def_share -> z-scored
     feature vectors.
  2. k-means sweep (k=6..14): silhouette + seed-stability (ARI). The
     final k is chosen for legibility among the statistical leaders —
     NOT forced to Futi's 6.
  3. Assign EVERY match signature to its nearest role centroid: the
     per-match role is the atomic unit. A player's season role is the
     DISTRIBUTION of their match roles (primary = most frequent), so
     players who change roles between games are described honestly.
  4. Validation: split-half role agreement (odd vs even matches),
     match-to-match stickiness, confusion vs lineup positions.
  5. Outputs:
       role_model.joblib                 (nmf, scaler, kmeans, names)
       role_assignments_season.parquet   (primary role + match-role mix)
       role_assignments_match.parquet    (per-match roles)
       plots_role_atlas.png              (mean maps + exemplars per role)

Run from the Dashboard dir: python models/roles/cluster_roles.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

_HERE = Path(__file__).resolve().parent
_DASH = _HERE.parent.parent
NX, NY = 8, 5
MAP_COLS = [f'ip_{i:02d}' for i in range(NX * NY)] + \
            [f'op_{i:02d}' for i in range(NX * NY)]
# v2 — within-team-match percentile ranks (teammate-relative position):
# scale-free orderings (deepest pivot, wide-CB vs wing-back) on top of the
# absolute maps. Measured: team context is only ~2% of player-x variance
# in these leagues, so ranks are a sharpener, not a fix.
RANK_COLS = ['rank_x_ip', 'rank_x_op', 'rank_yf_ip']
N_NMF = 10
DEF_SHARE_WEIGHT = 1.5
TAXONOMY_MIN_MINS = 600
RANDOM_STATE = 7

sea = pd.read_parquet(_HERE / 'role_features_season.parquet')
mat = pd.read_parquet(_HERE / 'role_features_match.parquet')
print(f"season rows {len(sea):,}  match rows {len(mat):,}")

# ---- 1. taxonomy sample + features ----------------------------------------
tax = sea[(sea['mins_played'].fillna(0) >= TAXONOMY_MIN_MINS)].copy()
X_maps = tax[MAP_COLS].fillna(0).to_numpy()
nmf = NMF(n_components=N_NMF, init='nndsvda', random_state=RANDOM_STATE,
            max_iter=600)
W = nmf.fit_transform(X_maps)
feat = np.column_stack([W, tax['def_share'].fillna(0).to_numpy(),
                          tax[RANK_COLS].fillna(0.5).to_numpy()])
scaler = StandardScaler().fit(feat)
Z = scaler.transform(feat)
Z[:, N_NMF] *= DEF_SHARE_WEIGHT
print(f"taxonomy sample: {len(tax):,} player-seasons, NMF reconstruction "
       f"err {nmf.reconstruction_err_:.3f}")

# ---- 2. k sweep -------------------------------------------------------------
print("\nk sweep (silhouette | seed-stability ARI):")
results = {}
for k in range(6, 15):
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(Z)
    sil = silhouette_score(Z, km.labels_, sample_size=3000,
                             random_state=RANDOM_STATE)
    aris = []
    for seed in (11, 23, 47):
        km2 = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z)
        aris.append(adjusted_rand_score(km.labels_, km2.labels_))
    results[k] = (sil, np.mean(aris), km)
    print(f"  k={k:2d}  sil={sil:.3f}  ARI={np.mean(aris):.3f}")

# pick: best silhouette among k with ARI >= 0.75 (stable solutions only)
stable = {k: v for k, v in results.items() if v[1] >= 0.75}
if not stable:
    stable = results
K = max(stable, key=lambda k: stable[k][0])
sil, ari, km = results[K]
print(f"\nchosen k={K} (sil={sil:.3f}, ARI={ari:.3f})")
tax['role_id'] = km.labels_

# ---- programmatic naming hints ---------------------------------------------
def role_hint(g):
    return (f"x_ip={g['x_ip'].mean():.0f} x_op={g['x_op'].mean():.0f} "
             f"yf={g['yf_ip'].mean():.0f} def%={g['def_share'].mean()*100:.0f} "
             f"box%={g['box_share_ip'].mean()*100:.0f}")

print("\nrole centroids (naming hints):")
hints = {}
for rid, g in tax.groupby('role_id'):
    pos_mix = g['raw_pos'].value_counts(normalize=True).head(3)
    hints[rid] = role_hint(g)
    print(f"  role {rid}: n={len(g):4d}  {hints[rid]}  "
           f"pos: {dict((p, round(v, 2)) for p, v in pos_mix.items())}")

# ---- 3. per-match assignment -------------------------------------------------
def transform(df):
    Wm = nmf.transform(df[MAP_COLS].fillna(0).to_numpy())
    f = np.column_stack([Wm, df['def_share'].fillna(0).to_numpy(),
                           df[RANK_COLS].fillna(0.5).to_numpy()])
    Zm = scaler.transform(f)
    Zm[:, N_NMF] *= DEF_SHARE_WEIGHT
    return Zm

Zm = transform(mat)
d = km.transform(Zm)                     # distances to centroids
mat['role_id'] = d.argmin(axis=1)
mat['role_margin'] = np.sort(d, axis=1)[:, 1] - np.sort(d, axis=1)[:, 0]

# season distribution over match roles
def season_mix(g):
    mix = g['role_id'].value_counts(normalize=True)
    out = {'n_matches': len(g),
            'primary_role': int(mix.index[0]),
            'primary_share': float(mix.iloc[0])}
    for rid in range(K):
        out[f'role_share_{rid}'] = float(mix.get(rid, 0.0))
    return pd.Series(out)

season_roles = (mat.groupby(['playerId', 'seasonId'])
                   .apply(season_mix).reset_index())
season_roles['primary_role'] = season_roles['primary_role'].astype(int)

# attach side attribute + season-map fallback role for low-match players
sea_z = transform(sea)
sea['season_role'] = km.predict(sea_z)
# Wyscout y: LOW y = left flank (verified against lineup positions)
side = np.where(sea['y_signed'] < -6, 'L',
          np.where(sea['y_signed'] > 6, 'R', 'C'))
sea['side'] = side
season_roles = season_roles.merge(
    sea[['playerId', 'seasonId', 'season_role', 'side', 'raw_pos',
          'mins_played', 'name']],
    on=['playerId', 'seasonId'], how='left')

# ---- 4. validation -----------------------------------------------------------
print("\n=== validation ===")
# split-half: aggregate odd/even match maps, assign, compare
val = []
for (pid, sid), g in mat.groupby(['playerId', 'seasonId']):
    if len(g) < 10:
        continue
    g = g.sort_values('matchId').reset_index(drop=True)
    halves = []
    for par in (0, 1):
        h = g[g.index % 2 == par]
        m = h[MAP_COLS + ['def_share'] + RANK_COLS].mean().to_frame().T
        halves.append(km.predict(transform(m))[0])
    val.append((halves[0], halves[1]))
val = np.array(val)
agree = (val[:, 0] == val[:, 1]).mean()
print(f"split-half role agreement (>=10 matches): {agree*100:.1f}% (n={len(val)})  "
       f"[chance ≈ {100/K:.0f}%]")

# match-to-match stickiness
trans = []
for (pid, sid), g in mat.groupby(['playerId', 'seasonId']):
    r = g.sort_values('matchId')['role_id'].to_numpy()
    if len(r) >= 2:
        trans.append((r[:-1] == r[1:]).mean())
print(f"match-to-match same-role rate: {np.mean(trans)*100:.1f}%")

# confusion vs lineup buckets
BUCKET = {'CB': 'CB', 'LCB': 'CB', 'RCB': 'CB', 'LCB3': 'CB', 'RCB3': 'CB',
           'LB': 'FB', 'RB': 'FB', 'LB5': 'FB', 'RB5': 'FB', 'LWB': 'FB', 'RWB': 'FB',
           'CMF': 'CM', 'LCMF': 'CM', 'RCMF': 'CM', 'LCMF3': 'CM', 'RCMF3': 'CM',
           'DMF': 'CM', 'LDMF': 'CM', 'RDMF': 'CM',
           'AMF': 'AM_WG', 'LAMF': 'AM_WG', 'RAMF': 'AM_WG', 'LMF': 'AM_WG',
           'RMF': 'AM_WG', 'LW': 'AM_WG', 'RW': 'AM_WG', 'LWF': 'AM_WG', 'RWF': 'AM_WG',
           'CF': 'ST', 'SS': 'ST'}
season_roles['bucket'] = season_roles['raw_pos'].map(BUCKET)
ct = pd.crosstab(season_roles['primary_role'], season_roles['bucket'],
                   normalize='index')
print("\nrole x lineup-bucket (row-normalized):")
print((ct * 100).round(0).to_string())

# ---- naming: carry names from the previous model when clusters correspond ---
ROLE_FALLBACK = None
try:
    prev = pd.read_parquet(_HERE / 'role_assignments_season.parquet')
    if 'primary_role_name' in prev.columns:
        j = season_roles.merge(prev[['playerId', 'seasonId', 'primary_role_name']]
                                  .rename(columns={'primary_role_name': 'prev_name'}),
                                  on=['playerId', 'seasonId'], how='inner')
        name_map = (j.groupby('primary_role')['prev_name']
                       .agg(lambda s: s.mode().iloc[0]).to_dict())
        if len(set(name_map.values())) == K:
            ROLE_FALLBACK = name_map
            print(f"\nauto-named from previous model: {name_map}")
        else:
            print(f"\n[warn] cluster->name mapping not 1:1 ({name_map}) — "
                   f"name manually from the atlas")
except Exception as e:
    print(f"[warn] no previous names to carry: {e}")
if ROLE_FALLBACK:
    season_roles['primary_role_name'] = season_roles['primary_role'].map(ROLE_FALLBACK)
    season_roles['season_role_name'] = season_roles['season_role'].map(ROLE_FALLBACK)
    mat['role_name'] = mat['role_id'].map(ROLE_FALLBACK)

# ---- 5. save + atlas ----------------------------------------------------------
joblib.dump({'nmf': nmf, 'scaler': scaler, 'kmeans': km, 'k': K,
              'def_share_weight': DEF_SHARE_WEIGHT, 'map_cols': MAP_COLS,
              'rank_cols': RANK_COLS, 'n_nmf': N_NMF,
              'role_names': ROLE_FALLBACK, 'hints': hints},
             _HERE / 'role_model.joblib')
season_roles.to_parquet(_HERE / 'role_assignments_season.parquet')
keep_m = ['playerId', 'matchId', 'seasonId', 'role_id', 'role_margin',
            'n_ip', 'n_op'] + (['role_name'] if ROLE_FALLBACK else [])
mat[keep_m].to_parquet(_HERE / 'role_assignments_match.parquet')
print(f"\nsaved model + assignments (k={K})")

# atlas: mean IP/OOP map per role + exemplars
fig, axes = plt.subplots(K, 3, figsize=(13, 2.6 * K),
                           gridspec_kw={'width_ratios': [1, 1, 1.4]})
for rid in range(K):
    g = tax[tax['role_id'] == rid]
    for c, ch in enumerate(['ip', 'op']):
        m = g[[f'{ch}_{i:02d}' for i in range(NX * NY)]].mean().to_numpy()
        ax = axes[rid, c]
        ax.imshow(m.reshape(NX, NY).T, origin='lower', aspect='auto',
                   cmap='Blues' if ch == 'ip' else 'Reds')
        ax.set_xticks([]); ax.set_yticks([])
        if rid == 0:
            ax.set_title('In possession' if ch == 'ip' else 'Out of possession',
                          fontsize=10)
        if c == 0:
            ax.set_ylabel(f'role {rid}\nn={len(g)}', fontsize=9)
    ax = axes[rid, 2]
    ax.axis('off')
    ex = (g.sort_values('mins_played', ascending=False)['name']
            .dropna().head(6).tolist())
    ax.text(0, 0.85, hints[rid], fontsize=8, color='#444', family='monospace')
    ax.text(0, 0.62, '  •  '.join(ex[:3]), fontsize=9)
    ax.text(0, 0.40, '  •  '.join(ex[3:6]), fontsize=9)
    pos_mix = g['raw_pos'].value_counts(normalize=True).head(3)
    ax.text(0, 0.12, 'lineup: ' + ', '.join(f'{p} {v*100:.0f}%'
              for p, v in pos_mix.items()), fontsize=8, color='#666')
fig.suptitle(f'Role atlas — k={K} (x: own goal → opp goal, y: centre → wide)',
              fontsize=13)
plt.tight_layout()
plt.savefig(_HERE / 'plots_role_atlas.png', dpi=110, bbox_inches='tight')
print("saved plots_role_atlas.png")
