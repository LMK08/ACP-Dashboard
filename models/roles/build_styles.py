#!/usr/bin/env python3
"""Player styles — archetypes WITHIN each role from on-ball action mix.

Futi's definition: "Styles group players within each role into archetypes
based on what they like to do on the ball." Roles say WHERE you operate;
styles say WHAT you do there.

Features per (player, season), open play only (restarts excluded):
  pass-mix rates (of passes):  long, forward, back, cross, progressive,
                                 to final third, to penalty area, shot assists
  volume rates (per 100 in-possession actions): dribbles, progressive
        runs, shots, box touches, aerial duels, offensive duels, carries
  pass_share: passes / all in-possession actions

Within each of the 6 roles: z-score features within-role -> k-means with
k chosen from {2,3,4} by silhouette. Split-half validation: odd vs even
matches' action mixes -> same style?

Outputs: style_assignments_season.parquet, style_profiles.csv (centroid
z-profiles for naming/audit).

Run from the Dashboard dir: python models/roles/build_styles.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import sys

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from build_role_features import load_events, flag_channels

TAGS = ['long_pass', 'forward_pass', 'back_pass', 'cross', 'progressive_pass',
         'pass_to_final_third', 'pass_to_penalty_area', 'shot_assist',
         'carry', 'progressive_run', 'touch_in_box', 'aerial_duel',
         'offensive_duel', 'dribble']
MIN_MINS = 600
MIN_PASSES = 80
RANDOM_STATE = 7

print("[1/4] events…", flush=True)
ev = load_events()
ev = flag_channels(ev)
ev = ev[~ev['_restart']].copy()
ev['playerId'] = ev['player.id']

print("[2/4] tag counts…", flush=True)
sec_sets = [frozenset(t) if isinstance(t, (list, tuple, np.ndarray)) else frozenset()
              for t in ev['type.secondary'].tolist()]
for tag in TAGS:
    ev[f't_{tag}'] = [tag in s for s in sec_sets]
ev['t_pass'] = ev['type.primary'] == 'pass'
ev['t_shot'] = ev['type.primary'] == 'shot'
ev['parity'] = (ev['matchId'].astype('int64') % 2).astype(int)

cnt_cols = [f't_{t}' for t in TAGS] + ['t_pass', 't_shot', '_ip']
g = (ev.groupby(['playerId', 'seasonId', 'parity'])[cnt_cols]
       .sum().reset_index())
full = g.groupby(['playerId', 'seasonId'])[cnt_cols].sum().reset_index()


def mix_features(df):
    """Raw counts -> style feature columns."""
    out = pd.DataFrame({'playerId': df['playerId'], 'seasonId': df['seasonId']})
    p = df['t_pass'].clip(lower=1)
    ip = df['_ip'].clip(lower=1)
    for t in ['long_pass', 'forward_pass', 'back_pass', 'cross',
                'progressive_pass', 'pass_to_final_third',
                'pass_to_penalty_area', 'shot_assist']:
        out[f'r_{t}'] = df[f't_{t}'] / p
    for t in ['dribble', 'progressive_run', 'touch_in_box', 'aerial_duel',
                'offensive_duel', 'carry']:
        out[f'v_{t}'] = df[f't_{t}'] / ip * 100
    out['v_shot'] = df['t_shot'] / ip * 100
    out['pass_share'] = df['t_pass'] / ip
    return out

F = mix_features(full)
FEATS = [c for c in F.columns if c not in ('playerId', 'seasonId')]

roles = pd.read_parquet(_HERE / 'role_assignments_season.parquet')
F = F.merge(roles[['playerId', 'seasonId', 'primary_role_name', 'side',
                      'mins_played', 'name']],
              on=['playerId', 'seasonId'], how='inner')
F = F.merge(full[['playerId', 'seasonId', 't_pass']], on=['playerId', 'seasonId'])
Q = F[(F['mins_played'].fillna(0) >= MIN_MINS) & (F['t_pass'] >= MIN_PASSES)
        & F['primary_role_name'].notna()].copy()
print(f"[3/4] clustering within roles… qualifying: {len(Q):,}")

profiles, assigns = [], []
models = {}
for role, gR in Q.groupby('primary_role_name'):
    X = gR[FEATS].fillna(0).to_numpy()
    sc = StandardScaler().fit(X)
    Z = sc.transform(X)
    best = None
    for k in (2, 3, 4):
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(Z)
        sil = silhouette_score(Z, km.labels_)
        share = np.bincount(km.labels_).min() / len(gR)
        if share < 0.10:
            continue
        if best is None or sil > best[1]:
            best = (k, sil, km)
    k, sil, km = best
    models[role] = (sc, km)
    lab = km.labels_
    print(f"  {role:<22} n={len(gR):4d}  k={k}  sil={sil:.3f}  "
           f"sizes={list(np.bincount(lab))}")
    gR = gR.copy(); gR['style_id'] = lab
    assigns.append(gR[['playerId', 'seasonId', 'primary_role_name',
                          'style_id']])
    for s in range(k):
        zc = km.cluster_centers_[s]
        top = sorted(zip(FEATS, zc), key=lambda t: -abs(t[1]))[:5]
        ex = (gR[gR['style_id'] == s].sort_values('mins_played',
                ascending=False)['name'].dropna().head(5).tolist())
        profiles.append({'role': role, 'style_id': s, 'n': int((lab == s).sum()),
                           'top_features': '; '.join(f"{f}{z:+.2f}" for f, z in top),
                           'exemplars': ', '.join(map(str, ex))})

A = pd.concat(assigns, ignore_index=True)
P = pd.DataFrame(profiles)

# ---- split-half validation ---------------------------------------------------
print("[4/4] split-half validation…", flush=True)
halves = {par: mix_features(g[g['parity'] == par]) for par in (0, 1)}
agree, tot = 0, 0
for role, (sc, km) in models.items():
    ids = A[A['primary_role_name'] == role][['playerId', 'seasonId', 'style_id']]
    h0 = ids.merge(halves[0], on=['playerId', 'seasonId'])
    h1 = ids.merge(halves[1], on=['playerId', 'seasonId'])
    common = set(map(tuple, h0[['playerId', 'seasonId']].to_numpy())) & \
               set(map(tuple, h1[['playerId', 'seasonId']].to_numpy()))
    h0 = h0[h0[['playerId', 'seasonId']].apply(tuple, axis=1).isin(common)]
    h1 = h1[h1[['playerId', 'seasonId']].apply(tuple, axis=1).isin(common)]
    if h0.empty:
        continue
    l0 = km.predict(sc.transform(h0[FEATS].fillna(0).to_numpy()))
    l1 = km.predict(sc.transform(h1[FEATS].fillna(0).to_numpy()))
    h0 = h0.assign(l0=l0).sort_values(['playerId', 'seasonId'])
    h1 = h1.assign(l1=l1).sort_values(['playerId', 'seasonId'])
    m = h0[['playerId', 'seasonId', 'l0']].merge(
        h1[['playerId', 'seasonId', 'l1']], on=['playerId', 'seasonId'])
    agree += (m['l0'] == m['l1']).sum(); tot += len(m)
print(f"split-half style agreement: {agree/tot*100:.1f}% (n={tot})")

A.to_parquet(_HERE / 'style_assignments_season.parquet')
P.to_csv(_HERE / 'style_profiles.csv', index=False)
import joblib
joblib.dump({'models': models, 'feats': FEATS}, _HERE / 'style_model.joblib')
print("saved style_assignments_season.parquet + style_profiles.csv")
print("\n" + P.to_string(index=False))
