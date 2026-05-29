#!/usr/bin/env python3
"""Train the v2 EUR transfer-value regression.

Pipeline:
  1. Load TM market-value snapshots (valuations/valuations.parquet)
     + TM player metadata + the per-(player, season) GPA values.
  2. For each (playerId, seasonId) where player is matched to TM,
     find the TM snapshot closest to the season's MIDPOINT and use
     log(value_eur) as the target.
  3. Compute features for that (player, season):
       - log(CVI) — the on-pitch quality kernel from CVI v2.7
       - position_group ONE-HOT (GK, CB, FB, CM, AM_WG, ST)
            → lets the model learn cross-position price premiums
              (wingers > CBs at the same CVI is what scouts see)
       - age_at_season
       - league_factor (1.0 L3, 0.85 Camp)
       - log(career_mins_to_date)
       - xG / xA per-90 residuals (career-cumulative)
       - n_seasons_played (cardinality, not position cardinality)
       - passport_PT (binary: Portuguese passport vs other)
       - team_opta_rating (current team strength, when available)
       - season_year (calendar year — captures macro market drift)
  4. Standardize numeric features, fit Ridge regression with
     cross-validated alpha. Ridge handles small-sample + correlated
     features gracefully.
  5. Save model.pkl + coefficient table + diagnostics to models/eur_v2/

Usage:
  python train_eur_v2.py            # train + save
  python train_eur_v2.py --dry      # just print diagnostics
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Import the CVI pipeline + helpers from the dashboard's app.py
# We import the bits we need by reading the module in a controlled way.
# This avoids spinning up Streamlit just to call the functions.


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--dry', action='store_true',
                     help='Print diagnostics without writing model')
    ap.add_argument('--min-mins', type=int, default=500,
                     help='Min minutes per season to include in training')
    ap.add_argument('--eur-min', type=int, default=25_000,
                     help='Min EUR to include (filters noise-floor entries)')
    ap.add_argument('--eur-max', type=int, default=5_000_000,
                     help='Max EUR to include (filters Sporting/Benfica academy '
                           'outliers that distort the model for Liga 3/Camp use)')
    args = ap.parse_args()

    print("=" * 60)
    print("v2 EUR REGRESSION — TRAINING PIPELINE")
    print("=" * 60)

    # --- Load data sources ---
    vals = pd.read_parquet(HERE / 'valuations' / 'valuations.parquet')
    print(f"\n[data] TM MV snapshots: {len(vals):,} rows for "
           f"{vals['playerId'].nunique()} players")

    gpa = pd.read_parquet(HERE / 'gpa_player_season_values.parquet')
    print(f"[data] GPA per-(player, season): {len(gpa):,} rows for "
           f"{gpa['playerId'].nunique()} players")

    # Player details for age/passport/birth area
    details = pd.read_pickle(HERE / 'player_details.pkl')
    if isinstance(details, list):
        details = pd.DataFrame(details)
    print(f"[data] player_details: {len(details):,} players")

    # TM metadata for richer features when available
    tm_meta_path = HERE / 'valuations' / 'tm_player_metadata.parquet'
    if tm_meta_path.exists():
        tm_meta = pd.read_parquet(tm_meta_path)
        print(f"[data] TM metadata: {len(tm_meta):,} players")
    else:
        tm_meta = pd.DataFrame()
        print(f"[data] TM metadata: not found")

    # Opta team strength
    opta_path = HERE / 'opta_ratings.parquet'
    if opta_path.exists():
        opta = pd.read_parquet(opta_path)
        print(f"[data] Opta team ratings: {len(opta):,} clubs")
    else:
        opta = pd.DataFrame()

    # --- Map seasons → dates ---
    SEASON_MIDPOINT = {
        # Approximate Dec 15 of the year covering the bulk of matches
        188221: '2021-12-15',
        188222: '2022-12-15',
        189147: '2023-12-15',
        190090: '2024-12-15',
        191782: '2025-12-15',
        # Campeonato
        190230: '2023-12-15',
        191779: '2025-12-15',
    }
    SEASON_LEAGUE_FACTOR = {
        188221: 1.00, 188222: 1.00, 189147: 1.00,
        190090: 1.00, 191782: 1.00,  # Liga 3
        190230: 0.85, 191779: 0.85,  # Campeonato
    }
    SEASON_YEAR = {
        188221: 2021, 188222: 2022, 189147: 2023,
        190090: 2024, 191782: 2025, 190230: 2023, 191779: 2025,
    }

    # --- Build training rows ---
    print(f"\n[build] Joining GPA seasons to nearest TM snapshot...")
    matched_pids = set(vals['playerId'].unique())
    gpa = gpa[gpa['playerId'].isin(matched_pids)].copy()
    print(f"[build] GPA rows for TM-matched players: {len(gpa):,}")

    # Filter to minimum minutes
    mins_col = next((c for c in ('mins_played', 'totalMinutes', 'Minutes')
                      if c in gpa.columns), None)
    if mins_col:
        gpa = gpa[gpa[mins_col] >= args.min_mins].copy()
        print(f"[build] After {args.min_mins}-min filter: {len(gpa):,}")

    # For each (player, season), find closest TM snapshot
    vals['_d'] = pd.to_datetime(vals['as_of_date'], errors='coerce')
    rows = []
    for _, r in gpa.iterrows():
        pid = int(r['playerId'])
        sid = int(r['seasonId'])
        mid_str = SEASON_MIDPOINT.get(sid)
        if mid_str is None:
            continue
        mid = pd.to_datetime(mid_str)
        snaps = vals[(vals['playerId'] == pid)
                       & (vals['value_eur'].notna())
                       & (vals['value_eur'] > 0)].copy()
        if snaps.empty:
            continue
        snaps['_diff'] = (snaps['_d'] - mid).abs()
        snaps = snaps.sort_values('_diff')
        best = snaps.iloc[0]
        # Reject if no snapshot within 18 months — too stale
        if best['_diff'] > pd.Timedelta(days=540):
            continue
        rows.append({
            'playerId': pid,
            'seasonId': sid,
            'mins_played': r.get(mins_col, 0),
            'primaryPosition': r.get('position', r.get('primaryPosition')),
            'Total Value': r.get('Total Value', r.get('total_v_per_90')),
            'value_eur': float(best['value_eur']),
            'mv_snapshot_date': best['_d'].date().isoformat(),
            'season_year': SEASON_YEAR.get(sid, 2024),
            'league_factor': SEASON_LEAGUE_FACTOR.get(sid, 1.0),
            'days_to_snapshot': int(best['_diff'].days),
        })
    train = pd.DataFrame(rows)
    print(f"[build] Joined training rows: {len(train):,} player-seasons")

    # Filter to a realistic EUR range. Liga 3 / Camp transfers cluster
    # in €25k - €2M; values outside that are noise (random €5k entries)
    # or academy-pipeline outliers (Sporting/Benfica loanees with
    # €10M+ TM valuations) that distort log-space variance.
    pre = len(train)
    train = train[(train['value_eur'] >= args.eur_min)
                    & (train['value_eur'] <= args.eur_max)].copy()
    print(f"[build] After EUR €{args.eur_min:,}-€{args.eur_max:,} filter: "
           f"{len(train):,}  (dropped {pre - len(train)})")

    # --- Map positions to CVI groups ---
    POS_GROUP = {
        'GK': 'GK',
        'CB': 'CB', 'LCB': 'CB', 'RCB': 'CB', 'LCB3': 'CB', 'RCB3': 'CB',
        'LB': 'FB', 'RB': 'FB', 'LB5': 'FB', 'RB5': 'FB', 'LWB': 'FB', 'RWB': 'FB',
        'CMF': 'CM', 'LCMF': 'CM', 'RCMF': 'CM', 'LCMF3': 'CM', 'RCMF3': 'CM',
        'DMF': 'CM', 'LDMF': 'CM', 'RDMF': 'CM',
        'AMF': 'AM_WG', 'LAMF': 'AM_WG', 'RAMF': 'AM_WG',
        'LMF': 'AM_WG', 'RMF': 'AM_WG',
        'LW': 'AM_WG', 'RW': 'AM_WG', 'LWF': 'AM_WG', 'RWF': 'AM_WG',
        'CF': 'ST', 'SS': 'ST',
    }
    train['position_group'] = train['primaryPosition'].map(POS_GROUP)
    train = train.dropna(subset=['position_group'])
    print(f"[build] After position-group mapping: {len(train):,}")

    # --- Compute age (player_details FIRST, fall back to TM metadata) ---
    dob_map = {}
    for _, r in details.iterrows():
        try:
            pid = int(r['playerId'])
            bd = r.get('birthDate')
            if bd and not pd.isna(bd):
                dob_map[pid] = pd.to_datetime(bd)
        except Exception:
            pass
    # Fallback: pull DOBs from TM metadata for the ~600 players missing
    # from player_details (e.g. cross-tier merges where the source DB
    # only carried current-league rosters).
    if not tm_meta.empty and 'dob' in tm_meta.columns:
        filled_from_tm = 0
        for _, r in tm_meta.iterrows():
            try:
                pid = int(r['playerId'])
                if pid in dob_map:
                    continue
                bd = r.get('dob')
                if bd and pd.notna(bd):
                    dob_map[pid] = pd.to_datetime(bd)
                    filled_from_tm += 1
            except Exception:
                pass
        print(f"[build] DOBs from player_details: "
               f"{len(dob_map) - filled_from_tm}, from TM metadata: {filled_from_tm}")
    train['age'] = train.apply(
        lambda r: ((pd.to_datetime(SEASON_MIDPOINT[r['seasonId']])
                     - dob_map[int(r['playerId'])]).days / 365.25
                     if int(r['playerId']) in dob_map else np.nan),
        axis=1,
    )
    train = train.dropna(subset=['age'])
    print(f"[build] After age filter: {len(train):,}")

    # --- Passport nationality flag ---
    pt_pids = set()
    for _, r in details.iterrows():
        try:
            pid = int(r['playerId'])
            pp = r.get('passportArea')
            if isinstance(pp, dict):
                if (pp.get('name', '') or '').lower() == 'portugal':
                    pt_pids.add(pid)
            ba = r.get('birthArea')
            if isinstance(ba, dict):
                if (ba.get('name', '') or '').lower() == 'portugal':
                    pt_pids.add(pid)
        except Exception:
            pass
    train['passport_pt'] = train['playerId'].apply(
        lambda p: 1 if int(p) in pt_pids else 0)

    # --- Career mins to date (career-cumulative up to season) ---
    gpa_career = (pd.read_parquet(HERE / 'gpa_player_season_values.parquet')
                     [['playerId', 'seasonId', mins_col]])
    season_orders = {sid: SEASON_YEAR.get(sid, 2024) for sid in SEASON_YEAR}
    gpa_career['_year'] = gpa_career['seasonId'].map(season_orders)
    career_mins_map = {}
    for pid, group in gpa_career.dropna(subset=['_year']).groupby('playerId'):
        sorted_g = group.sort_values('_year')
        cum = sorted_g[mins_col].cumsum().tolist()
        years = sorted_g['_year'].tolist()
        sids = sorted_g['seasonId'].tolist()
        for sid, c in zip(sids, cum):
            career_mins_map[(int(pid), int(sid))] = float(c)
    train['career_mins_to_date'] = train.apply(
        lambda r: career_mins_map.get((int(r['playerId']), int(r['seasonId'])),
                                         r['mins_played']),
        axis=1,
    )

    # --- Goals / assists / xG residuals (career + per-season) ---
    # Per-season + career-cumulative goals and assists are durable
    # output signals that markets price into transfer fees.
    try:
        ev = pd.read_parquet(HERE / 'raw_events.parquet',
                              columns=['player.id', 'matchId', 'type.primary',
                                        'type.secondary',
                                        'shot.xg', 'shot.isGoal'])
        ms_for_xg = pd.read_parquet(HERE / 'matches_summary.parquet',
                                       columns=['matchId', 'seasonId'])
        ev = ev.merge(ms_for_xg, on='matchId', how='left').dropna(subset=['player.id'])
        ev['player.id'] = ev['player.id'].astype('int64')
        sid_order = pd.DataFrame({'seasonId': list(SEASON_YEAR.keys()),
                                     '_yr': list(SEASON_YEAR.values())})

        # GOALS — from shots with isGoal
        shots = ev[ev['type.primary'].isin(['shot'])].copy()
        shots['xg'] = pd.to_numeric(shots.get('shot.xg'), errors='coerce').fillna(0)
        shots['goal'] = shots.get('shot.isGoal', False).fillna(False).astype(int)

        # ASSISTS — from passes with 'assist' tag in type.secondary
        passes = ev[ev['type.primary'] == 'pass'].copy()
        passes['is_assist'] = passes['type.secondary'].apply(
            lambda v: 1 if (isinstance(v, np.ndarray)
                              and 'assist' in v)
                       else (1 if isinstance(v, list) and 'assist' in v else 0)
        )

        per_ss_g = (shots.groupby(['player.id', 'seasonId'])
                          .agg(xg_sum=('xg', 'sum'),
                                goals=('goal', 'sum'))
                          .reset_index()
                          .rename(columns={'player.id': 'playerId'}))
        per_ss_a = (passes.groupby(['player.id', 'seasonId'])
                          .agg(assists=('is_assist', 'sum'))
                          .reset_index()
                          .rename(columns={'player.id': 'playerId'}))
        per_ss = per_ss_g.merge(per_ss_a, on=['playerId', 'seasonId'], how='outer').fillna(0)
        per_ss = per_ss.merge(sid_order, on='seasonId', how='left')
        per_ss = per_ss.sort_values(['playerId', '_yr'])
        per_ss['xg_career'] = per_ss.groupby('playerId')['xg_sum'].cumsum()
        per_ss['goals_career'] = per_ss.groupby('playerId')['goals'].cumsum()
        per_ss['assists_career'] = per_ss.groupby('playerId')['assists'].cumsum()
        per_ss['xg_residual_career'] = per_ss['goals_career'] - per_ss['xg_career']

        # Per-(player, season) feature maps
        xg_resid_map = {(int(r['playerId']), int(r['seasonId'])): float(r['xg_residual_career'])
                          for _, r in per_ss.iterrows()}
        goals_career_map = {(int(r['playerId']), int(r['seasonId'])): float(r['goals_career'])
                              for _, r in per_ss.iterrows()}
        goals_season_map = {(int(r['playerId']), int(r['seasonId'])): float(r['goals'])
                              for _, r in per_ss.iterrows()}
        assists_season_map = {(int(r['playerId']), int(r['seasonId'])): float(r['assists'])
                                for _, r in per_ss.iterrows()}
        assists_career_map = {(int(r['playerId']), int(r['seasonId'])): float(r['assists_career'])
                                for _, r in per_ss.iterrows()}

        train['xg_residual_career'] = train.apply(
            lambda r: xg_resid_map.get((int(r['playerId']), int(r['seasonId'])), 0.0),
            axis=1)
        train['goals_career'] = train.apply(
            lambda r: goals_career_map.get((int(r['playerId']), int(r['seasonId'])), 0.0),
            axis=1)
        train['goals_season'] = train.apply(
            lambda r: goals_season_map.get((int(r['playerId']), int(r['seasonId'])), 0.0),
            axis=1)
        train['assists_season'] = train.apply(
            lambda r: assists_season_map.get((int(r['playerId']), int(r['seasonId'])), 0.0),
            axis=1)
        train['assists_career'] = train.apply(
            lambda r: assists_career_map.get((int(r['playerId']), int(r['seasonId'])), 0.0),
            axis=1)
        print(f"[build] Goals/assists/xG computed for "
               f"{len(goals_career_map):,} player-seasons")
    except Exception as e:
        print(f"[build] Goals/assists computation failed: {e}; setting to 0")
        for c in ('xg_residual_career', 'goals_career', 'goals_season',
                   'assists_season', 'assists_career'):
            train[c] = 0.0

    # --- n_seasons_played (in our data, up to and including this season) ---
    seasons_to_date = {}
    for pid, group in gpa_career.dropna(subset=['_year']).groupby('playerId'):
        sorted_g = group.sort_values('_year')
        years = sorted_g['_year'].tolist()
        sids = sorted_g['seasonId'].tolist()
        for i, sid in enumerate(sids):
            seasons_to_date[(int(pid), int(sid))] = i + 1
    train['n_seasons_played'] = train.apply(
        lambda r: seasons_to_date.get((int(r['playerId']), int(r['seasonId'])), 1),
        axis=1,
    )

    # --- log target + features ---
    train['log_value_eur'] = np.log(train['value_eur'].clip(lower=1000))
    # log Total Value — use a small offset to handle 0/negative
    train['Total Value'] = pd.to_numeric(train['Total Value'], errors='coerce')
    train['signed_log_tv'] = train['Total Value'].apply(
        lambda v: np.sign(v) * np.log1p(abs(v) * 100) if pd.notna(v) else 0)
    train['log_career_mins'] = np.log(
        train['career_mins_to_date'].clip(lower=1))
    train['log_mins_season'] = np.log(train['mins_played'].clip(lower=1))
    # Raw career mins (in thousands so coef stays interpretable;
    # complements log_career_mins which captures concavity)
    train['career_mins_k'] = train['career_mins_to_date'] / 1000.0
    # Raw career goals + log(career goals) so the model can pick up
    # both 'has scored at all' (log lifts 0→1) and 'has scored a lot'.
    train['goals_career'] = pd.to_numeric(train['goals_career'],
                                              errors='coerce').fillna(0)
    train['log_goals_career'] = np.log1p(train['goals_career'])

    # --- Position-group one-hot (drop CM as reference) ---
    pos_dummies = pd.get_dummies(train['position_group'], prefix='pos',
                                     drop_first=False).astype(int)
    if 'pos_CM' in pos_dummies.columns:
        pos_dummies = pos_dummies.drop(columns=['pos_CM'])
    train = pd.concat([train, pos_dummies], axis=1)

    # --- Final feature list ---
    num_features = ['signed_log_tv', 'age', 'league_factor',
                     'log_career_mins', 'career_mins_k',
                     'log_mins_season',
                     'passport_pt', 'n_seasons_played', 'season_year',
                     'xg_residual_career',
                     'goals_career', 'goals_season',
                     'assists_career', 'assists_season']
    cat_features = [c for c in train.columns if c.startswith('pos_')]
    features = num_features + cat_features
    print(f"\n[features] {len(features)} features: {features}")

    X = train[features].astype(float).values
    y = train['log_value_eur'].values
    print(f"[features] X shape: {X.shape}, y shape: {y.shape}")

    # --- Cross-validate Ridge ---
    print(f"\n[model] Fitting Ridge with 5-fold CV...")
    model = Pipeline([
        ('scale', StandardScaler()),
        ('ridge', RidgeCV(alphas=[0.01, 0.1, 1, 10, 100], cv=5)),
    ])
    model.fit(X, y)
    chosen_alpha = model.named_steps['ridge'].alpha_
    print(f"[model] Chosen alpha: {chosen_alpha}")

    # Coefficients (back-mapped from standardized space)
    scaler = model.named_steps['scale']
    coefs = model.named_steps['ridge'].coef_
    intercept = model.named_steps['ridge'].intercept_

    print(f"\n[coefficients] Standardized (relative importance):")
    coef_df = pd.DataFrame({
        'feature': features,
        'coef_std': coefs,
        'abs_coef': np.abs(coefs),
    }).sort_values('abs_coef', ascending=False)
    print(coef_df[['feature', 'coef_std']].to_string(index=False))

    # --- Out-of-fold predictions for honest R² + MAE ---
    print(f"\n[validation] Computing 5-fold out-of-sample predictions...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    for fold_i, (tr, te) in enumerate(kf.split(X)):
        fm = Pipeline([
            ('scale', StandardScaler()),
            ('ridge', RidgeCV(alphas=[0.01, 0.1, 1, 10, 100], cv=3)),
        ])
        fm.fit(X[tr], y[tr])
        oof[te] = fm.predict(X[te])
    oof_r2 = r2_score(y, oof)
    oof_mae_log = mean_absolute_error(y, oof)
    # Back-transform to EUR for an interpretable MAE
    pred_eur = np.exp(oof)
    actual_eur = np.exp(y)
    # Median absolute % error is robust to outliers
    mape = np.median(np.abs(pred_eur - actual_eur) / actual_eur) * 100
    # Spearman rank correlation — captures "ordering correctness" which
    # matters more than absolute EUR accuracy for scouting purposes.
    from scipy.stats import spearmanr
    spearman_rho, _ = spearmanr(y, oof)
    # Within-position Spearman (so position one-hots don't drive the rank)
    pos_spearman = []
    train_idx = train.reset_index(drop=True)
    for pos in train_idx['position_group'].unique():
        mask = train_idx['position_group'] == pos
        if mask.sum() >= 10:
            r, _ = spearmanr(y[mask.values], oof[mask.values])
            pos_spearman.append((pos, int(mask.sum()), r))
    print(f"  Out-of-fold R²:           {oof_r2:.3f}")
    print(f"  Out-of-fold MAE (log):    {oof_mae_log:.3f}")
    print(f"  Out-of-fold MAPE:         {mape:.1f}%")
    print(f"  Out-of-fold MAE (EUR):    €{np.mean(np.abs(pred_eur - actual_eur)):,.0f}")
    print(f"  Out-of-fold Spearman ρ:   {spearman_rho:.3f}  (all positions pooled)")
    print(f"  Out-of-fold Spearman by position:")
    for pos, n, r in sorted(pos_spearman, key=lambda x: -x[2]):
        print(f"    {pos:<8}  n={n:>3}  ρ={r:+.3f}")

    # Top predictions vs actual
    train['predicted_eur'] = np.exp(model.predict(X))
    train['oof_predicted_eur'] = pred_eur
    train['log_actual'] = y
    train['oof_residual'] = y - oof
    print(f"\n[spot-check] 5 random rows with residuals:")
    sample = train.sample(min(5, len(train)), random_state=1)
    for _, r in sample.iterrows():
        print(f"  pid={r['playerId']:<8}  s={r['seasonId']}  pos={r['position_group']:<6} "
              f"age={r['age']:.1f}  actual=€{r['value_eur']:>10,.0f}  "
              f"pred=€{r['oof_predicted_eur']:>10,.0f}  residual={r['oof_residual']:+.2f}")

    if args.dry:
        print(f"\n[done] Dry run — no model saved")
        return 0

    # --- Save ---
    out_dir = HERE / 'models' / 'eur_v2'
    out_dir.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump({
        'model': model,
        'features': features,
        'chosen_alpha': float(chosen_alpha),
        'n_training_rows': len(train),
        'oof_r2': float(oof_r2),
        'oof_mae_log': float(oof_mae_log),
        'oof_mape_pct': float(mape),
    }, out_dir / 'eur_v2_ridge.joblib')
    coef_df.to_csv(out_dir / 'coefficients.csv', index=False)
    train[['playerId', 'seasonId', 'position_group', 'age',
            'value_eur', 'predicted_eur', 'oof_predicted_eur']].to_csv(
        out_dir / 'training_set.csv', index=False)
    with open(out_dir / 'meta.json', 'w') as f:
        json.dump({
            'features': features,
            'chosen_alpha': float(chosen_alpha),
            'n_training_rows': int(len(train)),
            'oof_r2': float(oof_r2),
            'oof_mae_log': float(oof_mae_log),
            'oof_mape_pct': float(mape),
            'pos_groups_one_hot': [c for c in features if c.startswith('pos_')],
            'reference_position_group': 'CM',
        }, f, indent=2)
    print(f"\n[done] Model + diagnostics saved to {out_dir}/")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
