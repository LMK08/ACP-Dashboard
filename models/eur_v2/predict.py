"""Prediction wrapper for the v2 EUR regression.

Usage from app.py:
    from models.eur_v2.predict import predict_eur, load_model

    model_bundle = load_model()  # cache me
    pred = predict_eur(model_bundle, features_dict)
    # → predicted EUR (float) or None if model unavailable

Features dict must include all keys in model_bundle['features'].
Missing or None values are filled with the column mean from training.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
# joblib is a transitive dep of scikit-learn but HF Spaces installs
# strip it sometimes. Importing at module-load makes a missing-joblib
# error fail loud + early instead of silently disabling MV / TV cells.
import joblib

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / 'eur_v2_ridge.joblib'
META_PATH  = HERE / 'meta.json'
TRUE_VALUE_MODEL_PATH = HERE / 'true_value_ridge.joblib'


def load_model() -> dict | None:
    """Returns dict with 'model', 'features', 'feature_means', 'meta'.
    None if the model file is missing (regression not yet trained).

    This is the FULL feature model — produces 'Predicted TMV'
    (Transfermarkt-style market value estimate).
    """
    if not MODEL_PATH.exists():
        return None
    bundle = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
    feature_means = {}
    ts_path = HERE / 'training_set.csv'
    if ts_path.exists():
        ts = pd.read_csv(ts_path)
        for f in bundle.get('features', []):
            if f in ts.columns:
                feature_means[f] = float(pd.to_numeric(ts[f],
                                                         errors='coerce').mean())
    return {**bundle, 'meta': meta, 'feature_means': feature_means}


def load_true_value_model() -> dict | None:
    """Returns the CVI-only True Value model bundle.
    Uses ONLY signed_log_tv + age + league + position_group — strips out
    market-noise features (goals/assists/xG/passport/career_mins).
    Answers 'pure on-pitch quality' EUR rather than market positioning.
    """
    if not TRUE_VALUE_MODEL_PATH.exists():
        return None
    bundle = joblib.load(TRUE_VALUE_MODEL_PATH)
    feature_means = {}
    ts_path = HERE / 'training_set.csv'
    if ts_path.exists():
        ts = pd.read_csv(ts_path)
        for f in bundle.get('features', []):
            if f in ts.columns:
                feature_means[f] = float(pd.to_numeric(ts[f],
                                                         errors='coerce').mean())
    return {**bundle, 'feature_means': feature_means}


def predict_eur(bundle: dict, features: dict) -> float | None:
    """Predict EUR market value from a features dict.

    Features keys must match bundle['features']; missing keys get the
    training-set mean. Returns predicted EUR (always > 0) or None on
    error.
    """
    if bundle is None or 'model' not in bundle:
        return None
    try:
        feat_list = bundle['features']
        means = bundle.get('feature_means', {})
        x = np.array([[features.get(f, means.get(f, 0.0))
                          for f in feat_list]], dtype=float)
        # Replace NaNs with means (defensive)
        for i, f in enumerate(feat_list):
            if np.isnan(x[0, i]):
                x[0, i] = means.get(f, 0.0)
        log_pred = bundle['model'].predict(x)[0]
        return float(np.exp(log_pred))
    except Exception as e:
        print(f"[eur_v2.predict] failed: {type(e).__name__}: {e}")
        return None


def build_features_for_player(*, age: float | None,
                                 position_group: str | None,
                                 total_value_per90: float | None,
                                 league_factor: float = 1.0,
                                 career_mins: float | None = None,
                                 mins_season: float | None = None,
                                 passport_pt: int = 0,
                                 n_seasons_played: int = 1,
                                 season_year: int = 2025,
                                 xg_residual_career: float = 0.0,
                                 goals_career: float = 0.0,
                                 goals_season: float = 0.0,
                                 assists_career: float = 0.0,
                                 assists_season: float = 0.0,
                                 ) -> dict:
    """Convenience builder mirroring the feature schema in
    train_eur_v2.py. Pass whatever you have; the rest fills with
    sensible defaults / training means.
    """
    feats = {
        'age': age,
        'league_factor': league_factor,
        'passport_pt': passport_pt,
        'n_seasons_played': n_seasons_played,
        'season_year': season_year,
        'xg_residual_career': xg_residual_career,
        'goals_career': goals_career,
        'goals_season': goals_season,
        'assists_career': assists_career,
        'assists_season': assists_season,
        # signed log of total_value_per90 with × 100 scaling (matches
        # train_eur_v2's signed_log_tv)
        'signed_log_tv': (np.sign(total_value_per90)
                            * np.log1p(abs(total_value_per90) * 100)
                            if total_value_per90 is not None and total_value_per90 == total_value_per90
                            else 0.0),
        'log_career_mins': (np.log(max(career_mins, 1))
                              if career_mins else None),
        'career_mins_k': (career_mins / 1000.0
                            if career_mins else None),
        'log_mins_season': (np.log(max(mins_season, 1))
                              if mins_season else None),
    }
    # Position one-hots (CM = reference; not included)
    for pos in ('AM_WG', 'CB', 'FB', 'GK', 'ST'):
        feats[f'pos_{pos}'] = 1 if position_group == pos else 0
    return feats
