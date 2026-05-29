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
# joblib is optional now — we ship .json fallbacks of the same
# coefficients so MV/TV work even when joblib is missing OR when
# sklearn version mismatches make the pickled Pipeline unloadable
# on HF Spaces.
try:
    import joblib
    _HAS_JOBLIB = True
except Exception:
    joblib = None
    _HAS_JOBLIB = False

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / 'eur_v2_ridge.joblib'
MODEL_JSON_PATH = HERE / 'eur_v2_ridge.json'
META_PATH  = HERE / 'meta.json'
TRUE_VALUE_MODEL_PATH = HERE / 'true_value_ridge.joblib'
TRUE_VALUE_JSON_PATH = HERE / 'true_value_ridge.json'


class _LinearPipeline:
    """Pure-numpy stand-in for sklearn Pipeline(StandardScaler+RidgeCV).
    Lets us load + predict from the .json bundle without any sklearn
    or joblib at runtime. Identical numeric output to the original.
    """
    def __init__(self, scaler_mean, scaler_scale, coef, intercept):
        self._mean = np.array(scaler_mean, dtype=float)
        self._scale = np.array(scaler_scale, dtype=float)
        self._coef = np.array(coef, dtype=float)
        self._intercept = float(intercept)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        X_std = (X - self._mean) / np.where(self._scale != 0, self._scale, 1.0)
        return X_std @ self._coef + self._intercept


def _load_json_bundle(json_path):
    if not json_path.exists():
        return None
    try:
        payload = json.loads(json_path.read_text())
        return {
            'model': _LinearPipeline(
                payload['scaler_mean'], payload['scaler_scale'],
                payload['ridge_coef'], payload['ridge_intercept']),
            'features': payload['features'],
            **payload.get('meta', {}),
        }
    except Exception as e:
        print(f"[predict] JSON bundle {json_path.name} unreadable: {e}")
        return None


def load_model() -> dict | None:
    """Returns dict with 'model', 'features', 'feature_means', 'meta'.
    None if the model file is missing (regression not yet trained).

    This is the FULL feature model — produces 'Predicted TMV'
    (Transfermarkt-style market value estimate).
    """
    bundle = None
    # Try joblib first (richest format)
    if _HAS_JOBLIB and MODEL_PATH.exists():
        try:
            bundle = joblib.load(MODEL_PATH)
        except Exception as e:
            print(f"[load_model] joblib load failed: "
                   f"{type(e).__name__}: {e} — falling back to JSON")
            bundle = None
    # Fallback to portable JSON (no sklearn / joblib needed)
    if bundle is None:
        bundle = _load_json_bundle(MODEL_JSON_PATH)
    if bundle is None:
        return None
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
    bundle = None
    if _HAS_JOBLIB and TRUE_VALUE_MODEL_PATH.exists():
        try:
            bundle = joblib.load(TRUE_VALUE_MODEL_PATH)
        except Exception as e:
            print(f"[load_true_value_model] joblib load failed: "
                   f"{type(e).__name__}: {e} — falling back to JSON")
            bundle = None
    if bundle is None:
        bundle = _load_json_bundle(TRUE_VALUE_JSON_PATH)
    if bundle is None:
        return None
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
                                 perf_blend: float | None = None,
                                 ) -> dict:
    """Convenience builder mirroring the feature schema in
    train_eur_v2.py. Pass whatever you have; the rest fills with
    sensible defaults / training means.
    """
    # signed log of total_value_per90 with × 100 scaling (matches
    # train_eur_v2's signed_log_tv)
    _slt = (np.sign(total_value_per90)
              * np.log1p(abs(total_value_per90) * 100)
              if total_value_per90 is not None and total_value_per90 == total_value_per90
              else 0.0)
    feats = {
        'age': age,
        # v3.6 — age centered at 24, then squared. Lets the regression
        # express a U-shape (peak prime, slight discount young AND old).
        'age_dev_sq': ((age - 24.0) ** 2) if age is not None else 4.0,
        'league_factor': league_factor,
        'passport_pt': passport_pt,
        'n_seasons_played': n_seasons_played,
        'season_year': season_year,
        'xg_residual_career': xg_residual_career,
        'goals_career': goals_career,
        'goals_season': goals_season,
        'assists_career': assists_career,
        'assists_season': assists_season,
        'signed_log_tv': _slt,
        # v3 — squared (sign-preserving) version of the perf signal.
        # Lets the model give convex weight to elite performances.
        'signed_log_tv_sq': _slt * abs(_slt),
        # v3.1 — CVI-style position-weighted perf blend, 2× boost
        # applied. The caller passes this in directly when available
        # (computed from CVI state); otherwise we estimate it from the
        # raw V/90 using a 50th-percentile prior (best we can do
        # without the full season's position-cohort percentile table).
        # See train_eur_v2.py PERF_BOOST.
        'perf_blend': ((perf_blend * 2.0) if perf_blend is not None
                        else 100.0),  # 50 × 2.0 boost
        'perf_blend_sq': (((perf_blend * perf_blend / 100.0) * 2.0)
                            if perf_blend is not None
                            else 50.0),  # (50² / 100) × 2.0
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
