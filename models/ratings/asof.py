"""As-of helpers: fits that a season-year's rows may use (2026-10-29).

A row of season-year y may only use what was known BEFORE y was played:
season pairs whose second season ended before y, fits on outcomes observed
by then (Lucas 2026-10-29: no data leakage, even where it buys no
accuracy). Too little earlier data -> the documented prior, never a later fit.
"""
import numpy as np
from scipy.stats import pearsonr

LAM_PRIOR, LAM_MIN_PAIRS = 0.15, 30


def _r(p):
    r = pearsonr(p['a'], p['b'])[0] if len(p) >= 3 else np.nan
    return r if np.isfinite(r) else np.nan


def lam_by_year(pairs, years, min_pairs=LAM_MIN_PAIRS, prior=LAM_PRIOR, pooled=None):
    """{y: reliability} — the lag-1 r of the (a, b) season pairs that END
    BEFORE y (pairs['yr'] < y; a pair ending in y contains the row's own
    season). Clipped to [0.05, 1].
    Without `pooled`: fewer than min_pairs -> prior.
    With `pooled` (the same category's pairs over every role): the role's r
    is shrunk toward the pooled r by n / (n + min_pairs); the pooled r itself
    needs min_pairs earlier pairs, else prior. Only earlier pairs either way."""
    out = {}
    for y in sorted(years):
        p = pairs[pairs['yr'] < y]
        if pooled is None:
            lam = _r(p) if len(p) >= min_pairs else prior
        else:
            q = pooled[pooled['yr'] < y]
            base = _r(q) if len(q) >= min_pairs else prior
            base = prior if np.isnan(base) else base
            r_role, n = _r(p), len(p)
            lam = base if np.isnan(r_role) else (n * r_role + min_pairs * base) / (n + min_pairs)
        out[y] = float(np.clip(lam, 0.05, 1.0)) if np.isfinite(lam) else prior
    return out


def fit_by_year(pairs, years, fit, prior, min_pairs, start_col='start_yr'):
    """{y: fit(pairs)} on the season pairs that had ENDED by the end of y — a
    pair (season s -> s + 1) has ended by y when s < y. Fewer than min_pairs
    -> prior (a documented design constant, never a later fit)."""
    out = {}
    for y in sorted(years):
        p = pairs[pairs[start_col] < y]
        out[y] = fit(p) if len(p) >= min_pairs else prior
    return out
