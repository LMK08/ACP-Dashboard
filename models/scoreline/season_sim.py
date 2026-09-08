"""Scoreline sampling for the season simulation (simulate_season.py).

Until 2026-09 the Monte Carlo season simulation drew win / draw / loss from
the sklearn simple predictor and invented the goals (2-1 for a win, 1-1 for
a draw). Now every remaining fixture is drawn as a full scoreline from the
Dixon-Coles model (models/scoreline/dixon_coles.py) — the same fit behind
the Match Predictor's scoreline forecast and its Team Strength Ratings — so
the Home page's promotion odds, the promotion / relegation tables and the
fixture forecast rest on ONE model, and goal difference and goals scored in
the simulated tables are real draws rather than placeholders.

Pure numpy / pandas; no Streamlit.
"""
import math
import os

import numpy as np

from models.scoreline.dixon_coles import DixonColes, MAX_GOALS, attach_xg, matches_from_summary

HERE = os.path.dirname(os.path.abspath(__file__))
PARAMS_PATH = os.path.join(HERE, 'dc_params.json')


class ScorelineSampler:
    """Per-fixture cumulative scoreline distributions from a DixonColes fit,
    for one league (it picks the league's scoring baseline). Rows are cached
    per (home, away) pair, so the chained phases that revisit a pairing
    thousands of times pay for the score matrix once."""

    def __init__(self, model, league_id, max_goals=MAX_GOALS):
        self.model = model
        self.league_id = int(league_id)
        self.goals = int(max_goals) + 1          # goals per side: 0..max_goals
        self._cache = {}

    def cumulative(self, home, away):
        """Flattened cumulative distribution over (home goals, away goals),
        row-major (index = home goals * goals + away goals); the last entry is
        exactly 1 so a uniform draw always lands on a cell."""
        key = (home, away)
        cum = self._cache.get(key)
        if cum is None:
            lam, mu, _, _ = self.model.rates(home, away, self.league_id)
            P = self.model.score_matrix(lam, mu, self.goals - 1)
            cum = np.cumsum(P.ravel())
            cum[-1] = 1.0
            self._cache[key] = cum
        return cum

    def rows(self, fixtures):
        """Cumulative rows stacked in fixture order — shape (n, goals**2)."""
        if not fixtures:
            return np.empty((0, self.goals * self.goals))
        return np.vstack([self.cumulative(h, a) for h, a in fixtures])

    def outcome_probs(self, home, away):
        """{'home', 'draw', 'away'} win probabilities implied by the same
        scoreline distribution the simulation samples from."""
        P = np.diff(np.concatenate([[0.0], self.cumulative(home, away)])).reshape(self.goals, self.goals)
        i, j = np.indices(P.shape)
        return {'home': float(P[i > j].sum()), 'draw': float(np.trace(P)), 'away': float(P[i < j].sum())}

    def unknown_teams(self, teams):
        """Teams the fit has never seen — they play at the league average."""
        return [t for t in teams if t not in self.model.teams]


def sample_scores(cum_rows, r):
    """Inverse-CDF draw of one scoreline per row. ``cum_rows`` (n, goals**2)
    from ScorelineSampler.rows, ``r`` n uniform draws in [0, 1). Returns
    (home goals, away goals) as int arrays."""
    r = np.asarray(r, dtype=float)
    if r.size == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    goals = math.isqrt(cum_rows.shape[1])
    k = (cum_rows < r[:, None]).sum(axis=1)
    k = np.minimum(k, cum_rows.shape[1] - 1)
    return k // goals, k % goals


def apply_results(fixtures, hg, ag, pts, gd, gf):
    """Add sampled scorelines to the points / goal-difference / goals-scored
    dicts in place (3-1-0 points)."""
    for (home, away), h, a in zip(fixtures, np.asarray(hg).tolist(), np.asarray(ag).tolist()):
        gf[home] += h
        gf[away] += a
        gd[home] += h - a
        gd[away] += a - h
        if h > a:
            pts[home] += 3
        elif h < a:
            pts[away] += 3
        else:
            pts[home] += 1
            pts[away] += 1


def load_params(path=PARAMS_PATH):
    """The committed fit (what the app's scoreline section reads)."""
    return DixonColes.load(path)


def refit(matches_summary_df, params_path=PARAMS_PATH, events_path=None, asof=None):
    """Refit the Dixon-Coles model on every played match in
    ``matches_summary_df`` with the hyperparameters (time decay xi, shrinkage
    l2, goals-vs-xG blend mix) chosen by the last build_dc.py backtest and
    stored in ``params_path``. xG is attached from ``events_path`` when the
    file exists, so the blend works as tuned; without it the rates fall back
    to goals. Returns (model, info)."""
    prior = DixonColes.load(params_path)
    matches = matches_from_summary(matches_summary_df)
    xg_attached = bool(events_path) and os.path.exists(events_path)
    if xg_attached:
        matches = attach_xg(matches, events_path)
    model = DixonColes.fit(matches, asof=asof, xi=prior.xi, l2=prior.l2, mix=prior.mix)
    info = {
        'refit': True,
        'xg_attached': xg_attached,
        'xg_coverage': float(matches['xg_h'].notna().mean()) if xg_attached else 0.0,
        'hyperparameters_from': os.path.basename(params_path),
    }
    return model, info


def model_meta(model, info=None):
    """What season_simulation.pkl records about the fit it drew from."""
    d = {
        'name': 'dixon_coles_v1', 'asof': model.asof, 'n_matches': int(model.n_matches),
        'home_adv': float(model.home_adv), 'rho': float(model.rho),
        'xi': float(model.xi), 'l2': float(model.l2), 'mix': float(model.mix),
        'refit': False, 'xg_attached': False,
    }
    if info:
        d.update(info)
    return d
