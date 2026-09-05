"""Dixon-Coles model: recovers a known generating process, produces proper
probabilities, and (when the fixture data is present) beats the base rate
out of sample."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

DASHBOARD_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, DASHBOARD_DIR)

from models.scoreline import dixon_coles as dc  # noqa: E402


def _synthetic(n_teams=12, rounds=6, seed=7, home_adv=0.25, rho=-0.08):
    rng = np.random.default_rng(seed)
    teams = [f'T{i:02d}' for i in range(n_teams)]
    att = rng.normal(0, 0.3, n_teams)
    dfn = rng.normal(0, 0.3, n_teams)
    rows, day = [], 0
    for _ in range(rounds):
        for i in range(n_teams):
            for j in range(n_teams):
                if i == j:
                    continue
                lam = np.exp(0.2 + home_adv + att[i] - dfn[j])
                mu = np.exp(0.2 + att[j] - dfn[i])
                x, y = rng.poisson(lam), rng.poisson(mu)
                rows.append({'match_id': len(rows), 'home': teams[i], 'away': teams[j], 'hg': x, 'ag': y,
                             'date': pd.Timestamp('2024-08-01') + pd.Timedelta(days=day), 'league': 1, 'season_id': 1})
                day += 1
    return pd.DataFrame(rows), teams, att, dfn


def test_fit_recovers_strength_ordering_and_home_advantage():
    df, teams, att, dfn = _synthetic()
    model = dc.DixonColes.fit(df, xi=0.0, l2=0.001)
    assert model.home_adv > 0.1
    est = pd.Series(model.att, index=model.teams).reindex(teams).values
    assert np.corrcoef(est, att)[0, 1] > 0.85
    est_d = pd.Series(model.dfn, index=model.teams).reindex(teams).values
    assert np.corrcoef(est_d, dfn)[0, 1] > 0.85
    assert -0.9 < model.rho < 0.9


def test_predict_is_a_proper_distribution():
    df, teams, *_ = _synthetic(rounds=3)
    model = dc.DixonColes.fit(df, xi=0.0)
    pr = model.predict(teams[0], teams[1], 1)
    assert abs(pr['matrix'].sum() - 1) < 1e-9
    assert abs(pr['p_home'] + pr['p_draw'] + pr['p_away'] - 1) < 1e-9
    assert 0 < pr['over_2_5'] < 1 and 0 < pr['btts'] < 1
    assert pr['top_scores'][0][1] >= pr['top_scores'][1][1]
    unknown = model.predict('Nobody FC', teams[1], 1)
    assert not unknown['known_home'] and unknown['known_away']


def test_save_load_roundtrip(tmp_path):
    df, teams, *_ = _synthetic(rounds=2)
    model = dc.DixonColes.fit(df, xi=0.001)
    path = tmp_path / 'dc.json'
    model.save(path)
    again = dc.DixonColes.load(path)
    a, b = model.predict(teams[2], teams[3], 1), again.predict(teams[2], teams[3], 1)
    assert abs(a['p_home'] - b['p_home']) < 1e-12 and again.xi == model.xi


def test_walk_forward_metrics_beat_base_rate_on_synthetic():
    df, *_ = _synthetic(rounds=8)
    pred = dc.walk_forward(df, xi=0.0, step_days=60, min_train=250)
    m = dc.metrics(pred)
    o = dc.outcome_index(df['hg'].values, df['ag'].values)
    base = dc.base_rate_metrics(pred, [float((o == k).mean()) for k in range(3)])
    assert m['n'] > 100
    assert m['log_loss'] < base['log_loss']
    rel = dc.reliability(pred)
    assert set(rel['outcome']) == {'home', 'draw', 'away'}


@pytest.mark.skipif(not os.path.exists(os.path.join(DASHBOARD_DIR, 'matches_summary.parquet')),
                    reason='fixture data not present')
def test_real_data_fit_and_backtest_beat_base_rate():
    ms = pd.read_parquet(os.path.join(DASHBOARD_DIR, 'matches_summary.parquet'))
    matches = dc.matches_from_summary(ms)
    assert len(matches) > 2000
    events = os.path.join(DASHBOARD_DIR, 'raw_events.parquet')
    if os.path.exists(events):
        matches = dc.attach_xg(matches, events)
        assert matches['xg_h'].notna().mean() > 0.8
    model = dc.DixonColes.fit(matches, xi=dc.DEFAULT_XI, l2=1.0, mix=0.25)
    assert model.home_adv > 0
    assert 'Atlético CP' in model.teams
    last_season_start = matches.groupby('season_id')['date'].min().sort_values().iloc[-2]
    pred = dc.walk_forward(matches, xi=dc.DEFAULT_XI, start=last_season_start, step_days=45, l2=1.0, mix=0.25)
    train = matches[matches['date'] < last_season_start]
    o = dc.outcome_index(train['hg'].values, train['ag'].values)
    base = dc.base_rate_metrics(pred, [float((o == k).mean()) for k in range(3)])
    assert dc.metrics(pred)['log_loss'] < base['log_loss'] - 0.01, 'the fitted model must clearly beat a constant forecast'


def test_shipped_params_load_and_predict():
    path = os.path.join(DASHBOARD_DIR, 'models', 'scoreline', 'dc_params.json')
    if not os.path.exists(path):
        pytest.skip('dc_params.json not built')
    model = dc.DixonColes.load(path)
    pr = model.predict('Atlético CP', 'Mafra', 43324)
    assert abs(pr['p_home'] + pr['p_draw'] + pr['p_away'] - 1) < 1e-9
    assert 0.5 < pr['lambda'] < 3 and 0.3 < pr['mu'] < 3
