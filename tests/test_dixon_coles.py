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


def test_strength_table_is_interpretable_and_neutral_for_unseen_teams():
    """scores_x / concedes_x are exp(±(param − mean of the teams shown)); xGD vs
    avg is the average side's goal rate times their difference; an unseen team
    sits at the average (1 / 1 / 0); best first."""
    df, _teams, _att, _dfn = _synthetic()
    m = dc.DixonColes.fit(df, xi=0.0, l2=0.001)
    tbl = m.strength_table(teams=list(m.teams) + ['Never Played'], league=m.leagues[0])
    assert list(tbl.columns) == ['team', 'attack', 'defence', 'rating', 'scores_x', 'concedes_x',
                                 'xgd_vs_avg', 'known']
    assert tbl['xgd_vs_avg'].is_monotonic_decreasing
    row = tbl.set_index('team').loc['Never Played']
    assert not row['known']
    assert row['scores_x'] == pytest.approx(1.0) and row['concedes_x'] == pytest.approx(1.0)
    assert row['xgd_vs_avg'] == pytest.approx(0.0)
    known = tbl[tbl['known']]
    a_bar, d_bar = known['attack'].mean(), known['defence'].mean()
    avg_goals = float(np.exp(m.base[0] + a_bar - d_bar))
    for _, r in known.iterrows():
        assert r['scores_x'] == pytest.approx(np.exp(r['attack'] - a_bar))
        assert r['concedes_x'] == pytest.approx(np.exp(-(r['defence'] - d_bar)))
        assert r['xgd_vs_avg'] == pytest.approx(avg_goals * (r['scores_x'] - r['concedes_x']))
    # Centred on the teams shown: the average side really is 1.0 on both.
    assert np.log(known['scores_x']).mean() == pytest.approx(0.0, abs=1e-9)
    assert np.log(known['concedes_x']).mean() == pytest.approx(0.0, abs=1e-9)
    # The strongest side beats the average on net; the weakest loses to it.
    assert known.iloc[0]['xgd_vs_avg'] > 0 > known.iloc[-1]['xgd_vs_avg']
    # centre=False reports against a zero-parameter side instead.
    raw = m.strength_table(teams=m.teams[:3], centre=False).set_index('team')
    for t in m.teams[:3]:
        a, d, _ = m._team_params(t)
        assert raw.loc[t, 'scores_x'] == pytest.approx(np.exp(a))
        assert raw.loc[t, 'concedes_x'] == pytest.approx(np.exp(-d))
    sub = m.strength_table(teams=m.teams[:3])
    assert set(sub['team']) == set(m.teams[:3]) and sub['xgd_vs_avg'].is_monotonic_decreasing
    assert m.strength_table(teams=[]).empty


# ---------------------------------------------------------------------------
# League-mean prior
# ---------------------------------------------------------------------------
def _two_leagues(seed=3, n_per=10, rounds=3, gap_att=0.3, gap_def=0.4, movers=4):
    """Two leagues with a true tier gap (league 2's sides weaker by gap_att /
    gap_def); in season 2 `movers` clubs go up and as many league-1 clubs go
    down. Returns the matches, a DixonColes holding the true parameters, the
    promoted clubs, the league-1 clubs that stayed, the date season 2 starts,
    and the true gap between the leagues' season-2 MEMBERSHIP means (attack,
    defence) — the quantity the league-mean prior estimates (the movers pull
    the memberships' means toward each other)."""
    rng = np.random.default_rng(seed)
    top = [f'T{i:02d}' for i in range(n_per)]
    low = [f'L{i:02d}' for i in range(n_per)]
    att = {t: rng.normal(0, 0.25) for t in top + low}
    dfn = {t: rng.normal(0, 0.25) for t in top + low}
    for t in low:
        att[t] -= gap_att
        dfn[t] -= gap_def
    rows, day = [], [0]

    def play(teams, league, season):
        for _ in range(rounds):
            pairs = [(i, j) for i in teams for j in teams if i != j]
            rng.shuffle(pairs)   # random fixture order: after k matches each side has played ~2k/n
            for i, j in pairs:
                lam = np.exp(0.2 + 0.25 + att[i] - dfn[j])
                mu = np.exp(0.2 + att[j] - dfn[i])
                rows.append({'match_id': len(rows), 'home': i, 'away': j, 'hg': rng.poisson(lam), 'ag': rng.poisson(mu),
                             'date': pd.Timestamp('2024-08-01') + pd.Timedelta(days=day[0]),
                             'league': league, 'season_id': season})
                day[0] += 1
    play(top, 1, 1)
    play(low, 2, 1)
    up, down = low[:movers], top[-movers:]
    season2 = pd.Timestamp('2024-08-01') + pd.Timedelta(days=day[0])
    league1 = [t for t in top if t not in down] + up
    league2 = [t for t in low if t not in up] + down
    play(league1, 1, 2)
    play(league2, 2, 2)
    truth = dc.DixonColes(teams=top + low, leagues=[1, 2], att=np.array([att[t] for t in top + low]),
                          dfn=np.array([dfn[t] for t in top + low]), base=np.array([0.2, 0.2]), home_adv=0.25, rho=0.0)
    true_gap = (float(np.mean([att[t] for t in league1]) - np.mean([att[t] for t in league2])),
                float(np.mean([dfn[t] for t in league1]) - np.mean([dfn[t] for t in league2])))
    return pd.DataFrame(rows), truth, up, [t for t in top if t not in down], season2, true_gap


def test_objective_gradient_matches_finite_differences():
    from scipy.optimize import approx_fprime
    df, *_ = _two_leagues(n_per=4, rounds=1)
    teams = sorted(set(df['home']) | set(df['away']))
    leagues = sorted(set(df['league']))
    t_idx = {t: i for i, t in enumerate(teams)}
    l_idx = {l: i for i, l in enumerate(leagues)}
    hi = df['home'].map(t_idx).values
    ai = df['away'].map(t_idx).values
    li = df['league'].map(l_idx).values
    x = df['hg'].values.astype(float)
    y = df['ag'].values.astype(float)
    w = np.exp(-0.002 * np.arange(len(df))[::-1].astype(float))
    n, L = len(teams), len(leagues)
    last = dc.team_leagues(df)
    tl = np.array([l_idx[int(last[t])] for t in teams])
    for league_mean in (False, True):
        k = 2 * n + 2 + L + (2 * L if league_mean else 0)
        theta = np.random.default_rng(1).normal(0, 0.3, k)
        theta[2 * n + 1] = -0.05
        args = (hi, ai, li, tl, x, y, x, y, w, n, L, 1.0, league_mean)
        _, grad = dc._objective(theta, *args)
        numeric = approx_fprime(theta, lambda th: dc._objective(th, *args)[0], 1e-6)
        assert np.allclose(grad, numeric, rtol=1e-4, atol=1e-4), (league_mean, np.abs(grad - numeric).max())


def test_league_mean_prior_recovers_the_tier_gap_and_stops_over_rating_movers():
    df, truth, up, stayers, season2, (gap_att, gap_def) = _two_leagues()
    zero = dc.DixonColes.fit(df, xi=0.0, l2=1.0, prior='zero')
    assert zero.tier_gap(1, 2) is None and zero.league_means == {1: (0.0, 0.0), 2: (0.0, 0.0)}
    lm = dc.DixonColes.fit(df, xi=0.0, l2=1.0, prior='league_mean')
    scores_x, concedes_x = lm.tier_gap(1, 2)
    # The gap between the leagues' membership means, read off the movers' matches in both leagues
    assert abs(np.log(scores_x) + gap_att) < 0.15, (scores_x, gap_att)
    assert abs(np.log(concedes_x) - gap_def) < 0.15, (concedes_x, gap_def)
    # Three matchdays into season 2 a promoted club's forecast still rests on
    # its league-2 record. Shrunk toward one shared centre it is over-rated
    # against league-1 stayers; toward its own league's mean it sits closer
    # to the truth.
    early = season2 + pd.Timedelta(days=15)
    fits = {'zero': dc.DixonColes.fit(df, asof=early, xi=0.0, l2=1.0, prior='zero'),
            'lm': dc.DixonColes.fit(df, asof=early, xi=0.0, l2=1.0, prior='league_mean')}
    err = {k: [] for k in fits}
    bias = {k: [] for k in fits}
    for u in up:
        for t in stayers:
            p_true = truth.predict(u, t, 1)['p_home']
            for name, model in fits.items():
                p = model.predict(u, t, 1)['p_home']
                err[name].append(abs(p - p_true))
                bias[name].append(p - p_true)
    assert np.mean(bias['zero']) > 0.05
    assert np.mean(bias['lm']) < np.mean(bias['zero']) - 0.03
    assert np.mean(err['lm']) < np.mean(err['zero'])


def test_league_mean_prior_roundtrip_and_unseen_teams(tmp_path):
    df, *_ = _two_leagues(n_per=5, rounds=1)
    lm = dc.DixonColes.fit(df, xi=0.001, l2=1.0, prior='league_mean')
    path = tmp_path / 'p.json'
    lm.save(path)
    back = dc.DixonColes.load(path)
    assert back.prior == 'league_mean'
    assert back.league_means.keys() == lm.league_means.keys()
    for l in lm.league_means:
        assert np.allclose(back.league_means[l], lm.league_means[l])
    # An unseen team plays at its league's mean, in either league
    for league in (1, 2):
        a, d, known = back._team_params('Nobody FC', league)
        assert not known and (a, d) == back.league_mean(league)
    assert back.predict('T00', 'Nobody FC', 1)['p_home'] == pytest.approx(lm.predict('T00', 'Nobody FC', 1)['p_home'])
    assert back.predict('T00', 'Nobody FC', 1)['p_home'] != pytest.approx(back.predict('T00', 'Nobody FC', 2)['p_home'])
    # The strength table still shows an unseen side at the average of the sides shown
    tbl = back.strength_table(teams=['T00', 'T01', 'Nobody FC'], league=1)
    row = tbl[tbl['team'] == 'Nobody FC'].iloc[0]
    assert not row['known'] and row['scores_x'] == pytest.approx(1.0) and row['concedes_x'] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        dc.DixonColes.fit(df, prior='nonsense')


def test_promoted_side_bias_table_shape():
    df, *_ = _two_leagues(n_per=5, rounds=1)
    # Re-label the leagues as the real ids so the category logic applies
    df['league'] = df['league'].map({1: 43324, 2: 702})
    cats = dc.team_season_categories(df)
    assert cats[('L00', 2)] == 'promoted from CdP' and cats[('T04', 2)] == 'relegated to CdP'
    assert cats[('T00', 1)] == 'new to the data' and cats[('T00', 2)] == 'stayed'
    pred = dc.walk_forward(df, xi=0.0, start=df[df['season_id'] == 2]['date'].min(), step_days=400, l2=1.0)
    rows = dc.promoted_side_bias(pred, df)
    assert rows and rows[0]['matches'] == '1-6' and rows[0]['n'] > 0
    assert {'n', 'exp_pts', 'act_pts', 'exp_gd', 'act_gd'} <= set(rows[0])
    assert 0.0 <= rows[0]['exp_pts'] <= 3.0 and 0.0 <= rows[0]['act_pts'] <= 3.0


def test_league_mean_prior_rejects_teams_without_matches():
    df, *_ = _two_leagues(n_per=4, rounds=1)
    with pytest.raises(ValueError):
        dc.DixonColes.fit(df, prior='league_mean', teams=['Ghost FC'])
    # the zero prior still accepts extras (they sit at zero, as before)
    m = dc.DixonColes.fit(df, prior='zero', teams=['Ghost FC'])
    assert 'Ghost FC' in m.teams
