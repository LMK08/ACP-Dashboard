"""Season simulation on the Dixon-Coles model: the sampler reproduces the
model's own probabilities, the table bookkeeping is exact, and (with the
fixture data present) a Liga 3 run yields the pickle the Home page and the
Match Predictor read, with the slot counts the league format implies."""
import os
import pickle
import sys

import numpy as np
import pandas as pd
import pytest

DASHBOARD_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, DASHBOARD_DIR)

from models.scoreline import dixon_coles as dc  # noqa: E402
from models.scoreline import season_sim  # noqa: E402
from models.scoreline.season_sim import ScorelineSampler  # noqa: E402

PARAMS = os.path.join(DASHBOARD_DIR, 'models', 'scoreline', 'dc_params.json')
MATCHES = os.path.join(DASHBOARD_DIR, 'matches_summary.parquet')
needs_data = pytest.mark.skipif(not (os.path.exists(PARAMS) and os.path.exists(MATCHES)),
                                reason='dc_params.json / matches_summary.parquet not present')


def _toy_model():
    return dc.DixonColes(teams=['A', 'B', 'C', 'D'], leagues=[1],
                         att=np.array([0.3, 0.0, -0.2, 0.1]), dfn=np.array([0.1, -0.1, 0.0, 0.2]),
                         base=np.array([0.2]), home_adv=0.25, rho=-0.08, asof='2026-01-01', n_matches=10)


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------
def test_sampler_rows_are_the_models_own_distribution():
    m = _toy_model()
    s = ScorelineSampler(m, 1)
    cum = s.cumulative('A', 'B')
    assert cum.shape == (81,) and cum[-1] == 1.0
    assert np.all(np.diff(cum) >= -1e-12)
    pr = m.predict('A', 'B', 1)
    op = s.outcome_probs('A', 'B')
    assert op['home'] == pytest.approx(pr['p_home'], abs=1e-9)
    assert op['draw'] == pytest.approx(pr['p_draw'], abs=1e-9)
    assert op['away'] == pytest.approx(pr['p_away'], abs=1e-9)
    # The 1-1 cell carries the Dixon-Coles correction, so it must match the matrix exactly
    P = np.diff(np.concatenate([[0.0], cum])).reshape(9, 9)
    assert P[1, 1] == pytest.approx(pr['matrix'][1, 1], abs=1e-12)
    rows = s.rows([('A', 'B'), ('C', 'D')])
    assert rows.shape == (2, 81)
    assert s.rows([]).shape == (0, 81)
    assert s.cumulative('A', 'B') is cum          # cached per pairing
    assert s.unknown_teams(['A', 'Zed']) == ['Zed']
    # An unseen team plays at zero parameters, like DixonColes.predict
    assert s.outcome_probs('A', 'Zed')['home'] == pytest.approx(m.predict('A', 'Zed', 1)['p_home'], abs=1e-9)


def test_sampling_reproduces_outcomes_and_goal_means():
    m = _toy_model()
    s = ScorelineSampler(m, 1)
    rng = np.random.default_rng(0)
    n = 60_000
    rows = np.repeat(s.cumulative('A', 'B')[None, :], n, axis=0)
    hg, ag = season_sim.sample_scores(rows, rng.random(n))
    pr = m.predict('A', 'B', 1)
    assert hg.dtype.kind == 'i' and ag.dtype.kind == 'i'
    assert (hg > ag).mean() == pytest.approx(pr['p_home'], abs=0.01)
    assert (hg == ag).mean() == pytest.approx(pr['p_draw'], abs=0.01)
    assert (hg < ag).mean() == pytest.approx(pr['p_away'], abs=0.01)
    assert hg.mean() == pytest.approx(pr['lambda'], abs=0.03)
    assert ag.mean() == pytest.approx(pr['mu'], abs=0.03)
    assert ((hg == 1) & (ag == 1)).mean() == pytest.approx(pr['matrix'][1, 1], abs=0.01)


def test_sample_scores_inverse_cdf_edges():
    # 2 goals per side: cells (0,0) (0,1) (1,0) (1,1)
    rows = np.array([[0.25, 0.5, 0.75, 1.0]] * 4)
    hg, ag = season_sim.sample_scores(rows, np.array([0.0, 0.3, 0.6, 0.999]))
    assert hg.tolist() == [0, 0, 1, 1] and ag.tolist() == [0, 1, 0, 1]
    hg, ag = season_sim.sample_scores(np.empty((0, 4)), np.empty(0))
    assert hg.size == 0 and ag.size == 0


def test_apply_results_bookkeeping_is_exact():
    pts = {'A': 1, 'B': 0, 'C': 0}
    gd = {'A': 0, 'B': 0, 'C': 0}
    gf = {'A': 0, 'B': 0, 'C': 0}
    season_sim.apply_results([('A', 'B'), ('B', 'C'), ('C', 'A')],
                             np.array([2, 1, 3]), np.array([0, 1, 0]), pts, gd, gf)
    assert pts == {'A': 4, 'B': 1, 'C': 4}
    assert gd == {'A': -1, 'B': -2, 'C': 3}
    assert gf == {'A': 2, 'B': 1, 'C': 4}
    assert all(isinstance(v, int) for v in pts.values())


# ---------------------------------------------------------------------------
# Real data
# ---------------------------------------------------------------------------
@needs_data
def test_liga3_run_matches_the_format():
    import simulate_season as ss
    matches = pd.read_parquet(MATCHES)
    sampler = ScorelineSampler(season_sim.load_params(PARAMS), 43324)
    res = ss.simulate_liga3(matches, sampler, n_sims=300)
    assert res, 'no Liga 3 groups simulated'
    for name, g in res.items():
        pos = g['position_probabilities']
        n = len(g['teams'])
        assert list(pos.columns) == [str(i + 1) for i in range(n)]
        assert set(pos.index) == set(g['teams'])
        assert np.allclose(pos.sum(axis=1), 1.0, atol=1e-9)
        st = g['current_standings']
        assert {'Pos', 'Team', 'P', 'Pts', 'GD', 'GF'} <= set(st.columns)
        assert int(g['matches_remaining']) >= 0
        xpts = g.get('expected_pts')
        assert xpts is not None and set(xpts) == set(g['teams'])
        for _, r in st.iterrows():
            remaining_for_team = 2 * (n - 1) - int(r['P'])
            lo = int(r['Pts']) + g.get('bonus_points', {}).get(r['Team'], 0)
            assert lo - 1e-9 <= xpts[r['Team']] <= lo + 3 * remaining_for_team + 1e-9

    series = [k for k in res if k.startswith('Série')]
    if series:  # first phase: chained second phase fills fixed slot counts
        assert len(series) == 2
        for k in series:
            g = res[k]
            assert sum(g['playoff_pct'].values()) == pytest.approx(4.0, abs=1e-9)
            assert sum(g['releg_pct'].values()) == pytest.approx(2.0, abs=1e-9)
            for t in g['teams']:
                if not ss.is_promotion_eligible(t):
                    assert g['promotion_pct'][t] == 0.0
        assert sum(sum(res[k]['promotion_pct'].values()) for k in series) == pytest.approx(2.0, abs=1e-9)
    # Deterministic: seeded generators, no process-salted hashes
    again = ss.simulate_liga3(matches, ScorelineSampler(season_sim.load_params(PARAMS), 43324), n_sims=300)
    for k in res:
        assert res[k]['promotion_pct'] == again[k]['promotion_pct']
        pd.testing.assert_frame_equal(res[k]['position_probabilities'], again[k]['position_probabilities'])


@needs_data
def test_refit_keeps_the_tuned_hyperparameters():
    matches = pd.read_parquet(MATCHES)
    prior = dc.DixonColes.load(PARAMS)
    model, info = season_sim.refit(matches, params_path=PARAMS, events_path=None)
    assert (model.xi, model.l2, model.mix) == (prior.xi, prior.l2, prior.mix)
    assert model.n_matches == len(dc.matches_from_summary(matches))
    assert set(prior.teams) <= set(model.teams)
    meta = season_sim.model_meta(model, info)
    assert meta['name'] == 'dixon_coles_v1' and meta['refit'] is True and meta['xg_attached'] is False
    assert meta['asof'] == model.asof and meta['n_matches'] == model.n_matches
    committed = season_sim.model_meta(prior)
    assert committed['refit'] is False and committed['asof'] == prior.asof


@needs_data
def test_main_writes_the_pickle_the_pages_read(tmp_path):
    import simulate_season as ss
    from league_config import COMPETITIONS
    out = tmp_path / 'sim.pkl'
    ss.main(['--sims', '40', '--no-refit', '--out', str(out)])
    with open(out, 'rb') as fh:
        d = pickle.load(fh)
    assert d['n_simulations'] == 40
    assert d['model']['name'] == 'dixon_coles_v1' and d['model']['refit'] is False
    assert d['season_id'] == COMPETITIONS[43324]['current_season']
    assert d['groups'] and d['competitions'][43324]['groups'] is d['groups']
    # --no-refit must leave the committed parameters untouched
    assert dc.DixonColes.load(PARAMS).asof == d['model']['asof']


@needs_data
def test_load_scoreline_model_write_params_semantics(tmp_path):
    """A refit never touches dc_params.json unless --write-params asks; a
    failed refit falls back to the committed parameters."""
    import shutil
    import simulate_season as ss
    params = tmp_path / 'dc_params.json'
    shutil.copy(PARAMS, params)
    before = params.read_bytes()
    matches = pd.read_parquet(MATCHES)
    model, meta = ss.load_scoreline_model(matches, refit=True, write_params=False,
                                          events_path=None, params_path=str(params))
    assert meta['refit'] is True and meta['xg_attached'] is False
    assert params.read_bytes() == before
    model2, meta2 = ss.load_scoreline_model(matches, refit=True, write_params=True,
                                            events_path=None, params_path=str(params))
    saved = dc.DixonColes.load(str(params))
    assert saved.asof == model2.asof == meta2['asof'] and saved.n_matches == model2.n_matches
    assert (saved.xi, saved.l2, saved.mix) == (model.xi, model.l2, model.mix)
    # Nothing to fit on -> the committed parameters, flagged as such
    model3, meta3 = ss.load_scoreline_model(matches.iloc[0:0], refit=True, write_params=True,
                                            events_path=None, params_path=str(params))
    assert meta3['refit'] is False and meta3['asof'] == saved.asof
    assert dc.DixonColes.load(str(params)).asof == saved.asof


def test_liga3_counts_only_matches_with_a_result_as_played():
    """A fixture listed ahead of time (no score) stays in the remaining set."""
    import simulate_season as ss
    from league_config import COMPETITIONS
    sid = COMPETITIONS[43324]['current_season']
    north = ss.FIRST_STAGE_GROUPS['North']
    rows = [
        {'matchId': 1, 'seasonId': sid, 'status': 'Played', 'roundId': 7, 'gameweek': 1,
         'dateutc': '2026-08-08 16:00:00', 'homeTeamName': north[0], 'awayTeamName': north[1],
         'score': '1-0', 'competitionId': 43324},
        {'matchId': 2, 'seasonId': sid, 'status': 'Played', 'roundId': 7, 'gameweek': 1,
         'dateutc': '2026-08-08 16:00:00', 'homeTeamName': north[2], 'awayTeamName': north[3],
         'score': '2-2', 'competitionId': 43324},
        {'matchId': 3, 'seasonId': sid, 'status': 'Fixture', 'roundId': 7, 'gameweek': 2,
         'dateutc': '2026-08-15 16:00:00', 'homeTeamName': north[0], 'awayTeamName': north[2],
         'score': '? - ?', 'competitionId': 43324},
        {'matchId': 4, 'seasonId': sid, 'status': 'Fixture', 'roundId': 8, 'gameweek': 1,
         'dateutc': '2027-03-06 16:00:00', 'homeTeamName': north[0], 'awayTeamName': north[1],
         'score': None, 'competitionId': 43324},
    ]
    res = ss.simulate_liga3(pd.DataFrame(rows), ScorelineSampler(_toy_model(), 43324), n_sims=20)
    assert set(res) == {'Série A (North)', 'Série B (South)'}   # the round-8 row did not start a second stage
    g = res['Série A (North)']
    assert g['matches_remaining'] == 90 - 2
    assert int(g['current_standings']['P'].sum()) == 4
    assert res['Série B (South)']['matches_remaining'] == 90
