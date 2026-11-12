"""Coach tables: no look-ahead in the ratings or the player baselines, and
the arithmetic of expected points, spells and the 'Coached by' shares."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.coaches import build_coaches as bc  # noqa: E402
from models.coaches import coach_metrics as cm  # noqa: E402


def _league(n_rounds=12, seed=0):
    """Six teams, a round a week; team 1 changes coach after round 6."""
    rng = np.random.default_rng(seed)
    rows, mid = [], 100
    teams = [1, 2, 3, 4, 5, 6]
    for rnd in range(n_rounds):
        order = list(rng.permutation(teams))
        for h, a in zip(order[:3], order[3:]):
            mid += 1
            xgd = float(rng.normal(0.2, 1.0))
            for t, o, side, sign in ((h, a, 'home', 1), (a, h, 'away', -1)):
                rows.append({'matchId': mid, 'teamId': int(t), 'oppId': int(o), 'side': side,
                             'competitionId': 1, 'seasonId': 1,
                             'date': pd.Timestamp('2025-01-01') + pd.Timedelta(days=7 * rnd),
                             'xgd': sign * xgd,
                             'coachId': 10 * int(t) + (1 if (t == 1 and rnd >= 6) else 0)})
    return pd.DataFrame(rows)


PRE = ['r_pre', 'r_opp_pre', 'hfa_pre', 'n_rated_pre']


def test_ratings_have_no_look_ahead():
    """A row's pre-match numbers must not move when later matches are removed
    or changed."""
    tm = _league()
    full = bc.add_ratings(tm).set_index(['matchId', 'teamId'])[PRE]
    cut = tm.date.sort_values().unique()[7]
    early = bc.add_ratings(tm[tm.date <= cut]).set_index(['matchId', 'teamId'])[PRE]
    pd.testing.assert_frame_equal(full.loc[early.index], early)
    # and changing the result of a match leaves its own pre-match row alone
    bumped = tm.copy()
    last = bumped.date == bumped.date.max()
    bumped.loc[last, 'xgd'] *= 5
    again = bc.add_ratings(bumped).set_index(['matchId', 'teamId'])[PRE]
    pd.testing.assert_frame_equal(full, again)


def test_first_match_starts_level():
    tm = bc.add_ratings(_league())
    first = tm[tm.date == tm.date.min()]
    assert (first.r_pre == 0).all() and (first.hfa_pre == 0).all() and (first.n_rated_pre == 0).all()


def test_spells_and_inherited_level():
    tm = bc.add_spells(bc.add_ratings(_league()))
    t1 = tm[tm.teamId == 1].sort_values('date')
    assert t1.spell.nunique() == 2
    second = t1[t1.coachId == 11]
    # 6 earlier matches < MIN_BASELINE_MATCHES → no baseline
    assert second.r_inherited.isna().all()
    tm = bc.add_spells(bc.add_ratings(_league(n_rounds=30)))
    t1 = tm[tm.teamId == 1].sort_values('date')
    late = t1.copy()
    late.loc[late.index[:15], 'coachId'] = 10
    late.loc[late.index[15:], 'coachId'] = 11
    other = tm[tm.teamId != 1]
    sp = bc.add_spells(pd.concat([other, late]).drop(columns=['spell', 'r_inherited']))
    new = sp[(sp.teamId == 1) & (sp.coachId == 11)].sort_values('date')
    assert new.r_inherited.notna().all()
    assert new.r_inherited.nunique() == 1 and new.r_inherited.iloc[0] == new.r_pre.iloc[0]


def test_expected_points():
    assert bc.expected_points([], []) == pytest.approx(1.0)
    assert bc.expected_points([0.99, 0.99], []) == pytest.approx(3.0, abs=1e-3)
    a, b = [0.3, 0.1, 0.45], [0.2, 0.6]
    pa, pb = bc.expected_points(a, b), bc.expected_points(b, a)
    # the two sides share 3 points for a decisive match and 2 for a draw
    assert 2.0 < pa + pb < 3.0
    assert bc.expected_points(a, a) == pytest.approx(bc.expected_points(a, a))


def test_coached_by_shares():
    cp = pd.DataFrame([{'coachId': 7, 'teamId': 1, 'playerId': 5, 'team': 'A',
                        'first_date': pd.Timestamp('2025-01-01'), 'last_date': pd.Timestamp('2025-03-01'),
                        'matches': 8, 'starts': 6, 'minutes': 540.0, 'obv_minutes': 540.0, 'obv': 0.6, 'points': 12,
                        'prior_minutes': 900.0, 'prior_obv': 0.5, 'team_matches_in_span': 10,
                        'named': 9, 'named_played': 8}])
    r = cm.coached_by(cp, None, 5).iloc[0]
    assert r.start_share == pytest.approx(0.6) and r.played_share == pytest.approx(0.8)
    assert r.obv90 == pytest.approx(0.1) and r.team_ppg_with == pytest.approx(1.5)
    dev, summ = cm.player_development(cp, 7)
    assert summ['n'] == 1 and dev.change.iloc[0] == pytest.approx(0.1 - 0.05)


BUILT = os.path.join(os.path.dirname(os.path.abspath(bc.__file__)), 'coach_team_match.parquet')


@pytest.mark.skipif(not os.path.exists(BUILT), reason='coach tables not built')
def test_built_table_is_consistent():
    tm = pd.read_parquet(BUILT)
    assert not tm.duplicated(['matchId', 'teamId']).any()
    ev = tm[tm.has_events]
    # the two sides of a match mirror each other
    both = ev.groupby('matchId').filter(lambda g: len(g) == 2)
    assert both.groupby('matchId').xgd.sum().abs().max() < 1e-9
    assert (both.groupby('matchId').xpts.sum().between(2.0, 3.0)).all()
    # the shipped pre-match columns are reproducible walk-forward from a prefix
    cut = tm.date.quantile(0.5)
    cols = ['matchId', 'teamId', 'competitionId', 'side', 'date', 'xgd']
    early = bc.add_ratings(tm.loc[tm.date <= cut, cols]).set_index(['matchId', 'teamId'])[PRE]
    shipped = tm.set_index(['matchId', 'teamId'])[PRE].loc[early.index]
    pd.testing.assert_frame_equal(shipped, early, check_exact=False, atol=1e-12)


def test_unknown_coach_does_not_split_a_spell():
    tm = bc.add_ratings(_league())
    t1 = tm[tm.teamId == 1].sort_values('date')
    tm.loc[t1.index[2], 'coachId'] = 0          # A, A, ?, A …
    sp = bc.add_spells(tm)
    first = sp[(sp.teamId == 1) & (sp.coachId == 10)]
    assert first.spell.nunique() == 1
    assert (sp.loc[sp.coachId == 0, 'spell'] == -1).all()


def _styled_league(n_rounds=140, seed=1):
    tm = bc.add_ratings(_league(n_rounds=n_rounds, seed=seed))
    rng = np.random.default_rng(seed)
    n = len(tm)
    tm['has_events'] = True
    for c in ('passes', 'opp_passes', 'passes_f3', 'opp_passes_f3', 'possessions', 'long_passes',
              'fwd_passes', 'def_actions', 'opp_press_passes', 'press_def'):
        tm[c] = rng.integers(50, 400, n).astype(float)
    tm['def_x_sum'] = tm.def_actions * rng.uniform(30, 40, n)
    return tm


def test_context_adjustment_has_no_look_ahead():
    """A match's z-scores must not move when later months are removed."""
    tm = _styled_league()
    full = bc.add_context_adjusted(tm)
    zc = ['z_' + m for m in bc.Z_METRICS]
    assert full[zc].notna().any().all(), 'the synthetic league never reached MIN_CONTEXT_ROWS'
    cut = tm.date.sort_values().iloc[int(len(tm) * 0.8)]
    early = bc.add_context_adjusted(tm[tm.date <= cut])
    a = full.set_index(['matchId', 'teamId']).loc[early.set_index(['matchId', 'teamId']).index, zc]
    pd.testing.assert_frame_equal(a, early.set_index(['matchId', 'teamId'])[zc])
    # the first month can never be adjusted
    first_month = full.date.dt.to_period('M') == full.date.dt.to_period('M').min()
    assert full.loc[first_month, zc].isna().all().all()


def test_flexibility_and_trust_arithmetic():
    rows = pd.DataFrame({
        'spell': [1] * 4, 'date': pd.date_range('2025-01-01', periods=4, freq='7D'),
        'formation': ['4-3-3', '4-3-3', '3-5-2', '4-3-3'],
        'u21_minutes': [99.0, 0, 99, 0], 'squad_minutes': [990.0] * 4, 'u21_named': [4, 4, 4, 4],
        'sheet_n': [20] * 4, 'u21_starts': [1, 0, 1, 0], 'xi_n': [11] * 4,
        'age_minutes_sum': [990 * 26.0] * 4, 'sheet_age_sum': [20 * 25.0] * 4})
    f = cm.flexibility(rows)
    assert f['formation_change_rate'] == pytest.approx(2 / 3)
    assert f['back_line_change_rate'] == pytest.approx(2 / 3)
    # 3 × 4-3-3 and 1 × 3-5-2: two matches share a shape in 3 of 6 pairs
    assert f['eff_formations'] == pytest.approx(2.0)
    y = cm.youth_trust(rows, 21)
    assert y['u21_minutes_share'] == pytest.approx(0.05) and y['u21_named_share'] == pytest.approx(0.2)
    assert y['u21_trust'] == pytest.approx(0.25)
    assert y['age_on_pitch'] == pytest.approx(26.0) and y['age_of_squad'] == pytest.approx(25.0)


def test_expected_for_age_uses_only_finished_spells():
    """Removing or changing spells that end later must not move an earlier
    pair's expected-for-age change."""
    rng = np.random.default_rng(3)
    n = 120
    start = pd.Timestamp('2022-01-01') + pd.to_timedelta(np.sort(rng.integers(0, 1200, n)), unit='D')
    cp = pd.DataFrame({'coachId': np.arange(n) % 7 + 1, 'playerId': np.arange(n) + 1, 'team': 'T',
                       'obv_minutes': 900.0, 'obv': rng.normal(0.5, 0.3, n), 'prior_minutes': 900.0,
                       'prior_obv': rng.normal(0.5, 0.3, n), 'age_first': 24.0,
                       'first_under_coach': start, 'last_date': start + pd.Timedelta(days=150)})
    full = cm.development_pairs(cp).set_index('playerId')
    assert full.expected.notna().sum() > 20
    cut = start[80]
    early = cm.development_pairs(cp[cp.last_date < cut]).set_index('playerId')
    pd.testing.assert_series_equal(full.loc[early.index, 'expected'], early.expected)
    assert full.sort_values('first_date').expected.iloc[:cm.MIN_PEERS].isna().all()
    # a coach's own earlier players are never his players' peers
    assert cm.development_pairs(cp.assign(coachId=1)).expected.isna().all()


def _state_league(n_rounds=140, seed=2):
    tm = bc.add_ratings(_league(n_rounds=n_rounds, seed=seed))
    rng = np.random.default_rng(seed)
    n = len(tm)
    cut = np.sort(rng.uniform(0, 90, (n, 2)), axis=1)
    tm['min_lead'], tm['min_draw'], tm['min_trail'] = cut[:, 0], cut[:, 1] - cut[:, 0], 90 - cut[:, 1]
    for s in bc.STATES:
        tm[f'xgd_{s}'] = rng.normal(0, 0.3, n)
    tm['state_ok'] = rng.uniform(size=n) > 0.1
    return tm


EXP = [f'exp_{k}_{s}' for k in ('min', 'xgd') for s in bc.STATES]


def test_state_expected_has_no_look_ahead():
    """A match's expected game-state numbers must not move when later months
    are removed or their results changed."""
    tm = _state_league()
    full = bc.add_state_expected(tm).set_index(['matchId', 'teamId'])
    assert full[EXP].notna().any().all(), 'the synthetic league never reached MIN_CONTEXT_ROWS'
    cut = tm.date.sort_values().iloc[int(len(tm) * 0.8)]
    early = bc.add_state_expected(tm[tm.date <= cut]).set_index(['matchId', 'teamId'])
    pd.testing.assert_frame_equal(full.loc[early.index, EXP], early[EXP])
    # a match's own result is not in its own expected minutes
    bumped = tm.copy()
    last = bumped.date.dt.to_period('M') == bumped.date.dt.to_period('M').max()
    bumped.loc[last, 'min_lead'], bumped.loc[last, 'min_trail'] = bumped.loc[last, 'min_trail'], bumped.loc[last, 'min_lead']
    again = bc.add_state_expected(bumped).set_index(['matchId', 'teamId'])
    cols = [f'exp_min_{s}' for s in bc.STATES]
    pd.testing.assert_frame_equal(full[cols], again[cols])
    # expected minutes add up to the match; rows without a clean timeline get none
    have = full[full.exp_min_lead.notna()]
    assert np.allclose(have[cols].sum(axis=1), have[['min_lead', 'min_draw', 'min_trail']].sum(axis=1))
    assert full.loc[~full.state_ok, EXP].isna().all().all()
    first_month = full.date.dt.to_period('M') == full.date.dt.to_period('M').min()
    assert full.loc[first_month, EXP].isna().all().all()


@pytest.mark.skipif(not os.path.exists(BUILT), reason='coach tables not built')
def test_state_vs_expected_arithmetic():
    tm = pd.read_parquet(BUILT)
    tm = tm[tm.coachId == tm[tm.coachId > 0].coachId.mode().iloc[0]].copy()
    tm['state_ok'] = True
    for s, m, x, em, ex in (('lead', 30.0, 0.3, 20.0, 0.1), ('draw', 45.0, 0.0, 50.0, 0.0), ('trail', 15.0, 0.2, 20.0, 0.3)):
        tm[f'min_{s}'], tm[f'xgd_{s}'], tm[f'exp_min_{s}'], tm[f'exp_xgd_{s}'] = m, x, em, ex
    s = cm.summarise(tm)
    assert s['exp_min_share_lead'] == pytest.approx(20 / 90) and s['min_share_vs_exp_lead'] == pytest.approx(10 / 90)
    assert s['exp_xgd90_lead'] == pytest.approx(0.1 * 90 / 30) and s['xgd90_vs_exp_lead'] == pytest.approx(0.2 * 90 / 30)
    assert s['xgd90_vs_exp_trail'] == pytest.approx(-0.1 * 90 / 15)
