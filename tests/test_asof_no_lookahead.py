"""As-of fits (models/ratings/asof.py and their users): what a season-year's
rows use is unchanged when that year's or any later year's data changes
(Lucas 2026-10-29: no data leakage, even where it buys no accuracy).
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

DASH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, DASH)

from models.ratings import asof as A  # noqa: E402


def _pairs(seed=0, years=(2022, 2023, 2024, 2025), n=40):
    rng = np.random.default_rng(seed)
    rows = []
    for y in years:
        a = rng.random(n)
        rows += [(x, 0.6 * x + 0.4 * rng.random(), y) for x in a]
    return pd.DataFrame(rows, columns=['a', 'b', 'yr'])


def test_reliability_ignores_the_rows_own_season_and_later():
    p = _pairs()
    lam = A.lam_by_year(p, [2022, 2023, 2024, 2025, 2026])
    q = p.copy()
    q.loc[q['yr'] >= 2024, 'b'] = 1 - q.loc[q['yr'] >= 2024, 'a']    # wreck 2024+ pairs
    lam2 = A.lam_by_year(q, [2022, 2023, 2024, 2025, 2026])
    assert lam2[2024] == lam[2024] and lam2[2023] == lam[2023]
    assert lam2[2025] != lam[2025]
    # no pair ended before 2022 or 2023 (only 2022 pairs exist for 2023: 40 >= 30)
    assert lam[2022] == A.LAM_PRIOR
    assert lam[2023] != A.LAM_PRIOR


def test_a_past_season_value_never_sees_later_seasons():
    """eur_intervals.engine_values: a 2021 or 2025 value is unchanged when
    later seasons' rows appear or change (the pool is the one of its year)."""
    from models.value import eur_intervals as ei
    e = pd.DataFrame({
        'playerId': [1, 2, 3, 9], 'seasonId': [191782, 191782, 188221, 191782],
        'role': ['Striker'] * 4, 'mins_played': [2000, 1800, 1500, 2000],
        'age': [24.0, 25.0, np.nan, 26.0], 'projection_abs': [50.0, 45.0, np.nan, 40.0],
        'regular_factor': [1.0] * 4, 'asof_abs': [50.0, 45.0, 52.0, 40.0],
        'asof_age': [24.0, 25.0, 21.0, 26.0], 'asof_live': [False] * 4, 'live_ok': [False] * 4},
        index=[10, 20, 30, 90])
    _, s0 = ei.engine_values(e, as_of='2026-01-01', pools=None)
    later = pd.concat([e.assign(projection_abs=e['projection_abs'] + 7.0),      # today's pool moves
                       pd.DataFrame({'playerId': [5], 'seasonId': [192831], 'role': ['Striker'],
                                     'mins_played': [900], 'age': [np.nan], 'projection_abs': [np.nan],
                                     'regular_factor': [1.0], 'asof_abs': [70.0], 'asof_age': [23.0],
                                     'asof_live': [True], 'live_ok': [True]}, index=[50])])
    _, s1 = ei.engine_values(later, as_of='2026-01-01', pools=None)
    assert s1[30] == s0[30] and s1[20] == s0[20] and s1[10] == s0[10]
    # a 2021 value doesn't move when 2025 rows change
    e2 = e.copy(); e2.loc[[10, 20, 90], 'asof_abs'] += 5.0
    assert ei.engine_values(e2, as_of='2026-01-01', pools=None)[1][30] == s0[30]


def test_pooled_reliability_ignores_the_rows_own_season_and_later():
    p, q = _pairs(1, n=12), _pairs(2, n=60)        # a thin role, the category over every role
    lam = A.lam_by_year(p, [2023, 2024, 2025], pooled=q)
    q2 = q.copy()
    q2.loc[q2['yr'] >= 2024, 'b'] = 1 - q2.loc[q2['yr'] >= 2024, 'a']
    p2 = p.copy()
    p2.loc[p2['yr'] >= 2024, 'b'] = p2.loc[p2['yr'] >= 2024, 'a']
    lam2 = A.lam_by_year(p2, [2023, 2024, 2025], pooled=q2)
    assert lam2[2023] == lam[2023] and lam2[2024] == lam[2024]
    # 12 own pairs before 2023 vs 60 pooled: shrunk 12/(12+30) toward the pooled r
    r_own = A._r(p[p['yr'] < 2023]); r_pool = A._r(q[q['yr'] < 2023])
    assert lam[2023] == pytest.approx((12 * r_own + 30 * r_pool) / 42)
    assert A.lam_by_year(p, [2022], pooled=q)[2022] == A.LAM_PRIOR


def test_playing_time_coefficients_only_from_finished_outcomes():
    """availability.coef_for: a row of year y uses fits whose outcome season
    (fit year + 1) was played by the end of y; none before -> factor 1.0."""
    from models.value import availability as av
    assert av.coef_for(2021) is None and av.coef_for(2022) is None
    assert av.coef_for(2023) == av.COEF_ASOF[2023] and av.coef_for(2024) == av.COEF_ASOF[2023]
    assert av.coef_for(2025) == av.COEF and av.coef_for(2026) == av.COEF
    for first, fit_years in ((2023, (2022,)), (2025, av.FIT_YEARS)):
        assert all(fy + 1 <= first for fy in fit_years)


def test_a_mid_season_sale_never_sees_its_season_after_the_sale():
    """eur_intervals.pair_fees_to_engine: a sale before its season's last
    match is paired with the sale-date projection (build_projection), never
    with the sale season's as-of row; a summer sale with its season's row."""
    from models.value import eur_intervals as ei
    e = pd.DataFrame({
        'playerId': [1, 1], 'seasonId': [190090, 191782], 'role': ['Striker'] * 2, 'team': ['X', 'X'],
        'mins_played': [2000, 900], 'age': [24.0, 25.0], 'projection_abs': [np.nan, 50.0],
        'regular_factor': [1.0, 1.0], 'asof_abs': [48.0, 70.0], 'asof_age': [24.0, 25.0],
        'asof_w_evidence': [0.6, 0.65], 'asof_evidence': [2000.0, 2600.0],
        'asof_live': [False, False], 'live_ok': [False, False]}, index=[10, 11])
    fees = pd.DataFrame({'playerId': [1], 'season_id': [191782], 'fee_eur': [100_000],
                         'transfer_type': ['permanent'], 'synthetic_flag': [0], 'player_name': ['P'],
                         'to_team': ['Y'], 'from_team': ['X'], 'as_of_date': ['2026-01-15']})
    sales = pd.DataFrame({'playerId': [1], 'season_id': [191782], 'as_of_date': ['2026-01-15'],
                          'base_seasonId': [190090], 'used_sale_season': [False],
                          'projection_sale': [49.0], 'projection_sale_abs': [49.0], 'age': [25.04],
                          'value_role': ['Striker'], 'w_evidence': [0.6], 'evidence': [2100.0]})
    ends = {2024: pd.Timestamp('2025-05-20'), 2025: pd.Timestamp('2026-05-20')}
    p = ei.pair_fees_to_engine(fees, e, ends=ends, sales=sales, pools=None).iloc[0]
    assert bool(p['mid_season']) and p['paired_season_id'] == 190090
    assert p['value_eur'] == pytest.approx(ei.engine_value_eur(49.0, ei.asof_pool(e, 2024), 'Striker', 25.04, 190090, 1.0))
    # the 25/26 row (asof 70: the season after the sale) changes nothing
    e2 = e.copy(); e2.loc[11, 'asof_abs'] = 30.0
    assert ei.pair_fees_to_engine(fees, e2, ends=ends, sales=sales, pools=None).iloc[0]['value_eur'] == p['value_eur']
    # no sale-date projection: no value, not the sale season's row
    q = ei.pair_fees_to_engine(fees, e, ends=ends, sales=sales.iloc[0:0], pools=None).iloc[0]
    assert q['exclusion'] == 'no_data_before_sale'
