"""The live-season / per-season engine value (2026-10-29):
eur_intervals.engine_values and the as-of sale pairing, on synthetic frames.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

DASH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, DASH)

from models.value import eur_intervals as ei  # noqa: E402

L3_25, L3_26 = 191782, 192831


def _engine():
    """Player 1: summer projection row (25/26) + a valued live row (26/27).
    Player 2: summer row + a live row whose club hasn't played enough matches.
    Player 3: a past season only (21/22). Player 4: live only (new player).
    A pool filler row keeps the percentiles sane."""
    return pd.DataFrame({
        'playerId':       [1, 1, 2, 2, 3, 4, 9],
        'seasonId':       [L3_25, L3_26, L3_25, L3_26, 188221, L3_26, L3_25],
        'role':           ['Striker'] * 7,
        'mins_played':    [2000, 700, 1800, 600, 1500, 650, 2000],
        'age':            [24.0, np.nan, 25.0, np.nan, np.nan, np.nan, 26.0],
        'projection_abs': [50.0, np.nan, 45.0, np.nan, np.nan, np.nan, 40.0],
        'regular_factor': [1.0, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0],
        'asof_abs':       [50.0, 55.0, 45.0, 47.0, 52.0, 48.0, 40.0],
        'asof_age':       [24.0, 24.8, 25.0, 25.8, 21.0, 22.0, 26.0],
        'asof_w_evidence': [0.6, 0.65, 0.6, 0.62, 0.5, 0.3, 0.6],
        'asof_live':      [False, True, False, True, False, True, False],
        'live_ok':        [False, True, False, False, False, True, False],
    }, index=[10, 11, 20, 21, 30, 40, 90])


def test_without_asof_columns_today_is_the_old_frame():
    e = _engine().drop(columns=['asof_abs', 'asof_age', 'asof_w_evidence', 'asof_live', 'live_ok'])
    now, season = ei.engine_values(e)
    old = pd.to_numeric(ei.engine_value_eur_frame(e), errors='coerce')
    assert np.allclose(now.fillna(-1), old.fillna(-1)) and season.isna().all()


def test_season_values_and_the_live_guard():
    e = _engine()
    now, season = ei.engine_values(e)
    pool = ei.sorted_pool(e)
    # a completed row: its as-of projection at its as-of age, own factor
    assert season[30] == pytest.approx(ei.engine_value_eur(52.0, pool, 'Striker', 21.0, 188221, 1.0))
    # the summer row: its season value IS today's projection value
    assert season[20] == pytest.approx(ei.engine_value_eur(45.0, pool, 'Striker', 25.0, L3_25, 1.0))
    # a live row is valued only once live_ok
    assert season[11] == pytest.approx(ei.engine_value_eur(55.0, pool, 'Striker', 24.8, L3_26, 0.8))
    assert np.isnan(season[21])


def test_today_is_live_once_valued_else_the_summer_value():
    e = _engine()
    now, season = ei.engine_values(e)
    old = pd.to_numeric(ei.engine_value_eur_frame(e), errors='coerce')
    assert now[10] == pytest.approx(season[11])        # player 1: live value on his projection row
    assert np.isnan(now[11])
    assert now[20] == pytest.approx(old[20])           # player 2: live not valued yet -> summer
    assert now[40] == pytest.approx(season[40])        # player 4: no projection row -> on the live row
    assert np.isnan(now[30])                           # player 3: no projection -> no value today


def test_pairing_uses_the_value_as_of_the_sale_season():
    e = _engine().assign(team='X')
    fees = pd.DataFrame({'playerId': [3, 1], 'season_id': [188221, L3_25], 'fee_eur': [100_000, 200_000],
                         'transfer_type': ['permanent', 'permanent'], 'synthetic_flag': [0, 0],
                         'player_name': ['Past', 'Now'], 'to_team': ['Y', 'Y'], 'from_team': ['X', 'X'],
                         'as_of_date': ['2022-07-01', '2026-07-01']})
    pairs = ei.pair_fees_to_engine(fees, e).set_index('playerId')
    _, season = ei.engine_values(e)
    # player 3's 21/22 row has no projection (the old pairing excluded him)
    assert bool(pairs.loc[3, 'included']) and pairs.loc[3, 'value_eur'] == pytest.approx(season[30])
    assert pairs.loc[3, 'age_at_sale'] == pytest.approx(21.0) and pairs.loc[3, 'w_evidence'] == pytest.approx(0.5)
    assert pairs.loc[1, 'value_eur'] == pytest.approx(season[10])


def test_a_value_below_the_pool_floor_is_nan_not_none():
    # pandas 3 refuses None into a float Series: an unvaluable row must be NaN
    e = _engine()
    e.loc[30, 'asof_abs'] = -100.0
    now, season = ei.engine_values(e)
    assert season.dtype == float and np.isnan(season[30])
