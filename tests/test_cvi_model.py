"""Behavioural checks on the CVI market-value model (models/value/cvi.py).

Pure Python, no data files, no Streamlit: these pin the shape of the model
so a refactor or a retune that breaks a monotonicity or a bound is caught
before it reaches a scout's screen.
"""
import os
import sys

import pytest

DASHBOARD_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, DASHBOARD_DIR)

from models.value import cvi  # noqa: E402

LIGA3, CAMPEONATO = 43324, 702
POSITION_GROUPS = ['GK', 'CB', 'FB', 'CM', 'AM_WG', 'ST']


def test_module_is_streamlit_free():
    assert 'streamlit' not in sys.modules, "importing the model must not pull in Streamlit"
    assert len(cvi.__all__) >= 30


def test_radar_templates_load_from_config():
    weights, groups = cvi._radar_templates()
    assert len(weights) >= 20 and set(weights) == set(groups)
    assert 'Poacher' in weights and 'Shot Stopper' in groups


@pytest.mark.parametrize('pos', POSITION_GROUPS)
def test_age_multiplier_bounded_and_declines_after_peak(pos):
    mults = {age: cvi._cvi_age_value_multiplier(age, pos) for age in range(17, 38)}
    assert all(0.5 <= m <= 1.4 for m in mults.values()), mults
    params = cvi.CVI_AGE_VALUE_PARAMS[pos]
    peak = params['peak_age'] if 'peak_age' in params else params.get('peak', 26)
    # Past the decline age the multiplier must not increase with age.
    decline = params.get('decline_age', params.get('decline', peak + 3))
    tail = [mults[a] for a in range(int(decline), 38)]
    assert all(b <= a + 1e-9 for a, b in zip(tail, tail[1:])), tail


@pytest.mark.parametrize('pos', POSITION_GROUPS)
def test_reliability_weight_monotone_in_minutes(pos):
    weights = [cvi._cvi_reliability_weight(m, pos)[0] for m in (0, 90, 450, 900, 1800, 3600, 7200)]
    assert all(0.0 <= w <= 1.0 for w in weights), weights
    assert all(b >= a - 1e-9 for a, b in zip(weights, weights[1:])), weights
    assert weights[-1] <= cvi.CVI_RELIAB_CEILING_BY_POS[pos] + 1e-9


def test_projected_eur_monotone_in_cvi_and_penalises_campeonato():
    eur = [cvi.cvi_to_projected_eur(c, 'CM', LIGA3) for c in (60, 80, 100, 120)]
    assert all(b > a for a, b in zip(eur, eur[1:])), eur
    l3, camp = cvi.cvi_to_projected_eur(110, 'CM', LIGA3), cvi.cvi_to_projected_eur(110, 'CM', CAMPEONATO)
    assert camp == pytest.approx(l3 * cvi.CAMP_PROJECTED_EUR_PENALTY, rel=1e-6)


def test_projected_eur_position_multipliers_apply():
    base = cvi.cvi_to_projected_eur(100, 'CM', LIGA3)
    for pos, mult in cvi.POSITION_EUR_MULTIPLIER.items():
        assert cvi.cvi_to_projected_eur(100, pos, LIGA3) == pytest.approx(
            base * mult / cvi.POSITION_EUR_MULTIPLIER['CM'], rel=1e-6)
