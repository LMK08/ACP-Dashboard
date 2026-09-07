"""The strength-of-schedule adjustment must CREDIT tough schedules.

Regression for the 2026-09 inversion: the factors were opponents' conceded ÷
league and opponents' scored ÷ league, so a team that had faced weak
opponents had its attack inflated and its conceded figure shrunk (Caldas
ranked above Atlético CP and Mafra after MW4 2026/27 on that basis).
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

DASHBOARD_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, DASHBOARD_DIR)

from models.strength import sos  # noqa: E402

TEAMS = ['Tough', 'Easy', 'Fresh', 'S1', 'S2', 'S3', 'W1', 'W2', 'W3']


def _team_strength():
    # Every team at 1.0 / 1.0 so both league averages are exactly 1.0.
    return pd.DataFrame(
        {'Attacking Strength': 1.0, 'Defending Strength': 1.0}, index=TEAMS)


def _rolling():
    """Pre-match strengths per (matchId, team).

    'Tough' played S1..S3, each entering with a potent attack (1.5) and a
    stingy defence (0.5). 'Easy' played W1..W3, each entering with a weak
    attack (0.5) and a leaky defence (1.5). 'Fresh' played one opponent whose
    pre-match strength is NaN (no prior match) and one with data — below the
    3-match gate.
    """
    rows = []
    for i, opp_name in enumerate(['S1', 'S2', 'S3']):
        rows += [dict(matchId=10 + i, team='Tough', att_strength=1.0, def_strength=1.0),
                 dict(matchId=10 + i, team=opp_name, att_strength=1.5, def_strength=0.5)]
    for i, opp_name in enumerate(['W1', 'W2', 'W3']):
        rows += [dict(matchId=20 + i, team='Easy', att_strength=1.0, def_strength=1.0),
                 dict(matchId=20 + i, team=opp_name, att_strength=0.5, def_strength=1.5)]
    rows += [dict(matchId=30, team='Fresh', att_strength=np.nan, def_strength=np.nan),
             dict(matchId=30, team='S1', att_strength=np.nan, def_strength=np.nan),
             dict(matchId=31, team='Fresh', att_strength=1.0, def_strength=1.0),
             dict(matchId=31, team='W1', att_strength=0.5, def_strength=1.5)]
    return pd.DataFrame(rows)


@pytest.fixture
def result():
    return sos.sos_adjust(_rolling(), _team_strength())


def test_columns_unchanged(result):
    assert list(result.columns) == sos.RESULT_COLUMNS
    assert set(result.index) == set(TEAMS)


def test_tough_schedule_is_credited(result):
    r = result.loc['Tough']
    assert r['matches_with_opp_data'] == 3
    assert r['sos_att'] > r['raw_att'], 'facing stingy defences must raise the attack rating'
    assert r['sos_def'] < r['raw_def'], 'facing potent attacks must lower the conceded rating'
    assert r['sos_att'] == pytest.approx(1.0 / 0.5)   # league_def / avg_opp_def
    assert r['sos_def'] == pytest.approx(1.0 / 1.5)   # league_att / avg_opp_att
    assert r['sos_factor'] > 1.0


def test_easy_schedule_is_discounted(result):
    r = result.loc['Easy']
    assert r['sos_att'] < r['raw_att'], 'facing leaky defences must lower the attack rating'
    assert r['sos_def'] > r['raw_def'], 'facing weak attacks must raise the conceded rating'
    assert r['sos_att'] == pytest.approx(1.0 / 1.5)
    assert r['sos_def'] == pytest.approx(1.0 / 0.5)
    assert r['sos_factor'] < 1.0


def test_tough_beats_easy_on_net_strength(result):
    net = result['sos_att'] - result['sos_def']
    assert net['Tough'] > net['Easy']
    # The pre-fix formula produced the opposite ordering on this fixture.
    old_att = result['raw_att'] * result['avg_opp_def']
    old_def = result['raw_def'] * result['avg_opp_att']
    assert (old_att - old_def)['Tough'] < (old_att - old_def)['Easy']


def test_below_gate_falls_back_to_raw(result):
    r = result.loc['Fresh']
    assert r['matches_with_opp_data'] == 1          # the NaN opponent is skipped
    assert r['sos_att'] == r['raw_att'] and r['sos_def'] == r['raw_def']
    assert np.isnan(r['sos_factor'])


def test_factors_are_bounded():
    """Three opponents that entered with near-zero conceded must not blow the attack up."""
    rolling = _rolling()
    extreme = rolling['team'].isin(['S1', 'S2', 'S3']) & rolling['matchId'].between(10, 12)
    rolling.loc[extreme, 'def_strength'] = 0.05      # raw factor would be 20x
    rolling.loc[extreme, 'att_strength'] = 10.0      # raw def factor would be 0.1x
    r = sos.sos_adjust(rolling, _team_strength()).loc['Tough']
    lo, hi = sos.FACTOR_BOUNDS
    assert r['sos_att_factor'] == pytest.approx(hi) and r['sos_def_factor'] == pytest.approx(lo)
    assert r['sos_att'] == pytest.approx(hi) and r['sos_def'] == pytest.approx(lo)


def test_gate_boundary_two_matches_is_raw():
    rolling = _rolling()
    rolling = rolling[rolling['matchId'] != 12]          # Tough keeps only 2 valid opponents
    r = sos.sos_adjust(rolling, _team_strength()).loc['Tough']
    assert r['matches_with_opp_data'] == 2
    assert r['sos_att'] == r['raw_att'] and r['sos_def'] == r['raw_def']
    assert np.isnan(r['sos_factor'])


def test_empty_inputs():
    assert sos.sos_adjust(pd.DataFrame(), _team_strength()).empty
    assert sos.sos_adjust(_rolling(), pd.DataFrame()).empty
