"""DefR within-season look-ahead (v7.1, 2026-10-29): an opposition event is
judged with response tables from EARLIER seasons only, and a player is placed
at the slot he played in THAT match — never with later matches' positions.
"""
import os
import sys

import pandas as pd
import pytest

DASH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, DASH)

from models.defr import compute_defr as CD  # noqa: E402

DATES = {1: pd.Timestamp('2024-09-01'), 2: pd.Timestamp('2024-10-01'), 3: pd.Timestamp('2024-11-01')}


def _ev(rows):
    """rows: (player, match, position code)"""
    return pd.DataFrame(rows, columns=['player.id', 'matchId', 'player.position'])


def test_a_match_slot_is_that_matchs_position():
    rows = [(7, 1, 'LB'), (7, 1, 'LB'), (7, 1, 'LCB'), (7, 2, 'RW'), (7, 2, 'RW')]
    msl = CD.compute_match_slots(_ev(rows), DATES)
    assert msl[(7, 1)] == CD.slot_of('LB') and msl[(7, 2)] == CD.slot_of('RW')


def test_later_matches_never_move_an_earlier_slot():
    base = [(7, 1, 'LB'), (7, 1, 'LB'), (7, 2, None)]
    later = base + [(7, 3, 'RW'), (7, 3, 'RW'), (7, 3, 'RW'), (7, 1, None)]
    a = CD.compute_match_slots(_ev(base), DATES)
    b = CD.compute_match_slots(_ev(later), DATES)
    assert a[(7, 1)] == b[(7, 1)] and a[(7, 2)] == b[(7, 2)]
    # match 2 has no coded position: the latest EARLIER match's slot, not match 3's
    assert b[(7, 2)] == CD.slot_of('LB')


def test_response_tables_come_from_earlier_seasons_only(monkeypatch):
    seen = {}

    def spy(r_y, fine, coarse):
        seen.setdefault('years', []).append(sorted(r_y['_yr'].unique()))
        return pd.DataFrame({'n': [len(r_y)]})
    monkeypatch.setattr(CD, 'build_prob_table', spy)
    monkeypatch.setattr(CD, 'build_prob_table_by_type', lambda r_y, f, c: {})
    resp = pd.DataFrame({'_yr': [2021, 2021, 2022, 2023, 2023]})
    pres = {188221: ({('a',): 1}, {}), 188222: ({('a',): 2}, {}), 189147: ({('a',): 4}, {})}
    t = CD.asof_tables(resp, pres, [2021, 2022, 2023])
    assert t[2021] is None                                   # nothing earlier
    assert seen['years'] == [[2021], [2021, 2022]]           # 2022 <- 2021; 2023 <- 2021-22
    assert int(t[2023][0]['n'].iloc[0]) == 3


def test_dwae_rates_never_see_their_own_year():
    """A year's expected win rates come from earlier years: flipping every
    outcome of 2023 leaves 2023's p_win unchanged; the first year has none."""
    import numpy as np
    rng = np.random.default_rng(0)
    n = 300
    eng = pd.DataFrame({'etype': rng.choice(['tackle', 'aerial'], n), 'oppgrp': rng.choice(['ST', 'W'], n),
                        'zx': rng.integers(0, 3, n), 'phase': rng.choice(['open', 'set'], n),
                        'won': rng.integers(0, 2, n), '_yr': np.repeat([2021, 2022, 2023], n // 3)})
    p = CD.dwae_win_rates(eng)
    flip = eng.copy()
    flip.loc[flip['_yr'] == 2023, 'won'] = 1 - flip.loc[flip['_yr'] == 2023, 'won']
    q = CD.dwae_win_rates(flip)
    assert p[eng['_yr'] == 2021].isna().all()
    assert np.array_equal(p[eng['_yr'] <= 2023].fillna(-1), q[eng['_yr'] <= 2023].fillna(-1))
    flip22 = eng.copy()
    flip22.loc[flip22['_yr'] == 2022, 'won'] = 1 - flip22.loc[flip22['_yr'] == 2022, 'won']
    assert not np.array_equal(p[eng['_yr'] == 2023], CD.dwae_win_rates(flip22)[eng['_yr'] == 2023])
