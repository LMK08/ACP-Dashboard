"""models/ratings/duel_expectations — a contest is judged only against
contests from matches that kicked off before it (v7.1, 2026-10-29): later
contests, its own outcome and its own match never enter its expectation.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

DASH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, DASH)

from models.ratings import duel_expectations as DE  # noqa: E402

S24, S25 = 190090, 191782
YR = {S24: 2024, S25: 2025}


def _ct(rows):
    """rows: (ladder, seasonId, 'YYYY-MM-DD HH:MM', playerA, playerB, scoreA)"""
    return pd.DataFrame([{'ladder': l, 'matchId': i, 'seasonId': s, 'date': pd.Timestamp(d),
                          'playerA': a, 'playerB': b, 'scoreA': w,
                          'att_kind': 'dribble', 'zx': 1, 'phase': 'open'}
                         for i, (l, s, d, a, b, w) in enumerate(rows)])


BASE = [('aerial', S24, '2024-09-01 15:00', 1, 2, 1.0),
        ('ground', S24, '2024-09-01 15:00', 3, 4, 1.0),
        ('aerial', S24, '2024-10-01 15:00', 1, 5, 0.0),
        ('ground', S24, '2024-10-01 15:00', 3, 6, 0.0),
        ('aerial', S25, '2025-09-01 15:00', 2, 5, 1.0),
        ('ground', S25, '2025-09-01 15:00', 4, 3, 1.0)]
LATER = [('aerial', S25, '2026-01-01 15:00', 1, 2, 0.0),
         ('ground', S25, '2026-01-01 15:00', 3, 4, 0.0),
         ('ground', S24, '2026-01-01 15:00', 6, 3, 1.0)]


def _exp(rows):
    a, g = DE.expectations(_ct(rows), set(), YR)
    return a[['pA', 'pB']].to_numpy(), g[['mu_b', 'phat', 'pdef']].to_numpy()


def test_later_contests_do_not_move_earlier_expectations():
    a0, g0 = _exp(BASE)
    a1, g1 = _exp(BASE + LATER)
    np.testing.assert_array_equal(a0, a1[:len(a0)])
    np.testing.assert_array_equal(g0, g1[:len(g0)])


def test_a_contest_never_sees_its_own_outcome_or_its_own_match():
    a0, g0 = _exp(BASE)
    flipped = [r if i not in (0, 1) else r[:5] + (1 - r[5],) for i, r in enumerate(BASE)]
    a1, g1 = _exp(flipped)
    # the first match (contests 0 and 1) keeps its expectations; later ones move
    np.testing.assert_array_equal(a0[0], a1[0])
    np.testing.assert_array_equal(g0[0], g1[0])
    assert not np.array_equal(a0[1], a1[1])


def test_no_contest_sees_its_own_outcome():
    """Flip one contest at a time: its own expectations never move."""
    rows = BASE + LATER
    a0, g0 = _exp(rows)
    ia = [i for i, r in enumerate(rows) if r[0] == 'aerial']
    ig = [i for i, r in enumerate(rows) if r[0] == 'ground']
    for i in range(len(rows)):
        flipped = [r if j != i else r[:5] + (1 - r[5],) for j, r in enumerate(rows)]
        a1, g1 = _exp(flipped)
        if i in ia:
            np.testing.assert_array_equal(a0[ia.index(i)], a1[ia.index(i)])
        else:
            np.testing.assert_array_equal(g0[ig.index(i)], g1[ig.index(i)])


def test_centring_uses_only_earlier_kickoffs():
    """A ground contest's centring never sees other contests of its own
    kick-off: adding one to the same match leaves the first unchanged."""
    one = [('ground', S24, '2024-09-01 15:00', 3, 4, 1.0),
           ('ground', S24, '2024-10-01 15:00', 3, 4, 0.0)]
    two = one + [('ground', S24, '2024-10-01 15:00', 5, 4, 1.0)]
    _, g1 = _exp(one)
    _, g2 = _exp(two)
    np.testing.assert_array_equal(g1[1], g2[1])


def test_first_contest_is_judged_at_the_neutral_prior():
    a, g = _exp(BASE)
    assert a[0] == pytest.approx([0.5, 0.5])
    assert g[0][0] == pytest.approx(0.5)      # no earlier contest: bucket rate = 0.5
    assert g[0][1] == pytest.approx(0.5)      # centred strengths: phat = mu_b


def test_strength_counts_the_whole_career_before_the_match():
    a, _ = _exp(BASE)       # aerial rows: 2024-09 1 beats 2; 2024-10 5 beats 1; 2025-09 2 beats 5
    # 2024-10, 1 vs 5: 1 has won 1 of 1 earlier aerials, 5 has none
    assert a[1] == pytest.approx([0.5, 1 - 21 / 41])
    # 2025-09, 2 vs 5: last season's contests still count (career, not season)
    assert a[2] == pytest.approx([1 - 21 / 41, 1 - 20 / 41])


def test_contests_in_the_same_match_do_not_judge_each_other():
    a, _ = _exp([('aerial', S24, '2024-09-01 15:00', 1, 2, 1.0),
                 ('aerial', S24, '2024-09-01 15:00', 1, 2, 1.0)])
    assert a[1] == pytest.approx([0.5, 0.5])


@pytest.mark.skipif(not os.path.exists(os.path.join(DASH, 'models/duels/contests.parquet')),
                    reason='contests.parquet not present')
def test_real_contests_truncated_at_a_date_keep_every_earlier_expectation():
    from league_config import SEASON_YEAR, CAMP_SEASON_IDS
    ct = pd.read_parquet(os.path.join(DASH, 'models/duels/contests.parquet'),
                         columns=['ladder', 'seasonId', 'date', 'playerA', 'playerB',
                                  'scoreA', 'att_kind', 'zx', 'phase'])
    cut = pd.Timestamp('2025-01-01')
    a, g = DE.expectations(ct, CAMP_SEASON_IDS, SEASON_YEAR)
    a2, g2 = DE.expectations(ct[ct['date'] < cut], CAMP_SEASON_IDS, SEASON_YEAR)
    keep_a, keep_g = (a['date'] < cut).to_numpy(), (g['date'] < cut).to_numpy()
    np.testing.assert_array_equal(a.loc[keep_a, ['pA', 'pB']].to_numpy(), a2[['pA', 'pB']].to_numpy())
    np.testing.assert_array_equal(g.loc[keep_g, ['phat', 'pdef']].to_numpy(), g2[['phat', 'pdef']].to_numpy())
