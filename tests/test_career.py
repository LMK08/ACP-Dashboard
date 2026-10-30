"""models/ratings/career — a player's career as one continuous sample for
prediction: season blocks at their minute-weighted match date, decayed by
time before the as-of time; no look-ahead. Plus the career playing-time share
built on it (models/value/availability.career_playing_time).
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

DASH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, DASH)

from models.ratings import career as C  # noqa: E402
from models.value import availability as av  # noqa: E402

S24, S25 = 190090, 191782       # Liga 3 2024/25, 2025/26
YR = {S24: 2024, S25: 2025}


def _ms(rows):
    """rows: (matchId, seasonId, 'YYYY-MM-DD')"""
    return pd.DataFrame([{'matchId': m, 'seasonId': s, 'dateutc': f'{d} 15:00:00', 'status': 'Played'}
                         for m, s, d in rows])


def test_decay_constants_are_the_old_per_season_rates():
    assert 0.5 ** (1 / C.H_FORM) == pytest.approx(0.5)
    assert 0.5 ** (1 / C.H_EVID) == pytest.approx(0.9)


def test_block_sits_at_its_minute_weighted_date_and_ignores_later_matches():
    ms = _ms([(1, S25, '2025-09-01'), (2, S25, '2025-10-01'), (3, S25, '2025-11-01')])
    mm = pd.DataFrame({'matchId': [1, 2, 3], 'playerId': [7, 7, 7], 'minutes': [90, 30, 90]})
    cen, mid = C.block_centroids(mm, ms)
    t = C.to_years(pd.Series(pd.to_datetime(['2025-09-01', '2025-10-01', '2025-11-01'])))
    assert cen[(7, S25)] == pytest.approx((90 * t[0] + 30 * t[1] + 90 * t[2]) / 210)
    assert mid[S25] == pytest.approx(t[1])
    # as of 15 Oct: the November match does not exist yet
    cen_oct, _ = C.block_centroids(mm, ms, upto=C.to_years(pd.Timestamp('2025-10-15')))
    assert cen_oct[(7, S25)] == pytest.approx((90 * t[0] + 30 * t[1]) / 120)


def test_career_rating_and_evidence_decay_by_time_not_by_season_count():
    t24, t25 = C.to_years(pd.Timestamp('2025-03-01')), C.to_years(pd.Timestamp('2026-03-01'))
    cen = pd.Series({(7, S24): t24, (7, S25): t25})
    rows = pd.DataFrame({'playerId': [7, 7], 'seasonId': [S24, S25], 'yr': [2024, 2025],
                         'mins_played': [2000.0, 1000.0], 'acp_rating': [60.0, 40.0]})
    out = C.career_features(rows, cen, pd.Series(dtype=float), pd.Series([t24, t25], index=rows.index))
    gap = t25 - t24
    wf = 2000 * 0.5 ** (gap / C.H_FORM)
    assert out.loc[1, 'career_asof'] == pytest.approx((wf * 60 + 1000 * 40) / (wf + 1000))
    assert out.loc[1, 'evidence'] == pytest.approx(2000 * 0.5 ** (gap / C.H_EVID) + 1000)
    # the earlier row sees only its own block (no look-ahead)
    assert (out.loc[0, 'career_asof'], out.loc[0, 'evidence']) == (pytest.approx(60.0), pytest.approx(2000.0))


def test_a_season_without_per_match_minutes_sits_at_its_median_date():
    rows = pd.DataFrame({'playerId': [7], 'seasonId': [S24], 'yr': [2024],
                         'mins_played': [1500.0], 'acp_rating': [50.0]})
    mid = pd.Series({S24: C.to_years(pd.Timestamp('2025-01-15'))})
    t = C.to_years(pd.Timestamp('2026-01-15'))
    out = C.career_features(rows, pd.Series(dtype=float), mid, pd.Series([t], index=rows.index))
    assert out.loc[0, 'evidence'] == pytest.approx(1500 * 0.5 ** ((t - mid[S24]) / C.H_EVID))


def test_career_share_carries_a_regular_through_a_slow_start():
    ms = _ms([(1, S24, '2025-03-01'), (2, S25, '2026-03-01')])
    mm = pd.DataFrame({'matchId': [1, 2], 'playerId': [7, 7], 'minutes': [90, 90]})
    rows = pd.DataFrame({'playerId': [7, 7], 'yr': [2024, 2025], 'mins_year': [2000.0, 100.0],
                         'avail_mins': [2850.0, 1000.0], 'starts': [22, 1], 'sub_apps': [2, 3]})
    share, sub = av.career_playing_time(rows, {'mins': mm, 'ms': ms})
    w = 0.5 ** ((C.to_years(pd.Timestamp('2026-03-01')) - C.to_years(pd.Timestamp('2025-03-01'))) / av.H_PLAYING)
    assert share[1] == pytest.approx((w * 2000 + 100) / (w * 2850 + 1000))
    assert share[1] > 100 / 1000                       # far above his season share
    assert sub[1] == pytest.approx((w * 2 + 3) / (w * 24 + 4))
    assert share[0] == pytest.approx(2000 / 2850)      # no look-ahead into 25/26
    assert np.isfinite(sub[0])
