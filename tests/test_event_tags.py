"""event_tags.TagIndex must match the per-row lambdas it replaced."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from event_tags import TagIndex, has_tag, tag_flags  # noqa: E402


def _lambda(sec, tag):
    """The original app.py / pitch_visualizations membership test."""
    return sec.apply(lambda x: isinstance(x, (list, np.ndarray, set, tuple)) and tag in x)


CELLS = [
    np.array(['recovery', 'loss'], dtype=object),
    ['progressive_pass'],
    ('recovery',),
    {'defensive_duel', 'aerial_duel'},
    np.array([], dtype=object),
    [],
    None,
    float('nan'),
    np.array(['loss'], dtype=object),
]
TAGS = ['recovery', 'loss', 'progressive_pass', 'defensive_duel', 'aerial_duel', 'absent_tag']


def test_matches_lambda_on_mixed_cells():
    sec = pd.Series(CELLS, dtype='object')
    for tag in TAGS:
        pd.testing.assert_series_equal(has_tag(sec, tag), _lambda(sec, tag), check_names=False)


def test_keeps_caller_index_even_when_duplicated():
    idx = pd.Index([10, 10, 7, 3, 3, 3, 1, 0, 0])
    sec = pd.Series(CELLS, index=idx, dtype='object')
    out = has_tag(sec, 'loss')
    assert out.index.equals(idx)
    assert out.tolist() == [True, False, False, False, False, False, False, False, True]


def test_flags_share_one_explode():
    sec = pd.Series(CELLS, dtype='object')
    flags = tag_flags(sec, TAGS)
    assert set(flags) == set(TAGS)
    for tag in TAGS:
        pd.testing.assert_series_equal(flags[tag], _lambda(sec, tag), check_names=False)


def test_empty_and_none_columns():
    assert has_tag(pd.Series(dtype='object'), 'loss').empty
    assert has_tag(None, 'loss').empty
    ti = TagIndex(pd.Series([None, None], dtype='object'))
    assert ti.has('loss').tolist() == [False, False]
