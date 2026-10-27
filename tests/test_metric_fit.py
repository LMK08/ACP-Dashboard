"""theme.metric_fit_key / metric_fit_css — the metric-card text fit behind
the profile's Player Information rows and the comparison header.

Widths below were measured in the browser (canvas, the app's bold Source
Sans, 2026-10); the estimate must never run narrower than the real word, or
the word breaks mid-letter in a card sized to it.
"""
import re

import pytest

import theme

MEASURED_EM = {
    'Conventional': 5.91, 'Conservative': 5.78, 'Midfielder': 4.52,
    'Oliveirense': 4.99, 'Netherlands': 5.44, 'Switzerland': 5.32,
    'Mozambique': 5.64, 'Atlético': 3.39, 'Académica': 4.77,
    'Sanjoanense': 5.63, '€179,204': 4.00, '€1,250,000': 4.82,
}


def _tenths(key):
    return int(re.match(r'mfit(\d+)_', key).group(1))


@pytest.mark.parametrize('word,em', sorted(MEASURED_EM.items()))
def test_estimate_is_never_narrower_than_the_word(word, em):
    est = theme._word_em(word)
    assert em <= est <= em * 1.2
    assert _tenths(theme.metric_fit_key('x', word)) / 10 >= em


def test_the_longest_word_of_all_values_sets_the_bucket():
    short = theme.metric_fit_key('row', 'RB', '23', 'Left')
    long_ = theme.metric_fit_key('row', 'RB', '23', 'Conventional Central Defender')
    assert _tenths(short) < _tenths(long_)
    assert long_.endswith('_row')


def test_a_line_breaks_after_a_hyphen():
    assert theme.metric_fit_key('x', 'Guinea-Bissau') == theme.metric_fit_key('x', 'Guinea-')
    assert theme.metric_fit_key('x', None, float('nan'), '').startswith('mfit')


def test_one_css_rule_per_bucket_and_every_key_has_one():
    css = theme.metric_fit_css()
    for t in theme.METRIC_FIT_TENTHS:
        assert f'st-key-mfit{t}_' in css
    huge = theme.metric_fit_key('x', 'Wwwwwwwwwwwwwwwwwwww')
    assert _tenths(huge) == theme.METRIC_FIT_TENTHS[-1]
    assert len(css.splitlines()) == len(theme.METRIC_FIT_TENTHS)
