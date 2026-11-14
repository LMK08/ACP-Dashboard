"""theme.py — one meaning per colour (2026-11): the parts a test can hold."""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import theme  # noqa: E402


def _rgb(hex_colour):
    h = hex_colour.lstrip('#')
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _source(path):
    return open(os.path.join(ROOT, path), encoding='utf-8').read()


def test_good_is_the_subject_green_and_bad_is_the_opponent_red():
    assert theme.GOOD == theme.FOCUS and theme.BAD == theme.OPPONENT
    r, g, b = _rgb(theme.GOOD)
    assert g > r and g > b                       # green
    r, g, b = _rgb(theme.BAD)
    assert r > g and r > b                       # red


def test_the_pairs_hold_no_red_and_no_green():
    """Home / away, the two leagues and the two ends of the value scale are
    blue and orange: neither side of a pair reads as good or bad."""
    pair = {theme.HOME_BLUE, theme.AWAY, theme.VALUE_NEG, theme.VALUE_POS, *theme.LEAGUE_COLORS.values()}
    for colour in pair:
        r, g, b = _rgb(colour)
        is_blue = b > r and b > g
        is_orange = r > g > b and g >= 0.3 * r
        assert is_blue or is_orange, colour
    assert theme.AWAY != theme.HOME_BLUE and theme.VALUE_POS != theme.VALUE_NEG
    assert theme.VALUE_COLORSCALE[0][1] == theme.VALUE_NEG and theme.VALUE_COLORSCALE[-1][1] == theme.VALUE_POS
    assert not hasattr(theme, 'AWAY_RED')        # the away side is never red: it may be us


def test_rgba_helper():
    assert theme.rgba('#1a472a', 0.18) == 'rgba(26,71,42,0.18)'
    r, g, b = _rgb(theme.BAD)
    assert theme.rgba(theme.BAD, 0.2) == f'rgba({r},{g},{b},0.2)'


def test_the_charts_read_these_colours_from_the_theme():
    """The clashes this pass removed stay removed: a good outcome typed in
    red, the other teams typed in green, the value scale typed apart from
    the theme."""
    assert 'theme.AWAY_RED' not in ''.join(
        _source(p) for p in ('app.py', 'team_interactive.py', 'scoreline_ui.py', 'eur_interval_ui.py',
                             'obv_viz.py', 'pitch_interactive.py', 'pitch_visualizations.py'))
    assert "'#3a8a3a'" not in _source('app.py')                       # the season report's other teams
    assert 'VAL_COLORSCALE = theme.VALUE_COLORSCALE' in _source('pitch_interactive.py')
    rolling = re.search(r"def plotly_rolling_xg.*?(?=\ndef |\Z)", _source('team_interactive.py'), re.S).group(0)
    assert 'theme.FOCUS' in rolling and 'theme.OPPONENT' in rolling and 'HOME_BLUE' not in rolling
    # the figure the team pages draw (shot assists beside final-third dribbles), and its two unused relatives
    for name in ('plot_shot_assists_and_dribbles', 'plot_through_passes', 'plot_final_third_dribbles'):
        body = re.search(rf"def {name}.*?(?=\ndef |\Z)", _source('pitch_visualizations.py'), re.S).group(0)
        assert 'theme.GOOD' in body and "'#e63946'" not in body and "'#2a9d8f'" not in body, name
    assert 'theme.BAD' in re.search(r"def plot_shot_assists_and_dribbles.*?(?=\ndef |\Z)",
                                    _source('pitch_visualizations.py'), re.S).group(0)


def test_the_radar_colours_are_untouched():
    """Radar colours name a kind of metric and are outside the rule; they
    change only after Lucas has seen the change (radar rule)."""
    assert (theme.RADAR_OFFENSIVE, theme.RADAR_DISTRIBUTION, theme.RADAR_DEFENSIVE, theme.RADAR_SET_PIECE) == (
        '#e60000', '#0077b6', '#52A736', '#ff8c00')
    assert theme.XG_COLORS == ['#03045e', '#ade8f4', '#fff3b0', '#ff8c00', '#e63946', '#800f2f']
