"""pitch_interactive.projection_series / plotly_projection_overlay — the
career & projection chart of the Player Comparison page (the profile's fan,
two players on one chart).

Synthetic engine rows; no app boot.
"""
import numpy as np
import pandas as pd
import pytest

import pitch_interactive as pi

LABELS = {1: '2023/24', 2: '2024/25', 3: '2025/26', 11: '2024/25', 12: '2025/26'}


def _rows(pid, seasons, leagues, ratings, abs_ratings=None, proj=None, age=24.0):
    n = len(seasons)
    return pd.DataFrame({
        'playerId': pid, 'seasonId': seasons, 'league': leagues,
        'acp_rating': ratings,
        'acp_rating_abs': abs_ratings if abs_ratings is not None else ratings,
        'mins_played': [1500.0] * n,
        'projection': [np.nan] * (n - 1) + [proj],
        'projection_abs': [np.nan] * (n - 1) + [(proj - 6) if proj is not None else np.nan],
        'band_sd': [np.nan] * (n - 1) + [10.0],
        'w_evidence': [np.nan] * (n - 1) + [0.6],
        'role': 'Wide Defender', 'age': age,
    })


def _marker_traces(fig):
    return [t for t in fig.data if t.mode == 'markers' and t.name]


def test_projection_series_decides_the_scale_per_player_unless_told():
    one_league = _rows(1, [1, 2, 3], ['L3'] * 3, [50.0, 55.0, 60.0], proj=62.0)
    ser = pi.projection_series(one_league, LABELS)
    assert ser['use_abs'] is False and ser['proj'] == 62.0
    forced = pi.projection_series(one_league, LABELS, use_abs=True)
    assert forced['use_abs'] is True and forced['proj'] == 56.0   # projection_abs


def test_overlay_puts_both_players_on_one_scale_when_leagues_differ():
    a = _rows(1, [1, 2, 3], ['L3'] * 3, [50.0, 55.0, 60.0], proj=62.0)
    b = _rows(2, [11, 12], ['CAMP'] * 2, [70.0, 72.0], abs_ratings=[58.0, 60.0], proj=74.0)
    fig, left = pi.plotly_projection_overlay(
        [('A', a, '#0077b6'), ('B', b, '#A67B5B')], LABELS)
    assert left == []
    assert 'L3-equivalent' in fig.layout.yaxis.title.text
    by_name = {t.name: list(t.y) for t in _marker_traces(fig)}
    assert by_name['B'] == [58.0, 60.0]        # CAMP plotted on the abs scale
    assert by_name['A'] == [50.0, 55.0, 60.0]


def test_overlay_keeps_native_scale_within_one_league():
    a = _rows(1, [1, 2, 3], ['L3'] * 3, [50.0, 55.0, 60.0], proj=62.0)
    b = _rows(2, [2, 3], ['L3'] * 2, [40.0, 45.0], proj=47.0)
    fig, _ = pi.plotly_projection_overlay([('A', a, '#0077b6'), ('B', b, '#A67B5B')], LABELS)
    assert fig.layout.yaxis.title.text == 'ACP rating'


def test_overlay_names_who_it_cannot_draw():
    a = _rows(1, [1, 2, 3], ['L3'] * 3, [50.0, 55.0, 60.0], proj=62.0)
    no_age = _rows(2, [2, 3], ['L3'] * 2, [40.0, 45.0], proj=47.0, age=np.nan)
    empty = a.iloc[0:0]
    fig, left = pi.plotly_projection_overlay(
        [('A', a, '#0077b6'), ('NoAge', no_age, '#A67B5B'), ('Keeper', empty, '#000000')],
        LABELS, x_axis='age')
    assert left == ['NoAge (age unknown)', 'Keeper (no rated season)']
    assert [t.name for t in _marker_traces(fig)] == ['A']
    fig, left = pi.plotly_projection_overlay([('Keeper', empty, '#000000')], LABELS)
    assert fig is None and left == ['Keeper (no rated season)']


def test_age_axis_places_seasons_by_age_and_projection_a_year_on():
    a = _rows(1, [1, 2, 3], ['L3'] * 3, [50.0, 55.0, 60.0], proj=62.0, age=24.0)
    fig, _ = pi.plotly_projection_overlay([('A', a, '#0077b6')], LABELS, x_axis='age')
    assert list(_marker_traces(fig)[0].x) == pytest.approx([22.0, 23.0, 24.0])
    diamond = [t for t in fig.data if t.mode == 'markers+text'][0]
    assert list(diamond.x) == pytest.approx([25.0]) and list(diamond.y) == [62.0]


def test_profile_fan_still_needs_a_projection():
    no_proj = _rows(1, [1, 2], ['L3'] * 2, [50.0, 55.0], proj=None)
    no_proj['projection'] = np.nan
    assert pi.plotly_projection_fan(no_proj, LABELS, 'X') is None
    fig, _ = pi.plotly_projection_overlay([('A', no_proj, '#0077b6')], LABELS)
    assert fig is not None          # the overlay still draws the career line
