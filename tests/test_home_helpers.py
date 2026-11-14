"""views/home.py — the pure helpers behind Home's cards (no app boot)."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from views import home  # noqa: E402

GROUP = ['Us', 'Alfa', 'Beta', 'Gama']


def test_remaining_legs_are_the_double_round_robin_minus_what_was_played():
    assert home.remaining_legs('Us', GROUP, []) == [
        ('Alfa', 'H'), ('Alfa', 'A'), ('Beta', 'H'), ('Beta', 'A'), ('Gama', 'H'), ('Gama', 'A')]
    # (home, away): we hosted Alfa and went to Beta; other teams' matches do not count
    played = [('Us', 'Alfa'), ('Beta', 'Us'), ('Alfa', 'Gama')]
    assert home.remaining_legs('Us', GROUP, played) == [
        ('Alfa', 'A'), ('Beta', 'H'), ('Gama', 'H'), ('Gama', 'A')]
    everything = [('Us', t) for t in GROUP[1:]] + [(t, 'Us') for t in GROUP[1:]]
    assert home.remaining_legs('Us', GROUP, everything) == []
    assert home.remaining_legs('Us', [], []) == []


def _matches(rows):
    return pd.DataFrame(rows, columns=['roundId', 'homeTeamName', 'awayTeamName'])


def test_phase_legs_are_the_first_phase_until_a_second_phase_match_exists():
    first = [(1, 'Us', 'Alfa'), (1, 'Beta', 'Us'), (1, 'Alfa', 'Gama'), (1, 'Gama', 'Beta')]
    legs, second = home.phase_legs_played(_matches(first), 'Us')
    assert (legs, second) == ([('Us', 'Alfa'), ('Beta', 'Us')], False)
    # one second-phase match anywhere in the league: the phase has turned, and
    # our first-phase meetings are no longer legs of "this phase"
    turned = first + [(2, 'Alfa', 'Gama')]
    assert home.phase_legs_played(_matches(turned), 'Us') == ([], True)
    # ...so a club that is in our second-phase group too is listed home AND away
    assert home.remaining_legs('Us', ['Us', 'Alfa', 'Delta'], []) == [
        ('Alfa', 'H'), ('Alfa', 'A'), ('Delta', 'H'), ('Delta', 'A')]
    ours_too = turned + [(2, 'Us', 'Delta')]
    assert home.phase_legs_played(_matches(ours_too), 'Us') == ([('Us', 'Delta')], True)
    assert home.phase_legs_played(_matches([]), 'Us') == ([], False)


def test_table_window_keeps_the_team_in_view_and_clips_at_the_ends():
    table = pd.DataFrame({'Pos': range(1, 11), 'Team': [f'T{i}' for i in range(1, 11)]})
    assert list(home.table_window(table, 'T5')['Pos']) == [3, 4, 5, 6, 7]       # centred
    assert list(home.table_window(table, 'T1')['Pos']) == [1, 2, 3, 4, 5]       # top of the table
    assert list(home.table_window(table, 'T2')['Pos']) == [1, 2, 3, 4, 5]
    assert list(home.table_window(table, 'T10')['Pos']) == [6, 7, 8, 9, 10]     # bottom
    assert len(home.table_window(table.head(4), 'T2')) == 4                     # a short table: all of it
    assert len(home.table_window(table, 'Nobody')) == 10                        # team not in it: all of it


def test_form_tile_shows_one_chip_per_result_with_its_letter_and_match():
    tile = home.form_tile_html([('W', 'W 1–0 vs Alfa (A), GW 9'), ('D', 'D 2–2 vs <Beta> (H), GW 8'), ('L', 'L 0–1')])
    body = tile.split('</style>')[-1]
    assert body.count('<span') == 3
    assert '>W</span>' in body and '>D</span>' in body and '>L</span>' in body
    assert home._FORM_CHIP['W'] in body and home._FORM_CHIP['L'] in body
    assert 'vs &lt;Beta&gt;' in body and '<Beta>' not in body                   # tooltips are escaped
    assert 'Form (last 5)' in body and 'Oldest to newest' in body
    # no result yet: a dash, not an empty tile
    assert '—' in home.form_tile_html([]).split('</style>')[-1]
