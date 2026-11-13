"""search.build_index — the list behind the sidebar search box (pure pandas).

No app boot: tiny frames in the shapes app.py holds. Season ids are the real
ones from league_config (the index reads league and season labels from it).
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import search  # noqa: E402

L3_NEW, L3_OLD, CAMP = 192831, 191782, 191779      # Liga 3 2026/27, Liga 3 2025/26, Campeonato 2025/26


def _minutes(rows):
    return pd.DataFrame(rows, columns=['playerId', 'playerName', 'teamName', 'primaryPosition', 'totalMinutes'])


def _index():
    minutes = {
        L3_NEW: _minutes([(1, 'Ana Um', 'Clube A', 'CB', 900),
                          (2, 'Rui Dois', 'Clube B', 'CF', 0)]),         # 0 minutes this season
        L3_OLD: _minutes([(1, 'Ana Um', 'Clube Z', 'CB', 2000),          # older club: not the label
                          (2, 'Rui Dois', 'Clube B', 'CF', 1500)]),
        CAMP: _minutes([(3, 'Carimo Três', 'Clube C', 'GK', 3000),
                        (3, 'Carimo Três', 'Clube D', 'GK', 100),        # two clubs in one season
                        (4, 'Unknown', 'Unknown', 'CM', 500),            # the minutes files' placeholder
                        (5, 'João Costa', 'Clube C', 'CB', 1200),        # two players, one name,
                        (6, 'João Costa', 'Clube C', 'CF', 700)]),       # one club, one season
    }
    matches = pd.DataFrame({
        'seasonId': [L3_NEW, L3_OLD, CAMP],
        'homeTeamName': ['Clube A', 'Clube B', 'Clube C'],
        'awayTeamName': ['Clube B', 'Clube Z', 'Clube D'],
    })
    coach_matches = pd.DataFrame({
        'coachId': [10, 10, 11, 0, 12, 13, 14, 14],
        'date': pd.to_datetime(['2025-09-01', '2026-10-01', '2026-01-01', '2026-10-02',
                                '2026-02-01', '2026-03-01', '2026-03-02', '2026-03-09']),
        'team': ['Clube Z', 'Clube A', 'Clube C', 'Clube B', 'Clube D', 'Clube C', 'Clube C', 'Clube C'],
        'competitionId': [43324, 43324, 702, 43324, 702, 702, 702, 702],
    })
    coaches = pd.DataFrame({
        'coachId': [10, 11, 12, 13, 14],
        'name': ['Z. Treinador', 'A. Mister', 'Coach #12', 'A. Mister', 'A. Mister'],   # 12: unnamed
    }).set_index('coachId')
    return search.build_index(minutes, matches, (coach_matches, coaches))


def test_one_entry_per_player_in_his_newest_season_with_minutes():
    index = _index()
    ana, rui, carimo = index['p:1'], index['p:2'], index['p:3']
    assert (ana['season_id'], ana['league'], ana['label']) == (L3_NEW, 'Liga 3', 'Ana Um — Clube A, 2026/27')
    # no minutes in the newest season: the season before is his newest WITH minutes
    assert (rui['season_id'], rui['label']) == (L3_OLD, 'Rui Dois — Clube B, 2025/26')
    # the club he played most for in that season; the other league is named
    assert (carimo['season_id'], carimo['league'], carimo['label']) == (
        CAMP, 'Campeonato', 'Carimo Três — Clube C, 2025/26')
    # newest season first, then most minutes
    assert [k for k in index if k.startswith('p:')] == ['p:1', 'p:3', 'p:2', 'p:5', 'p:6']
    assert 'p:4' not in index                                                 # no name, no entry


def test_teams_open_in_their_newest_season_and_league():
    index = _index()
    assert index['t:Clube A'] == {'kind': 'team', 'label': 'Clube A — team, Liga 3', 'team': 'Clube A',
                                  'league': 'Liga 3', 'season': '2026/27'}
    assert index['t:Clube Z']['season'] == '2025/26'                      # only played the older season
    assert (index['t:Clube C']['league'], index['t:Clube C']['season']) == ('Campeonato', '2025/26')


def test_coaches_carry_their_latest_team_and_league_and_unknown_ones_are_left_out():
    index = _index()
    assert index['c:10'] == {'kind': 'coach', 'label': 'Z. Treinador — coach, Clube A', 'coach_id': 10,
                             'league': 'Liga 3', 'detail': '2 matches'}
    assert index['c:11']['league'] == 'Campeonato'
    assert 'c:0' not in index                                             # coachId 0 = unknown
    assert 'c:12' not in index                                            # 'Coach #12' = no name to search


def test_every_label_is_unique():
    """Streamlit maps the picked LABEL back to an option: two entries with one
    label would open the same (last) one. Namesakes get their detail."""
    index = _index()
    labels = [entry['label'] for entry in index.values()]
    assert len(set(labels)) == len(labels), sorted(labels)
    # two players, one name, one club, one season: position and minutes tell them apart
    assert index['p:5']['label'] == 'João Costa — Clube C, 2025/26 (CB, 1,200 min)'
    assert index['p:6']['label'] == 'João Costa — Clube C, 2025/26 (CF, 700 min)'
    # three coaches called A. Mister at Clube C: matches coached, then a counter
    assert index['c:14']['label'] == 'A. Mister — coach, Clube C (2 matches)'
    assert sorted(index[k]['label'] for k in ('c:11', 'c:13')) == [
        'A. Mister — coach, Clube C (1 match) [1]', 'A. Mister — coach, Clube C (1 match) [2]']
    # a label that is already unique is left alone
    assert index['p:1']['label'] == 'Ana Um — Clube A, 2026/27'


def test_teams_come_first_then_coaches_then_players_and_search_works_without_coaches():
    kinds = [entry['kind'] for entry in _index().values()]
    assert kinds == sorted(kinds, key=['team', 'coach', 'player'].index)
    minutes = {L3_NEW: _minutes([(1, 'Ana Um', 'Clube A', 'CB', 900)])}
    matches = pd.DataFrame({'seasonId': [L3_NEW], 'homeTeamName': ['Clube A'], 'awayTeamName': ['Clube B']})
    index = search.build_index(minutes, matches, None)
    assert {e['kind'] for e in index.values()} == {'team', 'player'}
    assert search.build_index({}, matches.iloc[0:0], None) == {}
