"""Minutes come only from Wyscout's per-match minutesOnField (Lucas
2026-10-20: never estimate). Pure-pandas checks of the alias collapse, the
position split and the season build — no API, no data files."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import precompute_minutes as pm  # noqa: E402
from match_minutes import _positions_str  # noqa: E402


def _mm(rows):
    return pd.DataFrame(rows, columns=['matchId', 'seasonId', 'competitionId',
                                       'playerId', 'minutes', 'started', 'positions'])


def test_positions_str():
    assert _positions_str([{'position': {'code': 'cf'}, 'percent': 83},
                           {'position': {'code': 'lw'}, 'percent': 16}]) == 'cf:83,lw:16'
    assert _positions_str([]) == ''
    assert _positions_str(None) == ''


def test_alias_same_match_is_one_appearance(monkeypatch):
    monkeypatch.setattr(pm, 'PLAYER_ID_ALIASES', {2: 1})
    mm = _mm([
        (10, 5, 1, 1, 90, True, 'rcb:100'),   # canonical, match 10
        (10, 5, 1, 2, 90, True, 'rcb:100'),   # duplicate record, SAME match
        (11, 5, 1, 2, 30, False, 'rcb:100'),  # duplicate record, other match
    ])
    out = pm.resolve_aliases(mm)
    assert set(out['playerId']) == {1}
    assert out['minutes'].sum() == 120  # 90 (not 180) + 30


def test_position_minutes_split():
    mm = _mm([(10, 5, 1, 7, 100, True, 'cf:80,lw:20')])
    pos = pm._position_minutes(mm).set_index('position')['minutes']
    assert pos['CF'] == 80 and pos['LW'] == 20


def test_build_season_uses_api_minutes_only(monkeypatch):
    monkeypatch.setattr(pm, '_event_identity', lambda sid: {7: ('Ev Name', 'Ev Team', 'CF')})
    mm = _mm([(10, 5, 1, 7, 67, False, 'rcb:100'),
              (11, 5, 1, 7, 97, True, 'rcb:100')])
    api = pd.DataFrame([{'playerId': 7, 'playerName': 'API Name', 'teamName': 'API Team',
                         'primaryPosition': 'RCB', 'totalMinutes': 999}])
    df = pm.build_season(5, mm, api, details={}, lineup_teams={})
    row = df.iloc[0]
    # minutes = the per-match API sum; labels come from the season's events
    # first (they follow mid-season transfers), the API season row second
    assert row['totalMinutes'] == 164
    assert row['playerName'] == 'Ev Name' and row['teamName'] == 'Ev Team'


def test_build_season_no_minutes_no_row(monkeypatch):
    monkeypatch.setattr(pm, '_event_identity', lambda sid: {8: ('Only Events', 'X', 'CF')})
    df = pm.build_season(5, _mm([]), None, details={}, lineup_teams={})
    assert df.empty  # a player with events but no API minutes is missing, not guessed
