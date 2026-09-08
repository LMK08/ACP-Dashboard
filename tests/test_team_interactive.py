"""Pure checks on team_interactive: the Plotly builders and the selection
parsing the click-throughs depend on. No data files, no Streamlit run."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

DASHBOARD_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, DASHBOARD_DIR)

import team_interactive as ti  # noqa: E402


def _event(customdata_rows):
    """The dict st.plotly_chart(on_select='rerun') returns (Streamlit 1.35+)."""
    return {'selection': {'points': [{'curve_number': 0, 'point_index': i, 'customdata': cd}
                                     for i, cd in enumerate(customdata_rows)],
                          'point_indices': list(range(len(customdata_rows))),
                          'box': [], 'lasso': []}}


def test_selected_customdata_reads_dict_and_object_forms():
    ev = _event([['a', 1], ['b', 2]])
    assert ti.selected_customdata(ev) == [['a', 1], ['b', 2]]

    class P:  # attribute form
        def __init__(self, cd): self.customdata = cd
    class S:
        points = [P([7, 'x'])]
    class E:
        selection = S()
    assert ti.selected_customdata(E()) == [[7, 'x']]
    assert ti.selected_customdata({}) == []
    assert ti.selected_customdata(None) == []


def test_match_id_from_shot_and_rolling_rows():
    shot_row = ['Player', '01 Jan 2026', 'Opp', '1-0', '0.12', 'Goal', 5740718, 'Atlético CP']
    rolling_row = ['01 Jan 2026', 'Opp', '1-0', '1.20', '0.80', 5740719]
    assert ti.match_id_from_rows([shot_row]) == 5740718
    assert ti.match_id_from_rows([rolling_row]) == 5740719
    assert ti.match_id_from_rows([np.array(shot_row, dtype=object)]) == 5740718
    assert ti.match_id_from_rows([['01 Jan', 'Opp', '', '1', '1', -1]]) is None  # no match id
    assert ti.match_id_from_rows([]) is None


def test_player_id_from_rows():
    assert ti.player_id_from_rows([[44479, 'Duarte Henriques']]) == 44479
    assert ti.player_id_from_rows([np.array([44479, 'X'], dtype=object)]) == 44479
    assert ti.player_id_from_rows([]) is None


def _events(n=30, team='Us', opp='Them'):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        shooter = team if i % 2 == 0 else opp
        rows.append({'matchId': 100 + i % 3, 'team.name': shooter, 'player.name': f'P{i}',
                     'type.primary': 'shot', 'location.x': 70 + rng.random() * 30,
                     'location.y': 20 + rng.random() * 60, 'shot.xg': rng.random() * 0.5,
                     'shot.isGoal': i % 7 == 0})
    return pd.DataFrame(rows)


def _matches():
    return pd.DataFrame([{'matchId': 100 + k, 'homeTeamName': 'Us' if k % 2 else 'Them',
                          'awayTeamName': 'Them' if k % 2 else 'Us', 'dateutc': f'2026-01-0{k + 1}',
                          'score': '1-1'} for k in range(3)])


def test_season_shot_map_builds_for_and_against_with_match_ids():
    ev, m = _events(), _matches()
    for mode in ('for', 'against'):
        fig = ti.plotly_season_shot_map(ev, m, 'Us', mode)
        assert len(fig.data) == 1 and len(fig.data[0].x) == 15
        rows = [list(r) for r in fig.data[0].customdata]
        assert ti.match_id_from_rows(rows[:1]) in (100, 101, 102)
        assert all(r[7] == ('Us' if mode == 'for' else 'Them') for r in rows)
    empty = ti.plotly_season_shot_map(ev.iloc[0:0], m, 'Us', 'for')
    assert len(empty.data) == 0 and empty.layout.annotations


def test_rolling_xg_builds_and_carries_match_ids():
    today = pd.Timestamp.today().normalize()
    hist = pd.DataFrame({
        'matchId': range(200, 212), 'teamName': 'Us', 'seasonId': [1] * 6 + [2] * 6,
        'roundId': [10] * 6 + [20] * 6,
        'date': [today - pd.Timedelta(days=7 * (12 - i)) for i in range(12)],
        'xG_For': np.linspace(0.5, 2.0, 12), 'xG_Against': np.linspace(1.5, 0.8, 12)})
    fig = ti.plotly_rolling_xg(hist, 'Us', None)
    line = fig.data[-1]
    assert len(line.x) == 12
    assert ti.match_id_from_rows([list(line.customdata[3])]) == 203
    assert ti.plotly_rolling_xg(hist.iloc[0:0], 'Us').layout.annotations


def test_match_shot_map_builds_and_carries_player_ids():
    """The match-level map uses the same builder style; customdata[0] is the
    playerId app.open_profile_from_selection reads."""
    import numpy as np, pandas as pd
    ev = pd.DataFrame({
        'matchId': [1] * 4, 'team.name': ['A', 'A', 'B', 'A'], 'type.primary': ['shot', 'penalty', 'shot', 'shot'],
        'location.x': [88.0, 90.0, 80.0, 70.0], 'location.y': [50.0, 50.0, 40.0, 60.0],
        'shot.xg': [0.3, 0.76, 0.05, 0.02], 'shot.isGoal': [True, False, False, False],
        'shot.onTarget': [True, True, False, False], 'player.id': [11, 12, 21, 11],
        'player.name': ['P11', 'P12', 'P21', 'P11'], 'minute': [12, 45, 60, 80], 'shot.bodyPart': ['right_foot'] * 4})
    info = {'homeTeamName': 'A', 'awayTeamName': 'B', 'score': '1-0'}
    fig = ti.plotly_match_shot_map(ev, info, 'A')
    pts = fig.data[0]
    assert len(pts.x) == 3 and int(pts.customdata[0][0]) == 11
    assert ti.player_id_from_rows([list(pts.customdata[1])]) == 12
    empty = ti.plotly_match_shot_map(ev, info, 'C')
    assert not empty.data
    # true pitch proportions: y units are 1.05/0.68 times x units
    assert abs(fig.layout.yaxis.scaleratio - 1.05 / 0.68) < 1e-9


def test_all_shot_maps_share_one_drawing():
    """Match, team-season and player maps must be the same drawing: marker
    size/opacity, true pitch proportions and axis ranges, xG colour axis."""
    import numpy as np, pandas as pd
    import pitch_interactive as pi
    ev = pd.DataFrame({'matchId': [1, 1], 'team.name': ['A', 'A'], 'type.primary': ['shot', 'shot'],
                       'location.x': [88.0, 80.0], 'location.y': [50.0, 40.0], 'shot.xg': [0.3, 0.05],
                       'shot.isGoal': [True, False], 'shot.onTarget': [True, False], 'player.id': [1, 2],
                       'player.name': ['P1', 'P2'], 'minute': [10, 20], 'shot.bodyPart': ['right_foot'] * 2})
    matches = pd.DataFrame({'matchId': [1], 'homeTeamName': ['A'], 'awayTeamName': ['B'], 'dateutc': ['2025-08-01'], 'score': ['1-0']})
    log = pd.DataFrame({'Shot Number': [1, 2], 'Date': ['2025-08-01'] * 2, 'Opponent': ['B'] * 2, 'Result': ['Goal', 'Miss'],
                        'xG': [0.3, 0.05], 'Body Part': ['Right foot'] * 2, 'Phase': ['Open play'] * 2, 'SCA': ['Pass'] * 2,
                        'location.x': [88.0, 80.0], 'location.y': [50.0, 40.0], 'shot.isGoal': [True, False], 'minute': [10, 20]})
    figs = [ti.plotly_match_shot_map(ev, {'homeTeamName': 'A', 'awayTeamName': 'B', 'score': '1-0'}, 'A', height=1000),
            ti.plotly_season_shot_map(ev, matches, 'A', 'for', height=1000),
            pi.plotly_shot_map(log, 'P1', height=1000)]
    refs = None
    for f in figs:
        m = f.data[0].marker
        sig = (m.size, m.opacity, f.layout.yaxis.scaleratio, tuple(f.layout.yaxis.range), tuple(f.layout.xaxis.range),
               f.layout.height, f.layout.coloraxis.cmax, f.layout.yaxis.constraintoward)
        refs = refs or sig
        assert sig == refs, sig
    assert refs[0] == pi.SHOT_MARKER['size'] and refs[5] == 1000 and refs[7] == 'top'
