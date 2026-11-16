"""The season totals behind the Match Predictor's team pickers
(app.build_season_cumulative_stats) are ONE season's totals, from that
season's rows, saved under that season's name and nothing else.

Until 2026-11 the function also attached every club's totals of the "prior"
season and took as prior the next-lower season id across BOTH leagues: Liga 3
2025/26 got the Campeonato 2025/26 (no club had any), the Campeonato 2023/24
a later Liga 3 season. Nothing read the value. Working it out called the
function for the other league's seasons with this league's frames and SAVED
the result: empty files over the other league's saved totals, rebuilt on the
next visit. The step was removed.
"""
import ast
import json
import os
import pickle
import types

import pandas as pd
import pytest

DASH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
L3_2526, L3_2425, CAMP_2526 = 191782, 190090, 191779


@pytest.fixture(scope='module')
def app():
    """app.py up to the data load (functions + constants only), as
    tests/test_projection_now.py loads it."""
    src = open(os.path.join(DASH, 'app.py'), encoding='utf-8').read()
    src = src[:src.index('# --- Load Data ---')]
    mod = types.ModuleType('app_under_test_season_totals')
    mod.__file__ = os.path.join(DASH, 'app.py')
    cwd = os.getcwd()
    os.chdir(DASH)
    try:
        exec(compile(src, 'app.py', 'exec'), mod.__dict__)
    finally:
        os.chdir(cwd)
    return mod


def _liga3_frames():
    """What the Match Predictor hands the function for Liga 3: the league's
    match list and events, two seasons, no Campeonato row."""
    matches = pd.DataFrame([
        {'matchId': 1, 'seasonId': L3_2526, 'homeTeamName': 'A', 'awayTeamName': 'B',
         'score': '2-1', 'dateutc': '2025-09-01 15:00:00', 'roundId': 7},
        {'matchId': 2, 'seasonId': L3_2526, 'homeTeamName': 'B', 'awayTeamName': 'C',
         'score': '0-0', 'dateutc': '2025-09-08 15:00:00', 'roundId': 7},
        {'matchId': 3, 'seasonId': L3_2425, 'homeTeamName': 'A', 'awayTeamName': 'C',
         'score': '1-3', 'dateutc': '2024-09-01 15:00:00', 'roundId': 5},
    ])
    shot = lambda mid, sid, team, xg, goal: {            # noqa: E731
        'matchId': mid, 'seasonId': sid, 'team.name': team, 'type.primary': 'shot',
        'shot.xg': xg, 'shot.isGoal': goal}
    events = pd.DataFrame([
        shot(1, L3_2526, 'A', 0.5, True), shot(1, L3_2526, 'A', 0.3, True),
        shot(1, L3_2526, 'B', 0.2, True),
        shot(2, L3_2526, 'C', 0.1, False),
        shot(3, L3_2425, 'C', 0.9, True),
        {'matchId': 1, 'seasonId': L3_2526, 'team.name': 'A', 'type.primary': 'pass',
         'shot.xg': None, 'shot.isGoal': None},
    ])
    return events, matches


def _saved(folder):
    return sorted(f for f in os.listdir(folder) if f.endswith('.pkl'))


def test_one_seasons_totals_and_only_that_seasons_file(app, tmp_path, monkeypatch):
    monkeypatch.setattr(app, 'STATS_CACHE_DIR', str(tmp_path))
    build = app.build_season_cumulative_stats.__wrapped__      # the function, without Streamlit's cache
    events, matches = _liga3_frames()
    got = build(events, matches, L3_2526)

    assert set(got) == {'A', 'B', 'C'}
    a, b, c = got['A'], got['B'], got['C']
    assert (a['matches'], a['wins'], a['points'], a['goals_for'], a['goals_against']) == (1, 1, 3, 2, 1)
    assert a['xG_for'] == pytest.approx(0.8) and a['xG_against'] == pytest.approx(0.2)
    assert (a['shots_for'], a['home_matches'], a['home_wins']) == (2, 1, 1)
    assert (b['matches'], b['losses'], b['draws'], b['points'], b['clean_sheets']) == (2, 1, 1, 1, 1)
    assert (c['matches'], c['draws'], c['points'], c['goals_for']) == (1, 1, 1, 0)
    # nothing of another season: 2024/25 (A lost 1-3, C scored 3) stays out
    assert a['losses'] == 0 and c['wins'] == 0

    # no "prior season" on any club
    assert not any('prior_stats' in club for club in got.values())
    # saved under this season's name only: not the season one id below it
    # (the Campeonato 2025/26), not the previous Liga 3 season
    assert _saved(tmp_path) == [f'season_cum_stats_{L3_2526}.pkl']
    assert not os.path.exists(tmp_path / f'season_cum_stats_{CAMP_2526}.pkl')

    # the saved copy is what a second call returns
    assert build(events, matches, L3_2526) == got


def test_a_file_saved_by_the_old_code_is_not_served(app, tmp_path, monkeypatch):
    """The old files carry 'prior_stats' (wrong for two seasons) under a
    stamp without a version. The stamp must differ so they are built again."""
    monkeypatch.setattr(app, 'STATS_CACHE_DIR', str(tmp_path))
    build = app.build_season_cumulative_stats.__wrapped__
    events, matches = _liga3_frames()
    fresh = build(events, matches, L3_2526)
    path = tmp_path / f'season_cum_stats_{L3_2526}.pkl'
    stamp = json.load(open(str(path) + '.meta.json'))
    assert stamp.get('v', 0) >= 2

    old_stamp = {k: v for k, v in stamp.items() if k != 'v'}        # what the old code wrote
    with open(path, 'wb') as fh:
        pickle.dump({'A': {'matches': 99, 'prior_stats': {'matches': 1}}}, fh)
    with open(str(path) + '.meta.json', 'w') as fh:
        json.dump(old_stamp, fh)
    assert build(events, matches, L3_2526) == fresh


def test_one_leagues_view_leaves_the_other_leagues_saved_totals_alone(app, tmp_path, monkeypatch):
    """The Liga 3 view saves a Liga 3 season; then the Campeonato view runs
    with the Campeonato's frames. The Liga 3 file must still hold its clubs.
    Before the fix the Campeonato call worked its way down the season ids,
    reached the Liga 3 season with frames that have none of its rows, and
    saved an empty file over it."""
    monkeypatch.setattr(app, 'STATS_CACHE_DIR', str(tmp_path))
    build = app.build_season_cumulative_stats.__wrapped__
    events, matches = _liga3_frames()
    assert set(build(events, matches, L3_2425)) == {'A', 'C'}

    camp_matches = pd.DataFrame([
        {'matchId': 9, 'seasonId': CAMP_2526, 'homeTeamName': 'X', 'awayTeamName': 'Y',
         'score': '1-0', 'dateutc': '2025-09-01 15:00:00', 'roundId': 3}])
    camp_events = pd.DataFrame([
        {'matchId': 9, 'seasonId': CAMP_2526, 'team.name': 'X', 'type.primary': 'shot',
         'shot.xg': 0.4, 'shot.isGoal': True}])
    assert set(build(camp_events, camp_matches, CAMP_2526)) == {'X', 'Y'}

    with open(tmp_path / f'season_cum_stats_{L3_2425}.pkl', 'rb') as fh:
        assert set(pickle.load(fh)) == {'A', 'C'}, "the Campeonato view overwrote a Liga 3 season's totals"
    assert _saved(tmp_path) == [f'season_cum_stats_{L3_2425}.pkl', f'season_cum_stats_{CAMP_2526}.pkl']


# ------------------------------------------------------------ source checks

def _function(name):
    with open(os.path.join(DASH, 'app.py'), encoding='utf-8') as fh:
        tree = ast.parse(fh.read())
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


def test_the_function_does_not_call_itself_or_order_seasons_by_id():
    body = _function('build_season_cumulative_stats').body[1:]          # without the docstring
    names = {n.id for b in body for n in ast.walk(b) if isinstance(n, ast.Name)}
    assert 'build_season_cumulative_stats' not in names, "the function calls itself again (another season)"
    assert 'SEASON_ID_MAP' not in names, "an order of season ids across both leagues is back"


def _code_strings(path):
    """String constants in a file's code, docstrings left out."""
    with open(path, encoding='utf-8') as fh:
        tree = ast.parse(fh.read())
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, 'body', [])
            if (body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                docstrings.add(id(body[0].value))
    return {n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings}


def test_no_page_reads_prior_stats():
    """The key is gone from the season totals: a page that reads it would
    raise KeyError (or, with .get, silently see nothing)."""
    files = ['app.py', 'opposition_report.py', 'scoreline_ui.py']
    views = os.path.join(DASH, 'views')
    files += [os.path.join('views', f) for f in sorted(os.listdir(views)) if f.endswith('.py')]
    readers = [f for f in files if 'prior_stats' in _code_strings(os.path.join(DASH, f))]
    assert not readers, f"these files use 'prior_stats', which the season totals no longer carry: {readers}"
