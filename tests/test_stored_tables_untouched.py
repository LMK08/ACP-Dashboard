"""League Analysis's season averages (app.calculate_expanded_team_stats) must
leave the stored match tables exactly as they were.

all_match_data is ONE object for every page and visitor. Until 2026-11 the
function converted the two team columns of every stored table to numbers IN
the table (`df[home] = pd.to_numeric(df[home], errors='coerce').fillna(0)`).
27 kinds of rows are text that is not one number ("Shots / on target" '14/5',
"Total passes / accurate" '450/317', "Pure possession time" '24:53'); each
became 0.0, and Match Analysis and the match report PDF showed 0.0 for every
match of the season until the app restarted. On the live Space that happened
on the first League Analysis view of a season after every restart.
"""
import ast
import copy
import os
import types

import pandas as pd
import pytest

DASH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEASON = 192831


@pytest.fixture(scope='module')
def app():
    """app.py up to the data load (functions + constants only), as
    tests/test_projection_now.py loads it."""
    src = open(os.path.join(DASH, 'app.py'), encoding='utf-8').read()
    src = src[:src.index('# --- Load Data ---')]
    mod = types.ModuleType('app_under_test_stored_tables')
    mod.__file__ = os.path.join(DASH, 'app.py')
    cwd = os.getcwd()
    os.chdir(DASH)
    try:
        exec(compile(src, 'app.py', 'exec'), mod.__dict__)
    finally:
        os.chdir(cwd)
    return mod


def _table(home, away, rows):
    """A stored match table: metric names down the side, one column per team,
    cells as the pipeline stores them (text, whole numbers, decimals mixed)."""
    return pd.DataFrame({home: [r[1] for r in rows], away: [r[2] for r in rows]},
                        index=[r[0] for r in rows], dtype=object)


def _data():
    matches = pd.DataFrame([
        {'matchId': 1, 'seasonId': SEASON, 'homeTeamName': 'A', 'awayTeamName': 'B'},
        {'matchId': 2, 'seasonId': SEASON, 'homeTeamName': 'B', 'awayTeamName': 'A'},
    ])
    stored = {
        1: {'stats_version': 3, 'team_stats': {
            'General': _table('A', 'B', [('Goals', 2, 1), ('xG', 1.5, 0.5),
                                         ('Shots / on target', '14/5', '7/2'), ('Corners', '5', '3')]),
            'Possession': _table('A', 'B', [('Pure possession time', '24:53', '17:51')])}},
        2: {'stats_version': 3, 'team_stats': {
            'General': _table('B', 'A', [('Goals', 0, 0), ('xG', 0.25, 1.0),
                                         ('Shots / on target', '9/4', '11/6'), ('Corners', '1', '7')])}},
    }
    return stored, matches


def _same(a, b):
    for mid in a:
        for cat, df in a[mid]['team_stats'].items():
            other = b[mid]['team_stats'][cat]
            if not (df.equals(other) and list(df.dtypes) == list(other.dtypes)
                    and list(df.index) == list(other.index) and list(df.columns) == list(other.columns)):
                return False
    return True


def test_the_season_averages_leave_the_stored_tables_as_they_were(app):
    build = app.calculate_expanded_team_stats.__wrapped__      # the function, without Streamlit's cache
    stored, matches = _data()
    before = copy.deepcopy(stored)
    got = build(stored, matches, season_id=None)               # no season: nothing saved, always computed

    assert _same(stored, before), "the function changed a stored match table"
    general = stored[1]['team_stats']['General']
    assert general.loc['Shots / on target', 'A'] == '14/5'     # what Match Analysis shows
    assert stored[1]['team_stats']['Possession'].loc['Pure possession time', 'B'] == '17:51'
    assert general['A'].dtype == object

    # the averages themselves: per match, as before
    assert got.loc['A', 'Goals'] == pytest.approx(1.0) and got.loc['B', 'Goals'] == pytest.approx(0.5)
    assert got.loc['A', 'xG'] == pytest.approx(1.25) and got.loc['B', 'xG'] == pytest.approx(0.375)
    assert got.loc['A', 'Corners'] == pytest.approx(6.0) and got.loc['B', 'Corners'] == pytest.approx(2.0)
    # a row that is never one number is not averaged at all (see the tests below)
    assert 'Shots / on target' not in got.columns and 'Pure possession time' not in got.columns


def test_with_a_saved_file_too(app, tmp_path, monkeypatch):
    """With a season the result is saved and the next call reads it back:
    the stored tables stay untouched on both paths."""
    monkeypatch.setattr(app, 'STATS_CACHE_DIR', str(tmp_path))
    build = app.calculate_expanded_team_stats.__wrapped__
    stored, matches = _data()
    before = copy.deepcopy(stored)
    first = build(stored, matches, season_id=SEASON)
    assert os.path.exists(tmp_path / f'expanded_team_stats_{SEASON}.parquet')
    second = build(stored, matches, season_id=SEASON)
    pd.testing.assert_frame_equal(first, second)
    assert _same(stored, before)


# ---------------------------------------------------------------------------
# Rows that are never one number are left out of the averages
# ---------------------------------------------------------------------------
# Until 2026-11 they came out as 0 for every club, and League Analysis offered
# 27 always-zero metrics ("Shots / on target", "Total passes / accurate",
# "Pure possession time" ...) in its custom charts and raw table. Lucas,
# 2026-11-16: leave them out.

def _mixed_data():
    matches = pd.DataFrame([
        {'matchId': 1, 'seasonId': SEASON, 'homeTeamName': 'A', 'awayTeamName': 'B'},
        {'matchId': 2, 'seasonId': SEASON, 'homeTeamName': 'B', 'awayTeamName': 'A'},
    ])
    stored = {
        1: {'stats_version': 3, 'team_stats': {'General': _table('A', 'B', [
            ('Goals', 2, 1), ('Shots / on target', '14/5', '7/2'), ('Corners', '5', '3'),
            ('Offsides', 0, 0), ('Pure possession time', '24:53', '17:51'),
            ('Sometimes a number', 4, '-'), ('Yellow / red cards', '1/0', '4/0')])}},
        2: {'stats_version': 3, 'team_stats': {'General': _table('B', 'A', [
            ('Goals', 0, 0), ('Shots / on target', '9/4', '11/6'), ('Corners', '1', '7'),
            ('Offsides', 0, 0), ('Pure possession time', '20:00', '21:00'),
            ('Sometimes a number', '-', '-'), ('Yellow / red cards', '0/0', '2/1')])}},
    }
    return stored, matches


def test_rows_that_are_never_one_number_are_left_out(app):
    build = app.calculate_expanded_team_stats.__wrapped__
    stored, matches = _mixed_data()
    got = build(stored, matches, season_id=None)

    # the order of the rows that stay is the order they are stored in
    assert list(got.columns) == ['Goals', 'Corners', 'Offsides', 'Sometimes a number']
    for row in ('Shots / on target', 'Pure possession time', 'Yellow / red cards'):
        assert row not in got.columns, f"{row!r} is never one number and must not be averaged"
    # numbers written as text are numbers
    assert got.loc['A', 'Corners'] == pytest.approx(6.0) and got.loc['B', 'Corners'] == pytest.approx(2.0)
    # a number that is 0 for every club is data, and stays
    assert (got['Offsides'] == 0).all()
    # a row that is a number in ONE cell stays; its other cells count as 0, as before
    assert got.loc['A', 'Sometimes a number'] == pytest.approx(2.0) and got.loc['B', 'Sometimes a number'] == 0


def test_a_season_with_no_number_at_all_gives_an_empty_table_not_an_error(app):
    build = app.calculate_expanded_team_stats.__wrapped__
    matches = pd.DataFrame([{'matchId': 1, 'seasonId': SEASON, 'homeTeamName': 'A', 'awayTeamName': 'B'}])
    stored = {1: {'team_stats': {'General': _table('A', 'B', [('Shots / on target', '14/5', '7/2')])}}}
    got = build(stored, matches, season_id=None)
    assert list(got.columns) == [] and set(got.index) == {'A', 'B'}


def test_a_saved_file_from_before_is_not_served(app, tmp_path, monkeypatch):
    """Files saved before this change hold the 27 zero columns under a stamp
    without a version: the stamp must differ, so they are built again."""
    import json
    monkeypatch.setattr(app, 'STATS_CACHE_DIR', str(tmp_path))
    build = app.calculate_expanded_team_stats.__wrapped__
    stored, matches = _mixed_data()
    fresh = build(stored, matches, season_id=SEASON)
    path = tmp_path / f'expanded_team_stats_{SEASON}.parquet'
    stamp = json.load(open(str(path) + '.meta.json'))
    assert stamp.get('v', 0) >= 2

    old_file = fresh.copy()
    old_file['Shots / on target'] = 0.0                    # what the old code saved
    old_file.to_parquet(path)
    with open(str(path) + '.meta.json', 'w') as fh:
        json.dump({k: v for k, v in stamp.items() if k != 'v'}, fh)
    again = build(stored, matches, season_id=SEASON)
    assert 'Shots / on target' not in again.columns
    pd.testing.assert_frame_equal(again, fresh)


def test_the_function_never_assigns_into_a_table():
    """A source check for the same rule: inside the function nothing is
    assigned into `df` (the stored table) or changed in place."""
    with open(os.path.join(DASH, 'app.py'), encoding='utf-8') as fh:
        tree = ast.parse(fh.read())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == 'calculate_expanded_team_stats')

    def root(node):
        while isinstance(node, (ast.Subscript, ast.Attribute)):
            node = node.value
        return node.id if isinstance(node, ast.Name) else None

    writes = []
    for node in ast.walk(fn):
        targets = (node.targets if isinstance(node, ast.Assign)
                   else [node.target] if isinstance(node, (ast.AugAssign, ast.AnnAssign)) else [])
        for t in targets:
            if isinstance(t, (ast.Subscript, ast.Attribute)) and root(t) in ('df', 'match_data', '_all_match_data'):
                writes.append(node.lineno)
        if (isinstance(node, ast.Call) and root(node.func) in ('df', 'match_data', '_all_match_data')
                and any(kw.arg == 'inplace' and isinstance(kw.value, ast.Constant) and kw.value.value
                        for kw in node.keywords)):
            writes.append(node.lineno)
    assert not writes, f"calculate_expanded_team_stats writes into a stored table at app.py line(s) {sorted(writes)}"
