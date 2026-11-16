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
    # a row that is not one number still counts as 0 in the averages (as it
    # always did: what League Analysis offers was not changed with this fix)
    assert got.loc['A', 'Shots / on target'] == 0 and got.loc['A', 'Pure possession time'] == 0


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
