"""No page builds its own copy of a league's events.

The app holds ONE events table for every page and visitor (4.8 million rows
by 46 columns once loaded). A page that needs one league's rows reads the
shared, cached table from app.get_filtered_events (cache_resource: the same
object on every run, at most five scopes kept). A page that filters the
master table itself, filter_by_league(raw_events_df, comps), makes a new copy
of the league on EVERY run of the page: 0.8 GB of new memory for Liga 3
(measured 2026-11-15; pandas' "deep" count says 2.0 GB because it includes
the text a copy shares with the master). That line stopped two deploys on
2026-11-13 (Opposition Report, fixed in 67bd409) and was still on the Match
Predictor until 2026-11-15.

These are source checks: they need no data and run everywhere. They stop the
one line that has done this twice. They cannot see the master table under
another name, or another way of copying it (a mask written out by hand,
get_season_events on the master).
"""
import ast
import os

DASH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# the names the master events table goes by in the app and its pages
MASTER_EVENT_NAMES = {'raw_events_df', '_raw_events_df', 'all_raw_events_df'}


def _page_files():
    files = [os.path.join(DASH, 'app.py'), os.path.join(DASH, 'opposition_report.py')]
    views = os.path.join(DASH, 'views')
    files += [os.path.join(views, f) for f in sorted(os.listdir(views)) if f.endswith('.py')]
    return files


def _name(node):
    """`x` for the name x, `y` for the attribute a.y, else None."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _argument(call, position, keyword):
    """The argument of a call, given by position or by keyword; else None."""
    if len(call.args) > position:
        return call.args[position]
    for kw in call.keywords:
        if kw.arg == keyword:
            return kw.value
    return None


def _league_copies_of_master_events(source):
    """Line numbers of every call filter_by_league(<the master events>, ...).
    Comments and docstrings do not count: only real calls."""
    return sorted(
        node.lineno for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and _name(node.func) == 'filter_by_league'
        and _name(_argument(node, 0, 'df')) in MASTER_EVENT_NAMES)


def _all_seasons_reads_of_the_shared_table(source):
    """Line numbers of every call get_filtered_events(<events>, None, <league>)."""
    lines = []
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.Call) and _name(node.func) == 'get_filtered_events'):
            continue
        season = _argument(node, 1, 'season_ids')
        if isinstance(season, ast.Constant) and season.value is None:
            lines.append(node.lineno)
    return sorted(lines)


def test_the_check_finds_a_league_copy_and_nothing_else():
    assert _league_copies_of_master_events(
        "pred_events = filter_by_league(raw_events_df, comps)\n") == [1]
    assert _league_copies_of_master_events(
        "x = 1\nev = app.filter_by_league(app.raw_events_df, comps)\n") == [2]
    assert _league_copies_of_master_events(
        "ev = filter_by_league(df=raw_events_df, comp_ids=comps)\n") == [1]
    # the match list is small: filtering it per run is fine
    assert _league_copies_of_master_events(
        "m = filter_by_league(matches_summary_df, comps)\n") == []
    # the shared table is the right way
    assert _league_copies_of_master_events(
        "ev = get_filtered_events(raw_events_df, None, comps)\n") == []
    # a mention in a docstring or a comment is not a call
    assert _league_copies_of_master_events(
        'def f():\n    """was filter_by_league(raw_events_df, comps)"""\n'
        "    # filter_by_league(raw_events_df, comps)\n    return 1\n") == []


def test_the_check_finds_an_all_seasons_read_of_the_shared_table():
    assert _all_seasons_reads_of_the_shared_table(
        "ev = get_filtered_events(raw_events_df, None, comps)\n") == [1]
    assert _all_seasons_reads_of_the_shared_table(
        "ev = app.get_filtered_events(events_df=raw, season_ids=None, comp_ids=comps)\n") == [1]
    # one season of the league is another scope
    assert _all_seasons_reads_of_the_shared_table(
        "ev = get_filtered_events(raw_events_df, sid, comps)\n") == []
    assert _all_seasons_reads_of_the_shared_table(
        "ev = filter_by_league(raw_events_df, comps)\n") == []


def test_no_page_copies_the_league_events_itself():
    found = []
    for path in _page_files():
        with open(path, encoding='utf-8') as fh:
            lines = _league_copies_of_master_events(fh.read())
        found += [f"{os.path.relpath(path, DASH)}:{n}" for n in lines]
    assert not found, (
        "these lines copy a league's events on every run of the page; read the "
        "shared table instead (app.get_filtered_events(raw_events_df, None, comps)): "
        + ", ".join(found))


def test_match_predictor_reads_the_shared_league_table():
    """The page hands the league's events, all seasons, to
    build_season_cumulative_stats. They come from the shared table, scope
    (None, league): the same rows in the same order as the old per-run copy,
    so the season totals do not change."""
    with open(os.path.join(DASH, 'views', 'match_predictor.py'), encoding='utf-8') as fh:
        source = fh.read()
    assert _all_seasons_reads_of_the_shared_table(source), (
        "views/match_predictor.py no longer reads the league's events from the "
        "shared table (get_filtered_events(raw_events_df, None, comps))")
