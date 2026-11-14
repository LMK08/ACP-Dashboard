"""glossary.py — one glossary, names as shown, nothing invented (no app boot)."""
import ast
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import glossary  # noqa: E402

ENGINE_HELP = {'ACP Rating': 'Scale: 50 = league average.',
               'Aerial Grade %': 'Ranks aerial duels won above expectation. Within-role percentile, 0–100 — not a success rate.',
               'Evidence Weight': 'How much the projection trusts the record.'}


def _config():
    with open(os.path.join(ROOT, 'config.yaml'), encoding='utf-8') as fh:
        return yaml.safe_load(fh)


def _player_names():
    return {name for rows in glossary.PLAYER_GROUPS.values() for name, _ in rows}


def _pending(group=None):
    return {name for (g, name) in glossary.PENDING if group is None or g == group}


def test_every_entry_is_complete_and_unique_in_its_group():
    groups = glossary.groups(ENGINE_HELP)
    assert sum(len(rows) for rows in groups.values()) > 250
    for group, rows in groups.items():
        names = [name for name, _, _ in rows]
        assert len(set(names)) == len(names), (group, sorted(n for n in names if names.count(n) > 1))
        for name, text, direction in rows:
            assert name.strip() and len(text.strip()) > 5, (group, name)
            assert direction in (glossary.HIGHER, glossary.LOWER, glossary.NOT_SCORED), (group, name)
            assert text.strip()[-1] in '.)"', (group, name, text[-12:])


def test_every_player_metric_in_the_config_is_defined_or_listed_as_pending():
    """The gap is tracked: a metric a role template weights, or a stat
    category lists, is explained here or waits in PENDING for Lucas. A new
    metric in config.yaml fails this test until one of the two is true."""
    cfg = _config()
    named = {m for template in cfg['weights'].values() for m in template}
    named |= {m for metrics in cfg['metric_categories'].values() for m in metrics}
    named |= set(cfg['invert_metrics'])
    missing = sorted(named - _player_names() - _pending())
    assert not missing, missing


def test_a_pending_metric_is_shown_nowhere():
    shown = {(group, name) for group, rows in glossary.groups(ENGINE_HELP).items() for name, _, _ in rows}
    assert not shown & set(glossary.PENDING), shown & set(glossary.PENDING)
    assert all(note.strip() for note in glossary.PENDING.values())


def test_player_direction_comes_from_the_config_only():
    cfg = _config()
    weighted = {m for template in cfg['weights'].values() for m in template}
    for name in cfg['invert_metrics']:
        assert glossary.scored_as(name) == glossary.LOWER, name
    for name in weighted - set(cfg['invert_metrics']):
        assert glossary.scored_as(name) == glossary.HIGHER, name
    assert glossary.scored_as('Yellow cards') == glossary.NOT_SCORED       # no template uses it
    assert glossary.scored_as('not a metric') == glossary.NOT_SCORED


def _module_lists(path, names):
    """Module-level list literals, read without importing the module."""
    tree = ast.parse(open(os.path.join(ROOT, path), encoding='utf-8').read())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and getattr(node.targets[0], 'id', None) in names:
            out[node.targets[0].id] = ast.literal_eval(node.value)
    return out


def test_every_team_radar_metric_is_defined_or_listed_as_pending():
    lists = _module_lists('opposition_report.py', {'OFFENSIVE_METRICS', 'DISTRIBUTION_METRICS',
                                                   'DEFENSIVE_METRICS_TEAM', 'SET_PIECE_METRICS_RADAR'})
    assert len(lists) == 4
    team = {name for name, _, _ in glossary.TEAM_GROUPS['Team radars']} | _pending('Team radars')
    for key in ('OFFENSIVE_METRICS', 'DISTRIBUTION_METRICS', 'DEFENSIVE_METRICS_TEAM'):
        assert not set(lists[key]) - team, (key, sorted(set(lists[key]) - team))
    set_piece = {name for name, _, _ in glossary.TEAM_GROUPS['Set pieces']}
    assert not set(lists['SET_PIECE_METRICS_RADAR']) - set_piece
    # the inverted radar metrics are the ones scored 'lower'
    lower = {name for name, _, d in glossary.TEAM_GROUPS['Team radars'] if d == glossary.LOWER}
    assert lower == {'Goals Against', 'xG Against', 'xG per Shot Against', 'Shots Against', 'PPDA', 'Losses'}


def test_every_page_group_and_area_exists():
    groups = set(glossary.groups(ENGINE_HELP))
    for page, wanted in glossary.PAGE_GROUPS.items():
        assert set(wanted) <= groups | {glossary.COACH_GROUP}, (page, set(wanted) - groups)
    for area, wanted in glossary.AREAS.items():
        assert set(wanted) <= groups | {glossary.COACH_GROUP}, (area, set(wanted) - groups)


def test_search_and_table():
    rows = glossary.groups(ENGINE_HELP)['Goalkeeping']
    assert [r[0] for r in glossary.search(rows, 'prevented')] == ['goalsPrevented', 'goalsPreventedPerSOT']
    assert glossary.search(rows, '') == rows and glossary.search(rows, 'zzz') == []
    assert [r[0] for r in glossary.search(rows, 'LONG accurate')] == ['longPassesSuccessful_gk',
                                                                      'GK Long passes successful %']
    table = glossary.table_markdown(rows)
    assert table.count('\n') == len(rows) + 1
    assert '**passes\\_gk**' in table                     # the name prints as typed
    # the engine texts are the app's own; the shared percentile sentence is said once, in the note
    engine = dict((n, (t, d)) for n, t, d in glossary.groups(ENGINE_HELP)[glossary.ENGINE_GROUP])
    assert engine['Aerial Grade %'] == ('Ranks aerial duels won above expectation.', glossary.HIGHER)
    assert engine['Evidence Weight'][1] == glossary.NOT_SCORED
