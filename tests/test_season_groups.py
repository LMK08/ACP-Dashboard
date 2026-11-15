"""Liga 3's group lists live in ONE place, league_config.SEASON_GROUPS, and
the Opposition Report offers only clubs of the chosen season.

Until 2026-11 there were three copies of the group lists. The Opposition
Report's was a hard-coded 2025/26 Série B, added to the Opponent list of
every Liga 3 season. In every season but 2025/26 the picker then offered
three to six clubs with no match in it (2026/27: 1º Dezembro, Amora,
Académica), kept such a club across a season switch and drew an empty report
for it, with the line "Offensive Threat: Average" about a club with no data.

Nothing here depends on which season is the current one or on how many
matches have been played: a season turnover or a data refresh must not fail
these tests for no reason.
"""
import ast
import os

import pandas as pd
import pytest

from league_config import COMPETITIONS, SEASON_GROUPS, group_of

DASH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIGA3 = 43324
OUR_TEAM = 'Atlético CP'
S2526, S2627 = 191782, 192831          # the two seasons the copies held
MATCH_LIST = os.path.join(DASH, 'matches_summary.parquet')


# ---------------------------------------------------------------- the lists

def test_the_current_liga3_season_has_its_groups():
    """A reminder, not a gate: the séries may not be published when a season
    is added. Without them nothing shows last season's clubs (see below)."""
    current = COMPETITIONS[LIGA3]['current_season']
    if current not in SEASON_GROUPS:
        pytest.skip(f"no séries in league_config.SEASON_GROUPS for the current Liga 3 "
                    f"season {current} yet: add them when they are known")


@pytest.mark.parametrize('season_id', sorted(SEASON_GROUPS))
def test_each_season_has_two_well_formed_groups(season_id):
    groups = SEASON_GROUPS[season_id]
    assert season_id in COMPETITIONS[LIGA3]['seasons'], "groups are Liga 3 seasons"
    assert set(groups) == {'Group A', 'Group B'}
    a, b = groups['Group A'], groups['Group B']
    assert isinstance(a, tuple) and isinstance(b, tuple), "tuples: no reader can change the shared lists"
    assert a and b
    assert len(a) == len(set(a)) and len(b) == len(set(b)), "a club is listed twice"
    assert not set(a) & set(b), f"in both groups: {sorted(set(a) & set(b))}"
    assert all(isinstance(c, str) and c and c == c.strip() for c in a + b)


def test_group_of():
    ours = group_of(S2627, OUR_TEAM)
    assert ours == SEASON_GROUPS[S2627]['Group B'] and OUR_TEAM in ours
    assert group_of(S2627, 'Fafe') == SEASON_GROUPS[S2627]['Group A']
    assert group_of(S2526, 'Amora') == SEASON_GROUPS[S2526]['Group B']
    assert group_of(S2627, 'Amora') == ()            # not in Liga 3 that season
    assert group_of(S2627, 'No Such Club') == ()
    assert group_of(188221, OUR_TEAM) == ()          # 2021/22: no groups here
    assert group_of(191779, OUR_TEAM) == ()          # a Campeonato season
    assert group_of(None, OUR_TEAM) == ()
    assert group_of([S2627], OUR_TEAM) == ()         # a list of seasons is not a season


def test_the_lists_are_what_the_three_copies_held():
    """The move changed where the lists live, not the lists: same clubs in
    the same order as the copies in views/league_analysis.py and
    simulate_season.py (and, for 2025/26, the Opposition Report's nine)."""
    assert SEASON_GROUPS[S2526]['Group A'] == (
        'Fafe', 'Varzim', 'Paredes', 'Sanjoanense', 'São João Ver',
        'Amarante', 'Vitória Guimarães II', 'Trofense', 'Sporting Braga II', 'AD Marco 09')
    assert SEASON_GROUPS[S2526]['Group B'] == (
        '1º Dezembro', 'Caldas', 'Sporting Covilhã', 'Mafra', 'União Santarém',
        'Amora', 'Académica', 'CF Os Belenenses', 'Lusitano Évora 1911', 'Atlético CP')
    assert SEASON_GROUPS[S2627]['Group A'] == (
        'Fafe', 'Varzim', 'Paredes', 'Paços de Ferreira', 'São João Ver',
        'Leça', 'Vitória Guimarães II', 'Trofense', 'Vianense', 'AD Marco 09')
    assert SEASON_GROUPS[S2627]['Group B'] == (
        'Louletano', 'Caldas', 'Sporting Covilhã', 'Mafra', 'União Santarém',
        'UD Oliveirense', 'Vitória de Sernache', 'CF Os Belenenses', 'Lusitano Évora 1911', 'Atlético CP')


# ------------------------------------------------- the season simulation

def test_the_season_simulation_reads_the_shared_lists():
    import simulate_season as ss
    assert ss._first_stage_groups(S2627) == {'North': list(SEASON_GROUPS[S2627]['Group A']),
                                             'South': list(SEASON_GROUPS[S2627]['Group B'])}
    assert ss.FIRST_STAGE_GROUPS == ss._first_stage_groups(COMPETITIONS[LIGA3]['current_season'])
    # its own lists: a change made by the simulation cannot reach the shared ones
    got = ss._first_stage_groups(S2627)
    got['North'].append('x')
    assert 'x' not in SEASON_GROUPS[S2627]['Group A']


def test_a_season_without_groups_skips_the_simulation_and_says_so(monkeypatch, capsys):
    """A new current season whose séries are not in league_config yet: the
    import still works, Liga 3 is skipped with a ::warning:: line (GitHub
    shows it on the run's page) and the nightly job goes on. It must never
    simulate the new season with last season's séries."""
    import simulate_season as ss
    assert ss._first_stage_groups(999999) == {'North': [], 'South': []}
    monkeypatch.setattr(ss, 'FIRST_STAGE_GROUPS', {'North': [], 'South': []})
    current = COMPETITIONS[LIGA3]['current_season']
    matches = pd.DataFrame({'seasonId': [current], 'homeTeamName': ['A'], 'awayTeamName': ['B'],
                            'score': ['1-0'], 'roundId': [1]})
    # sampler=None: the skip comes before anything is drawn
    assert ss.simulate_liga3(matches, None, n_sims=10) == {}
    out = capsys.readouterr().out
    assert '::warning::' in out and 'SEASON_GROUPS' in out and str(current) in out


def _long_club_lists(path):
    """Every list, tuple or set written out in a file's code that holds five
    or more clubs of ONE group: a group list spelled out again."""
    with open(path, encoding='utf-8') as fh:
        tree = ast.parse(fh.read())
    groups = [set(g) - {OUR_TEAM} for season in SEASON_GROUPS.values() for g in season.values()]
    found = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            names = {e.value for e in node.elts
                     if isinstance(e, ast.Constant) and isinstance(e.value, str)}
            if any(len(names & g) >= 5 for g in groups):
                found.append(node.lineno)
    return sorted(found)


@pytest.mark.parametrize('path', ['opposition_report.py', 'views/league_analysis.py',
                                  'simulate_season.py'])
def test_no_other_file_keeps_its_own_group_list(path):
    found = _long_club_lists(os.path.join(DASH, path))
    assert not found, f"{path} spells out a group list itself at line(s) {found}: read league_config.SEASON_GROUPS"


def test_the_copy_check_sees_a_list_and_ignores_a_few_names(tmp_path):
    copy = tmp_path / 'copy.py'
    copy.write_text("G = ['Caldas', 'Mafra', 'Louletano', 'UD Oliveirense', 'Sporting Covilhã']\n",
                    encoding='utf-8')
    assert _long_club_lists(str(copy)) == [1]
    few = tmp_path / 'few.py'
    few.write_text("RELEGATED = ('Mafra', 'UD Oliveirense')\nTIES = {'Fafe': 1, 'Varzim': 2}\n"
                   "# ['Caldas', 'Mafra', 'Louletano', 'UD Oliveirense', 'Sporting Covilhã']\n",
                   encoding='utf-8')
    assert _long_club_lists(str(few)) == []


# ------------------------------------------------------- the Opponent list

def _matches(rows):
    return pd.DataFrame(rows, columns=['homeTeamName', 'awayTeamName'])


def test_opponents_are_the_seasons_clubs_plus_that_seasons_own_group():
    import opposition_report as opp
    # early in 2026/27: two matches played, most of our group not seen yet
    early = _matches([(OUR_TEAM, 'Caldas'), ('Fafe', 'Varzim')])
    got = opp._season_opponents(early, S2627)
    assert got == sorted((set(SEASON_GROUPS[S2627]['Group B']) - {OUR_TEAM}) | {'Fafe', 'Varzim'})
    assert OUR_TEAM not in got
    # a season with no groups here: its clubs and nothing else
    old = _matches([('Torreense', 'Alverca'), ('Pevidém', 'Fafe'), (None, 'Fafe')])
    assert opp._season_opponents(old, 188221) == ['Alverca', 'Fafe', 'Pevidém', 'Torreense']
    # last season's group never reaches another season
    assert 'Amora' in opp._season_opponents(_matches([]), S2526)
    assert 'Amora' not in opp._season_opponents(early, S2627)
    assert 'Amora' not in opp._season_opponents(old, 188221)
    # nothing listed and no group: an empty list (the page then says so)
    assert opp._season_opponents(_matches([]), 188221) == []
    assert opp._season_opponents(pd.DataFrame(), 188221) == []


@pytest.mark.skipif(not os.path.exists(MATCH_LIST), reason='match list not present')
def test_real_match_list_every_offered_club_belongs_to_the_season():
    """On the real match list, every Liga 3 season: a club is offered only if
    it has a match in that season or is in that season's own group. The old
    list failed this in five of the six seasons."""
    import opposition_report as opp
    matches = pd.read_parquet(MATCH_LIST, columns=['seasonId', 'homeTeamName', 'awayTeamName'])
    seen = 0
    for season_id in COMPETITIONS[LIGA3]['seasons']:
        season = matches[matches['seasonId'] == season_id]
        if season.empty:
            continue
        seen += 1
        played = set(season['homeTeamName'].dropna()) | set(season['awayTeamName'].dropna())
        offered = opp._season_opponents(season, season_id)
        strangers = sorted(set(offered) - played - set(group_of(season_id, OUR_TEAM)))
        assert not strangers, f"season {season_id} offers clubs that are not of it: {strangers}"
        assert set(offered) >= played - {OUR_TEAM}, "a club of the season is missing"
        assert OUR_TEAM not in offered
    assert seen, "no Liga 3 season found in the match list"


@pytest.mark.skipif(not os.path.exists(MATCH_LIST), reason='match list not present')
@pytest.mark.parametrize('season_id', [S2526, S2627])
def test_real_match_list_the_two_known_seasons(season_id):
    """The two seasons whose groups were moved: every club of our group has
    matches there, so the group adds no name the match list does not have
    (a second spelling of a club would show up here as an extra option)."""
    import opposition_report as opp
    matches = pd.read_parquet(MATCH_LIST, columns=['seasonId', 'homeTeamName', 'awayTeamName'])
    season = matches[matches['seasonId'] == season_id]
    if season.empty:
        pytest.skip(f"season {season_id} is not in this match list")
    played = set(season['homeTeamName'].dropna()) | set(season['awayTeamName'].dropna())
    assert set(group_of(season_id, OUR_TEAM)) <= played
    assert opp._season_opponents(season, season_id) == sorted(played - {OUR_TEAM})


def test_the_page_uses_the_season_list_and_no_hard_coded_group():
    with open(os.path.join(DASH, 'opposition_report.py'), encoding='utf-8') as fh:
        tree = ast.parse(fh.read())
    assert 'GROUP_B_TEAMS' not in {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    render = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == 'render_opposition_report')
    called = {n.func.id for n in ast.walk(render)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert '_season_opponents' in called, "the Opponent list is no longer built by _season_opponents"
    assert '_club_has_match' in called, "the takeaways are no longer told whether the club has a match"


# ------------------------------------------------- the line about no matches

def test_club_has_match():
    import opposition_report as opp
    season = _matches([(OUR_TEAM, 'Caldas'), ('Fafe', None)])
    assert opp._club_has_match(season, 'Caldas') is True
    assert opp._club_has_match(season, 'Fafe') is True
    assert opp._club_has_match(season, 'Amora') is False
    assert opp._club_has_match(pd.DataFrame(), 'Caldas') is False


def test_a_club_with_no_match_is_not_rated_average():
    import opposition_report as opp
    args = ([], [], [], [], [], None, pd.DataFrame())
    average = ["**Offensive Threat:** Average; no standout attacking "
               "metrics above 65th percentile."]
    # a club that is in the match list and has no standout strength: the old
    # line, unchanged, whether or not the flag is given
    assert opp._generate_key_takeaways('Caldas', *args) == average
    assert opp._generate_key_takeaways('Caldas', *args, has_matches=True) == average
    # a club with no match in the season: said plainly, no verdict
    assert opp._generate_key_takeaways('Amora', *args, has_matches=False) == [
        "**No matches yet:** Amora has no match in this season's data, "
        "so there is nothing to rate."]
    # the name is trimmed (the events spell one club with a trailing space)
    assert 'Mortágua has no' in opp._generate_key_takeaways('Mortágua ', *args, has_matches=False)[0]
    # a strength still comes first, whatever the flag
    strong = opp._generate_key_takeaways('Caldas', [('Goals', 80.0)], [], [], [], [], None,
                                         pd.DataFrame(), has_matches=False)
    assert strong[0].startswith('**Offensive Strengths:**')
