"""team_rules.py — the rules more than one page shares (no app boot)."""
import math
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import team_rules  # noqa: E402


def _events(rows):
    return pd.DataFrame(rows, columns=['type.primary', 'type.secondary', 'location.x', 'groundDuel.takeOn'])


def test_a_dribble_is_the_dribblers_row_never_the_defenders():
    """Wyscout sets groundDuel.takeOn on BOTH rows of a take-on. The old rule
    ("tagged dribble OR take-on") counted the defender's row as his dribble."""
    ev = _events([
        ('duel', np.array(['dribble', 'offensive_duel']), 70, True),     # the dribbler, a take-on
        ('duel', np.array(['defensive_duel']), 30, True),                # the defender of that take-on
        ('duel', ['dribble'], 55, False),                                # a dribble that is not a take-on
        ('pass', ['forward_pass'], 40, None),
        ('duel', None, 40, None),
    ])
    assert team_rules.dribble_attempts(ev).tolist() == [True, False, True, False, False]


def test_stored_flags_are_brought_to_the_rule():
    ev = _events([
        ('duel', ['dribble'], 70, True),
        ('duel', ['defensive_duel'], 30, True),
        ('duel', ['dribble'], 55, False),
    ])
    ev['is_dribble_attempt'] = [True, True, True]              # as stored before 2026-11-14
    ev['is_custom_dribble_success'] = [True, True, False]
    out = team_rules.correct_dribble_flags(ev)
    assert out['is_dribble_attempt'].tolist() == [True, False, True]
    assert out['is_custom_dribble_success'].tolist() == [True, False, False]   # the dribbler's own result is kept


def test_ppda_counts_both_sides_in_the_opponents_own_sixty_per_cent():
    opponent = _events([
        ('pass', None, 10, None), ('pass', None, 60, None),      # in its own 60 %: counted
        ('pass', None, 61, None), ('pass', None, 95, None),      # its attacking 40 %: not
        ('shot', None, 20, None),
    ])
    team = _events([
        ('interception', None, 40, None),                        # x >= 40 in OUR coordinates: counted
        ('clearance', None, 39, None),                           # too deep: not
        ('duel', ['defensive_duel'], 80, None),                  # counted, won or lost
        ('duel', ['sliding_tackle'], 50, None),                  # counted
        ('infraction', None, 70, None),                          # a foul is not a defensive action here
        ('duel', ['aerial_duel'], 70, None),
    ])
    assert team_rules.ppda(team, opponent) == 2 / 3
    assert team_rules.ppda(team.iloc[[1, 4]], opponent) is None  # no action in the zone
    assert team_rules.ppda(team.iloc[:0], opponent) is None


def test_directness_is_the_share_of_passes_played_forward():
    passes = _events([('pass', ['forward_pass'], 1, None), ('pass', ['back_pass'], 1, None),
                      ('pass', None, 1, None), ('pass', ['lateral_pass', 'long_pass'], 1, None)])
    assert team_rules.forward_pass_share(passes) == 25.0
    assert team_rules.forward_pass_share(passes.iloc[:0]) == 0.0


def test_shot_distance_is_in_metres():
    d = team_rules.shot_distance_m(pd.Series([100, 90, 100, np.nan]), pd.Series([50, 50, 0, 50]))
    assert d.iloc[0] == 0 and math.isclose(d.iloc[1], 10.5) and math.isclose(d.iloc[2], 34.0) and np.isnan(d.iloc[3])


def _source(path):
    return open(os.path.join(ROOT, path), encoding='utf-8').read()


def test_the_pages_use_these_rules_and_not_a_copy():
    """The mismatches of 2026-11 stay fixed: no radar axis filled from a
    Wyscout field that means something else, no dribble counted off the
    take-on flag, one PPDA rule, shot distance through the metre helper."""
    app, pdata, pv = _source('app.py'), _source('process_data.py'), _source('pitch_visualizations.py')
    for wrong in ("row.get('passes_to_final_third'", "row.get('progressive_run'", "row.get('successful_dribbles'"):
        assert wrong not in app, wrong
    assert "RADAR_AXES_FROM_EVENTS = ('Passes into Box', 'Progressive Passes', 'Dribbles')" in app
    assert 'directness = team_rules.forward_pass_share(team_passes)' in app
    for src in (app, pdata):
        assert "| (df.get('groundDuel.takeOn') == True)" not in src
        assert "df['is_dribble_attempt'] = team_rules.dribble_attempts(df)" in src
        assert 'team_rules.ppda(' in src
    assert "[team_duels.get('groundDuel.takeOn') == True]" not in pdata
    assert 'team_rules.shot_distance_m(' in pdata and 'team_rules.correct_dribble_flags(all_raw_events_df)' in pdata
    assert pv.count('team_rules.dribble_attempts(te)') >= 2
    # the events radar must not write into the frame it is given (every session shares it)
    assert 'season_events_df = season_events_df.copy(deep=False)' in app
    # a failed recompute of a match keeps its cached tables and is not stamped as done
    assert "'stats_version': MATCH_STATS_VERSION if _computed else None" in pdata
    assert 'keeping the cached tables' in pdata


def _calculate_match_data(monkeypatch, match_df, home, away):
    """process_data.calculate_match_data, quietly, where python-dotenv may be missing."""
    import contextlib
    import io
    import types
    # process_data imports python-dotenv at the top, for its main(). Only the
    # nightly job installs that package: the Space and the deploy check do
    # not (a test failed the check for that reason, 2026-11-14). The
    # function under test never reads the environment, so a stand-in will do.
    try:
        import dotenv  # noqa: F401
    except ModuleNotFoundError:
        monkeypatch.setitem(sys.modules, 'dotenv',
                            types.SimpleNamespace(load_dotenv=lambda *a, **k: False))
    # importing process_data switches a pandas warning off for the whole
    # process; put the setting back for the tests that run after this one
    _chained = pd.get_option('mode.chained_assignment')
    try:
        import process_data
        with contextlib.redirect_stdout(io.StringIO()):
            return process_data.calculate_match_data(match_df, home, away), process_data.MATCH_STATS_VERSION
    finally:
        pd.set_option('mode.chained_assignment', _chained)


def test_a_match_that_could_not_be_computed_is_not_stamped_as_done(monkeypatch):
    """calculate_match_data swallows its own errors and returns empty tables.
    Such a result carries no stamp, so the cached tables are kept and the
    next run tries again (process_data.main)."""
    broken = pd.DataFrame({'matchId': [1], 'team.name': ['A']})          # nothing a match needs
    out, _ = _calculate_match_data(monkeypatch, broken, 'A', 'B')
    assert out['stats_version'] is None
    assert not any(isinstance(df, pd.DataFrame) and not df.empty for df in out['team_stats'].values())


# --- one match, one spelling of each club -------------------------------------

def _match(home='Montalegre', away='Pevidém', as_category=False):
    df = pd.DataFrame({
        'matchId': [1] * 4,
        'team.name': [home, away, away, home],
        'opponentTeam.name': [away, home, home, away],
        'possession.team.name': [home, home, away, None],
        'player.name': ['a', 'b', 'c', 'd'],
    })
    if as_category:
        for c in team_rules.TEAM_NAME_COLUMNS:
            df[c] = df[c].astype('category')
    return df


def test_two_spellings_of_a_club_share_a_key():
    assert team_rules.team_name_key('Pevidém') == team_rules.team_name_key('Pevidem')
    assert team_rules.team_name_key('Mortágua ') == team_rules.team_name_key('Mortágua')
    assert team_rules.team_name_key('Sporting CP') != team_rules.team_name_key('Sporting CP II')
    assert team_rules.team_name_key('Atlético CP') != team_rules.team_name_key('Atlético')


def test_a_match_whose_names_agree_is_returned_as_it_is():
    """No copy and no change for every match but the 75: the page calls this on every run."""
    df = _match(away='Pevidem')
    assert team_rules.align_match_team_names(df, 'Montalegre', 'Pevidem') is df
    empty = df.iloc[0:0]
    assert team_rules.align_match_team_names(empty, 'Montalegre', 'Pevidem') is empty
    assert team_rules.align_match_team_names(None, 'Montalegre', 'Pevidem') is None


def test_a_club_spelt_differently_takes_the_match_lists_name():
    """Only the spelling changes, in every team-name column; the frame given is not written into."""
    for as_category in (False, True):
        df = _match(as_category=as_category)
        before = df.copy()
        out = team_rules.align_match_team_names(df, 'Montalegre', 'Pevidem')
        assert out is not df and len(out) == len(df)
        assert out['team.name'].tolist() == ['Montalegre', 'Pevidem', 'Pevidem', 'Montalegre']
        assert out['opponentTeam.name'].tolist() == ['Pevidem', 'Montalegre', 'Montalegre', 'Pevidem']
        assert out['possession.team.name'].tolist()[:3] == ['Montalegre', 'Montalegre', 'Pevidem']
        assert pd.isna(out['possession.team.name'].iloc[3])
        assert out['player.name'].tolist() == ['a', 'b', 'c', 'd']
        pd.testing.assert_frame_equal(df, before)                      # the shared frame is untouched
        assert (out['team.name'] == 'Pevidem').sum() == 2              # a plain comparison finds the side
    spaced = team_rules.align_match_team_names(_match(home='Gouveia', away='Mortágua '), 'Gouveia', 'Mortágua')
    assert set(spaced['team.name']) == {'Gouveia', 'Mortágua'}


def test_a_name_that_is_not_clearly_the_same_club_is_left_alone():
    # another club altogether: nothing in the events has the key
    df = _match(away='Fafe')
    assert team_rules.align_match_team_names(df, 'Montalegre', 'Pevidem') is df
    # two names in the events with the key: no guess
    assert team_rules.match_team_name_map(['Montalegre', 'Pevidém', 'PEVIDEM'], 'Montalegre', 'Pevidem') == {}
    # the other side of the match is never renamed, whatever its key
    assert team_rules.match_team_name_map(['Braga', 'Braga '], 'Braga', 'Braga ') == {}
    assert team_rules.match_team_name_map(['Montalegre', 'Pevidém'], None, float('nan')) == {}


def test_a_line_up_sheet_is_found_under_the_events_spelling():
    sheets = {'Montalegre': 'home sheet', 'Pevidém': 'away sheet'}
    assert team_rules.entry_for_team(sheets, 'Montalegre') == 'home sheet'
    assert team_rules.entry_for_team(sheets, 'Pevidem') == 'away sheet'
    assert team_rules.entry_for_team({'Mortágua ': 1, 'Eléctrico': 2}, 'Mortágua') == 1
    assert team_rules.entry_for_team(sheets, 'Fafe') is None
    assert team_rules.entry_for_team({}, 'Fafe') is None and team_rules.entry_for_team(None, 'Fafe') is None
    assert team_rules.entry_for_team(sheets, None) is None


def test_the_match_report_uses_the_one_spelling():
    """Both builders of a match report go through the rule: the stored tables
    (process_data) and the page's events and line-up sheets (the view)."""
    pdata, view = _source('process_data.py'), _source('views/match_analysis.py')
    assert 'match_df = team_rules.align_match_team_names(match_df, home_team, away_team)' in pdata
    assert 'match_events_df = team_rules.align_match_team_names(' in view
    assert 'team_rules.entry_for_team(match_lineup_data, home_team)' in view
    assert 'team_rules.entry_for_team(match_lineup_data, away_team)' in view
    assert 'match_lineup_data.get(' not in view


# One match of each kind: the events say 'Pevidém' (Montalegre v Pevidem, Liga 3
# 2021/22) and 'Mortágua ' with a trailing space (Mortágua v Eléctrico, Campeonato 2025/26).
MISMATCHED_MATCHES = (5263405, 5740512)


def _one_match_events(path, match_id, batch_size=20_000):
    """One match's events, read in small batches. The deploy check is
    memory-bound: a filtered pd.read_parquet decodes a whole row group of a
    million events (1.2 GB at its peak, measured 2026-11-14); this peaks at
    0.24 GB and returns the same frame."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    f = pq.ParquetFile(path)
    groups = []
    for i in range(f.metadata.num_row_groups):
        rg = f.metadata.row_group(i)
        # the column is looked up by its PATH: a position in the Arrow schema
        # is not a position among the file's leaf columns once one is nested
        st = next((rg.column(j).statistics for j in range(rg.num_columns)
                   if rg.column(j).path_in_schema == 'matchId'), None)
        if st is None or not st.has_min_max or st.min <= match_id <= st.max:
            groups.append(i)
    parts = []
    for batch in f.iter_batches(batch_size=batch_size, row_groups=groups, use_threads=False):
        keep = batch.filter(pc.equal(batch.column('matchId'), match_id))
        if keep.num_rows:
            parts.append(keep)
    return pa.Table.from_batches(parts).to_pandas() if parts else pd.DataFrame()


def test_a_club_spelt_differently_gets_its_match_tables(monkeypatch):
    """On the real events of two of the 75 matches: both columns are filled
    and neither side's PPDA reads 0. Until 2026-11-14 the club's column was
    all zeros and both PPDA cells read 0.0."""
    import pytest
    events_path = os.path.join(ROOT, 'raw_events.parquet')
    list_path = os.path.join(ROOT, 'matches_summary.parquet')
    if not (os.path.exists(events_path) and os.path.exists(list_path)):
        pytest.skip('needs raw_events.parquet and matches_summary.parquet')
    match_list = pd.read_parquet(list_path).set_index('matchId')
    for match_id in MISMATCHED_MATCHES:
        events = _one_match_events(events_path, match_id)
        # a finished season's match is in every copy of the events: an empty
        # read means the reader broke, and a skip here would hide it
        assert len(events) > 1000, match_id
        home, away = match_list.at[match_id, 'homeTeamName'], match_list.at[match_id, 'awayTeamName']
        out, version = _calculate_match_data(monkeypatch, events, home, away)
        assert out['stats_version'] == version, match_id
        general, defence = out['team_stats']['General'], out['team_stats']['Defence']
        assert list(general.columns) == [home, away]                      # still the match list's names
        for team in (home, away):
            assert general.at['Shots / on target', team] != '0/0', (match_id, team)
            assert float(defence.at['Passes allowed per def. action (PPDA)', team]) > 0, (match_id, team)
        assert len(out['player_stats']['home']) >= 11 and len(out['player_stats']['away']) >= 11, match_id


# --- the app's match list takes the events' spelling (season pages) ------------

def _list_and_events():
    matches = pd.DataFrame({
        'matchId':      [1, 2, 3, 4, 5, 6],
        'seasonId':     [10, 10, 10, 20, 20, 30],
        'homeTeamName': ['Montalegre', 'Pevidem', 'Fafe', 'Pevidém', 'Gouveia', 'Pevidem'],
        'awayTeamName': ['Pevidem', 'Fafe', 'Pevidem', 'Fafe', 'Mortágua', 'Fafe'],
        'status':       ['Played'] * 6,
    })
    events = pd.DataFrame({
        'matchId':   [1, 1, 2, 2, 4, 4, 5, 5, 6, 6],          # match 3 has no events
        'team.name': ['Montalegre', 'Pevidém', 'Pevidém', 'Fafe', 'Pevidém', 'Fafe',
                      'Gouveia', 'Mortágua ', 'Pevidem', 'Fafe'],
    })
    return matches, events


def test_the_match_list_takes_the_events_spelling():
    matches, events = _list_and_events()
    before = matches.copy()
    for as_category in (False, True):
        ev = events.copy()
        if as_category:
            ev['team.name'] = ev['team.name'].astype('category')
        out, renames = team_rules.match_list_in_events_spelling(matches, ev)
        pd.testing.assert_frame_equal(matches, before)                     # the frame given is untouched
        assert out['homeTeamName'].tolist() == ['Montalegre', 'Pevidém', 'Fafe', 'Pevidém', 'Gouveia', 'Pevidem']
        # match 3 has no events and follows its season (10); season 30 spells the
        # club 'Pevidem' in its events too, so nothing is changed there
        assert out['awayTeamName'].tolist() == ['Pevidém', 'Fafe', 'Pevidém', 'Fafe', 'Mortágua ', 'Fafe']
        assert renames == {1: {'Pevidem': 'Pevidém'}, 2: {'Pevidem': 'Pevidém'}, 3: {'Pevidem': 'Pevidém'},
                           5: {'Mortágua': 'Mortágua '}}
        again, none = team_rules.match_list_in_events_spelling(out, ev)
        assert again is out and none == {}                                 # a second pass changes nothing


def test_a_match_without_events_is_renamed_only_when_its_season_is_clear():
    matches, events = _list_and_events()
    # season 10 now also has a match whose events spell the club as the list does
    matches.loc[len(matches)] = [7, 10, 'Pevidem', 'Anadia', 'Played']
    events = pd.concat([events, pd.DataFrame({'matchId': [7, 7], 'team.name': ['Pevidem', 'Anadia']})],
                       ignore_index=True)
    out, renames = team_rules.match_list_in_events_spelling(matches, events)
    assert 3 not in renames and out.set_index('matchId').at[3, 'awayTeamName'] == 'Pevidem'
    assert renames[1] == {'Pevidem': 'Pevidém'}                           # a match WITH events is still its own proof
    # nothing to do: the frame itself comes back
    same, none = team_rules.match_list_in_events_spelling(out, events)
    assert same is out and none == {}
    assert team_rules.match_list_in_events_spelling(None, events) == (None, {})
    empty_events = events.iloc[0:0]
    assert team_rules.match_list_in_events_spelling(matches, empty_events)[0] is matches


def test_a_club_missing_from_its_matchs_events_follows_its_season():
    """Events that hold only the OTHER side: the match itself proves nothing
    about the club's spelling, the season does."""
    matches, events = _list_and_events()
    matches.loc[len(matches)] = [8, 10, 'Pevidem', 'Anadia', 'Played']
    events = pd.concat([events, pd.DataFrame({'matchId': [8], 'team.name': ['Anadia']})], ignore_index=True)
    out, renames = team_rules.match_list_in_events_spelling(matches, events)
    assert renames[8] == {'Pevidem': 'Pevidém'}
    row = out.set_index('matchId').loc[8]
    assert (row['homeTeamName'], row['awayTeamName']) == ('Pevidém', 'Anadia')


def test_the_stored_match_tables_follow_the_renamed_list():
    table = pd.DataFrame({'Montalegre': [7, 5.6], 'Pevidem': [9, 5.0]}, index=['Shots', 'PPDA'])
    untouched = pd.DataFrame({'Fafe': [1], 'Anadia': [2]}, index=['Shots'])
    data = {1: {'team_stats': {'General': table, 'Defence': table.copy()},
                'player_stats': {'home': pd.DataFrame(), 'away': pd.DataFrame()}, 'stats_version': 3},
            2: {'team_stats': {'General': untouched}},
            3: {'team_stats': {}}, 4: None}
    n = team_rules.relabel_match_tables(data, {1: {'Pevidem': 'Pevidém'}, 3: {'Pevidem': 'Pevidém'}, 9: {'A': 'B'}})
    assert n == 1
    assert list(data[1]['team_stats']['General'].columns) == ['Montalegre', 'Pevidém']
    assert list(data[1]['team_stats']['Defence'].columns) == ['Montalegre', 'Pevidém']
    assert data[1]['team_stats']['General']['Pevidém'].tolist() == [9, 5.0]
    assert list(table.columns) == ['Montalegre', 'Pevidem']               # the old frame was not written into
    assert data[2]['team_stats']['General'] is untouched and data[1]['stats_version'] == 3
    assert team_rules.relabel_match_tables(None, {1: {'a': 'b'}}) == 0 and team_rules.relabel_match_tables(data, {}) == 0


def test_the_app_loads_one_spelling_and_its_name_keyed_caches_know_it():
    app = _source('app.py')
    assert 'team_rules.match_list_in_events_spelling(' in app
    assert 'team_rules.relabel_match_tables(all_match_data, _club_renames)' in app
    assert app.count("'teams': _season_team_names_fp(") == 2


def test_the_real_match_list_agrees_with_the_events_after_the_rule():
    """On the shipped files: the two clubs take the events' spelling, and
    every match with events then names its two sides as its events do.
    (The stored tables are not loaded here: 0.5 GB, and the check is
    memory-bound. Their relabelling is tested above on small frames.)"""
    import warnings
    import pytest
    import pyarrow.parquet as pq
    paths = [os.path.join(ROOT, f) for f in ('raw_events.parquet', 'matches_summary.parquet')]
    if not all(os.path.exists(p) for p in paths):
        pytest.skip('needs raw_events.parquet and matches_summary.parquet')
    # two columns, the names as a dictionary column
    events = pq.read_table(paths[0], columns=['matchId', 'team.name'], read_dictionary=['team.name']).to_pandas()
    matches = pd.read_parquet(paths[1])
    out, renames = team_rules.match_list_in_events_spelling(matches, events)

    def listed(frame, season, name):
        return int(((frame['seasonId'] == season)
                    & ((frame['homeTeamName'] == name) | (frame['awayTeamName'] == name))).sum())

    for season, old, new in ((188221, 'Pevidem', 'Pevidém'), (190230, 'Mortágua', 'Mortágua '),
                             (191779, 'Mortágua', 'Mortágua ')):
        n_before = listed(matches, season, old)
        if n_before:                      # a match list already in the events' spelling needs nothing
            assert listed(out, season, old) == 0 and listed(out, season, new) >= n_before, (season, old)
    pairs = events.drop_duplicates()
    del events
    names = {}
    for mid, name in zip(pairs['matchId'].tolist(), pairs['team.name'].astype(str).tolist()):
        names.setdefault(mid, set()).add(name)
    left = [mid for mid, h, a in zip(out['matchId'], out['homeTeamName'], out['awayTeamName'])
            if mid in names and len(names[mid]) == 2 and {h, a} != names[mid]]
    if left:                              # a NEW spelling: reported, not a reason to block every deploy
        warnings.warn(f'{len(left)} matches still name a club differently from their events: {left[:5]}')
    assert len(left) <= 5
    assert set(renames) <= set(out['matchId'])


def test_a_bold_caption_never_ends_in_an_untrimmed_team_name():
    """Markdown does not close `**` after a space, and the events spell one
    club with a trailing space ('Mortágua '): its captions printed as
    "**Shots FOR Mortágua **" with the asterisks showing (2026-11-15). A team
    name that sits right before the closing `**` is trimmed."""
    import re
    raw = re.compile(r"\{[^{}]*(team|opponent)[^{}]*\}\*\*", re.IGNORECASE)
    for path in ('views/team_analysis.py', 'opposition_report.py', 'views/match_analysis.py',
                 'views/player_profile.py'):
        for n, line in enumerate(_source(path).splitlines(), 1):
            for m in raw.finditer(line):
                assert '.strip()' in m.group(0), f'{path}:{n}: {m.group(0)}'


# --- the nightly job's season corner tables (season_team_stats.pkl) -------------

def _corner(season, match, team, possession, xg=0.1):
    """One corner possession: the corner, then a shot from it."""
    base = {'seasonId': season, 'matchId': match, 'team.name': team, 'possession.id': possession,
            'pass.endLocation.x': None, 'pass.endLocation.y': None, 'shot.xg': None, 'shot.isGoal': None}
    return [dict(base, **{'type.primary': 'corner', 'possession.eventIndex': 0, 'location.x': 100.0, 'location.y': 0.0,
                          'pass.endLocation.x': 95.0, 'pass.endLocation.y': 50.0}),
            dict(base, **{'type.primary': 'shot', 'possession.eventIndex': 1, 'location.x': 95.0, 'location.y': 50.0,
                          'shot.xg': xg, 'shot.isGoal': False})]


def test_a_club_that_took_corners_and_has_no_table_is_reported():
    matches = pd.DataFrame({'homeTeamName': ['Montalegre', 'Fafe'], 'awayTeamName': ['Pevidém', 'Anadia']})
    corners = pd.DataFrame({'team.name': ['Montalegre', 'Pevidém', 'Fafe', 'Anadia', 'Elsewhere'],
                            'possession.id': [1, 2, 3, None, 5], 'possession.eventIndex': [0, 0, 0, 0, 0]})
    have = {'Montalegre': {'corners': 'table'}}
    # Anadia's only corner cannot be used (no possession): it can never get a table, so it is not asked for
    assert team_rules.clubs_missing_corner_entry(matches, corners, have) == ['Fafe', 'Pevidém']
    # nor is a club whose only corner has no place in its possession (the table needs the next event)
    no_index = corners.assign(**{'possession.eventIndex': [0, 0, None, 0, 0]})
    assert team_rules.clubs_missing_corner_entry(matches, no_index, have) == ['Pevidém']
    assert team_rules.clubs_missing_corner_entry(matches, corners, dict(have, Fafe={}, **{'Pevidém': {}})) == []
    assert team_rules.clubs_missing_corner_entry(matches, corners, None) == ['Fafe', 'Montalegre', 'Pevidém']
    assert team_rules.clubs_missing_corner_entry(matches, corners.iloc[0:0], have) == []
    assert team_rules.clubs_missing_corner_entry(matches.iloc[0:0], corners, have) == []
    assert team_rules.clubs_missing_corner_entry(matches, corners[['team.name']], have) == []


def test_the_job_fills_a_finished_season_that_lacks_a_club(monkeypatch):
    """Season 10 is finished and its file lacks the club the match list spells
    differently: it is computed again, under the events' spelling. Season 20
    is finished and complete: not touched. Season 30 is in progress: always
    computed."""
    import contextlib
    import io
    events = pd.DataFrame(
        _corner(10, 1, 'Montalegre', 101) + _corner(10, 1, 'Pevidém', 102, xg=0.3) + _corner(10, 1, 'Pevidém', 103)
        + _corner(20, 2, 'Fafe', 201) + [dict(_corner(20, 2, 'Anadia', 202)[1], **{'type.primary': 'pass'})]
        + _corner(30, 3, 'Braga', 301) + _corner(30, 3, 'Amora', 302)
        + [dict(_corner(30, 4, 'Canelas', 303)[1], **{'type.primary': 'pass'})])     # Canelas took no corner
    matches = pd.DataFrame({'matchId': [1, 2, 3, 4], 'seasonId': [10, 20, 30, 30],
                            'homeTeamName': ['Montalegre', 'Fafe', 'Braga', 'Braga'],
                            'awayTeamName': ['Pevidem', 'Anadia', 'Amora', 'Canelas'], 'status': ['Played'] * 4})
    untouched = {'Fafe': {'corners': 'the table computed once'}}        # Anadia took no corner: nothing is missing
    stored = {10: {'Montalegre': {'corners': 'an old table'}}, 20: untouched}
    events_before, matches_before = events.copy(), matches.copy()

    _, version = _calculate_match_data(monkeypatch, pd.DataFrame({'matchId': [1], 'team.name': ['A']}), 'A', 'B')
    import process_data                                               # importable now (see _calculate_match_data)
    chained = pd.get_option('mode.chained_assignment')
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            out, computed = process_data.compute_season_team_stats(stored, events, matches, [30], [10, 20])
    finally:
        pd.set_option('mode.chained_assignment', chained)
    assert computed == [30, 10]
    assert out[20] is untouched
    assert sorted(out[10]) == ['Montalegre', 'Pevidém'] and sorted(out[30]) == ['Amora', 'Braga']
    # a club with no table gets NO entry, not an empty one: the app merges seasons
    # with dict.update, and an empty entry of a later season would hide an earlier table
    assert 'Canelas' not in out[30] and all(entry.get('corners') is not None for entry in out[30].values())
    table = out[10]['Pevidém']['corners']
    assert list(table.columns) == ['Pevidém']                         # the name the page asks for
    assert table.at['Total Corners', 'Pevidém'] == 2 and table.at['Total xG', 'Pevidém'] == '0.400'
    assert table.at['Corners/Match', 'Pevidém'] == '2.0'              # its ONE match in the list is found
    pd.testing.assert_frame_equal(events, events_before)              # the events given are not written into
    pd.testing.assert_frame_equal(matches, matches_before)            # nor is the stored match list
    # a second run finds nothing to fill
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        again, computed = process_data.compute_season_team_stats(out, events, matches, [], [10, 20])
    assert computed == [] and again[10] is out[10]


def test_the_job_builds_the_corner_tables_through_the_one_step():
    pdata = _source('process_data.py')
    assert 'season_team_stats, _recomputed = compute_season_team_stats(' in pdata
    assert 'team_rules.clubs_missing_corner_entry(' in pdata
    assert 'calculate_team_corner_stats(season_events.copy(), season_matches, team)' not in pdata



# ---------------------------------------------------------------------------
# League Analysis: one number per metric name (2026-11-16)
# ---------------------------------------------------------------------------
# The page joins the radar numbers, the set-piece table and the match-table
# averages by club. pandas kept a repeated name twice, as "xG_x" / "xG_y":
# two different numbers, and plain "xG" was not offered. Lucas: the radar
# version only.

def test_join_club_tables_keeps_the_first_table_that_has_a_name():
    import pandas as pd
    radar = pd.DataFrame({'xG': [1.17, 0.9], 'Goals': [1.84, 1.0]}, index=['A', 'B'])
    set_pieces = pd.DataFrame({'Corners': [5.0, 3.0], 'xG from Corners': [0.2, 0.1]}, index=['A', 'B'])
    match_tables = pd.DataFrame({'xG': [1.25, 0.95], 'Corners': [4.0, 3.5], 'Offsides': [2.0, 1.0]},
                                index=['A', 'B'])
    before = [t.copy() for t in (radar, set_pieces, match_tables)]
    got = team_rules.join_club_tables(radar, set_pieces, match_tables)
    assert list(got.columns) == ['xG', 'Goals', 'Corners', 'xG from Corners', 'Offsides']
    assert got.loc['A', 'xG'] == 1.17 and got.loc['A', 'Corners'] == 5.0 and got.loc['B', 'Offsides'] == 1.0
    assert not [c for c in got.columns if c.endswith('_x') or c.endswith('_y')]
    for t, b in zip((radar, set_pieces, match_tables), before):
        pd.testing.assert_frame_equal(t, b)          # the tables given are not changed


def test_join_club_tables_fills_missing_clubs_and_skips_missing_tables():
    import pandas as pd
    radar = pd.DataFrame({'xG': [1.0]}, index=['A'])
    other = pd.DataFrame({'Offsides': [2.0]}, index=['B'])
    got = team_rules.join_club_tables(radar, None, other)
    assert set(got.index) == {'A', 'B'}
    assert got.loc['B', 'xG'] == 0 and got.loc['A', 'Offsides'] == 0
    assert team_rules.join_club_tables(None, None).empty
    pd.testing.assert_frame_equal(team_rules.join_club_tables(radar, None), radar)


def test_league_analysis_joins_its_tables_through_the_rule():
    src = open(os.path.join(ROOT, 'views', 'league_analysis.py'), encoding='utf-8').read()
    assert 'team_rules.join_club_tables(' in src
    assert 'pd.merge(stats_df_raw' not in src and 'pd.merge(combined_stats_df' not in src
