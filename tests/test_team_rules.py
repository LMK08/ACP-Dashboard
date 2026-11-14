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


def test_a_match_that_could_not_be_computed_is_not_stamped_as_done(monkeypatch):
    """calculate_match_data swallows its own errors and returns empty tables.
    Such a result carries no stamp, so the cached tables are kept and the
    next run tries again (process_data.main)."""
    import contextlib
    import io
    import sys
    import types
    # process_data imports python-dotenv at the top, for its main(). Only the
    # nightly job installs that package: the Space and the deploy check do
    # not (this test failed the check for that reason, 2026-11-14). The
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
        broken = pd.DataFrame({'matchId': [1], 'team.name': ['A']})          # nothing a match needs
        with contextlib.redirect_stdout(io.StringIO()):
            out = process_data.calculate_match_data(broken, 'A', 'B')
    finally:
        pd.set_option('mode.chained_assignment', _chained)
    assert out['stats_version'] is None
    assert not any(isinstance(df, pd.DataFrame) and not df.empty for df in out['team_stats'].values())
