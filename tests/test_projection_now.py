"""Today's projection on the boards (2026-11-12): models/ratings/projection_now
on synthetic engine frames, and the scored stats frame keeping the engine
rating / projection blank instead of 0 (the 2026/27 role board ranked every
player on a tie of zero projections).
"""
import os
import sys
import re
import types

import numpy as np
import pandas as pd
import pytest

DASH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, DASH)

from models.ratings import projection_now as pn  # noqa: E402

import league_config as lc  # noqa: E402

# the live season and the latest completed one of Liga 3, from the config, so
# the tests follow the season rollover
L3 = 43324
L3_26 = lc.COMPETITIONS[L3]['current_season']
L3_25 = max((int(s) for s in lc.COMPETITIONS[L3]['seasons'] if int(s) != L3_26), key=lc.SEASON_YEAR.get)


def _engine():
    # A: summer + live (valued)      -> live forward
    # B: summer + live (not valued)  -> summer
    # C: newcomer, live (valued)     -> live forward
    # D: newcomer, live (not valued) -> none
    # E: winter mover, two valued live rows -> the higher-minutes row
    # F: summer only (left the league) -> summer
    rows = [
        # pid, season, mins, projection, projection_abs, band, live_ok, fwd, fwd_abs
        (1, L3_25, 2000, 60.0, 60.0, 5.0, False, np.nan, np.nan),
        (1, L3_26, 600, np.nan, np.nan, np.nan, True, 62.0, 62.0),
        (2, L3_25, 1500, 55.0, 55.0, 6.0, False, np.nan, np.nan),
        (2, L3_26, 300, np.nan, np.nan, np.nan, False, 58.0, 58.0),
        (3, L3_26, 400, np.nan, np.nan, np.nan, True, 50.0, 50.0),
        (4, L3_26, 200, np.nan, np.nan, np.nan, False, 47.0, 47.0),
        (5, L3_26, 100, np.nan, np.nan, np.nan, True, 40.0, 40.0),
        (5, L3_26 + 1, 500, np.nan, np.nan, np.nan, True, 45.0, 45.0),
        (6, L3_25, 900, 52.0, 51.0, 7.0, False, np.nan, np.nan),
    ]
    df = pd.DataFrame(rows, columns=['playerId', 'seasonId', 'mins_played', 'projection', 'projection_abs',
                                     'band_sd', 'live_ok', 'fwd', 'fwd_abs'])
    df['playerId'] = df['playerId'].astype('Int64')
    df['seasonId'] = df['seasonId'].astype('Int64')
    return df


def _pick(df):
    # the app's row pick (load_player_engine / engine_rows_for_scope(None))
    return (df.assign(_hp=df['projection'].notna())
              .sort_values(['playerId', '_hp', 'seasonId', 'mins_played'])
              .drop_duplicates('playerId', keep='last').set_index('playerId'))


def _now(df):
    out = pn.attach(df, _pick(df))
    return out.groupby('playerId').agg(proj=('projection_now', 'first'), abs_=('projection_now_abs', 'first'),
                                       band=('band_now', 'first'), live=('projection_now_live', 'first'),
                                       n_values=('projection_now', 'nunique'))


def test_live_forward_once_valued_else_summer():
    g = _now(_engine())
    assert g.loc[1, 'proj'] == 62.0 and g.loc[1, 'live'] and np.isnan(g.loc[1, 'band'])
    assert g.loc[2, 'proj'] == 55.0 and not g.loc[2, 'live'] and g.loc[2, 'band'] == 6.0
    assert g.loc[3, 'proj'] == 50.0 and g.loc[3, 'live']
    assert np.isnan(g.loc[4, 'proj']) and not g.loc[4, 'live']
    assert g.loc[5, 'proj'] == 45.0                      # the higher-minutes live row
    assert g.loc[6, 'proj'] == 52.0 and g.loc[6, 'abs_'] == 51.0 and not g.loc[6, 'live']


def test_every_row_of_a_player_carries_the_same_value():
    g = _now(_engine())
    assert (g['n_values'] <= 1).all()


def test_engine_without_forward_columns_is_unchanged():
    old = _engine().drop(columns=['fwd', 'fwd_abs'])
    out = pn.attach(old, _pick(old))
    assert not set(pn.NOW_COLS) & set(out.columns)
    pd.testing.assert_frame_equal(out, old)


def test_scopes_that_show_todays_projection():
    live = frozenset({L3_26})
    assert pn.includes_live_season(None, live)
    assert pn.includes_live_season(L3_26, live)
    assert pn.includes_live_season([L3_25, L3_26], live)
    assert not pn.includes_live_season(L3_25, live)
    assert not pn.includes_live_season([L3_25], live)


# ---- the scored stats frame keeps the engine rating / projection blank -------

@pytest.fixture(scope='module')
def app():
    """app.py up to the data load (functions + constants only)."""
    src = open(os.path.join(DASH, 'app.py'), encoding='utf-8').read()
    src = src[:src.index('# --- Load Data ---')]
    mod = types.ModuleType('app_under_test')
    mod.__file__ = os.path.join(DASH, 'app.py')
    cwd = os.getcwd()
    os.chdir(DASH)
    try:
        exec(compile(src, 'app.py', 'exec'), mod.__dict__)
    finally:
        os.chdir(cwd)
    return mod


def test_scored_frame_keeps_engine_columns_nan(app):
    df = pd.DataFrame({
        'playerId': [1, 2, 3], 'playerName': ['a', 'b', 'c'], 'teamName': ['x', 'x', 'y'],
        'primaryPosition': ['CF', 'CF', 'CF'], 'totalMinutes': [900, 800, 700],
        'ACP Rating': [55.0, np.nan, 48.0], 'ACP Rating (abs)': [55.0, np.nan, 48.0],
        'ACP Projection': [np.nan, np.nan, 50.0], 'ACP Projection (abs)': [np.nan, np.nan, 50.0],
        'Projection Band': [np.nan] * 3, 'Evidence Weight': [0.4, np.nan, 0.6],
        'Goals': [np.nan, 1.0, 0.5],
    })
    out = app.calculate_player_percentiles_and_scores(
        df, app.POSITION_GROUPS, app.WEIGHTS, app.INVERT_METRICS, min_minutes=500,
        season_id=-12345, disk_cache=False)
    out = out.set_index('playerId')
    for c in app.ENGINE_NAN_COLUMNS:
        pd.testing.assert_series_equal(out[c].astype(float), df.set_index('playerId')[c].astype(float),
                                       check_names=False)
    assert out.loc[1, 'Goals'] == 0                     # other columns still fill with 0


def test_latest_completed_and_headline_scopes():
    comps = {1: {'seasons': {10: 'a', 11: 'b', 12: 'c'}}, 2: {'seasons': {20: 'x', 21: 'y'}}}
    year = {10: 2023, 11: 2024, 12: 2025, 20: 2023, 21: 2024}
    live = frozenset({12})                      # comp 2 is inactive: its newest season is complete
    assert pn.latest_completed_seasons(comps, live, year) == {11, 21}
    latest = {11, 21}
    assert pn.projection_is_headline(None, live, latest)
    assert pn.projection_is_headline(12, live, latest)          # today's projection
    assert pn.projection_is_headline([11], live, latest)        # summer projection covers it
    assert pn.projection_is_headline([11, 21], live, latest)
    assert not pn.projection_is_headline(10, live, latest)      # older: only players last seen then
    assert not pn.projection_is_headline([10, 11], live, latest)


# ---- the merge: which projection a scope shows -------------------------------

def _app_engine(df):
    """_engine() plus every column merge_engine_values_into_stats reads, and
    load_player_engine's value / projection_now columns."""
    src = open(os.path.join(DASH, 'app.py'), encoding='utf-8').read()
    body = src[src.index('def merge_engine_values_into_stats'):]
    body = body[:body.index('\ndef ', 10)]
    need = set(re.findall(r"e\['([^']+)'\]", body))
    need -= set(pn.NOW_COLS)        # those come from attach() only (absent on an old engine)
    out = df.copy()
    for c in need - set(out.columns):
        out[c] = np.nan
    out['acp_rating'] = out['acp_rating_abs'] = 50.0
    out['engine_value_eur_now'] = np.nan
    out['value_w_now'] = out['value_mins_now'] = np.nan
    out['value_label_now'] = ''
    return out


def _merged(app, monkeypatch, eng, scope):
    eng = pn.attach(eng, _pick(eng))
    monkeypatch.setattr(app, 'load_player_engine', lambda: (eng, {}))
    stats = pd.DataFrame({'playerId': [1, 2, 3, 4, 6]})
    return app.merge_engine_values_into_stats(stats, scope).set_index('playerId')['ACP Projection (abs)']


def test_merge_live_and_all_seasons_show_todays_projection(app, monkeypatch):
    eng = _app_engine(_engine())
    live = _merged(app, monkeypatch, eng, L3_26)
    assert live[1] == 62.0 and live[2] == 55.0 and live[3] == 50.0 and np.isnan(live[4])
    allp = _merged(app, monkeypatch, eng, None)
    assert allp[1] == 62.0 and allp[6] == 51.0
    assert app.scope_shows_todays_projection(L3_26) and app.scope_shows_todays_projection(None)


def test_merge_past_season_keeps_its_own_projection(app, monkeypatch):
    past = _merged(app, monkeypatch, _app_engine(_engine()), L3_25)
    assert past[1] == 60.0 and past[2] == 55.0 and past[6] == 51.0
    assert not app.scope_shows_todays_projection(L3_25)


def test_old_engine_never_shows_a_summer_projection_as_live(app, monkeypatch):
    old = _app_engine(_engine().drop(columns=['fwd', 'fwd_abs']))
    live = _merged(app, monkeypatch, old, L3_26)
    assert live.isna().all()                    # live rows carry no projection
    assert not app.scope_shows_todays_projection(L3_26)
    assert not app.scope_shows_todays_projection(None)   # the caption's 'live' wording is off


# ---- the profile / comparison card (card_scope.card_row) --------------------

def _card_engine(with_fwd=True):
    df = _engine()
    if not with_fwd:
        df = df.drop(columns=['fwd', 'fwd_abs'])
    df['acp_rating'] = [58.0, 61.0, 54.0, 50.0, 49.0, 45.0, 44.0, 46.0, 52.0]
    df['acp_rating_abs'] = df['acp_rating']
    df['w_evidence'] = [0.6, np.nan, 0.5, np.nan, np.nan, np.nan, np.nan, np.nan, 0.4]
    df['asof_w_evidence'] = [0.6, 0.7, 0.5, 0.55, 0.2, 0.1, 0.1, 0.15, 0.4]
    df['proj_delta'] = df['projection'] - df['acp_rating']
    df['seasons_ago'] = np.where(df['projection'].notna(), 0, np.nan)
    df['league'] = 'L3'
    return pn.attach(df, _pick(df))


def test_card_shows_the_live_projection_once_valued():
    from models.ratings.card_scope import card_row
    eng = _card_engine()
    e = card_row(eng, 1, L3_26)['e']
    assert e['projection'] == 62.0 and e['projection_live'] and np.isnan(e['band_sd'])
    assert e['w_evidence'] == 0.7                       # the evidence behind the live number
    assert e['proj_delta'] == 62.0 - 61.0
    assert card_row(eng, 1, L3_25)['e']['projection'] == 62.0   # one forward look, any scope
    e2 = card_row(eng, 2, L3_26)['e']
    assert e2['projection'] == 55.0 and not e2['projection_live'] and e2['band_sd'] == 6.0


def test_card_on_an_old_engine_keeps_the_summer_projection():
    from models.ratings.card_scope import card_row
    e = card_row(_card_engine(with_fwd=False), 1, L3_26)['e']
    assert e['projection'] == 60.0 and not e['projection_live'] and e['band_sd'] == 5.0


# ---- the band and the fan chart (2026-11-12: "switch the fan chart") ---------

def test_live_band_is_the_forward_band_when_the_engine_has_it():
    df = _engine()
    df['fwd_band'] = np.where(df['fwd'].notna(), 9.0, np.nan)
    g = pn.attach(df, _pick(df)).groupby('playerId')['band_now'].first()
    assert g[1] == 9.0 and g[3] == 9.0          # live forward
    assert g[2] == 6.0                          # not valued yet: the summer band


def _fan_rows(live=True):
    import pitch_interactive as pi  # noqa: F401 (import check)
    labels = {101: '2024/25', 102: '2025/26', 103: '2026/27'}
    df = pd.DataFrame({
        'playerId': 7, 'seasonId': [101, 102, 103], 'league': 'L3',
        'acp_rating': [55.0, 60.0, 64.0], 'acp_rating_abs': [55.0, 60.0, 64.0],
        'mins_played': [2000.0, 2100.0, 700.0],
        'projection': [np.nan, 61.0, np.nan], 'projection_abs': [np.nan, 61.0, np.nan],
        'band_sd': [np.nan, 8.0, np.nan], 'w_evidence': [np.nan, 0.6, np.nan],
        'role': 'Striker', 'age': [np.nan, 24.5, np.nan],
        'live_ok': [False, False, live], 'fwd': [np.nan, np.nan, 63.0],
        'fwd_abs': [np.nan, np.nan, 63.0], 'fwd_band': [np.nan, np.nan, 9.0],
        'asof_age': [np.nan, np.nan, 25.3], 'asof_w_evidence': [np.nan, np.nan, 0.66],
    })
    return pn.attach(df, _pick(df)), labels


def test_fan_plots_the_live_projection_half_a_season_on():
    import pitch_interactive as pi
    rows, labels = _fan_rows(live=True)
    ser = pi.projection_series(rows, labels)
    assert ser['live'] and ser['proj'] == 63.0 and ser['band'] == 9.0
    assert ser['x_last'] == 2026 and ser['x_proj'] == 2026.5
    assert ser['age_now'] == 25.3 and ser['w_ev'] == 0.66
    fig = pi.plotly_projection_fan(rows, labels, 'P')
    assert 'next 12 mo (proj)' in ''.join(map(str, fig.layout.xaxis.ticktext))
    diamond = [t for t in fig.data if t.name == 'projection'][0]
    assert list(diamond.x) == [2026.5] and list(diamond.y) == [63.0]


def test_fan_before_the_live_switch_keeps_the_summer_projection():
    import pitch_interactive as pi
    rows, labels = _fan_rows(live=False)
    ser = pi.projection_series(rows, labels)
    assert not ser['live'] and ser['proj'] == 61.0 and ser['band'] == 8.0 and ser['x_proj'] == 2027


# ---- the card's 'Career (as-of)' (2026-11-12: "Yes, fix it") ----------------

def test_card_career_asof_is_filled_for_every_season():
    from models.ratings.card_scope import card_row
    eng = _card_engine()
    # the engine's own career_asof sits on projection rows only; asof_career on every row
    eng['career_asof'] = np.where(eng['projection'].notna(), 57.0, np.nan)
    eng['asof_career'] = np.where(eng['projection'].notna(), 57.0, 59.5)
    assert card_row(eng, 1, L3_26)['e']['career_asof'] == 59.5     # live season: was blank
    assert card_row(eng, 1, L3_25)['e']['career_asof'] == 57.0     # its own season's career
    assert card_row(eng, 1, None)['e']['career_asof'] == 59.5      # All Seasons: through the latest season
    old = eng.drop(columns=['asof_career'])                          # engine before the rebuild
    assert np.isnan(card_row(old, 1, L3_26)['e']['career_asof'])
