"""Headless page walk of the dashboard through Streamlit's AppTest.

Runs app.py without a browser, opens every analysis type on its default
selections, then forces the newest fixture-only season (no events yet) on the
pages that used to crash there. A page FAILS when it raises an uncaught
exception or renders an ``st.error`` box — the two ways a user sees a broken
page.

Needs the real data files next to app.py (raw_events.parquet is the big one,
HF-only; CI downloads it before running). Skips cleanly without them so the
suite can run on a code-only checkout.

Run:  python -m pytest tests/test_smoke_pages.py -v      (about 2 minutes)
"""
import os
import sys

import pytest

DASHBOARD_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REQUIRED_DATA = ['raw_events.parquet', 'matches_summary.parquet',
                 'all_match_data.pkl', 'player_details.pkl']

PAGES = ['Home', 'Match Analysis', 'Team Analysis', 'League Analysis', 'Player Profile',
         'Player Comparison', 'Player Analysis', 'Match Predictor', 'Shadow Team',
         'Opposition Report']
# Pages that read the player-stats engine and used to raise KeyError
# 'next_shot_id' on a season with fixtures but no events.
EMPTY_SEASON_PAGES = ['Player Profile', 'Player Comparison', 'Player Analysis',
                      'Opposition Report', 'Team Analysis']

pytestmark = pytest.mark.skipif(
    any(not os.path.exists(os.path.join(DASHBOARD_DIR, f)) for f in REQUIRED_DATA),
    reason="dashboard data files not present (raw_events.parquet is HF-only)")


@pytest.fixture(scope='module')
def app():
    """One booted AppTest shared by every test in this module (boot loads
    ~4.7M events; doing it per test would take minutes)."""
    from streamlit.testing.v1 import AppTest
    cwd = os.getcwd()
    os.chdir(DASHBOARD_DIR)
    sys.path.insert(0, DASHBOARD_DIR)
    try:
        at = AppTest.from_file('app.py', default_timeout=1200)
        at.run()
        yield at
    finally:
        os.chdir(cwd)


def _problems(at):
    return ([f"exception: {str(e.value)[:300]}" for e in at.exception]
            + [f"st.error: {str(e.value)[:200]}" for e in at.error])


def _nav_radio(at, page):
    """The sidebar navigation is one radio per group (navigation.NAV_GROUPS)."""
    return next(r for r in at.sidebar.radio if page in r.options)


def _open_page(at, page):
    _nav_radio(at, page).set_value(page).run()
    at.run()  # some pages rerun once on first visit (nav / session seeding)
    assert _nav_radio(at, page).value == page


def _newest_season_label(at):
    boxes = [s for s in at.sidebar.selectbox if s.label == 'Season']
    assert boxes, "no Season selector on this page"
    labels = [o for o in boxes[0].options if o != 'All Seasons']
    return boxes[0], labels[0]


def test_landing_page_renders_clean(app):
    assert not _problems(app), _problems(app)


@pytest.mark.parametrize('page', PAGES)
def test_page_renders_on_defaults(app, page):
    _open_page(app, page)
    assert not _problems(app), _problems(app)


def test_default_season_has_event_data(app):
    """The default season must be one the events file actually covers, not
    merely the newest label in the fixture list. Checked against the parquet
    directly (importing app.py here would boot a second copy of the app)."""
    import pyarrow.parquet as pq
    import league_config
    _open_page(app, 'Team Analysis')
    season_box, newest = _newest_season_label(app)
    table = pq.read_table(os.path.join(DASHBOARD_DIR, 'raw_events.parquet'),
                          columns=['seasonId', 'matchId']).to_pandas()
    counts = table.groupby('seasonId', observed=True)['matchId'].nunique().to_dict()
    chosen_ids = [sid for sid, lab in league_config.all_season_id_map().items() if lab == season_box.value]
    min_matches = 5  # app.MIN_MATCHES_FOR_DEFAULT_SEASON
    assert any(counts.get(sid, 0) >= min_matches for sid in chosen_ids), (
        f"default season {season_box.value} has too few matches with events: "
        f"{[counts.get(s, 0) for s in chosen_ids]}")


def test_club_first_defaults(app):
    _open_page(app, 'Team Analysis')
    team = [s for s in app.sidebar.selectbox if s.label == 'Select a Team'][0]
    assert team.value == 'Atlético CP'
    _open_page(app, 'Player Profile')
    player = [s for s in app.sidebar.selectbox if s.label == 'Select Player:'][0]
    assert 'Atlético CP' in player.value


def test_context_persists_across_pages(app):
    """League/season are chosen once in the context bar. The choice must
    survive pages that don't draw the bar (Home) or the season (Match
    Predictor) — Streamlit drops widget state for widgets absent from a run."""
    _open_page(app, 'Team Analysis')
    season_box, _ = _newest_season_label(app)
    options = [o for o in season_box.options if o != 'All Seasons']
    assert len(options) >= 2
    chosen = options[1]  # not the default
    season_box.set_value(chosen)
    app.run()
    for page in ('League Analysis', 'Match Predictor', 'Home', 'Player Profile', 'Team Analysis'):
        _open_page(app, page)
        boxes = [s for s in app.sidebar.selectbox if s.label == 'Season']
        if boxes:
            assert boxes[0].value == chosen, f"{page}: season reset to {boxes[0].value}"
    # "All Seasons" stays selected everywhere (stable widget options); a page
    # that can't show it resolves to one season and says so, without erroring.
    _open_page(app, 'Player Profile')
    [s for s in app.sidebar.selectbox if s.label == 'Season'][0].set_value('All Seasons')
    app.run()
    _open_page(app, 'Team Analysis')
    assert [s for s in app.sidebar.selectbox if s.label == 'Season'][0].value == 'All Seasons'
    assert any('one season at a time' in str(c.value) for c in app.sidebar.caption)
    assert any('Team Report' in str(h.value) for h in app.header)
    assert not _problems(app), _problems(app)
    # Switching league swaps to that league's own season key: a valid season
    # is selected, the page renders, and switching back restores the choice.
    league_box = [s for s in app.sidebar.selectbox if s.label == 'League'][0]
    league_box.set_value('Campeonato')
    app.run()
    season_camp = [s for s in app.sidebar.selectbox if s.label == 'Season'][0]
    assert season_camp.value in season_camp.options
    assert not _problems(app), _problems(app)
    [s for s in app.sidebar.selectbox if s.label == 'League'][0].set_value('Liga 3')
    app.run()
    assert [s for s in app.sidebar.selectbox if s.label == 'Season'][0].value == 'All Seasons'


def test_player_profile_bridge_carries_season(app):
    """A view that selects a player (Team Analysis / Home squad table) sets
    nav_to_profile + nav_season_id and reruns. The prelude must route to the
    profile AND apply that season to the context bar BEFORE the bar draws its
    widget (writing a widget key after instantiation raises)."""
    import league_config
    _open_page(app, 'Team Analysis')
    player = [s for s in app.sidebar.selectbox if s.label == 'Select Player:']
    # pick any rated player id from the profile page's own list
    _open_page(app, 'Player Profile')
    options = [s for s in app.sidebar.selectbox if s.label == 'Select Player:'][0].options
    assert options
    # drive the bridge exactly as views do
    labels = league_config.all_season_id_map()
    target_sid = next(sid for sid, lab in labels.items() if lab == '2025/26')
    _open_page(app, 'Team Analysis')
    app.session_state['nav_to_profile'] = True
    app.session_state['nav_has_season'] = True
    app.session_state['nav_season_id'] = target_sid
    app.run()
    app.run()
    assert not _problems(app), _problems(app)
    assert _nav_radio(app, 'Player Profile').value == 'Player Profile'
    season = [s for s in app.sidebar.selectbox if s.label == 'Season'][0]
    assert season.value == '2025/26'


@pytest.mark.parametrize('page,label', [('Team Analysis', 'Select a Team'), ('Opposition Report', 'Opponent')])
def test_keyed_selectors_are_stable_across_reruns(app, page, label):
    """A keyed selectbox whose parameters differ between runs is a NEW widget
    to Streamlit 1.41 and snaps back to its first option. The team and
    opponent selectors seed their default through session state and take no
    index=, so a plain rerun must leave them where they were."""
    _open_page(app, page)
    box = [s for s in app.selectbox if s.label == label][0]
    first = box.value
    app.run()
    again = [s for s in app.selectbox if s.label == label][0]
    assert again.value == first, f"{page}: {label} moved from {first!r} to {again.value!r} on a rerun"


@pytest.mark.parametrize('page', ['Team Analysis', 'Opposition Report'])
def test_team_visuals_are_interactive(app, page):
    """Both team pages show the shot maps, rolling xG, passing network and
    the season-report dot plots as Plotly charts (parity rule), on top of the
    static PNGs that remain for the other visuals / the PDF."""
    _open_page(app, page)
    charts = app.get('plotly_chart')
    assert len(charts) >= 4, f"{page}: only {len(charts)} plotly charts"
    assert not _problems(app), _problems(app)


def test_chart_size_control_scales_every_pitch_chart(app):
    """The sidebar Chart size control sets the height of every pitch chart on
    the page (shot maps + passing network) — Huge = 1300 px."""
    import json
    _open_page(app, 'Team Analysis')
    sliders = [s for s in app.sidebar.select_slider if s.label == 'Chart size']
    assert sliders, 'Chart size control missing from the context bar'
    sliders[0].set_value('Huge')
    app.run()
    heights = []
    for ch in app.get('plotly_chart'):
        spec = json.loads(ch.spec) if isinstance(ch.spec, str) else ch.spec
        h = (spec.get('layout') or {}).get('height')
        if h in (760, 1000, 1300):
            heights.append(h)
    assert heights and all(h == 1300 for h in heights), heights
    assert not _problems(app), _problems(app)


@pytest.mark.parametrize('page', EMPTY_SEASON_PAGES)
def test_empty_season_degrades_gracefully(app, page):
    """Forcing the newest season (fixtures, few or no events) must give a
    warning or an empty state — never an exception or an st.error box."""
    _open_page(app, page)
    season_box, newest = _newest_season_label(app)
    season_box.set_value(newest)
    app.run()
    assert not _problems(app), _problems(app)
