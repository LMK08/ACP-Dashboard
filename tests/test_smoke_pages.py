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

PAGES = ['Match Analysis', 'Team Analysis', 'League Analysis', 'Player Profile',
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


def _open_page(at, page):
    at.sidebar.radio[0].set_value(page).run()
    at.run()  # some pages rerun once on first visit (nav / session seeding)
    assert at.sidebar.radio[0].value == page


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


@pytest.mark.parametrize('page', EMPTY_SEASON_PAGES)
def test_empty_season_degrades_gracefully(app, page):
    """Forcing the newest season (fixtures, few or no events) must give a
    warning or an empty state — never an exception or an st.error box."""
    _open_page(app, page)
    season_box, newest = _newest_season_label(app)
    season_box.set_value(newest)
    app.run()
    assert not _problems(app), _problems(app)
