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
import html
import os
import re
import sys

import pytest

DASHBOARD_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# the stats-cache version the app writes (no literal to bump by hand)
STATS_VER = re.search(r"^STATS_CACHE_VERSION = '([^']+)'",
                      open(os.path.join(DASHBOARD_DIR, 'app.py')).read(), re.M).group(1)
REQUIRED_DATA = ['raw_events.parquet', 'matches_summary.parquet',
                 'all_match_data.pkl', 'player_details.pkl']

PAGES = ['Home', 'Match Analysis', 'Team Analysis', 'League Analysis', 'Player Profile',
         'Player Comparison', 'Player Analysis', 'Match Predictor', 'Shadow Team',
         'Opposition Report', 'Coach Profile']
# Pages that read the player-stats engine and used to raise KeyError
# 'next_shot_id' on a season with fixtures but no events.
EMPTY_SEASON_PAGES = ['Player Profile', 'Player Comparison', 'Player Analysis',
                      'Opposition Report', 'Team Analysis']

pytestmark = pytest.mark.skipif(
    any(not os.path.exists(os.path.join(DASHBOARD_DIR, f)) for f in REQUIRED_DATA),
    reason="dashboard data files not present (raw_events.parquet is HF-only)")


@pytest.fixture(scope='module')
def app(request):
    """One booted AppTest shared by every test in this module (boot loads
    ~4.7M events; doing it per test would take minutes)."""
    from streamlit.testing.v1 import AppTest
    cwd = os.getcwd()
    os.chdir(DASHBOARD_DIR)
    sys.path.insert(0, DASHBOARD_DIR)
    try:
        at = AppTest.from_file('app.py', default_timeout=1200)
        at.run()
        total = '?'
        try:
            with open('/proc/meminfo') as fh:
                total = f"{int(fh.readline().split()[1]) // 1024} MB"
        except OSError:
            pass
        _say(request, f"[mem] machine {total}; {_rss_mb()} MB after booting the app")
        yield at
    finally:
        os.chdir(cwd)


# GitHub's hosted runner was shut down twice in the middle of this module on
# 2026-11-13 ("The runner has received a shutdown signal", the repo's
# out-of-memory pattern) after the team pages gained sections and this file
# gained page runs. ONE booted app serves the whole module, so memory only
# ever goes up unless it is handed back:
#   * a page run leaves large frames in reference cycles -> gc.collect()
#   * every page run is a new thread and glibc keeps what a thread freed in
#     that thread's arena -> malloc_trim(0) returns it to the system
# and each test prints what the process holds, so the next failure (or the
# next added test) can be judged from the log instead of guessed at.
#
# First read-out from the runner (run 43516986474, 2026-11-13): 16 GB machine,
# 4.3 GB after booting the app, and 14.3 GB at the end of ONE walk of the
# Opposition Report's six sections — released again by the next test's
# clean-up. So memory is also released after every page change and section
# change (_open_page / _open_section), not only between tests.
#
# Second read-out, after the Opposition Report stopped copying the league's
# events on every run (run 43520667881): that walk ends at 7.0 GB and the
# module peaks at 9.0 GB. The two tests that had been kept off the runner
# (the PDF completeness test, the every-section walk on the fixtures-only
# season) are back in the gate.


def _rss_mb():
    """Resident memory of this process in MB (Linux); the peak on macOS."""
    try:
        with open('/proc/self/status') as fh:
            for line in fh:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak // (1024 * 1024) if sys.platform == 'darwin' else peak // 1024


def _say(request, text):
    reporter = request.config.pluginmanager.get_plugin('terminalreporter')
    if reporter is not None:
        reporter.write_line(text)


def _release_memory():
    """Hand back what the last page run left behind (see the note above)."""
    import gc
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except OSError:          # not glibc (macOS)
        pass


@pytest.fixture(autouse=True)
def _free_memory_after_each_test(request):
    yield
    _release_memory()
    _say(request, f"[mem] {_rss_mb():>6} MB after {request.node.name}")


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
    _release_memory()


def _newest_season_label(at):
    boxes = [s for s in at.selectbox if s.label == 'Season']
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
    team = [s for s in app.selectbox if s.label == 'Team'][0]
    assert team.value == 'Atlético CP'
    _open_page(app, 'Player Profile')
    player = [s for s in app.selectbox if s.label == 'Player'][0]
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
        boxes = [s for s in app.selectbox if s.label == 'Season']
        if boxes:
            assert boxes[0].value == chosen, f"{page}: season reset to {boxes[0].value}"
    # "All Seasons" stays selected everywhere (stable widget options); a page
    # that can't show it resolves to one season and says so, without erroring.
    _open_page(app, 'Player Profile')
    [s for s in app.selectbox if s.label == 'Season'][0].set_value('All Seasons')
    app.run()
    _open_page(app, 'Team Analysis')
    assert [s for s in app.selectbox if s.label == 'Season'][0].value == 'All Seasons'
    assert any('one season at a time' in str(c.value) for c in app.caption)
    assert any('Team Analysis' in str(h.value) for h in app.header)
    assert not _problems(app), _problems(app)
    # Switching league swaps to that league's own season key: a valid season
    # is selected, the page renders, and switching back restores the choice.
    league_box = [s for s in app.selectbox if s.label == 'League'][0]
    league_box.set_value('Campeonato')
    app.run()
    season_camp = [s for s in app.selectbox if s.label == 'Season'][0]
    assert season_camp.value in season_camp.options
    assert not _problems(app), _problems(app)
    [s for s in app.selectbox if s.label == 'League'][0].set_value('Liga 3')
    app.run()
    assert [s for s in app.selectbox if s.label == 'Season'][0].value == 'All Seasons'


def test_player_profile_bridge_carries_season(app):
    """A view that selects a player (Team Analysis / Home squad table) sets
    nav_to_profile + nav_season_id and reruns. The prelude must route to the
    profile AND apply that season to the context bar BEFORE the bar draws its
    widget (writing a widget key after instantiation raises)."""
    import league_config
    _open_page(app, 'Team Analysis')
    player = [s for s in app.selectbox if s.label == 'Player']
    # pick any rated player id from the profile page's own list
    _open_page(app, 'Player Profile')
    options = [s for s in app.selectbox if s.label == 'Player'][0].options
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
    season = [s for s in app.selectbox if s.label == 'Season'][0]
    assert season.value == '2025/26'


@pytest.mark.parametrize('page,label', [('Team Analysis', 'Team'), ('Opposition Report', 'Opponent')])
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


def test_page_head_carries_title_league_and_season(app):
    """Every page but Home opens on ONE head row: its name as the page's only
    header, with League (and Season where the page uses one) beside it in the
    main area. The sidebar holds no page picker, only the search box, so
    nothing above the navigation changes height and it never moves."""
    import context_bar
    pages = PAGES + (['Coach Comparison'] if os.path.exists(COACH_TABLES) else [])
    for page in pages:
        _open_page(app, page)
        assert not _problems(app), (page, _problems(app))
        # the sidebar's only selectbox is the search box (a fixed height on every page)
        assert [s.label for s in app.sidebar.selectbox] == ['Search'], page
        headers = [str(h.value) for h in app.header]
        # the head row is the first thing in the main area, so its pickers are
        # the first selectboxes on the page (a section further down may have
        # a 'League' picker of its own, e.g. Similar Players)
        first = [s.label for s in app.selectbox][:2]
        if page in context_bar.PAGES_WITHOUT_CONTEXT:
            assert 'League' not in first and 'Season' not in first and page not in headers, (page, headers)
            continue
        assert headers == [page], (page, headers)
        assert first[:1] == ['League'], (page, first)
        assert (first[1:2] == ['Season']) == (page not in context_bar.PAGES_WITHOUT_SEASON), (page, first)


TEAM_PAGE_SECTION_RADIO = {'Team Analysis': 'Team section', 'Opposition Report': 'Report section'}


def _open_section(app, page, name):
    """Open a section of a team page. Re-finds the radio every time: an
    element handle goes stale after a run."""
    [r for r in app.radio if r.label == TEAM_PAGE_SECTION_RADIO[page]][0].set_value(name)
    app.run()
    _release_memory()


@pytest.mark.parametrize('page', ['Team Analysis', 'Opposition Report'])
def test_team_pages_share_one_section_row(app, page):
    """Both team pages draw ONE section at a time and share the five team
    sections in the same order (parity rule); the Opposition Report puts its
    own Summary in front and opens on it. Every section renders clean, and
    the interactive visuals (parity rule again: shot maps, rolling xG,
    passing network, season-report dot plots) sit in their sections as
    Plotly charts — a closed section builds none.

    ONE walk of the sections does all of it: each section is a full run of
    a heavy page, and this module's memory on the CI runner is finite."""
    import page_sections
    _open_page(app, page)
    assert not _problems(app), _problems(app)
    radio = [r for r in app.radio if r.label == TEAM_PAGE_SECTION_RADIO[page]][0]
    names = list(radio.options)
    assert names[-5:] == list(page_sections.TEAM_SECTIONS), names
    assert names == list(page_sections.OPP_SECTIONS if page == 'Opposition Report'
                         else page_sections.TEAM_SECTIONS)
    assert radio.value == names[0]
    charts = {names[0]: len(app.get('plotly_chart'))}
    for name in names[1:]:
        _open_section(app, page, name)
        assert not _problems(app), (name, _problems(app))
        charts[name] = len(app.get('plotly_chart'))
    assert charts['Overview'] >= 1, charts          # the season-report dot plots
    assert charts['Form & shots'] >= 2, charts      # shots for / against (+ rolling xG when it has data)
    assert charts['Tactics'] >= 1, charts           # the passing network
    assert charts['Set pieces'] == 0 and charts['Players'] == 0, charts
    if page == 'Opposition Report':
        assert charts['Summary'] == 0, charts
    _open_section(app, page, names[0])


def test_opposition_pdf_is_complete_from_any_section(app, monkeypatch):
    """The Opposition Report draws one section at a time, but its PDF must
    carry every section's figures and texts whichever section is open: a
    closed section's figures are queued and built by the button, with the
    very arguments the screen would have drawn them with.

    No real figure is built here. The renderer is replaced by a stand-in
    that returns a 2x2 PNG coloured by a digest of the KEY the real one
    caches on, so "same picture" means "same arguments". The first version
    of this test built the real figures: the PDF button renders about 26 of
    them in ONE run, which added some 5 GB at the peak, and GitHub's runner
    was shut down in the middle of it (2026-11-13, the deploy of cc74c0b).
    The real figures are still drawn, section by section, by the tests above.
    """
    import hashlib
    import io

    from PIL import Image

    import generate_pdf
    import opposition_report
    import page_sections

    def stand_in_png(kind, season_key, comp_key, team_name, extra, day_key,
                     fig_ver, _events_df, _matches_df, _payload):
        key = repr((kind, season_key, comp_key, team_name, extra, day_key, fig_ver))
        buf = io.BytesIO()
        Image.new('RGB', (2, 2), tuple(hashlib.md5(key.encode()).digest()[:3])).save(buf, format='PNG')
        return buf.getvalue()

    seen = []

    def spy(**kwargs):
        seen.append(({k: hashlib.md5(v).hexdigest() for k, v in kwargs['figures'].items()},
                     set(kwargs['texts'])))
        return b'%PDF-stub'      # the generator itself: tests/test_pdf_layout.py

    monkeypatch.setattr(opposition_report, '_render_opp_figure_png', stand_in_png)
    monkeypatch.setattr(generate_pdf, 'generate_opposition_report_pdf', spy)
    _open_page(app, 'Opposition Report')
    # every section open once: each figure is then drawn in one build and
    # queued in the five others
    for name in page_sections.OPP_SECTIONS:
        # the section change and the click travel in ONE run
        [r for r in app.radio if r.label == 'Report section'][0].set_value(name)
        [b for b in app.button if b.label == 'Generate PDF Report'][0].click()
        app.run()
        assert not _problems(app), (name, _problems(app))
    assert len(seen) == len(page_sections.OPP_SECTIONS), len(seen)
    figures, texts = seen[0]
    for name, (other_figures, other_texts) in zip(page_sections.OPP_SECTIONS[1:], seen[1:]):
        assert set(other_figures) == set(figures), (name, set(figures) ^ set(other_figures))
        differ = sorted(k for k in figures if figures[k] != other_figures[k])
        assert not differ, (name, differ)
        assert other_texts == texts, (name, texts ^ other_texts)
    assert {'radar_offensive', 'radar_distribution', 'radar_defensive', 'formation',
            'shotmap_for', 'shotmap_against', 'passing_network', 'avg_positions',
            'defensive_structure', 'zone_recovery', 'zone_loss',
            'shot_assists_dribbles'} <= set(figures), sorted(figures)
    # distinct figures really are distinct pictures (the stand-in is not constant)
    assert len(set(figures.values())) == len(figures), sorted(figures)
    assert {'takeaways', 'strengths', 'weaknesses', 'key_players'} <= texts, sorted(texts)
    _open_section(app, 'Opposition Report', 'Summary')


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
    # the pitch charts live in two sections: the shot maps and the passing network
    for section in ('Form & shots', 'Tactics'):
        _open_section(app, 'Team Analysis', section)
        assert not _problems(app), (section, _problems(app))
        for ch in app.get('plotly_chart'):
            spec = json.loads(ch.spec) if isinstance(ch.spec, str) else ch.spec
            h = (spec.get('layout') or {}).get('height')
            if h in (760, 1000, 1300):
                heights.append(h)
    assert len(heights) >= 3 and all(h == 1300 for h in heights), heights
    _open_section(app, 'Team Analysis', 'Overview')


def test_projected_value_ranges_render_in_player_analysis(app):
    """'Show Projected value' is off by default, so the bracketed likely-fee
    range path only runs here: some cells carry a range, some (outside the
    calibration's support) do not, and the page stays clean."""
    _open_page(app, 'Player Analysis')
    cb = [c for c in app.checkbox if c.label == 'Show Projected value'][0]
    cb.set_value(True)
    app.run()
    cells = [str(v) for d in app.dataframe for col in d.value.columns
             if str(col) == 'Projected Value' or (isinstance(col, tuple) and 'Proj. Value' in col)
             for v in d.value[col].tolist()]
    euros = [c for c in cells if c.startswith('€')]
    assert euros, 'no projected values rendered'
    assert any('(' in c for c in euros), 'no bracketed likely-fee range in Player Analysis'
    assert any('(' not in c for c in euros), 'every value got a range — the support gate is not applied'
    assert not _problems(app), _problems(app)
    cb.set_value(False)
    app.run()


def test_profile_value_section_and_calibration_panel(app):
    """The Value section and the fee-calibration toggle sit behind a section
    radio the default walk never selects."""
    _open_page(app, 'Player Profile')
    section = [r for r in app.radio if r.label == 'Profile section'][0]
    section.set_value('Value')
    app.run()
    toggles = [t for t in app.toggle if 'fee calibration' in t.label]
    assert toggles, 'fee-calibration toggle missing from the Value section'
    toggles[0].set_value(True)
    app.run()
    assert any(m.label == 'Real sales' for m in app.metric)
    assert any('Projected value is computed by the ACP engine' in str(c.value) for c in app.caption)
    # The range line exists only for an outfielder WITH an engine value; the
    # default player differs between the local and the canonical (CI) data.
    position = next((str(m.value) for m in app.metric if m.label == 'Position'), '')
    value = next((str(m.value) for m in app.metric if m.label == 'Projected value'), '')
    if value.startswith('€') and not position.upper().startswith('GK'):
        assert any(('Likely fee' in str(c.value)) or ('No likely-fee range' in str(c.value))
                   for c in app.caption), 'outfielder with a value but no range line'
    assert not _problems(app), _problems(app)
    toggles[0].set_value(False)
    app.run()
    section.set_value('Player Radar')
    app.run()


def test_profile_similar_players_section(app):
    """The Similar Players section (models/similarity) sits behind the
    profile's section radio: it must render a table with a similarity column
    (or an honest info box for a keeper / a player without a qualifying
    season) and stay clean; a row click opens that player's profile."""
    _open_page(app, 'Player Profile')
    section = [r for r in app.radio if r.label == 'Profile section'][0]
    section.set_value('Similar Players')
    app.run()
    assert not _problems(app), _problems(app)
    tables = [d for d in app.dataframe if 'Similarity' in list(d.value.columns)]
    infos = [i for i in app.info if 'qualifying season' in str(i.value) or 'Goalkeepers' in str(i.value)
             or 'No candidates' in str(i.value)]
    assert not any('unavailable' in str(w.value) for w in app.warning), 'the similarity pool failed to build'
    assert not any('stats caches' in str(i.value) for i in app.info), 'no per-season caches found'
    assert tables or infos, 'neither a similarity table nor an explanatory info box rendered'
    if tables:
        t = tables[0].value
        assert t['Similarity'].between(0, 100).all()
        assert (t['Similar because'].astype(str) != '').all()
        assert any('How similarity is measured' in str(e.label) for e in app.expander)
    section.set_value('Player Radar')
    app.run()


def test_comparison_selectors_are_keyed_and_seedable(app):
    """The Comparison page's selectors are keyed (stable across reruns) and
    the similar-players bridge can seed Player A before the widgets draw."""
    import pyarrow.parquet as pq
    _open_page(app, 'Player Comparison')
    box_a = [s for s in app.selectbox if s.label == 'Player A'][0]
    first = box_a.value
    app.run()
    assert [s for s in app.selectbox if s.label == 'Player A'][0].value == first
    # seed A with a real player of the page's scope: the third-most-used
    # player in the cache of the season the page is showing (the landing
    # season differs between the local and the canonical data)
    import league_config
    season_label = [s for s in app.selectbox if s.label == 'Season'][0].value
    sid = next((int(k) for k, v in league_config.COMPETITIONS[43324]['seasons'].items()
                if v == season_label), None)
    cache = os.path.join(DASHBOARD_DIR, 'stats_cache', f'player_percentiles_{STATS_VER}_{sid}.parquet')
    if sid is None or not os.path.exists(cache):
        pytest.skip(f'percentiles cache for {season_label} not present')
    t = pq.read_table(cache, columns=['playerId', 'playerName', 'totalMinutes']).to_pandas()
    t = t.sort_values('totalMinutes', ascending=False)
    seed = t.iloc[2]
    app.session_state['compare_seed_a'] = int(seed['playerId'])
    app.run()
    box_a = [s for s in app.selectbox if s.label == 'Player A'][0]
    assert str(box_a.value).startswith(str(seed['playerName'])), (box_a.value, seed['playerName'])
    assert not _problems(app), _problems(app)
    # the template dropdown offers the templates of the positions A played:
    # all of his stats position's, and nothing for a position he never held
    import yaml
    with open(os.path.join(DASHBOARD_DIR, 'config.yaml')) as f:
        groups = yaml.safe_load(f)['position_groups']
    pos = pq.read_table(cache, columns=['playerId', 'primaryPosition']).to_pandas()
    pos_a = pos.loc[pos['playerId'] == int(seed['playerId']), 'primaryPosition'].iloc[0]
    tpl = [s for s in app.selectbox if s.label == 'Template'][0]
    if any(pos_a in v for v in groups.values()):
        own = {g for g, v in groups.items() if pos_a in v}
        assert own <= set(tpl.options), (pos_a, tpl.options)
        assert len(tpl.options) < len(groups), tpl.options


def test_shadow_queue_adds_in_scope_players_and_keeps_others(app):
    """A neighbour queued from the Similar Players section lands in the
    Shadow Team slot's multiselect when he is in that page's scope; one
    outside it stays queued with a warning instead of breaking the widget."""
    import pyarrow.parquet as pq
    import league_config
    _open_page(app, 'Shadow Team')
    season_label = [s for s in app.selectbox if s.label == 'Season'][0].value
    sid = next((int(k) for k, v in league_config.COMPETITIONS[43324]['seasons'].items()
                if v == season_label), None)
    cache = os.path.join(DASHBOARD_DIR, 'stats_cache', f'player_percentiles_{STATS_VER}_{sid}.parquet')
    if sid is None or not os.path.exists(cache):
        pytest.skip(f'percentiles cache for {season_label} not present')
    t = pq.read_table(cache, columns=['playerId', 'playerName', 'totalMinutes']).to_pandas()
    top = t.sort_values('totalMinutes', ascending=False).iloc[0]
    # first slot of the current formation, read from the page's own multiselects
    slot = [m for m in app.multiselect if m.label == 'Players'][0].key.replace('shadow_players_', '')
    app.session_state['shadow_pending_adds'] = [
        {'slot': slot, 'playerId': int(top['playerId']), 'playerName': str(top['playerName']),
         'teamName': '', 'seasonId': sid, 'competitionId': 43324},
        {'slot': slot, 'playerId': 1, 'playerName': 'Nobody', 'teamName': '', 'seasonId': sid,
         'competitionId': 43324},
    ]
    app.run()
    assert not _problems(app), _problems(app)
    ms = [m for m in app.multiselect if m.key == f'shadow_players_{slot}'][0]
    assert any(str(top['playerName']) in v for v in ms.value), (ms.value, top['playerName'])
    assert any('Nobody' in str(w.value) for w in app.warning)
    assert [i['playerId'] for i in app.session_state['shadow_pending_adds']] == [1]
    app.session_state['shadow_pending_adds'] = []
    app.session_state[f'shadow_players_{slot}'] = []
    app.run()
    # Load Team queues its formation (a widget key cannot be written after
    # the widget draws): saving then loading must not raise
    name = [t for t in app.text_input if 'name' in t.label.lower()]
    if name:
        name[0].set_value('smoke-team').run()
        save = [b for b in app.button if 'Save' in b.label]
        if save:
            save[0].click().run()
            load = [b for b in app.button if b.label == 'Load Team']
            if load:
                load[0].click().run()
                app.run()
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
    # The team pages draw one section at a time: a section that is not open
    # does not run (Team Analysis) or builds no figure (Opposition Report),
    # so each has to be opened on the empty season too.
    if page in TEAM_PAGE_SECTION_RADIO:
        names = list([r for r in app.radio if r.label == TEAM_PAGE_SECTION_RADIO[page]][0].options)
        for name in names:
            _open_section(app, page, name)
            assert not _problems(app), (name, _problems(app))
        _open_section(app, page, names[0])


def test_promotion_odds_cite_the_scoreline_model(app):
    """Home's promotion tile and the Match Predictor's promotion tables rest
    on the Dixon-Coles fit recorded in season_simulation.pkl (simulate_season.py);
    both pages say so next to the numbers."""
    import pickle
    with open(os.path.join(DASHBOARD_DIR, 'season_simulation.pkl'), 'rb') as fh:
        sim = pickle.load(fh)
    model = sim.get('model') or {}
    if not str(model.get('name', '')).startswith('dixon_coles'):
        pytest.skip('season_simulation.pkl predates the Dixon-Coles simulation')
    _open_page(app, 'Match Predictor')
    captions = ' '.join(str(c.value) for c in app.caption)
    assert 'Dixon-Coles' in captions and f"{sim['n_simulations']:,} Monte Carlo" in captions
    assert not _problems(app), _problems(app)
    _open_page(app, 'Home')
    captions = ' '.join(str(c.value) for c in app.caption)
    assert f"Dixon-Coles fit through {model.get('asof')}" in captions
    assert not _problems(app), _problems(app)


COACH_TABLES = os.path.join(DASHBOARD_DIR, 'models', 'coaches', 'coach_team_match.parquet')


def test_match_analysis_scoreboard_and_sections(app):
    """Match Analysis opens on a scoreboard (names, score, xG) and draws ONE
    section at a time; every section renders clean, the shot maps are the
    Shots section's two Plotly charts, and the report PDF builds from a
    section other than the first (it renders its own figures and reads
    tables prepared above the section branches)."""
    _open_page(app, 'Match Analysis')
    assert not _problems(app), _problems(app)
    match = [s for s in app.selectbox if s.label == 'Match'][0].value
    board = [str(m.value) for m in app.markdown if 'class="acp-sb"' in str(m.value)]
    assert board, 'scoreboard missing'
    board = board[0].split('</style>')[-1]           # the markup, not the stylesheet
    assert 'Match Report' in board
    names = re.findall(r'class="acp-sb-name">([^<]+)<', board)
    assert len(names) == 2 and all(html.unescape(n) in match for n in names), (names, match)
    score = re.search(r'class="acp-sb-score">(\d+)&nbsp;–&nbsp;(\d+)<', board)
    assert score and f"({score.group(1)}-{score.group(2)})" in match, (board[-400:], match)
    assert len(re.findall(r'class="acp-sb-xg (?:home|away)">\d+\.\d\d<', board)) == 2

    sections = list([r for r in app.radio if r.label == 'Match section'][0].options)
    assert sections[0] == 'Summary' and [r for r in app.radio if r.label == 'Match section'][0].value == 'Summary'
    charts = {}
    for name in sections:
        # re-find the radio every time: an element handle goes stale after a run
        [r for r in app.radio if r.label == 'Match section'][0].set_value(name)
        app.run()
        assert not _problems(app), (name, _problems(app))
        assert [s for s in app.selectbox if s.label == 'Match'][0].value == match, name
        assert any(b.label == 'Build match report PDF' for b in app.button), name
        charts[name] = len(app.get('plotly_chart'))
    assert charts['Shots'] == 2, charts
    assert charts['Summary'] == 0 and charts['Tactics'] == 0, charts

    [r for r in app.radio if r.label == 'Match section'][0].set_value('Players')
    app.run()
    [b for b in app.button if b.label == 'Build match report PDF'][0].click()
    app.run()
    assert not _problems(app), _problems(app)
    pdfs = [app.session_state[k] for k in app.session_state.filtered_state
            if str(k).startswith('match_report_pdf_')]
    assert pdfs and bytes(pdfs[0][:5]) == b'%PDF-', 'no match report PDF built from the Players section'


def test_match_deep_link_survives_a_section_click(app):
    """A match opened by a deep link (Home's button, a click on a shot) stays
    selected when a section is clicked: the picker is keyed and seeded, so
    popping nav_match_id does not turn it into a new widget."""
    _open_page(app, 'Match Analysis')
    box = [s for s in app.selectbox if s.label == 'Match'][0]
    other = [o for o in box.options if o != box.value][-1]      # an old match, not the default
    mid = _match_id_of(app, other)
    _open_page(app, 'Home')
    app.session_state['nav_match_id'] = mid
    _open_page(app, 'Match Analysis')
    assert [s for s in app.selectbox if s.label == 'Match'][0].value == other
    [r for r in app.radio if r.label == 'Match section'][0].set_value('Team stats')
    app.run()
    assert not _problems(app), _problems(app)
    assert [s for s in app.selectbox if s.label == 'Match'][0].value == other


def _match_id_of(app, display_name):
    """matchId of a Match Analysis option ('GW n: Home vs Away (h-a) - yyyy-mm-dd')."""
    import pandas as pd
    m = re.match(r'(?:\[\w+\] )?GW (\S+): (.+) vs (.+) \((\S+)\) - (\d{4}-\d{2}-\d{2})$', display_name)
    assert m, display_name
    ms = pd.read_parquet(os.path.join(DASHBOARD_DIR, 'matches_summary.parquet'))
    hit = ms[(ms['homeTeamName'] == m.group(2)) & (ms['awayTeamName'] == m.group(3))
             & (pd.to_datetime(ms['dateutc']).dt.strftime('%Y-%m-%d') == m.group(5))]
    assert len(hit) == 1, (display_name, len(hit))
    return int(hit['matchId'].iloc[0])


@pytest.mark.skipif(not os.path.exists(COACH_TABLES), reason='coach tables not built')
def test_coach_profile_sections(app):
    """Every Coach Profile section renders clean on the default coach, and
    the page opens on the club's own coach."""
    _open_page(app, 'Coach Profile')
    assert not _problems(app), _problems(app)
    coach = [s for s in app.selectbox if s.label == 'Coach'][0]
    assert 'Atlético CP' in coach.format_func(coach.value)

    def show(name):
        # re-find the radio every time: an element handle goes stale after a run
        [r for r in app.radio if r.label == 'Coach section'][0].set_value(name)
        app.run()
        assert not _problems(app), (name, _problems(app))
        return {m.label for m in app.metric}

    names = list([r for r in app.radio if r.label == 'Coach section'][0].options)
    seen = {name: show(name) for name in names}
    assert {'Team level', 'Change since arrival', 'Points − expected'} <= seen['Overview']
    assert 'Settled ↔ fluid' in seen['Flexibility']
    assert 'Trust ratio' in seen['Young players']
    show('Flexibility')
    flex = [d for d in app.dataframe if 'How stable the measure is' in list(d.value.columns)][0].value
    assert len(flex) == 3 and flex['Percentile'].dropna().between(0, 100).all()
    show('Young players')
    [r for r in app.radio if r.label == 'Young means'][0].set_value(23)
    app.run()
    assert not _problems(app), _problems(app)
    show('Overview')


@pytest.mark.skipif(not os.path.exists(COACH_TABLES), reason='coach tables not built')
def test_profile_coached_by_section(app):
    """'Coached by' on the Player Profile: a table of coaches with exact
    minutes, or an honest info box."""
    _open_page(app, 'Player Profile')
    section = [r for r in app.radio if r.label == 'Profile section'][0]
    section.set_value('Coached by')
    app.run()
    assert not _problems(app), _problems(app)
    tables = [d for d in app.dataframe if 'Coach' in list(d.value.columns)]
    assert tables or any('known coach' in str(i.value) for i in app.info)
    if tables:
        t = tables[0].value
        assert (t['Minutes'] > 0).all() and t['Share of minutes'].dropna().between(0, 1).all() \
            and t['Matches started'].dropna().between(0, 1).all()
        print(t.to_string())
    section.set_value('Player Radar')
    app.run()


@pytest.mark.skipif(not os.path.exists(COACH_TABLES), reason='coach tables not built')
def test_coach_comparison_sections(app):
    """Coach Comparison opens on the club's own coach against another coach,
    and every section renders clean with both coaches in its tables."""
    _open_page(app, 'Coach Comparison')
    assert not _problems(app), _problems(app)
    a = [s for s in app.selectbox if s.label == 'Coach A'][0]
    b = [s for s in app.selectbox if s.label == 'Coach B'][0]
    assert 'Atlético CP' in a.format_func(a.value)
    assert a.value != b.value

    names = list([r for r in app.radio if r.label == 'Comparison section'][0].options)
    for name in names:
        [r for r in app.radio if r.label == 'Comparison section'][0].set_value(name)
        app.run()
        assert not _problems(app), (name, _problems(app))
        tables = [d.value for d in app.dataframe if 'League median' in list(d.value.columns)]
        assert tables, name
        for t in tables:
            pct = [c for c in t.columns if c.endswith('percentile')]
            assert len(pct) == 2, (name, list(t.columns))
            assert t[pct].stack().dropna().between(0, 100).all()

    # another Coach B, preset the way the Coach Profile's 'Compare' button presets Coach A:
    # the page follows the seed and still renders clean
    import pandas as pd
    a = [s for s in app.selectbox if s.label == 'Coach A'][0]
    b = [s for s in app.selectbox if s.label == 'Coach B'][0]
    tm = pd.read_parquet(COACH_TABLES, columns=['coachId', 'competitionId', 'team'])
    comp = tm.loc[tm.coachId == a.value, 'competitionId'].mode().iloc[0]
    pool = tm[(tm.competitionId == comp) & (tm.coachId > 0) & ~tm.coachId.isin([a.value, b.value])]
    other = int(pool.coachId.value_counts().index[0])
    app.session_state['coach_compare_seed_b'] = other
    app.run()
    assert not _problems(app), _problems(app)
    assert [s for s in app.selectbox if s.label == 'Coach B'][0].value == other
    assert [s for s in app.selectbox if s.label == 'Coach A'][0].value == a.value


@pytest.mark.skipif(not os.path.exists(COACH_TABLES), reason='coach tables not built')
def test_coach_comparison_same_coach_and_other_league(app):
    """Picking one coach twice is said, not silently swapped; a coach sent
    from his profile in the other league is followed there as Coach A."""
    import pandas as pd
    _open_page(app, 'Coach Comparison')
    a = [s for s in app.selectbox if s.label == 'Coach A'][0]
    a_id = a.value
    app.session_state['coach_comparison_b'] = a_id            # the user picks Coach A again as Coach B
    app.run()
    assert not _problems(app), _problems(app)
    assert [s for s in app.selectbox if s.label == 'Coach B'][0].value == a_id
    assert any('two different coaches' in i.value for i in app.info)

    tm = pd.read_parquet(COACH_TABLES, columns=['coachId', 'competitionId'])
    here = int(tm.loc[tm.coachId == a_id, 'competitionId'].mode().iloc[0])
    only_there = tm[tm.coachId > 0].groupby('coachId').competitionId.agg(lambda c: set(c))
    other = [int(c) for c, comps in only_there.items() if here not in comps]
    if not other:
        pytest.skip('no coach with matches only in the other league')
    counts = tm[tm.coachId.isin(other)].coachId.value_counts()
    seed = int(counts.index[0])
    app.session_state['coach_compare_seed_a'] = seed
    app.run()
    assert not _problems(app), _problems(app)
    assert [s for s in app.selectbox if s.label == 'Coach A'][0].value == seed
    assert any('no matches in the league selected' in i.value for i in app.info)
    b = [s for s in app.selectbox if s.label == 'Coach B'][0]
    assert b.value != seed


def test_search_opens_the_page_on_the_pick(app):
    """The sidebar search box: a pick opens the right page ON that player /
    team / coach, in the league and season where the data is, and the box
    empties itself. ONE test, three picks. It is the LAST test of the module:
    it changes league, season and page, and nothing after it inherits that."""
    import pickle

    import pandas as pd

    import search
    with open(os.path.join(DASHBOARD_DIR, 'complete_player_minutes.pkl'), 'rb') as fh:
        minutes = pickle.load(fh)
    matches = pd.read_parquet(os.path.join(DASHBOARD_DIR, 'matches_summary.parquet'))
    coach_tables = None
    if os.path.exists(COACH_TABLES):
        coach_tables = (pd.read_parquet(COACH_TABLES),
                        pd.read_parquet(os.path.join(os.path.dirname(COACH_TABLES), 'coaches.parquet'))
                        .set_index('coachId'))
    index = search.build_index(minutes, matches, coach_tables)

    def box():
        return [s for s in app.sidebar.selectbox if s.label == 'Search'][0]

    def pick(option):
        box().set_value(option).run()
        app.run()
        assert not _problems(app), (option, _problems(app))
        assert box().value is None, 'the search box should empty itself'
        _release_memory()

    def head(label):
        found = [s for s in app.selectbox if s.label == label]
        return found[0].value if found else None

    _open_page(app, 'Home')
    # (app.py repairs a few names in the minutes it loads, so the app's list is
    # not this one entry for entry: every pick below is checked against the box)
    shown = set(box().options)
    assert len(shown) > 100 and {e['kind'] for e in index.values()} >= {'team', 'player'}

    # a player whose newest season is in the OTHER league: the page follows him
    # there. NOT the one with most minutes: the profile opens on that one in
    # the Campeonato anyway, which would hide a broken hand-off.
    campeonato = [e for e in index.values() if e['kind'] == 'player' and e['league'] == 'Campeonato'
                  and e['label'] in shown]
    player = campeonato[7]
    pick(f"p:{player['player_id']}")
    assert app.session_state.current_page == 'Player Profile'
    assert head('League') == 'Campeonato'
    assert head('Season') == player['label'].rsplit(', ', 1)[1]
    assert player['label'].split(' — ')[0] in [s for s in app.selectbox if s.label == 'Player'][0].value

    # a team: Team Analysis on that team, in its league and newest season
    team = next(e for e in index.values() if e['kind'] == 'team' and e['league'] == 'Liga 3'
                and e['season'] == '2026/27' and e['team'] != 'Atlético CP' and e['label'] in shown)
    # a stage filter left on Team Analysis must not swallow the pick
    app.session_state['team_analysis_stage_selector'] = 'Promotion'
    pick(f"t:{team['team']}")
    assert 'team_analysis_stage_selector' not in app.session_state
    assert app.session_state.current_page == 'Team Analysis'
    assert (head('League'), head('Season')) == ('Liga 3', '2026/27')
    assert [s for s in app.selectbox if s.label == 'Team'][0].value == team['team']

    # a coach: his profile, in his league, with no "showing another league" note
    coaches = [e for e in index.values() if e['kind'] == 'coach' and e['league'] == 'Campeonato'
               and e['label'] in shown]
    if coaches:
        coach = coaches[0]
        pick(f"c:{coach['coach_id']}")
        assert app.session_state.current_page == 'Coach Profile'
        assert head('League') == 'Campeonato'
        assert [s for s in app.selectbox if s.label == 'Coach'][0].value == coach['coach_id']
        assert not [i for i in app.info if 'has no matches in the league' in str(i.value)]
    [s for s in app.selectbox if s.label == 'League'][0].set_value('Liga 3').run()
    _release_memory()
