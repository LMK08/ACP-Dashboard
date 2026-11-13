"""The global context: league and season, chosen once, read by every page.

Before this, each page drew its own League / Season selectors with per-page
keys (seeded from a shared value, but still a widget per page). Now ONE pair
is drawn per run and the pages resolve their scope from it through
app.league_selector / app.season_selector, whose signatures are unchanged so
the views did not have to move. Since 2026-11 the pair sits in the page's
head row in the main area, beside the title (render_page_head); it was at
the top of the sidebar, above the navigation, which it pushed up and down.

Per-page differences are handled here, not in the pages:
- Home has no context bar (it is always our club, now).
- Match Predictor has a league but no season.
- "All Seasons" is always offered so the widget's option list is identical
  on every page (Streamlit resets a keyed widget whose options change — the
  value would snap to the newest season on every page switch). The player
  pages and Shadow Team honour it; the team, match, league and opposition
  pages resolve it to the default season and say so under the head row.

Deep links and cross-page bridges call set_context(league, season) BEFORE the
bar's widgets exist in the run (from Home, from app.py's prelude, or right
before st.rerun()); the guards in render_page_head() drop anything the league's season
list can't honour.
"""
import sys

import streamlit as st

LEAGUE_KEY = 'ctx_league'
# The season widget key is PER LEAGUE (season_key(league)): each league has a
# different season list, and Streamlit resets a keyed widget whose options
# change — one key per league keeps every key's options stable forever.
SEASON_KEY_PREFIX = 'ctx_season__'
# Streamlit drops a widget's state when a run doesn't render it (Home has no
# bar, Match Predictor no season). These plain keys remember the last choice
# so the bar re-seeds from them instead of falling back to the default.
LEAGUE_MEMORY = 'ctx_league_memory'
SEASON_MEMORY = 'ctx_season_memory'
ALL_SEASONS = 'All Seasons'
# Chart size: a Plotly figure's height is fixed server-side while its width
# follows the browser, so on a wide screen a pitch chart is height-limited
# and leaves side margins. This control (kept in session state) lets the
# user pick the height every pitch chart uses; pages call pitch_height().
SIZE_KEY = 'ctx_chart_size'
SIZE_MEMORY = 'ctx_chart_size_memory'
SIZE_OPTIONS = ('Standard', 'Large', 'Huge')
PITCH_HEIGHTS = {'Standard': 760, 'Large': 1000, 'Huge': 1300}
# Large (1,000 px) is taller than a 900 px laptop screen; Standard (760 px)
# fits one. Standard was tried as the default on 2026-11-13 and Lucas kept
# Large. The choice is remembered for the session.
SIZE_DEFAULT = 'Large'
LEAGUE_OPTIONS = ('Liga 3', 'Campeonato')

PAGES_WITHOUT_CONTEXT = {'Home'}
PAGES_WITHOUT_SEASON = {'Match Predictor', 'Coach Profile'}
PAGES_WITH_ALL_SEASONS = {'Player Profile', 'Player Comparison', 'Player Analysis', 'Shadow Team'}
# The pages whose pitch charts read pitch_height(): the Chart size control is
# drawn only there (add a page here when it gains a pitch chart).
PAGES_WITH_PITCH_CHARTS = {'Team Analysis', 'Opposition Report', 'Match Analysis', 'Player Profile'}

_HEADER = (
    '<div style="font-size:0.68rem;letter-spacing:0.14em;text-transform:uppercase;'
    'color:#9a948d;margin:0.4rem 0 0.1rem 0.15rem;font-weight:600;">Context</div>'
)


def _app():
    return sys.modules['__main__']


def season_key(league_label):
    return f'{SEASON_KEY_PREFIX}{league_label}'


def current_league():
    return st.session_state.get(LEAGUE_KEY) or st.session_state.get(LEAGUE_MEMORY) or LEAGUE_OPTIONS[0]


def set_context(league=None, season=None):
    """Preset the bar for the NEXT run (deep links, cross-page bridges). Must be
    called before the bar's widgets exist in the current run — i.e. from a
    page without the bar (Home), from app.py's prelude, or right before
    st.rerun()."""
    if league:
        st.session_state[LEAGUE_KEY] = league
        st.session_state[LEAGUE_MEMORY] = league
    if season:
        st.session_state[season_key(league or current_league())] = season
        st.session_state[SEASON_MEMORY] = season


def pitch_height():
    """Figure height (px) for every pitch chart, from the Chart size control."""
    label = st.session_state.get(SIZE_KEY) or st.session_state.get(SIZE_MEMORY) or SIZE_DEFAULT
    return PITCH_HEIGHTS.get(label, PITCH_HEIGHTS[SIZE_DEFAULT])


def comp_ids_for(league_label):
    """Competition ids for a league label; falls back to the first competition."""
    comps = _app().COMPETITIONS
    ids = [cid for cid, cfg in comps.items() if cfg['name'] == league_label]
    return ids or [next(iter(comps))]


def current_comp_ids():
    return comp_ids_for(current_league())


def season_choices(comp_ids, include_all_seasons=True):
    """(available {season_id: label}, options newest-first) for the league(s).
    include_all_seasons is always True for the WIDGET (stable options); pages
    that can't use it resolve it in current_season_id."""
    comps = _app().COMPETITIONS
    available = {}
    for cid in comp_ids or []:
        if cid in comps:
            available.update(comps[cid]['seasons'])
    if not available:
        available = dict(_app().SEASON_ID_MAP)
    labels = list(dict.fromkeys(available.values()))
    return available, ([ALL_SEASONS] if include_all_seasons else []) + labels


def current_season_id(comp_ids, include_all_seasons):
    """Season id for the page's scope: None for All Seasons (when the page
    accepts it), else the chosen season, else the data-aware default."""
    app = _app()
    available, options = season_choices(comp_ids, include_all_seasons)
    label = st.session_state.get(season_key(current_league())) or st.session_state.get(SEASON_MEMORY)
    if label == ALL_SEASONS:
        if include_all_seasons:
            return None
        label = app._default_season_label(available, options)
    if label not in options:
        label = app._default_season_label(available, options)
    for sid, lab in available.items():
        if lab == label:
            return sid
    return next(iter(available), app.CURRENT_SEASON_ID)


HEAD_KEY = 'page_head'
# The head row: the title's rule runs under the whole row (title + pickers),
# so the heading's own underline and bottom spacing are taken off inside it.
# (Streamlit pads a heading 1rem on top and pulls the next element up with a
# -1rem bottom margin on its markdown container: left in, the heading
# overflowed the row and sat on the rule.)
HEAD_CSS = (
    f'[class*="st-key-{HEAD_KEY}"] {{ border-bottom: 1.5px solid #1a1a1a; '
    'padding-bottom: 0.5rem; margin-bottom: 0.35rem; }\n'
    f'[class*="st-key-{HEAD_KEY}"] h2 {{ border-bottom: none !important; '
    'padding: 0 !important; margin: 0 !important; line-height: 1.15 !important; }\n'
    f'[class*="st-key-{HEAD_KEY}"] [data-testid="stHeading"] [data-testid="stMarkdownContainer"] '
    '{ margin-bottom: 0 !important; }'
)


def render_page_head(current_page):
    """The head row of a page, in the MAIN area: its title on the left, the
    League and Season pickers on the right (2026-11). Nothing on Home, which
    writes its own heading and has no context; no Season on the pages that
    do not use one.

    Until then the two pickers sat in the sidebar above the navigation, so
    the page list moved by up to 250 px between Home, the pages without a
    season and the rest. app.py calls this once per run, BEFORE the view
    renders: a view reads the choice through league_selector() /
    season_selector() and never draws a title of its own.
    """
    if current_page in PAGES_WITHOUT_CONTEXT:
        return
    app = _app()
    with_season = current_page not in PAGES_WITHOUT_SEASON
    note = None
    with st.container(key=HEAD_KEY):
        cols = st.columns([5, 2, 2] if with_season else [7, 2], vertical_alignment="bottom")
        cols[0].header(current_page)

        if st.session_state.get(LEAGUE_KEY) not in LEAGUE_OPTIONS:
            st.session_state.pop(LEAGUE_KEY, None)
        if LEAGUE_KEY not in st.session_state:
            remembered = st.session_state.get(LEAGUE_MEMORY)
            st.session_state[LEAGUE_KEY] = remembered if remembered in LEAGUE_OPTIONS else LEAGUE_OPTIONS[0]
        # labels are collapsed, not removed: "Liga 3" and "2026/27" say what
        # they are, and screen readers and the tests still find 'League' / 'Season'
        league = cols[1].selectbox('League', LEAGUE_OPTIONS, key=LEAGUE_KEY, label_visibility='collapsed')
        st.session_state[LEAGUE_MEMORY] = league

        if with_season:
            available, options = season_choices(comp_ids_for(league), True)
            key = season_key(league)
            if st.session_state.get(key) not in options:
                st.session_state.pop(key, None)
            if key not in st.session_state:
                remembered = st.session_state.get(SEASON_MEMORY)
                # Else: newest season with enough EVENT data, not merely the newest
                # fixture list (see app._default_season_label).
                st.session_state[key] = (remembered if remembered in options
                                         else app._default_season_label(available, options))
            cols[2].selectbox('Season', options, key=key, label_visibility='collapsed')
            chosen = st.session_state[key]
            st.session_state[SEASON_MEMORY] = chosen
            if chosen == ALL_SEASONS and current_page not in PAGES_WITH_ALL_SEASONS:
                note = (f"This page shows one season at a time — showing "
                        f"{app._default_season_label(available, options)}.")
        # inside the head row's container: an element that comes and goes
        # here does not shift the position of anything the view draws
        if note:
            st.caption(note)


_DISPLAY_HEADER = _HEADER.replace('Context', 'Display').replace('margin:0.4rem', 'margin:0.9rem')


def render_display(current_page):
    """The Chart size control, drawn by app.py UNDER the navigation and only
    on the pages with pitch charts. It has nothing to do with league or
    season, and sat between the two until 2026-11. The choice is kept in
    SIZE_MEMORY, so it survives the pages that do not draw the control."""
    if current_page not in PAGES_WITH_PITCH_CHARTS:
        return
    st.markdown(_DISPLAY_HEADER, unsafe_allow_html=True)
    if st.session_state.get(SIZE_KEY) not in SIZE_OPTIONS:
        st.session_state.pop(SIZE_KEY, None)
    if SIZE_KEY not in st.session_state:
        remembered = st.session_state.get(SIZE_MEMORY)
        st.session_state[SIZE_KEY] = remembered if remembered in SIZE_OPTIONS else SIZE_DEFAULT
    st.select_slider('Chart size', SIZE_OPTIONS, key=SIZE_KEY,
                     help='Height of the pitch charts. Pick Large or Huge on a wide monitor.')
    st.session_state[SIZE_MEMORY] = st.session_state[SIZE_KEY]
