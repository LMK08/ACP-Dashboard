"""The global context bar: league and season, chosen once, read by every page.

Before this, each page drew its own League / Season selectors with per-page
keys (seeded from a shared value, but still a widget per page). Now the
sidebar shows ONE pair at the top and the pages resolve their scope from it
through app.league_selector / app.season_selector, whose signatures are
unchanged so the views did not have to move.

Per-page differences are handled here, not in the pages:
- Home has no context bar (it is always our club, now).
- Match Predictor has a league but no season.
- "All Seasons" is always offered so the widget's option list is identical
  on every page (Streamlit resets a keyed widget whose options change — the
  value would snap to the newest season on every page switch). The player
  pages and Shadow Team honour it; the team, match, league and opposition
  pages resolve it to the default season and say so under the selector.

Deep links and cross-page bridges call set_context(league, season) BEFORE the
bar's widgets exist in the run (from Home, from app.py's prelude, or right
before st.rerun()); the guards in render() drop anything the league's season
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
SIZE_DEFAULT = 'Large'
LEAGUE_OPTIONS = ('Liga 3', 'Campeonato')

PAGES_WITHOUT_CONTEXT = {'Home'}
PAGES_WITHOUT_SEASON = {'Match Predictor'}
PAGES_WITH_ALL_SEASONS = {'Player Profile', 'Player Comparison', 'Player Analysis', 'Shadow Team'}

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


def render(current_page):
    """Draw the bar for `current_page` (nothing on Home). Container-relative:
    app.py calls this inside a sidebar container placed ABOVE the navigation,
    so widgets land in that slot rather than at the sidebar's end."""
    if current_page in PAGES_WITHOUT_CONTEXT:
        return
    app = _app()
    st.markdown(_HEADER, unsafe_allow_html=True)

    if st.session_state.get(LEAGUE_KEY) not in LEAGUE_OPTIONS:
        st.session_state.pop(LEAGUE_KEY, None)
    if LEAGUE_KEY not in st.session_state:
        remembered = st.session_state.get(LEAGUE_MEMORY)
        st.session_state[LEAGUE_KEY] = remembered if remembered in LEAGUE_OPTIONS else LEAGUE_OPTIONS[0]
    league = st.selectbox('League', LEAGUE_OPTIONS, key=LEAGUE_KEY)
    st.session_state[LEAGUE_MEMORY] = league

    if st.session_state.get(SIZE_KEY) not in SIZE_OPTIONS:
        st.session_state.pop(SIZE_KEY, None)
    if SIZE_KEY not in st.session_state:
        remembered = st.session_state.get(SIZE_MEMORY)
        st.session_state[SIZE_KEY] = remembered if remembered in SIZE_OPTIONS else SIZE_DEFAULT
    st.select_slider('Chart size', SIZE_OPTIONS, key=SIZE_KEY,
                     help='Height of the pitch charts. Pick Huge on a wide monitor.')
    st.session_state[SIZE_MEMORY] = st.session_state[SIZE_KEY]

    if current_page in PAGES_WITHOUT_SEASON:
        return
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
    st.selectbox('Season', options, key=key)
    chosen = st.session_state[key]
    st.session_state[SEASON_MEMORY] = chosen
    if chosen == ALL_SEASONS and current_page not in PAGES_WITH_ALL_SEASONS:
        st.caption(f"This page shows one season at a time — showing "
                   f"{app._default_season_label(available, options)}.")
