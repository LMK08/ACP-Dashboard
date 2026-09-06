"""Sidebar navigation: the analysis pages grouped by who reaches for them.

The app keeps ONE piece of navigation state, ``st.session_state.current_page``.
Each group below is drawn as a radio whose value mirrors that state (``None``
when the current page sits in another group); picking a page in any group
updates the state through ``on_change``. A view that wants to send the user
somewhere calls :func:`go_to`, optionally pre-setting the destination's
selector keys so it opens on the right season / opponent / match.

Why plain widgets and not ``st.navigation``: the views read the app's
globals through ``sys.modules['__main__']`` and the test harness drives the
sidebar radios. Once views take an explicit context object, ``st.navigation``
(real URLs per page, browser back) is a drop-in replacement for this module.
"""
import streamlit as st
import streamlit.components.v1 as components

HOME = 'Home'

NAV_GROUPS = (
    ('Club', ('Home', 'Team Analysis', 'Match Analysis', 'League Analysis')),
    ('Opposition', ('Opposition Report', 'Match Predictor')),
    ('Players', ('Player Profile', 'Player Comparison', 'Player Analysis')),
    ('Recruitment', ('Shadow Team',)),
)
ALL_PAGES = tuple(page for _, pages in NAV_GROUPS for page in pages)

_GROUP_HEADER = (
    '<div style="font-size:0.68rem;letter-spacing:0.14em;text-transform:uppercase;'
    'color:#9a948d;margin:0.9rem 0 0.1rem 0.15rem;font-weight:600;">{}</div>'
)


def _key(group):
    return f'nav_group_{group}'


def _on_change(group):
    chosen = st.session_state.get(_key(group))
    if chosen:
        st.session_state.current_page = chosen


def render_sidebar_nav():
    """Draw the grouped navigation in the sidebar; return the current page."""
    current = st.session_state.get('current_page')
    if current not in ALL_PAGES:
        current = HOME
        st.session_state.current_page = HOME
    for group, pages in NAV_GROUPS:
        # Mirror the single source of truth into this group's radio BEFORE the
        # widget is created (a click has already moved current_page in
        # _on_change, which Streamlit runs ahead of the script body).
        st.session_state[_key(group)] = current if current in pages else None
        st.sidebar.markdown(_GROUP_HEADER.format(group), unsafe_allow_html=True)
        st.sidebar.radio(group, pages, key=_key(group),
                         label_visibility='collapsed',
                         on_change=_on_change, args=(group,))
    return current


def go_to(page, **state):
    """Open `page` on the next run. Extra keyword arguments are written to
    ``st.session_state`` first, so a destination's selectors can be preset —
    e.g. ``go_to('Opposition Report', season_select_opposition_report='2026/27',
    opposition_report_team='Mafra')``."""
    if page not in ALL_PAGES:
        raise ValueError(f'unknown page {page!r}')
    for key, value in state.items():
        st.session_state[key] = value
    st.session_state.current_page = page
    st.rerun()


_PAGE_DRAWN_KEY = '_page_drawn'


def scroll_to_top_on_page_change(page):
    """Scroll the main area to the top the first time a page is drawn.

    Streamlit streams a page in element by element and, until the run ends,
    keeps the PREVIOUS page's elements on screen where the new ones have not
    arrived yet. Click a page while scrolled halfway down and the viewport
    sits on stale content that gets replaced piece by piece for the whole
    cold render — the "shaking". From the top, the new page simply builds
    downward from a stable header.

    Mechanics: a zero-height component iframe is emitted on EVERY run, at
    the same tree position, so the element count of the main area never
    changes between runs (a transient element would shift every later
    element's slot and remount them on the next interaction). Its HTML
    carries a nonce that only changes when the page changed: React keeps an
    iframe whose srcdoc is unchanged and never re-runs its script, so
    ordinary widget interactions never move the scroll position. The script
    (same origin, so it may reach the parent document) hides its own
    container with an injected :has() rule — no extra style element — and
    scrolls the main area to the top.
    """
    if st.session_state.get(_PAGE_DRAWN_KEY) != page:
        st.session_state[_PAGE_DRAWN_KEY] = page
        st.session_state['_page_drawn_n'] = st.session_state.get('_page_drawn_n', 0) + 1
    nonce = st.session_state.get('_page_drawn_n', 0)
    components.html(_SCROLL_TOP_HTML.replace('__NONCE__', f'{nonce} {page}'), height=0)


_SCROLL_TOP_HTML = """<!-- acp-scroll-top __NONCE__ --><script>
(function () {
  try {
    var d = window.parent.document;
    if (!d.getElementById('acp-scroll-top-style')) {
      var s = d.createElement('style');
      s.id = 'acp-scroll-top-style';
      s.textContent = '[data-testid="stElementContainer"]:has(iframe[srcdoc*="acp-scroll-top"]){display:none}';
      d.head.appendChild(s);
    }
    var m = d.querySelector('[data-testid="stMain"]') || d.querySelector('section.main');
    if (m) { m.scrollTop = 0; }
    d.documentElement.scrollTop = 0;
    if (d.body) { d.body.scrollTop = 0; }
  } catch (e) {}
})();
</script>"""
