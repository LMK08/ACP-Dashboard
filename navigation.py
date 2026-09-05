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
