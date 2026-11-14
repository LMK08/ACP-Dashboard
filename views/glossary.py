"""Glossary: every number the dashboard shows, by its name as shown (2026-11).

The entries live in glossary.py (one place; the rules are in its docstring).
This page only draws them: an area row, a search box, one table per group.
"""
import sys

import streamlit as st

import glossary

AREA_KEY = 'glossary_area'
QUERY_KEY = 'glossary_query'


def render():
    app = sys.modules['__main__']
    engine_help = getattr(app, 'ENGINE_METRIC_HELP', None)

    st.caption(glossary.INTRO)
    query = st.text_input("Find a metric", key=QUERY_KEY, placeholder="for example: PPDA, Deep Completions, xT")
    if str(query or '').strip():
        # a search looks in every area
        wanted = [group for names in glossary.AREAS.values() for group in names]
        st.caption("Results from every area.")
    else:
        area = st.radio("Area", list(glossary.AREAS), horizontal=True, key=AREA_KEY,
                        label_visibility='collapsed')
        wanted = glossary.AREAS[area]
    if not glossary.render_groups(st, wanted, query, engine_help):
        st.info("No metric matches that search. A metric that is not here has no agreed definition yet.")
