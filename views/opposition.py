"""Opposition Report view — extracted verbatim from app.py's `elif analysis_type == 'Opposition Report'` branch (2026-09).

Collaborators are read from the running app module at call time (the
pattern opposition_report.py uses), so importing this module never imports
app.py. The binding block at the top of render() IS the page's dependency
list: everything it reads from app.py, nothing else.
"""
import streamlit as st
import sys


def _app():
    return sys.modules['__main__']


def render():
    app = _app()
    COMPETITIONS = app.COMPETITIONS
    CURRENT_SEASON_ID = app.CURRENT_SEASON_ID
    all_match_data = app.all_match_data
    filter_by_league = app.filter_by_league
    league_selector = app.league_selector
    logger = app.logger
    matches_summary_df = app.matches_summary_df
    player_minutes_data = app.player_minutes_data
    raw_events_df = app.raw_events_df
    season_team_stats = app.season_team_stats

    selected_comp_ids = league_selector("opposition_report")
    from opposition_report import render_opposition_report
    opp_events = filter_by_league(raw_events_df, selected_comp_ids)
    opp_matches = filter_by_league(matches_summary_df, selected_comp_ids)
    # Build season map for selected leagues
    opp_season_map = {}
    for cid in selected_comp_ids:
        if cid in COMPETITIONS:
            opp_season_map.update(COMPETITIONS[cid]["seasons"])
    opp_current_sid = COMPETITIONS[selected_comp_ids[0]]["current_season"] if selected_comp_ids else CURRENT_SEASON_ID
    try:
        render_opposition_report(
            opp_events, opp_matches, all_match_data,
            season_team_stats, player_minutes_data,
            opp_current_sid, opp_season_map,
            comp_ids=selected_comp_ids,
        )
    except Exception as _opp_exc:
        # Error boundary: a failure inside the report used to print a raw
        # Python traceback into the page. Log it for us, explain it to them.
        logger.exception("Opposition Report failed")
        st.error(
            "The Opposition Report could not be built for this league and season "
            f"({type(_opp_exc).__name__}: {_opp_exc}). Try the previous season, "
            "or check back after the next data refresh."
        )
