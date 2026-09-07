"""Player Comparison view — extracted verbatim from app.py's `elif analysis_type == 'Player Comparison'` branch (2026-09).

Collaborators are read from the running app module at call time (the
pattern opposition_report.py uses), so importing this module never imports
app.py. The binding block at the top of render() IS the page's dependency
list: everything it reads from app.py, nothing else.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import sys


def _app():
    return sys.modules['__main__']


def render():
    app = _app()
    MPL_LOCK = app.MPL_LOCK
    OUR_TEAM = app.OUR_TEAM
    POSITION_GROUPS = app.POSITION_GROUPS
    RADAR_HIDDEN_METRICS = app.RADAR_HIDDEN_METRICS
    WEIGHTS = app.WEIGHTS
    get_filtered_events = app.get_filtered_events
    get_season_ids_for_selection = app.get_season_ids_for_selection
    get_season_player_minutes = app.get_season_player_minutes
    league_selector = app.league_selector
    load_and_score_player_stats = app.load_and_score_player_stats
    logger = app.logger
    player_minutes_data = app.player_minutes_data
    plot_comparison_radar = app.plot_comparison_radar
    raw_events_df = app.raw_events_df
    season_selector = app.season_selector
    player_stats_with_scores_df = app.player_stats_with_scores_df


    # --- League & Season Selector ---
    selected_comp_ids = league_selector("player_comparison")
    selected_season_id = season_selector("player_comparison", include_all_seasons=True, comp_ids=selected_comp_ids)
    active_season_ids = get_season_ids_for_selection(selected_season_id, selected_comp_ids)
    comp_events_df = get_filtered_events(raw_events_df, active_season_ids, selected_comp_ids)
    comp_player_minutes_df = get_season_player_minutes(player_minutes_data, active_season_ids, comp_ids=selected_comp_ids)

    # --- 1. Load Data ---
    try:
        with st.spinner("Loading player statistics..."):
            player_stats_df, player_stats_with_scores_df = load_and_score_player_stats(
                comp_events_df, comp_player_minutes_df, selected_season_id, active_season_ids, selected_comp_ids
            )
    except Exception as e:
        st.error(f"An error occurred calculating player stats: {e}")
        logger.exception("Error in Player Comparison stats calculation")
        st.stop()

    if player_stats_with_scores_df.empty:
        st.warning("No players found with sufficient minutes for comparison.")
        st.stop()

    # --- 2. Player Selectors (NEW LOGIC) ---
    st.sidebar.subheader("Comparison Options")

    # --- Step A: Select Player A (from all players) ---
    # FIX: Include 'playerId' in the columns so we can use it for lookup
    player_list_df = player_stats_with_scores_df[['playerId', 'playerName', 'teamName', 'totalMinutes']].sort_values(by='totalMinutes', ascending=False)
    player_list_df['display_name'] = player_list_df['playerName'].astype(str) + " (" + player_list_df['teamName'].astype(str) + ", " + pd.to_numeric(player_list_df['totalMinutes'], errors='coerce').fillna(0).astype(int).astype(str) + " min)"

    # Default Player A to our own club's most-used player (list is sorted
    # by minutes, so the first OUR_TEAM row is that player).
    _our_positions = np.flatnonzero(player_list_df['teamName'].astype(str).values == OUR_TEAM)
    _default_a_idx = int(_our_positions[0]) if len(_our_positions) else 0
    # Keyed selectors seeded through session state BEFORE the widgets exist
    # (never index=: a keyed widget whose parameters change is a new widget
    # to Streamlit). The Similar Players section's 'Compare on radar' button
    # sets compare_seed_a / compare_seed_b (playerIds) via navigation.go_to.
    _a_options = player_list_df['display_name'].tolist()
    _seed_a = st.session_state.pop('compare_seed_a', None)
    _seed_b = st.session_state.pop('compare_seed_b', None)
    if _seed_a is not None:
        _m = player_list_df[player_list_df['playerId'] == int(_seed_a)]
        if not _m.empty:
            st.session_state['player_comparison_a'] = _m['display_name'].iloc[0]
        else:
            st.sidebar.warning("The player sent here is not in this league/season scope — "
                               "showing the default comparison instead.")
            _seed_b = None
    if st.session_state.get('player_comparison_a') not in _a_options:
        st.session_state['player_comparison_a'] = _a_options[_default_a_idx] if _a_options else None
    selected_player_a_display = st.sidebar.selectbox(
        "Select Player A:",
        _a_options,
        key='player_comparison_a',
    )

    # FIX: Lookup by ID instead of Name
    selected_player_a_id = player_list_df[player_list_df['display_name'] == selected_player_a_display]['playerId'].values[0]
    player_a_data = player_stats_with_scores_df[player_stats_with_scores_df['playerId'] == selected_player_a_id]

    # Get the name safely from the ID-filtered data
    selected_player_a_name = player_a_data.iloc[0]['playerName']

    # --- Step B: Select Template ---
    all_templates = sorted(list(POSITION_GROUPS.keys()))

    # Find Player A's best-fit template as default
    primary_pos_a = player_a_data.iloc[0]['primaryPosition']
    eligible_groups_a = [pos_group for pos_group, pos_roles in POSITION_GROUPS.items() if primary_pos_a in pos_roles]
    highest_score = -1; default_template = all_templates[0]
    for group in eligible_groups_a:
        score_col = group + '_Score'
        if score_col in player_a_data.columns:
            player_score = player_a_data[score_col].values[0]
            if player_score > highest_score:
                highest_score = player_score; default_template = group

    # Template: default to Player A's best-fit template on first visit / a
    # change of Player A; a seeded Player B forces a template both fit.
    _tpl_key = 'player_comparison_template'
    if st.session_state.get('player_comparison_last_a') != selected_player_a_display or _seed_b is not None:
        st.session_state[_tpl_key] = default_template
        st.session_state['player_comparison_last_a'] = selected_player_a_display
    if st.session_state.get(_tpl_key) not in all_templates:
        st.session_state[_tpl_key] = default_template if default_template in all_templates else all_templates[0]
    if _seed_b is not None:
        _b_pos = player_stats_with_scores_df.loc[
            player_stats_with_scores_df['playerId'] == int(_seed_b), 'primaryPosition']
        _b_pos = str(_b_pos.iloc[0]) if len(_b_pos) else None
        if _b_pos is not None and _b_pos not in POSITION_GROUPS.get(st.session_state[_tpl_key], []):
            for _t in all_templates:
                if _b_pos in POSITION_GROUPS.get(_t, []):
                    st.session_state[_tpl_key] = _t
                    break
    selected_template = st.sidebar.selectbox(
        "Select Comparison Template:",
        all_templates,
        key=_tpl_key,
    )

    # --- Step C: Filter Player B list based on Template ---
    positions_in_group = POSITION_GROUPS.get(selected_template, [])

    filtered_player_df = player_stats_with_scores_df[
        player_stats_with_scores_df['primaryPosition'].isin(positions_in_group)
    ]

    # Create the display list for Player B from the filtered df
    # FIX: Include 'playerId' here too
    player_b_list_df = filtered_player_df[['playerId', 'playerName', 'teamName', 'totalMinutes']].sort_values(by='totalMinutes', ascending=False)
    player_b_list_df['display_name'] = player_b_list_df['playerName'].astype(str) + " (" + player_b_list_df['teamName'].astype(str) + ", " + pd.to_numeric(player_b_list_df['totalMinutes'], errors='coerce').fillna(0).astype(int).astype(str) + " min)"

    # Player B: the seeded player when the bridge set one, else the second
    # player of the group (or the first if the group has one)
    _b_options = player_b_list_df['display_name'].tolist()
    if _seed_b is not None:
        _mb = player_b_list_df[player_b_list_df['playerId'] == int(_seed_b)]
        if not _mb.empty:
            st.session_state['player_comparison_b'] = _mb['display_name'].iloc[0]
        else:
            st.sidebar.warning("The second player sent here is not in this scope or position "
                               "group — showing the default Player B instead.")
    if st.session_state.get('player_comparison_b') not in _b_options:
        st.session_state['player_comparison_b'] = (_b_options[1] if len(_b_options) > 1
                                                   else (_b_options[0] if _b_options else None))

    # --- Step D: Select Player B (from filtered list) ---
    selected_player_b_display = st.sidebar.selectbox(
        "Select Player B (Same Position Group):",
        _b_options,
        key='player_comparison_b',
    )

    # FIX: Lookup by ID instead of Name
    selected_player_b_id = player_b_list_df[player_b_list_df['display_name'] == selected_player_b_display]['playerId'].values[0]
    player_b_data = player_stats_with_scores_df[player_stats_with_scores_df['playerId'] == selected_player_b_id]

    # Get the name safely
    selected_player_b_name = player_b_data.iloc[0]['playerName']


    # --- 4. Plot Radar ---
    st.subheader(f"Comparing: {selected_player_a_name} vs. {selected_player_b_name}")

    _mins_a = pd.to_numeric(player_a_data.iloc[0].get('totalMinutes', 0), errors='coerce') or 0
    _mins_b = pd.to_numeric(player_b_data.iloc[0].get('totalMinutes', 0), errors='coerce') or 0
    _below_threshold = []
    if _mins_a < 300:
        _below_threshold.append(f"{selected_player_a_name} ({int(_mins_a)} min)")
    if _mins_b < 300:
        _below_threshold.append(f"{selected_player_b_name} ({int(_mins_b)} min)")

    if _below_threshold:
        st.info(f"⚠️ **Insufficient sample size** — {' and '.join(_below_threshold)}.")

    metrics_to_plot = list(WEIGHTS[selected_template].keys())
    metrics_to_plot = [m for m in metrics_to_plot
                       if m in player_stats_with_scores_df.columns
                       and m not in RADAR_HIDDEN_METRICS]

    # --- FIX: Use a square figure to prevent distortion ---
    with MPL_LOCK:
        fig = plt.figure(figsize=(15, 15))
        # [left, bottom, width, height] - This centers the radar
        ax_radar = fig.add_axes([0.15, 0.15, 0.7, 0.7], polar=True)

        plot_comparison_radar(
            ax_radar,
            player_a_data,
            player_b_data,
            metrics_to_plot,
            selected_template
        )

        st.pyplot(fig, use_container_width=True)
