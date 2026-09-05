"""Shadow Team view — extracted verbatim from app.py's `elif analysis_type == 'Shadow Team'` branch (2026-09).

Collaborators are read from the running app module at call time (the
pattern opposition_report.py uses), so importing this module never imports
app.py. The binding block at the top of render() IS the page's dependency
list: everything it reads from app.py, nothing else.
"""
import io
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import sys


def _app():
    return sys.modules['__main__']


def render():
    app = _app()
    FORMATION_COORDS = app.FORMATION_COORDS
    MPL_LOCK = app.MPL_LOCK
    SHADOW_TAG_CATEGORIES = app.SHADOW_TAG_CATEGORIES
    _calculate_age = app._calculate_age
    auto_column_config = app.auto_column_config
    create_shadow_team_graphic = app.create_shadow_team_graphic
    get_filtered_events = app.get_filtered_events
    get_player_role_scores = app.get_player_role_scores
    get_season_ids_for_selection = app.get_season_ids_for_selection
    get_season_player_minutes = app.get_season_player_minutes
    league_selector = app.league_selector
    load_and_score_player_stats = app.load_and_score_player_stats
    load_player_details = app.load_player_details
    logger = app.logger
    player_minutes_data = app.player_minutes_data
    raw_events_df = app.raw_events_df
    season_selector = app.season_selector
    player_stats_with_scores_df = app.player_stats_with_scores_df


    # --- League & Season Selector ---
    selected_comp_ids = league_selector("shadow_team")
    selected_season_id = season_selector("shadow_team", include_all_seasons=True, comp_ids=selected_comp_ids)
    active_season_ids = get_season_ids_for_selection(selected_season_id, selected_comp_ids)
    shadow_events_df = get_filtered_events(raw_events_df, active_season_ids, selected_comp_ids)
    shadow_player_minutes_df = get_season_player_minutes(player_minutes_data, active_season_ids, comp_ids=selected_comp_ids)

    # --- Load Data ---
    player_details_df = load_player_details()

    try:
        with st.spinner("Loading player statistics..."):
            player_stats_df, player_stats_with_scores_df = load_and_score_player_stats(
                shadow_events_df, shadow_player_minutes_df, selected_season_id, active_season_ids, selected_comp_ids
            )
    except Exception as e:
        st.error(f"An error occurred calculating player stats: {e}")
        logger.exception("Error in Shadow Team stats calculation")
        st.stop()

    if player_stats_with_scores_df.empty:
        st.warning("No players found with sufficient minutes for analysis.")
        st.stop()

    # Build player options sorted by minutes desc, using "Name (Team)" for uniqueness
    player_list_df = player_stats_with_scores_df[['playerId', 'playerName', 'teamName', 'totalMinutes']].copy()
    player_list_df = player_list_df.sort_values('totalMinutes', ascending=False)
    player_list_df['display_key'] = player_list_df['playerName'] + ' (' + player_list_df['teamName'] + ')'
    player_display_names = player_list_df['display_key'].tolist()
    # Map display_key -> index in stats df for exact lookups
    display_key_to_idx = {}
    for idx, row in player_list_df.iterrows():
        display_key_to_idx[row['display_key']] = idx

    # --- Sidebar Controls ---
    st.sidebar.subheader("Formation")
    formation_key = st.sidebar.selectbox("Select Formation", list(FORMATION_COORDS.keys()), key="shadow_formation")
    formation_data = FORMATION_COORDS[formation_key]

    st.sidebar.markdown("---")
    st.sidebar.subheader("Save / Load")
    team_name_input = st.sidebar.text_input("Team Name", value="My Shadow Team", key="shadow_team_name_input")

    if st.sidebar.button("Save Team", key="shadow_save_btn"):
        players = {}
        tags = {}
        for slot in formation_data['positions']:
            p_key = f"shadow_players_{slot}"
            selected = st.session_state.get(p_key, [])
            players[slot] = selected
            slot_tags = {}
            for p_name in selected:
                t_key = f"shadow_tag_{slot}_{p_name}"
                l_key = f"shadow_label_{slot}_{p_name}"
                slot_tags[p_name] = {
                    'category': st.session_state.get(t_key, 'Current Starter'),
                    'label': st.session_state.get(l_key, ''),
                }
            tags[slot] = slot_tags
        st.session_state.shadow_teams[team_name_input] = {
            'formation': formation_key,
            'players': players,
            'tags': tags,
        }
        st.sidebar.success(f"Saved '{team_name_input}'!")

    saved_names = list(st.session_state.shadow_teams.keys())
    if saved_names:
        load_name = st.sidebar.selectbox("Load Saved Team", saved_names, key="shadow_load_select")
        if st.sidebar.button("Load Team", key="shadow_load_btn"):
            saved = st.session_state.shadow_teams[load_name]
            st.session_state['shadow_formation'] = saved['formation']
            for slot, player_list in saved['players'].items():
                st.session_state[f"shadow_players_{slot}"] = player_list
            for slot, slot_tags in saved['tags'].items():
                for p_name, tag_info in slot_tags.items():
                    st.session_state[f"shadow_tag_{slot}_{p_name}"] = tag_info.get('category', 'Current Starter')
                    st.session_state[f"shadow_label_{slot}_{p_name}"] = tag_info.get('label', '')
            st.rerun()

    # --- Tag Legend (sidebar) ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Tag Legend")
    for tag_name, tag_color in SHADOW_TAG_CATEGORIES.items():
        st.sidebar.markdown(
            f'<span style="display:inline-block;width:12px;height:12px;'
            f'background-color:{tag_color};border-radius:50%;margin-right:6px;'
            f'vertical-align:middle;"></span>'
            f'<span style="vertical-align:middle;">{tag_name}</span>',
            unsafe_allow_html=True
        )

    # --- Main Content: Two Columns ---
    left_col, right_col = st.columns([3, 2])

    # Gather current assignments from widget state
    player_assignments = {}  # {slot: [player_name, ...]}
    tag_assignments = {}     # {slot: {player_name: {category, label}}}
    tag_categories_list = list(SHADOW_TAG_CATEGORIES.keys())

    with right_col:
        st.subheader("Assign Players")
        for slot in formation_data['positions']:
            with st.expander(f"{slot}", expanded=False):
                selected_players = st.multiselect(
                    "Players", player_display_names,
                    key=f"shadow_players_{slot}"
                )
                player_assignments[slot] = selected_players
                slot_tags = {}
                for p_name in selected_players:
                    st.markdown(f"**{p_name}**")
                    c1, c2 = st.columns(2)
                    with c1:
                        selected_tag = st.selectbox(
                            "Tag", tag_categories_list,
                            key=f"shadow_tag_{slot}_{p_name}"
                        )
                    with c2:
                        custom_label = st.text_input(
                            "Label", value="",
                            key=f"shadow_label_{slot}_{p_name}"
                        )
                    slot_tags[p_name] = {
                        'category': selected_tag,
                        'label': custom_label,
                    }
                tag_assignments[slot] = slot_tags

    with left_col:
        st.subheader("Formation View")
        with MPL_LOCK:
            fig = create_shadow_team_graphic(formation_key, player_assignments, tag_assignments, team_name_input, player_stats_with_scores_df)
            st.pyplot(fig)
            plt.close(fig)

            # Export PNG
            buf = io.BytesIO()
            fig_export = create_shadow_team_graphic(formation_key, player_assignments, tag_assignments, team_name_input, player_stats_with_scores_df)
            fig_export.savefig(buf, format='png', dpi=200, bbox_inches='tight', facecolor='#1a472a')
            plt.close(fig_export)
        buf.seek(0)
        st.download_button(
            label="Download PNG",
            data=buf,
            file_name=f"shadow_team_{team_name_input.replace(' ', '_')}.png",
            mime="image/png",
            key="shadow_download_png"
        )

    # --- Player Details Panel ---
    st.markdown("---")
    st.subheader("Player Details")

    for slot in formation_data['positions']:
        slot_players = player_assignments.get(slot, [])
        if not slot_players:
            continue

        for p_display_key in slot_players:
            # Find player row via display_key index lookup
            row_idx = display_key_to_idx.get(p_display_key)
            if row_idx is None:
                continue
            player_row = player_stats_with_scores_df.loc[row_idx]
            player_id = player_row.get('playerId', None)
            p_name = player_row.get('playerName', p_display_key)
            team = player_row.get('teamName', 'N/A')
            minutes = player_row.get('totalMinutes', 0)
            primary_pos = player_row.get('primaryPosition', 'N/A')

            # Get tag info for display
            p_tag_info = tag_assignments.get(slot, {}).get(p_display_key, {})
            p_tag = p_tag_info.get('category', '')
            p_tag_color = SHADOW_TAG_CATEGORIES.get(p_tag, '#ffffff')

            # Get age from player_details
            age_str = "N/A"
            if player_id is not None and not player_details_df.empty:
                pid = int(player_id) if pd.notna(player_id) else None
                if pid is not None and pid in player_details_df.index:
                    birth_date = player_details_df.loc[pid].get('birthDate', None)
                    age_val = _calculate_age(birth_date)
                    if age_val != "N/A":
                        age_str = f"{age_val:.1f}"

            with st.expander(f"{slot}: {p_name} ({team})", expanded=False):
                # Tag badge
                if p_tag:
                    st.markdown(
                        f'<span style="background-color:{p_tag_color};color:#fff;'
                        f'padding:2px 8px;border-radius:10px;font-size:0.8em;">'
                        f'{p_tag}</span>',
                        unsafe_allow_html=True
                    )
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Team", team)
                m2.metric("Age", age_str)
                m3.metric("Minutes", f"{int(minutes):,}")
                m4.metric("Position", primary_pos)

                # Role scores table
                role_scores = get_player_role_scores(player_row, slot)
                if role_scores:
                    scores_df = pd.DataFrame(list(role_scores.items()), columns=['Role', 'Score'])
                    st.dataframe(scores_df, use_container_width=True, hide_index=True, column_config=auto_column_config(scores_df))
                else:
                    st.caption("No role scores available for this slot.")
