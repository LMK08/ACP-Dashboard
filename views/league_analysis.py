"""League Analysis view — extracted verbatim from app.py's `elif analysis_type == 'League Analysis'` branch (2026-09).

Collaborators are read from the running app module at call time (the
pattern opposition_report.py uses), so importing this module never imports
app.py. The binding block at the top of render() IS the page's dependency
list: everything it reads from app.py, nothing else.
"""
import datetime
import pandas as pd
import streamlit as st
import sys


def _app():
    return sys.modules['__main__']


def render():
    app = _app()
    COMPETITIONS = app.COMPETITIONS
    FIGURE_CACHE_VERSION = app.FIGURE_CACHE_VERSION
    SEASON_ID_MAP = app.SEASON_ID_MAP
    _STRENGTH_COLS = app._STRENGTH_COLS
    _plot_values_key = app._plot_values_key
    _render_league_figure_png = app._render_league_figure_png
    all_match_data = app.all_match_data
    calculate_all_team_radars_stats = app.calculate_all_team_radars_stats
    calculate_expanded_team_stats = app.calculate_expanded_team_stats
    calculate_league_table = app.calculate_league_table
    calculate_set_piece_metrics = app.calculate_set_piece_metrics
    calculate_team_strength = app.calculate_team_strength
    filter_by_league = app.filter_by_league
    get_filtered_events = app.get_filtered_events
    get_league_label = app.get_league_label
    get_season_ids_for_selection = app.get_season_ids_for_selection
    get_season_matches = app.get_season_matches
    league_selector = app.league_selector
    matches_summary_df = app.matches_summary_df
    raw_events_df = app.raw_events_df
    season_selector = app.season_selector


    # --- League & Season Selector ---
    selected_comp_ids = league_selector("league_analysis")
    selected_season_id = season_selector("league_analysis", comp_ids=selected_comp_ids)
    active_season_ids = get_season_ids_for_selection(selected_season_id, selected_comp_ids)
    league_events_df = get_filtered_events(raw_events_df, active_season_ids, selected_comp_ids)
    league_matches_df = filter_by_league(get_season_matches(matches_summary_df, active_season_ids), selected_comp_ids)

    # --- 1. ALL DATA CALCS ---
    stats_df_raw, stats_df_pct = calculate_all_team_radars_stats(league_events_df, league_matches_df, season_id=active_season_ids if isinstance(active_season_ids, list) else selected_season_id)
    team_strength_df = calculate_team_strength(league_events_df, league_matches_df, season_id=active_season_ids if isinstance(active_season_ids, list) else selected_season_id).copy()

    # Filter all_match_data to only include matches from selected season
    season_match_ids = set(league_matches_df['matchId'].dropna().unique())
    season_match_data = {mid: data for mid, data in all_match_data.items() if mid in season_match_ids}

    try:
        expanded_stats_df = calculate_expanded_team_stats(season_match_data, league_matches_df, season_id=selected_season_id)
        combined_stats_df = pd.merge(stats_df_raw, expanded_stats_df, left_index=True, right_index=True, how='outer').fillna(0)
    except Exception as e:
        st.warning(f"Could not calculate expanded match stats: {e}")
        combined_stats_df = stats_df_raw.copy()

    # Calculate and merge set piece metrics
    try:
        set_piece_df = calculate_set_piece_metrics(league_events_df, season_id=selected_season_id)
        combined_stats_df = pd.merge(combined_stats_df, set_piece_df, left_index=True, right_index=True, how='outer').fillna(0)
    except Exception as e:
        st.warning(f"Could not calculate set piece metrics: {e}")

    # --- 2. Define Team Lists (Season-dependent groups) ---
    # Seasons with defined Group A/B; other seasons show all teams together
    SEASON_GROUPS = {
        191782: {
            'Group A': ['Fafe', 'Varzim', 'Paredes', 'Sanjoanense', 'São João Ver',
                        'Amarante', 'Vitória Guimarães II', 'Trofense', 'Sporting Braga II', 'AD Marco 09'],
            'Group B': ['1º Dezembro', 'Caldas', 'Sporting Covilhã', 'Mafra', 'União Santarém',
                        'Amora', 'Académica', 'CF Os Belenenses', 'Lusitano Évora 1911', 'Atlético CP'],
        },
        192831: {
            'Group A': ['Fafe', 'Varzim', 'Paredes', 'Paços de Ferreira', 'São João Ver',
                        'Leça', 'Vitória Guimarães II', 'Trofense', 'Vianense', 'AD Marco 09'],
            'Group B': ['Louletano', 'Caldas', 'Sporting Covilhã', 'Mafra', 'União Santarém',
                        'UD Oliveirense', 'Vitória de Sernache', 'CF Os Belenenses', 'Lusitano Évora 1911', 'Atlético CP'],
        }
    }

    has_groups = selected_season_id in SEASON_GROUPS
    all_season_teams = sorted(pd.concat([league_matches_df.get('homeTeamName'), league_matches_df.get('awayTeamName')]).dropna().unique())

    if has_groups:
        GROUP_A_TEAMS = SEASON_GROUPS[selected_season_id]['Group A']
        GROUP_B_TEAMS = SEASON_GROUPS[selected_season_id]['Group B']
        valid_group_a_teams = [t for t in GROUP_A_TEAMS if t in combined_stats_df.index]
        valid_group_b_teams = [t for t in GROUP_B_TEAMS if t in combined_stats_df.index]
        ALL_TEAMS_TO_HIGHLIGHT = list(set(GROUP_A_TEAMS + GROUP_B_TEAMS))
    else:
        ALL_TEAMS_TO_HIGHLIGHT = all_season_teams

    valid_all_teams = [t for t in ALL_TEAMS_TO_HIGHLIGHT if t in combined_stats_df.index]

    # --- Cached-PNG figure plumbing for this page ---------------------
    # These figures do NOT key on the data scope the way Match/Team do.
    # Both plotters read only their frame's index and the columns they
    # plot, so _plot_values_key() of that slice is the data half of the
    # key — see _render_league_figure_png. What no scope key could supply
    # is the WIDGET STATE: which metric sits on each axis, whether it is
    # inverted, which teams are drawn, which seasons the multi-season
    # chart concatenated. That rides in `extra`.
    _fig_day_key = datetime.date.today().isoformat()

    # Titles carry the league/season ACTUALLY selected on this page
    # (the plotters used to default to "Liga 3, 2025/26" whatever was chosen).
    _league_label = get_league_label(selected_comp_ids)
    _season_label_page = SEASON_ID_MAP.get(selected_season_id, "All Seasons")

    def _show_strength_png(stats_df, teams=None, icon_zoom=0.25,
                            season_label=None):
        _png = _render_league_figure_png(
            'team_strength', _plot_values_key(stats_df, _STRENGTH_COLS),
            (tuple(teams) if teams is not None else None, icon_zoom,
             season_label or _season_label_page, _league_label),
            _fig_day_key, FIGURE_CACHE_VERSION, stats_df)
        if _png:
            st.image(_png, use_container_width=True)

    def _show_scatter_png(stats_df, x_metric, y_metric, invert_x, invert_y,
                          season_label=None):
        _png = _render_league_figure_png(
            'custom_scatter', _plot_values_key(stats_df, (x_metric, y_metric)),
            (x_metric, y_metric, bool(invert_x), bool(invert_y),
             _league_label, season_label or _season_label_page),
            _fig_day_key, FIGURE_CACHE_VERSION, stats_df)
        if _png:
            st.image(_png, use_container_width=True)

    # --- 3. League Tables ---
    st.subheader("League Standings")

    league_table_config = {
        'Pos': st.column_config.NumberColumn('Pos', width='small'),
        'Team': st.column_config.TextColumn('Team', width='medium'),
        'P': st.column_config.NumberColumn('P', help='Played', width='small'),
        'W': st.column_config.NumberColumn('W', help='Won', width='small'),
        'D': st.column_config.NumberColumn('D', help='Drawn', width='small'),
        'L': st.column_config.NumberColumn('L', help='Lost', width='small'),
        'GF': st.column_config.NumberColumn('GF', help='Goals For', width='small'),
        'GA': st.column_config.NumberColumn('GA', help='Goals Against', width='small'),
        'GD': st.column_config.NumberColumn('GD', help='Goal Difference', width='small'),
        'Pts': st.column_config.NumberColumn('Pts', help='Points', width='small'),
    }

    if has_groups:
        col_table_a, col_table_b = st.columns(2)
        with col_table_a:
            st.markdown("**Group A**")
            table_a = calculate_league_table(league_matches_df, GROUP_A_TEAMS)
            st.dataframe(table_a, use_container_width=True, hide_index=True, column_config=league_table_config)
        with col_table_b:
            st.markdown("**Group B**")
            table_b = calculate_league_table(league_matches_df, GROUP_B_TEAMS)
            st.dataframe(table_b, use_container_width=True, hide_index=True, column_config=league_table_config)
    else:
        st.markdown("**All Teams**")
        table_all = calculate_league_table(league_matches_df, all_season_teams)
        st.dataframe(table_all, use_container_width=True, hide_index=True, column_config=league_table_config)

    # --- 4. Strength Charts ---
    if has_groups:
        st.subheader(f"Team Strength Scatterplot ({get_league_label(selected_comp_ids)} - Group B)")
        if not team_strength_df.empty:
            valid_group_b_strength_teams = [t for t in GROUP_B_TEAMS if t in team_strength_df.index]
            _show_strength_png(team_strength_df,
                                teams=valid_group_b_strength_teams,
                                icon_zoom=0.4)
            with st.expander("View Group B Raw Strength Data"):
                if valid_group_b_strength_teams:
                    st.dataframe(team_strength_df.loc[valid_group_b_strength_teams, ['Attacking Strength', 'Defending Strength']].round(2))
        else:
            st.warning("Could not calculate team strength data for Group B.")

        # Group B Custom Scatterplot
        st.subheader("Group B Custom Scatterplot")
        if not combined_stats_df.empty and valid_group_b_teams:
            group_b_stats_df = combined_stats_df.loc[valid_group_b_teams]
            metrics_to_exclude = ['teamName', 'matchId', 'seasonId', 'teamId']
            available_metrics_gb = sorted([col for col in group_b_stats_df.columns if col not in metrics_to_exclude])

            col_x_gb, col_y_gb = st.columns(2)
            with col_x_gb:
                default_x_gb_index = available_metrics_gb.index('xG') if 'xG' in available_metrics_gb else 0
                x_metric_gb = st.selectbox("Select X-Axis Metric:", available_metrics_gb, index=default_x_gb_index, key='x_metric_group_b')
            with col_y_gb:
                default_y_gb_index = available_metrics_gb.index('xG Against') if 'xG Against' in available_metrics_gb else 1
                y_metric_gb = st.selectbox("Select Y-Axis Metric:", available_metrics_gb, index=default_y_gb_index, key='y_metric_group_b')

            col_inv_x_gb, col_inv_y_gb = st.columns(2)
            with col_inv_x_gb:
                invert_x_gb = st.checkbox("Invert X-Axis (Lower is Better)", key='invert_x_group_b')
            with col_inv_y_gb:
                default_invert_y_gb = 'Against' in y_metric_gb or 'PPDA' in y_metric_gb or 'Losses' in y_metric_gb
                invert_y_gb = st.checkbox("Invert Y-Axis (Lower is Better)", value=default_invert_y_gb, key='invert_y_group_b')

            if x_metric_gb and y_metric_gb:
                _show_scatter_png(group_b_stats_df, x_metric_gb, y_metric_gb,
                                   invert_x_gb, invert_y_gb)
        else:
            st.info("No data available for Group B custom plot.")

    # --- 5. All Teams Strength Chart ---
    st.subheader("Team Strength Scatterplot (All Teams)")

    # Multi-season comparison option (filtered to selected league)
    league_scatter_map = {}
    for cid in selected_comp_ids:
        if cid in COMPETITIONS:
            league_scatter_map.update(COMPETITIONS[cid]["seasons"])
    scatter_season_labels = list(league_scatter_map.values())
    scatter_default = league_scatter_map.get(selected_season_id)
    if not scatter_default and scatter_season_labels:
        scatter_default = scatter_season_labels[0]
    scatter_seasons = st.multiselect(
        "Compare seasons", scatter_season_labels,
        default=[scatter_default] if scatter_default else [],
        key="scatter_seasons"
    )
    season_name_to_id_scatter = {v: k for k, v in league_scatter_map.items()}

    if len(scatter_seasons) > 1:
        # Multi-season: combine team_strength_df from each season
        combined_strength_frames = []
        for sname in scatter_seasons:
            sid = season_name_to_id_scatter[sname]
            s_events = get_filtered_events(raw_events_df, sid, selected_comp_ids)
            s_matches = filter_by_league(get_season_matches(matches_summary_df, sid), selected_comp_ids)
            s_df = calculate_team_strength(s_events, s_matches, season_id=sid).copy()
            if not s_df.empty:
                s_df.index = [f"{t} ({sname})" for t in s_df.index]
                s_df['Season'] = sname
                combined_strength_frames.append(s_df)
        if combined_strength_frames:
            multi_strength_df = pd.concat(combined_strength_frames)
            # Plot with text labels (no logos since same team appears multiple times)
            # scatter_seasons is not named in the key: it reaches the picture
            # only through multi_strength_df, whose index is
            # "{team} ({season})" per row — so values_key already carries
            # every season drawn, in the order they were concatenated.
            _show_strength_png(multi_strength_df, season_label="Multi-Season")
            with st.expander("View Multi-Season Raw Strength Data"):
                st.dataframe(multi_strength_df[['Attacking Strength', 'Defending Strength', 'Season']].round(2))
        else:
            st.warning("No team strength data for selected seasons.")
    else:
        # Single season (original behavior)
        if not team_strength_df.empty:
            valid_all_strength_teams = [t for t in ALL_TEAMS_TO_HIGHLIGHT if t in team_strength_df.index]
            _show_strength_png(team_strength_df, teams=valid_all_strength_teams)
            with st.expander("View All Teams Raw Strength Data"):
                 st.dataframe(team_strength_df[['Attacking Strength', 'Defending Strength']].round(2))
        else:
            st.warning("Could not calculate team strength data.")

    # --- 6. All Teams Custom Scatterplot ---
    st.subheader("All Teams Custom Scatterplot")
    if not combined_stats_df.empty:
        metrics_to_exclude = ['teamName', 'matchId', 'seasonId', 'teamId']
        available_metrics_all = sorted([col for col in combined_stats_df.columns if col not in metrics_to_exclude])

        col_x_all, col_y_all = st.columns(2)
        with col_x_all:
            default_x_all_index = available_metrics_all.index('xG') if 'xG' in available_metrics_all else 0
            x_metric_all = st.selectbox("Select X-Axis Metric:", available_metrics_all, index=default_x_all_index, key='x_metric_all')
        with col_y_all:
            default_y_all_index = available_metrics_all.index('xG Against') if 'xG Against' in available_metrics_all else 1
            y_metric_all = st.selectbox("Select Y-Axis Metric:", available_metrics_all, index=default_y_all_index, key='y_metric_all')

        col_inv_x_all, col_inv_y_all = st.columns(2)
        with col_inv_x_all:
            invert_x_all = st.checkbox("Invert X-Axis (Lower is Better)", key='invert_x_all')
        with col_inv_y_all:
            default_invert_y_all = 'Against' in y_metric_all or 'PPDA' in y_metric_all or 'Losses' in y_metric_all
            invert_y_all = st.checkbox("Invert Y-Axis (Lower is Better)", value=default_invert_y_all, key='invert_y_all')

        if x_metric_all and y_metric_all:
            _show_scatter_png(combined_stats_df, x_metric_all, y_metric_all,
                               invert_x_all, invert_y_all,
                               season_label=" + ".join(scatter_seasons) if scatter_seasons else None)

        with st.expander("View All Teams Raw Radar & Expanded Stats Data"):
            st.dataframe(combined_stats_df.round(2))
    else:
        st.warning("Could not calculate raw league stats for custom plot.")
