"""Team Analysis view — extracted verbatim from app.py's `elif analysis_type == 'Team Analysis'` branch (2026-09).

Collaborators are read from the running app module at call time (the
pattern opposition_report.py uses), so importing this module never imports
app.py. The binding block at the top of render() IS the page's dependency
list: everything it reads from app.py, nothing else.
"""
import pandas as pd
import streamlit as st
import sys
import context_bar
import team_interactive as ti
import theme


def _app():
    return sys.modules['__main__']


def render():
    app = _app()
    FIGURE_CACHE_VERSION = app.FIGURE_CACHE_VERSION
    OUR_TEAM = app.OUR_TEAM
    SEASON_ID_MAP = app.SEASON_ID_MAP
    STAGE_ALL = app.STAGE_ALL
    _calculate_age = app._calculate_age
    _render_team_figure_png = app._render_team_figure_png
    _season_id_list = app._season_id_list
    auto_column_config = app.auto_column_config
    calculate_all_team_radars_stats = app.calculate_all_team_radars_stats
    calculate_set_piece_metrics = app.calculate_set_piece_metrics
    calculate_xg_history_data = app.calculate_xg_history_data
    filter_by_league = app.filter_by_league
    filter_by_stage = app.filter_by_stage
    get_filtered_events = app.get_filtered_events
    get_league_label = app.get_league_label
    get_season_ids_for_selection = app.get_season_ids_for_selection
    get_season_matches = app.get_season_matches
    get_season_player_minutes = app.get_season_player_minutes
    get_season_team_stats = app.get_season_team_stats
    get_team_primary_formation = app.get_team_primary_formation
    get_team_starting_xi = app.get_team_starting_xi
    league_selector = app.league_selector
    load_obv_viz_data = app.load_obv_viz_data
    load_player_details = app.load_player_details
    matches_summary_df = app.matches_summary_df
    player_minutes_data = app.player_minutes_data
    raw_events_df = app.raw_events_df
    render_season_report_section = app.render_season_report_section
    season_selector = app.season_selector
    season_team_stats = app.season_team_stats
    stage_selector = app.stage_selector


    # --- League & Season Selector ---
    selected_comp_ids = league_selector("team_analysis")
    selected_season_id = season_selector("team_analysis", comp_ids=selected_comp_ids)
    active_season_ids = get_season_ids_for_selection(selected_season_id, selected_comp_ids)
    season_label = SEASON_ID_MAP.get(selected_season_id, "Unknown") if isinstance(selected_season_id, int) else "Unknown"
    # Stage selector (Regular / Promotion / Maintenance / Promotion playoff)
    selected_stage = stage_selector(
        "team_analysis",
        matches_summary_df,
        selected_comp_ids,
        active_season_ids,
    )
    team_events_df = get_filtered_events(raw_events_df, active_season_ids, selected_comp_ids)
    team_matches_df = filter_by_league(get_season_matches(matches_summary_df, active_season_ids), selected_comp_ids)
    # Apply stage filter — narrows the working set to the chosen stage's matches.
    team_events_df, team_matches_df = filter_by_stage(
        team_events_df, team_matches_df, matches_summary_df,
        selected_comp_ids, active_season_ids, selected_stage,
    )
    team_player_minutes_df = get_season_player_minutes(player_minutes_data, active_season_ids, comp_ids=selected_comp_ids)
    team_season_stats = get_season_team_stats(season_team_stats, active_season_ids, comp_ids=selected_comp_ids)

    all_teams_t = sorted(pd.concat([team_matches_df.get('homeTeamName'), team_matches_df.get('awayTeamName')]).dropna().unique())
    # Our own club first, not whichever team sorts first alphabetically.
    # Default (our club) and deep links (a team's dot in the season report)
    # both go through session state BEFORE the widget exists, and the widget
    # takes no index=: a keyed widget whose parameters change between runs
    # is a NEW widget to Streamlit (1.41 reset it to the first option).
    _nav_team = st.session_state.pop('nav_team', None)
    if _nav_team in all_teams_t:
        st.session_state['team_select_tab'] = _nav_team
    if st.session_state.get('team_select_tab') not in all_teams_t:
        st.session_state['team_select_tab'] = OUR_TEAM if OUR_TEAM in all_teams_t else all_teams_t[0]
    selected_team_t = st.sidebar.selectbox("Select a Team", all_teams_t,
                                           key="team_select_tab")
    _stage_suffix = "" if selected_stage in (STAGE_ALL, None) else f" — {selected_stage}"
    st.header(f"Team Report: {selected_team_t}{_stage_suffix}")
    if selected_stage not in (STAGE_ALL, None):
        _n_team_matches = team_matches_df[
            (team_matches_df['homeTeamName'] == selected_team_t)
            | (team_matches_df['awayTeamName'] == selected_team_t)
        ].shape[0]
        st.caption(
            f"Filtered to **{selected_stage}** — {len(team_matches_df)} matches in scope, "
            f"{_n_team_matches} for {selected_team_t}."
        )

    # Load player details for roster table
    player_details_df = load_player_details()

    # When a stage filter is active, force events-based radars so they
    # reflect only the matches in that stage (Wyscout's table is
    # season-aggregated and would otherwise leak full-season numbers).
    _stage_active = selected_stage not in (STAGE_ALL, None)
    _radar_cache_key = (active_season_ids if isinstance(active_season_ids, list) else selected_season_id)
    if _stage_active:
        _radar_cache_key = f"{_radar_cache_key}_{selected_stage}"
    stats_df_raw, stats_df_pct = calculate_all_team_radars_stats(
        team_events_df, team_matches_df,
        season_id=_radar_cache_key,
        force_events=_stage_active,
    )

    # Compute set piece radar data (all rate metrics — higher = better, no inversions)
    sp_df_raw = None
    sp_df_pct = None
    try:
        # team_events_df is stage-filtered above, so the stage has to ride
        # in the key — it is unhashed inside (leading underscore), and the
        # season alone would serve the All-Stages numbers here and cache
        # them to a stage-blind parquet.
        sp_df_raw = calculate_set_piece_metrics(
            team_events_df,
            season_id=active_season_ids if isinstance(active_season_ids, list) else selected_season_id,
            stage=selected_stage if _stage_active else None,
        )
        if sp_df_raw is not None and not sp_df_raw.empty:
            sp_df_pct = sp_df_raw.copy()
            for col in sp_df_pct.columns:
                sp_df_pct[col] = sp_df_pct[col].rank(pct=True) * 100
    except Exception:
        pass

    league_label = get_league_label(selected_comp_ids)

    # --- Cached-PNG figure plumbing for this page ---------------------
    # Every figure below reads team_events_df / team_matches_df, which are
    # fully determined by (active_season_ids, selected_comp_ids,
    # selected_stage) — see _render_team_figure_png for the full argument.
    # These three become the scope half of every cache key; team_name and
    # `extra` supply the rest. _season_id_list normalises the
    # None|int|list active_season_ids ('All Seasons' -> () stays distinct
    # from any real season).
    _fig_season_key = tuple(sorted(_season_id_list(active_season_ids)))
    _fig_comp_key = tuple(sorted(int(c) for c in (selected_comp_ids or [])))
    _fig_stage_key = '' if selected_stage in (STAGE_ALL, None) else str(selected_stage)

    def _show_team_png(kind, extra=(), payload=None):
        _png = _render_team_figure_png(
            kind, selected_team_t, _fig_season_key, _fig_comp_key,
            _fig_stage_key, extra, FIGURE_CACHE_VERSION,
            team_events_df, team_matches_df, payload)
        if _png:
            st.image(_png, use_container_width=True)

    def _show_team_radar(title, params, values_raw, values_pct, color):
        # The plotted values ride in `extra` (hashed), not just the scope:
        # ~10 floats, and it makes the radar a pure function of its key
        # regardless of how the upstream stat caches key themselves.
        # league_label/season_label are drawn onto the image, so they
        # belong in the key too — season_label derives from
        # selected_season_id, which season_key does not capture.
        _show_team_png(
            'radar',
            extra=(title, tuple(params), color, league_label, season_label),
            payload=(tuple(values_raw), tuple(values_pct)))

    st.subheader(f"Team Style Radars (Percentile Ranks vs {league_label})")
    if selected_team_t in stats_df_raw.index and selected_team_t in stats_df_pct.index:
        offensive_params = ['Goals', 'xG', 'xG per Shot', 'Shots', 'Actions in Box', 'Passes into Box', 'Crosses', 'Dribbles']
        distribution_params = ['Passes', 'Progressive Passes', 'Directness', 'Ball Possession', 'Losses']
        defensive_params = ['Goals Against', 'xG Against', 'xG per Shot Against', 'Shots Against', 'Aerial Duel Win %', 'Defensive Duel Win %', 'Interceptions', 'Fouls', 'PPDA']
        set_piece_params = [
            'Corners', 'xG per Corner', 'Goals per Corner', 'Short Corner %',  # corner cluster
            'Long Throws', 'Long Throw %', 'xG per Long Throw',  # throw-in cluster
            'First Contact %', 'xG per FK Delivery', 'Penalties', 'Non-Pen SP Goals',  # general
        ]
        team_stats_raw = stats_df_raw.loc[selected_team_t]
        team_stats_pct = stats_df_pct.loc[selected_team_t]
        current_league = get_league_label(selected_comp_ids); current_season = season_label

        # Row 1: Offensive + Distribution
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown("**Offensive Radar**")
            valid_offensive_params = [p for p in offensive_params if p in team_stats_raw.index]
            if valid_offensive_params:
                 _show_team_radar("Offensive Radar", valid_offensive_params,
                                  team_stats_raw[valid_offensive_params].tolist(),
                                  team_stats_pct[valid_offensive_params].tolist(),
                                  theme.RADAR_OFFENSIVE)
        with col_r2:
            st.markdown("**Distribution Radar**")
            valid_distribution_params = [p for p in distribution_params if p in team_stats_raw.index]
            if valid_distribution_params:
                 raw_dist_values = team_stats_raw[valid_distribution_params].tolist()
                 try: poss_index = valid_distribution_params.index('Ball Possession'); raw_dist_values[poss_index] = f"{raw_dist_values[poss_index]:.0f}%"
                 except ValueError: pass
                 _show_team_radar("Distribution Radar", valid_distribution_params,
                                  raw_dist_values,
                                  team_stats_pct[valid_distribution_params].tolist(),
                                  theme.RADAR_DISTRIBUTION)

        # Row 2: Defensive + Set Piece
        col_r3, col_r4 = st.columns(2)
        with col_r3:
            st.markdown("**Defensive Radar**")
            valid_defensive_params = [p for p in defensive_params if p in team_stats_raw.index]
            if valid_defensive_params:
                 raw_def_values = team_stats_raw[valid_defensive_params].tolist()
                 try: aerial_idx = valid_defensive_params.index('Aerial Duel Win %'); raw_def_values[aerial_idx] = f"{raw_def_values[aerial_idx]:.0f}%"
                 except ValueError: pass
                 try: def_idx = valid_defensive_params.index('Defensive Duel Win %'); raw_def_values[def_idx] = f"{raw_def_values[def_idx]:.0f}%"
                 except ValueError: pass
                 _show_team_radar("Defensive Radar", valid_defensive_params,
                                  raw_def_values,
                                  team_stats_pct[valid_defensive_params].tolist(),
                                  theme.RADAR_DEFENSIVE)
        with col_r4:
            st.markdown("**Set Piece Radar**")
            if sp_df_raw is not None and not sp_df_raw.empty and selected_team_t in sp_df_raw.index:
                sp_team_raw = sp_df_raw.loc[selected_team_t]
                sp_team_pct = sp_df_pct.loc[selected_team_t]
                valid_sp_params = [p for p in set_piece_params if p in sp_team_raw.index]
                if valid_sp_params:
                    raw_sp_values = sp_team_raw[valid_sp_params].tolist()
                    # Format percentage params with % suffix
                    for _pct_name in ['Short Corner %', 'Long Throw %', 'First Contact %']:
                        try:
                            _idx = valid_sp_params.index(_pct_name)
                            raw_sp_values[_idx] = f"{raw_sp_values[_idx]:.0f}%"
                        except ValueError:
                            pass
                    _show_team_radar("Set Piece Radar", valid_sp_params,
                                     raw_sp_values,
                                     sp_team_pct[valid_sp_params].tolist(),
                                     theme.RADAR_SET_PIECE)
                else:
                    st.info("Set piece data not available.")
            else:
                st.info("Set piece data not available for this team.")
    else:
        st.warning(f"Could not find calculated radar statistics for {selected_team_t}.")

    # ── Season Report (7-dimension dot plots, replicates the Twelve format) ─
    st.divider()
    with st.expander("📊 Season Report — performance across 7 dimensions",
                      expanded=False):
        st.caption(
            "Each row shows every team in the current league/stage as a green dot, "
            f"with **{selected_team_t}** highlighted as a white hexagon. "
            "Values are the team's raw per-match / per-90 numbers."
        )
        _sr_cache_key = (
            f"sr_{','.join(map(str, selected_comp_ids))}"
            f"_{active_season_ids if not isinstance(active_season_ids, list) else ','.join(map(str, active_season_ids))}"
            f"_{selected_stage or 'all'}"
        )
        def _open_team(team):
            st.session_state['nav_team'] = team
            st.rerun()

        render_season_report_section(
            team_events_df, team_matches_df, selected_team_t,
            season_ids=active_season_ids,
            stage=selected_stage,
            cache_key=_sr_cache_key,
            on_team_select=_open_team,
        )

    # Primary Formation XI Graphic
    # =============================================================
    # On-Ball Value & Phases (engine OBV + phases-of-play v1)
    # =============================================================
    st.subheader("On-Ball Value & Phases")
    _team_obv_all = load_obv_viz_data()
    if selected_season_id is not None:
        _prof = _team_obv_all.get('phase_profile')
        _tseason = _team_obv_all.get('team_season')
        if _prof is not None:
            _prof = _prof[(_prof['seasonId'] == selected_season_id)
                          & (_prof['competitionId'].isin(selected_comp_ids))]
        if _tseason is not None:
            _tseason = _tseason[(_tseason['seasonId'] == selected_season_id)
                                & (_tseason['competitionId'].isin(selected_comp_ids))]
        # team id via the phases profile (the scoped events frame drops team.id)
        _team_obv_id = None
        _prof_full = _team_obv_all.get('phase_profile')
        if _prof_full is not None:
            _tid_rows = _prof_full.loc[
                _prof_full['teamName'] == selected_team_t, 'teamId'].dropna()
            if len(_tid_rows):
                _team_obv_id = int(_tid_rows.iloc[0])

        if _tseason is not None and not _tseason.empty and _team_obv_id is not None:
            _show_team_png('obv_categories',
                           payload={'team_season': _tseason,
                                    'team_id': _team_obv_id})
        else:
            st.caption("Team OBV for this season appears after the next engine rebuild.")

        if _prof is not None and not _prof.empty:
            _show_team_png('phase_profile', payload={'profile': _prof})
            with st.expander("How phases are defined (v1)"):
                st.markdown(
                    "- **Buildup / Progression / Finishing** — organized-possession "
                    "segments by pitch third; success = advancing a third "
                    "(finishing: shot or box entry)\n"
                    "- **Fast break** — Wyscout counterattack possessions; "
                    "success = shot or box entry\n"
                    "- **Set piece** — corner / free-kick / penalty possessions; "
                    "success = shot or box entry\n"
                    "- Uncontrolled possessions (≤2 events, no shot) are excluded.\n"
                    "- **OBV per phase** — engine action values summed over the phase.")
        else:
            st.caption("Phase profile for this season appears after the next engine rebuild.")
    else:
        st.caption("On-Ball Value & Phases are per-season views — select a single season.")

    st.subheader("Primary Formation")
    primary_formation = get_team_primary_formation(team_events_df, selected_team_t)
    starting_xi = get_team_starting_xi(team_events_df, selected_team_t)

    col_xi1, col_xi2 = st.columns([1, 1])

    with col_xi1:
        if primary_formation and starting_xi:
            # starting_xi is the render payload; the formation string rides
            # in the key alongside the scope. Both are derived from
            # (team_events_df, team) i.e. the scope key + team, so the
            # scope pins the XI too — the formation is included because it
            # is cheap and makes the key self-evident.
            _show_team_png('formation_xi', extra=(primary_formation,),
                           payload=starting_xi)
        else:
            st.info("Formation data not available for this team.")

    with col_xi2:
        st.write(f"**Formation:** {primary_formation}")

        # Build roster table with unique players
        if starting_xi:
            # Get unique players (same player may appear at multiple positions)
            unique_players = {}
            for pos, player in starting_xi.items():
                pid = player['id']
                if pid not in unique_players:
                    unique_players[pid] = {'name': player['name'], 'positions': [pos], 'id': pid}
                else:
                    unique_players[pid]['positions'].append(pos)

            # Build table data
            roster_data = []
            player_id_list = []
            for pid, pinfo in unique_players.items():
                row = {'Player': pinfo['name'], 'Position': pinfo['positions'][0]}

                # Get age and nationality from player_details
                if pid in player_details_df.index:
                    details = player_details_df.loc[pid]
                    age = _calculate_age(details.get('birthDate'))
                    row['Age'] = int(age) if isinstance(age, (int, float)) and age != "N/A" else "N/A"
                    row['Nationality'] = details.get('passportArea', 'N/A')
                else:
                    row['Age'] = "N/A"
                    row['Nationality'] = "N/A"

                # Get minutes from team_player_minutes_df
                player_mins = team_player_minutes_df[team_player_minutes_df['playerId'] == pid] if not team_player_minutes_df.empty else pd.DataFrame()
                if not player_mins.empty:
                    row['Minutes'] = int(player_mins['totalMinutes'].values[0])
                else:
                    row['Minutes'] = 0

                roster_data.append(row)
                player_id_list.append(pid)

            roster_df = pd.DataFrame(roster_data)
            roster_df = roster_df.sort_values('Minutes', ascending=False)
            # Reorder player_id_list to match sorted dataframe
            player_id_list = [player_id_list[i] for i in roster_df.index] if len(roster_data) > 0 else []
            roster_df = roster_df.reset_index(drop=True)

            st.write("**Squad Roster** (click to view profile):")
            selection = st.dataframe(
                roster_df,
                use_container_width=True,
                on_select="rerun",
                selection_mode="single-row",
                key="team_roster_table",
                hide_index=True,
                column_config=auto_column_config(roster_df)
            )

            # Handle row selection for navigation to Player Profile
            if selection and selection.selection and selection.selection.rows:
                selected_row_idx = selection.selection.rows[0]
                if selected_row_idx < len(player_id_list):
                    selected_player_id = player_id_list[selected_row_idx]
                    st.session_state.selected_player_id = selected_player_id
                    st.session_state.nav_to_profile = True
                    st.session_state.nav_season_id = selected_season_id
                    st.session_state.nav_has_season = True
                    st.rerun()

    st.subheader("Season Shot Maps (Non-Penalty)")
    # Full width, stacked: the half-pitch locks its aspect, so in a half-width
    # column it rendered a few hundred px tall.
    if True:
        st.markdown(f"**Shots FOR {selected_team_t}**")
        _ev = st.plotly_chart(ti.plotly_season_shot_map(team_events_df, team_matches_df, selected_team_t, 'for', height=context_bar.pitch_height()),
                              use_container_width=True, key='ta_shots_for', on_select='rerun',
                              selection_mode='points', config=app._PLOTLY_CFG, theme=None)
        app.open_match_from_selection(_ev)
    if True:
        st.markdown(f"**Shots AGAINST {selected_team_t}**")
        _ev = st.plotly_chart(ti.plotly_season_shot_map(team_events_df, team_matches_df, selected_team_t, 'against', height=context_bar.pitch_height()),
                              use_container_width=True, key='ta_shots_against', on_select='rerun',
                              selection_mode='points', config=app._PLOTLY_CFG, theme=None)
        app.open_match_from_selection(_ev)

    # --- Rolling xG History ---
    st.subheader("Rolling xG (5-Game Average)")
    if True:  # was an expander: a Plotly chart first drawn hidden keeps a collapsed height
        try:
            # Use the stage-filtered events/matches so the rolling
            # series only covers matches in the active stage. Both frames
            # are underscore-prefixed inside, so scope_key is what makes
            # the cache follow the scope — reuse the same triple the
            # figure cache keys on.
            rolling_xg_data_for_plot = calculate_xg_history_data(
                team_events_df, team_matches_df,
                scope_key=(_fig_season_key, _fig_comp_key, _fig_stage_key))
            if not rolling_xg_data_for_plot.empty:
                _ev = st.plotly_chart(ti.plotly_rolling_xg(rolling_xg_data_for_plot, selected_team_t, team_matches_df),
                                      use_container_width=True, key='ta_rolling_xg', on_select='rerun',
                                      selection_mode='points', config=app._PLOTLY_CFG, theme=None)
                app.open_match_from_selection(_ev)
            else:
                st.warning("No data available to calculate xG history.")
        except Exception as e:
            st.error(f"Error loading xG history: {e}")

    st.subheader("Corner Kick Analysis")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown("**Corners from Left Side**")
        # (These two were also the only figures on the page with no
        # plt.close — they leaked until the plt.close('all') at the end of
        # the script. Going through _fig_png_bytes closes them properly.)
        _show_team_png('corner_analysis', extra=('left',))
    with col_c2:
        st.markdown("**Corners from Right Side**")
        _show_team_png('corner_analysis', extra=('right',))

    st.subheader("Season-Long Stats")
    if selected_team_t in team_season_stats and 'corners' in team_season_stats[selected_team_t]:
        st.markdown("**Corner Kick Summary**")
        if selected_stage not in (STAGE_ALL, None):
            st.caption(
                f"⚠️ Showing full-season aggregates; this table is pre-computed "
                f"and doesn't filter to the **{selected_stage}** stage."
            )
        st.dataframe(team_season_stats[selected_team_t]['corners'])
    else:
        st.write("No season-long stats available for this team.")

    # =============================================================
    # Tactical Zone Analysis (Wyscout-style)
    # =============================================================
    st.subheader("Tactical Zone Analysis")

    # 0. Average Player Positions (season) — parity with the Opposition
    # Report: restricted to the primary XI so the map stays readable.
    st.markdown("**Average Player Positions (Season)**")
    try:
        _xi_names = tuple(sorted({p['name'] for p in starting_xi.values()
                                  if p.get('name')})) if starting_xi else None
        _show_team_png('avg_positions', extra=(_xi_names,))
    except Exception as e:
        st.caption(f"Could not render average positions: {e}")

    # 1. Ball Recovery Zones (vs league average)
    st.markdown("**Ball Recovery Zones** (vs League Average)")
    try:
        _show_team_png('zone_heatmap', extra=('recovery',))
    except Exception as e:
        st.caption(f"Could not render recovery zones: {e}")

    # 2. Ball Loss Zones (vs league average)
    st.markdown("**Ball Loss Zones** (vs League Average)")
    try:
        _show_team_png('zone_heatmap', extra=('loss',))
    except Exception as e:
        st.caption(f"Could not render loss zones: {e}")

    # 3. Passing Network (Season)
    st.markdown("**Passing Network (Season)**")
    try:
        _np_payload = None
        _np_pairs = _team_obv_all.get('pairs')
        if (_np_pairs is not None and selected_season_id is not None
                and _team_obv_id is not None):
            _np_pairs = _np_pairs[
                (_np_pairs['seasonId'] == selected_season_id)
                & (_np_pairs['competitionId'].isin(selected_comp_ids))
                & (_np_pairs['teamId'] == _team_obv_id)]
            _np_players = _team_obv_all.get('players')
            if _np_players is not None:
                _np_players = _np_players[
                    (_np_players['seasonId'] == selected_season_id)
                    & (_np_players['competitionId'].isin(selected_comp_ids))]
            if not _np_pairs.empty:
                _np_payload = {'pairs': _np_pairs, 'players': _np_players}
        _net = app.cached_passing_network(_fig_season_key, _fig_comp_key, _fig_stage_key, selected_team_t,
                                          FIGURE_CACHE_VERSION, team_events_df,
                                          _np_payload['pairs'] if _np_payload else None)
        _ev = st.plotly_chart(ti.plotly_passing_network(_net, selected_team_t, height=context_bar.pitch_height()), use_container_width=True,
                              key='ta_passnet', on_select='rerun', selection_mode='points', config=app._PLOTLY_CFG, theme=None)
        app.open_profile_from_selection(_ev, selected_season_id)
    except Exception as e:
        st.caption(f"Could not render passing network: {e}")

    # 4. Defensive Structure
    st.markdown("**Defensive Structure**")
    try:
        _show_team_png('defensive_structure')
    except Exception as e:
        st.caption(f"Could not render defensive structure: {e}")

    # 5. Shot Assists + Dribbles in Final Third — parity with the
    # Opposition Report's section of the same name.
    st.markdown("**Shot Assists & Dribbles in Final Third**")
    try:
        _show_team_png('shot_assists')
    except Exception as e:
        st.caption(f"Could not render shot assists & dribbles: {e}")
