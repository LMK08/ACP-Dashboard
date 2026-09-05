"""Match Analysis view — extracted verbatim from app.py's `elif analysis_type == 'Match Analysis'` branch (2026-09).

Collaborators are read from the running app module at call time (the
pattern opposition_report.py uses), so importing this module never imports
app.py. The binding block at the top of render() IS the page's dependency
list: everything it reads from app.py, nothing else.
"""
import pandas as pd
import streamlit as st
import sys


def _app():
    return sys.modules['__main__']


def render():
    app = _app()
    FIGURE_CACHE_VERSION = app.FIGURE_CACHE_VERSION
    _render_match_figure_png = app._render_match_figure_png
    all_match_data = app.all_match_data
    auto_column_config = app.auto_column_config
    filter_by_league = app.filter_by_league
    get_filtered_events = app.get_filtered_events
    get_season_ids_for_selection = app.get_season_ids_for_selection
    get_season_matches = app.get_season_matches
    league_selector = app.league_selector
    load_obv_viz_data = app.load_obv_viz_data
    logger = app.logger
    match_lineups = app.match_lineups
    matches_summary_df = app.matches_summary_df
    raw_events_df = app.raw_events_df
    season_selector = app.season_selector

    # --- League & Season Selector ---
    selected_comp_ids = league_selector("match_analysis")
    selected_season_id = season_selector("match_analysis", comp_ids=selected_comp_ids)
    active_season_ids = get_season_ids_for_selection(selected_season_id, selected_comp_ids)
    season_matches_df = filter_by_league(get_season_matches(matches_summary_df, active_season_ids), selected_comp_ids).copy()

    # --- Match Selection (Using correct column names) ---
    if 'dateutc' in season_matches_df.columns:
        season_matches_df['display_date'] = pd.to_datetime(season_matches_df['dateutc']).dt.strftime('%Y-%m-%d')
    else: season_matches_df['display_date'] = 'Unknown Date'

    # Create a display-ready gameweek column
    season_matches_df['gw_display'] = "GW " + season_matches_df.get('gameweek', pd.Series(dtype='str')).fillna('?').astype(str)

    # --- Determine stage labels (Promotion League vs Maintenance Stage) ---
    if 'roundId' in season_matches_df.columns and len(season_matches_df) > 0:
        # Group matches by roundId to find the first stage (most matches)
        round_counts = season_matches_df.groupby('roundId').size()
        first_stage_round = round_counts.idxmax()

        # Determine second stage rounds and their labels
        second_stage_rounds = round_counts.drop(first_stage_round, errors='ignore')

        def get_stage_label(row):
            if row['roundId'] == first_stage_round:
                return row['gw_display']  # Regular season: no prefix
            else:
                # Second stage: determine if Promotion or Maintenance
                if len(second_stage_rounds) > 0:
                    min_round = second_stage_rounds.idxmin()
                    is_promotion = row['roundId'] == min_round
                    prefix = "[P] " if is_promotion else "[M] "
                else:
                    prefix = "[S2] "
                return prefix + row['gw_display']

        season_matches_df['gw_display_with_stage'] = season_matches_df.apply(get_stage_label, axis=1)
    else:
        season_matches_df['gw_display_with_stage'] = season_matches_df['gw_display']

    # Build the full display name using the new columns (GW: Teams (Score) - Date)
    season_matches_df['display_name'] = season_matches_df['gw_display_with_stage'] + ": " + \
                                         season_matches_df.get('homeTeamName', '?').fillna('?') + " vs " + \
                                         season_matches_df.get('awayTeamName', '?').fillna('?') + \
                                         " (" + season_matches_df.get('score', '?-?').fillna('?-?') + ") - " + \
                                         season_matches_df['display_date']

    sort_key = 'dateutc' if 'dateutc' in season_matches_df.columns else 'matchId'
    # Sort descending to show newest matches first
    season_matches_df.sort_values(by=[sort_key, 'matchId'], inplace=True, ascending=False, na_position='last')

    # Default to the newest match that HAS event data. The newest fixture
    # in the list can predate its event ingest, which used to open the app
    # on two "No shots found" panels.
    _match_ids_with_events = set(
        get_filtered_events(raw_events_df, active_season_ids, selected_comp_ids)['matchId'].unique())
    _default_match_idx = next(
        (i for i, mid in enumerate(season_matches_df['matchId'].tolist())
         if mid in _match_ids_with_events), 0)
    # Deep link from Home ("Open match report"): open on that match.
    _nav_mid = st.session_state.pop('nav_match_id', None)
    if _nav_mid is not None:
        _ids = season_matches_df['matchId'].tolist()
        if _nav_mid in _ids:
            _default_match_idx = _ids.index(_nav_mid)
    selected_match_display = st.sidebar.selectbox(
        "Select a Match", season_matches_df['display_name'], index=_default_match_idx)
    matching_matches = season_matches_df[season_matches_df['display_name'] == selected_match_display]
    if matching_matches.empty:
        st.error("Selected match not found. Please refresh the page and try again.")
        st.stop()
    selected_match_info = matching_matches.iloc[0]
    selected_match_id = selected_match_info['matchId']

    # --- Display stage badge for the selected match ---
    if 'roundId' in season_matches_df.columns and len(season_matches_df) > 0:
        round_counts = season_matches_df.groupby('roundId').size()
        first_stage_round = round_counts.idxmax()
        second_stage_rounds = round_counts.drop(first_stage_round, errors='ignore')

        current_round_id = selected_match_info.get('roundId')
        if current_round_id == first_stage_round:
            badge_text = "Regular Season"
            badge_bg = "rgba(255,255,255,0.08)"
            badge_fg = "rgba(255,255,255,0.45)"
            badge_border = "rgba(255,255,255,0.12)"
        else:
            if len(second_stage_rounds) > 0:
                min_round = second_stage_rounds.idxmin()
                is_promotion = current_round_id == min_round
                if is_promotion:
                    badge_text = "Promotion League"
                    badge_bg = "rgba(255,255,255,0.12)"
                    badge_fg = "#fff"
                    badge_border = "rgba(255,255,255,0.2)"
                else:
                    badge_text = "Maintenance Stage"
                    badge_bg = "rgba(255,255,255,0.08)"
                    badge_fg = "#fff"
                    badge_border = "rgba(255,255,255,0.25)"
            else:
                badge_text = "Second Stage"
                badge_bg = "rgba(255,255,255,0.08)"
                badge_fg = "rgba(255,255,255,0.45)"
                badge_border = "rgba(255,255,255,0.12)"

        st.sidebar.markdown(
            f'<div style="background:{badge_bg}; color:{badge_fg}; border:1px solid {badge_border}; padding:7px 12px; border-radius:6px; text-align:center; font-weight:600; font-size:0.8rem; margin-top:8px; letter-spacing:0.3px;">{badge_text}</div>',
            unsafe_allow_html=True
        )

    st.header(f"Match Report: {selected_match_info['homeTeamName']} vs {selected_match_info['awayTeamName']}")



    match_data = all_match_data.get(selected_match_id)
    if match_data:
        st.subheader("Shot Maps")
        import team_interactive as ti
        import context_bar

        # --- Get the match events ONCE ---
        match_events_df = raw_events_df[raw_events_df['matchId'] == selected_match_id]

        # Every pitch figure on this page goes through _show_match_png ->
        # _render_match_figure_png, which caches the PNG bytes on
        # (kind, matchId, team, FIGURE_CACHE_VERSION). See that function
        # for why those four components are the complete key. int() the
        # matchId so a numpy int64 and a Python int can't key separately.
        _mid = int(selected_match_id)

        # Engine OBV aggregates for this match (None-safe: files may not
        # exist until the next engine rebuild ships them)
        _obv_all = load_obv_viz_data()

        def _slice_obv(df):
            if df is None or df.empty:
                return None
            s = df[df['matchId'] == _mid]
            return s if not s.empty else None

        _obv_match = {
            'minute': _slice_obv(_obv_all.get('minute')),
            'pairs': _slice_obv(_obv_all.get('pairs')),
            'players': _slice_obv(_obv_all.get('players')),
        }

        def _show_match_png(kind, team=None, lineup=None):
            _png = _render_match_figure_png(
                kind, _mid, team, FIGURE_CACHE_VERSION,
                match_events_df, selected_match_info, lineup,
                _obv=_obv_match)
            if _png:
                st.image(_png, use_container_width=True)

        # ---- Merge engine OBV into the match stats tables ----
        # Display-time COPIES (the cached all_match_data frames must not
        # be mutated); used by both the page and the match report PDF.
        _disp_team_stats = {k: v.copy() if isinstance(v, pd.DataFrame) else v
                            for k, v in (match_data.get('team_stats') or {}).items()}
        _ps_src = match_data.get('player_stats') or {}
        _disp_player_stats = {k: v.copy() if isinstance(v, pd.DataFrame) else v
                              for k, v in _ps_src.items()}
        _obv_cols_ok = {'player.id', 'player.name',
                        'team.id', 'team.name'}.issubset(match_events_df.columns)
        if _obv_match['players'] is not None and _obv_cols_ok:
            _id_name = (match_events_df
                        .dropna(subset=['player.id', 'player.name'])
                        .drop_duplicates('player.id'))
            _id2name = dict(zip(_id_name['player.id'].astype(int),
                                _id_name['player.name']))
            _pobv = _obv_match['players'].copy()
            _pobv['name'] = _pobv['playerId'].astype(int).map(_id2name)
            _name2obv = (_pobv.dropna(subset=['name'])
                         .groupby('name')['obv'].sum().to_dict())
            for _side, _df in _disp_player_stats.items():
                if isinstance(_df, pd.DataFrame) and not _df.empty:
                    _vals = [_name2obv.get(nm) for nm in _df.index]
                    _df.insert(min(1, len(_df.columns)), 'OBV',
                               [f'{v:+.2f}' if v is not None else '-'
                                for v in _vals])
            # team totals -> a row atop the General table
            _tid2name = (match_events_df.dropna(subset=['team.id', 'team.name'])
                         .drop_duplicates('team.id'))
            _tid2name = dict(zip(_tid2name['team.id'].astype(int),
                                 _tid2name['team.name']))
            _team_obv = (_obv_match['players'].groupby('teamId')['obv'].sum())
            _gen = _disp_team_stats.get('General')
            if isinstance(_gen, pd.DataFrame):
                _row = {c: '-' for c in _gen.columns}
                for _tid, _v in _team_obv.items():
                    _nm = _tid2name.get(int(_tid))
                    if _nm in _row:
                        _row[_nm] = f'{_v:+.2f}'
                _gen.loc['On-Ball Value (engine)'] = _row
                _disp_team_stats['General'] = _gen.reindex(
                    ['On-Ball Value (engine)']
                    + [i for i in _gen.index if i != 'On-Ball Value (engine)'])

        # Interactive, stacked at full width (the same drawing as the team and
        # player maps; the static PNG stays for the match-report PDF).
        for _side, _team in (('home', selected_match_info['homeTeamName']),
                             ('away', selected_match_info['awayTeamName'])):
            _ev = st.plotly_chart(ti.plotly_match_shot_map(match_events_df, selected_match_info, _team, height=context_bar.pitch_height()),
                                  use_container_width=True, key=f'ma_shots_{_side}', on_select='rerun',
                                  selection_mode='points', config=app._PLOTLY_CFG, theme=None)
            app.open_profile_from_selection(_ev, selected_season_id)

        # --- NEW: Shot Details Tables ---
        st.markdown("---") # Add a separator
        st.subheader("Shot Details")

        def get_shot_table(df, team_name):
            """Helper function to create the shot detail table."""
            shots = df[
                (df.get('team.name') == team_name) & 
                (df.get('type.primary').isin(['shot', 'penalty']))
            ].copy()

            if shots.empty:
                return pd.DataFrame(columns=["#", "Shooter", "Minute", "Body Part", "xG", "PSxG"])

            # Select and rename columns
            # Use .get() for safety, in case a column is missing
            shots_table = pd.DataFrame()
            shots_table['Shooter'] = shots.get('player.name', 'N/A')
            shots_table['Minute'] = shots.get('minute', 0).astype(int)
            # Use .get() on the dictionary-like column 'shot.bodyPart'
            shots_table['Body Part'] = shots.get('shot.bodyPart', {}).apply(lambda x: x.get('name', 'unknown') if isinstance(x, dict) else x)
            shots_table['xG'] = shots.get('shot.xg', 0).fillna(0).round(2)
            shots_table['PSxG'] = shots.get('shot.postShotXg', 0).fillna(0).round(2)

            # Add the shot number (#)
            shots_table.reset_index(drop=True, inplace=True)
            shots_table.index = shots_table.index + 1
            shots_table.reset_index(inplace=True)
            shots_table = shots_table.rename(columns={'index': '#'})

            return shots_table.set_index('#')

        col1_table, col2_table = st.columns(2)
        with col1_table:
            st.markdown(f"**{selected_match_info['homeTeamName']}**")
            home_shots_table = get_shot_table(match_events_df, selected_match_info['homeTeamName'])
            st.dataframe(home_shots_table, column_config=auto_column_config(home_shots_table))

        with col2_table:
            st.markdown(f"**{selected_match_info['awayTeamName']}**")
            away_shots_table = get_shot_table(match_events_df, selected_match_info['awayTeamName'])
            st.dataframe(away_shots_table, column_config=auto_column_config(away_shots_table))
        # --- END NEW SECTION ---

        # --- Momentum (engine OBV) ---
        st.subheader("Momentum")
        if _obv_match['minute'] is not None:
            try:
                _show_match_png('obv_momentum')
            except Exception as e:
                st.warning(f"Could not generate momentum chart: {e}")
        else:
            st.caption("Momentum uses the engine's on-ball values — this match "
                       "appears after the next engine rebuild.")

        # --- xG Flowchart ---
        st.subheader("xG Flowchart")
        match_events_df = raw_events_df[raw_events_df['matchId'] == selected_match_id]
        if not match_events_df.empty:
            try:
                _show_match_png('xg_flowchart')
            except Exception as e:
                st.warning(f"Could not generate xG flowchart: {e}")
        else:
            st.info("No event data found for flowchart.")

        st.subheader("Team Stats")
        if _disp_team_stats:
            for stat_category, df in _disp_team_stats.items():
                st.markdown(f"**{stat_category}**")
                if isinstance(df, pd.DataFrame): st.dataframe(df, column_config=auto_column_config(df))
                else: st.warning(f"Data for '{stat_category}' is not a DataFrame.")
        else: st.warning("Team stats data not found.")

        st.subheader("Player Stats")
        if 'home' in _disp_player_stats and 'away' in _disp_player_stats:
            st.markdown(f"**{selected_match_info['homeTeamName']}**")
            if isinstance(_disp_player_stats['home'], pd.DataFrame): st.dataframe(_disp_player_stats['home'])
            else: st.warning("Home player stats data not a DataFrame.")
            st.markdown(f"**{selected_match_info['awayTeamName']}**")
            if isinstance(_disp_player_stats['away'], pd.DataFrame): st.dataframe(_disp_player_stats['away'])
            else: st.warning("Away player stats data not a DataFrame.")
        else: st.warning("Player stats data not found.")

        # =============================================================
        # Tactical Analysis (Wyscout-style pitch visualizations)
        # =============================================================
        st.subheader("Tactical Analysis")
        home_team = selected_match_info['homeTeamName']
        away_team = selected_match_info['awayTeamName']

        # Get lineup/substitution data for this match (if available)
        match_lineup_data = match_lineups.get(selected_match_id, {}) if match_lineups else {}
        home_lineup = match_lineup_data.get(home_team)
        away_lineup = match_lineup_data.get(away_team)

        # 1. Average Player Positions
        st.markdown("**Average Player Positions**")
        col_ap1, col_ap2 = st.columns(2)
        with col_ap1:
            try:
                _show_match_png('avg_positions', home_team, home_lineup)
            except Exception as e:
                st.caption(f"Could not render: {e}")
        with col_ap2:
            try:
                _show_match_png('avg_positions', away_team, away_lineup)
            except Exception as e:
                st.caption(f"Could not render: {e}")

        # 2. Average Positions by Substitution Phase
        st.markdown(f"**{home_team} — Avg Positions by Phase**")
        try:
            _show_match_png('avg_positions_by_subs', home_team, home_lineup)
        except Exception as e:
            st.caption(f"Could not render: {e}")

        st.markdown(f"**{away_team} — Avg Positions by Phase**")
        try:
            _show_match_png('avg_positions_by_subs', away_team, away_lineup)
        except Exception as e:
            st.caption(f"Could not render: {e}")

        # 3. Passing Network
        st.markdown("**Passing Network**")
        col_pn1, col_pn2 = st.columns(2)
        with col_pn1:
            try:
                _show_match_png('passing_network', home_team)
            except Exception as e:
                st.caption(f"Could not render: {e}")
        with col_pn2:
            try:
                _show_match_png('passing_network', away_team)
            except Exception as e:
                st.caption(f"Could not render: {e}")

        # 4. Ball Recoveries & Losses
        st.markdown("**Ball Recoveries & Losses**")
        tac_team = st.selectbox(
            "Select team for recovery/loss maps",
            [home_team, away_team],
            key="tac_recovery_team",
        )
        col_rl1, col_rl2 = st.columns(2)
        with col_rl1:
            try:
                # tac_team is the selectbox above — it IS the team component
                # of the key, so the toggle needs nothing extra.
                _show_match_png('recovery_map', tac_team)
            except Exception as e:
                st.caption(f"Could not render: {e}")
        with col_rl2:
            try:
                _show_match_png('loss_map', tac_team)
            except Exception as e:
                st.caption(f"Could not render: {e}")

        # 5. Defensive Duels
        st.markdown("**Defensive Duels**")
        col_dd1, col_dd2 = st.columns(2)
        with col_dd1:
            try:
                _show_match_png('defensive_duels', home_team)
            except Exception as e:
                st.caption(f"Could not render: {e}")
        with col_dd2:
            try:
                _show_match_png('defensive_duels', away_team)
            except Exception as e:
                st.caption(f"Could not render: {e}")

        # 6. Shot Assists + Dribbles in Final Third
        st.markdown("**Shot Assists & Dribbles in Final Third**")
        col_sa1, col_sa2 = st.columns(2)
        with col_sa1:
            try:
                _show_match_png('shot_assists', home_team)
            except Exception as e:
                st.caption(f"Could not render: {e}")
        with col_sa2:
            try:
                _show_match_png('shot_assists', away_team)
            except Exception as e:
                st.caption(f"Could not render: {e}")

        # =============================================================
        # Download Match Report (PDF)
        # =============================================================
        st.divider()
        st.subheader("Download Match Report")
        _pdf_key = f"match_report_pdf_{_mid}"

        def _match_fig_or_none(kind, team=None, lineup=None):
            """Same cached renderer the page uses; None on any failure so
            one broken figure never blocks the whole report."""
            try:
                return _render_match_figure_png(
                    kind, _mid, team, FIGURE_CACHE_VERSION,
                    match_events_df, selected_match_info, lineup,
                    _obv=_obv_match)
            except Exception:
                return None

        if st.button("Build match report PDF", key=f"build_{_pdf_key}"):
            with st.spinner("Assembling match report..."):
                from match_report_pdf import generate_match_report_pdf
                _report_figures = {
                    'obv_momentum': _match_fig_or_none('obv_momentum'),
                    'shotmap_home': _match_fig_or_none('shotmap', home_team),
                    'shotmap_away': _match_fig_or_none('shotmap', away_team),
                    'xg_flowchart': (_match_fig_or_none('xg_flowchart')
                                     if not match_events_df.empty else None),
                    'avg_positions_home': _match_fig_or_none('avg_positions', home_team, home_lineup),
                    'avg_positions_away': _match_fig_or_none('avg_positions', away_team, away_lineup),
                    'avg_positions_by_subs_home': _match_fig_or_none('avg_positions_by_subs', home_team, home_lineup),
                    'avg_positions_by_subs_away': _match_fig_or_none('avg_positions_by_subs', away_team, away_lineup),
                    'passing_network_home': _match_fig_or_none('passing_network', home_team),
                    'passing_network_away': _match_fig_or_none('passing_network', away_team),
                    'defensive_duels_home': _match_fig_or_none('defensive_duels', home_team),
                    'defensive_duels_away': _match_fig_or_none('defensive_duels', away_team),
                    'shot_assists_home': _match_fig_or_none('shot_assists', home_team),
                    'shot_assists_away': _match_fig_or_none('shot_assists', away_team),
                    'recovery_map_home': _match_fig_or_none('recovery_map', home_team),
                    'recovery_map_away': _match_fig_or_none('recovery_map', away_team),
                    'loss_map_home': _match_fig_or_none('loss_map', home_team),
                    'loss_map_away': _match_fig_or_none('loss_map', away_team),
                }
                try:
                    st.session_state[_pdf_key] = generate_match_report_pdf(
                        selected_match_info,
                        _report_figures,
                        team_stats=_disp_team_stats,
                        player_stats=_disp_player_stats,
                        shot_details={'home': home_shots_table,
                                      'away': away_shots_table},
                    )
                except Exception as e:
                    st.error(f"Could not build the match report PDF: {e}")
                    logger.exception("match report PDF build failed")

        if st.session_state.get(_pdf_key):
            _fname = (f"{selected_match_info['homeTeamName']}_v_"
                      f"{selected_match_info['awayTeamName']}_"
                      f"{selected_match_info.get('display_date', '')}"
                      f"_match_report.pdf").replace(' ', '_').replace('/', '-')
            st.download_button(
                "Download match report (PDF)",
                data=st.session_state[_pdf_key],
                file_name=_fname,
                mime="application/pdf",
                key=f"dl_{_pdf_key}",
            )

    else:
         st.warning(f"No detailed match data found for Match ID {selected_match_id}.")
