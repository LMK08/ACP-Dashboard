"""Player Profile view — extracted verbatim from app.py's `elif analysis_type == 'Player Profile'` branch (2026-09).

Collaborators are read from the running app module at call time (the
pattern opposition_report.py uses), so importing this module never imports
app.py. The binding block at the top of render() IS the page's dependency
list: everything it reads from app.py, nothing else.
"""
from pathlib import Path
import datetime
import hashlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pitch_visualizations as pv
import plotly.graph_objects as go
import streamlit as st
import sys

import context_bar
import eur_interval_ui


def _app():
    return sys.modules['__main__']


def render():
    app = _app()
    COMPETITIONS = app.COMPETITIONS
    DEFENSIVE_METRICS = app.DEFENSIVE_METRICS
    DEFR_DISPLAY_METRICS = app.DEFR_DISPLAY_METRICS
    DEFR_RADAR_MAP = app.DEFR_RADAR_MAP
    DRIBBLING_METRICS = app.DRIBBLING_METRICS
    ENGINE_DISPLAY_METRICS = app.ENGINE_DISPLAY_METRICS
    GOALKEEPING_METRICS = app.GOALKEEPING_METRICS
    INVERT_METRICS = app.INVERT_METRICS
    MPL_LOCK = app.MPL_LOCK
    OUR_TEAM = app.OUR_TEAM
    OUTPUT_METRICS = app.OUTPUT_METRICS
    PASSING_METRICS = app.PASSING_METRICS
    POSITION_GROUPS = app.POSITION_GROUPS
    RADAR_HIDDEN_METRICS = app.RADAR_HIDDEN_METRICS
    RATINGS_EXPLAINER_MD = app.RATINGS_EXPLAINER_MD
    SEASON_ID_MAP = app.SEASON_ID_MAP
    STATS_CACHE_VERSION = app.STATS_CACHE_VERSION
    THOUSANDTHS_METRICS = app.THOUSANDTHS_METRICS
    WEIGHTS = app.WEIGHTS
    _ATTACK_TEMPLATE_ROLES = app._ATTACK_TEMPLATE_ROLES
    _build_player_season_perf_table = app._build_player_season_perf_table
    _calculate_age = app._calculate_age
    _canonical_engine_role = app._canonical_engine_role
    _compute_peer_density_stack = app._compute_peer_density_stack
    _cvi_position_group = app._cvi_position_group
    _engine_role_is_attacking = app._engine_role_is_attacking
    _next_shot_id_by_match = app._next_shot_id_by_match
    _render_acp_index_card_png = app._render_acp_index_card_png
    _role_key_stats = app._role_key_stats
    _season_id_list = app._season_id_list
    all_match_data = app.all_match_data
    auto_column_config = app.auto_column_config
    build_player_priors_lookup = app.build_player_priors_lookup
    calculate_all_player_stats = app.calculate_all_player_stats
    calculate_player_percentiles_and_scores = app.calculate_player_percentiles_and_scores
    competition_for_season = app.competition_for_season
    compute_career_cvi = app.compute_career_cvi
    compute_cvi_columns = app.compute_cvi_columns
    compute_market_features = app.compute_market_features
    create_player_shotmap = app.create_player_shotmap
    create_radar_with_distributions = app.create_radar_with_distributions
    cvi_to_projected_eur = app.cvi_to_projected_eur
    filter_by_league = app.filter_by_league
    fmt_val = app.fmt_val
    get_all_players_minutes_by_position = app.get_all_players_minutes_by_position
    get_filtered_events = app.get_filtered_events
    get_player_match_stats = app.get_player_match_stats
    get_player_minutes_by_position = app.get_player_minutes_by_position
    get_scoped_style = app.get_scoped_style
    get_scoped_tendencies = app.get_scoped_tendencies
    get_season_events = app.get_season_events
    get_season_ids_for_selection = app.get_season_ids_for_selection
    get_season_matches = app.get_season_matches
    get_season_player_minutes = app.get_season_player_minutes
    league_selector = app.league_selector
    load_and_score_player_stats = app.load_and_score_player_stats
    load_box_passes = app.load_box_passes
    load_gpa_values = app.load_gpa_values
    load_player_details = app.load_player_details
    load_player_engine = app.load_player_engine
    load_eur_calibration = app.load_eur_calibration
    engine_rows_for_scope = app.engine_rows_for_scope
    logger = app.logger
    make_opta_team_strength_lookup = app.make_opta_team_strength_lookup
    matches_summary_df = app.matches_summary_df
    merge_defr_values_into_stats = app.merge_defr_values_into_stats
    merge_gpa_values_into_stats = app.merge_gpa_values_into_stats
    most_recent_season_for_player = app.most_recent_season_for_player
    mpl_box_passes_map = app.mpl_box_passes_map
    player_minutes_data = app.player_minutes_data
    plotly_box_passes_map = app.plotly_box_passes_map
    plotly_shot_map = app.plotly_shot_map
    raw_events_df = app.raw_events_df
    render_tendencies_panel = app.render_tendencies_panel
    scipy = app.scipy
    season_selector = app.season_selector
    player_stats_with_scores_df = app.player_stats_with_scores_df
    __file__ = app.__file__


    # (The season carried by a cross-page bridge is applied in app.py's prelude,
    #  before the context bar draws its widget — a widget key cannot be set after.)

    # --- League & Season Selector ---
    selected_comp_ids = league_selector("player_profile")
    selected_season_id = season_selector("player_profile", include_all_seasons=True, comp_ids=selected_comp_ids)
    active_season_ids = get_season_ids_for_selection(selected_season_id, selected_comp_ids)
    profile_season_changed = (selected_season_id != st.session_state.player_profile_last_season)
    st.session_state.player_profile_last_season = selected_season_id
    profile_events_df = get_filtered_events(raw_events_df, active_season_ids, selected_comp_ids)
    profile_matches_df = filter_by_league(get_season_matches(matches_summary_df, active_season_ids), selected_comp_ids)
    profile_player_minutes_df = get_season_player_minutes(player_minutes_data, active_season_ids, comp_ids=selected_comp_ids)

    # --- 1. Load All Necessary Data ---
    player_details_df = load_player_details()

    try:
        with st.spinner("Calculating player statistics (this may take a moment on first load)..."):
            player_stats_df, player_stats_with_scores_df = load_and_score_player_stats(
                profile_events_df, profile_player_minutes_df, selected_season_id, active_season_ids, selected_comp_ids
            )
    except Exception as e:
        st.error(f"An error occurred calculating overall player stats: {e}")
        logger.exception("Error in calculate_all_player_stats")
        player_stats_df = pd.DataFrame()
        player_stats_with_scores_df = pd.DataFrame()

    if player_stats_df.empty or player_details_df.empty or player_stats_with_scores_df.empty:
        st.warning("Player data not available for this selection. Early in a season this is expected — "
                   "player stats appear once Wyscout event/minutes data starts flowing. "
                   "Try selecting a previous season, or check back after the next data refresh.")
        st.stop()

    # --- 2. Player Selector ---
    st.sidebar.subheader("Player Analysis Options")

    # FIX: Include 'playerId' in the list dataframe so we can grab it later
    # (Added 'playerId' to the list of columns below)
    player_list_df = player_stats_with_scores_df[['playerId', 'playerName', 'teamName', 'totalMinutes']].sort_values(by='totalMinutes', ascending=False)

    # Create unique display names
    player_list_df['display_name'] = player_list_df['playerName'].astype(str) + " (" + player_list_df['teamName'].astype(str) + ", " + pd.to_numeric(player_list_df['totalMinutes'], errors='coerce').fillna(0).astype(int).astype(str) + " min)"

    # If navigating from another section, set the player selectbox value directly
    if st.session_state.selected_player_id is not None:
        sorted_player_ids = player_list_df['playerId'].tolist()
        target_id = st.session_state.selected_player_id
        for i, pid in enumerate(sorted_player_ids):
            if int(pid) == int(target_id):
                st.session_state['player_profile_selector'] = player_list_df['display_name'].iloc[i]
                st.session_state.player_profile_current_id = int(target_id)
                break
        st.session_state.selected_player_id = None
    # Re-seed the selector when the season changed (the stored display
    # name may not exist in the new list) OR when the widget's state is
    # gone because this page wasn't rendered last run — Streamlit drops
    # widget state for widgets absent from a run, so a page round-trip
    # used to reset the selection to the top of the league-wide list.
    elif profile_season_changed or 'player_profile_selector' not in st.session_state:
        _seeded = False
        target_id = st.session_state.player_profile_current_id
        if target_id is not None:
            for i, pid in enumerate(player_list_df['playerId'].tolist()):
                if int(pid) == int(target_id):
                    st.session_state['player_profile_selector'] = player_list_df['display_name'].iloc[i]
                    _seeded = True
                    break
        if not _seeded:
            _cur = st.session_state.get('player_profile_selector')
            if _cur is not None and _cur not in set(player_list_df['display_name']):
                st.session_state.pop('player_profile_selector', None)
            # First visit (or the remembered player isn't in this scope):
            # open on our own club's most-used player rather than whoever
            # tops the league-wide minutes list.
            if 'player_profile_selector' not in st.session_state:
                _our_rows = player_list_df[player_list_df['teamName'].astype(str) == OUR_TEAM]
                if not _our_rows.empty:
                    st.session_state['player_profile_selector'] = _our_rows['display_name'].iloc[0]

    selected_player_display = st.sidebar.selectbox(
        "Select Player:",
        player_list_df['display_name'],
        key="player_profile_selector"
    )

    try:
        # FIX: Get the UNIQUE ID corresponding to the selected display name
        # (We use .values[0] to grab the actual integer ID)
        selected_player_id = player_list_df[player_list_df['display_name'] == selected_player_display]['playerId'].values[0]
        st.session_state.player_profile_current_id = int(selected_player_id)

        # FIX: Filter the main dataframe by ID, not by Name
        # This ensures we get the exact Miguel Lopes the user clicked on
        player_data_row = player_stats_with_scores_df[player_stats_with_scores_df['playerId'] == selected_player_id]

        # Extract stats series
        player_per_90_stats = player_data_row.iloc[0] 

        # Define player_id variable for use in other sections
        player_id = selected_player_id

        # Load Bio
        player_bio = player_details_df.loc[player_id] if player_id in player_details_df.index else pd.Series(dtype='object')
        total_minutes = player_per_90_stats.get('totalMinutes', 0)

        # Also update the 'selected_player_name' variable for the Match Log function
        selected_player_name = player_per_90_stats.get('playerName')

    except Exception as e:
        st.error(f"Could not load data for {selected_player_display}. Error: {e}")
        st.stop()

    # --- 3. Get Player's Match Log ---
    player_match_log_df = get_player_match_stats(selected_player_name, all_match_data, profile_matches_df, season_id=selected_season_id)

    # Enrich match log with per-match xA (xAOP/xASP) and xT (xTOP/xTSP) from raw events
    if not player_match_log_df.empty and not profile_events_df.empty:
        try:
            _p_events = profile_events_df[profile_events_df['player.name'] == selected_player_name].copy()
            if not _p_events.empty:
                # Per-match xA: find shot assists, map to shot xG
                _p_events['shot_event_id'] = np.where(_p_events['shot.xg'].notna(), _p_events['id'], np.nan)
                _p_events['next_shot_id'] = _next_shot_id_by_match(_p_events)
                _shot_xg_map = _p_events[_p_events['shot.xg'].notna()].set_index('id')['shot.xg'].to_dict()
                # Get all events in matches this player played (need all players for shot assists)
                _player_match_ids = _p_events['matchId'].unique()
                _match_events = profile_events_df[profile_events_df['matchId'].isin(_player_match_ids)].copy()
                _match_events['shot_event_id'] = np.where(_match_events['shot.xg'].notna(), _match_events['id'], np.nan)
                _match_events['next_shot_id'] = _next_shot_id_by_match(_match_events)
                _all_shot_xg = _match_events[_match_events['shot.xg'].notna()].set_index('id')['shot.xg'].to_dict()
                _assists = _match_events[
                    (_match_events['player.name'] == selected_player_name) &
                    (_match_events.get('type.secondary', pd.Series(dtype='object')).apply(lambda x: isinstance(x, (list, np.ndarray)) and 'shot_assist' in x))
                ].copy()
                _assists['xA'] = _assists['next_shot_id'].map(_all_shot_xg)
                _sp_types = ['corner', 'free_kick', 'throw_in', 'goal_kick']
                _assists['_xa_type'] = np.where(_assists['type.primary'].isin(_sp_types), 'xASP', 'xAOP')
                # Aggregate per match
                _xa_per_match = _assists.groupby(['matchId', '_xa_type'])['xA'].sum().unstack(fill_value=0).reset_index()
                for _c in ['xAOP', 'xASP']:
                    if _c not in _xa_per_match.columns:
                        _xa_per_match[_c] = 0.0
                _xa_total = _assists.groupby('matchId')['xA'].sum().reset_index().rename(columns={'xA': 'xA_total'})
                _xa_per_match = _xa_per_match.merge(_xa_total, on='matchId', how='left')

                # Per-match xT: calculate from passes/touches/accelerations + set pieces
                _xt_data = [[0.01,0.01,0.01,0.01,0.01,0.01,0.02,0.02,0.03,0.03,0.04,0.04],[0.01,0.01,0.01,0.01,0.01,0.02,0.02,0.02,0.03,0.04,0.05,0.05],[0.01,0.01,0.01,0.01,0.01,0.02,0.02,0.02,0.03,0.05,0.06,0.06],[0.01,0.01,0.01,0.01,0.01,0.02,0.02,0.02,0.04,0.11,0.26,0.26],[0.01,0.01,0.01,0.01,0.01,0.02,0.02,0.02,0.04,0.11,0.26,0.26],[0.01,0.01,0.01,0.01,0.01,0.02,0.02,0.02,0.03,0.05,0.06,0.06],[0.01,0.01,0.01,0.01,0.01,0.02,0.02,0.02,0.03,0.04,0.05,0.05],[0.01,0.01,0.01,0.01,0.01,0.01,0.02,0.02,0.03,0.03,0.04,0.04]]
                _xt_grid = np.array(_xt_data); _r, _c = _xt_grid.shape
                _sp_xt = ['corner', 'free_kick', 'throw_in']
                _moves = _p_events[_p_events['type.primary'].isin(['pass', 'touch', 'acceleration'] + _sp_xt)].copy()
                _suc_pass = (_moves['type.primary'] == 'pass') & (_moves.get('pass.accurate') == True)
                _other_suc = _moves['type.primary'].isin(['touch', 'acceleration'] + _sp_xt)
                _moves = _moves[_suc_pass | _other_suc]
                _is_pass_like = _moves['type.primary'].isin(['pass'] + _sp_xt)
                _moves['_ex'] = np.where(_is_pass_like, _moves.get('pass.endLocation.x'), _moves.get('carry.endLocation.x'))
                _moves['_ey'] = np.where(_is_pass_like, _moves.get('pass.endLocation.y'), _moves.get('carry.endLocation.y'))
                _moves = _moves.dropna(subset=['_ex', '_ey'])
                _moves['_sc'] = np.clip((_moves['location.x'].astype(float).fillna(0) / 100 * _c).astype(int), 0, _c-1)
                _moves['_sr'] = np.clip((_moves['location.y'].astype(float).fillna(0) / 100 * _r).astype(int), 0, _r-1)
                _moves['_ec'] = np.clip((_moves['_ex'].astype(float).fillna(0) / 100 * _c).astype(int), 0, _c-1)
                _moves['_er'] = np.clip((_moves['_ey'].astype(float).fillna(0) / 100 * _r).astype(int), 0, _r-1)
                _moves['_xT'] = _xt_grid[_moves['_er'].values, _moves['_ec'].values] - _xt_grid[_moves['_sr'].values, _moves['_sc'].values]
                _pos_xt = _moves[_moves['_xT'] > 0].copy()
                _pos_xt['_xt_type'] = np.where(_pos_xt['type.primary'].isin(_sp_xt), 'xTSP', 'xTOP')
                _xt_per_match = _pos_xt.groupby(['matchId', '_xt_type'])['_xT'].sum().unstack(fill_value=0).reset_index()
                for _c_name in ['xTOP', 'xTSP']:
                    if _c_name not in _xt_per_match.columns:
                        _xt_per_match[_c_name] = 0.0
                _xt_total = _pos_xt.groupby('matchId')['_xT'].sum().reset_index().rename(columns={'_xT': 'xT_total'})
                _xt_per_match = _xt_per_match.merge(_xt_total, on='matchId', how='left')

                # Map matchId to match log rows via Date + opponent lookup
                _mid_map = profile_matches_df.set_index('matchId')['dateutc'].to_dict()
                _xa_per_match['_date'] = _xa_per_match['matchId'].map(_mid_map)
                _xa_per_match['_date'] = pd.to_datetime(_xa_per_match['_date'], errors='coerce').apply(lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else 'N/A')
                _xt_per_match['_date'] = _xt_per_match['matchId'].map(_mid_map)
                _xt_per_match['_date'] = pd.to_datetime(_xt_per_match['_date'], errors='coerce').apply(lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else 'N/A')

                # Merge into match log
                player_match_log_df = player_match_log_df.merge(
                    _xa_per_match[['_date', 'xAOP', 'xASP']].rename(columns={'_date': 'Date'}),
                    on='Date', how='left'
                )
                player_match_log_df = player_match_log_df.merge(
                    _xt_per_match[['_date', 'xTOP', 'xTSP']].rename(columns={'_date': 'Date'}),
                    on='Date', how='left'
                )
                # Round and fill
                for _mc in ['xAOP', 'xASP', 'xTOP', 'xTSP']:
                    if _mc in player_match_log_df.columns:
                        player_match_log_df[_mc] = player_match_log_df[_mc].fillna(0).round(2)
        except Exception as e:
            print(f"Warning: Could not enrich match log with xA/xT: {e}")

    # --- 4. Display Player Bio ---
    current_team = player_per_90_stats.get('teamName', 'N/A')
    current_pos = player_per_90_stats.get('primaryPosition', 'N/A')

    # If 'Unknown', try to force a lookup in the minutes file
    if (current_team in ['Unknown', 'N/A'] or current_pos in ['Unknown', 'N/A']) and not profile_player_minutes_df.empty:
        try:
            pid_int = int(player_id)
            min_row = profile_player_minutes_df[profile_player_minutes_df['playerId'] == pid_int]
            if not min_row.empty:
                current_team = min_row.iloc[0]['teamName']
                current_pos = min_row.iloc[0]['primaryPosition']
        except Exception:
            pass # Fallback to original values if lookup fails

    st.header(f"{player_per_90_stats.get('playerName', 'N/A')}")

    # ---- Compute Transfer Value primitives ONCE per page load ----
    # Used twice: (a) compact projected / true / Δ metrics in the
    # Player Information bio row below, (b) full CVI breakdown +
    # Market Context detail section just above Career Trajectory.
    _tv_age = _calculate_age(player_bio.get('birthDate')) \
               if isinstance(player_bio, pd.Series) else None
    _tv_comp_id = competition_for_season(selected_season_id) \
                   if selected_season_id else None
    _tv_player_row = player_data_row.iloc[0] if not player_data_row.empty else None
    _tv_single = (player_data_row.copy()
                   if not player_data_row.empty else None)
    if _tv_single is not None and 'Total Value' not in _tv_single.columns:
        # GPA merge may not be present in profile-mode stats; backfill.
        try:
            _gpa = load_gpa_values()
            if _gpa is not None and not _gpa.empty:
                _r = _gpa[(_gpa['playerId'] == player_id)
                           & (_gpa['seasonId'] == selected_season_id)]
                if not _r.empty:
                    _val_col = next((c for c in ('Total Value',
                                      'total_v_per_90')
                                      if c in _r.columns), None)
                    if _val_col:
                        _tv_single['Total Value'] = float(_r[_val_col].iloc[0])
        except Exception:
            pass

    _tv_cvi_block = pd.DataFrame()
    if _tv_single is not None and 'primaryPosition' in _tv_single.columns:
        try:
            # Build empirical-Bayes prior for THIS player based on
            # their strictly-prior-season career data. Pulls the
            # same perf_table used for Career CVI below so we only
            # pay the cost once.
            _tv_prior_lookup = None
            try:
                _pt = _build_player_season_perf_table(
                    load_gpa_values(),
                    profile_player_minutes_df
                      if 'profile_player_minutes_df' in dir() else None,
                )
                _prior_map = build_player_priors_lookup(
                    _pt, selected_season_id,
                ) if selected_season_id is not None else {}
                _tv_prior_lookup = lambda pid: _prior_map.get(int(pid)) \
                                                if pid is not None else None
            except Exception:
                _tv_prior_lookup = None
            _tv_cvi_block = compute_cvi_columns(
                _tv_single,
                age_lookup=lambda pid: _tv_age,
                comp_id_lookup=lambda pid: _tv_comp_id,
                prior_lookup=_tv_prior_lookup,
            )
        except Exception:
            _tv_cvi_block = pd.DataFrame()

    # ---- Career CVI (cross-season, cross-league, decay-weighted) ----
    # Two anchors:
    #   • _tv_career_current — anchored to the player's most recent
    #     season → headline "Current CVI" in the bio row
    #   • _tv_career_season  — anchored to the selected season → shown
    #     in Transfer Value Detail with full per-season breakdown
    # Both apply 0.6 decay per season back and never peek at seasons
    # after the anchor.
    _tv_career_current = None
    _tv_career_season = None
    try:
        _gpa_for_career = load_gpa_values()
        _perf_table = _build_player_season_perf_table(
            _gpa_for_career,
            profile_player_minutes_df
              if 'profile_player_minutes_df' in dir() else None,
        )
        _dob_for_career = None
        if isinstance(player_bio, pd.Series):
            _dob_for_career = player_bio.get('birthDate')
        _dob_lookup_for_career = (lambda pid:
                                     pd.to_datetime(_dob_for_career,
                                                      errors='coerce')
                                     if _dob_for_career is not None
                                     else None)
        # Current CVI: anchor at player's most recent season w/ GPA data
        _most_recent_sid = most_recent_season_for_player(_perf_table,
                                                            player_id)
        if _most_recent_sid is not None:
            _tv_career_current = compute_career_cvi(
                player_id, _most_recent_sid,
                perf_table=_perf_table,
                dob_lookup=_dob_lookup_for_career,
            )
        # Season CVI: anchor at the user-selected season
        if selected_season_id is not None:
            _tv_career_season = compute_career_cvi(
                player_id, selected_season_id,
                perf_table=_perf_table,
                dob_lookup=_dob_lookup_for_career,
            )
    except Exception as _car_exc:
        print(f"Warning: career CVI failed: "
               f"{type(_car_exc).__name__}: {_car_exc}")

    _tv_valuations_rows = pd.DataFrame()
    try:
        from valuations.load_valuations import load_all_valuations
        _all_val = load_all_valuations()
        if _all_val is not None and not _all_val.empty:
            _tv_valuations_rows = _all_val[
                _all_val['playerId'] == player_id
            ].sort_values('as_of_date', ascending=False)
    except Exception:
        pass

    # Legacy CVI-path value (goalkeepers' headline; outfielders use the
    # engine value below). The curve — PROJECTED_EUR_COEF × CVI^PROJECTED_EUR_EXP
    # × position multiplier × Camp penalty, uncapped — lives in
    # models/value/cvi.cvi_to_projected_eur; never restate its constants here.
    _tv_projected_eur = None
    _current_cvi_for_eur = None
    if _tv_career_current is not None:
        _v = _tv_career_current.get('career_cvi')
        if _v is not None and not pd.isna(_v):
            _current_cvi_for_eur = float(_v)
    if _current_cvi_for_eur is None and not _tv_cvi_block.empty:
        _v = _tv_cvi_block.iloc[0].get('_CVI')
        if _v is not None and not pd.isna(_v):
            _current_cvi_for_eur = float(_v)
    if _current_cvi_for_eur is not None and _current_cvi_for_eur > 0:
        # v2.9 — single helper handles power curve + position
        # multiplier + Camp penalty + cap. See cvi_to_projected_eur.
        _pos_grp_for_eur = _cvi_position_group(current_pos)
        _tv_projected_eur = cvi_to_projected_eur(
            _current_cvi_for_eur,
            position_group=_pos_grp_for_eur,
            competition_id=_tv_comp_id,
        )
    _tv_true_eur = None
    _tv_true_source = None
    if not _tv_valuations_rows.empty:
        _latest = _tv_valuations_rows.iloc[0]
        _tv_true_eur = _latest.get('value_eur')
        _tv_true_source = _latest.get('source')
    _tv_delta_eur = (_tv_projected_eur - _tv_true_eur
                      if _tv_projected_eur is not None and _tv_true_eur is not None
                      else None)

    # Engine projected value — computed centrally in
    # load_player_engine() (engine_value_eur); just look it up.
    _eng_proj_eur = None
    _eng_w_ev = _eng_mins = None   # support inputs for the likely-fee range
    try:
        _eng_tv_df, _ = load_player_engine()
        if not _eng_tv_df.empty:
            # ONE row-pick rule for engine columns (app.engine_rows_for_scope:
            # projection-first, then seasonId, then minutes) — the row the
            # analysis tables merge, so the two pages can never disagree.
            _p_rows = engine_rows_for_scope(None)
            _p_rows = (_p_rows[_p_rows['playerId'] == int(selected_player_id)]
                       .dropna(subset=['engine_value_eur']) if not _p_rows.empty else _p_rows)
            if not _p_rows.empty:
                _p_row = _p_rows.iloc[0]
                _eng_proj_eur = float(_p_row['engine_value_eur'])
                _eng_w_ev = _p_row.get('w_evidence')
                _eng_mins = _p_row.get('mins_played')
    except Exception:
        logger.exception("engine projected value failed")
    _eurcal = load_eur_calibration()

    col1_bio, col2_bio = st.columns([1, 3])
    with col1_bio:
        image_url = player_bio.get('imageDataURL', None)
        if image_url:
            st.image(image_url, width=150)
        else:
            st.image("https://t3.ftcdn.net/jpg/05/16/27/58/360_F_516275801_f3Fsp17x6HQK0xQgDQEELoGau0sJzEf4.jpg", width=150)

    with col2_bio:
        st.subheader("Player Information")
        bio_row1 = st.columns(4)
        bio_row1[0].metric("Team", current_team)
        bio_row1[1].metric("Position", current_pos)
        bio_row1[2].metric("Nationality", player_bio.get('passportArea', 'N/A'))

        age = _calculate_age(player_bio.get('birthDate'))
        age_display = f"{age:.1f}" if isinstance(age, float) else "N/A"
        bio_row1[3].metric("Age", age_display)

        bio_row2 = st.columns(4)
        foot_value = player_bio.get('foot')
        foot_display = "N/A"
        if foot_value and not pd.isna(foot_value):
            foot_display = foot_value.capitalize()
        bio_row2[0].metric("Foot", foot_display)
        bio_row2[1].metric("Height", f"{player_bio.get('height', 0)} cm")
        bio_row2[2].metric("Weight", f"{player_bio.get('weight', 0)} kg")
        bio_row2[3].metric("Birthplace", player_bio.get('birthArea', 'N/A'))

        # Headline value — ENGINE projected value for outfielders;
        # GOALKEEPERS keep the prior CVI→EUR value (Lucas) since the
        # outfield engine does not cover keepers.
        _is_gk = str(player_per_90_stats.get('primaryPosition', '')).upper().startswith('GK')
        bio_row3 = st.columns(3)
        if _is_gk:
            bio_row3[0].metric(
                "Projected value",
                ("—" if _tv_projected_eur is None else f"€{_tv_projected_eur:,.0f}"),
                help=eur_interval_ui.help_text(_eurcal, gk=True),
            )
        else:
            bio_row3[0].metric(
                "Projected value",
                ("—" if _eng_proj_eur is None else f"€{_eng_proj_eur:,.0f}"),
                help=eur_interval_ui.help_text(_eurcal),
            )
            # The likely-fee range goes in a caption, NOT the metric delta:
            # Streamlit draws an up-arrow for any non-negative delta text.
            _rng_line = eur_interval_ui.range_sentence(
                _eng_proj_eur, _eurcal, w_evidence=_eng_w_ev, mins=_eng_mins)
            if _rng_line:
                bio_row3[0].caption(_rng_line)
            # Observed role + style in the header (Lucas 2026-07-17).
            # Scope-aware; hidden when the player has no style row (GKs
            # never reach here, sub-300' players degrade gracefully).
            try:
                _hdr_style = get_scoped_style(int(selected_player_id),
                                              active_season_ids)
            except Exception:
                _hdr_style = None
            if _hdr_style:
                bio_row3[1].metric(
                    "Role", str(_hdr_style.get('role') or '—'),
                    help="Observed engine role — learned from where his "
                         "events actually happen match by match, not the "
                         "lineup-card position.",
                )
                _stl = _hdr_style.get('style')
                _fit = _hdr_style.get('style_fit')
                bio_row3[2].metric(
                    "Style", (str(_stl) if _stl else '—'),
                    delta=(f"{float(_fit):.0f}% fit"
                           if _fit is not None and pd.notna(_fit) else None),
                    delta_color="off",
                    help="Tendency-derived archetype — how he plays the "
                         "role, not how well. Never part of the rating. "
                         "Fit = how strongly he expresses the style "
                         "(percentile vs the role cohort).",
                )

    st.divider()

    with st.expander("ℹ️ How these ratings work"):
        st.markdown(RATINGS_EXPLAINER_MD)

    # --- ACP Engine card: rating + projection + components ---------
    try:
        _eng_df, _eng_meta = load_player_engine()
        _erows = _eng_df[_eng_df['playerId'] == int(selected_player_id)] if not _eng_df.empty else pd.DataFrame()
        # active_season_ids can be None (All Seasons), an int, or a list
        if isinstance(active_season_ids, (list, tuple, set)):
            _e_sids = [int(s) for s in active_season_ids if s is not None]
        elif active_season_ids is not None:
            _e_sids = [int(active_season_ids)]
        else:
            _e_sids = None
        _career_view = False
        if _erows.empty:
            _escope = _erows
        elif _e_sids:
            _escope = _erows[_erows['seasonId'].isin(_e_sids)]
        else:   # All Seasons → CAREER view: aggregate every rated season
            _escope = _erows
            _career_view = len(_erows) > 1
        _eng_stale = False
        if _escope.empty and not _erows.empty:
            # not rated in the selected scope — fall back to last rated season
            _escope = _erows[_erows['seasonId'] == _erows['seasonId'].max()]
            _eng_stale = True
        _is_gk_card = str(player_per_90_stats.get('primaryPosition', '')).upper().startswith('GK')
        st.subheader("Goalkeeper Rating (legacy system)" if _is_gk_card
                      else "ACP Index")
        if _escope.empty:
            if _is_gk_card:
                # Prior rating + value system, retained for keepers.
                _gk_templates = ['Shot Stopper', 'Cross Claimer', 'Ball-playing GK']
                _gk_scored = []
                for _t in _gk_templates:
                    _s = player_per_90_stats.get(f'{_t}_Score')
                    if _s is not None and pd.notna(_s):
                        _gk_scored.append((_t, float(_s)))
                if _gk_scored:
                    _gk_best = max(_gk_scored, key=lambda x: x[1])
                    _gkc = st.columns(3)
                    _gkc[0].metric(
                        "GK Rating (legacy)", f"{_gk_best[1]:.0f}",
                        help="Best-fit goalkeeper template score — the "
                             "bespoke weighted-percentile system (the prior "
                             "rating engine), retained for keepers since the "
                             "outfield ACP engine does not cover them.")
                    _gkc[1].metric("Best-fit template", _gk_best[0])
                    _gkc[2].metric(
                        "Projected value",
                        "—" if _tv_projected_eur is None else f"€{_tv_projected_eur:,.0f}",
                        help="Legacy CVI→EUR value.")
                    st.caption("Template scores — " + " · ".join(
                        f"**{_t}** {_s:.0f}" for _t, _s in _gk_scored)
                        + "  ·  full goalkeeping metrics in the Player "
                          "Radar and Stats tabs below.")
                else:
                    st.info("Goalkeeper — insufficient minutes for the "
                            "legacy GK rating in this scope.")
            else:
                st.info("Not rated by the engine for this scope "
                        "(below the 90-minute floor, or no "
                        "role assignment yet).")
        else:
            _e = _escope.sort_values('mins_played').iloc[-1]
            if _career_view:
                # All-seasons CAREER view: representative row = highest-
                # minutes season (for role/league/shares); rating + every
                # percentile/grade aggregated minutes-weighted across all
                # the player's rated seasons.
                _e = _e.copy()
                _wts = pd.to_numeric(_escope['mins_played'], errors='coerce').fillna(0.0).to_numpy()
                _pct_cols = ['off_pct', 'qual_pct', 'rapm_pct', 'defr_pct',
                              'datt_pct', 'setpiece_pct', 'aerial_grade_pct',
                              'ground_grade_pct', 'Shooting_pct', 'Creating_pct',
                              'Linking_pct', 'Receiving_pct', 'Dribbling_pct',
                              'career_asof', 'w_evidence']
                for _c in _pct_cols:
                    if _c in _escope.columns and _wts.sum() > 0:
                        _v = pd.to_numeric(_escope[_c], errors='coerce').to_numpy()
                        _m = ~np.isnan(_v)
                        if _m.any():
                            _e[_c] = float(np.average(_v[_m], weights=_wts[_m]))
                if pd.notna(_e.get('acp_rating_career')):
                    _e['acp_rating'] = float(_e['acp_rating_career'])
                _e['mins_played'] = float(_wts.sum())
                # projection = the LATEST season's forward look; lineup
                # minutes = career total (for the radar header)
                _latest_row = _escope.sort_values('seasonId').iloc[-1]
                for _pc in ('projection', 'projection_abs', 'band_sd',
                             'proj_delta', 'seasons_ago', 'age'):
                    if _pc in _latest_row:
                        _e[_pc] = _latest_row[_pc]
                if 'mins_lineup' in _escope.columns:
                    _e['mins_lineup'] = float(pd.to_numeric(
                        _escope['mins_lineup'], errors='coerce').fillna(0.0).sum())
                st.caption(f"📚 Career view — minutes-weighted across "
                           f"{len(_escope)} rated seasons "
                           f"({int(_wts.sum())} total minutes). Select a "
                           f"single season for that season's rating.")
            if _eng_stale:
                st.caption(f"⏳ Not rated in the selected season — showing "
                           f"last rated season ({SEASON_ID_MAP.get(int(_e['seasonId']), _e['seasonId'])}).")
            # The projection is a single player-level forward-look anchored to
            # the player's most-recent season. seasonId is NOT chronological
            # across leagues (Camp 23/24 = 190230 > L3 24/25 = 190090), so the
            # in-scope "latest" row picked above can be the wrong one and carry
            # no projection (e.g. M. Konaté). Always source the projection from
            # the row that actually has one. (Lucas 2026-06-24)
            _proj_src = (_erows[_erows['projection'].notna()]
                         if not _erows.empty else _erows)
            if not _proj_src.empty:
                _pr = _proj_src.sort_values('seasonId').iloc[-1]
                for _pc in ('projection', 'projection_abs', 'band_sd',
                             'proj_delta', 'seasons_ago'):
                    if _pc in _pr.index:
                        _e[_pc] = _pr[_pc]
            _ec1, _ec2, _ec3, _ec4 = st.columns(4)
            _abs_note = (f"abs {_e['acp_rating_abs']:.0f}"
                         if pd.notna(_e.get('acp_rating_abs'))
                         and _e.get('league') == 'CAMP' else None)
            _ec1.metric("ACP Rating", f"{_e['acp_rating']:.0f}", _abs_note,
                        delta_color="off",
                        help="Role-blended 5-axis rating, 50±17 scale, "
                             "within-league. 'abs' = Liga-3-equivalent "
                             "(descriptive league delta).")
            if pd.notna(_e.get('projection')):
                _proj_note = (f"abs {_e['projection_abs']:.0f}"
                              if pd.notna(_e.get('projection_abs'))
                              and _e.get('league') == 'CAMP' else None)
                _ec2.metric("Projection (next season)",
                            f"{_e['projection']:.0f} ± {_e['band_sd']:.0f}",
                            _proj_note, delta_color="off",
                            help="Career + evidence pull + role age curve. "
                                 "Band = role-specific 1 SD of realized "
                                 "error. 'abs' applies the stricter "
                                 "recruit league delta.")
            else:
                _ec2.metric("Projection", "—",
                            help="No projection (age unknown or below floor).")
            _ec3.metric("Career (as-of)",
                        f"{_e['career_asof']:.0f}" if pd.notna(_e.get('career_asof')) else "—",
                        help="Recency-weighted career rating through this season.")
            _ec4.metric("Evidence",
                        f"{_e['w_evidence']:.0%}" if pd.notna(_e.get('w_evidence')) else "—",
                        help="Career evidence weight w = eff_mins/(eff_mins+K). "
                             "Low = projection leans on the pull terms.")

            # badges
            _badges = []
            _shares = sorted(
                [(c[3:], float(_e[c])) for c in _escope.columns
                 if c.startswith('sh_') and pd.notna(_e[c]) and float(_e[c]) > 0.15],
                key=lambda x: -x[1])
            if _shares:
                _badges.append(" · ".join(f"**{n}** {s:.0%}" for n, s in _shares[:3]))
            if pd.notna(_e.get('seasons_ago')) and int(_e['seasons_ago']) >= 1:
                _badges.append("🕐 last rated 24/25 — projection carries "
                               "extra age step + wider band")
            if float(_e['mins_played']) < 500:
                _badges.append(f"⚠️ thin sample this league "
                               f"({int(_e['mins_played'])}′ — admitted via "
                               f"cross-competition season pooling)")
            if len(_escope) > 1:
                _others = _escope[_escope.index != _e.name]
                for _, _o in _others.iterrows():
                    _badges.append(f"also rated in {_o['league']} "
                                   f"({int(_o['mins_played'])}′, rating "
                                   f"{_o['acp_rating']:.0f})")
            if _badges:
                st.markdown("  \n".join(_badges))

            # --- style chip: role + tendency-derived archetype ----------
            # Descriptive only (never in the rating). Scope-aware: the
            # style of the player's highest-minutes row in the selected
            # scope. GKs / sub-300' players simply get no chip.
            try:
                _sty = get_scoped_style(int(selected_player_id), _e_sids)
                _role_disp = _e.get('role')
                if _sty and pd.notna(_sty.get('style')):
                    _fit = _sty.get('style_fit')
                    _fit_txt = (f" · {float(_fit):.0f}% fit"
                                if _fit is not None and pd.notna(_fit) else "")
                    _thin = " · thin sample" if _sty.get('thin_sample') else ""
                    st.markdown(
                        f"<span style='background:#eef2ff;color:#3730a3;"
                        f"padding:3px 10px;border-radius:12px;font-size:0.9em;"
                        f"font-weight:600'>{_role_disp} · "
                        f"{_sty['style']}{_fit_txt}{_thin}</span>",
                        unsafe_allow_html=True)
            except Exception:
                logger.exception("style chip failed")

            # component bars
            _comp_cols = st.columns(6)
            for _i, (_lbl, _col) in enumerate([
                    ("Offensive Value", 'off_pct'),
                    ("Def Quality Grade", 'qual_pct'),
                    ("RAPM", 'rapm_pct'),
                    ("Def Volume Grade", 'defr_pct'),
                    ("Off Duel Grade", 'datt_pct'),
                    ("Set piece", 'setpiece_pct')]):
                _v = _e.get(_col)
                with _comp_cols[_i]:
                    if pd.notna(_v):
                        st.progress(min(max(float(_v), 0.0), 1.0))
                        st.caption(f"{_lbl} · {float(_v)*100:.0f}")
                    else:
                        st.caption(f"{_lbl} · —")
            # --- ACP Index radar card: cached PNG render (see
            # _render_acp_index_card_png). Building the matplotlib
            # radar+KDE figure inline cost ~2-3 s on every rerun /
            # profile section toggle; the PNG bytes are cached on
            # (player, season scope, stats-cache ver, engine ver).
            _card_png = _render_acp_index_card_png(
                int(selected_player_id),
                tuple(sorted(_e_sids)) if _e_sids else (),
                STATS_CACHE_VERSION,
                str(_eng_meta.get('rating_version', '')),
                _career_view,
                _eng_df, _e, _eng_meta)
            if _card_png:
                st.image(_card_png, use_container_width=True)

            # --- Tendencies panel: futi-style bipolar sliders ----------
            # The role's menu of attempt-composition leanings, each a
            # within-role percentile (50 = role-typical). Descriptive, never
            # a rating. Hidden entirely for players without tendencies.
            try:
                _has_tend = get_scoped_tendencies(
                    int(selected_player_id), _e_sids) is not None
                if _has_tend:
                    with st.expander("🎚️ Tendencies — how he plays the role",
                                     expanded=False):
                        st.caption(
                            "Each bar is a **within-role percentile** of "
                            "what he attempts (50 = typical for the role) — "
                            "style, not quality, and never part of the "
                            "rating. The leaning side is coloured.")
                        render_tendencies_panel(int(selected_player_id),
                                                _e_sids)
            except Exception:
                logger.exception("tendencies panel failed")

            # --- Projection outlook fan chart (prototype) ---
            try:
                from pitch_interactive import plotly_projection_fan
                _fan = plotly_projection_fan(
                    _erows, SEASON_ID_MAP, selected_player_name)
                if _fan is not None:
                    st.plotly_chart(_fan, use_container_width=True,
                                    config={'displayModeBar': False})
                    st.caption(
                        "Career ratings by season (league and age on "
                        "each tick) flowing into next season's "
                        "projection. Shaded fan = ±1 SD of the "
                        "projection; a faded segment bridges seasons "
                        "missing from our data; green band = typical "
                        "peak ages for the role (literature-based "
                        "curve used by the projection model); "
                        "evidence % = how much data underwrites the "
                        "starting point. Cross-league careers plot on "
                        "the L3-equivalent scale (CAMP seasons "
                        "discounted by the league conversion — hover "
                        "a dot for the native rating).")
            except Exception:
                logger.exception("Projection fan chart failed")
            st.caption(
                f"Engine {_eng_meta.get('rating_version', '?')} · "
                f"projection {_eng_meta.get('projection_version', '?')} · "
                f"data through {_eng_meta.get('data_through', '?')} · "
                f"percentiles within league × season × role cohort; "
                f"set piece shown separately (not in the rating)")
    except Exception:
        logger.exception("Engine card failed")
    st.divider()

    # --- Exportable one-pager PDF ----------------------------------
    _op_cols = st.columns([1.4, 1.4, 3])
    with _op_cols[0]:
        _op_clicked = st.button("📄 Build player report PDF",
                                 key="onepager_build")
    if _op_clicked:
        try:
            with st.spinner("Composing player report…"):
                from player_onepager import build_player_onepager
                from pitch_interactive import mpl_projection_fan
                # The whole build runs under ONE MPL_LOCK: every figure is
                # created here and rasterised inside build_player_onepager,
                # so build + render + close must not be split (a figure
                # freed by another session's plt.close('all') mid-render is
                # a segfault — see mpl_safety). Each figure has its own
                # try/except so one failure drops that panel, not the PDF.
                _op_figs = []          # every fig we create, for cleanup
                with MPL_LOCK:
                    # --- resolve the player's stats row + best-fit role ---
                    _op_row = None
                    _op_role = None      # config template role (radar/stats)
                    _op_elig = []
                    _op_pop = player_stats_with_scores_df
                    _op_matches = player_stats_with_scores_df[
                        player_stats_with_scores_df['playerId']
                        == int(selected_player_id)]
                    if not _op_matches.empty:
                        _op_row = _op_matches.iloc[0]
                        _op_pos = _op_row.get('primaryPosition')
                        _op_elig = [r for r in WEIGHTS
                                    if _op_pos in POSITION_GROUPS.get(r, [])]
                        if _op_elig:
                            _op_role = max(_op_elig, key=lambda r: float(
                                _op_row.get(f'{r}_Score', 0) or 0))
                            _op_pop = player_stats_with_scores_df[
                                player_stats_with_scores_df['primaryPosition']
                                .isin(POSITION_GROUPS.get(_op_role, [_op_pos]))]
                            if len(_op_pop) < 5:
                                _op_pop = player_stats_with_scores_df

                    # --- engine row: reuse the header's `_e` when present
                    # (it is sorted/scoped correctly); fall back defensively
                    # for keepers, who have no engine row at all. ---
                    _op_e = None
                    _op_eng_role = None
                    if not _erows.empty:
                        _op_prow = _erows[_erows['projection'].notna()]
                        if not _op_prow.empty:
                            # match the header's chronological pick, NOT the
                            # unsorted .iloc[-1] the old build used
                            _op_e = _op_prow.sort_values('seasonId').iloc[-1]
                        else:
                            _op_e = _erows.sort_values('mins_played').iloc[-1]
                        _op_eng_role = _canonical_engine_role(
                            _op_e.get('role'))
                    # attacker vs defender: engine role first (the split
                    # Lucas asked for), then fall back to the config
                    # template allowlist for players the engine does not
                    # rate (keepers). Unknown -> defensive layout.
                    _op_attacking = (
                        _engine_role_is_attacking(_op_eng_role)
                        if _op_eng_role is not None
                        else (_op_role in _ATTACK_TEMPLATE_ROLES))

                    # 1) template radar (raw mean ± 2σ mode)
                    _op_fig_radar = None
                    try:
                        if _op_row is not None and _op_role:
                            _op_metrics = [m for m in WEIGHTS[_op_role]
                                           if m in _op_row.index
                                           and m not in RADAR_HIDDEN_METRICS]
                            if _op_metrics:
                                _op_seasons = _season_id_list(active_season_ids)
                                _op_season_lbl = (
                                    SEASON_ID_MAP.get(_op_seasons[0], '')
                                    if len(_op_seasons) == 1 else 'All Seasons')
                                _op_fig_radar = create_radar_with_distributions(
                                    pd.DataFrame([_op_row]), _op_metrics,
                                    _op_pos, _op_elig, _op_pop,
                                    full_df_for_ranking=player_stats_with_scores_df,
                                    season_label=_op_season_lbl,
                                    radar_mode='raw')
                                _op_figs.append(_op_fig_radar)
                    except Exception:
                        logger.exception("one-pager radar failed")

                    # 2) projection outlook (matplotlib twin of the fan)
                    _op_fig_proj = None
                    try:
                        if not _erows.empty:
                            _op_fig_proj = mpl_projection_fan(
                                _erows, SEASON_ID_MAP, selected_player_name)
                            if _op_fig_proj is not None:
                                _op_figs.append(_op_fig_proj)
                    except Exception:
                        logger.exception("one-pager projection failed")

                    # 3) position-conditional maps
                    _op_fig_shots = _op_fig_passes = _op_fig_def = None
                    if _op_attacking:
                        try:   # shot map
                            _op_shots = profile_events_df[
                                (profile_events_df['player.name'] == selected_player_name)
                                & (profile_events_df['type.primary'] == 'shot')].copy()
                            if not _op_shots.empty:
                                _op_shots = _op_shots.sort_values(
                                    ['matchId', 'minute', 'second'])
                                _op_shots.reset_index(drop=True, inplace=True)
                                _op_shots['Shot Number'] = _op_shots.index + 1
                                _op_fig_shots = create_player_shotmap(
                                    _op_shots, selected_player_name)
                                _op_figs.append(_op_fig_shots)
                        except Exception:
                            logger.exception("one-pager shotmap failed")
                        try:   # box-pass creativity map
                            _op_bp = load_box_passes()
                            if not _op_bp.empty:
                                _op_bp_seasons = _season_id_list(active_season_ids)
                                _op_bp = _op_bp[
                                    _op_bp['player.id'] == int(selected_player_id)]
                                if _op_bp_seasons:
                                    _op_bp = _op_bp[
                                        _op_bp['seasonId'].isin(_op_bp_seasons)]
                                if not _op_bp.empty:
                                    _op_fig_passes = mpl_box_passes_map(
                                        _op_bp, selected_player_name)
                                    _op_figs.append(_op_fig_passes)
                        except Exception:
                            logger.exception("one-pager box passes failed")
                    else:
                        try:   # defensive action heatmap
                            # Same peer-group resolution + cached density
                            # stack the Shots & Creation tab uses, so the
                            # PDF heatmap is normalised identically.
                            _DEF_PEERS = {
                                'GK': ['GK'],
                                'CB': ['CB', 'LCB', 'RCB', 'LCB3', 'RCB3'],
                                'FB': ['LB', 'RB', 'LB5', 'RB5', 'LWB', 'RWB'],
                                'CM': ['DMF', 'LDMF', 'RDMF', 'LCMF', 'RCMF',
                                       'LCMF3', 'RCMF3'],
                                'AM/Wing': ['AMF', 'LAMF', 'RAMF', 'LW', 'RW',
                                            'LWF', 'RWF'],
                                'ST': ['CF', 'SS'],
                            }
                            _op_pos_codes = [current_pos]
                            for _gc in _DEF_PEERS.values():
                                if current_pos in _gc:
                                    _op_pos_codes = _gc
                                    break
                            _op_ev_hash = hashlib.md5(
                                f"{len(profile_events_df)}_"
                                f"{tuple(sorted(_op_pos_codes))}".encode()
                            ).hexdigest()
                            _op_stack = _compute_peer_density_stack(
                                _op_ev_hash, profile_events_df,
                                tuple(sorted(_op_pos_codes)),
                                _player_minutes_df=profile_player_minutes_df,
                                include_recoveries=True)
                            _op_fig_def = pv.plot_defensive_action_heatmap(
                                profile_events_df, player_id,
                                selected_player_name,
                                position_codes=_op_pos_codes,
                                player_minutes_df=profile_player_minutes_df,
                                peer_density_stack=_op_stack,
                                include_recoveries=True)
                            if _op_fig_def is not None:
                                _op_figs.append(_op_fig_def)
                        except Exception:
                            logger.exception("one-pager def heatmap failed")

                    # 4) role-relevant stats (percentile-washed table)
                    _op_role_stats = _role_key_stats(_op_row, _op_role) \
                        if _op_role else []

                    # 5) engine card PNG — reuse the header's cached render
                    _op_card_png = None
                    _op_card_aspect = 2.0    # figsize (20, 10)
                    try:
                        if _op_e is not None and not _erows.empty:
                            _op_card_png = _render_acp_index_card_png(
                                int(selected_player_id),
                                tuple(sorted(_e_sids)) if _e_sids else (),
                                STATS_CACHE_VERSION,
                                str(_eng_meta.get('rating_version', '')),
                                _career_view,
                                _eng_df, _op_e, _eng_meta)
                    except Exception:
                        logger.exception("one-pager engine card failed")

                    # --- header tiles + bio strip ---
                    _op_tiles = [("Team", current_team),
                                 ("Position", current_pos),
                                 ("Age", age_display),
                                 ("Minutes", f"{total_minutes:,.0f}")]
                    _op_val = (_tv_projected_eur if str(current_pos).upper().startswith('GK')
                               else _eng_proj_eur)
                    if _op_val is not None:
                        _op_tiles.append(("Proj. value",
                                          f"EUR {_op_val:,.0f}"))
                        if not str(current_pos).upper().startswith('GK'):
                            _op_rng = eur_interval_ui.pdf_range_text(
                                _op_val, _eurcal, w_evidence=_eng_w_ev, mins=_eng_mins)
                            if _op_rng:
                                _op_tiles.append(("Likely fee (EUR)", _op_rng))
                    if _op_e is not None:
                        _op_tiles.append(
                            ("ACP Rating", f"{float(_op_e['acp_rating']):.0f}"))
                        if pd.notna(_op_e.get('projection')):
                            _op_tiles.append(
                                ("Projection",
                                 f"{float(_op_e['projection']):.0f} "
                                 f"+/- {float(_op_e.get('band_sd', 0) or 0):.0f}"))
                    _op_bio = [
                        ("Nationality", player_bio.get('passportArea')),
                        ("Foot", str(player_bio.get('foot', '')).capitalize()),
                        ("Height", f"{player_bio.get('height')} cm"
                         if player_bio.get('height') else None),
                        ("Weight", f"{player_bio.get('weight')} kg"
                         if player_bio.get('weight') else None),
                        ("Born", player_bio.get('birthArea')),
                    ]
                    _op_footer = (
                        f"Engine {_eng_meta.get('rating_version', '?')} · "
                        f"projection {_eng_meta.get('projection_version', '?')} · "
                        f"data through {_eng_meta.get('data_through', '?')} · "
                        f"generated {datetime.date.today().isoformat()}"
                    ) if not _erows.empty else (
                        f"Generated {datetime.date.today().isoformat()}")

                    _op_bytes = build_player_onepager(
                        selected_player_name,
                        f"{current_team} · {current_pos}",
                        _op_tiles, _op_fig_radar, _op_fig_shots,
                        _op_fig_passes, footer_note=_op_footer,
                        bio=_op_bio, role_stats=_op_role_stats,
                        role_label=_op_role or '',
                        fig_projection=_op_fig_proj,
                        fig_defensive=_op_fig_def,
                        engine_card_png=_op_card_png,
                        engine_card_aspect=_op_card_aspect)
                    for _f in _op_figs:
                        if _f is not None:
                            plt.close(_f)
                    st.session_state['onepager_pdf'] = (
                        int(selected_player_id), _op_bytes)
        except Exception:
            logger.exception("one-pager build failed")
            st.error("Could not build the one-pager for this player.")
    _op_cached = st.session_state.get('onepager_pdf')
    if _op_cached and _op_cached[0] == int(selected_player_id):
        with _op_cols[1]:
            st.download_button(
                "⬇️ Download player report",
                data=_op_cached[1],
                file_name=f"{selected_player_name.replace(' ', '_')}_report.pdf",
                mime="application/pdf", key="onepager_dl")
    st.divider()

    # Hoisted so every lazy section can access it: the Stats section's
    # positional-peer population (radar_stats_df at ~10840) was previously
    # populated by the Player Radar tab body, which always ran under st.tabs.
    # With lazy if/elif rendering only one section runs, so compute the base
    # (player_stats_with_scores_df + DefR columns) up front. The Player Radar
    # section may still override radar_stats_df locally via its "Show Only
    # Position" filter for its own rendering.
    radar_stats_df = merge_defr_values_into_stats(
        player_stats_with_scores_df, active_season_ids, selected_comp_ids)

    _profile_sections = ["Player Radar", "Stats", "Value", "Shots & Creation", "Match Log"]
    _active_tab = st.radio("Profile section", _profile_sections,
                            horizontal=True, label_visibility="collapsed",
                            key="profile_active_tab")

    if _active_tab == "Player Radar":
        # --- Transfer Value Detail moved to just above Career Trajectory ---

       # --- 5. NEW: DISPLAY PLAYER RADAR ---
        st.subheader("Player Radar")

        # Gate: 300-minute minimum for radar charts
        _MIN_RADAR_MINUTES = 300
        _show_radar = total_minutes >= _MIN_RADAR_MINUTES
        if not _show_radar:
            st.info(f"⚠️ **Insufficient sample size** — {player_per_90_stats.get('playerName', 'This player')} has only played **{int(total_minutes)} minutes** this season.")

        # 1. Detect Raw Positions (What did they actually play?)
        try:
            player_events = profile_events_df[profile_events_df['player.id'] == player_id]
            if 'player.position' in player_events.columns:
                raw_positions = player_events['player.position'].unique()
            elif 'position_name' in player_events.columns:
                raw_positions = player_events['position_name'].unique()
            else:
                raw_positions = []

            # Filter out None/Nan
            raw_positions = [x for x in raw_positions if x and str(x) != 'nan']

        except Exception:
            logger.exception("position extraction failed")
            raw_positions = []

        # Ensure we at least have the primary position from the bio
        if current_pos and current_pos not in raw_positions:
            raw_positions.append(current_pos)

        # Sort for the dropdown
        raw_positions = sorted([str(p) for p in raw_positions])

        # 2. Position Selector (Simple Raw Codes)
        col_rad_sel1, col_rad_sel2 = st.columns([1, 3])
        with col_rad_sel1:
            st.markdown("##### Show Radar For:")
        with col_rad_sel2:
            selected_raw_pos = st.selectbox(
                "Select Position:",
                raw_positions,
                label_visibility="collapsed",
                key="radar_pos_selector"
            )

        # --- Show Only Position toggle ---
        # When active, recompute stats using only events at the selected position
        profile_pos_filter = st.checkbox(
            "Show Only Position",
            key="profile_pos_played_filter",
            help=f"When checked, the radar uses only events where the player played as **{selected_raw_pos}** (selected above), instead of all events."
        )
        radar_stats_df = player_stats_with_scores_df
        radar_player_data_row = player_data_row
        if profile_pos_filter and 'player.position' in profile_events_df.columns:
            pos_filtered_events = profile_events_df[
                profile_events_df['player.position'] == selected_raw_pos
            ]
            if not pos_filtered_events.empty:
                # Build position-adjusted minutes: replace totalMinutes with minutes at this position
                all_pos_minutes = get_all_players_minutes_by_position(profile_events_df)
                pos_minutes = all_pos_minutes[all_pos_minutes['Position'] == selected_raw_pos][['playerId', 'Minutes']]
                pos_player_minutes_df = profile_player_minutes_df.copy()
                pos_player_minutes_df = pos_player_minutes_df.merge(pos_minutes, on='playerId', how='inner')
                pos_player_minutes_df['totalMinutes'] = pos_player_minutes_df['Minutes']
                pos_player_minutes_df = pos_player_minutes_df.drop(columns=['Minutes'])

                # Use a distinct season_id so the cache differentiates from unfiltered stats
                pos_cache_key = f"{selected_season_id}_pos_{selected_raw_pos}"
                pos_filtered_stats = calculate_all_player_stats(
                    pos_filtered_events, pos_player_minutes_df, season_id=pos_cache_key
                )
                # Merge GPA Value columns (same season × competition scope as profile)
                pos_filtered_stats = merge_gpa_values_into_stats(pos_filtered_stats, active_season_ids, selected_comp_ids)
                pos_filtered_scores = calculate_player_percentiles_and_scores(
                    pos_filtered_stats, POSITION_GROUPS, WEIGHTS, INVERT_METRICS, min_minutes=500, season_id=pos_cache_key
                )
                if not pos_filtered_scores.empty:
                    radar_stats_df = pos_filtered_scores
                    pos_player_row = pos_filtered_scores[pos_filtered_scores['playerId'] == player_id]
                    if not pos_player_row.empty:
                        radar_player_data_row = pos_player_row

        # Ensure the radar population carries DefR columns (the percentile
        # function is disk-cached and may drop them) — used by the radar
        # population, the DefR-mode toggle, and Overall Season Stats coloring.
        radar_stats_df = merge_defr_values_into_stats(
            radar_stats_df, active_season_ids, selected_comp_ids)

        # 3. Find the "Best Fit" Template for this Raw Position
        # (e.g. If 'CF' is selected, check 'Target Man', 'Poacher', etc. and pick the best one)

        # Find all roles that include this raw position
        eligible_roles = []
        for role, valid_codes in POSITION_GROUPS.items():
            if selected_raw_pos in valid_codes:
                eligible_roles.append(role)

        if not eligible_roles:
            st.warning(f"No radar templates defined for position '{selected_raw_pos}'.")
        else:
            # Calculate Scores to find the best fit
            best_role = None
            best_score = -1

            # We calculate a simple score (sum of percentiles) for each eligible role
            for role in eligible_roles:
                # Get metrics and weights
                role_weights = WEIGHTS.get(role, {})
                if not role_weights: continue

                # Define Population for this role (for percentile calculation)
                role_codes = POSITION_GROUPS[role]
                population = radar_stats_df[
                    radar_stats_df['primaryPosition'].isin(role_codes)
                ]
                if len(population) < 5: population = radar_stats_df # Fallback

                # Calculate Score
                role_score = 0
                total_weight = 0

                for metric, weight in role_weights.items():
                    if metric in radar_player_data_row.columns and metric in population.columns:
                        val = radar_player_data_row[metric].values[0]
                        pop_vals = population[metric].fillna(0)

                        # Percentile
                        pct = (pop_vals < val).mean()
                        if metric in INVERT_METRICS: pct = 1.0 - pct

                        role_score += (pct * weight)
                        total_weight += weight

                final_score = (role_score / total_weight) if total_weight > 0 else 0

                if final_score > best_score:
                    best_score = final_score
                    best_role = role

            # Handle case where no score could be calculated
            if best_role is None: best_role = eligible_roles[0]

            # 4. Generate Chart for the Winner
            st.caption(f"Best Template Match: **{best_role}**")
            _radar_style = st.radio("Radar Style", ["Percentile", "Raw Values (mean ± 2σ)"], index=1, horizontal=True, key=f"radar_style_{player_id}")
            _use_defr = st.toggle(
                "Defensive metrics → DefR",
                value=False, key=f"defr_mode_{player_id}",
                help="Swap each defensive axis (tackles, interceptions, recoveries, "
                     "clearances, aerials) to its Defensive Responsibility value — "
                     "actions above/below what the player's role is expected to make.")

            # Prepare data for plotting
            metrics_to_plot = list(WEIGHTS[best_role].keys())
            metrics_to_plot = [m for m in metrics_to_plot
                               if m in radar_player_data_row.columns
                               and m not in RADAR_HIDDEN_METRICS]

            # Get Population for distribution
            final_population = radar_stats_df[
                radar_stats_df['primaryPosition'].isin(POSITION_GROUPS[best_role])
            ]
            if len(final_population) < 5: final_population = radar_stats_df

            # --- DefR mode: swap each defensive axis to its DefR value ---
            if _use_defr:
                final_population = merge_defr_values_into_stats(
                    final_population, active_season_ids, selected_comp_ids)
                radar_player_data_row = merge_defr_values_into_stats(
                    radar_player_data_row, active_season_ids, selected_comp_ids)
                _mapped, _seen = [], set()
                for _m in metrics_to_plot:
                    _mm = DEFR_RADAR_MAP.get(_m, _m)
                    if _mm in radar_player_data_row.columns and _mm not in _seen:
                        _mapped.append(_mm); _seen.add(_mm)
                if _mapped:
                    metrics_to_plot = _mapped
                # Percentile for each DefR axis vs same-position peers
                for _m in metrics_to_plot:
                    if _m in DEFR_DISPLAY_METRICS \
                            and _m in radar_player_data_row.columns \
                            and _m in final_population.columns:
                        _pv = final_population[_m].dropna()
                        if not _pv.empty:
                            _val = radar_player_data_row[_m].values[0]
                            radar_player_data_row[_m + '_percentile'] = (
                                scipy.stats.percentileofscore(_pv, _val, kind='weak') / 100.0)
                if _mapped:
                    st.caption("🛡️ Defensive axes show **DefR** (actions above/below role expectation).")

            # --- NEW: Recalculate percentiles and scores for ALL eligible roles ---
            # This ensures that if the user selects a raw position that maps to multiple templates
            # (e.g., 'CF' -> Mobile Striker, Poacher, etc.), ALL those scores are updated
            # based on the new comparison group.
            radar_player_data_row = radar_player_data_row.copy()

            for role in eligible_roles:
                # 1. Get Population for this specific role
                role_population = radar_stats_df[
                    radar_stats_df['primaryPosition'].isin(POSITION_GROUPS[role])
                ]
                if len(role_population) < 5: role_population = radar_stats_df

                # 2. Get metrics and weights for this role
                role_weights = WEIGHTS.get(role, {})
                new_total_score = 0
                total_weight = 0

                # 3. Recalculate percentiles for all metrics used in this role
                for metric, weight in role_weights.items():
                    if metric not in role_population.columns:
                        continue
                    # Get population values for this metric
                    pop_values = role_population[metric].dropna()

                    if metric in radar_player_data_row.columns:
                        player_val = radar_player_data_row[metric].values[0]

                        if not pop_values.empty:
                            # Calculate percentile (0-100)
                            pct_score = scipy.stats.percentileofscore(pop_values, player_val, kind='weak')

                            # Handle Inverted Metrics
                            if metric in INVERT_METRICS:
                                pct_score = 100.0 - pct_score

                            # Update the row's percentile column (0-1) used for plotting
                            # Note: This overwrites the column. If multiple roles use the same metric,
                            # the last one wins. This is generally acceptable as they are usually 
                            # compared against similar populations if they share a raw position.
                            # Ideally, we'd plot based on the 'best_role' metrics specifically.
                            radar_player_data_row[metric + '_percentile'] = pct_score / 100.0

                            # Add to weighted score
                            new_total_score += ((pct_score / 100.0) * weight)
                            total_weight += weight

                # 4. Update the Role Score column
                if total_weight > 0:
                    final_new_score = (new_total_score / total_weight) * 100
                    radar_player_data_row[role + '_Score'] = final_new_score

            # --- NEW: Update Position Label for Chart ---
            # This ensures the chart displays "CF" if we selected "CF", even if their bio says "RW"
            radar_player_data_row['primaryPosition'] = selected_raw_pos
            # -----------------------------------------------------------------------

            # Plot
            _radar_season_label = SEASON_ID_MAP.get(selected_season_id, 'All Seasons') if selected_season_id else 'All Seasons'
            _radar_mode = 'raw' if _radar_style == "Raw Values (mean ± 2σ)" else 'percentile'
            with MPL_LOCK:
                fig_radar = create_radar_with_distributions(
                    radar_player_data_row,
                    metrics_to_plot,
                    best_role,
                    eligible_roles,
                    all_position_data=final_population,
                    full_df_for_ranking=radar_stats_df,
                    season_label=_radar_season_label,
                    radar_mode=_radar_mode
                )
                st.pyplot(fig_radar, use_container_width=True)

        # Career radar section disabled to reduce memory usage
        # TODO: Re-enable when Streamlit Cloud resources are upgraded

        # --- Minutes by Position + Outlier Stats side-by-side ---
        _col_mins, _col_outliers = st.columns([1, 2])

        with _col_mins:
            minutes_by_pos = get_player_minutes_by_position(profile_events_df, player_id, player_match_log_df)
            if not minutes_by_pos.empty and len(minutes_by_pos) > 1:
                st.caption("Minutes by Position")
                st.dataframe(
                    minutes_by_pos,
                    column_config={
                        "Position": st.column_config.TextColumn("Position"),
                        "Minutes": st.column_config.NumberColumn("Minutes", format="%d"),
                        "Percentage": st.column_config.ProgressColumn(
                            "% of Minutes",
                            min_value=0,
                            max_value=100,
                            format="%.1f%%",
                        ),
                    },
                    hide_index=True,
                    use_container_width=False,
                )

        with _col_outliers:
            # Compute outlier stats: metrics beyond 2σ / 3σ from positional mean
            try:
                # Use the best role's position group as the population
                _outlier_pop = radar_stats_df[
                    radar_stats_df['primaryPosition'].isin(POSITION_GROUPS.get(best_role, [selected_raw_pos]))
                ].copy() if 'best_role' in dir() and best_role else radar_stats_df.copy()
                if 'totalMinutes' in _outlier_pop.columns:
                    _outlier_pop = _outlier_pop[pd.to_numeric(_outlier_pop['totalMinutes'], errors='coerce').fillna(0) >= 500]

                # Gather all numeric per-90 metrics (exclude info/intermediate columns)
                _skip_cols = {'playerName', 'teamName', 'totalMinutes', 'primaryPosition', 'secondaryPosition',
                              'tertiaryPosition', 'playerId', 'player.id', 'Defensive Area', 'Expected xT at Center',
                              'competitionId', 'competitionId_per_90'}
                _skip_suffixes = ('_percentile', '_Score', '_TotalScore', '_Rank')
                _all_metrics = [c for c in radar_player_data_row.columns
                                if pd.api.types.is_numeric_dtype(radar_player_data_row[c])
                                and c not in _skip_cols
                                and not c.endswith(_skip_suffixes)]

                _outliers_2s = []  # (metric, value, z_score, direction)
                _outliers_3s = []
                for _m in _all_metrics:
                    if _m not in _outlier_pop.columns:
                        continue
                    _pop_vals = pd.to_numeric(_outlier_pop[_m], errors='coerce').dropna()
                    if len(_pop_vals) < 5:
                        continue
                    _mean = _pop_vals.mean()
                    _std = _pop_vals.std()
                    if _std == 0:
                        continue
                    _player_val = float(radar_player_data_row[_m].values[0])
                    _z = (_player_val - _mean) / _std
                    # For inverted metrics, negative z is "good" (below average = better)
                    if _m in INVERT_METRICS:
                        _direction = "⬇️" if _z < 0 else "⬆️"
                    else:
                        _direction = "⬆️" if _z > 0 else "⬇️"
                    if abs(_z) >= 3:
                        _outliers_3s.append((_m, _player_val, _z, _direction))
                    elif abs(_z) >= 2:
                        _outliers_2s.append((_m, _player_val, _z, _direction))

                # Sort by absolute z-score descending
                _outliers_3s.sort(key=lambda x: abs(x[2]), reverse=True)
                _outliers_2s.sort(key=lambda x: abs(x[2]), reverse=True)

                if _outliers_3s or _outliers_2s:
                    st.caption("Statistical Outliers (vs. positional avg)")
                    if _outliers_3s:
                        st.markdown("**🔴 Super-outliers (> 3σ)**")
                        for _m, _v, _z, _d in _outliers_3s:
                            st.markdown(f"&nbsp;&nbsp;{_d} **{_m}**: {fmt_val(_m, _v)} p90 &nbsp;({_z:+.1f}σ)")
                    if _outliers_2s:
                        st.markdown("**🟡 Outliers (> 2σ)**")
                        for _m, _v, _z, _d in _outliers_2s:
                            st.markdown(f"&nbsp;&nbsp;{_d} **{_m}**: {fmt_val(_m, _v)} p90 &nbsp;({_z:+.1f}σ)")
                else:
                    st.caption("No statistical outliers (all metrics within 2σ of positional mean)")
            except Exception as e:
                print(f"Warning: Could not compute outlier stats: {e}")



    elif _active_tab == "Stats":
        # --- 5b. Career Trajectory ----------------------------------------
        # Per-season summary of the player's appearances + per-season
        # strip plots of the chosen metric (Action V/90 or Best-fit
        # Rating) showing the full league-wide distribution with the
        # selected player highlighted. Lives just above Overall Season
        # Stats so the trajectory context flows into the per-season
        # breakdown right below it.
        st.subheader("Career Trajectory")

        from plotly.subplots import make_subplots as _make_subplots

        def _season_start_year(season_label: str) -> int | None:
            """Parse a season label like '2025/26' or '2025/2026' and
            return the start year. Used to sort seasons chronologically
            because raw seasonIds aren't comparable across competitions
            (e.g. Camp 23/24 seasonId 190230 sorts AFTER Liga 3 24/25
            seasonId 190090 by numeric value)."""
            try:
                return int(str(season_label).split('/')[0])
            except (ValueError, AttributeError, IndexError):
                return None

        def _age_at_season_label(birth, season_label) -> float | None:
            """Age at midpoint of the season (start year + 0.5)."""
            if not birth or pd.isna(birth):
                return None
            try:
                from datetime import datetime
                bd = pd.to_datetime(birth, errors='coerce')
                if pd.isna(bd):
                    return None
                start_year = _season_start_year(season_label)
                if start_year is None:
                    return None
                mid_season = datetime(start_year, 12, 31)
                return round((mid_season - bd.to_pydatetime()).days / 365.25, 1)
            except Exception:
                return None

        # ---- 1) Walk player_minutes_data to build per-season rows ----
        career_rows = []
        birth = player_bio.get('birthDate') if isinstance(player_bio, pd.Series) else None
        for _sid, _pm_df in player_minutes_data.items():
            if not isinstance(_pm_df, pd.DataFrame) or _pm_df.empty:
                continue
            if 'playerId' not in _pm_df.columns:
                continue
            sub = _pm_df[_pm_df['playerId'] == player_id]
            if sub.empty:
                continue
            for _, row in sub.iterrows():
                _comp_id = competition_for_season(_sid)
                _season_label = SEASON_ID_MAP.get(_sid, str(_sid))
                _comp_name = (COMPETITIONS.get(_comp_id, {}).get('name')
                               if _comp_id else 'Other')
                career_rows.append({
                    'Season': _season_label,
                    '_seasonId': _sid,
                    '_startYear': _season_start_year(_season_label) or 0,
                    'Competition': _comp_name or 'Other',
                    '_compId': _comp_id,
                    'Team': row.get('teamName', 'N/A'),
                    'Position': row.get('primaryPosition', 'N/A'),
                    'Minutes': int(row.get('totalMinutes', 0) or 0),
                    'Age': _age_at_season_label(birth, _season_label),
                })

        if not career_rows:
            st.caption("No multi-season history available for this player.")
        else:
            # Chronological sort: by parsed start year, then by competition
            # id as a stable tiebreaker (so Liga 3 23/24 + Camp 23/24
            # land next to each other deterministically).
            career_df = (pd.DataFrame(career_rows)
                          .sort_values(['_startYear', '_compId', '_seasonId']))

            # ---- 2) Merge in GPA Total Value per season for the player ----
            _gpa_full = None
            try:
                _gpa_full = load_gpa_values()
                if _gpa_full is not None and not _gpa_full.empty \
                        and 'playerId' in _gpa_full.columns:
                    _gpa_player = _gpa_full[_gpa_full['playerId'] == player_id].copy()
                    if not _gpa_player.empty:
                        _val_col = next((c for c in ('Total Value', 'total_v_per_90')
                                          if c in _gpa_player.columns), None)
                        if _val_col:
                            career_df = career_df.merge(
                                _gpa_player[['seasonId', _val_col]]
                                    .rename(columns={'seasonId': '_seasonId',
                                                      _val_col: 'Action V/90'})
                                    .drop_duplicates('_seasonId'),
                                on='_seasonId', how='left'
                            )
            except Exception:
                pass

            # ---- 3) Per-season best-fit rating + per-season population ----
            # For each season the player has data in, also collect the
            # FULL league-wide distribution of the metric so we can plot
            # a violin per season with the player highlighted.
            _role_rating_by_season: dict[int, float] = {}
            _role_name_by_season:   dict[int, str]   = {}
            _rating_population: dict[int, np.ndarray] = {}  # sid -> array of all players' best-fit scores

            def _best_fit_score(row, _weights, _groups):
                pos = row.get('primaryPosition')
                if pd.isna(pos):
                    return None
                eligible = [r for r in _weights if pos in _groups.get(r, [])]
                vals = [row.get(f"{r}_Score") for r in eligible
                         if f"{r}_Score" in row.index]
                vals = [float(v) for v in vals if v is not None and not pd.isna(v)]
                return max(vals) if vals else None

            def _position_group_of(pos):
                """Map a Wyscout primaryPosition to its top-level position
                group (GK/CB/FB/CM/AM/WG/ST). Best-fit Rating is a
                position-specific composite, so the per-season violin
                should compare against same-position peers only — not
                fullbacks vs centerbacks vs forwards."""
                if pos is None or pd.isna(pos):
                    return None
                p = str(pos)
                if p == 'GK': return 'GK'
                if p in ('CB', 'LCB', 'RCB', 'LCB3', 'RCB3'): return 'CB'
                if p in ('LB', 'RB', 'LB5', 'RB5', 'LWB', 'RWB'): return 'FB'
                if p in ('CMF', 'LCMF', 'RCMF', 'LCMF3', 'RCMF3',
                          'DMF', 'LDMF', 'RDMF'): return 'CM'
                if p in ('AMF', 'LAMF', 'RAMF', 'LMF', 'RMF'): return 'AM'
                if p in ('LW', 'RW', 'LWF', 'RWF'): return 'WG'
                if p in ('CF', 'SS'): return 'ST'
                return None

            try:
                for _sid in career_df['_seasonId'].unique():
                    _events_sid = get_season_events(raw_events_df, [_sid])
                    _minutes_sid = player_minutes_data.get(_sid)
                    if _events_sid.empty or _minutes_sid is None or _minutes_sid.empty:
                        continue
                    _stats_sid = calculate_all_player_stats(
                        _events_sid, _minutes_sid, season_id=_sid
                    )
                    if _stats_sid.empty:
                        continue
                    _scores_sid = calculate_player_percentiles_and_scores(
                        _stats_sid, POSITION_GROUPS, WEIGHTS, INVERT_METRICS,
                        min_minutes=500, season_id=_sid
                    )
                    if _scores_sid.empty:
                        continue
                    # Per-player best-fit role score (vectorized via apply).
                    _scores_sid = _scores_sid.copy()
                    _scores_sid['_best_fit'] = _scores_sid.apply(
                        _best_fit_score, axis=1,
                        args=(WEIGHTS, POSITION_GROUPS),
                    )
                    # The selected player's row — needed up-front so we
                    # can filter the population to same-position peers.
                    _player_rows = _scores_sid[_scores_sid['playerId'] == player_id]
                    if _player_rows.empty:
                        continue
                    _player_row = _player_rows.iloc[0]
                    _pos = _player_row.get('primaryPosition')
                    _player_pos_group = _position_group_of(_pos)

                    # Population for violin = ≥500-min players in the
                    # SAME position group as the selected player this
                    # season. Best-fit Rating is position-specific so
                    # comparing a CB to wingers isn't meaningful.
                    if 'totalMinutes' in _scores_sid.columns:
                        _qualified = _scores_sid[
                            (_scores_sid['totalMinutes'].fillna(0) >= 500)
                            & _scores_sid['_best_fit'].notna()
                        ]
                    else:
                        _qualified = _scores_sid[_scores_sid['_best_fit'].notna()]
                    if _player_pos_group and 'primaryPosition' in _qualified.columns:
                        _qualified = _qualified[
                            _qualified['primaryPosition'].map(_position_group_of)
                            == _player_pos_group
                        ]
                    _pop = _qualified['_best_fit'].values
                    if len(_pop) > 0:
                        _rating_population[int(_sid)] = _pop
                    _eligible = [r for r in WEIGHTS
                                  if _pos in POSITION_GROUPS.get(r, [])]
                    _scored = [(r, float(_player_row.get(f"{r}_Score", 0) or 0))
                                for r in _eligible
                                if f"{r}_Score" in _player_row.index]
                    if not _scored:
                        continue
                    _best_role, _best_score = max(_scored, key=lambda t: t[1])
                    _role_rating_by_season[int(_sid)] = round(_best_score, 1)
                    _role_name_by_season[int(_sid)] = _best_role
            except Exception as _exc:
                logger.warning(f"Could not compute per-season role ratings: {_exc}")

            if _role_rating_by_season:
                career_df['Best-fit Rating'] = career_df['_seasonId'].map(_role_rating_by_season)
                career_df['Best-fit Role']   = career_df['_seasonId'].map(_role_name_by_season)

            # ---- 4) Display per-season table ----
            _display_career = career_df.drop(columns=[c for c in career_df.columns if c.startswith('_')])
            if 'Action V/90' in _display_career.columns:
                _display_career['Action V/90'] = _display_career['Action V/90'].round(3)
            st.dataframe(_display_career, use_container_width=True, hide_index=True, column_config=auto_column_config(_display_career))

            # ---- 5) Per-season strip plots with player highlighted ----
            _chart_y_candidates = [m for m in ('Action V/90', 'Best-fit Rating')
                                    if m in career_df.columns
                                    and career_df[m].notna().any()]
            if not _chart_y_candidates:
                st.caption(
                    "No Action V/90 or Best-fit Rating data available across "
                    "this player's seasons — chart suppressed."
                )
            else:
                chart_y = st.radio(
                    "Trajectory metric:",
                    _chart_y_candidates,
                    horizontal=True,
                    key=f"_traj_metric_{player_id}",
                )
                # Build (season, label, population, player_value) list in
                # chronological order. Skip seasons where the player has
                # no value for the chosen metric — the violin without a
                # highlighted dot is just visual noise.
                _panels = []
                _seen_sids = set()
                for _, _row in career_df.iterrows():
                    _sid = int(_row['_seasonId'])
                    if _sid in _seen_sids:
                        continue  # de-dupe rows where player had multi teams
                    _seen_sids.add(_sid)
                    _pv = _row.get(chart_y)
                    if pd.isna(_pv):
                        continue
                    if chart_y == 'Action V/90' and _gpa_full is not None:
                        _val_col = next((c for c in ('Total Value', 'total_v_per_90')
                                          if c in _gpa_full.columns), None)
                        if _val_col:
                            # Filter to ≥500-min sample so the population
                            # represents proper regular-rotation players.
                            _gpa_season = _gpa_full.loc[
                                (_gpa_full['seasonId'] == _sid)
                                & (_gpa_full.get('mins_played', 0) >= 500)
                            ]
                            # Filter to same position group as the
                            # selected player in this season. Prefer
                            # GPA's own position_group column (source-
                            # of-truth for this dataset); fall back to
                            # deriving from the `position` column via
                            # _position_group_of if missing.
                            _player_pg = None
                            _gpa_player_row = _gpa_full[
                                (_gpa_full['playerId'] == player_id)
                                & (_gpa_full['seasonId'] == _sid)
                            ]
                            if not _gpa_player_row.empty:
                                if 'position_group' in _gpa_player_row.columns:
                                    _v = _gpa_player_row['position_group'].iloc[0]
                                    if pd.notna(_v):
                                        _player_pg = str(_v)
                                if _player_pg is None and 'position' in _gpa_player_row.columns:
                                    _player_pg = _position_group_of(
                                        _gpa_player_row['position'].iloc[0]
                                    )
                            if _player_pg:
                                if 'position_group' in _gpa_season.columns:
                                    _gpa_season = _gpa_season[
                                        _gpa_season['position_group'] == _player_pg
                                    ]
                                elif 'position' in _gpa_season.columns:
                                    _gpa_season = _gpa_season[
                                        _gpa_season['position'].map(_position_group_of)
                                        == _player_pg
                                    ]
                            _pop = _gpa_season[_val_col].dropna().values
                        else:
                            _pop = np.array([])
                    else:
                        _pop = _rating_population.get(_sid, np.array([]))
                    if len(_pop) < 5:
                        continue
                    _age_str = (f" · age {_row['Age']:.0f}"
                                 if pd.notna(_row.get('Age')) else "")
                    _comp_short = ('L3' if _row.get('_compId') == 43324
                                    else 'CP' if _row.get('_compId') == 702
                                    else (_row.get('Competition') or '')[:6])
                    _panels.append({
                        'sid': _sid,
                        'label': f"{_row['Season']}<br>{_comp_short}{_age_str}",
                        'population': _pop,
                        'player_value': float(_pv),
                        'team': _row.get('Team', ''),
                    })

                if not _panels:
                    st.caption(
                        f"No seasons have both a {chart_y} value for this "
                        f"player AND a comparable population to plot."
                    )
                else:
                    # One subplot per season, shared y-axis so the player's
                    # trajectory is easy to follow across seasons.
                    _fig = _make_subplots(
                        rows=1, cols=len(_panels),
                        shared_yaxes=True,
                        subplot_titles=[p['label'] for p in _panels],
                        horizontal_spacing=0.01,
                    )
                    # Common y-range — driven by the ≥500-min POPULATION
                    # only (per user request). If the highlighted player
                    # is outside that range we extend slightly so the gold
                    # dot is still visible, but the scale is anchored to
                    # the regular-rotation population.
                    _pop_concat = np.concatenate([p['population'] for p in _panels])
                    _y_lo = float(np.nanmin(_pop_concat))
                    _y_hi = float(np.nanmax(_pop_concat))
                    _y_pad = 0.05 * (_y_hi - _y_lo if _y_hi > _y_lo else 1.0)
                    # Allow the highlighted dot to bleed up to 1 pad
                    # outside the population range without rescaling the
                    # whole panel.
                    _player_vals = [p['player_value'] for p in _panels]
                    _y_lo = min(_y_lo, float(np.nanmin(_player_vals)) - _y_pad)
                    _y_hi = max(_y_hi, float(np.nanmax(_player_vals)) + _y_pad)
                    # Width scaling: seasons with more ≥500-min players get
                    # visibly wider violins so the user can tell apart a
                    # 200-player Liga 3 season from a 600-player Camp
                    # season. Use a power < 1 so small samples don't
                    # collapse to slivers.
                    _max_n = max(len(p['population']) for p in _panels) or 1
                    for _i, _p in enumerate(_panels, start=1):
                        _n = len(_p['population'])
                        _scaled_w = 0.85 * (_n / _max_n) ** 0.5
                        # 1) Violin: density shape, no built-in points (we
                        # render our own colored dots in the next trace).
                        # Explicit x=0 anchor — without this, plotly
                        # picks categorical mode and the highlight dot's
                        # numeric x positioning silently breaks.
                        _fig.add_trace(go.Violin(
                            x=np.zeros(len(_p['population'])),
                            y=_p['population'],
                            points=False,
                            box_visible=False,
                            meanline_visible=False,
                            side='both',
                            width=_scaled_w,
                            line_color='rgba(80,80,80,0.55)',
                            fillcolor='rgba(140,140,140,0.18)',
                            showlegend=False,
                            hoverinfo='skip',
                            name='',
                        ), row=1, col=_i)

                        # 2) Colored dots: deterministic jitter so the
                        # layout doesn't shift on rerun (seeded by sid),
                        # bounded so dots stay inside the violin.
                        _rng = np.random.default_rng(seed=int(_p['sid']) & 0xFFFFFFFF)
                        _half = max(0.05, _scaled_w / 2 - 0.04)
                        _jitter = _rng.uniform(-_half, _half, size=_n)
                        _fig.add_trace(go.Scatter(
                            x=_jitter,
                            y=_p['population'],
                            mode='markers',
                            marker=dict(
                                size=5,
                                color=_p['population'],
                                colorscale='RdYlGn',
                                cmin=_y_lo, cmax=_y_hi,
                                opacity=0.6,
                                line=dict(width=0),
                                showscale=False,
                            ),
                            showlegend=False,
                            hoverinfo='y',
                            name='',
                        ), row=1, col=_i)

                        # 3) Highlight: the selected player's point on top.
                        # Bumped to a larger diamond with a thicker dark
                        # ring so it stays unmistakable against the
                        # dense violin shape.
                        _fig.add_trace(go.Scatter(
                            y=[_p['player_value']],
                            x=[0.0],
                            mode='markers',
                            marker=dict(
                                size=18,
                                color='#FFC400',
                                line=dict(color='black', width=2),
                                symbol='diamond',
                            ),
                            showlegend=False,
                            hovertemplate=(
                                f"<b>{selected_player_name}</b><br>"
                                f"Team: {_p['team']}<br>"
                                f"{chart_y}: %{{y:.2f}}<extra></extra>"
                            ),
                            name='',
                        ), row=1, col=_i)
                        # Force the x-axis to linear — adding go.Violin
                        # without an explicit x array flips Plotly into
                        # categorical-axis mode, which silently clobbers
                        # the numeric x positions we use for jitter and
                        # the highlight dot (so they collapse onto a
                        # single phantom category and the gold dot got
                        # buried under the colored cloud).
                        _fig.update_xaxes(
                            type='linear',
                            showticklabels=False, zeroline=False,
                            range=[-0.5, 0.5], row=1, col=_i,
                        )
                    _fig.update_yaxes(range=[_y_lo - _y_pad, _y_hi + _y_pad])
                    _fig.update_layout(
                        title=f"{chart_y} by season (gold dot = {selected_player_name})",
                        height=420,
                        margin=dict(t=70, b=30, l=40, r=20),
                        showlegend=False,
                    )
                    # Make subplot titles smaller so they fit when there
                    # are 5+ panels.
                    for _ann in _fig['layout']['annotations']:
                        _ann['font'] = dict(size=10)
                    st.plotly_chart(_fig, use_container_width=True)

        st.divider()

        # --- 6. STATS TOGGLE ---
        st.subheader("Overall Season Stats")
        show_totals = st.toggle("Show Season Totals", value=False)
        stats_to_display = pd.Series(dtype='object')

        per_90_stats = player_per_90_stats.copy()

        if show_totals:
            st.text(f"Displaying TOTAL stats from {total_minutes:.0f} minutes played.")
            total_stats = per_90_stats.copy()
            rate_cols = [col for col in total_stats.index if '%' in col or 'per' in col.lower() or 'index' in col or 'Percentage' in col]
            # goalsConceded is conceptually a defensive rate (goals against per 90)
            # rather than a count — it stays per-90 even in season-totals mode,
            # the same way 'goalsPrevented' / xG family already do.
            if 'goalsConceded' in total_stats.index and 'goalsConceded' not in rate_cols:
                rate_cols.append('goalsConceded')
            # engine metrics are levels, never season-totaled
            for _ec in ENGINE_DISPLAY_METRICS:
                if _ec in total_stats.index and _ec not in rate_cols:
                    rate_cols.append(_ec)

            for col in total_stats.index:
                if col not in rate_cols and pd.api.types.is_numeric_dtype(total_stats[col]):
                    total_val = (total_stats[col] * total_minutes) / 90
                    if col in ['xG', 'xA', 'xT', 'xTOP', 'xTSP', 'npxG', 'xAOP', 'xASP', 'psxG_faced', 'goalsPrevented']:
                         total_stats[col] = total_val
                    else:
                         total_stats[col] = np.round(total_val)

            for col in rate_cols:
                if col in per_90_stats.index:
                    total_stats[col] = per_90_stats[col]

            stats_to_display = total_stats

        else: # Show Per 90
            st.text(f"Displaying PER 90 stats from {total_minutes:.0f} minutes played.")
            stats_to_display = per_90_stats

        # --- 7. Display Stats (Using all global groups) ---
        stat_groups = {
            "Output": OUTPUT_METRICS,
            "Passing": PASSING_METRICS,
            "Defensive": DEFENSIVE_METRICS,
            "Defensive Responsibility (DefR)": DEFR_DISPLAY_METRICS,
            "Dribbling": DRIBBLING_METRICS,
            "ACP Index": ENGINE_DISPLAY_METRICS,
            "Goalkeeping": GOALKEEPING_METRICS
        }

        player_is_gk = (per_90_stats.get('primaryPosition', 'N/A') == 'GK')

        # ── Build positional-peer population for percentile comparisons ──
        # Union of all raw positions that share at least one role template
        # with the player's primary position; filter to 500+ minutes.
        _primary_pos = player_per_90_stats.get('primaryPosition', None)
        _peer_pop = pd.DataFrame()
        if _primary_pos and _primary_pos not in ('N/A', 'Unknown', None, ''):
            _peer_positions = set()
            for _role, _positions in POSITION_GROUPS.items():
                if _primary_pos in _positions:
                    _peer_positions.update(_positions)
            if not _peer_positions:
                _peer_positions = {_primary_pos}
            _peer_pop = radar_stats_df[radar_stats_df['primaryPosition'].isin(_peer_positions)]
            if 'totalMinutes' in _peer_pop.columns:
                _peer_pop = _peer_pop[
                    pd.to_numeric(_peer_pop['totalMinutes'], errors='coerce').fillna(0) >= 500
                ]

        def _percentile_for(metric_name, p90_value):
            """Percentile of this player's per-90 value against same-position peers (≥500 min)."""
            if _peer_pop.empty or metric_name not in _peer_pop.columns:
                return None
            if pd.isna(p90_value):
                return None
            pop_vals = pd.to_numeric(_peer_pop[metric_name], errors='coerce').dropna()
            if len(pop_vals) < 5 or pop_vals.std() == 0:
                return None
            pct = scipy.stats.percentileofscore(pop_vals, p90_value, kind='weak')
            if metric_name in INVERT_METRICS:
                pct = 100.0 - pct
            return float(pct)

        def _percentile_color(p):
            """Red (0) → yellow (50) → green (100) HSL gradient."""
            if p is None or pd.isna(p):
                return ''
            p_clamped = max(0.0, min(100.0, float(p)))
            hue = (p_clamped / 100.0) * 120.0   # 0=red, 60=yellow, 120=green
            return f'background-color: hsl({hue:.0f}, 65%, 72%); color: black;'

        for group_name, group_metrics in stat_groups.items():

            if player_is_gk and group_name != 'Goalkeeping':
                continue
            if not player_is_gk and group_name == 'Goalkeeping':
                continue

            if player_is_gk and group_name == 'Goalkeeping':
                group_metrics = GOALKEEPING_METRICS + ['GK Passes successful %', 'GK Long passes successful %']

            metrics_to_show = [m for m in group_metrics if m in stats_to_display.index]

            if metrics_to_show:
                default_expanded = (group_name == 'Output')
                with st.expander(f"**{group_name} Stats**", expanded=default_expanded):

                    stats_subset_series = stats_to_display[metrics_to_show]
                    stats_subset_series = stats_subset_series[stats_subset_series != 0]

                    if stats_subset_series.empty:
                        st.text("No data for this category.")
                        continue

                    def _fmt_stat(metric_name, x):
                        if not isinstance(x, (int, float)):
                            return str(x)
                        if metric_name in THOUSANDTHS_METRICS:
                            return f"{x:.3f}"
                        if np.round(x) == x and '%' not in str(x):
                            return f"{x:.0f}"
                        return f"{x:.2f}"

                    # Build display DataFrame: Value (formatted str) + Percentile (numeric)
                    _rows = []
                    for _metric in stats_subset_series.index:
                        _disp_val = stats_subset_series[_metric]
                        _p90_val = per_90_stats.get(_metric, np.nan)
                        _pct = _percentile_for(_metric, _p90_val)
                        _rows.append({
                            'Metric': _metric,
                            'Value': _fmt_stat(_metric, _disp_val),
                            'Percentile': _pct,
                        })
                    stats_subset = pd.DataFrame(_rows).set_index('Metric')

                    _styled = (
                        stats_subset.style
                        .applymap(_percentile_color, subset=['Percentile'])
                        .format({'Percentile': lambda v: f"{int(round(v))}" if pd.notna(v) else '—'})
                    )
                    st.dataframe(_styled, use_container_width=True)


    elif _active_tab == "Value":
        st.divider()

        # --- Transfer Value Detail ----------------------------------------
        # Deep-dive on market value: Reported fees & manual entries
        # expander + the Market Context features block. The legacy CVI
        # breakdown display panels were retired (Lucas 2026-06-12) — the
        # ACP engine provides the headline Projected value in the bio
        # card. CVI computation helpers stay at module level for other
        # pages.
        st.subheader("Transfer Value Detail")
        _vt_rng = (None if _is_gk else eur_interval_ui.range_sentence(
            _eng_proj_eur, _eurcal, w_evidence=_eng_w_ev, mins=_eng_mins))
        st.caption("Projected value is computed by the ACP engine (see bio card)."
                   + (f" {_vt_rng}" if _vt_rng else ""))
        # A toggle, not an expander: the panel holds a Plotly chart, which
        # keeps its collapsed height when first drawn inside an expander.
        if st.toggle("How reliable is the projected value? Show the fee calibration",
                     value=False, key=f"eurcal_toggle_{player_id}"):
            eur_interval_ui.render_eur_calibration_section(_eurcal, key=f"eurcal_{player_id}")
        try:
            with st.expander("Reported transfer fees & manual entries",
                              expanded=False):
                st.caption("**Market value sources**")
                if _tv_valuations_rows.empty:
                    st.caption("No data yet. Populates from reported transfer "
                                "fees + manual entries.")
                else:
                    _src_view = (_tv_valuations_rows
                                  .groupby('source', as_index=False)
                                  .first()[['source', 'value_eur', 'as_of_date']])
                    _src_view['value_eur'] = _src_view['value_eur'].apply(
                        lambda v: f"€{v:,.0f}" if pd.notna(v) else "—"
                    )
                    st.dataframe(_src_view, use_container_width=True, hide_index=True, column_config=auto_column_config(_src_view))
                    if len(_tv_valuations_rows) > len(_src_view):
                        with st.expander(f"Full history ({len(_tv_valuations_rows)} entries)"):
                            _hist_view = _tv_valuations_rows[
                                ['source', 'value_eur', 'as_of_date', 'notes']
                            ].copy()
                            _hist_view['value_eur'] = _hist_view['value_eur'].apply(
                                lambda v: f"€{v:,.0f}" if pd.notna(v) else "—"
                            )
                            st.dataframe(_hist_view, use_container_width=True,
                                          hide_index=True,
                                          column_config=auto_column_config(_hist_view))

                # ---- Manual valuation entry ----
                # Add a hand-entered figure from club / agent conversations.
                # Highest-authority source (weight 4.0 in the loader's blend).
                with st.expander("➕ Add manual valuation", expanded=False):
                    with st.form(f"manual_val_{player_id}_{selected_season_id}",
                                  clear_on_submit=True):
                        _mv_col_a, _mv_col_b = st.columns(2)
                        _mv_eur = _mv_col_a.number_input(
                            "Value (EUR)", min_value=0, step=10_000,
                            value=0, help="Hand-entered figure from club "
                                          "or agent conversation. €0 = skip.",
                        )
                        from datetime import date as _date_cls
                        _mv_date = _mv_col_b.date_input(
                            "As-of date", value=_date_cls.today(),
                            help="When this valuation was given to you.",
                        )
                        _mv_notes = st.text_input(
                            "Notes (optional)",
                            placeholder="e.g. 'agent quote', 'club asking price', "
                                        "'rejected bid from X'",
                        )
                        _mv_submitted = st.form_submit_button("Save",
                                                                type="primary")
                        if _mv_submitted:
                            if _mv_eur <= 0:
                                st.warning("Value must be > €0 — skipping.")
                            else:
                                try:
                                    import csv
                                    _man_path = (Path(__file__).resolve().parent
                                                  / 'valuations'
                                                  / 'manual_entries.csv')
                                    _man_path.parent.mkdir(exist_ok=True)
                                    _new_file = not _man_path.exists()
                                    with open(_man_path, 'a', newline='') as _f:
                                        _w = csv.writer(_f)
                                        if _new_file:
                                            _w.writerow(['playerId', 'value_eur',
                                                          'as_of_date', 'season_id',
                                                          'source_url', 'notes'])
                                        _w.writerow([
                                            int(player_id), int(_mv_eur),
                                            _mv_date.isoformat(),
                                            (int(selected_season_id)
                                             if selected_season_id else ''),
                                            '',
                                            (f"{_mv_notes} | added via dashboard"
                                             if _mv_notes else "added via dashboard"),
                                        ])
                                    st.success(
                                        f"Saved: €{_mv_eur:,} as of {_mv_date} "
                                        f"for {selected_player_name}. "
                                        f"Refresh the page to see it in the True value."
                                    )
                                except Exception as _save_exc:
                                    st.error(f"Could not save: "
                                              f"{type(_save_exc).__name__}: {_save_exc}")

            # ---- Market Context features ----
            st.markdown("##### Market Context")
            try:
                _tv_team = (str(_tv_player_row.get('teamName'))
                             if _tv_player_row is not None
                             and pd.notna(_tv_player_row.get('teamName'))
                             else None)
                _opta_fn = (make_opta_team_strength_lookup()
                             if 'make_opta_team_strength_lookup' in globals()
                             else (lambda _t: None))
                _mc = compute_market_features(
                    player_id=player_id,
                    season_id=selected_season_id,
                    raw_events_df=raw_events_df,
                    matches_summary_df=matches_summary_df,
                    player_details_df=player_details_df,
                    player_minutes_data=player_minutes_data,
                    team_name=_tv_team,
                    opta_team_lookup=_opta_fn,
                )
                _mc_c1, _mc_c2, _mc_c3, _mc_c4 = st.columns(4)
                def _fmt_resid(v, n_dec=1):
                    if v is None or pd.isna(v): return "—"
                    return f"{v:+.{n_dec}f}"

                _mc_c1.metric("xG O/U (season)",
                                _fmt_resid(_mc['xg_residual_season']),
                                help="Goals minus xG, non-penalty, this season. "
                                     "Positive = outperforming xG (clinical "
                                     "finishing or variance); negative = "
                                     "underperforming.")
                _mc_c1.metric("xG O/U (career)",
                                _fmt_resid(_mc['xg_residual_career']),
                                help="Cumulative across all seasons in our "
                                     "data. More stable than single-season "
                                     "residuals.")
                _mc_c2.metric("xA O/U (season)",
                                _fmt_resid(_mc['ass_residual_season']),
                                help="Assists minus xA proxy (sum of xG of "
                                     "shots the player set up).")
                _mc_c2.metric("xA O/U (career)",
                                _fmt_resid(_mc['ass_residual_career']))

                _nat_p = _mc.get('passport_nationality') or '—'
                _nat_b = _mc.get('birth_nationality') or '—'
                _mc_c3.metric("Nationality (passport)", _nat_p)
                if _nat_b != _nat_p:
                    _mc_c3.metric("Birthplace", _nat_b)

                _team_opta = _mc.get('team_opta_rating')
                _team_ppm = _mc.get('team_ppm_season')
                _team_pos = _mc.get('team_league_position')
                _mc_c4.metric(
                    "Team Opta",
                    f"{_team_opta:.1f}" if _team_opta is not None else "—",
                    help="Current team's Opta Power Ranking — proxy for "
                         "scouting visibility and tier-internal team strength.",
                )
                _mc_c4.metric(
                    "Team this season",
                    (f"{_team_ppm:.2f} PPM" if _team_ppm is not None else "—")
                    + (f" · {_team_pos}." if _team_pos is not None else ""),
                    help="Points per match + league position from parsed scores. "
                         "Successful-team players typically carry a market premium.",
                )

                _ver = _mc.get('positions_played_career')
                _sea = _mc.get('seasons_played')
                if _ver is not None or _sea is not None:
                    _bits = []
                    if _ver is not None:
                        _bits.append(f"{_ver} position{'s' if _ver != 1 else ''} played")
                    if _sea is not None:
                        _bits.append(f"{_sea} season{'s' if _sea != 1 else ''} in data")
                    st.caption("· ".join(_bits))
                st.caption(
                    "📌 These features feed the v2 EUR regression "
                    "(currently pending). They don't change CVI itself."
                )
            except Exception as _mc_exc:
                st.caption(f"Market Context error: "
                            f"{type(_mc_exc).__name__}: {_mc_exc}")
        except Exception as _tv_exc:
            st.caption(f"Transfer Value Detail error: "
                        f"{type(_tv_exc).__name__}: {_tv_exc}")


    elif _active_tab == "Shots & Creation":
        st.divider()

        # --- 7. SHOT ANALYSIS (UPDATED) ---
        st.subheader("Shot Analysis")

        # 1. Get all player events
        player_events_all = profile_events_df[profile_events_df['player.name'] == selected_player_name].copy()

        # 2. Filter for shots (non-penalty) for the map/analysis
        shot_log = player_events_all[
            (player_events_all['type.primary'] == 'shot') &
            (player_events_all['type.primary'] != 'penalty')
        ].copy()

        if not shot_log.empty:
            # --- DATA PROCESSING START ---

            # Sort chronologically for numbering (oldest first)
            if 'dateutc' in shot_log.columns:
                shot_log = shot_log.sort_values(by=['dateutc', 'minute', 'second'], ascending=True)
            else:
                shot_log = shot_log.sort_values(by=['matchId', 'minute', 'second'], ascending=True)

            # Assign Shot Numbers (1 to N)
            shot_log.reset_index(drop=True, inplace=True)
            shot_log['Shot Number'] = shot_log.index + 1

            # Basic formatting
            shot_log['Date'] = pd.to_datetime(shot_log['dateutc']).dt.strftime('%Y-%m-%d') if 'dateutc' in shot_log.columns else "N/A"
            shot_log['Opponent'] = shot_log.get('opponentTeam.name', 'Unknown')
            shot_log['xG'] = pd.to_numeric(shot_log['shot.xg'], errors='coerce').fillna(0)
            shot_log['Result'] = np.where(shot_log['shot.isGoal'] == True, 'Goal', 
                                 np.where(shot_log['shot.onTarget'] == True, 'Saved', 'Off Target'))

            # Body Part Extraction
            if 'shot.bodyPart.name' in shot_log.columns:
                shot_log['Body Part'] = shot_log['shot.bodyPart.name']
            elif 'shot.bodyPart' in shot_log.columns:
                shot_log['Body Part'] = shot_log['shot.bodyPart'].apply(
                    lambda x: x.get('name', 'Unknown') if isinstance(x, dict) else str(x)
                )
                shot_log['Body Part'] = shot_log['Body Part'].str.replace('_', ' ').str.title()
            else:
                shot_log['Body Part'] = 'Unknown'

            # Phase of Play
            def get_phase(possession_types):
                if not isinstance(possession_types, (list, np.ndarray)): return "Open Play"
                if 'counter_attack' in possession_types: return "Counter Attack"
                if 'corner' in possession_types or 'free_kick' in possession_types or 'penalty' in possession_types: return "Set Piece"
                if 'positional_attack' in possession_types: return "Positional Attack"
                return "Open Play"

            if 'possession.types' in shot_log.columns:
                shot_log['Phase'] = shot_log['possession.types'].apply(get_phase)
            else:
                shot_log['Phase'] = "Unknown"

            # Shot Creating Action (SCA)
            relevant_match_ids = shot_log['matchId'].unique()
            context_events = profile_events_df[
                (profile_events_df['matchId'].isin(relevant_match_ids)) &
                (profile_events_df['team.name'] == shot_log.iloc[0]['team.name'])
            ].copy()

            shot_log['prev_event_idx'] = shot_log['possession.eventIndex'] - 1

            sca_merge = pd.merge(
                shot_log[['id', 'matchId', 'possession.id', 'prev_event_idx']],
                context_events[['matchId', 'possession.id', 'possession.eventIndex', 'type.primary', 'type.secondary']],
                left_on=['matchId', 'possession.id', 'prev_event_idx'],
                right_on=['matchId', 'possession.id', 'possession.eventIndex'],
                how='left',
                suffixes=('', '_prev')
            )

            def label_sca(row):
                if pd.isna(row['type.primary']): return "Recovery/None"
                sec_types = row['type.secondary'] if isinstance(row['type.secondary'], (list, np.ndarray)) else []
                if 'cross' in sec_types: return "Cross"
                if 'through_pass' in sec_types: return "Through Pass"
                if 'deep_completion' in sec_types: return "Deep Completion"
                prim = row['type.primary']
                if prim == 'pass': return "Pass"
                if prim == 'duel': return "Dribble/Duel"
                if prim == 'acceleration' or prim == 'touch': return "Carry"
                if prim == 'clearance': return "Clearance"
                if prim == 'interception': return "Interception"
                return prim.replace('_', ' ').title()

            sca_merge['SCA'] = sca_merge.apply(label_sca, axis=1)
            shot_log = shot_log.merge(sca_merge[['id', 'SCA']], on='id', how='left')

            # --- DATA PROCESSING END ---

            # --- VISUALIZATION: one combined visual — full-width
            # StatsBomb-style map (shape = creating action, color =
            # xG, ring = goal), shot log directly beneath. Static mpl
            # version kept for the PDF one-pager.
            st.plotly_chart(
                plotly_shot_map(shot_log, selected_player_name,
                                height=context_bar.pitch_height()),
                use_container_width=True,
                config={'displayModeBar': False})

            st.markdown("**Shot Log**")
            display_cols = ['Shot Number', 'Date', 'Opponent', 'Result', 'xG', 'Body Part', 'SCA']
            table_display = shot_log[display_cols].rename(columns={
                'Shot Number': '#',
                'SCA': 'Creating Action'
            }).sort_values(by='#', ascending=False) # Show newest first (highest number)

            st.dataframe(table_display, use_container_width=True, height=380, hide_index=True, column_config=auto_column_config(table_display))

            # --- NEW: SUMMARY TABLES ---
            st.markdown("---")
            col_sum1, col_sum2 = st.columns(2)

            with col_sum1:
                st.markdown("**Stats by Body Part**")
                body_summary = shot_log.groupby('Body Part').agg(
                    Shots=('id', 'count'),
                    Goals=('shot.isGoal', 'sum'),
                    Total_xG=('xG', 'sum')
                ).sort_values(by='Total_xG', ascending=False)
                body_summary['xG/Shot'] = (body_summary['Total_xG'] / body_summary['Shots']).round(2)
                body_summary['Total_xG'] = body_summary['Total_xG'].round(2)
                st.dataframe(body_summary, use_container_width=True, column_config=auto_column_config(body_summary))

            with col_sum2:
                st.markdown("**Stats by Creating Action**")
                sca_summary = shot_log.groupby('SCA').agg(
                    Shots=('id', 'count'),
                    Goals=('shot.isGoal', 'sum'),
                    Total_xG=('xG', 'sum')
                ).sort_values(by='Total_xG', ascending=False)
                sca_summary['xG/Shot'] = (sca_summary['Total_xG'] / sca_summary['Shots']).round(2)
                sca_summary['Total_xG'] = sca_summary['Total_xG'].round(2)
                st.dataframe(sca_summary, use_container_width=True, column_config=auto_column_config(sca_summary))

        else:
            st.info("No shots recorded for this player.")

        st.divider()

        # --- 7a-bis. CREATION — passes into the attacking box ---
        st.subheader("Creation — Passes into the Box")
        _bp_all = load_box_passes()
        if _bp_all.empty:
            st.caption("Box-pass data not available in this deployment.")
        else:
            _bp_seasons = _season_id_list(active_season_ids)
            _bp_p = _bp_all[
                _bp_all['player.id'] == int(selected_player_id)].copy()
            if _bp_seasons:
                _bp_p = _bp_p[_bp_p['seasonId'].isin(_bp_seasons)]
            if _bp_p.empty:
                st.info("No passes into the box recorded for this player "
                        "in the selected season(s).")
            else:
                _cc1, _cc2, _cc3 = st.columns([1, 1, 2])
                with _cc1:
                    _bp_no_sp = st.checkbox(
                        "Open play only", value=False,
                        help="Exclude corner / free-kick / throw-in deliveries")
                with _cc2:
                    _bp_acc_only = st.checkbox("Completed only", value=False)
                _bp_view = _bp_p
                if _bp_no_sp:
                    _bp_view = _bp_view[_bp_view['phase'] != 'set_piece']
                if _bp_acc_only:
                    _bp_view = _bp_view[_bp_view['pass.accurate'] == True]  # noqa: E712
                if _bp_view.empty:
                    st.info("No box passes match the current filters.")
                else:
                    st.plotly_chart(
                        plotly_box_passes_map(_bp_view, selected_player_name, height=context_bar.pitch_height()),
                        use_container_width=True,
                        config={'displayModeBar': False})
                    _n_bp = len(_bp_view)
                    _mets = st.columns(4)
                    _mets[0].metric("Box passes", f"{_n_bp}")
                    _mets[1].metric(
                        "Completed",
                        f"{(_bp_view['pass.accurate'] == True).mean():.0%}")  # noqa: E712
                    _mets[2].metric(
                        "Total pass value",
                        f"{pd.to_numeric(_bp_view['action_value'], errors='coerce').sum():+.3f}",
                        help="Sum of GPA action values of these passes")
                    _mets[3].metric(
                        "Value / pass",
                        f"{pd.to_numeric(_bp_view['action_value'], errors='coerce').mean():+.4f}")
                    st.caption(
                        "Arrow color = GPA pass value (red = value created, "
                        "blue = negative value); arrow weight scales with "
                        "|value|. Hover an endpoint for pass details.")

        st.divider()

        # --- 7b. Shot Assists & Dribbles in Final Third ---
        st.subheader("Shot Assists & Dribbles in Final Third")
        try:
            with MPL_LOCK:
                fig_sa_player = pv.plot_shot_assists_and_dribbles(
                    profile_events_df, current_team,
                    player_name=selected_player_name,
                )
                st.pyplot(fig_sa_player, use_container_width=True)
                plt.close(fig_sa_player)
        except Exception as e:
            st.caption(f"Could not render shot assists & dribbles: {e}")

        st.divider()

        # --- 7c. Defensive Action Heatmap ---
        st.subheader("Defensive Action Heatmap")
        try:
            # Resolve positional peer group for defensive heatmap
            _DEFENSIVE_PEER_GROUPS = {
                'GK': ['GK'],
                'CB': ['CB', 'LCB', 'RCB', 'LCB3', 'RCB3'],
                'FB': ['LB', 'RB', 'LB5', 'RB5', 'LWB', 'RWB'],
                'CM': ['DMF', 'LDMF', 'RDMF', 'LCMF', 'RCMF', 'LCMF3', 'RCMF3'],
                'AM/Wing': ['AMF', 'LAMF', 'RAMF', 'LW', 'RW', 'LWF', 'RWF'],
                'ST': ['CF', 'SS'],
            }
            _heatmap_pos_codes = [current_pos]
            _heatmap_peer_label = current_pos
            for _grp_name, _grp_codes in _DEFENSIVE_PEER_GROUPS.items():
                if current_pos in _grp_codes:
                    _heatmap_pos_codes = _grp_codes
                    _heatmap_peer_label = _grp_name
                    break

            # Compute peer density stack (cached)
            _events_hash = hashlib.md5(
                f"{len(profile_events_df)}_{tuple(sorted(_heatmap_pos_codes))}".encode()
            ).hexdigest()

            _peer_stack = _compute_peer_density_stack(
                _events_hash, profile_events_df,
                tuple(sorted(_heatmap_pos_codes)),
                _player_minutes_df=profile_player_minutes_df,
                include_recoveries=True,
            )

            with MPL_LOCK:
                fig_def_heatmap = pv.plot_defensive_action_heatmap(
                    profile_events_df, player_id, selected_player_name,
                    position_codes=_heatmap_pos_codes,
                    player_minutes_df=profile_player_minutes_df,
                    peer_density_stack=_peer_stack,
                    include_recoveries=True,
                )
                st.pyplot(fig_def_heatmap, use_container_width=True)
                plt.close(fig_def_heatmap)
            st.caption(f"Colour intensity normalised across **{_heatmap_peer_label}** peers.")
        except Exception as e:
            st.caption(f"Could not render defensive action heatmap: {e}")

        st.divider()

        # --- 7a. Throw-In Analysis ---
        st.subheader("Throw-In Analysis")

        try:
            player_throwin_df = profile_events_df[
                (profile_events_df['player.id'] == player_id) &
                (profile_events_df['type.primary'] == 'throw_in')
            ].copy()

            if not player_throwin_df.empty and 'pass.length' in player_throwin_df.columns:
                total_throwins = len(player_throwin_df)

                # Avg of top 10 longest throw-ins (overall distance)
                top_10_all = player_throwin_df.nlargest(min(10, total_throwins), 'pass.length')
                avg_top10_length = top_10_all['pass.length'].mean()

                # Throw-ins into the attacking penalty box (end x >= 84, 20 <= end y <= 80)
                into_box = player_throwin_df[
                    (player_throwin_df['pass.endLocation.x'] >= 84) &
                    (player_throwin_df['pass.endLocation.y'] >= 20) &
                    (player_throwin_df['pass.endLocation.y'] <= 80)
                ]
                if not into_box.empty:
                    top_10_box = into_box.nlargest(min(10, len(into_box)), 'pass.length')
                    avg_top10_into_box = top_10_box['pass.length'].mean()
                else:
                    avg_top10_into_box = 0.0

                # Throw-ins into box where next action is an aerial duel
                avg_top10_into_box_aerial = 0.0
                if not into_box.empty:
                    sorted_match_events = profile_events_df.sort_values(by=['matchId', 'minute', 'second']).reset_index(drop=True)
                    aerial_box_throws = []
                    for _, ti_row in into_box.iterrows():
                        m_id = ti_row.get('matchId')
                        if m_id is None:
                            continue
                        m_events = sorted_match_events[sorted_match_events['matchId'] == m_id]
                        pos_mask = (m_events['minute'] == ti_row['minute']) & (m_events['second'] == ti_row['second']) & (m_events['type.primary'] == 'throw_in')
                        positions = m_events[pos_mask].index
                        if len(positions) == 0:
                            continue
                        next_pos = positions[0] + 1
                        if next_pos in m_events.index:
                            next_sec = m_events.loc[next_pos].get('type.secondary', '')
                            if isinstance(next_sec, (list, set)):
                                is_aerial = 'aerial_duel' in next_sec
                            else:
                                is_aerial = 'aerial_duel' in str(next_sec)
                            if is_aerial:
                                aerial_box_throws.append(ti_row)
                    if aerial_box_throws:
                        aerial_df = pd.DataFrame(aerial_box_throws)
                        top_10_aerial = aerial_df.nlargest(min(10, len(aerial_df)), 'pass.length')
                        avg_top10_into_box_aerial = top_10_aerial['pass.length'].mean()

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Throw-Ins", int(total_throwins))
                with col2:
                    st.metric("Avg Max Distance", f"{avg_top10_length:.1f}m")
                with col3:
                    st.metric("Avg Max Into Box", f"{avg_top10_into_box:.1f}m")
                with col4:
                    st.metric("Avg Max Into Box → Aerial", f"{avg_top10_into_box_aerial:.1f}m")
            else:
                st.info(f"{selected_player_name} has no throw-ins in the selected period.")

        except Exception as e:
            st.caption(f"Could not render throw-in analysis: {e}")


    elif _active_tab == "Match Log":
        st.divider()

        # --- 8. Display Individual Match Stats (Unchanged) ---
        st.subheader("Individual Match Log")

        if player_match_log_df.empty:
            st.info("No individual match stats found for this player.")
        else:
            key_match_stats = ['Date', 'Match', 'Score', 'Minutes', 'Goals / xG', 'xAOP', 'xASP', 'xTOP', 'xTSP', 'Actions / successful', 'Passes / accurate', 'Duels / won']
            cols_to_show = [c for c in key_match_stats if c in player_match_log_df.columns]
            st.dataframe(player_match_log_df[cols_to_show].set_index('Date'))
            with st.expander("View Full Match Log (All Stats)"):
                st.dataframe(player_match_log_df.set_index('Date'))
