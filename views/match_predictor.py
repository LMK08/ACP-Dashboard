"""Match Predictor view — extracted verbatim from app.py's `elif analysis_type == 'Match Predictor'` branch (2026-09).

Collaborators are read from the running app module at call time (the
pattern opposition_report.py uses), so importing this module never imports
app.py. The binding block at the top of render() IS the page's dependency
list: everything it reads from app.py, nothing else.
"""
import numpy as np
import pandas as pd
import pickle
import streamlit as st
import sys


def _app():
    return sys.modules['__main__']


def render():
    app = _app()
    COMPETITIONS = app.COMPETITIONS
    CURRENT_SEASON_ID = app.CURRENT_SEASON_ID
    SEASON_ID_MAP = app.SEASON_ID_MAP
    _predictor_default_labels = app._predictor_default_labels
    auto_column_config = app.auto_column_config
    build_season_cumulative_stats = app.build_season_cumulative_stats
    calculate_rolling_team_strength = app.calculate_rolling_team_strength
    calculate_sos_adjusted_strength = app.calculate_sos_adjusted_strength
    calculate_team_strength = app.calculate_team_strength
    filter_by_league = app.filter_by_league
    get_filtered_events = app.get_filtered_events
    get_season_matches = app.get_season_matches
    league_selector = app.league_selector
    matches_summary_df = app.matches_summary_df
    raw_events_df = app.raw_events_df

    selected_comp_ids = league_selector("match_predictor")
    st.markdown("Predict the outcome of upcoming matches based on team performance data with season-specific priors.")

    # Load prediction model
    @st.cache_resource
    def load_prediction_model():
        try:
            with open('match_predictor_model.pkl', 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None

    # Helper functions for decay priors
    def get_decay_weight(matches_played, decay_rate=0.15):
        """Exponential decay weight for prior season stats"""
        return np.exp(-decay_rate * matches_played)

    def get_blended_stat(current_value, current_matches, prior_per_game, decay_rate=0.15, default_prior=None):
        """Blend current season stat with decaying prior"""
        if current_matches == 0:
            return prior_per_game if prior_per_game is not None else (default_prior if default_prior else 0.0)
        current_per_game = current_value / current_matches
        if prior_per_game is None:
            return current_per_game
        prior_weight = get_decay_weight(current_matches, decay_rate)
        current_weight = 1 - prior_weight
        return current_weight * current_per_game + prior_weight * prior_per_game

    def calculate_prediction_features(team_stats, prior_stats, league_avg, is_home):
        """Calculate features for a team with decaying priors"""
        curr = team_stats
        m = curr['matches']

        # Get prior per-game stats
        if prior_stats and prior_stats.get('matches', 0) > 0:
            pm = prior_stats['matches']
            prior_ppg = prior_stats['points'] / pm
            prior_gpg = prior_stats['goals_for'] / pm
            prior_gapg = prior_stats['goals_against'] / pm
            prior_xgpg = prior_stats['xG_for'] / pm if prior_stats['xG_for'] > 0 else 1.0
            prior_xgapg = prior_stats['xG_against'] / pm if prior_stats['xG_against'] > 0 else 1.0
            prior_winrate = prior_stats['wins'] / pm
            prior_csrate = prior_stats['clean_sheets'] / pm
            prior_shot_conv = prior_stats['goals_for'] / max(prior_stats['shots_for'], 1)
            prior_sot_rate = prior_stats['sot_for'] / max(prior_stats['shots_for'], 1)
            if is_home and prior_stats['home_matches'] > 0:
                prior_venue_wr = prior_stats['home_wins'] / prior_stats['home_matches']
                prior_venue_gpg = prior_stats['home_goals'] / prior_stats['home_matches']
            elif not is_home and prior_stats['away_matches'] > 0:
                prior_venue_wr = prior_stats['away_wins'] / prior_stats['away_matches']
                prior_venue_gpg = prior_stats['away_goals'] / prior_stats['away_matches']
            else:
                prior_venue_wr = prior_winrate
                prior_venue_gpg = prior_gpg
        else:
            # Promoted team: use league average (slightly below)
            prior_ppg = league_avg.get('ppg', 1.0) * 0.85
            prior_gpg = league_avg.get('gpg', 1.0) * 0.85
            prior_gapg = league_avg.get('gapg', 1.0) * 1.15
            prior_xgpg = league_avg.get('xgpg', 1.0) * 0.85
            prior_xgapg = league_avg.get('xgapg', 1.0) * 1.15
            prior_winrate = 0.28
            prior_csrate = league_avg.get('csrate', 0.25) * 0.85
            prior_shot_conv = league_avg.get('shot_conv', 0.1) * 0.9
            prior_sot_rate = league_avg.get('sot_rate', 0.35) * 0.95
            prior_venue_wr = 0.28
            prior_venue_gpg = prior_gpg

        decay_rate = 0.15
        # Blend current and prior stats
        ppg = get_blended_stat(curr['points'], m, prior_ppg, decay_rate)
        gpg = get_blended_stat(curr['goals_for'], m, prior_gpg, decay_rate)
        gapg = get_blended_stat(curr['goals_against'], m, prior_gapg, decay_rate)
        xgpg = get_blended_stat(curr['xG_for'], m, prior_xgpg, decay_rate)
        xgapg = get_blended_stat(curr['xG_against'], m, prior_xgapg, decay_rate)
        win_rate = get_blended_stat(curr['wins'], m, prior_winrate, decay_rate)
        cs_rate = get_blended_stat(curr['clean_sheets'], m, prior_csrate, decay_rate)

        curr_shot_conv = curr['goals_for'] / max(curr['shots_for'], 1) if m > 0 else 0
        curr_sot_rate = curr['sot_for'] / max(curr['shots_for'], 1) if m > 0 else 0
        shot_conv = curr_shot_conv if m > 3 else prior_shot_conv
        sot_rate = curr_sot_rate if m > 3 else prior_sot_rate

        venue_key = 'home' if is_home else 'away'
        venue_wr = get_blended_stat(
            curr[f'{venue_key}_wins'], curr[f'{venue_key}_matches'],
            prior_venue_wr, decay_rate
        )
        venue_gpg = get_blended_stat(
            curr[f'{venue_key}_goals'], curr[f'{venue_key}_matches'],
            prior_venue_gpg, decay_rate
        )

        gd = gpg - gapg
        xg_diff = xgpg - xgapg
        form = np.mean(curr['last_5_results'][-5:]) if curr['last_5_results'] else 1.0
        xg_form = np.mean(curr['last_5_xG'][-5:]) if curr['last_5_xG'] else prior_xgpg

        return {
            'ppg': ppg, 'gpg': gpg, 'gapg': gapg, 'xgpg': xgpg, 'xgapg': xgapg,
            'win_rate': win_rate, 'cs_rate': cs_rate, 'shot_conv': shot_conv,
            'sot_rate': sot_rate, 'gd': gd, 'xg_diff': xg_diff, 'form': form,
            'xg_form': xg_form, 'venue_wr': venue_wr, 'venue_gpg': venue_gpg
        }

    model_data = load_prediction_model()

    if model_data is None:
        st.error("Prediction model not found. Please ensure 'match_predictor_model.pkl' exists.")
    else:
        model = model_data['model']
        scaler = model_data['scaler']
        team_stats = model_data['team_stats']
        prior_season_stats = model_data.get('prior_season_stats', {})
        league_avg_stats = model_data.get('league_avg_stats', {'ppg': 1.0, 'gpg': 1.19, 'gapg': 1.19, 'xgpg': 1.0, 'xgapg': 1.0, 'csrate': 0.25, 'shot_conv': 0.1, 'sot_rate': 0.35})
        team_ratings = model_data.get('team_ratings', {})

        # Display Team Strength Ratings (Multi-Season with SOS)
        with st.expander("Team Strength Ratings", expanded=False):
            # Build season options filtered to selected league(s)
            league_season_map = {}
            for cid in selected_comp_ids:
                if cid in COMPETITIONS:
                    league_season_map.update(COMPETITIONS[cid]["seasons"])
            league_season_labels = list(league_season_map.values())
            # Default to current season for selected league
            default_label = league_season_map.get(
                COMPETITIONS[selected_comp_ids[0]].get("current_season") if selected_comp_ids else CURRENT_SEASON_ID,
                league_season_labels[0] if league_season_labels else None
            )
            rating_seasons = st.multiselect(
                "Seasons", league_season_labels,
                default=[default_label] if default_label else [],
                key="rating_seasons"
            )
            # Reverse-lookup season IDs from display names (league-filtered)
            season_name_to_id = {v: k for k, v in league_season_map.items()}
            rating_rows = []
            for season_name in rating_seasons:
                sid = season_name_to_id[season_name]
                s_events = get_filtered_events(raw_events_df, sid, selected_comp_ids)
                s_matches = filter_by_league(get_season_matches(matches_summary_df, sid), selected_comp_ids)
                ts_df = calculate_team_strength(s_events, s_matches, season_id=sid)
                if ts_df.empty:
                    continue
                rolling_df = calculate_rolling_team_strength(s_events, s_matches, season_id=sid)
                sos_df = calculate_sos_adjusted_strength(rolling_df, ts_df, season_id=sid)
                for team in ts_df.index:
                    raw_att = ts_df.loc[team, 'Attacking Strength']
                    raw_def = ts_df.loc[team, 'Defending Strength']
                    sos_att = sos_df.loc[team, 'sos_att'] if team in sos_df.index else raw_att
                    sos_def = sos_df.loc[team, 'sos_def'] if team in sos_df.index else raw_def
                    sos_factor = sos_df.loc[team, 'sos_factor'] if team in sos_df.index else np.nan
                    # Count matches from rolling data
                    team_rolling = rolling_df[rolling_df['team'] == team] if not rolling_df.empty else pd.DataFrame()
                    n_matches = int(team_rolling['match_number'].max()) + 1 if not team_rolling.empty else 0
                    rating_rows.append({
                        'Rank': 0,
                        'Team': team,
                        'Season': season_name,
                        'Att Strength': round(raw_att, 3),
                        'Def Strength': round(raw_def, 3),
                        'SOS Att': round(float(sos_att), 3),
                        'SOS Def': round(float(sos_def), 3),
                        'SOS Factor': round(float(sos_factor), 3) if not np.isnan(float(sos_factor)) else None,
                        'Matches': n_matches,
                    })
            if rating_rows:
                ratings_combined = pd.DataFrame(rating_rows)
                # Overall = att - def, rescaled to 0-100
                raw_overall = ratings_combined['SOS Att'] - ratings_combined['SOS Def']
                ov_min = raw_overall.min()
                ov_max = raw_overall.max()
                if ov_max > ov_min:
                    ratings_combined['Overall'] = round(((raw_overall - ov_min) / (ov_max - ov_min)) * 100, 1)
                else:
                    ratings_combined['Overall'] = 50.0
                ratings_combined = ratings_combined.sort_values('Overall', ascending=False).reset_index(drop=True)
                ratings_combined['Rank'] = range(1, len(ratings_combined) + 1)
                st.dataframe(ratings_combined, use_container_width=True, hide_index=True, column_config=auto_column_config(ratings_combined))
            else:
                st.info("No team strength data available for selected seasons.")

        # Season Simulation - Promotion/Relegation Probabilities
        @st.cache_data
        def load_simulation_data():
            try:
                with open('season_simulation.pkl', 'rb') as f:
                    return pickle.load(f)
            except FileNotFoundError:
                return None

        sim_data = load_simulation_data()
        if sim_data is None:
            st.info("Season simulation not yet available. Run simulate_season.py to generate probabilities.")
        else:
            st.subheader("Promotion & Relegation Probabilities")
            sim_ts = sim_data.get('timestamp', '')
            n_sims = sim_data.get('n_simulations', 0)
            st.caption(f"Based on {n_sims:,} Monte Carlo simulations | Updated: {sim_ts[:16].replace('T', ' ')}")

            def render_probability_table(group_name, prob_df, matches_remaining, bonus_points=None, expanded=False, current_standings=None, playoff_pct=None, promotion_pct=None, releg_pct=None, serie_col_labels=None):
                """Render a color-coded probability table for a second-stage group."""
                n_teams = len(prob_df)
                pos_cols = [str(i+1) for i in range(n_teams)]
                is_serie = group_name.startswith('Série')
                is_playoff_group = group_name.startswith('Promotion Playoff')

                # Build lookup for points and matches played from current standings
                standings_lookup = {}
                if current_standings is not None:
                    for _, row in current_standings.iterrows():
                        standings_lookup[row['Team']] = {'P': row['P'], 'Pts': row['Pts']}

                with st.expander(f"{group_name} ({matches_remaining} matches remaining)", expanded=expanded):
                    # Build HTML table
                    html = '<table style="width:100%;border-collapse:collapse;font-size:0.85em;text-align:center;">'

                    # Header row
                    html += '<tr style="border-bottom:2px solid #444;">'
                    html += '<th style="text-align:left;padding:6px 10px;">Team</th>'
                    html += '<th style="padding:6px 8px;">P</th>'
                    html += '<th style="padding:6px 8px;">Pts</th>'
                    for p in pos_cols:
                        html += f'<th style="padding:6px 8px;">{p}</th>'

                    # Summary column header
                    if group_name == 'Promotion':
                        html += '<th style="padding:6px 8px;border-left:2px solid #444;">Promo %</th>'
                        html += '<th style="padding:6px 8px;">Playoff %</th>'
                    elif is_playoff_group:
                        html += '<th style="padding:6px 8px;border-left:2px solid #444;">Promo %</th>'
                    elif is_serie:
                        _slabels = serie_col_labels or ('Playoff %', 'Promo %', 'Releg %')
                        html += f'<th style="padding:6px 8px;border-left:2px solid #444;">{_slabels[0]}</th>'
                        html += f'<th style="padding:6px 8px;">{_slabels[1]}</th>'
                        html += f'<th style="padding:6px 8px;border-left:2px solid #444;">{_slabels[2]}</th>'
                    else:
                        html += '<th style="padding:6px 8px;border-left:2px solid #444;">Releg %</th>'
                    html += '</tr>'

                    # Data rows
                    for team in prob_df.index:
                        html += '<tr style="border-bottom:1px solid #ddd;">'
                        html += f'<td style="text-align:left;padding:6px 10px;font-weight:bold;white-space:nowrap;">{team}</td>'
                        team_info = standings_lookup.get(team, {'P': 0, 'Pts': 0})
                        html += f'<td style="padding:6px 8px;color:#888;">{team_info["P"]}</td>'
                        total_pts = team_info["Pts"] + (bonus_points.get(team, 0) if bonus_points else 0)
                        html += f'<td style="padding:6px 8px;font-weight:bold;">{total_pts}</td>'

                        for p in pos_cols:
                            val = prob_df.loc[team, p]
                            pos_num = int(p)

                            # Determine cell color
                            bg = ''
                            if group_name == 'Promotion':
                                if pos_num <= 2:
                                    intensity = min(val * 1.2, 1.0)
                                    bg = f'background-color:rgba(46,204,113,{intensity:.2f});'
                                elif pos_num == 3:
                                    intensity = min(val * 1.2, 1.0)
                                    bg = f'background-color:rgba(241,196,15,{intensity:.2f});'
                            elif is_playoff_group:
                                if pos_num <= 2:
                                    intensity = min(val * 1.2, 1.0)
                                    bg = f'background-color:rgba(46,204,113,{intensity:.2f});'
                            elif is_serie:
                                if pos_num <= 2:
                                    # Green for playoff qualification positions
                                    intensity = min(val * 1.2, 1.0)
                                    bg = f'background-color:rgba(46,204,113,{intensity:.2f});'
                                elif pos_num >= n_teams - 4:
                                    # Red for relegation positions (bottom 5)
                                    intensity = min(val * 1.2, 1.0)
                                    bg = f'background-color:rgba(231,76,60,{intensity:.2f});'
                            else:
                                if pos_num >= n_teams - 1:
                                    intensity = min(val * 1.2, 1.0)
                                    bg = f'background-color:rgba(231,76,60,{intensity:.2f});'

                            cell_text = f'{val:.1%}' if val >= 0.005 else ''
                            html += f'<td style="padding:6px 8px;{bg}">{cell_text}</td>'

                        # Summary columns
                        if group_name == 'Promotion':
                            promo_pct = prob_df.loc[team, '1'] + prob_df.loc[team, '2']
                            po_pct = prob_df.loc[team, '3']
                            promo_bg = f'background-color:rgba(46,204,113,{min(promo_pct * 1.2, 1.0):.2f});'
                            po_bg = f'background-color:rgba(241,196,15,{min(po_pct * 1.2, 1.0):.2f});'
                            html += f'<td style="padding:6px 8px;border-left:2px solid #444;font-weight:bold;{promo_bg}">{promo_pct:.1%}</td>'
                            html += f'<td style="padding:6px 8px;font-weight:bold;{po_bg}">{po_pct:.1%}</td>'
                        elif is_playoff_group:
                            team_promo = promotion_pct.get(team, 0) if promotion_pct else 0
                            promo_bg = f'background-color:rgba(46,204,113,{min(team_promo * 1.2, 1.0):.2f});'
                            html += f'<td style="padding:6px 8px;border-left:2px solid #444;font-weight:bold;{promo_bg}">{team_promo:.1%}</td>'
                        elif is_serie:
                            # Playoff % = chance of finishing top 2 in série
                            team_playoff = playoff_pct.get(team, 0) if playoff_pct else 0
                            # Promotion % = chance of top 2 in série AND top 2 in playoff group
                            team_promo = promotion_pct.get(team, 0) if promotion_pct else 0
                            # Relegation %: chained simulation value when the
                            # sim provides one (Liga 3 first phase), else the
                            # positional bottom-5 heuristic (Campeonato séries)
                            if releg_pct:
                                team_releg = releg_pct.get(team, 0)
                            else:
                                releg_positions = [str(i) for i in range(n_teams - 4, n_teams + 1)]
                                team_releg = sum(prob_df.loc[team, p] for p in releg_positions if p in prob_df.columns)

                            playoff_bg = f'background-color:rgba(46,204,113,{min(team_playoff * 1.2, 1.0):.2f});'
                            promo_bg = f'background-color:rgba(46,204,113,{min(team_promo * 2.0, 1.0):.2f});'
                            releg_bg = f'background-color:rgba(231,76,60,{min(team_releg * 1.2, 1.0):.2f});'
                            html += f'<td style="padding:6px 8px;border-left:2px solid #444;font-weight:bold;{playoff_bg}">{team_playoff:.1%}</td>'
                            html += f'<td style="padding:6px 8px;font-weight:bold;{promo_bg}">{team_promo:.1%}</td>'
                            html += f'<td style="padding:6px 8px;border-left:2px solid #444;font-weight:bold;{releg_bg}">{team_releg:.1%}</td>'
                        else:
                            releg_pct = prob_df.loc[team, str(n_teams - 1)] + prob_df.loc[team, str(n_teams)]
                            releg_bg = f'background-color:rgba(231,76,60,{min(releg_pct * 1.2, 1.0):.2f});'
                            html += f'<td style="padding:6px 8px;border-left:2px solid #444;font-weight:bold;{releg_bg}">{releg_pct:.1%}</td>'

                        html += '</tr>'

                    html += '</table>'
                    st.markdown(html, unsafe_allow_html=True)

            # Display simulation results for selected competition(s)
            competitions_data = sim_data.get('competitions', {})
            if not competitions_data:
                # Backward compat: old format has 'groups' at top level (Liga 3 only)
                competitions_data = {43324: {'competition_name': 'Liga 3', 'groups': sim_data.get('groups', {})}}

            for comp_id in selected_comp_ids:
                comp_sim = competitions_data.get(comp_id, {})
                sim_groups = comp_sim.get('groups', {})
                if not sim_groups:
                    continue

                comp_name = comp_sim.get('competition_name', COMPETITIONS.get(comp_id, {}).get('name', ''))
                if len(selected_comp_ids) > 1:
                    st.markdown(f"#### {comp_name}")

                for group_name, g in sim_groups.items():
                    expanded = (
                        group_name == 'Promotion'
                        or group_name.startswith('Promotion Playoff')
                        or (comp_id == 43324 and group_name.startswith('Série'))
                        or (comp_id == 702 and group_name == list(sim_groups.keys())[0])
                    )
                    render_probability_table(
                        group_name, g['position_probabilities'], g['matches_remaining'],
                        bonus_points=g.get('bonus_points'), expanded=expanded,
                        current_standings=g.get('current_standings'),
                        playoff_pct=g.get('playoff_pct'),
                        promotion_pct=g.get('promotion_pct'),
                        releg_pct=g.get('releg_pct'),
                        serie_col_labels=g.get('serie_col_labels'),
                    )

        # Team selection — cross-season team-season combos
        all_season_options = {}  # {"Team (Season)": (team_name, season_id)}
        # Filter seasons by selected league(s)
        available_sids = set()
        for cid in selected_comp_ids:
            if cid in COMPETITIONS:
                available_sids.update(COMPETITIONS[cid]["seasons"].keys())
        pred_events = filter_by_league(raw_events_df, selected_comp_ids)
        pred_matches = filter_by_league(matches_summary_df, selected_comp_ids)
        for sid in sorted(available_sids, reverse=True):
            season_cum = build_season_cumulative_stats(pred_events, pred_matches, sid)
            for team_name, stats in season_cum.items():
                if stats['matches'] >= 3:
                    label = f"{team_name} ({SEASON_ID_MAP[sid]})"
                    all_season_options[label] = (team_name, sid)

        sorted_options = sorted(all_season_options.keys())
        # Defaults: our club in its newest available season, against the
        # next (else most recent) opponent from the fixture list — not the
        # alphabetically-first team in two different seasons.
        _home_default, _away_default = _predictor_default_labels(
            all_season_options, sorted(available_sids, reverse=True), pred_matches)
        col1, col2 = st.columns(2)
        with col1:
            home_label = st.selectbox(
                "Home Team", sorted_options, key="pred_home",
                index=sorted_options.index(_home_default) if _home_default in sorted_options else 0)
        with col2:
            away_options = [t for t in sorted_options if t != home_label]
            away_label = st.selectbox(
                "Away Team", away_options, key="pred_away",
                index=away_options.index(_away_default) if _away_default in away_options else 0)

        # --- Scoreline model (Dixon-Coles): always shown for the chosen fixture ---
        try:
            import scoreline_ui
            _dc_home, _ = all_season_options[home_label]
            _dc_away, _ = all_season_options[away_label]
            scoreline_ui.render_scoreline_section(_dc_home, _dc_away, selected_comp_ids[0], SEASON_ID_MAP)
        except Exception as _dc_exc:
            st.caption(f"Scoreline model unavailable: {type(_dc_exc).__name__}: {_dc_exc}")
        st.divider()

        if st.button("Predict Match Outcome", type="primary"):
            home_team_name, home_sid = all_season_options[home_label]
            away_team_name, away_sid = all_season_options[away_label]
            home_cum = build_season_cumulative_stats(raw_events_df, matches_summary_df, home_sid)[home_team_name]
            away_cum = build_season_cumulative_stats(raw_events_df, matches_summary_df, away_sid)[away_team_name]

            # Tier-aware priors + calibrated Bayes blend from
            # simulate_season — the SAME feature path the current model
            # was trained on (the page's legacy inline blend produced
            # train/serve drift for the simple-strength model)
            import simulate_season as _ssim

            @st.cache_resource(show_spinner=False)
            def _predictor_tier_priors(comp_id):
                return _ssim.build_prior_strengths(
                    matches_summary_df, league_avg_stats, comp_id)

            _pp_comp = int(selected_comp_ids[0]) if selected_comp_ids else 43324
            _priors = _predictor_tier_priors(_pp_comp)
            home_feats = _ssim.calculate_prediction_features(
                home_cum, _priors.get(home_team_name), league_avg_stats, is_home=True)
            away_feats = _ssim.calculate_prediction_features(
                away_cum, _priors.get(away_team_name), league_avg_stats, is_home=False)

            if model_data.get('feature_mode') == 'simple_strength_v1':
                _mix = model_data.get('strength_mix', 0.7)
                def _S(f):
                    return (_mix * (f['xgpg'] - f['xgapg'])
                            + (1 - _mix) * (f['gpg'] - f['gapg']))
                feature_vector = [_S(home_feats) - _S(away_feats)]
            else:
                feature_vector = [
                home_feats['ppg'], away_feats['ppg'], home_feats['ppg'] - away_feats['ppg'],
                home_feats['form'], away_feats['form'], home_feats['form'] - away_feats['form'],
                home_feats['gpg'], away_feats['gpg'], home_feats['gpg'] - away_feats['gpg'],
                home_feats['gd'], away_feats['gd'], home_feats['gd'] - away_feats['gd'],
                home_feats['xgpg'], away_feats['xgpg'], home_feats['xgpg'] - away_feats['xgpg'],
                home_feats['xgapg'], away_feats['xgapg'],
                home_feats['xg_diff'], away_feats['xg_diff'], home_feats['xg_diff'] - away_feats['xg_diff'],
                home_feats['xg_form'], away_feats['xg_form'],
                home_feats['win_rate'], away_feats['win_rate'], home_feats['win_rate'] - away_feats['win_rate'],
                home_feats['venue_wr'], away_feats['venue_wr'],
                home_feats['shot_conv'], away_feats['shot_conv'],
                home_feats['sot_rate'], away_feats['sot_rate'],
                home_feats['cs_rate'], away_feats['cs_rate'],
                home_feats['venue_gpg'], away_feats['venue_gpg'],
            ]

            X = scaler.transform([feature_vector])
            proba = model.predict_proba(X)[0]
            pred = model.predict(X)[0]

            # Display results
            st.subheader(f"{home_label} vs {away_label}")

            # Show team strength ratings
            if team_ratings:
                home_rating = team_ratings.get(home_team_name, {})
                away_rating = team_ratings.get(away_team_name, {})
                if home_rating and away_rating:
                    rcol1, rcol2 = st.columns(2)
                    with rcol1:
                        st.caption(f"**{home_label}** Rating: {home_rating['overall']:.1f}")
                    with rcol2:
                        st.caption(f"**{away_label}** Rating: {away_rating['overall']:.1f}")

            # Probability bars
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Home Win", f"{proba[1]:.1%}")
                st.progress(proba[1])
            with col2:
                st.metric("Draw", f"{proba[0]:.1%}")
                st.progress(proba[0])
            with col3:
                st.metric("Away Win", f"{proba[2]:.1%}")
                st.progress(proba[2])

            # Prediction
            if pred == 1:
                st.success(f"**Predicted Outcome: {home_label} Win**")
            elif pred == 2:
                st.success(f"**Predicted Outcome: {away_label} Win**")
            else:
                st.info(f"**Predicted Outcome: Draw**")

            # Team stats comparison (using blended stats)
            st.subheader("Team Comparison (Season Stats with Priors)")
            comparison_data = {
                'Metric': ['Points/Game', 'Goals/Game', 'xG/Game', 'xG Against/Game', 'Win Rate', 'Form (Last 5)', 'Clean Sheet Rate'],
                home_label: [f"{home_feats['ppg']:.2f}", f"{home_feats['gpg']:.2f}", f"{home_feats['xgpg']:.2f}", f"{home_feats['xgapg']:.2f}", f"{home_feats['win_rate']:.1%}", f"{home_feats['form']:.2f}", f"{home_feats['cs_rate']:.1%}"],
                away_label: [f"{away_feats['ppg']:.2f}", f"{away_feats['gpg']:.2f}", f"{away_feats['xgpg']:.2f}", f"{away_feats['xgapg']:.2f}", f"{away_feats['win_rate']:.1%}", f"{away_feats['form']:.2f}", f"{away_feats['cs_rate']:.1%}"]
            }
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True, column_config=auto_column_config(comparison_df))

            # Show matches played
            home_matches = home_cum['matches']
            away_matches = away_cum['matches']
            st.caption(f"Based on {home_matches} matches for {home_label} and {away_matches} matches for {away_label}")
