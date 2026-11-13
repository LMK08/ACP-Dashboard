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
    st.markdown("Every forecast on this page comes from one model, the Dixon-Coles "
                "scoreline model: the team strength ratings, the promotion and "
                "relegation odds, and the scoreline forecast for a chosen fixture.")

    # Team Strength Ratings — the Dixon-Coles attack / defence parameters
    # (2026-09). They replaced the per-match xG rates × schedule ratio as
    # the rating: that table treated four games as certainty and its
    # schedule step was a ratio of noisy early-season numbers. The
    # per-season rates stay available behind a toggle for context.
    with st.expander("Team Strength Ratings", expanded=False):
        import scoreline_ui
        _lg = selected_comp_ids[0] if selected_comp_ids else None
        _cur_sid = (COMPETITIONS[_lg].get("current_season", CURRENT_SEASON_ID)
                    if _lg in COMPETITIONS else CURRENT_SEASON_ID)
        _cur_matches = filter_by_league(get_season_matches(matches_summary_df, _cur_sid), selected_comp_ids)
        _cur_teams = sorted(set(_cur_matches['homeTeamName'].dropna()) | set(_cur_matches['awayTeamName'].dropna()))
        _cur_played = (_cur_matches[_cur_matches['status'] == 'Played']
                       if 'status' in _cur_matches.columns else _cur_matches)
        _played_counts = pd.concat([_cur_played['homeTeamName'], _cur_played['awayTeamName']]).value_counts()
        try:
            scoreline_ui.render_strength_table(
                _lg, _cur_teams, played=_played_counts,
                league_name=COMPETITIONS[_lg]["name"] if _lg in COMPETITIONS else None,
                season_label=SEASON_ID_MAP.get(_cur_sid, str(_cur_sid)),
                key='mp_dc_strength')
        except Exception as _dc_err:  # a malformed params file must not take the page down
            st.warning(f"Team strength table unavailable: {_dc_err}")

        if st.toggle("Show per-match xG rates and schedule adjustment by season",
                     value=False, key="rating_show_rates"):
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
                # Net schedule-adjusted rate (goals + xG per match); no min-max
                # rescale — the model-based table above is the rating.
                ratings_combined['SOS Net'] = (ratings_combined['SOS Att'] - ratings_combined['SOS Def']).round(3)
                ratings_combined = ratings_combined.sort_values('SOS Net', ascending=False).reset_index(drop=True)
                ratings_combined['Rank'] = range(1, len(ratings_combined) + 1)
                st.dataframe(ratings_combined, width="stretch", hide_index=True, column_config=auto_column_config(ratings_combined))
                st.caption(
                    "Att / Def Strength = 30% goals + 70% xG per match (Def: lower is better). "
                    "SOS columns credit a tough schedule: attack × (league conceded ÷ opponents' conceded), "
                    "defence × (league scored ÷ opponents' scored), using each opponent's pre-match strength; "
                    "SOS Factor > 1 = tougher-than-average schedule. Switches on after 3 matches against "
                    "opponents with prior data, so it is noisy early in a season. "
                    "SOS Net = SOS Att − SOS Def. Descriptive per-season rates: a team with four "
                    "matches is treated like one with thirty — rank by the model table above."
                )
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
        _mm = sim_data.get('model') or {}
        if str(_mm.get('name', '')).startswith('dixon_coles'):
            _src = (f"every remaining fixture drawn as a scoreline from the Dixon-Coles model "
                    f"({_mm.get('n_matches', 0):,} matches through {_mm.get('asof', '?')}"
                    f"{', refitted at this data refresh' if _mm.get('refit') else ''}) — the same model "
                    f"as the scoreline forecast and Team Strength Ratings on this page; xPts = expected "
                    f"final points")
        else:
            _src = "match outcomes from the simple predictor (older simulation format)"
        st.caption(f"{n_sims:,} Monte Carlo simulations of the rest of the season, {_src} | "
                   f"Updated: {sim_ts[:16].replace('T', ' ')}")

        def render_probability_table(group_name, prob_df, matches_remaining, bonus_points=None, expanded=False, current_standings=None, playoff_pct=None, promotion_pct=None, releg_pct=None, serie_col_labels=None, expected_pts=None):
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
                if expected_pts:
                    html += '<th style="padding:6px 8px;" title="Expected final points over the simulations">xPts</th>'
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
                    if expected_pts:
                        _xp = expected_pts.get(team)
                        html += (f'<td style="padding:6px 8px;color:#888;">{_xp:.1f}</td>'
                                 if _xp is not None else '<td></td>')

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
                    expected_pts=g.get('expected_pts'),
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
