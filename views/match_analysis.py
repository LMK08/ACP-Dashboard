"""Match Analysis view — extracted verbatim from app.py's `elif analysis_type == 'Match Analysis'` branch (2026-09).

Collaborators are read from the running app module at call time (the
pattern opposition_report.py uses), so importing this module never imports
app.py. The binding block at the top of render() IS the page's dependency
list: everything it reads from app.py, nothing else.
"""
import re

import pandas as pd
import streamlit as st
import sys


def _app():
    return sys.modules['__main__']


MATCH_SECTIONS = ("Summary", "Team stats", "Shots", "Players", "Tactics")


# The scoreboard's layout. Classes, not inline styles, because the phone
# layout needs a media query: under 640 px each side stacks its crest above
# its name (side by side, a 375 px screen left the names about 40 px and
# they broke letter by letter).
_SCOREBOARD_CSS = """<style>
.acp-sb { background:#fff; border:1px solid #EBE7E1; border-radius:8px; padding:0.9rem 1.25rem 1rem;
          margin:0.3rem 0 0.9rem; box-shadow:0 1px 3px rgba(0,0,0,0.04); }
.acp-sb-detail, .acp-sb-xg-label { font-size:0.72rem; letter-spacing:0.12em; text-transform:uppercase;
          color:#5c5650; font-weight:600; text-align:center; }
.acp-sb-detail { margin-bottom:0.7rem; }
.acp-sb-row { display:grid; grid-template-columns:minmax(0,1fr) auto minmax(0,1fr); align-items:center;
          column-gap:1.4rem; }
.acp-sb-row + .acp-sb-row { margin-top:0.55rem; }
.acp-sb-side { display:flex; align-items:center; gap:0.8rem; min-width:0; }
.acp-sb-side.home { justify-content:flex-end; text-align:right; }
.acp-sb-side.away { justify-content:flex-start; text-align:left; }
.acp-sb-side img { width:52px; height:52px; object-fit:contain; flex:none; }
.acp-sb-name { font-size:1.3rem; font-weight:700; color:#1a1a1a; line-height:1.2; min-width:0;
          overflow-wrap:break-word; }
.acp-sb-score { font-size:2.5rem; font-weight:700; line-height:1; color:#1a1a1a; white-space:nowrap;
          font-variant-numeric:tabular-nums; }
.acp-sb-xg { font-size:1rem; font-weight:600; color:#1a1a1a; font-variant-numeric:tabular-nums; }
.acp-sb-xg.home { text-align:right; } .acp-sb-xg.away { text-align:left; }
@media (max-width: 640px) {
  .acp-sb { padding:0.8rem 0.7rem 0.9rem; }
  .acp-sb-row { column-gap:0.6rem; align-items:start; }
  .acp-sb-side.home { flex-direction:column-reverse; }
  .acp-sb-side.away { flex-direction:column; }
  .acp-sb-side.home, .acp-sb-side.away { justify-content:flex-start; text-align:center; gap:0.35rem; }
  .acp-sb-side img { width:44px; height:44px; }
  .acp-sb-name { font-size:0.95rem; }
  .acp-sb-score { font-size:2rem; margin-top:0.55rem; }
  .acp-sb-xg.home, .acp-sb-xg.away { text-align:center; }
}
</style>"""


def _scoreboard_html(app, info, events, stage):
    """The match header: crests, names, score, xG per side, and one line of
    detail. xG is the sum of `shot.xg` in the match events — the figure the
    shot maps' subtitles and Home's "Last match" card show. A fixture with
    no score reads "vs"; a side with no crest file shows its name alone."""
    import html
    home, away = str(info.get('homeTeamName', '?')), str(info.get('awayTeamName', '?'))
    _m = re.match(r'\s*(\d+)\s*-\s*(\d+)', str(info.get('score')))
    score = f"{_m.group(1)}&nbsp;–&nbsp;{_m.group(2)}" if _m else "vs"
    xg = {}
    if events is not None and not events.empty and {'team.name', 'shot.xg'}.issubset(events.columns):
        xg = events.groupby('team.name', observed=True)['shot.xg'].sum().to_dict()
    date = pd.to_datetime(info.get('dateutc'), errors='coerce')
    detail = " · ".join(x for x in (
        "Match Report",
        f"GW {info.get('gameweek')}" if pd.notna(info.get('gameweek')) else None,
        date.strftime('%d %b %Y') if pd.notna(date) else None,
        stage) if x)

    def crest(team):
        uri = app.crest_data_uri(team)
        return f'<img src="{uri}" alt="">' if uri else ''

    def name(team):
        return f'<div class="acp-sb-name">{html.escape(team)}</div>'

    def xg_cell(team, side):
        value = f"{xg[team]:.2f}" if team in xg else "–"
        return f'<div class="acp-sb-xg {side}">{value}</div>'

    return (
        _SCOREBOARD_CSS
        + '<div class="acp-sb">'
        f'<div class="acp-sb-detail">{html.escape(detail)}</div>'
        '<div class="acp-sb-row">'
        f'<div class="acp-sb-side home">{name(home)}{crest(home)}</div>'
        f'<div class="acp-sb-score">{score}</div>'
        f'<div class="acp-sb-side away">{crest(away)}{name(away)}</div>'
        '</div>'
        + (f'<div class="acp-sb-row">{xg_cell(home, "home")}'
           f'<div class="acp-sb-xg-label">xG</div>{xg_cell(away, "away")}</div>' if xg else '')
        + '</div>')


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
    load_match_player_minutes = app.load_match_player_minutes
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
    # ...and, within those, OUR club's newest match (it used to open on
    # whichever match happened to be newest, e.g. Fafe vs Trofense).
    _our = getattr(app, 'OUR_TEAM', 'Atlético CP')
    _is_ours = ((season_matches_df.get('homeTeamName') == _our)
                | (season_matches_df.get('awayTeamName') == _our)
                if {'homeTeamName', 'awayTeamName'}.issubset(season_matches_df.columns)
                else pd.Series(False, index=season_matches_df.index))
    _with_ev = [mid in _match_ids_with_events for mid in season_matches_df['matchId'].tolist()]
    _default_match_idx = next(
        (i for i, (ev, ours) in enumerate(zip(_with_ev, _is_ours.tolist())) if ev and ours),
        next((i for i, ev in enumerate(_with_ev) if ev), 0))
    # The match picker, first thing under the page head (it sat at the
    # foot of the sidebar, below the screen edge, until 2026-11). The option
    # text carries the score and the date.
    _match_options = season_matches_df['display_name'].tolist()
    if not _match_options:
        st.warning("No matches for this league and season yet.")
        st.stop()
    # The picker is KEYED and seeded through session state, never index=
    # (widget-seeding rule). Un-keyed with index=, a match opened by a deep
    # link (Home's "Open match report", a click on a shot or a rolling-xG
    # point) was a different widget on the next run, once nav_match_id had
    # been popped — so the first click on a section jumped back to the
    # default match. The value is re-set every run: the option list follows
    # the league and season.
    _nav_mid = st.session_state.pop('nav_match_id', None)
    if _nav_mid is not None:
        _ids = season_matches_df['matchId'].tolist()
        if _nav_mid in _ids:
            st.session_state['match_analysis_match'] = _match_options[_ids.index(_nav_mid)]
    _cur_match = st.session_state.get('match_analysis_match')
    st.session_state['match_analysis_match'] = (
        _cur_match if _cur_match in _match_options else _match_options[_default_match_idx])
    selected_match_display = st.selectbox("Match", _match_options, key='match_analysis_match')
    matching_matches = season_matches_df[season_matches_df['display_name'] == selected_match_display]
    if matching_matches.empty:
        st.error("Selected match not found. Please refresh the page and try again.")
        st.stop()
    selected_match_info = matching_matches.iloc[0]
    selected_match_id = selected_match_info['matchId']

    # --- Stage of the selected match (once a badge at the foot of the sidebar,
    # now part of the scoreboard's detail line) ---
    badge_text = None
    if 'roundId' in season_matches_df.columns and len(season_matches_df) > 0:
        round_counts = season_matches_df.groupby('roundId').size()
        first_stage_round = round_counts.idxmax()
        second_stage_rounds = round_counts.drop(first_stage_round, errors='ignore')

        current_round_id = selected_match_info.get('roundId')
        if current_round_id == first_stage_round:
            badge_text = "Regular Season"
        elif len(second_stage_rounds) > 0:
            badge_text = ("Promotion League" if current_round_id == second_stage_rounds.idxmin()
                          else "Maintenance Stage")
        else:
            badge_text = "Second Stage"

    # --- Get the match events ONCE ---
    match_events_df = raw_events_df[raw_events_df['matchId'] == selected_match_id]

    # --- Scoreboard (2026-11): the score used to appear only in the small
    # print above the first shot map. It carries the "Match Report" line
    # Lucas kept, the gameweek, date and stage.
    st.markdown(_scoreboard_html(app, selected_match_info, match_events_df, badge_text),
                unsafe_allow_html=True)

    match_data = all_match_data.get(selected_match_id)
    if match_data:
        import team_interactive as ti
        import context_bar

        home_team = selected_match_info['homeTeamName']
        away_team = selected_match_info['awayTeamName']
        # Get lineup/substitution data for this match (if available)
        match_lineup_data = match_lineups.get(selected_match_id, {}) if match_lineups else {}
        home_lineup = match_lineup_data.get(home_team)
        away_lineup = match_lineup_data.get(away_team)

        # --- Sections (2026-11): ONE is drawn at a time; the page was 12 screens
        # of everything. The PDF button sits beside the row and builds the
        # same report from any section: it renders its own figures
        # (_match_fig_or_none) and reads the tables prepared below, which are
        # therefore built on every run whatever the section.
        _sec_col, _pdf_col = st.columns([3, 1], vertical_alignment="center")
        with _sec_col:
            _section = st.radio("Match section", MATCH_SECTIONS, horizontal=True,
                                key="match_section", label_visibility="collapsed")

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
                st.image(_png, width="stretch")

        # ---- Merge engine OBV into the match stats tables ----
        # Display-time COPIES (the cached all_match_data frames must not
        # be mutated); used by both the page and the match report PDF.
        _disp_team_stats = {k: v.copy() if isinstance(v, pd.DataFrame) else v
                            for k, v in (match_data.get('team_stats') or {}).items()}
        _ps_src = match_data.get('player_stats') or {}
        _disp_player_stats = {k: v.copy() if isinstance(v, pd.DataFrame) else v
                              for k, v in _ps_src.items()}
        # Minutes = Wyscout's exact minutesOnField for this match. The
        # 'Minutes' stored in all_match_data is an event-timestamp estimate
        # and is never shown (Lucas 2026-10-20); no API value -> '-'.
        if {'player.id', 'player.name'}.issubset(match_events_df.columns):
            _mm = load_match_player_minutes()
            _mm = _mm[_mm['matchId'] == int(selected_match_id)] if not _mm.empty else _mm
            _pid_mins = dict(zip(_mm['playerId'], _mm['minutes'])) if not _mm.empty else {}
            _nm_ids = (match_events_df.dropna(subset=['player.id', 'player.name'])
                       .drop_duplicates('player.name'))
            _name2pid = {n: app._resolve_pid(int(p)) for n, p in
                         zip(_nm_ids['player.name'], _nm_ids['player.id'])}
            for _side, _df in _disp_player_stats.items():
                if isinstance(_df, pd.DataFrame) and 'Minutes' in _df.columns:
                    _df['Minutes'] = [
                        f"{int(_pid_mins[_name2pid[nm]])}'"
                        if _name2pid.get(nm) in _pid_mins else '-'
                        for nm in _df.index]
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

        # built on every run: the match report PDF prints them
        home_shots_table = get_shot_table(match_events_df, home_team)
        away_shots_table = get_shot_table(match_events_df, away_team)

        if _section == "Shots":
            st.subheader("Shot Maps")
            # Interactive, stacked at full width (the same drawing as the team and
            # player maps; the static PNG stays for the match-report PDF).
            for _side, _team in (('home', home_team), ('away', away_team)):
                _ev = st.plotly_chart(ti.plotly_match_shot_map(match_events_df, selected_match_info, _team, height=context_bar.pitch_height()),
                                      use_container_width=True, key=f'ma_shots_{_side}', on_select='rerun',
                                      selection_mode='points', config=app._PLOTLY_CFG, theme=None)
                app.open_profile_from_selection(_ev, selected_season_id)

            st.markdown("---") # Add a separator
            st.subheader("Shot Details")
            col1_table, col2_table = st.columns(2)
            with col1_table:
                st.markdown(f"**{home_team}**")
                st.dataframe(home_shots_table, column_config=auto_column_config(home_shots_table))

            with col2_table:
                st.markdown(f"**{away_team}**")
                st.dataframe(away_shots_table, column_config=auto_column_config(away_shots_table))

        if _section == "Summary":
            # the headline table first (goals, xG, shots, possession ...); the
            # other seven categories are the "Team stats" section
            _general = _disp_team_stats.get('General')
            if isinstance(_general, pd.DataFrame):
                st.subheader("Key numbers")
                st.dataframe(_general, column_config=auto_column_config(_general))

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
            if not match_events_df.empty:
                try:
                    _show_match_png('xg_flowchart')
                except Exception as e:
                    st.warning(f"Could not generate xG flowchart: {e}")
            else:
                st.info("No event data found for flowchart.")

        if _section == "Team stats":
            st.subheader("Team Stats")
            if _disp_team_stats:
                for stat_category, df in _disp_team_stats.items():
                    st.markdown(f"**{stat_category}**")
                    if isinstance(df, pd.DataFrame): st.dataframe(df, column_config=auto_column_config(df))
                    else: st.warning(f"Data for '{stat_category}' is not a DataFrame.")
            else: st.warning("Team stats data not found.")

        if _section == "Players":
            st.subheader("Player Stats")
            if 'home' in _disp_player_stats and 'away' in _disp_player_stats:
                st.markdown(f"**{selected_match_info['homeTeamName']}**")
                if isinstance(_disp_player_stats['home'], pd.DataFrame): st.dataframe(_disp_player_stats['home'])
                else: st.warning("Home player stats data not a DataFrame.")
                st.markdown(f"**{selected_match_info['awayTeamName']}**")
                if isinstance(_disp_player_stats['away'], pd.DataFrame): st.dataframe(_disp_player_stats['away'])
                else: st.warning("Away player stats data not a DataFrame.")
            else: st.warning("Player stats data not found.")

        if _section == "Tactics":
            # =============================================================
            # Tactical Analysis (Wyscout-style pitch visualizations)
            # =============================================================
            st.subheader("Tactical Analysis")
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

        with _pdf_col:
            if st.button("Build match report PDF", key=f"build_{_pdf_key}", width="stretch"):
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
                    width="stretch",
                )

    else:
         st.warning(f"No detailed match data found for Match ID {selected_match_id}.")
