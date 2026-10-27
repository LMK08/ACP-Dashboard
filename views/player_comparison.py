"""Player Comparison view — two players side by side, with the Player
Profile's features: the bio header (projected value + likely-fee range,
observed role / style), the ACP Index card, the template radar (percentile
or raw mean ± 2σ, DefR axes) and the grouped season stats.

Everything numeric is ranked against ONE sample — the selected template's
position group at or above the scope's minutes floor
(`template_pct.rank_in_template`) — so the two columns compare like with
like; the profile ranks a player in their own group. The ACP Index card row
comes from the profile's own rule (`engine_card_row`).

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

import eur_interval_ui
import theme

_PLACEHOLDER_IMG = ("https://t3.ftcdn.net/jpg/05/16/27/58/"
                    "360_F_516275801_f3Fsp17x6HQK0xQgDQEELoGau0sJzEf4.jpg")
# (label, engine column, ENGINE_METRIC_HELP key) — the profile's component bars
_ENGINE_COMPONENTS = [
    ("Offensive Value", 'off_pct', 'Offensive Value %'),
    ("Def Quality Grade", 'qual_pct', 'Def Quality Grade %'),
    ("RAPM", 'rapm_pct', 'Engine RAPM %'),
    ("Def Volume Grade", 'defr_pct', 'Def Volume Grade %'),
    ("Off Duel Grade", 'datt_pct', 'Off Duel Grade %'),
    ("Set piece", 'setpiece_pct', 'Engine Set Piece %'),
]
_GK_TEMPLATES = ['Shot Stopper', 'Cross Claimer', 'Ball-playing GK']
# season-total mode: these stay rates (the profile's list)
_DECIMAL_TOTALS = ['xG', 'xA', 'xT', 'xTOP', 'xTSP', 'npxG', 'xAOP', 'xASP',
                   'psxG_faced', 'goalsPrevented']


def _app():
    return sys.modules['__main__']


def _is_gk(pos):
    return str(pos or '').upper().startswith('GK')


def _season_totals(per90, minutes, engine_metrics):
    """The profile's 'Show Season Totals' conversion of one stats row."""
    total = per90.copy()
    rate_cols = [c for c in total.index if '%' in c or 'per' in c.lower()
                 or 'index' in c or 'Percentage' in c]
    for c in ['goalsConceded'] + list(engine_metrics):
        if c in total.index and c not in rate_cols:
            rate_cols.append(c)
    for c in total.index:
        if c not in rate_cols and pd.api.types.is_numeric_dtype(total[c]):
            v = total[c] * minutes / 90
            total[c] = v if c in _DECIMAL_TOTALS else np.round(v)
    return total


def _fmt_stat(metric, x, thousandths):
    if not isinstance(x, (int, float, np.number)) or pd.isna(x):
        return '—' if pd.isna(x) else str(x)
    if metric in thousandths:
        return f"{x:.3f}"
    if np.round(x) == x:
        return f"{x:.0f}"
    return f"{x:.2f}"


def _txt(v):
    """A bio field as display text, or None when it is missing / NaN."""
    if v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() in ('', 'nan', 'None'):
        return None
    return str(v)


def _percentile_color(p):
    """Red (0) → yellow (50) → green (100) — the profile's Stats colouring."""
    if p is None or isinstance(p, str) or pd.isna(p):
        return ''
    hue = max(0.0, min(100.0, float(p))) / 100.0 * 120.0
    return f'background-color: hsl({hue:.0f}, 65%, 72%); color: black;'


def render():
    app = _app()
    DEFENSIVE_METRICS = app.DEFENSIVE_METRICS
    DEFR_DISPLAY_METRICS = app.DEFR_DISPLAY_METRICS
    DEFR_RADAR_MAP = app.DEFR_RADAR_MAP
    DRIBBLING_METRICS = app.DRIBBLING_METRICS
    ENGINE_DISPLAY_METRICS = app.ENGINE_DISPLAY_METRICS
    ENGINE_ROLE_PCT_METRICS = app.ENGINE_ROLE_PCT_METRICS
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
    _build_player_season_perf_table = app._build_player_season_perf_table
    _calculate_age = app._calculate_age
    _cvi_position_group = app._cvi_position_group
    _render_acp_index_card_png = app._render_acp_index_card_png
    competition_for_season = app.competition_for_season
    build_player_priors_lookup = app.build_player_priors_lookup
    compute_career_cvi = app.compute_career_cvi
    compute_cvi_columns = app.compute_cvi_columns
    cvi_to_projected_eur = app.cvi_to_projected_eur
    engine_card_row = app.engine_card_row
    engine_metric_help_md = app.engine_metric_help_md
    engine_rows_for_scope = app.engine_rows_for_scope
    get_filtered_events = app.get_filtered_events
    get_scoped_style = app.get_scoped_style
    get_scoped_tendencies = app.get_scoped_tendencies
    get_season_ids_for_selection = app.get_season_ids_for_selection
    get_season_player_minutes = app.get_season_player_minutes
    league_selector = app.league_selector
    load_and_score_player_stats = app.load_and_score_player_stats
    load_eur_calibration = app.load_eur_calibration
    load_gpa_values = app.load_gpa_values
    load_player_details = app.load_player_details
    load_player_engine = app.load_player_engine
    logger = app.logger
    merge_defr_values_into_stats = app.merge_defr_values_into_stats
    most_recent_season_for_player = app.most_recent_season_for_player
    player_minutes_data = app.player_minutes_data
    plot_comparison_radar = app.plot_comparison_radar
    raw_events_df = app.raw_events_df
    render_tendencies_panel = app.render_tendencies_panel
    scipy = app.scipy
    season_selector = app.season_selector
    template_pct = app.template_pct
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

    # Player B: the seeded player when the bridge set one, else the group's
    # most-used player who is not Player A (the old 'second of the group'
    # was Player A whenever A was that player — Janico vs Janico, 2026/27)
    _b_options = player_b_list_df['display_name'].tolist()
    _b_default = next((o for o in _b_options if o != selected_player_a_display),
                      _b_options[0] if _b_options else None)
    if _seed_b is not None:
        _mb = player_b_list_df[player_b_list_df['playerId'] == int(_seed_b)]
        if not _mb.empty:
            st.session_state['player_comparison_b'] = _mb['display_name'].iloc[0]
        else:
            st.sidebar.warning("The second player sent here is not in this scope or position "
                               "group — showing the default Player B instead.")
    # (B is moved off A too: a player against himself compares nothing)
    if st.session_state.get('player_comparison_b') in (None, selected_player_a_display) \
            or st.session_state.get('player_comparison_b') not in _b_options:
        st.session_state['player_comparison_b'] = _b_default

    # --- Step D: Select Player B (from filtered list) ---
    selected_player_b_display = st.sidebar.selectbox(
        "Select Player B (Same Position Group):",
        _b_options,
        key='player_comparison_b',
    )

    if not _b_options:
        st.warning(f"No players in the {selected_template} positions for this league/season — "
                   "pick another template in the sidebar.")
        st.stop()

    # FIX: Lookup by ID instead of Name
    selected_player_b_id = player_b_list_df[player_b_list_df['display_name'] == selected_player_b_display]['playerId'].values[0]
    player_b_data = player_stats_with_scores_df[player_stats_with_scores_df['playerId'] == selected_player_b_id]

    # Get the name safely
    selected_player_b_name = player_b_data.iloc[0]['playerName']

    # --- 3. Per-player context: bio, value, role / style, ACP Index row ---
    player_details_df = load_player_details()
    try:
        _eng_df, _eng_meta = load_player_engine()
    except Exception:
        logger.exception("engine load failed")
        _eng_df, _eng_meta = pd.DataFrame(), {}
    _eng_meta = _eng_meta or {}
    _eurcal = load_eur_calibration()
    try:
        _eng_value_rows = engine_rows_for_scope(None)
    except Exception:
        logger.exception("engine value rows failed")
        _eng_value_rows = pd.DataFrame()
    _tv_comp_id = competition_for_season(selected_season_id) if selected_season_id else None

    def _legacy_gk_value(row_df, pid, pos, bio):
        """The profile's keeper headline: career CVI (anchored at the most
        recent season), else the selected season's CVI → EUR (the profile's
        own fallback). None when neither exists."""
        try:
            gpa = load_gpa_values()
            perf = _build_player_season_perf_table(gpa, comp_player_minutes_df)
            dob = bio.get('birthDate') if isinstance(bio, pd.Series) else None
            cvi = None
            sid = most_recent_season_for_player(perf, pid)
            if sid is not None:
                cc = compute_career_cvi(
                    pid, sid, perf_table=perf,
                    dob_lookup=lambda _p: pd.to_datetime(dob, errors='coerce') if dob is not None else None)
                cvi = (cc or {}).get('career_cvi')
            if cvi is None or pd.isna(cvi):
                single = row_df.copy()
                if 'Total Value' not in single.columns and gpa is not None and not gpa.empty:
                    # GPA merge may not be present in these stats; backfill
                    _r = gpa[(gpa['playerId'] == pid) & (gpa['seasonId'] == selected_season_id)]
                    _vc = next((c for c in ('Total Value', 'total_v_per_90') if c in _r.columns), None)
                    if not _r.empty and _vc:
                        single['Total Value'] = float(_r[_vc].iloc[0])
                _prior = (build_player_priors_lookup(perf, selected_season_id)
                          if selected_season_id is not None else {})
                _age = _calculate_age(dob)
                blk = compute_cvi_columns(
                    single, age_lookup=lambda _p: _age,
                    comp_id_lookup=lambda _p: _tv_comp_id,
                    prior_lookup=lambda _p: _prior.get(int(_p)) if _p is not None else None)
                cvi = blk.iloc[0].get('_CVI') if not blk.empty else None
            if cvi is None or pd.isna(cvi) or float(cvi) <= 0:
                return None
            return cvi_to_projected_eur(float(cvi), position_group=_cvi_position_group(pos),
                                        competition_id=_tv_comp_id)
        except Exception:
            logger.exception("comparison GK value failed")
            return None

    def _context(row_df, colour, slot):
        row = row_df.iloc[0]
        pid = int(row['playerId'])
        bio = (player_details_df.loc[pid] if not player_details_df.empty
               and pid in player_details_df.index else pd.Series(dtype='object'))
        pos = row.get('primaryPosition', 'N/A')
        ctx = dict(pid=pid, name=row.get('playerName', 'N/A'), team=row.get('teamName', 'N/A'),
                   pos=pos, bio=bio, colour=colour, slot=slot, is_gk=_is_gk(pos),
                   minutes=float(np.nan_to_num(pd.to_numeric(row.get('totalMinutes', 0),
                                                             errors='coerce'), nan=0.0)))
        ctx['card'] = engine_card_row(_eng_df, pid, active_season_ids)
        ctx['value'] = ctx['w_ev'] = ctx['eng_mins'] = None
        if ctx['is_gk']:
            ctx['value'] = _legacy_gk_value(row_df, pid, pos, bio)
        elif not _eng_value_rows.empty:
            _v = _eng_value_rows[_eng_value_rows['playerId'] == pid].dropna(subset=['engine_value_eur'])
            if not _v.empty:
                ctx['value'] = float(_v['engine_value_eur'].iloc[0])
                ctx['w_ev'] = _v['w_evidence'].iloc[0]
                ctx['eng_mins'] = _v['mins_played'].iloc[0]
        try:
            ctx['style'] = None if ctx['is_gk'] else get_scoped_style(pid, active_season_ids)
        except Exception:
            ctx['style'] = None
        return ctx

    ctx_a = _context(player_a_data, theme.COMPARE_A, 'a')
    ctx_b = _context(player_b_data, theme.COMPARE_B, 'b')
    players = (ctx_a, ctx_b)
    if ctx_a['pid'] == ctx_b['pid']:
        st.info("Player A and Player B are the same player — pick another Player B "
                "in the sidebar.")

    # --- 4. Header: the profile's player information, side by side ---
    st.subheader(f"Comparing: {selected_player_a_name} vs. {selected_player_b_name}")

    _head_cols = st.columns(2, gap="large")
    for ctx, col in zip(players, _head_cols):
        bio = ctx['bio']
        with col:
            _img, _info = st.columns([1, 2.6])
            _img.image(_txt(bio.get('imageDataURL')) or _PLACEHOLDER_IMG, width=110)
            age = _calculate_age(bio.get('birthDate'))
            age_txt = f"{int(age)}" if isinstance(age, (int, float)) and age == age else "N/A"
            foot = _txt(bio.get('foot'))
            _h, _w = _txt(bio.get('height')), _txt(bio.get('weight'))
            _bits = [f"Age {age_txt}", _txt(bio.get('passportArea')),
                     f"{foot.capitalize()} foot" if foot else None,
                     f"{_h} cm" if _h and _h != '0' else None,
                     f"{_w} kg" if _w and _w != '0' else None]
            _info.markdown(
                f"<div style='font-size:1.3rem;font-weight:700;color:{ctx['colour']};"
                f"line-height:1.2'>{ctx['name']}</div>"
                f"<div style='margin-top:0.2rem'>{ctx['team']} · {ctx['pos']} · "
                f"{int(ctx['minutes'])}′</div>",
                unsafe_allow_html=True)
            _info.caption(" · ".join(str(b) for b in _bits if b))
            if _info.button("Open profile", key=f"cmp_open_profile_{ctx['slot']}"):
                st.session_state.selected_player_id = ctx['pid']
                st.session_state.nav_to_profile = True
                st.rerun()

            _m = st.columns(3)
            _m[0].metric(
                "Projected value",
                "—" if ctx['value'] is None else f"€{ctx['value']:,.0f}",
                help=eur_interval_ui.help_text(_eurcal, gk=ctx['is_gk']))
            if not ctx['is_gk']:
                # range as a CAPTION, never a metric delta (Streamlit draws an
                # arrow for any delta text)
                _rng = eur_interval_ui.range_sentence(
                    ctx['value'], _eurcal, w_evidence=ctx['w_ev'], mins=ctx['eng_mins'])
                if _rng:
                    _m[0].caption(_rng)
            _sty = ctx['style']
            if _sty:
                _m[1].metric("Role", str(_sty.get('role') or '—'),
                             help="Observed engine role — learned from where his events "
                                  "actually happen match by match, not the lineup-card position.")
                _fit = _sty.get('style_fit')
                _m[2].metric("Style", str(_sty.get('style') or '—'),
                             delta=(f"{float(_fit):.0f}% fit"
                                    if _fit is not None and pd.notna(_fit) else None),
                             delta_color="off",
                             help="Tendency-derived archetype — how he plays the role, not how "
                                  "well. Never part of the rating. Fit = how strongly he "
                                  "expresses the style (percentile vs the role cohort).")

    st.divider()

    # --- 5. ACP Index, side by side ---
    with st.expander("ℹ️ How these ratings work"):
        st.markdown(RATINGS_EXPLAINER_MD)
    st.subheader("ACP Index")

    # one floor for the whole page (template percentiles, radar, stats)
    _floor = template_pct.minutes_floor(player_stats_with_scores_df)

    def _gk_legacy(row_df):
        # a keeper under the floor sits at the pipeline's placeholder 0 —
        # place him into the sample first (the minutes-floor RULE)
        provisional = float(np.nan_to_num(pd.to_numeric(
            row_df.iloc[0].get('totalMinutes', 0), errors='coerce'), nan=0.0)) < _floor
        if provisional:
            row_df = template_pct.score_below_floor(
                row_df, player_stats_with_scores_df, POSITION_GROUPS, WEIGHTS,
                INVERT_METRICS, _floor)
        _scored = []
        for _t in _GK_TEMPLATES:
            _s = row_df.iloc[0].get(f'{_t}_Score')
            if _s is not None and pd.notna(_s):
                _scored.append((_t, float(_s)))
        return (max(_scored, key=lambda x: x[1]) + (provisional,)) if _scored else None

    _head_rows = {}
    _comp_rows = {}
    _notes = []
    for ctx, row_df in ((ctx_a, player_a_data), (ctx_b, player_b_data)):
        e = ctx['card']['e']
        col = ctx['name'] if ctx_a['name'] != ctx_b['name'] else f"{ctx['name']} ({ctx['slot'].upper()})"
        ctx['col'] = col
        if e is None:
            _gk = _gk_legacy(row_df) if ctx['is_gk'] else None
            _head_rows[col] = {
                'ACP Rating': '—', 'Projection (next season)': '—',
                'Career (as-of)': '—', 'Evidence': '—',
                **({'GK Rating (legacy)': f"{_gk[1]:.0f} ({_gk[0]})"
                    + (f" · provisional, < {_floor}′" if _gk[2] else "")} if _gk else {})}
            _comp_rows[col] = {lbl: np.nan for lbl, _c, _h in _ENGINE_COMPONENTS}
            _notes.append(f"**{ctx['name']}** — " + (
                "goalkeeper: the engine does not rate keepers (legacy GK rating shown)."
                if ctx['is_gk'] else
                "not rated by the engine for this scope (below the 90-minute floor, or "
                "no role assignment yet)."))
            continue
        _abs = (f" (abs {e['acp_rating_abs']:.0f})" if pd.notna(e.get('acp_rating_abs'))
                and e.get('league') == 'CAMP' else "")
        _proj = (f"{e['projection']:.0f} ± {e['band_sd']:.0f}" if pd.notna(e.get('projection'))
                 else "—")
        if _proj != "—" and pd.notna(e.get('projection_abs')) and e.get('league') == 'CAMP':
            _proj += f" (abs {e['projection_abs']:.0f})"
        _head_rows[col] = {
            'ACP Rating': f"{e['acp_rating']:.0f}{_abs}",
            'Projection (next season)': _proj,
            'Career (as-of)': f"{e['career_asof']:.0f}" if pd.notna(e.get('career_asof')) else "—",
            'Evidence': f"{e['w_evidence']:.0%}" if pd.notna(e.get('w_evidence')) else "—",
        }
        _comp_rows[col] = {lbl: (float(e[c]) * 100 if pd.notna(e.get(c)) else np.nan)
                           for lbl, c, _h in _ENGINE_COMPONENTS}
        _n = []
        if ctx['card']['career_view']:
            _n.append(f"career view, minutes-weighted across {len(ctx['card']['escope'])} "
                      f"rated seasons ({int(e['mins_played'])}′)")
        if ctx['card']['stale']:
            _n.append("not rated in the selected season — showing last rated season "
                      f"({SEASON_ID_MAP.get(int(e['seasonId']), e['seasonId'])})")
        if pd.notna(e.get('seasons_ago')) and int(e['seasons_ago']) >= 1:
            _n.append(f"last rated {int(e['seasons_ago'])} season(s) ago — the projection "
                      f"carries an extra age step + wider band")
        if float(e['mins_played']) < 500:
            _n.append(f"thin sample ({int(e['mins_played'])}′)")
        _esc = ctx['card']['escope']
        if len(_esc) > 1 and not ctx['card']['career_view']:   # a winter mover
            for _, _o in _esc[_esc.index != e.name].iterrows():
                _n.append(f"also rated in {_o['league']} ({int(_o['mins_played'])}′, "
                          f"rating {_o['acp_rating']:.0f})")
        if pd.notna(e.get('role')):
            _n.append(f"rated as {e['role']}")
        if _n:
            _notes.append(f"**{ctx['name']}** — " + "; ".join(_n) + ".")

    _acp_l, _acp_r = st.columns([1, 1.3], gap="large")
    with _acp_l:
        _hdf = pd.DataFrame(_head_rows).fillna('—')
        _hdf.index.name = 'Metric'
        st.dataframe(_hdf, width="stretch")
    with _acp_r:
        _cdf = pd.DataFrame(_comp_rows)
        _cdf.index.name = 'Component'
        st.dataframe(
            _cdf, width="stretch",
            column_config={c: st.column_config.ProgressColumn(
                c, min_value=0, max_value=100, format="%.0f") for c in _cdf.columns})
        with st.popover("What each component measures"):
            st.markdown(engine_metric_help_md([h for _l, _c, h in _ENGINE_COMPONENTS]))
    for _note in _notes:
        st.caption(_note)
    st.caption(
        f"Engine {_eng_meta.get('rating_version', '?')} · "
        f"projection {_eng_meta.get('projection_version', '?')} · "
        f"data through {_eng_meta.get('data_through', '?')} · "
        f"ratings within league × season × role cohort (each player in his own); "
        f"set piece shown separately (not in the rating)")

    st.divider()

    # --- 6. One sample for the radar and the stats tables ---
    # Both players ranked against the template's position group at/above the
    # scope's minutes floor, with the DefR columns merged in (the percentile
    # cache may drop them).
    radar_stats_df = merge_defr_values_into_stats(
        player_stats_with_scores_df, active_season_ids, selected_comp_ids)
    _rs_mins = pd.to_numeric(radar_stats_df['totalMinutes'], errors='coerce').fillna(0)
    _sample = radar_stats_df[radar_stats_df['primaryPosition'].isin(positions_in_group)
                             & (_rs_mins >= _floor)]
    # ignore_index: A and B may be the same player (one index label twice)
    _rows_ab = pd.concat([radar_stats_df[radar_stats_df['playerId'] == ctx_a['pid']].head(1),
                          radar_stats_df[radar_stats_df['playerId'] == ctx_b['pid']].head(1)],
                         ignore_index=True)
    if len(_rows_ab) < 2:   # merge dropped a row: fall back to the scored rows
        _rows_ab = pd.concat([player_a_data.head(1), player_b_data.head(1)], ignore_index=True)
    row_a, row_b = _rows_ab.iloc[[0]], _rows_ab.iloc[[1]]

    _sections = ["Radar", "ACP Index cards", "Stats"]
    _active = st.radio("Comparison section", _sections, horizontal=True,
                       label_visibility="collapsed", key="comparison_active_section")

    if _active == "Radar":
        st.subheader("Template Radar")
        _below_threshold = [f"{c['name']} ({int(c['minutes'])} min)" for c in players
                            if c['minutes'] < 300]
        if _below_threshold:
            st.info(f"⚠️ **Insufficient sample size** — {' and '.join(_below_threshold)}.")

        _rc1, _rc2 = st.columns([2, 1])
        with _rc1:
            _radar_style = st.radio("Radar Style", ["Percentile", "Raw Values (mean ± 2σ)"],
                                    index=1, horizontal=True, key="cmp_radar_style")
        with _rc2:
            _use_defr = st.toggle(
                "Defensive metrics → DefR", value=False, key="cmp_defr_mode",
                help="Swap each defensive axis (tackles, interceptions, recoveries, "
                     "clearances, aerials) to its Defensive Responsibility value — "
                     "actions above/below what the player's role is expected to make.")

        metrics_to_plot = [m for m in WEIGHTS[selected_template]
                           if m in radar_stats_df.columns and m not in RADAR_HIDDEN_METRICS]
        if _use_defr:
            _mapped, _seen = [], set()
            for _m in metrics_to_plot:
                _mm = DEFR_RADAR_MAP.get(_m, _m)
                if _mm in radar_stats_df.columns and _mm not in _seen:
                    _mapped.append(_mm); _seen.add(_mm)
            if _mapped:
                metrics_to_plot = _mapped
                st.caption("🛡️ Defensive axes show **DefR** (actions above/below role expectation).")

        ranked, placed = template_pct.rank_in_template(
            _rows_ab, radar_stats_df, selected_template, POSITION_GROUPS, WEIGHTS,
            INVERT_METRICS, _floor,
            extra_metrics=[m for m in metrics_to_plot if m not in WEIGHTS[selected_template]])
        r_a, r_b = ranked.iloc[[0]], ranked.iloc[[1]]

        _provisional = [c['name'] for c in players if c['minutes'] < _floor]
        _outside = [c['name'] for c in players
                    if c['pid'] in placed and c['minutes'] >= _floor]
        _season_lbl = (SEASON_ID_MAP.get(selected_season_id, 'All Seasons')
                       if selected_season_id else 'All Seasons')
        _note = (f"Provisional: {', '.join(_provisional)} (< {_floor}′)" if _provisional else None)

        if len(metrics_to_plot) < 3:
            st.warning(f"Not enough radar metrics for the {selected_template} template.")
        else:
            with MPL_LOCK:
                fig = plt.figure(figsize=(15, 15))
                # [left, bottom, width, height] - This centers the radar
                ax_radar = fig.add_axes([0.15, 0.15, 0.7, 0.7], polar=True)
                plot_comparison_radar(
                    ax_radar, r_a, r_b, metrics_to_plot, selected_template,
                    radar_mode='raw' if _radar_style.startswith('Raw') else 'percentile',
                    population_data=_sample, season_label=_season_lbl, note=_note)
                st.pyplot(fig, width="stretch")

        _cap = (f"Both players ranked against the **{selected_template}** positions "
                f"with ≥ {_floor}′ ({len(_sample)} players)")
        if _provisional:
            _cap += (f"; **provisional** for {' and '.join(_provisional)} — below the "
                     f"{_floor}′ floor, placed into that sample on their own")
        if _outside:
            _cap += (f"; {' and '.join(_outside)} "
                     f"{'plays' if len(_outside) == 1 else 'play'} outside this position group, "
                     f"so the template score is placed on the group's range")
        st.caption(_cap + ".")

    elif _active == "ACP Index cards":
        st.caption("Each player's ACP Index radar is drawn against his OWN league × season "
                   "× role cohort (raw per-90 on a mean ± 2σ scale), as on the profile.")
        for ctx in players:
            st.markdown(f"<div style='font-size:1.15rem;font-weight:700;color:{ctx['colour']}'>"
                        f"{ctx['name']}</div>", unsafe_allow_html=True)
            card = ctx['card']
            if card['e'] is None:
                st.info("No ACP Index card — " + ("goalkeepers are not rated by the engine."
                        if ctx['is_gk'] else "not rated by the engine for this scope."))
                continue
            try:
                _png = _render_acp_index_card_png(
                    ctx['pid'], tuple(sorted(card['e_sids'])) if card['e_sids'] else (),
                    STATS_CACHE_VERSION, str(_eng_meta.get('rating_version', '')),
                    card['career_view'], _eng_df, card['e'], _eng_meta)
            except Exception:
                logger.exception("comparison ACP card failed")
                _png = None
            if _png:
                st.image(_png, width="stretch")
            else:
                st.caption("Too few ACP Index axes with data for a radar.")
            try:
                if get_scoped_tendencies(ctx['pid'], card['e_sids']) is not None:
                    with st.expander(f"🎚️ Tendencies — how {ctx['name']} plays the role"):
                        st.caption(
                            "Each bar is a **within-role percentile** of what he attempts "
                            "(50 = typical for the role) — style, not quality, and never "
                            "part of the rating. The leaning side is coloured.")
                        render_tendencies_panel(ctx['pid'], card['e_sids'])
            except Exception:
                logger.exception("comparison tendencies panel failed")

    elif _active == "Stats":
        st.subheader("Overall Season Stats")
        show_totals = st.toggle("Show Season Totals", value=False, key="cmp_show_totals")
        st.caption(
            f"{'TOTAL' if show_totals else 'PER 90'} stats — {ctx_a['name']} "
            f"{int(ctx_a['minutes'])}′, {ctx_b['name']} {int(ctx_b['minutes'])}′. Percentile "
            f"ranks each player's per-90 value against the **{selected_template}** positions "
            f"with ≥ {_floor}′ ({len(_sample)} players), so the two columns compare like "
            f"with like.")

        def _percentile_for(metric, p90_value, pid):
            if metric not in _sample.columns or pd.isna(p90_value):
                return None
            pop = _sample[_sample['playerId'] != pid]
            vals = pd.to_numeric(pop[metric], errors='coerce').dropna()
            if len(vals) < 5 or vals.std() == 0:
                return None
            pct = scipy.stats.percentileofscore(vals, p90_value, kind='weak')
            return float(100.0 - pct if metric in INVERT_METRICS else pct)

        per90 = {ctx['slot']: r.iloc[0] for ctx, r in ((ctx_a, row_a), (ctx_b, row_b))}
        shown = {ctx['slot']: (_season_totals(per90[ctx['slot']], ctx['minutes'],
                                              ENGINE_DISPLAY_METRICS)
                               if show_totals else per90[ctx['slot']]) for ctx in players}
        stat_groups = {
            "Output": OUTPUT_METRICS,
            "Passing": PASSING_METRICS,
            "Defensive": DEFENSIVE_METRICS,
            "Defensive Responsibility (DefR)": DEFR_DISPLAY_METRICS,
            "Dribbling": DRIBBLING_METRICS,
            "ACP Index": ENGINE_DISPLAY_METRICS,
            "Goalkeeping": GOALKEEPING_METRICS + ['GK Passes successful %',
                                                  'GK Long passes successful %'],
        }
        _any_gk = any(c['is_gk'] for c in players)
        _any_outfield = any(not c['is_gk'] for c in players)
        _pct_names = [f"{ctx['col']} %ile" for ctx in players]
        for group_name, group_metrics in stat_groups.items():
            if group_name == 'Goalkeeping' and not _any_gk:
                continue
            if group_name != 'Goalkeeping' and not _any_outfield:
                continue
            _rows, _pcts = [], {n: [] for n in _pct_names}
            for _metric in dict.fromkeys(group_metrics):
                _vals = [shown[c['slot']].get(_metric, np.nan) for c in players]
                _nums = [pd.to_numeric(pd.Series([v]), errors='coerce').iloc[0] for v in _vals]
                if all(pd.isna(v) or v == 0 for v in _nums):
                    continue
                _row = {'Metric': _metric}
                for ctx, v, pname in zip(players, _nums, _pct_names):
                    _p = (None if _metric in ENGINE_ROLE_PCT_METRICS else
                          _percentile_for(_metric, pd.to_numeric(pd.Series(
                              [per90[ctx['slot']].get(_metric, np.nan)]),
                              errors='coerce').iloc[0], ctx['pid']))
                    _row[ctx['col']] = _fmt_stat(_metric, v, THOUSANDTHS_METRICS)
                    # text column: Streamlit draws an empty numeric cell as a
                    # grey "None" whatever the Styler format says
                    _row[pname] = '—' if _p is None else f"{int(round(_p))}"
                    _pcts[pname].append(_p)
                _rows.append(_row)
            if not _rows:
                continue
            with st.expander(f"**{group_name} Stats**", expanded=(group_name == 'Output')):
                _df = pd.DataFrame(_rows).set_index('Metric')
                _df = _df[[ctx_a['col'], _pct_names[0], ctx_b['col'], _pct_names[1]]]
                _styled = _df.style
                for pname in _pct_names:
                    _styled = _styled.apply(
                        lambda _c, _p=_pcts[pname]: [_percentile_color(v) for v in _p],
                        subset=[pname])
                if group_name == "ACP Index":
                    st.caption("% values are already within-role percentiles (50 = typical "
                               "for the role, each player in his own cohort), so they get no "
                               "second percentile (—).")
                    with st.popover("What each metric measures"):
                        st.markdown(engine_metric_help_md(_df.index))
                st.dataframe(_styled, width="stretch")
