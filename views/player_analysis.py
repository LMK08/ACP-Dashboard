"""Player Analysis view — extracted verbatim from app.py's `elif analysis_type == 'Player Analysis'` branch (2026-09).

Collaborators are read from the running app module at call time (the
pattern opposition_report.py uses), so importing this module never imports
app.py. The binding block at the top of render() IS the page's dependency
list: everything it reads from app.py, nothing else.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import sys


def _app():
    return sys.modules['__main__']


def render():
    app = _app()
    COMPETITIONS = app.COMPETITIONS
    DEFENSIVE_METRICS = app.DEFENSIVE_METRICS
    DEFR_DISPLAY_METRICS = app.DEFR_DISPLAY_METRICS
    DRIBBLING_METRICS = app.DRIBBLING_METRICS
    ENGINE_DISPLAY_METRICS = app.ENGINE_DISPLAY_METRICS
    GOALKEEPING_METRICS = app.GOALKEEPING_METRICS
    INVERT_METRICS = app.INVERT_METRICS
    OUTPUT_METRICS = app.OUTPUT_METRICS
    PASSING_METRICS = app.PASSING_METRICS
    POSITION_GROUPS = app.POSITION_GROUPS
    RATINGS_EXPLAINER_MD = app.RATINGS_EXPLAINER_MD
    SEASON_ID_MAP = app.SEASON_ID_MAP
    THOUSANDTHS_METRICS = app.THOUSANDTHS_METRICS
    WEIGHTS = app.WEIGHTS
    WHOLE_NUMBER_METRICS = app.WHOLE_NUMBER_METRICS
    _build_player_season_perf_table = app._build_player_season_perf_table
    _calculate_age = app._calculate_age
    auto_column_config = app.auto_column_config
    build_player_priors_lookup = app.build_player_priors_lookup
    bulk_export_radars = app.bulk_export_radars
    calculate_all_player_stats = app.calculate_all_player_stats
    calculate_player_percentiles_and_scores = app.calculate_player_percentiles_and_scores
    competition_for_season = app.competition_for_season
    compute_cvi_columns = app.compute_cvi_columns
    get_all_players_minutes_by_position = app.get_all_players_minutes_by_position
    get_career_engine_role_map = app.get_career_engine_role_map
    get_filtered_events = app.get_filtered_events
    get_season_events = app.get_season_events
    get_season_ids_for_selection = app.get_season_ids_for_selection
    get_season_player_minutes = app.get_season_player_minutes
    league_selector = app.league_selector
    load_and_score_player_stats = app.load_and_score_player_stats
    load_gpa_values = app.load_gpa_values
    load_player_details = app.load_player_details
    load_player_engine = app.load_player_engine
    load_styles = app.load_styles
    logger = app.logger
    player_minutes_data = app.player_minutes_data
    raw_events_df = app.raw_events_df
    season_selector = app.season_selector
    player_stats_with_scores_df = app.player_stats_with_scores_df


    # --- League & Season Selector ---
    selected_comp_ids = league_selector("player_analysis")
    selected_season_id = season_selector("player_analysis", include_all_seasons=True, comp_ids=selected_comp_ids)
    active_season_ids = get_season_ids_for_selection(selected_season_id, selected_comp_ids)
    analysis_events_df = get_filtered_events(raw_events_df, active_season_ids, selected_comp_ids)
    analysis_player_minutes_df = get_season_player_minutes(player_minutes_data, active_season_ids, comp_ids=selected_comp_ids)

    # --- 1. Load Data ---
    try:
        with st.spinner("Loading player statistics..."):
            player_stats_df, player_stats_with_scores_df = load_and_score_player_stats(
                analysis_events_df, analysis_player_minutes_df, selected_season_id, active_season_ids, selected_comp_ids
            )
    except Exception as e:
        st.error(f"An error occurred calculating player stats: {e}")
        logger.exception("Error in Player Analysis stats calculation")
        st.stop()

    if player_stats_with_scores_df.empty:
        st.warning("No players found with sufficient minutes for analysis.")
        st.stop()

    # --- 2. Sidebar Controls ---
    st.sidebar.subheader("Analysis Options")

    # Template / mode selector — Overview is the default landing page
    _TEMPLATE_GROUPS = {
        'Goalkeepers': ['Shot Stopper', 'Cross Claimer', 'Ball-playing GK'],
        'Center Backs': ['Ball-Playing Centerback', 'Stopper', 'Athletic Centerback'],
        'Full Backs': ['Full Back', 'Wingback', 'Inverted Full Back'],
        'Central Midfielders': ['Box-to-Box', 'Holding Mid', 'Ball-Winning Mid', 'Deep-lying Playmaker'],
        'Attacking Mids / Wingers': ['Advanced Playmaker', 'Wide Winger', 'Creative Winger', 'Inside Forward'],
        'Forwards': ['Shadow Striker', 'Mobile Striker', 'Poacher', 'Target Man', 'Pressing Forward'],
    }
    # Build ordered template list from groups (preserving group order)
    _ordered_templates = []
    for _grp_templates in _TEMPLATE_GROUPS.values():
        _ordered_templates.extend([t for t in _grp_templates if t in POSITION_GROUPS])
    _selector_options = ["Overview"] + _ordered_templates + ["Individual Metric", "Peer Scatter"]
    _selected_view = st.sidebar.selectbox(
        "View:",
        _selector_options,
        index=0,
        key="player_analysis_view"
    )

    # Minimum minutes filter. Floor lowered 500->90 (Lucas 2026-06): the
    # ACP engine now rates players down to 90 min (scored against the
    # >=500 cohort, then minutes-shrunk toward replacement), so low-minute
    # players are penalised by the shrink rather than excluded. Default 90
    # surfaces them; the shrinkage keeps them off the top of the boards.
    max_minutes = int(player_stats_with_scores_df['totalMinutes'].max())
    min_minutes_filter = st.sidebar.slider(
        "Minimum Minutes Played:",
        min_value=90,
        max_value=max(max_minutes, 500),
        value=90,
        step=45,
        key="player_analysis_min_minutes",
        help="ACP ratings exist down to 90 min (heavily shrunk toward "
             "replacement). Bespoke template scores still need 500+.",
    )

    # Number of players to display
    num_players = st.sidebar.slider(
        "Number of Players to Display:",
        min_value=5,
        max_value=50,
        value=20,
        step=5,
        key="player_analysis_num_players"
    )

    # --- Bulk Radar Export ---
    st.sidebar.markdown("---")
    # Per-radar PNG files in a render directory — each completed
    # radar is independently committed to disk, so a kill mid-render
    # leaves a directory of valid PNGs the user can still download
    # as a ZIP. The ZIP is built lazily at download time, never
    # streamed/written during render → no central-directory
    # truncation problem.
    import os as _os, io as _io, time as _time, pickle as _pickle
    import hashlib as _hashlib, json as _json, zipfile as _zipfile
    _BULK_CACHE_DIR = "/tmp/dashboard_bulk_export"
    _BULK_CACHE_ERROR = None
    try:
        _os.makedirs(_BULK_CACHE_DIR, exist_ok=True)
        _probe = _os.path.join(_BULK_CACHE_DIR, '.writable_probe')
        with open(_probe, 'w') as _pf:
            _pf.write('ok')
        _os.unlink(_probe)
    except Exception as _cache_dir_exc:
        _BULK_CACHE_ERROR = f"{type(_cache_dir_exc).__name__}: {_cache_dir_exc}"

    def _bulk_cache_key(season_lbl, groups, mode, min_mins):
        payload = _json.dumps({
            "season": str(season_lbl),
            "groups": sorted([str(g) for g in groups]),
            "mode": str(mode),
            "min_mins": int(min_mins),
        }, sort_keys=True)
        return _hashlib.md5(payload.encode("utf-8")).hexdigest()[:12]

    def _bulk_render_dir(key):
        return _os.path.join(_BULK_CACHE_DIR, f"radars__{key}")

    def _bulk_meta_path(key):
        return _os.path.join(_bulk_render_dir(key), 'meta.pkl')

    def _list_cached_renders():
        """Return one entry per render directory under the cache
        dir, regardless of meta state. Each entry surfaces the PNG
        count, total bytes on disk, and last-modified time so
        partial/crashed runs are still visible and downloadable."""
        entries = []
        try:
            names = _os.listdir(_BULK_CACHE_DIR)
        except (FileNotFoundError, Exception):
            return entries
        for name in names:
            full = _os.path.join(_BULK_CACHE_DIR, name)
            if not _os.path.isdir(full):
                continue
            try:
                pngs = [f for f in _os.listdir(full) if f.endswith('.png')]
            except Exception:
                continue
            if not pngs:
                continue
            meta_path = _os.path.join(full, 'meta.pkl')
            meta = None
            if _os.path.exists(meta_path):
                try:
                    with open(meta_path, 'rb') as _f:
                        meta = _pickle.load(_f)
                except Exception:
                    meta = None
            if meta is None:
                meta = {'status': 'incomplete', 'label': name}
            try:
                total_bytes = sum(_os.path.getsize(_os.path.join(full, f))
                                   for f in pngs)
                mtime = max((_os.path.getmtime(_os.path.join(full, f))
                              for f in pngs), default=0)
            except Exception:
                total_bytes, mtime = 0, 0
            entries.append({
                'path': full,
                'meta': meta,
                'mtime': mtime,
                'size': total_bytes,
                'png_count': len(pngs),
            })
        entries.sort(key=lambda e: e['mtime'], reverse=True)
        return entries

    def _build_zip_from_dir(render_dir):
        """Build a ZIP byte-string on the fly from every PNG in the
        render directory. Memory cost is proportional to ZIP size at
        click time only — not held during rendering."""
        buf = _io.BytesIO()
        with _zipfile.ZipFile(buf, 'w', _zipfile.ZIP_DEFLATED,
                               compresslevel=1) as _zf:
            for fn in sorted(_os.listdir(render_dir)):
                if not fn.endswith('.png'):
                    continue
                _fp = _os.path.join(render_dir, fn)
                with open(_fp, 'rb') as _ff:
                    _zf.writestr(fn, _ff.read())
        return buf.getvalue()

    # Apply minutes filter
    filtered_df = player_stats_with_scores_df[
        player_stats_with_scores_df['totalMinutes'] >= min_minutes_filter
    ].copy()

    if filtered_df.empty:
        st.warning(f"No players found with {min_minutes_filter}+ minutes. Try lowering the threshold.")
        st.stop()

    # --- Age filter ---
    analysis_player_details_df = load_player_details()
    if not analysis_player_details_df.empty and 'birthDate' in analysis_player_details_df.columns:
        _ages_series = analysis_player_details_df['birthDate'].apply(_calculate_age)
        _numeric_ages = pd.to_numeric(_ages_series, errors='coerce').dropna()
        if not _numeric_ages.empty:
            min_age_available = int(_numeric_ages.min())
            max_age_available = int(_numeric_ages.max()) + 1
            age_range = st.sidebar.slider(
                "Age Range:",
                min_value=min_age_available,
                max_value=max_age_available,
                value=(min_age_available, max_age_available),
                key="player_analysis_age_range"
            )
            if age_range != (min_age_available, max_age_available):
                valid_ids = _numeric_ages[
                    (_numeric_ages >= age_range[0]) & (_numeric_ages <= age_range[1])
                ].index.tolist()
                filtered_df = filtered_df[filtered_df['playerId'].isin(valid_ids)]
                if filtered_df.empty:
                    st.warning(f"No players found in age range {age_range[0]}-{age_range[1]}.")
                    st.stop()

    # --- Observed role / Style filters (Lucas 2026-07-17) ------------
    # Two scope-aware maps (playerId -> observed engine role / tendency-
    # derived style, highest-minutes row in the selected scope). They
    # drive sidebar filters that subset the WHOLE analysis population —
    # every view downstream (role board, template tables, scatter,
    # individual metric) inherits them — and Role/Style columns on the
    # per-template tables so the dataframe search finds them. Both maps
    # degrade to empty (filters hidden) if the parquets are missing.
    _ANALYSIS_ROLE_ORDER = ['Striker', 'Wide Attacker', 'Advanced Midfielder',
                            'Deep Midfielder', 'Wide Defender', 'Central Defender']
    _scope_sids = None
    if active_season_ids is not None:
        _scope_sids = [int(s) for s in (active_season_ids
                       if isinstance(active_season_ids, (list, tuple, set))
                       else [active_season_ids])]
    _obs_role_map, _an_style_map = {}, {}
    try:
        # SAME career share-weighted role the Best-Players-by-Role board
        # buckets by — one source of truth, so filtering to "Striker"
        # keeps exactly the players the board files under Striker.
        _obs_role_map = get_career_engine_role_map()
    except Exception:
        logger.exception("observed-role filter map failed")
    try:
        _st_flt = load_styles()
        if _st_flt is not None and not _st_flt.empty:
            _sf = _st_flt
            if _scope_sids:
                _sfs = _sf[_sf['seasonId'].isin(_scope_sids)]
                _sf = _sfs if not _sfs.empty else _sf
            _sf = _sf.sort_values('mins_played').drop_duplicates(
                'playerId', keep='last')
            _an_style_map = {int(p): s for p, s in
                             zip(_sf['playerId'], _sf['style']) if pd.notna(s)}
    except Exception:
        logger.exception("style filter map failed")

    _roles_present = [r for r in _ANALYSIS_ROLE_ORDER
                      if r in set(_obs_role_map.values())]
    if _roles_present:
        _sel_obs_role = st.sidebar.selectbox(
            "Observed role:", ['All roles'] + _roles_present, index=0,
            key="analysis_obs_role_filter",
            help="The engine's data-derived role (where his events happen "
                 "match by match) — not the lineup position.")
        if _sel_obs_role != 'All roles':
            filtered_df = filtered_df[filtered_df['playerId'].map(
                _obs_role_map) == _sel_obs_role]
            if filtered_df.empty:
                st.warning(f"No players with observed role "
                           f"“{_sel_obs_role}” in this scope.")
                st.stop()
    _styles_present = sorted({s for s in _an_style_map.values()
                              if isinstance(s, str)})
    if _styles_present:
        _sel_an_style = st.sidebar.selectbox(
            "Style:", ['All styles'] + _styles_present, index=0,
            key="analysis_style_filter",
            help="Tendency-derived archetype (descriptive, never in the "
                 "rating). Conventional = no pronounced lean.")
        if _sel_an_style != 'All styles':
            filtered_df = filtered_df[filtered_df['playerId'].map(
                _an_style_map) == _sel_an_style]
            if filtered_df.empty:
                st.warning(f"No players with style “{_sel_an_style}” "
                           f"in this scope.")
                st.stop()

    # --- Show Only Position toggle (HIDDEN per Lucas 2026-06) ---
    # Toggle removed from the sidebar; pos-played filtering stays off.
    analysis_pos_played_filter = False
    analysis_pos_played_active = False
    analysis_selected_positions = []
    if analysis_pos_played_filter:
        all_pos_minutes = get_all_players_minutes_by_position(analysis_events_df)
        if not all_pos_minutes.empty:
            available_positions = sorted(all_pos_minutes['Position'].unique().tolist())
            analysis_selected_positions = st.sidebar.multiselect(
                "Position(s):",
                available_positions,
                default=available_positions[:1],
                key="analysis_pos_played_position"
            )
            if analysis_selected_positions:
                pos_min_for_positions = all_pos_minutes[all_pos_minutes['Position'].isin(analysis_selected_positions)].groupby('playerId')['Minutes'].sum().reset_index().rename(columns={'Minutes': 'posMinutes'})
                filtered_df = filtered_df.merge(pos_min_for_positions, on='playerId', how='inner')
                analysis_pos_played_active = not filtered_df.empty
                if filtered_df.empty:
                    pos_label = "/".join(analysis_selected_positions)
                    st.warning(f"No players found who played at {pos_label} with current filters.")
                    st.stop()

    # --- Global rating-adjustment toggles ----------------------------
    # These adjustments operate on the Rating column (Role_Score) used
    # in Overview + per-template views. They are intentionally NOT
    # offered on Individual Metric (per user — they only make sense
    # for overall profile ratings, not per-metric leaderboards).
    st.sidebar.markdown("---")
    # Same-age-peers + cross-tier toggles REMOVED (Lucas 2026-06-12):
    # the engine's age curve handles age context in the projection,
    # and the abs columns handle cross-league translation. Flags stay
    # pinned off; their dormant downstream code paths were deleted.
    age_adjusted = False

    # Cross-tier translation — Opta strength multiplier only. The
    # empirical-median variant is per-metric, so it doesn't apply
    # to a composite Role_Score.
    cross_tier_mode = 'Off'
    _trans_src_comp = _trans_tgt_comp = None
    _rating_multiplier = None
    _rating_caption = None

    # --- CVI (Composite Value Index) toggle -----------------------
    # When ON, a CVI column is appended to the right of the Rating
    # column in Overview + per-template tables, and (optionally) the
    # ranking re-sorts by CVI. Position-tuned age curve
    # (see CVI_AGE_VALUE_PARAMS) calibrated off the 27 reported transfers.
    show_cvi = st.sidebar.checkbox(
        "Show Projected value",
        value=False,
        key="player_analysis_show_cvi",
        help="ACP engine projection → EUR: percentile of the "
             "next-season projection (abs scale, recruit-discounted "
             "for Camp) × career-NPV age multiplier (ST 1.30, AM/WG "
             "1.25, CM 1.00, CB 0.90, FB 0.85), through the "
             "fee-calibrated CVI→EUR curve, capped at €500k.",
    )
    sort_by_cvi = False
    if show_cvi:
        sort_by_cvi = st.sidebar.checkbox(
            "Sort by Projected value",
            value=False,
            key="player_analysis_sort_by_cvi",
            help="Replace the Rating-based sort with a Projected-value sort.",
        )

    # Pre-compute age column for the full filtered pool — used by
    # the same-age peer computation inside _build_template_table
    # AND by CVI's age-value lookup.
    _has_age = (not analysis_player_details_df.empty
                 and 'birthDate' in analysis_player_details_df.columns)
    if show_cvi and _has_age:
        _filtered_age = filtered_df['playerId'].map(
            lambda pid: _calculate_age(analysis_player_details_df.loc[pid, 'birthDate'])
            if pid in analysis_player_details_df.index else None
        )
        filtered_df = filtered_df.assign(_age=_filtered_age.values)

    # Pre-compute CVI columns once for the full filtered_df. The
    # helper does its own position-grouped percentile internally;
    # we slice the result per-template inside _build_template_table.
    if show_cvi:
        try:
            _age_map = (filtered_df.set_index('playerId')['_age'].to_dict()
                         if '_age' in filtered_df.columns else {})
            # Map player → competitionId. selected_comp_ids is the
            # league filter active in the sidebar; when one league
            # is selected, every visible player belongs to it.
            # Otherwise fall back to competition_for_season.
            if selected_comp_ids and len(selected_comp_ids) == 1:
                _the_comp = int(selected_comp_ids[0])
                _comp_lookup = lambda _pid: _the_comp
            elif 'seasonId' in filtered_df.columns:
                _season_to_comp = {
                    sid: competition_for_season(sid)
                    for sid in filtered_df['seasonId'].dropna().unique()
                }
                _ssid_map = filtered_df.set_index('playerId')['seasonId'].to_dict()
                _comp_lookup = lambda pid: _season_to_comp.get(_ssid_map.get(pid))
            else:
                _comp_lookup = lambda _pid: None
            # Build empirical-Bayes prior lookup so each player's
            # season perf is shrunk toward THEIR OWN career prior
            # (not generic 40). Single perf_table build covers
            # every player in filtered_df → cheap bulk lookup.
            _bulk_prior_lookup = None
            try:
                _bulk_pt = _build_player_season_perf_table(
                    load_gpa_values(), None,
                )
                if (_bulk_pt is not None and not _bulk_pt.empty
                        and selected_season_id is not None):
                    _bulk_prior_map = build_player_priors_lookup(
                        _bulk_pt, selected_season_id,
                    )
                    _bulk_prior_lookup = (
                        lambda pid: _bulk_prior_map.get(int(pid))
                                      if pid is not None else None
                    )
            except Exception as _prior_exc:
                print(f"[CVI prior] bulk lookup build failed: "
                       f"{type(_prior_exc).__name__}: {_prior_exc}")
            _cvi_block = compute_cvi_columns(
                filtered_df,
                age_lookup=lambda pid: _age_map.get(pid),
                comp_id_lookup=_comp_lookup,
                prior_lookup=_bulk_prior_lookup,
            )
            if not _cvi_block.empty:
                filtered_df = pd.concat(
                    [filtered_df.reset_index(drop=True),
                     _cvi_block.reset_index(drop=True)],
                    axis=1,
                )
        except Exception as _cvi_exc:
            import traceback as _tb
            _tb_str = _tb.format_exc()
            # Dtype diagnostic for the input frame — the most likely
            # source of comparison errors is a mixed-type column.
            _dtype_lines = []
            try:
                for _c in ('primaryPosition', 'totalMinutes',
                            'Total Value', 'playerId', 'seasonId',
                            'competitionId', '_age'):
                    if _c in filtered_df.columns:
                        _samp = filtered_df[_c].dropna().head(3).tolist()
                        _dtype_lines.append(
                            f"  {_c:<18} dtype={filtered_df[_c].dtype} "
                            f"sample={_samp}")
                _score_cols = [c for c in filtered_df.columns
                                if c.endswith('_Score')][:5]
                for _c in _score_cols:
                    _samp = filtered_df[_c].dropna().head(3).tolist()
                    _dtype_lines.append(
                        f"  {_c:<18} dtype={filtered_df[_c].dtype} "
                        f"sample={_samp}")
            except Exception:
                pass
            _diag = "\n".join(_dtype_lines)
            st.sidebar.warning(f"CVI compute failed: "
                                f"{type(_cvi_exc).__name__}: {_cvi_exc}")
            # Full traceback + dtype diagnostic to BOTH sidebar
            # (always visible) and server-side stdout. Don't gate
            # on an expander — Streamlit's expander_state can hide
            # the message on some deploys.
            _full_diag = (f"{_tb_str}\n"
                            f"---- input column diagnostics ----\n"
                            f"{_diag}")
            print(f"[CVI ERROR] {type(_cvi_exc).__name__}: {_cvi_exc}\n"
                   f"{_full_diag}")
            st.sidebar.code(_full_diag, language='python')
            show_cvi = False
            sort_by_cvi = False

    # --- Helper: build a display table for a given template ---
    def _build_template_table(template_name, source_df, n_players, compact=False):
        """Build a display DataFrame for a template. Returns (display_df, player_ids) or (None, [])."""
        positions_in_group = POSITION_GROUPS.get(template_name, [])
        score_col = f"{template_name}_Score"
        if score_col not in source_df.columns:
            return None, []

        if analysis_pos_played_active:
            tdf = source_df
        else:
            tdf = source_df[source_df['primaryPosition'].isin(positions_in_group)]

        if tdf.empty:
            return None, []

        _sort_col = score_col

        # Overview defaults to ACP PROJECTION order (Lucas) —
        # bespoke template score still drives the per-template views
        # and remains sortable via Individual Metric mode. GK
        # templates (no engine coverage) keep the score sort.
        if (compact and 'ACP Projection (abs)' in tdf.columns
                and tdf['ACP Projection (abs)'].notna().any()):
            _sort_col = 'ACP Projection (abs)'

        # Projected-value sort override.
        if show_cvi and sort_by_cvi and 'Engine Value EUR' in tdf.columns:
            _sort_col = 'Engine Value EUR'

        sorted_tdf = tdf.sort_values(by=_sort_col, ascending=False, na_position='last').head(n_players)

        if compact:
            # Overview mode (Lucas): ACP Projection first, then ACP
            # Rating, REPLACING the bespoke template score. GK
            # templates fall back to the bespoke score.
            _eng_over = [c for c in ('ACP Projection (abs)', 'ACP Rating')
                         if c in sorted_tdf.columns
                         and sorted_tdf[c].notna().any()]
            if _eng_over:
                cols = ['playerName', 'teamName', 'primaryPosition',
                        'totalMinutes'] + _eng_over
            else:
                cols = ['playerName', 'teamName', 'primaryPosition',
                        'totalMinutes', score_col]
        else:
            # Template-specific mode: include all weighted metrics (weight > 0) sorted by weight desc
            template_weights = WEIGHTS.get(template_name, {})
            weighted_metrics = sorted(
                [(m, w) for m, w in template_weights.items() if w > 0],
                key=lambda x: x[1], reverse=True
            )
            metric_cols = [m for m, _ in weighted_metrics if m in sorted_tdf.columns]
            _eng_cols = [c for c in ('ACP Rating', 'ACP Projection (abs)')
                         if c in sorted_tdf.columns]
            cols = (['playerName', 'teamName', 'primaryPosition', 'totalMinutes',
                      score_col] + _eng_cols + metric_cols)

        cols = [c for c in cols if c in sorted_tdf.columns]
        display = sorted_tdf[cols].copy()
        _ren = {
            'playerName': 'Player',
            'teamName': 'Team',
            'primaryPosition': 'Position',
            'totalMinutes': 'Minutes',
            score_col: 'Rating'
        }
        if compact and 'ACP Rating' in cols:
            # engine columns take the headline names in Overview
            _ren.update({'ACP Projection (abs)': 'Projection',
                          'ACP Rating': 'Rating'})
        display = display.rename(columns=_ren)
        if 'Rating' in display.columns:
            display['Rating'] = pd.to_numeric(display['Rating'], errors='coerce').round(1)
        if 'Projection' in display.columns:
            display['Projection'] = pd.to_numeric(display['Projection'], errors='coerce').round(1)
        display['Minutes'] = display['Minutes'].astype(int)

        # Observed role + style columns (full mode only — the compact
        # Overview pivot stays narrow). Searchable via the dataframe
        # toolbar; the sidebar filters subset the population upstream.
        if not compact and 'Position' in display.columns:
            _rs_idx = display.columns.get_loc('Position') + 1
            display.insert(_rs_idx, 'Role',
                           [(_obs_role_map.get(int(p), '—') if pd.notna(p)
                             else '—') for p in sorted_tdf['playerId']])
            display.insert(_rs_idx + 1, 'Style',
                           [(_an_style_map.get(int(p), '—') if pd.notna(p)
                             else '—') for p in sorted_tdf['playerId']])

        # Projected value: insert next to Rating in both compact
        # and full modes. EUR computed from CVI × position mult ×
        # Camp penalty, capped at €500k. In full mode also surface
        # the Trajectory flag (perf - same-age-position median).
        # Projected Value column — ENGINE value (legacy CVI→EUR
        # removed per Lucas 2026-06-12). Computed centrally in
        # load_player_engine(); merged in as 'Engine Value EUR'.
        if show_cvi and 'Engine Value EUR' in sorted_tdf.columns:
            pv_vals = pd.Series(sorted_tdf['Engine Value EUR'].values,
                                 index=display.index)
            pv_display = [
                (f"€{int(v):,}" if v is not None and pd.notna(v) else '')
                for v in pv_vals
            ]
            _r_idx = display.columns.get_loc('Rating')
            display.insert(_r_idx + 1, 'Projected Value', pv_display)

        # Add Pos. Minutes
        if analysis_pos_played_active and 'posMinutes' in sorted_tdf.columns:
            display.insert(display.columns.get_loc('Minutes') + 1, 'Pos. Minutes', sorted_tdf['posMinutes'].astype(int).values)

        # Add Age
        if not analysis_player_details_df.empty and 'birthDate' in analysis_player_details_df.columns:
            age_pos = display.columns.get_loc('Pos. Minutes') + 1 if 'Pos. Minutes' in display.columns else display.columns.get_loc('Minutes') + 1
            display.insert(age_pos, 'Age', sorted_tdf['playerId'].map(
                lambda pid: _calculate_age(analysis_player_details_df.loc[pid, 'birthDate']) if pid in analysis_player_details_df.index else None
            ).apply(lambda x: round(x, 1) if isinstance(x, float) else None))

        # Round metric columns
        for col in display.columns:
            if pd.api.types.is_numeric_dtype(display[col]) and col not in ['Minutes', 'Rating', 'Pos. Minutes', 'Age', 'Rank']:
                decimals = 3 if col in THOUSANDTHS_METRICS else (0 if col in WHOLE_NUMBER_METRICS else 2)
                display[col] = display[col].round(decimals)

        display.insert(0, 'Rank', range(1, len(display) + 1))
        return display, sorted_tdf['playerId'].tolist()

    # --- Helper: handle row selection from a dataframe ---
    def _handle_row_selection(selection, player_ids):
        if selection and selection.selection and selection.selection.rows:
            selected_row_idx = selection.selection.rows[0]
            if selected_row_idx < len(player_ids):
                st.session_state.selected_player_id = player_ids[selected_row_idx]
                st.session_state.nav_to_profile = True
                st.session_state.nav_season_id = selected_season_id
                st.session_state.nav_has_season = True
                st.rerun()

    # --- 3. Display based on selected view ---
    if _selected_view == "Overview":
        # --- Engine-role board (shown ABOVE the template Overview) ---
        # Best players per OBSERVED engine role (the 6 data-derived
        # clusters), ranked by ACP Projection, with Minutes + Age.
        #
        # CONSISTENCY: the engine ROLE is the ONLY thing taken from
        # player_engine. Age, Min and Proj are read from the SAME scoped
        # stats frame (filtered_df) + player_details that the template table
        # below uses — so every shared column is identical across the two
        # tables. In particular: Age = age TODAY (_calculate_age), Min =
        # totalMinutes (TRUE lineup minutes). We deliberately do NOT use the
        # engine's `age` (as-of-season, ~5.5 mo stale) or `mins_played`
        # (event-derived undercount) for display.
        _eng_role_df, _ = load_player_engine()
        if (_eng_role_df is not None and not _eng_role_df.empty
                and filtered_df is not None and not filtered_df.empty):
            # role per player = their MOST COMMON ACP engine role across ALL
            # seasons and BOTH leagues (Lucas 2026-06-24) — computed by the
            # shared get_career_engine_role_map() helper, which the sidebar
            # role filter and the template-table Role column also use, so
            # a role filter keeps exactly the players this board files
            # under that role.
            _role_map_d = get_career_engine_role_map()
            _role_map = pd.DataFrame(
                {'playerId': list(_role_map_d.keys()),
                 'role': list(_role_map_d.values())})
            # join role onto the scoped, minutes-filtered stats pool — the
            # board now inherits filtered_df's totalMinutes + ACP columns
            _rb = filtered_df.merge(_role_map, on='playerId', how='inner')
            # canonical age TODAY (same lookup the template table uses)
            if (not analysis_player_details_df.empty
                    and 'birthDate' in analysis_player_details_df.columns):
                _rb_age = _rb['playerId'].map(
                    lambda pid: _calculate_age(analysis_player_details_df.loc[pid, 'birthDate'])
                    if pid in analysis_player_details_df.index else None)
                _rb = _rb.assign(_age_disp=_rb_age.values)
            else:
                _rb = _rb.assign(_age_disp=None)
            # Projection is the headline. Rank by it and show only players
            # who HAVE one (current + recent-lapsed), so inactive historical
            # players — whose face-value rating would otherwise outrank the
            # mean-shrunk projections — don't pollute the board. A purely
            # past-season scope carries no projections, so there we fall
            # back to rating and relabel the column honestly.
            # .get on a missing column returns None -> pd.to_numeric gives a
            # SCALAR nan whose .notna() crashes; normalise to a Series first
            _proj_series = _rb['ACP Projection (abs)'] \
                if 'ACP Projection (abs)' in _rb.columns \
                else pd.Series(np.nan, index=_rb.index)
            _proj_num = pd.to_numeric(_proj_series, errors='coerce')
            if _proj_num.notna().any():
                _rb = _rb[_proj_num.notna()].copy()
                _metric_label, _metric_col = 'Proj', 'ACP Projection (abs)'
            else:
                _metric_label, _metric_col = 'Rating', 'ACP Rating (abs)'
            if _metric_col in _rb.columns:
                _rb['_rankval'] = pd.to_numeric(_rb[_metric_col], errors='coerce')
            else:
                _rb['_rankval'] = np.nan

            # --- style per board player (scope-aware, descriptive) ------
            # Highest-minutes style row within the scoped seasons; falls
            # back to the player's most-played season overall so a purely
            # historical scope still shows a style.
            _styles_all = load_styles()
            _style_map = {}
            if _styles_all is not None and not _styles_all.empty:
                _sc = _styles_all
                if active_season_ids is not None:
                    _ssids = [int(s) for s in (active_season_ids
                              if isinstance(active_season_ids, (list, tuple, set))
                              else [active_season_ids])]
                    _sc_scope = _sc[_sc['seasonId'].isin(_ssids)]
                    _sc = _sc_scope if not _sc_scope.empty else _sc
                _sc = _sc.sort_values('mins_played').drop_duplicates(
                    'playerId', keep='last')
                _style_map = dict(zip(_sc['playerId'], _sc['style']))
            _rb['_style'] = _rb['playerId'].map(_style_map)

            # --- style filter selectbox ---------------------------------
            _style_opts = ['All styles'] + sorted(
                {s for s in _rb['_style'].dropna().unique()})
            _sel_style = 'All styles'
            if len(_style_opts) > 1:
                _sel_style = st.selectbox(
                    "Filter board by style", _style_opts, index=0,
                    key="role_board_style_filter",
                    help="Descriptive tendency-derived archetype (never in "
                         "the rating). Filters the board to one style.")
                if _sel_style != 'All styles':
                    _rb = _rb[_rb['_style'] == _sel_style].copy()

            _ENGINE_ROLE_ORDER = ['Striker', 'Wide Attacker', 'Advanced Midfielder',
                                   'Deep Midfielder', 'Wide Defender', 'Central Defender']
            _role_sub = ['Player', _metric_label, 'Min', 'Age', 'Style']
            _role_cols = {}
            for _role in _ENGINE_ROLE_ORDER:
                _rsub = _rb[_rb['role'] == _role].sort_values('_rankval', ascending=False).head(num_players)
                if _rsub.empty:
                    continue
                _rows = []
                for _, _r in _rsub.iterrows():
                    _mval = pd.to_numeric(_r.get(_metric_col), errors='coerce')
                    _age = _r.get('_age_disp')
                    _age = float(_age) if isinstance(_age, (int, float)) and pd.notna(_age) else None
                    _mins = pd.to_numeric(_r.get('totalMinutes'), errors='coerce')
                    _stl = _r.get('_style')
                    _rows.append((
                        _r.get('playerName', ''),
                        (round(float(_mval), 1) if pd.notna(_mval) else ''),
                        (int(_mins) if pd.notna(_mins) else 0),
                        (round(_age, 1) if _age is not None else ''),
                        (str(_stl) if _stl is not None and pd.notna(_stl) else '—'),
                    ))
                _role_cols[_role] = _rows

            if _role_cols:
                _max_r = max(len(v) for v in _role_cols.values())
                _ctups, _cdata = [], {}
                _present_roles = [r for r in _ENGINE_ROLE_ORDER if r in _role_cols]
                for _role in _present_roles:
                    for _s in _role_sub:
                        _ctups.append((_role, _s)); _cdata[(_role, _s)] = []
                for _i in range(_max_r):
                    for _role in _present_roles:
                        _rws = _role_cols[_role]
                        if _i < len(_rws):
                            for _s, _v in zip(_role_sub, _rws[_i]):
                                _cdata[(_role, _s)].append(_v)
                        else:
                            for _s in _role_sub:
                                _cdata[(_role, _s)].append('')
                _erole_df = pd.DataFrame(_cdata, columns=pd.MultiIndex.from_tuples(_ctups))
                _erole_df.index = range(1, len(_erole_df) + 1)
                _erole_df.index.name = 'Rank'
                st.subheader("Best Players by Role")
                with st.expander("ℹ️ How these ratings work"):
                    st.markdown(RATINGS_EXPLAINER_MD)
                if _metric_label == 'Proj':
                    st.caption(
                        "Observed ACP engine roles (data-derived from playing patterns), "
                        "ranked by ACP Projection. **Proj** = projected level next season "
                        "(absolute / cross-league scale). **Age** (current) and **Min** "
                        "(total minutes in scope) match the table below. **Style** = the "
                        "player's tendency-derived archetype (descriptive, never in the "
                        "rating). Only players with a live projection appear."
                    )
                else:
                    st.caption(
                        "Observed ACP engine roles (data-derived from playing patterns), "
                        "ranked by ACP **Rating** (absolute scale) — the selected season is "
                        "historical, so no forward projection exists. **Age** is current; "
                        "**Min** is total minutes in scope. **Style** = tendency-derived "
                        "archetype (descriptive, never in the rating)."
                    )
                st.dataframe(_erole_df, use_container_width=True)
                st.markdown("---")
            elif _sel_style != 'All styles':
                st.subheader("Best Players by Role")
                st.info(f"No players match the style “{_sel_style}” in this "
                        f"scope. Clear the style filter to see the full board.")
                st.markdown("---")

        # Build wide pivot table: each template is a column group with Player, Team, Minutes, Rating
        _OVERVIEW_ORDER = [
            'Forwards', 'Attacking Mids / Wingers', 'Central Midfielders',
            'Full Backs', 'Center Backs', 'Goalkeepers',
        ]
        # Collect per-template data as lists aligned by rank.
        # When Projected Value is on, append it as a 5th sub-column
        # per template (already EUR-formatted by _build_template_table).
        _sub_cols = ['Player', 'Team', 'Min', 'Proj', 'Rating']
        if show_cvi:
            _sub_cols.append('Proj. Value')
        template_columns = {}
        for group_name in _OVERVIEW_ORDER:
            group_templates = _TEMPLATE_GROUPS.get(group_name, [])
            for tmpl in [t for t in group_templates if t in POSITION_GROUPS]:
                display_df, _ = _build_template_table(tmpl, filtered_df, num_players, compact=True)
                if display_df is not None and not display_df.empty:
                    rows = []
                    for _, row in display_df.iterrows():
                        _proj_v = row.get('Projection')
                        _rat_v = row.get('Rating')
                        tup = [
                            row.get('Player', ''),
                            row.get('Team', ''),
                            int(row.get('Minutes', 0)),
                            (round(float(_proj_v), 1)
                             if _proj_v is not None and pd.notna(_proj_v) else ''),
                            (round(float(_rat_v), 1)
                             if _rat_v is not None and pd.notna(_rat_v) else ''),
                        ]
                        if show_cvi:
                            _pv = row.get('Projected Value', '')
                            tup.append(_pv if _pv else '')
                        rows.append(tuple(tup))
                    template_columns[tmpl] = rows

        if template_columns:
            max_rows = max(len(v) for v in template_columns.values())
            # Build MultiIndex columns DataFrame
            col_tuples = []
            data_dict = {}
            for tmpl in template_columns:
                for sub in _sub_cols:
                    col_tuples.append((tmpl, sub))
                    data_dict[(tmpl, sub)] = []

            for rank_idx in range(max_rows):
                for tmpl in template_columns:
                    rows = template_columns[tmpl]
                    if rank_idx < len(rows):
                        tup = rows[rank_idx]
                        for sub, val in zip(_sub_cols, tup):
                            data_dict[(tmpl, sub)].append(val)
                    else:
                        for sub in _sub_cols:
                            data_dict[(tmpl, sub)].append('')

            multi_idx = pd.MultiIndex.from_tuples(col_tuples)
            overview_df = pd.DataFrame(data_dict, columns=multi_idx)
            overview_df.index = range(1, len(overview_df) + 1)
            overview_df.index.name = 'Rank'

            st.subheader("Player Overview — Template Roles")
            st.caption(
                "Bespoke scouting templates (Shadow Striker, Mobile Striker, "
                "Creative Winger, …) — same players, scored against each role's "
                "weighting. Proj/Rating columns are the ACP engine values."
            )
            if show_cvi:
                st.caption(
                    "🟩 Projected Value = CVI → EUR mapping "
                    "(2.5 × CVI^2.5 × position multiplier, capped at €500k). "
                    "Calibrated against the 27 reported transfer fees."
                    + (" Sort is by Projected Value." if sort_by_cvi else "")
                )
            st.dataframe(overview_df, use_container_width=True)
        else:
            st.warning("No players match current filters.")

    elif _selected_view == "Peer Scatter":
        # --- Peer Scatter: any metric vs any metric, full peer cloud ---
        _SC_SET_PIECE = ['Set Piece Value', 'Corner Value',
                         'Free Kick Value', 'Throw-In Value',
                         'xASP', 'xTSP']
        _sc_categories = {
            "Output": OUTPUT_METRICS,
            "Passing": PASSING_METRICS,
            "Defensive": DEFENSIVE_METRICS,
            "Defensive Responsibility (DefR)": DEFR_DISPLAY_METRICS,
            "Dribbling": DRIBBLING_METRICS,
            "Goalkeeping": GOALKEEPING_METRICS,
            "Set Pieces": _SC_SET_PIECE,
            "ACP Index": ENGINE_DISPLAY_METRICS,
            "Template Ratings": sorted(
                [c for c in filtered_df.columns if c.endswith('_Score')]),
        }

        def _sc_metric_picker(axis_label, default_cat, default_metric):
            cat = st.sidebar.selectbox(
                f"{axis_label} category:", list(_sc_categories.keys()),
                index=list(_sc_categories.keys()).index(default_cat),
                key=f"peer_scatter_cat_{axis_label}")
            opts = [m for m in _sc_categories[cat]
                    if m in filtered_df.columns]
            if not opts:
                return None
            idx = opts.index(default_metric) if default_metric in opts else 0
            return st.sidebar.selectbox(
                f"{axis_label} metric:", opts, index=idx,
                key=f"peer_scatter_metric_{axis_label}")

        _sc_x = _sc_metric_picker("X", "Defensive", "Interceptions")
        _sc_y = _sc_metric_picker("Y", "Output", "npxG")

        _sc_positions = sorted(
            filtered_df['primaryPosition'].dropna().unique().tolist())
        _sc_pos_filter = st.sidebar.multiselect(
            "Filter by Position (optional):", _sc_positions, default=[],
            key="peer_scatter_positions")

        if _sc_x is None or _sc_y is None:
            st.warning("No metrics available for the selected categories.")
            st.stop()
        if _sc_x == _sc_y:
            st.info("Pick two different metrics to compare.")
            st.stop()

        _sc_df = filtered_df
        if _sc_pos_filter:
            _sc_df = _sc_df[_sc_df['primaryPosition'].isin(_sc_pos_filter)]
        _sc_df = _sc_df.dropna(subset=[_sc_x, _sc_y])
        if len(_sc_df) < 5:
            st.warning("Not enough players with both metrics under the "
                       "current filters.")
            st.stop()

        st.subheader(f"{_sc_y} vs {_sc_x} (per 90)")
        _sc_c1, _sc_c2 = st.columns([2, 1])
        with _sc_c1:
            _sc_hl = st.selectbox(
                "Highlight player:",
                ["(none)"] + sorted(_sc_df['playerName'].dropna().unique().tolist()),
                key="peer_scatter_highlight")
        with _sc_c2:
            _sc_fit = st.checkbox("Show line of best fit", value=True,
                                  key="peer_scatter_fit")

        _sx = pd.to_numeric(_sc_df[_sc_x], errors='coerce')
        _sy = pd.to_numeric(_sc_df[_sc_y], errors='coerce')
        _sc_fig = go.Figure()
        _sc_hover = [
            f"<b>{r.playerName}</b> · {r.teamName}<br>"
            f"{r.primaryPosition} · {int(r.totalMinutes)}'<br>"
            f"{_sc_x}: {x:.3f}<br>{_sc_y}: {y:.3f}"
            for r, x, y in zip(_sc_df.itertuples(), _sx, _sy)]
        _sc_fig.add_trace(go.Scatter(
            x=_sx, y=_sy, mode='markers',
            marker=dict(size=8, color='rgba(110,125,118,0.45)',
                        line=dict(color='rgba(255,255,255,0.6)', width=0.5)),
            text=_sc_hover, hovertemplate='%{text}<extra></extra>',
            name='peers'))
        _sc_fig.add_vline(x=float(_sx.median()), line_dash='dot',
                          line_color='rgba(128,128,128,0.5)')
        _sc_fig.add_hline(y=float(_sy.median()), line_dash='dot',
                          line_color='rgba(128,128,128,0.5)')
        if _sc_fit and len(_sc_df) >= 3:
            _b, _a = np.polyfit(_sx, _sy, 1)
            _xline = np.linspace(_sx.min(), _sx.max(), 50)
            _r = float(np.corrcoef(_sx, _sy)[0, 1])
            _sc_fig.add_trace(go.Scatter(
                x=_xline, y=_b * _xline + _a, mode='lines',
                line=dict(color='#3987e5', width=2, dash='dash'),
                hoverinfo='skip', name=f'fit (r = {_r:+.2f})'))
        if _sc_hl != "(none)":
            _hrow = _sc_df[_sc_df['playerName'] == _sc_hl]
            _sc_fig.add_trace(go.Scatter(
                x=pd.to_numeric(_hrow[_sc_x], errors='coerce'),
                y=pd.to_numeric(_hrow[_sc_y], errors='coerce'),
                mode='markers+text',
                marker=dict(size=14, color='#2aa876',
                            line=dict(color='white', width=2)),
                text=[_sc_hl] * len(_hrow), textposition='top center',
                textfont=dict(size=12),
                hoverinfo='skip', name=_sc_hl))
        _sc_fig.update_layout(
            height=640, showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.01),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(title=_sc_x, gridcolor='rgba(128,128,128,0.15)',
                       zeroline=False),
            yaxis=dict(title=_sc_y, gridcolor='rgba(128,128,128,0.15)',
                       zeroline=False))
        st.plotly_chart(_sc_fig, use_container_width=True)
        _sc_notes = []
        if _sc_x in INVERT_METRICS or _sc_y in INVERT_METRICS:
            _inv = [m for m in (_sc_x, _sc_y) if m in INVERT_METRICS]
            _sc_notes.append(f"lower is better for: {', '.join(_inv)}")
        _sc_notes.append("dotted lines = peer medians")
        _sc_notes.append(f"{len(_sc_df)} players shown "
                         "(sidebar minutes/age filters apply)")
        st.caption(" · ".join(_sc_notes))

    elif _selected_view == "Individual Metric":
        # --- Individual Metric mode (preserved from original) ---
        # Set-piece metrics — the four GPA action-value columns
        # ("Set Piece Value" = sum of the other three) plus the
        # set-piece-only flavors of xA and xT. xASP/xTSP also live
        # in OUTPUT_METRICS but are duplicated here so users can
        # find every set-piece metric in one place.
        SET_PIECE_METRICS = [
            'Set Piece Value', 'Corner Value', 'Free Kick Value',
            'Throw-In Value', 'xASP', 'xTSP',
        ]
        metric_categories = {
            "Output": OUTPUT_METRICS,
            "Passing": PASSING_METRICS,
            "Defensive": DEFENSIVE_METRICS,
            "Defensive Responsibility (DefR)": DEFR_DISPLAY_METRICS,
            "Dribbling": DRIBBLING_METRICS,
            "Goalkeeping": GOALKEEPING_METRICS,
            "Set Pieces": SET_PIECE_METRICS,
            "ACP Index": ENGINE_DISPLAY_METRICS,
            # bespoke template ratings stay sortable here (Lucas) —
            # they left the Overview headline but remain a metric
            "Template Ratings": sorted(
                [c for c in filtered_df.columns if c.endswith('_Score')]),
        }

        selected_category = st.sidebar.selectbox(
            "Metric Category:",
            list(metric_categories.keys()),
            key="player_analysis_metric_category"
        )

        available_metrics = [m for m in metric_categories[selected_category]
                           if m in filtered_df.columns]
        if not available_metrics:
            st.warning(f"No metrics available for {selected_category} category.")
            st.stop()

        selected_metric = st.sidebar.selectbox(
            "Select Metric:",
            available_metrics,
            key="player_analysis_metric_v2"
        )

        all_positions = sorted(filtered_df['primaryPosition'].dropna().unique().tolist())
        position_filter = st.sidebar.multiselect(
            "Filter by Position (optional):",
            all_positions,
            default=[],
            key="player_analysis_position_filter"
        )

        if position_filter:
            metric_filtered_df = filtered_df[filtered_df['primaryPosition'].isin(position_filter)]
        else:
            metric_filtered_df = filtered_df

        if metric_filtered_df.empty:
            st.warning("No players found with current filters.")
            st.stop()

        # Individual Metric mode intentionally does NOT apply the
        # age-adjusted / cross-tier-translation toggles — those are
        # for overall profile ratings only (Overview + per-template).
        _sort_ascending = selected_metric in INVERT_METRICS
        sorted_df = metric_filtered_df.sort_values(by=selected_metric, ascending=_sort_ascending).head(num_players)

        st.subheader(f"Top Players by {selected_metric} (per 90)")

        related_metrics = []
        if selected_metric in OUTPUT_METRICS:
            related_metrics = ['Goals', 'xG', 'npxG', 'Shots', 'Assists', 'xAOP']
        elif selected_metric in PASSING_METRICS:
            related_metrics = ['Passes', 'Passes successful %', 'Progressive Passes', 'xTOP']
        elif selected_metric in DEFENSIVE_METRICS:
            related_metrics = ['Interceptions', 'Recoveries', 'Defensive duels', 'Aerial duels']
        elif selected_metric in DRIBBLING_METRICS:
            related_metrics = ['Dribbles', 'Dribbles successful %', 'Progressive runs']
        elif selected_metric in GOALKEEPING_METRICS:
            related_metrics = ['goalsPrevented', 'savePercentage', 'exits']
        elif selected_metric in SET_PIECE_METRICS:
            # Show the other set-piece value flavors + the set-
            # piece xA/xT pair so you can see, e.g., which corner
            # specialists are also creating high-xT deliveries.
            related_metrics = ['Set Piece Value', 'Corner Value',
                                'Free Kick Value', 'Throw-In Value',
                                'xASP', 'xTSP']

        related_metrics = [m for m in related_metrics if m in sorted_df.columns and m != selected_metric][:4]

        display_cols = ['playerName', 'teamName', 'primaryPosition', 'totalMinutes', selected_metric] + related_metrics
        display_cols = [c for c in display_cols if c in sorted_df.columns]

        display_df = sorted_df[display_cols].copy()
        display_df = display_df.rename(columns={
            'playerName': 'Player',
            'teamName': 'Team',
            'primaryPosition': 'Position',
            'totalMinutes': 'Minutes'
        })
        display_df['Minutes'] = display_df['Minutes'].astype(int)

        if analysis_pos_played_active and 'posMinutes' in sorted_df.columns:
            display_df.insert(display_df.columns.get_loc('Minutes') + 1, 'Pos. Minutes', sorted_df['posMinutes'].astype(int).values)

        if not analysis_player_details_df.empty and 'birthDate' in analysis_player_details_df.columns:
            age_col_pos = display_df.columns.get_loc('Pos. Minutes') + 1 if 'Pos. Minutes' in display_df.columns else display_df.columns.get_loc('Minutes') + 1
            display_df.insert(age_col_pos, 'Age', sorted_df['playerId'].map(
                lambda pid: _calculate_age(analysis_player_details_df.loc[pid, 'birthDate']) if pid in analysis_player_details_df.index else None
            ).apply(lambda x: round(x, 1) if isinstance(x, float) else None))

        # Individual Metric no longer applies the cross-tier / age
        # toggles — those are now scoped to overall ratings only.
        for col in display_df.columns:
            if pd.api.types.is_numeric_dtype(display_df[col]) and col not in ['Minutes', 'Pos. Minutes', 'Age']:
                decimals = 3 if col in THOUSANDTHS_METRICS else 2
                display_df[col] = display_df[col].round(decimals)

        display_df.insert(0, 'Rank', range(1, len(display_df) + 1))
        player_ids = sorted_df['playerId'].tolist()

        st.caption("Click on a row to view that player's profile")
        selection = st.dataframe(
            display_df.set_index('Rank'),
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row",
            key="individual_metric_table"
        )
        _handle_row_selection(selection, player_ids)

    else:
        # --- Template-specific view ---
        selected_template = _selected_view
        st.subheader(f"Top {selected_template}s by Rating")

        display_df, player_ids = _build_template_table(selected_template, filtered_df, num_players, compact=False)

        if display_df is not None and not display_df.empty:
            if show_cvi:
                st.caption(
                    "🟩 CVI = composite scout-facing value · 'Traj vs age' = "
                    "performance vs same-position-same-age median "
                    "(e.g. '+25' = 25pt ahead of age peer median). "
                    "Currently uses placeholder parameters; will be calibrated "
                    "against scraped market values."
                    + (" Sort is by CVI." if sort_by_cvi else "")
                )
            st.caption("Click on a row to view that player's profile")
            selection = st.dataframe(
                display_df.set_index('Rank'),
                use_container_width=True,
                on_select="rerun",
                selection_mode="single-row",
                key="template_detail_table"
            )
            _handle_row_selection(selection, player_ids)

            # Display weight reference table below
            template_weights = WEIGHTS.get(selected_template, {})
            weighted_items = sorted(
                [(m, w) for m, w in template_weights.items() if w > 0],
                key=lambda x: x[1], reverse=True
            )
            if weighted_items:
                with st.expander("Template Weights", expanded=False):
                    weight_df = pd.DataFrame(weighted_items, columns=['Metric', 'Weight'])
                    weight_df['Weight'] = weight_df['Weight'].apply(lambda w: f"{w:.1f}")
                    st.dataframe(weight_df, use_container_width=True, hide_index=True, column_config=auto_column_config(weight_df))
        else:
            st.warning(f"No players found for {selected_template} template with current filters.")

    # ===== Distribution violins (appended below either Overview or
    # template table). Skips Individual Metric because that view is
    # already a per-metric leaderboard. =====
    if _selected_view not in ("Individual Metric", "Peer Scatter"):
        st.markdown("---")

        from plotly.subplots import make_subplots as _make_subplots

        def _pos_to_group_full(pos):
            """primaryPosition → top-level position-group label
            (matches _TEMPLATE_GROUPS keys)."""
            if pos is None or pd.isna(pos):
                return None
            p = str(pos)
            if p == 'GK': return 'Goalkeepers'
            if p in ('CB','LCB','RCB','LCB3','RCB3'): return 'Center Backs'
            if p in ('LB','RB','LB5','RB5','LWB','RWB'): return 'Full Backs'
            if p in ('CMF','LCMF','RCMF','LCMF3','RCMF3',
                     'DMF','LDMF','RDMF'): return 'Central Midfielders'
            if p in ('AMF','LAMF','RAMF','LMF','RMF',
                     'LW','RW','LWF','RWF'): return 'Attacking Mids / Wingers'
            if p in ('CF','SS'): return 'Forwards'
            return None

        def _best_fit_in_group(row, group_roles):
            """Best Role_Score among the templates in this group."""
            vals = []
            for r in group_roles:
                col = f"{r}_Score"
                if col in row.index:
                    v = row.get(col)
                    if v is not None and not pd.isna(v):
                        vals.append(float(v))
            return max(vals) if vals else None

        def _add_strip(fig, row_idx, col_idx, values, names, teams,
                        metric_label, y_lo, y_hi, scaled_w, sid):
            """Append a violin + jittered dots to the (row, col)
            subplot. Hover on a dot shows player + team + value."""
            if len(values) < 3:
                return
            # 1) Violin shape
            fig.add_trace(go.Violin(
                x=np.zeros(len(values)),
                y=values,
                points=False,
                box_visible=False,
                meanline_visible=False,
                side='both',
                width=scaled_w,
                line_color='rgba(80,80,80,0.55)',
                fillcolor='rgba(140,140,140,0.18)',
                showlegend=False,
                hoverinfo='skip',
                name='',
            ), row=row_idx, col=col_idx)
            # 2) Colored jittered dots with rich hover text
            _rng = np.random.default_rng(seed=int(sid) & 0xFFFFFFFF)
            _half = max(0.06, scaled_w / 2 - 0.04)
            _jit = _rng.uniform(-_half, _half, size=len(values))
            _custom = np.array(list(zip(names, teams)), dtype=object)
            fig.add_trace(go.Scatter(
                x=_jit, y=values,
                mode='markers',
                marker=dict(
                    size=6,
                    color=values,
                    colorscale='RdYlGn',
                    cmin=y_lo, cmax=y_hi,
                    opacity=0.7,
                    line=dict(width=0),
                    showscale=False,
                ),
                customdata=_custom,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Team: %{customdata[1]}<br>"
                    f"{metric_label}: " + "%{y:.3f}<extra></extra>"
                ),
                showlegend=False,
                name='',
            ), row=row_idx, col=col_idx)
            fig.update_xaxes(
                type='linear', showticklabels=False, zeroline=False,
                range=[-0.5, 0.5], row=row_idx, col=col_idx,
            )

        def _build_panel_for_group(group_name, source_df, metric_key):
            """Return (values, names, teams) for the panel at this
            position group. metric_key ∈ {'action_v', 'best_fit',
            'acp_rating', 'acp_proj'}."""
            grp_pop = source_df[
                source_df['primaryPosition'].map(_pos_to_group_full)
                == group_name
            ]
            if grp_pop.empty:
                return np.array([]), [], []
            if metric_key in ('action_v', 'acp_rating', 'acp_proj'):
                if metric_key == 'action_v':
                    col = next((c for c in ('Total Value', 'total_v_per_90')
                                  if c in grp_pop.columns), None)
                elif metric_key == 'acp_rating':
                    col = 'ACP Rating' if 'ACP Rating' in grp_pop.columns else None
                else:
                    col = ('ACP Projection (abs)'
                           if 'ACP Projection (abs)' in grp_pop.columns else None)
                if col is None:
                    return np.array([]), [], []
                sub = grp_pop[grp_pop[col].notna()]
                return (sub[col].astype(float).values,
                         sub.get('playerName', sub.index).astype(str).tolist(),
                         sub.get('teamName', pd.Series([''] * len(sub))).fillna('').astype(str).tolist())
            # best_fit
            group_roles = _TEMPLATE_GROUPS.get(group_name, [])
            if not group_roles:
                return np.array([]), [], []
            _scores = grp_pop.apply(_best_fit_in_group, axis=1,
                                      args=(group_roles,))
            mask = _scores.notna()
            vals = _scores[mask].astype(float).values
            names = grp_pop.loc[mask].get('playerName',
                        grp_pop.loc[mask].index).astype(str).tolist()
            teams = grp_pop.loc[mask].get(
                'teamName', pd.Series([''] * mask.sum())
            ).fillna('').astype(str).tolist()
            return vals, names, teams

        # --- View-dependent rendering ---
        if _selected_view == "Overview":
            st.subheader("Distribution by Position Group")
            _viz_metric = st.radio(
                "Distribution metric:",
                ["ACP Projection", "ACP Rating", "Action V/90", "Best-fit Rating"],
                horizontal=True,
                key="player_analysis_viz_metric_overview",
            )
            _metric_key = {'Action V/90': 'action_v',
                            'Best-fit Rating': 'best_fit',
                            'ACP Rating': 'acp_rating',
                            'ACP Projection': 'acp_proj'}[_viz_metric]

            _groups = ['Goalkeepers', 'Center Backs', 'Full Backs',
                        'Central Midfielders',
                        'Attacking Mids / Wingers', 'Forwards']
            _panels = []
            for g in _groups:
                vals, names, teams = _build_panel_for_group(
                    g, filtered_df, _metric_key
                )
                if len(vals) >= 5:
                    _panels.append({
                        'label': f"{g}<br><span style='font-size:0.85em;color:#777'>n={len(vals)}</span>",
                        'group': g,
                        'values': vals,
                        'names': names,
                        'teams': teams,
                    })

            if not _panels:
                st.caption(
                    f"No {_viz_metric} data available for any position "
                    f"group in the current selection."
                )
            else:
                _pop_concat = np.concatenate([p['values'] for p in _panels])
                _y_lo = float(np.nanmin(_pop_concat))
                _y_hi = float(np.nanmax(_pop_concat))
                _y_pad = 0.05 * (_y_hi - _y_lo or 1.0)
                _max_n = max(len(p['values']) for p in _panels) or 1
                _fig = _make_subplots(
                    rows=1, cols=len(_panels),
                    shared_yaxes=True,
                    subplot_titles=[p['label'] for p in _panels],
                    horizontal_spacing=0.01,
                )
                for _i, _p in enumerate(_panels, start=1):
                    _scaled_w = 0.85 * (len(_p['values']) / _max_n) ** 0.5
                    # Use a stable per-group seed for jitter (so layout
                    # doesn't dance on rerun).
                    _seed = abs(hash(_p['group'])) & 0xFFFFFFFF
                    _add_strip(
                        _fig, 1, _i,
                        _p['values'], _p['names'], _p['teams'],
                        _viz_metric, _y_lo, _y_hi, _scaled_w, _seed,
                    )
                _fig.update_yaxes(range=[_y_lo - _y_pad, _y_hi + _y_pad])
                _fig.update_layout(
                    title=(f"{_viz_metric} distribution by position "
                            f"group · ≥{min_minutes_filter:.0f} min"),
                    height=460,
                    margin=dict(t=70, b=30, l=40, r=20),
                    showlegend=False,
                )
                for _ann in _fig['layout']['annotations']:
                    _ann['font'] = dict(size=11)
                st.plotly_chart(_fig, use_container_width=True)

        else:
            # Template view — one panel per season for that template's
            # position group. We compute role scores for each season
            # (cached), filter to the template's eligible positions,
            # and use THAT template's Role_Score as the y-value.
            _template = _selected_view
            _eligible_positions = POSITION_GROUPS.get(_template, [])
            if not _eligible_positions:
                st.caption(f"No position list defined for template '{_template}'.")
            else:
                st.subheader(f"{_template} distribution across seasons")
                _viz_metric = st.radio(
                    "Distribution metric:",
                    ["ACP Projection", "ACP Rating", "Action V/90", "Best-fit Rating"],
                    horizontal=True,
                    key="player_analysis_viz_metric_template",
                )

                # Iterate over every season we have data for, sorted
                # chronologically by parsed start year.
                def _season_start_year(label):
                    try: return int(str(label).split('/')[0])
                    except (ValueError, AttributeError, IndexError): return None
                _season_panels = []
                _all_sids = sorted(
                    [int(s) for s in SEASON_ID_MAP.keys()],
                    key=lambda s: (
                        _season_start_year(SEASON_ID_MAP.get(s, '')) or 0,
                        competition_for_season(s) or 0,
                        s,
                    ),
                )

                with st.spinner(f"Computing {_template} distributions across "
                                 f"{len(_all_sids)} seasons…"):
                    for _sid in _all_sids:
                        _evs = get_season_events(raw_events_df, [_sid])
                        _mins = player_minutes_data.get(_sid)
                        if _evs.empty or _mins is None or _mins.empty:
                            continue
                        if _viz_metric == 'Action V/90':
                            # GPA path: filter by season + position
                            if 'load_gpa_values' in globals():
                                _gpa_all = load_gpa_values()
                            else:
                                _gpa_all = None
                            if _gpa_all is None or _gpa_all.empty:
                                continue
                            _val_col = next((c for c in ('Total Value',
                                              'total_v_per_90')
                                              if c in _gpa_all.columns), None)
                            if _val_col is None:
                                continue
                            _sub = _gpa_all[
                                (_gpa_all['seasonId'] == _sid)
                                & (_gpa_all.get('mins_played', 0)
                                   >= min_minutes_filter)
                                & (_gpa_all.get('position', '').astype(str)
                                   .isin(_eligible_positions))
                            ]
                            _sub = _sub[_sub[_val_col].notna()]
                            vals = _sub[_val_col].astype(float).values
                            names = _sub.get('name', pd.Series([''] * len(_sub))).astype(str).tolist()
                            # GPA doesn't carry teamName — leave blank
                            teams = [''] * len(_sub)
                        elif _viz_metric in ('ACP Rating', 'ACP Projection'):
                            # Engine path: per-season engine values,
                            # position-filtered via the GPA table
                            # (engine rows don't carry raw position
                            # codes). Projections exist only for the
                            # current + lapsed seasons, so older
                            # panels skip naturally (<5 rows).
                            _eng_all, _ = load_player_engine()
                            _gpa_all = (load_gpa_values()
                                         if 'load_gpa_values' in globals() else None)
                            if _eng_all.empty or _gpa_all is None or _gpa_all.empty:
                                continue
                            _ecol = ('acp_rating' if _viz_metric == 'ACP Rating'
                                      else 'projection_abs')
                            _sub = _gpa_all[
                                (_gpa_all['seasonId'] == _sid)
                                & (_gpa_all.get('mins_played', 0)
                                   >= min_minutes_filter)
                                & (_gpa_all.get('position', '').astype(str)
                                   .isin(_eligible_positions))
                            ][['playerId', 'name']].merge(
                                _eng_all[_eng_all['seasonId'] == _sid][
                                    ['playerId', _ecol]],
                                on='playerId', how='inner')
                            _sub = _sub[_sub[_ecol].notna()]
                            vals = _sub[_ecol].astype(float).values
                            names = _sub['name'].astype(str).tolist()
                            teams = [''] * len(_sub)
                        else:
                            # Best-fit (this template's specific Role_Score)
                            _stats = calculate_all_player_stats(
                                _evs, _mins, season_id=_sid
                            )
                            if _stats.empty:
                                continue
                            _scored = calculate_player_percentiles_and_scores(
                                _stats, POSITION_GROUPS, WEIGHTS, INVERT_METRICS,
                                min_minutes=int(min_minutes_filter),
                                season_id=_sid,
                            )
                            if _scored.empty:
                                continue
                            _score_col = f"{_template}_Score"
                            if _score_col not in _scored.columns:
                                continue
                            _sub = _scored[
                                _scored['primaryPosition'].isin(_eligible_positions)
                                & _scored[_score_col].notna()
                                & (_scored.get('totalMinutes', 0)
                                   >= min_minutes_filter)
                            ]
                            vals = _sub[_score_col].astype(float).values
                            names = _sub.get('playerName',
                                        pd.Series([''] * len(_sub))).astype(str).tolist()
                            teams = _sub.get('teamName',
                                        pd.Series([''] * len(_sub))).fillna('').astype(str).tolist()

                        if len(vals) < 5:
                            continue
                        _comp = competition_for_season(_sid)
                        _comp_short = ('L3' if _comp == 43324
                                        else 'CP' if _comp == 702
                                        else (COMPETITIONS.get(_comp, {}).get('name', '') or '')[:6])
                        _season_panels.append({
                            'sid': _sid,
                            'label': (f"{SEASON_ID_MAP.get(_sid, str(_sid))}<br>"
                                       f"<span style='font-size:0.85em;color:#777'>"
                                       f"{_comp_short} · n={len(vals)}</span>"),
                            'values': vals,
                            'names': names,
                            'teams': teams,
                        })

                if not _season_panels:
                    st.caption(
                        f"No {_viz_metric} data available across seasons "
                        f"for the {_template} template."
                    )
                else:
                    _pop_concat = np.concatenate([p['values'] for p in _season_panels])
                    _y_lo = float(np.nanmin(_pop_concat))
                    _y_hi = float(np.nanmax(_pop_concat))
                    _y_pad = 0.05 * (_y_hi - _y_lo or 1.0)
                    _max_n = max(len(p['values']) for p in _season_panels) or 1
                    _fig = _make_subplots(
                        rows=1, cols=len(_season_panels),
                        shared_yaxes=True,
                        subplot_titles=[p['label'] for p in _season_panels],
                        horizontal_spacing=0.01,
                    )
                    for _i, _p in enumerate(_season_panels, start=1):
                        _scaled_w = 0.85 * (len(_p['values']) / _max_n) ** 0.5
                        _add_strip(
                            _fig, 1, _i,
                            _p['values'], _p['names'], _p['teams'],
                            _viz_metric, _y_lo, _y_hi, _scaled_w, _p['sid'],
                        )
                    _fig.update_yaxes(range=[_y_lo - _y_pad, _y_hi + _y_pad])
                    _fig.update_layout(
                        title=(f"{_template} · {_viz_metric} distribution "
                                f"by season · ≥{min_minutes_filter:.0f} min"),
                        height=460,
                        margin=dict(t=70, b=30, l=40, r=20),
                        showlegend=False,
                    )
                    for _ann in _fig['layout']['annotations']:
                        _ann['font'] = dict(size=11)
                    st.plotly_chart(_fig, use_container_width=True)

    # Always keep the expander open on the Player Analysis page so the
    # Cached ZIPs section, debug info, and any errors are unmissable.
    _bulk_expander_open = True
    with st.sidebar.expander("📥 Bulk Export Radars", expanded=_bulk_expander_open):
        _bulk_groups_default = list(_TEMPLATE_GROUPS.keys())
        _bulk_groups = st.multiselect(
            "Position groups:",
            _bulk_groups_default,
            default=_bulk_groups_default,
            key="bulk_export_groups",
        )
        _bulk_mode_label = st.radio(
            "Radar style:",
            ["Percentile", "Raw (mean ± 2σ)"],
            index=1,
            key="bulk_export_mode",
        )
        _bulk_min_mins = st.number_input(
            "Min minutes:",
            min_value=0,
            max_value=int(max_minutes) if max_minutes else 5000,
            value=int(min_minutes_filter),
            step=45,
            key="bulk_export_min_mins",
            help="Default uses the Minimum Minutes Played slider above."
        )
        _bulk_generate = st.button("Generate ZIP", key="bulk_export_btn", use_container_width=True)

        if _bulk_generate:
            # Resolve the multi-select group labels to raw position codes.
            _bulk_raw_codes = set()
            for _grp in _bulk_groups:
                for _role in _TEMPLATE_GROUPS.get(_grp, []):
                    if _role in POSITION_GROUPS:
                        _bulk_raw_codes.update(POSITION_GROUPS[_role])

            _export_df = player_stats_with_scores_df[
                (pd.to_numeric(player_stats_with_scores_df['totalMinutes'], errors='coerce').fillna(0) >= _bulk_min_mins) &
                (player_stats_with_scores_df['primaryPosition'].isin(_bulk_raw_codes))
            ].copy()

            if _export_df.empty:
                st.warning("No players match the selection.")
            else:
                _n_total = len(_export_df)
                _progress = st.progress(0.0, text=f"Rendering 0/{_n_total} radars…")

                def _on_progress(i, n, name, resumed=0):
                    if resumed:
                        _progress.progress(
                            i / max(n, 1),
                            text=f"Rendering {i}/{n} (resumed {resumed}): {name}"
                        )
                    else:
                        _progress.progress(
                            i / max(n, 1),
                            text=f"Rendering {i}/{n}: {name}"
                        )

                _radar_mode = 'raw' if _bulk_mode_label.startswith("Raw") else 'percentile'
                _season_lbl = SEASON_ID_MAP.get(selected_season_id, 'All Seasons') if selected_season_id else 'All Seasons'

                # Write each PNG into a per-render directory; the
                # download ZIP is built lazily at click time. Sentinel
                # meta.pkl is written first so even crashed runs
                # appear in the Cached list.
                _cache_key = _bulk_cache_key(_season_lbl, _bulk_groups, _radar_mode, _bulk_min_mins)
                _render_dir = _bulk_render_dir(_cache_key)
                _meta_path = _bulk_meta_path(_cache_key)
                try:
                    _os.makedirs(_render_dir, exist_ok=True)
                    with open(_meta_path, 'wb') as _f:
                        _pickle.dump({
                            'status': 'running',
                            'rendered': 0,
                            'skipped': [],
                            'label': f"{_season_lbl}__{_radar_mode}",
                            'season': _season_lbl,
                            'mode': _radar_mode,
                            'groups': list(_bulk_groups),
                            'min_mins': int(_bulk_min_mins),
                            'started_at': _time.time(),
                        }, _f)
                except Exception as _sentinel_exc:
                    st.warning(f"⚠️ Could not write sentinel meta: "
                               f"{type(_sentinel_exc).__name__}: {_sentinel_exc}")

                try:
                    _result_path, _rendered, _skipped, _resumed = bulk_export_radars(
                        _export_df,
                        player_stats_with_scores_df,
                        radar_mode=_radar_mode,
                        season_label=_season_lbl,
                        progress_cb=_on_progress,
                        output_path=_render_dir,
                    )
                    try:
                        with open(_meta_path, 'wb') as _f:
                            _pickle.dump({
                                'status': 'complete',
                                'rendered': _rendered,
                                'skipped': _skipped,
                                'resumed': _resumed,
                                'label': f"{_season_lbl}__{_radar_mode}",
                                'season': _season_lbl,
                                'mode': _radar_mode,
                                'groups': list(_bulk_groups),
                                'min_mins': int(_bulk_min_mins),
                            }, _f)
                    except Exception as _meta_exc:
                        st.warning(f"⚠️ Render finished but completion-meta "
                                   f"write failed: {type(_meta_exc).__name__}: "
                                   f"{_meta_exc}")
                    _progress.empty()
                    _new_count = _rendered - _resumed
                    _resume_note = (f" (resumed {_resumed}, rendered {_new_count} new)"
                                     if _resumed else "")
                    st.success(
                        f"Rendered {_rendered} radars to disk{_resume_note}"
                        + (f" · {len(_skipped)} skipped" if _skipped else "")
                        + ". Use the Prepare ZIP button below."
                    )
                except Exception as _gen_exc:
                    _progress.empty()
                    import traceback as _tb
                    st.error(f"Render failed: {type(_gen_exc).__name__}: {_gen_exc}")
                    with st.popover("Traceback (for debugging)", use_container_width=True):
                        st.code(_tb.format_exc())

        # --- Cached Renders section — always shown. ---
        st.markdown("---")
        _cached = _list_cached_renders()
        if _BULK_CACHE_ERROR:
            st.error(f"⚠️ Cache directory unusable: {_BULK_CACHE_ERROR}. "
                     f"Generated renders will not survive the page render. "
                     f"This usually means /tmp/ is not writable in this runtime.")
        if not _cached:
            st.caption("💾 No cached renders yet. Run Generate ZIP to create one.")
        else:
            st.caption(f"💾 Cached renders ({len(_cached)})")
        _now = _time.time()
        for _idx, _entry in enumerate(_cached):
            _meta = _entry['meta']
            _rd = _entry['path']
            _age = _now - _entry['mtime']
            _age_str = (f"{int(_age)}s ago" if _age < 60 else
                        f"{int(_age/60)} min ago" if _age < 3600 else
                        f"{int(_age/3600)} h ago" if _age < 86400 else
                        f"{int(_age/86400)} d ago")
            _size_mb = _entry['size'] / (1024 * 1024)
            _status = _meta.get('status', 'complete')
            _png_count = _entry['png_count']
            _season = _meta.get('season', '?')
            _mode = _meta.get('mode', '?')
            _mm = _meta.get('min_mins', '?')
            _ngroups = len(_meta.get('groups', []) or [])
            _badge = ""
            if _status == 'running':
                _badge = " · 🟡 interrupted (partial download still works)"
            elif _status == 'incomplete':
                _badge = " · 🟠 metadata missing (download still works)"
            st.markdown(
                f"**{_season}** · {_mode} · {_ngroups} groups · ≥{_mm} min{_badge}  \n"
                f"<span style='color:#888;font-size:0.85em'>{_png_count} radars · "
                f"{_size_mb:.1f} MB on disk · {_age_str}</span>",
                unsafe_allow_html=True,
            )
            # Build ZIP lazily from the directory contents. This is
            # the moment we pay the in-memory cost for the ZIP, not
            # during render. Even if the render was interrupted,
            # every PNG that made it to disk gets bundled cleanly.
            _zip_btn_key = f"bulk_export_dl_{_idx}"
            _prep_key = f"bulk_export_prep_{_idx}"
            _zip_bytes_key = f"bulk_export_bytes_{_idx}"
            _zip_fp_key = f"bulk_export_fp_{_idx}"
            # Fingerprint the on-disk state so a previously-built
            # ZIP gets invalidated when the directory has grown
            # (e.g. after a resume run). Without this the download
            # button would happily serve stale bytes.
            _current_fp = f"{_png_count}_{int(_entry['size'])}"
            _cached_fp = st.session_state.get(_zip_fp_key)
            if (_zip_bytes_key in st.session_state
                    and _cached_fp == _current_fp):
                _cached_size_mb = len(st.session_state[_zip_bytes_key]) / (1024*1024)
                st.download_button(
                    label=f"⬇️ Download ({_cached_size_mb:.0f} MB)",
                    data=st.session_state[_zip_bytes_key],
                    file_name=f"radars__{_meta.get('label', 'export')}.zip",
                    mime="application/zip",
                    key=_zip_btn_key,
                    use_container_width=True,
                )
            else:
                # If we have stale cached bytes, surface that so the
                # user understands why the prepare button is back.
                if _zip_bytes_key in st.session_state and _cached_fp:
                    try:
                        _stale_count = int(_cached_fp.split('_')[0])
                        st.caption(f"⚠️ Cached ZIP is stale "
                                   f"({_stale_count} files vs {_png_count} on disk). "
                                   f"Re-prepare to refresh.")
                    except Exception:
                        pass
                if st.button(f"📦 Prepare ZIP ({_png_count} files, ~{_size_mb:.0f} MB)",
                              key=_prep_key, use_container_width=True):
                    try:
                        with st.spinner("Building ZIP from rendered PNGs…"):
                            st.session_state[_zip_bytes_key] = _build_zip_from_dir(_rd)
                            st.session_state[_zip_fp_key] = _current_fp
                        st.rerun()
                    except Exception as _zip_exc:
                        st.error(f"ZIP build failed: "
                                 f"{type(_zip_exc).__name__}: {_zip_exc}")
            if _meta.get('skipped'):
                _sk = _meta['skipped']
                with st.popover(f"View skipped ({len(_sk)})", use_container_width=True):
                    for _name, _reason in _sk[:50]:
                        st.caption(f"• **{_name}** — {_reason}")
                    if len(_sk) > 50:
                        st.caption(f"…and {len(_sk) - 50} more")

        # --- Diagnostics: surfaces the actual on-disk state ---
        # popover instead of expander — Streamlit forbids nested expanders.
        with st.popover("🔍 Diagnostics", use_container_width=True):
            st.caption(f"Cache dir: `{_BULK_CACHE_DIR}`")
            if _BULK_CACHE_ERROR:
                st.error(f"Cache dir setup error: {_BULK_CACHE_ERROR}")
            try:
                _entries = sorted(_os.listdir(_BULK_CACHE_DIR))
                if not _entries:
                    st.caption("Directory is empty.")
                else:
                    _rows = []
                    for _en in _entries:
                        _ep = _os.path.join(_BULK_CACHE_DIR, _en)
                        try:
                            if _os.path.isdir(_ep):
                                _png = [f for f in _os.listdir(_ep)
                                         if f.endswith('.png')]
                                _sz = sum(_os.path.getsize(_os.path.join(_ep, f))
                                           for f in _os.listdir(_ep))
                                _mt = _os.path.getmtime(_ep)
                                _rows.append({
                                    'entry': _en + '/',
                                    'type': 'dir',
                                    'pngs': len(_png),
                                    'size_MB': f"{_sz/(1024*1024):.2f}",
                                    'mtime': _time.strftime('%Y-%m-%d %H:%M:%S',
                                                            _time.localtime(_mt)),
                                })
                            else:
                                _sz = _os.path.getsize(_ep)
                                _mt = _os.path.getmtime(_ep)
                                _rows.append({
                                    'entry': _en,
                                    'type': 'file',
                                    'pngs': 0,
                                    'size_MB': f"{_sz/(1024*1024):.2f}",
                                    'mtime': _time.strftime('%Y-%m-%d %H:%M:%S',
                                                            _time.localtime(_mt)),
                                })
                        except Exception:
                            _rows.append({'entry': _en, 'type': '?',
                                          'pngs': 0, 'size_MB': '?', 'mtime': '?'})
                    st.dataframe(pd.DataFrame(_rows),
                                 use_container_width=True, hide_index=True)
            except Exception as _diag_exc:
                st.error(f"Cannot list cache dir: "
                         f"{type(_diag_exc).__name__}: {_diag_exc}")
            # Disk usage info — useful if /tmp/ is filling up
            try:
                import shutil as _shutil
                _du = _shutil.disk_usage(_BULK_CACHE_DIR if _os.path.exists(_BULK_CACHE_DIR) else '/tmp')
                st.caption(
                    f"`/tmp/` disk: total {_du.total/(1024**3):.1f} GB · "
                    f"used {_du.used/(1024**3):.1f} GB · "
                    f"free {_du.free/(1024**3):.1f} GB"
                )
            except Exception as _du_exc:
                st.caption(f"disk_usage error: {_du_exc}")
            # Process RAM, if psutil is available — early-warning for OOM
            try:
                import psutil as _psutil
                _proc = _psutil.Process()
                _rss = _proc.memory_info().rss / (1024**3)
                _vmem = _psutil.virtual_memory()
                st.caption(
                    f"Process RSS: {_rss:.2f} GB · "
                    f"system RAM: {_vmem.used/(1024**3):.1f} GB used / "
                    f"{_vmem.total/(1024**3):.1f} GB total "
                    f"({_vmem.percent:.0f}%)"
                )
            except Exception:
                pass
