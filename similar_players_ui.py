"""Similar Players — the profile section (and the Shadow Team entry point)
for the like-for-like search in models/similarity/similar_players.py.

Mirrors eur_interval_ui.py: the maths lives in the model module, this file
only renders. The section never presents a neighbour as "the replacement":
scores, shared traits and the measured validation sit together.
"""
import pandas as pd
import streamlit as st

import navigation
from league_config import all_season_id_map
from models.similarity import similar_players as sp
from models.value.eur_intervals import format_eur_short

_LEAGUE_ID = {'Liga 3': 43324, 'Campeonato': 702}
_LEAGUE_NAME = {43324: 'Liga 3', 702: 'Campeonato'}
SECTION_NAME = 'Similar Players'


def _season_label(sid):
    return all_season_id_map().get(int(sid), str(sid)) if pd.notna(sid) else '—'


def query_row(pool, player_id, selected_season_id):
    """The pool row the search starts from: the page's season when the
    player qualifies in it, else his latest qualifying season."""
    qi = None
    if isinstance(selected_season_id, int):
        qi = pool.index_of(player_id, selected_season_id)
    if qi is None:
        qi = pool.index_of(player_id)
    return qi


def profile_bridge_state(record):
    """Session keys that open a neighbour's profile in HIS league and season
    (the pool spans both leagues; the season label alone would land in the
    current league)."""
    comp = int(record['competitionId']) if pd.notna(record.get('competitionId')) else -1
    return {'selected_player_id': int(record['playerId']), 'nav_to_profile': True,
            'nav_season_id': int(record['seasonId']), 'nav_has_season': True,
            'nav_league': _LEAGUE_NAME.get(comp)}


def can_compare(query, record):
    """'Compare on radar' only when both rows are in the SAME league and
    season: the Comparison page's radars are that scope's percentiles."""
    try:
        return (int(query['competitionId']) == int(record['competitionId'])
                and int(query['seasonId']) == int(record['seasonId']))
    except (TypeError, ValueError):
        return False


def render(app, player_id, selected_season_id, key='sim', current_pos=None):
    sig = app.similarity_cache_signature()
    pool = app.load_similarity_pool(sig)
    if pool.meta.empty:
        if getattr(pool, 'error', ''):
            st.warning(f"Similar-player search is unavailable: {pool.error}")
        else:
            st.info("Similar-player search needs the per-season stats caches, and none are "
                    "available here.")
        return
    if str(current_pos or '').upper().startswith('GK'):
        st.info("Goalkeepers are not covered by the similar-player search yet: their metric "
                "set is different from outfield players'.")
        return
    qi = query_row(pool, player_id, selected_season_id)
    if qi is None:
        st.info(f"No qualifying season for this player (needs {sp.POOL_MIN_MINUTES} minutes "
                f"in one season) — similarity hidden.")
        return
    q = pool.meta.iloc[qi]
    st.caption(
        f"Profile searched: **{_season_label(q['seasonId'])}** season "
        f"({int(q['totalMinutes']):,} min at {q['teamName']}) · position bucket **{q['bucket']}** · "
        f"pool: {len(pool.meta):,} player-seasons from {pool.n_seasons} cached seasons, both leagues, "
        f"≥ {sp.POOL_MIN_MINUTES} min. Neighbours are like-for-like in what they do and how well, "
        f"not a recommendation — open a profile before judging.")

    # two control rows: seven widgets in one row wrap on a laptop width
    c = st.columns([1.0, 1.5, 1.2, 1.0])
    league = c[0].selectbox('League', ['Both', 'Liga 3', 'Campeonato'], key=f'{key}_league')
    profile = c[1].selectbox('Candidates', ['Latest season per player', 'Any player-season'],
                             key=f'{key}_profile',
                             help="'Latest season per player' gives a target list (one row per "
                                  "player, his most recent qualifying season); 'Any player-season' "
                                  "also finds 'he looked like this in 2023/24'.")
    min_minutes = c[2].select_slider('Min minutes', options=[500, 700, 900, 1200, 1500],
                                     value=500, key=f'{key}_mins')
    max_age = c[3].number_input('Max age (0 = any)', min_value=0, max_value=45, value=0, step=1,
                                key=f'{key}_age')
    c2 = st.columns([1.5, 1.2, 0.8, 1.5])
    _q_abs = pd.to_numeric(pd.Series([q.get('ACP Rating (abs)')]), errors='coerce').iloc[0]
    quality = c2[0].selectbox('Quality', ['Any level', 'Similar level (±5)', 'At least as good'],
                             key=f'{key}_quality',
                             help="On the league-ABSOLUTE ACP rating, so a Campeonato player is "
                                  "judged on the same scale as a Liga 3 one. Disabled when the "
                                  "searched player has no engine rating.")
    exclude_own = c2[1].checkbox('Exclude own club', value=True, key=f'{key}_excl')
    k = c2[2].selectbox('Show', [5, 10, 20], index=1, key=f'{key}_k')
    rmin = rmax = None
    if quality != 'Any level':
        if pd.isna(_q_abs):
            st.caption("Quality filter ignored: the searched player has no engine rating.")
        elif quality.startswith('Similar'):
            rmin, rmax = float(_q_abs) - 5.0, float(_q_abs) + 5.0
        else:
            rmin = float(_q_abs)

    ages = app.player_ages()
    nb = sp.neighbours(
        pool, int(player_id), season_id=int(q['seasonId']), k=int(k),
        profile='latest' if profile.startswith('Latest') else 'any',
        league=_LEAGUE_ID.get(league), min_minutes=int(min_minutes),
        exclude_team=(app.OUR_TEAM if exclude_own else None),
        max_age=(float(max_age) if max_age else None), ages=ages,
        rating_abs_min=rmin, rating_abs_max=rmax)
    if nb.empty:
        st.info("No candidates match these filters.")
        return
    role_map = {}
    try:
        role_map = app.get_career_engine_role_map() or {}
    except Exception:
        role_map = {}
    records = nb.to_dict('records')
    table = pd.DataFrame({
        'Rank': [r['rank'] for r in records],
        'Player': [r['playerName'] for r in records],
        'Team': [r['teamName'] for r in records],
        'Season': [_season_label(r['seasonId']) for r in records],
        'League': [_LEAGUE_NAME.get(int(r['competitionId']) if pd.notna(r['competitionId']) else -1, '—') for r in records],
        'Pos': [r['primaryPosition'] for r in records],
        'Min': [int(r['totalMinutes']) for r in records],
        'Age (today)': [(f"{ages[int(r['playerId'])]:.0f}" if int(r['playerId']) in ages else '—') for r in records],
        'Similarity': [round(float(r['similarity']), 0) for r in records],
        'Career role': [role_map.get(int(r['playerId']), '—') for r in records],
        'ACP Rating (abs)': [(round(float(r['ACP Rating (abs)']), 0) if 'ACP Rating (abs)' in r and pd.notna(r['ACP Rating (abs)']) else None) for r in records],
        'Proj. value': [(format_eur_short(r['Engine Value EUR']) if 'Engine Value EUR' in r and pd.notna(r['Engine Value EUR']) else '—') for r in records],
        'Similar because': ['; '.join(sp.explain(pool, int(r['query_row']), int(r['_row']), n_alike=2)['alike']) or '—'
                            for r in records],
    })
    event = st.dataframe(
        table, hide_index=True, use_container_width=True, key=f'{key}_table',
        on_select='rerun', selection_mode='single-row',
        column_config={
            'Similarity': st.column_config.ProgressColumn(
                'Similarity', min_value=0, max_value=100, format='%.0f',
                help='100 = identical profile, 50 = a typical pair of this position bucket, 0 = far apart.'),
            'ACP Rating (abs)': st.column_config.NumberColumn(
                'ACP Rating (abs)', format='%.0f',
                help='League-absolute engine rating — comparable across Liga 3 and Campeonato.'),
            'Age (today)': st.column_config.TextColumn(
                'Age (today)', help="The player's age now, whatever season the row shows."),
            'Career role': st.column_config.TextColumn(
                'Career role', help='Engine role over the whole career (minutes-weighted), not that season.'),
        })
    st.caption("Click a row for the shared traits and the differences, then open the profile "
               "or compare on the radar.")
    sel = event.selection.rows if event and event.selection else []
    if sel:
        r = records[sel[0]]
        ex = sp.explain(pool, int(r['query_row']), int(r['_row']), n_alike=4, n_differ=3)
        st.markdown(f"**{r['playerName']}** ({r['teamName']}, {_season_label(r['seasonId'])}) — "
                    f"similarity {float(r['similarity']):.0f}, closer than "
                    f"{min(99.9, float(r['closer_than_pct'])):.1f}% of {q['bucket']} pairs.")
        col_a, col_b = st.columns(2)
        col_a.markdown("**Alike on:** " + (", ".join(ex['alike']) if ex['alike'] else "nothing stands out"))
        col_b.markdown("**Differs on:** " + (", ".join(ex['differs']) if ex['differs'] else "nothing large"))
        b1, b2 = st.columns([1, 1])
        if b1.button("Open profile", key=f'{key}_open'):
            for k_, v_ in profile_bridge_state(r).items():
                st.session_state[k_] = v_
            st.rerun()
        if can_compare(q, r):
            if b2.button("Compare on radar", key=f'{key}_compare'):
                navigation.go_to('Player Comparison', compare_seed_a=int(player_id),
                                 compare_seed_b=int(r['playerId']))
        else:
            b2.caption("Compare on radar needs both players in the same league and season "
                       "(the radar percentiles are that scope's).")

    with st.expander("How similarity is measured", expanded=False):
        v = app.similarity_validation(sig) or {}
        st.markdown(
            "One vector per player-season: 36 per-90 style metrics (what he does), 8 value "
            "metrics at half weight (how well) and 6 pitch-occupancy scalars (where). Each is a "
            "percentile within the position bucket over the whole pool, so 'both top 20% for "
            "counter-pressing' is literally what the distance used; similarity = 100 − 50 × "
            "distance ÷ the bucket's median pair distance.")
        sm, ra, sa = v.get('self_match') or {}, v.get('role_agreement') or {}, v.get('style_agreement') or {}
        if sm.get('top_k_rate') is not None:
            m = st.columns(3)
            m[0].metric("Finds a player's other season in his top 10",
                        f"{sm['top_k_rate']:.0%}", help=f"Chance level {sm.get('chance_top_k', 0):.1%}; "
                        f"median rank {sm.get('median_rank')} among the other seasons of his bucket. "
                        f"n = {sm.get('n')} player-seasons with another season in the same bucket.")
            if ra.get('top_k_share') is not None:
                m[1].metric("Top 10 sharing the engine role", f"{ra['top_k_share']:.0%}",
                            help=f"Pool share (chance) {ra.get('chance', 0):.0%}. The role label is not in the "
                                 "vector, though the role model is clustered from similar pitch-occupancy inputs.")
            if sa.get('top_k_share') is not None:
                m[2].metric("Top 10 sharing the style label", f"{sa['top_k_share']:.0%}",
                            help=f"Pool share (chance) {sa.get('chance', 0):.0%}. The style label is not in the vector.")
        _self_rate = (f"{sm['top_k_rate']:.0%}" if sm.get('top_k_rate') is not None else 'well under half')
        st.caption(
            f"Pool: {v.get('n_rows', len(pool.meta)):,} player-seasons, {v.get('n_players', 0):,} players, "
            f"{v.get('n_seasons', pool.n_seasons)} seasons; spatial scalars cover "
            f"{100 * v.get('spatial_coverage', pool.spatial_coverage):.0f}% of rows. Caveats: the position "
            f"bucket is the first position a player was seen in, so a converted player sits with his "
            f"old peers; Liga 3 and Campeonato rows are pooled without a tier adjustment (use the "
            f"League filter and the rating column); neighbour sets are noisy at this sample size — a "
            f"player's own other season makes his top 10 {_self_rate} "
            f"of the time — so read the shared traits, not the rank.")
