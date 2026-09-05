"""Home — the page a coach or director opens on Monday.

Everything here is about OUR club in its current league season: where we
stand, how the last match went, who is next, who is in form, and how fresh
the data is. Every card links into the analysis page that has the detail.

Reads the running app module for data and helpers (see views/__init__.py).
"""
import os
import pickle
import sys

import pandas as pd
import streamlit as st

import context_bar
import navigation
import theme


def _app():
    return sys.modules['__main__']


@st.cache_data(ttl=3600, show_spinner=False)
def _load_simulation(app_dir):
    """season_simulation.pkl from simulate_season.py, or None."""
    try:
        with open(os.path.join(app_dir, 'season_simulation.pkl'), 'rb') as fh:
            return pickle.load(fh)
    except (FileNotFoundError, OSError, pickle.UnpicklingError):
        return None


def _score_for(row, team):
    """(goals for, goals against) of `team` in a fixture row, or None."""
    try:
        home_goals, away_goals = (int(x) for x in str(row['score']).split('-'))
    except (ValueError, TypeError):
        return None
    return (home_goals, away_goals) if row['homeTeamName'] == team else (away_goals, home_goals)


def _outcome(gf, ga):
    return 'W' if gf > ga else 'D' if gf == ga else 'L'


def _fmt_date(value):
    ts = pd.to_datetime(value, errors='coerce')
    return ts.strftime('%d %b %Y') if pd.notna(ts) else '?'


def render():
    app = _app()
    OUR_TEAM = app.OUR_TEAM
    COMPETITIONS = app.COMPETITIONS
    SEASON_ID_MAP = app.SEASON_ID_MAP
    matches_summary_df = app.matches_summary_df
    raw_events_df = app.raw_events_df
    get_filtered_events = app.get_filtered_events
    next_fixture_for_team = app.next_fixture_for_team
    last_fixture_for_team = app.last_fixture_for_team
    load_player_engine = app.load_player_engine
    _is_played = app._is_played
    app_dir = os.path.dirname(os.path.abspath(app.__file__))

    # ------------------------------------------------------------------
    # Scope: our league, its current season
    # ------------------------------------------------------------------
    league_id = next((cid for cid, cfg in COMPETITIONS.items() if cfg['name'] == 'Liga 3'),
                     next(iter(COMPETITIONS)))
    league_name = COMPETITIONS[league_id]['name']
    season_id = COMPETITIONS[league_id]['current_season']
    season_label = SEASON_ID_MAP.get(season_id, '?')

    fixtures = matches_summary_df[matches_summary_df['seasonId'] == season_id].copy()
    ours = fixtures[(fixtures['homeTeamName'] == OUR_TEAM) | (fixtures['awayTeamName'] == OUR_TEAM)].copy()
    ours['_date'] = pd.to_datetime(ours['dateutc'], errors='coerce')
    ours = ours.sort_values('_date')
    played = ours[ours['score'].apply(_is_played)]
    results = [(_outcome(*gf_ga), gf_ga, row) for _, row in played.iterrows()
               if (gf_ga := _score_for(row, OUR_TEAM)) is not None]

    season_events = get_filtered_events(raw_events_df, season_id, [league_id])
    event_match_ids = set(season_events['matchId'].unique()) if season_events is not None and not season_events.empty else set()

    sim = _load_simulation(app_dir)
    sim_group = None
    if sim and sim.get('season_id') == season_id:
        sim_group = next((g for g in sim.get('groups', {}).values() if OUR_TEAM in g.get('teams', [])), None)

    # ------------------------------------------------------------------
    # Header + freshness
    # ------------------------------------------------------------------
    st.header(f"{OUR_TEAM} — {league_name} {season_label}")
    _, engine_meta = load_player_engine()
    fresh_bits = []
    if not ours.empty:
        fresh_bits.append(f"fixtures through {_fmt_date(ours['_date'].max())}")
    if event_match_ids:
        ev_dates = pd.to_datetime(fixtures[fixtures['matchId'].isin(event_match_ids)]['dateutc'], errors='coerce')
        fresh_bits.append(f"event data through {_fmt_date(ev_dates.max())}")
    else:
        fresh_bits.append("no event data for this season yet")
    if engine_meta:
        fresh_bits.append(f"engine {engine_meta.get('rating_version', '?')} through {engine_meta.get('data_through', '?')}")
    if sim:
        fresh_bits.append(f"simulation {str(sim.get('timestamp', ''))[:10]}")
    st.caption(" · ".join(fresh_bits))

    # ------------------------------------------------------------------
    # Tiles
    # ------------------------------------------------------------------
    pts = sum(3 if o == 'W' else 1 if o == 'D' else 0 for o, _, _ in results)
    gd = sum(gf - ga for _, (gf, ga), _ in results)
    form = " ".join(o for o, _, _ in results[-5:]) or "—"

    pos_txt, promo_txt, promo_help = "—", "—", "Run simulate_season.py to refresh"
    if sim_group is not None:
        standings = sim_group.get('current_standings')
        if standings is not None and OUR_TEAM in set(standings['Team']):
            row = standings[standings['Team'] == OUR_TEAM].iloc[0]
            pos_txt = f"{int(row['Pos'])} of {len(standings)}"
        promo = sim_group.get('promotion_pct', {}).get(OUR_TEAM)
        playoff = sim_group.get('playoff_pct', {}).get(OUR_TEAM)
        if promo is not None:
            promo_txt = f"{promo:.0%}"
            promo_help = (f"{playoff:.0%} to reach the promotion series" if playoff is not None else "")
            promo_help += f" · {sim.get('n_simulations', 0):,} simulations"

    t1, t2, t3, t4 = st.columns(4)
    t1.metric("League position", pos_txt, help="Current standing in our série")
    t2.metric("Points", f"{pts}", f"{len(results)} played · GD {gd:+d}", delta_color="off")
    t3.metric("Form (last 5)", form, help="Oldest to newest")
    t4.metric("Promotion", promo_txt, help=promo_help)

    # ------------------------------------------------------------------
    # Last match + next opponent
    # ------------------------------------------------------------------
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.subheader("Last match")
            if results:
                outcome, (gf, ga), row = results[-1]
                home = row['homeTeamName'] == OUR_TEAM
                opponent = row['awayTeamName'] if home else row['homeTeamName']
                st.markdown(f"**{outcome} {gf}–{ga}** vs {opponent} ({'H' if home else 'A'}) · "
                            f"GW {row.get('gameweek', '?')} · {_fmt_date(row['_date'])}")
                mid = row['matchId']
                if mid in event_match_ids:
                    match_ev = season_events[season_events['matchId'] == mid]
                    xg = match_ev.groupby('team.name', observed=True)['shot.xg'].sum()
                    st.markdown(f"xG **{xg.get(OUR_TEAM, 0):.2f}** for · **{xg.drop(OUR_TEAM, errors='ignore').sum():.2f}** against")
                else:
                    st.caption("Event data for this match not loaded yet — xG and maps appear after the next refresh.")
                if st.button("Open match report", key="home_open_match"):
                    context_bar.set_context(league_name, season_label)
                    navigation.go_to('Match Analysis', nav_match_id=int(mid))
            else:
                st.caption("No matches played yet this season.")

    with right:
        with st.container(border=True):
            st.subheader("Next opponent")
            nxt = next_fixture_for_team(matches_summary_df, season_id, OUR_TEAM)
            target = None
            if nxt:
                st.markdown(f"**{nxt['opponent']}** ({nxt['home_away']}) · GW {nxt['gameweek']} · {_fmt_date(nxt['date'])}")
                target = nxt['opponent']
            else:
                last = last_fixture_for_team(matches_summary_df, season_id, OUR_TEAM)
                st.caption("The fixture list only fills in as matches are played, so the next opponent isn't in the data yet.")
                if last:
                    st.markdown(f"Most recent opponent: **{last['opponent']}** ({last['home_away']})")
                    target = last['opponent']
            if target and st.button("Open Opposition Report", key="home_open_opp", type="primary"):
                context_bar.set_context(league_name, season_label)
                navigation.go_to('Opposition Report', opposition_report_team=target)

    # ------------------------------------------------------------------
    # Results this season
    # ------------------------------------------------------------------
    if results:
        with st.expander(f"Results this season ({len(results)})", expanded=False):
            rows = []
            for outcome, (gf, ga), row in results[::-1]:
                home = row['homeTeamName'] == OUR_TEAM
                rows.append({'GW': row.get('gameweek', ''), 'Date': _fmt_date(row['_date']),
                             'Opponent': row['awayTeamName'] if home else row['homeTeamName'],
                             'H/A': 'H' if home else 'A', 'Score': f"{gf}–{ga}", 'Result': outcome})
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # ------------------------------------------------------------------
    # Players in form (ACP Index, newest rated season for our squad)
    # ------------------------------------------------------------------
    engine_df, _ = load_player_engine()
    st.subheader("Squad — ACP Index")
    if engine_df is not None and not engine_df.empty and 'team' in engine_df.columns:
        squad = engine_df[engine_df['team'] == OUR_TEAM]
        if not squad.empty:
            rated_season = int(squad['seasonId'].max())
            latest = squad[(squad['seasonId'] == rated_season) & (squad['mins_played'] >= 300)]
            top = latest.nlargest(8, 'acp_rating')[['playerId', 'name', 'role', 'mins_played', 'acp_rating']]
            st.caption(f"Season {SEASON_ID_MAP.get(rated_season, rated_season)}, players with 300+ minutes. "
                       "Select a row to open the profile.")
            selection = st.dataframe(
                top.drop(columns=['playerId']).rename(columns={
                    'name': 'Player', 'role': 'Role', 'mins_played': 'Minutes', 'acp_rating': 'ACP Index'}),
                hide_index=True, use_container_width=True,
                on_select='rerun', selection_mode='single-row', key='home_squad_table',
                column_config={
                    'Minutes': st.column_config.NumberColumn(format='%d'),
                    'ACP Index': st.column_config.ProgressColumn(min_value=0, max_value=100, format='%.0f'),
                })
            rows_sel = getattr(getattr(selection, 'selection', None), 'rows', None) or []
            if rows_sel:
                pid = int(top['playerId'].iloc[rows_sel[0]])
                st.session_state.selected_player_id = pid
                st.session_state.nav_to_profile = True
                st.session_state.nav_season_id = rated_season
                st.session_state.nav_has_season = True
                navigation.go_to('Player Profile')
        else:
            st.caption("No rated players for our squad in the engine export yet.")
    else:
        st.caption("Player engine export not available.")

    # ------------------------------------------------------------------
    # Quick links
    # ------------------------------------------------------------------
    st.divider()
    q1, q2, q3 = st.columns(3)
    if q1.button("Team report", use_container_width=True, key="home_q_team"):
        context_bar.set_context(league_name, season_label)
        navigation.go_to('Team Analysis')
    if q2.button("League table & strength", use_container_width=True, key="home_q_league"):
        context_bar.set_context(league_name, season_label)
        navigation.go_to('League Analysis')
    if q3.button("Promotion probabilities", use_container_width=True, key="home_q_pred"):
        navigation.go_to('Match Predictor')
