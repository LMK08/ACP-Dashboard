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


def remaining_legs(team, group_teams, played):
    """The legs of the group's double round-robin that `team` has not played:
    [(opponent, 'H' | 'A'), ...], opponents in alphabetical order, the home
    leg before the away leg. `played` is an iterable of (home, away) names.

    This is what the data KNOWS about what is left. The match feed lists a
    match only once it is played, so neither the date nor the order of these
    legs is in it: "who is next" is not derivable. The list is exact when
    `group_teams` is the group of the current phase and `played` holds that
    phase's legs only (phase_legs_played): the caller checks both."""
    done = set(played)
    legs = []
    for opponent in sorted(t for t in group_teams if t != team):
        if (team, opponent) not in done:
            legs.append((opponent, 'H'))
        if (opponent, team) not in done:
            legs.append((opponent, 'A'))
    return legs


def phase_legs_played(season_matches, team):
    """((home, away) legs `team` has played in the CURRENT phase, whether the
    second phase has started).

    `season_matches` are the season's PLAYED matches, every team. The phases
    are told apart exactly as simulate_season.py does it: the roundId with
    the most matches is the first phase (the two séries), any other roundId
    is the second (promotion / maintenance series). Once a second-phase
    match exists, only second-phase matches count as legs of "this phase":
    a first-phase meeting with a club that is in our second-phase group too
    is not a leg of that group's round-robin."""
    if season_matches is None or len(season_matches) == 0 or 'roundId' not in season_matches.columns:
        return [], False
    first_phase_round = season_matches.groupby('roundId').size().idxmax()
    in_second = season_matches['roundId'] != first_phase_round
    started = bool(in_second.any())
    phase = season_matches[in_second] if started else season_matches
    ours = phase[(phase['homeTeamName'] == team) | (phase['awayTeamName'] == team)]
    return list(zip(ours['homeTeamName'], ours['awayTeamName'])), started


def table_window(standings, team, size=5):
    """`size` consecutive rows of `standings` with `team` as near the middle
    as the ends of the table allow (the whole table when it is shorter)."""
    teams = list(standings['Team'])
    if team not in teams or len(teams) <= size:
        return standings
    at = teams.index(team)
    start = max(0, min(at - size // 2, len(teams) - size))
    return standings.iloc[start:start + size]


# W / D / L chips: the letter carries the meaning, the colour repeats it
# (theme.FOCUS = ours / good, theme.OPPONENT = against / bad, grey = neither).
_FORM_CHIP = {'W': theme.FOCUS, 'D': '#6f6a63', 'L': theme.OPPONENT}

_TILE_CSS = """<style>
/* a tile that is not an st.metric (the form chips) but sits in a row of them:
   same card, same height as its neighbours */
.acp-tile { background:#fff; border:1px solid #EBE7E1; border-radius:8px; padding:1.4rem;
            box-shadow:0 1px 3px rgba(0,0,0,0.04); min-height:100%; box-sizing:border-box; }
.acp-tile-label { color:#9a948d; font-weight:500; font-size:0.82rem; text-transform:uppercase;
                  letter-spacing:0.4px; }
.acp-form { display:flex; gap:0.3rem; margin:0.45rem 0 0.35rem; flex-wrap:wrap; }
.acp-form span { display:inline-flex; align-items:center; justify-content:center; width:1.75rem;
                 height:1.75rem; border-radius:5px; color:#fff; font-weight:700; font-size:0.95rem; }
.acp-tile-note { color:#5c5650; font-size:0.8rem; }
[data-testid="stColumn"] > [data-testid="stVerticalBlock"]:has(.acp-tile),
[data-testid="stElementContainer"]:has(.acp-tile),
[data-testid="stElementContainer"]:has(.acp-tile) > [data-testid="stMarkdown"],
[data-testid="stMarkdownContainer"]:has(> .acp-tile) { height:100%; }
@media (max-width: 1200px) { .acp-tile { padding:1rem 0.75rem; } }
</style>"""


def form_tile_html(outcomes):
    """The "Form (last 5)" tile: one chip per result, oldest first."""
    import html
    chips = ''.join(
        f'<span style="background:{_FORM_CHIP.get(o, "#6f6a63")};" title="{html.escape(tip)}">{html.escape(o)}</span>'
        for o, tip in outcomes) or '<span style="background:none;color:#1a1a1a;">—</span>'
    return (_TILE_CSS + '<div class="acp-tile"><div class="acp-tile-label">Form (last 5)</div>'
            f'<div class="acp-form">{chips}</div><div class="acp-tile-note">Oldest to newest</div></div>')


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
    crest_data_uri = app.crest_data_uri
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
    # which legs of the CURRENT phase we have played, and whether the season
    # is in its second phase (see phase_legs_played)
    _phase_played, _second_phase = phase_legs_played(
        fixtures[fixtures['score'].apply(_is_played)], OUR_TEAM)

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
    # Freshness, in two plain dates under the title (the line used to list
    # five data and model dates). The simulation and its Dixon-Coles fit are
    # cited under the tiles, next to the promotion number they explain; the
    # engine's version and date are in the sidebar ("About the data").
    fresh_bits = []
    if not ours.empty:
        fresh_bits.append(f"Fixtures through {_fmt_date(ours['_date'].max())}")
    if event_match_ids:
        ev_dates = pd.to_datetime(fixtures[fixtures['matchId'].isin(event_match_ids)]['dateutc'], errors='coerce')
        fresh_bits.append(f"match data through {_fmt_date(ev_dates.max())}")
    else:
        fresh_bits.append("no match data for this season yet")
    st.caption(" · ".join(fresh_bits))

    # ------------------------------------------------------------------
    # Tiles
    # ------------------------------------------------------------------
    pts = sum(3 if o == 'W' else 1 if o == 'D' else 0 for o, _, _ in results)
    gd = sum(gf - ga for _, (gf, ga), _ in results)

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
            promo_help += f" · {sim.get('n_simulations', 0):,} simulations of the remaining fixtures"
            _mm = sim.get('model') or {}
            if str(_mm.get('name', '')).startswith('dixon_coles'):
                promo_help += (f", each scoreline drawn from the Dixon-Coles model fitted through "
                               f"{_mm.get('asof', '?')} — the same model as the Match Predictor's "
                               f"scoreline forecast and strength ratings")

    # keyed: theme.metric_note_css hides the arrow in the Points card's note
    with st.container(key="home_tiles"):
        t1, t2, t3, t4 = st.columns(4)
    t1.metric("League position", pos_txt, help="Current standing in our série")
    t2.metric("Points", f"{pts}", f"{len(results)} played · goal difference {gd:+d}", delta_color="off")
    # coloured W / D / L chips, oldest first (hover a chip for the match)
    _form_chips = []
    for _o, (_gf, _ga), _row in results[-5:]:
        _home = _row['homeTeamName'] == OUR_TEAM
        _opp = _row['awayTeamName'] if _home else _row['homeTeamName']
        _form_chips.append((_o, f"{_o} {_gf}–{_ga} vs {_opp} ({'H' if _home else 'A'}), GW {_row.get('gameweek', '?')}"))
    t3.markdown(form_tile_html(_form_chips), unsafe_allow_html=True)
    t4.metric("Promotion", promo_txt, help=promo_help)
    # Under the whole row (the cards keep one height): what the promotion
    # number rests on. Both pages that show these odds cite the fit
    # (tests/test_smoke_pages.py::test_promotion_odds_cite_the_scoreline_model).
    if sim:
        _mm = sim.get('model') or {}
        _fit_txt = (f" (Dixon-Coles fit through {_mm.get('asof', '?')})"
                    if str(_mm.get('name', '')).startswith('dixon_coles') else '')
        _sim_date = str(sim.get('timestamp', ''))[:10]
        if sim_group is not None and promo_txt != "—":
            st.caption(
                f"Promotion: the share of {sim.get('n_simulations', 0):,} simulated runs of the "
                f"remaining fixtures in which we go up, run {_sim_date}{_fit_txt}.")
        else:
            # a simulation file exists but holds no number for us this season
            st.caption(f"Season simulation run {_sim_date}{_fit_txt}; it has no promotion "
                       "number for this season yet.")

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

        # The table around us: five rows of our série, with crests. First
        # phase only: in the second phase `current_standings` leaves out the
        # FPF bonus points the maintenance series carries, so its points and
        # order would differ from the official table.
        _standings = sim_group.get('current_standings') if sim_group is not None else None
        if (_standings is not None and OUR_TEAM in set(_standings['Team'])
                and not _second_phase):
            with st.container(border=True):
                st.subheader("Table")
                _window = table_window(_standings, OUR_TEAM)[['Pos', 'Team', 'P', 'Pts', 'GD']].copy()
                _window.insert(1, 'Crest', [crest_data_uri(t) for t in _window['Team']])
                _styled = _window.style.apply(
                    lambda r: ['font-weight: 700; background-color: #F0ECE4' if r['Team'] == OUR_TEAM else ''
                               for _ in r], axis=1)
                st.dataframe(
                    _styled, hide_index=True, width="stretch",
                    column_config={
                        # no fixed widths on the numbers: at half the page five
                        # 'small' columns pushed Pts out of view
                        'Crest': st.column_config.ImageColumn('', width='small'),
                        'P': st.column_config.NumberColumn('P', help='Played'),
                        'Pts': st.column_config.NumberColumn('Pts', help='Points'),
                        'GD': st.column_config.NumberColumn('GD', help='Goal difference', format='%+d'),
                    })
                st.caption(f"Places {int(_window['Pos'].min())} to {int(_window['Pos'].max())} of "
                           f"{len(_standings)} in our série, as of the last simulation run. "
                           "The full table is on League Analysis.")

    with right:
        with st.container(border=True):
            st.subheader("Next opponent")
            nxt = next_fixture_for_team(matches_summary_df, season_id, OUR_TEAM)
            target = None
            if nxt:
                st.markdown(f"**{nxt['opponent']}** ({nxt['home_away']}) · GW {nxt['gameweek']} · {_fmt_date(nxt['date'])}")
                target = nxt['opponent']
            else:
                # The feed lists a match only once it is played: the date and
                # order of the next one are not in the data. What IS known
                # exactly is who is still to be played in this phase.
                _group = list(sim_group.get('teams', [])) if sim_group is not None else []
                # the group and the phase's matches must describe the same
                # phase (a simulation file older than the first second-phase
                # match would not): every leg played has to be inside the group
                _consistent = all(h in _group and a in _group for h, a in _phase_played)
                _legs = remaining_legs(OUR_TEAM, _group, _phase_played) if _group and _consistent else []
                if _legs:
                    st.caption("The match feed lists a fixture only once it is played, so the date and order "
                               "of the next one are not known here. Still to play in this phase:")
                    _met = {}      # our latest meeting with each club this season
                    for _o, (_gf, _ga), _row in results:
                        _home = _row['homeTeamName'] == OUR_TEAM
                        _met[_row['awayTeamName'] if _home else _row['homeTeamName']] = (
                            f"{_o} {_gf}–{_ga} ({'H' if _home else 'A'})")
                    _left_df = pd.DataFrame({
                        'Crest': [crest_data_uri(o) for o, _ in _legs],
                        'Opponent': [o for o, _ in _legs],
                        'Venue': ['Home' if v == 'H' else 'Away' for _, v in _legs],
                        'Already this season': [_met.get(o, '—') for o, _ in _legs],
                    })
                    _pick = st.dataframe(
                        _left_df, hide_index=True, width="stretch",
                        on_select='rerun', selection_mode='single-row', key='home_remaining_table',
                        column_config={'Crest': st.column_config.ImageColumn('', width='small')})
                    st.caption("Select a row to open that team's Opposition Report.")
                    _rows = getattr(getattr(_pick, 'selection', None), 'rows', None) or []
                    if _rows:
                        context_bar.set_context(league_name, season_label)
                        navigation.go_to('Opposition Report',
                                         opposition_report_team=_left_df['Opponent'].iloc[_rows[0]])
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
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

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
                hide_index=True, width="stretch",
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
    if q1.button("Team report", width="stretch", key="home_q_team"):
        context_bar.set_context(league_name, season_label)
        navigation.go_to('Team Analysis')
    if q2.button("League table & strength", width="stretch", key="home_q_league"):
        context_bar.set_context(league_name, season_label)
        navigation.go_to('League Analysis')
    if q3.button("Promotion probabilities", width="stretch", key="home_q_pred"):
        navigation.go_to('Match Predictor')
