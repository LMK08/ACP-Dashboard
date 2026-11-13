"""One search box for players, teams and coaches (2026-11 UX pass).

It sits at the top of the sidebar, above the navigation, on every page (a
fixed height, so the page list still never moves). Type a name, pick it,
and the app opens the right page ON that player / team / coach, in the
league and season where the data is:

* a player  -> Player Profile, in the newest season he has minutes in
              (the same session keys the Similar Players bridge sets)
* a team    -> Team Analysis, in the newest season the team played
* a coach   -> Coach Profile, in the league of his latest match

The index is built once per data state (`build_index`, pure pandas, tested
without the app) and the pick is handled in the selectbox's `on_change`
callback, which runs BEFORE the script body: that is where widget keys and
the League / Season context may still be written. The box empties itself
after a pick. Nothing here renames or computes a football number.
"""
import sys

import pandas as pd
import streamlit as st

import context_bar
import navigation
from league_config import COMPETITIONS, competition_for_season

KEY = 'global_search'
PLACEHOLDER = 'Player, team or coach'      # the sidebar is 223 px wide: longer text is cut


def _app():
    return sys.modules['__main__']


def _year(label):
    try:
        return int(str(label)[:4])
    except ValueError:
        return 0


def build_index(player_minutes_data, matches_summary_df, coach_tables=None):
    """{option id: entry} in display order: teams, coaches, players.

    player_minutes_data: {season_id: frame(playerId, playerName, teamName,
    totalMinutes, ...)}; matches_summary_df: the fixtures frame;
    coach_tables: (coach_team_match, coaches indexed by coachId) or None.
    An entry holds 'kind', 'label' and what the jump needs.
    """
    entries = {}

    # --- teams: the newest season each team played in --------------------
    sides = [matches_summary_df[['seasonId', side]].rename(columns={side: 'team'})
             for side in ('homeTeamName', 'awayTeamName') if side in matches_summary_df.columns]
    if sides:
        teams = pd.concat(sides).dropna()
        teams['comp'] = teams['seasonId'].map(competition_for_season)
        teams = teams.dropna(subset=['comp'])
        teams['season'] = [COMPETITIONS[int(c)]['seasons'][int(s)]
                           for c, s in zip(teams['comp'], teams['seasonId'])]
        teams['year'] = teams['season'].map(_year)
        newest = teams.sort_values('year', kind='stable').drop_duplicates('team', keep='last')
        for row in newest.sort_values('team', kind='stable').itertuples():
            league = COMPETITIONS[int(row.comp)]['name']
            entries[f't:{row.team}'] = {
                'kind': 'team', 'label': f'{row.team} — team, {league}',
                'team': str(row.team), 'league': league, 'season': row.season}

    # --- coaches: named ones, with the team of their latest match ---------
    if coach_tables is not None:
        matches, coaches = coach_tables
        known = matches[matches['coachId'] > 0].sort_values('date', kind='stable')
        latest = known.drop_duplicates('coachId', keep='last')
        n_matches = known.groupby('coachId').size()
        named = []
        for row in latest.itertuples():
            comp = int(row.competitionId) if pd.notna(row.competitionId) else None
            if int(row.coachId) not in coaches.index or comp not in COMPETITIONS:
                continue
            name = coaches.at[int(row.coachId), 'name']
            # the coach tables carry a 'Coach #<id>' row for every coach Wyscout
            # does not name: nobody can search for those
            if pd.isna(name) or not str(name).strip() or str(name).startswith('Coach #'):
                continue
            named.append((str(name), int(row.coachId), str(row.team), COMPETITIONS[comp]['name']))
        for name, coach_id, team, league in sorted(named):
            entries[f'c:{coach_id}'] = {
                'kind': 'coach', 'label': f'{name} — coach, {team}', 'coach_id': coach_id, 'league': league,
                'detail': (lambda n: f"{n} match" if n == 1 else f"{n} matches")(
                    int(n_matches.get(coach_id, 0)))}

    # --- players: the newest season each has minutes in --------------------
    frames = []
    for season_id, frame in (player_minutes_data or {}).items():
        comp = competition_for_season(season_id)
        if comp is None or frame is None or len(frame) == 0:
            continue
        part = frame[['playerId', 'playerName', 'teamName', 'totalMinutes']].copy()
        part['position'] = frame['primaryPosition'] if 'primaryPosition' in frame.columns else None
        part['seasonId'] = int(season_id)
        part['league'] = COMPETITIONS[comp]['name']
        part['season'] = COMPETITIONS[comp]['seasons'][season_id]
        frames.append(part)
    if frames:
        players = pd.concat(frames, ignore_index=True)
        players['totalMinutes'] = pd.to_numeric(players['totalMinutes'], errors='coerce').fillna(0)
        # a row without a real name cannot be searched for ('Unknown' is the
        # minutes files' placeholder)
        players = players[(players['totalMinutes'] > 0) & players['playerId'].notna()
                          & players['playerName'].notna()
                          & ~players['playerName'].astype(str).str.strip().isin(['', 'Unknown'])].copy()
        players['year'] = players['season'].map(_year)
        # one row per player: his newest season, and within it the club he played most for
        newest = (players.sort_values(['year', 'totalMinutes'], kind='stable')
                  .drop_duplicates('playerId', keep='last'))
        newest = newest.sort_values(['year', 'totalMinutes'], ascending=False, kind='stable')
        for row in newest.itertuples():
            position = None if pd.isna(row.position) or str(row.position) == 'Unknown' else str(row.position)
            entries[f'p:{int(row.playerId)}'] = {
                'kind': 'player', 'label': f'{row.playerName} — {row.teamName}, {row.season}',
                'player_id': int(row.playerId), 'season_id': int(row.seasonId), 'league': row.league,
                'detail': ', '.join(x for x in (position, f"{row.totalMinutes:,.0f} min") if x)}
    return _unique_labels(entries)


def _unique_labels(entries):
    """Every label must be unique: Streamlit sends the LABEL of the pick back
    and maps it to an option, so of two entries with one label only the last
    could ever be opened (nine such pairs in the 2026-11 data: namesakes at
    one club in one season, two coaches called Carlos Pinto). A label that
    repeats gets the entry's detail (position and minutes, or matches
    coached), and a counter if that is still not enough."""
    def repeats():
        seen = {}
        for entry in entries.values():
            seen[entry['label']] = seen.get(entry['label'], 0) + 1
        return {label for label, n in seen.items() if n > 1}

    shared = repeats()
    for entry in entries.values():
        if entry['label'] in shared and entry.get('detail'):
            entry['label'] = f"{entry['label']} ({entry['detail']})"
    shared, count = repeats(), {}
    for entry in entries.values():
        if entry['label'] in shared:
            count[entry['label']] = count.get(entry['label'], 0) + 1
            entry['label'] = f"{entry['label']} [{count[entry['label']]}]"
    return entries


@st.cache_resource(show_spinner=False)
def _cached_index(signature, _player_minutes_data, _matches_summary_df, _load_coach_tables):
    return build_index(_player_minutes_data, _matches_summary_df, _load_coach_tables())


def _index():
    app = _app()
    minutes = app.player_minutes_data if isinstance(app.player_minutes_data, dict) else {}
    coach_sig = None
    try:
        import views.coach_profile as coach_view
        coach_sig = coach_view.tables_signature()
    except Exception:           # the search works without coaches
        app.logger.exception('search: coach tables unavailable')

    def load_coach_tables():
        # called only when the index is (re)built: load_tables is cache_data,
        # which would unpickle the three coach tables on every run of every page
        if coach_sig is None:
            return None
        try:
            tables = coach_view.load_tables(coach_sig)
        except Exception:
            app.logger.exception('search: coach tables unavailable')
            return None
        return None if tables is None else (tables[0], tables[1])

    signature = (tuple(sorted((int(s), len(f)) for s, f in minutes.items() if f is not None)),
                 len(app.matches_summary_df), coach_sig)
    return _cached_index(signature, minutes, app.matches_summary_df, load_coach_tables)


def _on_pick(index):
    """The jump. Runs as the selectbox's on_change callback, before the
    script body, so the keys below are written before any widget exists."""
    picked = st.session_state.get(KEY)
    st.session_state[KEY] = None             # the box empties itself
    entry = index.get(picked)
    if not entry:
        return
    if entry['kind'] == 'player':
        # the keys app.py's prelude reads to open a profile in HIS league and
        # season (similar_players_ui.profile_bridge_state sets the same ones)
        st.session_state.selected_player_id = entry['player_id']
        st.session_state.nav_to_profile = True
        st.session_state.nav_season_id = entry['season_id']
        st.session_state.nav_has_season = True
        st.session_state['nav_league'] = entry['league']
    elif entry['kind'] == 'team':
        context_bar.set_context(entry['league'], entry['season'])
        st.session_state['nav_team'] = entry['team']
        # Team Analysis builds its team list AFTER the stage filter and drops
        # a nav_team that is not in it: with "Promotion" still selected, a
        # side from the maintenance stage would not open. Back to all stages.
        st.session_state.pop('team_analysis_stage_selector', None)
        st.session_state.current_page = 'Team Analysis'
    elif entry['kind'] == 'coach':
        import views.coach_profile as coach_view
        # his league in the page head, so the profile does not have to say
        # "showing another league than the one selected"
        context_bar.set_context(entry['league'])
        st.session_state[coach_view.SEED_KEY] = entry['coach_id']
        st.session_state.current_page = coach_view.PAGE


def render():
    """Draw the box in the sidebar (app.py calls this between the club block
    and the navigation)."""
    try:
        index = _index()
    except Exception:
        # drawn on every page, ahead of everything else (and ahead of the
        # --precompute branch): a broken index costs the box, not the app
        _app().logger.exception('search: index could not be built')
        return
    st.sidebar.markdown(navigation._GROUP_HEADER.format('Search'), unsafe_allow_html=True)
    st.sidebar.selectbox(
        'Search', list(index), index=None, key=KEY, placeholder=PLACEHOLDER,
        label_visibility='collapsed', format_func=lambda option: index[option]['label'],
        on_change=_on_pick, args=(index,),
        help='Type a name. A player opens his profile, a team its analysis, a coach his profile.')
