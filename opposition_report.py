# opposition_report.py
"""Opposition Report module for the ACP Dashboard.

Analyzes the next opponent and generates a comprehensive scouting report.
"""

import streamlit as st
import pandas as pd
import numpy as np
# Selects the Agg backend before pyplot is imported, and owns MPL_LOCK, the
# process-wide lock serialising all matplotlib work (see mpl_safety.py).
from mpl_safety import MPL_LOCK
import matplotlib.pyplot as plt
import datetime
import sys
import io
import pitch_visualizations as pv

# ---------------------------------------------------------------------------
# Helper: access functions defined in the main app module (app.py / __main__)
# ---------------------------------------------------------------------------
def _get_app():
    """Return the main app module so we can reuse its functions."""
    return sys.modules['__main__']


def _get_transferred_players():
    """Return the set of player names who transferred out of Liga 3."""
    app = _get_app()
    if hasattr(app, 'load_transferred_players'):
        return set(app.load_transferred_players())
    return set()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GROUP_B_TEAMS = [
    '1º Dezembro', 'Caldas', 'Sporting Covilhã', 'Mafra',
    'União Santarém', 'Amora', 'Académica', 'CF Os Belenenses',
    'Lusitano Évora 1911',
]

OUR_TEAM = 'Atlético CP'

OFFENSIVE_METRICS = [
    'Goals', 'xG', 'xG per Shot', 'Shots',
    'Actions in Box', 'Passes into Box', 'Crosses', 'Dribbles',
]
DISTRIBUTION_METRICS = [
    'Passes', 'Progressive Passes', 'Directness',
    'Ball Possession', 'Losses',
]
DEFENSIVE_METRICS_TEAM = [
    'Goals Against', 'xG Against', 'xG per Shot Against',
    'Shots Against', 'Aerial Duel Win %', 'Defensive Duel Win %',
    'Interceptions', 'Fouls', 'PPDA',
]
SET_PIECE_METRICS_RADAR = [
    'Corners', 'xG per Corner', 'Goals per Corner', 'Short Corner %',
    'Long Throws', 'Long Throw %', 'xG per Long Throw',
    'First Contact %', 'xG per FK Delivery', 'Penalties', 'Non-Pen SP Goals',
]


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def _fig_to_png_bytes(fig):
    """Convert a matplotlib figure to PNG bytes (for PDF embedding)."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def _find_next_fixture(matches_df, season_id):
    """Auto-detect the next unplayed fixture for Atletico CP."""
    season_matches = matches_df[matches_df['seasonId'] == season_id].copy()
    acp_matches = season_matches[
        (season_matches['homeTeamName'] == OUR_TEAM) |
        (season_matches['awayTeamName'] == OUR_TEAM)
    ].copy()

    if acp_matches.empty:
        return None

    def _is_unplayed(score):
        if pd.isna(score):
            return True
        return '-' not in str(score)

    acp_matches['_unplayed'] = acp_matches['score'].apply(_is_unplayed)
    unplayed = acp_matches[acp_matches['_unplayed']].copy()

    if unplayed.empty:
        return None

    unplayed['dateutc'] = pd.to_datetime(unplayed['dateutc'], errors='coerce')
    unplayed = unplayed.sort_values('dateutc')
    nxt = unplayed.iloc[0]

    opponent = (nxt['awayTeamName']
                if nxt['homeTeamName'] == OUR_TEAM
                else nxt['homeTeamName'])
    home_away = 'Home' if nxt['homeTeamName'] == OUR_TEAM else 'Away'

    gw = nxt.get('gameweek', '?')
    # If gameweek is NaN/None/"?", derive it from the team's fixture position
    if pd.isna(gw) or str(gw).strip() in ('', '?', 'nan', 'None'):
        acp_tmp = acp_matches.copy()
        acp_tmp['_sort_date'] = pd.to_datetime(acp_tmp['dateutc'], errors='coerce')
        acp_sorted = acp_tmp.sort_values('_sort_date')
        match_ids = acp_sorted['matchId'].tolist()
        try:
            gw = match_ids.index(nxt.get('matchId')) + 1
        except ValueError:
            gw = '?'

    return {
        'opponent': opponent,
        'date': nxt['dateutc'],
        'gameweek': gw,
        'home_away': home_away,
        'matchId': nxt.get('matchId'),
    }


def _get_team_recent_form(matches_df, team_name, season_id, n=5):
    """Return last *n* played results for *team_name* in the given season."""
    season_matches = matches_df[matches_df['seasonId'] == season_id].copy()
    tm = season_matches[
        (season_matches['homeTeamName'] == team_name) |
        (season_matches['awayTeamName'] == team_name)
    ].copy()

    def _is_played(score):
        if pd.isna(score):
            return False
        return '-' in str(score)

    tm = tm[tm['score'].apply(_is_played)].copy()
    tm['dateutc'] = pd.to_datetime(tm['dateutc'], errors='coerce')
    tm = tm.sort_values('dateutc', ascending=False).head(n)

    results = []
    for _, match in tm.iterrows():
        parts = str(match['score']).split('-')
        if len(parts) != 2:
            continue
        try:
            hg = int(parts[0].strip())
            ag = int(parts[1].strip())
        except ValueError:
            continue
        is_home = match['homeTeamName'] == team_name
        gf = hg if is_home else ag
        ga = ag if is_home else hg
        opp = match['awayTeamName'] if is_home else match['homeTeamName']

        if gf > ga:
            res = 'W'
        elif gf == ga:
            res = 'D'
        else:
            res = 'L'

        results.append({
            'date': match['dateutc'],
            'opponent': opp,
            'score': match['score'],
            'result': res,
            'home_away': 'H' if is_home else 'A',
            'goals_for': gf,
            'goals_against': ga,
        })
    return results


def _get_projected_starting_xi(events_df, matches_df, team_name, season_id):
    """Projected starting XI from the last 3 completed matches (recency-weighted)."""
    from collections import Counter

    season_matches = matches_df[matches_df['seasonId'] == season_id].copy()
    tm = season_matches[
        (season_matches['homeTeamName'] == team_name) |
        (season_matches['awayTeamName'] == team_name)
    ].copy()

    def _is_played(score):
        if pd.isna(score):
            return False
        return '-' in str(score)

    tm = tm[tm['score'].apply(_is_played)].copy()
    tm['dateutc'] = pd.to_datetime(tm['dateutc'], errors='coerce')
    tm = tm.sort_values('dateutc', ascending=False).head(3)

    if tm.empty:
        return '4-4-2', {}

    # Formation from most recent match
    most_recent_id = tm.iloc[0]['matchId']
    mr_events = events_df[
        (events_df['matchId'] == most_recent_id) &
        (events_df['team.name'] == team_name)
    ]

    formation = '4-4-2'
    if 'team.formation' in mr_events.columns:
        fc = mr_events['team.formation'].dropna().value_counts()
        # team.formation is categorical: value_counts reports zero-count
        # categories too — drop them or an absent formation wins the fallback
        fc = fc[fc > 0]
        if len(fc) > 0:
            formation = fc.index[0]

    # Weighted position-player mapping
    weights = [3, 2, 1]
    position_player_weights = {}

    for i, (_, match) in enumerate(tm.iterrows()):
        if i >= 3:
            break
        w = weights[i]
        me = events_df[
            (events_df['matchId'] == match['matchId']) &
            (events_df['team.name'] == team_name)
        ]
        if 'player.position' not in me.columns or 'player.name' not in me.columns:
            continue

        pp = me.drop_duplicates(subset=['player.name'])[
            ['player.position', 'player.name', 'player.id']
        ].dropna()

        for _, row in pp.iterrows():
            pos = row['player.position']
            name = row['player.name']
            pid = row['player.id']
            if pos not in position_player_weights:
                position_player_weights[pos] = Counter()
            position_player_weights[pos][(name, pid)] += w

    # Filter out transferred players before building the XI
    transferred = _get_transferred_players()
    for pos in position_player_weights:
        position_player_weights[pos] = Counter({
            k: v for k, v in position_player_weights[pos].items()
            if k[0] not in transferred
        })

    # Build raw XI from best player per position
    raw_xi = {}
    for pos, counter in position_player_weights.items():
        if counter:
            best = counter.most_common(1)[0][0]
            raw_xi[pos] = {'name': best[0], 'id': best[1]}

    # Constrain to exactly 11 unique players (10 outfield + 1 GK)
    # Use the formation slots to pick the best 11 via map_players_to_formation
    # which already handles dedup and tier-based matching.
    # If there are >11 unique players, the mapping naturally selects 11.
    # Remove any duplicate names (same player listed under multiple positions)
    seen_names = set()
    starting_xi = {}
    for pos, info in raw_xi.items():
        if info['name'] not in seen_names:
            starting_xi[pos] = info
            seen_names.add(info['name'])

    # If still >11, let map_players_to_formation in app.py handle it
    # (it maps to exactly 11 formation slots and discards extras)

    return formation, starting_xi


def _get_projected_subs(events_df, player_minutes_df, team_name,
                        starting_xi_names, season_id):
    """Players with minutes that are NOT in the projected XI.

    Filters out GK and CB positions since they rarely come off the bench.
    """
    # Positions that almost never sub in
    _UNLIKELY_SUB_POSITIONS = {'GK', 'LCB', 'RCB', 'CB', 'LCB3', 'RCB3'}

    transferred = _get_transferred_players()

    team_mins = player_minutes_df[player_minutes_df['teamName'] == team_name].copy()
    team_mins['totalMinutes'] = pd.to_numeric(team_mins['totalMinutes'], errors='coerce')
    # Exclude transferred players and starting XI
    team_mins = team_mins[~team_mins['playerName'].isin(transferred)]
    subs = team_mins[~team_mins['playerName'].isin(starting_xi_names)].copy()
    subs = subs[subs['totalMinutes'] > 0].sort_values('totalMinutes', ascending=False)

    # Filter out unlikely sub positions
    if 'primaryPosition' in subs.columns:
        subs = subs[~subs['primaryPosition'].isin(_UNLIKELY_SUB_POSITIONS)]

    season_events = events_df[events_df['seasonId'] == season_id]
    te = season_events[season_events['team.name'] == team_name]
    if not te.empty and 'player.name' in te.columns:
        apps = te.groupby('player.name')['matchId'].nunique().reset_index()
        apps.columns = ['playerName', 'Appearances']
        subs = subs.merge(apps, on='playerName', how='left')
    else:
        subs['Appearances'] = 0

    subs['Appearances'] = subs['Appearances'].fillna(0).astype(int)
    cols = ['playerName', 'primaryPosition', 'totalMinutes', 'Appearances']
    available = [c for c in cols if c in subs.columns]
    return subs[available].head(10)


def _identify_key_players(player_stats_df, player_percentiles_df,
                          team_name, n=5):
    """Top *n* outfield key players by composite (minutes + position-relative role score)."""
    app = _get_app()

    transferred = _get_transferred_players()

    ts = player_stats_df[player_stats_df['teamName'] == team_name].copy()
    ts['totalMinutes'] = pd.to_numeric(ts['totalMinutes'], errors='coerce')

    # Exclude transferred players
    if transferred and 'playerName' in ts.columns:
        ts = ts[~ts['playerName'].isin(transferred)]

    # 200+ minutes, no GKs
    ts_filtered = ts[ts['totalMinutes'] >= 200]
    if 'primaryPosition' in ts_filtered.columns:
        ts_filtered = ts_filtered[ts_filtered['primaryPosition'] != 'GK']

    # Fallback to lower threshold
    if ts_filtered.empty:
        ts_filtered = ts[ts['totalMinutes'] >= 90]
        if 'primaryPosition' in ts_filtered.columns:
            ts_filtered = ts_filtered[ts_filtered['primaryPosition'] != 'GK']

    if ts_filtered.empty:
        return []

    # Build reverse mapping: Wyscout position -> list of eligible role names
    pos_to_roles = {}
    for role_name, positions in app.POSITION_GROUPS.items():
        for pos in positions:
            if pos not in pos_to_roles:
                pos_to_roles[pos] = []
            pos_to_roles[pos].append(role_name)

    score_cols = [c for c in ts_filtered.columns if c.endswith('_Score')]
    ts_filtered = ts_filtered.copy()

    if score_cols:
        # For each player, only consider _Score columns matching their position
        def _best_role_score(row):
            pri_pos = row.get('primaryPosition', '')
            eligible_roles = pos_to_roles.get(pri_pos, [])
            eligible_score_cols = [r + '_Score' for r in eligible_roles
                                   if r + '_Score' in score_cols]
            if eligible_score_cols:
                vals = [row[c] for c in eligible_score_cols
                        if pd.notna(row.get(c))]
                return max(vals) if vals else 0
            # Fallback: use max across all score columns
            return row[score_cols].max()

        ts_filtered['best_role_score'] = ts_filtered.apply(_best_role_score, axis=1)
    else:
        ts_filtered['best_role_score'] = 0

    # Assign coarse position buckets and normalize within each bucket
    def _pos_bucket(pos):
        if pos in ('GK',):
            return 'GK'
        if pos in ('LCB', 'RCB', 'CB', 'LCB3', 'RCB3', 'LB', 'RB',
                    'LB5', 'RB5', 'LWB', 'RWB'):
            return 'DEF'
        if pos in ('LCMF', 'RCMF', 'LCMF3', 'RCMF3', 'DMF', 'LDMF',
                    'RDMF', 'AMF', 'LAMF', 'RAMF'):
            return 'MID'
        return 'FWD'

    ts_filtered['_bucket'] = ts_filtered['primaryPosition'].apply(_pos_bucket)

    # Min-max normalize best_role_score within each bucket (0->1)
    def _normalize_within_bucket(group):
        mn = group['best_role_score'].min()
        mx = group['best_role_score'].max()
        if mx > mn:
            group['norm_score'] = (group['best_role_score'] - mn) / (mx - mn)
        else:
            group['norm_score'] = 1.0 if mx > 0 else 0.0
        return group

    ts_filtered = ts_filtered.groupby('_bucket', group_keys=False).apply(
        _normalize_within_bucket)

    mx_min = ts_filtered['totalMinutes'].max()
    ts_filtered['norm_min'] = ts_filtered['totalMinutes'] / mx_min if mx_min > 0 else 0

    # Weight toward attacking players: FWD/MID get a bonus, DEF only if standout
    # Bonus: FWD +0.20, MID +0.10, DEF +0.0 (but standout DEF with norm_score >= 0.9 gets +0.10)
    def _attack_bonus(row):
        bucket = row['_bucket']
        if bucket == 'FWD':
            return 0.20
        if bucket == 'MID':
            return 0.10
        # DEF: only boost if they're a standout (top of their bucket)
        if row.get('norm_score', 0) >= 0.9:
            return 0.10
        return 0.0

    ts_filtered['_atk_bonus'] = ts_filtered.apply(_attack_bonus, axis=1)
    ts_filtered['composite'] = (0.35 * ts_filtered['norm_min']
                                + 0.50 * ts_filtered['norm_score']
                                + 0.15 * 1.0  # base
                                + ts_filtered['_atk_bonus'])
    # Normalize composite to 0-1 for clean sorting
    c_min, c_max = ts_filtered['composite'].min(), ts_filtered['composite'].max()
    if c_max > c_min:
        ts_filtered['composite'] = (ts_filtered['composite'] - c_min) / (c_max - c_min)
    ts_filtered = ts_filtered.sort_values('composite', ascending=False)

    key_players = []
    for _, player in ts_filtered.head(n).iterrows():
        pid = player.get('playerId', None)
        if pd.isna(pid):
            continue
        pid = int(pid)

        pct_row = None
        if not player_percentiles_df.empty:
            if 'playerId' in player_percentiles_df.columns:
                pct_row = player_percentiles_df[player_percentiles_df['playerId'] == pid]
            elif player_percentiles_df.index.name == 'playerId':
                if pid in player_percentiles_df.index:
                    pct_row = player_percentiles_df.loc[[pid]]

        key_players.append({
            'player_id': pid,
            'name': player.get('playerName', 'Unknown'),
            'position': player.get('primaryPosition', '?'),
            'team': team_name,
            'minutes': player.get('totalMinutes', 0),
            'best_role_score': player.get('best_role_score', 0),
            'stats_row': player,
            'percentiles_row': pct_row,
        })
    return key_players


def _get_player_strengths_weaknesses(player_data, metrics_list):
    """Strengths (>=70th pct) and weaknesses (<=30th pct) for a player."""
    strengths, weaknesses = [], []
    for metric in metrics_list:
        pct_col = metric + '_percentile'
        if pct_col not in player_data.columns:
            continue
        pct_val = player_data[pct_col].values[0]
        if not isinstance(pct_val, (int, float)):
            continue
        pct_100 = pct_val * 100 if pct_val <= 1 else pct_val
        raw = player_data[metric].values[0] if metric in player_data.columns else None
        if pct_100 >= 70:
            strengths.append((metric, pct_100, raw))
        elif pct_100 <= 30:
            weaknesses.append((metric, pct_100, raw))

    strengths.sort(key=lambda x: x[1], reverse=True)
    weaknesses.sort(key=lambda x: x[1])
    return strengths, weaknesses


def _generate_team_synopsis(stats_pct_df, team_name):
    """Categorise each metric and detect tactical profile."""
    if team_name not in stats_pct_df.index:
        return [], [], []

    team_pct = stats_pct_df.loc[team_name]
    strengths, weaknesses, average = [], [], []

    for metric in team_pct.index:
        v = team_pct[metric]
        if v >= 65:
            strengths.append((metric, v))
        elif v <= 35:
            weaknesses.append((metric, v))
        else:
            average.append((metric, v))

    strengths.sort(key=lambda x: x[1], reverse=True)
    weaknesses.sort(key=lambda x: x[1])

    profiles = []
    poss = team_pct.get('Ball Possession', 50)
    passes = team_pct.get('Passes', 50)
    ppda = team_pct.get('PPDA', 50)
    ga = team_pct.get('Goals Against', 50)
    xga = team_pct.get('xG Against', 50)
    drib = team_pct.get('Dribbles', 50)

    if poss >= 65 and passes >= 60:
        profiles.append("Possession-based")
    if ppda >= 70:
        profiles.append("High-pressing")
    if ga >= 65 and xga >= 65:
        profiles.append("Defensively solid")
    if poss <= 40 and drib >= 60:
        profiles.append("Counter-attacking")
    if not profiles:
        profiles.append("Balanced")

    return strengths, weaknesses, profiles


def _generate_key_takeaways(team_name, strengths, weaknesses, profiles,
                            form_results, key_players, set_piece_df,
                            stats_pct_df):
    """5-8 auto-generated bullet points combining all analysis."""
    takeaways = []

    # 1. Tactical profile
    if profiles:
        takeaways.append(
            f"**Tactical Profile:** {team_name} play a "
            f"{', '.join(profiles).lower()} style."
        )

    # 2. Offensive strengths
    off_str = [s for s in strengths if s[0] in OFFENSIVE_METRICS]
    if off_str:
        ms = ', '.join(f"{s[0]} ({s[1]:.0f}th pct)" for s in off_str[:3])
        takeaways.append(f"**Offensive Strengths:** Strong in {ms}.")
    else:
        takeaways.append(
            "**Offensive Threat:** Average; no standout attacking "
            "metrics above 65th percentile."
        )

    # 3. Defensive vulnerabilities
    def_wk = [w for w in weaknesses if w[0] in DEFENSIVE_METRICS_TEAM]
    if def_wk:
        ms = ', '.join(f"{w[0]} ({w[1]:.0f}th pct)" for w in def_wk[:3])
        takeaways.append(
            f"**Defensive Vulnerabilities:** Weak in {ms} — areas to exploit."
        )

    # 4. Distribution weaknesses
    dist_wk = [w for w in weaknesses if w[0] in DISTRIBUTION_METRICS]
    if dist_wk:
        ms = ', '.join(f"{w[0]} ({w[1]:.0f}th pct)" for w in dist_wk[:2])
        takeaways.append(f"**Distribution Weaknesses:** {ms}.")

    # 5. Form trend
    if form_results:
        form_str = ''.join(r['result'] for r in form_results)
        wins = sum(1 for r in form_results if r['result'] == 'W')
        draws = sum(1 for r in form_results if r['result'] == 'D')
        losses = sum(1 for r in form_results if r['result'] == 'L')
        takeaways.append(
            f"**Form:** {form_str} — {wins}W {draws}D {losses}L "
            f"in the last {len(form_results)} matches."
        )

    # 6. Set piece threat (league-relative using percentile rank)
    if set_piece_df is not None and team_name in set_piece_df.index:
        sp = set_piece_df.loc[team_name]
        sp_xg = sp.get('xG from Set Pieces', 0)
        sp_goals = sp.get('Goals from Set Pieces', 0)
        sp_conceded = sp.get('xG Conceded Set Pieces', 0)

        # Compute percentile rank within the league
        n_teams = len(set_piece_df)
        if n_teams > 1 and 'xG from Set Pieces' in set_piece_df.columns:
            xg_rank = (set_piece_df['xG from Set Pieces'] <= sp_xg).sum()
            sp_xg_pct = 100 * xg_rank / n_teams
        else:
            sp_xg_pct = 50

        if n_teams > 1 and 'xG Conceded Set Pieces' in set_piece_df.columns:
            conc_rank = (set_piece_df['xG Conceded Set Pieces'] <= sp_conceded).sum()
            sp_conc_pct = 100 * conc_rank / n_teams
        else:
            sp_conc_pct = 50

        if sp_xg_pct >= 65:
            takeaways.append(
                f"**Set Piece Threat:** {sp_xg:.1f} xG and "
                f"{int(sp_goals)} goals from set pieces "
                f"({sp_xg_pct:.0f}th pct in league)."
            )
        if sp_conc_pct >= 65:
            takeaways.append(
                f"**Set Piece Vulnerability:** Conceded {sp_conceded:.1f} xG "
                f"from set pieces ({sp_conc_pct:.0f}th pct in league)."
            )

    # 7. Key players
    if key_players:
        strs = [
            f"{kp['name']} ({kp['position']}, {kp['minutes']:.0f} mins)"
            for kp in key_players[:3]
        ]
        takeaways.append(f"**Key Players:** {'; '.join(strs)}.")

    return takeaways


# ===========================================================================
# MAIN RENDER FUNCTION
# ===========================================================================
def render_opposition_report(raw_events_df, matches_summary_df,
                             all_match_data, season_team_stats,
                             player_minutes_data,
                             current_season_id, season_id_map):
    """Entry point called from app.py."""
    app = _get_app()

    st.header("Opposition Report")

    # --- Season selector ---
    selected_season_id = app.season_selector("opposition_report")

    # --- Season data ---
    season_events_df = app.get_season_events(raw_events_df, selected_season_id)
    season_matches_df = app.get_season_matches(matches_summary_df, selected_season_id)
    season_player_minutes = app.get_season_player_minutes(
        player_minutes_data, selected_season_id
    )

    # ===================================================================
    # Opponent selection
    # ===================================================================
    st.subheader("Select Opponent")

    next_fixture = _find_next_fixture(matches_summary_df, selected_season_id)

    all_opponents = sorted(
        set(GROUP_B_TEAMS)
        | set(season_matches_df['homeTeamName'].dropna().unique())
        | set(season_matches_df['awayTeamName'].dropna().unique())
    )
    all_opponents = [t for t in all_opponents if t != OUR_TEAM]

    if not all_opponents:
        st.warning("No opponent teams found for this season.")
        return

    default_idx = 0
    if next_fixture and next_fixture['opponent'] in all_opponents:
        default_idx = all_opponents.index(next_fixture['opponent'])

    selected_opponent = st.selectbox(
        "Opponent", all_opponents, index=default_idx,
        key="opposition_report_team",
    )

    fixture_info = None
    if next_fixture and next_fixture['opponent'] == selected_opponent:
        date_str = (next_fixture['date'].strftime('%d %b %Y')
                    if pd.notna(next_fixture['date']) else '?')
        st.info(
            f"**Next Fixture:** {OUR_TEAM} vs {selected_opponent} | "
            f"GW {next_fixture['gameweek']} | {date_str} | "
            f"{next_fixture['home_away']}"
        )
        fixture_info = next_fixture
    else:
        st.info(f"Analysing: **{selected_opponent}**")

    st.divider()

    # Collections for PDF
    pdf_figures = {}   # key -> PNG bytes
    pdf_texts = {}

    # Initialise variables used across sections
    strengths, weaknesses, profiles = [], [], []
    key_players = []
    set_piece_df = pd.DataFrame()
    form_results = []
    stats_df_pct = pd.DataFrame()

    # ===================================================================
    # Step 3: Team Overview — Three Radar Charts
    # ===================================================================
    st.subheader(f"{selected_opponent} — Team Overview")

    stats_df_raw, stats_df_pct = app.calculate_all_team_radars_stats(
        season_events_df, season_matches_df, season_id=selected_season_id,
    )

    # Compute set piece radar data
    sp_df_raw = None
    sp_df_pct = None
    try:
        sp_df_raw = app.calculate_set_piece_metrics(season_events_df, season_id=selected_season_id)
        if sp_df_raw is not None and not sp_df_raw.empty:
            sp_df_pct = sp_df_raw.copy()
            for col in sp_df_pct.columns:
                sp_df_pct[col] = sp_df_pct[col].rank(pct=True) * 100
    except Exception:
        pass

    if selected_opponent in stats_df_pct.index:
        team_raw = stats_df_raw.loc[selected_opponent]
        team_pct = stats_df_pct.loc[selected_opponent]

        # Row 1: Offensive + Distribution
        col1, col2 = st.columns(2)

        # Offensive
        off_m = [m for m in OFFENSIVE_METRICS if m in team_pct.index]
        if off_m:
            with MPL_LOCK:
                fig = app.plot_radar_chart(
                    off_m, [team_raw[m] for m in off_m],
                    [team_pct[m] for m in off_m],
                    selected_opponent, "Offensive", '#e63946',
                )
                with col1:
                    st.pyplot(fig, use_container_width=True)
                pdf_figures['radar_offensive'] = _fig_to_png_bytes(fig)
                plt.close(fig)

        # Distribution
        dist_m = [m for m in DISTRIBUTION_METRICS if m in team_pct.index]
        if dist_m:
            with MPL_LOCK:
                fig = app.plot_radar_chart(
                    dist_m, [team_raw[m] for m in dist_m],
                    [team_pct[m] for m in dist_m],
                    selected_opponent, "Distribution", '#0077b6',
                )
                with col2:
                    st.pyplot(fig, use_container_width=True)
                pdf_figures['radar_distribution'] = _fig_to_png_bytes(fig)
                plt.close(fig)

        # Row 2: Defensive + Set Piece
        col3, col4 = st.columns(2)

        # Defensive
        def_m = [m for m in DEFENSIVE_METRICS_TEAM if m in team_pct.index]
        if def_m:
            with MPL_LOCK:
                fig = app.plot_radar_chart(
                    def_m, [team_raw[m] for m in def_m],
                    [team_pct[m] for m in def_m],
                    selected_opponent, "Defensive", '#2a9d8f',
                )
                with col3:
                    st.pyplot(fig, use_container_width=True)
                pdf_figures['radar_defensive'] = _fig_to_png_bytes(fig)
                plt.close(fig)

        # Set Piece
        if sp_df_raw is not None and not sp_df_raw.empty and selected_opponent in sp_df_raw.index:
            sp_team_raw = sp_df_raw.loc[selected_opponent]
            sp_team_pct = sp_df_pct.loc[selected_opponent]
            sp_m = [m for m in SET_PIECE_METRICS_RADAR if m in sp_team_raw.index]
            if sp_m:
                raw_sp_values = [sp_team_raw[m] for m in sp_m]
                for _pct_name in ['Short Corner %', 'Long Throw %', 'First Contact %']:
                    try:
                        _idx = sp_m.index(_pct_name)
                        raw_sp_values[_idx] = f"{raw_sp_values[_idx]:.0f}%"
                    except ValueError:
                        pass
                with MPL_LOCK:
                    fig = app.plot_radar_chart(
                        sp_m, raw_sp_values,
                        [sp_team_pct[m] for m in sp_m],
                        selected_opponent, "Set Piece Radar", '#ff8c00',
                    )
                    with col4:
                        st.pyplot(fig, use_container_width=True)
                    pdf_figures['radar_set_piece'] = _fig_to_png_bytes(fig)
                    plt.close(fig)
    else:
        st.warning(f"No radar data available for {selected_opponent}.")

    st.divider()

    # ===================================================================
    # Step 4: Projected Lineup & Formation
    # ===================================================================
    st.subheader(f"{selected_opponent} — Projected Lineup")

    formation, starting_xi = _get_projected_starting_xi(
        season_events_df, season_matches_df, selected_opponent,
        selected_season_id,
    )
    xi_names = set()  # will be populated with exactly 11 names below

    if starting_xi:
        # Map to exactly 11 formation slots (dedup + constrain)
        from pitch_visualizations import FORMATION_COORDS
        _fkey = formation if formation in FORMATION_COORDS else '4-4-2'
        _fslots = FORMATION_COORDS[_fkey]['positions']
        _mapped_xi = app.map_players_to_formation(starting_xi, _fslots)
        xi_names = {v['name'] for v in _mapped_xi.values()
                    if 'name' in v and v.get('id') is not None}

        col_form, col_subs = st.columns([2, 1])

        with col_form:
            with MPL_LOCK:
                fig = app.create_formation_graphic(
                    formation, starting_xi, selected_opponent,
                )
                st.pyplot(fig)
                pdf_figures['formation'] = _fig_to_png_bytes(fig)
                plt.close(fig)

        with col_subs:
            st.markdown("**Projected Substitutes**")
            if season_player_minutes is not None and not season_player_minutes.empty:
                subs_df = _get_projected_subs(
                    season_events_df, season_player_minutes,
                    selected_opponent, xi_names, selected_season_id,
                )
                if not subs_df.empty:
                    rename_map = {
                        'playerName': 'Player', 'primaryPosition': 'Position',
                        'totalMinutes': 'Minutes', 'Appearances': 'Apps',
                    }
                    rm = {k: v for k, v in rename_map.items() if k in subs_df.columns}
                    st.dataframe(
                        subs_df.rename(columns=rm),
                        use_container_width=True, hide_index=True,
                    )
                    pdf_texts['subs'] = subs_df.rename(columns=rm)
                else:
                    st.caption("No substitute data available.")
            else:
                st.caption("Player minutes data not available.")
    else:
        st.warning(f"No lineup data available for {selected_opponent}.")

    st.divider()

    # ===================================================================
    # Step 5: Key Players Deep Dive
    # ===================================================================
    st.subheader(f"{selected_opponent} — Key Players")

    if season_player_minutes is not None and not season_player_minutes.empty:
        player_stats_df = app.calculate_all_player_stats(
            season_events_df, season_player_minutes,
            season_id=selected_season_id,
        )

        if not player_stats_df.empty:
            player_percentiles_df = app.calculate_player_percentiles_and_scores(
                player_stats_df, app.POSITION_GROUPS, app.WEIGHTS,
                app.INVERT_METRICS, min_minutes=90,
                season_id=selected_season_id,
            )

            key_players = _identify_key_players(
                player_stats_df, player_percentiles_df,
                selected_opponent, n=5,
            )

            if key_players:
                for i, kp in enumerate(key_players):
                    with st.expander(
                        f"**{i+1}. {kp['name']}** — {kp['position']} | "
                        f"{kp['minutes']:.0f} mins",
                        expanded=(i == 0),
                    ):
                        player_pos = kp['position']
                        eligible_groups = [
                            g for g, positions in app.POSITION_GROUPS.items()
                            if player_pos in positions
                        ]

                        if (kp['percentiles_row'] is not None
                                and not kp['percentiles_row'].empty
                                and eligible_groups):
                            player_data = kp['percentiles_row']

                            # Find best role
                            best_group = eligible_groups[0]
                            scores = {}
                            for g in eligible_groups:
                                sc = g + '_Score'
                                if sc in player_data.columns:
                                    scores[g] = player_data[sc].values[0]
                            if scores:
                                best_group = max(scores, key=scores.get)

                            metrics = list(
                                app.WEIGHTS.get(best_group, {}).keys()
                            )
                            metrics = [
                                m for m in metrics if m in player_data.columns
                            ]

                            if metrics:
                                col_r, col_a = st.columns([2, 1])

                                with col_r:
                                    try:
                                        pos_grp = app.POSITION_GROUPS.get(
                                            best_group, [player_pos]
                                        )
                                        all_pos = player_percentiles_df[
                                            player_percentiles_df[
                                                'primaryPosition'
                                            ].isin(pos_grp)
                                        ]
                                        with MPL_LOCK:
                                            fig = app.create_radar_with_distributions(
                                                player_data, metrics, player_pos,
                                                eligible_groups, all_pos,
                                                full_df_for_ranking=player_percentiles_df,
                                            )
                                            st.pyplot(fig)
                                            pdf_figures[f'player_{i}'] = (
                                                _fig_to_png_bytes(fig)
                                            )
                                            plt.close(fig)
                                    except Exception as e:
                                        st.caption(
                                            f"Could not render radar: {e}"
                                        )

                                with col_a:
                                    st.markdown(f"**Best Role:** {best_group}")

                                    s, w = _get_player_strengths_weaknesses(
                                        player_data, metrics,
                                    )
                                    # Store for PDF
                                    kp['strengths_lines'] = [
                                        f"{m}: {p:.0f}th pct ({r:.2f} p90)"
                                        if r is not None else f"{m}: {p:.0f}th pct"
                                        for m, p, r in s[:5]
                                    ]
                                    kp['weaknesses_lines'] = [
                                        f"{m}: {p:.0f}th pct ({r:.2f} p90)"
                                        if r is not None else f"{m}: {p:.0f}th pct"
                                        for m, p, r in w[:5]
                                    ]
                                    if s:
                                        st.markdown("**Strengths:**")
                                        for met, pct, raw in s[:5]:
                                            rs = (f" ({raw:.2f} p90)"
                                                  if raw is not None else "")
                                            st.markdown(
                                                f"- :green[{met}] — "
                                                f"{pct:.0f}th pct{rs}"
                                            )
                                    if w:
                                        st.markdown("**Weaknesses:**")
                                        for met, pct, raw in w[:5]:
                                            rs = (f" ({raw:.2f} p90)"
                                                  if raw is not None else "")
                                            st.markdown(
                                                f"- :red[{met}] — "
                                                f"{pct:.0f}th pct{rs}"
                                            )

                                    st.markdown("**Key Stats (per 90):**")
                                    stat_names = [
                                        'Goals', 'npxG', 'Assists',
                                        'xAOP', 'Passes', 'Interceptions',
                                    ]
                                    rows = []
                                    for sn in stat_names:
                                        if sn in player_data.columns:
                                            v = player_data[sn].values[0]
                                            rows.append({
                                                'Metric': sn,
                                                'Value': f"{v:.2f}",
                                            })
                                    if rows:
                                        st.dataframe(
                                            pd.DataFrame(rows),
                                            hide_index=True,
                                            use_container_width=True,
                                        )
                            else:
                                st.caption(
                                    "No metrics available for this "
                                    "player's position."
                                )
                        else:
                            st.caption(
                                "Percentile data not available "
                                "for this player."
                            )

                pdf_texts['key_players'] = key_players
            else:
                st.info("Not enough data to identify key players.")
        else:
            st.warning("Player stats could not be calculated.")
    else:
        st.warning("Player minutes data not available for this season.")

    st.divider()

    # ===================================================================
    # Step 6: Strengths & Weaknesses Synopsis
    # ===================================================================
    st.subheader(f"{selected_opponent} — Strengths & Weaknesses")

    if selected_opponent in stats_df_pct.index:
        strengths, weaknesses, profiles = _generate_team_synopsis(
            stats_df_pct, selected_opponent,
        )

        if profiles:
            st.markdown(f"**Tactical Profile:** {', '.join(profiles)}")

        col_s, col_w = st.columns(2)

        with col_s:
            st.markdown("**Strengths** (65th+ percentile)")
            if strengths:
                for met, val in strengths:
                    tier = "Elite" if val >= 80 else "Above Average"
                    st.markdown(
                        f"- :green[{met}] — {val:.0f}th pct ({tier})"
                    )
            else:
                st.caption("No standout strengths identified.")

        with col_w:
            st.markdown("**Weaknesses** (35th- percentile)")
            if weaknesses:
                for met, val in weaknesses:
                    tier = "Poor" if val <= 20 else "Below Average"
                    st.markdown(
                        f"- :red[{met}] — {val:.0f}th pct ({tier})"
                    )
            else:
                st.caption("No clear weaknesses identified.")

        pdf_texts['strengths'] = strengths
        pdf_texts['weaknesses'] = weaknesses
        pdf_texts['profiles'] = profiles
    else:
        st.warning("No percentile data available for synopsis.")

    st.divider()

    # ===================================================================
    # Step 7: Set Piece Analysis
    # ===================================================================
    st.subheader(f"{selected_opponent} — Set Piece Analysis")

    try:
        set_piece_df = app.calculate_set_piece_metrics(
            season_events_df, season_id=selected_season_id,
        )
    except Exception:
        set_piece_df = pd.DataFrame()

    if not set_piece_df.empty and selected_opponent in set_piece_df.index:
        sp = set_piece_df.loc[selected_opponent]

        sp_data = {
            'Category': [
                'Corners', 'Free Kicks (Att 3rd)',
                'Throw-ins (Att 3rd)', 'Total Set Pieces',
            ],
            'xG For': [
                sp.get('xG from Corners', 0),
                sp.get('xG from Free Kicks', 0),
                sp.get('xG from Att Throw-ins', 0),
                sp.get('xG from Set Pieces', 0),
            ],
            'Goals For': [
                int(sp.get('Goals from Corners', 0)),
                int(sp.get('Goals from Free Kicks', 0)),
                int(sp.get('Goals from Att Throw-ins', 0)),
                int(sp.get('Goals from Set Pieces', 0)),
            ],
            'xG Conceded': [
                sp.get('xG Conceded Corners', 0),
                sp.get('xG Conceded Free Kicks', 0),
                sp.get('xG Conceded Att Throw-ins', 0),
                sp.get('xG Conceded Set Pieces', 0),
            ],
            'Goals Conceded': [
                int(sp.get('Goals Conceded Corners', 0)),
                int(sp.get('Goals Conceded Free Kicks', 0)),
                int(sp.get('Goals Conceded Att Throw-ins', 0)),
                int(sp.get('Goals Conceded Set Pieces', 0)),
            ],
        }
        sp_table = pd.DataFrame(sp_data)
        for c in ['xG For', 'xG Conceded']:
            sp_table[c] = sp_table[c].round(2)
        st.dataframe(sp_table, use_container_width=True, hide_index=True)
        pdf_texts['set_piece_table'] = sp_table

        col_l, col_r = st.columns(2)
        with col_l:
            try:
                with MPL_LOCK:
                    fig = app.plot_corner_analysis(
                        season_events_df, selected_opponent, 'left',
                    )
                    st.pyplot(fig)
                    pdf_figures['corner_left'] = _fig_to_png_bytes(fig)
                    plt.close(fig)
            except Exception as e:
                st.caption(f"Could not render left corner analysis: {e}")

        with col_r:
            try:
                with MPL_LOCK:
                    fig = app.plot_corner_analysis(
                        season_events_df, selected_opponent, 'right',
                    )
                    st.pyplot(fig)
                    pdf_figures['corner_right'] = _fig_to_png_bytes(fig)
                    plt.close(fig)
            except Exception as e:
                st.caption(f"Could not render right corner analysis: {e}")

        # Set piece scatter plots (league context)
        st.markdown("**Set Piece Efficiency vs League**")

        scatter_pairs = [
            ('xG per Corner', 'Goals per Corner',
             'Corner Attacking Efficiency'),
            ('xG Conceded per Corner', 'Goals Conceded per Corner',
             'Corner Defensive Vulnerability'),
            ('xG from Att Throw-ins', 'Goals from Att Throw-ins',
             'Attacking 3rd Throw-in Efficiency'),
            ('Long Throws', 'xG from Att Throw-ins',
             'Long Throws vs Throw-in xG'),
        ]

        for pair_idx in range(0, len(scatter_pairs), 2):
            cols = st.columns(2)
            for ci, col in enumerate(cols):
                si = pair_idx + ci
                if si >= len(scatter_pairs):
                    break
                x_met, y_met, scatter_title = scatter_pairs[si]
                with col:
                    try:
                        with MPL_LOCK:
                            fig = app.plot_custom_scatter(
                                set_piece_df, x_met, y_met,
                            )
                            fig.axes[0].set_title(scatter_title,
                                                  fontsize=14, weight='bold')
                            st.pyplot(fig)
                            pdf_figures[f'sp_scatter_{si}'] = (
                                _fig_to_png_bytes(fig))
                            plt.close(fig)
                    except Exception as e:
                        st.caption(f"Could not render {scatter_title}: {e}")

    else:
        st.info("No set piece data available.")

    st.divider()

    # ===================================================================
    # Step 8: Season Form & Shot Maps
    # ===================================================================
    st.subheader(f"{selected_opponent} — Season Form & Shot Maps")

    form_results = _get_team_recent_form(
        season_matches_df, selected_opponent, selected_season_id, n=5,
    )

    if form_results:
        form_str = ' '.join(
            f":{'green' if r['result'] == 'W' else 'red' if r['result'] == 'L' else 'orange'}[{r['result']}]"
            for r in form_results
        )
        st.markdown(f"**Last {len(form_results)} results:** {form_str}")

        ft = pd.DataFrame(form_results)[
            ['date', 'opponent', 'score', 'result', 'home_away']
        ]
        ft.columns = ['Date', 'Opponent', 'Score', 'Result', 'H/A']
        ft['Date'] = pd.to_datetime(ft['Date']).dt.strftime('%d %b %Y')
        st.dataframe(ft, use_container_width=True, hide_index=True)
        pdf_texts['form_results'] = form_results

    # xG history
    try:
        xg_hist = app.calculate_xg_history_data(
            raw_events_df, matches_summary_df,
        )
        if not xg_hist.empty:
            with MPL_LOCK:
                fig = app.plot_match_xg_history(xg_hist, selected_opponent)
                st.pyplot(fig)
                pdf_figures['xg_history'] = _fig_to_png_bytes(fig)
                plt.close(fig)
    except Exception as e:
        st.caption(f"Could not render xG history: {e}")

    # Shot maps
    col_sf, col_sa = st.columns(2)

    with col_sf:
        st.markdown("**Shots For**")
        try:
            with MPL_LOCK:
                fig = app.create_season_shotmap(
                    season_events_df, selected_opponent,
                )
                st.pyplot(fig, use_container_width=True)
                pdf_figures['shotmap_for'] = _fig_to_png_bytes(fig)
                plt.close(fig)
        except Exception as e:
            st.caption(f"Could not render shot map: {e}")

    with col_sa:
        st.markdown("**Shots Conceded**")
        try:
            with MPL_LOCK:
                fig = app.create_season_shots_against_shotmap(
                    season_events_df, season_matches_df, selected_opponent,
                )
                st.pyplot(fig, use_container_width=True)
                pdf_figures['shotmap_against'] = _fig_to_png_bytes(fig)
                plt.close(fig)
        except Exception as e:
            st.caption(f"Could not render shots against map: {e}")

    st.divider()

    # ===================================================================
    # Step 8b: Tactical Zone Analysis (Wyscout-style)
    # ===================================================================
    st.subheader(f"{selected_opponent} — Tactical Zone Analysis")

    # Average Player Positions (season) — restricted to projected XI (exactly 11)
    st.markdown("**Average Player Positions (Season)**")
    try:
        with MPL_LOCK:
            fig = pv.plot_average_positions(
                season_events_df, selected_opponent,
                player_names=xi_names if xi_names else None,
            )
            st.pyplot(fig)
            pdf_figures['avg_positions'] = _fig_to_png_bytes(fig)
            plt.close(fig)
    except Exception as e:
        st.caption(f"Could not render average positions: {e}")

    # Defensive Structure
    st.markdown("**Defensive Structure**")
    try:
        with MPL_LOCK:
            fig = pv.plot_defensive_structure(season_events_df, selected_opponent,
                                              league_events_df=season_events_df)
            st.pyplot(fig, use_container_width=True)
            pdf_figures['defensive_structure'] = _fig_to_png_bytes(fig)
            plt.close(fig)
    except Exception as e:
        st.caption(f"Could not render defensive structure: {e}")

    # Recovery / Loss zone heatmaps side by side
    col_rz, col_lz = st.columns(2)
    with col_rz:
        st.markdown("**Recovery Zones vs League**")
        try:
            with MPL_LOCK:
                fig = pv.plot_zone_heatmap(
                    season_events_df, selected_opponent, 'recovery',
                    league_events_df=season_events_df,
                )
                st.pyplot(fig)
                pdf_figures['zone_recovery'] = _fig_to_png_bytes(fig)
                plt.close(fig)
        except Exception as e:
            st.caption(f"Could not render: {e}")

    with col_lz:
        st.markdown("**Loss Zones vs League**")
        try:
            with MPL_LOCK:
                fig = pv.plot_zone_heatmap(
                    season_events_df, selected_opponent, 'loss',
                    league_events_df=season_events_df,
                )
                st.pyplot(fig)
                pdf_figures['zone_loss'] = _fig_to_png_bytes(fig)
                plt.close(fig)
        except Exception as e:
            st.caption(f"Could not render: {e}")

    # Shot Assists + Dribbles in Final Third
    st.markdown("**Shot Assists & Dribbles in Final Third**")
    try:
        with MPL_LOCK:
            fig = pv.plot_shot_assists_and_dribbles(season_events_df, selected_opponent)
            st.pyplot(fig)
            pdf_figures['shot_assists_dribbles'] = _fig_to_png_bytes(fig)
            plt.close(fig)
    except Exception as e:
        st.caption(f"Could not render shot assists & dribbles: {e}")

    st.divider()

    # ===================================================================
    # Step 9: Key Takeaways
    # ===================================================================
    st.subheader("Key Takeaways")

    takeaways = _generate_key_takeaways(
        selected_opponent, strengths, weaknesses, profiles,
        form_results, key_players, set_piece_df if not set_piece_df.empty else None,
        stats_df_pct,
    )

    for t in takeaways:
        st.markdown(f"- {t}")

    pdf_texts['takeaways'] = takeaways

    st.divider()

    # ===================================================================
    # Step 10: PDF Download
    # ===================================================================
    gw = fixture_info['gameweek'] if fixture_info else '?'
    m_date = (
        fixture_info['date'].strftime('%d %b %Y')
        if fixture_info and pd.notna(fixture_info.get('date'))
        else datetime.date.today().strftime('%d %b %Y')
    )

    if st.button("Generate PDF Report", type="primary"):
        try:
            from generate_pdf import generate_opposition_report_pdf

            with st.spinner("Generating PDF..."):
                pdf_bytes = generate_opposition_report_pdf(
                    opponent_name=selected_opponent,
                    match_date=m_date,
                    gameweek=gw,
                    figures=pdf_figures,
                    texts=pdf_texts,
                )

            fname = (
                f"Opposition_Report_"
                f"{selected_opponent.replace(' ', '_')}_GW{gw}.pdf"
            )
            st.download_button(
                label="Download PDF Report",
                data=pdf_bytes,
                file_name=fname,
                mime="application/pdf",
            )
        except ImportError:
            st.warning("PDF generation not available (fpdf2 not installed).")
        except Exception as e:
            st.error(f"PDF generation failed: {e}")
