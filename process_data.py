# ==============================================================================
# SECTION 1: IMPORTS
# ==============================================================================
import pandas as pd
import numpy as np
import datetime
import requests
from requests.auth import HTTPBasicAuth
from tqdm import tqdm
import os
import pickle
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ==============================================================================
# SECTION 2: DATA FETCHING FUNCTIONS
# ==============================================================================

def fetch_match_ids(username, password, competition_id, season_id):
    """Fetches all match IDs for a specific season by querying the competition endpoint."""
    url = f"https://apirest.wyscout.com/v3/competitions/{competition_id}/matches"
    auth = HTTPBasicAuth(username, password)
    print(f"Attempting to fetch matches for competitionId: {competition_id}...")
    try:
        r = requests.get(url, auth=auth, timeout=15)
        if r.status_code == 200:
            all_matches_data = r.json().get("matches", [])
            season_matches = [m for m in all_matches_data if m.get("seasonId") == season_id]
            match_ids = [m['matchId'] for m in season_matches]
            print(f"✅ Found {len(match_ids)} matches for seasonId {season_id}.")
            return match_ids
        else:
            print(f"❌ FAILED to get matches. Status: {r.status_code}, Response: {r.text}")
            return []
    except requests.exceptions.RequestException as e:
        print(f"❌ An error occurred: {e}")
        return []

def fetch_events(username, password, match_ids):
    """Fetches all event data for a given list of match IDs with retries."""
    base_url_v3 = "https://apirest.wyscout.com/v3"
    auth = HTTPBasicAuth(username, password)
    all_events = []
    session = requests.Session()
    retry_strategy = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    print(f"\nFetching events for {len(match_ids)} matches...")
    for match_id in tqdm(match_ids, desc="Fetching Events"):
        url = f"{base_url_v3}/matches/{match_id}/events"
        try:
            r = session.get(url, auth=auth, timeout=30)
            r.raise_for_status()
            all_events.extend(r.json().get('events', []))
        except requests.exceptions.RequestException as e:
            print(f"  -> ⚠️ Failed to fetch match {match_id} after multiple retries: {e}")
    if not all_events: return pd.DataFrame()
    events_df = pd.json_normalize(all_events)
    print(f"\n✅ Retrieved {len(events_df)} total events.")
    return events_df

# ==============================================================================
# SECTION 3: DATA PROCESSING FUNCTIONS
# ==============================================================================

def create_match_summary(events_df):
    """Creates a summary DataFrame with details for each match."""
    print("Processing: Creating match summary...")
    matches_summary = []
    for match_id in tqdm(events_df['matchId'].unique(), desc="Summarizing Matches"):
        match_df = events_df[events_df['matchId'] == match_id].copy()
        teams = match_df['team.name'].unique()
        if len(teams) < 2: continue
        home_team, away_team = teams[0], teams[1]
        
        # --- FIX FOR DATES ---
        # Ensure timestamps are datetimes, then sort to find the earliest one
        match_df['matchTimestamp'] = pd.to_datetime(match_df['matchTimestamp'])
        match_df.sort_values(by='matchTimestamp', inplace=True)
        match_date = match_df['matchTimestamp'].iloc[0].strftime('%Y-%m-%d')
        # --- END FIX ---

        goals_df = match_df[match_df['shot.isGoal'] == True]
        goal_counts = goals_df['team.name'].value_counts()
        home_score = goal_counts.get(home_team, 0)
        away_score = goal_counts.get(away_team, 0)
        matches_summary.append({
            'matchId': match_id, 'home_team': home_team, 'away_team': away_team,
            'date': match_date, 'score': f"{home_score} - {away_score}"
        })
    return pd.DataFrame(matches_summary)

def calculate_team_corner_stats(events_df, matches_summary_df, selected_team):
    """Calculates detailed corner kick statistics for a single team for the whole season."""
    # (This function is from your corner kick logic)
    def is_short_corner_by_distance(start_x_w, start_y_w, end_x_w, end_y_w, threshold_m=20):
        PITCH_LENGTH_M, PITCH_WIDTH_M = 105.0, 68.0
        if pd.isna(start_x_w) or pd.isna(start_y_w) or pd.isna(end_x_w) or pd.isna(end_y_w): return False
        distance = np.sqrt(((end_x_w - start_x_w) * (PITCH_LENGTH_M / 100.0))**2 + ((end_y_w - start_y_w) * (PITCH_WIDTH_M / 100.0))**2)
        return distance <= threshold_m
    def map_location_to_zone(x, y):
        if pd.isna(x) or pd.isna(y): return 'Unknown'
        if x >= 94 and 36 <= y <= 64: return '6Y Box'
        if 88 <= x < 94 and 36 <= y <= 64: return 'Front Area'
        if x >= 90 and (21 <= y < 36 or 64 < y <= 79): return 'Near Post'
        if 83 <= x < 90 and (21 <= y < 36 or 64 < y <= 79): return 'Far Post'
        if 83 <= x < 88 and 36 <= y <= 64: return 'Middle Area'
        return 'Back Area'
    
    team_matches_df = matches_summary_df[(matches_summary_df['home_team'] == selected_team) | (matches_summary_df['away_team'] == selected_team)]
    total_matches = team_matches_df['matchId'].nunique()
    if total_matches == 0: return None
    
    events_df['is_corner'] = (events_df['type.primary'] == 'corner')
    corners_df = events_df[(events_df['is_corner']) & (events_df['team.name'] == selected_team)].copy()
    if corners_df.empty: return None

    corner_possession_ids = corners_df['possession.id'].dropna().unique()
    corner_possessions_df = events_df[events_df['possession.id'].isin(corner_possession_ids)].sort_values(by=['possession.id', 'possession.eventIndex'])

    processed_corners = []
    if not corner_possessions_df.empty:
        for possession_id, group in corner_possessions_df.groupby('possession.id'):
            corner_event = group[group['is_corner']].iloc[0]
            if corner_event['team.name'] != selected_team: continue
            
            corner_index = corner_event['possession.eventIndex']
            delivery_type = 'Direct Corner'
            if is_short_corner_by_distance(corner_event['location.x'], corner_event['location.y'], corner_event['pass.endLocation.x'], corner_event['pass.endLocation.y']):
                delivery_type = 'Short Corner'
            
            first_contact_event = group[group['possession.eventIndex'] == corner_index + 1].copy()
            first_contact_zone = 'Unknown'
            if not first_contact_event.empty:
                first_contact_zone = map_location_to_zone(first_contact_event.iloc[0]['location.x'], first_contact_event.iloc[0]['location.y'])
            
            shot_details = {
                '1st Contact': {'shots': 0, 'goals': 0, 'xg': 0}, '2nd Contact': {'shots': 0, 'goals': 0, 'xg': 0},
                '2nd Phase': {'shots': 0, 'goals': 0, 'xg': 0},
            }
            if not first_contact_event.empty and pd.notna(first_contact_event.iloc[0]['shot.xg']):
                event = first_contact_event.iloc[0]
                shot_details['1st Contact'].update({'shots': 1, 'goals': 1 if event['shot.isGoal'] == True else 0, 'xg': event['shot.xg']})
            
            second_contact_event = group[group['possession.eventIndex'] == corner_index + 2].copy()
            if not second_contact_event.empty and pd.notna(second_contact_event.iloc[0]['shot.xg']):
                event = second_contact_event.iloc[0]
                shot_details['2nd Contact'].update({'shots': 1, 'goals': 1 if event['shot.isGoal'] == True else 0, 'xg': event['shot.xg']})
            
            second_phase_events = group[group['possession.eventIndex'] > corner_index + 2].copy()
            if not second_phase_events.empty:
                shots_in_phase = second_phase_events.dropna(subset=['shot.xg']).copy()
                if not shots_in_phase.empty:
                    shots_in_phase['isGoal'] = (shots_in_phase['shot.isGoal'] == True)
                    shot_details['2nd Phase'].update({'shots': len(shots_in_phase), 'goals': shots_in_phase['isGoal'].sum(), 'xg': shots_in_phase['shot.xg'].sum()})
            
            processed_corners.append({
                'possession.id': possession_id, 'delivery_type': delivery_type, 'first_contact_zone': first_contact_zone,
                'shots_1st_contact': shot_details['1st Contact']['shots'], 'goals_1st_contact': shot_details['1st Contact']['goals'], 'xg_1st_contact': shot_details['1st Contact']['xg'],
                'shots_2nd_contact': shot_details['2nd Contact']['shots'], 'goals_2nd_contact': shot_details['2nd Contact']['goals'], 'xg_2nd_contact': shot_details['2nd Contact']['xg'],
                'shots_2nd_phase': shot_details['2nd Phase']['shots'], 'goals_2nd_phase': shot_details['2nd Phase']['goals'], 'xg_2nd_phase': shot_details['2nd Phase']['xg'],
            })
    
    results_df = pd.DataFrame(processed_corners)
    if results_df.empty: return None

    # Build the final DataFrame
    total_corners = len(results_df)
    total_xg = results_df[['xg_1st_contact', 'xg_2nd_contact', 'xg_2nd_phase']].sum().sum()
    corners_per_match = total_corners / total_matches
    xg_per_90 = total_xg / total_matches
    xg_per_corner = total_xg / total_corners if total_corners > 0 else 0
    first_contact_pct = results_df['first_contact_zone'].value_counts(normalize=True) * 100
    total_goals = results_df[['goals_1st_contact', 'goals_2nd_contact', 'goals_2nd_phase']].sum().sum()
    total_shots = results_df[['shots_1st_contact', 'shots_2nd_contact', 'shots_2nd_phase']].sum().sum()
    delivery_counts = results_df['delivery_type'].value_counts(normalize=True) * 100

    corner_stats = {
        selected_team: {
            "Corners/Match": f"{corners_per_match:.1f}", "Total Corners": total_corners,
            "Total xG": f"{total_xg:.3f}", "xG/90min": f"{xg_per_90:.2f}", "xG/Corner": f"{xg_per_corner:.3f}",
            "1st Contact Zone: 6Y Box %": f"{first_contact_pct.get('6Y Box', 0):.1f}",
            "1st Contact Zone: Front Area %": f"{first_contact_pct.get('Front Area', 0):.1f}",
            "1st Contact Zone: Near Post %": f"{first_contact_pct.get('Near Post', 0):.1f}",
            "Total Goals": int(total_goals), "Total Shots": int(total_shots),
            "Short Corner %": f"{delivery_counts.get('Short Corner', 0):.1f}",
            "Direct Corner %": f"{delivery_counts.get('Direct Corner', 0):.1f}",
        }
    }
    return pd.DataFrame(corner_stats).rename_axis("Corner Kicks")


def calculate_match_data(match_df, home_team, away_team):
    """Calculates all team and player stats for a single match."""
    
    # --- NESTED FUNCTION: TEAM STATS ---
    def _calculate_team_stats(match_df, home_team, away_team):
        teams = [home_team, away_team]
        all_tables = {}
        
        # Pre-computation
        match_df['matchTimestamp'] = pd.to_datetime(match_df['matchTimestamp'])
        match_df['possession.duration_sec'] = pd.to_numeric(match_df['possession.duration'].str.replace('s', ''), errors='coerce')
        
        PENALTY_AREA_X = 83
        PENALTY_AREA_Y1, PENALTY_AREA_Y2 = (21, 79)
        DEFENSIVE_THIRD_X = 33.3
        ATTACKING_THIRD_X = 66.6
        FINAL_THIRD_X = 66
        RECOVERY_EVENTS = ['interception', 'duel', 'clearance', 'goalkeeper_exit']
        SET_PIECE_EVENTS = ['corner', 'free_kick', 'goal_kick', 'throw_in', 'penalty']

        # --- 2. Calculate "General" Stats ---
        general_stats = {}
        shots_df = match_df[match_df['type.primary'] == 'shot'].copy()
        infractions_df = match_df[match_df['type.primary'] == 'infraction'].copy()
        offsides_df = match_df[match_df['type.primary'] == 'offside'].copy()
        free_kicks_df = match_df[match_df['type.primary'] == 'free_kick'].copy()
        corners_df = match_df[match_df['type.primary'] == 'corner'].copy()

        for team in teams:
            team_shots = shots_df[shots_df['team.name'] == team].copy()
            team_shots.loc[:, 'distance'] = np.sqrt((100 - team_shots['location.x'])**2 + (50 - team_shots['location.y'])**2)
            goals = team_shots[team_shots['shot.isGoal'] == True].shape[0]
            xg = team_shots['shot.xg'].sum()
            total_shots = team_shots.shape[0]
            on_target = team_shots[team_shots['shot.onTarget'] == True].shape[0]
            shots_in_box = team_shots[(team_shots['location.x'] >= PENALTY_AREA_X) & (team_shots['location.y'].between(PENALTY_AREA_Y1, PENALTY_AREA_Y2))]
            shots_in_box_on_target = shots_in_box[shots_in_box['shot.onTarget'] == True].shape[0]
            shots_out_box = team_shots[team_shots['location.x'] < PENALTY_AREA_X]
            shots_out_box_on_target = shots_out_box[shots_out_box['shot.onTarget'] == True].shape[0]
            avg_shot_dist = team_shots['distance'].mean()
            corners = corners_df[corners_df['team.name'] == team].shape[0]
            team_free_kicks = free_kicks_df[free_kicks_df['team.name'] == team].copy()
            team_free_kicks['distance_to_goal'] = np.sqrt((100 - team_free_kicks['location.x'])**2 + (50 - team_free_kicks['location.y'])**2)
            attacking_free_kicks = team_free_kicks[team_free_kicks['distance_to_goal'] <= 45].shape[0]
            offsides = offsides_df[offsides_df['team.name'] == team].shape[0]
            fouls_committed = infractions_df[infractions_df['team.name'] == team].shape[0]
            yellow_cards = infractions_df[(infractions_df['team.name'] == team) & (infractions_df['infraction.yellowCard'] == True)].shape[0]
            red_cards = infractions_df[(infractions_df['team.name'] == team) & (infractions_df['infraction.redCard'] == True)].shape[0]
            general_stats[team] = {
                "Goals": goals, "xG": round(xg, 2), "Shots / on target": f"{total_shots}/{on_target}",
                "From penalty area / on target": f"{shots_in_box.shape[0]}/{shots_in_box_on_target}",
                "Outside penalty area / on target": f"{shots_out_box.shape[0]}/{shots_out_box_on_target}",
                "Average shot distance (m)": round(avg_shot_dist, 1), "Corners": corners,
                "Attacking free kicks": attacking_free_kicks, "Offsides": offsides,
                "Fouls committed": fouls_committed, "Yellow / red cards": f"{yellow_cards}/{red_cards}"
            }
        home_fouls_committed = general_stats[home_team].pop('Fouls committed')
        away_fouls_committed = general_stats[away_team].pop('Fouls committed')
        general_stats[home_team]['Fouls committed / suffered'] = f"{home_fouls_committed}/{away_fouls_committed}"
        general_stats[away_team]['Fouls committed / suffered'] = f"{away_fouls_committed}/{home_fouls_committed}"
        general_stats_df = pd.DataFrame(general_stats).rename_axis("General")
        all_tables["General"] = general_stats_df.reindex([
            "Goals", "xG", "Shots / on target", "From penalty area / on target",
            "Outside penalty area / on target", "Average shot distance (m)",
            "Corners", "Attacking free kicks", "Offsides",
            "Fouls committed / suffered", "Yellow / red cards"
        ])

        # --- 3. "Attacks" Stats ---
        attack_stats = {}
        for team in teams:
            team_match_events = match_df[match_df['team.name'] == team]
            possessions_grouped = team_match_events.groupby('possession.id')['location.x']
            final_third_entries = possessions_grouped.filter(lambda x: x.min() < FINAL_THIRD_X and x.max() >= FINAL_THIRD_X).nunique()
            possessions_with_box_event = team_match_events[(team_match_events['location.x'] >= PENALTY_AREA_X) & (team_match_events['location.y'].between(PENALTY_AREA_Y1, PENALTY_AREA_Y2))]['possession.id'].unique()
            box_entries = len(possessions_with_box_event)
            first_events = team_match_events.loc[team_match_events.groupby('possession.id')['matchTimestamp'].idxmin()].copy()
            recoveries = first_events[first_events['type.primary'].isin(RECOVERY_EVENTS)].copy()
            recoveries['start_distance'] = np.sqrt((100 - recoveries['location.x'])**2 + (50 - recoveries['location.y'])**2)
            counter_candidates = recoveries[recoveries['start_distance'] >= 40].copy()
            counter_attacks = 0
            if not counter_candidates.empty:
                candidate_events = team_match_events[team_match_events['possession.id'].isin(counter_candidates['possession.id'])].copy()
                candidate_events = candidate_events.merge(counter_candidates[['possession.id', 'matchTimestamp']], on='possession.id', suffixes=('', '_start'))
                candidate_events['time_in_possession'] = (candidate_events['matchTimestamp'] - candidate_events['matchTimestamp_start']).dt.total_seconds()
                fast_events = candidate_events[candidate_events['time_in_possession'] <= 5].copy()
                fast_events['distance_to_goal'] = np.sqrt((100 - fast_events['location.x'])**2 + (50 - fast_events['location.y'])**2)
                min_distances = fast_events.groupby('possession.id')['distance_to_goal'].min().reset_index().rename(columns={'distance_to_goal': 'min_distance'})
                progression_df = counter_candidates.merge(min_distances, on='possession.id')
                progression_df['progression'] = progression_df['start_distance'] - progression_df['min_distance']
                counter_attacks = progression_df[progression_df['progression'] >= 30].shape[0]
            attack_stats[team] = {
                "Positional attacks": match_df.drop_duplicates(subset='possession.id')[(match_df['possession.team.name'] == team) & (match_df['possession.attack.flank'].isin(['central', 'left', 'right']))].shape[0],
                "Final 1/3 Entries": final_third_entries, "Box Entries": box_entries, "Counterattacks": counter_attacks,
            }
        all_tables["Attacks"] = pd.DataFrame(attack_stats).rename_axis("Attacks")

        # --- 4. "Defence" Stats ---
        defence_stats = {}
        for team in teams:
            opponent = away_team if team == home_team else home_team
            interceptions = match_df[(match_df['type.primary'] == 'interception') & (match_df['team.name'] == team)].shape[0]
            clearances = match_df[(match_df['type.primary'] == 'clearance') & (match_df['team.name'] == team)].shape[0]
            in_press_zone = match_df['location.x'] >= 40
            opponent_passes_df = match_df[(match_df['team.name'] == opponent) & (match_df['type.primary'] == 'pass') & in_press_zone]
            num_opponent_passes = opponent_passes_df.shape[0]
            team_def_actions_df = match_df[(match_df['team.name'] == team) & in_press_zone]
            fouls = team_def_actions_df[team_def_actions_df['type.primary'] == 'infraction'].shape[0]
            def_interceptions = team_def_actions_df[team_def_actions_df['type.primary'] == 'interception'].shape[0]
            duels_in_zone_df = team_def_actions_df[team_def_actions_df['type.primary'] == 'duel']
            defensive_duels_in_zone = duels_in_zone_df[duels_in_zone_df['groundDuel.duelType'] == 'defensive_duel']
            won_defensive_duels = defensive_duels_in_zone[(defensive_duels_in_zone['groundDuel.recoveredPossession'] == True) | (defensive_duels_in_zone['groundDuel.stoppedProgress'] == True)].shape[0]
            sliding_tackles = duels_in_zone_df[duels_in_zone_df['groundDuel.duelType'].astype(str).str.contains('sliding_tackle', na=False)].shape[0]
            num_defensive_actions = fouls + def_interceptions + won_defensive_duels + sliding_tackles
            ppda = round(num_opponent_passes / num_defensive_actions, 1) if num_defensive_actions > 0 else 0
            defence_stats[team] = {
                "Interceptions": interceptions, "Clearances": clearances,
                "Passes allowed per def. action (PPDA)": ppda,
            }
        all_tables["Defence"] = pd.DataFrame(defence_stats).rename_axis("Defence")

        # --- 5. "Transitions" Stats ---
        transitions_stats = {}
        match_df['next_possession.id'] = match_df['possession.id'].shift(-1)
        possession_changes = match_df[match_df['possession.id'] != match_df['next_possession.id']]
        losses_df_base = possession_changes[possession_changes['infraction.type'] != 'foul_suffered'].copy()
        unsuccessful_pass_mask = (losses_df_base['type.primary'] == 'pass') & (losses_df_base['pass.accurate'] == False)
        losses_df_base.loc[unsuccessful_pass_mask, 'location.x'] = losses_df_base.loc[unsuccessful_pass_mask, 'pass.endLocation.x']
        losses_df_base.loc[unsuccessful_pass_mask, 'location.y'] = losses_df_base.loc[unsuccessful_pass_mask, 'pass.endLocation.y']
        for team in teams:
            interceptions_df = match_df[(match_df['type.primary'] == 'interception') & (match_df['team.name'] == team)]
            clearances_df = match_df[(match_df['type.primary'] == 'clearance') & (match_df['team.name'] == team)]
            won_duels_df = match_df[(match_df['type.primary'] == 'duel') & (match_df['team.name'] == team) & (match_df['groundDuel.recoveredPossession'] == True)]
            recoveries_df = pd.concat([interceptions_df, clearances_df, won_duels_df])
            total_recoveries = recoveries_df.shape[0]
            low_recoveries = recoveries_df[recoveries_df['location.x'] <= DEFENSIVE_THIRD_X].shape[0]
            mid_recoveries = recoveries_df[recoveries_df['location.x'].between(DEFENSIVE_THIRD_X, ATTACKING_THIRD_X)].shape[0]
            high_recoveries = recoveries_df[recoveries_df['location.x'] > ATTACKING_THIRD_X].shape[0]
            opponent_half_recoveries = recoveries_df[recoveries_df['location.x'] > 50].shape[0]
            team_losses = losses_df_base[losses_df_base['team.name'] == team]
            total_losses = team_losses.shape[0]
            low_losses = team_losses[team_losses['location.x'] <= DEFENSIVE_THIRD_X].shape[0]
            mid_losses = team_losses[team_losses['location.x'].between(DEFENSIVE_THIRD_X, ATTACKING_THIRD_X)].shape[0]
            high_losses = team_losses[team_losses['location.x'] > ATTACKING_THIRD_X].shape[0]
            transitions_stats[team] = {
                "Recoveries / low / medium / high": f"{total_recoveries}/{low_recoveries}/{mid_recoveries}/{high_recoveries}",
                "Opponent half recoveries": opponent_half_recoveries,
                "Losses / low / medium / high": f"{total_losses}/{low_losses}/{mid_losses}/{high_losses}",
            }
        all_tables["Transitions"] = pd.DataFrame(transitions_stats).rename_axis("Transitions")

        # --- 6. "Duels" Stats ---
        duels_stats = {}
        all_duels_df = match_df[match_df['type.primary'] == 'duel'].copy()
        total_possession_time = match_df.drop_duplicates(subset='possession.id').groupby('possession.team.name')['possession.duration_sec'].sum()
        for team in teams:
            opponent = away_team if team == home_team else home_team
            team_duels = all_duels_df[all_duels_df['team.name'] == team]
            won_ground_duels_df = team_duels[(team_duels['groundDuel.keptPossession'] == True) | (team_duels['groundDuel.recoveredPossession'] == True)]
            aerial_duels = team_duels[team_duels['type.secondary'].astype(str).str.contains('aerial', na=False)]
            won_aerial_duels_df = aerial_duels[aerial_duels['aerialDuel.firstTouch'] == True]
            total_duels_won = won_ground_duels_df.shape[0] + won_aerial_duels_df.shape[0]
            offensive_duels = team_duels[team_duels['groundDuel.duelType'] == 'offensive_duel']
            won_offensive_duels = offensive_duels[(offensive_duels['groundDuel.keptPossession'] == True) | (offensive_duels['groundDuel.recoveredPossession'] == True)].shape[0]
            defensive_duels = team_duels[team_duels['groundDuel.duelType'] == 'defensive_duel']
            won_defensive_duels = defensive_duels[(defensive_duels['groundDuel.recoveredPossession'] == True) | (defensive_duels['groundDuel.stoppedProgress'] == True)].shape[0]
            dribbles = team_duels[team_duels['groundDuel.takeOn'] == True]
            successful_dribbles = dribbles[dribbles['groundDuel.progressedWithBall'] == True].shape[0]
            defensive_duels_ci = defensive_duels.shape[0]
            interceptions_ci = match_df[(match_df['type.primary'] == 'interception') & (match_df['team.name'] == team)].shape[0]
            total_def_actions = defensive_duels_ci + interceptions_ci
            opponent_possession_minutes = total_possession_time.get(opponent, 0) / 60
            challenge_intensity = round(total_def_actions / opponent_possession_minutes, 1) if opponent_possession_minutes > 0 else 0
            duels_stats[team] = {
                "Total duels / won": f"{team_duels.shape[0]}/{total_duels_won}",
                "Offensive duels / won": f"{offensive_duels.shape[0]}/{won_offensive_duels}",
                "Defensive duels / won": f"{defensive_duels.shape[0]}/{won_defensive_duels}",
                "Aerial duels / won": f"{aerial_duels.shape[0]}/{won_aerial_duels_df.shape[0]}",
                "Challenge intensity": challenge_intensity,
                "Dribbles / successful": f"{dribbles.shape[0]}/{successful_dribbles}",
            }
        all_tables["Duels"] = pd.DataFrame(duels_stats).rename_axis("Duels")

        # --- 7. "Possession" Stats ---
        possession_stats = {}
        home_possessions = match_df[match_df['possession.team.name'] == home_team].drop_duplicates(subset='possession.id')
        away_possessions = match_df[match_df['possession.team.name'] == away_team].drop_duplicates(subset='possession.id')
        home_time_sec = home_possessions['possession.duration_sec'].sum()
        away_time_sec = away_possessions['possession.duration_sec'].sum()
        total_time_in_possession = home_time_sec + away_time_sec
        total_match_duration_sec = (match_df['matchTimestamp'].max() - match_df['matchTimestamp'].min()).total_seconds()
        dead_time_sec = total_match_duration_sec - total_time_in_possession
        for team in teams:
            team_possessions_df = match_df[match_df['possession.team.name'] == team]
            unique_team_possessions = team_possessions_df.drop_duplicates(subset='possession.id')
            time_sec = unique_team_possessions['possession.duration_sec'].sum()
            possession_pct = round(time_sec / total_time_in_possession * 100) if total_time_in_possession > 0 else 0
            pure_time_str = str(datetime.timedelta(seconds=int(time_sec)))[2:]
            num_possessions = unique_team_possessions.shape[0]
            reaching_half = team_possessions_df.groupby('possession.id')['location.x'].filter(lambda x: x.max() > 50).nunique()
            possessions_in_box_ids = team_possessions_df[(team_possessions_df['location.x'] >= PENALTY_AREA_X) & (team_possessions_df['location.y'].between(PENALTY_AREA_Y1, PENALTY_AREA_Y2))]['possession.id'].unique()
            reaching_box = len(possessions_in_box_ids)
            avg_duration_sec = unique_team_possessions['possession.duration_sec'].mean()
            avg_duration_str = str(datetime.timedelta(seconds=int(avg_duration_sec)))[-5:]
            possession_stats[team] = {
                "Possession %": possession_pct, "Pure possession time": pure_time_str,
                "Number of possessions": num_possessions,
                "Possessions reaching opponent half": f"{reaching_half}",
                "Possessions reaching opponent penalty area": f"{reaching_box}",
                "Average possession duration": avg_duration_str,
            }
        possession_stats_df = pd.DataFrame(possession_stats).rename_axis("Possession")
        dead_time_str = str(datetime.timedelta(seconds=int(dead_time_sec)))[2:]
        possession_stats_df.loc['Dead time'] = [dead_time_str, dead_time_str]
        all_tables["Possession"] = possession_stats_df

        # --- 8. "Open play possessions" Stats ---
        open_play_stats = {}
        first_events_df = match_df.loc[match_df.groupby('possession.id')['matchTimestamp'].idxmin()]
        open_play_possessions = first_events_df[~first_events_df['type.primary'].isin(SET_PIECE_EVENTS)]
        for team in teams:
            team_open_play = open_play_possessions[open_play_possessions['possession.team.name'] == team]
            total = team_open_play.shape[0]
            short = team_open_play[team_open_play['possession.duration_sec'] <= 10].shape[0]
            medium = team_open_play[team_open_play['possession.duration_sec'].between(10, 20)].shape[0]
            long = team_open_play[team_open_play['possession.duration_sec'].between(20, 45)].shape[0]
            very_long = team_open_play[team_open_play['possession.duration_sec'] > 45].shape[0]
            open_play_stats[team] = {
                "Total": total, "Short (0-10 sec)": short, "Medium (10-20 sec)": medium,
                "Long (20-45 sec)": long, "Very long (45+ sec)": very_long,
            }
        all_tables["Open play possessions"] = pd.DataFrame(open_play_stats).rename_axis("Open play possessions")

        # --- 9. "Passes" Stats ---
        passes_stats = {}
        all_passes_df = match_df[match_df['type.primary'] == 'pass'].copy()
        for team in teams:
            team_passes = all_passes_df[all_passes_df['team.name'] == team].copy()
            accurate_passes = team_passes[team_passes['pass.accurate'] == True]
            forward_passes = team_passes[team_passes['pass.endLocation.x'] > team_passes['location.x']]
            back_passes = team_passes[team_passes['pass.endLocation.x'] < team_passes['location.x']]
            lateral_passes = team_passes[team_passes['pass.endLocation.x'] == team_passes['location.x']]
            prog_cond1 = (team_passes['location.x'] < 60) & (team_passes['pass.endLocation.x'] >= 60)
            prog_cond2 = (team_passes['location.x'] >= 60) & (team_passes['pass.endLocation.x'] >= 60) & (team_passes['pass.length'] >= 10)
            progressive_passes = team_passes[prog_cond1 | prog_cond2]
            long_passes = team_passes[team_passes['pass.length'] > 40]
            passes_to_final_third = team_passes[(team_passes['location.x'] < ATTACKING_THIRD_X) & (team_passes['pass.endLocation.x'] >= ATTACKING_THIRD_X)]
            passes_to_box = team_passes[(team_passes['pass.endLocation.x'] >= PENALTY_AREA_X) & (team_passes['pass.endLocation.y'].between(PENALTY_AREA_Y1, PENALTY_AREA_Y2))]
            smart_passes = team_passes[team_passes['type.secondary'].astype(str).str.contains('smart_pass', na=False)]
            shot_assists = team_passes[team_passes['type.secondary'].astype(str).str.contains('shot_assist', na=False)]
            through_passes = team_passes[team_passes['type.secondary'].astype(str).str.contains('through_pass', na=False)]
            crosses = team_passes[team_passes['type.secondary'].astype(str).str.contains('cross', na=False)]
            low_crosses = crosses[crosses['pass.height'].astype(str).str.contains('low', na=False)].shape[0]
            high_crosses = crosses[crosses['pass.height'].astype(str).str.contains('high', na=False)].shape[0]
            team_passes['dist_to_goal_end'] = np.sqrt((100 - team_passes['pass.endLocation.x'])**2 + (50 - team_passes['pass.endLocation.y'])**2)
            deep_completions = team_passes[(team_passes['pass.accurate'] == True) & (team_passes['dist_to_goal_end'] <= 20) & (~team_passes['type.secondary'].astype(str).str.contains('cross', na=False))].shape[0]
            pure_possession_minutes = total_possession_time.get(team, 0) / 60
            match_tempo = round(team_passes.shape[0] / pure_possession_minutes, 1) if pure_possession_minutes > 0 else 0
            passes_stats[team] = {
                "Total passes / accurate": f"{team_passes.shape[0]}/{accurate_passes.shape[0]}",
                "Forward passes / accurate": f"{forward_passes.shape[0]}/{forward_passes[forward_passes['pass.accurate'] == True].shape[0]}",
                "Back passes / accurate": f"{back_passes.shape[0]}/{back_passes[back_passes['pass.accurate'] == True].shape[0]}",
                "Lateral passes / accurate": f"{lateral_passes.shape[0]}/{lateral_passes[lateral_passes['pass.accurate'] == True].shape[0]}",
                "Progressive passes / accurate": f"{progressive_passes.shape[0]}/{progressive_passes[progressive_passes['pass.accurate'] == True].shape[0]}",
                "Long passes / accurate": f"{long_passes.shape[0]}/{long_passes[long_passes['pass.accurate'] == True].shape[0]}",
                "Passes to final third / accurate": f"{passes_to_final_third.shape[0]}/{passes_to_final_third[passes_to_final_third['pass.accurate'] == True].shape[0]}",
                "Avg pass to final third length (m)": round(passes_to_final_third['pass.length'].mean(), 1),
                "Passes to penalty area / accurate": f"{passes_to_box.shape[0]}/{passes_to_box[passes_to_box['pass.accurate'] == True].shape[0]}",
                "Smart passes / accurate": f"{smart_passes.shape[0]}/{smart_passes[smart_passes['pass.accurate'] == True].shape[0]}",
                "Shot assists": shot_assists.shape[0],
                "Through passes / accurate": f"{through_passes.shape[0]}/{through_passes[through_passes['pass.accurate'] == True].shape[0]}",
                "Crosses / accurate": f"{crosses.shape[0]}/{crosses[crosses['pass.accurate'] == True].shape[0]}",
                "Crosses: low / high / blocked": f"{low_crosses}/{high_crosses}/{'N/A'}",
                "Deep completions": deep_completions, "Match tempo": match_tempo,
                "Average pass length (m)": round(team_passes['pass.length'].mean(), 1),
            }
        all_tables["Passes"] = pd.DataFrame(passes_stats).rename_axis("Passes")
        
        return all_tables

    # --- NESTED FUNCTION: PLAYER STATS ---
    def _calculate_player_stats(match_df, home_team, away_team):
        
        def _calculate_stats_for_team(team_name, match_df):
            team_events = match_df[match_df['team.name'] == team_name].copy()
            if team_events.empty:
                return pd.DataFrame()
                
            starters = team_events['player.name'].dropna().unique()[:11]
            match_end_minute = match_df['minute'].max()
            player_minutes = {}
            for player in team_events['player.name'].dropna().unique():
                player_events_for_mins = team_events[team_events['player.name'] == player]
                if player_events_for_mins.empty:
                    continue
                if player in starters:
                    player_minutes[player] = player_events_for_mins['minute'].max()
                else:
                    minute_on = player_events_for_mins['minute'].min()
                    player_minutes[player] = match_end_minute - minute_on
            
            player_stats_list = []
            players = team_events['player.name'].dropna().unique()
            for player in players:
                player_events = team_events[team_events['player.name'] == player]
                if player_events.empty:
                    continue
                    
                player_shots = player_events[player_events['type.primary'] == 'shot']
                goals = player_shots[player_shots['shot.isGoal'] == True].shape[0]
                xg = player_shots['shot.xg'].sum()
                total_actions = player_events.shape[0]
                successful_actions = player_events[(player_events['pass.accurate'] == True) | (player_events['shot.onTarget'] == True)].shape[0]
                total_shots = player_shots.shape[0]
                shots_on_target = player_shots[player_shots['shot.onTarget'] == True].shape[0]
                player_passes = player_events[player_events['type.primary'] == 'pass']
                total_passes = player_passes.shape[0]
                accurate_passes = player_passes[player_passes['pass.accurate'] == True].shape[0]
                crosses = player_passes[player_passes['type.secondary'].astype(str).str.contains('cross', na=False)]
                accurate_crosses = crosses[crosses['pass.accurate'] == True].shape[0]
                player_duels = player_events[player_events['type.primary'] == 'duel']
                dribbles = player_duels[player_duels['groundDuel.takeOn'] == True]
                successful_dribbles = dribbles[dribbles['groundDuel.progressedWithBall'] == True].shape[0]
                total_duels = player_duels.shape[0]
                won_ground_duels = player_duels[(player_duels['groundDuel.keptPossession'] == True) | (player_duels['groundDuel.recoveredPossession'] == True)].shape[0]
                won_aerial_duels = player_duels[player_duels['aerialDuel.firstTouch'] == True].shape[0]
                duels_won = won_ground_duels + won_aerial_duels
                recoveries = player_events[player_events['type.primary'].isin(['interception', 'clearance'])].shape[0] + won_ground_duels
                recoveries_opp_half = player_events[(player_events['type.primary'].isin(['interception', 'clearance'])) & (player_events['location.x'] > 50)].shape[0]
                losses = player_passes[player_passes['pass.accurate'] == False].shape[0]
                losses_own_half = player_passes[(player_passes['pass.accurate'] == False) & (player_passes['location.x'] <= 50)].shape[0]
                touches_in_box = player_events[(player_events['type.primary'] == 'touch') & (player_events['location.x'] >= 83)].shape[0]
                offsides = player_events[player_events['type.primary'] == 'offside'].shape[0]
                yellow_cards = player_events[player_events['infraction.yellowCard'] == True].shape[0]
                red_cards = player_events[player_events['infraction.redCard'] == True].shape[0]

                player_stats_list.append({
                    "Player": player, "Minutes with Actions": f"{player_minutes.get(player, 0)}'", "Goals / xG": f"{goals}/{round(xg, 2) if xg > 0 else '-'}",
                    "Actions / successful": f"{total_actions}/{successful_actions}", "Shots / on target": f"{total_shots}/{shots_on_target}",
                    "Passes / accurate": f"{total_passes}/{accurate_passes}", "Crosses / accurate": f"{crosses.shape[0]}/{accurate_crosses}",
                    "Dribbles / successful": f"{dribbles.shape[0]}/{successful_dribbles}", "Duels / won": f"{total_duels}/{duels_won}",
                    "Losses / own half": f"{losses}/{losses_own_half}", "Recoveries / opponent half": f"{recoveries}/{recoveries_opp_half}",
                    "Touches in penalty area": touches_in_box if touches_in_box > 0 else "-", "Offsides": offsides if offsides > 0 else "-",
                    "Yellow / Red cards": f"{int(yellow_cards)}/{int(red_cards)}" if (yellow_cards > 0 or red_cards > 0) else "-",
                })
            return pd.DataFrame(player_stats_list).set_index('Player')
        
        home_stats = _calculate_stats_for_team(home_team, match_df)
        away_stats = _calculate_stats_for_team(away_team, match_df)
        return {'home': home_stats, 'away': away_stats}

    # --- EXECUTE AND RETURN ---
    team_stats = _calculate_team_stats(match_df, home_team, away_team)
    player_stats = _calculate_player_stats(match_df, home_team, away_team)
    
    return {'team_stats': team_stats, 'player_stats': player_stats}

# ==============================================================================
# SECTION 4: MAIN FUNCTION
# ==============================================================================
def main():
    """Main function to run the entire data processing pipeline."""
    wyscout_user = "ggm0zzt-jidg1g5bv-ofdye2m-huk6ii8kkd"
    wyscout_pass = ",Xzas52XAavPLHNK8sSJLJNhHEP!NY"
    competition_id = 43324
    season_id = 191782

    # --- 1. Fetch Data ---
    match_ids = fetch_match_ids(wyscout_user, wyscout_pass, competition_id, season_id)
    if not match_ids: 
        print("No match IDs found. Exiting.")
        return
        
    raw_events_df = fetch_events(wyscout_user, wyscout_pass, match_ids)
    if raw_events_df.empty: 
        print("No event data fetched. Exiting.")
        return
    
    # Save the raw events for the app to use
    raw_events_df.to_parquet('raw_events.parquet', index=False)
    print("✅ Raw event data saved.")

    # --- 2. Create and Save High-Level Summary ---
    matches_summary_df = create_match_summary(raw_events_df)
    matches_summary_df.to_parquet('matches_summary.parquet', index=False)
    print("✅ Match summary saved.")

    # --- 3. Process and Save All Per-Match Data (Team and Player) ---
    all_match_data = {}
    for index, match_summary in tqdm(matches_summary_df.iterrows(), total=matches_summary_df.shape[0], desc="Processing All Matches"):
        match_id = match_summary['matchId']
        home_team = match_summary['home_team']
        away_team = match_summary['away_team']
        match_events_df = raw_events_df[raw_events_df['matchId'] == match_id].copy()
        
        if match_events_df.empty:
            continue
            
        # Calculate and store stats for the current match
        match_data = calculate_match_data(match_events_df, home_team, away_team)
        all_match_data[match_id] = match_data

    with open('all_match_data.pkl', 'wb') as f:
        pickle.dump(all_match_data, f)
    print("✅ All detailed match data saved to 'all_match_data.pkl'")

    # --- 4. Process and Save Season-Long Team Stats ---
    all_teams = pd.concat([matches_summary_df['home_team'], matches_summary_df['away_team']]).unique()
    season_team_stats = {}
    for team in tqdm(all_teams, desc="Processing Season-Long Team Stats"):
        team_corners = calculate_team_corner_stats(raw_events_df, matches_summary_df, team)
        if team_corners is not None:
            season_team_stats[team] = {'corners': team_corners} 
            
    with open('season_team_stats.pkl', 'wb') as f:
        pickle.dump(season_team_stats, f)
    print("✅ All season-long team stats saved to 'season_team_stats.pkl'")
    
    print("\n🎉 Data processing pipeline complete!")


if __name__ == "__main__":
    main()