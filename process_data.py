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
        match_date = pd.to_datetime(match_df['matchTimestamp'].iloc[0]).strftime('%Y-%m-%d')
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
    for possession_id, group in corner_possessions_df.groupby('possession.id'):
        corner_event = group[group['is_corner']].iloc[0]
        if corner_event['team.name'] != selected_team: continue
        delivery_type = 'Direct Corner'
        if is_short_corner_by_distance(corner_event['location.x'], corner_event['location.y'], corner_event['pass.endLocation.x'], corner_event['pass.endLocation.y']):
            delivery_type = 'Short Corner'
        first_contact_event = group[group['possession.eventIndex'] == corner_event['possession.eventIndex'] + 1]
        first_contact_zone = 'Unknown'
        if not first_contact_event.empty:
            first_contact_zone = map_location_to_zone(first_contact_event.iloc[0]['location.x'], first_contact_event.iloc[0]['location.y'])
        
        # ... (Your full detailed shot and phase logic would continue here) ...
        # For simplicity, we'll keep the output struct consistent with your notebook
        processed_corners.append({'possession.id': possession_id, 'delivery_type': delivery_type, 'first_contact_zone': first_contact_zone,
                                  'xg_1st_contact': 0, 'goals_1st_contact': 0, 'shots_1st_contact': 0, # etc. for other phases
                                  })
    
    results_df = pd.DataFrame(processed_corners)
    if results_df.empty: return None

    total_corners = len(results_df)
    total_xg = results_df['xg_1st_contact'].sum() # Simplified, use your full logic
    corners_per_match = total_corners / total_matches
    
    corner_stats = {selected_team: {"Total Corners": total_corners, "Corners/Match": f"{corners_per_match:.1f}", "Total xG": f"{total_xg:.3f}"}}
    return pd.DataFrame(corner_stats).rename_axis("Corner Kicks")

def calculate_match_data(match_df, home_team, away_team):
    """Calculates all team and player stats for a single match."""
    # This is a large wrapper function for your two main analysis types
    
    # --- NESTED FUNCTION: TEAM STATS ---
    def _calculate_team_stats(match_df, home_team, away_team):
        teams = [home_team, away_team]
        all_tables = {}
        match_df['matchTimestamp'] = pd.to_datetime(match_df['matchTimestamp'])
        match_df['possession.duration_sec'] = pd.to_numeric(match_df['possession.duration'].str.replace('s', ''), errors='coerce')
        # ... (Paste your ENTIRE team stat calculation logic here, from "General" to "Passes") ...
        # Example for "General":
        general_stats = {}
        shots_df = match_df[match_df['type.primary'] == 'shot']
        for team in teams:
            team_shots = shots_df[shots_df['team.name'] == team]
            goals = team_shots[team_shots['shot.isGoal'] == True].shape[0]
            xg = team_shots['shot.xg'].sum()
            general_stats[team] = {"Goals": goals, "xG": round(xg, 2)}
        all_tables["General"] = pd.DataFrame(general_stats).rename_axis("General")
        # ... Continue for all other stat categories ...
        return all_tables

    # --- NESTED FUNCTION: PLAYER STATS ---
    def _calculate_player_stats(match_df, home_team, away_team):
        def _calculate_for_team(team_name, match_df):
            team_events = match_df[match_df['team.name'] == team_name].copy()
            player_stats_list = []
            players = team_events['player.name'].dropna().unique()
            # ... (Paste your ENTIRE player stat calculation logic here) ...
            for player in players:
                player_events = team_events[team_events['player.name'] == player]
                goals = player_events[player_events['shot.isGoal'] == True].shape[0]
                xg = player_events['shot.xg'].sum()
                player_stats_list.append({"Player": player, "Goals / xG": f"{goals}/{round(xg, 2)}"})
            return pd.DataFrame(player_stats_list).set_index('Player')
        
        home_stats = _calculate_for_team(home_team, match_df)
        away_stats = _calculate_for_team(away_team, match_df)
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
    if not match_ids: return
    raw_events_df = fetch_events(wyscout_user, wyscout_pass, match_ids)
    if raw_events_df.empty: return

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
            # You can add more season-long stats to this dictionary
            season_team_stats[team] = {'corners': team_corners} 
            
    with open('season_team_stats.pkl', 'wb') as f:
        pickle.dump(season_team_stats, f)
    print("✅ All season-long team stats saved to 'season_team_stats.pkl'")
    
    print("\n🎉 Data processing pipeline complete!")


if __name__ == "__main__":
    main()