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
import warnings
import json # Added for JSON decoding error handling
from dotenv import load_dotenv # Added for .env support
from collections import defaultdict

# Suppress SettingWithCopyWarning, use cautiously
pd.options.mode.chained_assignment = None

# forcing update
# ==============================================================================
# SECTION 2: DATA FETCHING FUNCTIONS
# ==============================================================================

# === REPLACED FUNCTION ===
def fetch_match_schedule(username, password, competition_id, season_id):
    """Fetches the full match schedule (IDs, dates, teams, GW, score) for a specific season."""
    url = f"https://apirest.wyscout.com/v3/seasons/{season_id}/matches"
    auth = HTTPBasicAuth(username, password)
    print(f"Attempting to fetch match schedule for seasonId: {season_id}...")
    try:
        r = requests.get(url, auth=auth, timeout=20)
        r.raise_for_status()

        season_matches = r.json().get("matches", [])
        print(f"✅ Found {len(season_matches)} matches for seasonId {season_id}.")

        if not season_matches:
            return pd.DataFrame() # Return empty if no matches

        extracted_data = []
        for match in tqdm(season_matches, desc="Extracting Schedule Data"):
            # --- Robust Team & Score Extraction ---
            home_team_name = "Unknown Home"
            away_team_name = "Unknown Away"
            home_team_id = None
            away_team_id = None
            score_str = "? - ?"

            # Try extracting from label first as a fallback
            label = match.get('label', '')
            try:
                parts = label.split(',')
                if len(parts) > 0:
                    teams_part = parts[0]
                    team_names = teams_part.split(' - ')
                    if len(team_names) == 2:
                        home_team_name = team_names[0].strip()
                        away_team_name = team_names[1].strip()
                if len(parts) > 1:
                     score_part = parts[1].strip()
                     if '-' in score_part and len(score_part) >= 3:
                          score_str = score_part
            except Exception:
                pass # Ignore errors during label parsing

            # Try extracting from teamsData
            teams_data = match.get('teamsData', {})
            team1_id_str = str(match.get('team1Id'))
            team2_id_str = str(match.get('team2Id'))

            if team1_id_str in teams_data:
                 home_team_info = teams_data[team1_id_str]
                 home_team_name = home_team_info.get('name', home_team_name)
                 home_team_id = home_team_info.get('teamId')
            if team2_id_str in teams_data:
                 away_team_info = teams_data[team2_id_str]
                 away_team_name = away_team_info.get('name', away_team_name)
                 away_team_id = away_team_info.get('teamId')

            # Try extracting score from score1/score2
            if score_str == "? - ?":
                 s1 = match.get('score1')
                 s2 = match.get('score2')
                 if s1 is not None and s2 is not None:
                      score_str = f"{int(s1)} - {int(s2)}"
            # --- End Robust Extraction ---


            extracted_data.append({
                'matchId': match.get('matchId'),
                'seasonId': match.get('seasonId'),
                'status': match.get('status'),
                'roundId': match.get('roundId'),
                'gameweek': match.get('gameweek'), # Use API gameweek
                'dateutc': match.get('dateutc'),   # Use API dateutc
                'homeTeamId': home_team_id,
                'homeTeamName': home_team_name,
                'awayTeamId': away_team_id,
                'awayTeamName': away_team_name,
                'score': score_str # Use extracted score string
            })

        schedule_df = pd.DataFrame(extracted_data)

        # Convert dateutc and sort
        schedule_df['dateutc'] = pd.to_datetime(schedule_df['dateutc'], errors='coerce')
        schedule_df.sort_values(by='dateutc', inplace=True, na_position='last')
        schedule_df.reset_index(drop=True, inplace=True)

        print("✅ Successfully created match schedule DataFrame sorted by date.")
        return schedule_df

    except requests.exceptions.RequestException as e:
        print(f"❌ An error occurred fetching schedule: {e}")
        return pd.DataFrame() # Return empty on error
    except json.JSONDecodeError:
        print(f"❌ Error decoding JSON response from schedule API.")
        raw_response = "Request failed or response empty"
        if 'r' in locals() and hasattr(r, 'text'):
            raw_response = r.text
        print(f"   Raw Response: {raw_response}")
        return pd.DataFrame()

def fetch_events(username, password, match_ids):
    """Fetches all event data for a given list of match IDs with retries.
    Processes in batches of 200 matches to avoid memory issues on CI runners."""
    base_url_v3 = "https://apirest.wyscout.com/v3"
    auth = HTTPBasicAuth(username, password)
    session = requests.Session()
    retry_strategy = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    print(f"\nFetching events for {len(match_ids)} matches...")

    BATCH_SIZE = 200
    batch_dfs = []
    batch_events = []
    total_events = 0

    for i, match_id in enumerate(tqdm(match_ids, desc="Fetching Events")):
        url = f"{base_url_v3}/matches/{match_id}/events"
        try:
            r = session.get(url, auth=auth, timeout=30)
            r.raise_for_status()
            events_list = r.json().get('events', [])
            for event in events_list:
                event['matchId'] = match_id
            batch_events.extend(events_list)
        except requests.exceptions.RequestException as e:
            print(f"  -> ⚠️ Failed to fetch match {match_id} after multiple retries: {e}")

        # Flush batch to DataFrame periodically to free memory
        if len(batch_events) > 0 and ((i + 1) % BATCH_SIZE == 0 or i == len(match_ids) - 1):
            batch_df = pd.json_normalize(batch_events)
            batch_dfs.append(batch_df)
            total_events += len(batch_df)
            batch_events = []  # Free raw dicts

    if not batch_dfs:
        return pd.DataFrame()
    events_df = pd.concat(batch_dfs, ignore_index=True)
    del batch_dfs  # Free batch list
    print(f"\n✅ Retrieved {total_events} total events.")
    return events_df

# In process_data.py...

# ==============================================================================
# PLAYER MINUTES: Using Player Advanced Stats Endpoint (Direct Minutes)
# ==============================================================================
def fetch_official_player_minutes(username, password, match_ids, raw_events_df, competition_id=43324, season_id=191782):
    """
    Fetches player minutes directly from Wyscout Player Advanced Stats endpoint.

    1. Collects unique player IDs from match lineups
    2. Fetches each player's advanced stats (includes direct minutesOnField)
    3. Uses event data for player name/team identity
    """
    base_url_v3 = "https://apirest.wyscout.com/v3"
    auth = HTTPBasicAuth(username, password)

    # --- STEP 1: Build Identity Map from Events ---
    print("Building Player Identity Map from Events...")

    valid_events = raw_events_df.dropna(subset=['player.id', 'team.name', 'player.position'])
    grouped = valid_events.groupby('player.id')

    names = grouped['player.name'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown")
    teams = grouped['team.name'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown")
    positions = grouped['player.position'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown")

    id_map = {}
    for pid in names.index:
        try:
            pid_int = int(pid)
            id_map[pid_int] = {
                'name': str(names.loc[pid]),
                'team': str(teams.loc[pid]),
                'pos': str(positions.loc[pid])
            }
        except:
            continue

    print(f"✅ Identity Map ready for {len(id_map)} players.")

    # --- STEP 2: Collect unique player IDs from match lineups ---
    session = requests.Session()
    retry_strategy = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)

    unique_player_ids = set()
    match_lineups = {}  # {match_id: {team_name: {lineup, bench, substitutions, formation}}}

    print(f"\n🔍 Collecting player IDs from {len(match_ids)} matches...")

    for match_id in tqdm(match_ids, desc="Scanning Lineups"):
        url = f"{base_url_v3}/matches/{match_id}"
        try:
            r = session.get(url, auth=auth, timeout=20)
            r.raise_for_status()
            match_data = r.json()

            teams_data = match_data.get('teamsData', {})
            match_lineups[match_id] = {}
            for team_id, team_info in teams_data.items():
                formation = team_info.get('formation')
                if not formation:
                    continue

                # Resolve team name from lineup player IDs via id_map
                team_name = team_info.get('name')  # v3 API may not have this
                if not team_name or team_name.startswith('team_'):
                    # Look up from id_map using any lineup player
                    for player in formation.get('lineup', []):
                        pid = player.get('playerId')
                        if pid and pid in id_map:
                            team_name = id_map[pid]['team']
                            break
                if not team_name or team_name.startswith('team_'):
                    team_name = f'team_{team_id}'

                # Collect all player IDs from lineup and bench
                for player in formation.get('lineup', []):
                    pid = player.get('playerId')
                    if pid:
                        unique_player_ids.add(pid)

                for player in formation.get('bench', []):
                    pid = player.get('playerId')
                    if pid:
                        unique_player_ids.add(pid)

                # Store lineup, bench, and substitution data for this team
                match_lineups[match_id][team_name] = {
                    'lineup': formation.get('lineup', []),
                    'bench': formation.get('bench', []),
                    'substitutions': formation.get('substitutions', []),
                }

        except Exception as e:
            print(f"  -> ⚠️ Error scanning match {match_id}: {e}")

    print(f"✅ Found {len(unique_player_ids)} unique players.")
    print(f"✅ Captured lineup/substitution data for {len(match_lineups)} matches.")

    # --- STEP 3: Fetch minutes from Player Advanced Stats endpoint ---
    print(f"\n🚀 Fetching player stats from Advanced Stats endpoint...")
    print(f"   (Competition: {competition_id}, Season: {season_id})")

    player_stats_list = []

    for pid in tqdm(unique_player_ids, desc="Fetching Player Stats"):
        url = f"{base_url_v3}/players/{pid}/advancedstats"
        params = {'compId': competition_id, 'seasonId': season_id}

        try:
            r = session.get(url, auth=auth, params=params, timeout=20)

            if r.status_code != 200:
                continue

            data = r.json()
            total_stats = data.get('total', {})

            # Get direct minutes from API
            minutes = total_stats.get('minutesOnField', 0)

            if minutes and minutes > 0:
                # Get position from API response (most played position)
                positions_data = data.get('positions', [])
                api_position = None
                if positions_data:
                    # First position is most played
                    api_position = positions_data[0].get('position', {}).get('code', '').upper()

                # Get identity from events (fallback to API position)
                identity = id_map.get(pid, {'name': 'Unknown', 'team': 'Unknown', 'pos': 'Unknown'})

                player_stats_list.append({
                    'playerId': pid,
                    'playerName': identity['name'],
                    'teamName': identity['team'],
                    'primaryPosition': api_position if api_position else identity['pos'],
                    'totalMinutes': minutes
                })

        except Exception as e:
            print(f"  -> ⚠️ Error fetching player {pid}: {e}")

    # --- STEP 4: Create DataFrame ---
    if not player_stats_list:
        print("❌ No player minutes data retrieved.")
        return pd.DataFrame(columns=['playerId', 'playerName', 'teamName', 'primaryPosition', 'totalMinutes'])

    total_minutes_df = pd.DataFrame(player_stats_list)
    total_minutes_df['playerId'] = total_minutes_df['playerId'].astype(int)

    print(f"✅ Retrieved direct minutes for {len(total_minutes_df)} players.")

    return total_minutes_df, match_lineups

# ==============================================================================
# SECTION 3: DATA PROCESSING FUNCTIONS
# ==============================================================================

# process_data.py (Add this new function in Section 3)

def calculate_player_minutes_and_positions(matches_df, events_df):
    """
    Calculates total minutes played and primary positions for each player
    using the direct 'minutesOnField' data from the matches_df (Wyscout API data).
    """
    print("Calculating player minutes and positions from API data...")
    
    player_stats_list = []

    # Iterate through every match to scrape minute data
    for index, match in matches_df.iterrows():
        match_id = match.get('wyId')
        teams_data = match.get('teamsData')
        
        if not teams_data:
            continue
            
        for team_id, team_data in teams_data.items():
            team_name = team_data.get('name', 'Unknown')
            formation = team_data.get('formation')
            
            # Safety Check: Skip if formation data is missing
            if not formation:
                continue
                
            # Combine starters (lineup) and subs (bench)
            # Both lists contain player objects with 'minutesOnField'
            all_players = formation.get('lineup', []) + formation.get('bench', [])
            
            for player in all_players:
                p_id = player.get('playerId')
                p_name = player.get('shortName') or player.get('lastName') or 'Unknown'
                
                # --- MINUTES CALCULATION ---
                # Wyscout usually uses 'minutesOnField'. We check 'minutesPlayed' as fallback.
                minutes_played = player.get('minutesOnField')
                if minutes_played is None:
                     minutes_played = player.get('minutesPlayed', 0)
                
                # Only record if they actually played
                if minutes_played > 0:
                    # --- POSITION EXTRACTION ---
                    # Wyscout provides a 'role' dict: {'code2': 'MD', 'name': 'Midfielder', ...}
                    role = player.get('role', {})
                    role_name = role.get('name') 
                    role_code = role.get('code2') # e.g., 'GK', 'DF', 'MD', 'FW'
                    
                    player_stats_list.append({
                        'playerId': p_id,
                        'playerName': p_name,
                        'teamName': team_name,
                        'minutes': minutes_played,
                        'matchId': match_id,
                        'position_code': role_code,
                        'position_name': role_name
                    })

    # Create DataFrame from the list
    if not player_stats_list:
        print("Warning: No player minute data found.")
        return pd.DataFrame(columns=['playerId', 'playerName', 'teamName', 'totalMinutes', 'primaryPosition'])

    stats_df = pd.DataFrame(player_stats_list)

    # --- AGGREGATION ---
    # 1. Group by Player to get Total Minutes
    minutes_df = stats_df.groupby(['playerId', 'playerName', 'teamName'])['minutes'].sum().reset_index()
    minutes_df.rename(columns={'minutes': 'totalMinutes'}, inplace=True)

    # 2. Determine Primary Position (Most common position played)
    # We group by player and position, count occurrences, and take the top one
    pos_counts = stats_df.groupby(['playerId', 'position_code']).size().reset_index(name='count')
    # Sort by count desc and drop duplicates to keep the most frequent
    primary_positions = pos_counts.sort_values(['playerId', 'count'], ascending=[True, False]).drop_duplicates('playerId')
    primary_positions.rename(columns={'position_code': 'primaryPosition'}, inplace=True)

    # 3. Merge Position back into Minutes
    final_df = pd.merge(minutes_df, primary_positions[['playerId', 'primaryPosition']], on='playerId', how='left')

    # Fill missing positions
    final_df['primaryPosition'] = final_df['primaryPosition'].fillna('Unknown')
    
    # Ensure IDs are integers
    final_df['playerId'] = final_df['playerId'].astype(int)

    return final_df


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

    team_matches_df = matches_summary_df[(matches_summary_df['homeTeamName'] == selected_team) | (matches_summary_df['awayTeamName'] == selected_team)]
    total_matches = team_matches_df['matchId'].nunique()
    if total_matches == 0: return None

    # Use .get() for safer column access
    events_df['is_corner'] = (events_df.get('type.primary') == 'corner')

    corners_df = events_df[(events_df['is_corner']) & (events_df.get('team.name') == selected_team)].copy()
    if corners_df.empty: return None

    corner_possession_ids = corners_df['possession.id'].dropna().unique()
    relevant_events_df = events_df[events_df['possession.id'].isin(corner_possession_ids)].sort_values(by=['possession.id', 'possession.eventIndex'])

    processed_corners = []
    if not relevant_events_df.empty:
        # Use try-except for robustness during iteration
        try:
            for possession_id, group in relevant_events_df.groupby('possession.id'):
                corner_event_rows = group[group['is_corner']]
                if corner_event_rows.empty: continue
                corner_event = corner_event_rows.iloc[0]

                if corner_event['team.name'] != selected_team: continue

                corner_index = corner_event.get('possession.eventIndex')
                # Check if corner_index is valid before proceeding
                if pd.isna(corner_index): continue

                delivery_type = 'Direct Corner'
                if is_short_corner_by_distance(corner_event.get('location.x'), corner_event.get('location.y'), corner_event.get('pass.endLocation.x'), corner_event.get('pass.endLocation.y')):
                     delivery_type = 'Short Corner'

                first_contact_event = group[group['possession.eventIndex'] == corner_index + 1].copy()
                first_contact_zone = 'Unknown'
                if not first_contact_event.empty:
                    first_contact_zone = map_location_to_zone(first_contact_event.iloc[0].get('location.x'), first_contact_event.iloc[0].get('location.y'))

                shot_details = {
                    '1st Contact': {'shots': 0, 'goals': 0, 'xg': 0}, '2nd Contact': {'shots': 0, 'goals': 0, 'xg': 0},
                    '2nd Phase': {'shots': 0, 'goals': 0, 'xg': 0},
                }
                if not first_contact_event.empty and pd.notna(first_contact_event.iloc[0].get('shot.xg')):
                    event = first_contact_event.iloc[0]
                    shot_details['1st Contact'].update({'shots': 1, 'goals': 1 if event.get('shot.isGoal') == True else 0, 'xg': event.get('shot.xg', 0)})

                second_contact_event = group[group['possession.eventIndex'] == corner_index + 2].copy()
                if not second_contact_event.empty and pd.notna(second_contact_event.iloc[0].get('shot.xg')):
                    event = second_contact_event.iloc[0]
                    shot_details['2nd Contact'].update({'shots': 1, 'goals': 1 if event.get('shot.isGoal') == True else 0, 'xg': event.get('shot.xg', 0)})

                second_phase_events = group[group['possession.eventIndex'] > corner_index + 2].copy()
                if not second_phase_events.empty:
                    # Ensure 'shot.xg' exists before dropping NA
                    if 'shot.xg' in second_phase_events.columns:
                        shots_in_phase = second_phase_events.dropna(subset=['shot.xg']).copy()
                        if not shots_in_phase.empty:
                            shots_in_phase['isGoal'] = (shots_in_phase.get('shot.isGoal') == True)
                            shot_details['2nd Phase'].update({'shots': len(shots_in_phase), 'goals': shots_in_phase['isGoal'].sum(), 'xg': shots_in_phase.get('shot.xg', 0).sum()})
                    else: # Handle case where shot.xg might be missing entirely
                         shot_details['2nd Phase'].update({'shots': 0, 'goals': 0, 'xg': 0})


                processed_corners.append({
                    'possession.id': possession_id, 'delivery_type': delivery_type, 'first_contact_zone': first_contact_zone,
                    'shots_1st_contact': shot_details['1st Contact']['shots'], 'goals_1st_contact': shot_details['1st Contact']['goals'], 'xg_1st_contact': shot_details['1st Contact']['xg'],
                    'shots_2nd_contact': shot_details['2nd Contact']['shots'], 'goals_2nd_contact': shot_details['2nd Contact']['goals'], 'xg_2nd_contact': shot_details['2nd Contact']['xg'],
                    'shots_2nd_phase': shot_details['2nd Phase']['shots'], 'goals_2nd_phase': shot_details['2nd Phase']['goals'], 'xg_2nd_phase': shot_details['2nd Phase']['xg'],
                })
        except Exception as e:
            print(f"Warning: Error processing corners for {selected_team} - {e}")
            # Continue processing other teams/data if possible

    results_df = pd.DataFrame(processed_corners)
    if results_df.empty: return None

    # Build the final DataFrame
    total_corners = len(results_df)
    total_xg = results_df[['xg_1st_contact', 'xg_2nd_contact', 'xg_2nd_phase']].sum().sum()
    corners_per_match = total_corners / total_matches
    xg_per_90 = total_xg / total_matches
    xg_per_corner = total_xg / total_corners if total_corners > 0 else 0
    first_contact_pct = results_df['first_contact_zone'].value_counts(normalize=True).mul(100)
    total_goals = results_df[['goals_1st_contact', 'goals_2nd_contact', 'goals_2nd_phase']].sum().sum()
    goals_1st_contact = results_df['goals_1st_contact'].sum()
    goals_2nd_contact = results_df['goals_2nd_contact'].sum()
    goals_2nd_phase = results_df['goals_2nd_phase'].sum()
    total_shots = results_df[['shots_1st_contact', 'shots_2nd_contact', 'shots_2nd_phase']].sum().sum()
    shots_1st_contact = results_df['shots_1st_contact'].sum()
    shots_2nd_contact = results_df['shots_2nd_contact'].sum()
    shots_2nd_phase = results_df['shots_2nd_phase'].sum()
    delivery_counts = results_df['delivery_type'].value_counts(normalize=True).mul(100)

    corner_stats_data = {
        "Corners/Match": f"{corners_per_match:.1f}", "Total Corners": total_corners,
        "Total xG": f"{total_xg:.3f}", "xG/90min": f"{xg_per_90:.2f}", "xG/Corner": f"{xg_per_corner:.3f}",
        "1st Contact Zone: 6Y Box %": f"{first_contact_pct.get('6Y Box', 0):.1f}",
        "1st Contact Zone: Front Area %": f"{first_contact_pct.get('Front Area', 0):.1f}",
        "1st Contact Zone: Near Post %": f"{first_contact_pct.get('Near Post', 0):.1f}",
        "1st Contact Zone: Far Post %": f"{first_contact_pct.get('Far Post', 0):.1f}",
        "1st Contact Zone: Middle Area %": f"{first_contact_pct.get('Middle Area', 0):.1f}",
        "1st Contact Zone: Back Area %": f"{first_contact_pct.get('Back Area', 0):.1f}",
        "Total Goals": int(total_goals),
        "Goals (1st Contact)": int(goals_1st_contact), "Goals (2nd Contact)": int(goals_2nd_contact), "Goals (2nd Phase)": int(goals_2nd_phase),
        "Total Shots": int(total_shots),
        "Shots (1st Contact)": int(shots_1st_contact), "Shots (2nd Contact)": int(shots_2nd_contact), "Shots (2nd Phase)": int(shots_2nd_phase),
        "Short Corner %": f"{delivery_counts.get('Short Corner', 0):.1f}",
        "Direct Corner %": f"{delivery_counts.get('Direct Corner', 0):.1f}",
    }
    corner_stats_df = pd.DataFrame.from_dict({selected_team: corner_stats_data}, orient='columns') # Team as column directly
    corner_stats_df.index.name = "Corner Kicks"
    
    return corner_stats_df


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
import warnings

# Suppress SettingWithCopyWarning, use cautiously
pd.options.mode.chained_assignment = None

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

# NOTE: fetch_events is defined once at the top of this file (batched version).
# Do not redefine it here.

# ==============================================================================
# SECTION 3: DATA PROCESSING FUNCTIONS
# ==============================================================================

def create_match_summary(events_df):
    """Creates summary (no date) with REVERSED gameweek labels based on natural order blocks."""
    print("Processing: Creating match summary (natural order, no date)...")
    matches_summary = []
    # Ensure 'matchId' exists before proceeding
    if 'matchId' not in events_df.columns:
        print("❌ Error: 'matchId' column not found in events data. Cannot create match summary.")
        return pd.DataFrame()

    # Get unique match IDs IN THE ORDER THEY APPEAR in the events_df
    # unique_match_ids = events_df['matchId'].unique() # This preserves order
    # Alternative using drop_duplicates which is usually faster and preserves order
    unique_match_ids = events_df.drop_duplicates(subset=['matchId'], keep='first')['matchId'].tolist()


    for match_id in tqdm(unique_match_ids, desc="Summarizing Matches"): # Iterate in natural order
        match_df = events_df[events_df['matchId'] == match_id].copy()
        if match_df.empty: continue

        teams = match_df['team.name'].unique()
        if len(teams) < 2: continue
        home_team, away_team = teams[0], teams[1] # Assumes consistent order

        # Get score
        goals_df = match_df[match_df.get('shot.isGoal') == True]
        goal_counts = goals_df.get('team.name', pd.Series(dtype='str')).value_counts()
        home_score = goal_counts.get(home_team, 0)
        away_score = goal_counts.get(away_team, 0)

        matches_summary.append({
            'matchId': match_id,
            'home_team': home_team,
            'away_team': away_team,
            # 'date': match_date, # REMOVED DATE
            'score': f"{home_score} - {away_score}"
        })

    if not matches_summary:
        print("Warning: No matches found to create summary.")
        return pd.DataFrame()

    matches_summary_df = pd.DataFrame(matches_summary)
    # The DataFrame index (0 to N) now reflects the natural order

    # --- Gameweek Calculation Logic (Blocks of 10 from the end of NATURAL order) ---
    print("Calculating Gameweeks (blocks of 10, reversed, natural order)...")
    n_matches = len(matches_summary_df)
    matches_per_gw = 10

    # Calculate gameweek number based on index from the END
    gameweek_numbers = ((n_matches - 1 - matches_summary_df.index) // matches_per_gw) + 1
    matches_summary_df['Gameweek'] = [f"GW{num}" for num in gameweek_numbers]
    # --- End Gameweek Logic ---

    return matches_summary_df

def calculate_team_corner_stats(events_df, matches_summary_df, selected_team):
    """Calculates detailed corner kick statistics for a single team for the whole season."""
    # (This function seems okay from previous context, keeping it as is)
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

    team_matches_df = matches_summary_df[(matches_summary_df['homeTeamName'] == selected_team) | (matches_summary_df['awayTeamName'] == selected_team)]
    total_matches = team_matches_df['matchId'].nunique()
    if total_matches == 0: return None

    # Use .get() for safer column access
    events_df['is_corner'] = (events_df.get('type.primary') == 'corner')

    corners_df = events_df[(events_df['is_corner']) & (events_df.get('team.name') == selected_team)].copy()
    if corners_df.empty: return None

    corner_possession_ids = corners_df['possession.id'].dropna().unique()
    relevant_events_df = events_df[events_df['possession.id'].isin(corner_possession_ids)].sort_values(by=['possession.id', 'possession.eventIndex'])

    processed_corners = []
    if not relevant_events_df.empty:
        # Use try-except for robustness during iteration
        try:
            for possession_id, group in relevant_events_df.groupby('possession.id'):
                corner_event_rows = group[group['is_corner']]
                if corner_event_rows.empty: continue
                corner_event = corner_event_rows.iloc[0]

                if corner_event['team.name'] != selected_team: continue

                corner_index = corner_event.get('possession.eventIndex')
                # Check if corner_index is valid before proceeding
                if pd.isna(corner_index): continue

                delivery_type = 'Direct Corner'
                if is_short_corner_by_distance(corner_event.get('location.x'), corner_event.get('location.y'), corner_event.get('pass.endLocation.x'), corner_event.get('pass.endLocation.y')):
                     delivery_type = 'Short Corner'

                first_contact_event = group[group['possession.eventIndex'] == corner_index + 1].copy()
                first_contact_zone = 'Unknown'
                if not first_contact_event.empty:
                    first_contact_zone = map_location_to_zone(first_contact_event.iloc[0].get('location.x'), first_contact_event.iloc[0].get('location.y'))

                shot_details = {
                    '1st Contact': {'shots': 0, 'goals': 0, 'xg': 0}, '2nd Contact': {'shots': 0, 'goals': 0, 'xg': 0},
                    '2nd Phase': {'shots': 0, 'goals': 0, 'xg': 0},
                }
                if not first_contact_event.empty and pd.notna(first_contact_event.iloc[0].get('shot.xg')):
                    event = first_contact_event.iloc[0]
                    shot_details['1st Contact'].update({'shots': 1, 'goals': 1 if event.get('shot.isGoal') == True else 0, 'xg': event.get('shot.xg', 0)})

                second_contact_event = group[group['possession.eventIndex'] == corner_index + 2].copy()
                if not second_contact_event.empty and pd.notna(second_contact_event.iloc[0].get('shot.xg')):
                    event = second_contact_event.iloc[0]
                    shot_details['2nd Contact'].update({'shots': 1, 'goals': 1 if event.get('shot.isGoal') == True else 0, 'xg': event.get('shot.xg', 0)})

                second_phase_events = group[group['possession.eventIndex'] > corner_index + 2].copy()
                if not second_phase_events.empty:
                    # Ensure 'shot.xg' exists before dropping NA
                    if 'shot.xg' in second_phase_events.columns:
                        shots_in_phase = second_phase_events.dropna(subset=['shot.xg']).copy()
                        if not shots_in_phase.empty:
                            # Use .get() for safe access within isGoal check
                            shots_in_phase['isGoal'] = (shots_in_phase.get('shot.isGoal') == True)
                            # Use .get() for safe access to shot.xg sum
                            shot_details['2nd Phase'].update({'shots': len(shots_in_phase), 'goals': shots_in_phase['isGoal'].sum(), 'xg': shots_in_phase.get('shot.xg', pd.Series(dtype='float')).sum()})
                    else: # Handle case where shot.xg might be missing entirely
                         shot_details['2nd Phase'].update({'shots': 0, 'goals': 0, 'xg': 0})


                processed_corners.append({
                    'possession.id': possession_id, 'delivery_type': delivery_type, 'first_contact_zone': first_contact_zone,
                    'shots_1st_contact': shot_details['1st Contact']['shots'], 'goals_1st_contact': shot_details['1st Contact']['goals'], 'xg_1st_contact': shot_details['1st Contact']['xg'],
                    'shots_2nd_contact': shot_details['2nd Contact']['shots'], 'goals_2nd_contact': shot_details['2nd Contact']['goals'], 'xg_2nd_contact': shot_details['2nd Contact']['xg'],
                    'shots_2nd_phase': shot_details['2nd Phase']['shots'], 'goals_2nd_phase': shot_details['2nd Phase']['goals'], 'xg_2nd_phase': shot_details['2nd Phase']['xg'],
                })
        except Exception as e:
            print(f"Warning: Error processing corners for {selected_team} - {e}")

    results_df = pd.DataFrame(processed_corners)
    if results_df.empty: return None

    # Build the final DataFrame
    total_corners = len(results_df)
    total_xg = results_df[['xg_1st_contact', 'xg_2nd_contact', 'xg_2nd_phase']].sum().sum()
    corners_per_match = total_corners / total_matches
    xg_per_90 = total_xg / total_matches
    xg_per_corner = total_xg / total_corners if total_corners > 0 else 0
    first_contact_pct = results_df['first_contact_zone'].value_counts(normalize=True).mul(100)
    total_goals = results_df[['goals_1st_contact', 'goals_2nd_contact', 'goals_2nd_phase']].sum().sum()
    goals_1st_contact = results_df['goals_1st_contact'].sum()
    goals_2nd_contact = results_df['goals_2nd_contact'].sum()
    goals_2nd_phase = results_df['goals_2nd_phase'].sum()
    total_shots = results_df[['shots_1st_contact', 'shots_2nd_contact', 'shots_2nd_phase']].sum().sum()
    shots_1st_contact = results_df['shots_1st_contact'].sum()
    shots_2nd_contact = results_df['shots_2nd_contact'].sum()
    shots_2nd_phase = results_df['shots_2nd_phase'].sum()
    delivery_counts = results_df['delivery_type'].value_counts(normalize=True).mul(100)

    corner_stats_data = {
        "Corners/Match": f"{corners_per_match:.1f}", "Total Corners": total_corners,
        "Total xG": f"{total_xg:.3f}", "xG/90min": f"{xg_per_90:.2f}", "xG/Corner": f"{xg_per_corner:.3f}",
        "1st Contact Zone: 6Y Box %": f"{first_contact_pct.get('6Y Box', 0):.1f}",
        "1st Contact Zone: Front Area %": f"{first_contact_pct.get('Front Area', 0):.1f}",
        "1st Contact Zone: Near Post %": f"{first_contact_pct.get('Near Post', 0):.1f}",
        "1st Contact Zone: Far Post %": f"{first_contact_pct.get('Far Post', 0):.1f}",
        "1st Contact Zone: Middle Area %": f"{first_contact_pct.get('Middle Area', 0):.1f}",
        "1st Contact Zone: Back Area %": f"{first_contact_pct.get('Back Area', 0):.1f}",
        "Total Goals": int(total_goals),
        "Goals (1st Contact)": int(goals_1st_contact), "Goals (2nd Contact)": int(goals_2nd_contact), "Goals (2nd Phase)": int(goals_2nd_phase),
        "Total Shots": int(total_shots),
        "Shots (1st Contact)": int(shots_1st_contact), "Shots (2nd Contact)": int(shots_2nd_contact), "Shots (2nd Phase)": int(shots_2nd_phase),
        "Short Corner %": f"{delivery_counts.get('Short Corner', 0):.1f}",
        "Direct Corner %": f"{delivery_counts.get('Direct Corner', 0):.1f}",
    }
    corner_stats_df = pd.DataFrame.from_dict({selected_team: corner_stats_data}, orient='columns')
    corner_stats_df.index.name = "Corner Kicks"

    return corner_stats_df

def add_custom_dribble_success(events_df):
    """
    Applies custom logic to determine if a dribble was successful.
    
    IMPROVEMENT: Looks for the next event *by the same team* to skip 
    interleaved opponent defensive events.
    """
    # 1. Work on a sorted copy
    df = events_df.sort_values(by=['matchId', 'matchTimestamp']).copy()

    
    # --- FIX: Force timestamp to datetime object to avoid string errors ---
    df['matchTimestamp'] = pd.to_datetime(df['matchTimestamp'], errors='coerce')
    
    # 2. Calculate Distance to Goal for the CURRENT event
    # (100, 50) is the center of the opponent's goal
    df['dist_to_goal'] = np.sqrt((100 - df['location.x'].fillna(100))**2 + (50 - df['location.y'].fillna(50))**2)
    
    # 3. Get details of the NEXT event for the SAME TEAM
    # This skips opponent events (like defensive duels) that happen in between
    team_grouped = df.groupby(['matchId', 'team.name'])
    
    df['next_team_player_id'] = team_grouped['player.id'].shift(-1)
    df['next_team_type'] = team_grouped['type.primary'].shift(-1)
    df['next_team_accurate'] = team_grouped['pass.accurate'].shift(-1)
    df['next_team_start_x'] = team_grouped['location.x'].shift(-1)
    df['next_team_start_y'] = team_grouped['location.y'].shift(-1)
    df['next_team_timestamp'] = team_grouped['matchTimestamp'].shift(-1)

    # Calculate distance for the NEXT team event start
    df['next_team_start_dist'] = np.sqrt((100 - df['next_team_start_x'].fillna(100))**2 + (50 - df['next_team_start_y'].fillna(50))**2)
    
    # For Pass Success Logic: We need where the next pass ENDED
    df['next_team_end_x'] = team_grouped['pass.endLocation.x'].shift(-1)
    df['next_team_end_y'] = team_grouped['pass.endLocation.y'].shift(-1)
    df['next_team_end_dist'] = np.sqrt((100 - df['next_team_end_x'].fillna(100))**2 + (50 - df['next_team_end_y'].fillna(50))**2)

    # 4. Time Delta Check
    # If the next team event is > 4 seconds later, the chain is broken
    df['time_diff'] = (df['next_team_timestamp'] - df['matchTimestamp']).dt.total_seconds()
    is_chain_intact = df['time_diff'] <= 4.0

    # 5. Identify Dribble Attempts
    df['is_dribble_attempt'] = df.get('type.secondary', pd.Series(dtype='object')).apply(
        lambda x: isinstance(x, (list, np.ndarray)) and 'dribble' in x
    ) | (df.get('groundDuel.takeOn') == True)

    # 6. Evaluate Success Conditions
    
    # Condition A: Same player, next action is closer to goal (Carry, Shot, etc.)
    cond_a = (
        is_chain_intact &
        (df['next_team_player_id'] == df['player.id']) & 
        (df['next_team_type'] != 'pass') & 
        (df['next_team_start_dist'] < df['dist_to_goal'])
    )
    
    # Condition B: "Successful forward pass from a dribble"
    cond_b = (
        is_chain_intact &
        (df['next_team_player_id'] == df['player.id']) & 
        (df['next_team_type'] == 'pass') & 
        (df['next_team_accurate'] == True) & 
        (df['next_team_end_dist'] < df['dist_to_goal'])
    )
    
    # Condition C: "Touch of an attacking teammate closer to goal"
    cond_c = (
        is_chain_intact &
        (df['next_team_player_id'] != df['player.id']) & 
        (df['next_team_start_dist'] < df['dist_to_goal'])
    )
    
    # Condition D: Foul Suffered
    cond_d = df.get('type.secondary', pd.Series(dtype='object')).apply(
        lambda x: isinstance(x, (list, np.ndarray)) and 'foul_suffered' in x
    )

    # 7. Assign Result
    df['is_custom_dribble_success'] = df['is_dribble_attempt'] & (cond_a | cond_b | cond_c | cond_d)
    
    # Cleanup
    cols_to_drop = [c for c in df.columns if 'next_team_' in c] + ['dist_to_goal', 'time_diff']
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    
    return df


def calculate_match_data(match_df, home_team, away_team):
    """Calculates all team and player stats for a single match."""

    print(f"\n---> Processing Match: {home_team} vs {away_team}") # ADDED PRINT

    # --- NESTED FUNCTION: TEAM STATS (FULL LOGIC) ---
    def _calculate_team_stats(match_df, home_team, away_team):
        print(f"      Calculating Team Stats...") # ADDED PRINT
        teams = [home_team, away_team]
        all_tables = {}
        try: # Added try block for entire team stat calculation
            # Pre-computation and error handling for missing columns
            match_df['matchTimestamp'] = pd.to_datetime(match_df.get('matchTimestamp'), errors='coerce')
            match_df['possession.duration_sec'] = pd.to_numeric(match_df.get('possession.duration', pd.Series(dtype='str')).str.replace('s', ''), errors='coerce')

            PENALTY_AREA_X = 83; PENALTY_AREA_Y1, PENALTY_AREA_Y2 = (21, 79)
            DEFENSIVE_THIRD_X = 33.3; ATTACKING_THIRD_X = 66.6; FINAL_THIRD_X = 66
            RECOVERY_EVENTS = ['interception', 'duel', 'clearance', 'goalkeeper_exit']
            SET_PIECE_EVENTS = ['corner', 'free_kick', 'goal_kick', 'throw_in', 'penalty']

            # --- Calculate "General" Stats ---
            general_stats = {}
            shots_df = match_df[match_df.get('type.primary') == 'shot'].copy()
            infractions_df = match_df[match_df.get('type.primary') == 'infraction'].copy()
            offsides_df = match_df[match_df.get('type.primary') == 'offside'].copy()
            free_kicks_df = match_df[match_df.get('type.primary') == 'free_kick'].copy()
            corners_df = match_df[match_df.get('type.primary') == 'corner'].copy()

            for team in teams:
                team_shots = shots_df[shots_df.get('team.name') == team].copy()
                # Use .loc with check for non-empty DataFrame before calculating distance
                if not team_shots.empty and 'location.x' in team_shots.columns and 'location.y' in team_shots.columns:
                     team_shots.loc[:, 'distance'] = np.sqrt((100 - team_shots['location.x'].fillna(100))**2 + (50 - team_shots['location.y'].fillna(50))**2)
                else:
                     team_shots['distance'] = np.nan # Add column if needed

                goals = team_shots[team_shots.get('shot.isGoal') == True].shape[0]
                xg = team_shots.get('shot.xg', pd.Series(dtype='float')).sum()
                total_shots = team_shots.shape[0]
                on_target = team_shots[team_shots.get('shot.onTarget') == True].shape[0]
                # Use .get with default 0 for location if column missing
                shots_in_box = team_shots[(team_shots.get('location.x', 0) >= PENALTY_AREA_X) & (team_shots.get('location.y', 0).between(PENALTY_AREA_Y1, PENALTY_AREA_Y2))]
                shots_in_box_on_target = shots_in_box[shots_in_box.get('shot.onTarget') == True].shape[0]
                shots_out_box = team_shots[team_shots.get('location.x', 101) < PENALTY_AREA_X]
                shots_out_box_on_target = shots_out_box[shots_out_box.get('shot.onTarget') == True].shape[0]
                avg_shot_dist = team_shots.get('distance', pd.Series(dtype='float')).mean()
                corners = corners_df[corners_df.get('team.name') == team].shape[0]
                team_free_kicks = free_kicks_df[free_kicks_df.get('team.name') == team].copy()
                # Use .loc with check for non-empty DataFrame before calculating distance
                if not team_free_kicks.empty and 'location.x' in team_free_kicks.columns and 'location.y' in team_free_kicks.columns:
                     team_free_kicks.loc[:,'distance_to_goal'] = np.sqrt((100 - team_free_kicks['location.x'].fillna(100))**2 + (50 - team_free_kicks['location.y'].fillna(50))**2)
                else:
                     team_free_kicks['distance_to_goal'] = np.nan
                attacking_free_kicks = team_free_kicks[team_free_kicks['distance_to_goal'] <= 45].shape[0]
                offsides = offsides_df[offsides_df.get('team.name') == team].shape[0]
                fouls_committed = infractions_df[infractions_df.get('team.name') == team].shape[0]
                yellow_cards = infractions_df[(infractions_df.get('team.name') == team) & (infractions_df.get('infraction.yellowCard') == True)].shape[0]
                red_cards = infractions_df[(infractions_df.get('team.name') == team) & (infractions_df.get('infraction.redCard') == True)].shape[0]

                general_stats[team] = {
                    "Goals": goals, "xG": round(xg, 2) if pd.notna(xg) else 0.0, # Handle potential NaN xG
                    "Shots / on target": f"{total_shots}/{on_target}",
                    "From penalty area / on target": f"{shots_in_box.shape[0]}/{shots_in_box_on_target}",
                    "Outside penalty area / on target": f"{shots_out_box.shape[0]}/{shots_out_box_on_target}",
                    "Average shot distance (m)": round(avg_shot_dist, 1) if pd.notna(avg_shot_dist) else '-', "Corners": corners,
                    "Attacking free kicks": attacking_free_kicks, "Offsides": offsides,
                    "Fouls committed": fouls_committed, "Yellow / red cards": f"{yellow_cards}/{red_cards}"
                }

            # Handle cases where one team might not have stats
            if home_team in general_stats and away_team in general_stats:
                home_fouls_committed = general_stats[home_team].pop('Fouls committed', 0)
                away_fouls_committed = general_stats[away_team].pop('Fouls committed', 0)
                general_stats[home_team]['Fouls committed / suffered'] = f"{home_fouls_committed}/{away_fouls_committed}"
                general_stats[away_team]['Fouls committed / suffered'] = f"{away_fouls_committed}/{home_fouls_committed}"
            elif home_team in general_stats:
                 general_stats[home_team].pop('Fouls committed', 0)
                 general_stats[home_team]['Fouls committed / suffered'] = f"{0}/{0}" # Placeholder
            elif away_team in general_stats:
                 general_stats[away_team].pop('Fouls committed', 0)
                 general_stats[away_team]['Fouls committed / suffered'] = f"{0}/{0}" # Placeholder

            general_stats_df = pd.DataFrame(general_stats).rename_axis("General")
            general_order = ["Goals", "xG", "Shots / on target", "From penalty area / on target",
                             "Outside penalty area / on target", "Average shot distance (m)",
                             "Corners", "Attacking free kicks", "Offsides",
                             "Fouls committed / suffered", "Yellow / red cards"]
            # Ensure columns exist before reindexing
            existing_cols = [col for col in general_order if col in general_stats_df.index]
            all_tables["General"] = general_stats_df.reindex(existing_cols).fillna('-')
            print(f"      ... General stats calculated.") # ADDED PRINT

            # --- Calculate "Attacks" Stats ---
            attack_stats = {}
            for team in teams:
                team_match_events = match_df[match_df.get('team.name') == team]
                final_third_entries = 0
                box_entries = 0
                counter_attacks = 0
                positional_attacks = 0

                if not team_match_events.empty and 'possession.id' in team_match_events.columns and 'location.x' in team_match_events.columns:
                    try: # Wrap groupby operations in try-except
                         # Ensure location.x is numeric before min/max
                         team_match_events['location.x_numeric'] = pd.to_numeric(team_match_events['location.x'], errors='coerce')
                         possessions_grouped = team_match_events.groupby('possession.id')[['location.x_numeric']]
                         # Filter out groups where min/max failed (all NaNs)
                         valid_groups = possessions_grouped.filter(lambda x: not x['location.x_numeric'].isna().all())
                         # Apply filter logic only to valid groups
                         if not valid_groups.empty:
                             final_third_entries_series = valid_groups.groupby('possession.id')['location.x_numeric'].transform(lambda x: x.min() < FINAL_THIRD_X and x.max() >= FINAL_THIRD_X)
                             final_third_entries = final_third_entries_series[final_third_entries_series].index.get_level_values('possession.id').nunique()
                         team_match_events.drop(columns=['location.x_numeric'], inplace=True) # Clean up temporary column
                    except Exception as e:
                        print(f"Warning: Could not calculate final third entries for {team} - {e}")
                        final_third_entries = 0


                    possessions_with_box_event = team_match_events[
                        (pd.to_numeric(team_match_events.get('location.x'), errors='coerce').fillna(0) >= PENALTY_AREA_X) &
                        (pd.to_numeric(team_match_events.get('location.y'), errors='coerce').fillna(0).between(PENALTY_AREA_Y1, PENALTY_AREA_Y2))
                    ]['possession.id'].unique()
                    box_entries = len(possessions_with_box_event)

                    # Counterattacks logic
                    if 'matchTimestamp' in team_match_events.columns:
                        try:
                             first_events_idx = team_match_events.groupby('possession.id')['matchTimestamp'].idxmin()
                             first_events = team_match_events.loc[first_events_idx].copy()
                        except KeyError: # Handle cases where idxmin might return index not present
                            first_events = pd.DataFrame() # Empty df

                        recoveries = first_events[first_events.get('type.primary', '').isin(RECOVERY_EVENTS)].copy()
                        if not recoveries.empty and 'location.x' in recoveries.columns and 'location.y' in recoveries.columns:
                            recoveries['start_distance'] = np.sqrt((100 - recoveries['location.x'].fillna(100))**2 + (50 - recoveries['location.y'].fillna(50))**2)
                            counter_candidates = recoveries[recoveries['start_distance'] >= 40].copy()
                            if not counter_candidates.empty:
                                candidate_events = team_match_events[team_match_events.get('possession.id', pd.NA).isin(counter_candidates['possession.id'])].copy()
                                if not candidate_events.empty and 'possession.id' in candidate_events.columns and 'matchTimestamp' in candidate_events.columns and 'possession.id' in counter_candidates.columns:
                                    candidate_events = candidate_events.merge(counter_candidates[['possession.id', 'matchTimestamp']], on='possession.id', suffixes=('', '_start'), how='left')
                                    if 'matchTimestamp_start' in candidate_events.columns and pd.api.types.is_datetime64_any_dtype(candidate_events['matchTimestamp']) and pd.api.types.is_datetime64_any_dtype(candidate_events['matchTimestamp_start']):
                                        # Ensure no NaT before subtraction
                                        valid_ts = candidate_events[['matchTimestamp', 'matchTimestamp_start']].dropna()
                                        if not valid_ts.empty:
                                            candidate_events.loc[valid_ts.index, 'time_in_possession'] = (valid_ts['matchTimestamp'] - valid_ts['matchTimestamp_start']).dt.total_seconds()
                                            fast_events = candidate_events[candidate_events.get('time_in_possession', np.inf) <= 5].copy()
                                            if not fast_events.empty and 'location.x' in fast_events.columns and 'location.y' in fast_events.columns:
                                                fast_events['distance_to_goal'] = np.sqrt((100 - fast_events['location.x'].fillna(100))**2 + (50 - fast_events['location.y'].fillna(50))**2)
                                                min_distances = fast_events.groupby('possession.id')['distance_to_goal'].min().reset_index().rename(columns={'distance_to_goal': 'min_distance'})
                                                progression_df = counter_candidates.merge(min_distances, on='possession.id', how='left')
                                                if 'min_distance' in progression_df.columns and 'start_distance' in progression_df.columns:
                                                    progression_df['progression'] = progression_df['start_distance'] - progression_df['min_distance']
                                                    counter_attacks = progression_df[progression_df['progression'].fillna(-1) >= 30].shape[0] # Fill NaN progression


                    # Positional attacks
                    if 'possession.id' in match_df.columns and 'possession.team.name' in match_df.columns:
                        unique_possessions = match_df.drop_duplicates(subset='possession.id')
                        positional_attacks_df = unique_possessions[
                            (unique_possessions.get('possession.team.name') == team) &
                            (unique_possessions.get('possession.attack.flank', '').isin(['central', 'left', 'right']))
                        ]
                        positional_attacks = positional_attacks_df.shape[0]

                attack_stats[team] = {
                    "Positional attacks": positional_attacks,
                    "Final 1/3 Entries": final_third_entries,
                    "Box Entries": box_entries,
                    "Counterattacks": counter_attacks,
                }
            all_tables["Attacks"] = pd.DataFrame(attack_stats).rename_axis("Attacks").fillna(0)
            print(f"      ... Attacks stats calculated.") # ADDED PRINT

            # --- Calculate "Defence" Stats ---
            defence_stats = {}
            for team in teams:
                opponent = away_team if team == home_team else home_team
                interceptions = match_df[(match_df.get('type.primary') == 'interception') & (match_df.get('team.name') == team)].shape[0]
                clearances = match_df[(match_df.get('type.primary') == 'clearance') & (match_df.get('team.name') == team)].shape[0]
                in_press_zone = match_df.get('location.x', 0) >= 40
                opponent_passes_df = match_df[(match_df.get('team.name') == opponent) & (match_df.get('type.primary') == 'pass') & in_press_zone]
                num_opponent_passes = opponent_passes_df.shape[0]
                team_def_actions_df = match_df[(match_df.get('team.name') == team) & in_press_zone]
                fouls = team_def_actions_df[team_def_actions_df.get('type.primary') == 'infraction'].shape[0]
                def_interceptions = team_def_actions_df[team_def_actions_df.get('type.primary') == 'interception'].shape[0]
                duels_in_zone_df = team_def_actions_df[team_def_actions_df.get('type.primary') == 'duel']
                defensive_duels_in_zone = duels_in_zone_df[duels_in_zone_df.get('groundDuel.duelType') == 'defensive_duel']
                won_defensive_duels = defensive_duels_in_zone[(defensive_duels_in_zone.get('groundDuel.recoveredPossession') == True) | (defensive_duels_in_zone.get('groundDuel.stoppedProgress') == True)].shape[0]
                sliding_tackles = duels_in_zone_df[duels_in_zone_df.get('groundDuel.duelType','').astype(str).str.contains('sliding_tackle', na=False)].shape[0]
                num_defensive_actions = fouls + def_interceptions + won_defensive_duels + sliding_tackles
                ppda = round(num_opponent_passes / num_defensive_actions, 1) if num_defensive_actions > 0 else 0
                defence_stats[team] = {
                    "Interceptions": interceptions, "Clearances": clearances,
                    "Passes allowed per def. action (PPDA)": ppda,
                }
            all_tables["Defence"] = pd.DataFrame(defence_stats).rename_axis("Defence").fillna(0)
            print(f"      ... Defence stats calculated.") # ADDED PRINT

            # --- Calculate "Transitions" Stats ---
            transitions_stats = {}
            if 'possession.id' in match_df.columns:
                match_df['next_possession.id'] = match_df['possession.id'].shift(-1)
                possession_changes = match_df[match_df['possession.id'] != match_df['next_possession.id']]
                losses_df_base = possession_changes[possession_changes.get('infraction.type') != 'foul_suffered'].copy()
                unsuccessful_pass_mask = (losses_df_base.get('type.primary') == 'pass') & (losses_df_base.get('pass.accurate') == False)
                # Use .loc with boolean mask for assignment, checking column existence
                if 'pass.endLocation.x' in losses_df_base.columns:
                     losses_df_base.loc[unsuccessful_pass_mask, 'location.x'] = losses_df_base.loc[unsuccessful_pass_mask, 'pass.endLocation.x']
                if 'pass.endLocation.y' in losses_df_base.columns:
                     losses_df_base.loc[unsuccessful_pass_mask, 'location.y'] = losses_df_base.loc[unsuccessful_pass_mask, 'pass.endLocation.y']

                for team in teams:
                    interceptions_df = match_df[(match_df.get('type.primary') == 'interception') & (match_df.get('team.name') == team)]
                    clearances_df = match_df[(match_df.get('type.primary') == 'clearance') & (match_df.get('team.name') == team)]
                    won_duels_df = match_df[(match_df.get('type.primary') == 'duel') & (match_df.get('team.name') == team) & (match_df.get('groundDuel.recoveredPossession') == True)]
                    recoveries_df = pd.concat([interceptions_df, clearances_df, won_duels_df])
                    total_recoveries = recoveries_df.shape[0]
                    low_recoveries = recoveries_df[pd.to_numeric(recoveries_df.get('location.x'), errors='coerce').fillna(0) <= DEFENSIVE_THIRD_X].shape[0]
                    mid_recoveries = recoveries_df[pd.to_numeric(recoveries_df.get('location.x'), errors='coerce').fillna(0).between(DEFENSIVE_THIRD_X, ATTACKING_THIRD_X)].shape[0]
                    high_recoveries = recoveries_df[pd.to_numeric(recoveries_df.get('location.x'), errors='coerce').fillna(0) > ATTACKING_THIRD_X].shape[0]
                    opponent_half_recoveries = recoveries_df[pd.to_numeric(recoveries_df.get('location.x'), errors='coerce').fillna(0) > 50].shape[0]
                    team_losses = losses_df_base[losses_df_base.get('team.name') == team]
                    total_losses = team_losses.shape[0]
                    low_losses = team_losses[pd.to_numeric(team_losses.get('location.x'), errors='coerce').fillna(0) <= DEFENSIVE_THIRD_X].shape[0]
                    mid_losses = team_losses[pd.to_numeric(team_losses.get('location.x'), errors='coerce').fillna(0).between(DEFENSIVE_THIRD_X, ATTACKING_THIRD_X)].shape[0]
                    high_losses = team_losses[pd.to_numeric(team_losses.get('location.x'), errors='coerce').fillna(0) > ATTACKING_THIRD_X].shape[0]
                    transitions_stats[team] = {
                        "Recoveries / low / medium / high": f"{total_recoveries}/{low_recoveries}/{mid_recoveries}/{high_recoveries}",
                        "Opponent half recoveries": opponent_half_recoveries,
                        "Losses / low / medium / high": f"{total_losses}/{low_losses}/{mid_losses}/{high_losses}",
                    }
                all_tables["Transitions"] = pd.DataFrame(transitions_stats).rename_axis("Transitions").fillna('0/0/0/0')
            else:
                 for team in teams: transitions_stats[team] = {"Recoveries / low / medium / high": 'N/A', "Opponent half recoveries": 'N/A', "Losses / low / medium / high": 'N/A'}
                 all_tables["Transitions"] = pd.DataFrame(transitions_stats).rename_axis("Transitions")
            print(f"      ... Transitions stats calculated.") # ADDED PRINT

            # --- Calculate "Duels" Stats ---
            duels_stats = {}
            all_duels_df = match_df[match_df.get('type.primary') == 'duel'].copy()
            total_possession_time = pd.Series(dtype='float')
            if 'possession.id' in match_df.columns and 'possession.team.name' in match_df.columns:
                total_possession_time = match_df.drop_duplicates(subset='possession.id').groupby('possession.team.name')['possession.duration_sec'].sum()

            for team in teams:
                opponent = away_team if team == home_team else home_team
                team_duels = all_duels_df[all_duels_df.get('team.name') == team]
                won_ground_duels_df = team_duels[(team_duels.get('groundDuel.keptPossession') == True) | (team_duels.get('groundDuel.recoveredPossession') == True)]
                aerial_duels = team_duels[team_duels.get('type.secondary','').astype(str).str.contains('aerial', na=False)]
                won_aerial_duels_df = aerial_duels[aerial_duels.get('aerialDuel.firstTouch') == True]
                total_duels_won = won_ground_duels_df.shape[0] + won_aerial_duels_df.shape[0]
                offensive_duels = team_duels[team_duels.get('groundDuel.duelType') == 'offensive_duel']
                won_offensive_duels = offensive_duels[(offensive_duels.get('groundDuel.keptPossession') == True) | (offensive_duels.get('groundDuel.recoveredPossession') == True)].shape[0]
                defensive_duels = team_duels[team_duels.get('groundDuel.duelType') == 'defensive_duel']
                won_defensive_duels = defensive_duels[(defensive_duels.get('groundDuel.recoveredPossession') == True) | (defensive_duels.get('groundDuel.stoppedProgress') == True)].shape[0]
                dribbles = team_duels[team_duels.get('groundDuel.takeOn') == True]
                successful_dribbles = dribbles[dribbles.get('groundDuel.progressedWithBall') == True].shape[0]
                defensive_duels_ci = defensive_duels.shape[0]
                interceptions_ci = match_df[(match_df.get('type.primary') == 'interception') & (match_df.get('team.name') == team)].shape[0]
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
            all_tables["Duels"] = pd.DataFrame(duels_stats).rename_axis("Duels").fillna('0/0')
            print(f"      ... Duels stats calculated.") # ADDED PRINT

            # --- Calculate "Possession" Stats ---
            possession_stats = {}
            home_possessions = match_df[match_df.get('possession.team.name') == home_team].drop_duplicates(subset='possession.id')
            away_possessions = match_df[match_df.get('possession.team.name') == away_team].drop_duplicates(subset='possession.id')
            home_time_sec = home_possessions.get('possession.duration_sec', 0).sum()
            away_time_sec = away_possessions.get('possession.duration_sec', 0).sum()
            total_time_in_possession = home_time_sec + away_time_sec
            total_match_duration_sec = 0
            if 'matchTimestamp' in match_df.columns and pd.api.types.is_datetime64_any_dtype(match_df['matchTimestamp']) and not match_df['matchTimestamp'].isna().all():
                 total_match_duration_sec = (match_df['matchTimestamp'].max() - match_df['matchTimestamp'].min()).total_seconds()
            dead_time_sec = total_match_duration_sec - total_time_in_possession if total_match_duration_sec > total_time_in_possession else 0

            for team in teams:
                team_possessions_df = match_df[match_df.get('possession.team.name') == team]
                unique_team_possessions = team_possessions_df.drop_duplicates(subset='possession.id')
                time_sec = unique_team_possessions.get('possession.duration_sec', 0).sum()
                possession_pct = round(time_sec / total_time_in_possession * 100) if total_time_in_possession > 0 else 0
                pure_time_str = str(datetime.timedelta(seconds=int(time_sec)))[2:] if pd.notna(time_sec) else '0:00'
                num_possessions = unique_team_possessions.shape[0]
                reaching_half = 0
                reaching_box = 0
                avg_duration_sec = np.nan
                if not team_possessions_df.empty and 'possession.id' in team_possessions_df.columns:
                     try:
                          half_entries = team_possessions_df.groupby('possession.id')['location.x'].transform('max') > 50
                          reaching_half = team_possessions_df.loc[half_entries, 'possession.id'].nunique()
                     except Exception: reaching_half = 0

                     possessions_in_box_ids = team_possessions_df[(pd.to_numeric(team_possessions_df.get('location.x'), errors='coerce').fillna(0) >= PENALTY_AREA_X) & (pd.to_numeric(team_possessions_df.get('location.y'), errors='coerce').fillna(0).between(PENALTY_AREA_Y1, PENALTY_AREA_Y2))]['possession.id'].unique()
                     reaching_box = len(possessions_in_box_ids)
                     avg_duration_sec = unique_team_possessions.get('possession.duration_sec', pd.Series(dtype='float')).mean()
                avg_duration_str = str(datetime.timedelta(seconds=int(avg_duration_sec)))[-5:] if pd.notna(avg_duration_sec) and avg_duration_sec > 0 else '00:00'
                possession_stats[team] = {
                    "Possession %": possession_pct, "Pure possession time": pure_time_str,
                    "Number of possessions": num_possessions,
                    "Possessions reaching opponent half": f"{reaching_half}",
                    "Possessions reaching opponent penalty area": f"{reaching_box}",
                    "Average possession duration": avg_duration_str,
                }
            possession_stats_df = pd.DataFrame(possession_stats).rename_axis("Possession")
            dead_time_str = str(datetime.timedelta(seconds=int(dead_time_sec)))[2:] if pd.notna(dead_time_sec) and dead_time_sec >=0 else '0:00'
            possession_stats_df.loc['Dead time'] = [dead_time_str, dead_time_str]
            all_tables["Possession"] = possession_stats_df.fillna('-')
            print(f"      ... Possession stats calculated.") # ADDED PRINT

            # --- Calculate "Open play possessions" Stats ---
            open_play_stats = {}
            if 'possession.id' in match_df.columns and 'matchTimestamp' in match_df.columns:
                try: # Protect idxmin operation
                    first_events_idx = match_df.groupby('possession.id')['matchTimestamp'].idxmin()
                    first_events_df = match_df.loc[first_events_idx]
                except KeyError:
                    first_events_df = pd.DataFrame() # Handle missing index

                open_play_possessions = first_events_df[~first_events_df.get('type.primary', '').isin(SET_PIECE_EVENTS)]
                for team in teams:
                    team_open_play = open_play_possessions[open_play_possessions.get('possession.team.name') == team]
                    total = team_open_play.shape[0]
                    # Ensure duration is numeric before comparisons
                    durations = pd.to_numeric(team_open_play.get('possession.duration_sec'), errors='coerce')
                    short = team_open_play[durations <= 10].shape[0]
                    medium = team_open_play[durations.between(10, 20, inclusive='right')].shape[0] # exclusive left, inclusive right
                    long = team_open_play[durations.between(20, 45, inclusive='right')].shape[0]
                    very_long = team_open_play[durations > 45].shape[0]
                    open_play_stats[team] = {
                        "Total": total, "Short (0-10 sec)": short, "Medium (10-20 sec)": medium,
                        "Long (20-45 sec)": long, "Very long (45+ sec)": very_long,
                    }
                all_tables["Open play possessions"] = pd.DataFrame(open_play_stats).rename_axis("Open play possessions").fillna(0)
            else:
                 for team in teams: open_play_stats[team] = {"Total": 0, "Short (0-10 sec)": 0, "Medium (10-20 sec)": 0, "Long (20-45 sec)": 0, "Very long (45+ sec)": 0}
                 all_tables["Open play possessions"] = pd.DataFrame(open_play_stats).rename_axis("Open play possessions")
            print(f"      ... Open Play stats calculated.") # ADDED PRINT

            # --- Calculate "Passes" Stats ---
            passes_stats = {}
            all_passes_df = match_df[match_df.get('type.primary') == 'pass'].copy()
            for team in teams:
                team_passes = all_passes_df[all_passes_df.get('team.name') == team].copy()
                accurate_passes = team_passes[team_passes.get('pass.accurate') == True]
                forward_passes = team_passes[pd.to_numeric(team_passes.get('pass.endLocation.x'), errors='coerce').fillna(-1) > pd.to_numeric(team_passes.get('location.x'), errors='coerce').fillna(-1)]
                back_passes = team_passes[pd.to_numeric(team_passes.get('pass.endLocation.x'), errors='coerce').fillna(-1) < pd.to_numeric(team_passes.get('location.x'), errors='coerce').fillna(-1)]
                lateral_passes = team_passes[pd.to_numeric(team_passes.get('pass.endLocation.x'), errors='coerce').fillna(-1) == pd.to_numeric(team_passes.get('location.x'), errors='coerce').fillna(-1)]

                # Convert locations to numeric before conditions
                loc_x_num = pd.to_numeric(team_passes.get('location.x'), errors='coerce')
                pass_end_x_num = pd.to_numeric(team_passes.get('pass.endLocation.x'), errors='coerce')
                pass_len_num = pd.to_numeric(team_passes.get('pass.length'), errors='coerce')

                prog_cond1 = (loc_x_num.fillna(101) < 60) & (pass_end_x_num.fillna(0) >= 60)
                prog_cond2 = (loc_x_num.fillna(0) >= 60) & (pass_end_x_num.fillna(0) >= 60) & (pass_len_num.fillna(0) >= 10)
                progressive_passes = team_passes[prog_cond1 | prog_cond2]
                long_passes = team_passes[pass_len_num.fillna(0) > 40]
                passes_to_final_third = team_passes[(loc_x_num.fillna(101) < ATTACKING_THIRD_X) & (pass_end_x_num.fillna(0) >= ATTACKING_THIRD_X)]

                pass_end_y_num = pd.to_numeric(team_passes.get('pass.endLocation.y'), errors='coerce')
                passes_to_box = team_passes[(pass_end_x_num.fillna(0) >= PENALTY_AREA_X) & (pass_end_y_num.fillna(0).between(PENALTY_AREA_Y1, PENALTY_AREA_Y2))]

                smart_passes = team_passes[team_passes.get('type.secondary','').astype(str).str.contains('smart_pass', na=False)]
                shot_assists = team_passes[team_passes.get('type.secondary','').astype(str).str.contains('shot_assist', na=False)]
                through_passes = team_passes[team_passes.get('type.secondary','').astype(str).str.contains('through_pass', na=False)]
                crosses = team_passes[team_passes.get('type.secondary','').astype(str).str.contains('cross', na=False)]
                low_crosses = crosses[crosses.get('pass.height','').astype(str).str.contains('low', na=False)].shape[0]
                high_crosses = crosses[crosses.get('pass.height','').astype(str).str.contains('high', na=False)].shape[0]

                # Add distance calculation only if pass location data exists
                if not team_passes.empty and 'pass.endLocation.x' in team_passes.columns and 'pass.endLocation.y' in team_passes.columns:
                     team_passes['dist_to_goal_end'] = np.sqrt((100 - pass_end_x_num.fillna(100))**2 + (50 - pass_end_y_num.fillna(50))**2)
                else:
                     team_passes['dist_to_goal_end'] = np.nan

                deep_completions = team_passes[(team_passes.get('pass.accurate') == True) & (team_passes.get('dist_to_goal_end', np.inf) <= 20) & (~team_passes.get('type.secondary','').astype(str).str.contains('cross', na=False))].shape[0]
                pure_possession_minutes = total_possession_time.get(team, 0) / 60
                match_tempo = round(team_passes.shape[0] / pure_possession_minutes, 1) if pure_possession_minutes > 0 else 0
                avg_pass_len = pass_len_num.mean()
                avg_pass_final_third_len = pd.to_numeric(passes_to_final_third.get('pass.length'), errors='coerce').mean()

                passes_stats[team] = {
                    "Total passes / accurate": f"{team_passes.shape[0]}/{accurate_passes.shape[0]}",
                    "Forward passes / accurate": f"{forward_passes.shape[0]}/{forward_passes[forward_passes.get('pass.accurate') == True].shape[0]}",
                    "Back passes / accurate": f"{back_passes.shape[0]}/{back_passes[back_passes.get('pass.accurate') == True].shape[0]}",
                    "Lateral passes / accurate": f"{lateral_passes.shape[0]}/{lateral_passes[lateral_passes.get('pass.accurate') == True].shape[0]}",
                    "Progressive passes / accurate": f"{progressive_passes.shape[0]}/{progressive_passes[progressive_passes.get('pass.accurate') == True].shape[0]}",
                    "Long passes / accurate": f"{long_passes.shape[0]}/{long_passes[long_passes.get('pass.accurate') == True].shape[0]}",
                    "Passes to final third / accurate": f"{passes_to_final_third.shape[0]}/{passes_to_final_third[passes_to_final_third.get('pass.accurate') == True].shape[0]}",
                    "Avg pass to final third length (m)": round(avg_pass_final_third_len, 1) if pd.notna(avg_pass_final_third_len) else '-',
                    "Passes to penalty area / accurate": f"{passes_to_box.shape[0]}/{passes_to_box[passes_to_box.get('pass.accurate') == True].shape[0]}",
                    "Smart passes / accurate": f"{smart_passes.shape[0]}/{smart_passes[smart_passes.get('pass.accurate') == True].shape[0]}",
                    "Shot assists": shot_assists.shape[0],
                    "Through passes / accurate": f"{through_passes.shape[0]}/{through_passes[through_passes.get('pass.accurate') == True].shape[0]}",
                    "Crosses / accurate": f"{crosses.shape[0]}/{crosses[crosses.get('pass.accurate') == True].shape[0]}",
                    "Crosses: low / high / blocked": f"{low_crosses}/{high_crosses}/{'N/A'}", # Blocked crosses unavailable
                    "Deep completions": deep_completions, "Match tempo": match_tempo,
                    "Average pass length (m)": round(avg_pass_len, 1) if pd.notna(avg_pass_len) else '-',
                }
            all_tables["Passes"] = pd.DataFrame(passes_stats).rename_axis("Passes").fillna('-')
            print(f"      ... Passes stats calculated.") # ADDED PRINT

            print(f"      ✅ Team Stats Calculation Successful.") # ADDED PRINT
            return all_tables

        except Exception as e:
            print(f"      ❌ ERROR calculating team stats for {home_team} vs {away_team}: {e}") # ADDED PRINT
            # Return empty structure on error
            return {cat: pd.DataFrame(columns=[home_team, away_team]).rename_axis(cat) for cat in ["General", "Attacks", "Defence", "Transitions", "Duels", "Possession", "Open play possessions", "Passes"]}


    # --- NESTED FUNCTION: PLAYER STATS (FULL LOGIC) ---
    def _calculate_player_stats(match_df, home_team, away_team):
        print(f"      Calculating Player Stats...") # ADDED PRINT

        def _calculate_stats_for_team(team_name, match_df):
            team_events = match_df[match_df.get('team.name') == team_name].copy()
            if team_events.empty:
                print(f"         ... No events found for {team_name}")
                return pd.DataFrame()

            # --- Minutes Played Logic ---
            starters = team_events['player.name'].dropna().unique()[:11]
            match_end_minute = match_df['minute'].max()
            player_minutes = {}
            for player in team_events['player.name'].dropna().unique():
                player_events_for_mins = team_events[team_events['player.name'] == player]
                if player_events_for_mins.empty:
                    continue
                # Use .get with default for safety
                minutes_series = pd.to_numeric(player_events_for_mins.get('minute'), errors='coerce')
                if player in starters:
                     player_minutes[player] = minutes_series.max() # Max minute they appeared in
                else:
                    minute_on = minutes_series.min() # Min minute they appeared in
                    if pd.notna(minute_on):
                         player_minutes[player] = match_end_minute - minute_on
                    else:
                         player_minutes[player] = 0 # Assign 0 if minute_on is NaN


            # --- Stat Calculation Loop ---
            player_stats_list = []
            players = team_events['player.name'].dropna().unique()
            for player in players:
                player_events = team_events[team_events['player.name'] == player]
                if player_events.empty:
                    continue

                player_shots = player_events[player_events.get('type.primary') == 'shot']
                goals = player_shots[player_shots.get('shot.isGoal') == True].shape[0]
                xg = player_shots.get('shot.xg', pd.Series(dtype='float')).sum()
                total_actions = player_events.shape[0]
                # Check for existence of pass.accurate and shot.onTarget before boolean logic
                pass_acc_series = player_events.get('pass.accurate') if 'pass.accurate' in player_events.columns else pd.Series([False]*len(player_events))
                shot_ot_series = player_events.get('shot.onTarget') if 'shot.onTarget' in player_events.columns else pd.Series([False]*len(player_events))
                successful_actions = player_events[(pass_acc_series == True) | (shot_ot_series == True)].shape[0]

                total_shots = player_shots.shape[0]
                shots_on_target = player_shots[player_shots.get('shot.onTarget') == True].shape[0]
                player_passes = player_events[player_events.get('type.primary') == 'pass']
                total_passes = player_passes.shape[0]
                accurate_passes = player_passes[player_passes.get('pass.accurate') == True].shape[0]
                crosses = player_passes[player_passes.get('type.secondary','').astype(str).str.contains('cross', na=False)]
                accurate_crosses = crosses[crosses.get('pass.accurate') == True].shape[0]
                player_duels = player_events[player_events.get('type.primary') == 'duel']
                # Use our new custom definitions
                dribbles = player_events[player_events.get('is_dribble_attempt') == True]
                successful_dribbles = dribbles[dribbles.get('is_custom_dribble_success') == True].shape[0]
                total_duels = player_duels.shape[0]
                won_ground_duels = player_duels[(player_duels.get('groundDuel.keptPossession') == True) | (player_duels.get('groundDuel.recoveredPossession') == True)].shape[0]
                won_aerial_duels = player_duels[player_duels.get('aerialDuel.firstTouch') == True].shape[0]
                duels_won = won_ground_duels + won_aerial_duels
                # Corrected recovery logic
                interceptions_clearances = player_events[player_events.get('type.primary', '').isin(['interception', 'clearance'])]
                recoveries = interceptions_clearances.shape[0] + won_ground_duels # Sum counts
                # Use .get with default 0 for location.x
                recoveries_opp_half = interceptions_clearances[pd.to_numeric(interceptions_clearances.get('location.x'), errors='coerce').fillna(0) > 50].shape[0] + \
                                      player_duels[(player_duels.get('groundDuel.recoveredPossession') == True) & (pd.to_numeric(player_duels.get('location.x'), errors='coerce').fillna(0) > 50)].shape[0]

                losses = player_passes[player_passes.get('pass.accurate') == False].shape[0]
                losses_own_half = player_passes[(player_passes.get('pass.accurate') == False) & (pd.to_numeric(player_passes.get('location.x'), errors='coerce').fillna(101) <= 50)].shape[0]
                # Use .get with default 0 for location.x
                touches_in_box = player_events[(player_events.get('type.primary') == 'touch') & (pd.to_numeric(player_events.get('location.x'), errors='coerce').fillna(0) >= 83)].shape[0]
                offsides = player_events[player_events.get('type.primary') == 'offside'].shape[0]
                yellow_cards = player_events[player_events.get('infraction.yellowCard') == True].shape[0]
                red_cards = player_events[player_events.get('infraction.redCard') == True].shape[0]
                
                # Ensure minutes value is valid before formatting
                minutes_val = player_minutes.get(player)
                minutes_display = f"{int(minutes_val)}'" if pd.notna(minutes_val) else "N/A"


                player_stats_list.append({
                    "Player": player, "Minutes": minutes_display, # Changed column name slightly for clarity
                    "Goals / xG": f"{goals}/{round(xg, 2) if pd.notna(xg) and xg > 0 else '-'}",
                    "Actions / successful": f"{total_actions}/{successful_actions}", "Shots / on target": f"{total_shots}/{shots_on_target}",
                    "Passes / accurate": f"{total_passes}/{accurate_passes}", "Crosses / accurate": f"{crosses.shape[0]}/{accurate_crosses}",
                    "Dribbles / successful": f"{dribbles.shape[0]}/{successful_dribbles}", "Duels / won": f"{total_duels}/{duels_won}",
                    "Losses / own half": f"{losses}/{losses_own_half}", "Recoveries / opponent half": f"{recoveries}/{recoveries_opp_half}",
                    "Touches in penalty area": touches_in_box if touches_in_box > 0 else "-", "Offsides": offsides if offsides > 0 else "-",
                    "Yellow / Red cards": f"{int(yellow_cards)}/{int(red_cards)}" if (yellow_cards > 0 or red_cards > 0) else "-",
                })
            print(f"         ... Calculated stats for {len(player_stats_list)} players on {team_name}") # ADDED PRINT
            return pd.DataFrame(player_stats_list).set_index('Player')

        try: # Added try block around player stat calculation
            home_stats = _calculate_stats_for_team(home_team, match_df)
            away_stats = _calculate_stats_for_team(away_team, match_df)
            print(f"      ✅ Player Stats Calculation Successful.") # ADDED PRINT
            return {'home': home_stats, 'away': away_stats}
        except Exception as e:
            print(f"      ❌ ERROR calculating player stats for {home_team} vs {away_team}: {e}") # ADDED PRINT
            return {'home': pd.DataFrame(), 'away': pd.DataFrame()} # Return empty structure

    # --- EXECUTE AND RETURN ---
    team_stats = _calculate_team_stats(match_df.copy(), home_team, away_team)
    player_stats = _calculate_player_stats(match_df.copy(), home_team, away_team)

    # Check if stats were successfully calculated before returning
    if not team_stats or not any(isinstance(df, pd.DataFrame) and not df.empty for df in team_stats.values()):
         print(f"  -> ⚠️ Warning: Team stats calculation returned empty or invalid data for {home_team} vs {away_team}")
    if not player_stats or (not isinstance(player_stats.get('home'), pd.DataFrame) or player_stats['home'].empty) and \
                           (not isinstance(player_stats.get('away'), pd.DataFrame) or player_stats['away'].empty):
         print(f"  -> ⚠️ Warning: Player stats calculation returned empty or invalid data for {home_team} vs {away_team}")


    return {'team_stats': team_stats, 'player_stats': player_stats}


# ==============================================================================
# SECTION 4: MAIN FUNCTION (REVISED FOR MULTI-SEASON)
# ==============================================================================
def main():
    """Main function to run the entire data processing pipeline."""
    # --- Credentials ---
    load_dotenv()
    from league_config import COMPETITIONS, get_credentials, competition_for_season

    # Validate credentials for all competitions
    for comp_id, comp_config in COMPETITIONS.items():
        user, password = get_credentials(comp_id)
        if not user or not password:
            print(f"⚠️ Warning: Credentials not found for {comp_config['name']} (comp {comp_id}). Skipping.")

    # Build combined season lists
    ALL_SEASON_IDS_TO_FETCH = []
    CURRENT_SEASON_IDS = []
    for comp_id, comp_config in COMPETITIONS.items():
        user, password = get_credentials(comp_id)
        if not user or not password:
            continue
        ALL_SEASON_IDS_TO_FETCH.extend(comp_config["seasons"].keys())
        CURRENT_SEASON_IDS.append(comp_config["current_season"])

    HISTORICAL_SEASON_IDS = [sid for sid in ALL_SEASON_IDS_TO_FETCH if sid not in CURRENT_SEASON_IDS]

    print(f"Starting data pipeline for {len(ALL_SEASON_IDS_TO_FETCH)} total seasons across {len(COMPETITIONS)} competitions.")
    for comp_id, comp_config in COMPETITIONS.items():
        user, _ = get_credentials(comp_id)
        status = "✅" if user else "❌ (no credentials)"
        print(f"  {comp_config['name']}: {len(comp_config['seasons'])} seasons, current={comp_config['current_season']} {status}")
    print(f"Historical seasons (cached): {HISTORICAL_SEASON_IDS}")

    # --- 2. FETCH MATCH SCHEDULES (INCREMENTAL) ---
    cached_schedule_df = None
    if os.path.exists('matches_summary.parquet'):
        print("📂 Loading cached schedule from matches_summary.parquet...")
        cached_schedule_df = pd.read_parquet('matches_summary.parquet')
        print(f"   Cached schedule has {len(cached_schedule_df)} matches.")

    all_schedules_list = []
    if cached_schedule_df is not None:
        # Keep historical season rows from cache
        historical_cached = cached_schedule_df[cached_schedule_df['seasonId'].isin(HISTORICAL_SEASON_IDS)]
        if not historical_cached.empty:
            all_schedules_list.append(historical_cached)
            print(f"   ✅ Reusing {len(historical_cached)} cached historical matches.")

    # Determine which seasons to fetch per competition
    for comp_id, comp_config in COMPETITIONS.items():
        user, password = get_credentials(comp_id)
        if not user or not password:
            continue
        comp_name = comp_config["name"]
        comp_seasons = list(comp_config["seasons"].keys())
        current_sid = comp_config["current_season"]

        if cached_schedule_df is not None:
            # Only fetch current season for this competition
            seasons_to_fetch = [current_sid]
            # Also fetch any historical seasons not in cache
            cached_season_ids = set(cached_schedule_df['seasonId'].unique())
            for sid in comp_seasons:
                if sid != current_sid and sid not in cached_season_ids:
                    seasons_to_fetch.append(sid)
        else:
            seasons_to_fetch = comp_seasons

        for season_id in seasons_to_fetch:
            print(f"\n--- Fetching Schedule for {comp_name}, Season ID: {season_id} ---")
            season_schedule_df = fetch_match_schedule(user, password, comp_id, season_id)
            if not season_schedule_df.empty:
                season_schedule_df['competitionId'] = comp_id
                all_schedules_list.append(season_schedule_df)
            else:
                print(f"Warning: No match schedule found for {comp_name} season {season_id}.")

    if not all_schedules_list:
        print("❌ No match schedules found for any season. Exiting.")
        return

    # Combine all schedules, deduplicate by matchId
    all_matches_summary_df = pd.concat(all_schedules_list, ignore_index=True)
    all_matches_summary_df.drop_duplicates(subset='matchId', keep='last', inplace=True)
    all_matches_summary_df['dateutc'] = pd.to_datetime(all_matches_summary_df['dateutc'], errors='coerce')
    all_matches_summary_df.sort_values(by='dateutc', inplace=True, na_position='last')
    all_matches_summary_df.reset_index(drop=True, inplace=True)

    # Backfill competitionId for cached rows that don't have it
    if 'competitionId' not in all_matches_summary_df.columns:
        all_matches_summary_df['competitionId'] = all_matches_summary_df['seasonId'].map(
            lambda sid: competition_for_season(sid)
        )
    elif all_matches_summary_df['competitionId'].isna().any():
        mask = all_matches_summary_df['competitionId'].isna()
        all_matches_summary_df.loc[mask, 'competitionId'] = all_matches_summary_df.loc[mask, 'seasonId'].map(
            lambda sid: competition_for_season(sid)
        )

    print(f"\n✅ Combined schedule: {len(all_matches_summary_df)} total matches ({len(seasons_to_fetch)} season(s) fetched from API).")

    # --- 3. SAVE UNIFIED MATCH SCHEDULE (ALL SEASONS) ---
    all_matches_summary_df.to_parquet('matches_summary.parquet', index=False)
    print(f"✅ Unified match schedule (all seasons) saved to 'matches_summary.parquet'")

    # --- 4. FETCH EVENT DATA (INCREMENTAL) ---
    # Memory-efficient approach: read only matchId column to determine cache,
    # fetch new events, clean & save them, then load full dataset only once.
    import pyarrow.parquet as pq

    cached_match_ids = set()
    has_cache = os.path.exists('raw_events.parquet')
    if has_cache:
        print("📂 Checking cached events in raw_events.parquet...")
        cached_match_col = pq.read_table('raw_events.parquet', columns=['matchId']).to_pandas()['matchId']
        cached_match_ids = set(cached_match_col.dropna().unique())
        del cached_match_col
        print(f"   Cached events cover {len(cached_match_ids)} matches.")

    # Only fetch events for "Played" matches not already in cache
    played_matches = all_matches_summary_df[all_matches_summary_df['status'] == 'Played']
    new_match_ids = [mid for mid in played_matches['matchId'].dropna().unique() if mid not in cached_match_ids]
    print(f"   {len(new_match_ids)} new played matches need event fetching.")

    # TEST_LIMIT: set env var to cap new matches per competition (e.g. TEST_FETCH_LIMIT=10)
    test_limit = int(os.environ.get('TEST_FETCH_LIMIT', 0))
    if test_limit:
        print(f"   ⚠️ TEST MODE: limiting to {test_limit} new matches per competition")

    if new_match_ids:
        # Group new matches by competition to use correct credentials
        match_comp_map = all_matches_summary_df.set_index('matchId')['competitionId'].to_dict()
        comp_match_groups = {}
        for mid in new_match_ids:
            cid = match_comp_map.get(mid)
            if cid is not None:
                comp_match_groups.setdefault(cid, []).append(mid)

        new_events_list = []
        for cid, mids in comp_match_groups.items():
            user, password = get_credentials(cid)
            if not user or not password:
                print(f"⚠️ Skipping {len(mids)} matches for comp {cid} (no credentials)")
                continue
            comp_name = COMPETITIONS[cid]["name"]
            if test_limit:
                mids = mids[:test_limit]
            print(f"\nFetching events for {len(mids)} {comp_name} matches...")
            comp_events = fetch_events(user, password, mids)
            if not comp_events.empty:
                new_events_list.append(comp_events)

        if new_events_list:
            new_events_df = pd.concat(new_events_list, ignore_index=True)
            del new_events_list  # Free memory

            # Clean new events before saving
            match_id_to_season_map = all_matches_summary_df.set_index('matchId')['seasonId']
            match_id_to_comp_map = all_matches_summary_df.set_index('matchId')['competitionId']
            new_events_df['seasonId'] = new_events_df['matchId'].map(match_id_to_season_map)
            new_events_df['competitionId'] = new_events_df['matchId'].map(match_id_to_comp_map)

            # Apply type conversions to new events
            numeric_cols = ['shot.xg', 'location.x', 'location.y', 'minute', 'second', 'pass.length',
                            'pass.endLocation.x', 'pass.endLocation.y', 'possession.duration_sec']
            bool_cols = ['shot.isGoal', 'shot.onTarget', 'pass.accurate', 'groundDuel.keptPossession',
                         'groundDuel.recoveredPossession', 'groundDuel.stoppedProgress', 'aerialDuel.firstTouch',
                         'groundDuel.takeOn', 'groundDuel.progressedWithBall', 'infraction.yellowCard', 'infraction.redCard']
            for col in numeric_cols:
                if col in new_events_df.columns:
                    new_events_df[col] = pd.to_numeric(new_events_df[col], errors='coerce')
            for col in bool_cols:
                if col in new_events_df.columns:
                    new_events_df[col] = new_events_df[col].replace({'true': True, 'false': False, 1: True, 0: False})
                    try: new_events_df[col] = new_events_df[col].astype('boolean')
                    except Exception: new_events_df[col] = new_events_df[col].apply(lambda x: True if x == True else (False if x == False else pd.NA))

            new_events_df = add_custom_dribble_success(new_events_df)
            fk_shot_mask = (new_events_df['type.primary'] == 'free_kick') & (new_events_df['shot.xg'].notna())
            if fk_shot_mask.sum() > 0:
                new_events_df.loc[fk_shot_mask, 'type.primary'] = 'shot'

            # Save new events to a temp file, then merge parquet files without loading both into memory
            new_events_df.to_parquet('_new_events_tmp.parquet', index=False, compression='zstd')
            print(f"   New events saved ({len(new_events_df)} rows)")
            del new_events_df  # Free memory

            if has_cache:
                # Merge parquet files using pyarrow — never load both into pandas
                # New events are for match IDs NOT in cache, so no dedup needed.
                print("Merging cached + new events via pyarrow (append-only)...")
                import pyarrow as pa

                cached_table = pq.read_table('raw_events.parquet')
                new_table = pq.read_table('_new_events_tmp.parquet')

                # Build a unified schema: start with cached types, add new-only columns
                target_fields = {f.name: f for f in cached_table.schema}
                for field in new_table.schema:
                    if field.name not in target_fields:
                        target_fields[field.name] = field
                target_schema = pa.schema(list(target_fields.values()))

                # Add missing columns as null to each table, then cast types
                for field in target_schema:
                    if field.name not in cached_table.schema.names:
                        cached_table = cached_table.append_column(
                            field, pa.nulls(cached_table.num_rows, type=field.type))
                    if field.name not in new_table.schema.names:
                        new_table = new_table.append_column(
                            field, pa.nulls(new_table.num_rows, type=field.type))

                # Reorder columns to match target schema, then cast types
                cached_table = cached_table.select(target_schema.names).cast(target_schema, safe=False)
                new_table = new_table.select(target_schema.names).cast(target_schema, safe=False)
                merged = pa.concat_tables([cached_table, new_table])
                del cached_table, new_table  # Free before writing
                pq.write_table(merged, 'raw_events.parquet', compression='zstd')
                print(f"   Merged: {merged.num_rows:,} total rows")
                del merged
            else:
                os.rename('_new_events_tmp.parquet', 'raw_events.parquet')

            if os.path.exists('_new_events_tmp.parquet'):
                os.remove('_new_events_tmp.parquet')
            print(f"✅ Unified event data saved to 'raw_events.parquet'")
        else:
            print("No new events fetched.")
    else:
        print("No new matches to fetch.")

    # --- 5. LOAD FULL EVENTS FOR DOWNSTREAM PROCESSING ---
    # Now load the complete parquet for match processing / season stats
    if not os.path.exists('raw_events.parquet'):
        print("No event data found. Exiting.")
        return
    print("📂 Loading full event data for downstream processing...")
    all_raw_events_df = pd.read_parquet('raw_events.parquet')

    # Ensure seasonId and competitionId are set (for old cached rows that may lack them)
    match_id_to_season_map = all_matches_summary_df.set_index('matchId')['seasonId']
    match_id_to_comp_map = all_matches_summary_df.set_index('matchId')['competitionId']
    all_raw_events_df['seasonId'] = all_raw_events_df['matchId'].map(match_id_to_season_map)
    all_raw_events_df['competitionId'] = all_raw_events_df['matchId'].map(match_id_to_comp_map)
    print(f"   Loaded {len(all_raw_events_df):,} events across {all_raw_events_df['matchId'].nunique()} matches.")

    # --- 7. PROCESS PER-MATCH DATA (INCREMENTAL) ---
    all_match_data = {}
    if os.path.exists('all_match_data.pkl'):
        print("📂 Loading cached per-match data from all_match_data.pkl...")
        with open('all_match_data.pkl', 'rb') as f:
            all_match_data = pickle.load(f)
        print(f"   Cached data for {len(all_match_data)} matches.")

    # Only process played matches (unplayed ones have no events)
    played_matches_for_data = all_matches_summary_df[all_matches_summary_df['status'] == 'Played']

    # Always reprocess current season (new matches may have been played since last run)
    current_season_match_ids = set(
        played_matches_for_data[played_matches_for_data['seasonId'].isin(CURRENT_SEASON_IDS)]['matchId']
    )
    cached_match_data_ids = set(all_match_data.keys()) - current_season_match_ids
    matches_to_process = played_matches_for_data[~played_matches_for_data['matchId'].isin(cached_match_data_ids)]
    print(f"   {len(matches_to_process)} matches to process (current season always refreshed).")

    for index, match_summary in tqdm(matches_to_process.iterrows(), total=len(matches_to_process), desc="Processing New Matches"):
        match_id = match_summary['matchId']
        home_team = match_summary['homeTeamName']
        away_team = match_summary['awayTeamName']
        if pd.isna(home_team) or pd.isna(away_team):
            print(f"  -> ⚠️ Warning: Missing team names for match {match_id}. Skipping stats calc.")
            continue

        match_events_df = all_raw_events_df[all_raw_events_df['matchId'] == match_id].copy()

        if match_events_df.empty:
            print(f"  -> ℹ️ Info: No event data for match {match_id}. Skipping stats.")
            all_match_data[match_id] = {'team_stats': {}, 'player_stats': {'home': pd.DataFrame(), 'away': pd.DataFrame()}}
            continue

        match_data = calculate_match_data(match_events_df, home_team, away_team)
        all_match_data[match_id] = match_data

    with open('all_match_data.pkl', 'wb') as f: pickle.dump(all_match_data, f)
    print("✅ All detailed match data (all seasons) saved to 'all_match_data.pkl'")

    # --- 8. PROCESS SEASON-LONG STATS (INCREMENTAL) ---
    season_team_stats = {}
    if os.path.exists('season_team_stats.pkl'):
        print("📂 Loading cached season team stats from season_team_stats.pkl...")
        with open('season_team_stats.pkl', 'rb') as f:
            season_team_stats = pickle.load(f)
        print(f"   Cached stats for {len(season_team_stats)} seasons.")

    # Determine which seasons need (re)computing
    seasons_to_compute = list(CURRENT_SEASON_IDS)  # Always recompute current seasons
    for sid in HISTORICAL_SEASON_IDS:
        if sid not in season_team_stats:
            seasons_to_compute.append(sid)  # First run: also compute missing historical
    print(f"   Recomputing team stats for {len(seasons_to_compute)} season(s): {seasons_to_compute}")

    for season_id in seasons_to_compute:
        season_events = all_raw_events_df[all_raw_events_df['seasonId'] == season_id].copy()
        season_matches = all_matches_summary_df[all_matches_summary_df['seasonId'] == season_id].copy()
        all_teams = pd.concat([season_matches['homeTeamName'], season_matches['awayTeamName']]).dropna().unique()
        season_team_stats[season_id] = {}
        for team in tqdm(all_teams, desc=f"Processing Season {season_id} Team Stats"):
            team_corners = calculate_team_corner_stats(season_events.copy(), season_matches, team)
            if team_corners is not None:
                season_team_stats[season_id][team] = {'corners': team_corners}

    with open('season_team_stats.pkl', 'wb') as f: pickle.dump(season_team_stats, f)
    print("✅ Per-season team stats saved to 'season_team_stats.pkl'")

    # Free large DataFrames no longer needed — critical for CI memory
    del all_raw_events_df, all_match_data
    import gc; gc.collect()
    print("🧹 Freed events & match data from memory for player minutes step.")

    # --- 9. FETCH OFFICIAL PLAYER MINUTES (INCREMENTAL) ---
    print("\nStarting official player minute retrieval (incremental)...")

    player_minutes_by_season = {}
    all_match_lineups = {}  # Accumulate lineup/substitution data across seasons
    if os.path.exists('player_minutes_and_positions.pkl'):
        print("📂 Loading cached player minutes from player_minutes_and_positions.pkl...")
        with open('player_minutes_and_positions.pkl', 'rb') as f:
            player_minutes_by_season = pickle.load(f)
        print(f"   Cached player minutes for {len(player_minutes_by_season)} seasons.")
    if os.path.exists('match_lineups.pkl'):
        print("📂 Loading cached match lineups from match_lineups.pkl...")
        with open('match_lineups.pkl', 'rb') as f:
            all_match_lineups = pickle.load(f)
        print(f"   Cached lineups for {len(all_match_lineups)} matches.")

    # Determine which seasons need fetching
    seasons_to_fetch_minutes = list(CURRENT_SEASON_IDS)  # Always re-fetch current seasons
    for sid in HISTORICAL_SEASON_IDS:
        if sid not in player_minutes_by_season:
            seasons_to_fetch_minutes.append(sid)
    print(f"   Fetching player minutes for {len(seasons_to_fetch_minutes)} season(s): {seasons_to_fetch_minutes}")

    for season_id in seasons_to_fetch_minutes:
        comp_id = competition_for_season(season_id)
        if comp_id is None:
            print(f"⚠️ Unknown competition for season {season_id}, skipping.")
            continue
        user, password = get_credentials(comp_id)
        if not user or not password:
            print(f"⚠️ No credentials for {COMPETITIONS[comp_id]['name']}, skipping season {season_id}.")
            continue

        season_matches_df = all_matches_summary_df[all_matches_summary_df['seasonId'] == season_id]
        season_match_ids = season_matches_df['matchId'].dropna().unique().tolist()
        # Load only this season's events from parquet (memory-efficient row filtering)
        season_events = pd.read_parquet('raw_events.parquet',
            filters=[('seasonId', '==', season_id)])

        print(f"\n--- Fetching player minutes for {COMPETITIONS[comp_id]['name']} season {season_id} ({len(season_match_ids)} matches) ---")
        season_minutes_df, season_lineups = fetch_official_player_minutes(
            user,
            password,
            season_match_ids,
            season_events,
            competition_id=comp_id,
            season_id=season_id
        )

        if not season_minutes_df.empty:
            season_minutes_df['competitionId'] = comp_id
            player_minutes_by_season[season_id] = season_minutes_df
            print(f"✅ Season {season_id}: {len(season_minutes_df)} player records.")
        else:
            print(f"⚠️ Season {season_id}: No player minutes data.")

        all_match_lineups.update(season_lineups)

    # Save match lineups (lineup, bench, substitutions per match per team)
    with open('match_lineups.pkl', 'wb') as f:
        pickle.dump(all_match_lineups, f)
    print(f"✅ Match lineups/substitutions saved to 'match_lineups.pkl' ({len(all_match_lineups)} matches)")

    # Save per-season dict
    with open('player_minutes_and_positions.pkl', 'wb') as f:
        pickle.dump(player_minutes_by_season, f)
    print("✅ Per-season player minutes saved to 'player_minutes_and_positions.pkl'")

    # Build all-time player minutes (aggregate across all seasons)
    all_season_minutes_list = [df for df in player_minutes_by_season.values() if not df.empty]
    if all_season_minutes_list:
        combined_df = pd.concat(all_season_minutes_list)
        all_time_df = combined_df.groupby('playerId').agg({
            'playerName': 'first',
            'teamName': 'first',
            'primaryPosition': 'first',
            'totalMinutes': 'sum'
        }).reset_index()
        all_time_df.to_pickle('all_time_player_minutes.pkl')
        print(f"✅ All-time career stats saved: {len(all_time_df)} total players.")
    else:
        print("❌ Failed to fetch player minutes for any season.")

    # --- 10. PRECOMPUTE COMPLETE PLAYER MINUTES (3-source merge) ---
    print("\nPrecomputing complete player minutes (API + lineup + event sources)...")
    from precompute_minutes import main as precompute_minutes_main
    precompute_minutes_main()

    print("\n🎉 Data processing pipeline complete!")


if __name__ == "__main__":
    main()