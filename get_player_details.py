import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
from tqdm import tqdm
import os
import pickle
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv
import warnings
from league_config import COMPETITIONS, get_credentials, competition_for_season

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')
pd.options.mode.chained_assignment = None

# --- Configuration ---
EVENTS_FILE = 'raw_events.parquet'
OUTPUT_FILE = 'player_details.pkl'

def setup_session():
    """Sets up a requests session with retries."""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

def fetch_player_details(player_ids, username, password, session):
    """
    Fetches details for a list of player IDs from the /players/{id} endpoint.
    """
    base_url = "https://apirest.wyscout.com/v3/players"
    auth = HTTPBasicAuth(username, password)
    player_data_list = []
    
    print(f"Fetching details for {len(player_ids)} new players...")
    for player_id in tqdm(player_ids, desc="Fetching Player Details"):
        if pd.isna(player_id):
            continue
            
        url = f"{base_url}/{int(player_id)}"
        
        try:
            r = session.get(url, auth=auth, timeout=10)
            if r.status_code == 404:
                continue
            r.raise_for_status()
            
            data = r.json()
            
            # --- NEW: Extract all biographical data ---
            birth_area = data.get('birthArea', {}).get('name', 'N/A')
            passport_area = data.get('passportArea', {}).get('name', 'N/A')
            role = data.get('role', {}).get('name', 'N/A')
            image_url = data.get('imageDataURL', None)
            
            player_data_list.append({
                'playerId': player_id,
                'firstName': data.get('firstName'),
                'lastName': data.get('lastName'),
                'shortName': data.get('shortName'),
                'foot': data.get('foot'),
                'height': data.get('height'),
                'weight': data.get('weight'),
                'birthDate': data.get('birthDate'),
                'birthArea': birth_area,         # <-- NEW
                'passportArea': passport_area,   # <-- NEW
                'role': role,                  # <-- NEW
                'imageDataURL': image_url      # <-- NEW
            })
            
        except requests.exceptions.RequestException as e:
            if '400 Client Error' not in str(e):
                print(f"Error fetching player {player_id}: {e}")
            
    return player_data_list

def main():
    """Main function to load events, find players, and fetch details."""
    # --- 1. Load Credentials ---
    load_dotenv()

    # Validate that at least one credential set is available
    creds_available = {}
    for comp_id, comp in COMPETITIONS.items():
        user, pw = get_credentials(comp_id)
        if user and pw:
            creds_available[comp_id] = (user, pw)
        else:
            print(f"Warning: No credentials for {comp['name']} (competition {comp_id}), skipping.")

    if not creds_available:
        print("Error: No Wyscout credentials found in .env file.")
        return

    # --- 2. Load Existing Player Details (if any) ---
    existing_data = []
    existing_ids = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'rb') as f:
                existing_data = pickle.load(f)
            existing_ids = {p['playerId'] for p in existing_data if 'playerId' in p and p['playerId'] is not None}
            print(f"Loaded {len(existing_data)} existing player detail records.")
        except Exception as e:
            print(f"Warning: Could not load existing file {OUTPUT_FILE}. Re-fetching all. Error: {e}")
            existing_data = []
            existing_ids = set()

    # --- 3. Find All Player IDs from Event Data, grouped by competition ---
    try:
        events_df = pd.read_parquet(EVENTS_FILE, columns=['player.id', 'competitionId'])
    except FileNotFoundError:
        print(f"Error: {EVENTS_FILE} not found. Run process_data.py first.")
        return
    except Exception as e:
        print(f"Error loading {EVENTS_FILE}: {e}")
        return

    events_df = events_df.dropna(subset=['player.id'])
    events_df = events_df[events_df['player.id'] != 0]
    events_df['player.id'] = events_df['player.id'].astype(int)
    events_df['competitionId'] = events_df['competitionId'].astype(int)

    all_player_ids = set(events_df['player.id'].unique())
    print(f"Found {len(all_player_ids)} unique player IDs in event data.")

    # Group player IDs by their competition
    players_by_comp = {}
    for comp_id in events_df['competitionId'].unique():
        comp_players = set(events_df.loc[events_df['competitionId'] == comp_id, 'player.id'].unique())
        players_by_comp[int(comp_id)] = comp_players

    # --- 4. Determine Which New Players to Fetch ---
    new_player_ids = all_player_ids - existing_ids

    if not new_player_ids:
        print("All player details are already up-to-date.")
        combined_data = existing_data
    else:
        # --- 5. Fetch New Data per competition ---
        session = setup_session()
        new_player_data = []
        fetched_ids = set()

        # Process each competition's players with the matching credentials
        for comp_id, comp_player_ids in players_by_comp.items():
            if comp_id not in creds_available:
                print(f"Skipping competition {comp_id} (no credentials).")
                continue

            # Only fetch players that are new AND not yet fetched by another competition
            ids_to_fetch = list(comp_player_ids & new_player_ids - fetched_ids)
            if not ids_to_fetch:
                continue

            user, pw = creds_available[comp_id]
            comp_name = COMPETITIONS[comp_id]["name"]
            print(f"\nFetching {len(ids_to_fetch)} players for {comp_name} (competition {comp_id})...")
            results = fetch_player_details(ids_to_fetch, user, pw, session)
            new_player_data.extend(results)
            fetched_ids.update(r['playerId'] for r in results)

        # Handle players that appear in both leagues but weren't found yet
        # (e.g. player in Campeonato events but only fetchable with Liga 3 creds)
        still_missing = new_player_ids - fetched_ids - existing_ids
        if still_missing:
            for comp_id, (user, pw) in creds_available.items():
                remaining = list(still_missing - fetched_ids)
                if not remaining:
                    break
                comp_name = COMPETITIONS[comp_id]["name"]
                print(f"\nRetrying {len(remaining)} missing players with {comp_name} credentials...")
                results = fetch_player_details(remaining, user, pw, session)
                new_player_data.extend(results)
                fetched_ids.update(r['playerId'] for r in results)

        # --- 6. Combine and Save ---
        combined_data = existing_data + new_player_data
        with open(OUTPUT_FILE, 'wb') as f:
            pickle.dump(combined_data, f)
        print(f"\nSuccessfully saved {len(combined_data)} total player records to {OUTPUT_FILE}.")

    # --- 7. Load and Show Results ---
    print("\n--- Player Details Summary ---")
    if not combined_data:
        print("No player data found or fetched.")
        return
        
    try:
        final_df = pd.DataFrame(combined_data)
        final_df = final_df.dropna(subset=['playerId'])
        
        final_df['fullName'] = final_df['firstName'].fillna('') + ' ' + final_df['lastName'].fillna('')
        final_df['fullName'] = final_df['fullName'].str.strip()
        
        # --- UPDATED: Display new fields in summary ---
        display_cols = ['playerId', 'fullName', 'shortName', 'foot', 'role', 'passportArea', 'birthArea']
        print(final_df[display_cols].head())
        
        print("\nFootedness distribution:")
        print(final_df['foot'].value_counts())
    except Exception as e:
        print(f"Could not load final DataFrame for summary: {e}")

if __name__ == "__main__":
    main()