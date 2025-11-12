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

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')
pd.options.mode.chained_assignment = None

# --- Configuration ---
EVENTS_FILE = 'historical_events.parquet'
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
    wyscout_user = os.getenv("WYSCOUT_USER")
    wyscout_pass = os.getenv("WYSCOUT_PASS")
    if not wyscout_user or not wyscout_pass:
        print("❌ Error: Wyscout credentials not found in .env file.")
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

    # --- 3. Find All Player IDs from Event Data ---
    try:
        events_df = pd.read_parquet(EVENTS_FILE, columns=['player.id'])
    except FileNotFoundError:
        print(f"❌ Error: {EVENTS_FILE} not found. Run process_data.py first.")
        return
    except Exception as e:
        print(f"❌ Error loading {EVENTS_FILE}: {e}")
        return
        
    all_player_ids = events_df['player.id'].dropna().unique()
    all_player_ids = {int(pid) for pid in all_player_ids if pid != 0}
    print(f"Found {len(all_player_ids)} unique player IDs in event data.")

    # --- 4. Determine Which New Players to Fetch ---
    new_player_ids_to_fetch = list(all_player_ids - existing_ids)
    
    if not new_player_ids_to_fetch:
        print("✅ All player details are already up-to-date.")
        combined_data = existing_data 
    else:
        # --- 5. Fetch New Data ---
        session = setup_session()
        new_player_data = fetch_player_details(new_player_ids_to_fetch, wyscout_user, wyscout_pass, session)
        
        # --- 6. Combine and Save ---
        combined_data = existing_data + new_player_data
        with open(OUTPUT_FILE, 'wb') as f:
            pickle.dump(combined_data, f)
        print(f"✅ Successfully saved {len(combined_data)} total player records to {OUTPUT_FILE}.")

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