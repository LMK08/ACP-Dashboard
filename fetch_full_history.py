import requests
import pandas as pd
import os
import time
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth
from tqdm import tqdm

# --- 1. SETUP ---
load_dotenv()
WYSCOUT_USERNAME = os.getenv("WYSCOUT_USER")
WYSCOUT_PASSWORD = os.getenv("WYSCOUT_PASS")

if not WYSCOUT_USERNAME or not WYSCOUT_PASSWORD:
    print("❌ Credentials not found in .env")
    exit()

BASE_URL = "https://apirest.wyscout.com/v3"
AUTH = HTTPBasicAuth(WYSCOUT_USERNAME, WYSCOUT_PASSWORD)

# --- HARDCODED SEASON LIST ---
TARGET_SEASON_IDS = [
    191782, # 2025/26 Season (Current)
    190090, # 2024/25 Season
    189147, # 2023/24 Season
    188222, # 2022/23 Season
    188221, # 2021/22 Season
]

def fetch_matches_for_season(season_id):
    """Fetch all matches for a specific season"""
    r = requests.get(f"{BASE_URL}/seasons/{season_id}/matches", auth=AUTH)
    if r.status_code == 200:
        return r.json().get('matches', [])
    print(f"⚠️ Failed to fetch matches for season {season_id}: {r.status_code}")
    return []

def fetch_events_batch(match_ids):
    """Fetch events for a list of matches"""
    all_events = []
    
    # Using tqdm to show progress bar
    for mid in tqdm(match_ids, desc="   Downloading Events"):
        try:
            # Add a small sleep to be nice to the API if needed, usually Wyscout is okay
            # time.sleep(0.1) 
            r = requests.get(f"{BASE_URL}/matches/{mid}/events", auth=AUTH, timeout=15)
            if r.status_code == 200:
                events = r.json().get('events', [])
                # Ensure matchId is attached to every event
                for e in events:
                    e['matchId'] = mid
                all_events.extend(events)
            else:
                # 404 means no events found (sometimes happens for postponed/cancelled matches)
                pass
        except Exception as e:
            print(f"   ⚠️ Error fetching match {mid}: {e}")
            continue
    return all_events

def main():
    all_historical_matches = []
    all_historical_events = []

    print(f"🚀 Starting History Download for {len(TARGET_SEASON_IDS)} seasons...")
    print(f"   Seasons: {TARGET_SEASON_IDS}\n")
    
    for season_id in TARGET_SEASON_IDS:
        print(f"📅 Processing Season ID: {season_id}")
        
        # A. Fetch Matches
        matches = fetch_matches_for_season(season_id)
        count = len(matches)
        print(f"   Found {count} matches.")
        
        if count == 0:
            continue
            
        all_historical_matches.extend(matches)
        
        # B. Fetch Events
        match_ids = [m['matchId'] for m in matches]
        print(f"   ⏳ Fetching events for {count} matches (this will take time)...")
        
        events = fetch_events_batch(match_ids)
        all_historical_events.extend(events)
        print(f"   ✅ Finished Season {season_id}.\n")

    # 3. Save to Parquet
    print(f"💾 Saving Total: {len(all_historical_matches)} matches and {len(all_historical_events)} events...")
    
    # Save Matches
    if all_historical_matches:
        df_matches = pd.DataFrame(all_historical_matches)
        # Drop duplicates just in case
        df_matches = df_matches.drop_duplicates(subset=['matchId'])
        df_matches.to_parquet('historical_matches.parquet')
        print("   -> saved 'historical_matches.parquet'")
    
    # Save Events
    if all_historical_events:
        # Normalize JSON to flat table
        # We process in chunks if memory is an issue, but for <100k events standard df is fine
        print("   🔨 Flattening event data...")
        df_events = pd.json_normalize(all_historical_events)
        df_events.to_parquet('historical_events.parquet')
        print("   -> saved 'historical_events.parquet'")
    
    print("\n✅ Full 5-Year History Downloaded Successfully!")
    print("👉 Next Step: Run 'python process_history.py' to generate the minutes file.")

if __name__ == "__main__":
    main()