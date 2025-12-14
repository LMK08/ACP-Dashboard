import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
import os

# --- IMPORT CREDENTIALS ---
try:
    from process_data import WYSCOUT_USERNAME, WYSCOUT_PASSWORD
except ImportError:
    try:
        from process_data import wyscout_user as WYSCOUT_USERNAME, wyscout_pass as WYSCOUT_PASSWORD
    except ImportError:
        print("❌ Could not import credentials. Please hardcode them below.")
        WYSCOUT_USERNAME = "ggm0zzt-jidg1g5bv-ofdye2m-huk6ii8kkd"
        WYSCOUT_PASSWORD = ",Xzas52XAavPLHNK8sSJLJNhHEP!NY"

def test_event_name_mapping():
    # 1. LOAD EVENTS & BUILD MAP
    print("📂 Loading 'raw_events.parquet' to build Name Map...")
    if not os.path.exists('raw_events.parquet'):
        print("❌ 'raw_events.parquet' not found. Cannot run test.")
        return

    events_df = pd.read_parquet('raw_events.parquet')
    
    # Create dictionary: {12345: "Yan Said", ...}
    # We drop NaNs to ensure clean data
    name_map = events_df.dropna(subset=['player.id', 'player.name']) \
        .set_index('player.id')['player.name'].to_dict()
    
    # Convert keys to integers for safe lookup
    name_map = {int(k): v for k, v in name_map.items() if str(k).replace('.','').isdigit()}
    
    print(f"✅ Built Name Map with {len(name_map)} players.")

    # 2. FETCH A REAL MATCH FROM API
    match_id = 5740709 # Using the ID that failed earlier
    print(f"\n🚀 Fetching Match {match_id} from API...")
    
    url = f"https://apirest.wyscout.com/v3/matches/{match_id}"
    auth = HTTPBasicAuth(WYSCOUT_USERNAME, WYSCOUT_PASSWORD)
    r = requests.get(url, auth=auth)
    
    if r.status_code != 200:
        print("❌ API Request failed.")
        return

    match_data = r.json()
    teams_data = match_data.get('teamsData', {})
    
    # 3. TEST LOOKUP
    print("\n🕵️‍♂️ Testing Lookup on Starters...")
    
    for team_id, team_info in teams_data.items():
        team_name = team_info.get('name')
        lineup = team_info.get('formation', {}).get('lineup', [])
        
        print(f"\n--- Team: {team_name} ({len(lineup)} starters) ---")
        
        found = 0
        missing = 0
        
        for player in lineup:
            pid = player.get('playerId')
            
            # --- THE CRITICAL CHECK ---
            if pid in name_map:
                name = name_map[pid]
                found += 1
                # Print first 3 found to verify
                if found <= 3:
                    print(f"   ✅ ID {pid} -> Found: '{name}'")
            else:
                missing += 1
                if missing <= 1:
                    print(f"   ❌ ID {pid} -> Not in map")
        
        print(f"   👉 Result: {found} Resolved / {missing} Missing")

if __name__ == "__main__":
    test_event_name_mapping()