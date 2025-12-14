import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
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

def test_details_mapping():
    # 1. LOAD EVENTS & BUILD FULL MAP
    print("📂 Loading 'raw_events.parquet' to build Details Map...")
    if not os.path.exists('raw_events.parquet'):
        print("❌ 'raw_events.parquet' not found. Cannot run test.")
        return

    events_df = pd.read_parquet('raw_events.parquet')
    
    # Create dictionary: {12345: {'name': 'Yan Said', 'team': 'Braga II', 'pos': 'FW'}, ...}
    unique_players = events_df[['player.id', 'player.name', 'team.name', 'player.position']].dropna(subset=['player.id'])
    
    details_map = {}
    for _, row in unique_players.iterrows():
        try:
            pid = int(row['player.id'])
            details_map[pid] = {
                'name': row['player.name'],
                'team': row['team.name'],
                'position': row['player.position']
            }
        except:
            continue
            
    print(f"✅ Built Details Map for {len(details_map)} players.")

    # 2. FETCH A PROBLEM MATCH FROM API
    match_id = 5740709 # The match where teams/positions were missing
    print(f"\n🚀 Fetching Match {match_id} from API...")
    
    url = f"https://apirest.wyscout.com/v3/matches/{match_id}"
    auth = HTTPBasicAuth(WYSCOUT_USERNAME, WYSCOUT_PASSWORD)
    
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1)))
    r = session.get(url, auth=auth)
    
    if r.status_code != 200:
        print("❌ API Request failed.")
        return

    match_data = r.json()
    teams_data = match_data.get('teamsData', {})
    
    # 3. TEST THE FIX
    print("\n🕵️‍♂️ comparing API Data vs. Local Map...")
    
    for team_id, team_info in teams_data.items():
        # Check Team Name
        api_team_name = team_info.get('name')
        
        # Check Players
        lineup = team_info.get('formation', {}).get('lineup', [])
        if not lineup: continue
        
        print(f"\n--- Checking Team ID: {team_id} ---")
        print(f"   API Team Name: {api_team_name} (Likely None)")
        
        # Check first 3 players
        for player in lineup[:3]:
            p_id = player.get('playerId')
            
            # API Data
            api_pos = player.get('role', {}).get('code2')
            
            # Local Map Data
            local_data = details_map.get(p_id, {})
            local_name = local_data.get('name', 'Unknown')
            local_team = local_data.get('team', 'Unknown')
            local_pos = local_data.get('position', 'Unknown')
            
            print(f"   Player ID {p_id}:")
            print(f"      API Position: {api_pos}")
            print(f"      👉 FIX FOUND: Name='{local_name}' | Team='{local_team}' | Pos='{local_pos}'")

if __name__ == "__main__":
    test_details_mapping()