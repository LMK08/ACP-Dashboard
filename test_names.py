import requests
import pandas as pd
from requests.auth import HTTPBasicAuth
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm
import os

def test_name_mapping(match_ids):
    # --- 1. ENTER YOUR CREDENTIALS HERE ---
    WYSCOUT_USERNAME = "ggm0zzt-jidg1g5bv-ofdye2m-huk6ii8kkd"
    WYSCOUT_PASSWORD = ",Xzas52XAavPLHNK8sSJLJNhHEP!NY"  
    # -------------------------------------

    if WYSCOUT_USERNAME == "YOUR_USERNAME_HERE":
        print("❌ Error: Please update the script with your actual username/password.")
        return

    base_url_v3 = "https://apirest.wyscout.com/v3"
    auth = HTTPBasicAuth(WYSCOUT_USERNAME, WYSCOUT_PASSWORD)
    
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1)))
    
    print(f"\n🚀 Testing Name Mapping on {len(match_ids)} matches...")
    
    for match_id in tqdm(match_ids, desc="Checking Names"):
        url = f"{base_url_v3}/matches/{match_id}"
        r = session.get(url, auth=auth)
        
        if r.status_code != 200:
            print(f"⚠️ Failed to fetch match {match_id}")
            continue
            
        match_data = r.json()
        
        # --- THE FIX BEING TESTED ---
        player_map = {}
        raw_players = match_data.get('players', [])
        
        print(f"\nMatch {match_id}: Found {len(raw_players)} players in lookup list.")
        
        # Check keys for the first player to be sure
        if raw_players:
            print(f"🔍 Keys available in player object: {list(raw_players[0].keys())}")
        
        for p in raw_players:
            # TRY BOTH KEYS
            p_id = p.get('playerId') or p.get('wyId')
            if p_id:
                name = p.get('shortName') or p.get('lastName') or "Unknown"
                player_map[p_id] = name
                
        # Now check a lineup to see if we can resolve the names
        teams_data = match_data.get('teamsData', {})
        for team_id, team_info in teams_data.items():
            lineup = team_info.get('formation', {}).get('lineup', [])
            if not lineup: continue
            
            print(f"   Checking {len(lineup)} starters for {team_info.get('name')}...")
            
            resolved_count = 0
            unknown_count = 0
            
            for player in lineup:
                pid = player.get('playerId')
                mapped_name = player_map.get(pid, "Unknown")
                
                if mapped_name == "Unknown":
                    unknown_count += 1
                else:
                    resolved_count += 1
            
            print(f"   ✅ Resolved: {resolved_count} | ❌ Unknown: {unknown_count}")
            
            # Print sample names
            sample_names = [player_map.get(p.get('playerId')) for p in lineup[:3]]
            print(f"   Examples: {sample_names}")
            break # Just check one team

if __name__ == "__main__":
    # Get 3 sample IDs locally
    if os.path.exists('matches_summary.parquet'):
        ids = pd.read_parquet('matches_summary.parquet')['matchId'].head(3).tolist()
    elif os.path.exists('raw_events.parquet'):
        ids = pd.read_parquet('raw_events.parquet')['matchId'].unique()[:3].tolist()
    else:
        # Fallback IDs if local data missing
        ids = [5740709, 5740714] 
        
    test_name_mapping(ids)