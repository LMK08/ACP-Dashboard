import os
import requests
import pandas as pd
from requests.auth import HTTPBasicAuth
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

# --- 1. CREDENTIALS (UPDATE THESE) ---
WYSCOUT_USERNAME = "ggm0zzt-jidg1g5bv-ofdye2m-huk6ii8kkd"
WYSCOUT_PASSWORD = ",Xzas52XAavPLHNK8sSJLJNhHEP!NY"
# -------------------------------------

def fetch_official_player_minutes_test(username, password, match_ids):
    base_url_v3 = "https://apirest.wyscout.com/v3"
    auth = HTTPBasicAuth(username, password)
    player_minutes_list = []
    
    session = requests.Session()
    retry_strategy = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    
    print(f"\n🚀 Stress testing {len(match_ids)} matches...")
    
    for match_id in tqdm(match_ids, desc="Fetching Match Info"):
        url = f"{base_url_v3}/matches/{match_id}"
        
        try:
            r = session.get(url, auth=auth, timeout=20)
            r.raise_for_status()
            match_data = r.json()
            
            # 1. BUILD PLAYER LOOKUP MAP
            player_map = {}
            for p in match_data.get('players', []):
                p_id = p.get('wyId')
                if p_id:
                    player_map[p_id] = p.get('shortName') or p.get('lastName') or "Unknown"

            match_duration = 96 
            teams_data = match_data.get('teamsData', {})
            
            for team_id, team_info in teams_data.items():
                team_name = team_info.get('name', 'Unknown')
                
                # --- SAFETY FIX ---
                formation = team_info.get('formation')
                if not formation:
                    # If we hit this, print a warning so we know the fix worked
                    # print(f"⚠️ Warning: Missing formation for team {team_name} in match {match_id}")
                    continue
                # ------------------
                
                lineup = formation.get('lineup', [])
                bench = formation.get('bench', [])
                substitutions = formation.get('substitutions', [])
                
                # 2. MAP SUBSTITUTIONS
                sub_out_map = {}
                sub_in_map = {}
                for sub in substitutions:
                    minute = sub.get('minute', 90)
                    player_out = sub.get('playerOut')
                    player_in = sub.get('playerIn')
                    if player_out: sub_out_map[player_out] = minute
                    if player_in: sub_in_map[player_in] = minute

                # 3. PROCESS STARTERS
                for player in lineup:
                    p_id = player.get('playerId')
                    p_name = player_map.get(p_id, "Unknown Player")
                    
                    if p_id in sub_out_map:
                        mins_played = sub_out_map[p_id]
                    else:
                        mins_played = match_duration
                    
                    if mins_played > 0:
                        player_minutes_list.append({'matchId': match_id, 'minutes': mins_played})

                # 4. PROCESS BENCH
                for player in bench:
                    p_id = player.get('playerId')
                    if p_id in sub_in_map:
                        time_in = sub_in_map[p_id]
                        if p_id in sub_out_map:
                            mins_played = sub_out_map[p_id] - time_in
                        else:
                            mins_played = match_duration - time_in
                        
                        if mins_played > 0:
                            player_minutes_list.append({'matchId': match_id, 'minutes': mins_played})
                    
        except requests.exceptions.RequestException as e:
            print(f"  -> ⚠️ Failed match {match_id}: {e}")
        except Exception as e:
            print(f"  -> ❌ CRITICAL ERROR on match {match_id}: {e}")
            
    return pd.DataFrame(player_minutes_list)

def main():
    if WYSCOUT_USERNAME == "YOUR_USERNAME_HERE":
        print("❌ Please update credentials in the script.")
        return

    # Load ALL match IDs from your summary or events file
    print("Loading all match IDs...")
    try:
        if os.path.exists('matches_summary.parquet'):
            matches_df = pd.read_parquet('matches_summary.parquet')
            all_ids = matches_df['matchId'].unique().tolist()
        elif os.path.exists('raw_events.parquet'):
            events_df = pd.read_parquet('raw_events.parquet')
            all_ids = events_df['matchId'].unique().tolist()
        else:
            print("❌ No local data found.")
            return
            
        print(f"ℹ️  Found {len(all_ids)} matches. Starting test run...")
        
        # RUN ON ALL MATCHES
        df = fetch_official_player_minutes_test(WYSCOUT_USERNAME, WYSCOUT_PASSWORD, all_ids)
        
        if not df.empty:
            print("\n✅ Success! Processed all matches without crashing.")
            print(f"Generated {len(df)} player-minute records.")
        else:
            print("\n❌ Failed: No data returned.")

    except Exception as e:
        print(f"❌ Script Error: {e}")

if __name__ == "__main__":
    main()