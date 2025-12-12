import pandas as pd
import requests
import os
from dotenv import load_dotenv # <--- Import this
from requests.auth import HTTPBasicAuth
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

# --- 1. LOAD CREDENTIALS FROM .ENV ---
load_dotenv() 

# FIX: Use the exact variable names from your .env file
WYSCOUT_USERNAME = os.getenv("WYSCOUT_USER") 
WYSCOUT_PASSWORD = os.getenv("WYSCOUT_PASS")

if not WYSCOUT_USERNAME or not WYSCOUT_PASSWORD:
    print("❌ Error: Credentials not found in .env file.")
    print("   Make sure WYSCOUT_USER and WYSCOUT_PASS are set.")
    exit() 
# -------------------------------------

def fetch_history_minutes(username, password, match_ids, raw_events_df):
    """
    Hybrid Method: Minutes from API + Identity from Events.
    Specific for Historical Data.
    """
    base_url_v3 = "https://apirest.wyscout.com/v3"
    auth = HTTPBasicAuth(username, password)
    
    # 1. Build Identity Map from Events
    print("Building Historical Identity Map...")
    valid_events = raw_events_df.dropna(subset=['player.id', 'team.name', 'player.position'])
    grouped = valid_events.groupby('player.id')
    
    # Get most common metadata per player
    names = grouped['player.name'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown")
    teams = grouped['team.name'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown")
    positions = grouped['player.position'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown")
    
    id_map = {}
    for pid in names.index:
        try:
            id_map[int(pid)] = {
                'name': str(names.loc[pid]),
                'team': str(teams.loc[pid]),
                'pos': str(positions.loc[pid])
            }
        except: continue

    # 2. Fetch Minutes from API
    player_minutes_list = []
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1)))
    
    print(f"🚀 Fetching history for {len(match_ids)} matches...")
    for match_id in tqdm(match_ids, desc="Processing History"):
        try:
            r = session.get(f"{base_url_v3}/matches/{match_id}", auth=auth, timeout=15)
            if r.status_code != 200: continue
            
            data = r.json()
            teams_data = data.get('teamsData', {})
            
            for t_id, t_info in teams_data.items():
                formation = t_info.get('formation', {})
                if not formation: continue
                
                # Subs logic
                sub_out = {}; sub_in = {}
                for sub in formation.get('substitutions', []):
                    m = sub.get('minute', 90)
                    if sub.get('playerOut'): sub_out[sub['playerOut']] = m
                    if sub.get('playerIn'): sub_in[sub['playerIn']] = m
                
                # Starters
                for p in formation.get('lineup', []):
                    pid = p.get('playerId')
                    mins = sub_out.get(pid, 96)
                    if mins > 0: player_minutes_list.append({'playerId': pid, 'minutes': mins})
                        
                # Bench
                for p in formation.get('bench', []):
                    pid = p.get('playerId')
                    if pid in sub_in:
                        t_in = sub_in[pid]
                        mins = sub_out.get(pid, 96) - t_in if pid in sub_out else 96 - t_in
                        if mins > 0: player_minutes_list.append({'playerId': pid, 'minutes': mins})
        except:
            continue

    # 3. Merge
    df = pd.DataFrame(player_minutes_list)
    
    def fill_identity(pid):
        data = id_map.get(pid, {'name': 'Unknown', 'team': 'Unknown', 'pos': 'Unknown'})
        return pd.Series([data['name'], data['team'], data['pos']])

    print("Merging historical identities...")
    df[['playerName', 'teamName', 'primaryPosition']] = df['playerId'].apply(fill_identity)
    
    # Aggregation
    total_minutes = df.groupby(['playerId', 'playerName', 'teamName', 'primaryPosition'])['minutes'].sum().reset_index()
    total_minutes.rename(columns={'minutes': 'totalMinutes'}, inplace=True)
    total_minutes['playerId'] = total_minutes['playerId'].astype(int)
    
    return total_minutes

def main():
    if WYSCOUT_USERNAME == "YOUR_USERNAME_HERE":
        print("❌ Update credentials in process_history.py")
        return

    # Load Historical Data
    if not os.path.exists('historical_matches.parquet') or not os.path.exists('historical_events.parquet'):
        print("❌ Historical files not found.")
        return

    hist_matches = pd.read_parquet('historical_matches.parquet')
    hist_events = pd.read_parquet('historical_events.parquet')
    
    # Run Calculation
    history_minutes_df = fetch_history_minutes(
        WYSCOUT_USERNAME, WYSCOUT_PASSWORD,
        hist_matches['matchId'].unique().tolist(),
        hist_events
    )
    
    # Save
    if not history_minutes_df.empty:
        history_minutes_df.to_pickle('history_player_minutes.pkl')
        print(f"✅ SUCCESS: Saved {len(history_minutes_df)} historical players to 'history_player_minutes.pkl'")
    else:
        print("❌ No data generated.")

if __name__ == "__main__":
    main()