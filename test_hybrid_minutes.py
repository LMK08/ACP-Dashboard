import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import os
from tqdm import tqdm

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

def test_hybrid_logic():
    print("🧪 STARTING TEST: Hybrid Minutes Calculation (API Time + Event Metadata)\n")

    # --- STEP 1: Build Identity Map from Events ---
    print("1. 📂 Loading 'raw_events.parquet' to build Identity Map...")
    if not os.path.exists('raw_events.parquet'):
        print("❌ 'raw_events.parquet' not found. Cannot run test.")
        return

    events_df = pd.read_parquet('raw_events.parquet')
    
    # Filter for rows with valid identity info
    valid_events = events_df.dropna(subset=['player.id', 'team.name', 'player.position'])
    
    # Group by ID and get the most common Name, Team, and Position
    # (This ensures we get their MAIN team, avoiding one-off errors)
    print("   🔨 Calculating most frequent metadata for each player...")
    grouped = valid_events.groupby('player.id')
    
    names = grouped['player.name'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown")
    teams = grouped['team.name'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown")
    positions = grouped['player.position'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown")
    
    # Build simple lookup dict
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
            
    print(f"✅ Identity Map ready for {len(id_map)} players.\n")

    # --- STEP 2: Fetch API Minutes for Sample Matches ---
    # Use the specific matches that were giving us trouble
    sample_match_ids = [5740709, 5740714] 
    print(f"2. 🚀 Fetching minutes for matches: {sample_match_ids}...")
    
    auth = HTTPBasicAuth(WYSCOUT_USERNAME, WYSCOUT_PASSWORD)
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1)))
    
    player_minutes_list = []
    
    for match_id in tqdm(sample_match_ids, desc="Fetching Lineups"):
        url = f"https://apirest.wyscout.com/v3/matches/{match_id}"
        r = session.get(url, auth=auth)
        if r.status_code != 200:
            print(f"⚠️ Failed to fetch match {match_id}")
            continue
            
        match_data = r.json()
        match_duration = 96
        
        for team_id, team_info in match_data.get('teamsData', {}).items():
            formation = team_info.get('formation', {})
            
            # Sub logic
            sub_out = {}; sub_in = {}
            for sub in formation.get('substitutions', []):
                minute = sub.get('minute', 90)
                if sub.get('playerOut'): sub_out[sub['playerOut']] = minute
                if sub.get('playerIn'): sub_in[sub['playerIn']] = minute
            
            # Starters
            for p in formation.get('lineup', []):
                pid = p.get('playerId')
                mins = sub_out.get(pid, match_duration)
                if mins > 0:
                    player_minutes_list.append({'playerId': pid, 'minutes': mins})
            
            # Bench
            for p in formation.get('bench', []):
                pid = p.get('playerId')
                if pid in sub_in:
                    t_in = sub_in[pid]
                    mins = sub_out.get(pid, match_duration) - t_in if pid in sub_out else match_duration - t_in
                    if mins > 0:
                        player_minutes_list.append({'playerId': pid, 'minutes': mins})

    # --- STEP 3: Merge and Verify ---
    print("\n3. 🔗 Merging API Minutes with Event Metadata...")
    minutes_df = pd.DataFrame(player_minutes_list)
    
    # Apply the map
    def fill_identity(pid):
        data = id_map.get(pid, {'name': 'Unknown', 'team': 'Unknown', 'pos': 'Unknown'})
        return pd.Series([data['name'], data['team'], data['pos']])

    minutes_df[['playerName', 'teamName', 'primaryPosition']] = minutes_df['playerId'].apply(fill_identity)
    
    # Show Results
    print("\n✅ FINAL RESULT PREVIEW (Top 10):")
    # Sort by minutes to see key players
    result = minutes_df.sort_values('minutes', ascending=False).head(10)
    print(result[['playerId', 'playerName', 'teamName', 'primaryPosition', 'minutes']])
    
    # Verify no unknowns in top results
    unknowns = result[result['teamName'] == 'Unknown']
    if unknowns.empty:
        print("\n🎉 SUCCESS! No 'Unknown' teams in the top sample.")
    else:
        print(f"\n⚠️ WARNING: {len(unknowns)} players still have Unknown teams. Check map coverage.")

if __name__ == "__main__":
    test_hybrid_logic()