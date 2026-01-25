import pandas as pd
import requests
import os
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

# --- 1. LOAD CREDENTIALS FROM .ENV ---
load_dotenv()

WYSCOUT_USERNAME = os.getenv("WYSCOUT_USER")
WYSCOUT_PASSWORD = os.getenv("WYSCOUT_PASS")

if not WYSCOUT_USERNAME or not WYSCOUT_PASSWORD:
    print("Error: Credentials not found in .env file.")
    print("   Make sure WYSCOUT_USER and WYSCOUT_PASS are set.")
    exit()

# Competition and Season IDs
COMPETITION_ID = 43324  # Liga 3 Portugal

# Historical seasons (excluding current)
HISTORICAL_SEASON_IDS = [
    190090,  # 2024/25 Season
    189147,  # 2023/24 Season
    188222,  # 2022/23 Season
    188221,  # 2021/22 Season
]


def fetch_history_minutes(username, password, match_ids, raw_events_df, competition_id, season_ids):
    """
    Fetches player minutes directly from Wyscout Player Advanced Stats endpoint.
    Handles multiple historical seasons.

    1. Collects unique player IDs from match lineups
    2. For each season, fetches player's advanced stats (includes direct minutesOnField)
    3. Uses event data for player name/team identity
    """
    base_url_v3 = "https://apirest.wyscout.com/v3"
    auth = HTTPBasicAuth(username, password)

    # --- STEP 1: Build Identity Map from Events ---
    print("Building Historical Identity Map from Events...")

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

    print(f"Identity Map ready for {len(id_map)} players.")

    # --- STEP 2: Collect unique player IDs from match lineups ---
    session = requests.Session()
    retry_strategy = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)

    unique_player_ids = set()

    print(f"\nCollecting player IDs from {len(match_ids)} historical matches...")

    for match_id in tqdm(match_ids, desc="Scanning Lineups"):
        url = f"{base_url_v3}/matches/{match_id}"
        try:
            r = session.get(url, auth=auth, timeout=20)
            if r.status_code != 200:
                continue

            match_data = r.json()
            teams_data = match_data.get('teamsData', {})

            for team_id, team_info in teams_data.items():
                formation = team_info.get('formation')
                if not formation:
                    continue

                for player in formation.get('lineup', []):
                    pid = player.get('playerId')
                    if pid:
                        unique_player_ids.add(pid)

                for player in formation.get('bench', []):
                    pid = player.get('playerId')
                    if pid:
                        unique_player_ids.add(pid)

        except Exception as e:
            continue

    print(f"Found {len(unique_player_ids)} unique players in historical data.")

    # --- STEP 3: Fetch minutes from Player Advanced Stats for each season ---
    player_minutes_dict = {}  # {player_id: total_minutes}

    for season_id in season_ids:
        print(f"\nFetching player stats for Season {season_id}...")

        for pid in tqdm(unique_player_ids, desc=f"Season {season_id}"):
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
                    # Accumulate minutes across seasons
                    if pid not in player_minutes_dict:
                        player_minutes_dict[pid] = {
                            'totalMinutes': 0,
                            'position': None
                        }

                    player_minutes_dict[pid]['totalMinutes'] += minutes

                    # Get position from API (use most recent if available)
                    positions_data = data.get('positions', [])
                    if positions_data and not player_minutes_dict[pid]['position']:
                        player_minutes_dict[pid]['position'] = positions_data[0].get('position', {}).get('code', '').upper()

            except Exception as e:
                continue

    # --- STEP 4: Create DataFrame ---
    player_stats_list = []

    for pid, stats in player_minutes_dict.items():
        if stats['totalMinutes'] > 0:
            identity = id_map.get(pid, {'name': 'Unknown', 'team': 'Unknown', 'pos': 'Unknown'})

            player_stats_list.append({
                'playerId': pid,
                'playerName': identity['name'],
                'teamName': identity['team'],
                'primaryPosition': stats['position'] if stats['position'] else identity['pos'],
                'totalMinutes': stats['totalMinutes']
            })

    if not player_stats_list:
        print("No historical player minutes data retrieved.")
        return pd.DataFrame(columns=['playerId', 'playerName', 'teamName', 'primaryPosition', 'totalMinutes'])

    total_minutes_df = pd.DataFrame(player_stats_list)
    total_minutes_df['playerId'] = total_minutes_df['playerId'].astype(int)

    print(f"Retrieved direct minutes for {len(total_minutes_df)} historical players.")

    return total_minutes_df


def main():
    # Load Historical Data
    if not os.path.exists('historical_matches.parquet') or not os.path.exists('historical_events.parquet'):
        print("Historical files not found. Run fetch_full_history.py first.")
        return

    hist_matches = pd.read_parquet('historical_matches.parquet')
    hist_events = pd.read_parquet('historical_events.parquet')

    print(f"Loaded {len(hist_matches)} historical matches and {len(hist_events)} events.")

    # Run Calculation
    history_minutes_df = fetch_history_minutes(
        WYSCOUT_USERNAME,
        WYSCOUT_PASSWORD,
        hist_matches['matchId'].unique().tolist(),
        hist_events,
        COMPETITION_ID,
        HISTORICAL_SEASON_IDS
    )

    # Save
    if not history_minutes_df.empty:
        history_minutes_df.to_pickle('history_player_minutes.pkl')
        print(f"SUCCESS: Saved {len(history_minutes_df)} historical players to 'history_player_minutes.pkl'")
    else:
        print("No data generated.")


if __name__ == "__main__":
    main()
