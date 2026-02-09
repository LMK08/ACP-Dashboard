#!/usr/bin/env python3
"""
Fetch team advanced stats from Wyscout API for radar charts.
Reads team IDs from raw_events.parquet, fetches /teams/{id}/advancedstats,
and saves results to team_advanced_stats.parquet.
"""

import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import os
from dotenv import load_dotenv
import warnings
import time

warnings.filterwarnings('ignore')

COMPETITION_ID = 43324
ALL_SEASON_IDS = [191782, 190090, 189147, 188222, 188221]
EVENTS_FILE = 'raw_events.parquet'
OUTPUT_FILE = 'team_advanced_stats.parquet'


def setup_session():
    session = requests.Session()
    retry_strategy = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def main():
    load_dotenv()
    wyscout_user = os.getenv("WYSCOUT_USER")
    wyscout_pass = os.getenv("WYSCOUT_PASS")

    if not wyscout_user or not wyscout_pass:
        print("Error: WYSCOUT_USER and WYSCOUT_PASS must be set.")
        exit(1)

    # Strip quotes that may come from .env
    wyscout_user = wyscout_user.strip().strip('"').strip("'")
    wyscout_pass = wyscout_pass.strip().strip('"').strip("'")

    print("Loading team IDs from events data...")
    events = pd.read_parquet(EVENTS_FILE, columns=['team.id', 'team.name', 'seasonId'])

    session = setup_session()
    auth = HTTPBasicAuth(wyscout_user, wyscout_pass)
    rows = []

    for season_id in ALL_SEASON_IDS:
        season_events = events[events['seasonId'] == season_id]
        teams = season_events.drop_duplicates(subset='team.id')[['team.id', 'team.name']].dropna()
        print(f"\nSeason {season_id}: Found {len(teams)} teams")

        for _, row in teams.iterrows():
            team_id = int(row['team.id'])
            team_name = row['team.name']
            url = f"https://apirest.wyscout.com/v3/teams/{team_id}/advancedstats"
            params = {'compId': COMPETITION_ID, 'seasonId': season_id}

            try:
                r = session.get(url, auth=auth, params=params, timeout=15)
                if r.status_code != 200:
                    print(f"  {team_name}: HTTP {r.status_code}")
                    continue

                data = r.json()
                avg = data.get('average', {})
                pct = data.get('percent', {})
                total = data.get('total', {})

                rows.append({
                    'team_id': team_id,
                    'team_name': team_name,
                    'seasonId': season_id,
                    # Offensive
                    'goals': avg.get('goals', 0),
                    'xg': avg.get('xgShot', 0),
                    'shots': avg.get('shots', 0),
                    'touch_in_box': avg.get('touchInBox', 0),
                    'passes_to_final_third': avg.get('passesToFinalThird', 0),
                    'crosses': avg.get('crosses', 0),
                    'successful_dribbles': avg.get('successfulDribbles', 0),
                    # Distribution
                    'passes': avg.get('passes', 0),
                    'progressive_run': avg.get('progressiveRun', 0),
                    'forward_passes': avg.get('forwardPasses', 0),
                    'possession_percent': avg.get('possessionPercent', 0),
                    'ball_losses': avg.get('ballLosses', 0),
                    # Defensive
                    'conceded_goals': avg.get('concededGoals', 0),
                    'xg_shot_against': avg.get('xgShotAgainst', 0),
                    'shots_against': avg.get('shotsAgainst', 0),
                    'aerial_duels_won_pct': pct.get('aerialDuelsWon', 0),
                    'defensive_duels_won_pct': pct.get('newDefensiveDuelsWon', 0),
                    'interceptions': avg.get('interceptions', 0),
                    'fouls': avg.get('fouls', 0),
                    'ppda': total.get('ppda', 0),
                })
                print(f"  {team_name}: OK")
            except Exception as e:
                print(f"  {team_name}: Error - {e}")

            time.sleep(0.3)

    df = pd.DataFrame(rows)
    df.to_parquet(OUTPUT_FILE, index=False)
    print(f"\nSaved {len(df)} team-season records to {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
