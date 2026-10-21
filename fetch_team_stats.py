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

from league_config import COMPETITIONS, get_credentials, competition_for_season

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

    print("Loading team IDs from events data...")
    events_cols = ['team.id', 'team.name', 'seasonId']
    # Load competitionId if available for per-competition credential lookup
    try:
        events = pd.read_parquet(EVENTS_FILE, columns=events_cols + ['competitionId'])
    except Exception:
        events = pd.read_parquet(EVENTS_FILE, columns=events_cols)
        # Backfill competitionId from season
        events['competitionId'] = events['seasonId'].map(competition_for_season)

    session = setup_session()
    rows = []
    _stored_seasons = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            _stored_seasons = set(pd.read_parquet(OUTPUT_FILE, columns=['seasonId'])['seasonId'].astype(int))
        except Exception:
            _stored_seasons = set()

    for comp_id, comp_config in COMPETITIONS.items():
        user, password = get_credentials(comp_id)
        if not user or not password:
            print(f"⚠️ No credentials for {comp_config['name']}, skipping.")
            continue
        if not comp_config.get("active", True):
            print(f"{comp_config['name']}: inactive (no Wyscout access), skipping.")
            continue
        user = user.strip().strip('"').strip("'")
        password = password.strip().strip('"').strip("'")
        auth = HTTPBasicAuth(user, password)

        for season_id in comp_config["seasons"]:
            # A finished season's team stats never change: re-fetch only the
            # current season (and any season not stored yet). All 8 seasons
            # used to be re-requested every run (~3.5 min).
            if (season_id != comp_config["current_season"] and season_id in _stored_seasons):
                continue
            season_events = events[events['seasonId'] == season_id]
            teams = season_events.drop_duplicates(subset='team.id')[['team.id', 'team.name']].dropna()
            if teams.empty:
                continue
            print(f"\n{comp_config['name']} Season {season_id}: Found {len(teams)} teams")

            for _, row in teams.iterrows():
                team_id = int(row['team.id'])
                team_name = row['team.name']
                url = f"https://apirest.wyscout.com/v3/teams/{team_id}/advancedstats"
                params = {'compId': comp_id, 'seasonId': season_id}

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
    # Keep previously fetched rows that this run did not refresh (a failed
    # call used to shrink the file, e.g. 128 -> 122 rows on 2026-10-14).
    if os.path.exists(OUTPUT_FILE):
        old_df = pd.read_parquet(OUTPUT_FILE)
        key = ['team_id', 'seasonId']
        if df.empty:
            # every call failed (API down / access lost): keep the file as is
            print("⚠️ No team stats fetched this run — keeping the previous file.")
            df = old_df
        elif not old_df.empty and all(c in old_df.columns for c in key):
            fresh = set(map(tuple, df[key].values.tolist()))
            keep = old_df[[tuple(r) not in fresh for r in old_df[key].values.tolist()]]
            df = pd.concat([df, keep], ignore_index=True)
    df.to_parquet(OUTPUT_FILE, index=False)
    print(f"\nSaved {len(df)} team-season records to {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
