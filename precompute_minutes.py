"""
Pre-compute complete player minutes for all seasons using three sources:
1. API-based minutes (from player_minutes_and_positions.pkl)
2. Lineup data (from all_match_data.pkl)
3. Event-based estimates (from raw_events.parquet)

Saves result to complete_player_minutes.pkl: {season_id: DataFrame}
"""
import pandas as pd
import pickle

ALL_SEASON_IDS = [191782, 190090, 189147, 188222, 188221]

def get_minutes_from_match_data(all_match_data, matches_summary_df, season_id):
    """Extract actual player minutes from match lineup data."""
    season_match_ids = set(matches_summary_df[matches_summary_df['seasonId'] == season_id]['matchId'].unique())
    match_teams = matches_summary_df.set_index('matchId')[['homeTeamName', 'awayTeamName']].to_dict('index')

    records = []
    for mid, data in all_match_data.items():
        if mid not in season_match_ids:
            continue
        ps = data.get('player_stats', {})
        teams_info = match_teams.get(mid, {})
        side_team_map = {
            'home': teams_info.get('homeTeamName', 'Unknown'),
            'away': teams_info.get('awayTeamName', 'Unknown'),
        }
        for side in ['home', 'away']:
            df = ps.get(side)
            if df is None or not isinstance(df, pd.DataFrame) or 'Minutes' not in df.columns:
                continue
            team_name = side_team_map[side]
            for player_name, row in df.iterrows():
                try:
                    mins = int(str(row['Minutes']).replace("'", '').strip())
                except (ValueError, TypeError):
                    mins = 0
                if mins > 0:
                    records.append({'playerName': player_name, 'teamName': team_name, 'totalMinutes': mins})

    if not records:
        return pd.DataFrame(columns=['playerName', 'teamName', 'totalMinutes'])

    result = pd.DataFrame(records)
    return result.groupby('playerName').agg(
        teamName=('teamName', lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown'),
        totalMinutes=('totalMinutes', 'sum'),
    ).reset_index()


def estimate_minutes_from_events(events_df):
    """Estimate player minutes from event timestamps as a last-resort fallback."""
    ev = events_df.dropna(subset=['player.id']).copy()
    ev['player.id'] = pd.to_numeric(ev['player.id'], errors='coerce').dropna().astype(int)
    ev['minute'] = pd.to_numeric(ev['minute'], errors='coerce').fillna(0)

    match_max = ev.groupby('matchId')['minute'].max().rename('match_max_minute')
    player_match = ev.groupby(['matchId', 'player.id']).agg(
        first_min=('minute', 'min'),
        last_min=('minute', 'max'),
        player_name=('player.name', 'first'),
        team_name=('team.name', 'first'),
        position=('player.position', 'first'),
    ).reset_index()
    player_match = player_match.merge(match_max, on='matchId', how='left')

    # Vectorized estimation
    player_match['est_start'] = player_match['first_min'].where(player_match['first_min'] > 5, 0)
    player_match['est_end'] = player_match['last_min'].where(
        player_match['last_min'] < player_match['match_max_minute'] - 5,
        player_match['match_max_minute']
    )
    player_match['est_minutes'] = (player_match['est_end'] - player_match['est_start']).clip(lower=1)

    return player_match.groupby('player.id').agg(
        playerName=('player_name', 'first'),
        teamName=('team_name', 'first'),
        primaryPosition=('position', 'first'),
        totalMinutes=('est_minutes', 'sum'),
    ).reset_index().rename(columns={'player.id': 'playerId'}).astype({'playerId': int})


def main():
    print("Loading data files...")
    events = pd.read_parquet('raw_events.parquet')
    matches = pd.read_parquet('matches_summary.parquet')
    with open('all_match_data.pkl', 'rb') as f:
        all_match_data = pickle.load(f)
    with open('player_minutes_and_positions.pkl', 'rb') as f:
        api_minutes = pickle.load(f)

    complete_minutes = {}

    for sid in ALL_SEASON_IDS:
        print(f"\n--- Season {sid} ---")
        season_events = events[events['seasonId'] == sid]

        # Source 1: API minutes
        minutes_df = api_minutes.get(sid, pd.DataFrame()).copy()
        api_count = len(minutes_df)
        existing_names = set(minutes_df['playerName'].unique()) if not minutes_df.empty else set()
        print(f"  API minutes: {api_count} players")

        # Build name -> (id, position) lookup from events
        ev_lookup = season_events.dropna(subset=['player.id', 'player.name']).copy()
        ev_lookup['player.id'] = pd.to_numeric(ev_lookup['player.id'], errors='coerce').dropna().astype(int)
        name_to_info = ev_lookup.groupby('player.name').agg(
            playerId=('player.id', 'first'),
            primaryPosition=('player.position', 'first'),
        ).to_dict('index')

        # Source 2: Lineup data
        lineup = get_minutes_from_match_data(all_match_data, matches, sid)
        if not lineup.empty:
            missing_lineup = lineup[~lineup['playerName'].isin(existing_names)].copy()
            missing_lineup['playerId'] = missing_lineup['playerName'].map(
                lambda n: name_to_info.get(n, {}).get('playerId', 0)
            )
            missing_lineup['primaryPosition'] = missing_lineup['playerName'].map(
                lambda n: name_to_info.get(n, {}).get('primaryPosition', 'Unknown')
            )
            missing_lineup = missing_lineup[missing_lineup['playerId'] > 0]
            if not missing_lineup.empty:
                print(f"  Lineup data: +{len(missing_lineup)} players")
                minutes_df = pd.concat([
                    minutes_df,
                    missing_lineup[['playerId', 'playerName', 'teamName', 'primaryPosition', 'totalMinutes']]
                ], ignore_index=True)

        # Source 3: Event estimates
        existing_ids = set(minutes_df['playerId'].unique()) if not minutes_df.empty else set()
        estimated = estimate_minutes_from_events(season_events)
        if not estimated.empty:
            still_missing = estimated[~estimated['playerId'].isin(existing_ids)]
            if not still_missing.empty:
                print(f"  Event estimates: +{len(still_missing)} players")
                minutes_df = pd.concat([
                    minutes_df,
                    still_missing[['playerId', 'playerName', 'teamName', 'primaryPosition', 'totalMinutes']]
                ], ignore_index=True)

        # Remove invalid entries (playerId=0, playerName='0')
        if not minutes_df.empty:
            minutes_df = minutes_df[(minutes_df['playerId'] != 0) & (minutes_df['playerName'].astype(str) != '0')].copy()

        print(f"  Total: {len(minutes_df)} players")
        complete_minutes[sid] = minutes_df

    # Save
    with open('complete_player_minutes.pkl', 'wb') as f:
        pickle.dump(complete_minutes, f)
    print(f"\nSaved complete_player_minutes.pkl with {len(complete_minutes)} seasons")

    # Also build all-time aggregation
    all_dfs = [df for df in complete_minutes.values() if not df.empty]
    if all_dfs:
        combined = pd.concat(all_dfs)
        all_time = combined.groupby('playerId').agg({
            'playerName': 'first',
            'teamName': 'first',
            'primaryPosition': 'first',
            'totalMinutes': 'sum',
        }).reset_index()
        all_time.to_pickle('all_time_player_minutes.pkl')
        print(f"Saved all_time_player_minutes.pkl with {len(all_time)} players")


if __name__ == '__main__':
    main()
