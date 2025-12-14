import pandas as pd
import json
from process_data import calculate_player_minutes_and_positions

def debug_main():
    print("🚀 Starting fast debug for minutes calculation...")

    # 1. Load Data
    print("Loading raw_events.parquet...")
    current_raw_events_df = pd.read_parquet('raw_events.parquet')

    print("Loading historical_matches.parquet (Checking for raw data)...")
    try:
        matches_df = pd.read_parquet('historical_matches.parquet')
    except Exception as e:
        print(f"❌ Could not load historical_matches.parquet: {e}")
        return

    # --- DIAGNOSTIC PRINT ---
    print(f"\n📊 Matches DataFrame Columns: {list(matches_df.columns)}")
    # ------------------------

    # 2. Smart Column Detection
    possible_id_cols = ['wyId', 'matchId', 'id', 'match_id', 'gameId']
    id_col = None
    for col in possible_id_cols:
        if col in matches_df.columns:
            id_col = col
            break
            
    if id_col is None:
        print(f"\n❌ CRITICAL ERROR: Could not find a match ID column.")
        return

    print(f"ℹ️  Using match ID column: '{id_col}'")

    # 3. Handle teamsData (Parquet often saves dicts as strings)
    if 'teamsData' in matches_df.columns:
        # Check if it needs parsing
        sample_val = matches_df.iloc[0]['teamsData']
        if isinstance(sample_val, str):
            print("⚠️ 'teamsData' is a string. Parsing JSON...")
            # We use a lambda to safely parse, handling potential None/NaN
            matches_df['teamsData'] = matches_df['teamsData'].apply(
                lambda x: json.loads(x.replace("'", '"')) if isinstance(x, str) else x
            )

    # 4. Filter
    current_match_ids = current_raw_events_df['matchId'].unique().tolist()
    current_matches_df = matches_df[matches_df[id_col].isin(current_match_ids)].copy()
    print(f"ℹ️  Found {len(current_matches_df)} matches for the current season.")

    # 5. Run Calculation
    print("Running calculate_player_minutes_and_positions...")
    player_minutes_df = calculate_player_minutes_and_positions(
        current_matches_df,
        current_raw_events_df
    )

    if not player_minutes_df.empty:
        print("\n✅ Success! Sample output:")
        print(player_minutes_df.head())
    else:
        print("\n❌ Failed: Resulting DataFrame is empty.")

if __name__ == "__main__":
    debug_main()