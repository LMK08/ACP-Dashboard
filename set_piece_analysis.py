import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm # Import tqdm for a progress bar

# --- 1. Load All Data ---

print("Loading data...")
# Load the new player details
try:
    with open('player_details.pkl', 'rb') as f:
        player_details_list = pickle.load(f)
    players_df = pd.DataFrame(player_details_list)
    players_df = players_df.dropna(subset=['playerId'])
    players_df['playerId'] = players_df['playerId'].astype(int)
    players_df = players_df.set_index('playerId')
    print(f"✅ Successfully loaded {len(players_df)} player detail records.")
except FileNotFoundError:
    print("❌ Error: player_details.pkl not found. Run get_player_details.py first.")
    exit()

# Load historical event data
try:
    # --- UPDATED: Load columns needed for this analysis ---
    event_cols = [
        'matchId', 'team.name', 'player.id', 'type.primary', 
        'location.x', 'location.y', 'matchTimestamp',
        'shot.xg', 'shot.isGoal'
    ]
    events_df = pd.read_parquet('raw_events.parquet', columns=event_cols)
    events_df['player.id'] = pd.to_numeric(events_df['player.id'], errors='coerce').fillna(0).astype(int)
    events_df['matchTimestamp'] = pd.to_datetime(events_df['matchTimestamp'], errors='coerce')
    events_df = events_df.dropna(subset=['matchTimestamp']) # Critical for time-based analysis
    print(f"✅ Successfully loaded {len(events_df)} events.")
except FileNotFoundError:
    print("❌ Error: raw_events.parquet not found. Run process_data.py first.")
    exit()

# --- 2. Merge Player Footedness with Events ---

# Join the event data with the player data
merged_df = events_df.join(players_df, on='player.id')
print("✅ Merged player data with events.")

# --- 3. Isolate Corners and Determine Swing Type ---

print("Analyzing corner swing types...")
corners_df = merged_df[merged_df['type.primary'] == 'corner'].copy()

# Determine side of the pitch
# location.y < 50 is the left side (from attacker's view)
# location.y >= 50 is the right side
corners_df['side'] = np.where(corners_df['location.y'] < 50, 'Left', 'Right')

# Define swing type
conditions = [
    # Inswinger
    (corners_df['foot'] == 'right') & (corners_df['side'] == 'Left'),
    (corners_df['foot'] == 'left') & (corners_df['side'] == 'Right'),
    # Outswinger
    (corners_df['foot'] == 'right') & (corners_df['side'] == 'Right'),
    (corners_df['foot'] == 'left') & (corners_df['side'] == 'Left')
]
choices = ['Inswinger', 'Inswinger', 'Outswinger', 'Outswinger']
corners_df['swing_type'] = np.select(conditions, choices, default='Unknown')

# --- 4. Isolate "Success" Events (Shots & Goals) ---

# Get all shots, penalties, and own goals
shots_df = merged_df[merged_df['type.primary'].isin(['shot', 'penalty', 'own_goal'])].copy()
shots_df['shot.xg'] = pd.to_numeric(shots_df['shot.xg'], errors='coerce').fillna(0)
# An 'own_goal' is a goal, or a shot.isGoal
shots_df['is_goal'] = (shots_df['shot.isGoal'] == True) | (shots_df['type.primary'] == 'own_goal')

# Sort both DataFrames for efficiency
corners_df = corners_df.sort_values(by=['matchId', 'matchTimestamp'])
shots_df = shots_df.sort_values(by=['matchId', 'matchTimestamp'])

# --- 5. Loop Through Corners to Find Success (20s window) ---

print(f"Processing {len(corners_df)} corners to find shots in a 20s window...")
results = []

# Use tqdm to show a progress bar
for _, corner in tqdm(corners_df.iterrows(), total=corners_df.shape[0]):
    # Define the 20-second window
    corner_time = corner['matchTimestamp']
    window_end = corner_time + pd.Timedelta(seconds=20)
    
    # Find all shots *by the same team* within the window
    shots_in_window = shots_df[
        (shots_df['matchId'] == corner['matchId']) &
        (shots_df['team.name'] == corner['team.name']) &
        (shots_df['matchTimestamp'] > corner_time) &
        (shots_df['matchTimestamp'] <= window_end)
    ]
    
    total_xg_in_window = shots_in_window['shot.xg'].sum()
    total_goals_in_window = shots_in_window['is_goal'].sum()
    
    results.append({
        'swing_type': corner['swing_type'],
        'goals_in_window': total_goals_in_window,
        'xg_in_window': total_xg_in_window
    })

print("✅ Analysis complete. Aggregating results...")

# --- 6. Aggregate and Display Results ---

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Group by swing_type and aggregate
summary = results_df.groupby('swing_type').agg(
    Total_Corners=('swing_type', 'size'),
    Total_Goals=('goals_in_window', 'sum'),
    Total_xG=('xg_in_window', 'sum')
)

# Calculate rates
summary['Goal %'] = (summary['Total_Goals'] / summary['Total_Corners']) * 100
summary['xG per Corner'] = summary['Total_xG'] / summary['Total_Corners']

# Format for printing
summary['Total_xG'] = summary['Total_xG'].round(2)
summary['Goal %'] = summary['Goal %'].round(2)
summary['xG per Corner'] = summary['xG per Corner'].round(3)

print("\n--- In-swing vs. Out-swing Corner Analysis (20s Window) ---")
print(summary)