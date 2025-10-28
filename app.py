# app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from mplsoccer import Pitch
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
from matplotlib.table import Table

# ==============================================================================
# 1. PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Soccer Match & Season Dashboard",
    layout="wide"
)

# ==============================================================================
# 2. DATA LOADING (with Caching)
# ==============================================================================
@st.cache_data
def load_data():
    """Load all pre-processed data files."""
    try:
        # We need the raw events for plotting shot coordinates
        raw_events_df = pd.read_parquet('raw_events.parquet') # Make sure your process_data saves this
        matches_summary_df = pd.read_parquet('matches_summary.parquet')
        
        with open('all_match_data.pkl', 'rb') as f:
            all_match_data = pickle.load(f)
            
        with open('season_team_stats.pkl', 'rb') as f:
            season_team_stats = pickle.load(f)

        return raw_events_df, matches_summary_df, all_match_data, season_team_stats
    except FileNotFoundError as e:
        st.error(f"❌ Error: A data file was not found. Please run `process_data.py` first. Missing file: {e.filename}")
        return None, None, None, None
# ==============================================================================
# 2. Trendline Viz
# ==============================================================================

# app.py (Add these functions)
import scipy.stats # For percentile ranks if needed, though pandas rank is used

# --- NEW FUNCTION: Calculate Season Team Stats (Combined for Radars) ---
@st.cache_data # Cache the results for performance
def calculate_all_team_radars_stats(season_events_df, matches_summary_df):
    """Calculates aggregated stats and percentiles for Offensive, Distribution, and Defensive radars."""
    
    print("Calculating team radar stats...") # Add print for debugging cache
    all_teams_stats = {}
    
    # --- Data Prep ---
    teams = season_events_df['team.name'].unique()
    # Ensure matchId exists before nunique
    matches_played = season_events_df.groupby('team.name')['matchId'].nunique() if 'matchId' in season_events_df.columns else pd.Series(dtype='int')

    # Convert relevant columns safely
    season_events_df['possession.duration_sec'] = pd.to_numeric(season_events_df.get('possession.duration', pd.Series(dtype='str')).str.replace('s', ''), errors='coerce')
    season_events_df['location.x'] = pd.to_numeric(season_events_df.get('location.x'), errors='coerce')
    season_events_df['location.y'] = pd.to_numeric(season_events_df.get('location.y'), errors='coerce')
    season_events_df['pass.endLocation.x'] = pd.to_numeric(season_events_df.get('pass.endLocation.x'), errors='coerce')
    season_events_df['pass.endLocation.y'] = pd.to_numeric(season_events_df.get('pass.endLocation.y'), errors='coerce')
    season_events_df['pass.length'] = pd.to_numeric(season_events_df.get('pass.length'), errors='coerce')
    season_events_df['shot.xg'] = pd.to_numeric(season_events_df.get('shot.xg'), errors='coerce')

    # Pre-calculate possession time and losses (from distribution radar logic)
    total_possession_time_per_team = season_events_df.drop_duplicates(subset='possession.id').groupby('possession.team.name')['possession.duration_sec'].sum()
    league_total_in_play_time = total_possession_time_per_team.sum()
    
    losses_df = pd.DataFrame() # Initialize empty
    if 'possession.id' in season_events_df.columns:
        season_events_df['next_possession.id'] = season_events_df['possession.id'].shift(-1)
        possession_changes = season_events_df[season_events_df['possession.id'] != season_events_df['next_possession.id']]
        losses_df = possession_changes[possession_changes.get('infraction.type') != 'foul_suffered'].copy()

    # Pre-calculate opponent events for defensive stats
    # Create opponent name column if it doesn't exist (handle potential merge issues)
    if 'opponentTeam.name' not in season_events_df.columns and 'matchId' in season_events_df.columns:
         temp_summary = matches_summary_df[['matchId', 'home_team', 'away_team']].copy()
         temp_summary.rename(columns={'home_team':'ht', 'away_team':'at'}, inplace=True) # Short names to avoid conflict
         season_events_df = season_events_df.merge(temp_summary, on='matchId', how='left')
         season_events_df['opponentTeam.name'] = np.where(season_events_df['team.name'] == season_events_df['ht'], season_events_df['at'], season_events_df['ht'])
         season_events_df.drop(columns=['ht', 'at'], inplace=True) # Drop temporary columns


    # --- Loop Through Teams ---
    for team in teams:
        team_events = season_events_df[season_events_df.get('team.name') == team]
        # Ensure opponentTeam.name exists before filtering
        opponent_events = season_events_df[season_events_df.get('opponentTeam.name') == team] if 'opponentTeam.name' in season_events_df.columns else pd.DataFrame()

        games = matches_played.get(team, 0)
        if games == 0: continue

        # --- Offensive Stats ---
        team_shots = team_events[team_events.get('type.primary') == 'shot']
        shots = team_shots.shape[0] / games
        goals = team_shots[team_shots.get('shot.isGoal') == True].shape[0] / games
        xg = team_shots['shot.xg'].sum() / games
        xg_per_shot = xg / shots if shots > 0 else 0
        PENALTY_AREA_X=83; PENALTY_AREA_Y1, PENALTY_AREA_Y2 = (21, 79)
        actions_in_box = team_events[(team_events['location.x'].fillna(0) >= PENALTY_AREA_X) & (team_events['location.y'].fillna(0).between(PENALTY_AREA_Y1, PENALTY_AREA_Y2))].shape[0] / games
        team_passes = team_events[team_events.get('type.primary') == 'pass']
        passes_into_box = team_passes[(team_passes['pass.endLocation.x'].fillna(0) >= PENALTY_AREA_X) & (team_passes['pass.endLocation.y'].fillna(0).between(PENALTY_AREA_Y1, PENALTY_AREA_Y2))].shape[0] / games
        crosses = team_passes[team_passes.get('type.secondary','').astype(str).str.contains('cross', na=False)].shape[0] / games
        team_duels_off = team_events[team_events.get('type.primary') == 'duel']
        dribbles = team_duels_off[team_duels_off.get('groundDuel.takeOn') == True].shape[0] / games

        # --- Distribution Stats ---
        passes_per_match = team_passes.shape[0] / games
        # Progressive Passes (using simpler definition for broader compatibility)
        prog_cond1 = (team_passes['location.x'].fillna(101) < 60) & (team_passes['pass.endLocation.x'].fillna(0) >= 60)
        prog_cond2 = (team_passes['location.x'].fillna(0) >= 60) & (team_passes['pass.endLocation.x'].fillna(0) >= 60) & (team_passes['pass.length'].fillna(0) >= 10)
        progressive_passes = team_passes[prog_cond1 | prog_cond2].shape[0] / games
        
        # Directness (Simplified - requires pass.length)
        directness = team_passes['pass.length'].mean() # Using average pass length as proxy

        team_possession_sec = total_possession_time_per_team.get(team, 0)
        ball_possession_pct = (team_possession_sec / league_total_in_play_time) * 100 if league_total_in_play_time > 0 else 0 # Corrected %

        final_third_entries = 0
        if 'possession.id' in team_events.columns and 'location.x' in team_events.columns:
            try:
                possessions_grouped = team_events.groupby('possession.id')[['location.x']]
                valid_groups = possessions_grouped.filter(lambda x: not x['location.x'].isna().all())
                if not valid_groups.empty:
                     final_third_entries_series = valid_groups.groupby('possession.id')['location.x'].transform(lambda x: x.min() < 66.6 and x.max() >= 66.6)
                     final_third_entries = final_third_entries_series[final_third_entries_series].index.get_level_values('possession.id').nunique() / games
            except Exception: final_third_entries = 0 # Handle potential errors

        avg_in_possession_height = team_events['location.x'].mean() # Simplified
        avg_out_of_possession_height = 0 # Placeholder - requires more complex opponent phase logic

        losses = losses_df[losses_df.get('team.name') == team].shape[0] / games if not losses_df.empty else 0

        # --- Defensive Stats ---
        goals_against=0; xg_against=0; shots_against=0; xg_per_shot_against=0;
        aerial_duel_win_pct=0; defensive_duel_win_pct=0; interceptions=0; fouls=0; ppda=np.inf;
        
        if not opponent_events.empty:
            opponent_shots = opponent_events[opponent_events.get('type.primary') == 'shot']
            goals_against = opponent_shots[opponent_shots.get('shot.isGoal') == True].shape[0] / games
            xg_against = opponent_shots['shot.xg'].sum() / games
            shots_against = opponent_shots.shape[0] / games
            xg_per_shot_against = xg_against / shots_against if shots_against > 0 else 0

        team_duels_def = team_events[team_events.get('type.primary') == 'duel']
        aerial_duels = team_duels_def[team_duels_def.get('type.secondary','').astype(str).str.contains('aerial', na=False)]
        total_aerial_duels = aerial_duels.shape[0]
        won_aerial_duels_count = aerial_duels[aerial_duels.get('aerialDuel.firstTouch') == True].shape[0]
        aerial_duel_win_pct = (won_aerial_duels_count / total_aerial_duels) * 100 if total_aerial_duels > 0 else 0

        defensive_duels = team_duels_def[team_duels_def.get('groundDuel.duelType') == 'defensive_duel']
        total_defensive_duels = defensive_duels.shape[0]
        won_defensive_duels_count = defensive_duels[(defensive_duels.get('groundDuel.recoveredPossession') == True) | (defensive_duels.get('groundDuel.stoppedProgress') == True)].shape[0]
        defensive_duel_win_pct = (won_defensive_duels_count / total_defensive_duels) * 100 if total_defensive_duels > 0 else 0
        
        interceptions = team_events[team_events.get('type.primary') == 'interception'].shape[0] / games
        fouls = team_events[team_events.get('type.primary') == 'infraction'].shape[0] / games

        # PPDA
        in_high_press_zone = season_events_df['location.x'].fillna(0) >= 40
        opponent_passes_df = opponent_events[(opponent_events.get('type.primary') == 'pass') & in_high_press_zone[opponent_events.index]] # Align index for boolean mask
        team_def_actions_df = team_events[in_high_press_zone[team_events.index]] # Align index
        def_actions_for_ppda = team_def_actions_df[team_def_actions_df.get('type.primary').isin(['infraction', 'interception', 'duel'])].shape[0]
        ppda = opponent_passes_df.shape[0] / def_actions_for_ppda if def_actions_for_ppda > 0 else np.inf # Use inf for zero actions

        # Store all calculated stats
        all_teams_stats[team] = {
            # Offensive
            'Goals': goals, 'xG': xg, 'xG per Shot': xg_per_shot, 'Shots': shots,
            'Actions in Box': actions_in_box, 'Passes into Box': passes_into_box,
            'Crosses': crosses, 'Dribbles': dribbles,
            # Distribution
            'Passes': passes_per_match, 'Progressive Passes': progressive_passes,
            'Directness': directness, 'Ball Possession': ball_possession_pct,
            'Final 1/3 Entries': final_third_entries,
            #'Avg In-Possession Action Height': avg_in_possession_height, # Simplified/removed
            #'Avg Out-of-Possession Action Height': avg_out_of_possession_height, # Simplified/removed
            'Losses': losses,
            # Defensive
            'Goals Against': goals_against, 'xG Against': xg_against,
            'xG per Shot Against': xg_per_shot_against, 'Shots Against': shots_against,
            'Aerial Duel Win %': aerial_duel_win_pct, 'Defensive Duel Win %': defensive_duel_win_pct,
            'Interceptions': interceptions, 'Fouls': fouls, 'PPDA': ppda,
        }

    # --- Convert to DataFrame ---
    stats_df_raw = pd.DataFrame.from_dict(all_teams_stats, orient='index').fillna(0).round(2)
    # Replace inf PPDA with a high number or NaN if preferred for ranking
    stats_df_raw.replace([np.inf, -np.inf], 999, inplace=True) # Replace inf with high value

    # --- Calculate Percentiles ---
    stats_df_pct = stats_df_raw.copy()
    # Invert metrics where lower is better before ranking
    metrics_to_invert_pct = ['Goals Against', 'xG Against', 'xG per Shot Against', 'Shots Against', 'PPDA', 'Losses']
    stats_df_pct[metrics_to_invert_pct] = -stats_df_pct[metrics_to_invert_pct]

    for col in stats_df_pct.columns:
        # Use pandas rank method for percentiles
        stats_df_pct[col] = stats_df_pct[col].rank(pct=True) * 100

    return stats_df_raw, stats_df_pct


# app.py (Replace the existing plot_radar_chart function)

def plot_radar_chart(params, values_raw, values_pct, team_name, title_suffix, color, league="Liga 3 Portugal", season="2025/26"):
    """Generates a single radar chart figure using Matplotlib, matching the notebook style."""

    num_params = len(params)
    angles = np.linspace(0, 2 * np.pi, num_params, endpoint=False).tolist()
    angles += angles[:1] # Close the plot

    plot_values_pct = values_pct + values_pct[:1] # Close the plot

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True)) # Adjusted size slightly
    fig.set_facecolor('#f5f1e9')
    ax.set_facecolor('#f5f1e9')

    # Plot grid and labels
    ax.set_xticks(angles[:-1])
    ax.set_ylim(0, 100)
    ax.grid(color='gray', linestyle='--', linewidth=0.5)
    ax.spines['polar'].set_color('gray')
    ax.set_yticks([25, 50, 75])
    ax.set_yticklabels(["25th", "50th", "75th"], color="grey", size=10)
    # R-label position might need slight adjustment depending on font size
    ax.set_rlabel_position(angles[0] * 180/np.pi + 10) # Position radial labels slightly off axis 0
    ax.set_thetagrids([], []) # Hide default angle labels

    # --- ADDED: Custom Label Distances (based on your examples) ---
    # Combine distances from all your examples into one dictionary
    LABEL_DISTANCES = {
        # Offensive
        "xG per Shot": 106, "Crosses": 107,
        # Distribution
        "Directness": 106, "Avg Out-of-Possession Action Height": 108,
        "Avg In-Possession Action Height": 122, "Final 1/3 Entries": 117,
        # Defensive
        "Shots Against": 106, "xG per Shot Against": 108, "PPDA": 110,
        "Quick Recoveries": 110,
        # Default
        "DEFAULT": 115 # Adjusted default slightly for potentially more labels
    }

    # Add parameter labels (percentiles) with custom distances
    for angle, param, percentile in zip(angles[:-1], params, values_pct):
        percentile_val = int(round(percentile, 0))
        # --- UPDATED: Label format ---
        label_text = f"{param}\n({percentile_val}th %-tile)"
        distance = LABEL_DISTANCES.get(param, LABEL_DISTANCES["DEFAULT"])
        # Adjust alignment based on angle to prevent overlap
        ha_align = 'left' if (np.degrees(angle) > 100 and np.degrees(angle) < 260) else 'right'
        ha_align = 'center' if (abs(np.degrees(angle) - 90) < 10 or abs(np.degrees(angle) - 270) < 10) else ha_align
        ax.text(angle, distance, label_text, ha=ha_align, va='center', size=10) # Adjusted size slightly

    # Plot the data
    ax.plot(angles, plot_values_pct, color=color, linewidth=2, linestyle='solid')
    # --- UPDATED: Alpha value ---
    ax.fill(angles, plot_values_pct, color=color, alpha=0.6) # Matched alpha=0.6

    # --- ADDED: Raw value labels ---
    for angle, value_raw, value_pct in zip(angles[:-1], values_raw, values_pct):
         # Format raw value appropriately (handle percentages)
         raw_display = f'{value_raw}%' if '%' in str(value_raw) else f'{value_raw}'
         # Place raw value inside the radar near the 95th percentile line for consistency
         ax.text(angle, 95, raw_display, ha='center', va='top', size=9, weight='bold', # Adjusted size
                 bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor='none', alpha=0.7))

    # --- ADDED: Footer text ---
    footer_text = "@lucaskimball | Data via Wyscout | Values in parentheses are percentile rank vs. other Liga 3 teams"
    fig.text(0.02, 0.02, footer_text, ha='left', va='bottom', fontsize=9, color='gray') # Adjusted size

    # --- UPDATED: Title formatting ---
    report_date = datetime.date.today().strftime("%Y-%m-%d")
    full_title = f"{team_name}\n{title_suffix} | {league} {season} (As of: {report_date})"
    ax.set_title(full_title, size=18, weight='bold', pad=40) # Adjusted size and padding

    return fig




# --- NEW FUNCTION: Plot Season Shots AGAINST Map ---
def create_season_shots_against_shotmap(season_events_df, matches_summary_df, team_to_analyze):
    """Generates a Matplotlib figure for a full season shots AGAINST map."""

    # 1. Identify opponent shots
    # Get matches involving the team
    team_match_ids = matches_summary_df[
        (matches_summary_df['home_team'] == team_to_analyze) | (matches_summary_df['away_team'] == team_to_analyze)
    ]['matchId'].unique()

    # Filter events for those matches
    relevant_events = season_events_df[season_events_df['matchId'].isin(team_match_ids)]

    # Filter for shots NOT taken by the team_to_analyze (i.e., opponent shots)
    opponent_shots_df = relevant_events[
        (relevant_events.get('type.primary') == 'shot') &
        (relevant_events.get('team.name') != team_to_analyze) &
        (~relevant_events.get('type.secondary','').astype(str).str.contains('penalty', na=False)) # Exclude penalties
    ].copy().reset_index(drop=True)

    if opponent_shots_df.empty:
        print(f"Debug: No opponent shots found for {team_to_analyze}") # Add print for debugging
        return None # Return None if no opponent shots found

    # 2. Create the plot (Similar to the 'Shots For' map)
    fig = plt.figure(figsize=(12, 12))
    fig.set_facecolor('#f5f1e9')
    pitch = Pitch(pitch_type='wyscout', pitch_color='#f5f1e9', line_color='black', line_zorder=2, half=True)
    # Flip coordinates if needed - Wyscout usually has shots going left-to-right
    # If shots appear on wrong half, invert pitch.draw() or coordinates
    ax_pitch = fig.add_subplot()
    pitch.draw(ax=ax_pitch)

    # 3. Plotting logic (using same colormap)
    XG_MAX = 0.8
    colors = ["#03045e", "#ade8f4", "#fff3b0", "#ff8c00", "#e63946", "#800f2f"]
    nodes = [0.0, 0.1 / XG_MAX, 0.2 / XG_MAX, 0.4 / XG_MAX, 0.6 / XG_MAX, 1.0]
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", list(zip(nodes, colors)))

    for index, shot in opponent_shots_df.iterrows():
        # Wyscout coordinates: (0,0) bottom-left, (100,100) top-right. Shots usually L->R.
        # Plotting on half pitch needs coordinates relative to attacking goal (usually right side)
        x, y = shot.get('location.x'), shot.get('location.y')
        xg = pd.to_numeric(shot.get('shot.xg'), errors='coerce')
        is_goal = shot.get('shot.isGoal') == True

        # Skip if coordinates or xG are invalid
        if pd.isna(x) or pd.isna(y) or pd.isna(xg):
            continue

        color = cmap(min(xg / XG_MAX, 1.0))
        edge_color = 'green' if is_goal else 'black' # Simpler edge: goal or not goal
        pitch.scatter(x, y, s=150, facecolor=color, edgecolor=edge_color, linewidth=1.5, ax=ax_pitch, zorder=3, alpha=0.7)

    # 4. Titles and Stats
    total_shots_against = len(opponent_shots_df)
    total_xg_against = round(pd.to_numeric(opponent_shots_df.get('shot.xg'), errors='coerce').sum(), 2)
    goals_against = opponent_shots_df[opponent_shots_df.get('shot.isGoal') == True].shape[0]
    xg_per_shot_against = round(total_xg_against / total_shots_against, 3) if total_shots_against > 0 else 0
    subtitle = f"Liga 3 Portugal, 2025/26 | Total xGA: {total_xg_against} | Goals: {goals_against}"

    ax_pitch.set_title(f"{team_to_analyze} Shots Against Map (Non-Penalty)\n{subtitle}", fontsize=18, weight='bold')

    return fig

# ==============================================================================
# 3. PLOTTING FUNCTIONS
# (Logic from your notebooks, refactored into functions)
# ==============================================================================

def create_match_shotmap(match_events_df, match_info, team_to_analyze):
    """Generates a Matplotlib figure for a single match shotmap."""
    team_shots_df = match_events_df[(match_events_df['team.name'] == team_to_analyze) & (match_events_df['type.primary'] == 'shot')].copy().reset_index(drop=True)
    if team_shots_df.empty:
        return None # Return None if there are no shots to plot

    home_team = match_info['home_team']
    opponent = match_info['away_team'] if team_to_analyze == home_team else home_team
    
    fig = plt.figure(figsize=(12, 12))
    fig.set_facecolor('#f5f1e9')
    pitch = Pitch(pitch_type='wyscout', pitch_color='#f5f1e9', line_color='black', line_zorder=2, half=True)
    ax_pitch = fig.add_subplot()
    pitch.draw(ax=ax_pitch)
    
    # Plotting logic from your notebook...
    XG_MAX = 0.8
    colors = ["#03045e", "#ade8f4", "#fff3b0", "#ff8c00", "#e63946", "#800f2f"]
    nodes = [0.0, 0.1 / XG_MAX, 0.2 / XG_MAX, 0.4 / XG_MAX, 0.6 / XG_MAX, 1.0]
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", list(zip(nodes, colors)))

    for index, shot in team_shots_df.iterrows():
        x, y, xg = shot['location.x'], shot['location.y'], shot['shot.xg']
        is_goal = shot['shot.isGoal'] == True
        is_on_target = shot['shot.onTarget'] == True
        color = cmap(min(xg / XG_MAX, 1.0))
        edge_color = 'gray'
        if is_goal: edge_color = 'green'
        elif is_on_target: edge_color = 'black'
        pitch.scatter(x, y, s=400, facecolor=color, edgecolor=edge_color, linewidth=2, ax=ax_pitch, zorder=3)
        pitch.text(x, y, str(index + 1), ax=ax_pitch, ha='center', va='center', fontsize=9, color='white', zorder=4)

    # Titles and Legends
    subtitle = f"vs. {opponent} | Score: {match_info['score']} | xG: {team_shots_df['shot.xg'].sum():.2f}"
    ax_pitch.set_title(f"{team_to_analyze} Shot Map\n{subtitle}", fontsize=18, weight='bold')
    
    return fig

def create_season_shotmap(season_events_df, team_to_analyze):
    """Generates a Matplotlib figure for a full season shotmap."""
    team_shots_df = season_events_df[
        (season_events_df['team.name'] == team_to_analyze) & 
        (season_events_df['type.primary'] == 'shot') &
        (~season_events_df['type.secondary'].astype(str).str.contains('penalty', na=False))
    ].copy().reset_index(drop=True)

    if team_shots_df.empty:
        return None

    fig = plt.figure(figsize=(12, 12))
    fig.set_facecolor('#f5f1e9')
    pitch = Pitch(pitch_type='wyscout', pitch_color='#f5f1e9', line_color='black', line_zorder=2, half=True)
    ax_pitch = fig.add_subplot()
    pitch.draw(ax=ax_pitch)

    # Plotting logic is very similar to the match shotmap
    XG_MAX = 0.8
    colors = ["#03045e", "#ade8f4", "#fff3b0", "#ff8c00", "#e63946", "#800f2f"]
    nodes = [0.0, 0.1 / XG_MAX, 0.2 / XG_MAX, 0.4 / XG_MAX, 0.6 / XG_MAX, 1.0]
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", list(zip(nodes, colors)))

    for index, shot in team_shots_df.iterrows():
        x, y, xg = shot['location.x'], shot['location.y'], shot['shot.xg']
        is_goal = shot['shot.isGoal'] == True
        color = cmap(min(xg / XG_MAX, 1.0))
        edge_color = 'green' if is_goal else 'black'
        pitch.scatter(x, y, s=150, facecolor=color, edgecolor=edge_color, linewidth=1.5, ax=ax_pitch, zorder=3, alpha=0.7)

    total_xg = team_shots_df['shot.xg'].sum()
    goals = team_shots_df['shot.isGoal'].sum()
    subtitle = f"Liga 3 Portugal, 2025/26 | Total xG: {total_xg:.2f} | Goals: {goals}"
    ax_pitch.set_title(f"{team_to_analyze} Season Shot Map (Non-Penalty)\n{subtitle}", fontsize=18, weight='bold')

    return fig

# ==============================================================================
# 4. STREAMLIT APP UI
# ==============================================================================
st.title("Atlético CP Analysis")

# --- Load Data ---
raw_events_df, matches_summary_df, all_match_data, season_team_stats = load_data()

if raw_events_df is not None:
    # --- Sidebar for Navigation ---
    st.sidebar.title("Dashboard Controls")
    analysis_type = st.sidebar.radio("Choose Analysis Type", ('Match Analysis', 'Season-Long Analysis'))

    if analysis_type == 'Match Analysis':
        # --- Match Selection ---
        # Create a more informative display name including Gameweek
        matches_summary_df['display_name'] = matches_summary_df['home_team'] + " vs " + matches_summary_df['away_team'] + " (" + matches_summary_df['score'] + ")"
        
        # Sort matches for the dropdown, potentially by Gameweek then date/ID
        matches_summary_df.sort_values(by=['matchId'], inplace=True)
                                 
        selected_match_display = st.sidebar.selectbox("Select a Match", matches_summary_df['display_name'])

        # Find the selected match info based on the unique display name
        selected_match_info = matches_summary_df[matches_summary_df['display_name'] == selected_match_display].iloc[0]
        selected_match_id = selected_match_info['matchId']
        
        st.header(f"Match Report: {selected_match_info['home_team']} vs {selected_match_info['away_team']}")
        
        # --- Display Data for Selected Match ---
        match_data = all_match_data.get(selected_match_id)
        if match_data:
            # Shotmaps
            st.subheader("Shot Maps")
            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(create_match_shotmap(raw_events_df[raw_events_df['matchId'] == selected_match_id], selected_match_info, selected_match_info['home_team']), use_container_width=True)
            with col2:
                st.pyplot(create_match_shotmap(raw_events_df[raw_events_df['matchId'] == selected_match_id], selected_match_info, selected_match_info['away_team']), use_container_width=True)

            # Team Stats
            st.subheader("Team Stats")
            if 'team_stats' in match_data and match_data['team_stats']:
                for stat_category, df in match_data['team_stats'].items():
                    st.markdown(f"**{stat_category}**")
                    st.dataframe(df)
            
            # Player Stats
            st.subheader("Player Stats")
            if 'player_stats' in match_data and match_data['player_stats']:
                st.markdown(f"**{selected_match_info['home_team']}**")
                st.dataframe(match_data['player_stats']['home'])
                st.markdown(f"**{selected_match_info['away_team']}**")
                st.dataframe(match_data['player_stats']['away'])

    elif analysis_type == 'Season-Long Analysis':
        # --- Team Selection ---
        all_teams = sorted(pd.concat([matches_summary_df['home_team'], matches_summary_df['away_team']]).unique())
        selected_team = st.sidebar.selectbox("Select a Team", all_teams)

        st.header(f"Season Report: {selected_team}")

        # --- Calculate Radar Stats (Run Once) ---
        # Ensure raw_events_df and matches_summary_df are available
        if raw_events_df is not None and matches_summary_df is not None:
             stats_df_raw, stats_df_pct = calculate_all_team_radars_stats(raw_events_df, matches_summary_df)
        else:
             stats_df_raw, stats_df_pct = pd.DataFrame(), pd.DataFrame() # Empty DFs if data failed load

        # --- NEW: Radar Charts ---
        st.subheader("Team Style Radars (Percentile Ranks vs Liga 3)")

        if selected_team in stats_df_raw.index and selected_team in stats_df_pct.index:
            col_r1, col_r2, col_r3 = st.columns(3)

            # Define parameters for each radar (ensure these match columns in stats_df_raw)
            offensive_params = ['Goals', 'xG', 'xG per Shot', 'Shots', 'Actions in Box', 'Passes into Box', 'Crosses', 'Dribbles']
            distribution_params = ['Passes', 'Progressive Passes', 'Directness', 'Ball Possession', 'Final 1/3 Entries', 'Losses']
            defensive_params = ['Goals Against', 'xG Against', 'xG per Shot Against', 'Shots Against', 'Aerial Duel Win %', 'Defensive Duel Win %', 'Interceptions', 'Fouls', 'PPDA'] # Removed Quick Recoveries, Avg Recovery Dist for now if missing

            # Get data for the selected team
            team_stats_raw = stats_df_raw.loc[selected_team]
            team_stats_pct = stats_df_pct.loc[selected_team]

            # Define league/season (can be dynamic later if needed)
            current_league = "Liga 3 Portugal"
            current_season = "2025/26"

            with col_r1:
                st.markdown("**Offensive Radar**")
                valid_offensive_params = [p for p in offensive_params if p in team_stats_raw.index]
                if valid_offensive_params:
                     # --- UPDATED CALL ---
                     fig_off = plot_radar_chart(
                         valid_offensive_params,
                         team_stats_raw[valid_offensive_params].tolist(), # Raw values
                         team_stats_pct[valid_offensive_params].tolist(), # Percentile values
                         selected_team, "Offensive Radar", '#e60000', # Title Suffix, Color
                         league=current_league, season=current_season # Pass league/season
                     )
                     st.pyplot(fig_off, use_container_width=True)
                else: st.warning("Missing data for offensive radar.")


            with col_r2:
                st.markdown("**Distribution Radar**")
                valid_distribution_params = [p for p in distribution_params if p in team_stats_raw.index]
                if valid_distribution_params:
                     # Format Ball Possession for raw display
                     raw_dist_values = team_stats_raw[valid_distribution_params].tolist()
                     poss_index = valid_distribution_params.index('Ball Possession') # Find index
                     raw_dist_values[poss_index] = f"{raw_dist_values[poss_index]:.0f}%" # Add % sign

                     # --- UPDATED CALL ---
                     fig_dist = plot_radar_chart(
                        valid_distribution_params,
                        raw_dist_values, # Use formatted raw values
                        team_stats_pct[valid_distribution_params].tolist(),
                        selected_team, "Distribution Radar", '#0077b6', # Blue
                        league=current_league, season=current_season
                    )
                     st.pyplot(fig_dist, use_container_width=True)
                else: st.warning("Missing data for distribution radar.")

            with col_r3:
                st.markdown("**Defensive Radar**")
                valid_defensive_params = [p for p in defensive_params if p in team_stats_raw.index]
                if valid_defensive_params:
                     # Format Duel Win % for raw display
                     raw_def_values = team_stats_raw[valid_defensive_params].tolist()
                     try: # Add try-except in case columns aren't present
                         aerial_idx = valid_defensive_params.index('Aerial Duel Win %')
                         raw_def_values[aerial_idx] = f"{raw_def_values[aerial_idx]:.0f}%"
                     except ValueError: pass # Ignore if column not found
                     try:
                         def_idx = valid_defensive_params.index('Defensive Duel Win %')
                         raw_def_values[def_idx] = f"{raw_def_values[def_idx]:.0f}%"
                     except ValueError: pass # Ignore if column not found

                     # --- UPDATED CALL ---
                     fig_def = plot_radar_chart(
                        valid_defensive_params,
                        raw_def_values, # Use formatted raw values
                        team_stats_pct[valid_defensive_params].tolist(),
                        selected_team, "Defensive Radar", '#52A736', # Green
                        league=current_league, season=current_season
                    )
                     st.pyplot(fig_def, use_container_width=True)
                else: st.warning("Missing data for defensive radar.")
        else:
            st.warning(f"Could not find calculated radar statistics for {selected_team}.")
        # --- END RADARS ---

        # --- NEW: Side-by-Side Shot Maps ---
        st.subheader("Season Shot Maps (Non-Penalty)")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**Shots FOR {selected_team}**")
            fig_shots_for = create_season_shotmap(raw_events_df, selected_team)
            if fig_shots_for:
                st.pyplot(fig_shots_for, use_container_width=True)
            else:
                st.warning("No shots found FOR this team.")

        with col2:
            st.markdown(f"**Shots AGAINST {selected_team}**")
            # Pass matches_summary_df to the new function
            fig_shots_against = create_season_shots_against_shotmap(raw_events_df, matches_summary_df, selected_team)
            if fig_shots_against:
                st.pyplot(fig_shots_against, use_container_width=True)
            else:
                st.warning("No shots found AGAINST this team.")
        # --- END SHOT MAPS ---

      

        # Season-long tables (Existing Code)
        st.subheader("Season-Long Stats")
        # ... (rest of your existing code for corner stats etc.) ...
        if selected_team in season_team_stats and 'corners' in season_team_stats[selected_team]:
            st.markdown("**Corner Kick Summary**")
            st.dataframe(season_team_stats[selected_team]['corners'])
        else:
            st.write("No season-long stats available for this team.")