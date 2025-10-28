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
import scipy.stats # For linear regression trend line

# --- NEW FUNCTION: Calculate Rolling xG ---
@st.cache_data # Cache the results for performance
def calculate_rolling_xg(season_events_df, matches_summary_df, team_name, window_size=10):
    """Calculates rolling xG For and Conceded for a team over the season."""
    
    # 1. Filter for shots and ensure xG is numeric
    shots_df = season_events_df[season_events_df.get('type.primary') == 'shot'].copy()
    shots_df['shot.xg'] = pd.to_numeric(shots_df.get('shot.xg'), errors='coerce').fillna(0)
    
    # 2. Calculate xG per team per match
    xg_per_match = shots_df.groupby(['matchId', 'team.name'])['shot.xg'].sum().unstack(fill_value=0)
    
    # 3. Merge with match summary to get dates and opponents
    team_matches_df = matches_summary_df[
        (matches_summary_df['home_team'] == team_name) | (matches_summary_df['away_team'] == team_name)
    ].copy()
    
    # Ensure date is datetime, handle potential errors
    team_matches_df['date'] = pd.to_datetime(team_matches_df['date'], errors='coerce')
    team_matches_df.dropna(subset=['date'], inplace=True) # Remove matches with invalid dates
    team_matches_df.sort_values(by='date', inplace=True)
    
    # Merge xG data
    team_matches_df = team_matches_df.merge(xg_per_match, on='matchId', how='left').fillna(0)
    
    # 4. Determine xG For and xG Conceded
    xg_for = []
    xg_conceded = []
    
    for index, row in team_matches_df.iterrows():
        home = row['home_team']
        away = row['away_team']
        # Use .get(column, 0) for safe access in case a team column doesn't exist after merge
        home_xg = row.get(home, 0)
        away_xg = row.get(away, 0)
        
        if home == team_name:
            xg_for.append(home_xg)
            xg_conceded.append(away_xg)
        else: # Team must be away team
            xg_for.append(away_xg)
            xg_conceded.append(home_xg)
            
    team_matches_df['xg_for'] = xg_for
    team_matches_df['xg_conceded'] = xg_conceded
    
    # 5. Calculate Rolling Averages
    team_matches_df['xg_for_roll'] = team_matches_df['xg_for'].rolling(window=window_size, min_periods=1).mean()
    team_matches_df['xg_conceded_roll'] = team_matches_df['xg_conceded'].rolling(window=window_size, min_periods=1).mean()
    
    # 6. Prepare data for trend line (numeric representation of dates)
    # Convert dates to ordinal numbers for regression
    team_matches_df['date_ordinal'] = team_matches_df['date'].apply(lambda date: date.toordinal())
    
    return team_matches_df[['date', 'date_ordinal', 'xg_for_roll', 'xg_conceded_roll']]


# app.py (Add this function)

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
    subtitle = f"Liga 3 Portugal, 2025/26 | Total xGA: {total_xg_against} | Goals Against: {goals_against}"

    ax_pitch.set_title(f"{team_to_analyze} Shots CONCEDED Map (Non-Penalty)\n{subtitle}", fontsize=18, weight='bold')

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
st.title("⚽ Liga 3 Match & Season Analysis")

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


        st.subheader("Rolling xG Trend")
      

        # Season-long tables (Existing Code)
        st.subheader("Season-Long Stats")
        # ... (rest of your existing code for corner stats etc.) ...
        if selected_team in season_team_stats and 'corners' in season_team_stats[selected_team]:
            st.markdown("**Corner Kick Summary**")
            st.dataframe(season_team_stats[selected_team]['corners'])
        else:
            st.write("No season-long stats available for this team.")