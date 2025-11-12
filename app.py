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
import datetime # For Radar dates
import matplotlib.gridspec as gridspec # For Corner plots
import scipy.stats # For Radar stats percentile rank
import os # For checking logo file paths
from PIL import Image # For scatter plot logos
from matplotlib.offsetbox import OffsetImage, AnnotationBbox # For scatter plot logos
from adjustText import adjust_text # For scatter plot logos
from math import pi # For player radar charts
import matplotlib.dates as mdates # <-- ADD THIS LINE
from matplotlib.gridspec import GridSpec # For player radar charts
from collections import defaultdict # For player radar calculations
import seaborn as sns # For player radar distributions
from collections import defaultdict # Make sure this is at the top with other imports
import plotly.graph_objects as go
import io # For saving the in-memory image
# ... after your other imports ...
import base64


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
        raw_events_df = pd.read_parquet('raw_events.parquet')
        matches_summary_df = pd.read_parquet('matches_summary.parquet')
        
        with open('all_match_data.pkl', 'rb') as f:
            all_match_data = pickle.load(f)
            
        with open('season_team_stats.pkl', 'rb') as f:
            season_team_stats = pickle.load(f)
            
        with open('player_minutes_and_positions.pkl', 'rb') as f:
            player_minutes_df = pickle.load(f)

        return raw_events_df, matches_summary_df, all_match_data, season_team_stats, player_minutes_df
    
    except FileNotFoundError as e:
        st.error(f"❌ Error: A data file was not found. Please run `process_data.py` (including the new player minutes step) first. Missing file: {e.filename}")
        return None, None, None, None, None
    except Exception as e:
        st.error(f"An error occurred loading data: {e}")
        return None, None, None, None, None

# ... after your @st.cache_data def load_data(): ...
@st.cache_data
def load_player_details():
    """Loads the player details (foot, height, etc.) from the pkl file."""
    try:
        with open('player_details.pkl', 'rb') as f:
            player_details_list = pickle.load(f)
        
        players_df = pd.DataFrame(player_details_list)
        players_df = players_df.dropna(subset=['playerId'])
        players_df['playerId'] = players_df['playerId'].astype(int)
        players_df = players_df.set_index('playerId')
        return players_df
    except FileNotFoundError:
        st.error("❌ Error: `player_details.pkl` not found. Please run `get_player_details.py` locally and push the file.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An error occurred loading player details: {e}")
        return pd.DataFrame()

def _calculate_age(birth_date):
    """Helper function to calculate age from birth_date string."""
    if not birth_date or pd.isna(birth_date):
        return "N/A"
    try:
        today = datetime.date.today()
        birth = datetime.datetime.strptime(birth_date, '%Y-%m-%d').date()
        # Calculate age as a float
        age_in_days = (today - birth).days
        age = age_in_days / 365.25 
        return age # Return the float
    except Exception:
        return "N/A"

@st.cache_data
def get_player_match_stats(player_name, _all_match_data, _matches_summary_df):
    """
    Goes through all match data and extracts the individual match stats
    for a single selected player.
    """
    player_matches = []
    
    # Create a quick lookup map for match info
    match_info_map = _matches_summary_df.set_index('matchId').to_dict('index')
    
    for match_id, match_data in _all_match_data.items():
        if not match_data or 'player_stats' not in match_data:
            continue
            
        home_df = match_data['player_stats'].get('home')
        away_df = match_data['player_stats'].get('away')
        
        player_stats_series = None
        opponent = "N/A"
        
        match_info = match_info_map.get(match_id, {})
        home_team = match_info.get('homeTeamName', 'Home')
        away_team = match_info.get('awayTeamName', 'Away')
        
        if home_df is not None and player_name in home_df.index:
            player_stats_series = home_df.loc[player_name]
            opponent = f"vs. {away_team} (H)"
        elif away_df is not None and player_name in away_df.index:
            player_stats_series = away_df.loc[player_name]
            opponent = f"vs. {home_team} (A)"
            
        if player_stats_series is not None:
            # Convert series to a dictionary and add match info
            stats_dict = player_stats_series.to_dict()
            stats_dict['Match'] = opponent
            stats_dict['Date'] = match_info.get('dateutc', 'N/A')
            stats_dict['Score'] = match_info.get('score', 'N/A')
            player_matches.append(stats_dict)

    if not player_matches:
        return pd.DataFrame()
        
    # Create and format the DataFrame
    match_log_df = pd.DataFrame(player_matches)
    
    # Format the date
    if 'Date' in match_log_df.columns:
        match_log_df['Date'] = pd.to_datetime(match_log_df['Date']).dt.strftime('%Y-%m-%d')
    
    # Reorder columns to put match info first
    cols_to_front = ['Date', 'Match', 'Score', 'Minutes']
    all_cols = cols_to_front + [col for col in match_log_df.columns if col not in cols_to_front]
    match_log_df = match_log_df[all_cols].fillna(0)
    match_log_df = match_log_df.sort_values(by='Date', ascending=False)
    
    return match_log_df

@st.cache_data
def calculate_player_profile_stats(_raw_events_df, _player_minutes_df):
    """
    A new, streamlined function to calculate ONLY the key stats for player profiles.
    Calculates: npxG, xA, xT, Progressive Passes, Deep Completions, and GK Stats.
    """
    print("--- STARTING: Streamlined player profile stats ---")
    
    events_df = _raw_events_df.copy()
    # Start with our base player list
    combined_df = _player_minutes_df.copy()

    # Ensure player.id is int for all merging
    events_df['player.id'] = pd.to_numeric(events_df['player.id'], errors='coerce')
    events_df = events_df.dropna(subset=['player.id'])
    events_df['player.id'] = events_df['player.id'].astype(int)

    # --- 1. Calculate npxG, xAOP, xASP ---
    print("Step 1: Calculating npxG, xAOP, xASP...")
    try:
        shots_df = events_df[
            (events_df['shot.xg'].notna()) &
            (events_df['type.primary'] != 'penalty')
        ].copy()
        npxg_totals = shots_df.groupby('player.id')['shot.xg'].sum().reset_index().rename(columns={'shot.xg': 'npxG'})

        events_df['shot_event_id'] = np.where(events_df['shot.xg'].notna(), events_df['id'], np.nan)
        events_df['next_shot_id'] = events_df.groupby('matchId')['shot_event_id'].bfill()
        shot_xg_map = events_df[events_df['shot.xg'].notna()].set_index('id')['shot.xg'].to_dict()

        assists_df = events_df[events_df.get('type.secondary', pd.Series(dtype='object')).apply(lambda x: isinstance(x, (list, np.ndarray)) and 'shot_assist' in x)].copy()
        assists_df['xA'] = assists_df['next_shot_id'].map(shot_xg_map)
        set_piece_types = ['corner', 'free_kick', 'throw_in', 'goal_kick']
        assists_df['assist_type'] = np.where(assists_df['type.primary'].isin(set_piece_types), 'xASP', 'xAOP')
        
        xa_split_totals = assists_df.groupby(['player.id', 'assist_type'])['xA'].sum()
        xa_final_df = xa_split_totals.unstack(fill_value=0).reset_index()

        final_stats_df = pd.merge(npxg_totals, xa_final_df, on='player.id', how='outer')
        combined_df = pd.merge(combined_df, final_stats_df, left_on='playerId', right_on='player.id', how='left')
        if 'player.id' in combined_df.columns: combined_df = combined_df.drop(columns=['player.id'])
    except Exception as e:
        print(f"  -> ❌ ERROR (Step 1): {e}")

    # --- 2. Calculate Deep Completions and Progressive Passes ---
    print("Step 2: Calculating Deep Completions and Progressive Passes...")
    try:
        passes_df = events_df[
            (events_df['type.primary'] == 'pass') & (events_df.get('pass.accurate') == True)
        ].dropna(subset=['location.x', 'pass.endLocation.x', 'player.id']).copy()
        
        passes_df['end_x_m'] = passes_df['pass.endLocation.x'] * 1.05
        passes_df['end_y_m'] = passes_df['pass.endLocation.y'] * 0.68
        passes_df['dist_to_goal_center'] = np.sqrt((passes_df['end_x_m'] - 105)**2 + (passes_df['end_y_m'] - 34)**2)
        passes_df['is_cross'] = passes_df.get('type.secondary', pd.Series(dtype='object')).apply(lambda x: isinstance(x, (list, np.ndarray)) and 'cross' in x)
        passes_df['is_deep_completion'] = (passes_df['dist_to_goal_center'] <= 20) & (passes_df['is_cross'] == False)
        deep_completions = passes_df.groupby('player.id')['is_deep_completion'].sum().reset_index().rename(columns={'is_deep_completion': 'Deep Completions'})

        start_x = passes_df['location.x']; end_x = passes_df['pass.endLocation.x']
        cond1 = (start_x < 50) & (end_x < 50) & (end_x - start_x >= 30)
        cond2 = (start_x < 50) & (end_x >= 50) & (end_x - start_x >= 15)
        cond3 = (start_x >= 50) & (end_x >= 50) & (end_x - start_x >= 10)
        passes_df['is_progressive_pass'] = cond1 | cond2 | cond3
        progressive_passes = passes_df.groupby('player.id')['is_progressive_pass'].sum().reset_index().rename(columns={'is_progressive_pass': 'Progressive Passes'})

        new_metrics_df = pd.merge(deep_completions, progressive_passes, on='player.id', how='outer')
        combined_df = pd.merge(combined_df, new_metrics_df, left_on='playerId', right_on='player.id', how='left')
        if 'player.id' in combined_df.columns: combined_df = combined_df.drop(columns=['player.id'])
    except Exception as e:
        print(f"  -> ❌ ERROR (Step 2): {e}")

    # --- 3. Calculate Goalkeeper Stats ---
    print("Step 3: Calculating Goalkeeper stats...")
    try:
        gk_ids = events_df[events_df.get('player.position') == 'GK']['player.id'].dropna().unique().astype(int)
        gk_events_df = events_df[events_df['player.id'].isin(gk_ids)].copy()
        
        shots_faced_df = events_df[(events_df.get('type.primary') == 'shot') & (events_df.get('shot.onTarget') == True) & (events_df.get('shot.goalkeeper.id').notna())].copy()
        shots_faced_df['shot.goalkeeper.id'] = shots_faced_df['shot.goalkeeper.id'].astype(int)
        gk_shot_stopping_stats = shots_faced_df.groupby('shot.goalkeeper.id').agg(shotsOnTargetAgainst=('shot.isGoal', 'count'), goalsConceded=('shot.isGoal', 'sum'), psxG_faced=('shot.postShotXg', 'sum')).reset_index().rename(columns={'shot.goalkeeper.id': 'player.id'})
        if not gk_shot_stopping_stats.empty:
            gk_shot_stopping_stats['goalsPrevented'] = gk_shot_stopping_stats['psxG_faced'] - gk_shot_stopping_stats['goalsConceded']
            gk_shot_stopping_stats['goalsPreventedPerSOT'] = (gk_shot_stopping_stats['goalsPrevented'] / gk_shot_stopping_stats['shotsOnTargetAgainst']).fillna(0)
            gk_shot_stopping_stats['savePercentage'] = ((gk_shot_stopping_stats['shotsOnTargetAgainst'] - gk_shot_stopping_stats['goalsConceded']) / gk_shot_stopping_stats['shotsOnTargetAgainst'] * 100).fillna(0)
        else:
            gk_shot_stopping_stats = gk_shot_stopping_stats.reindex(columns=['player.id', 'shotsOnTargetAgainst', 'goalsConceded', 'psxG_faced', 'goalsPrevented', 'goalsPreventedPerSOT', 'savePercentage']).fillna(0)

        exits = gk_events_df[gk_events_df['type.primary'] == 'goalkeeper_exit'].groupby('player.id').size().reset_index(name='exits')
        recoveries_gk = gk_events_df[gk_events_df.get('type.secondary', pd.Series(dtype='object')).apply(lambda x: isinstance(x, (list, np.ndarray)) and 'recovery' in x)].groupby('player.id').size().reset_index(name='recoveries_gk')
        gk_passes = gk_events_df[gk_events_df['type.primary'] == 'pass']
        passes_total_gk = gk_passes.groupby('player.id').size().reset_index(name='passes_gk')
        passes_succ_gk = gk_passes[gk_passes['pass.accurate'] == True].groupby('player.id').size().reset_index(name='passesSuccessful_gk')
        long_passes_total_gk = gk_passes[gk_passes.get('type.secondary', pd.Series(dtype='object')).apply(lambda x: isinstance(x, (list, np.ndarray)) and 'long_pass' in x)].groupby('player.id').size().reset_index(name='longPasses_gk')
        long_passes_succ_gk = gk_passes[gk_passes.get('type.secondary', pd.Series(dtype='object')).apply(lambda x: isinstance(x, (list, np.ndarray)) and 'long_pass' in x) & (gk_passes['pass.accurate'] == True)].groupby('player.id').size().reset_index(name='longPassesSuccessful_gk')

        gk_report_df = pd.DataFrame({'player.id': gk_ids}); gk_report_df = pd.merge(gk_report_df, gk_shot_stopping_stats, on='player.id', how='left'); gk_report_df = pd.merge(gk_report_df, exits, on='player.id', how='left'); gk_report_df = pd.merge(gk_report_df, recoveries_gk, on='player.id', how='left'); gk_report_df = pd.merge(gk_report_df, passes_total_gk, on='player.id', how='left'); gk_report_df = pd.merge(gk_report_df, passes_succ_gk, on='player.id', how='left'); gk_report_df = pd.merge(gk_report_df, long_passes_total_gk, on='player.id', how='left'); gk_report_df = pd.merge(gk_report_df, long_passes_succ_gk, on='player.id', how='left')
        
        # Add % stats
        gk_report_df['Passes successful %'] = (gk_report_df['passesSuccessful_gk'] / gk_report_df['passes_gk'] * 100).fillna(0)
        gk_report_df['Long passes successful %'] = (gk_report_df['longPassesSuccessful_gk'] / gk_report_df['longPasses_gk'] * 100).fillna(0)

        combined_df = pd.merge(combined_df, gk_report_df, left_on='playerId', right_on='player.id', how='left')
        if 'player.id' in combined_df.columns: combined_df = combined_df.drop(columns=['player.id'])
    except Exception as e:
        print(f"  -> ❌ ERROR (Step 3): {e}")

    # --- 4. Calculate xT (Expected Threat) ---
    print("Step 4: Calculating Expected Threat (xT)...")
    try:
        xt_data_from_image = [[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.03, 0.03, 0.04, 0.04], [0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.03, 0.04, 0.05, 0.05], [0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.03, 0.05, 0.06, 0.06], [0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.04, 0.11, 0.26, 0.26], [0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.04, 0.11, 0.26, 0.26], [0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.03, 0.05, 0.06, 0.06], [0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.03, 0.04, 0.05, 0.05], [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.03, 0.03, 0.04, 0.04]]
        xt_grid = np.array(xt_data_from_image); rows, cols = xt_grid.shape
        move_df = events_df[events_df['type.primary'].isin(['pass', 'touch', 'acceleration'])].copy()
        successful_pass = (move_df['type.primary'] == 'pass') & (move_df.get('pass.accurate') == True)
        other_successful_moves = move_df['type.primary'].isin(['touch', 'acceleration'])
        move_df = move_df[successful_pass | other_successful_moves]
        move_df['start_x'] = move_df['location.x']; move_df['start_y'] = move_df['location.y']
        move_df['end_x'] = np.where(move_df['type.primary'] == 'pass', move_df.get('pass.endLocation.x'), move_df.get('carry.endLocation.x'))
        move_df['end_y'] = np.where(move_df['type.primary'] == 'pass', move_df.get('pass.endLocation.y'), move_df.get('carry.endLocation.y'))
        move_df = move_df.dropna(subset=['end_x', 'end_y', 'player.id'])
        def get_xt_zone(x, y, xt_rows, xt_cols):
            if pd.isna(x) or pd.isna(y): return None, None
            col = min(int(x / 100 * xt_cols), xt_cols - 1); row = min(int(y / 100 * xt_rows), xt_rows - 1)
            return row, col
        move_df[['start_row', 'start_col']] = move_df.apply(lambda row: get_xt_zone(row['start_x'], row['start_y'], rows, cols), axis=1, result_type='expand')
        move_df[['end_row', 'end_col']] = move_df.apply(lambda row: get_xt_zone(row['end_x'], row['end_y'], rows, cols), axis=1, result_type='expand')
        move_df['xt_start'] = move_df.apply(lambda row: xt_grid[int(row['start_row']), int(row['start_col'])] if pd.notna(row['start_row']) else 0, axis=1)
        move_df['xt_end'] = move_df.apply(lambda row: xt_grid[int(row['end_row']), int(row['end_col'])] if pd.notna(row['end_row']) else 0, axis=1)
        move_df['xT'] = move_df['xt_end'] - move_df['xt_start']
        successful_threat = move_df[move_df['xT'] > 0]
        player_xt = successful_threat.groupby('player.id')['xT'].sum().reset_index()
        combined_df = pd.merge(combined_df, player_xt, left_on='playerId', right_on='player.id', how='left')
        if 'player.id' in combined_df.columns: combined_df = combined_df.drop(columns=['player.id'])
    except Exception as e:
        print(f"  -> ❌ ERROR (Step 4): {e}")

    # --- 5. Normalize to Per 90 ---
    print("Step 5: Normalizing stats to per 90...")
    combined_df = combined_df.fillna(0)
    
    # Define only the metrics we just calculated
    metrics_to_normalize = [
        'npxG', 'xAOP', 'xASP', 'xT', 'Deep Completions', 'Progressive Passes',
        'shotsOnTargetAgainst', 'goalsConceded', 'psxG_faced', 'goalsPrevented', 'exits',
        'recoveries_gk', 'passes_gk', 'passesSuccessful_gk', 'longPasses_gk', 'longPassesSuccessful_gk'
    ]
    # Get only the metrics that actually exist in the df
    existing_metrics_to_normalize = [m for m in metrics_to_normalize if m in combined_df.columns]
    
    combined_df['totalMinutes'] = pd.to_numeric(combined_df['totalMinutes'], errors='coerce').fillna(0)
    minutes_gt_0 = combined_df['totalMinutes'] > 0
    
    for metric in existing_metrics_to_normalize:
        combined_df[metric] = pd.to_numeric(combined_df[metric], errors='coerce').fillna(0)
        combined_df[metric] = np.where(
            minutes_gt_0,
            (combined_df[metric].astype(float) / combined_df['totalMinutes']) * 90,
            0
        )

    print("--- FINISHED: Streamlined player stats ---")
    return combined_df.fillna(0)

@st.cache_data
def load_historical_data():
    """
    Load all historical data files for rolling charts.
    --- OPTIMIZED to only load necessary columns to save memory. ---
    """
    try:
        # 1. Define only the columns we absolutely need
        events_cols = ['type.primary', 'shot.xg', 'matchId', 'team.name']
        # --- ADDED 'seasonId' to this list ---
        matches_cols = ['matchId', 'dateutc', 'gameweek', 'homeTeamName', 'awayTeamName', 'seasonId']
        
        # 2. Load *only* those columns
        hist_events_df = pd.read_parquet('historical_events.parquet', columns=events_cols)
        hist_matches_df = pd.read_parquet('historical_matches.parquet', columns=matches_cols)
        
        return hist_events_df, hist_matches_df
    
    except FileNotFoundError as e:
        st.error(f"❌ Error: A historical data file was not found. Please run `process_data.py` (and force-push the files). Missing file: {e.filename}")
        return None, None
    except Exception as e:
        st.error(f"An error occurred loading historical data: {e}")
        return None, None
    
# ==============================================================================
# 3. GLOBAL CONSTANTS FOR PLAYER RADARS
# ==============================================================================
POSITION_GROUPS = {
    'Shot Stopper': ['GK'], 'Cross Claimer': ['GK'], 'Ball-playing GK': ['GK'],
    'Mobile Striker': ['CF', 'SS'], 'Shadow Striker': ['CF', 'SS'], 'Poacher': ['CF', 'SS'], 'Target Man': ['CF', 'SS'], 'Pressing Forward': ['CF', 'SS'],
    'Box-to-Box': ['LCMF', 'RCMF', 'AMF', 'LCMF3', 'RCMF3', 'DMF', 'LDMF', 'RDMF'],
    'Ball-Winning Mid': ['LCMF', 'RCMF', 'LCMF3', 'RCMF3', 'DMF', 'LDMF', 'RDMF'],
    'Holding Mid': ['DMF', 'LDMF', 'RDMF'],
    'Deep-lying Playmaker': ['LCMF', 'RCMF', 'LCMF3', 'RCMF3', 'DMF', 'LDMF', 'RDMF'],
    'Advanced Playmaker': ['AMF', 'RAMF', 'LAMF', 'LW', 'RW'],
    'Wide Winger': ['LW', 'RW', 'LWF', 'RWF', 'LWB', 'RWB'],
    'Creative Winger': ['LW', 'RW', 'LWF', 'RWF', 'RAMF', 'LAMF'],
    'Inside Forward': ['LW', 'RW', 'LWF', 'RWF'],
    'Full Back': ['LB', 'RB', 'LB5', 'RB5', 'LWB', 'RWB'],
    'Wingback': ['LWB', 'RWB', 'LB5', 'RB5'],
    'Inverted Full Back': ['LB', 'RB', 'LWB', 'RWB', 'LB5', 'RB5'],
    'Ball-Playing Centerback': ['LCB', 'RCB', 'CB', 'LCB3', 'RCB3'],
    'Stopper': ['LCB', 'RCB', 'CB', 'LCB3', 'RCB3'],
    'Athletic Centerback': ['LCB', 'RCB', 'CB', 'LCB3', 'RCB3'],
}
WEIGHTS = {
    'Shot Stopper': {'goalsPrevented': 10.0, 'goalsPreventedPerSOT': 10.0, 'goalsConceded': 1.0, 'exits': 2.0, 'Passes successful %': 1.0, 'Long passes successful %': 1.0, 'passes_gk': 1.0, 'recoveries_gk': 2.0},
    'Cross Claimer': {'goalsPrevented': 10.0, 'goalsPreventedPerSOT': 10.0, 'goalsConceded': 1.0, 'exits': 20.0, 'Passes successful %': 1.0, 'longPassesSuccessful_gk': 1, 'passes_gk': 1.0, 'recoveries_gk': 10},
    'Ball-playing GK': {'goalsPrevented': 10.0, 'goalsPreventedPerSOT': 10.0, 'goalsConceded': 1.0, 'exits': 2.0, 'Passes successful %': 10.0, 'longPassesSuccessful_gk': 6.0, 'passes_gk': 4.0, 'recoveries_gk': 3.0},
    'Ball-Playing Centerback': {'npxG': 1.0, 'xAOP': 1.0, 'xT': 5.0, 'Passes': 20, 'Passes successful %': 10, 'Progressive Passes': 20, 'Progressive runs': 6, 'Aerial duels': 2, 'Aerial duels successful %': 6, 'Defensive duels': 2, 'Defensive duels successful %': 8, 'Interceptions': 6, 'Recoveries': 6, 'Clearances': 2},
    'Stopper': {'npxG': 3.0, 'xAOP': 1.0, 'xT': 1.0, 'Passes': 1.0, 'Passes successful %': 1.0, 'Progressive Passes': 1.0, 'Progressive runs': 1.0, 'Aerial duels': 8, 'Aerial duels successful %': 10, 'Defensive duels': 8, 'Defensive duels successful %': 10, 'Interceptions': 8, 'Recoveries': 8, 'Clearances': 6},
    'Athletic Centerback': {'npxG': 3.0, 'xAOP': 1.0, 'xT': 1.0, 'Passes': 1.0, 'Passes successful %': 1.0, 'Progressive Passes': 1.0, 'Progressive runs': 6, 'Aerial duels': 6, 'Aerial duels successful %': 10, 'Defensive duels': 8, 'Defensive duels successful %': 10, 'Interceptions': 10, 'Recoveries': 10, 'Clearances': 6},
    'Box-to-Box': {'Passes': 4, 'Passes successful %': 3, 'Progressive Passes': 2, 'xT': 4.0, 'Goals': 1.0, 'npxG': 4, 'Shots': 2, 'xG per Shot': 1.0, 'Assists': 1.0, 'xAOP': 4, 'Progressive runs': 5, 'Dribbles successful': 4, 'Aerial duels successful': 1.0, 'Defensive duels successful': 2, 'Interceptions': 3, 'Recoveries': 4},
    'Holding Mid': {'Passes': 6, 'Passes successful %': 6, 'Progressive Passes': 2, 'xT': 4.0, 'npxG': 1.0, 'xAOP': 1.0, 'Progressive runs': 1.0, 'Dribbles successful': 1.0, 'Aerial duels successful': 4, 'Defensive duels successful': 6, 'Interceptions': 6, 'Recoveries': 6},
    'Ball-Winning Mid': {'Passes': 4, 'Passes successful %': 6, 'Progressive Passes': 2, 'xT': 2.0, 'npxG': 1.0, 'xAOP': 1.0, 'Progressive runs': 1.0, 'Aerial duels': 4, 'Aerial duels successful %': 6, 'Defensive duels': 6, 'Defensive duels successful %': 10, 'Interceptions': 10, 'Recoveries': 10, 'Recoveries Opp Half': 4},
    'Deep-lying Playmaker': {'Passes': 10, 'Passes successful %': 6, 'Progressive Passes': 10, 'Passes to final third successful': 8, 'xT': 10,  'npxG': 1.0, 'xAOP': 8, 'Progressive runs': 2, 'Dribbles successful': 1.0, 'Aerial duels successful': 1.0, 'Defensive duels successful': 4, 'Interceptions': 4, 'Recoveries': 6},
    'Advanced Playmaker': {'Passes': 6, 'Passes successful %': 2, 'Progressive Passes': 4, 'xT': 8, 'Goals': 2, 'npxG': 8, 'Shots': 2, 'xG per Shot': 2, 'Assists': 2, 'xAOP': 8, 'Progressive runs': 2, 'Dribbles successful': 2, 'Aerial duels successful': 1.0, 'Defensive duels successful': 1.0, 'Interceptions': 1.0, 'Recoveries': 1.0, 'Counterpressing Recoveries': 1},
    'Full Back': {'npxG': 4, 'xAOP': 4, 'xT': 3, 'Passes': 2, 'Passes successful %': 2, 'Progressive Passes': 2, 'Progressive runs': 2, 'Aerial duels': 2, 'Aerial duels successful %': 8, 'Defensive duels': 4, 'Defensive duels successful %': 10, 'Interceptions': 8, 'Recoveries': 8, 'Clearances': 2},
    'Wingback': {'Goals': 2, 'npxG': 4, 'Shots': 2, 'xG per Shot': 1, 'Assists': 6, 'xAOP': 8, 'xT': 6, 'Passes': 1.0, 'Passes successful %': 1.0, 'Progressive Passes': 2, 'Crosses successful': 2, 'Progressive runs': 3, 'Aerial duels': 1.0, 'Aerial duels successful %': 1.0, 'Defensive duels': 1.0, 'Defensive duels successful %': 4, 'Interceptions': 4, 'Recoveries': 4, 'Clearances': 1.0},
    'Inverted Full Back': {'npxG': 1.0, 'xAOP': 1.0, 'xT': 12, 'Passes': 16, 'Passes successful %': 6, 'Progressive Passes': 8, 'Progressive runs': 2, 'Aerial duels': 1.0, 'Aerial duels successful %': 4, 'Defensive duels': 4, 'Defensive duels successful %': 6, 'Interceptions': 6, 'Recoveries': 4, 'Clearances': 1.0},
    'Wide Winger': {'Goals': 4, 'npxG': 8, 'Shots': 2, 'xG per Shot': 2, 'Assists': 4, 'xAOP': 8, 'xT': 6, 'Passes': 1.0, 'Passes successful %': 1.0, 'Progressive Passes': 1.0, 'Deep Completions': 2, 'Crosses successful': 2, 'Progressive runs': 2, 'Dribbles': 4, 'Dribbles successful %': 2, 'Loss index': 5, 'Aerial duels successful': 1.0, 'Defensive duels successful': 1.0, 'Interceptions': 1.0, 'Recoveries': 1.0, 'Counterpressing Recoveries': 1},
    'Creative Winger': {'Goals': 4, 'npxG': 8, 'Shots': 2, 'xG per Shot': 2, 'Assists': 6, 'xAOP': 12, 'xT': 10, 'Passes': 2, 'Passes successful %': 1.0, 'Progressive Passes': 2, 'Deep Completions': 3, 'Crosses successful': 2, 'Progressive runs': 2, 'Dribbles': 2, 'Dribbles successful %': 4, 'Loss index': 5, 'Aerial duels successful': 1.0, 'Defensive duels successful': 1.0, 'Interceptions': 1.0, 'Recoveries': 1.0, 'Counterpressing Recoveries': 1},
    'Inside Forward': {'Goals': 15, 'npxG': 30, 'Shots': 6, 'xG per Shot': 8, 'Assists': 10, 'xAOP': 20, 'xT': 2, 'Passes': 1.0, 'Passes successful %': 1.0, 'Progressive runs': 2, 'Dribbles': 4, 'Dribbles successful %': 4, 'Loss index': 5, 'Aerial duels successful': 4, 'Defensive duels successful': 1.0, 'Interceptions': 1.0, 'Recoveries': 1.0, 'Counterpressing Recoveries': 1},
    'Shadow Striker': {'Goals': 15, 'npxG': 30, 'Shots': 10, 'xG per Shot': 8, 'Assists': 10, 'xAOP': 20, 'xT': 4, 'Passes': 2, 'Passes successful %': 2, 'Progressive Passes': 3, 'Deep Completions': 3, 'Progressive runs': 2, 'Dribbles': 4, 'Dribbles successful %': 4, 'Loss index': 5, 'Aerial duels successful': 2, 'Defensive duels successful': 1.0, 'Interceptions': 1.0, 'Recoveries': 3},
    'Mobile Striker': {'Goals': 15, 'npxG': 30, 'Shots': 10, 'xG per Shot': 8, 'Assists': 10, 'xAOP': 20, 'xT': 4, 'Passes': 2, 'Passes successful %': 1.0, 'Progressive Passes': 1.0, 'Deep Completions': 1.0, 'Progressive runs': 8, 'Dribbles': 8, 'Dribbles successful %': 6, 'Loss index': 5, 'Aerial duels': 1.0, 'Aerial duels successful %': 1.0, 'Defensive duels successful': 1.0, 'Interceptions': 1.0, 'Recoveries': 6},
    'Poacher': {'Goals': 20, 'npxG': 40, 'Shots': 10, 'xG per Shot': 10, 'Assists': 10, 'xAOP': 20, 'Passes': 1.0, 'Passes successful %': 1.0, 'Progressive Passes': 1.0, 'Deep Completions': 1.0, 'Progressive runs': 1.0, 'Dribbles successful': 1.0, 'Loss index': 5, 'Aerial duels': 5, 'Aerial duels successful %': 5, 'Defensive duels successful': 1.0, 'Interceptions': 1.0, 'Recoveries': 1.0},
    'Target Man': {'Goals': 15, 'npxG': 30, 'Shots': 10, 'xG per Shot': 8, 'Assists': 10, 'xAOP': 20, 'xT': 2, 'Passes': 2, 'Passes successful %': 2, 'Progressive Passes': 1.0, 'Deep Completions': 1.0, 'Progressive runs': 1.0, 'Dribbles': 1.0, 'Dribbles successful %': 1.0, 'Loss index': 5, 'Aerial duels': 10, 'Aerial duels successful %': 10, 'Defensive duels successful': 1.0, 'Interceptions': 1.0, 'Recoveries': 1.0, 'Clearances': 10},
    'Pressing Forward': {'Goals': 15, 'npxG': 30, 'Shots': 10, 'xG per Shot': 8, 'Assists': 10, 'xAOP': 20, 'xT': 2, 'Passes': 2, 'Passes successful %': 1.0, 'Progressive Passes': 1.0, 'Deep Completions': 1.0, 'Progressive runs': 2, 'Dribbles': 2, 'Dribbles successful %': 2, 'Loss index': 5, 'Aerial duels': 1.0, 'Aerial duels successful %': 1.0, 'Defensive duels successful': 1.0, 'Interceptions': 8, 'Recoveries': 10, 'Counterpressing Recoveries': 4}
}
INVERT_METRICS = ['Loss index', 'goalsConceded']
OUTPUT_METRICS = ['Goals', 'Assists', 'xG', 'npxG', 'xA', 'xAOP', 'xASP', 'xT', 'Second assists', 'Shots', 'xG per Shot']
PASSING_METRICS = ['Passes', 'Passes successful', 'Passes successful %', 'Long passes', 'Long passes successful', 'Long passes successful %', 'Crosses', 'Crosses successful', 'Crosses successful %', 'Through passes', 'Through passes successful', 'Progressive Passes', 'Passes to final third', 'Passes to final third successful', 'Forward passes', 'Forward passes successful', 'Back passes', 'Back passes successful', 'Passes to penalty area', 'Passes to penalty area successful', 'Deep Completions']
DEFENSIVE_METRICS = ['Interceptions', 'Aerial duels', 'Aerial duels successful', 'Aerial duels successful %', 'Sliding tackles', 'Sliding tackles successful', 'Sliding tackles successful %', 'Recoveries', 'Recoveries Opp Half', 'Counterpressing Recoveries', 'Defensive duels', 'Defensive duels successful', 'Defensive duels successful %', 'Clearances', 'Fouls', 'Yellow cards', 'Red cards']
DRIBBLING_METRICS = ['Dribbles', 'Dribbles successful', 'Dribbles successful %', 'Touches in penalty area', 'Progressive runs', 'Fouls suffered']
GOALKEEPING_METRICS = ['shotsOnTargetAgainst', 'goalsConceded', 'exits', 'saves', 'goalsPrevented', 'goalsPreventedPerSOT', 'savePercentage', 'recoveries_gk', 'passes_gk', 'passesSuccessful_gk', 'Long passes successful %', 'longPasses_gk', 'longPassesSuccessful_gk']
DISTRIBUTION_METRICS_BY_POSITION = {
    'Shot Stopper': ['goalsPrevented', 'goalsPreventedPerSOT', 'exits', 'Long passes successful %', 'recoveries_gk'],
    'Cross Claimer': ['goalsPrevented', 'goalsPreventedPerSOT', 'exits', 'Long passes successful %', 'recoveries_gk'],
    'Ball-playing GK': ['goalsPrevented', 'goalsPreventedPerSOT', 'exits', 'recoveries_gk', 'passes_gk', 'Passes successful %', 'longPassesSuccessful_gk'],
    'Ball-Playing Centerback': ['xT', 'Passes', 'Passes successful %', 'Progressive Passes', 'Progressive runs'],
    'Stopper': ['Aerial duels', 'Aerial duels successful %', 'Defensive duels', 'Defensive duels successful %','Interceptions', 'Recoveries', 'Clearances'],
    'Athletic Centerback': ['npxG', 'Progressive runs', 'Aerial duels', 'Aerial duels successful %', 'Defensive duels', 'Defensive duels successful %','Interceptions', 'Recoveries', 'Clearances'],
    'Box-to-Box': ['Progressive Passes', 'npxG', 'Shots', 'xAOP', 'xT', 'Progressive runs', 'Dribbles successful', 'Aerial duels successful',  'Defensive duels successful', 'Interceptions', 'Recoveries'],
    'Holding Mid':['Passes', 'Passes successful %',  'Progressive Passes', 'xT', 'Aerial duels successful',  'Defensive duels successful', 'Interceptions', 'Recoveries'],
    'Ball-Winning Mid': ['Aerial duels', 'Aerial duels successful %',  'Defensive duels', 'Defensive duels successful %', 'Interceptions', 'Recoveries', 'Recoveries Opp Half'],
    'Deep-lying Playmaker': ['Passes', 'Passes successful %',  'Progressive Passes', 'xT','xAOP', 'Progressive runs'],
    'Advanced Playmaker': ['Goals', 'npxG', 'Shots', 'xG per Shot', 'Assists', 'xAOP', 'xT', 'Progressive runs', 'Dribbles successful'],
    'Full Back': ['Aerial duels', 'Aerial duels successful %', 'Defensive duels', 'Defensive duels successful %','Interceptions', 'Recoveries', 'Clearances'],
    'Wingback': ['Assists', 'xAOP', 'xT', 'Passes', 'Crosses successful', 'Progressive runs','Interceptions', 'Recoveries'],
    'Inverted Full Back': ['Progressive Passes', 'xT', 'Progressive runs', 'Defensive duels', 'Defensive duels successful %','Interceptions', 'Recoveries'],
    'Wide Winger': ['Goals', 'npxG', 'Shots', 'xG per Shot',  'Assists', 'xAOP', 'xT', 'Crosses successful', 'Progressive runs', 'Dribbles', 'Dribbles successful %',  'Loss index'],
    'Creative Winger': ['Goals', 'npxG', 'Shots', 'xG per Shot',  'Assists', 'xAOP', 'xT','Progressive runs', 'Dribbles', 'Dribbles successful %',  'Loss index'],
    'Inside Forward': ['Goals', 'npxG', 'Shots', 'xG per Shot',  'Assists', 'xAOP','Loss index'],
    'Shadow Striker': ['Goals', 'npxG', 'Shots', 'xG per Shot',  'Assists', 'xAOP', 'xT', 'Progressive runs', 'Dribbles', 'Dribbles successful %',  'Loss index'],
    'Mobile Striker': ['Goals', 'npxG', 'Shots', 'xG per Shot',  'Assists', 'xAOP', 'xT', 'Progressive runs', 'Dribbles', 'Dribbles successful %',  'Loss index'],
    'Poacher': ['Goals', 'npxG', 'Shots', 'xG per Shot',  'Assists', 'xAOP', 'Loss index'],
    'Target Man': ['Goals', 'npxG', 'Shots', 'xG per Shot',  'Assists', 'xAOP', 'Loss index', 'Aerial duels', 'Aerial duels successful %','Clearances'],
    'Pressing Forward': ['Goals', 'npxG', 'Shots', 'xG per Shot',  'Assists', 'xAOP', 'Loss index', 'Defensive duels successful', 'Interceptions', 'Recoveries', 'Counterpressing Recoveries']
}


# ==============================================================================
# 4. HELPER & PLOTTING FUNCTIONS
# ==============================================================================

# --- Helper for Player Radars (from Cell 11) ---
def calculate_and_merge(base_df, events_df, stat_name, primary_type=None, bool_condition=None):
    """
    Helper function from notebook Cell 11.
    Calculates a stat based on a filter and merges it into the base DataFrame.
    (FIXED to accept primary_type and bool_condition)
    """
    # Start with a base condition (all True)
    filter_condition = pd.Series(True, index=events_df.index)
    
    if primary_type:
        filter_condition &= (events_df.get('type.primary') == primary_type)
    
    if bool_condition is not None:
        # Align indices before combining
        bool_condition_aligned = bool_condition.reindex(filter_condition.index, fill_value=False)
        filter_condition &= bool_condition_aligned
        
    # Ensure player.id is numeric for groupby
    events_df['player.id'] = pd.to_numeric(events_df['player.id'], errors='coerce')
    safe_condition = filter_condition & events_df['player.id'].notna()
    
    if safe_condition.empty or not safe_condition.any():
        stat_series = pd.Series(dtype='int').rename(stat_name)
    else:
        # Group by the integer version of player.id
        stat_series = events_df[safe_condition].groupby(events_df['player.id'].astype(int)).size()
        stat_series.name = stat_name
        
    base_df = base_df.merge(stat_series, left_index=True, right_index=True, how='left')
    return base_df

# --- NEW Helper for Robust List Checking ---
def calculate_and_merge_list(base_df, events_df, stat_name, tag_to_find, primary_type=None, and_condition=None):
    """
    Robust helper to count stats by checking for a tag in the 'type.secondary' list.
    """
    # Base condition: Check if the tag is in the list (if the list exists)
    condition = events_df.get('type.secondary', pd.Series(dtype='object')).apply(
        lambda x: isinstance(x, (list, np.ndarray)) and tag_to_find in x
    )
    
    if primary_type:
        condition &= (events_df.get('type.primary') == primary_type)
        
    if and_condition is not None:
        # Align indices before combining conditions
        and_condition_aligned = and_condition.reindex(condition.index, fill_value=False)
        condition = condition & and_condition_aligned
        
    # --- THIS IS THE FIX ---
    # We must pass 'primary_type' and the 'condition' as a 'bool_condition'
    # to the main helper function.
    return calculate_and_merge(
        base_df, 
        events_df, 
        stat_name, 
        primary_type=primary_type, # <-- Pass the primary_type
        bool_condition=condition   # <-- Pass the condition as a bool_condition
    )

@st.cache_data
def calculate_all_player_stats(_raw_events_df, _player_minutes_df):
    """
    A new, streamlined, and correct function to calculate all player stats 
    for the player profile page (Per 90 and Totals).
    """
    print("--- STARTING: New All-Player-Stats Calculation ---")
    
    events_df = _raw_events_df.copy()
    base_df = _player_minutes_df.copy().set_index('playerId') # Use playerId as index
    
    # Ensure player.id is int for all merging
    events_df['player.id'] = pd.to_numeric(events_df['player.id'], errors='coerce')
    events_df = events_df.dropna(subset=['player.id'])
    events_df['player.id'] = events_df['player.id'].astype(int)

    # --- Helper Function for Aggregation ---
    def count_and_merge(base_df, events_df, stat_name, filter_condition):
        """Groups events by player.id, counts them, and merges into base_df."""
        # Align index if bool_condition is passed
        if isinstance(filter_condition, pd.Series):
             filter_condition = filter_condition.reindex(events_df.index, fill_value=False)
        
        safe_condition = filter_condition & events_df['player.id'].notna()
        
        if safe_condition.empty or not safe_condition.any():
            stat_series = pd.Series(dtype='int', name=stat_name)
        else:
            stat_series = events_df[safe_condition].groupby(events_df['player.id']).size()
            stat_series.name = stat_name
            
        base_df = base_df.merge(stat_series, left_index=True, right_index=True, how='left')
        return base_df

    # --- Helper for list-checking ---
    def check_secondary_list(tag):
        """Returns a boolean Series if tag is in the 'type.secondary' list."""
        return events_df.get('type.secondary', pd.Series(dtype='object')).apply(
            lambda x: isinstance(x, (list, np.ndarray)) and tag in x
        )

    # --- Step 1: Calculate All Counting Stats (Totals) ---
    print("Step 1: Calculating counting stats...")
    
    # -- Basic Event Counts --
    base_df = count_and_merge(base_df, events_df, 'Goals', events_df.get('shot.isGoal') == True)
    base_df = count_and_merge(base_df, events_df, 'Assists', check_secondary_list('assist'))
    base_df = count_and_merge(base_df, events_df, 'Shots', events_df['type.primary'] == 'shot')
    base_df = count_and_merge(base_df, events_df, 'Shots on target', (events_df['type.primary'] == 'shot') & (events_df['shot.onTarget'] == True))
    base_df = count_and_merge(base_df, events_df, 'Interceptions', events_df['type.primary'] == 'interception')
    base_df = count_and_merge(base_df, events_df, 'Clearances', events_df['type.primary'] == 'clearance')
    base_df = count_and_merge(base_df, events_df, 'Fouls', events_df['type.primary'] == 'infraction')
    base_df = count_and_merge(base_df, events_df, 'Offsides', events_df['type.primary'] == 'offside')
    base_df = count_and_merge(base_df, events_df, 'Yellow cards', events_df.get('infraction.yellowCard') == True)
    base_df = count_and_merge(base_df, events_df, 'Red cards', events_df.get('infraction.redCard') == True)
    base_df = count_and_merge(base_df, events_df, 'Touches in penalty area', check_secondary_list('touch_in_box'))
    base_df = count_and_merge(base_df, events_df, 'Progressive runs', check_secondary_list('progressive_run'))
    base_df = count_and_merge(base_df, events_df, 'Fouls suffered', check_secondary_list('foul_suffered'))
    base_df = count_and_merge(base_df, events_df, 'Second assists', check_secondary_list('second_assist'))

    # -- Passing Metrics --
    pass_events = events_df[events_df['type.primary'] == 'pass']
    pass_accurate = pass_events.get('pass.accurate') == True
    base_df = count_and_merge(base_df, pass_events, 'Passes', pd.Series(True, index=pass_events.index))
    base_df = count_and_merge(base_df, pass_events, 'Passes successful', pass_accurate)
    base_df = count_and_merge(base_df, pass_events, 'Long passes', check_secondary_list('long_pass'))
    base_df = count_and_merge(base_df, pass_events, 'Long passes successful', check_secondary_list('long_pass') & pass_accurate)
    base_df = count_and_merge(base_df, pass_events, 'Crosses', check_secondary_list('cross'))
    base_df = count_and_merge(base_df, pass_events, 'Crosses successful', check_secondary_list('cross') & pass_accurate)
    base_df = count_and_merge(base_df, pass_events, 'Through passes', check_secondary_list('through_pass'))
    base_df = count_and_merge(base_df, pass_events, 'Through passes successful', check_secondary_list('through_pass') & pass_accurate)
    base_df = count_and_merge(base_df, pass_events, 'Passes to final third', check_secondary_list('pass_to_final_third'))
    base_df = count_and_merge(base_df, pass_events, 'Passes to final third successful', check_secondary_list('pass_to_final_third') & pass_accurate)
    base_df = count_and_merge(base_df, pass_events, 'Forward passes', check_secondary_list('forward_pass'))
    base_df = count_and_merge(base_df, pass_events, 'Forward passes successful', check_secondary_list('forward_pass') & pass_accurate)
    base_df = count_and_merge(base_df, pass_events, 'Back passes', check_secondary_list('back_pass'))
    base_df = count_and_merge(base_df, pass_events, 'Back passes successful', check_secondary_list('back_pass') & pass_accurate)
    base_df = count_and_merge(base_df, pass_events, 'Passes to penalty area', check_secondary_list('pass_to_penalty_area'))
    base_df = count_and_merge(base_df, pass_events, 'Passes to penalty area successful', check_secondary_list('pass_to_penalty_area') & pass_accurate)

    # -- Dueling & Defensive Metrics --
    duel_events = events_df[events_df['type.primary'] == 'duel']
    base_df = count_and_merge(base_df, duel_events, 'Duels', pd.Series(True, index=duel_events.index))
    base_df = count_and_merge(base_df, duel_events, 'Duels successful', (duel_events.get('groundDuel.keptPossession') == True) | (duel_events.get('groundDuel.recoveredPossession') == True) | (duel_events.get('aerialDuel.firstTouch') == True))
    base_df = count_and_merge(base_df, duel_events, 'Aerial duels', check_secondary_list('aerial_duel'))
    base_df = count_and_merge(base_df, duel_events, 'Aerial duels successful', check_secondary_list('aerial_duel') & (duel_events.get('aerialDuel.firstTouch') == True))
    base_df = count_and_merge(base_df, duel_events, 'Defensive duels', check_secondary_list('defensive_duel'))
    base_df = count_and_merge(base_df, duel_events, 'Defensive duels successful', check_secondary_list('defensive_duel') & (duel_events.get('groundDuel.recoveredPossession') == True))
    base_df = count_and_merge(base_df, duel_events, 'Offensive duels', check_secondary_list('offensive_duel'))
    base_df = count_and_merge(base_df, duel_events, 'Offensive duels successful', check_secondary_list('offensive_duel') & (duel_events.get('groundDuel.progressedWithBall') == True))
    base_df = count_and_merge(base_df, duel_events, 'Sliding tackles', check_secondary_list('sliding_tackle'))
    base_df = count_and_merge(base_df, duel_events, 'Sliding tackles successful', check_secondary_list('sliding_tackle') & (duel_events.get('groundDuel.recoveredPossession') == True))
    base_df = count_and_merge(base_df, duel_events, 'Dribbles', check_secondary_list('dribble'))
    base_df = count_and_merge(base_df, duel_events, 'Dribbles successful', check_secondary_list('dribble') & (duel_events.get('groundDuel.takeOn') == True))

    # -- Losses & Recoveries --
    base_df = count_and_merge(base_df, events_df, 'Losses', check_secondary_list('loss'))
    base_df = count_and_merge(base_df, events_df, 'Losses Opp Half', check_secondary_list('loss') & (events_df.get('location.x', 0) >= 50))
    base_df = count_and_merge(base_df, events_df, 'Recoveries', check_secondary_list('recovery'))
    base_df = count_and_merge(base_df, events_df, 'Recoveries Opp Half', check_secondary_list('recovery') & (events_df.get('location.x', 0) >= 50))
    base_df = count_and_merge(base_df, events_df, 'Counterpressing Recoveries', check_secondary_list('counterpressing_recovery'))

    # --- Step 2: Calculate xG, xA, xT, and special passing ---
    print("Step 2: Calculating xG, xA, xT...")
    # This is the same logic from your (working) `calculate_player_profile_stats`
    # -- xG --
    xg_series = events_df.groupby('player.id')['shot.xg'].sum()
    xg_series.name = 'xG'
    base_df = base_df.merge(xg_series, left_index=True, right_index=True, how='left')

    # -- npxG, xAOP, xASP --
    shots_df = events_df[(events_df['shot.xg'].notna()) & (events_df['type.primary'] != 'penalty')].copy()
    npxg_totals = shots_df.groupby('player.id')['shot.xg'].sum().reset_index().rename(columns={'shot.xg': 'npxG'})
    events_df['shot_event_id'] = np.where(events_df['shot.xg'].notna(), events_df['id'], np.nan)
    events_df['next_shot_id'] = events_df.groupby('matchId')['shot_event_id'].bfill()
    shot_xg_map = events_df[events_df['shot.xg'].notna()].set_index('id')['shot.xg'].to_dict()
    assists_df = events_df[check_secondary_list('shot_assist')].copy()
    assists_df['xA'] = assists_df['next_shot_id'].map(shot_xg_map)
    set_piece_types = ['corner', 'free_kick', 'throw_in', 'goal_kick']
    assists_df['assist_type'] = np.where(assists_df['type.primary'].isin(set_piece_types), 'xASP', 'xAOP')
    xa_split_totals = assists_df.groupby(['player.id', 'assist_type'])['xA'].sum()
    xa_final_df = xa_split_totals.unstack(fill_value=0).reset_index()
    final_stats_df = pd.merge(npxg_totals, xa_final_df, on='player.id', how='outer')
    base_df = base_df.merge(final_stats_df.set_index('player.id'), left_index=True, right_index=True, how='left')


    # -- Deep Completions and Progressive Passes --
    passes_df = events_df[(events_df['type.primary'] == 'pass') & (events_df.get('pass.accurate') == True)].dropna(subset=['location.x', 'pass.endLocation.x', 'player.id']).copy()
    passes_df['end_x_m'] = passes_df['pass.endLocation.x'] * 1.05
    passes_df['end_y_m'] = passes_df['pass.endLocation.y'] * 0.68
    passes_df['dist_to_goal_center'] = np.sqrt((passes_df['end_x_m'] - 105)**2 + (passes_df['end_y_m'] - 34)**2)
    passes_df['is_cross'] = passes_df.get('type.secondary', pd.Series(dtype='object')).apply(lambda x: isinstance(x, (list, np.ndarray)) and 'cross' in x)
    passes_df['is_deep_completion'] = (passes_df['dist_to_goal_center'] <= 20) & (passes_df['is_cross'] == False)
    deep_completions = passes_df.groupby('player.id')['is_deep_completion'].sum().reset_index().rename(columns={'is_deep_completion': 'Deep Completions'})
    start_x = passes_df['location.x']; end_x = passes_df['pass.endLocation.x']
    cond1 = (start_x < 50) & (end_x < 50) & (end_x - start_x >= 30)
    cond2 = (start_x < 50) & (end_x >= 50) & (end_x - start_x >= 15)
    cond3 = (start_x >= 50) & (end_x >= 50) & (end_x - start_x >= 10)
    passes_df['is_progressive_pass'] = cond1 | cond2 | cond3
    progressive_passes = passes_df.groupby('player.id')['is_progressive_pass'].sum().reset_index().rename(columns={'is_progressive_pass': 'Progressive Passes'})
    new_metrics_df = pd.merge(deep_completions, progressive_passes, on='player.id', how='outer')
    base_df = base_df.merge(new_metrics_df.set_index('player.id'), left_index=True, right_index=True, how='left')
    
    # -- xT --
    xt_data_from_image = [[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.03, 0.03, 0.04, 0.04], [0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.03, 0.04, 0.05, 0.05], [0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.03, 0.05, 0.06, 0.06], [0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.04, 0.11, 0.26, 0.26], [0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.04, 0.11, 0.26, 0.26], [0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.03, 0.05, 0.06, 0.06], [0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.03, 0.04, 0.05, 0.05], [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.03, 0.03, 0.04, 0.04]]
    xt_grid = np.array(xt_data_from_image); rows, cols = xt_grid.shape
    move_df = events_df[events_df['type.primary'].isin(['pass', 'touch', 'acceleration'])].copy()
    successful_pass = (move_df['type.primary'] == 'pass') & (move_df.get('pass.accurate') == True)
    other_successful_moves = move_df['type.primary'].isin(['touch', 'acceleration'])
    move_df = move_df[successful_pass | other_successful_moves]
    move_df['start_x'] = move_df['location.x']; move_df['start_y'] = move_df['location.y']
    move_df['end_x'] = np.where(move_df['type.primary'] == 'pass', move_df.get('pass.endLocation.x'), move_df.get('carry.endLocation.x'))
    move_df['end_y'] = np.where(move_df['type.primary'] == 'pass', move_df.get('pass.endLocation.y'), move_df.get('carry.endLocation.y'))
    move_df = move_df.dropna(subset=['end_x', 'end_y', 'player.id'])
    def get_xt_zone(x, y, xt_rows, xt_cols):
        if pd.isna(x) or pd.isna(y): return None, None
        col = min(int(x / 100 * xt_cols), xt_cols - 1); row = min(int(y / 100 * xt_rows), xt_rows - 1)
        return row, col
    move_df[['start_row', 'start_col']] = move_df.apply(lambda row: get_xt_zone(row['start_x'], row['start_y'], rows, cols), axis=1, result_type='expand')
    move_df[['end_row', 'end_col']] = move_df.apply(lambda row: get_xt_zone(row['end_x'], row['end_y'], rows, cols), axis=1, result_type='expand')
    move_df['xt_start'] = move_df.apply(lambda row: xt_grid[int(row['start_row']), int(row['start_col'])] if pd.notna(row['start_row']) else 0, axis=1)
    move_df['xt_end'] = move_df.apply(lambda row: xt_grid[int(row['end_row']), int(row['end_col'])] if pd.notna(row['end_row']) else 0, axis=1)
    move_df['xT'] = move_df['xt_end'] - move_df['xt_start']
    successful_threat = move_df[move_df['xT'] > 0]
    player_xt = successful_threat.groupby('player.id')['xT'].sum().reset_index()
    base_df = base_df.merge(player_xt.set_index('player.id'), left_index=True, right_index=True, how='left')

    # --- Step 3: Calculate Goalkeeper Stats ---
    print("Step 3: Calculating Goalkeeper stats...")
    gk_ids = events_df[events_df.get('player.position') == 'GK']['player.id'].dropna().unique().astype(int)
    gk_events_df = events_df[events_df['player.id'].isin(gk_ids)].copy()
    shots_faced_df = events_df[(events_df.get('type.primary') == 'shot') & (events_df.get('shot.onTarget') == True) & (events_df.get('shot.goalkeeper.id').notna())].copy()
    shots_faced_df['shot.goalkeeper.id'] = shots_faced_df['shot.goalkeeper.id'].astype(int)
    gk_shot_stopping_stats = shots_faced_df.groupby('shot.goalkeeper.id').agg(shotsOnTargetAgainst=('shot.isGoal', 'count'), goalsConceded=('shot.isGoal', 'sum'), psxG_faced=('shot.postShotXg', 'sum')).reset_index().rename(columns={'shot.goalkeeper.id': 'player.id'})
    if not gk_shot_stopping_stats.empty:
        gk_shot_stopping_stats['goalsPrevented'] = gk_shot_stopping_stats['psxG_faced'] - gk_shot_stopping_stats['goalsConceded']
        gk_shot_stopping_stats['goalsPreventedPerSOT'] = (gk_shot_stopping_stats['goalsPrevented'] / gk_shot_stopping_stats['shotsOnTargetAgainst']).fillna(0)
        gk_shot_stopping_stats['savePercentage'] = ((gk_shot_stopping_stats['shotsOnTargetAgainst'] - gk_shot_stopping_stats['goalsConceded']) / gk_shot_stopping_stats['shotsOnTargetAgainst'] * 100).fillna(0)
    else:
        gk_shot_stopping_stats = gk_shot_stopping_stats.reindex(columns=['player.id', 'shotsOnTargetAgainst', 'goalsConceded', 'psxG_faced', 'goalsPrevented', 'goalsPreventedPerSOT', 'savePercentage']).fillna(0)
    exits = gk_events_df[gk_events_df['type.primary'] == 'goalkeeper_exit'].groupby('player.id').size().reset_index(name='exits')
    recoveries_gk = gk_events_df[check_secondary_list('recovery')].groupby('player.id').size().reset_index(name='recoveries_gk')
    gk_passes = gk_events_df[gk_events_df['type.primary'] == 'pass']
    passes_total_gk = gk_passes.groupby('player.id').size().reset_index(name='passes_gk')
    passes_succ_gk = gk_passes[gk_passes['pass.accurate'] == True].groupby('player.id').size().reset_index(name='passesSuccessful_gk')
    long_passes_total_gk = gk_passes[check_secondary_list('long_pass')].groupby('player.id').size().reset_index(name='longPasses_gk')
    long_passes_succ_gk = gk_passes[check_secondary_list('long_pass') & (gk_passes['pass.accurate'] == True)].groupby('player.id').size().reset_index(name='longPassesSuccessful_gk')
    gk_report_df = pd.DataFrame({'player.id': gk_ids}); gk_report_df = pd.merge(gk_report_df, gk_shot_stopping_stats, on='player.id', how='left'); gk_report_df = pd.merge(gk_report_df, exits, on='player.id', how='left'); gk_report_df = pd.merge(gk_report_df, recoveries_gk, on='player.id', how='left'); gk_report_df = pd.merge(gk_report_df, passes_total_gk, on='player.id', how='left'); gk_report_df = pd.merge(gk_report_df, passes_succ_gk, on='player.id', how='left'); gk_report_df = pd.merge(gk_report_df, long_passes_total_gk, on='player.id', how='left'); gk_report_df = pd.merge(gk_report_df, long_passes_succ_gk, on='player.id', how='left')
    base_df = base_df.merge(gk_report_df.set_index('player.id'), left_index=True, right_index=True, how='left')

    # --- Step 4: Finalize, Calculate Percentages, and Normalize ---
    print("Step 4: Finalizing and normalizing...")
    base_df = base_df.fillna(0)
    
    # Calculate % stats from the totals
    def safe_divide_perc(n, d): return (base_df[n] / base_df[d] * 100).replace([np.inf, -np.inf], 0).fillna(0)
    base_df['xG per Shot'] = (base_df['xG'] / base_df['Shots']).replace([np.inf, -np.inf], 0).fillna(0)
    base_df['Passes successful %'] = safe_divide_perc('Passes successful', 'Passes')
    base_df['Long passes successful %'] = safe_divide_perc('Long passes successful', 'Long passes')
    base_df['Crosses successful %'] = safe_divide_perc('Crosses successful', 'Crosses')
    base_df['Dribbles successful %'] = safe_divide_perc('Dribbles successful', 'Dribbles')
    base_df['Duels successful %'] = safe_divide_perc('Duels successful', 'Duels')
    base_df['Aerial duels successful %'] = safe_divide_perc('Aerial duels successful', 'Aerial duels')
    base_df['Offensive duels successful %'] = safe_divide_perc('Offensive duels successful', 'Offensive duels')
    base_df['Defensive duels successful %'] = safe_divide_perc('Defensive duels successful', 'Defensive duels')
    base_df['Sliding tackles successful %'] = safe_divide_perc('Sliding tackles successful', 'Sliding tackles')
    successful_attacking_actions = base_df['Shots on target'] + base_df['Crosses successful'] + base_df['Dribbles successful']
    base_df['Loss index'] = (base_df['Losses'] / successful_attacking_actions).replace([np.inf, -np.inf], 0).fillna(0)
    
    # GK % stats
    base_df['Passes successful %_gk'] = safe_divide_perc('passesSuccessful_gk', 'passes_gk')
    base_df['Long passes successful %_gk'] = safe_divide_perc('longPassesSuccessful_gk', 'longPasses_gk')
    # Rename for consistency
    base_df = base_df.rename(columns={
        'Passes successful %_gk': 'GK Passes successful %',
        'Long passes successful %_gk': 'GK Long passes successful %'
    })

    # Get a list of ALL calculated metrics
    all_calculated_metrics = list(base_df.columns)
    
    # Normalize to Per 90
    total_minutes = base_df['totalMinutes']
    minutes_gt_0 = total_minutes > 0
    
    # Define cols that should NOT be normalized
    rate_cols = [col for col in base_df.columns if '%' in col or 'per' in col or 'index' in col or 'Percentage' in col]
    info_cols = ['playerName', 'teamName', 'totalMinutes', 'primaryPosition', 'secondaryPosition', 'tertiaryPosition', 'player.id', 'player.id_x', 'player.id_y']
    dont_normalize = rate_cols + info_cols

    for col in all_calculated_metrics:
        if col not in dont_normalize and pd.api.types.is_numeric_dtype(base_df[col]):
            base_df[col] = np.where(
                minutes_gt_0,
                (base_df[col].astype(float) / total_minutes) * 90,
                0
            )
            
    # Clean up and return
    base_df = base_df.reset_index() # 'playerId' is now a column
    # Drop all the junk 'player.id' columns
    cols_to_drop = [col for col in base_df.columns if 'player.id' in str(col)]
    if 'playerId' in base_df.columns and 'player.id' in cols_to_drop:
        cols_to_drop.remove('player.id') # Keep the real one
        
    base_df = base_df.drop(columns=cols_to_drop, errors='ignore')
    
    print("--- FINISHED: New All-Player-Stats Calculation ---")
    return base_df.fillna(0)

@st.cache_data
def calculate_player_percentiles_and_scores(_player_data_df, _position_groups, _weights, _invert_metrics, min_minutes=90):
    """Calculates percentiles and scores for all players based on position."""
    print("Calculating player percentiles and scores...")
    data = _player_data_df.copy()
    
    data['totalMinutes'] = pd.to_numeric(data['totalMinutes'], errors='coerce')
    data = data[data['totalMinutes'] >= min_minutes]
    if data.empty:
        print(f"Warning: No players found with >= {min_minutes} minutes.")
        return pd.DataFrame()

    # Calculate percentiles
    for position, group in _position_groups.items():
        metrics = list(_weights[position].keys())
        position_data_mask = data['primaryPosition'].isin(group)
        position_data_indices = data[position_data_mask].index
        
        if position_data_indices.empty: continue

        for metric in metrics:
            if metric in data.columns:
                data[metric] = pd.to_numeric(data[metric], errors='coerce').fillna(0)
                percentiles = data.loc[position_data_indices, metric].rank(pct=True)
                if metric in _invert_metrics:
                    percentiles = 1 - (percentiles.fillna(0.5))
                
                data.loc[position_data_indices, metric + '_percentile'] = percentiles
            
    # Calculate Scores
    for position, group in _position_groups.items():
        metrics = list(_weights[position].keys())
        position_data_mask = data['primaryPosition'].isin(group)
        position_data_indices = data[position_data_mask].index
        if position_data_indices.empty: continue

        total_score = pd.Series(0.0, index=position_data_indices, dtype='float64')
        for metric in metrics:
            percentile_col = metric + '_percentile'
            if percentile_col in data.columns:
                weight = _weights[position].get(metric, 0)
                total_score = total_score.add(data.loc[position_data_indices, percentile_col].fillna(0) * weight, fill_value=0)
        
        data.loc[position_data_indices, position + '_TotalScore'] = total_score
        
        min_score = total_score.min()
        max_score = total_score.max()
        if (max_score - min_score) != 0:
            data.loc[position_data_indices, position + '_Score'] = (total_score - min_score) / (max_score - min_score) * 100
        else:
            data.loc[position_data_indices, position + '_Score'] = 0.0

    print("✅ Player percentiles and scores calculated.")
    return data.fillna(0)


def _create_base_radar_chart(ax, player_data, metrics, position, eligible_groups, full_df_for_ranking=None):
    """Helper function to create the base radar chart."""
    
    num_metrics = len(metrics)
    angles = [n / float(num_metrics) * 2 * pi for n in range(num_metrics)]
    angles += angles[:1]
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])  

    values = []
    for metric in metrics:
        col = metric + '_percentile'
        if col in player_data.columns:
            # Explicitly convert to float and handle NaNs
            val = float(player_data[col].values[0])
            # --- THIS IS THE FIX: Scale 0-1 value to 0-100 ---
            values.append(np.nan_to_num(val, nan=0.0) * 100)
        else:
            values.append(0.0) # Default to 0 if metric is missing
            print(f"Warning: Missing percentile column {col} for radar.")
            
    values += values[:1] # Close the loop

    # --- UPDATED: Set zorder to force plot on top of grid ---
    ax.plot(angles, values, linewidth=2, linestyle='solid', color='#0077b6', zorder=3)  
    ax.fill(angles, values, '#0077b6', alpha=0.25, zorder=2)

    ax.set_rlabel_position(0)
    # --- UPDATED: Set grid zorder to 1 (behind the plot) ---
    plt.yticks([25, 50, 75, 100], ["25%", "50%", "75%", "100%"], color="grey", size=7, zorder=1)  
    plt.ylim(0, 100)

    category_colors = {'output': 'green', 'passing': 'orange', 'defensive': 'red', 'dribbling': 'purple', 'goalkeeping': 'cyan'}

    # Plot raw values
    for i, metric in enumerate(metrics):
        angle_rad = angles[i]
        label = f"{player_data[metric].values[0]:.2f}"
        ax.text(angle_rad, 85, label, size=8, ha='center', va='center', color='blue')

    # Plot metric names
    for i, metric in enumerate(metrics):
        angle_rad = angles[i]
        if metric in OUTPUT_METRICS: color = category_colors['output']
        elif metric in PASSING_METRICS: color = category_colors['passing']
        elif metric in DEFENSIVE_METRICS: color = category_colors['defensive']
        elif metric in DRIBBLING_METRICS: color = category_colors['dribbling']
        elif metric in GOALKEEPING_METRICS: color = category_colors['goalkeeping']
        else: color = 'grey'
        ax.text(angle_rad, 115, metric, size=8, ha='center', va='center', rotation=0, color=color, fontweight='bold')

    ax.set_rlabel_position(0)
    plt.yticks([25, 50, 75, 100], ["25%", "50%", "75%", "100%"], color="grey", size=7)  
    plt.ylim(0, 100)  

    player_name = player_data['playerName'].values[0]
    player_position = player_data['primaryPosition'].values[0]
    player_minutes = player_data['totalMinutes'].values[0]
    player_team = player_data['teamName'].values[0]
    
    ax.text(-0.1, 1.15, f"{player_name} | {player_team}", size=15, color='black', ha='left', va='top', transform=ax.transAxes, weight='bold')
    ax.text(-0.1, 1.11, f"{player_position} | {player_minutes:.0f} minutes played", horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, color='black', size=12)

    today = datetime.date.today()
    plt.figtext(0.90, 0.90, f'Stats are per 90 mins \n25-26 \nLiga 3 \nData via Wyscout \n@lucaskimball\nDate: {today}', horizontalalignment='left', fontsize=10, color='black')
    legend_labels = ['Output Metrics', 'Passing Metrics', 'Defensive Metrics', 'Dribbling Metrics', 'Goalkeeping Metrics']
    legend_colors = [category_colors['output'], category_colors['passing'], category_colors['defensive'], category_colors['dribbling'], category_colors['goalkeeping']]
    patches = [plt.Line2D([0], [0], color=color, lw=4) for color in legend_colors]
    ax.legend(patches, legend_labels, loc='lower right', bbox_to_anchor=(1.7, 1), frameon=False)

    score_text = "\n"
    for group in eligible_groups:
        score_col = group + '_Score'
        rank_col = group + '_Rank'
        if score_col in player_data.columns:
            player_score = player_data[score_col].values[0]
            player_rank_str = ""
            try:
                if full_df_for_ranking is not None and not full_df_for_ranking.empty:
                    group_players = full_df_for_ranking[full_df_for_ranking['primaryPosition'].isin(POSITION_GROUPS[group])]
                    if score_col in group_players.columns:
                        group_players[rank_col] = group_players[score_col].rank(ascending=False, method='dense').astype(int)
                        if player_data.index[0] in group_players.index:
                            player_rank = group_players.loc[player_data.index[0], rank_col]
                            player_rank_str = f" (Rank: {player_rank})"
            except Exception as e:
                print(f"Warning: Could not calculate rank for {group}. Error: {e}")
            score_text += f"{group}: {player_score:.2f}{player_rank_str}\n"

    outside_background_color = (0.95, 0.92, 0.87); inside_radar_color = (0.99, 0.98, 0.95); score_box_color = (1.0, 0.99, 0.97)
    ax.set_facecolor(inside_radar_color)
    if ax.figure: ax.figure.patch.set_facecolor(outside_background_color)
    plt.figtext(.55, 1, score_text, horizontalalignment='left', verticalalignment='top', fontsize=12, bbox=dict(facecolor=score_box_color, alpha=0.5))


def get_percentile_suffix(value):
    """Function to add the appropriate suffix for the percentile."""
    value = int(value)
    if 10 <= value % 100 <= 20: suffix = 'th'
    else: suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(value % 10, 'th')
    return suffix

def create_radar_with_distributions(player_data, metrics, position, eligible_groups, all_position_data, full_df_for_ranking=None):
    """Creates the combined figure with radar and distribution plots."""
    
    player_name = player_data['playerName'].values[0]
    highest_scoring_group = None; highest_score = -1; scores_by_group = {}

    for group in eligible_groups:
        score_col = group + '_Score'
        if score_col in player_data.columns:
            player_score = player_data[score_col].values[0]
            scores_by_group[group] = player_score
            if player_score > highest_score:
                highest_score = player_score; highest_scoring_group = group

    if highest_scoring_group is None:
        print(f"No highest scoring group found for {player_name}. Using default.")
        highest_scoring_group = eligible_groups[0] if eligible_groups else "Default"  

    relevant_metrics = DISTRIBUTION_METRICS_BY_POSITION.get(highest_scoring_group, metrics)
    relevant_metrics = [m for m in relevant_metrics if m in player_data.columns]

    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(1, 2, width_ratios=[2.5, 1.2], figure=fig)
    ax_radar = plt.subplot(gs[0], polar=True)
    
    _create_base_radar_chart(ax_radar, player_data, metrics, position, eligible_groups, full_df_for_ranking=full_df_for_ranking)
    
    ax_radar.text(-0.1, 1.065, f"{highest_scoring_group} Template",
                  horizontalalignment='left', verticalalignment='center', transform=ax_radar.transAxes,
                  fontsize=14, fontweight='bold', color='black')

    # --- Distribution Plots ---
    primary_pos_group = POSITION_GROUPS.get(eligible_groups[0], [player_data['primaryPosition'].values[0]])
    relevant_players_data = all_position_data[all_position_data['primaryPosition'].isin(primary_pos_group)]
    
    if relevant_metrics and not relevant_players_data.empty:
        gs_distributions = GridSpec(len(relevant_metrics), 1, left=0.70, right=0.98, top=0.82, bottom=0.07, hspace=0.7, figure=fig)
        for i, metric in enumerate(relevant_metrics):
            ax_dist = plt.subplot(gs_distributions[i])
            if metric in OUTPUT_METRICS: color = 'green'
            elif metric in PASSING_METRICS: color = 'orange'
            elif metric in DEFENSIVE_METRICS: color = 'red'
            elif metric in DRIBBLING_METRICS: color = 'purple'
            elif metric in GOALKEEPING_METRICS: color = 'cyan'
            else: color = 'blue'
            
            valid_relevant_players = relevant_players_data[relevant_players_data[metric].notna()][metric]
            if len(valid_relevant_players) > 1: sns.kdeplot(valid_relevant_players, ax=ax_dist, fill=True, color=color, cut=0)
            elif len(valid_relevant_players) == 1: ax_dist.axvline(valid_relevant_players.iloc[0], color=color, linestyle='-')
            
            player_value = player_data[metric].values[0]
            
            percentile_rank = 0
            if len(valid_relevant_players) > 0: percentile_rank = scipy.stats.percentileofscore(valid_relevant_players, player_value, kind='strict')
            percentile_rank_int = int(percentile_rank); suffix = get_percentile_suffix(percentile_rank_int)
            min_value = valid_relevant_players.min(); max_value = valid_relevant_players.max()
            if pd.isna(min_value) or pd.isna(max_value) or min_value == max_value: min_value = player_value - 0.1; max_value = player_value + 0.1
            if min_value == max_value: max_value = min_value + 1.0 # Handle 0 case
            
            ax_dist.set_xlim(min_value, max_value); ax_dist.set_xticks([min_value, max_value]); ax_dist.set_xticklabels([f"{min_value:.2f}", f"{max_value:.2f}"], fontsize=8)
            ax_dist.axvline(player_value, color='blue', linestyle='--')
            raw_value = f"{player_value:.2f}"
            ax_dist.text(1.05, 0.5, f"%-tile: {percentile_rank_int}{suffix}\np/90 value: {raw_value}", transform=ax_dist.transAxes, fontsize=8, verticalalignment='center')
            ax_dist.set_yticks([]); ax_dist.set_ylabel(""); ax_dist.set_title(""); ax_dist.set_xlabel("");
            legend = ax_dist.get_legend();
            if legend is not None: legend.remove()
            ax_dist.text(-0.05, 0.5, metric, transform=ax_dist.transAxes, fontsize=9, fontweight='bold', va='center', ha='right')

    return fig

def plot_comparison_radar(ax, player_a_data, player_b_data, metrics, position_template):
    """Helper function to create a 2-player comparison radar chart."""
    
    num_metrics = len(metrics)
    angles = [n / float(num_metrics) * 2 * pi for n in range(num_metrics)]
    angles += angles[:1]
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])  

    # --- Player A Values ---
    values_a = []
    for metric in metrics:
        col = metric + '_percentile'
        val = float(player_a_data[col].values[0]) if col in player_a_data.columns else 0.0
        values_a.append(np.nan_to_num(val, nan=0.0) * 100)
    values_a += values_a[:1]

    # --- Player B Values ---
    values_b = []
    for metric in metrics:
        col = metric + '_percentile'
        val = float(player_b_data[col].values[0]) if col in player_b_data.columns else 0.0
        values_b.append(np.nan_to_num(val, nan=0.0) * 100)
    values_b += values_b[:1]

    # --- Plot Polygons ---
    color_a = '#0077b6' # Blue
    color_b = '#e63946' # Red
    
    player_a_name = player_a_data['playerName'].values[0]
    player_b_name = player_b_data['playerName'].values[0]

    ax.plot(angles, values_a, linewidth=2, linestyle='solid', color=color_a, zorder=3, label=player_a_name)  
    ax.fill(angles, values_a, color_a, alpha=0.2, zorder=2)
    
    ax.plot(angles, values_b, linewidth=2, linestyle='solid', color=color_b, zorder=3, label=player_b_name)  
    ax.fill(angles, values_b, color_b, alpha=0.2, zorder=2)

    # --- Plot Metric Names (No raw values for comparison) ---
    category_colors = {'output': 'green', 'passing': 'orange', 'defensive': 'red', 'dribbling': 'purple', 'goalkeeping': 'cyan'}
    for i, metric in enumerate(metrics):
        angle_rad = angles[i]
        if metric in OUTPUT_METRICS: color = category_colors['output']
        elif metric in PASSING_METRICS: color = category_colors['passing']
        elif metric in DEFENSIVE_METRICS: color = category_colors['defensive']
        elif metric in DRIBBLING_METRICS: color = category_colors['dribbling']
        elif metric in GOALKEEPING_METRICS: color = category_colors['goalkeeping']
        else: color = 'grey'
        ax.text(angle_rad, 115, metric, size=8, ha='center', va='center', rotation=0, color=color, fontweight='bold')

    ax.set_rlabel_position(0)
    plt.yticks([25, 50, 75, 100], ["25%", "50%", "75%", "100%"], color="grey", size=7, zorder=1)  
    plt.ylim(0, 100)  

    # --- Titles and Info ---
    player_a_team = player_a_data['teamName'].values[0]
    player_a_mins = player_a_data['totalMinutes'].values[0]
    
    player_b_team = player_b_data['teamName'].values[0]
    player_b_mins = player_b_data['totalMinutes'].values[0]
    
    ax.text(-0.1, 1.20, f"{player_a_name} | {player_a_team}", size=15, color=color_a, ha='left', va='top', transform=ax.transAxes, weight='bold')
    ax.text(-0.1, 1.16, f"{player_a_mins:.0f} minutes played", horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, color='black', size=12)
    
    ax.text(-0.1, 1.11, f"{player_b_name} | {player_b_team}", size=15, color=color_b, ha='left', va='top', transform=ax.transAxes, weight='bold')
    ax.text(-0.1, 1.07, f"{player_b_mins:.0f} minutes played", horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, color='black', size=12)

    ax.text(-0.1, 1.02, f"Comparison Template: {position_template}", horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, color='black', size=12, style='italic')

   # --- Legend ---
    # Get archetype scores for the legend
    score_col = position_template + '_Score'
    score_a = player_a_data[score_col].values[0] if score_col in player_a_data.columns else 0.0
    score_b = player_b_data[score_col].values[0] if score_col in player_b_data.columns else 0.0
    
    legend_elements = [
        plt.Line2D([0], [0], color=color_a, lw=4, label=f"{player_a_name} (Score: {score_a:.0f})"),
        plt.Line2D([0], [0], color=color_b, lw=4, label=f"{player_b_name} (Score: {score_b:.0f})")
    ]
    ax.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(1.5, 1), frameon=False)
    
    outside_background_color = (0.95, 0.92, 0.87); inside_radar_color = (0.99, 0.98, 0.95)
    ax.set_facecolor(inside_radar_color)
    if ax.figure: ax.figure.patch.set_facecolor(outside_background_color)

# --- Radar Stats Calculation ---
@st.cache_data
def calculate_all_team_radars_stats(season_events_df, matches_summary_df):
    """Calculates aggregated stats and percentiles for Offensive, Distribution, and Defensive radars."""
    
    print("Calculating team radar stats...") # Debug print
    all_teams_stats = {}
    
    # --- Data Prep ---
    # Ensure 'team.name' exists before using it
    if 'team.name' not in season_events_df.columns:
         print("Warning: 'team.name' column missing from events_df, cannot calculate radar stats.")
         return pd.DataFrame(), pd.DataFrame() # Return empty DataFrames

    teams = season_events_df['team.name'].unique()
    matches_played = season_events_df.groupby('team.name')['matchId'].nunique() if 'matchId' in season_events_df.columns else pd.Series(dtype='int')

    # Convert relevant columns safely
    season_events_df['possession.duration_sec'] = pd.to_numeric(season_events_df.get('possession.duration', pd.Series(dtype='str')).str.replace('s', ''), errors='coerce')
    season_events_df['location.x'] = pd.to_numeric(season_events_df.get('location.x'), errors='coerce')
    season_events_df['location.y'] = pd.to_numeric(season_events_df.get('location.y'), errors='coerce')
    season_events_df['pass.endLocation.x'] = pd.to_numeric(season_events_df.get('pass.endLocation.x'), errors='coerce')
    season_events_df['pass.endLocation.y'] = pd.to_numeric(season_events_df.get('pass.endLocation.y'), errors='coerce')
    season_events_df['pass.length'] = pd.to_numeric(season_events_df.get('pass.length'), errors='coerce')
    season_events_df['shot.xg'] = pd.to_numeric(season_events_df.get('shot.xg'), errors='coerce')

    # Pre-calculate possession time and losses
    total_possession_time_per_team = season_events_df.drop_duplicates(subset='possession.id').groupby('possession.team.name')['possession.duration_sec'].sum()
    league_total_in_play_time = total_possession_time_per_team.sum()
    
    losses_df = pd.DataFrame()
    if 'possession.id' in season_events_df.columns:
        season_events_df['next_possession.id'] = season_events_df['possession.id'].shift(-1)
        possession_changes = season_events_df[season_events_df['possession.id'] != season_events_df['next_possession.id']]
        losses_df = possession_changes[possession_changes.get('infraction.type') != 'foul_suffered'].copy()

    # Pre-calculate opponent events for defensive stats
    if 'opponentTeam.name' not in season_events_df.columns and 'matchId' in season_events_df.columns:
         # Use the correct column names from the new summary df
         temp_summary = matches_summary_df[['matchId', 'homeTeamName', 'awayTeamName']].copy()
         temp_summary.rename(columns={'homeTeamName':'ht', 'awayTeamName':'at'}, inplace=True)
         season_events_df = season_events_df.merge(temp_summary, on='matchId', how='left')
         season_events_df['opponentTeam.name'] = np.where(season_events_df['team.name'] == season_events_df['ht'], season_events_df['at'], season_events_df['ht'])
         season_events_df.drop(columns=['ht', 'at'], inplace=True, errors='ignore')

    # --- Loop Through Teams ---
    for team in teams:
        team_events = season_events_df[season_events_df.get('team.name') == team]
        opponent_events = season_events_df[season_events_df.get('opponentTeam.name') == team] if 'opponentTeam.name' in season_events_df.columns else pd.DataFrame()
        games = matches_played.get(team, 0)
        if games == 0: continue

        # --- Offensive Stats ---
        team_shots = team_events[team_events.get('type.primary') == 'shot']
        shots = team_shots.shape[0] / games
        goals = team_shots[team_shots.get('shot.isGoal') == True].shape[0] / games
        xg = team_shots['shot.xg'].sum() / games
        xg_per_shot = xg / shots if shots > 0 else 0
        PENALTY_AREA_X=83; PENALTY_AREA_Y1, PENALTY_AREA_Y2 = (21, 79) # Note: Wyscout PA Y is ~21-79
        actions_in_box = team_events[(team_events['location.x'].fillna(0) >= PENALTY_AREA_X) & (team_events['location.y'].fillna(0).between(PENALTY_AREA_Y1, PENALTY_AREA_Y2))].shape[0] / games
        team_passes = team_events[team_events.get('type.primary') == 'pass']
        passes_into_box = team_passes[(team_passes['pass.endLocation.x'].fillna(0) >= PENALTY_AREA_X) & (team_passes['pass.endLocation.y'].fillna(0).between(PENALTY_AREA_Y1, PENALTY_AREA_Y2))].shape[0] / games
        crosses = team_passes[team_passes.get('type.secondary','').astype(str).str.contains('cross', na=False)].shape[0] / games
        team_duels_off = team_events[team_events.get('type.primary') == 'duel']
        dribbles = team_duels_off[team_duels_off.get('groundDuel.takeOn') == True].shape[0] / games

        # --- Distribution Stats ---
        passes_per_match = team_passes.shape[0] / games
        # Use your notebook definition for Progressive Passes
        team_passes['start_dist_to_goal'] = np.sqrt((100 - team_passes['location.x'])**2 + (50 - team_passes['location.y'])**2)
        team_passes['end_dist_to_goal'] = np.sqrt((100 - team_passes['pass.endLocation.x'])**2 + (50 - team_passes['pass.endLocation.y'])**2)
        team_passes['progression'] = team_passes['start_dist_to_goal'] - team_passes['end_dist_to_goal']
        cond1 = (team_passes['location.x'] <= 50) & (team_passes['pass.endLocation.x'] <= 50) & (team_passes['progression'] >= 30)
        cond2 = (team_passes['location.x'] <= 50) & (team_passes['pass.endLocation.x'] > 50) & (team_passes['progression'] >= 15)
        cond3 = (team_passes['location.x'] > 50) & (team_passes['pass.endLocation.x'] > 50) & (team_passes['progression'] >= 10)
        progressive_passes = team_passes[cond1 | cond2 | cond3].shape[0] / games
        directness = team_passes['progression'].mean() # Use your notebook definition of directness
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
            except Exception: final_third_entries = 0
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
        total_aerial_duels = aerial_duels.shape[0]; won_aerial_duels_count = aerial_duels[aerial_duels.get('aerialDuel.firstTouch') == True].shape[0]
        aerial_duel_win_pct = (won_aerial_duels_count / total_aerial_duels) * 100 if total_aerial_duels > 0 else 0
        defensive_duels = team_duels_def[team_duels_def.get('groundDuel.duelType') == 'defensive_duel']
        total_defensive_duels = defensive_duels.shape[0]; won_defensive_duels_count = defensive_duels[(defensive_duels.get('groundDuel.recoveredPossession') == True) | (defensive_duels.get('groundDuel.stoppedProgress') == True)].shape[0]
        defensive_duel_win_pct = (won_defensive_duels_count / total_defensive_duels) * 100 if total_defensive_duels > 0 else 0
        interceptions = team_events[team_events.get('type.primary') == 'interception'].shape[0] / games
        fouls = team_events[team_events.get('type.primary') == 'infraction'].shape[0] / games
        # PPDA
        in_high_press_zone = season_events_df['location.x'].fillna(0) >= 40
        # Align index for boolean mask
        opponent_passes_df = opponent_events[(opponent_events.get('type.primary') == 'pass') & in_high_press_zone.reindex(opponent_events.index, fill_value=False)] 
        team_def_actions_df = team_events[in_high_press_zone.reindex(team_events.index, fill_value=False)] # Align index
        def_actions_for_ppda = team_def_actions_df[team_def_actions_df.get('type.primary').isin(['infraction', 'interception', 'duel'])].shape[0]
        ppda = opponent_passes_df.shape[0] / def_actions_for_ppda if def_actions_for_ppda > 0 else np.inf

        all_teams_stats[team] = {
            'Goals': goals, 'xG': xg, 'xG per Shot': xg_per_shot, 'Shots': shots,
            'Actions in Box': actions_in_box, 'Passes into Box': passes_into_box,
            'Crosses': crosses, 'Dribbles': dribbles,
            'Passes': passes_per_match, 'Progressive Passes': progressive_passes,
            'Directness': directness, 'Ball Possession': ball_possession_pct,
            'Final 1/3 Entries': final_third_entries, 'Losses': losses,
            'Goals Against': goals_against, 'xG Against': xg_against,
            'xG per Shot Against': xg_per_shot_against, 'Shots Against': shots_against,
            'Aerial Duel Win %': aerial_duel_win_pct, 'Defensive Duel Win %': defensive_duel_win_pct,
            'Interceptions': interceptions, 'Fouls': fouls, 'PPDA': ppda,
        }

    stats_df_raw = pd.DataFrame.from_dict(all_teams_stats, orient='index').fillna(0).round(2)
    stats_df_raw.replace([np.inf, -np.inf], 999, inplace=True)
    stats_df_pct = stats_df_raw.copy()
    metrics_to_invert_pct = ['Goals Against', 'xG Against', 'xG per Shot Against', 'Shots Against', 'PPDA', 'Losses']
    # Ensure columns exist before inverting
    valid_metrics_to_invert = [col for col in metrics_to_invert_pct if col in stats_df_pct.columns]
    stats_df_pct[valid_metrics_to_invert] = -stats_df_pct[valid_metrics_to_invert]
    for col in stats_df_pct.columns:
        stats_df_pct[col] = stats_df_pct[col].rank(pct=True) * 100
    return stats_df_raw, stats_df_pct

# --- Radar Plotting Function (Unchanged) ---
def plot_radar_chart(params, values_raw, values_pct, team_name, title_suffix, color, league="Liga 3 Portugal", season="2025/26"):
    # (This is the full function from the previous step)
    num_params = len(params); angles = np.linspace(0, 2 * np.pi, num_params, endpoint=False).tolist(); angles += angles[:1]
    plot_values_pct = values_pct + values_pct[:1]; fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    fig.set_facecolor('#f5f1e9'); ax.set_facecolor('#f5f1e9'); ax.set_xticks(angles[:-1]); ax.set_ylim(0, 100)
    ax.grid(color='gray', linestyle='--', linewidth=0.5); ax.spines['polar'].set_color('gray'); ax.set_yticks([25, 50, 75])
    ax.set_yticklabels(["25th", "50th", "75th"], color="grey", size=10); ax.set_rlabel_position(angles[0] * 180/np.pi + 10); ax.set_thetagrids([], [])
    LABEL_DISTANCES = {"xG per Shot": 106, "Crosses": 107, "Directness": 106, "Avg Out-of-Possession Action Height": 108, "Avg In-Possession Action Height": 122, "Final 1/3 Entries": 117, "Shots Against": 106, "xG per Shot Against": 108, "PPDA": 110, "Quick Recoveries": 110, "DEFAULT": 115}
    for angle, param, percentile in zip(angles[:-1], params, values_pct):
        percentile_val = int(round(percentile, 0)); label_text = f"{param}\n({percentile_val}th %-tile)"; distance = LABEL_DISTANCES.get(param, LABEL_DISTANCES["DEFAULT"])
        ha_align = 'left' if (np.degrees(angle) > 100 and np.degrees(angle) < 260) else 'right'; ha_align = 'center' if (abs(np.degrees(angle) - 90) < 10 or abs(np.degrees(angle) - 270) < 10) else ha_align
        ax.text(angle, distance, label_text, ha=ha_align, va='center', size=10)
    ax.plot(angles, plot_values_pct, color=color, linewidth=2, linestyle='solid'); ax.fill(angles, plot_values_pct, color=color, alpha=0.6)
    for angle, value_raw, value_pct in zip(angles[:-1], values_raw, values_pct):
         raw_display = f'{value_raw}%' if '%' in str(value_raw) else f'{value_raw}'; ax.text(angle, 95, raw_display, ha='center', va='top', size=9, weight='bold', bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor='none', alpha=0.7))
    footer_text = "@lucaskimball | Data via Wyscout | Values in parentheses are percentile rank vs. other Liga 3 teams"; fig.text(0.02, 0.02, footer_text, ha='left', va='bottom', fontsize=9, color='gray')
    report_date = datetime.date.today().strftime("%Y-%m-%d"); full_title = f"{team_name}\n{title_suffix} | {league} {season} (As of: {report_date})"; ax.set_title(full_title, size=18, weight='bold', pad=40)
    return fig

# --- Corner Analysis Plotting Function (Unchanged) ---
def plot_corner_analysis(season_events_df, team_to_analyze, side, league="Liga 3 Portugal", season="2025/26"):
    # (This is the full function from the previous step)
    def categorize_corner(row, side):
        end_x = row.get('pass.endLocation.x'); end_y = row.get('pass.endLocation.y'); pass_len = row.get('pass.length')
        if pd.isna(pass_len) and pd.notna(row.get('location.x')): start_x = row.get('location.x', 0); start_y = row.get('location.y', 0); PITCH_LENGTH_M, PITCH_WIDTH_M = 105.0, 68.0; pass_len = np.sqrt(((end_x - start_x) * (PITCH_LENGTH_M / 100.0))**2 + ((end_y - start_y) * (PITCH_WIDTH_M / 100.0))**2)
        if pd.isna(end_x) or pd.isna(end_y): return 'Other'
        PENALTY_AREA_X = 83; SIX_YARD_BOX_Y1, SIX_YARD_BOX_Y2 = (36, 64); SHORT_CORNER_MAX_DIST_FROM_START = 20
        if end_x < PENALTY_AREA_X or (pd.notna(pass_len) and pass_len < SHORT_CORNER_MAX_DIST_FROM_START): return 'Short'
        third_of_box = (SIX_YARD_BOX_Y2 - SIX_YARD_BOX_Y1) / 3; near_thresh = SIX_YARD_BOX_Y1 + third_of_box; far_thresh = SIX_YARD_BOX_Y2 - third_of_box
        if side == 'left':
            if end_y < near_thresh: return 'Near Post'
            elif end_y > far_thresh: return 'Far Post'
            else: return 'Middle'
        elif side == 'right':
             if end_y > far_thresh: return 'Near Post'
             elif end_y < near_thresh: return 'Far Post'
             else: return 'Middle'
        return 'Other'
    if side == 'left': side_corners_df = season_events_df[(season_events_df.get('team.name') == team_to_analyze) & (season_events_df.get('type.primary') == 'corner') & (season_events_df.get('location.y', 101) < 50)].copy()
    else: side_corners_df = season_events_df[(season_events_df.get('team.name') == team_to_analyze) & (season_events_df.get('type.primary') == 'corner') & (season_events_df.get('location.y', -1) >= 50)].copy()
    if side_corners_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5)); fig.set_facecolor('#f5f1e9'); ax.set_facecolor('#f5f1e9'); ax.text(0.5, 0.5, f'No {side} corners found for {team_to_analyze}', ha='center', va='center', fontsize=14); ax.axis('off'); return fig
    side_corners_df['zone'] = side_corners_df.apply(categorize_corner, axis=1, side=side)
    if 'player.name' in side_corners_df.columns: corner_takers = side_corners_df.groupby('player.name').agg(Total=('id', 'count'), Short=('zone', lambda x: (x == 'Short').sum()), Near=('zone', lambda x: (x == 'Near Post').sum()), Middle=('zone', lambda x: (x == 'Middle').sum()), Far=('zone', lambda x: (x == 'Far Post').sum())).sort_values(by='Total', ascending=False).fillna(0).astype(int)
    else: corner_takers = pd.DataFrame(columns=['Total', 'Short', 'Near', 'Middle', 'Far'])
    fig = plt.figure(figsize=(16, 8)); fig.set_facecolor('#f5f1e9'); gs = gridspec.GridSpec(1, 2, width_ratios=[0.6, 0.4]); ax_pitch = fig.add_subplot(gs[0, 0]); ax_table = fig.add_subplot(gs[0, 1]); ax_table.axis('off')
    pitch = Pitch(pitch_type='wyscout', pitch_color='#f5f1e9', line_color='black', line_zorder=2); pitch.draw(ax=ax_pitch); zone_colors = {'Short': 'blue', 'Near Post': 'orange', 'Middle': 'red', 'Far Post': 'yellow', 'Other': 'grey'}
    for idx, corner in side_corners_df.iterrows():
         if pd.notna(corner.get('pass.endLocation.x')) and pd.notna(corner.get('pass.endLocation.y')): pitch.scatter(x=corner['pass.endLocation.x'], y=corner['pass.endLocation.y'], s=200, color=zone_colors.get(corner['zone'], 'gray'), edgecolor='black', ax=ax_pitch, zorder=3, alpha=0.7)
    ax_pitch.set_title(f"Corners from the {side.capitalize()} Side | {league} {season}", fontsize=14); legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Short'), Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Near Post'), Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Middle'), Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=10, label='Far Post'), Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', markersize=10, label='Other/Outside PA')]; ax_pitch.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(0.01, 0.01), frameon=False, fontsize=10)
    ax_table.set_title("Corner Taker Summary", fontsize=14, weight='bold')
    if not corner_takers.empty:
        table = Table(ax_table, bbox=[0, 0, 1, 0.9], loc='center'); table.auto_set_font_size(False); table.set_fontsize(10)
        table_data = [['Player'] + list(corner_takers.columns)] + [[idx] + list(row) for idx, row in corner_takers.iterrows()]
        col_widths = [0.4] + [0.12] * 5
        for i, row_list in enumerate(table_data):
            for j, cell_text in enumerate(row_list):
                is_header = (i == 0); weight = 'bold' if is_header or j == 0 else 'normal'; facecolor = '#e0e0e0' if is_header else ['#fdfdfd', '#f0f0f0'][i % 2]; loc = 'left' if j == 0 else 'center'
                cell = table.add_cell(i, j, width=col_widths[j], height=1.0/len(table_data), text=cell_text, loc=loc, facecolor=facecolor, edgecolor='w', fontproperties={'weight': weight})
        ax_table.add_table(table)
    else: ax_table.text(0.5, 0.5, "No corner takers found.", ha='center', va='center')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); return fig

# --- (ORIGINAL MATPLOTLIB FUNCTION 1) ---
def create_match_shotmap(match_events_df, match_info, team_to_analyze):
 
    team_shots_df = match_events_df[(match_events_df.get('team.name') == team_to_analyze) & (match_events_df.get('type.primary').isin(['shot', 'penalty']))].copy().reset_index(drop=True)
    if team_shots_df.empty:
        fig, ax = plt.subplots(figsize=(10, 8)); fig.set_facecolor('#f5f1e9'); ax.set_facecolor('#f5f1e9'); ax.text(0.5, 0.5, 'No shots found for this team in this match.', ha='center', va='center', fontsize=12); ax.axis('off'); return fig
    
    home_team = match_info.get('homeTeamName', '?'); away_team = match_info.get('awayTeamName', '?'); opponent = away_team if team_to_analyze == home_team else home_team
    fig = plt.figure(figsize=(12, 12)); fig.set_facecolor('#f5f1e9'); pitch = Pitch(pitch_type='wyscout', pitch_color='#f5f1e9', line_color='black', line_zorder=2, half=True); ax_pitch = fig.add_subplot(); pitch.draw(ax=ax_pitch)
    
    # Your original colormap
    XG_MAX = 0.8; colors = ["#03045e", "#ade8f4", "#fff3b0", "#ff8c00", "#e63946", "#800f2f"]; nodes = [0.0, 0.1 / XG_MAX, 0.2 / XG_MAX, 0.4 / XG_MAX, 0.6 / XG_MAX, 1.0]; cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", list(zip(nodes, colors)))
    
    for index, shot in team_shots_df.iterrows():
        x = shot.get('location.x'); y = shot.get('location.y'); xg = pd.to_numeric(shot.get('shot.xg'), errors='coerce')
        if pd.isna(x) or pd.isna(y) or pd.isna(xg): continue
        is_goal = shot.get('shot.isGoal') == True; is_on_target = shot.get('shot.onTarget') == True
        
        color = cmap(min(xg / XG_MAX, 1.0)); edge_color = 'gray'; line_width = 1.5
        
        if is_goal: edge_color = 'green'; line_width = 2.5
        elif is_on_target: edge_color = 'black'; line_width = 2.5
            
        pitch.scatter(x, y, s=400, facecolor=color, edgecolor=edge_color, linewidth=line_width, ax=ax_pitch, zorder=3)
        pitch.text(x, y, str(index + 1), ax=ax_pitch, ha='center', va='center', fontsize=9, color='white', zorder=4)
        
    subtitle = f"vs. {opponent} | Score: {match_info.get('score', '?-?')} | xG: {pd.to_numeric(team_shots_df['shot.xg'], errors='coerce').sum():.2f}"; 
    ax_pitch.set_title(f"{team_to_analyze} Shot Map\n{subtitle}", fontsize=18, weight='bold')
    return fig

# --- (ORIGINAL MATPLOTLIB FUNCTION 2) ---
def create_season_shotmap(season_events_df, team_to_analyze):
    
    team_shots_df = season_events_df[(season_events_df.get('team.name') == team_to_analyze) & (season_events_df.get('type.primary') == 'shot')].copy().reset_index(drop=True)
    if team_shots_df.empty:
        fig, ax = plt.subplots(figsize=(10, 8)); fig.set_facecolor('#f5f1e9'); ax.set_facecolor('#f5f1e9'); ax.text(0.5, 0.5, 'No shots found for this team this season.', ha='center', va='center', fontsize=12); ax.axis('off'); return fig
    
    fig = plt.figure(figsize=(12, 12)); fig.set_facecolor('#f5f1e9'); pitch = Pitch(pitch_type='wyscout', pitch_color='#f5f1e9', line_color='black', line_zorder=2, half=True); ax_pitch = fig.add_subplot(); pitch.draw(ax=ax_pitch)
    
    XG_MAX = 0.8; colors = ["#03045e", "#ade8f4", "#fff3b0", "#ff8c00", "#e63946", "#800f2f"]; nodes = [0.0, 0.1 / XG_MAX, 0.2 / XG_MAX, 0.4 / XG_MAX, 0.6 / XG_MAX, 1.0]; cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", list(zip(nodes, colors)))
    
    for index, shot in team_shots_df.iterrows():
        x = shot.get('location.x'); y = shot.get('location.y'); xg = pd.to_numeric(shot.get('shot.xg'), errors='coerce')
        if pd.isna(x) or pd.isna(y) or pd.isna(xg): continue
        is_goal = shot.get('shot.isGoal') == True; color = cmap(min(xg / XG_MAX, 1.0)); edge_color = 'green' if is_goal else 'black'
        pitch.scatter(x, y, s=150, facecolor=color, edgecolor=edge_color, linewidth=1.5, ax=ax_pitch, zorder=3, alpha=0.7)
        
    total_xg = pd.to_numeric(team_shots_df['shot.xg'], errors='coerce').sum(); goals = team_shots_df['shot.isGoal'].sum(); subtitle = f"Liga 3 Portugal, 2025/26 | Total xG: {total_xg:.2f} | Goals: {goals}"; 
    ax_pitch.set_title(f"{team_to_analyze} Season Shot Map (Non-Penalty)\n{subtitle}", fontsize=18, weight='bold')
    return fig

# --- (ORIGINAL MATPLOTLIB FUNCTION 3) ---
def create_season_shots_against_shotmap(season_events_df, matches_summary_df, team_to_analyze):
    
    team_match_ids = matches_summary_df[(matches_summary_df.get('homeTeamName') == team_to_analyze) | (matches_summary_df.get('awayTeamName') == team_to_analyze)]['matchId'].unique()
    relevant_events = season_events_df[season_events_df['matchId'].isin(team_match_ids)]
    opponent_shots_df = relevant_events[(relevant_events.get('type.primary') == 'shot') & (relevant_events.get('team.name') != team_to_analyze)].copy().reset_index(drop=True)
    if opponent_shots_df.empty:
        fig, ax = plt.subplots(figsize=(10, 8)); fig.set_facecolor('#f5f1e9'); ax.set_facecolor('#f5f1e9'); ax.text(0.5, 0.5, 'No shots against found for this team.', ha='center', va='center', fontsize=12); ax.axis('off'); return fig
    
    fig = plt.figure(figsize=(12, 12)); fig.set_facecolor('#f5f1e9'); pitch = Pitch(pitch_type='wyscout', pitch_color='#f5f1e9', line_color='black', line_zorder=2, half=True); ax_pitch = fig.add_subplot(); pitch.draw(ax=ax_pitch)
    
    XG_MAX = 0.8; colors = ["#03045e", "#ade8f4", "#fff3b0", "#ff8c00", "#e63946", "#800f2f"]; nodes = [0.0, 0.1 / XG_MAX, 0.2 / XG_MAX, 0.4 / XG_MAX, 0.6 / XG_MAX, 1.0]; cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", list(zip(nodes, colors)))
    
    for index, shot in opponent_shots_df.iterrows():
        x = shot.get('location.x'); y = shot.get('location.y'); xg = pd.to_numeric(shot.get('shot.xg'), errors='coerce'); is_goal = shot.get('shot.isGoal') == True
        if pd.isna(x) or pd.isna(y) or pd.isna(xg): continue
        color = cmap(min(xg / XG_MAX, 1.0)); edge_color = 'green' if is_goal else 'black'; pitch.scatter(x, y, s=150, facecolor=color, edgecolor=edge_color, linewidth=1.5, ax=ax_pitch, zorder=3, alpha=0.7)
        
    total_shots_against = len(opponent_shots_df); total_xg_against = round(pd.to_numeric(opponent_shots_df.get('shot.xg'), errors='coerce').sum(), 2); goals_against = opponent_shots_df[opponent_shots_df.get('shot.isGoal') == True].shape[0]
    subtitle = f"Liga 3 Portugal, 2025/26 | Total xGA: {total_xg_against} | Goals Against: {goals_against}"; 
    ax_pitch.set_title(f"{team_to_analyze} Shots CONCEDED Map (Non-Penalty)\n{subtitle}", fontsize=18, weight='bold')
    return fig

# --- NEW FUNCTION: Calculate Team Strength ---
@st.cache_data
def calculate_team_strength(season_events_df, matches_summary_df):
    """Calculates Attacking and Defending Strength metrics for all teams."""
    print("Calculating team strength stats...") # Debug print
    team_stats = {}

    # Ensure necessary columns exist and are numeric
    all_shots = season_events_df[season_events_df.get('type.primary') == 'shot'].copy()
    all_shots['shot.xg'] = pd.to_numeric(all_shots.get('shot.xg'), errors='coerce').fillna(0)
    all_shots['shot.isGoal'] = all_shots.get('shot.isGoal') == True # Ensure boolean

    all_teams_in_data = season_events_df['team.name'].unique()
    # Ensure matchId exists before grouping
    matches_played = season_events_df.groupby('team.name')['matchId'].nunique() if 'matchId' in season_events_df.columns else pd.Series(dtype='int')

    # Add opponent name if missing (needed for GA/xGA)
    if 'opponentTeam.name' not in all_shots.columns and 'matchId' in all_shots.columns:
         if matches_summary_df is not None and not matches_summary_df.empty:
             # --- UPDATED Column Names ---
             temp_summary = matches_summary_df[['matchId', 'homeTeamName', 'awayTeamName']].copy()
             temp_summary.rename(columns={'homeTeamName':'ht', 'awayTeamName':'at'}, inplace=True)
             # ---
             all_shots = all_shots.merge(temp_summary, on='matchId', how='left')
             all_shots['opponentTeam.name'] = np.where(all_shots['team.name'] == all_shots['ht'], all_shots['at'], all_shots['ht'])
             all_shots.drop(columns=['ht', 'at'], inplace=True, errors='ignore')
         else:
             print("Warning: Cannot calculate GA/xGA reliably without opponent names.")
             all_shots['opponentTeam.name'] = "Unknown Opponent"


    for team in all_teams_in_data:
        team_shots = all_shots[all_shots.get('team.name') == team]
        goals_for = team_shots['shot.isGoal'].sum()
        xg_for = team_shots['shot.xg'].sum()

        opponent_shots = all_shots[all_shots.get('opponentTeam.name') == team]
        goals_against = opponent_shots['shot.isGoal'].sum()
        xg_against = opponent_shots['shot.xg'].sum()

        games = matches_played.get(team, 0)
        if games > 0:
            team_stats[team] = {
                'GF_per_match': goals_for / games,
                'GA_per_match': goals_against / games,
                'xGF_per_match': xg_for / games,
                'xGA_per_match': xg_against / games
            }

    stats_df = pd.DataFrame.from_dict(team_stats, orient='index').fillna(0)
    if stats_df.empty:
        return pd.DataFrame() # Return empty if no stats calculated

    # Calculate Strength Metrics
    stats_df['Attacking Strength'] = (stats_df['GF_per_match'] * 0.3) + (stats_df['xGF_per_match'] * 0.7)
    stats_df['Defending Strength'] = (stats_df['GA_per_match'] * 0.3) + (stats_df['xGA_per_match'] * 0.7)

    return stats_df


# --- NEW FUNCTION: Plot Team Strength Scatter ---
def plot_team_strength(stats_df, teams_to_include=None, league="Liga 3 Portugal", season="2025/26", icon_zoom=0.25): # <-- ADDED icon_zoom
    """Generates the Matplotlib figure for the team strength scatter plot."""

    if stats_df.empty or 'Attacking Strength' not in stats_df.columns or 'Defending Strength' not in stats_df.columns:
         fig, ax = plt.subplots(figsize=(10, 8)); fig.set_facecolor('#f5f1e9'); ax.set_facecolor('#f5f1e9')
         ax.text(0.5, 0.5, 'Team strength data unavailable.', ha='center', va='center', fontsize=14); ax.axis('off'); return fig

    fig, ax = plt.subplots(figsize=(16, 12)); fig.set_facecolor('#f5f1e9'); ax.set_facecolor('#f5f1e9'); ax.invert_yaxis()
    x_min, x_max = stats_df['Attacking Strength'].min(), stats_df['Attacking Strength'].max()
    y_min, y_max = stats_df['Defending Strength'].min(), stats_df['Defending Strength'].max()
    x_padding = (x_max - x_min) * 0.1; y_padding = (y_max - y_min) * 0.1
    ax.set_xlim(x_min - x_padding, x_max + x_padding); ax.set_ylim(y_max + y_padding, y_min - y_padding) # Inverted Y
    # --- (NEW) Diagonal Lines Logic ---
    # Get the final plot limits *after* setting them
    x_min, x_max = ax.get_xlim()
    y_max, y_min = ax.get_ylim() # Remember y-axis is inverted (y_max < y_min)

    # Calculate the 'c' value (c = x - y) for all four corners of the plot
    c_top_left = x_min - y_max
    c_top_right = x_max - y_max
    c_bottom_left = x_min - y_min
    c_bottom_right = x_max - y_min

    # Find the minimum and maximum 'c' values, rounded to nearest 0.1
    min_c = np.floor(min([c_top_left, c_top_right, c_bottom_left, c_bottom_right]) * 10) / 10
    max_c = np.ceil(max([c_top_left, c_top_right, c_bottom_left, c_bottom_right]) * 10) / 10

    # Draw lines for every 'c' value in the calculated range
    for c in np.arange(min_c, max_c + 0.1, 0.1):
        # Use axline to draw an infinite line with slope 1 passing through (0, -c)
        # Matplotlib will automatically clip it to the plot boundaries
        ax.axline((0, -c), slope=1, color='lightgray', linestyle=':', zorder=1, lw=1)
    # --- (END NEW) Diagonal Lines Logic ---


    stats_df_to_plot = stats_df
    if teams_to_include:
        valid_teams = [t for t in teams_to_include if t in stats_df.index]
        stats_df_to_plot = stats_df.loc[valid_teams]

    texts = []; logos_plotted = 0; base_icon_path = "icons" # ASSUMES 'icons' FOLDER
    for team_name, row in stats_df_to_plot.iterrows():
        safe_team_name = team_name.replace('/', '_').replace('\\', '_')
        logo_path = os.path.join(base_icon_path, f"{safe_team_name}.png")
        try:
            if os.path.exists(logo_path):
                 img = Image.open(logo_path); # --- USE THE NEW PARAMETER HERE ---
                 imagebox = OffsetImage(img, zoom=icon_zoom) # <-- USE icon_zoom
                 ab = AnnotationBbox(imagebox, (row['Attacking Strength'], row['Defending Strength']), frameon=False, zorder=2)
                 ax.add_artist(ab)
                 logos_plotted +=1
            else: texts.append(ax.text(row['Attacking Strength'], row['Defending Strength'], team_name, zorder=3, fontsize=9))
        except Exception as e: print(f"Error loading logo for {team_name}: {e}. Using text."); texts.append(ax.text(row['Attacking Strength'], row['Defending Strength'], team_name, zorder=3, fontsize=9))
    if texts: adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
    if logos_plotted == 0 and not texts: ax.scatter(stats_df_to_plot['Attacking Strength'], stats_df_to_plot['Defending Strength'], s=50, zorder=2)

    report_date = datetime.date.today().strftime("%Y-%m-%d")
    ax.set_title(f'Team Strength Scatterplot | {league}, {season} (As of: {report_date})', fontsize=18, weight='bold')
    ax.set_xlabel('Attacking Strength (30% NP Goals, 70% NPxG)', fontsize=12)
    ax.set_ylabel('Defending Strength (30% NP Goals Against, 70% NPxG Against)', fontsize=12)
    #ax.grid(True, linestyle='--', alpha=0.5); plt.tight_layout(); return fig

# app.py (Add this new function)

# --- NEW FUNCTION: Plot Custom Scatter Plot ---
def plot_custom_scatter(stats_df, x_metric, y_metric, invert_x=False, invert_y=False, league="Liga 3 Portugal", season="2025/26"):
    """Generates a dynamic Matplotlib scatter plot with logos."""

    # Ensure the selected metrics exist in the DataFrame
    if x_metric not in stats_df.columns or y_metric not in stats_df.columns:
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.set_facecolor('#f5f1e9'); ax.set_facecolor('#f5f1e9')
        ax.text(0.5, 0.5, f"Error: Metric not found.\nCheck data processing script.", ha='center', va='center', fontsize=12, color='red')
        ax.axis('off'); return fig

    fig, ax = plt.subplots(figsize=(16, 12))
    fig.set_facecolor('#f5f1e9')
    ax.set_facecolor('#f5f1e9')

    x_data = stats_df[x_metric]
    y_data = stats_df[y_metric]

    # --- 1. Set Axis Limits & Padding ---
    x_min, x_max = x_data.min(), x_data.max()
    y_min, y_max = y_data.min(), y_data.max()
    x_padding = (x_max - x_min) * 0.1
    y_padding = (y_max - y_min) * 0.1

    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)

    # --- 2. Invert Axis (if user checked the box) ---
    if invert_x:
        ax.set_xlim(x_max + x_padding, x_min - x_padding)
    if invert_y:
        ax.set_ylim(y_max + y_padding, y_min - y_padding)
        
    # --- 3. Add Mean Quadrant Lines for Context ---
    x_mean = x_data.mean()
    y_mean = y_data.mean()
    ax.axhline(y_mean, color='gray', linestyle='--', lw=1, zorder=1)
    ax.axvline(x_mean, color='gray', linestyle='--', lw=1, zorder=1)

    # --- 4. Plot Logos (re-using logic from plot_team_strength) ---
    stats_df_to_plot = stats_df.copy()
    texts = []; logos_plotted = 0; base_icon_path = "icons" 

    for team_name, row in stats_df_to_plot.iterrows():
        safe_team_name = team_name.replace('/', '_').replace('\\', '_')
        logo_path = os.path.join(base_icon_path, f"{safe_team_name}.png")
        try:
            if os.path.exists(logo_path):
                 img = Image.open(logo_path)
                 # Use the increased zoom factor
                 imagebox = OffsetImage(img, zoom=0.25) 
                 # Plot using the dynamic x_metric and y_metric
                 ab = AnnotationBbox(imagebox, (row[x_metric], row[y_metric]), frameon=False, zorder=2)
                 ax.add_artist(ab)
                 logos_plotted +=1
            else:
                 texts.append(ax.text(row[x_metric], row[y_metric], team_name, zorder=3, fontsize=9))
        except Exception as e:
            print(f"Error loading logo for {team_name}: {e}. Using text.")
            texts.append(ax.text(row[x_metric], row[y_metric], team_name, zorder=3, fontsize=9))
    
    if texts:
        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
    if logos_plotted == 0 and not texts: 
         ax.scatter(stats_df_to_plot[x_metric], stats_df_to_plot[y_metric], s=50, zorder=2)

    # --- 5. Styling ---
    report_date = datetime.date.today().strftime("%Y-%m-%d")
    ax.set_title(f'League Scatterplot | {league}, {season} (As of: {report_date})', fontsize=18, weight='bold')
    ax.set_xlabel(x_metric, fontsize=12) # Dynamic X Label
    ax.set_ylabel(y_metric, fontsize=12) # Dynamic Y Label

    plt.tight_layout()
    return fig

# ... after plot_custom_scatter ...

def plot_xg_flowchart(match_events_df, match_info):
    """
    Generates a Matplotlib figure for the match xG flowchart.
    Includes markers for goals (regular, penalty, own) and red cards.
    """
    
    # 1. Get team names and set colors
    home_team = match_info.get('homeTeamName', 'Home')
    away_team = match_info.get('awayTeamName', 'Away')
    home_color = '#0077b6' # Blue
    away_color = '#e63946' # Red
    
    # 2. Filter for relevant events (shots, penalties, own goals, and red cards)
    # --- UPDATED: Include 'own_goal' type ---
    events_df = match_events_df[match_events_df['type.primary'].isin(['shot', 'penalty', 'own_goal'])].copy()
    events_df = events_df[['minute', 'team.name', 'shot.xg', 'shot.isGoal', 'type.primary']]
    
    reds_df = match_events_df[match_events_df.get('infraction.redCard') == True].copy()
    if not reds_df.empty:
        reds_df = reds_df[['minute', 'team.name']]
        reds_df['isRedCard'] = True
    else:
        reds_df = pd.DataFrame(columns=['minute', 'team.name', 'isRedCard'])
    
    # 3. Combine and sort
    # --- UPDATED: Use events_df ---
    df = pd.concat([events_df, reds_df]).sort_values(by='minute')
    
    # 4. Clean NaNs and identify event types
    df['shot.xg'] = pd.to_numeric(df['shot.xg'], errors='coerce').fillna(0)
    df['isRedCard'] = df['isRedCard'].fillna(False)
    
    # --- UPDATED: Identify Goal Types (using type.primary) ---
    df['shot.isGoal'] = df['shot.isGoal'].fillna(False) # For regular shots/penalties
    df['isPenalty'] = df['type.primary'] == 'penalty'
    df['isOwnGoal'] = df['type.primary'] == 'own_goal' # <-- THE FIX
    
    # A 'Regular Goal' is a scored shot, but not a penalty
    # (An own goal will have shot.isGoal=False, so it's excluded)
    df['isRegularGoal'] = df['shot.isGoal'] & (df['type.primary'] == 'shot')
    
    # A "Goal" for plotting is a scored shot/penalty OR an own goal
    df['isGoal'] = df['shot.isGoal'] | df['isOwnGoal']
    
    # 5. Create xG columns per team
    df['home_xG'] = np.where(df['team.name'] == home_team, df['shot.xg'], 0)
    df['away_xG'] = np.where(df['team.name'] == away_team, df['shot.xg'], 0)
    
    # 6. Add start row
    start_row = pd.DataFrame([{
        'minute': 0, 'home_xG': 0, 'away_xG': 0, 
        'shot.isGoal': False, 'isRedCard': False, 'team.name': 'Start',
        'isPenalty': False, 'isOwnGoal': False, 'isRegularGoal': False, 'isGoal': False
    }])
    df = pd.concat([start_row, df]).sort_values(by='minute')
    
    # 7. Calculate cumulative xG
    df['home_xG_cum'] = df['home_xG'].cumsum()
    df['away_xG_cum'] = df['away_xG'].cumsum()

    # 8. Get max minute for plot limit
    max_minute = df['minute'].max()
    if max_minute < 90: max_minute = 90
    
    # 9. Plotting
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.set_facecolor('#f5f1e9')
    ax.set_facecolor('#f5f1e9')
    
    ax.step(df['minute'], df['home_xG_cum'], label=home_team, color=home_color, where='post', linewidth=2.5)
    ax.step(df['minute'], df['away_xG_cum'], label=away_team, color=away_color, where='post', linewidth=2.5)
    
    # 10. Add Goal and Red Card Markers (This logic remains the same, but now uses the corrected types)
    goals_df = df[df['isGoal'] == True]
    for _, row in goals_df.iterrows():
        if row['isRegularGoal']:
            label = f"Goal ({row['minute']}')"
            marker_size = 250
            marker_shape = 'o'
        elif row['isPenalty'] and row['shot.isGoal']: # Check if penalty was scored
            label = f"Penalty Goal ({row['minute']}')"
            marker_size = 350
            marker_shape = 'X'
        elif row['isOwnGoal']:
            label = f"Own Goal ({row['minute']}')"
            marker_size = 250
            marker_shape = 's'
        else:
            continue # Don't plot missed penalties as "goals"
            
        if row['team.name'] == home_team:
            ax.scatter(row['minute'], row['home_xG_cum'], c=home_color, s=marker_size, marker=marker_shape, 
                       edgecolor='black', zorder=5, label=label)
        else:
            ax.scatter(row['minute'], row['away_xG_cum'], c=away_color, s=marker_size, marker=marker_shape, 
                       edgecolor='black', zorder=5, label=label)

    reds_df_plot = df[df['isRedCard'] == True]
    for _, row in reds_df_plot.iterrows():
        label = f"Red Card ({row['minute']}')"
        y_val = row['home_xG_cum'] if row['team.name'] == home_team else row['away_xG_cum']
        ax.scatter(row['minute'], y_val, c='red', s=150, marker='D', 
                   edgecolor='black', linewidth=1.5, zorder=5, label=label)

    # 11. Styling
    ax.set_xlabel('Minute', fontsize=12)
    ax.set_ylabel('Cumulative xG', fontsize=12)
    ax.set_title(f"xG Flowchart: {home_team} vs {away_team}\nScore: {match_info.get('score', '?-?')}", fontsize=16, weight='bold')
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), loc='upper left', frameon=False)
    
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlim(0, max_minute + 2)
    ax.set_ylim(0)

    plt.tight_layout()
    return fig

@st.cache_data
def calculate_rolling_xg_data(_raw_events_df, _matches_summary_df):
    """
    Aggregates xG For and Against for every team for every match
    to be used in rolling average charts.
    """
    print("Calculating rolling xG data...") # Debug print
    all_team_matches = []
    
    shots_df = _raw_events_df[_raw_events_df['type.primary'].isin(['shot', 'penalty'])].copy()
    shots_df['shot.xg'] = pd.to_numeric(shots_df['shot.xg'], errors='coerce').fillna(0)
    xg_by_match_team = shots_df.groupby(['matchId', 'team.name'])['shot.xg'].sum()

    for _, match in _matches_summary_df.iterrows():
        matchId = match.get('matchId')
        date = match.get('dateutc')
        season_marker = f"GW {match.get('gameweek', '1')}" # Keep this for labels
        season_id = match.get('seasonId') # <-- GET THE SEASON ID
        home_team = match.get('homeTeamName')
        away_team = match.get('awayTeamName')
        
        if not all([matchId, date, home_team, away_team, season_id]): # <-- ADD season_id CHECK
            continue 
            
        home_xg_for = xg_by_match_team.get((matchId, home_team), 0)
        away_xg_for = xg_by_match_team.get((matchId, away_team), 0)
        
        all_team_matches.append({
            'date': date, 
            'season_marker': season_marker, 
            'seasonId': season_id, # <-- ADD IT HERE
            'teamName': home_team, 
            'xG_For': home_xg_for, 
            'xG_Against': away_xg_for
        })
        all_team_matches.append({
            'date': date, 
            'season_marker': season_marker, 
            'seasonId': season_id, # <-- AND ADD IT HERE
            'teamName': away_team, 
            'xG_For': away_xg_for, 
            'xG_Against': home_xg_for
        })
    
    if not all_team_matches:
        print("Warning: No matches for rolling xG data.")
        return pd.DataFrame()

    result_df = pd.DataFrame(all_team_matches)
    result_df['date'] = pd.to_datetime(result_df['date'])
    result_df = result_df.sort_values(by='date')
    
    print("✅ Rolling xG data calculated.")
    return result_df

def plot_rolling_xg(all_matches_df, selected_team):
    """
    Plots the rolling xG For and Against for a team over the last year.
    Trendline is only for the most recent season.
    """
    # 1. Filter for selected team
    team_df = all_matches_df[all_matches_df['teamName'] == selected_team].copy()
    if team_df.empty:
        fig, ax = plt.subplots(figsize=(14, 7)); ax.text(0.5, 0.5, 'No match data found for this team.', ha='center'); return fig
        
    # 2. Filter for last 365 days
    today = pd.to_datetime(datetime.date.today())
    one_year_ago = today - pd.DateOffset(years=1)
    team_df = team_df[(team_df['date'] >= one_year_ago) & (team_df['date'] <= today)]
    
    if team_df.empty:
        fig, ax = plt.subplots(figsize=(14, 7)); ax.text(0.5, 0.5, 'No match data in the last 365 days.', ha='center'); return fig

    # 3. Calculate 10-game rolling average
    rolling_window = 10 
    team_df = team_df.sort_values(by='date')
    team_df['xG_For_Roll'] = team_df['xG_For'].rolling(window=rolling_window, min_periods=1).mean()
    team_df['xG_Against_Roll'] = team_df['xG_Against'].rolling(window=rolling_window, min_periods=1).mean()
    
    # 4. Calculate trendlines
    team_df = team_df.dropna(subset=['xG_For_Roll', 'xG_Against_Roll', 'seasonId'])
    if team_df.empty:
        fig, ax = plt.subplots(figsize=(14, 7)); ax.text(0.5, 0.5, 'Not enough data for rolling average.', ha='center'); return fig
        
    team_df['date_numeric'] = mdates.date2num(team_df['date'])
    
    # --- NEW: Filter for current season data FOR TRENDLINE ONLY ---
    current_season_id = team_df.iloc[-1]['seasonId'] # Get seasonId of the most recent game
    current_season_df = team_df[team_df['seasonId'] == current_season_id].copy()

    # 4b. Calculate trendlines ONLY for the current season
    if len(current_season_df) > 1: # Need at least 2 points for a line
        z_for = np.polyfit(current_season_df['date_numeric'], current_season_df['xG_For_Roll'], 1)
        p_for = np.poly1d(z_for)
        current_season_df['xG_For_Trend'] = p_for(current_season_df['date_numeric'])
        
        z_against = np.polyfit(current_season_df['date_numeric'], current_season_df['xG_Against_Roll'], 1)
        p_against = np.poly1d(z_against)
        current_season_df['xG_Against_Trend'] = p_against(current_season_df['date_numeric'])
    else:
        current_season_df['xG_For_Trend'] = np.nan
        current_season_df['xG_Against_Trend'] = np.nan
    # --- END NEW TRENDLINE LOGIC ---

    # 5. Get season markers (for Gameweek 1)
    season_starts = team_df[team_df['season_marker'] == 'GW 1'].drop_duplicates(subset=['date'])

    # 6. Plotting
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.set_facecolor('#f5f1e9')
    ax.set_facecolor('#f5f1e9')
    
    # Plot rolling averages (uses the full 365-day team_df)
    ax.plot(team_df['date'], team_df['xG_For_Roll'], label=f'{rolling_window}-Game Rolling xG For', color='#0077b6', lw=2.5)
    ax.plot(team_df['date'], team_df['xG_Against_Roll'], label=f'{rolling_window}-Game Rolling xG Against', color='#e63946', lw=2.5)
    
    # --- UPDATED: Plot trendlines (uses the smaller current_season_df) ---
    ax.plot(current_season_df['date'], current_season_df['xG_For_Trend'], label='xG For Trend (Current Season)', color='#0077b6', linestyle='--', lw=1.5)
    ax.plot(current_season_df['date'], current_season_df['xG_Against_Trend'], label='xG Against Trend (Current Season)', color='#e63946', linestyle='--', lw=1.5)
    
    # 7. Plot season markers
    ylim_top = ax.get_ylim()[1]
    for _, row in season_starts.iterrows():
        ax.axvline(row['date'], color='gray', linestyle=':', lw=1.5, zorder=0)
        month = row['date'].month
        if month in [7, 8, 9]: 
            label = ' Regular Season Start'
        else: 
            label = ' Post Season Start'
        ax.text(row['date'] + pd.Timedelta(days=2), ylim_top, label, 
                ha='left', va='top', color='gray', rotation=90, fontsize=10)

    # 8. Styling
    ax.set_title(f"{selected_team} - Rolling xG (Last 365 Days)", fontsize=16, weight='bold')
    ax.set_ylabel(f'{rolling_window}-Game Rolling Avg')
    ax.legend(loc='upper left', frameon=False)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlim(one_year_ago, today)
    ax.set_ylim(bottom=0)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    
    plt.tight_layout()
    return fig

@st.cache_data
def calculate_expanded_team_stats(_all_match_data, _matches_summary_df):
    """
    Aggregates all per-match team stats from all_match_data into a
    season-long per-game average.
    """
    print("Calculating expanded team stats...") # Debug print
    
    # Use defaultdict for easy aggregation
    all_stats_agg = defaultdict(lambda: defaultdict(float))
    games_played = defaultdict(int)

    # Get team names from matches_summary
    team_name_map = {} # map matchId to (homeName, awayName)
    if 'homeTeamName' in _matches_summary_df.columns and 'awayTeamName' in _matches_summary_df.columns:
        for _, row in _matches_summary_df.iterrows():
            team_name_map[row['matchId']] = (row['homeTeamName'], row['awayTeamName'])
            
    if not team_name_map:
        st.error("Could not build team name map from matches_summary_df")
        return pd.DataFrame()

    # Loop through every match in the dataset
    for match_id, match_data in _all_match_data.items():
        if match_id not in team_name_map:
            continue # Skip if match not in summary
            
        home_team, away_team = team_name_map[match_id]
        
        # Increment games played
        games_played[home_team] += 1
        games_played[away_team] += 1
        
        # Check if 'team_stats' (the dict of DataFrames) exists
        if 'team_stats' in match_data and isinstance(match_data['team_stats'], dict):
            # Loop through each stat DataFrame (e.g., 'Passing', 'Defense')
            for category, df in match_data['team_stats'].items():
                
                # We expect df format: ['Metric', homeTeamName, awayTeamName]
                if isinstance(df, pd.DataFrame) and not df.empty and 'Metric' in df.columns:
                    if home_team in df.columns and away_team in df.columns:
                        try:
                            # Convert all stats to numeric, coercing errors
                            df[home_team] = pd.to_numeric(df[home_team], errors='coerce').fillna(0)
                            df[away_team] = pd.to_numeric(df[away_team], errors='coerce').fillna(0)
                            
                            # Loop through each metric row (e.g., 'Passes', 'Duels')
                            for _, row in df.iterrows():
                                metric_name = row['Metric']
                                # Add to the aggregate totals
                                all_stats_agg[home_team][metric_name] += row[home_team]
                                all_stats_agg[away_team][metric_name] += row[away_team]
                        except Exception as e:
                            print(f"Warning: Could not process df in match {match_id}. Error: {e}")
                    else:
                        print(f"Warning: Team columns '{home_team}' or '{away_team}' not in df for match {match_id}")

    # --- Convert aggregated totals to a DataFrame ---
    if not all_stats_agg:
        print("No stats aggregated.")
        return pd.DataFrame()
        
    stats_df = pd.DataFrame.from_dict(all_stats_agg, orient='index').fillna(0)
    
    # --- Normalize to Per-Game ---
    games_series = pd.Series(games_played, name='games')
    
    # Ensure we only divide teams that have games logged
    stats_df = stats_df.loc[games_series.index] 
    
    # Divide all stats by the number of games played
    stats_per_game_df = stats_df.div(games_series, axis=0)
    
    # Clean up any potential inf/-inf values
    stats_per_game_df.replace([np.inf, -np.inf], 0, inplace=True)
    
    print("✅ Expanded team stats calculated.")
    return stats_per_game_df.fillna(0)

# ==============================================================================
# 5. STREAMLIT APP UI
# ==============================================================================
st.title("Atlético CP Analysis") # You can change this title

# --- Load Data ---
raw_events_df, matches_summary_df, all_match_data, season_team_stats, player_minutes_df = load_data()


# --- Declare player_stats_with_scores_df globally for the app session ---
# This ensures it's accessible inside the plotting function
player_stats_with_scores_df = pd.DataFrame()


# --- Main App Logic ---
if raw_events_df is not None and matches_summary_df is not None and player_minutes_df is not None:
    # --- Sidebar for Navigation ---
    st.sidebar.title("Dashboard Controls")
    analysis_type = st.sidebar.radio("Choose Analysis Type", ('Match Analysis', 'Team Analysis', 'League Analysis', 'Player Profile', 'Player Comparison'))
    if analysis_type == 'Match Analysis':
        st.header("Match Analysis")
        
        # --- Match Selection (Using correct column names) ---
        if 'dateutc' in matches_summary_df.columns:
            matches_summary_df['display_date'] = pd.to_datetime(matches_summary_df['dateutc']).dt.strftime('%Y-%m-%d')
        else: matches_summary_df['display_date'] = 'Unknown Date'
        
        # Create a display-ready gameweek column
        matches_summary_df['gw_display'] = "GW " + matches_summary_df.get('gameweek', pd.Series(dtype='str')).fillna('?').astype(str)
        
        # Build the full display name using the new columns (GW: Teams (Score) - Date)
        matches_summary_df['display_name'] = matches_summary_df['gw_display'] + ": " + \
                                             matches_summary_df.get('homeTeamName', '?').fillna('?') + " vs " + \
                                             matches_summary_df.get('awayTeamName', '?').fillna('?') + \
                                             " (" + matches_summary_df.get('score', '?-?').fillna('?-?') + ") - " + \
                                             matches_summary_df['display_date']

        sort_key = 'dateutc' if 'dateutc' in matches_summary_df.columns else 'matchId'
        # Sort descending to show newest matches first
        matches_summary_df.sort_values(by=[sort_key, 'matchId'], inplace=True, ascending=False, na_position='last')
        
        selected_match_display = st.sidebar.selectbox("Select a Match", matches_summary_df['display_name'])
        selected_match_info = matches_summary_df[matches_summary_df['display_name'] == selected_match_display].iloc[0]
        selected_match_id = selected_match_info['matchId']
        
        st.header(f"Match Report: {selected_match_info['homeTeamName']} vs {selected_match_info['awayTeamName']}")
        
    

        match_data = all_match_data.get(selected_match_id)
        if match_data:
            st.subheader("Shot Maps")
            col1, col2 = st.columns(2)
            
            # --- Get the match events ONCE ---
            match_events_df = raw_events_df[raw_events_df['matchId'] == selected_match_id]

            with col1:
                # --- (REVERTED) ---
                st.pyplot(create_match_shotmap(match_events_df, selected_match_info, selected_match_info['homeTeamName']), use_container_width=True)
            with col2:
                # --- (REVERTED) ---
                st.pyplot(create_match_shotmap(match_events_df, selected_match_info, selected_match_info['awayTeamName']), use_container_width=True)

            # --- NEW: Shot Details Tables ---
            st.markdown("---") # Add a separator
            st.subheader("Shot Details")
            
            def get_shot_table(df, team_name):
                """Helper function to create the shot detail table."""
                shots = df[
                    (df.get('team.name') == team_name) & 
                    (df.get('type.primary').isin(['shot', 'penalty']))
                ].copy()
                
                if shots.empty:
                    return pd.DataFrame(columns=["#", "Shooter", "Minute", "Body Part", "xG", "PSxG"])

                # Select and rename columns
                # Use .get() for safety, in case a column is missing
                shots_table = pd.DataFrame()
                shots_table['Shooter'] = shots.get('player.name', 'N/A')
                shots_table['Minute'] = shots.get('minute', 0).astype(int)
                # Use .get() on the dictionary-like column 'shot.bodyPart'
                shots_table['Body Part'] = shots.get('shot.bodyPart', {}).apply(lambda x: x.get('name', 'unknown') if isinstance(x, dict) else x)
                shots_table['xG'] = shots.get('shot.xg', 0).fillna(0).round(2)
                shots_table['PSxG'] = shots.get('shot.postShotXg', 0).fillna(0).round(2)
                
                # Add the shot number (#)
                shots_table.reset_index(drop=True, inplace=True)
                shots_table.index = shots_table.index + 1
                shots_table.reset_index(inplace=True)
                shots_table = shots_table.rename(columns={'index': '#'})
                
                return shots_table.set_index('#')

            col1_table, col2_table = st.columns(2)
            with col1_table:
                st.markdown(f"**{selected_match_info['homeTeamName']}**")
                home_shots_table = get_shot_table(match_events_df, selected_match_info['homeTeamName'])
                st.dataframe(home_shots_table)

            with col2_table:
                st.markdown(f"**{selected_match_info['awayTeamName']}**")
                away_shots_table = get_shot_table(match_events_df, selected_match_info['awayTeamName'])
                st.dataframe(away_shots_table)
            # --- END NEW SECTION ---
            
            # --- xG Flowchart ---
            st.subheader("xG Flowchart")
            match_events_df = raw_events_df[raw_events_df['matchId'] == selected_match_id]
            if not match_events_df.empty:
                try:
                    fig_flowchart = plot_xg_flowchart(match_events_df, selected_match_info)
                    st.pyplot(fig_flowchart, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not generate xG flowchart: {e}")
            else:
                st.info("No event data found for flowchart.")

            st.subheader("Team Stats")
            if 'team_stats' in match_data and isinstance(match_data['team_stats'], dict) and match_data['team_stats']:
                for stat_category, df in match_data['team_stats'].items():
                    st.markdown(f"**{stat_category}**")
                    if isinstance(df, pd.DataFrame): st.dataframe(df)
                    else: st.warning(f"Data for '{stat_category}' is not a DataFrame.")
            else: st.warning("Team stats data not found.")

            st.subheader("Player Stats")
            if 'player_stats' in match_data and isinstance(match_data['player_stats'], dict) and 'home' in match_data['player_stats'] and 'away' in match_data['player_stats']:
                st.markdown(f"**{selected_match_info['homeTeamName']}**")
                if isinstance(match_data['player_stats']['home'], pd.DataFrame): st.dataframe(match_data['player_stats']['home'])
                else: st.warning("Home player stats data not a DataFrame.")
                st.markdown(f"**{selected_match_info['awayTeamName']}**")
                if isinstance(match_data['player_stats']['away'], pd.DataFrame): st.dataframe(match_data['player_stats']['away'])
                else: st.warning("Away player stats data not a DataFrame.")
            else: st.warning("Player stats data not found.")
        else:
             st.warning(f"No detailed match data found for Match ID {selected_match_id}.")


    elif analysis_type == 'Team Analysis':
        st.header("Team Analysis")
        all_teams_t = sorted(pd.concat([matches_summary_df.get('homeTeamName'), matches_summary_df.get('awayTeamName')]).dropna().unique())
        selected_team_t = st.sidebar.selectbox("Select a Team", all_teams_t, key="team_select_tab")
        st.header(f"Team Report: {selected_team_t}")
        
        stats_df_raw, stats_df_pct = calculate_all_team_radars_stats(raw_events_df, matches_summary_df)

        st.subheader("Team Style Radars (Percentile Ranks vs Liga 3)")
        if selected_team_t in stats_df_raw.index and selected_team_t in stats_df_pct.index:
            col_r1, col_r2, col_r3 = st.columns(3)
            offensive_params = ['Goals', 'xG', 'xG per Shot', 'Shots', 'Actions in Box', 'Passes into Box', 'Crosses', 'Dribbles']
            distribution_params = ['Passes', 'Progressive Passes', 'Directness', 'Ball Possession', 'Final 1/3 Entries', 'Losses']
            defensive_params = ['Goals Against', 'xG Against', 'xG per Shot Against', 'Shots Against', 'Aerial Duel Win %', 'Defensive Duel Win %', 'Interceptions', 'Fouls', 'PPDA']
            team_stats_raw = stats_df_raw.loc[selected_team_t]
            team_stats_pct = stats_df_pct.loc[selected_team_t]
            current_league = "Liga 3 Portugal"; current_season = "2025/26"
            
            with col_r1:
                st.markdown("**Offensive Radar**")
                valid_offensive_params = [p for p in offensive_params if p in team_stats_raw.index]
                if valid_offensive_params:
                     fig_off = plot_radar_chart(valid_offensive_params, team_stats_raw[valid_offensive_params].tolist(), team_stats_pct[valid_offensive_params].tolist(), selected_team_t, "Offensive Radar", '#e60000', league=current_league, season=current_season)
                     st.pyplot(fig_off, use_container_width=True)
            with col_r2:
                st.markdown("**Distribution Radar**")
                valid_distribution_params = [p for p in distribution_params if p in team_stats_raw.index]
                if valid_distribution_params:
                     raw_dist_values = team_stats_raw[valid_distribution_params].tolist()
                     try: poss_index = valid_distribution_params.index('Ball Possession'); raw_dist_values[poss_index] = f"{raw_dist_values[poss_index]:.0f}%"
                     except ValueError: pass
                     fig_dist = plot_radar_chart(valid_distribution_params, raw_dist_values, team_stats_pct[valid_distribution_params].tolist(), selected_team_t, "Distribution Radar", '#0077b6', league=current_league, season=current_season)
                     st.pyplot(fig_dist, use_container_width=True)
            with col_r3:
                st.markdown("**Defensive Radar**")
                valid_defensive_params = [p for p in defensive_params if p in team_stats_raw.index]
                if valid_defensive_params:
                     raw_def_values = team_stats_raw[valid_defensive_params].tolist()
                     try: aerial_idx = valid_defensive_params.index('Aerial Duel Win %'); raw_def_values[aerial_idx] = f"{raw_def_values[aerial_idx]:.0f}%"
                     except ValueError: pass
                     try: def_idx = valid_defensive_params.index('Defensive Duel Win %'); raw_def_values[def_idx] = f"{raw_def_values[def_idx]:.0f}%"
                     except ValueError: pass
                     fig_def = plot_radar_chart(valid_defensive_params, raw_def_values, team_stats_pct[valid_defensive_params].tolist(), selected_team_t, "Defensive Radar", '#52A736', league=current_league, season=current_season)
                     st.pyplot(fig_def, use_container_width=True)
        else:
            st.warning(f"Could not find calculated radar statistics for {selected_team_t}.")
        
        st.subheader("Season Shot Maps (Non-Penalty)")
        col1_shot, col2_shot = st.columns(2)
        with col1_shot:
            st.markdown(f"**Shots FOR {selected_team_t}**")
            # --- (REVERTED) ---
            fig_shots_for = create_season_shotmap(raw_events_df, selected_team_t)
            st.pyplot(fig_shots_for, use_container_width=True)
        with col2_shot:
            st.markdown(f"**Shots AGAINST {selected_team_t}**")
            # --- (REVERTED) ---
            fig_shots_against = create_season_shots_against_shotmap(raw_events_df, matches_summary_df, selected_team_t)
            st.pyplot(fig_shots_against, use_container_width=True)

        # --- ADD NEW SECTION HERE ---
        st.subheader("Rolling xG (Last 365 Days)")
        
        # Load the NEW historical data
        hist_events_df, hist_matches_df = load_historical_data()
        
        if hist_events_df is not None and hist_matches_df is not None:
            # Calculate the rolling data using the new historical files
            rolling_xg_data_for_plot = calculate_rolling_xg_data(hist_events_df, hist_matches_df)
            
            if not rolling_xg_data_for_plot.empty:
                try:
                    fig_rolling_xg = plot_rolling_xg(rolling_xg_data_for_plot, selected_team_t)
                    st.pyplot(fig_rolling_xg, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not generate rolling xG chart: {e}")
            else:
                st.warning("No data available to calculate rolling xG.")
        else:
            st.warning("Historical data files not loaded, cannot display rolling xG chart.")
        # --- END NEW SECTION ---

        st.subheader("Corner Kick Analysis")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("**Corners from Left Side**")
            fig_corner_left = plot_corner_analysis(raw_events_df, selected_team_t, 'left')
            st.pyplot(fig_corner_left, use_container_width=True)
        with col_c2:
            st.markdown("**Corners from Right Side**")
            fig_corner_right = plot_corner_analysis(raw_events_df, selected_team_t, 'right')
            st.pyplot(fig_corner_right, use_container_width=True)

        st.subheader("Season-Long Stats")
        if selected_team_t in season_team_stats and 'corners' in season_team_stats[selected_team_t]:
            st.markdown("**Corner Kick Summary**")
            st.dataframe(season_team_stats[selected_team_t]['corners'])
        else:
            st.write("No season-long stats available for this team.")
            

    elif analysis_type == 'League Analysis':
        st.header("League Analysis")
        
        # --- 1. ALL DATA CALCS ---
        stats_df_raw, stats_df_pct = calculate_all_team_radars_stats(raw_events_df, matches_summary_df)
        team_strength_df = calculate_team_strength(raw_events_df, matches_summary_df).copy()

        try:
            expanded_stats_df = calculate_expanded_team_stats(all_match_data, matches_summary_df)
            combined_stats_df = pd.merge(stats_df_raw, expanded_stats_df, left_index=True, right_index=True, how='outer').fillna(0)
        except Exception as e:
            st.warning(f"Could not calculate expanded match stats: {e}")
            combined_stats_df = stats_df_raw.copy() 

        # --- 2. Define Team Lists ---
        GROUP_B_TEAMS = [
            '1º Dezembro', 'Caldas', 'Sporting Covilhã', 'Mafra', 'União Santarém',
            'Amora', 'Académica', 'CF Os Belenenses', 'Lusitano Évora 1911', 'Atlético CP'
        ]
        valid_group_b_teams = [t for t in GROUP_B_TEAMS if t in combined_stats_df.index]
        
        ALL_TEAMS_TO_HIGHLIGHT = [ '1º Dezembro', 'Caldas', 'Sporting Covilhã', 'Mafra', 'União Santarém', 'Amora', 'Académica', 'CF Os Belenenses', 'Lusitano Évora 1911', 'Atlético CP', 'Fafe', 'Varzim', 'Atlético CP', 'Mafra', 'Caldas', 'Paredes', 'Sanjoanense', 'São João Ver', 'Amarante', 'Vitória Guimarães II', 'Trofense', 'Sporting Braga II', 'AD Marco 09' ]
        valid_all_teams = [t for t in ALL_TEAMS_TO_HIGHLIGHT if t in combined_stats_df.index]


        # --- 3. Group B Strength Chart (NOW FIRST) ---
        st.subheader("Team Strength Scatterplot (Liga 3 - Group B)")
        if not team_strength_df.empty:
            valid_group_b_strength_teams = [t for t in GROUP_B_TEAMS if t in team_strength_df.index]
            fig_group_b_strength = plot_team_strength(team_strength_df, teams_to_include=valid_group_b_strength_teams, icon_zoom=0.4)
            st.pyplot(fig_group_b_strength, use_container_width=True)
            with st.expander("View Group B Raw Strength Data"):
                if valid_group_b_strength_teams:
                    st.dataframe(team_strength_df.loc[valid_group_b_strength_teams, ['Attacking Strength', 'Defending Strength']].round(2))
        else:
            st.warning("Could not calculate team strength data for Group B.")

        
        # --- 4. Group B Custom Scatterplot (NOW SECOND) ---
        st.subheader("Group B Custom Scatterplot")
        if not combined_stats_df.empty and valid_group_b_teams:
            group_b_stats_df = combined_stats_df.loc[valid_group_b_teams]
            
            metrics_to_exclude = ['teamName', 'matchId', 'seasonId', 'teamId']
            available_metrics_gb = sorted([col for col in group_b_stats_df.columns if col not in metrics_to_exclude])
            
            col_x_gb, col_y_gb = st.columns(2)
            with col_x_gb:
                default_x_gb_index = available_metrics_gb.index('xG') if 'xG' in available_metrics_gb else 0
                x_metric_gb = st.selectbox("Select X-Axis Metric:", available_metrics_gb, index=default_x_gb_index, key='x_metric_group_b')
            with col_y_gb:
                default_y_gb_index = available_metrics_gb.index('xG Against') if 'xG Against' in available_metrics_gb else 1
                y_metric_gb = st.selectbox("Select Y-Axis Metric:", available_metrics_gb, index=default_y_gb_index, key='y_metric_group_b')
            
            col_inv_x_gb, col_inv_y_gb = st.columns(2)
            with col_inv_x_gb:
                invert_x_gb = st.checkbox("Invert X-Axis (Lower is Better)", key='invert_x_group_b')
            with col_inv_y_gb:
                default_invert_y_gb = 'Against' in y_metric_gb or 'PPDA' in y_metric_gb or 'Losses' in y_metric_gb
                invert_y_gb = st.checkbox("Invert Y-Axis (Lower is Better)", value=default_invert_y_gb, key='invert_y_group_b')
            
            if x_metric_gb and y_metric_gb:
                fig_custom_gb = plot_custom_scatter(group_b_stats_df, x_metric_gb, y_metric_gb, invert_x_gb, invert_y_gb)
                st.pyplot(fig_custom_gb, use_container_width=True)
        else:
            st.info("No data available for Group B custom plot.")


        # --- 5. All Teams Strength Chart (Unchanged) ---
        st.subheader("Team Strength Scatterplot (All Highlighted Teams)")
        if not team_strength_df.empty:
            valid_all_strength_teams = [t for t in ALL_TEAMS_TO_HIGHLIGHT if t in team_strength_df.index]
            fig_all_strength = plot_team_strength(team_strength_df, teams_to_include=valid_all_strength_teams)
            st.pyplot(fig_all_strength, use_container_width=True)
            with st.expander("View All Teams Raw Strength Data"):
                 st.dataframe(team_strength_df[['Attacking Strength', 'Defending Strength']].round(2))
        else:
            st.warning("Could not calculate team strength data.")

        
        # --- 6. All Teams Custom Scatterplot (Unchanged) ---
        st.subheader("All Teams Custom Scatterplot")
        if not combined_stats_df.empty:
            metrics_to_exclude = ['teamName', 'matchId', 'seasonId', 'teamId']
            available_metrics_all = sorted([col for col in combined_stats_df.columns if col not in metrics_to_exclude])
            
            col_x_all, col_y_all = st.columns(2)
            with col_x_all:
                default_x_all_index = available_metrics_all.index('xG') if 'xG' in available_metrics_all else 0
                x_metric_all = st.selectbox("Select X-Axis Metric:", available_metrics_all, index=default_x_all_index, key='x_metric_all')
            with col_y_all:
                default_y_all_index = available_metrics_all.index('xG Against') if 'xG Against' in available_metrics_all else 1
                y_metric_all = st.selectbox("Select Y-Axis Metric:", available_metrics_all, index=default_y_all_index, key='y_metric_all')
            
            col_inv_x_all, col_inv_y_all = st.columns(2)
            with col_inv_x_all:
                invert_x_all = st.checkbox("Invert X-Axis (Lower is Better)", key='invert_x_all')
            with col_inv_y_all:
                default_invert_y_all = 'Against' in y_metric_all or 'PPDA' in y_metric_all or 'Losses' in y_metric_all
                invert_y_all = st.checkbox("Invert Y-Axis (Lower is Better)", value=default_invert_y_all, key='invert_y_all')
            
            if x_metric_all and y_metric_all:
                fig_custom_all = plot_custom_scatter(combined_stats_df, x_metric_all, y_metric_all, invert_x_all, invert_y_all)
                st.pyplot(fig_custom_all, use_container_width=True)
            
            with st.expander("View All Teams Raw Radar & Expanded Stats Data"):
                st.dataframe(combined_stats_df.round(2))
        else:
            st.warning("Could not calculate raw league stats for custom plot.")

    
    # --- UPDATED: Renamed to Player Profile ---
    elif analysis_type == 'Player Profile':
        st.header("Player Profile")
        
        # --- 1. Load All Necessary Data ---
        player_details_df = load_player_details()
        
        try:
            player_stats_df = calculate_all_player_stats(raw_events_df, player_minutes_df)
            # --- NEW: Calculate percentiles ---
            player_stats_with_scores_df = calculate_player_percentiles_and_scores(
                player_stats_df, POSITION_GROUPS, WEIGHTS, INVERT_METRICS, min_minutes=90
            )
        except Exception as e:
            st.error(f"An error occurred calculating overall player stats: {e}")
            st.exception(e)
            player_stats_df = pd.DataFrame()
            player_stats_with_scores_df = pd.DataFrame()
            
        if player_stats_df.empty or player_details_df.empty or player_stats_with_scores_df.empty:
            st.warning("Player data not available. Please ensure all processing scripts have run and data is loaded.")
            st.stop()

        # --- 2. Player Selector ---
        st.sidebar.subheader("Player Analysis Options")
        
        # --- Use the percentile DF for the list, as it's pre-filtered by min_minutes ---
        player_list_df = player_stats_with_scores_df[['playerName', 'teamName', 'totalMinutes']].sort_values(by='totalMinutes', ascending=False)
        player_list_df['display_name'] = player_list_df['playerName'] + " (" + player_list_df['teamName'] + ", " + player_list_df['totalMinutes'].astype(int).astype(str) + " min)"
        
        selected_player_display = st.sidebar.selectbox("Select Player:", player_list_df['display_name'])
        selected_player_name = player_list_df[player_list_df['display_name'] == selected_player_display]['playerName'].values[0]
        
        try:
            # Get data from the 'with_scores' df
            player_data_row = player_stats_with_scores_df[player_stats_with_scores_df['playerName'] == selected_player_name]
            player_per_90_stats = player_data_row.iloc[0] # This is the series for the stats tables
            player_id = player_per_90_stats.get('playerId')
            player_bio = player_details_df.loc[player_id] if player_id in player_details_df.index else pd.Series(dtype='object')
            total_minutes = player_per_90_stats.get('totalMinutes', 0)
        except Exception as e:
            st.error(f"Could not load data for {selected_player_name}. Error: {e}")
            st.stop()
            
        # --- 3. Get Player's Match Log ---
        player_match_log_df = get_player_match_stats(selected_player_name, all_match_data, matches_summary_df)
        
        # --- 4. Display Player Bio ---
        st.header(f"{player_per_90_stats.get('playerName', 'N/A')}")
        
        col1_bio, col2_bio = st.columns([1, 3])
        with col1_bio:
            image_url = player_bio.get('imageDataURL', None)
            if image_url:
                st.image(image_url, width=150)
            else:
                st.image("https://t3.ftcdn.net/jpg/05/16/27/58/360_F_516275801_f3Fsp17x6HQK0xQgDQEELoGau0sJzEf4.jpg", width=150)

        with col2_bio:
            st.subheader("Player Information")
            bio_row1 = st.columns(4)
            bio_row1[0].metric("Team", player_per_90_stats.get('teamName', 'N/A'))
            bio_row1[1].metric("Position", player_per_90_stats.get('primaryPosition', 'N/A'))
            bio_row1[2].metric("Nationality", player_bio.get('passportArea', 'N/A'))
            
            age = _calculate_age(player_bio.get('birthDate'))
            age_display = f"{age:.1f}" if isinstance(age, float) else "N/A"
            bio_row1[3].metric("Age", age_display)
            
            bio_row2 = st.columns(4)
            foot_value = player_bio.get('foot')
            foot_display = "N/A"
            if foot_value and not pd.isna(foot_value):
                foot_display = foot_value.capitalize()
            bio_row2[0].metric("Foot", foot_display)
            bio_row2[1].metric("Height", f"{player_bio.get('height', 0)} cm")
            bio_row2[2].metric("Weight", f"{player_bio.get('weight', 0)} kg")
            bio_row2[3].metric("Birthplace", player_bio.get('birthArea', 'N/A'))
        
        st.divider()

        # --- 5. NEW: DISPLAY PLAYER RADAR ---
        st.subheader("Player Radar")
        primary_pos = player_per_90_stats.get('primaryPosition', 'N/A')
        eligible_groups = [pos_group for pos_group, pos_roles in POSITION_GROUPS.items() if primary_pos in pos_roles]

        if not eligible_groups:
            st.warning(f"No radar templates found for player's primary position: {primary_pos}")
        else:
            # Find best-fit archetype
            highest_score = -1; highest_scoring_group = None;
            for group in eligible_groups:
                score_col = group + '_Score'
                if score_col in player_data_row.columns:
                    player_score = player_data_row[score_col].values[0]
                    if player_score > highest_score:
                        highest_score = player_score; highest_scoring_group = group
            
            if highest_scoring_group is None: highest_scoring_group = eligible_groups[0]
            
            metrics_to_plot = list(WEIGHTS[highest_scoring_group].keys())
            metrics_to_plot = [m for m in metrics_to_plot if m in player_data_row.columns]
            
            position_data_for_dist = player_stats_with_scores_df[player_stats_with_scores_df['primaryPosition'].isin(POSITION_GROUPS[highest_scoring_group])]
            
            fig_radar = create_radar_with_distributions(
                player_data_row, # The 1-row DataFrame for the selected player
                metrics_to_plot, 
                highest_scoring_group, 
                eligible_groups,
                all_position_data=position_data_for_dist, # df for distribution plots
                full_df_for_ranking=player_stats_with_scores_df # full df for ranking
            )
            st.pyplot(fig_radar, use_container_width=True)

        st.divider()
        
        # --- 6. STATS TOGGLE ---
        st.subheader("Overall Season Stats")
        show_totals = st.toggle("Show Season Totals", value=False)
        stats_to_display = pd.Series(dtype='object')
        
        per_90_stats = player_per_90_stats.copy()
        
        if show_totals:
            st.text(f"Displaying TOTAL stats from {total_minutes:.0f} minutes played.")
            total_stats = per_90_stats.copy()
            rate_cols = [col for col in total_stats.index if '%' in col or 'per' in col or 'index' in col or 'Percentage' in col]
            
            for col in total_stats.index:
                if col not in rate_cols and pd.api.types.is_numeric_dtype(total_stats[col]):
                    total_val = (total_stats[col] * total_minutes) / 90
                    if col in ['xG', 'xA', 'xT', 'npxG', 'xAOP', 'xASP', 'psxG_faced', 'goalsPrevented']:
                         total_stats[col] = total_val
                    else:
                         total_stats[col] = np.round(total_val)
            
            for col in rate_cols:
                if col in per_90_stats.index:
                    total_stats[col] = per_90_stats[col]
            
            stats_to_display = total_stats
            
        else: # Show Per 90
            st.text(f"Displaying PER 90 stats from {total_minutes:.0f} minutes played.")
            stats_to_display = per_90_stats

        # --- 7. Display Stats (Using all global groups) ---
        stat_groups = {
            "Output": OUTPUT_METRICS,
            "Passing": PASSING_METRICS,
            "Defensive": DEFENSIVE_METRICS,
            "Dribbling": DRIBBLING_METRICS,
            "Goalkeeping": GOALKEEPING_METRICS
        }

        player_is_gk = (per_90_stats.get('primaryPosition', 'N/A') == 'GK')

        for group_name, group_metrics in stat_groups.items():
            
            if player_is_gk and group_name != 'Goalkeeping':
                continue 
            if not player_is_gk and group_name == 'Goalkeeping':
                continue
                
            if player_is_gk and group_name == 'Goalkeeping':
                group_metrics = GOALKEEPING_METRICS + ['GK Passes successful %', 'GK Long passes successful %']
                
            metrics_to_show = [m for m in group_metrics if m in stats_to_display.index]
            
            if metrics_to_show:
                default_expanded = (group_name == 'Output') 
                with st.expander(f"**{group_name} Stats**", expanded=default_expanded):
                    
                    stats_subset_series = stats_to_display[metrics_to_show]
                    stats_subset_series = stats_subset_series[stats_subset_series != 0]
                    
                    if stats_subset_series.empty:
                        st.text("No data for this category.")
                        continue
                        
                    stats_subset = stats_subset_series.to_frame(name='Value')
                    stats_subset['Value'] = stats_subset['Value'].apply(
                        lambda x: f"{x:.0f}" if (isinstance(x, (int, float)) and np.round(x) == x and '%' not in str(x)) else (f"{x:.2f}" if isinstance(x, (float)) else str(x))
                    )
                    st.dataframe(stats_subset, use_container_width=True)
        
        st.divider()
        
        # --- 8. Display Individual Match Stats (Unchanged) ---
        st.subheader("Individual Match Log")
        
        if player_match_log_df.empty:
            st.info("No individual match stats found for this player.")
        else:
            key_match_stats = ['Date', 'Match', 'Score', 'Minutes', 'Goals / xG', 'Actions / successful', 'Passes / accurate', 'Duels / won']
            cols_to_show = [c for c in key_match_stats if c in player_match_log_df.columns]
            st.dataframe(player_match_log_df[cols_to_show].set_index('Date'))
            with st.expander("View Full Match Log (All Stats)"):
                st.dataframe(player_match_log_df.set_index('Date'))

# --- NEW: Player Comparison Section ---
    elif analysis_type == 'Player Comparison':
        st.header("Player Comparison")

        # --- 1. Load Data ---
        try:
            player_stats_df = calculate_all_player_stats(raw_events_df, player_minutes_df)
            player_stats_with_scores_df = calculate_player_percentiles_and_scores(
                player_stats_df, POSITION_GROUPS, WEIGHTS, INVERT_METRICS, min_minutes=90
            )
        except Exception as e:
            st.error(f"An error occurred calculating player stats: {e}")
            st.stop()
            
        if player_stats_with_scores_df.empty:
            st.warning("No players found with sufficient minutes for comparison.")
            st.stop()

        # --- 2. Player Selectors ---
        st.sidebar.subheader("Comparison Options")
        
        player_list_df = player_stats_with_scores_df[['playerName', 'teamName', 'totalMinutes']].sort_values(by='totalMinutes', ascending=False)
        player_list_df['display_name'] = player_list_df['playerName'] + " (" + player_list_df['teamName'] + ", " + player_list_df['totalMinutes'].astype(int).astype(str) + " min)"
        
        # Player A
        selected_player_a_display = st.sidebar.selectbox(
            "Select Player A:", 
            player_list_df['display_name'], 
            index=0 # Default to first player
        )
        selected_player_a_name = player_list_df[player_list_df['display_name'] == selected_player_a_display]['playerName'].values[0]
        player_a_data = player_stats_with_scores_df[player_stats_with_scores_df['playerName'] == selected_player_a_name]

        # Player B
        selected_player_b_display = st.sidebar.selectbox(
            "Select Player B:", 
            player_list_df['display_name'],
            index=1 # Default to second player
        )
        selected_player_b_name = player_list_df[player_list_df['display_name'] == selected_player_b_display]['playerName'].values[0]
        player_b_data = player_stats_with_scores_df[player_stats_with_scores_df['playerName'] == selected_player_b_name]

        # --- 3. Template Selector ---
        # Get all possible templates (archetypes)
        all_templates = sorted(list(POSITION_GROUPS.keys()))
        # Find a good default: Player A's best-fit template
        primary_pos_a = player_a_data['primaryPosition'].values[0]
        eligible_groups_a = [pos_group for pos_group, pos_roles in POSITION_GROUPS.items() if primary_pos_a in pos_roles]
        
        highest_score = -1; default_template = all_templates[0]
        for group in eligible_groups_a:
            score_col = group + '_Score'
            if score_col in player_a_data.columns:
                player_score = player_a_data[score_col].values[0]
                if player_score > highest_score:
                    highest_score = player_score; default_template = group
        
        default_index = all_templates.index(default_template) if default_template in all_templates else 0
        
        selected_template = st.sidebar.selectbox(
            "Select Comparison Template:",
            all_templates,
            index=default_index
        )

        # --- 4. Plot Radar ---
        st.subheader(f"Comparing: {selected_player_a_name} vs. {selected_player_b_name}")
        
        metrics_to_plot = list(WEIGHTS[selected_template].keys())
        # Ensure metrics exist in the base data
        metrics_to_plot = [m for m in metrics_to_plot if m in player_stats_with_scores_df.columns]
        
        fig = plt.figure(figsize=(14, 7))
        ax_radar = plt.subplot(111, polar=True)
        
        plot_comparison_radar(
            ax_radar,
            player_a_data,
            player_b_data,
            metrics_to_plot,
            selected_template
        )
        
        st.pyplot(fig, use_container_width=True)

else:
    st.error("Data files not loaded. Please run `process_data.py` locally and ensure all artifacts are pushed to GitHub.")