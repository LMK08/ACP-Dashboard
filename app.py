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
from matplotlib.gridspec import GridSpec # For player radar charts
from collections import defaultdict # For player radar calculations
import seaborn as sns # For player radar distributions

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
            
        # --- NEW: Load Player Minutes & Position Data ---
        with open('player_minutes_and_positions.pkl', 'rb') as f:
            player_minutes_df = pickle.load(f)

        return raw_events_df, matches_summary_df, all_match_data, season_team_stats, player_minutes_df
    
    except FileNotFoundError as e:
        st.error(f"❌ Error: A data file was not found. Please run `process_data.py` (including the new player minutes step) first. Missing file: {e.filename}")
        return None, None, None, None, None
    except Exception as e:
        st.error(f"An error occurred loading data: {e}")
        return None, None, None, None, None

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
GOALKEEPING_METRICS = ['shotsOnTargetAgainst', 'goalsConceded', 'exits', 'saves', 'goalsPrevented', 'goalsPreventedPerSOT', 'savePercentage', 'recoveries_gk', 'passes_gk', 'passesSuccessful_gk', 'longPasses_gk', 'longPassesSuccessful_gk']
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
def calculate_and_merge(base_df, events_df, stat_name, filter_condition):
    """
    Helper function from notebook Cell 11.
    Calculates a stat based on a filter and merges it into the base DataFrame.
    """
    # Ensure filter condition is a Series with the same index as events_df
    if not isinstance(filter_condition, pd.Series):
        # This handles cases like pd.Series(True, index=pass_events.index)
        # We need to reindex it to the full events_df
        filter_condition = filter_condition.reindex(events_df.index, fill_value=False)

    safe_condition = filter_condition & events_df['player.id'].notna()
    
    # Group by player.id (which should be numeric)
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
        condition = condition & and_condition.reindex(condition.index, fill_value=False)
        
    return calculate_and_merge(base_df, events_df, stat_name, condition)


# --- Player Radar Data Calculation (V-ROBUST) ---
@st.cache_data
def calculate_player_radar_data(_raw_events_df, _player_minutes_df):
    """
    Runs the entire player-level data processing pipeline from the notebooks.
    V-ROBUST: Skips one-hot encoding and uses robust list-checking.
    """
    print("--- STARTING: Player radar data calculation (V-Robust) ---")
    
    events_df = _raw_events_df.copy()
    combined_df = _player_minutes_df.copy()

    print("Step 1: Skipping one-hot encoding (using raw list search).")
    events_df['player.id'] = pd.to_numeric(events_df['player.id'], errors='coerce')
    events_df = events_df.dropna(subset=['player.id'])
    events_df['player.id'] = events_df['player.id'].astype(int)

    print("Step 2: Calculating npxG, xAOP, xASP...")
    try:
        shots_df = events_df[(events_df['shot.xg'].notna()) & (events_df['player.id'].notna()) & (events_df['type.primary'] != 'penalty')].copy()
        npxg_totals = shots_df.groupby('player.id')['shot.xg'].sum().reset_index().rename(columns={'shot.xg': 'npxG'})
        events_df['shot_event_id'] = np.where(events_df['shot.xg'].notna(), events_df['id'], np.nan)
        events_df['next_shot_id'] = events_df.groupby('matchId')['shot_event_id'].bfill()
        shot_xg_map = shots_df.set_index('id')['shot.xg'].to_dict()
        assists_df = events_df[events_df.get('type.secondary', pd.Series(dtype='object')).apply(lambda x: isinstance(x, (list, np.ndarray)) and 'shot_assist' in x)].copy()
        assists_df['player.id'] = assists_df['player.id'].astype(int)
        assists_df['xA'] = assists_df['next_shot_id'].map(shot_xg_map)
        set_piece_types = ['corner', 'free_kick', 'throw_in', 'goal_kick']
        assists_df['assist_type'] = np.where(assists_df['type.primary'].isin(set_piece_types), 'xASP', 'xAOP')
        xa_split_totals = assists_df.groupby(['player.id', 'assist_type'])['xA'].sum()
        xa_final_df = xa_split_totals.unstack(fill_value=0).reset_index()
        final_stats_df = pd.merge(npxg_totals, xa_final_df, on='player.id', how='outer')
        final_stats_df['playerId'] = final_stats_df['player.id']
        combined_df = pd.merge(combined_df, final_stats_df, on='playerId', how='left')
        if 'player.id_x' in combined_df.columns: combined_df = combined_df.drop(columns=['player.id_x'])
        if 'player.id_y' in combined_df.columns: combined_df = combined_df.drop(columns=['player.id_y'])
        if 'player.id' in combined_df.columns: combined_df = combined_df.rename(columns={'player.id': 'player.id_temp'})
    except Exception as e:
        print(f"  -> ❌ ERROR (Step 2): Failed calculating xA/npxG: {e}")
        if 'npxG' not in combined_df.columns: combined_df['npxG'] = 0
        if 'xAOP' not in combined_df.columns: combined_df['xAOP'] = 0
        if 'xASP' not in combined_df.columns: combined_df['xASP'] = 0
        if 'xA' not in combined_df.columns: combined_df['xA'] = 0

    print("Step 3: Calculating Deep Completions and Progressive Passes...")
    try:
        passes_df = events_df[(events_df['type.primary'] == 'pass') & (events_df.get('pass.accurate') == True)].dropna(subset=['location.x', 'pass.endLocation.x', 'player.id']).copy()
        passes_df['end_x_m'] = passes_df['pass.endLocation.x'] * 1.05
        passes_df['end_y_m'] = passes_df['pass.endLocation.y'] * 0.68
        passes_df['dist_to_goal_center'] = np.sqrt((passes_df['end_x_m'] - 105)**2 + (passes_df['end_y_m'] - 34)**2)
        passes_df['is_cross'] = passes_df.get('type.secondary', pd.Series(dtype='object')).apply(lambda x: isinstance(x, (list, np.ndarray)) and 'cross' in x)
        passes_df['is_deep_completion'] = (passes_df['dist_to_goal_center'] <= 20) & (passes_df['is_cross'] == False)
        deep_completions = passes_df.groupby('player.id')['is_deep_completion'].sum().reset_index().rename(columns={'is_deep_completion': 'Deep Completions'})
        start_x = passes_df['location.x']; end_x = passes_df['pass.endLocation.x']
        cond1 = (start_x < 50) & (end_x < 50) & (end_x - start_x >= 30); cond2 = (start_x < 50) & (end_x >= 50) & (end_x - start_x >= 15); cond3 = (start_x >= 50) & (end_x >= 50) & (end_x - start_x >= 10)
        passes_df['is_progressive_pass'] = cond1 | cond2 | cond3
        progressive_passes = passes_df.groupby('player.id')['is_progressive_pass'].sum().reset_index().rename(columns={'is_progressive_pass': 'Progressive Passes'})
        new_metrics_df = pd.merge(deep_completions, progressive_passes, on='player.id', how='outer')
        combined_df = pd.merge(combined_df, new_metrics_df, left_on='playerId', right_on='player.id', how='left')
        if 'player.id' in combined_df.columns: combined_df = combined_df.drop(columns=['player.id'])
    except Exception as e:
        print(f"  -> ❌ ERROR (Step 3): Failed calculating passing stats: {e}")
        if 'Deep Completions' not in combined_df.columns: combined_df['Deep Completions'] = 0
        if 'Progressive Passes' not in combined_df.columns: combined_df['Progressive Passes'] = 0

    print("Step 4: Calculating comprehensive counting stats (Robust)...")
    try:
        player_stats_df = events_df.dropna(subset=['player.id', 'player.name'])[['player.id', 'player.name']].drop_duplicates()
        player_stats_df['player.id'] = player_stats_df['player.id'].astype(int)
        player_stats_df = player_stats_df.set_index('player.id')
        
        # -- Basic Event Counts --
        player_stats_df = calculate_and_merge(player_stats_df, events_df, 'Goals', bool_condition=(events_df.get('shot.isGoal') == True))
        player_stats_df = calculate_and_merge_list(player_stats_df, events_df, 'Assists', 'assist')
        player_stats_df = calculate_and_merge(player_stats_df, events_df, 'Shots', primary_type='shot')
        player_stats_df = calculate_and_merge(player_stats_df, events_df, 'Shots on target', primary_type='shot', bool_condition=(events_df.get('shot.onTarget') == True))
        player_stats_df = calculate_and_merge(player_stats_df, events_df, 'Interceptions', primary_type='interception')
        player_stats_df = calculate_and_merge(player_stats_df, events_df, 'Clearances', primary_type='clearance')
        player_stats_df = calculate_and_merge(player_stats_df, events_df, 'Fouls', primary_type='infraction')
        player_stats_df = calculate_and_merge(player_stats_df, events_df, 'Offsides', primary_type='offside')
        player_stats_df = calculate_and_merge(player_stats_df, events_df, 'Yellow cards', bool_condition=(events_df.get('infraction.yellowCard') == True))
        player_stats_df = calculate_and_merge(player_stats_df, events_df, 'Red cards', bool_condition=(events_df.get('infraction.redCard') == True))
        player_stats_df = calculate_and_merge_list(player_stats_df, events_df, 'Touches in penalty area', 'touch_in_box')
        player_stats_df = calculate_and_merge_list(player_stats_df, events_df, 'Progressive runs', 'progressive_run')
        player_stats_df = calculate_and_merge_list(player_stats_df, events_df, 'Fouls suffered', 'foul_suffered')
        player_stats_df = calculate_and_merge_list(player_stats_df, events_df, 'Second assists', 'second_assist')
        
        # -- Passing Metrics --
        pass_events = events_df[events_df.get('type.primary') == 'pass'].copy()
        pass_events['player.id'] = pass_events['player.id'].astype(int); pass_accurate_condition = pass_events.get('pass.accurate') == True
        player_stats_df = calculate_and_merge(player_stats_df, pass_events, 'Passes', primary_type='pass')
        player_stats_df = calculate_and_merge(player_stats_df, pass_events, 'Passes successful', primary_type='pass', bool_condition=pass_accurate_condition)
        player_stats_df = calculate_and_merge_list(player_stats_df, pass_events, 'Long passes', 'long_pass')
        player_stats_df = calculate_and_merge_list(player_stats_df, pass_events, 'Long passes successful', 'long_pass', and_condition=pass_accurate_condition)
        player_stats_df = calculate_and_merge_list(player_stats_df, pass_events, 'Crosses', 'cross')
        player_stats_df = calculate_and_merge_list(player_stats_df, pass_events, 'Crosses successful', 'cross', and_condition=pass_accurate_condition)
        player_stats_df = calculate_and_merge_list(player_stats_df, pass_events, 'Through passes', 'through_pass')
        player_stats_df = calculate_and_merge_list(player_stats_df, pass_events, 'Through passes successful', 'through_pass', and_condition=pass_accurate_condition)
        player_stats_df = calculate_and_merge_list(player_stats_df, pass_events, 'Passes to final third', 'pass_to_final_third')
        player_stats_df = calculate_and_merge_list(player_stats_df, pass_events, 'Passes to final third successful', 'pass_to_final_third', and_condition=pass_accurate_condition)
        player_stats_df = calculate_and_merge_list(player_stats_df, pass_events, 'Forward passes', 'forward_pass')
        player_stats_df = calculate_and_merge_list(player_stats_df, pass_events, 'Forward passes successful', 'forward_pass', and_condition=pass_accurate_condition)
        player_stats_df = calculate_and_merge_list(player_stats_df, pass_events, 'Back passes', 'back_pass')
        player_stats_df = calculate_and_merge_list(player_stats_df, pass_events, 'Back passes successful', 'back_pass', and_condition=pass_accurate_condition)
        player_stats_df = calculate_and_merge_list(player_stats_df, pass_events, 'Passes to penalty area', 'pass_to_penalty_area')
        player_stats_df = calculate_and_merge_list(player_stats_df, pass_events, 'Passes to penalty area successful', 'pass_to_penalty_area', and_condition=pass_accurate_condition)
        
        # -- Dueling & Defensive Metrics --
        duel_events = events_df[events_df.get('type.primary') == 'duel'].copy()
        duel_events['player.id'] = duel_events['player.id'].astype(int)
        player_stats_df = calculate_and_merge(player_stats_df, duel_events, 'Duels', primary_type='duel')
        player_stats_df = calculate_and_merge(player_stats_df, duel_events, 'Duels successful', primary_type='duel', bool_condition=(duel_events.get('groundDuel.keptPossession') == True) | (duel_events.get('groundDuel.recoveredPossession') == True) | (duel_events.get('aerialDuel.firstTouch') == True))
        player_stats_df = calculate_and_merge_list(player_stats_df, duel_events, 'Aerial duels', 'aerial_duel')
        player_stats_df = calculate_and_merge_list(player_stats_df, duel_events, 'Aerial duels successful', 'aerial_duel', and_condition=(duel_events.get('aerialDuel.firstTouch') == True))
        player_stats_df = calculate_and_merge_list(player_stats_df, duel_events, 'Defensive duels', 'defensive_duel')
        player_stats_df = calculate_and_merge_list(player_stats_df, duel_events, 'Defensive duels successful', 'defensive_duel', bool_condition=(duel_events.get('groundDuel.recoveredPossession') == True))
        player_stats_df = calculate_and_merge_list(player_stats_df, duel_events, 'Offensive duels', 'offensive_duel')
        player_stats_df = calculate_and_merge_list(player_stats_df, duel_events, 'Offensive duels successful', 'offensive_duel', bool_condition=(duel_events.get('groundDuel.progressedWithBall') == True))
        player_stats_df = calculate_and_merge_list(player_stats_df, duel_events, 'Sliding tackles', 'sliding_tackle')
        player_stats_df = calculate_and_merge_list(player_stats_df, duel_events, 'Sliding tackles successful', 'sliding_tackle', bool_condition=(duel_events.get('groundDuel.recoveredPossession') == True))
        player_stats_df = calculate_and_merge_list(player_stats_df, duel_events, 'Dribbles', 'dribble')
        player_stats_df = calculate_and_merge_list(player_stats_df, duel_events, 'Dribbles successful', 'dribble', bool_condition=(duel_events.get('groundDuel.takeOn') == True))
        
        player_stats_df = calculate_and_merge_list(player_stats_df, events_df, 'Losses', 'loss')
        player_stats_df = calculate_and_merge_list(player_stats_df, events_df, 'Losses Opp Half', 'loss', and_condition=(events_df.get('location.x', 0) >= 50))
        player_stats_df = calculate_and_merge_list(player_stats_df, events_df, 'Recoveries', 'recovery')
        player_stats_df = calculate_and_merge_list(player_stats_df, events_df, 'Recoveries Opp Half', 'recovery', and_condition=(events_df.get('location.x', 0) >= 50))
        player_stats_df = calculate_and_merge_list(player_stats_df, events_df, 'Counterpressing Recoveries', 'counterpressing_recovery')
        
        xg_series = events_df.groupby(events_df['player.id'].astype(int))['shot.xg'].sum(); xg_series.name = 'xG'; player_stats_df = player_stats_df.merge(xg_series, left_index=True, right_index=True, how='left')
        player_stats_df = player_stats_df.fillna(0); cols_to_int = [col for col in player_stats_df.columns if col not in ['player.name', 'xG']]; player_stats_df[cols_to_int] = player_stats_df[cols_to_int].astype(int)
        
        def safe_divide(n, d): return (n / d * 100).replace([np.inf, -np.inf], 0).fillna(0)
        player_stats_df['xG per Shot'] = (player_stats_df['xG'] / player_stats_df['Shots']).replace([np.inf, -np.inf], 0).fillna(0)
        player_stats_df['Passes successful %'] = safe_divide(player_stats_df['Passes successful'], player_stats_df['Passes'])
        player_stats_df['Long passes successful %'] = safe_divide(player_stats_df['Long passes successful'], player_stats_df['Long passes'])
        player_stats_df['Crosses successful %'] = safe_divide(player_stats_df['Crosses successful'], player_stats_df['Crosses'])
        player_stats_df['Dribbles successful %'] = safe_divide(player_stats_df['Dribbles successful'], player_stats_df['Dribbles'])
        player_stats_df['Duels successful %'] = safe_divide(player_stats_df['Duels successful'], player_stats_df['Duels'])
        player_stats_df['Aerial duels successful %'] = safe_divide(player_stats_df['Aerial duels successful'], player_stats_df['Aerial duels'])
        player_stats_df['Offensive duels successful %'] = safe_divide(player_stats_df['Offensive duels successful'], player_stats_df['Offensive duels'])
        player_stats_df['Defensive duels successful %'] = safe_divide(player_stats_df['Defensive duels successful'], player_stats_df['Defensive duels'])
        player_stats_df['Sliding tackles successful %'] = safe_divide(player_stats_df['Sliding tackles successful'], player_stats_df['Sliding tackles'])
        successful_attacking_actions = player_stats_df['Shots on target'] + player_stats_df['Crosses successful'] + player_stats_df['Dribbles successful']
        player_stats_df['Loss index'] = (player_stats_df['Losses'] / successful_attacking_actions).replace([np.inf, -np.inf], 0).fillna(0)
        player_stats_df = player_stats_df.reset_index().rename(columns={'index':'player.id'})

        combined_df = pd.merge(combined_df, player_stats_df, left_on='playerId', right_on='player.id', how='left')
        if 'player.id' in combined_df.columns: combined_df = combined_df.drop(columns=['player.id'])
        
    except Exception as e:
        print(f"  -> ❌ ERROR (Cell 11-12): Failed calculating counting stats: {e}")

    # --- Cell 14 & 15: Calculate Goalkeeper Stats ---
    print("Step 5: Calculating Goalkeeper stats...")
    try:
        gk_ids = events_df[events_df.get('player.position') == 'GK']['player.id'].dropna().unique().astype(int)
        gk_events_df = events_df[events_df['player.id'].isin(gk_ids)].copy()
        
        shots_faced_df = events_df[(events_df.get('type.primary') == 'shot') & (events_df.get('shot.onTarget') == True) & (events_df.get('shot.goalkeeper.id').notna())].copy()
        shots_faced_df['shot.goalkeeper.id'] = shots_faced_df['shot.goalkeeper.id'].astype(int)
        gk_shot_stopping_stats = shots_faced_df.groupby('shot.goalkeeper.id').agg(shotsOnTargetAgainst=('shot.isGoal', 'count'), goalsConceded=('shot.isGoal', 'sum'), psxG_faced=('shot.postShotXg', 'sum')).reset_index().rename(columns={'shot.goalkeeper.id': 'player.id'})
        if not gk_shot_stopping_stats.empty:
            gk_shot_stopping_stats['goalsPrevented'] = gk_shot_stopping_stats['psxG_faced'] - gk_shot_stopping_stats['goalsConceded']
            gk_shot_stopping_stats['goalsPreventedPerSOT'] = (gk_shot_stopping_stats['goalsPrevented'] / gk_shot_stopping_stats['shotsOnTargetAgainst']).fillna(0)
        else:
            gk_shot_stopping_stats = gk_shot_stopping_stats.reindex(columns=['player.id', 'shotsOnTargetAgainst', 'goalsConceded', 'psxG_faced', 'goalsPrevented', 'goalsPreventedPerSOT']).fillna(0)

        gk_events_df['player.id'] = gk_events_df['player.id'].astype(int)
        exits = gk_events_df[gk_events_df['type.primary'] == 'goalkeeper_exit'].groupby('player.id').size().reset_index(name='exits')
        recoveries_gk = gk_events_df[gk_events_df.get('type.secondary', pd.Series(dtype='object')).apply(lambda x: isinstance(x, (list, np.ndarray)) and 'recovery' in x)].groupby('player.id').size().reset_index(name='recoveries_gk')
        gk_passes = gk_events_df[gk_events_df['type.primary'] == 'pass']
        passes_total_gk = gk_passes.groupby('player.id').size().reset_index(name='passes_gk')
        passes_succ_gk = gk_passes[gk_passes['pass.accurate'] == True].groupby('player.id').size().reset_index(name='passesSuccessful_gk')
        long_passes_total_gk = gk_passes[gk_passes.get('type.secondary', pd.Series(dtype='object')).apply(lambda x: isinstance(x, (list, np.ndarray)) and 'long_pass' in x)].groupby('player.id').size().reset_index(name='longPasses_gk')
        long_passes_succ_gk = gk_passes[gk_passes.get('type.secondary', pd.Series(dtype='object')).apply(lambda x: isinstance(x, (list, np.ndarray)) and 'long_pass' in x) & (gk_passes['pass.accurate'] == True)].groupby('player.id').size().reset_index(name='longPassesSuccessful_gk')

        gk_report_df = pd.DataFrame({'player.id': gk_ids}); gk_report_df = pd.merge(gk_report_df, gk_shot_stopping_stats, on='player.id', how='left'); gk_report_df = pd.merge(gk_report_df, exits, on='player.id', how='left'); gk_report_df = pd.merge(gk_report_df, recoveries_gk, on='player.id', how='left'); gk_report_df = pd.merge(gk_report_df, passes_total_gk, on='player.id', how='left'); gk_report_df = pd.merge(gk_report_df, passes_succ_gk, on='player.id', how='left'); gk_report_df = pd.merge(gk_report_df, long_passes_total_gk, on='player.id', how='left'); gk_report_df = pd.merge(gk_report_df, long_passes_succ_gk, on='player.id', how='left')
        combined_df = pd.merge(combined_df, gk_report_df, left_on='playerId', right_on='player.id', how='left')
        if 'player.id' in combined_df.columns: combined_df = combined_df.drop(columns=['player.id'])
    except Exception as e:
        print(f"  -> ❌ ERROR (Cell 14-15): Failed calculating GK stats: {e}")

    # --- Cell 18-20: Calculate xT (Expected Threat) ---
    print("Step 6: Calculating Expected Threat (xT)...")
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
        move_df['player.id'] = move_df['player.id'].astype(int)
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
        print(f"  -> ❌ ERROR (Cell 18-20): Failed calculating xT: {e}")
        if 'xT' not in combined_df.columns: combined_df['xT'] = 0

    # --- Cell 21: Normalize to Per 90 ---
    print("Step 7: Normalizing stats to per 90...")
    try:
        combined_df = combined_df.fillna(0)
        combined_df.rename(columns={'recoveries': 'recoveries_gk', 'passes': 'passes_gk', 'passesSuccessful': 'passesSuccessful_gk', 'longPasses': 'longPasses_gk', 'longPassesSuccessful': 'longPassesSuccessful_gk'}, inplace=True, errors='ignore')
        metrics_to_normalize = ['npxG', 'xAOP', 'xASP', 'xT', 'Goals', 'Assists', 'Shots', 'Shots on target', 'Interceptions', 'Clearances', 'Fouls', 'Offsides', 'Yellow cards', 'Red cards', 'Touches in penalty area', 'Progressive runs', 'Fouls suffered', 'Second assists', 'Passes', 'Passes successful', 'Long passes', 'Long passes successful', 'Crosses', 'Crosses successful', 'Through passes', 'Through passes successful', 'Passes to final third', 'Passes to final third successful', 'Forward passes', 'Forward passes successful', 'Back passes', 'Back passes successful', 'Passes to penalty area', 'Passes to penalty area successful', 'Duels', 'Duels successful', 'Aerial duels', 'Aerial duels successful', 'Defensive duels', 'Defensive duels successful', 'Offensive duels', 'Offensive duels successful', 'Sliding tackles', 'Sliding tackles successful', 'Dribbles', 'Dribbles successful', 'Losses', 'Losses Opp Half', 'Recoveries', 'Recoveries Opp Half', 'Counterpressing Recoveries', 'xG', 'Deep Completions', 'Progressive Passes', 'shotsOnTargetAgainst', 'goalsConceded', 'psxG_faced', 'goalsPrevented', 'exits', 'recoveries_gk', 'passes_gk', 'passesSuccessful_gk', 'longPasses_gk', 'longPassesSuccessful_gk']
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
    except Exception as e:
        print(f"  -> ❌ ERROR (Cell 21): Failed normalizing to per 90: {e}")

    print("--- FINISHED: Player radar data calculation ---")
    return combined_df.fillna(0)


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


# --- PLAYER RADAR PLOTTING FUNCTIONS ---
def _create_base_radar_chart(fig, ax, player_data, metrics, position, eligible_groups, full_df_for_ranking=None):
    """Helper function to create the base radar chart (Cell 23 logic)."""
    
    num_metrics = len(metrics)
    angles = [n / float(num_metrics) * 2 * pi for n in range(num_metrics)]
    angles += angles[:1]
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([]) 

    values = [player_data[metric + '_percentile'].values[0] for metric in metrics if metric + '_percentile' in player_data.columns]
    if len(values) != num_metrics:
        print(f"Warning: Mismatch in metrics ({num_metrics}) vs values ({len(values)}) for {player_data['playerName'].values[0]}")
        missing_pct = [m + '_percentile' for m in metrics if m + '_percentile' not in player_data.columns]
        print(f"  -> Missing percentile columns: {missing_pct}")
        values.extend([0] * (num_metrics - len(values)))
    values += values[:1] 

    ax.plot(angles, values, linewidth=1, linestyle='solid', color='#0077b6') 
    ax.fill(angles, values, '#0077b6', alpha=0.1) 

    category_colors = {'output': 'green', 'passing': 'orange', 'defensive': 'red', 'dribbling': 'purple', 'goalkeeping': 'cyan'}

    for i, metric in enumerate(metrics):
        angle_rad = angles[i]
        label = f"{player_data[metric].values[0]:.2f}"
        ax.text(angle_rad, 85, label, size=8, ha='center', va='center', color='blue')

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
                    # Check if 'Score' column exists before ranking
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
    
    # --- CORRECTED CALL: Pass ax_radar to 'ax' keyword ---
    _create_base_radar_chart(fig, ax_radar, player_data, metrics, position, eligible_groups, full_df_for_ranking=full_df_for_ranking)
    
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

# --- Shotmap Functions (Unchanged) ---
def create_match_shotmap(match_events_df, match_info, team_to_analyze):
    # (This is the full function from the previous step)
    team_shots_df = match_events_df[(match_events_df.get('team.name') == team_to_analyze) & (match_events_df.get('type.primary') == 'shot')].copy().reset_index(drop=True)
    if team_shots_df.empty:
        fig, ax = plt.subplots(figsize=(10, 8)); fig.set_facecolor('#f5f1e9'); ax.set_facecolor('#f5f1e9'); ax.text(0.5, 0.5, 'No shots found for this team in this match.', ha='center', va='center', fontsize=12); ax.axis('off'); return fig
    home_team = match_info.get('homeTeamName', '?'); away_team = match_info.get('awayTeamName', '?'); opponent = away_team if team_to_analyze == home_team else home_team
    fig = plt.figure(figsize=(12, 12)); fig.set_facecolor('#f5f1e9'); pitch = Pitch(pitch_type='wyscout', pitch_color='#f5f1e9', line_color='black', line_zorder=2, half=True); ax_pitch = fig.add_subplot(); pitch.draw(ax=ax_pitch)
    XG_MAX = 0.8; colors = ["#03045e", "#ade8f4", "#fff3b0", "#ff8c00", "#e63946", "#800f2f"]; nodes = [0.0, 0.1 / XG_MAX, 0.2 / XG_MAX, 0.4 / XG_MAX, 0.6 / XG_MAX, 1.0]; cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", list(zip(nodes, colors)))
    for index, shot in team_shots_df.iterrows():
        x = shot.get('location.x'); y = shot.get('location.y'); xg = shot.get('shot.xg')
        if pd.isna(x) or pd.isna(y) or pd.isna(xg): continue # Skip shots with missing data
        is_goal = shot.get('shot.isGoal') == True; is_on_target = shot.get('shot.onTarget') == True
        color = cmap(min(xg / XG_MAX, 1.0)); edge_color = 'gray'; line_width = 1.5
        if is_goal: edge_color = 'green'; line_width = 2.5
        elif is_on_target: edge_color = 'black'; line_width = 2.5
        pitch.scatter(x, y, s=400, facecolor=color, edgecolor=edge_color, linewidth=line_width, ax=ax_pitch, zorder=3)
        pitch.text(x, y, str(index + 1), ax=ax_pitch, ha='center', va='center', fontsize=9, color='white', zorder=4)
    subtitle = f"vs. {opponent} | Score: {match_info.get('score', '?-?')} | xG: {team_shots_df['shot.xg'].sum():.2f}"; ax_pitch.set_title(f"{team_to_analyze} Shot Map\n{subtitle}", fontsize=18, weight='bold')
    return fig

def create_season_shotmap(season_events_df, team_to_analyze):
    # (This is the full function from the previous step)
    team_shots_df = season_events_df[(season_events_df.get('team.name') == team_to_analyze) & (season_events_df.get('type.primary') == 'shot') & (~season_events_df.get('type.secondary','').astype(str).str.contains('penalty', na=False))].copy().reset_index(drop=True)
    if team_shots_df.empty:
        fig, ax = plt.subplots(figsize=(10, 8)); fig.set_facecolor('#f5f1e9'); ax.set_facecolor('#f5f1e9'); ax.text(0.5, 0.5, 'No shots found for this team this season.', ha='center', va='center', fontsize=12); ax.axis('off'); return fig
    fig = plt.figure(figsize=(12, 12)); fig.set_facecolor('#f5f1e9'); pitch = Pitch(pitch_type='wyscout', pitch_color='#f5f1e9', line_color='black', line_zorder=2, half=True); ax_pitch = fig.add_subplot(); pitch.draw(ax=ax_pitch)
    XG_MAX = 0.8; colors = ["#03045e", "#ade8f4", "#fff3b0", "#ff8c00", "#e63946", "#800f2f"]; nodes = [0.0, 0.1 / XG_MAX, 0.2 / XG_MAX, 0.4 / XG_MAX, 0.6 / XG_MAX, 1.0]; cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", list(zip(nodes, colors)))
    for index, shot in team_shots_df.iterrows():
        x = shot.get('location.x'); y = shot.get('location.y'); xg = shot.get('shot.xg')
        if pd.isna(x) or pd.isna(y) or pd.isna(xg): continue
        is_goal = shot.get('shot.isGoal') == True; color = cmap(min(xg / XG_MAX, 1.0)); edge_color = 'green' if is_goal else 'black'
        pitch.scatter(x, y, s=150, facecolor=color, edgecolor=edge_color, linewidth=1.5, ax=ax_pitch, zorder=3, alpha=0.7)
    total_xg = team_shots_df['shot.xg'].sum(); goals = team_shots_df['shot.isGoal'].sum(); subtitle = f"Liga 3 Portugal, 2025/26 | Total xG: {total_xg:.2f} | Goals: {goals}"; ax_pitch.set_title(f"{team_to_analyze} Season Shot Map (Non-Penalty)\n{subtitle}", fontsize=18, weight='bold')
    return fig

def create_season_shots_against_shotmap(season_events_df, matches_summary_df, team_to_analyze):
    # (This is the full function from the previous step)
    team_match_ids = matches_summary_df[(matches_summary_df.get('homeTeamName') == team_to_analyze) | (matches_summary_df.get('awayTeamName') == team_to_analyze)]['matchId'].unique()
    relevant_events = season_events_df[season_events_df['matchId'].isin(team_match_ids)]
    opponent_shots_df = relevant_events[(relevant_events.get('type.primary') == 'shot') & (relevant_events.get('team.name') != team_to_analyze) & (~relevant_events.get('type.secondary','').astype(str).str.contains('penalty', na=False))].copy().reset_index(drop=True)
    if opponent_shots_df.empty:
        fig, ax = plt.subplots(figsize=(10, 8)); fig.set_facecolor('#f5f1e9'); ax.set_facecolor('#f5f1e9'); ax.text(0.5, 0.5, 'No shots against found for this team.', ha='center', va='center', fontsize=12); ax.axis('off'); return fig
    fig = plt.figure(figsize=(12, 12)); fig.set_facecolor('#f5f1e9'); pitch = Pitch(pitch_type='wyscout', pitch_color='#f5f1e9', line_color='black', line_zorder=2, half=True); ax_pitch = fig.add_subplot(); pitch.draw(ax=ax_pitch)
    XG_MAX = 0.8; colors = ["#03045e", "#ade8f4", "#fff3b0", "#ff8c00", "#e63946", "#800f2f"]; nodes = [0.0, 0.1 / XG_MAX, 0.2 / XG_MAX, 0.4 / XG_MAX, 0.6 / XG_MAX, 1.0]; cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", list(zip(nodes, colors)))
    for index, shot in opponent_shots_df.iterrows():
        x = shot.get('location.x'); y = shot.get('location.y'); xg = pd.to_numeric(shot.get('shot.xg'), errors='coerce'); is_goal = shot.get('shot.isGoal') == True
        if pd.isna(x) or pd.isna(y) or pd.isna(xg): continue
        color = cmap(min(xg / XG_MAX, 1.0)); edge_color = 'green' if is_goal else 'black'; pitch.scatter(x, y, s=150, facecolor=color, edgecolor=edge_color, linewidth=1.5, ax=ax_pitch, zorder=3, alpha=0.7)
    total_shots_against = len(opponent_shots_df); total_xg_against = round(pd.to_numeric(opponent_shots_df.get('shot.xg'), errors='coerce').sum(), 2); goals_against = opponent_shots_df[opponent_shots_df.get('shot.isGoal') == True].shape[0]; xg_per_shot_against = round(total_xg_against / total_shots_against, 3) if total_shots_against > 0 else 0
    subtitle = f"Liga 3 Portugal, 2025/26 | Total xGA: {total_xg_against} | Goals Against: {goals_against}"; ax_pitch.set_title(f"{team_to_analyze} Shots CONCEDED Map (Non-Penalty)\n{subtitle}", fontsize=18, weight='bold')
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
def plot_team_strength(stats_df, teams_to_include=None, league="Liga 3 Portugal", season="2025/26"):
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
                 img = Image.open(logo_path); imagebox = OffsetImage(img, zoom=0.25); ab = AnnotationBbox(imagebox, (row['Attacking Strength'], row['Defending Strength']), frameon=False, zorder=2); ax.add_artist(ab); logos_plotted +=1
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


# ==============================================================================
# 5. STREAMLIT APP UI
# ==============================================================================
st.title("Atlético CP Analysis") # You can change this title

# --- Load Data ---
raw_events_df, matches_summary_df, all_match_data, season_team_stats, player_minutes_df = load_data()

# --- Main App Logic ---
if raw_events_df is not None and matches_summary_df is not None and player_minutes_df is not None:
    # --- Sidebar for Navigation ---
    st.sidebar.title("Dashboard Controls")
    analysis_type = st.sidebar.radio("Choose Analysis Type", ('Match Analysis', 'Team Analysis', 'League Analysis', 'Player Analysis'))

    if analysis_type == 'Match Analysis':
        st.header("Match Analysis")
        
        # --- Match Selection (Using correct column names) ---
        if 'dateutc' in matches_summary_df.columns:
            matches_summary_df['display_date'] = pd.to_datetime(matches_summary_df['dateutc']).dt.strftime('%Y-%m-%d')
        else: matches_summary_df['display_date'] = 'Unknown Date'
        
        gw_series = matches_summary_df.get('gameweek', pd.Series(dtype='str')).fillna('?').astype(str)
        # --- FIX: Use corrected homeTeamName/awayTeamName ---
        matches_summary_df['display_name'] = matches_summary_df.get('homeTeamName', '?').fillna('?') + " vs " + \
                                             matches_summary_df.get('awayTeamName', '?').fillna('?') + \
                                             " (" + matches_summary_df.get('score', '?-?').fillna('?-?') + ")"

        sort_key = 'dateutc' if 'dateutc' in matches_summary_df.columns else 'matchId'
        matches_summary_df.sort_values(by=[sort_key, 'matchId'], inplace=True, na_position='last')
        
        selected_match_display = st.sidebar.selectbox("Select a Match", matches_summary_df['display_name'])
        selected_match_info = matches_summary_df[matches_summary_df['display_name'] == selected_match_display].iloc[0]
        selected_match_id = selected_match_info['matchId']
        
        st.header(f"Match Report: {selected_match_info['homeTeamName']} vs {selected_match_info['awayTeamName']}")
        
        match_data = all_match_data.get(selected_match_id)
        if match_data:
            st.subheader("Shot Maps")
            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(create_match_shotmap(raw_events_df[raw_events_df['matchId'] == selected_match_id], selected_match_info, selected_match_info['homeTeamName']), use_container_width=True)
            with col2:
                st.pyplot(create_match_shotmap(raw_events_df[raw_events_df['matchId'] == selected_match_id], selected_match_info, selected_match_info['awayTeamName']), use_container_width=True)

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
            # (Your full radar logic here, it seems correct)
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
            fig_shots_for = create_season_shotmap(raw_events_df, selected_team_t)
            if fig_shots_for: st.pyplot(fig_shots_for, use_container_width=True)
            else: st.warning("No shots found FOR this team.")
        with col2_shot:
            st.markdown(f"**Shots AGAINST {selected_team_t}**")
            fig_shots_against = create_season_shots_against_shotmap(raw_events_df, matches_summary_df, selected_team_t)
            if fig_shots_against: st.pyplot(fig_shots_against, use_container_width=True)
            else: st.warning("No shots found AGAINST this team.")

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
        
        stats_df_raw, stats_df_pct = calculate_all_team_radars_stats(raw_events_df, matches_summary_df)
        team_strength_df = calculate_team_strength(raw_events_df, matches_summary_df).copy()

        st.subheader("Team Strength Scatterplot")
        if not team_strength_df.empty:
            TEAMS_TO_INCLUDE = [ '1º Dezembro', 'Caldas', 'Sporting Covilhã', 'Mafra', 'União Santarém', 'Amora', 'Académica', 'CF Os Belenenses', 'Lusitano Évora 1911', 'Atlético CP', 'Fafe', 'Varzim', 'Atlético CP', 'Mafra', 'Caldas', 'Paredes', 'Sanjoanense', 'São João Ver', 'Amarante', 'Vitória Guimarães II', 'Trofense', 'Sporting Braga II', 'AD Marco 09' ]
            valid_teams_to_plot = [team for team in TEAMS_TO_INCLUDE if team in team_strength_df.index]
            fig_strength = plot_team_strength(team_strength_df, teams_to_include=valid_teams_to_plot) 
            st.pyplot(fig_strength, use_container_width=True)
            with st.expander("View Raw Strength Data"):
                 st.dataframe(team_strength_df[['Attacking Strength', 'Defending Strength']].round(2))
        else:
            st.warning("Could not calculate team strength data.")
        
        st.subheader("Custom League Scatterplot")
        if not stats_df_raw.empty:
            metrics_to_exclude = ['teamName', 'matchId', 'seasonId', 'teamId'] 
            available_metrics = sorted([col for col in stats_df_raw.columns if col not in metrics_to_exclude])
            col_x, col_y = st.columns(2)
            with col_x:
                default_x_index = available_metrics.index('xG') if 'xG' in available_metrics else 0
                x_metric = st.selectbox("Select X-Axis Metric:", available_metrics, index=default_x_index) 
            with col_y:
                default_y_index = available_metrics.index('xG Against') if 'xG Against' in available_metrics else 1
                y_metric = st.selectbox("Select Y-Axis Metric:", available_metrics, index=default_y_index)
            col_inv_x, col_inv_y = st.columns(2)
            with col_inv_x:
                invert_x = st.checkbox("Invert X-Axis (Lower is Better)", key='invert_x')
            with col_inv_y:
                default_invert_y = 'Against' in y_metric or 'PPDA' in y_metric
                invert_y = st.checkbox("Invert Y-Axis (Lower is Better)", value=default_invert_y, key='invert_y')
            if x_metric and y_metric:
                fig_custom = plot_custom_scatter( stats_df_raw, x_metric, y_metric, invert_x, invert_y )
                st.pyplot(fig_custom, use_container_width=True)
            with st.expander("View Raw Radar Stats Data"):
                 st.dataframe(stats_df_raw.round(2))
        else:
            st.warning("Could not calculate raw league stats for custom plot.")

    # --- NEW: Player Analysis Section ---
    elif analysis_type == 'Player Analysis':
        st.header("Player Analysis")
        
        st.sidebar.subheader("Player Analysis Options")
        
        if player_minutes_df is None:
             st.error("Player minutes data file (`player_minutes_and_positions.pkl`) not loaded.")
             st.stop()
        
        # 1. Run the heavy calculations
        try:
            player_stats_df = calculate_player_radar_data(raw_events_df, player_minutes_df) 
            player_stats_with_scores_df = calculate_player_percentiles_and_scores(
                player_stats_df, POSITION_GROUPS, WEIGHTS, INVERT_METRICS, min_minutes=90
            )
        except Exception as e:
            st.error(f"An error occurred during player stat calculation: {e}")
            st.exception(e) 
            st.stop() 

        # --- DEBUG EXPANDER ---
        with st.expander("🕵️‍♂️ **Click to View Player Data Processing Steps**"):
            st.subheader("1. Base Player Data (from `player_minutes_and_positions.pkl`)")
            st.dataframe(player_minutes_df.head())
            st.subheader("2. Raw Per-90 Stats (from `calculate_player_radar_data`)")
            st.write("Check here if metrics like 'npxG', 'xAOP', 'xT', 'Passes', 'Duels' are all 0 or missing.")
            st.dataframe(player_stats_df) # This is the crucial table
            st.subheader("3. Final Percentiles & Scores (from `calculate_player_percentiles_and_scores`)")
            st.write("Check here if percentile/score columns (e.g., `npxG_percentile`, `Stopper_Score`) are all 0 or missing.")
            st.dataframe(player_stats_with_scores_df)
        # --- END DEBUG EXPANDER ---

        if player_stats_with_scores_df.empty:
            st.warning("No players found matching the criteria (e.g., >= 90 minutes).")
        else:
            # --- START: UI CODE TO DISPLAY PLOT ---
            player_list_df = player_stats_with_scores_df[['playerName', 'teamName', 'totalMinutes']].sort_values(by='totalMinutes', ascending=False)
            player_list_df['display_name'] = player_list_df['playerName'] + " (" + player_list_df['teamName'] + ", " + player_list_df['totalMinutes'].astype(int).astype(str) + " min)"
            selected_player_display = st.sidebar.selectbox("Select Player:", player_list_df['display_name'])
            
            selected_player_name = player_list_df[player_list_df['display_name'] == selected_player_display]['playerName'].values[0]
            player_data = player_stats_with_scores_df.loc[player_stats_with_scores_df['playerName'] == selected_player_name].copy()

            if player_data.empty:
                st.warning(f"No data found for {selected_player_name}.")
            else:
                primary_pos = player_data['primaryPosition'].values[0]
                eligible_groups = [pos_group for pos_group, pos_roles in POSITION_GROUPS.items() if primary_pos in pos_roles]
                
                if not eligible_groups:
                    st.warning(f"No radar templates found for player's primary position: {primary_pos}")
                else:
                    st.subheader(f"Player Radar: {selected_player_name}")
                    highest_score = -1; highest_scoring_group = None; scores_by_group = {}
                    for group in eligible_groups:
                        score_col = group + '_Score'
                        if score_col in player_data.columns:
                            player_score = player_data[score_col].values[0]
                            scores_by_group[group] = player_score
                            if player_score > highest_score:
                                highest_score = player_score; highest_scoring_group = group
                    if highest_scoring_group is None: highest_scoring_group = eligible_groups[0]
                    
                    metrics_to_plot = list(WEIGHTS[highest_scoring_group].keys())
                    metrics_to_plot = [m for m in metrics_to_plot if m in player_data.columns]
                    
                    st.markdown(f"Displaying radar for best-fit archetype: **{highest_scoring_group}**")
                    
                    position_data_for_dist = player_stats_with_scores_df[player_stats_with_scores_df['primaryPosition'].isin(POSITION_GROUPS[highest_scoring_group])]
                    
                    if position_data_for_dist.empty:
                         st.warning(f"No other players found for position group '{POSITION_GROUPS[highest_scoring_group]}' for percentile comparison.")
                    else:
                        # --- CORRECTED CALL (Fixing global var and argument order) ---
                        fig = create_radar_with_distributions(
                            player_data, 
                            metrics_to_plot, 
                            highest_scoring_group, 
                            eligible_groups,
                            all_position_data=position_data_for_dist,
                            full_df_for_ranking=player_stats_with_scores_df # Pass the full DF
                        )
                        st.pyplot(fig, use_container_width=True)
                    
                    with st.expander("View Raw Player Data (Per 90)"):
                         display_cols = [m for m in metrics_to_plot if m in player_data.columns] + ['totalMinutes', 'primaryPosition']
                         st.dataframe(player_data[display_cols].round(2).T)
                         
                    with st.expander("View Percentile Data"):
                         percentile_cols = [m + '_percentile' for m in metrics_to_plot if m + '_percentile' in player_data.columns]
                         st.dataframe(player_data[percentile_cols].round(2).T)
                         
                    with st.expander("View Archetype Scores"):
                         score_cols = [g + '_Score' for g in eligible_groups if g + '_Score' in player_data.columns]
                         st.dataframe(player_data[score_cols].round(2).T)
            # --- END: UI CODE ---

else:
    st.error("Data files not loaded. Please run `process_data.py` locally (including the new player minutes step) and ensure all artifacts are pushed to GitHub.")