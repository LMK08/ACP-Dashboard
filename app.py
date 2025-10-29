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
        return raw_events_df, matches_summary_df, all_match_data, season_team_stats
    except FileNotFoundError as e:
        st.error(f"❌ Error: A data file was not found. Please run `process_data.py` first. Missing file: {e.filename}")
        return None, None, None, None


# ==============================================================================
# 3. HELPER & PLOTTING FUNCTIONS
# ==============================================================================

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
    net_strength_diff = stats_df['Attacking Strength'] - stats_df['Defending Strength']
    min_diff = np.floor(net_strength_diff.quantile(0.05) * 10) / 10; max_diff = np.ceil(net_strength_diff.quantile(0.95) * 10) / 10
    plot_xlim = np.array(ax.get_xlim())
    for c in np.arange(min_diff, max_diff + 0.1, 0.1): y = plot_xlim - c; ax.plot(plot_xlim, y, color='lightgray', linestyle=':', zorder=1, lw=1)

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
    ax.set_xlabel('Attacking Strength (Goals & xG For per Match)', fontsize=12)
    ax.set_ylabel('Defending Strength (Goals & xG Against per Match)', fontsize=12)
    #ax.grid(True, linestyle='--', alpha=0.5); plt.tight_layout(); return fig


# ==============================================================================
# 4. STREAMLIT APP UI
# ==============================================================================
st.title("Atlético CP Analysis")

# --- Load Data ---
raw_events_df, matches_summary_df, all_match_data, season_team_stats = load_data()

if raw_events_df is not None and matches_summary_df is not None:
    # --- Sidebar for Navigation ---
    st.sidebar.title("Dashboard Controls")
    # --- UPDATED: Renamed 'Season-Long Analysis' to 'Team Analysis', Added 'League Analysis' ---
    analysis_type = st.sidebar.radio("Choose Analysis Type", ('Match Analysis', 'Team Analysis', 'League Analysis'))

    if analysis_type == 'Match Analysis':
        # --- Match Selection ---
        # --- UPDATED Column Names ---
        # Convert dateutc to just date for display if needed
        if 'dateutc' in matches_summary_df.columns:
            matches_summary_df['display_date'] = pd.to_datetime(matches_summary_df['dateutc']).dt.strftime('%Y-%m-%d')
        else:
            matches_summary_df['display_date'] = 'Unknown Date' # Fallback
            
        # Create display name using API Gameweek and Date (if available)
        matches_summary_df['display_name'] = matches_summary_df.get('gameweek', 'GW?').astype(str).apply(lambda x: f"GW{x}" if x.isdigit() else x) + " | " + \
                                             matches_summary_df.get('homeTeamName', '?').fillna('?') + " vs " + \
                                             matches_summary_df.get('awayTeamName', '?').fillna('?') + \
                                             " (" + matches_summary_df.get('score', '?-?').fillna('?-?') + ")" + " | " + \
                                             matches_summary_df['display_date']

        # Sort matches chronologically using dateutc (if available) or matchId
        sort_key = 'dateutc' if 'dateutc' in matches_summary_df.columns else 'matchId'
        matches_summary_df.sort_values(by=[sort_key, 'matchId'], inplace=True, na_position='last')

        selected_match_display = st.sidebar.selectbox("Select a Match", matches_summary_df['display_name'])

        # Find selected match info
        selected_match_info = matches_summary_df[matches_summary_df['display_name'] == selected_match_display].iloc[0]
        selected_match_id = selected_match_info['matchId']
        
        st.header(f"Match Report: {selected_match_info['homeTeamName']} vs {selected_match_info['awayTeamName']}")
        
        # --- Display Data for Selected Match ---
        match_data = all_match_data.get(selected_match_id)
        if match_data:
            # Shotmaps
            st.subheader("Shot Maps")
            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(create_match_shotmap(raw_events_df[raw_events_df['matchId'] == selected_match_id], selected_match_info, selected_match_info['homeTeamName']), use_container_width=True)
            with col2:
                st.pyplot(create_match_shotmap(raw_events_df[raw_events_df['matchId'] == selected_match_id], selected_match_info, selected_match_info['awayTeamName']), use_container_width=True)

            # Team Stats
            st.subheader("Team Stats")
            if 'team_stats' in match_data and isinstance(match_data['team_stats'], dict) and match_data['team_stats']:
                for stat_category, df in match_data['team_stats'].items():
                    st.markdown(f"**{stat_category}**")
                    if isinstance(df, pd.DataFrame):
                        st.dataframe(df)
                    else: st.warning(f"Data for '{stat_category}' is not a DataFrame.")
            else: st.warning("Team stats data not found or is in unexpected format for this match.")

            # Player Stats
            st.subheader("Player Stats")
            if 'player_stats' in match_data and isinstance(match_data['player_stats'], dict) and 'home' in match_data['player_stats'] and 'away' in match_data['player_stats']:
                st.markdown(f"**{selected_match_info['homeTeamName']}**")
                if isinstance(match_data['player_stats']['home'], pd.DataFrame):
                    st.dataframe(match_data['player_stats']['home'])
                else: st.warning("Home player stats data is not a DataFrame.")
                
                st.markdown(f"**{selected_match_info['awayTeamName']}**")
                if isinstance(match_data['player_stats']['away'], pd.DataFrame):
                    st.dataframe(match_data['player_stats']['away'])
                else: st.warning("Away player stats data is not a DataFrame.")
            else: st.warning("Player stats data not found or is in unexpected format for this match.")
        else:
             st.warning(f"No detailed match data found for Match ID {selected_match_id}. Data might still be processing.")

    # --- UPDATED: Renamed 'Season-Long Analysis' to 'Team Analysis' ---
    elif analysis_type == 'Team Analysis': 
        # --- Team Selection ---
        all_teams_t = sorted(pd.concat([matches_summary_df.get('homeTeamName'), matches_summary_df.get('awayTeamName')]).dropna().unique())
        selected_team_t = st.sidebar.selectbox("Select a Team", all_teams_t, key="team_select_tab") # Use unique key

        st.header(f"Team Report: {selected_team_t}")

        # --- Calculate Radar Stats (Run Once) ---
        stats_df_raw, stats_df_pct = calculate_all_team_radars_stats(raw_events_df, matches_summary_df)

        # --- Radar Charts ---
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
                else: st.warning("Missing data for offensive radar.")
            with col_r2:
                st.markdown("**Distribution Radar**")
                valid_distribution_params = [p for p in distribution_params if p in team_stats_raw.index]
                if valid_distribution_params:
                     raw_dist_values = team_stats_raw[valid_distribution_params].tolist()
                     try: poss_index = valid_distribution_params.index('Ball Possession'); raw_dist_values[poss_index] = f"{raw_dist_values[poss_index]:.0f}%"
                     except ValueError: pass # Ignore if Ball Possession not in params
                     fig_dist = plot_radar_chart(valid_distribution_params, raw_dist_values, team_stats_pct[valid_distribution_params].tolist(), selected_team_t, "Distribution Radar", '#0077b6', league=current_league, season=current_season)
                     st.pyplot(fig_dist, use_container_width=True)
                else: st.warning("Missing data for distribution radar.")
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
                else: st.warning("Missing data for defensive radar.")
        else:
            st.warning(f"Could not find calculated radar statistics for {selected_team_t}.")

        # --- Side-by-Side Shot Maps ---
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

        # --- Corner Analysis ---
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

        # --- Season-long tables ---
        st.subheader("Season-Long Stats")
        if selected_team_t in season_team_stats and 'corners' in season_team_stats[selected_team_t]:
            st.markdown("**Corner Kick Summary**")
            st.dataframe(season_team_stats[selected_team_t]['corners'])
        else:
            st.write("No season-long stats available for this team.")
            

    # --- NEW: League Analysis Section ---
    elif analysis_type == 'League Analysis':
        st.header("League Analysis")

        # --- Team Strength Scatter Plot ---
        st.subheader("Team Strength Scatterplot")

        # Calculate strength stats for ALL teams
        # Use .copy() to avoid modifying cached data if filtering/manipulating later
        team_strength_df = calculate_team_strength(raw_events_df, matches_summary_df).copy()

        # Your list of teams to include
        TEAMS_TO_INCLUDE = [
            '1º Dezembro', 'Caldas', 'Sporting Covilhã', 'Mafra', 'União Santarém',
            'Amora', 'Académica', 'CF Os Belenenses', 'Lusitano Évora 1911', 'Atlético CP',
            'Fafe', 'Varzim', 'Atlético CP', 'Mafra', 'Caldas', 'Paredes',
            'Sanjoanense', 'São João Ver', 'Amarante', 'Vitória Guimarães II', 'Trofense',
            'Sporting Braga II', 'AD Marco 09'
        ]
        
        # Ensure teams in the list are also in the calculated data
        valid_teams_to_plot = [team for team in TEAMS_TO_INCLUDE if team in team_strength_df.index]
        
        # You could also offer a multiselect in the sidebar for this tab
        # st.sidebar.subheader("League Analysis Options")
        # teams_to_plot = st.sidebar.multiselect("Select Teams to Plot:", 
        #                                       options=team_strength_df.index.tolist(), 
        #                                       default=valid_teams_to_plot)


        if not team_strength_df.empty:
            # Generate and display the plot
            # Pass the full df (for axes) and the list of teams to actually plot
            fig_strength = plot_team_strength(team_strength_df, teams_to_include=valid_teams_to_plot) 
            st.pyplot(fig_strength, use_container_width=True)
            
            # Optionally display the raw data table
            with st.expander("View Raw Strength Data (All Teams)"):
                 st.dataframe(team_strength_df[['Attacking Strength', 'Defending Strength']].round(2))
        else:
            st.warning("Could not calculate team strength data.")
        
        # --- Add more league-wide plots/tables below ---

else:
    st.error("Data files not loaded. Please run `process_data.py` locally and ensure artifacts are pushed to GitHub.")