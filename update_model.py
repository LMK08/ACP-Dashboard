#!/usr/bin/env python3
"""
Update the match predictor model with latest match data.
Run after process_data.py to refresh team stats and ratings.
"""

import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

print("Loading current season data...")
try:
    current_events = pd.read_parquet('raw_events.parquet')
    current_matches = pd.read_parquet('matches_summary.parquet')
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Run process_data.py first to generate data files.")
    exit(1)

print(f"Loaded {len(current_events):,} events, {len(current_matches)} matches")

# Load existing model
try:
    with open('match_predictor_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    print(f"Loaded existing model (version {model_data.get('version', 'unknown')})")
except FileNotFoundError:
    print("Error: match_predictor_model.pkl not found. Run train_predictor_v2.py first.")
    exit(1)

# Helper functions
def get_match_xg(events_df, match_id, team_name):
    # Non-penalty xG: type 'shot' only (penalties are their own primary type).
    # Aligned 2026-08 with the simple-strength predictor spec — NPxGD is the
    # more predictive and less noisy signal (see calibrate_strength_metrics).
    match_events = events_df[events_df['matchId'] == match_id]
    shots = match_events[(match_events['shot.isGoal'].notna())
                         & (match_events['team.name'] == team_name)
                         & (match_events['type.primary'] == 'shot')]
    if 'shot.xg' in shots.columns:
        return shots['shot.xg'].sum()
    return 0.0

def get_team_shots(events_df, match_id, team_name):
    match_events = events_df[events_df['matchId'] == match_id]
    team_shots = match_events[(match_events['shot.isGoal'].notna()) & (match_events['team.name'] == team_name)]
    total_shots = len(team_shots)
    if 'shot.isGoal' in team_shots.columns and len(team_shots) > 0:
        goals = team_shots['shot.isGoal'].sum() if team_shots['shot.isGoal'].notna().any() else 0
        sot = min(total_shots, int(goals + 0.3 * (total_shots - goals)))
    else:
        sot = 0
    return total_shots, sot

def initialize_team_stats():
    return {
        'matches': 0, 'wins': 0, 'draws': 0, 'losses': 0, 'points': 0,
        'goals_for': 0, 'goals_against': 0, 'xG_for': 0.0, 'xG_against': 0.0,
        'shots_for': 0, 'sot_for': 0, 'home_matches': 0, 'home_wins': 0,
        'home_goals': 0, 'away_matches': 0, 'away_wins': 0, 'away_goals': 0,
        'clean_sheets': 0, 'last_5_results': [], 'last_5_xG': []
    }

# Process current season matches.
# Hard cut to current-season matches only — at this tier roster turnover and
# cross-tier moves between seasons make prior-season stats unreliable, so we
# rebuild team strength from this year's matches alone. Prior-season blending
# is handled separately (and now decays out fast) in simulate_season.py.
from league_config import COMPETITIONS
CURRENT_SEASON_IDS = {comp["current_season"] for comp in COMPETITIONS.values()}
print(f"\nProcessing current-season matches only (seasonIds: {sorted(CURRENT_SEASON_IDS)})...")
matches = current_matches[current_matches['seasonId'].isin(CURRENT_SEASON_IDS)].copy()
matches = matches.sort_values('dateutc' if 'dateutc' in matches.columns else 'matchId')
print(f"  filtered: {len(matches)} matches in scope (was {len(current_matches)} total)")

team_stats = defaultdict(initialize_team_stats)

for _, match in matches.iterrows():
    match_id = match['matchId']
    home = match['homeTeamName']
    away = match['awayTeamName']

    # Parse score
    score = match.get('score', '')
    if pd.isna(score) or '-' not in str(score):
        continue
    try:
        home_goals, away_goals = map(int, str(score).split('-'))
    except:
        continue

    # Update stats
    team_stats[home]['matches'] += 1
    team_stats[home]['home_matches'] += 1
    team_stats[home]['goals_for'] += home_goals
    team_stats[home]['goals_against'] += away_goals
    team_stats[home]['home_goals'] += home_goals

    team_stats[away]['matches'] += 1
    team_stats[away]['away_matches'] += 1
    team_stats[away]['goals_for'] += away_goals
    team_stats[away]['goals_against'] += home_goals
    team_stats[away]['away_goals'] += away_goals

    if home_goals > away_goals:
        team_stats[home]['wins'] += 1
        team_stats[home]['home_wins'] += 1
        team_stats[home]['points'] += 3
        team_stats[home]['last_5_results'].append(3)
        team_stats[away]['losses'] += 1
        team_stats[away]['last_5_results'].append(0)
    elif away_goals > home_goals:
        team_stats[away]['wins'] += 1
        team_stats[away]['away_wins'] += 1
        team_stats[away]['points'] += 3
        team_stats[away]['last_5_results'].append(3)
        team_stats[home]['losses'] += 1
        team_stats[home]['last_5_results'].append(0)
    else:
        team_stats[home]['draws'] += 1
        team_stats[home]['points'] += 1
        team_stats[home]['last_5_results'].append(1)
        team_stats[away]['draws'] += 1
        team_stats[away]['points'] += 1
        team_stats[away]['last_5_results'].append(1)

    if away_goals == 0:
        team_stats[home]['clean_sheets'] += 1
    if home_goals == 0:
        team_stats[away]['clean_sheets'] += 1

    # xG
    home_xg = get_match_xg(current_events, match_id, home)
    away_xg = get_match_xg(current_events, match_id, away)
    team_stats[home]['xG_for'] += home_xg
    team_stats[home]['xG_against'] += away_xg
    team_stats[home]['last_5_xG'].append(home_xg)
    team_stats[away]['xG_for'] += away_xg
    team_stats[away]['xG_against'] += home_xg
    team_stats[away]['last_5_xG'].append(away_xg)

    # Shots
    home_shots, home_sot = get_team_shots(current_events, match_id, home)
    away_shots, away_sot = get_team_shots(current_events, match_id, away)
    team_stats[home]['shots_for'] += home_shots
    team_stats[home]['sot_for'] += home_sot
    team_stats[away]['shots_for'] += away_shots
    team_stats[away]['sot_for'] += away_sot

team_stats = dict(team_stats)
print(f"Processed {len(team_stats)} teams")

# Calculate team ratings
print("\nCalculating team strength ratings...")

model = model_data['model']
scaler = model_data['scaler']

league_avg = {
    'ppg': 1.0, 'gpg': 1.2, 'gapg': 1.2, 'xgpg': 1.0, 'xgapg': 1.0,
    'win_rate': 0.33, 'form': 1.0, 'xg_form': 1.0, 'cs_rate': 0.25,
    'shot_conv': 0.10, 'sot_rate': 0.35, 'venue_wr': 0.33, 'venue_gpg': 1.2
}

team_ratings = {}

for team, stats in team_stats.items():
    if stats['matches'] < 3:
        continue

    m = stats['matches']

    t_ppg = stats['points'] / m
    t_gpg = stats['goals_for'] / m
    t_gapg = stats['goals_against'] / m
    t_xgpg = stats['xG_for'] / m if stats['xG_for'] > 0 else 1.0
    t_xgapg = stats['xG_against'] / m if stats['xG_against'] > 0 else 1.0
    t_wr = stats['wins'] / m
    t_form = np.mean(stats['last_5_results'][-5:]) if stats['last_5_results'] else 1.0
    t_xg_form = np.mean(stats['last_5_xG'][-5:]) if stats['last_5_xG'] else 1.0
    t_cs = stats['clean_sheets'] / m
    t_shot_conv = stats['goals_for'] / max(stats['shots_for'], 1)
    t_sot_rate = stats['sot_for'] / max(stats['shots_for'], 1)
    t_home_wr = stats['home_wins'] / max(stats['home_matches'], 1)
    t_away_wr = stats['away_wins'] / max(stats['away_matches'], 1)
    t_home_gpg = stats['home_goals'] / max(stats['home_matches'], 1)
    t_away_gpg = stats['away_goals'] / max(stats['away_matches'], 1)

    # Features as HOME team vs league average
    home_features = [
        t_ppg, league_avg['ppg'], t_ppg - league_avg['ppg'],
        t_form, league_avg['form'], t_form - league_avg['form'],
        t_gpg, league_avg['gpg'], t_gpg - league_avg['gpg'],
        t_gpg - t_gapg, 0, (t_gpg - t_gapg),
        t_xgpg, league_avg['xgpg'], t_xgpg - league_avg['xgpg'],
        t_xgapg, league_avg['xgapg'],
        t_xgpg - t_xgapg, 0, (t_xgpg - t_xgapg),
        t_xg_form, league_avg['xg_form'],
        t_wr, league_avg['win_rate'], t_wr - league_avg['win_rate'],
        t_home_wr, league_avg['venue_wr'],
        t_shot_conv, league_avg['shot_conv'],
        t_sot_rate, league_avg['sot_rate'],
        t_cs, league_avg['cs_rate'],
        t_home_gpg, league_avg['venue_gpg'],
    ]

    # Features as AWAY team vs league average
    away_features = [
        league_avg['ppg'], t_ppg, league_avg['ppg'] - t_ppg,
        league_avg['form'], t_form, league_avg['form'] - t_form,
        league_avg['gpg'], t_gpg, league_avg['gpg'] - t_gpg,
        0, t_gpg - t_gapg, -(t_gpg - t_gapg),
        league_avg['xgpg'], t_xgpg, league_avg['xgpg'] - t_xgpg,
        league_avg['xgapg'], t_xgapg,
        0, t_xgpg - t_xgapg, -(t_xgpg - t_xgapg),
        league_avg['xg_form'], t_xg_form,
        league_avg['win_rate'], t_wr, league_avg['win_rate'] - t_wr,
        league_avg['venue_wr'], t_away_wr,
        league_avg['shot_conv'], t_shot_conv,
        league_avg['sot_rate'], t_sot_rate,
        league_avg['cs_rate'], t_cs,
        league_avg['venue_gpg'], t_away_gpg,
    ]

    try:
        home_scaled = scaler.transform([home_features])
        away_scaled = scaler.transform([away_features])

        home_proba = model.predict_proba(home_scaled)[0]
        away_proba = model.predict_proba(away_scaled)[0]

        home_strength = home_proba[1] - home_proba[2]
        away_strength = away_proba[2] - away_proba[1]
        overall_rating = (home_strength + away_strength) / 2
        rating_scaled = max(0, min(100, 50 + (overall_rating * 50)))

        team_ratings[team] = {
            'overall': rating_scaled,
            'home_strength': 50 + (home_strength * 50),
            'away_strength': 50 + (away_strength * 50),
            'attack': t_xgpg,
            'defense': t_xgapg,
            'form': t_form,
            'matches': m
        }
    except Exception as e:
        print(f"Warning: Could not calculate rating for {team}: {e}")

# Sort and display ratings
sorted_ratings = sorted(team_ratings.items(), key=lambda x: x[1]['overall'], reverse=True)
print("\nUpdated Team Strength Ratings:")
print("-" * 50)
for rank, (team, r) in enumerate(sorted_ratings, 1):
    print(f"{rank:2}. {team:25} {r['overall']:.1f}")

# Update model data
model_data['team_stats'] = team_stats
model_data['team_ratings'] = team_ratings

# Save updated model
with open('match_predictor_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print(f"\nModel updated with {len(team_ratings)} team ratings")
print("Saved to match_predictor_model.pkl")
