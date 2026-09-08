#!/usr/bin/env python3
"""
Match Outcome Predictor v2 - Season-Specific Features with Decay Priors

Key improvements over v1:
1. Season-specific team stats (reset at season boundary)
2. Decay priors from previous season performance
3. Proper handling of promoted/relegated teams
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
import os
from sklearn.metrics import log_loss

# Experiment switches (2026-08 OBV feature experiment):
#   OBV_FEATURES=1  -> add engine OBV-difference features (see
#                      calibrate_strength_metrics.py for the motivating evidence)
#   MODEL_OUT=path  -> write the trained pkl somewhere other than production
USE_OBV_FEATURES = os.environ.get('OBV_FEATURES') == '1'
MODEL_OUT = os.environ.get('MODEL_OUT', 'match_predictor_model.pkl')

# Load data
print("Loading data...")
historical_events = pd.read_parquet('historical_events.parquet')
historical_matches_raw = pd.read_parquet('historical_matches.parquet')
current_events = pd.read_parquet('raw_events.parquet')
current_matches_raw = pd.read_parquet('matches_summary.parquet')

OBV_LOOKUP = {}
if USE_OBV_FEATURES:
    try:
        _obv = (pd.read_parquet('obv_match_minute.parquet')
                .groupby(['matchId', 'teamId'])['obv'].sum().reset_index())
        _idn = (pd.read_parquet('raw_events.parquet', columns=['team.id', 'team.name'])
                .dropna().drop_duplicates())
        _id2name = dict(zip(_idn['team.id'].astype('int64'), _idn['team.name']))
        for _r in _obv.itertuples(index=False):
            _nm = _id2name.get(int(_r.teamId))
            if _nm:
                OBV_LOOKUP[(int(_r.matchId), _nm)] = float(_r.obv)
        print(f"OBV features ON — {len(OBV_LOOKUP):,} match-team OBV values")
    except Exception as _e:
        print(f"OBV lookup unavailable ({_e}) — OBV features disabled")
        USE_OBV_FEATURES = False

# Normalize column names
def normalize_match_df(df):
    """Standardize column names and extract scores"""
    df = df.copy()
    # Rename columns to standard names
    rename_map = {
        'matchId': 'match_id',
        'seasonId': 'season_id',
        'homeTeamName': 'home_team',
        'awayTeamName': 'away_team',
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Parse label format: "Home Team - Away Team, H-A" (historical data format)
    if 'label' in df.columns and 'home_team' not in df.columns:
        def parse_label(label):
            if pd.isna(label) or not isinstance(label, str):
                return None, None, None, None
            try:
                # Format: "Home Team - Away Team, H-A"
                # Split by comma to get teams and score
                if ', ' in label:
                    teams_part, score_part = label.rsplit(', ', 1)
                else:
                    return None, None, None, None

                # Split teams by " - "
                if ' - ' in teams_part:
                    home_team, away_team = teams_part.split(' - ', 1)
                else:
                    return None, None, None, None

                # Parse score
                if '-' in score_part:
                    home_score, away_score = score_part.split('-')
                    return home_team.strip(), away_team.strip(), int(home_score), int(away_score)
                else:
                    return None, None, None, None
            except:
                return None, None, None, None

        parsed = df['label'].apply(parse_label)
        df['home_team'] = parsed.apply(lambda x: x[0])
        df['away_team'] = parsed.apply(lambda x: x[1])
        df['home_score'] = parsed.apply(lambda x: x[2])
        df['away_score'] = parsed.apply(lambda x: x[3])

    # Parse score if it's combined (e.g., "2-1")
    elif 'score' in df.columns and 'home_score' not in df.columns:
        def parse_score(s):
            if pd.isna(s) or s == '' or '-' not in str(s):
                return None, None
            parts = str(s).split('-')
            try:
                return int(parts[0]), int(parts[1])
            except:
                return None, None

        scores = df['score'].apply(parse_score)
        df['home_score'] = scores.apply(lambda x: x[0])
        df['away_score'] = scores.apply(lambda x: x[1])

    # Remove matches with missing scores
    df = df.dropna(subset=['home_score', 'away_score'])
    df['home_score'] = df['home_score'].astype(int)
    df['away_score'] = df['away_score'].astype(int)

    return df

historical_matches = normalize_match_df(historical_matches_raw)
current_matches = normalize_match_df(current_matches_raw)

print(f"Historical events: {len(historical_events):,}")
print(f"Historical matches: {len(historical_matches)}")
print(f"Current events: {len(current_events):,}")
print(f"Current matches: {len(current_matches)}")

# Identify seasons
print("\n--- Season Analysis ---")
historical_seasons = sorted(historical_matches['season_id'].unique())
current_season_id = current_matches['season_id'].iloc[0] if 'season_id' in current_matches.columns else max(historical_seasons)

print(f"Historical seasons: {historical_seasons}")
print(f"Current season: {current_season_id}")

# Separate prior season from current season in historical data
# The historical data might include current season
prior_seasons = [s for s in historical_seasons if s != current_season_id]
print(f"Prior seasons for priors: {prior_seasons}")

# Get teams by season
teams_by_season = {}
for season in historical_seasons:
    season_matches = historical_matches[historical_matches['season_id'] == season]
    teams = set(season_matches['home_team'].unique()) | set(season_matches['away_team'].unique())
    teams_by_season[season] = teams
    print(f"Season {season}: {len(teams)} teams, {len(season_matches)} matches")

current_teams = set(current_matches['home_team'].unique()) | set(current_matches['away_team'].unique())
print(f"Current season ({current_season_id}): {len(current_teams)} teams, {len(current_matches)} matches")

# Identify team status (returning, promoted, relegated)
if len(prior_seasons) > 0:
    # Use the most recent prior season for priors
    prior_season_id = max(prior_seasons)
    prior_season_teams = teams_by_season[prior_season_id]

    returning_teams = current_teams & prior_season_teams
    promoted_teams = current_teams - prior_season_teams
    relegated_teams = prior_season_teams - current_teams

    print(f"\nUsing season {prior_season_id} for priors")
    print(f"Returning teams: {len(returning_teams)}")
    print(f"Promoted/new teams ({len(promoted_teams)}): {promoted_teams}")
    print(f"Relegated/departed teams ({len(relegated_teams)}): {relegated_teams}")
else:
    prior_season_id = None
    prior_season_teams = set()
    returning_teams = set()
    promoted_teams = current_teams
    relegated_teams = set()
    print("No prior season data available - all teams treated as new")


def get_match_xg(events_df, match_id, team_name, against=False):
    """Calculate xG for a team in a match"""
    match_events = events_df[events_df['matchId'] == match_id]
    if against:
        shots = match_events[(match_events['shot.isGoal'].notna()) & (match_events['team.name'] != team_name)]
    else:
        shots = match_events[(match_events['shot.isGoal'].notna()) & (match_events['team.name'] == team_name)]

    if 'shot.xg' in shots.columns:
        return shots['shot.xg'].sum()
    return 0.0


def get_team_shots(events_df, match_id, team_name):
    """Get shot stats for a team in a match"""
    match_events = events_df[events_df['matchId'] == match_id]
    team_shots = match_events[(match_events['shot.isGoal'].notna()) & (match_events['team.name'] == team_name)]

    total_shots = len(team_shots)
    sot = len(team_shots[team_shots['shot.isGoal'].isin([True, False]) | team_shots['shot.bodyPart'].notna()])

    # Approximate SOT: shots that hit target
    if 'shot.isGoal' in team_shots.columns:
        goals = team_shots['shot.isGoal'].sum() if team_shots['shot.isGoal'].notna().any() else 0
        # Rough approximation: goals + ~30% of non-goals on target
        sot = min(total_shots, int(goals + 0.3 * (total_shots - goals)))

    return total_shots, sot


def calculate_end_of_season_stats(matches_df, events_df, season_id=None):
    """Calculate final season stats for each team"""
    if season_id:
        matches = matches_df[matches_df['season_id'] == season_id].copy()
    else:
        matches = matches_df.copy()

    team_stats = defaultdict(lambda: {
        'matches': 0, 'wins': 0, 'draws': 0, 'losses': 0, 'points': 0,
        'goals_for': 0, 'goals_against': 0,
        'xG_for': 0.0, 'xG_against': 0.0,
        'shots_for': 0, 'sot_for': 0,
        'home_matches': 0, 'home_wins': 0, 'home_goals': 0,
        'away_matches': 0, 'away_wins': 0, 'away_goals': 0,
        'clean_sheets': 0
    })

    for _, match in matches.iterrows():
        match_id = match['match_id']
        home = match['home_team']
        away = match['away_team']
        home_goals = match['home_score']
        away_goals = match['away_score']

        # Home team stats
        team_stats[home]['matches'] += 1
        team_stats[home]['home_matches'] += 1
        team_stats[home]['goals_for'] += home_goals
        team_stats[home]['goals_against'] += away_goals
        team_stats[home]['home_goals'] += home_goals

        # Away team stats
        team_stats[away]['matches'] += 1
        team_stats[away]['away_matches'] += 1
        team_stats[away]['goals_for'] += away_goals
        team_stats[away]['goals_against'] += home_goals
        team_stats[away]['away_goals'] += away_goals

        # Results
        if home_goals > away_goals:
            team_stats[home]['wins'] += 1
            team_stats[home]['home_wins'] += 1
            team_stats[home]['points'] += 3
            team_stats[away]['losses'] += 1
        elif away_goals > home_goals:
            team_stats[away]['wins'] += 1
            team_stats[away]['away_wins'] += 1
            team_stats[away]['points'] += 3
            team_stats[home]['losses'] += 1
        else:
            team_stats[home]['draws'] += 1
            team_stats[home]['points'] += 1
            team_stats[away]['draws'] += 1
            team_stats[away]['points'] += 1

        # Clean sheets
        if away_goals == 0:
            team_stats[home]['clean_sheets'] += 1
        if home_goals == 0:
            team_stats[away]['clean_sheets'] += 1

        # xG (from events)
        home_xg = get_match_xg(events_df, match_id, home)
        away_xg = get_match_xg(events_df, match_id, away)
        team_stats[home]['xG_for'] += home_xg
        team_stats[home]['xG_against'] += away_xg
        team_stats[away]['xG_for'] += away_xg
        team_stats[away]['xG_against'] += home_xg

        # Shots
        home_shots, home_sot = get_team_shots(events_df, match_id, home)
        away_shots, away_sot = get_team_shots(events_df, match_id, away)
        team_stats[home]['shots_for'] += home_shots
        team_stats[home]['sot_for'] += home_sot
        team_stats[away]['shots_for'] += away_shots
        team_stats[away]['sot_for'] += away_sot

    return dict(team_stats)


def get_decay_weight(matches_played, decay_rate=0.1):
    """
    Calculate decay weight for prior season stats
    Weight decays exponentially as team plays more matches in current season

    decay_rate=0.1 means after 10 matches, prior weight is ~37% (1/e)
    """
    return np.exp(-decay_rate * matches_played)


def initialize_team_stats():
    """Create empty team stats dictionary"""
    return {
        'matches': 0, 'wins': 0, 'draws': 0, 'losses': 0, 'points': 0,
        'goals_for': 0, 'goals_against': 0,
        'xG_for': 0.0, 'xG_against': 0.0,
        'shots_for': 0, 'sot_for': 0,
        'home_matches': 0, 'home_wins': 0, 'home_goals': 0,
        'away_matches': 0, 'away_wins': 0, 'away_goals': 0,
        'clean_sheets': 0,
        'last_5_results': [],  # 3=win, 1=draw, 0=loss
        'last_5_xG': [],
        'obv_for': 0.0, 'obv_against': 0.0, 'obv_matches': 0
    }


def get_blended_stat(current_value, current_matches, prior_per_game, decay_rate=0.1, default_prior=None):
    """
    Blend current season stat with decaying prior from previous season

    If current_matches == 0: use prior only
    As matches increase: current stats dominate more
    """
    if current_matches == 0:
        if prior_per_game is not None:
            return prior_per_game
        elif default_prior is not None:
            return default_prior
        return 0.0

    current_per_game = current_value / current_matches

    if prior_per_game is None:
        return current_per_game

    # Decay weight for prior
    prior_weight = get_decay_weight(current_matches, decay_rate)
    current_weight = 1 - prior_weight

    return current_weight * current_per_game + prior_weight * prior_per_game


def calculate_features(team_stats, team_name, prior_stats, is_home, league_avg_stats):
    """
    Calculate features for a team with decaying priors

    team_stats: current season cumulative stats
    prior_stats: end of prior season stats (or league average for promoted teams)
    is_home: boolean for home/away
    """
    curr = team_stats
    m = curr['matches']

    # Get prior per-game stats (if team was in league last season)
    if prior_stats and prior_stats.get('matches', 0) > 0:
        pm = prior_stats['matches']
        prior_ppg = prior_stats['points'] / pm
        prior_gpg = prior_stats['goals_for'] / pm
        prior_gapg = prior_stats['goals_against'] / pm
        prior_xgpg = prior_stats['xG_for'] / pm if prior_stats['xG_for'] > 0 else 1.0
        prior_xgapg = prior_stats['xG_against'] / pm if prior_stats['xG_against'] > 0 else 1.0
        prior_winrate = prior_stats['wins'] / pm
        prior_csrate = prior_stats['clean_sheets'] / pm
        prior_shot_conv = prior_stats['goals_for'] / max(prior_stats['shots_for'], 1)
        prior_sot_rate = prior_stats['sot_for'] / max(prior_stats['shots_for'], 1)

        # Home/away specific priors
        if is_home and prior_stats['home_matches'] > 0:
            prior_venue_wr = prior_stats['home_wins'] / prior_stats['home_matches']
            prior_venue_gpg = prior_stats['home_goals'] / prior_stats['home_matches']
        elif not is_home and prior_stats['away_matches'] > 0:
            prior_venue_wr = prior_stats['away_wins'] / prior_stats['away_matches']
            prior_venue_gpg = prior_stats['away_goals'] / prior_stats['away_matches']
        else:
            prior_venue_wr = prior_winrate
            prior_venue_gpg = prior_gpg
    else:
        # Use league average for promoted teams (slightly below average)
        prior_ppg = league_avg_stats.get('ppg', 1.0) * 0.85
        prior_gpg = league_avg_stats.get('gpg', 1.0) * 0.85
        prior_gapg = league_avg_stats.get('gapg', 1.0) * 1.15
        prior_xgpg = league_avg_stats.get('xgpg', 1.0) * 0.85
        prior_xgapg = league_avg_stats.get('xgapg', 1.0) * 1.15
        prior_winrate = 0.28  # Below league average
        prior_csrate = league_avg_stats.get('csrate', 0.25) * 0.85
        prior_shot_conv = league_avg_stats.get('shot_conv', 0.1) * 0.9
        prior_sot_rate = league_avg_stats.get('sot_rate', 0.35) * 0.95
        prior_venue_wr = 0.28
        prior_venue_gpg = prior_gpg

    # Current season per-game stats
    if m > 0:
        curr_ppg = curr['points'] / m
        curr_gpg = curr['goals_for'] / m
        curr_gapg = curr['goals_against'] / m
        curr_xgpg = curr['xG_for'] / m
        curr_xgapg = curr['xG_against'] / m
        curr_winrate = curr['wins'] / m
        curr_csrate = curr['clean_sheets'] / m
        curr_shot_conv = curr['goals_for'] / max(curr['shots_for'], 1)
        curr_sot_rate = curr['sot_for'] / max(curr['shots_for'], 1)
        curr_gd = (curr['goals_for'] - curr['goals_against']) / m
        curr_xg_diff = (curr['xG_for'] - curr['xG_against']) / m

        # Venue-specific current stats
        if is_home and curr['home_matches'] > 0:
            curr_venue_wr = curr['home_wins'] / curr['home_matches']
            curr_venue_gpg = curr['home_goals'] / curr['home_matches']
        elif not is_home and curr['away_matches'] > 0:
            curr_venue_wr = curr['away_wins'] / curr['away_matches']
            curr_venue_gpg = curr['away_goals'] / curr['away_matches']
        else:
            curr_venue_wr = curr_winrate
            curr_venue_gpg = curr_gpg
    else:
        curr_ppg = curr_gpg = curr_gapg = curr_xgpg = curr_xgapg = 0
        curr_winrate = curr_csrate = curr_shot_conv = curr_sot_rate = 0
        curr_gd = curr_xg_diff = 0
        curr_venue_wr = curr_venue_gpg = 0

    # Blend with decaying priors (faster decay = current stats dominate sooner)
    decay_rate = 0.15  # After ~7 matches, current season dominates

    ppg = get_blended_stat(curr['points'], m, prior_ppg, decay_rate)
    gpg = get_blended_stat(curr['goals_for'], m, prior_gpg, decay_rate)
    gapg = get_blended_stat(curr['goals_against'], m, prior_gapg, decay_rate)
    xgpg = get_blended_stat(curr['xG_for'], m, prior_xgpg, decay_rate)
    xgapg = get_blended_stat(curr['xG_against'], m, prior_xgapg, decay_rate)
    win_rate = get_blended_stat(curr['wins'], m, prior_winrate, decay_rate)
    cs_rate = get_blended_stat(curr['clean_sheets'], m, prior_csrate, decay_rate)
    shot_conv = curr_shot_conv if m > 3 else prior_shot_conv  # Need enough shots for meaningful conversion
    sot_rate = curr_sot_rate if m > 3 else prior_sot_rate
    venue_wr = get_blended_stat(curr['home_wins'] if is_home else curr['away_wins'],
                                 curr['home_matches'] if is_home else curr['away_matches'],
                                 prior_venue_wr, decay_rate)
    venue_gpg = get_blended_stat(curr['home_goals'] if is_home else curr['away_goals'],
                                  curr['home_matches'] if is_home else curr['away_matches'],
                                  prior_venue_gpg, decay_rate)

    gd = gpg - gapg
    xg_diff = xgpg - xgapg

    # Form (recent matches only, no prior blending)
    form = np.mean(curr['last_5_results'][-5:]) if curr['last_5_results'] else 1.0
    xg_form = np.mean(curr['last_5_xG'][-5:]) if curr['last_5_xG'] else prior_xgpg if prior_stats else 1.0

    return {
        'ppg': ppg,
        'gpg': gpg,
        'gapg': gapg,
        'xgpg': xgpg,
        'xgapg': xgapg,
        'win_rate': win_rate,
        'cs_rate': cs_rate,
        'shot_conv': shot_conv,
        'sot_rate': sot_rate,
        'gd': gd,
        'xg_diff': xg_diff,
        'form': form,
        'xg_form': xg_form,
        'venue_wr': venue_wr,
        'venue_gpg': venue_gpg,
        'obvd': _blended_obvd(curr),
        'matches_played': m
    }


def _blended_obvd(curr, decay_rate=0.1):
    """Blended OBV-difference per game; prior is 0 (league-mean OBV diff)."""
    om = curr.get('obv_matches', 0)
    if om == 0:
        return 0.0
    per_game = (curr.get('obv_for', 0.0) - curr.get('obv_against', 0.0)) / om
    w_prior = get_decay_weight(om, decay_rate)
    return (1 - w_prior) * per_game


def build_training_data(matches_df, events_df, prior_season_stats=None, league_avg_stats=None):
    """
    Build training data with season-specific cumulative stats and decaying priors

    prior_season_stats: Dict of {team_name: end_of_season_stats} from previous season
    """
    matches = matches_df.sort_values('dateutc' if 'dateutc' in matches_df.columns else 'match_id').copy()

    # Initialize cumulative stats for all teams
    team_cumulative = defaultdict(initialize_team_stats)

    X = []
    y = []
    match_info = []

    for idx, match in matches.iterrows():
        match_id = match['match_id']
        home = match['home_team']
        away = match['away_team']
        home_goals = match['home_score']
        away_goals = match['away_score']

        home_stats = dict(team_cumulative[home])
        away_stats = dict(team_cumulative[away])

        # Get prior season stats for each team
        home_prior = prior_season_stats.get(home, None) if prior_season_stats else None
        away_prior = prior_season_stats.get(away, None) if prior_season_stats else None

        # Only create training example if teams have some data (current or prior)
        home_has_data = home_stats['matches'] > 0 or home_prior is not None
        away_has_data = away_stats['matches'] > 0 or away_prior is not None

        if home_has_data and away_has_data:
            # Calculate features with decaying priors
            home_feats = calculate_features(home_stats, home, home_prior, is_home=True, league_avg_stats=league_avg_stats)
            away_feats = calculate_features(away_stats, away, away_prior, is_home=False, league_avg_stats=league_avg_stats)

            # Create feature vector
            features = [
                # Points per game
                home_feats['ppg'], away_feats['ppg'], home_feats['ppg'] - away_feats['ppg'],
                # Form
                home_feats['form'], away_feats['form'], home_feats['form'] - away_feats['form'],
                # Goals per game
                home_feats['gpg'], away_feats['gpg'], home_feats['gpg'] - away_feats['gpg'],
                # Goal difference
                home_feats['gd'], away_feats['gd'], home_feats['gd'] - away_feats['gd'],
                # xG per game
                home_feats['xgpg'], away_feats['xgpg'], home_feats['xgpg'] - away_feats['xgpg'],
                # xG against
                home_feats['xgapg'], away_feats['xgapg'],
                # xG difference
                home_feats['xg_diff'], away_feats['xg_diff'], home_feats['xg_diff'] - away_feats['xg_diff'],
                # xG form
                home_feats['xg_form'], away_feats['xg_form'],
                # Win rates
                home_feats['win_rate'], away_feats['win_rate'], home_feats['win_rate'] - away_feats['win_rate'],
                # Venue-specific win rate
                home_feats['venue_wr'], away_feats['venue_wr'],
                # Shot conversion
                home_feats['shot_conv'], away_feats['shot_conv'],
                # SOT rate
                home_feats['sot_rate'], away_feats['sot_rate'],
                # Clean sheet rate
                home_feats['cs_rate'], away_feats['cs_rate'],
                # Venue goals per game
                home_feats['venue_gpg'], away_feats['venue_gpg'],
            ]
            if USE_OBV_FEATURES:
                features += [home_feats['obvd'], away_feats['obvd'],
                             home_feats['obvd'] - away_feats['obvd']]

            X.append(features)

            # Outcome: 0=Draw, 1=Home Win, 2=Away Win
            if home_goals > away_goals:
                y.append(1)
            elif away_goals > home_goals:
                y.append(2)
            else:
                y.append(0)

            match_info.append({
                'match_id': match_id,
                'home': home,
                'away': away,
                'home_matches': home_stats['matches'],
                'away_matches': away_stats['matches']
            })

        # Update cumulative stats AFTER using them for prediction
        team_cumulative[home]['matches'] += 1
        team_cumulative[home]['home_matches'] += 1
        team_cumulative[home]['goals_for'] += home_goals
        team_cumulative[home]['goals_against'] += away_goals
        team_cumulative[home]['home_goals'] += home_goals

        team_cumulative[away]['matches'] += 1
        team_cumulative[away]['away_matches'] += 1
        team_cumulative[away]['goals_for'] += away_goals
        team_cumulative[away]['goals_against'] += home_goals
        team_cumulative[away]['away_goals'] += away_goals

        if USE_OBV_FEATURES:
            _ho = OBV_LOOKUP.get((match_id, home))
            _ao = OBV_LOOKUP.get((match_id, away))
            if _ho is not None and _ao is not None:
                team_cumulative[home]['obv_for'] += _ho
                team_cumulative[home]['obv_against'] += _ao
                team_cumulative[home]['obv_matches'] += 1
                team_cumulative[away]['obv_for'] += _ao
                team_cumulative[away]['obv_against'] += _ho
                team_cumulative[away]['obv_matches'] += 1

        if home_goals > away_goals:
            team_cumulative[home]['wins'] += 1
            team_cumulative[home]['home_wins'] += 1
            team_cumulative[home]['points'] += 3
            team_cumulative[home]['last_5_results'].append(3)
            team_cumulative[away]['losses'] += 1
            team_cumulative[away]['last_5_results'].append(0)
        elif away_goals > home_goals:
            team_cumulative[away]['wins'] += 1
            team_cumulative[away]['away_wins'] += 1
            team_cumulative[away]['points'] += 3
            team_cumulative[away]['last_5_results'].append(3)
            team_cumulative[home]['losses'] += 1
            team_cumulative[home]['last_5_results'].append(0)
        else:
            team_cumulative[home]['draws'] += 1
            team_cumulative[home]['points'] += 1
            team_cumulative[home]['last_5_results'].append(1)
            team_cumulative[away]['draws'] += 1
            team_cumulative[away]['points'] += 1
            team_cumulative[away]['last_5_results'].append(1)

        if away_goals == 0:
            team_cumulative[home]['clean_sheets'] += 1
        if home_goals == 0:
            team_cumulative[away]['clean_sheets'] += 1

        # xG
        home_xg = get_match_xg(events_df, match_id, home)
        away_xg = get_match_xg(events_df, match_id, away)
        team_cumulative[home]['xG_for'] += home_xg
        team_cumulative[home]['xG_against'] += away_xg
        team_cumulative[home]['last_5_xG'].append(home_xg)
        team_cumulative[away]['xG_for'] += away_xg
        team_cumulative[away]['xG_against'] += home_xg
        team_cumulative[away]['last_5_xG'].append(away_xg)

        # Shots
        home_shots, home_sot = get_team_shots(events_df, match_id, home)
        away_shots, away_sot = get_team_shots(events_df, match_id, away)
        team_cumulative[home]['shots_for'] += home_shots
        team_cumulative[home]['sot_for'] += home_sot
        team_cumulative[away]['shots_for'] += away_shots
        team_cumulative[away]['sot_for'] += away_sot

    return np.array(X), np.array(y), match_info, dict(team_cumulative)


# Calculate league average stats for priors (from prior season only)
print("\n--- Calculating League Average Stats ---")
if prior_season_id:
    prior_season_matches = historical_matches[historical_matches['season_id'] == prior_season_id]
    total_matches = len(prior_season_matches)
    total_goals = prior_season_matches['home_score'].sum() + prior_season_matches['away_score'].sum()
else:
    total_matches = len(current_matches)
    total_goals = current_matches['home_score'].sum() + current_matches['away_score'].sum()

league_avg_stats = {
    'ppg': 1.0,  # League average is always 1 ppg (total points / total games / 2 teams)
    'gpg': total_goals / (total_matches * 2) if total_matches > 0 else 1.0,  # Goals per team per game
    'gapg': total_goals / (total_matches * 2) if total_matches > 0 else 1.0,
    'xgpg': 1.0,
    'xgapg': 1.0,
    'csrate': 0.25,  # Rough estimate
    'shot_conv': 0.10,
    'sot_rate': 0.35
}
print(f"League avg goals/game: {league_avg_stats['gpg']:.2f}")

# Calculate prior season end stats
print("\n--- Calculating Prior Season Stats ---")
if prior_season_id:
    prior_season_stats = calculate_end_of_season_stats(
        historical_matches, historical_events, season_id=prior_season_id
    )
    print(f"Prior season ({prior_season_id}) teams: {len(prior_season_stats)}")

    # Show sample stats
    if prior_season_stats:
        sample_team = list(prior_season_stats.keys())[0]
        print(f"Sample team '{sample_team}': {prior_season_stats[sample_team]['matches']} matches, "
              f"{prior_season_stats[sample_team]['points']} pts, "
              f"{prior_season_stats[sample_team]['goals_for']} GF")
else:
    prior_season_stats = {}
    print("No prior season data - all teams will use league average priors")

# Build training data for ALL historical seasons
# Train on prior seasons (188221, 188222, 189147, 190090), hold out current (191782) for testing
print("\n--- Building Training Data (Multi-Season with Held-Out Test) ---")

# Get all seasons sorted chronologically
all_seasons = sorted(historical_matches['season_id'].unique())
current_season_id = 191782  # Held-out test season
training_seasons = [s for s in all_seasons if s != current_season_id]

print(f"Training seasons: {training_seasons}")
print(f"Held-out test season: {current_season_id}")

# Build training data season by season, accumulating priors
X_all = []
y_all = []
season_end_stats = {}  # Track end-of-season stats for priors

for i, season_id in enumerate(training_seasons):
    season_matches = historical_matches[historical_matches['season_id'] == season_id]
    season_events = historical_events[historical_events['matchId'].isin(season_matches['match_id'])]

    # Get priors from previous season (if exists)
    if i > 0:
        prior_season = training_seasons[i-1]
        prior_stats = season_end_stats.get(prior_season, None)
    else:
        prior_stats = None

    print(f"\nSeason {season_id}: {len(season_matches)} matches")
    if prior_stats:
        print(f"  Using priors from season {training_seasons[i-1]} ({len(prior_stats)} teams)")
    else:
        print(f"  No priors (first season)")

    X_season, y_season, _, final_stats = build_training_data(
        season_matches, season_events,
        prior_season_stats=prior_stats,
        league_avg_stats=league_avg_stats
    )

    print(f"  Samples: {len(X_season)}")

    if len(X_season) > 0:
        X_all.append(X_season)
        y_all.append(y_season)

    # Calculate and store end-of-season stats for next season's priors
    season_end_stats[season_id] = calculate_end_of_season_stats(
        historical_matches, historical_events, season_id=season_id
    )

# Combine all training data
X = np.vstack(X_all) if X_all else np.array([])
y = np.concatenate(y_all) if y_all else np.array([])

print(f"\nTotal training samples: {len(X)}")
print(f"Outcome distribution: Home={sum(y==1)}, Draw={sum(y==0)}, Away={sum(y==2)}")

# Build test data from held-out current season
print("\n--- Building Held-Out Test Data ---")
# Get priors from most recent training season
prior_season_id = max(training_seasons)
prior_season_stats = season_end_stats.get(prior_season_id, {})
print(f"Using priors from season {prior_season_id} ({len(prior_season_stats)} teams)")

test_matches = historical_matches[historical_matches['season_id'] == current_season_id]
test_events = historical_events[historical_events['matchId'].isin(test_matches['match_id'])]

X_test, y_test, test_match_info, final_team_stats = build_training_data(
    test_matches, test_events,
    prior_season_stats=prior_season_stats,
    league_avg_stats=league_avg_stats
)
print(f"Held-out test samples: {len(X_test)}")

# Train model
print("\n--- Training Models ---")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Calculate class weights (to handle imbalanced outcomes)
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = dict(zip(np.unique(y), class_weights))
print(f"Class weights: {class_weight_dict}")

# Try multiple models with optimized parameters
models = {
    'GradientBoosting': GradientBoostingClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.1,
        min_samples_split=5, min_samples_leaf=3, random_state=42
    ),
    'RandomForest': RandomForestClassifier(
        n_estimators=300, max_depth=5, min_samples_split=5,
        min_samples_leaf=3, class_weight='balanced', random_state=42
    ),
    'LogisticRegression': LogisticRegression(
        max_iter=1000, C=0.5, class_weight='balanced', random_state=42
    )
}

best_model = None
best_cv_score = 0
best_test_score = 0
best_name = ""

print("\nCross-validation on training data + Held-out test evaluation:")
for name, model in models.items():
    # Cross-validation on training data
    scores = cross_val_score(model, X_scaled, y, cv=5)
    cv_score = scores.mean()

    # Fit on training data and evaluate on held-out test set
    model.fit(X_scaled, y)
    test_pred = model.predict(X_test_scaled)
    test_score = (test_pred == y_test).mean()
    _proba = model.predict_proba(X_test_scaled)
    _ll = log_loss(y_test, _proba, labels=[0, 1, 2])
    _onehot = np.eye(3)[np.asarray(y_test, dtype=int)]
    _brier = float(np.mean(np.sum((_proba - _onehot) ** 2, axis=1)))
    print(f"  Held-out log-loss: {_ll:.4f} | Brier: {_brier:.4f}")

    print(f"{name}:")
    print(f"  CV (train): {cv_score:.3f} (+/- {scores.std()*2:.3f})")
    print(f"  Held-out test: {test_score:.3f} ({test_score:.1%})")

    if test_score > best_test_score:
        best_test_score = test_score
        best_cv_score = cv_score
        best_model = model
        best_name = name

print(f"\nBest model: {best_name}")
print(f"  Training CV accuracy: {best_cv_score:.1%}")
print(f"  Held-out test accuracy: {best_test_score:.1%}")
print(f"  (Random baseline: 33.3%)")

# Re-fit best model on training data
best_model.fit(X_scaled, y)

# Show detailed test set results
print("\n--- Held-Out Test Set Analysis ---")
test_pred = best_model.predict(X_test_scaled)
from sklearn.metrics import classification_report, confusion_matrix
print("\nConfusion Matrix (rows=actual, cols=predicted):")
print("         Draw  Home  Away")
cm = confusion_matrix(y_test, test_pred)
for i, label in enumerate(['Draw', 'Home', 'Away']):
    print(f"{label:5s}  {cm[i]}")

print("\nClassification Report:")
print(classification_report(y_test, test_pred, target_names=['Draw', 'Home Win', 'Away Win']))

# Prepare team stats for prediction (include priors for newly promoted teams)
prediction_team_stats = {}
for team_name, stats in final_team_stats.items():
    prediction_team_stats[team_name] = {
        **stats,
        'prior_stats': prior_season_stats.get(team_name, None)
    }

# Save model
model_data = {
    'model': best_model,
    'scaler': scaler,
    'team_stats': prediction_team_stats,
    'prior_season_stats': prior_season_stats,
    'league_avg_stats': league_avg_stats,
    'cv_accuracy': best_cv_score,
    'test_accuracy': best_test_score,
    'accuracy': best_test_score,  # Use test accuracy as primary metric
    'model_type': best_name,
    'training_seasons': training_seasons,
    'test_season': current_season_id,
    'training_samples': len(X),
    'test_samples': len(X_test),
    'version': 3
}

with open(MODEL_OUT, 'wb') as f:
    pickle.dump(model_data, f)

print(f"\nModel saved to {MODEL_OUT}")
print(f"Teams included: {len(prediction_team_stats)}")
print(f"Returning teams with prior: {len([t for t, s in prediction_team_stats.items() if s.get('prior_stats')])}")
print(f"New/promoted teams: {len([t for t, s in prediction_team_stats.items() if not s.get('prior_stats')])}")

# Show some predictions
print("\n--- Sample Predictions ---")
teams = list(prediction_team_stats.keys())[:6]
for i in range(3):
    home = teams[i*2]
    away = teams[i*2 + 1]

    home_stats = prediction_team_stats[home]
    away_stats = prediction_team_stats[away]

    home_feats = calculate_features(
        home_stats, home, home_stats.get('prior_stats'),
        is_home=True, league_avg_stats=league_avg_stats
    )
    away_feats = calculate_features(
        away_stats, away, away_stats.get('prior_stats'),
        is_home=False, league_avg_stats=league_avg_stats
    )

    features = [
        home_feats['ppg'], away_feats['ppg'], home_feats['ppg'] - away_feats['ppg'],
        home_feats['form'], away_feats['form'], home_feats['form'] - away_feats['form'],
        home_feats['gpg'], away_feats['gpg'], home_feats['gpg'] - away_feats['gpg'],
        home_feats['gd'], away_feats['gd'], home_feats['gd'] - away_feats['gd'],
        home_feats['xgpg'], away_feats['xgpg'], home_feats['xgpg'] - away_feats['xgpg'],
        home_feats['xgapg'], away_feats['xgapg'],
        home_feats['xg_diff'], away_feats['xg_diff'], home_feats['xg_diff'] - away_feats['xg_diff'],
        home_feats['xg_form'], away_feats['xg_form'],
        home_feats['win_rate'], away_feats['win_rate'], home_feats['win_rate'] - away_feats['win_rate'],
        home_feats['venue_wr'], away_feats['venue_wr'],
        home_feats['shot_conv'], away_feats['shot_conv'],
        home_feats['sot_rate'], away_feats['sot_rate'],
        home_feats['cs_rate'], away_feats['cs_rate'],
        home_feats['venue_gpg'], away_feats['venue_gpg'],
    ]
    if USE_OBV_FEATURES:
        features += [home_feats['obvd'], away_feats['obvd'],
                     home_feats['obvd'] - away_feats['obvd']]

    X_pred = scaler.transform([features])
    proba = best_model.predict_proba(X_pred)[0]
    pred = best_model.predict(X_pred)[0]

    outcome = {0: 'Draw', 1: 'Home Win', 2: 'Away Win'}[pred]
    print(f"{home} vs {away}: {outcome} (H:{proba[1]:.0%}, D:{proba[0]:.0%}, A:{proba[2]:.0%})")
