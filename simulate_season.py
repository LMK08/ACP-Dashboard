#!/usr/bin/env python3
"""
Monte Carlo season simulation for Liga 3 and Campeonato de Portugal.
Produces position probability tables saved to season_simulation.pkl.
Run after update_model.py in the scheduled pipeline.
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from itertools import product
from collections import defaultdict

from league_config import COMPETITIONS

# ── Configuration ──────────────────────────────────────────────────────────────

N_SIMULATIONS = 10000

FIRST_STAGE_GROUPS = {
    'North': ['Fafe', 'Varzim', 'Paredes', 'Sanjoanense', 'São João Ver',
              'Amarante', 'Vitória Guimarães II', 'Trofense', 'Sporting Braga II', 'AD Marco 09'],
    'South': ['1º Dezembro', 'Caldas', 'Sporting Covilhã', 'Mafra', 'União Santarém',
              'Amora', 'Académica', 'CF Os Belenenses', 'Lusitano Évora 1911', 'Atlético CP'],
}

# Head-to-head tiebreaker overrides for first-stage positions.
# calculate_league_table uses GD, but FPF rules use head-to-head first.
# Format: {team: correct_position} — only needed for teams where h2h differs from GD order.
FIRST_STAGE_POSITION_OVERRIDES = {
    'Atlético CP': 5,          # h2h winner vs Lusitano (both 22 pts)
    'Lusitano Évora 1911': 6,
}


# ── Replicated helpers from app.py ────────────────────────────────────────────

def calculate_league_table(matches_df, team_list):
    """Calculate league standings for a list of teams."""
    standings = {}
    for team in team_list:
        standings[team] = {'P': 0, 'W': 0, 'D': 0, 'L': 0, 'GF': 0, 'GA': 0, 'GD': 0, 'Pts': 0}

    for _, match in matches_df.iterrows():
        home_team = match['homeTeamName']
        away_team = match['awayTeamName']
        score = match.get('score', '')

        if home_team not in team_list or away_team not in team_list:
            continue
        if not score or pd.isna(score) or '-' not in str(score):
            continue
        try:
            home_goals, away_goals = map(int, str(score).split('-'))
        except (ValueError, AttributeError):
            continue

        standings[home_team]['P'] += 1
        standings[home_team]['GF'] += home_goals
        standings[home_team]['GA'] += away_goals
        standings[away_team]['P'] += 1
        standings[away_team]['GF'] += away_goals
        standings[away_team]['GA'] += home_goals

        if home_goals > away_goals:
            standings[home_team]['W'] += 1
            standings[home_team]['Pts'] += 3
            standings[away_team]['L'] += 1
        elif home_goals < away_goals:
            standings[away_team]['W'] += 1
            standings[away_team]['Pts'] += 3
            standings[home_team]['L'] += 1
        else:
            standings[home_team]['D'] += 1
            standings[home_team]['Pts'] += 1
            standings[away_team]['D'] += 1
            standings[away_team]['Pts'] += 1

    for team in standings:
        standings[team]['GD'] = standings[team]['GF'] - standings[team]['GA']

    table_df = pd.DataFrame.from_dict(standings, orient='index')
    table_df.index.name = 'Team'
    table_df = table_df.reset_index()
    table_df = table_df.sort_values(by=['Pts', 'GD', 'GF'], ascending=[False, False, False]).reset_index(drop=True)
    table_df.insert(0, 'Pos', range(1, len(table_df) + 1))
    return table_df


def get_decay_weight(matches_played, decay_rate=0.15):
    return np.exp(-decay_rate * matches_played)


def get_blended_stat(current_value, current_matches, prior_per_game, decay_rate=0.15, default_prior=None):
    if current_matches == 0:
        return prior_per_game if prior_per_game is not None else (default_prior if default_prior else 0.0)
    current_per_game = current_value / current_matches
    if prior_per_game is None:
        return current_per_game
    prior_weight = get_decay_weight(current_matches, decay_rate)
    current_weight = 1 - prior_weight
    return current_weight * current_per_game + prior_weight * prior_per_game


def calculate_prediction_features(team_stats, prior_stats, league_avg, is_home):
    curr = team_stats
    m = curr['matches']

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
        prior_ppg = league_avg.get('ppg', 1.0) * 0.85
        prior_gpg = league_avg.get('gpg', 1.0) * 0.85
        prior_gapg = league_avg.get('gapg', 1.0) * 1.15
        prior_xgpg = league_avg.get('xgpg', 1.0) * 0.85
        prior_xgapg = league_avg.get('xgapg', 1.0) * 1.15
        prior_winrate = 0.28
        prior_csrate = league_avg.get('csrate', 0.25) * 0.85
        prior_shot_conv = league_avg.get('shot_conv', 0.1) * 0.9
        prior_sot_rate = league_avg.get('sot_rate', 0.35) * 0.95
        prior_venue_wr = 0.28
        prior_venue_gpg = prior_gpg

    decay_rate = 0.15
    ppg = get_blended_stat(curr['points'], m, prior_ppg, decay_rate)
    gpg = get_blended_stat(curr['goals_for'], m, prior_gpg, decay_rate)
    gapg = get_blended_stat(curr['goals_against'], m, prior_gapg, decay_rate)
    xgpg = get_blended_stat(curr['xG_for'], m, prior_xgpg, decay_rate)
    xgapg = get_blended_stat(curr['xG_against'], m, prior_xgapg, decay_rate)
    win_rate = get_blended_stat(curr['wins'], m, prior_winrate, decay_rate)
    cs_rate = get_blended_stat(curr['clean_sheets'], m, prior_csrate, decay_rate)

    curr_shot_conv = curr['goals_for'] / max(curr['shots_for'], 1) if m > 0 else 0
    curr_sot_rate = curr['sot_for'] / max(curr['shots_for'], 1) if m > 0 else 0
    shot_conv = curr_shot_conv if m > 3 else prior_shot_conv
    sot_rate = curr_sot_rate if m > 3 else prior_sot_rate

    venue_key = 'home' if is_home else 'away'
    venue_wr = get_blended_stat(curr[f'{venue_key}_wins'], curr[f'{venue_key}_matches'], prior_venue_wr, decay_rate)
    venue_gpg = get_blended_stat(curr[f'{venue_key}_goals'], curr[f'{venue_key}_matches'], prior_venue_gpg, decay_rate)

    gd = gpg - gapg
    xg_diff = xgpg - xgapg
    form = np.mean(curr['last_5_results'][-5:]) if curr['last_5_results'] else 1.0
    xg_form = np.mean(curr['last_5_xG'][-5:]) if curr['last_5_xG'] else prior_xgpg

    return {
        'ppg': ppg, 'gpg': gpg, 'gapg': gapg, 'xgpg': xgpg, 'xgapg': xgapg,
        'win_rate': win_rate, 'cs_rate': cs_rate, 'shot_conv': shot_conv,
        'sot_rate': sot_rate, 'gd': gd, 'xg_diff': xg_diff, 'form': form,
        'xg_form': xg_form, 'venue_wr': venue_wr, 'venue_gpg': venue_gpg,
    }


def calculate_maintenance_bonus(first_stage_table):
    """Calculate starting bonus points for maintenance group teams per FPF rules.

    Two cumulative bonuses based on 1st-phase classification and points:

    Classification bonus: 5th→6, 6th→5, 7th→4, 8th→3, 9th→2, 10th→1
    Points bonus: <15→0, 15-19→1, 20-24→2, 25-29→3, >29→4

    Exceptions:
      - <10 pts: starts at 0 (no bonuses at all)
      - 11-14 pts: only classification bonus (no points bonus)
      - >14 pts: both bonuses cumulated
    """
    classification_bonus = {5: 6, 6: 5, 7: 4, 8: 3, 9: 2, 10: 1}
    points_bonus_tiers = [
        (15, 0),   # < 15
        (20, 1),   # 15-19
        (25, 2),   # 20-24
        (30, 3),   # 25-29
        (999, 4),  # > 29
    ]

    bonuses = {}
    for _, row in first_stage_table.iterrows():
        team = row['Team']
        pos = row['Pos']
        pts = row['Pts']

        # Only maintenance teams (positions 5-10)
        if pos < 5:
            continue

        if pts < 10:
            # Exception i: starts at 0, no bonuses
            bonuses[team] = 0
        elif pts <= 14:
            # Exception ii: only classification bonus
            bonuses[team] = classification_bonus.get(pos, 0)
        else:
            # Exception iii: both bonuses cumulated
            cls_bonus = classification_bonus.get(pos, 0)
            pts_bonus = 0
            for threshold, bonus in points_bonus_tiers:
                if pts < threshold:
                    pts_bonus = bonus
                    break
            bonuses[team] = cls_bonus + pts_bonus

    return bonuses


def build_feature_vector(home_feats, away_feats):
    return [
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


# ── Campeonato group detection ───────────────────────────────────────────────

def detect_campeonato_groups(matches_df):
    """Detect the 4 regional groups in Campeonato de Portugal via opponent adjacency.
    Returns dict of {group_label: [team_names]}.
    """
    opponents = defaultdict(set)
    for _, m in matches_df.iterrows():
        home = m.get('homeTeamName')
        away = m.get('awayTeamName')
        if pd.isna(home) or pd.isna(away):
            continue
        opponents[home].add(away)
        opponents[away].add(home)

    groups = []
    assigned = set()
    for team in sorted(opponents):
        if team in assigned:
            continue
        group = {team} | opponents[team]
        groups.append(sorted(group))
        assigned.update(group)

    # Label groups alphabetically (Série A, B, C, D)
    labels = ['Série A', 'Série B', 'Série C', 'Série D']
    result = {}
    for i, g in enumerate(sorted(groups, key=lambda x: x[0])):
        label = labels[i] if i < len(labels) else f'Group {i+1}'
        result[label] = g

    return result


def simulate_campeonato(matches_df, model, scaler, team_stats, prior_season_stats, league_avg_stats):
    """Run Campeonato de Portugal simulation (Phase 1 group standings)."""
    print("=" * 60)
    print("Campeonato de Portugal Season Simulation")
    print("=" * 60)

    played_matches = matches_df[matches_df['status'] == 'Played']
    groups = detect_campeonato_groups(played_matches)
    print(f"\nDetected {len(groups)} groups")

    results = {}

    for group_name, group_teams in groups.items():
        print(f"\n{'─' * 60}")
        print(f"Processing: {group_name} ({len(group_teams)} teams)")

        current_table = calculate_league_table(played_matches, group_teams)
        print(f"  Played matches: {current_table['P'].sum() // 2}")

        # Full double round-robin fixtures
        full_fixtures = [(h, a) for h, a in product(group_teams, group_teams) if h != a]

        played = set()
        for _, m in played_matches.iterrows():
            h, a = m['homeTeamName'], m['awayTeamName']
            if h in group_teams and a in group_teams:
                played.add((h, a))

        remaining_fixtures = [f for f in full_fixtures if f not in played]
        print(f"  Total fixtures: {len(full_fixtures)}, Played: {len(played)}, Remaining: {len(remaining_fixtures)}")

        # Pre-compute match probabilities
        match_probs = {}
        for home, away in remaining_fixtures:
            home_cum = team_stats.get(home)
            away_cum = team_stats.get(away)

            if not home_cum or not away_cum:
                match_probs[(home, away)] = np.array([0.25, 0.45, 0.30])
                continue

            home_prior = home_cum.get('prior_stats') or prior_season_stats.get(home)
            away_prior = away_cum.get('prior_stats') or prior_season_stats.get(away)

            home_feats = calculate_prediction_features(home_cum, home_prior, league_avg_stats, is_home=True)
            away_feats = calculate_prediction_features(away_cum, away_prior, league_avg_stats, is_home=False)

            fv = build_feature_vector(home_feats, away_feats)
            X = scaler.transform([fv])
            proba = model.predict_proba(X)[0]
            match_probs[(home, away)] = proba

        # Monte Carlo
        print(f"  Running {N_SIMULATIONS:,} simulations...")
        n_teams = len(group_teams)
        position_counts = {team: np.zeros(n_teams, dtype=int) for team in group_teams}

        starting_pts = {}
        starting_gd = {}
        starting_gf = {}
        for _, row in current_table.iterrows():
            team = row['Team']
            starting_pts[team] = row['Pts']
            starting_gd[team] = row['GD']
            starting_gf[team] = row['GF']

        rng = np.random.default_rng(seed=42)

        fixture_list = list(remaining_fixtures)
        if fixture_list:
            prob_matrix = np.array([match_probs[f] for f in fixture_list])
            cumulative_probs = np.cumsum(prob_matrix, axis=1)
        else:
            cumulative_probs = np.array([])

        for sim in range(N_SIMULATIONS):
            pts = dict(starting_pts)
            gd = dict(starting_gd)
            gf = dict(starting_gf)

            if len(fixture_list) > 0:
                rand_vals = rng.random(len(fixture_list))
                for idx, (home, away) in enumerate(fixture_list):
                    r = rand_vals[idx]
                    cp = cumulative_probs[idx]

                    if r < cp[0]:
                        pts[home] += 1
                        pts[away] += 1
                        gf[home] += 1
                        gf[away] += 1
                    elif r < cp[1]:
                        pts[home] += 3
                        gd[home] += 1
                        gd[away] -= 1
                        gf[home] += 2
                        gf[away] += 1
                    else:
                        pts[away] += 3
                        gd[away] += 1
                        gd[home] -= 1
                        gf[away] += 2
                        gf[home] += 1

            final = sorted(group_teams, key=lambda t: (pts[t], gd[t], gf[t]), reverse=True)
            for pos, team in enumerate(final):
                position_counts[team][pos] += 1

        pos_prob_df = pd.DataFrame(
            {team: position_counts[team] / N_SIMULATIONS for team in group_teams},
        ).T
        pos_prob_df.columns = [f'{i+1}' for i in range(n_teams)]
        pos_prob_df.index.name = 'Team'

        pos_prob_df['expected_pos'] = sum((i + 1) * pos_prob_df[f'{i+1}'] for i in range(n_teams))
        pos_prob_df = pos_prob_df.sort_values('expected_pos')
        pos_prob_df = pos_prob_df.drop(columns='expected_pos')

        print(f"\n  Position probabilities:")
        print(pos_prob_df.to_string(float_format=lambda x: f'{x:.1%}'))

        results[group_name] = {
            'teams': group_teams,
            'position_probabilities': pos_prob_df,
            'current_standings': current_table,
            'matches_remaining': len(remaining_fixtures),
            'bonus_points': {},
        }

    return results


# ── Main simulation pipeline ─────────────────────────────────────────────────

def simulate_liga3(matches_df, model, scaler, team_stats, prior_season_stats, league_avg_stats):
    """Run Liga 3 season simulation (second-stage groups)."""
    SEASON_ID = COMPETITIONS[43324]["current_season"]

    print("=" * 60)
    print("Liga 3 Season Simulation")
    print("=" * 60)

    season_matches = matches_df[matches_df['seasonId'] == SEASON_ID].copy()
    print(f"  {len(season_matches)} matches in season {SEASON_ID}")

    if season_matches.empty:
        print("  No matches found, skipping Liga 3 simulation.")
        return {}

    # Determine first-stage roundId (the one with the most matches)
    round_counts = season_matches.groupby('roundId').size()
    first_stage_round_id = round_counts.idxmax()
    print(f"  First-stage roundId: {first_stage_round_id} ({round_counts[first_stage_round_id]} matches)")

    first_stage_matches = season_matches[season_matches['roundId'] == first_stage_round_id]
    second_stage_matches = season_matches[season_matches['roundId'] != first_stage_round_id]

    all_north = FIRST_STAGE_GROUPS['North']
    all_south = FIRST_STAGE_GROUPS['South']

    north_table = calculate_league_table(first_stage_matches, all_north)
    south_table = calculate_league_table(first_stage_matches, all_south)

    # Apply head-to-head tiebreaker overrides
    for table in [north_table, south_table]:
        overridden = table['Team'].map(FIRST_STAGE_POSITION_OVERRIDES)
        if overridden.notna().any():
            for idx in table.index:
                team = table.loc[idx, 'Team']
                if team in FIRST_STAGE_POSITION_OVERRIDES:
                    table.loc[idx, 'Pos'] = FIRST_STAGE_POSITION_OVERRIDES[team]
            table.sort_values('Pos', inplace=True)
            table.reset_index(drop=True, inplace=True)

    print(f"\n  North first-stage standings ({len(north_table)} teams):")
    for _, r in north_table.iterrows():
        print(f"    {r['Pos']:2}. {r['Team']:25} {r['Pts']:2} pts  GD {r['GD']:+d}")

    print(f"\n  South first-stage standings ({len(south_table)} teams):")
    for _, r in south_table.iterrows():
        print(f"    {r['Pos']:2}. {r['Team']:25} {r['Pts']:2} pts  GD {r['GD']:+d}")

    promotion_teams = list(north_table.head(4)['Team']) + list(south_table.head(4)['Team'])
    north_maintenance_teams = list(north_table.tail(6)['Team'])
    south_maintenance_teams = list(south_table.tail(6)['Team'])

    north_bonuses = calculate_maintenance_bonus(north_table)
    south_bonuses = calculate_maintenance_bonus(south_table)
    all_bonuses = {**north_bonuses, **south_bonuses}

    groups = {
        'Promotion': promotion_teams,
        'North Maintenance': north_maintenance_teams,
        'South Maintenance': south_maintenance_teams,
    }

    print(f"\n  Promotion group: {promotion_teams}")
    print(f"  North Maintenance: {north_maintenance_teams}")
    print(f"  South Maintenance: {south_maintenance_teams}")

    print(f"\n  Maintenance bonuses:")
    for team, bonus in sorted(all_bonuses.items(), key=lambda x: -x[1]):
        print(f"    {team:25} +{bonus} pts")

    results = {}

    for group_name, group_teams in groups.items():
        print(f"\n{'─' * 60}")
        print(f"Processing: {group_name} ({len(group_teams)} teams)")

        current_table = calculate_league_table(second_stage_matches, group_teams)
        print(f"  Played second-stage matches: {current_table['P'].sum() // 2}")

        full_fixtures = [(h, a) for h, a in product(group_teams, group_teams) if h != a]

        played = set()
        for _, m in second_stage_matches.iterrows():
            h, a = m['homeTeamName'], m['awayTeamName']
            if h in group_teams and a in group_teams:
                played.add((h, a))

        remaining_fixtures = [f for f in full_fixtures if f not in played]
        print(f"  Total fixtures: {len(full_fixtures)}, Played: {len(played)}, Remaining: {len(remaining_fixtures)}")

        match_probs = {}
        for home, away in remaining_fixtures:
            home_cum = team_stats.get(home)
            away_cum = team_stats.get(away)

            if not home_cum or not away_cum:
                match_probs[(home, away)] = np.array([0.25, 0.45, 0.30])
                continue

            home_prior = home_cum.get('prior_stats') or prior_season_stats.get(home)
            away_prior = away_cum.get('prior_stats') or prior_season_stats.get(away)

            home_feats = calculate_prediction_features(home_cum, home_prior, league_avg_stats, is_home=True)
            away_feats = calculate_prediction_features(away_cum, away_prior, league_avg_stats, is_home=False)

            fv = build_feature_vector(home_feats, away_feats)
            X = scaler.transform([fv])
            proba = model.predict_proba(X)[0]
            match_probs[(home, away)] = proba

        print(f"  Running {N_SIMULATIONS:,} simulations...")
        n_teams = len(group_teams)
        position_counts = {team: np.zeros(n_teams, dtype=int) for team in group_teams}

        starting_pts = {}
        starting_gd = {}
        starting_gf = {}
        bonus_pts = {}
        for _, row in current_table.iterrows():
            team = row['Team']
            bonus = all_bonuses.get(team, 0) if group_name != 'Promotion' else 0
            bonus_pts[team] = bonus
            starting_pts[team] = row['Pts'] + bonus
            starting_gd[team] = row['GD']
            starting_gf[team] = row['GF']

        if group_name != 'Promotion':
            print(f"  Starting points (with bonuses):")
            for team in sorted(starting_pts, key=lambda t: -starting_pts[t]):
                print(f"    {team:25} {starting_pts[team]:2} pts (bonus: +{bonus_pts[team]})")

        rng = np.random.default_rng(seed=42)

        fixture_list = list(remaining_fixtures)
        if fixture_list:
            prob_matrix = np.array([match_probs[f] for f in fixture_list])
            cumulative_probs = np.cumsum(prob_matrix, axis=1)
        else:
            cumulative_probs = np.array([])

        for sim in range(N_SIMULATIONS):
            pts = dict(starting_pts)
            gd = dict(starting_gd)
            gf = dict(starting_gf)

            if len(fixture_list) > 0:
                rand_vals = rng.random(len(fixture_list))
                for idx, (home, away) in enumerate(fixture_list):
                    r = rand_vals[idx]
                    cp = cumulative_probs[idx]

                    if r < cp[0]:
                        pts[home] += 1
                        pts[away] += 1
                        gf[home] += 1
                        gf[away] += 1
                    elif r < cp[1]:
                        pts[home] += 3
                        gd[home] += 1
                        gd[away] -= 1
                        gf[home] += 2
                        gf[away] += 1
                    else:
                        pts[away] += 3
                        gd[away] += 1
                        gd[home] -= 1
                        gf[away] += 2
                        gf[home] += 1

            final = sorted(group_teams, key=lambda t: (pts[t], gd[t], gf[t]), reverse=True)
            for pos, team in enumerate(final):
                position_counts[team][pos] += 1

        pos_prob_df = pd.DataFrame(
            {team: position_counts[team] / N_SIMULATIONS for team in group_teams},
        ).T
        pos_prob_df.columns = [f'{i+1}' for i in range(n_teams)]
        pos_prob_df.index.name = 'Team'

        pos_prob_df['expected_pos'] = sum((i + 1) * pos_prob_df[f'{i+1}'] for i in range(n_teams))
        pos_prob_df = pos_prob_df.sort_values('expected_pos')
        pos_prob_df = pos_prob_df.drop(columns='expected_pos')

        print(f"\n  Position probabilities:")
        print(pos_prob_df.to_string(float_format=lambda x: f'{x:.1%}'))

        row_sums = pos_prob_df.sum(axis=1)
        assert all(abs(s - 1.0) < 0.01 for s in row_sums), f"Row sums not ~1.0: {row_sums.to_dict()}"

        results[group_name] = {
            'teams': group_teams,
            'position_probabilities': pos_prob_df,
            'current_standings': current_table,
            'matches_remaining': len(remaining_fixtures),
            'bonus_points': bonus_pts,
        }

    return results


def main():
    print("\nLoading model and match data...")
    with open('match_predictor_model.pkl', 'rb') as f:
        model_data = pickle.load(f)

    model = model_data['model']
    scaler = model_data['scaler']
    team_stats = model_data['team_stats']
    prior_season_stats = model_data.get('prior_season_stats', {})
    league_avg_stats = model_data.get('league_avg_stats', {
        'ppg': 1.0, 'gpg': 1.19, 'gapg': 1.19, 'xgpg': 1.0,
        'xgapg': 1.0, 'csrate': 0.25, 'shot_conv': 0.1, 'sot_rate': 0.35,
    })

    matches_df = pd.read_parquet('matches_summary.parquet')

    all_results = {}

    # Liga 3 simulation
    try:
        liga3_results = simulate_liga3(matches_df, model, scaler, team_stats, prior_season_stats, league_avg_stats)
        if liga3_results:
            all_results[43324] = {
                'competition_name': 'Liga 3',
                'season_id': COMPETITIONS[43324]["current_season"],
                'groups': liga3_results,
            }
    except Exception as e:
        print(f"\n⚠️ Liga 3 simulation failed: {e}")

    # Campeonato simulation
    try:
        camp_season_id = COMPETITIONS[702]["current_season"]
        camp_matches = matches_df[matches_df['seasonId'] == camp_season_id]
        if not camp_matches.empty:
            camp_results = simulate_campeonato(camp_matches, model, scaler, team_stats, prior_season_stats, league_avg_stats)
            if camp_results:
                all_results[702] = {
                    'competition_name': 'Campeonato',
                    'season_id': camp_season_id,
                    'groups': camp_results,
                }
        else:
            print("\n⚠️ No Campeonato matches found, skipping simulation.")
    except Exception as e:
        print(f"\n⚠️ Campeonato simulation failed: {e}")

    # Save combined results (backward compatible: also keep 'groups' at top level for Liga 3)
    output = {
        'timestamp': datetime.now().isoformat(),
        'n_simulations': N_SIMULATIONS,
        'competitions': all_results,
        # Backward compat: keep top-level 'groups' and 'season_id' for Liga 3
        'season_id': COMPETITIONS[43324]["current_season"],
        'groups': all_results.get(43324, {}).get('groups', {}),
    }

    with open('season_simulation.pkl', 'wb') as f:
        pickle.dump(output, f)

    print(f"\n{'=' * 60}")
    print(f"Saved season_simulation.pkl ({len(all_results)} competition(s))")
    print(f"Timestamp: {output['timestamp']}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
