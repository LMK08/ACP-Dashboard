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

# Reserve / B teams ineligible for promotion. They keep their league standings
# but the playoff qualification slot drops to the next eligible team.
PROMOTION_INELIGIBLE_TEAMS = {
    'FC Alverca II',
    'Vitória Guimarães II',
    'Sporting Braga II',
}


def is_promotion_eligible(team_name: str) -> bool:
    """Whether this team can be promoted (excludes reserve / B teams)."""
    if team_name in PROMOTION_INELIGIBLE_TEAMS:
        return False
    # Heuristic fallback for teams ending in " II" or " B" not on the list above.
    s = team_name.strip()
    return not (s.endswith(' II') or s.endswith(' B'))


def top_n_eligible(standings_in_order, n=2):
    """Pick the top n teams from an ordered standings list, skipping any
    team that isn't promotion-eligible (reserve / B teams)."""
    out = []
    for t in standings_in_order:
        if is_promotion_eligible(t):
            out.append(t)
            if len(out) >= n:
                break
    return out


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


# Decay rate for the league-average prior. At rate 0.30, prior weight drops
# to 22% after 5 matches, 5% after 10, ~1% after 15, ~0.04% after 26. This
# reflects that at this tier (Campeonato / Liga 3) heavy roster turnover and
# cross-tier moves make prior-season data unreliable, so the start-of-season
# baseline is league-average and current-season form takes over within ~10
# matches.
DEFAULT_DECAY_RATE = 0.30


def get_decay_weight(matches_played, decay_rate=DEFAULT_DECAY_RATE):
    return np.exp(-decay_rate * matches_played)


def get_blended_stat(current_value, current_matches, prior_per_game, decay_rate=DEFAULT_DECAY_RATE, default_prior=None):
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

    # League-average prior baseline. We deliberately do NOT use the team's
    # prior_stats: at this tier the data is too unreliable (different
    # competition, year-old, heavy roster turnover). Instead every team
    # starts the season anchored to a slight-below-average baseline and the
    # decay below pulls in current-season form quickly.
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

    decay_rate = DEFAULT_DECAY_RATE
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
    """Run Campeonato de Portugal simulation (Phase 1 + promotion playoffs)."""
    print("=" * 60)
    print("Campeonato de Portugal Season Simulation")
    print("=" * 60)

    played_matches = matches_df[matches_df['status'] == 'Played']
    # Separate first-stage matches (the largest round) from promotion-playoff
    # matches so first-stage group detection and standings aren't polluted by
    # cross-group playoff fixtures.
    if 'roundId' in played_matches.columns and played_matches['roundId'].nunique() > 1:
        first_stage_round = played_matches['roundId'].value_counts().idxmax()
        first_stage_played = played_matches[played_matches['roundId'] == first_stage_round]
        playoff_played = played_matches[played_matches['roundId'] != first_stage_round]
        print(f"\nFirst-stage round: {first_stage_round} ({len(first_stage_played)} matches)")
        print(f"Playoff matches detected: {len(playoff_played)}")
    else:
        first_stage_played = played_matches
        playoff_played = played_matches.iloc[0:0]

    groups = detect_campeonato_groups(first_stage_played)
    print(f"\nDetected {len(groups)} groups")

    results = {}
    group_names_sorted = sorted(groups.keys())  # Série A, B, C, D

    # ── Phase 1: Simulate each group and store per-sim standings ──
    group_sim_data = {}  # {group_name: {sim_idx: [team1, team2, ...] sorted by final pos}}

    for group_name in group_names_sorted:
        group_teams = groups[group_name]
        print(f"\n{'─' * 60}")
        print(f"Processing: {group_name} ({len(group_teams)} teams)")

        current_table = calculate_league_table(first_stage_played, group_teams)
        print(f"  Played matches: {current_table['P'].sum() // 2}")

        # Full double round-robin fixtures
        full_fixtures = [(h, a) for h, a in product(group_teams, group_teams) if h != a]

        played = set()
        for _, m in first_stage_played.iterrows():
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

        rng = np.random.default_rng(seed=42 + hash(group_name) % 1000)

        fixture_list = list(remaining_fixtures)
        if fixture_list:
            prob_matrix = np.array([match_probs[f] for f in fixture_list])
            cumulative_probs = np.cumsum(prob_matrix, axis=1)
        else:
            cumulative_probs = np.array([])

        per_sim_standings = []  # Store final standings per sim for playoff pairing

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
            per_sim_standings.append(final)

        group_sim_data[group_name] = per_sim_standings

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
            'playoff_pct': {},
            'promotion_pct': {},
        }

    # ── Phase 2: Promotion playoff simulation ──
    # Playoff pairings (FPF 2025/26): Série A + Série C → Playoff Group 1,
    # Série B + Série D → Playoff Group 2. Top 2 promotion-eligible teams
    # from each série qualify; top 2 from each playoff group get promoted.
    playoff_pairings = []
    if len(group_names_sorted) >= 4:
        playoff_pairings = [
            (group_names_sorted[0], group_names_sorted[2]),  # Série A + C
            (group_names_sorted[1], group_names_sorted[3]),  # Série B + D
        ]
    elif len(group_names_sorted) == 2:
        playoff_pairings = [(group_names_sorted[0], group_names_sorted[1])]

    if playoff_pairings:
        print(f"\n{'=' * 60}")
        print("Promotion Playoff Simulation")
        print(f"{'=' * 60}")

        # Track playoff qualification and promotion counts per team
        playoff_counts = {}  # team -> count of making top 2 in série
        promotion_counts = {}  # team -> count of finishing top 2 in playoff group
        all_teams = set()
        for g in groups.values():
            for t in g:
                all_teams.add(t)
                playoff_counts[t] = 0
                promotion_counts[t] = 0

        # Pre-compute playoff match probabilities for all possible team pairs
        # (we don't know which teams will qualify until each sim)
        playoff_match_probs_cache = {}

        def get_playoff_match_prob(home, away):
            key = (home, away)
            if key not in playoff_match_probs_cache:
                home_cum = team_stats.get(home)
                away_cum = team_stats.get(away)
                if not home_cum or not away_cum:
                    playoff_match_probs_cache[key] = np.array([0.25, 0.45, 0.30])
                else:
                    home_prior = home_cum.get('prior_stats') or prior_season_stats.get(home)
                    away_prior = away_cum.get('prior_stats') or prior_season_stats.get(away)
                    home_feats = calculate_prediction_features(home_cum, home_prior, league_avg_stats, is_home=True)
                    away_feats = calculate_prediction_features(away_cum, away_prior, league_avg_stats, is_home=False)
                    fv = build_feature_vector(home_feats, away_feats)
                    X = scaler.transform([fv])
                    playoff_match_probs_cache[key] = model.predict_proba(X)[0]
            return playoff_match_probs_cache[key]

        playoff_rng = np.random.default_rng(seed=99)

        for sim in range(N_SIMULATIONS):
            for group_a_name, group_b_name in playoff_pairings:
                standings_a = group_sim_data[group_a_name][sim]
                standings_b = group_sim_data[group_b_name][sim]
                # Skip ineligible (reserve) teams; the slot drops to the next.
                top2_a = top_n_eligible(standings_a, 2)
                top2_b = top_n_eligible(standings_b, 2)

                # Track playoff qualification
                for t in top2_a + top2_b:
                    playoff_counts[t] += 1

                # Simulate 4-team round-robin playoff (6 matches)
                playoff_teams = top2_a + top2_b
                playoff_pts = {t: 0 for t in playoff_teams}
                playoff_gd = {t: 0 for t in playoff_teams}
                playoff_gf = {t: 0 for t in playoff_teams}

                for h in playoff_teams:
                    for a in playoff_teams:
                        if h == a:
                            continue
                        proba = get_playoff_match_prob(h, a)
                        r = playoff_rng.random()
                        if r < proba[0]:  # Draw
                            playoff_pts[h] += 1
                            playoff_pts[a] += 1
                            playoff_gf[h] += 1
                            playoff_gf[a] += 1
                        elif r < proba[0] + proba[1]:  # Home win
                            playoff_pts[h] += 3
                            playoff_gd[h] += 1
                            playoff_gd[a] -= 1
                            playoff_gf[h] += 2
                            playoff_gf[a] += 1
                        else:  # Away win
                            playoff_pts[a] += 3
                            playoff_gd[a] += 1
                            playoff_gd[h] -= 1
                            playoff_gf[a] += 2
                            playoff_gf[h] += 1

                # Top 2 in playoff group get promoted
                playoff_final = sorted(playoff_teams, key=lambda t: (playoff_pts[t], playoff_gd[t], playoff_gf[t]), reverse=True)
                for t in playoff_final[:2]:
                    promotion_counts[t] += 1

        # Store playoff and promotion percentages in each group's results
        for group_name, group_teams in groups.items():
            results[group_name]['playoff_pct'] = {t: playoff_counts[t] / N_SIMULATIONS for t in group_teams}
            results[group_name]['promotion_pct'] = {t: promotion_counts[t] / N_SIMULATIONS for t in group_teams}

        # Print summary
        for pairing_idx, (ga, gb) in enumerate(playoff_pairings):
            print(f"\n  Playoff Group {pairing_idx + 1} ({ga} + {gb}):")
            combined = list(groups[ga]) + list(groups[gb])
            combined.sort(key=lambda t: promotion_counts[t], reverse=True)
            for t in combined[:8]:
                pq = playoff_counts[t] / N_SIMULATIONS
                pp = promotion_counts[t] / N_SIMULATIONS
                if pq > 0.005:
                    print(f"    {t:30s}  Playoff: {pq:6.1%}  Promotion: {pp:6.1%}")

    # ── Phase 3: actual promotion-playoff group simulation ───────────────
    # Once playoff matches start, use real fixtures + results rather than
    # the séries-based projection. Adds 'Promotion Playoff Group N' entries
    # to results.
    canonical_playoff_groups: list[list[str]] = []
    if playoff_pairings:
        for ga, gb in playoff_pairings:
            # Final first-stage standings determine the qualifiers.
            final_a = list(results[ga]['current_standings']['Team'])
            final_b = list(results[gb]['current_standings']['Team'])
            qualifiers = top_n_eligible(final_a, 2) + top_n_eligible(final_b, 2)
            if len(qualifiers) == 4:
                canonical_playoff_groups.append(qualifiers)

    playoff_groups = simulate_promotion_playoff(
        matches_df, model, scaler, team_stats,
        prior_season_stats, league_avg_stats,
        first_stage_round=(first_stage_played['roundId'].iloc[0]
                            if 'roundId' in first_stage_played.columns and len(first_stage_played) > 0
                            else None),
        canonical_groups=canonical_playoff_groups,
    )
    results.update(playoff_groups)

    return results


def simulate_promotion_playoff(matches_df, model, scaler, team_stats,
                                 prior_season_stats, league_avg_stats,
                                 first_stage_round=None,
                                 canonical_groups=None):
    """Detect Campeonato promotion-playoff matches in `matches_df` and
    simulate the remaining 4-team round-robin per playoff group.

    `canonical_groups` (optional): pre-computed list of [team_a, team_b,
    team_c, team_d] for each playoff group, derived from the final
    first-stage standings. If supplied, used directly; otherwise the
    grouping is inferred from played playoff matches via adjacency.

    A "playoff" match is any match in this season's data whose roundId
    is not the main first-stage round. Returns a dict keyed by
    'Promotion Playoff Group {N}' (empty if no playoff matches yet)."""
    if 'roundId' not in matches_df.columns:
        return {}
    if first_stage_round is None:
        first_stage_round = matches_df['roundId'].value_counts().idxmax()

    playoff_all = matches_df[matches_df['roundId'] != first_stage_round].copy()
    if playoff_all.empty:
        return {}

    played = playoff_all[playoff_all['status'] == 'Played'].copy()
    if played.empty:
        # Playoff round exists in fixture list but nothing played yet —
        # don't try to guess groups. Return empty so the séries-based
        # projection above remains the source of truth.
        return {}

    print(f"\n{'=' * 60}")
    print("Promotion Playoff (actual matches) Simulation")
    print(f"{'=' * 60}")
    print(f"  Played playoff matches: {len(played)}  Total playoff fixtures: {len(playoff_all)}")

    # Prefer caller-supplied canonical groups (4 teams each, derived from
    # final séries standings + reserve-team filter). Preserve the supplied
    # group order (A+C → Group 1, B+D → Group 2 for the Campeonato).
    components = None
    if canonical_groups:
        components = [sorted(g) for g in canonical_groups if len(g) == 4]
        canonical_supplied = bool(components)
    else:
        canonical_supplied = False

    if not components:
        # Fall back to adjacency-based detection
        adj = defaultdict(set)
        teams_seen = set()
        for _, m in played.iterrows():
            h, a = m['homeTeamName'], m['awayTeamName']
            if pd.isna(h) or pd.isna(a):
                continue
            adj[h].add(a); adj[a].add(h)
            teams_seen.add(h); teams_seen.add(a)

        components = []
        visited = set()
        for t in sorted(teams_seen):
            if t in visited:
                continue
            comp = set(); queue = [t]
            while queue:
                x = queue.pop(0)
                if x in comp:
                    continue
                comp.add(x)
                queue.extend(adj[x] - comp)
            components.append(sorted(comp))
            visited |= comp

        # If still too small, fall through to the full-fixture adjacency
        if any(len(c) < 4 for c in components):
            full_adj = defaultdict(set)
            for _, m in playoff_all.iterrows():
                h, a = m['homeTeamName'], m['awayTeamName']
                if pd.isna(h) or pd.isna(a):
                    continue
                full_adj[h].add(a); full_adj[a].add(h)
            merged_components = []
            merged_visited = set()
            for t in sorted(full_adj):
                if t in merged_visited:
                    continue
                comp = set(); queue = [t]
                while queue:
                    x = queue.pop(0)
                    if x in comp:
                        continue
                    comp.add(x)
                    queue.extend(full_adj[x] - comp)
                merged_components.append(sorted(comp))
                merged_visited |= comp
            components = merged_components

    if not canonical_supplied:
        components.sort()
    if not components:
        return {}

    results = {}
    for idx, group_teams in enumerate(components, 1):
        group_name = f'Promotion Playoff Group {idx}'
        n_teams = len(group_teams)
        print(f"\n  {group_name}: {group_teams}")

        # Current standings within group, only counting group-internal matches
        group_played = played[
            played['homeTeamName'].isin(group_teams)
            & played['awayTeamName'].isin(group_teams)
        ]
        current_table = calculate_league_table(group_played, group_teams)

        # Remaining fixtures = full double round-robin minus already-played
        # matches. (FPF Campeonato playoff format is double round-robin: each
        # team plays every other in the group home and away.) If the schedule
        # parquet later includes unplayed fixtures we'd rather rely on those,
        # but with the current data we don't have unplayed playoff fixtures
        # listed as separate rows, so a constructive DRR is the correct
        # default once any playoff match has been played.
        played_pairs = set(zip(group_played['homeTeamName'], group_played['awayTeamName']))
        full_fixtures = [(h, a) for h, a in product(group_teams, group_teams) if h != a]
        scheduled_unplayed = playoff_all[
            playoff_all['homeTeamName'].isin(group_teams)
            & playoff_all['awayTeamName'].isin(group_teams)
            & (playoff_all['status'] != 'Played')
        ]
        if not scheduled_unplayed.empty:
            scheduled_pairs = set(zip(scheduled_unplayed['homeTeamName'],
                                        scheduled_unplayed['awayTeamName']))
            remaining_fixtures = [
                f for f in full_fixtures
                if f in scheduled_pairs and f not in played_pairs
            ]
        else:
            remaining_fixtures = [f for f in full_fixtures if f not in played_pairs]

        # Match probabilities
        match_probs = {}
        for h, a in remaining_fixtures:
            home_cum = team_stats.get(h); away_cum = team_stats.get(a)
            if not home_cum or not away_cum:
                match_probs[(h, a)] = np.array([0.25, 0.45, 0.30])
                continue
            home_prior = home_cum.get('prior_stats') or prior_season_stats.get(h)
            away_prior = away_cum.get('prior_stats') or prior_season_stats.get(a)
            home_feats = calculate_prediction_features(home_cum, home_prior, league_avg_stats, is_home=True)
            away_feats = calculate_prediction_features(away_cum, away_prior, league_avg_stats, is_home=False)
            fv = build_feature_vector(home_feats, away_feats)
            X = scaler.transform([fv])
            match_probs[(h, a)] = model.predict_proba(X)[0]

        # Starting points/GD/GF from current_table
        starting_pts = {}; starting_gd = {}; starting_gf = {}
        for _, row in current_table.iterrows():
            t = row['Team']
            starting_pts[t] = row['Pts']
            starting_gd[t] = row['GD']
            starting_gf[t] = row['GF']

        position_counts = {t: np.zeros(n_teams, dtype=int) for t in group_teams}
        promo_counts = {t: 0 for t in group_teams}
        rng = np.random.default_rng(seed=2026 + idx)

        if remaining_fixtures:
            prob_matrix = np.array([match_probs[f] for f in remaining_fixtures])
            cumulative_probs = np.cumsum(prob_matrix, axis=1)
        else:
            cumulative_probs = np.array([])

        for sim in range(N_SIMULATIONS):
            pts = dict(starting_pts); gd = dict(starting_gd); gf = dict(starting_gf)
            if len(remaining_fixtures) > 0:
                rand_vals = rng.random(len(remaining_fixtures))
                for i, (h, a) in enumerate(remaining_fixtures):
                    r = rand_vals[i]; cp = cumulative_probs[i]
                    if r < cp[0]:
                        pts[h] += 1; pts[a] += 1; gf[h] += 1; gf[a] += 1
                    elif r < cp[1]:
                        pts[h] += 3; gd[h] += 1; gd[a] -= 1; gf[h] += 2; gf[a] += 1
                    else:
                        pts[a] += 3; gd[a] += 1; gd[h] -= 1; gf[a] += 2; gf[h] += 1
            final = sorted(group_teams, key=lambda t: (pts[t], gd[t], gf[t]), reverse=True)
            for pos, t in enumerate(final):
                position_counts[t][pos] += 1
            for t in final[:2]:
                promo_counts[t] += 1

        pos_prob_df = pd.DataFrame(
            {t: position_counts[t] / N_SIMULATIONS for t in group_teams}
        ).T
        pos_prob_df.columns = [f'{i+1}' for i in range(n_teams)]
        pos_prob_df.index.name = 'Team'
        pos_prob_df['expected_pos'] = sum((i + 1) * pos_prob_df[f'{i+1}'] for i in range(n_teams))
        pos_prob_df = pos_prob_df.sort_values('expected_pos').drop(columns='expected_pos')

        results[group_name] = {
            'teams': group_teams,
            'position_probabilities': pos_prob_df,
            'current_standings': current_table,
            'matches_remaining': len(remaining_fixtures),
            'bonus_points': {},
            'promotion_pct': {t: promo_counts[t] / N_SIMULATIONS for t in group_teams},
        }

        # Print summary
        for t in pos_prob_df.index:
            pp = promo_counts[t] / N_SIMULATIONS
            print(f"    {t:30s}  P {current_table.loc[current_table['Team']==t,'P'].iloc[0]:>2}  "
                  f"Pts {current_table.loc[current_table['Team']==t,'Pts'].iloc[0]:>2}  "
                  f"Promo {pp:6.1%}")

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
