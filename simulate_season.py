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

# SEASON-SPECIFIC — update at every season turnover (must match the current
# season's Série A/B rosters; app.py SEASON_GROUPS carries the same lists).
# 2026/27 (seasonId 192831):
FIRST_STAGE_GROUPS = {
    'North': ['Fafe', 'Varzim', 'Paredes', 'Paços de Ferreira', 'São João Ver',
              'Leça', 'Vitória Guimarães II', 'Trofense', 'Vianense', 'AD Marco 09'],
    'South': ['Louletano', 'Caldas', 'Sporting Covilhã', 'Mafra', 'União Santarém',
              'UD Oliveirense', 'Vitória de Sernache', 'CF Os Belenenses',
              'Lusitano Évora 1911', 'Atlético CP'],
}

# Head-to-head tiebreaker overrides for first-stage positions.
# calculate_league_table uses GD, but FPF rules use head-to-head first.
# Format: {team: correct_position} — only needed for teams where h2h differs
# from GD order. SEASON-SPECIFIC: reset at turnover, add as ties emerge.
FIRST_STAGE_POSITION_OVERRIDES = {}

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


# ── Prior/current blending: empirical-Bayes shrinkage ────────────────────────
# Calibrated 2026-08 two ways (see decay_backtest): (a) backtest on three Liga 3
# season pairs — optimal current-weight for POINTS is only ~0.15 at 3 matches
# (last season's table out-predicts the current one until ~MW13), while xGD
# earns trust much faster (optimal k≈3); (b) literature — Dixon-Coles time
# decay implies ~1-year half-life, FiveThirtyEight carries 67% of prior-season
# rating across seasons, crossover studies put current-alone parity at 10-15
# matches, and ASA/11tegen11 show xG ratio is maximally predictive from ~game
# 4-5. The old exp(-0.30·m) decay (5% prior weight at 10 matches) was ~3x too
# fast for points-like rates.
#
# Form: prior_weight = k/(k+m)  — never reaches zero (prior info still helps
# late; Bundesliga study finds significant gains at matchday 17), with
# metric-specific k ("effective games of prior evidence"):
K_POINTS = 10.0   # ppg, win rate, clean sheets, venue rates
K_GOALS = 8.0     # goals for/against per game
K_XG = 4.0        # xG for/against per game — signal-rich early

# Prior-quality multiplier on k: a fuzzy prior earns less patience (literature:
# promoted/relegated sides get wider uncertainty and faster updating).
K_SCALE_BY_SOURCE = {'same_tier': 1.0, 'cross_tier': 0.7, 'from_above': 0.5}
K_SCALE_ANCHOR = 0.5  # no personal prior at all — generic league anchor


def get_blended_stat(current_value, current_matches, prior_per_game, k=K_POINTS,
                     k_scale=1.0, default_prior=None):
    if current_matches == 0:
        return prior_per_game if prior_per_game is not None else (default_prior if default_prior else 0.0)
    current_per_game = current_value / current_matches
    if prior_per_game is None:
        return current_per_game
    k_eff = max(k * k_scale, 1.0)
    prior_weight = k_eff / (k_eff + current_matches)
    return (1 - prior_weight) * current_per_game + prior_weight * prior_per_game


def calculate_prediction_features(team_stats, prior_stats, league_avg, is_home):
    curr = team_stats
    m = curr['matches']

    # Start-of-season anchor: slight-below-average baseline. When the team
    # has a USABLE prior (built tier-aware by build_prior_strengths — full
    # weight for returning clubs, discounted toward this anchor for
    # cross-tier movers), the prior overrides the anchor per rate. Either
    # way the exponential decay below hands over to current-season form
    # within ~10 matches.
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
    k_scale = K_SCALE_ANCHOR
    if prior_stats and prior_stats.get('per_game'):
        pg = prior_stats['per_game']
        prior_ppg = pg.get('ppg', prior_ppg)
        prior_gpg = pg.get('gpg', prior_gpg)
        prior_gapg = pg.get('gapg', prior_gapg)
        prior_xgpg = pg.get('xgpg', prior_xgpg)
        prior_xgapg = pg.get('xgapg', prior_xgapg)
        prior_winrate = pg.get('winrate', prior_winrate)
        prior_csrate = pg.get('csrate', prior_csrate)
        prior_venue_gpg = prior_gpg
        k_scale = K_SCALE_BY_SOURCE.get(prior_stats.get('source'), 1.0)

    ppg = get_blended_stat(curr['points'], m, prior_ppg, K_POINTS, k_scale)
    gpg = get_blended_stat(curr['goals_for'], m, prior_gpg, K_GOALS, k_scale)
    gapg = get_blended_stat(curr['goals_against'], m, prior_gapg, K_GOALS, k_scale)
    xgpg = get_blended_stat(curr['xG_for'], m, prior_xgpg, K_XG, k_scale)
    xgapg = get_blended_stat(curr['xG_against'], m, prior_xgapg, K_XG, k_scale)
    win_rate = get_blended_stat(curr['wins'], m, prior_winrate, K_POINTS, k_scale)
    cs_rate = get_blended_stat(curr['clean_sheets'], m, prior_csrate, K_POINTS, k_scale)

    curr_shot_conv = curr['goals_for'] / max(curr['shots_for'], 1) if m > 0 else 0
    curr_sot_rate = curr['sot_for'] / max(curr['shots_for'], 1) if m > 0 else 0
    shot_conv = curr_shot_conv if m > 3 else prior_shot_conv
    sot_rate = curr_sot_rate if m > 3 else prior_sot_rate

    venue_key = 'home' if is_home else 'away'
    venue_wr = get_blended_stat(curr[f'{venue_key}_wins'], curr[f'{venue_key}_matches'], prior_venue_wr, K_POINTS, k_scale)
    venue_gpg = get_blended_stat(curr[f'{venue_key}_goals'], curr[f'{venue_key}_matches'], prior_venue_gpg, K_POINTS, k_scale)

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


# Sides arriving from ABOVE the target tier (no data in our lake). They get a
# mildly-strong anchor instead of the pessimistic one. SEASON-SPECIFIC.
RELEGATED_INTO_LIGA3 = {'Paços de Ferreira', 'UD Oliveirense'}

# Prior-season ids per competition for prior building. SEASON-SPECIFIC.
PRIOR_SEASON = {43324: 191782, 702: 191779}
CROSS_TIER_SHRINK = 0.5  # weight on personal record for cross-tier movers


def build_prior_strengths(matches_df, league_avg, target_comp_id):
    """Tier-aware per-team priors from LAST season's full record.

    Returns {team: {'per_game': {...}}} for calculate_prediction_features:
      - played the SAME competition last season -> full personal rates
      - moved UP a tier (e.g. Campeonato -> Liga 3) -> personal rates shrunk
        50% toward the pessimistic league anchor
      - moved DOWN a tier (e.g. Liga 3 -> Campeonato) -> personal rates
        shrunk 50% toward an optimistic anchor
      - arrived from ABOVE our data (Liga 2 -> Liga 3): optimistic anchor
      - no record anywhere: no entry (caller's pessimistic anchor applies)

    xG comes from shot-level raw_events when available; otherwise goal rates
    stand in for xG rates (same scale at season aggregation).
    """
    other_comp = 702 if target_comp_id == 43324 else 43324
    frames = {}
    for cid in (target_comp_id, other_comp):
        sid = PRIOR_SEASON.get(cid)
        f = matches_df[(matches_df['seasonId'] == sid)
                       & (matches_df['status'] == 'Played')].copy() if sid else None
        if f is not None and not f.empty:
            score = f['score'].astype(str).str.extract(r'(\d+)\s*-\s*(\d+)')
            f['hg'] = pd.to_numeric(score[0], errors='coerce')
            f['ag'] = pd.to_numeric(score[1], errors='coerce')
            f = f.dropna(subset=['hg', 'ag'])
            frames[cid] = f

    # per-team season xG from shot events (best effort)
    team_match_xg = {}
    try:
        sids = [PRIOR_SEASON[c] for c in frames]
        ev = pd.read_parquet(
            'raw_events.parquet',
            columns=['matchId', 'seasonId', 'team.name', 'type.primary', 'shot.xg'],
            filters=[('seasonId', 'in', sids), ('type.primary', '==', 'shot')])
        xg = (ev.dropna(subset=['shot.xg', 'team.name'])
              .groupby(['matchId', 'team.name'])['shot.xg'].sum())
        team_match_xg = xg.to_dict()
    except Exception as e:
        print(f"  (prior xG unavailable — using goal rates: {e})")

    anchor_pess = {
        'ppg': league_avg.get('ppg', 1.0) * 0.85,
        'gpg': league_avg.get('gpg', 1.0) * 0.85,
        'gapg': league_avg.get('gapg', 1.0) * 1.15,
        'xgpg': league_avg.get('xgpg', 1.0) * 0.85,
        'xgapg': league_avg.get('xgapg', 1.0) * 1.15,
        'winrate': 0.28, 'csrate': league_avg.get('csrate', 0.25) * 0.85,
    }
    anchor_opt = {
        'ppg': league_avg.get('ppg', 1.0) * 1.15,
        'gpg': league_avg.get('gpg', 1.0) * 1.10,
        'gapg': league_avg.get('gapg', 1.0) * 0.90,
        'xgpg': league_avg.get('xgpg', 1.0) * 1.10,
        'xgapg': league_avg.get('xgapg', 1.0) * 0.90,
        'winrate': 0.40, 'csrate': league_avg.get('csrate', 0.25) * 1.10,
    }

    def team_rates(f, team):
        home = f[f['homeTeamName'] == team]
        away = f[f['awayTeamName'] == team]
        n = len(home) + len(away)
        if n < 10:  # need a meaningful sample
            return None
        gf = home['hg'].sum() + away['ag'].sum()
        ga = home['ag'].sum() + away['hg'].sum()
        wins = (home['hg'] > home['ag']).sum() + (away['ag'] > away['hg']).sum()
        draws = (home['hg'] == home['ag']).sum() + (away['ag'] == away['hg']).sum()
        cs = (home['ag'] == 0).sum() + (away['hg'] == 0).sum()
        xg_for = xg_against = 0.0
        have_xg = True
        for _, r in pd.concat([home, away]).iterrows():
            mid = r['matchId']
            us = team_match_xg.get((mid, team))
            opp_name = r['awayTeamName'] if r['homeTeamName'] == team else r['homeTeamName']
            them = team_match_xg.get((mid, opp_name))
            if us is None and them is None:
                have_xg = False
                break
            xg_for += us or 0.0
            xg_against += them or 0.0
        return {
            'ppg': (3 * wins + draws) / n, 'gpg': gf / n, 'gapg': ga / n,
            'xgpg': (xg_for / n) if have_xg else gf / n,
            'xgapg': (xg_against / n) if have_xg else ga / n,
            'winrate': wins / n, 'csrate': cs / n,
        }

    def shrink(rates, anchor, w):
        return {k: w * rates[k] + (1 - w) * anchor[k] for k in rates}

    priors = {}
    same = frames.get(target_comp_id)
    other = frames.get(other_comp)
    all_current_teams = set()  # caller filters; build for any team seen
    for f in frames.values():
        all_current_teams |= set(f['homeTeamName']) | set(f['awayTeamName'])
    for team in all_current_teams:
        rates = team_rates(same, team) if same is not None else None
        if rates is not None:
            priors[team] = {'per_game': rates, 'source': 'same_tier'}
            continue
        rates = team_rates(other, team) if other is not None else None
        if rates is not None:
            # moved between our two tiers: shrink toward the anchor matching
            # the direction (up a tier -> pessimistic, down -> optimistic)
            moved_up = (target_comp_id == 43324)
            anchor = anchor_pess if moved_up else anchor_opt
            priors[team] = {'per_game': shrink(rates, anchor, CROSS_TIER_SHRINK),
                            'source': 'cross_tier'}
    if target_comp_id == 43324:
        for team in RELEGATED_INTO_LIGA3:
            priors.setdefault(team, {'per_game': dict(anchor_opt),
                                     'source': 'from_above'})
    return priors


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


# Set from the loaded model artifact in main(); 'simple_strength_v1' switches
# the feature vector to the interpretable single-strength model (0.7 NPxGD +
# 0.3 GD, blended with priors upstream in calculate_prediction_features).
MODEL_FEATURE_MODE = None
MODEL_STRENGTH_MIX = 0.7


def team_strength(feats, mix=None):
    mix = MODEL_STRENGTH_MIX if mix is None else mix
    return (mix * (feats['xgpg'] - feats['xgapg'])
            + (1 - mix) * (feats['gpg'] - feats['gapg']))


def build_feature_vector(home_feats, away_feats):
    if MODEL_FEATURE_MODE == 'simple_strength_v1':
        return [team_strength(home_feats) - team_strength(away_feats)]
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

            home_prior = prior_season_stats.get(home) or home_cum.get('prior_stats')
            away_prior = prior_season_stats.get(away) or away_cum.get('prior_stats')

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
                    home_prior = prior_season_stats.get(home) or home_cum.get('prior_stats')
                    away_prior = prior_season_stats.get(away) or away_cum.get('prior_stats')
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
            home_prior = prior_season_stats.get(h) or home_cum.get('prior_stats')
            away_prior = prior_season_stats.get(a) or away_cum.get('prior_stats')
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

def _bonus_from_pos_pts(pos, pts):
    """FPF maintenance bonus for one team (see calculate_maintenance_bonus)."""
    classification_bonus = {5: 6, 6: 5, 7: 4, 8: 3, 9: 2, 10: 1}
    if pos < 5 or pts < 10:
        return 0
    if pts <= 14:
        return classification_bonus.get(pos, 0)
    pts_bonus = 4
    for threshold, bonus in [(15, 0), (20, 1), (25, 2), (30, 3)]:
        if pts < threshold:
            pts_bonus = bonus
            break
    return classification_bonus.get(pos, 0) + pts_bonus


def _pairwise_probs(teams, model, scaler, team_stats, prior_season_stats, league_avg_stats):
    """Model win/draw/loss probabilities for every ordered pairing."""
    probs = {}
    for home, away in product(teams, teams):
        if home == away:
            continue
        home_cum = team_stats.get(home)
        away_cum = team_stats.get(away)
        if not home_cum or not away_cum:
            probs[(home, away)] = np.array([0.25, 0.45, 0.30])
            continue
        home_prior = prior_season_stats.get(home) or home_cum.get('prior_stats')
        away_prior = prior_season_stats.get(away) or away_cum.get('prior_stats')
        home_feats = calculate_prediction_features(home_cum, home_prior, league_avg_stats, is_home=True)
        away_feats = calculate_prediction_features(away_cum, away_prior, league_avg_stats, is_home=False)
        fv = build_feature_vector(home_feats, away_feats)
        probs[(home, away)] = model.predict_proba(scaler.transform([fv]))[0]
    return probs


def _sim_fixtures(fixtures, cum_probs, pts, gd, gf, rng):
    """Sample outcomes for a fixture list, mutating pts/gd/gf dicts in place.
    cum_probs: precomputed np.cumsum rows aligned with fixtures (draw/home/away)."""
    if not fixtures:
        return
    rand_vals = rng.random(len(fixtures))
    for idx, (home, away) in enumerate(fixtures):
        r = rand_vals[idx]
        cp = cum_probs[idx]
        if r < cp[0]:
            pts[home] += 1; pts[away] += 1; gf[home] += 1; gf[away] += 1
        elif r < cp[1]:
            pts[home] += 3; gd[home] += 1; gd[away] -= 1; gf[home] += 2; gf[away] += 1
        else:
            pts[away] += 3; gd[away] += 1; gd[home] -= 1; gf[away] += 2; gf[home] += 1


def _simulate_liga3_first_phase(first_stage_matches, model, scaler, team_stats,
                                prior_season_stats, league_avg_stats):
    """Liga 3 during the FIRST PHASE (two séries of 10, double round robin,
    Aug–Feb): simulate the remaining série matches, then chain the whole
    second phase per simulation — top 4 of each série to the Promotion
    Series (fresh points), bottom 6 to the two Maintenance series with FPF
    bonus points from their simulated first-phase finish. Per team we report
    the série position probabilities plus P(reach promotion series),
    P(promotion slot: eligible top-2 of the promotion series), and
    P(relegation: bottom 2 of a maintenance series)."""
    series = {
        'Série A (North)': FIRST_STAGE_GROUPS['North'],
        'Série B (South)': FIRST_STAGE_GROUPS['South'],
    }
    all_teams = [t for teams in series.values() for t in teams]
    print("\n  First phase in progress — simulating séries + chained second phase")
    pair_probs = _pairwise_probs(all_teams, model, scaler, team_stats,
                                 prior_season_stats, league_avg_stats)

    prep = {}
    for name, teams in series.items():
        current_table = calculate_league_table(first_stage_matches, teams)
        full = [(h, a) for h, a in product(teams, teams) if h != a]
        played = set()
        for _, m in first_stage_matches.iterrows():
            h, a = m['homeTeamName'], m['awayTeamName']
            if h in teams and a in teams:
                played.add((h, a))
        remaining = [f for f in full if f not in played]
        start = {r['Team']: (r['Pts'], r['GD'], r['GF'])
                 for _, r in current_table.iterrows()}
        prep[name] = {
            'teams': teams,
            'current_table': current_table,
            'remaining': remaining,
            'cum': np.cumsum(np.array([pair_probs[f] for f in remaining]), axis=1)
                   if remaining else np.array([]),
            'start': start,
        }
        print(f"  {name}: {len(played)} played, {len(remaining)} remaining")

    pos_counts = {name: {t: np.zeros(len(p['teams']), dtype=int) for t in p['teams']}
                  for name, p in prep.items()}
    reach_promo = {t: 0 for t in all_teams}
    promo_slot = {t: 0 for t in all_teams}
    releg = {t: 0 for t in all_teams}

    rng = np.random.default_rng(seed=42)
    print(f"  Running {N_SIMULATIONS:,} chained simulations...")
    for _ in range(N_SIMULATIONS):
        serie_final = {}
        serie_pts = {}
        for name, p in prep.items():
            pts = {t: v[0] for t, v in p['start'].items()}
            gd = {t: v[1] for t, v in p['start'].items()}
            gf = {t: v[2] for t, v in p['start'].items()}
            _sim_fixtures(p['remaining'], p['cum'], pts, gd, gf, rng)
            order = sorted(p['teams'], key=lambda t: (pts[t], gd[t], gf[t]), reverse=True)
            for pos, team in enumerate(order):
                pos_counts[name][team][pos] += 1
            serie_final[name] = order
            serie_pts[name] = pts

        # ---- Promotion Series: top 4 of each série, fresh points ----
        promo_teams = serie_final['Série A (North)'][:4] + serie_final['Série B (South)'][:4]
        for t in promo_teams:
            reach_promo[t] += 1
        fixtures = [(h, a) for h, a in product(promo_teams, promo_teams) if h != a]
        cum = np.cumsum(np.array([pair_probs[f] for f in fixtures]), axis=1)
        pts = {t: 0 for t in promo_teams}
        gd = {t: 0 for t in promo_teams}
        gf = {t: 0 for t in promo_teams}
        _sim_fixtures(fixtures, cum, pts, gd, gf, rng)
        promo_order = sorted(promo_teams, key=lambda t: (pts[t], gd[t], gf[t]), reverse=True)
        for t in top_n_eligible(promo_order, 2):
            promo_slot[t] += 1

        # ---- Maintenance series: bottom 6 of each série, FPF bonuses ----
        for name in series:
            order = serie_final[name]
            maint = order[4:]
            pts = {t: _bonus_from_pos_pts(order.index(t) + 1, serie_pts[name][t])
                   for t in maint}
            gd = {t: 0 for t in maint}
            gf = {t: 0 for t in maint}
            fixtures = [(h, a) for h, a in product(maint, maint) if h != a]
            cum = np.cumsum(np.array([pair_probs[f] for f in fixtures]), axis=1)
            _sim_fixtures(fixtures, cum, pts, gd, gf, rng)
            maint_order = sorted(maint, key=lambda t: (pts[t], gd[t], gf[t]), reverse=True)
            for t in maint_order[-2:]:
                releg[t] += 1

    results = {}
    for name, p in prep.items():
        teams = p['teams']
        n_teams = len(teams)
        pos_prob_df = pd.DataFrame(
            {t: pos_counts[name][t] / N_SIMULATIONS for t in teams}).T
        pos_prob_df.columns = [f'{i+1}' for i in range(n_teams)]
        pos_prob_df.index.name = 'Team'
        pos_prob_df['expected_pos'] = sum((i + 1) * pos_prob_df[f'{i+1}'] for i in range(n_teams))
        pos_prob_df = pos_prob_df.sort_values('expected_pos').drop(columns='expected_pos')

        print(f"\n  {name} position probabilities:")
        print(pos_prob_df.to_string(float_format=lambda x: f'{x:.1%}'))

        results[name] = {
            'teams': teams,
            'position_probabilities': pos_prob_df,
            'current_standings': p['current_table'],
            'matches_remaining': len(p['remaining']),
            'bonus_points': {},
            'playoff_pct': {t: reach_promo[t] / N_SIMULATIONS for t in teams},
            'promotion_pct': {t: promo_slot[t] / N_SIMULATIONS for t in teams},
            'releg_pct': {t: releg[t] / N_SIMULATIONS for t in teams},
            'serie_col_labels': ('Promo Series %', 'Promo %', 'Releg %'),
        }
    return results


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

    # While the FIRST PHASE is running (Aug–Feb) the season is just the two
    # séries — simulate those (with the second phase chained inside each sim)
    # instead of pretending today's table already decided the qualifiers.
    if second_stage_matches.empty:
        return _simulate_liga3_first_phase(first_stage_matches, model, scaler,
                                           team_stats, prior_season_stats,
                                           league_avg_stats)

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

            home_prior = prior_season_stats.get(home) or home_cum.get('prior_stats')
            away_prior = prior_season_stats.get(away) or away_cum.get('prior_stats')

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
    global MODEL_FEATURE_MODE, MODEL_STRENGTH_MIX
    MODEL_FEATURE_MODE = model_data.get('feature_mode')
    MODEL_STRENGTH_MIX = model_data.get('strength_mix', 0.7)
    if MODEL_FEATURE_MODE:
        print(f"Predictor feature mode: {MODEL_FEATURE_MODE} "
              f"(mix={MODEL_STRENGTH_MIX})")
    team_stats = model_data['team_stats']
    # Tier-aware priors built fresh from last season's records (the pkl's
    # prior_season_stats is a stale training-time artifact — see
    # build_prior_strengths). Falls back to {} on any failure.
    prior_season_stats = model_data.get('prior_season_stats', {})
    league_avg_stats = model_data.get('league_avg_stats', {
        'ppg': 1.0, 'gpg': 1.19, 'gapg': 1.19, 'xgpg': 1.0,
        'xgapg': 1.0, 'csrate': 0.25, 'shot_conv': 0.1, 'sot_rate': 0.35,
    })

    matches_df = pd.read_parquet('matches_summary.parquet')

    liga3_priors, camp_priors = {}, {}
    try:
        liga3_priors = build_prior_strengths(matches_df, league_avg_stats, 43324)
        camp_priors = build_prior_strengths(matches_df, league_avg_stats, 702)
        print(f"Tier-aware priors built: Liga 3 {len(liga3_priors)} teams, "
              f"Campeonato {len(camp_priors)} teams")
    except Exception as e:
        print(f"⚠️ Prior build failed ({e}) — falling back to league anchors")

    all_results = {}

    # Liga 3 simulation
    try:
        liga3_results = simulate_liga3(matches_df, model, scaler, team_stats, liga3_priors, league_avg_stats)
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
            camp_results = simulate_campeonato(camp_matches, model, scaler, team_stats, camp_priors, league_avg_stats)
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
