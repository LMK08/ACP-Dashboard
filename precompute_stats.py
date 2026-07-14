"""
!!! DEPRECATED — DO NOT USE !!!

This script is a stale replica of the stats pipeline in app.py and its
outputs are NOT read by anything:

  * It writes legacy un-versioned cache files
    (stats_cache/player_stats_{season_id}.parquet). The app only loads the
    versioned player_stats_v14_* files written by
    calculate_all_player_stats() in app.py.
  * Its GK aggregation predates the numeric-coercion fix for
    shot.isGoal (app.py commit 424852e), so the goalsConceded /
    savePercentage numbers it computes are wrong.
  * Its hardcoded ALL_SEASON_IDS list is missing seasons (e.g. 190230,
    191779) that the canonical cache covers.

To regenerate the stats caches, use the canonical pipeline instead:

    python app.py --precompute

(which is also what .github/workflows/precompute_caches.yml runs).

The runtime guard below aborts execution; pass --force-deprecated to run
anyway at your own risk.
"""
import os
import sys
import pandas as pd
import pickle
import numpy as np

if '--force-deprecated' not in sys.argv:
    sys.exit(
        "precompute_stats.py is DEPRECATED: its outputs are unused and its GK "
        "stats are wrong (see module docstring). Use `python app.py --precompute` "
        "instead, or pass --force-deprecated to run anyway."
    )

# We need the computation logic from app.py but can't import it directly
# due to Streamlit decorators. So we replicate the essential functions here.

STATS_CACHE_DIR = 'stats_cache'
ALL_SEASON_IDS = [191782, 190090, 189147, 188222, 188221]

# --- Import position/weight config from app.py by reading it ---
# These are hardcoded in app.py, so we define them here too
POSITION_GROUPS = {
    'Forward': ['CF', 'SS', 'LW', 'RW', 'LWF', 'RWF'],
    'Midfielder': ['AMF', 'CMF', 'DMF', 'LCMF', 'RCMF', 'LCMF3', 'RCMF3', 'LDMF', 'RDMF', 'LAMF', 'RAMF'],
    'Defender': ['CB', 'LCB', 'RCB', 'LCB3', 'RCB3', 'LB', 'RB', 'LWB', 'RWB', 'LB5', 'RB5'],
    'Goalkeeper': ['GK'],
}

# We need to get WEIGHTS and INVERT_METRICS from app.py
# Let's extract them programmatically
def extract_config_from_app():
    """Extract WEIGHTS, INVERT_METRICS, and POSITION_GROUPS from app.py"""
    import ast

    with open('app.py', 'r') as f:
        content = f.read()

    # Find WEIGHTS
    weights_start = content.find('WEIGHTS = {')
    if weights_start == -1:
        weights_start = content.find('WEIGHTS= {')
    if weights_start == -1:
        print("Could not find WEIGHTS in app.py, using defaults")
        return None, None, None

    # Find INVERT_METRICS
    invert_start = content.find('INVERT_METRICS = ')
    if invert_start == -1:
        invert_start = content.find('INVERT_METRICS= ')

    # Find POSITION_GROUPS
    pg_start = content.find('POSITION_GROUPS = {')
    if pg_start == -1:
        pg_start = content.find('POSITION_GROUPS= {')

    return weights_start, invert_start, pg_start


def add_custom_dribble_success(events_df):
    """Applies custom logic to determine if a dribble was successful."""
    df = events_df.sort_values(['matchId', 'minute', 'second']).copy()

    dribble_mask = (df['type.primary'] == 'duel') & (df.get('groundDuel.takeOn') == True)
    df['is_dribble_attempt'] = dribble_mask
    df['is_custom_dribble_success'] = False

    if not dribble_mask.any():
        return df

    dribble_indices = df[dribble_mask].index.tolist()

    for idx in dribble_indices:
        try:
            pos = df.index.get_loc(idx)
            dribbler_team = df.at[idx, 'team.name']

            # Look at next few events for same team
            for next_pos in range(pos + 1, min(pos + 4, len(df))):
                next_idx = df.index[next_pos]
                if df.at[next_idx, 'matchId'] != df.at[idx, 'matchId']:
                    break
                if df.at[next_idx, 'team.name'] == dribbler_team:
                    next_type = df.at[next_idx, 'type.primary']
                    if next_type in ['pass', 'shot', 'carry', 'touch', 'cross']:
                        df.at[idx, 'is_custom_dribble_success'] = True
                    break
        except Exception:
            continue

    cols_to_drop = [c for c in df.columns if 'next_team_' in c] + ['dist_to_goal', 'time_diff']
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    return df


def main():
    print("Loading data...")

    # Load events
    events_columns = [
        'id', 'matchId', 'seasonId', 'minute', 'second', 'matchTimestamp',
        'type.primary', 'type.secondary', 'player.id', 'player.name', 'player.position',
        'team.name', 'opponentTeam.name', 'location.x', 'location.y',
        'pass', 'pass.accurate', 'pass.endLocation.x', 'pass.endLocation.y', 'pass.length',
        'shot', 'shot.xg', 'shot.isGoal', 'shot.onTarget', 'shot.bodyPart', 'shot.postShotXg', 'shot.goalkeeper.id',
        'groundDuel.duelType', 'groundDuel.keptPossession', 'groundDuel.progressedWithBall',
        'groundDuel.recoveredPossession', 'groundDuel.stoppedProgress', 'groundDuel.takeOn',
        'aerialDuel.firstTouch', 'carry.endLocation.x', 'carry.endLocation.y',
        'possession.id', 'possession.eventIndex', 'possession.duration', 'possession.team.name', 'possession.types',
        'infraction', 'infraction.type', 'infraction.yellowCard', 'infraction.redCard',
        'is_dribble_attempt', 'is_custom_dribble_success', 'relatedEventId',
        'team.formation'
    ]
    raw_events_df = pd.read_parquet('raw_events.parquet', columns=events_columns)

    # Load player minutes
    if os.path.exists('complete_player_minutes.pkl'):
        with open('complete_player_minutes.pkl', 'rb') as f:
            player_minutes_data = pickle.load(f)
    else:
        with open('player_minutes_and_positions.pkl', 'rb') as f:
            player_minutes_data = pickle.load(f)

    # We need to exec the WEIGHTS/INVERT_METRICS/POSITION_GROUPS from app.py
    # Read them from the file
    print("Extracting config from app.py...")
    with open('app.py', 'r') as f:
        app_content = f.read()

    # Extract globals we need
    config_globals = {}

    # Find and exec POSITION_GROUPS
    for var_name in ['POSITION_GROUPS', 'WEIGHTS', 'INVERT_METRICS']:
        start = app_content.find(f'{var_name} = ')
        if start == -1:
            start = app_content.find(f'{var_name}= ')
        if start == -1:
            print(f"Could not find {var_name} in app.py!")
            return

        # Find the end of the assignment (look for next unindented line)
        lines = app_content[start:].split('\n')
        block_lines = [lines[0]]
        for line in lines[1:]:
            if line and not line[0].isspace() and not line.startswith('#') and '=' in line and not line.strip().startswith("'") and not line.strip().startswith('"'):
                break
            block_lines.append(line)

        block = '\n'.join(block_lines)
        try:
            exec(block, config_globals)
        except Exception as e:
            print(f"Error parsing {var_name}: {e}")
            print(f"Block was: {block[:200]}")
            return

    POSITION_GROUPS = config_globals['POSITION_GROUPS']
    WEIGHTS = config_globals['WEIGHTS']
    INVERT_METRICS = config_globals['INVERT_METRICS']

    print(f"Config loaded: {len(POSITION_GROUPS)} position groups, {len(WEIGHTS)} weight sets")

    os.makedirs(STATS_CACHE_DIR, exist_ok=True)

    for season_id in ALL_SEASON_IDS:
        stats_cache = os.path.join(STATS_CACHE_DIR, f'player_stats_{season_id}.parquet')
        pct_cache = os.path.join(STATS_CACHE_DIR, f'player_percentiles_{season_id}.parquet')

        if os.path.exists(stats_cache) and os.path.exists(pct_cache):
            print(f"\nSeason {season_id}: Already cached, skipping")
            continue

        print(f"\n--- Computing stats for season {season_id} ---")

        season_events = raw_events_df[raw_events_df['seasonId'] == season_id].copy()
        season_minutes = player_minutes_data.get(season_id, pd.DataFrame())

        if season_events.empty or season_minutes.empty:
            print(f"  Skipping: no data")
            continue

        # Apply dribble logic
        print(f"  Applying dribble logic to {len(season_events)} events...")
        season_events = add_custom_dribble_success(season_events)

        # Build base_df from minutes
        season_minutes['totalMinutes'] = pd.to_numeric(season_minutes['totalMinutes'], errors='coerce').fillna(0)
        base_df = season_minutes.copy().set_index('playerId')

        # Ensure player.id is int
        season_events['player.id'] = pd.to_numeric(season_events['player.id'], errors='coerce')
        season_events = season_events.dropna(subset=['player.id'])
        season_events['player.id'] = season_events['player.id'].astype(int)

        # --- Counting stats (same logic as app.py) ---
        print(f"  Computing counting stats...")

        def count_and_merge(base_df, events_df, stat_name, filter_condition):
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

        def sum_and_merge(base_df, events_df, stat_name, filter_condition, value_col):
            if isinstance(filter_condition, pd.Series):
                filter_condition = filter_condition.reindex(events_df.index, fill_value=False)
            safe_condition = filter_condition & events_df['player.id'].notna()
            if safe_condition.empty or not safe_condition.any():
                stat_series = pd.Series(dtype='float', name=stat_name)
            else:
                stat_series = events_df.loc[safe_condition].groupby(events_df['player.id'])[value_col].sum()
                stat_series.name = stat_name
            base_df = base_df.merge(stat_series, left_index=True, right_index=True, how='left')
            return base_df

        events_df = season_events

        _secondary_col = events_df.get('type.secondary', pd.Series(dtype='object'))
        _secondary_sets = _secondary_col.apply(
            lambda x: set(x) if isinstance(x, (list, np.ndarray)) else set()
        )
        def check_secondary_list(tag):
            return _secondary_sets.apply(lambda s: tag in s)

        # Basic counts
        pass_events = events_df[events_df['type.primary'] == 'pass']
        base_df = count_and_merge(base_df, pass_events, 'Passes', pd.Series(True, index=pass_events.index))
        base_df = count_and_merge(base_df, pass_events, 'Passes accurate', pass_events.get('pass.accurate') == True)

        shot_events = events_df[events_df['type.primary'] == 'shot']
        base_df = count_and_merge(base_df, shot_events, 'Shots', pd.Series(True, index=shot_events.index))
        base_df = count_and_merge(base_df, shot_events, 'Shots on target', shot_events.get('shot.onTarget') == True)
        base_df = count_and_merge(base_df, shot_events, 'Goals', shot_events.get('shot.isGoal') == True)

        shot_events['shot.xg'] = pd.to_numeric(shot_events.get('shot.xg'), errors='coerce').fillna(0)
        base_df = sum_and_merge(base_df, shot_events, 'xG', pd.Series(True, index=shot_events.index), 'shot.xg')

        non_penalty = ~check_secondary_list('penalty').reindex(shot_events.index, fill_value=False)
        base_df = sum_and_merge(base_df, shot_events, 'npxG', non_penalty, 'shot.xg')
        base_df = count_and_merge(base_df, shot_events, 'npGoals',
                                   (shot_events.get('shot.isGoal') == True) & non_penalty)

        # Assists / Key passes / xA
        is_assist = check_secondary_list('assist')
        base_df = count_and_merge(base_df, events_df, 'Assists', is_assist)

        is_key_pass = check_secondary_list('key_pass')
        base_df = count_and_merge(base_df, events_df, 'Key passes', is_key_pass)

        # xA
        if 'relatedEventId' in events_df.columns and not shot_events.empty:
            shot_xg_map = shot_events.set_index('id')['shot.xg'].to_dict()
            events_df['_xa_value'] = events_df['relatedEventId'].map(shot_xg_map)
            xa_mask = events_df['_xa_value'].notna() & (events_df['type.primary'] == 'pass')
            base_df = sum_and_merge(base_df, events_df, 'xA', xa_mask, '_xa_value')
            events_df.drop(columns=['_xa_value'], inplace=True, errors='ignore')

        # Crosses
        is_cross = check_secondary_list('cross')
        cross_events = events_df[is_cross]
        base_df = count_and_merge(base_df, cross_events, 'Crosses', pd.Series(True, index=cross_events.index))
        base_df = count_and_merge(base_df, cross_events, 'Crosses accurate', cross_events.get('pass.accurate') == True)

        # Progressive passes
        pass_events_prog = pass_events.copy()
        pass_events_prog['pass.endLocation.x'] = pd.to_numeric(pass_events_prog.get('pass.endLocation.x'), errors='coerce').fillna(0)
        pass_events_prog['location.x'] = pd.to_numeric(pass_events_prog.get('location.x'), errors='coerce').fillna(0)
        pass_events_prog['_x_progress'] = pass_events_prog['pass.endLocation.x'] - pass_events_prog['location.x']
        base_df = count_and_merge(base_df, pass_events_prog, 'Progressive passes',
                                   (pass_events_prog.get('pass.accurate') == True) & (pass_events_prog['_x_progress'] >= 10))

        # Passes into final third / penalty area
        base_df = count_and_merge(base_df, pass_events, 'Passes to final third',
                                   (pass_events.get('pass.accurate') == True) &
                                   (pd.to_numeric(pass_events.get('pass.endLocation.x'), errors='coerce').fillna(0) >= 66))
        base_df = count_and_merge(base_df, pass_events, 'Passes to penalty area',
                                   (pass_events.get('pass.accurate') == True) &
                                   (pd.to_numeric(pass_events.get('pass.endLocation.x'), errors='coerce').fillna(0) >= 83) &
                                   (pd.to_numeric(pass_events.get('pass.endLocation.y'), errors='coerce').fillna(0).between(21, 79)))

        # Deep completions
        base_df = count_and_merge(base_df, pass_events, 'Deep completions',
                                   (pass_events.get('pass.accurate') == True) &
                                   (pd.to_numeric(pass_events.get('pass.endLocation.x'), errors='coerce').fillna(0) >= 80))

        # Dribbles
        dribble_events = events_df[events_df.get('is_dribble_attempt') == True]
        base_df = count_and_merge(base_df, dribble_events, 'Dribbles', pd.Series(True, index=dribble_events.index))
        base_df = count_and_merge(base_df, dribble_events, 'Dribbles successful',
                                   dribble_events.get('is_custom_dribble_success') == True)

        # Touches in box
        base_df = count_and_merge(base_df, events_df, 'Touches in penalty area',
                                   (events_df['type.primary'] == 'touch') &
                                   (pd.to_numeric(events_df.get('location.x'), errors='coerce').fillna(0) >= 83) &
                                   (pd.to_numeric(events_df.get('location.y'), errors='coerce').fillna(0).between(21, 79)))

        # Duels
        duel_events = events_df[events_df['type.primary'] == 'duel']
        base_df = count_and_merge(base_df, duel_events, 'Duels', pd.Series(True, index=duel_events.index))
        base_df = count_and_merge(base_df, duel_events, 'Duels successful',
                                   (duel_events.get('groundDuel.keptPossession') == True) |
                                   (duel_events.get('groundDuel.recoveredPossession') == True) |
                                   (duel_events.get('groundDuel.stoppedProgress') == True) |
                                   (duel_events.get('aerialDuel.firstTouch') == True))

        # Aerial duels
        aerial_mask = check_secondary_list('aerial_duel').reindex(duel_events.index, fill_value=False)
        aerial_events = duel_events[aerial_mask]
        base_df = count_and_merge(base_df, aerial_events, 'Aerial duels', pd.Series(True, index=aerial_events.index))
        base_df = count_and_merge(base_df, aerial_events, 'Aerial duels won',
                                   aerial_events.get('aerialDuel.firstTouch') == True)

        # Defensive
        base_df = count_and_merge(base_df, events_df, 'Interceptions', events_df['type.primary'] == 'interception')
        base_df = count_and_merge(base_df, events_df, 'Clearances', events_df['type.primary'] == 'clearance')

        # Fouls
        infraction_events = events_df[events_df['type.primary'] == 'infraction']
        is_foul = check_secondary_list('foul').reindex(infraction_events.index, fill_value=False)
        base_df = count_and_merge(base_df, infraction_events, 'Fouls committed', is_foul)
        base_df = count_and_merge(base_df, infraction_events, 'Yellow cards', infraction_events.get('infraction.yellowCard') == True)
        base_df = count_and_merge(base_df, infraction_events, 'Red cards', infraction_events.get('infraction.redCard') == True)

        # GK stats
        gk_events = events_df[events_df['type.primary'] == 'shot_against']
        base_df = count_and_merge(base_df, gk_events, 'Shots faced', pd.Series(True, index=gk_events.index))
        base_df = count_and_merge(base_df, gk_events, 'Saves', gk_events.get('shot.onTarget') == True)
        base_df = count_and_merge(base_df, gk_events, 'Goals conceded', gk_events.get('shot.isGoal') == True)

        if not gk_events.empty:
            gk_events_copy = gk_events.copy()
            gk_events_copy['shot.xg'] = pd.to_numeric(gk_events_copy.get('shot.xg'), errors='coerce').fillna(0)
            gk_events_copy['shot.postShotXg'] = pd.to_numeric(gk_events_copy.get('shot.postShotXg'), errors='coerce').fillna(0)
            base_df = sum_and_merge(base_df, gk_events_copy, 'xG faced', pd.Series(True, index=gk_events_copy.index), 'shot.xg')
            base_df = sum_and_merge(base_df, gk_events_copy, 'PSxG faced', pd.Series(True, index=gk_events_copy.index), 'shot.postShotXg')

        # Progressive carries
        carry_events = events_df[events_df['type.primary'] == 'carry'].copy() if 'carry' in events_df.get('type.primary', pd.Series()).values else pd.DataFrame()
        if not carry_events.empty:
            carry_events['carry.endLocation.x'] = pd.to_numeric(carry_events.get('carry.endLocation.x'), errors='coerce').fillna(0)
            carry_events['location.x'] = pd.to_numeric(carry_events.get('location.x'), errors='coerce').fillna(0)
            carry_events['_carry_progress'] = carry_events['carry.endLocation.x'] - carry_events['location.x']
            base_df = count_and_merge(base_df, carry_events, 'Progressive carries', carry_events['_carry_progress'] >= 10)

        # --- GK Advanced Stats ---
        print(f"  Computing GK advanced stats...")
        gk_ids = events_df[events_df.get('player.position') == 'GK']['player.id'].dropna().unique().astype(int)
        gk_events_all = events_df[events_df['player.id'].isin(gk_ids)].copy()
        shots_faced = events_df[
            (events_df.get('type.primary') == 'shot') &
            (events_df.get('shot.onTarget') == True) &
            (events_df.get('shot.goalkeeper.id').notna())
        ].copy()
        if not shots_faced.empty:
            shots_faced['shot.goalkeeper.id'] = shots_faced['shot.goalkeeper.id'].astype(int)
            shots_faced['shot.postShotXg'] = pd.to_numeric(shots_faced.get('shot.postShotXg'), errors='coerce').fillna(0)
            gk_shot_stats = shots_faced.groupby('shot.goalkeeper.id').agg(
                shotsOnTargetAgainst=('shot.isGoal', 'count'),
                goalsConceded=('shot.isGoal', 'sum'),
                psxG_faced=('shot.postShotXg', 'sum'),
            ).reset_index().rename(columns={'shot.goalkeeper.id': 'player.id'})
            gk_shot_stats['goalsPrevented'] = gk_shot_stats['psxG_faced'] - gk_shot_stats['goalsConceded']
            gk_shot_stats['goalsPreventedPerSOT'] = (gk_shot_stats['goalsPrevented'] / gk_shot_stats['shotsOnTargetAgainst']).fillna(0)
            gk_shot_stats['savePercentage'] = ((gk_shot_stats['shotsOnTargetAgainst'] - gk_shot_stats['goalsConceded']) / gk_shot_stats['shotsOnTargetAgainst'] * 100).fillna(0)
        else:
            gk_shot_stats = pd.DataFrame(columns=['player.id', 'shotsOnTargetAgainst', 'goalsConceded', 'psxG_faced', 'goalsPrevented', 'goalsPreventedPerSOT', 'savePercentage'])

        exits = gk_events_all[gk_events_all['type.primary'] == 'goalkeeper_exit'].groupby('player.id').size().reset_index(name='exits')
        _gk_secondary = gk_events_all.get('type.secondary', pd.Series(dtype='object')).apply(
            lambda x: set(x) if isinstance(x, (list, np.ndarray)) else set()
        )
        recovery_mask = _gk_secondary.apply(lambda s: 'recovery' in s)
        recoveries_gk = gk_events_all[recovery_mask].groupby('player.id').size().reset_index(name='recoveries_gk')
        gk_passes = gk_events_all[gk_events_all['type.primary'] == 'pass']
        passes_total_gk = gk_passes.groupby('player.id').size().reset_index(name='passes_gk')
        passes_succ_gk = gk_passes[gk_passes['pass.accurate'] == True].groupby('player.id').size().reset_index(name='passesSuccessful_gk')
        _gkp_secondary = gk_passes.get('type.secondary', pd.Series(dtype='object')).apply(
            lambda x: set(x) if isinstance(x, (list, np.ndarray)) else set()
        )
        long_pass_mask = _gkp_secondary.apply(lambda s: 'long_pass' in s)
        long_passes_total_gk = gk_passes[long_pass_mask].groupby('player.id').size().reset_index(name='longPasses_gk')
        long_passes_succ_gk = gk_passes[long_pass_mask & (gk_passes['pass.accurate'] == True)].groupby('player.id').size().reset_index(name='longPassesSuccessful_gk')

        gk_report_df = pd.DataFrame({'player.id': gk_ids})
        for gk_merge_df in [gk_shot_stats, exits, recoveries_gk, passes_total_gk, passes_succ_gk, long_passes_total_gk, long_passes_succ_gk]:
            gk_report_df = pd.merge(gk_report_df, gk_merge_df, on='player.id', how='left')
        base_df = base_df.merge(gk_report_df.set_index('player.id'), left_index=True, right_index=True, how='left')

        # Per 90 normalization
        print(f"  Normalizing to per-90...")
        base_df['totalMinutes'] = pd.to_numeric(base_df['totalMinutes'], errors='coerce').fillna(0)
        stat_cols = [c for c in base_df.columns if c not in ['playerName', 'teamName', 'primaryPosition', 'totalMinutes', 'playerId']]

        for col in stat_cols:
            base_df[col] = pd.to_numeric(base_df[col], errors='coerce').fillna(0)
            per90_col = col + '_per90'
            base_df[per90_col] = base_df.apply(
                lambda row: (row[col] / row['totalMinutes']) * 90 if row['totalMinutes'] > 0 else 0, axis=1
            )

        # Derived metrics
        base_df['Pass accuracy'] = base_df.apply(
            lambda r: (r['Passes accurate'] / r['Passes'] * 100) if r.get('Passes', 0) > 0 else 0, axis=1)
        base_df['Shot accuracy'] = base_df.apply(
            lambda r: (r['Shots on target'] / r['Shots'] * 100) if r.get('Shots', 0) > 0 else 0, axis=1)
        base_df['Cross accuracy'] = base_df.apply(
            lambda r: (r['Crosses accurate'] / r['Crosses'] * 100) if r.get('Crosses', 0) > 0 else 0, axis=1)
        base_df['Dribble success rate'] = base_df.apply(
            lambda r: (r['Dribbles successful'] / r['Dribbles'] * 100) if r.get('Dribbles', 0) > 0 else 0, axis=1)
        base_df['Aerial duel win rate'] = base_df.apply(
            lambda r: (r['Aerial duels won'] / r['Aerial duels'] * 100) if r.get('Aerial duels', 0) > 0 else 0, axis=1)

        # Cleanup
        cols_to_drop = [col for col in base_df.columns if 'player.id' in str(col)]
        base_df = base_df.drop(columns=cols_to_drop, errors='ignore')

        result = base_df.fillna(0)
        # Ensure string columns are actually strings before saving to parquet
        for col in ['playerName', 'teamName', 'primaryPosition']:
            if col in result.columns:
                result[col] = result[col].astype(str)
        result.to_parquet(os.path.join(STATS_CACHE_DIR, f'player_stats_{season_id}.parquet'))
        print(f"  Saved player_stats_{season_id}.parquet ({len(result)} players)")

        # --- Percentiles ---
        print(f"  Computing percentiles...")
        data = result.copy()
        data['totalMinutes'] = pd.to_numeric(data['totalMinutes'], errors='coerce')
        data = data[data['totalMinutes'] >= 90]

        if not data.empty:
            for position, group in POSITION_GROUPS.items():
                metrics = list(WEIGHTS[position].keys())
                position_data_mask = data['primaryPosition'].isin(group)
                position_data_indices = data[position_data_mask].index
                if position_data_indices.empty:
                    continue
                for metric in metrics:
                    if metric in data.columns:
                        data[metric] = pd.to_numeric(data[metric], errors='coerce').fillna(0)
                        percentiles = data.loc[position_data_indices, metric].rank(pct=True)
                        if metric in INVERT_METRICS:
                            percentiles = 1 - (percentiles.fillna(0.5))
                        data.loc[position_data_indices, metric + '_percentile'] = percentiles

            for position, group in POSITION_GROUPS.items():
                metrics = list(WEIGHTS[position].keys())
                position_data_mask = data['primaryPosition'].isin(group)
                position_data_indices = data[position_data_mask].index
                if position_data_indices.empty:
                    continue
                total_score = pd.Series(0.0, index=position_data_indices, dtype='float64')
                for metric in metrics:
                    percentile_col = metric + '_percentile'
                    if percentile_col in data.columns:
                        weight = WEIGHTS[position].get(metric, 0)
                        total_score = total_score.add(data.loc[position_data_indices, percentile_col].fillna(0) * weight, fill_value=0)
                data.loc[position_data_indices, position + '_TotalScore'] = total_score
                min_score = total_score.min()
                max_score = total_score.max()
                if (max_score - min_score) != 0:
                    data.loc[position_data_indices, position + '_Score'] = (total_score - min_score) / (max_score - min_score) * 100
                else:
                    data.loc[position_data_indices, position + '_Score'] = 0.0

            pct_result = data.fillna(0)
            for col in ['playerName', 'teamName', 'primaryPosition']:
                if col in pct_result.columns:
                    pct_result[col] = pct_result[col].astype(str)
            pct_result.to_parquet(os.path.join(STATS_CACHE_DIR, f'player_percentiles_{season_id}.parquet'))
            print(f"  Saved player_percentiles_{season_id}.parquet ({len(pct_result)} players)")
        else:
            print(f"  No players with >= 90 minutes, skipping percentiles")

    # =====================================================================
    # PHASE 2: Pre-compute TEAM stats for all seasons
    # =====================================================================
    print("\n=== Pre-computing team stats ===")

    # Load additional data needed for team stats
    matches_summary_df = pd.read_parquet('matches_summary.parquet')
    with open('all_match_data.pkl', 'rb') as f:
        all_match_data = pickle.load(f)

    for season_id in ALL_SEASON_IDS:
        print(f"\n--- Team stats for season {season_id} ---")
        season_events = raw_events_df[raw_events_df['seasonId'] == season_id].copy()
        season_matches = matches_summary_df[matches_summary_df['seasonId'] == season_id].copy()

        if season_events.empty or season_matches.empty:
            print(f"  Skipping: no data")
            continue

        # --- Team Strength ---
        ts_cache = os.path.join(STATS_CACHE_DIR, f'team_strength_{season_id}.parquet')
        if os.path.exists(ts_cache):
            print(f"  team_strength: already cached")
        else:
            print(f"  Computing team strength...")
            all_shots = season_events[season_events['type.primary'] == 'shot'].copy()
            all_shots['shot.xg'] = pd.to_numeric(all_shots.get('shot.xg'), errors='coerce').fillna(0)
            all_shots['shot.isGoal'] = all_shots.get('shot.isGoal') == True

            matches_played = season_events.groupby('team.name')['matchId'].nunique()
            all_teams_in_data = season_events['team.name'].unique()

            if 'opponentTeam.name' not in all_shots.columns and 'matchId' in all_shots.columns:
                temp_summary = season_matches[['matchId', 'homeTeamName', 'awayTeamName']].copy()
                temp_summary.rename(columns={'homeTeamName': 'ht', 'awayTeamName': 'at'}, inplace=True)
                all_shots = all_shots.merge(temp_summary, on='matchId', how='left')
                all_shots['opponentTeam.name'] = np.where(
                    all_shots['team.name'] == all_shots['ht'], all_shots['at'], all_shots['ht']
                )
                all_shots.drop(columns=['ht', 'at'], inplace=True, errors='ignore')

            team_stats = {}
            for team in all_teams_in_data:
                team_shots = all_shots[all_shots['team.name'] == team]
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
                        'xGA_per_match': xg_against / games,
                    }

            ts_df = pd.DataFrame.from_dict(team_stats, orient='index').fillna(0)
            if not ts_df.empty:
                ts_df['Attacking Strength'] = (ts_df['GF_per_match'] * 0.3) + (ts_df['xGF_per_match'] * 0.7)
                ts_df['Defending Strength'] = (ts_df['GA_per_match'] * 0.3) + (ts_df['xGA_per_match'] * 0.7)
                ts_df.to_parquet(ts_cache)
                print(f"  Saved team_strength ({len(ts_df)} teams)")

        # --- Expanded Team Stats ---
        ets_cache = os.path.join(STATS_CACHE_DIR, f'expanded_team_stats_{season_id}.parquet')
        if os.path.exists(ets_cache):
            print(f"  expanded_team_stats: already cached")
        else:
            print(f"  Computing expanded team stats...")
            from collections import defaultdict
            season_match_ids = set(season_matches['matchId'].dropna().unique())
            season_match_data = {mid: data for mid, data in all_match_data.items() if mid in season_match_ids}

            all_stats_agg = defaultdict(lambda: defaultdict(float))
            games_played = defaultdict(int)
            team_name_map = {}
            for _, row in season_matches.iterrows():
                team_name_map[row['matchId']] = (row['homeTeamName'], row['awayTeamName'])

            for match_id, match_data in season_match_data.items():
                if match_id not in team_name_map:
                    continue
                home_team, away_team = team_name_map[match_id]
                games_played[home_team] += 1
                games_played[away_team] += 1

                if 'team_stats' in match_data and isinstance(match_data['team_stats'], dict):
                    for category, df in match_data['team_stats'].items():
                        if isinstance(df, pd.DataFrame) and not df.empty:
                            if home_team in df.columns and away_team in df.columns:
                                try:
                                    df[home_team] = pd.to_numeric(df[home_team], errors='coerce').fillna(0)
                                    df[away_team] = pd.to_numeric(df[away_team], errors='coerce').fillna(0)
                                    for metric_name, row in df.iterrows():
                                        all_stats_agg[home_team][metric_name] += row[home_team]
                                        all_stats_agg[away_team][metric_name] += row[away_team]
                                except Exception as e:
                                    print(f"    Warning: match {match_id}: {e}")

            if all_stats_agg:
                stats_df = pd.DataFrame.from_dict(all_stats_agg, orient='index').fillna(0)
                games_series = pd.Series(games_played, name='games')
                stats_df = stats_df.loc[games_series.index]
                stats_per_game_df = stats_df.div(games_series, axis=0)
                stats_per_game_df.replace([np.inf, -np.inf], 0, inplace=True)
                result = stats_per_game_df.fillna(0)
                result.to_parquet(ets_cache)
                print(f"  Saved expanded_team_stats ({len(result)} teams)")

        # --- Set Piece Metrics ---
        sp_cache = os.path.join(STATS_CACHE_DIR, f'set_piece_metrics_{season_id}.parquet')
        if os.path.exists(sp_cache):
            print(f"  set_piece_metrics: already cached")
        else:
            print(f"  Computing set piece metrics...")
            events_df = season_events.copy()
            events_df['total_seconds'] = events_df['minute'] * 60 + events_df['second']
            ATTACKING_THIRD = 66.67
            LONG_THROW_THRESHOLD = 25

            results = {}
            teams = events_df['team.name'].dropna().unique()
            for team in teams:
                results[team] = {
                    'xG from Corners': 0, 'Goals from Corners': 0, 'Corners': 0,
                    'xG from Att Throw-ins': 0, 'Goals from Att Throw-ins': 0, 'Att Throw-ins': 0,
                    'xG from Free Kicks': 0, 'Goals from Free Kicks': 0, 'Free Kicks Att Third': 0,
                    'xG from Set Pieces': 0, 'Goals from Set Pieces': 0, 'Set Pieces Att Third': 0,
                    'Long Throws': 0,
                    'xG Conceded Corners': 0, 'Goals Conceded Corners': 0, 'Corners Against': 0,
                    'xG Conceded Att Throw-ins': 0, 'Goals Conceded Att Throw-ins': 0,
                    'xG Conceded Free Kicks': 0, 'Goals Conceded Free Kicks': 0,
                    'xG Conceded Set Pieces': 0, 'Goals Conceded Set Pieces': 0,
                }

            for match_id in events_df['matchId'].unique():
                match_events = events_df[events_df['matchId'] == match_id].sort_values('total_seconds')
                match_teams = match_events['team.name'].dropna().unique()
                shots = match_events[match_events['type.primary'] == 'shot'].copy()

                # Corners
                corners = match_events[match_events['type.primary'] == 'corner']
                for _, corner in corners.iterrows():
                    team = corner['team.name']
                    if pd.isna(team) or team not in results:
                        continue
                    corner_time = corner['total_seconds']
                    results[team]['Corners'] += 1
                    team_shots = shots[(shots['team.name'] == team) &
                                      (shots['total_seconds'] >= corner_time) &
                                      (shots['total_seconds'] <= corner_time + 15)]
                    xg = team_shots['shot.xg'].sum() if not team_shots.empty else 0
                    goals = int(team_shots['shot.isGoal'].sum()) if not team_shots.empty else 0
                    results[team]['xG from Corners'] += xg
                    results[team]['Goals from Corners'] += goals
                    results[team]['xG from Set Pieces'] += xg
                    results[team]['Goals from Set Pieces'] += goals
                    results[team]['Set Pieces Att Third'] += 1
                    for opp_team in match_teams:
                        if opp_team != team and opp_team in results:
                            results[opp_team]['Corners Against'] += 1
                            results[opp_team]['xG Conceded Corners'] += xg
                            results[opp_team]['Goals Conceded Corners'] += goals
                            results[opp_team]['xG Conceded Set Pieces'] += xg
                            results[opp_team]['Goals Conceded Set Pieces'] += goals

                # Attacking throw-ins
                throw_ins = match_events[match_events['type.primary'] == 'throw_in']
                att_throw_ins = throw_ins[throw_ins['location.x'] >= ATTACKING_THIRD]
                for _, throw in att_throw_ins.iterrows():
                    team = throw['team.name']
                    if pd.isna(team) or team not in results:
                        continue
                    throw_time = throw['total_seconds']
                    results[team]['Att Throw-ins'] += 1
                    results[team]['Set Pieces Att Third'] += 1
                    team_shots = shots[(shots['team.name'] == team) &
                                      (shots['total_seconds'] >= throw_time) &
                                      (shots['total_seconds'] <= throw_time + 15)]
                    xg = team_shots['shot.xg'].sum() if not team_shots.empty else 0
                    goals = int(team_shots['shot.isGoal'].sum()) if not team_shots.empty else 0
                    results[team]['xG from Att Throw-ins'] += xg
                    results[team]['Goals from Att Throw-ins'] += goals
                    results[team]['xG from Set Pieces'] += xg
                    results[team]['Goals from Set Pieces'] += goals
                    for opp_team in match_teams:
                        if opp_team != team and opp_team in results:
                            results[opp_team]['xG Conceded Att Throw-ins'] += xg
                            results[opp_team]['Goals Conceded Att Throw-ins'] += goals
                            results[opp_team]['xG Conceded Set Pieces'] += xg
                            results[opp_team]['Goals Conceded Set Pieces'] += goals

                # Long throws
                long_throws = throw_ins[throw_ins['pass.length'] >= LONG_THROW_THRESHOLD]
                for _, throw in long_throws.iterrows():
                    team = throw['team.name']
                    if pd.isna(team) or team not in results:
                        continue
                    results[team]['Long Throws'] += 1

                # Free kicks in attacking third
                free_kicks = match_events[match_events['type.primary'] == 'free_kick']
                att_free_kicks = free_kicks[free_kicks['location.x'] >= ATTACKING_THIRD]
                for _, fk in att_free_kicks.iterrows():
                    team = fk['team.name']
                    if pd.isna(team) or team not in results:
                        continue
                    fk_time = fk['total_seconds']
                    results[team]['Free Kicks Att Third'] += 1
                    results[team]['Set Pieces Att Third'] += 1
                    team_shots = shots[(shots['team.name'] == team) &
                                      (shots['total_seconds'] >= fk_time) &
                                      (shots['total_seconds'] <= fk_time + 15)]
                    xg = team_shots['shot.xg'].sum() if not team_shots.empty else 0
                    goals = int(team_shots['shot.isGoal'].sum()) if not team_shots.empty else 0
                    results[team]['xG from Free Kicks'] += xg
                    results[team]['Goals from Free Kicks'] += goals
                    results[team]['xG from Set Pieces'] += xg
                    results[team]['Goals from Set Pieces'] += goals
                    for opp_team in match_teams:
                        if opp_team != team and opp_team in results:
                            results[opp_team]['xG Conceded Free Kicks'] += xg
                            results[opp_team]['Goals Conceded Free Kicks'] += goals
                            results[opp_team]['xG Conceded Set Pieces'] += xg
                            results[opp_team]['Goals Conceded Set Pieces'] += goals

            # Per-event metrics
            for team in results:
                r = results[team]
                r['xG per Corner'] = round(r['xG from Corners'] / r['Corners'], 3) if r['Corners'] > 0 else 0
                r['xG per Att Set Piece'] = round(r['xG from Set Pieces'] / r['Set Pieces Att Third'], 3) if r['Set Pieces Att Third'] > 0 else 0
                r['xG Conceded per Corner'] = round(r['xG Conceded Corners'] / r['Corners Against'], 3) if r['Corners Against'] > 0 else 0
                r['Goals per Corner'] = round(r['Goals from Corners'] / r['Corners'], 3) if r['Corners'] > 0 else 0
                r['Goals Conceded per Corner'] = round(r['Goals Conceded Corners'] / r['Corners Against'], 3) if r['Corners Against'] > 0 else 0

            sp_df = pd.DataFrame.from_dict(results, orient='index')
            sp_df.to_parquet(sp_cache)
            print(f"  Saved set_piece_metrics ({len(sp_df)} teams)")

    print("\nDone! All seasons cached.")


if __name__ == '__main__':
    main()
