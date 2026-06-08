#!/usr/bin/env python3
"""Defensive Responsibility (DefR) — StatsBomb-style empirical model.

Replicates StatsBomb's DefR concept from the Hudl blog post:
  https://www.hudl.com/blog/defensive-responsibility-defr-statsbomb

Pure event-based (no tracking data). For every opposition event
(pass / carry / shot), predict which formation slot is most likely to
respond defensively in the next ~5 seconds, then compare expected vs
actual defensive actions per player.

Outputs models/defr/defr_per_player_season.parquet with one row per
(playerId, seasonId) and columns:
    expected_def_actions  — sum of predicted P(this player responds)
                              across every opposition event in matches
                              the player participated in
    actual_def_actions    — count of the player's defensive events
    defr                  — actual − expected (signed, in absolute action count)
    defr_per90            — DefR / (mins_played / 90)
    n_opp_events          — opposition events the player was on pitch for
                              (denominator for the expected sum)
    position              — player's most-frequent position-slot in season
    mins_played           — minutes from gpa_player_season_values.parquet

Usage:
    python models/defr/compute_defr.py
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DASHBOARD_DIR = HERE.parent.parent

# ----- Tunables ------------------------------------------------------------
# Time window (seconds) after an opposition event in which we count a
# defensive action as the "response" to that event. v2: tightened from
# 5s to 3s so distant-in-time defensive events aren't mis-tagged as
# responses.
RESPONSE_WINDOW_SEC = 3.0

# Zone grid for opposition events. 4 columns × 3 rows = 12 zones, standard
# tactical grid. Coordinates are 0-100 in Wyscout, from the acting team's
# attacking direction (so opp x=100 → defending team's defensive third).
N_ZONE_X = 4
N_ZONE_Y = 3

# Opposition event types we consider "could trigger a defensive response"
OPP_EVENT_TYPES = {'pass', 'carry', 'shot'}

# Position-slot bucketing. Wyscout has 30+ position codes; we collapse to
# the level StatsBomb's framework cares about (formation slot, not pos group).
# A slot is the COMBINATION (side × line × specialism) — granular enough
# to distinguish LCB from RCB but coarse enough to share data across
# matches with slightly different formations.
SLOT_MAP = {
    # Goalkeeper
    'GK': 'GK',
    # Center-backs
    'CB': 'CB',   'LCB': 'LCB',  'RCB': 'RCB',
    'LCB3': 'LCB','RCB3': 'RCB',
    # Fullbacks / wingbacks
    'LB': 'LB',   'RB': 'RB',
    'LB5': 'LB',  'RB5': 'RB',
    'LWB': 'LB',  'RWB': 'RB',
    # Defensive midfielders
    'DMF': 'DMF', 'LDMF': 'LCM', 'RDMF': 'RCM',
    # Central midfielders
    'CMF': 'CM',  'LCMF': 'LCM','RCMF': 'RCM',
    'LCMF3': 'LCM','RCMF3': 'RCM',
    # Attacking mids
    'AMF': 'AM',  'LAMF': 'LAM','RAMF': 'RAM',
    'LMF': 'LCM', 'RMF': 'RCM',
    # Wingers / wide forwards
    'LW': 'LW',   'RW': 'RW',
    'LWF': 'LW',  'RWF': 'RW',
    # Strikers
    'CF': 'ST',   'SS': 'AM',
}


# ----- Helpers -------------------------------------------------------------
def slot_of(pos: str | None) -> str | None:
    if pos is None or (isinstance(pos, float) and pd.isna(pos)):
        return None
    return SLOT_MAP.get(str(pos))


def zone_of(x: float, y: float) -> tuple[int, int]:
    """Return (zx, zy) ∈ [0,N_ZONE_X)×[0,N_ZONE_Y)."""
    zx = int(min(max(x, 0), 99.999) / (100.0 / N_ZONE_X))
    zy = int(min(max(y, 0), 99.999) / (100.0 / N_ZONE_Y))
    return zx, zy


def event_time_seconds(row) -> float:
    """Combine matchPeriod + minute + second into a single sortable seconds
    value within the match. matchPeriod is '1H','2H','E1','E2','P'."""
    period_offset = {
        '1H': 0, '2H': 45 * 60, 'E1': 90 * 60,
        'E2': 105 * 60, 'P': 120 * 60,
    }.get(row.get('matchPeriod'), 0)
    minute = float(row.get('minute') or 0)
    second = float(row.get('second') or 0)
    return period_offset + minute * 60 + second


def is_defensive_action(row) -> bool:
    """Defensive event types (StatsBomb-equivalent): interception,
    clearance, defensive duel. Aerial duels excluded because Wyscout
    doesn't distinguish offensive vs defensive aerials — a striker
    winning a header for a chance would otherwise count as defense."""
    tp = row.get('type.primary')
    if tp in ('interception', 'clearance'):
        return True
    if row.get('groundDuel.duelType') == 'defensive_duel':
        return True
    return False


# ----- Step 1 — Load + filter events --------------------------------------
def load_events() -> pd.DataFrame:
    cols = [
        'matchId', 'matchPeriod', 'minute', 'second', 'seasonId',
        'team.id', 'team.name', 'player.id', 'player.position',
        'type.primary', 'location.x', 'location.y',
        'pass.endLocation.x', 'pass.endLocation.y',
        'carry.endLocation.x', 'carry.endLocation.y',
        'possession.duration', 'possession.eventIndex',
        'groundDuel.duelType', 'aerialDuel.firstTouch',
        'shot.onTarget',
    ]
    df = pd.read_parquet(DASHBOARD_DIR / 'raw_events.parquet', columns=cols)
    # Drop events with no actor or no location
    df = df.dropna(subset=['player.id', 'team.id'])
    df['player.id'] = df['player.id'].astype(int)
    df['team.id'] = df['team.id'].astype(int)
    df['seasonId'] = pd.to_numeric(df['seasonId'], errors='coerce').astype('Int64')
    df['_t'] = df.apply(event_time_seconds, axis=1)
    # v2 — for each event, the EFFECTIVE LOCATION for defensive responsibility:
    #   pass:  pass.endLocation (where the ball ends up = defensive demand)
    #   carry: carry.endLocation (where the carrier moved to)
    #   shot:  location (the shooter's spot)
    #   other: location
    df['_eff_x'] = df['location.x']
    df['_eff_y'] = df['location.y']
    pass_mask = (df['type.primary'] == 'pass') & df['pass.endLocation.x'].notna()
    df.loc[pass_mask, '_eff_x'] = df.loc[pass_mask, 'pass.endLocation.x']
    df.loc[pass_mask, '_eff_y'] = df.loc[pass_mask, 'pass.endLocation.y']
    carry_mask = (df['type.primary'] == 'carry') & df['carry.endLocation.x'].notna()
    df.loc[carry_mask, '_eff_x'] = df.loc[carry_mask, 'carry.endLocation.x']
    df.loc[carry_mask, '_eff_y'] = df.loc[carry_mask, 'carry.endLocation.y']
    return df


def _mirror_x(x):
    """Wyscout coordinates are from acting team's perspective; opp x=80
    (their attacking third) = defending team's x=20 (their defensive third).
    Mirror so all locations can be compared in the same frame."""
    return 100.0 - x


def build_response_table(ev: pd.DataFrame) -> pd.DataFrame:
    """For every opposition event with a location, find the next defensive
    event by the opposing team within RESPONSE_WINDOW_SEC AND in the same
    or adjacent zone (after mirroring the responder's coords into the opp
    team's frame). The spatial constraint stops chronologically-close but
    spatially-unrelated events from being tagged as responses."""
    rows = []
    for mid, mdf in ev.groupby('matchId'):
        mdf = mdf.sort_values('_t').reset_index(drop=True)
        mdf['_is_def'] = mdf.apply(is_defensive_action, axis=1)
        for i, row in mdf.iterrows():
            if row['type.primary'] not in OPP_EVENT_TYPES:
                continue
            # v2 — use effective location (pass.endLocation for passes,
            # carry.endLocation for carries) instead of event origin.
            x, y = row.get('_eff_x'), row.get('_eff_y')
            if pd.isna(x) or pd.isna(y):
                continue
            opp_team = row['team.id']
            t0 = row['_t']
            zx, zy = zone_of(x, y)
            # Candidate responders: time window + other team + defensive
            window = mdf[(mdf['_t'] >= t0)
                          & (mdf['_t'] <= t0 + RESPONSE_WINDOW_SEC)
                          & (mdf['team.id'] != opp_team)
                          & (mdf['_is_def'])
                          & (mdf['location.x'].notna())
                          & (mdf['location.y'].notna())]
            # Spatial filter: responder's event (in OPP coordinate frame)
            # must be in same or adjacent zone as the opp event.
            responder_slot = None
            responder_pid = None
            def_team = None
            if not window.empty:
                # Mirror responder coords to opp frame, then bucket
                r_x_opp = window['location.x'].apply(_mirror_x)
                r_y_opp = 100 - window['location.y']  # mirror y too (same convention)
                r_zx = (r_x_opp / (100.0 / N_ZONE_X)).clip(0, N_ZONE_X - 1).astype(int)
                r_zy = (r_y_opp / (100.0 / N_ZONE_Y)).clip(0, N_ZONE_Y - 1).astype(int)
                # Same or adjacent (Chebyshev distance ≤ 1) zone
                spatial_ok = ((r_zx - zx).abs() <= 1) & ((r_zy - zy).abs() <= 1)
                close = window[spatial_ok]
                if not close.empty:
                    r0 = close.iloc[0]
                    responder_slot = slot_of(r0.get('player.position'))
                    responder_pid = int(r0['player.id'])
                    def_team = int(r0['team.id'])
            rows.append({
                'matchId': mid,
                'seasonId': int(row['seasonId']) if pd.notna(row['seasonId']) else None,
                'opp_team': opp_team,
                'def_team': def_team,
                'opp_type': row['type.primary'],
                'opp_x': float(x), 'opp_y': float(y),
                'zx': zx, 'zy': zy,
                't': t0,
                'responder_slot': responder_slot,
                'responder_pid': responder_pid,
            })
    return pd.DataFrame(rows)


# ----- Step 2 — Empirical conditional probability -------------------------
def build_prob_table(resp: pd.DataFrame) -> pd.DataFrame:
    """For each (zx, zy, opp_type, slot) bucket, compute the
    UNCONDITIONAL empirical probability that the slot is the responder
    given this kind of opposition event happens. About 70% of opp
    events get NO response within the window, so the sum across slots
    per (zone, type) is ~0.30, not 1.0. That's the correct number to
    use for expected — multiplying by it across all opp events the
    player faced gives a calibrated expected count."""
    matched = resp.dropna(subset=['responder_slot'])
    # Numerator: events responded by this slot in this bucket
    by_slot = (matched.groupby(['zx', 'zy', 'opp_type', 'responder_slot'])
                  .size().reset_index(name='n_responded'))
    # Denominator: TOTAL opp events in the bucket (matched + unmatched)
    by_bucket = (resp.groupby(['zx', 'zy', 'opp_type'])
                    .size().reset_index(name='n_total'))
    grp = by_slot.merge(by_bucket, on=['zx', 'zy', 'opp_type'])
    grp['prob'] = grp['n_responded'] / grp['n_total']
    return grp


# ----- Step 3 — Per-player expected + actual ------------------------------
def compute_per_player(ev: pd.DataFrame,
                         resp: pd.DataFrame,
                         prob_table: pd.DataFrame) -> pd.DataFrame:
    """Compute expected + actual defensive actions per (player, season).
    Returns DataFrame with one row per (playerId, seasonId)."""
    # Most-frequent slot per (player, season) — handles slight pos changes
    ev_with_slot = ev.copy()
    ev_with_slot['slot'] = ev_with_slot['player.position'].map(slot_of)
    player_slot = (ev_with_slot.dropna(subset=['slot'])
                                  .groupby(['player.id', 'seasonId', 'slot'])
                                  .size().reset_index(name='n')
                                  .sort_values('n', ascending=False)
                                  .drop_duplicates(['player.id', 'seasonId'])
                                  [['player.id', 'seasonId', 'slot']])

    # Per (player, season): which matches did they participate in?
    player_matches = (ev_with_slot[['player.id', 'matchId', 'seasonId']]
                        .drop_duplicates())

    # Index prob_table for fast lookup
    prob_lookup = prob_table.set_index(['zx', 'zy', 'opp_type',
                                           'responder_slot'])['prob']

    # For each opposition event in resp, distribute expected probability
    # across players on the pitch at that match in the relevant slot.
    # "On the pitch" approximation: any player with at least one event in
    # the match. (We'll refine for substitutions in v2.)
    # v2 — mins-share weighting. For each (player, match) compute the
    # approximate minutes played as (last_event_time - first_event_time)/60.
    # An opp event is only counted in a player's expected if it happened
    # while they were on the pitch (t between first and last event time).
    # This replaces v1's "any player with an event = on for the whole match"
    # which over-attributed expected to subs.
    pmt = (ev_with_slot.groupby(['player.id', 'matchId'])['_t']
              .agg(['min', 'max']).reset_index()
              .rename(columns={'min': 't_in', 'max': 't_out'}))
    pmt_lookup = {(int(r['player.id']), r['matchId']):
                    (float(r['t_in']), float(r['t_out']))
                    for _, r in pmt.iterrows()}

    # Sum expected per (player, season) by iterating through resp events
    expected_acc = {}   # (pid, sid) → sum of expected_p
    n_opp_acc = {}      # (pid, sid) → count of opposition events seen
    # Pre-compute (match → defending team) for each row, defended by joining
    # via opposition team. For a match with teams A and B, when team A is the
    # opp_team, the defending team is B.
    # Build match_id → set of teams
    match_teams = ev_with_slot.groupby('matchId')['team.id'].unique().to_dict()
    # Build (matchId, defending team) → list of players
    def_team_players = (ev_with_slot[['matchId', 'team.id', 'player.id',
                                          'seasonId']]
                          .drop_duplicates()
                          .groupby(['matchId', 'team.id'])
                          [['player.id', 'seasonId']]
                          .apply(lambda g: list(zip(g['player.id'], g['seasonId'])))
                          .to_dict())
    # Player → slot lookup
    player_slot_lookup = player_slot.set_index(['player.id', 'seasonId'])['slot']

    print(f"  Processing {len(resp):,} opposition events…", flush=True)
    for _, r in resp.iterrows():
        mid = r['matchId']
        opp_team = r['opp_team']
        teams = match_teams.get(mid)
        if teams is None or len(teams) < 2:
            continue
        def_team = next((t for t in teams if t != opp_team), None)
        if def_team is None:
            continue
        t_event = float(r['t'])
        players_in_match = def_team_players.get((mid, def_team), [])
        for pid, sid in players_in_match:
            # v2 — mins-share: skip the event if the player wasn't on the
            # pitch (t outside their first/last event timestamp window).
            t_in_out = pmt_lookup.get((pid, mid))
            if t_in_out is None:
                continue
            t_in, t_out = t_in_out
            if t_event < t_in or t_event > t_out:
                continue
            slot = player_slot_lookup.get((pid, sid))
            if slot is None:
                continue
            p = prob_lookup.get((r['zx'], r['zy'], r['opp_type'], slot), 0.0)
            if p > 0:
                expected_acc[(pid, sid)] = expected_acc.get((pid, sid), 0) + p
            n_opp_acc[(pid, sid)] = n_opp_acc.get((pid, sid), 0) + 1

    # Actual defensive actions per (player, season)
    ev_with_slot['_is_def'] = ev_with_slot.apply(is_defensive_action, axis=1)
    actual = (ev_with_slot[ev_with_slot['_is_def']]
                .groupby(['player.id', 'seasonId']).size()
                .reset_index(name='actual_def_actions')
                .rename(columns={'player.id': 'playerId'}))

    # Build output
    rows = []
    for (pid, sid), exp_v in expected_acc.items():
        rows.append({
            'playerId': pid,
            'seasonId': sid,
            'expected_def_actions': exp_v,
            'n_opp_events': n_opp_acc.get((pid, sid), 0),
        })
    out = pd.DataFrame(rows)
    out = out.merge(actual.rename(columns={'player.id': 'playerId'}),
                      on=['playerId', 'seasonId'], how='left')
    out['actual_def_actions'] = out['actual_def_actions'].fillna(0).astype(int)
    out['defr'] = out['actual_def_actions'] - out['expected_def_actions']
    # Most-common slot
    ps = player_slot.rename(columns={'player.id': 'playerId',
                                       'slot': 'position'})
    out = out.merge(ps, on=['playerId', 'seasonId'], how='left')

    # Attach minutes from GPA parquet (so we can compute per-90)
    try:
        gpa = pd.read_parquet(DASHBOARD_DIR /
                               'gpa_player_season_values.parquet',
                               columns=['playerId', 'seasonId', 'mins_played'])
        gpa['playerId'] = pd.to_numeric(gpa['playerId'], errors='coerce').astype('Int64')
        gpa['seasonId'] = pd.to_numeric(gpa['seasonId'], errors='coerce').astype('Int64')
        out['playerId'] = out['playerId'].astype('Int64')
        out['seasonId'] = out['seasonId'].astype('Int64')
        out = out.merge(gpa, on=['playerId', 'seasonId'], how='left')
        out['defr_per90'] = np.where(
            out['mins_played'].fillna(0) > 0,
            out['defr'] / (out['mins_played'] / 90.0),
            np.nan,
        )
    except Exception as e:
        print(f"  [warn] could not attach minutes: {e}")
        out['mins_played'] = np.nan
        out['defr_per90'] = np.nan

    # v2 — position-normalized DefR. The raw DefR/90 is biased positive
    # (actual includes responses to set pieces / throw-ins that aren't
    # in our opposition event set) AND varies systematically by position
    # (DMFs median +9.8, GKs median +1.1). Normalize by subtracting the
    # position-median so 0 = median defender at that position, positive
    # = above the typical for their slot.
    #
    # Use ONLY players with ≥800 mins as the normalization cohort to
    # avoid noisy small-sample skewing the medians.
    valid = out[(out['mins_played'].fillna(0) >= 800) & out['defr_per90'].notna()]
    pos_median = valid.groupby('position')['defr_per90'].median().to_dict()
    pos_std = valid.groupby('position')['defr_per90'].std().to_dict()
    out['defr_per90_vs_position'] = out.apply(
        lambda r: (r['defr_per90'] - pos_median.get(r['position'], 0))
                    if pd.notna(r['defr_per90']) else np.nan,
        axis=1,
    )
    # z-score within position (how many SDs above/below typical) — useful
    # for cross-position comparisons.
    out['defr_z_vs_position'] = out.apply(
        lambda r: ((r['defr_per90'] - pos_median.get(r['position'], 0))
                     / pos_std.get(r['position'], 1.0))
                    if pd.notna(r['defr_per90']) and pos_std.get(r['position'], 0) > 0
                    else np.nan,
        axis=1,
    )

    return out


# ----- Main ----------------------------------------------------------------
def main():
    print("=" * 60)
    print("DefR — Defensive Responsibility — empirical event-based model")
    print("=" * 60)
    print("\n[1/4] Loading events…", flush=True)
    ev = load_events()
    print(f"  Loaded {len(ev):,} events across "
           f"{ev['matchId'].nunique():,} matches.")

    print("\n[2/4] Building opposition-event response table…", flush=True)
    resp = build_response_table(ev)
    matched = resp['responder_slot'].notna().sum()
    print(f"  {len(resp):,} opposition events; "
           f"{matched:,} ({matched/len(resp)*100:.0f}%) had a defensive "
           f"response within {RESPONSE_WINDOW_SEC:.0f}s.")

    print("\n[3/4] Estimating empirical P(responder_slot | zone, type)…",
            flush=True)
    prob = build_prob_table(resp)
    print(f"  Probability table: {len(prob):,} bucket × slot combinations.")

    print("\n[4/4] Computing per-(player, season) expected + actual…", flush=True)
    out = compute_per_player(ev, resp, prob)
    print(f"  {len(out):,} (player, season) rows produced.")

    out_path = HERE / 'defr_per_player_season.parquet'
    out.to_parquet(out_path)
    print(f"\n✅ Saved {out_path}")

    # Sample output — sort by position-normalized DefR z-score
    print("\nSample — top 15 by DefR z-score vs position (min 1000 mins):")
    sample = out[out['mins_played'] >= 1000].sort_values(
        'defr_z_vs_position', ascending=False).head(15)
    print(sample[['playerId', 'seasonId', 'position', 'mins_played',
                    'expected_def_actions', 'actual_def_actions',
                    'defr_per90', 'defr_per90_vs_position',
                    'defr_z_vs_position']].round(2).to_string(index=False))

    print("\nSample — bottom 15 by DefR z-score vs position (min 1000 mins):")
    sample = out[out['mins_played'] >= 1000].sort_values(
        'defr_z_vs_position', ascending=True).head(15)
    print(sample[['playerId', 'seasonId', 'position', 'mins_played',
                    'expected_def_actions', 'actual_def_actions',
                    'defr_per90', 'defr_per90_vs_position',
                    'defr_z_vs_position']].round(2).to_string(index=False))

    print("\nDistribution by position (z-scores should be ~normal(0, 1)):")
    by_pos = out[out['mins_played'] >= 800].groupby('position')[
        'defr_z_vs_position'].agg(['count', 'mean', 'std']).round(2)
    print(by_pos.to_string())


if __name__ == '__main__':
    main()
