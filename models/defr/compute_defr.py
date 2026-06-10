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

# v5 — ATTACK MOMENTUM (StatsBomb factor 2) from Wyscout's possession.types
# momentum tags. We use ONLY the momentum tags (counterattack / transition_*)
# which describe the state of play, NOT the set-piece-origin tags
# (throw_in / free_kick / corner). Those are possession-LEVEL markers of
# how the possession STARTED — a possession that begins with a throw-in
# then recycles into open play would (wrongly) tag all its events as a set
# piece, which over-counts set pieces to ~45% of events. The momentum tags
# are far more stable across a possession's events.
#   counter    — 'counterattack' OR 'transition_high'  (fast break)
#   transition — 'transition_medium' OR 'transition_low'  (slower build off a turnover)
#   settled    — everything else ('attack' / set-piece-origin / untagged)
_PHASE_COUNTER_TAGS = {'counterattack', 'transition_high'}
_PHASE_TRANSITION_TAGS = {'transition_medium', 'transition_low'}

# v4 — DYNAMIC DEFENSIVE SHAPE (StatsBomb factor 3). At each opposition
# event we estimate the defending team's line height from the rolling
# median x-position (own frame) of their recent DEFENSIVE-ENGAGEMENT
# events. v3 used only interception/clearance/tackle (sparse). v4 adds
# recoveries + fouls so the estimate is denser and more stable.
# Bucket into:
#   high_line  — defending team pressing high up the pitch
#   mid_block  — standard mid-block
#   low_block  — sitting deep, compact near own goal
# v6.1 — recalibrated. Diagnostic showed the 30s/min-4 estimate left
# 98% of events at the mid_block fallback (too few recent engagements).
# Widened the window to 90s and dropped the min to 3 so far more events
# get a real estimate. Thresholds set to the empirical terciles of the
# line-height distribution (median engagement x, own frame: p33≈20,
# p67≈38) so the three buckets are BALANCED and the factor carries
# information. (When it does differentiate, responder-distribution shift
# is TVD 0.21-0.24 — real signal that was being wasted.)
SHAPE_WINDOW_SEC = 90.0   # rolling window for estimating line height
SHAPE_HIGH_THRESHOLD = 38.0   # engagement x (own frame) >= this = higher line
SHAPE_LOW_THRESHOLD = 20.0    # <= this = deep/low block
SHAPE_MIN_EVENTS = 3          # need >= this many recent engagements to estimate

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
# v3 — unified event source: BOTH Liga 3 AND Campeonato. The dashboard's
# own raw_events.parquet is Liga-3-only, so DefR previously couldn't cover
# Camp. The GPA v2 project's parquet_data/ has both leagues from the same
# scrape pipeline (consistent vintage). We read from there for the DefR
# precompute; the output parquet (which is what ships) covers both leagues.
ACP_ROOT = DASHBOARD_DIR.parent.parent   # …/ACP_Official
GPA_V2_DATA = ACP_ROOT / 'GPA Model Project v2' / 'parquet_data'
EVENT_SOURCES = [
    GPA_V2_DATA / 'liga3_portugal_events.parquet',         # Liga 3 (all 5 seasons)
    GPA_V2_DATA / 'campeonato_portugal_events.parquet',    # Campeonato (2 seasons)
]
# Fallback if the v2 project dir isn't present (e.g. running elsewhere):
# the dashboard's Liga-3-only events.
FALLBACK_SOURCE = DASHBOARD_DIR / 'raw_events.parquet'

_EVENT_COLS = [
    'id', 'relatedEventId',
    'matchId', 'matchPeriod', 'minute', 'second', 'seasonId',
    'team.id', 'team.name', 'player.id', 'player.position',
    'type.primary', 'type.secondary', 'location.x', 'location.y',
    'pass.endLocation.x', 'pass.endLocation.y',
    'carry.endLocation.x', 'carry.endLocation.y',
    'possession.types',
    'groundDuel.duelType', 'aerialDuel.firstTouch',
    'groundDuel.stoppedProgress', 'groundDuel.recoveredPossession',
    'groundDuel.opponent.position', 'aerialDuel.opponent.position',
    'competitionId',
]

# v7 — recovery is the most reliable defensive action (within-season
# split-half full-season r = 0.936, vs 0.933 for the int+clear+tackle
# set). Recoveries live in type.secondary as 'recovery' or
# 'counterpressing_recovery' (~254k events). Adding them to the
# defensive-action set both improves reliability and densifies `actual`,
# which directly cuts noise in the DefR = actual − expected difference.
_RECOVERY_TAGS = {'recovery', 'counterpressing_recovery'}

# v8.1 — DEFENSIVE AERIAL DUELS. Wyscout has no defensive-aerial flag, so
# we define one. An aerial duel is defensive — regardless of whether the
# player won it (firstTouch True OR False) — when:
#   • it's in the player's OWN HALF (location.x ≤ 50, own attacking
#     frame), OR
#   • it's in the OPPONENT'S half (x > 50) AND the immediately preceding
#     action was by the opponent (i.e. the player is contesting a header
#     during the opponent's phase of play — a high defensive header /
#     pressing duel, not an own-team flick-on).
# "Preceded by opponent action" is read from the team of the most recent
# non-aerial event before the duel (the cross / long ball / touch that
# led to it). For the two sides of one physical duel this correctly keeps
# the defender's side and drops the attacker's side.
#
# Double-counting: aerial duels are always type.primary == 'duel' and are
# NEVER also logged as a clearance/interception (verified: 0 of 137,851
# int/clear events carry an aerial flag). Same-player aerial↔clearance
# links are ~0.01% of all defensive actions — negligible. Recoveries are
# split so an aerial only ever enters via the aerial branch.
DEFENSIVE_AERIAL_MAX_X = 50.0


def _row_is_recovery(types_secondary) -> bool:
    if isinstance(types_secondary, (list, tuple, np.ndarray)):
        for t in types_secondary:
            if t in _RECOVERY_TAGS:
                return True
        return False
    return types_secondary in _RECOVERY_TAGS


def load_events() -> pd.DataFrame:
    frames = []
    sources = [p for p in EVENT_SOURCES if p.exists()]
    if not sources:
        print(f"  [warn] v2 event sources not found; "
               f"falling back to dashboard raw_events (Liga 3 only)")
        sources = [FALLBACK_SOURCE]
    for src in sources:
        # Read only the columns that exist in this file (schemas differ
        # slightly between the two leagues' files)
        import pyarrow.parquet as pq
        avail = set(pq.read_schema(src).names)
        use_cols = [c for c in _EVENT_COLS if c in avail]
        part = pd.read_parquet(src, columns=use_cols)
        # Ensure every expected column exists (fill missing with NaN)
        for c in _EVENT_COLS:
            if c not in part.columns:
                part[c] = np.nan
        print(f"  Loaded {len(part):,} events from {src.name} "
               f"(seasons {sorted(part['seasonId'].dropna().unique().astype(int).tolist())})")
        frames.append(part)
    df = pd.concat(frames, ignore_index=True)

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
    # v5 — precompute event-level phase from possession.types (vectorized
    # via list-comprehension; ~one pass over the column).
    df['_phase'] = [phase_of(t) for t in df['possession.types'].tolist()]
    # v7/v8 — precompute defensive-action flags once (vectorized).
    #   _is_recovery   — type.secondary recovery tag (any event)
    #   _is_aerial     — an aerial-duel event
    #   _is_def_aerial — WON aerial in own half (defensive header)
    #   _is_def        — counted defensive action:
    #                      interception | clearance | defensive duel
    #                      | GROUND recovery (recovery & not aerial)
    #                      | defensive aerial
    #                    Aerials enter ONLY via _is_def_aerial, never via the
    #                    recovery branch, so each aerial is counted once.
    #   _is_engage     — denser set for the line-height estimate (adds fouls)
    df['_is_recovery'] = [_row_is_recovery(t) for t in df['type.secondary'].tolist()]
    _tp = df['type.primary']
    df['_is_aerial'] = df['aerialDuel.firstTouch'].notna()
    # v8.1 — team of the most recent NON-aerial event before each event
    # (the action that led into the duel). Used to test "preceded by
    # opponent action" for opponent-half aerials. ffill within match,
    # over rows in chronological (load) order; aerial rows are NaN so
    # ffill carries the preceding non-aerial team into them.
    _na_team = df['team.id'].where(~df['_is_aerial'])
    df['_prev_team'] = _na_team.groupby(df['matchId']).ffill()
    _own_half = df['location.x'] <= DEFENSIVE_AERIAL_MAX_X
    _opp_half_def = ((df['location.x'] > DEFENSIVE_AERIAL_MAX_X)
                       & df['_prev_team'].notna()
                       & (df['_prev_team'] != df['team.id']))
    df['_is_def_aerial'] = df['_is_aerial'] & (_own_half | _opp_half_def)
    # v8.1 — dedup the ~0.02% of defensive aerials whose relatedEventId
    # points to a SAME-player clearance/interception (one physical header
    # logged as both a duel and a clearance). Drop the aerial side so the
    # moment counts once (via the clearance). Verified to be the only
    # same-player double-count path — aerials are never themselves a
    # clearance/interception type.
    if 'id' in df.columns and 'relatedEventId' in df.columns:
        _ci = df.loc[df['type.primary'].isin(['interception', 'clearance']),
                       ['id', 'player.id']].rename(
                         columns={'id': '_rid', 'player.id': '_rpid'})
        _da = df.loc[df['_is_def_aerial'],
                       ['relatedEventId', 'player.id']].reset_index()
        _m = _da.merge(_ci, left_on='relatedEventId', right_on='_rid',
                         how='left')
        _dup = _m.loc[(_m['_rpid'].notna())
                        & (_m['_rpid'] == _m['player.id']), 'index']
        if len(_dup):
            df.loc[_dup, '_is_def_aerial'] = False
            print(f"  [defr] deduped {len(_dup)} aerial↔clearance "
                   f"same-player double-counts")
    _ground_recovery = df['_is_recovery'] & ~df['_is_aerial']
    df['_is_def'] = ((_tp == 'interception') | (_tp == 'clearance')
                       | (df['groundDuel.duelType'] == 'defensive_duel')
                       | _ground_recovery
                       | df['_is_def_aerial'])
    df['_is_engage'] = df['_is_def'] | (_tp == 'infraction')
    # v9 — mutually-exclusive defensive TYPE per event (priority order
    # avoids any event being two types; the five sum to _is_def). Used to
    # produce per-type DefR (expected vs actual for each action type).
    _dtype = np.full(len(df), None, dtype=object)
    _dtype[df['_is_def_aerial'].to_numpy()] = 'def_aerial'
    _gd = df['groundDuel.duelType'].to_numpy()
    _tpn = _tp.to_numpy()
    _gr = _ground_recovery.to_numpy()
    _m = (_dtype == None) & (_tpn == 'interception'); _dtype[_m] = 'interception'
    _m = (_dtype == None) & (_tpn == 'clearance');    _dtype[_m] = 'clearance'
    _m = (_dtype == None) & (_gd == 'defensive_duel'); _dtype[_m] = 'tackle'
    _m = (_dtype == None) & _gr;                       _dtype[_m] = 'recovery'
    df['_dtype'] = _dtype
    return df


def _mirror_x(x):
    """Wyscout coordinates are from acting team's perspective; opp x=80
    (their attacking third) = defending team's x=20 (their defensive third).
    Mirror so all locations can be compared in the same frame."""
    return 100.0 - x


def phase_of(types) -> str:
    """v5 — Attack momentum from Wyscout possession.types momentum tags.
    Priority: counter > transition > settled."""
    if types is None:
        return 'settled'
    if isinstance(types, float) and pd.isna(types):
        return 'settled'
    try:
        tset = set(types) if not isinstance(types, str) else {types}
    except TypeError:
        return 'settled'
    if tset & _PHASE_COUNTER_TAGS:
        return 'counter'
    if tset & _PHASE_TRANSITION_TAGS:
        return 'transition'
    return 'settled'


def build_response_table(ev: pd.DataFrame) -> pd.DataFrame:
    """For every opposition event, find the responder AND tag the event
    with attack momentum (phase) and dynamic defensive shape (line_state).

    Responder match: next defensive action by the OTHER team within
    RESPONSE_WINDOW_SEC AND in the same/adjacent zone (coords mirrored
    into the opp frame).

    v3 adds:
      phase      — counter | settled  (from possession.duration)
      line_state — high_line | mid_block | low_block  (from the defending
                    team's recent defensive-action x-position, own frame)
    """
    rows = []
    zsx = 100.0 / N_ZONE_X
    zsy = 100.0 / N_ZONE_Y
    for mid, mdf in ev.groupby('matchId'):
        mdf = mdf.sort_values('_t').reset_index(drop=True)
        teams = mdf['team.id'].unique()
        if len(teams) < 2:
            continue
        # Pull everything into numpy arrays (vectorized; no per-row .apply)
        t_arr = mdf['_t'].to_numpy()
        team_arr = mdf['team.id'].to_numpy()
        type_arr = mdf['type.primary'].to_numpy()
        locx = mdf['location.x'].to_numpy(dtype=float)
        locy = mdf['location.y'].to_numpy(dtype=float)
        effx = mdf['_eff_x'].to_numpy(dtype=float)
        effy = mdf['_eff_y'].to_numpy(dtype=float)
        pos_arr = mdf['player.position'].to_numpy()
        pid_arr = mdf['player.id'].to_numpy()
        phase_col = mdf['_phase'].to_numpy()
        season = mdf['seasonId'].to_numpy()
        dtype_arr = mdf['_dtype'].to_numpy()         # responder action type
        n = len(mdf)
        # v7 — precomputed defensive-action masks (now include recoveries)
        is_def = mdf['_is_def'].to_numpy()           # responder match + actual
        is_engage = mdf['_is_engage'].to_numpy()     # line-height estimate
        # Per-team time-sorted engagement x (own frame) for line height
        team_def = {}
        for tid in teams:
            m = (team_arr == tid) & is_engage & ~np.isnan(locx)
            team_def[tid] = (t_arr[m], locx[m])

        def line_state_for(def_team_id, t):
            ts, xs = team_def.get(def_team_id, (np.array([]), np.array([])))
            if ts.size == 0:
                return 'mid_block'
            lo = np.searchsorted(ts, t - SHAPE_WINDOW_SEC, side='left')
            hi = np.searchsorted(ts, t, side='right')
            if hi - lo < SHAPE_MIN_EVENTS:
                return 'mid_block'
            med = float(np.median(xs[lo:hi]))
            if med >= SHAPE_HIGH_THRESHOLD:
                return 'high_line'
            if med <= SHAPE_LOW_THRESHOLD:
                return 'low_block'
            return 'mid_block'

        # Mirrored responder zones for the whole match (vectorized)
        r_x_opp = 100.0 - locx
        r_y_opp = 100.0 - locy
        r_zx = np.clip((r_x_opp / zsx).astype('float'), 0, N_ZONE_X - 1)
        r_zy = np.clip((r_y_opp / zsy).astype('float'), 0, N_ZONE_Y - 1)
        # NaN-safe int conversion (NaN → -99 so it never matches a zone)
        r_zx = np.where(np.isnan(locx), -99, r_zx).astype(int)
        r_zy = np.where(np.isnan(locy), -99, r_zy).astype(int)

        for i in range(n):
            if type_arr[i] not in OPP_EVENT_TYPES:
                continue
            x, y = effx[i], effy[i]
            if np.isnan(x) or np.isnan(y):
                continue
            opp_team = team_arr[i]
            t0 = t_arr[i]
            zx = min(max(int(x / zsx), 0), N_ZONE_X - 1)
            zy = min(max(int(y / zsy), 0), N_ZONE_Y - 1)
            def_team_known = next((tt for tt in teams if tt != opp_team), None)
            phase = phase_col[i]
            line_state = (line_state_for(def_team_known, t0)
                            if def_team_known is not None else 'mid_block')
            # Responder window: searchsorted on the time-sorted slice
            hi = np.searchsorted(t_arr, t0 + RESPONSE_WINDOW_SEC, side='right')
            responder_slot = None
            responder_pid = None
            responder_type = None
            def_team = def_team_known
            # Scan the (small) slice [i, hi) for first adjacent-zone def
            # event by the other team.
            for j in range(i, hi):
                if team_arr[j] == opp_team or not is_def[j]:
                    continue
                if r_zx[j] < 0:   # responder had no location
                    continue
                if abs(r_zx[j] - zx) <= 1 and abs(r_zy[j] - zy) <= 1:
                    responder_slot = slot_of(pos_arr[j])
                    responder_pid = int(pid_arr[j])
                    responder_type = dtype_arr[j]
                    def_team = int(team_arr[j])
                    break
            rows.append({
                'matchId': mid,
                'seasonId': int(season[i]) if not pd.isna(season[i]) else None,
                'opp_team': opp_team,
                'def_team': def_team,
                'opp_type': type_arr[i],
                'phase': phase,
                'line_state': line_state,
                'zx': zx, 'zy': zy,
                't': t0,
                'responder_slot': responder_slot,
                'responder_pid': responder_pid,
                'responder_type': responder_type,
            })
    return pd.DataFrame(rows)


# ----- Step 2 — Empirical conditional probability -------------------------
# Empirical-Bayes backoff constant. A fine (zone × type × phase × line)
# bucket with n events gets weight n/(n+K) on its own estimate and
# K/(n+K) on the coarse (zone × type) estimate. K=60 means a fine bucket
# needs ~60 events to be trusted at ~50%.
PROB_SMOOTHING_K = 60.0

FINE_KEYS = ['zx', 'zy', 'opp_type', 'phase', 'line_state']
COARSE_KEYS = ['zx', 'zy', 'opp_type']


# ----- v10 shared lookups ---------------------------------------------------
def compute_modal_slots(ev: pd.DataFrame):
    """Most-frequent slot per (player, season). Returns (df, dict)."""
    e = ev.copy()
    e['slot'] = e['player.position'].map(slot_of)
    player_slot = (e.dropna(subset=['slot'])
                     .groupby(['player.id', 'seasonId', 'slot'])
                     .size().reset_index(name='n')
                     .sort_values('n', ascending=False)
                     .drop_duplicates(['player.id', 'seasonId'])
                     [['player.id', 'seasonId', 'slot']])
    psl = player_slot.set_index(['player.id', 'seasonId'])['slot'].to_dict()
    return player_slot, psl


def match_lookups(ev: pd.DataFrame):
    """(match → teams, (match, team) → players, (player, match) → on-pitch
    window from first/last event timestamps)."""
    match_teams = ev.groupby('matchId')['team.id'].unique().to_dict()
    def_team_players = (ev[['matchId', 'team.id', 'player.id', 'seasonId']]
                          .drop_duplicates()
                          .groupby(['matchId', 'team.id'])
                          [['player.id', 'seasonId']]
                          .apply(lambda g: list(zip(g['player.id'], g['seasonId'])))
                          .to_dict())
    pmt = (ev.groupby(['player.id', 'matchId'])['_t']
              .agg(['min', 'max']).reset_index())
    pmt_lookup = {(int(r['player.id']), r['matchId']):
                    (float(r['min']), float(r['max']))
                    for _, r in pmt.iterrows()}
    return match_teams, def_team_players, pmt_lookup


def build_presence_tables(resp, match_teams, def_team_players, pmt_lookup, psl):
    """v10 — per-context counts of opposition events where each slot was
    actually ON THE PITCH for the defending team. These are the correct
    probability denominators: the old all-events denominator diluted slots
    that aren't in every formation (DMF present for only ~28% of events,
    back-3 CB ~24%, LAM/RAM ~10%), understating their expected ~3-10x.
    Returns (fine, coarse) dicts keyed ctx+(slot,)."""
    R_mid = resp['matchId'].to_numpy()
    R_opp = resp['opp_team'].to_numpy()
    R_zx = resp['zx'].to_numpy(); R_zy = resp['zy'].to_numpy()
    R_ot = resp['opp_type'].to_numpy()
    R_ph = resp['phase'].to_numpy(); R_ls = resp['line_state'].to_numpy()
    R_t = resp['t'].to_numpy(dtype=float)
    fine, coarse = {}, {}
    for k in range(len(resp)):
        mid = R_mid[k]; opp = R_opp[k]
        teams = match_teams.get(mid)
        if teams is None or len(teams) < 2:
            continue
        dft = next((t for t in teams if t != opp), None)
        if dft is None:
            continue
        te = R_t[k]
        slots_here = set()
        for pid, sid in def_team_players.get((mid, dft), []):
            w = pmt_lookup.get((pid, mid))
            if w is None or te < w[0] or te > w[1]:
                continue
            s = psl.get((pid, sid))
            if s is not None:
                slots_here.add(s)
        fk = (R_zx[k], R_zy[k], R_ot[k], R_ph[k], R_ls[k])
        ck = (R_zx[k], R_zy[k], R_ot[k])
        for s in slots_here:
            fine[fk + (s,)] = fine.get(fk + (s,), 0) + 1
            coarse[ck + (s,)] = coarse.get(ck + (s,), 0) + 1
    return fine, coarse


def build_prob_table(resp: pd.DataFrame, presence_fine: dict,
                       presence_coarse: dict) -> pd.DataFrame:
    """v10 — P(slot responds | context, SLOT PRESENT ON PITCH), with
    empirical-Bayes backoff toward the coarse (zone × opp_type) layer.

    Two v10 fixes vs the v3-v9 table:
      - Denominators are presence-conditioned (events where the slot was
        actually on the pitch), not all events — removes the formation-
        dilution artifact for DMF / back-3 CB / LAM / RAM.
      - Numerators use the responder's seasonal MODAL slot
        (resp['responder_modal']) instead of the event-moment position —
        33% of matched rows carried a rotated position code, which leaked
        probability mass between slots.
    Output keeps the 'responder_slot' column name for compatibility."""
    matched = resp.dropna(subset=['responder_modal'])

    fine_num = (matched.groupby(FINE_KEYS + ['responder_modal'])
                   .size().reset_index(name='f_n'))
    fine_num['f_total'] = [
        presence_fine.get((r.zx, r.zy, r.opp_type, r.phase, r.line_state,
                             r.responder_modal), 0)
        for r in fine_num.itertuples(index=False)]
    fine_num = fine_num[fine_num['f_total'] > 0].copy()
    fine_num['fine_prob'] = (fine_num['f_n'] / fine_num['f_total']).clip(upper=1.0)

    coarse_num = (matched.groupby(COARSE_KEYS + ['responder_modal'])
                     .size().reset_index(name='c_n'))
    coarse_num['c_total'] = [
        presence_coarse.get((r.zx, r.zy, r.opp_type, r.responder_modal), 0)
        for r in coarse_num.itertuples(index=False)]
    coarse_num = coarse_num[coarse_num['c_total'] > 0].copy()
    coarse_num['coarse_prob'] = (coarse_num['c_n'] / coarse_num['c_total']).clip(upper=1.0)

    grp = fine_num.merge(
        coarse_num[COARSE_KEYS + ['responder_modal', 'coarse_prob']],
        on=COARSE_KEYS + ['responder_modal'], how='left')
    grp['coarse_prob'] = grp['coarse_prob'].fillna(0.0)
    w = grp['f_total'] / (grp['f_total'] + PROB_SMOOTHING_K)
    grp['prob'] = w * grp['fine_prob'] + (1 - w) * grp['coarse_prob']
    return grp.rename(columns={'responder_modal': 'responder_slot'})


# v9 — the five mutually-exclusive defensive action types we score
# per-type DefR for. Names match _dtype values.
DEFR_TYPES = ['interception', 'clearance', 'tackle', 'recovery', 'def_aerial']


def build_prob_table_by_type(resp: pd.DataFrame, presence_fine: dict,
                               presence_coarse: dict) -> dict:
    """v10 — P(slot responds WITH action-type T | context, slot present);
    same presence denominators + modal numerators as build_prob_table.
    Returns dict keyed ctx+(slot,) -> {T: prob}; the per-type probs sum to
    the combined table's prob for that (ctx, slot)."""
    matched = resp.dropna(subset=['responder_modal', 'responder_type'])
    KT = ['responder_modal', 'responder_type']

    fine_num = (matched.groupby(FINE_KEYS + KT).size().reset_index(name='f_n'))
    fine_num['f_total'] = [
        presence_fine.get((r.zx, r.zy, r.opp_type, r.phase, r.line_state,
                             r.responder_modal), 0)
        for r in fine_num.itertuples(index=False)]
    fine_num = fine_num[fine_num['f_total'] > 0].copy()
    fine_num['fine_prob'] = (fine_num['f_n'] / fine_num['f_total']).clip(upper=1.0)

    coarse_num = (matched.groupby(COARSE_KEYS + KT).size().reset_index(name='c_n'))
    coarse_num['c_total'] = [
        presence_coarse.get((r.zx, r.zy, r.opp_type, r.responder_modal), 0)
        for r in coarse_num.itertuples(index=False)]
    coarse_num = coarse_num[coarse_num['c_total'] > 0].copy()
    coarse_num['coarse_prob'] = (coarse_num['c_n'] / coarse_num['c_total']).clip(upper=1.0)

    grp = fine_num.merge(coarse_num[COARSE_KEYS + KT + ['coarse_prob']],
                           on=COARSE_KEYS + KT, how='left')
    grp['coarse_prob'] = grp['coarse_prob'].fillna(0.0)
    w = grp['f_total'] / (grp['f_total'] + PROB_SMOOTHING_K)
    grp['prob'] = w * grp['fine_prob'] + (1 - w) * grp['coarse_prob']
    out = {}
    for r in grp.itertuples(index=False):
        key = (r.zx, r.zy, r.opp_type, r.phase, r.line_state, r.responder_modal)
        out.setdefault(key, {})[r.responder_type] = r.prob
    return out


# ----- v11 — defensive QUALITY: wins above expectation (DWAE) ---------------
# DefR measures workload (actions above role expectation); DWAE measures
# QUALITY: of the contested engagements a player actually took on, how many
# did he win versus how many an average player would have won in the same
# spots? Success flags: defensive ground duel won := stoppedProgress OR
# recoveredPossession; defensive aerial won := firstTouch. Interceptions /
# recoveries / clearances are success-by-definition, so quality lives in
# the contested engagements only.
DWAE_K_FINE = 60.0     # EB shrink: (etype, oppgrp, zx, phase) -> (etype, oppgrp)
DWAE_K_MID = 200.0     # EB shrink: (etype, oppgrp) -> (etype)

# Opponent position-group for MATCHUP conditioning. Without it, DWAE
# carried a structural position gap (CB median +0.49/90 vs FB +0.02):
# CBs duel strikers, wingers duel fullbacks — different base win rates
# that aren't the defender's skill.
_OPP_GRP = {'GK': 'GK',
             'CB': 'CB', 'LCB': 'CB', 'RCB': 'CB', 'LCB3': 'CB', 'RCB3': 'CB',
             'LB': 'FB', 'RB': 'FB', 'LB5': 'FB', 'RB5': 'FB', 'LWB': 'FB', 'RWB': 'FB',
             'CMF': 'CM', 'LCMF': 'CM', 'RCMF': 'CM', 'LCMF3': 'CM', 'RCMF3': 'CM',
             'DMF': 'CM', 'LDMF': 'CM', 'RDMF': 'CM',
             'AMF': 'AM', 'LAMF': 'AM', 'RAMF': 'AM', 'LMF': 'AM', 'RMF': 'AM',
             'LW': 'AM', 'RW': 'AM', 'LWF': 'AM', 'RWF': 'AM',
             'CF': 'ST', 'SS': 'ST'}


def build_defensive_engagements(ev: pd.DataFrame) -> pd.DataFrame:
    """One row per contested defensive engagement with the EB-shrunk
    expected win probability p_win, conditioned on engagement type,
    OPPONENT position-group (matchup), zone depth and phase.
    DWAE = won − p_win sums ≈ 0 league-wide by construction and is
    volume-free (conditioned on the engagements actually contested)."""
    is_tackle = (ev['groundDuel.duelType'] == 'defensive_duel')
    eng = ev[is_tackle | ev['_is_def_aerial']].copy()
    eng['etype'] = np.where(eng['groundDuel.duelType'] == 'defensive_duel',
                              'tackle', 'aerial')
    won_tackle = ((eng['groundDuel.stoppedProgress'] == True)
                    | (eng['groundDuel.recoveredPossession'] == True))
    won_aerial = (eng['aerialDuel.firstTouch'] == True)
    eng['won'] = np.where(eng['etype'] == 'tackle',
                            won_tackle, won_aerial).astype(int)
    opp_pos = np.where(eng['etype'] == 'tackle',
                         eng['groundDuel.opponent.position'],
                         eng['aerialDuel.opponent.position'])
    eng['oppgrp'] = pd.Series(opp_pos, index=eng.index).map(_OPP_GRP).fillna('UNK')
    x = eng['location.x'].to_numpy(dtype=float)
    eng['zx'] = np.clip((x / (100.0 / N_ZONE_X)).astype(int), 0, N_ZONE_X - 1)
    eng['phase'] = eng['_phase']

    # EB-shrunk expected win rate: (etype,oppgrp,zx,phase) -> (etype,oppgrp) -> (etype)
    g1 = eng.groupby('etype')['won'].agg(['sum', 'size'])
    p1 = (g1['sum'] / g1['size']).to_dict()
    g2 = eng.groupby(['etype', 'oppgrp'])['won'].agg(['sum', 'size']).reset_index()
    g2['p1'] = g2['etype'].map(p1)
    g2['p2'] = (g2['sum'] + DWAE_K_MID * g2['p1']) / (g2['size'] + DWAE_K_MID)
    p2 = g2.set_index(['etype', 'oppgrp'])['p2'].to_dict()
    g4 = (eng.groupby(['etype', 'oppgrp', 'zx', 'phase'])['won']
             .agg(['sum', 'size']).reset_index())
    g4['p2'] = [p2[(e, o)] for e, o in zip(g4['etype'], g4['oppgrp'])]
    g4['p_win'] = (g4['sum'] + DWAE_K_FINE * g4['p2']) / (g4['size'] + DWAE_K_FINE)
    pw = g4.set_index(['etype', 'oppgrp', 'zx', 'phase'])['p_win'].to_dict()
    eng['p_win'] = [pw[(e, o, a, c)] for e, o, a, c in
                      zip(eng['etype'], eng['oppgrp'], eng['zx'], eng['phase'])]
    eng = eng.rename(columns={'player.id': 'playerId'})
    return eng[['playerId', 'seasonId', 'matchId', 'etype', 'oppgrp', 'won', 'p_win']]


# ----- Step 3 — Per-player expected + actual ------------------------------
def compute_per_player(ev: pd.DataFrame,
                         resp: pd.DataFrame,
                         prob_table: pd.DataFrame,
                         ptype_lookup: dict,
                         player_slot: pd.DataFrame,
                         psl: dict,
                         match_teams: dict,
                         def_team_players: dict,
                         pmt_lookup: dict) -> pd.DataFrame:
    """Compute expected + actual defensive actions per (player, season).
    Returns DataFrame with one row per (playerId, seasonId).

    v10 — probabilities are presence-conditioned P(slot responds | slot on
    pitch), so an opp event only adds expected for players actually on the
    pitch at that moment, and when 2+ players of the SAME slot are on
    simultaneously (subs/rotations — 15-21% of events for the main slots)
    the slot's expected is SPLIT between them instead of double-credited.
    Sum of expected across all players ≈ matched responses (calibrated)."""
    ev_with_slot = ev.copy()
    ev_with_slot['slot'] = ev_with_slot['player.position'].map(slot_of)

    prob_lookup = prob_table.set_index(
        FINE_KEYS + ['responder_slot'])['prob'].to_dict()
    expected_acc = {}        # (pid, sid) → sum of expected_p
    expected_type_acc = {}   # (pid, sid, type) → sum of per-type expected_p
    n_opp_acc = {}           # (pid, sid) → count of opposition events seen

    print(f"  Processing {len(resp):,} opposition events…", flush=True)
    R_mid = resp['matchId'].to_numpy()
    R_opp = resp['opp_team'].to_numpy()
    R_zx = resp['zx'].to_numpy()
    R_zy = resp['zy'].to_numpy()
    R_type = resp['opp_type'].to_numpy()
    R_phase = resp['phase'].to_numpy()
    R_line = resp['line_state'].to_numpy()
    R_t = resp['t'].to_numpy(dtype=float)
    for k in range(len(resp)):
        mid = R_mid[k]
        opp_team = R_opp[k]
        teams = match_teams.get(mid)
        if teams is None or len(teams) < 2:
            continue
        def_team = next((t for t in teams if t != opp_team), None)
        if def_team is None:
            continue
        t_event = R_t[k]
        ctx = (R_zx[k], R_zy[k], R_type[k], R_phase[k], R_line[k])
        # Who is on the pitch for the defending team right now, and at
        # which modal slot?
        onpitch = []
        for pid, sid in def_team_players.get((mid, def_team), []):
            t_in_out = pmt_lookup.get((pid, mid))
            if t_in_out is None:
                continue
            if t_event < t_in_out[0] or t_event > t_in_out[1]:
                continue
            slot = psl.get((pid, sid))
            if slot is None:
                continue
            onpitch.append((pid, sid, slot))
        if not onpitch:
            continue
        slot_cnt = {}
        for _, _, s in onpitch:
            slot_cnt[s] = slot_cnt.get(s, 0) + 1
        for pid, sid, slot in onpitch:
            share = slot_cnt[slot]
            p = prob_lookup.get(ctx + (slot,), 0.0)
            if p > 0:
                expected_acc[(pid, sid)] = expected_acc.get((pid, sid), 0) + p / share
            d = ptype_lookup.get(ctx + (slot,))
            if d:
                for T, pT in d.items():
                    kk = (pid, sid, T)
                    expected_type_acc[kk] = expected_type_acc.get(kk, 0) + pT / share
            n_opp_acc[(pid, sid)] = n_opp_acc.get((pid, sid), 0) + 1

    # Actual defensive actions per (player, season) — v7 precomputed mask
    # (interception | clearance | defensive duel | recovery).
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

    # v9 — per-type actual + expected + DefR (count level; per-90 added
    # after minutes attach).
    act_by_type = (ev_with_slot[ev_with_slot['_dtype'].notna()]
                     .groupby(['player.id', 'seasonId', '_dtype']).size()
                     .reset_index(name='n'))
    for T in DEFR_TYPES:
        sub = (act_by_type[act_by_type['_dtype'] == T]
                 .rename(columns={'player.id': 'playerId', 'n': f'act_{T}'})
                 [['playerId', 'seasonId', f'act_{T}']])
        out = out.merge(sub, on=['playerId', 'seasonId'], how='left')
        out[f'act_{T}'] = out[f'act_{T}'].fillna(0).astype(int)
    # expected per type from the accumulator
    exp_rows = {}
    for (pid, sid, T), v in expected_type_acc.items():
        exp_rows.setdefault((pid, sid), {})[T] = v
    for T in DEFR_TYPES:
        out[f'exp_{T}'] = [exp_rows.get((p, s), {}).get(T, 0.0)
                             for p, s in zip(out['playerId'], out['seasonId'])]
        out[f'defr_{T}'] = out[f'act_{T}'] - out[f'exp_{T}']

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
        _mp90 = out['mins_played'].fillna(0) / 90.0
        out['defr_per90'] = np.where(_mp90 > 0, out['defr'] / _mp90, np.nan)
        # v9 — per-type DefR per 90 (the dashboard display metrics)
        for T in DEFR_TYPES:
            out[f'defr_{T}_p90'] = np.where(
                _mp90 > 0, out[f'defr_{T}'] / _mp90, np.nan)
            out[f'act_{T}_p90'] = np.where(
                _mp90 > 0, out[f'act_{T}'] / _mp90, np.nan)
    except Exception as e:
        print(f"  [warn] could not attach minutes: {e}")
        out['mins_played'] = np.nan
        out['defr_per90'] = np.nan

    # DefR is an OUTFIELD metric — the "respond to opp pass/carry/shot"
    # model doesn't fit keepers. Drop GK rows entirely so DefR carries no
    # goalkeeper data.
    out = out[out['position'] != 'GK'].reset_index(drop=True)

    # v4 — position-fair DefR with minutes shrinkage.
    #
    # Stability analysis (year-over-year Pearson r on same-position pairs):
    #   defr_per90 (raw)            r = 0.745   ← stable but position-biased
    #   defr_per90_vs_position      r = 0.484   ← position-fair, noisier
    #   defr_z_vs_position (÷std)   r = 0.455   ← std-division HURTS — dropped
    #
    # The std-division in the old z-score amplified noise, so v4 drops it.
    # Instead we keep the position-median-subtracted value and SHRINK it
    # toward 0 by a minutes factor mins/(mins+K). Low-minute players (whose
    # DefR is noisiest) get pulled toward "typical for position"; full-
    # season players keep most of their signal. K=900 ≈ half a season.
    SHRINK_K = 900.0
    # v10 — LEAGUE-AWARE normalization. The validation battery measured a
    # +0.86/90 bias: with pooled position medians, Campeonato players sat
    # systematically above Liga 3 on the "position-fair" metric (Camp games
    # have more defensive events). Medians are now per (position, league),
    # falling back to the pooled median when a league cell has <8 players.
    SEASON_LEAGUE = {190230: 'CAMP', 191779: 'CAMP'}   # everything else L3
    out['league'] = out['seasonId'].map(
        lambda s: SEASON_LEAGUE.get(int(s), 'L3') if pd.notna(s) else 'L3')
    valid = out[(out['mins_played'].fillna(0) >= 800) & out['defr_per90'].notna()]
    med_lg = valid.groupby(['position', 'league'])['defr_per90'].median().to_dict()
    n_lg = valid.groupby(['position', 'league']).size().to_dict()
    med_pooled = valid.groupby('position')['defr_per90'].median().to_dict()

    def _pos_median(r):
        key = (r['position'], r['league'])
        if n_lg.get(key, 0) >= 8:
            return med_lg[key]
        return med_pooled.get(r['position'], 0)

    out['defr_per90_vs_position'] = out.apply(
        lambda r: (r['defr_per90'] - _pos_median(r))
                    if pd.notna(r['defr_per90']) else np.nan,
        axis=1,
    )
    # Minutes-shrunk position-fair DefR — single-season display metric.
    out['defr_adj'] = out.apply(
        lambda r: (r['defr_per90_vs_position']
                     * (r['mins_played'] / (r['mins_played'] + SHRINK_K)))
                    if pd.notna(r['defr_per90_vs_position'])
                    and pd.notna(r['mins_played']) else np.nan,
        axis=1,
    )

    # ===== RELIABILITY & VALIDATION FINDINGS (v10, measured) =====
    # Within-season split-half (odd/even matches, each half ≥450 min,
    # n=3,078), Spearman-Brown corrected to full-season:
    #     defr_per90 (raw)        r = 0.857
    #     defr_per90_vs_position  r = 0.847
    # Year-over-year (same-position consecutive pairs, n=326):
    #     defr_per90 / vs_position  r ≈ 0.49 / 0.46
    # NOTE: v9 showed higher reliabilities (0.89-0.91 within-season) but
    # that stability was partly BORROWED from raw volume — v9's expected
    # was under-scaled (~17% of actual) due to formation dilution, so
    # DefR ≈ actual volume in disguise. v10's calibrated expected
    # (Σ expected ≈ matched responses, ratio 0.955) measures the real
    # over/under-performance signal; ~0.85 is the honest reliability.
    #
    # Validation battery (see git history):
    #   - LOSO out-of-sample: prob table w/o a season ranks it identically
    #     (Spearman 0.9999) — no overfitting.
    #   - Window sensitivity: 2s vs 3s rank corr 0.94; 5s vs 3s 0.85.
    #   - Discriminant: ~zero corr with all offensive GPA values; +0.02
    #     with Interrupting Value (DefR = workload, not value — they are
    #     complementary, not redundant).
    #
    # The within-vs-YoY gap is GENUINE system-dependence: defensive
    # responsibility changes with the team's pressing scheme / block
    # height / role. DefR describes the player in his CURRENT system.
    #
    # Implication for the career aggregate: a flat minutes-weighted mean
    # across seasons blends incompatible systems. We RECENCY-WEIGHT so the
    # player's current/most-recent system dominates, with older seasons as
    # a mild prior — mirrors the CVI career-decay approach.
    CAREER_DECAY = 0.5   # weight = mins × decay^(seasons_back)
    SEASON_YEAR = {188221: 2021, 188222: 2022, 189147: 2023, 190090: 2024,
                    191782: 2025, 190230: 2023, 191779: 2025}
    qual = out[(out['mins_played'].fillna(0) >= 500)
                 & out['defr_per90_vs_position'].notna()].copy()
    qual['_yr'] = qual['seasonId'].map(SEASON_YEAR)
    qual = qual.dropna(subset=['_yr'])
    car_rows = []
    for pid, g in qual.groupby('playerId'):
        latest = g['_yr'].max()
        w = (g['mins_played'].to_numpy()
              * (CAREER_DECAY ** (latest - g['_yr'].to_numpy())))
        v = g['defr_per90_vs_position'].to_numpy()
        if w.sum() > 0:
            car_rows.append({'playerId': pid,
                              'defr_career': float((v * w).sum() / w.sum()),
                              'n_seasons': int(g['seasonId'].nunique())})
    car = pd.DataFrame(car_rows)
    out = out.merge(car, on='playerId', how='left')

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

    print("\n[3/5] Modal slots, on-pitch lookups, slot-presence scan…",
            flush=True)
    player_slot, psl = compute_modal_slots(ev)
    match_teams, def_team_players, pmt_lookup = match_lookups(ev)
    # v10 — numerators use the responder's seasonal MODAL slot (the
    # event-moment position disagrees with it on 33% of matched rows).
    resp['responder_modal'] = [
        psl.get((int(p), s)) if pd.notna(p) else None
        for p, s in zip(resp['responder_pid'], resp['seasonId'])]
    presence_fine, presence_coarse = build_presence_tables(
        resp, match_teams, def_team_players, pmt_lookup, psl)
    print(f"  Presence tables: {len(presence_fine):,} fine cells.")

    print("\n[4/5] Estimating presence-conditioned P(slot responds | ctx)…",
            flush=True)
    prob = build_prob_table(resp, presence_fine, presence_coarse)
    ptype_lookup = build_prob_table_by_type(resp, presence_fine, presence_coarse)
    print(f"  Probability table: {len(prob):,} bucket × slot combinations.")

    print("\n[5/5] Computing per-(player, season) expected + actual…", flush=True)
    out = compute_per_player(ev, resp, prob, ptype_lookup, player_slot, psl,
                               match_teams, def_team_players, pmt_lookup)
    print(f"  {len(out):,} (player, season) rows produced.")

    # Calibration: with presence-conditioned probs + concurrent splitting,
    # the sum of expected across all players should ≈ matched responses.
    total_exp = float(out['expected_def_actions'].sum())
    total_matched = int(resp['responder_pid'].notna().sum())
    print(f"  Calibration: Σ expected = {total_exp:,.0f} vs matched responses "
           f"= {total_matched:,} (ratio {total_exp/total_matched:.3f}; "
           f"GK rows dropped from output keep ratio slightly under 1)")

    print("\n[+] Defensive quality — wins above expectation (DWAE)…", flush=True)
    eng = build_defensive_engagements(ev)
    eng.to_parquet(HERE / 'defensive_engagements.parquet')
    agg = (eng.groupby(['playerId', 'seasonId'])
              .agg(dwae_n=('won', 'size'), dwae_wins=('won', 'sum'),
                    dwae_exp=('p_win', 'sum')).reset_index())
    agg['defr_dwae'] = agg['dwae_wins'] - agg['dwae_exp']
    per_t = (eng.assign(_d=eng['won'] - eng['p_win'])
                .groupby(['playerId', 'seasonId', 'etype'])['_d']
                .sum().unstack(fill_value=0.0))
    per_t.columns = [f'defr_dwae_{c}' for c in per_t.columns]
    agg = agg.merge(per_t.reset_index(), on=['playerId', 'seasonId'], how='left')
    agg['playerId'] = agg['playerId'].astype('Int64')
    agg['seasonId'] = pd.to_numeric(agg['seasonId'], errors='coerce').astype('Int64')
    out = out.merge(agg, on=['playerId', 'seasonId'], how='left')
    _m90 = out['mins_played'].fillna(0) / 90.0
    for c in ['defr_dwae', 'defr_dwae_tackle', 'defr_dwae_aerial']:
        if c in out.columns:
            out[f'{c}_p90'] = np.where(_m90 > 0, out[c] / _m90, np.nan)
    print(f"  {len(eng):,} engagements; league DWAE sum = "
           f"{agg['defr_dwae'].sum():+.1f} (≈0 by construction)")

    out_path = HERE / 'defr_per_player_season.parquet'
    out.to_parquet(out_path)
    print(f"\n✅ Saved {out_path}")

    # Sample output — sort by minutes-shrunk position-fair DefR (defr_adj)
    cols = ['playerId', 'seasonId', 'position', 'mins_played',
             'expected_def_actions', 'actual_def_actions',
             'defr_per90', 'defr_per90_vs_position', 'defr_adj']
    print("\nSample — top 15 by defr_adj (min 1000 mins):")
    print(out[out['mins_played'] >= 1000].sort_values(
        'defr_adj', ascending=False).head(15)[cols].round(2).to_string(index=False))

    print("\nSample — bottom 15 by defr_adj (min 1000 mins):")
    print(out[out['mins_played'] >= 1000].sort_values(
        'defr_adj', ascending=True).head(15)[cols].round(2).to_string(index=False))


if __name__ == '__main__':
    main()
