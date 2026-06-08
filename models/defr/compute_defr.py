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
    'shot.onTarget', 'shot.isGoal', 'shot.postShotXg',
    'shot.goalkeeper.id', 'competitionId',
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


def build_prob_table(resp: pd.DataFrame) -> pd.DataFrame:
    """v3 — probability of each responder slot given the FULL context
    (zone × opp_type × phase × line_state), with empirical-Bayes backoff
    toward the coarse (zone × opp_type) probability so sparse fine
    buckets don't produce noisy estimates.

    Returns one row per (FINE_KEYS, responder_slot) with column
    'prob' = the shrunk unconditional P(this slot responds | context).
    The unconditional framing (denominator = ALL opp events in the
    bucket, ~70% of which get no response) keeps expected calibrated."""
    matched = resp.dropna(subset=['responder_slot'])

    # --- Coarse layer: P(slot | zone, type) ---
    coarse_num = (matched.groupby(COARSE_KEYS + ['responder_slot'])
                     .size().reset_index(name='c_n'))
    coarse_den = (resp.groupby(COARSE_KEYS).size()
                     .reset_index(name='c_total'))
    coarse = coarse_num.merge(coarse_den, on=COARSE_KEYS)
    coarse['coarse_prob'] = coarse['c_n'] / coarse['c_total']

    # --- Fine layer: P(slot | zone, type, phase, line) ---
    fine_num = (matched.groupby(FINE_KEYS + ['responder_slot'])
                   .size().reset_index(name='f_n'))
    fine_den = (resp.groupby(FINE_KEYS).size()
                   .reset_index(name='f_total'))
    fine = fine_num.merge(fine_den, on=FINE_KEYS)
    fine['fine_prob'] = fine['f_n'] / fine['f_total']

    # --- Backoff blend ---
    grp = fine.merge(coarse[COARSE_KEYS + ['responder_slot', 'coarse_prob']],
                       on=COARSE_KEYS + ['responder_slot'], how='left')
    grp['coarse_prob'] = grp['coarse_prob'].fillna(0.0)
    w = grp['f_total'] / (grp['f_total'] + PROB_SMOOTHING_K)
    grp['prob'] = w * grp['fine_prob'] + (1 - w) * grp['coarse_prob']
    return grp


# v9 — the five mutually-exclusive defensive action types we score
# per-type DefR for. Names match _dtype values.
DEFR_TYPES = ['interception', 'clearance', 'tackle', 'recovery', 'def_aerial']


def build_prob_table_by_type(resp: pd.DataFrame) -> dict:
    """P(responder_slot responds WITH action-type T | full context), same
    empirical-Bayes backoff as build_prob_table but split by the
    responder's action type. Returns a dict keyed
    (zx, zy, opp_type, phase, line_state, slot) -> {T: prob} so the
    expected loop can add each type's contribution in one pass."""
    matched = resp.dropna(subset=['responder_slot', 'responder_type'])
    KT = ['responder_slot', 'responder_type']
    coarse_num = (matched.groupby(COARSE_KEYS + KT).size()
                     .reset_index(name='c_n'))
    coarse_den = resp.groupby(COARSE_KEYS).size().reset_index(name='c_total')
    coarse = coarse_num.merge(coarse_den, on=COARSE_KEYS)
    coarse['coarse_prob'] = coarse['c_n'] / coarse['c_total']
    fine_num = (matched.groupby(FINE_KEYS + KT).size()
                   .reset_index(name='f_n'))
    fine_den = resp.groupby(FINE_KEYS).size().reset_index(name='f_total')
    fine = fine_num.merge(fine_den, on=FINE_KEYS)
    fine['fine_prob'] = fine['f_n'] / fine['f_total']
    grp = fine.merge(coarse[COARSE_KEYS + KT + ['coarse_prob']],
                       on=COARSE_KEYS + KT, how='left')
    grp['coarse_prob'] = grp['coarse_prob'].fillna(0.0)
    w = grp['f_total'] / (grp['f_total'] + PROB_SMOOTHING_K)
    grp['prob'] = w * grp['fine_prob'] + (1 - w) * grp['coarse_prob']
    out = {}
    for r in grp.itertuples(index=False):
        key = (r.zx, r.zy, r.opp_type, r.phase, r.line_state, r.responder_slot)
        out.setdefault(key, {})[r.responder_type] = r.prob
    return out


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

    # Index prob_table for fast lookup — full v3 key
    # (zone × type × phase × line_state × slot)
    prob_lookup = prob_table.set_index(
        FINE_KEYS + ['responder_slot'])['prob'].to_dict()
    # v9 — per-type expected probabilities, keyed (ctx, slot) -> {type: p}
    ptype_lookup = build_prob_table_by_type(resp)
    expected_type_acc = {}   # (pid, sid, type) → sum of per-type expected_p

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
    # Vectorized column access (numpy) — ~10x faster than .iterrows()
    R_mid = resp['matchId'].to_numpy()
    R_opp = resp['opp_team'].to_numpy()
    R_zx = resp['zx'].to_numpy()
    R_zy = resp['zy'].to_numpy()
    R_type = resp['opp_type'].to_numpy()
    R_phase = resp['phase'].to_numpy()
    R_line = resp['line_state'].to_numpy()
    R_t = resp['t'].to_numpy(dtype=float)
    psl = player_slot_lookup.to_dict()
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
        for pid, sid in def_team_players.get((mid, def_team), []):
            t_in_out = pmt_lookup.get((pid, mid))
            if t_in_out is None:
                continue
            t_in, t_out = t_in_out
            if t_event < t_in or t_event > t_out:
                continue
            slot = psl.get((pid, sid))
            if slot is None:
                continue
            p = prob_lookup.get(ctx + (slot,), 0.0)
            if p > 0:
                expected_acc[(pid, sid)] = expected_acc.get((pid, sid), 0) + p
            # v9 — per-type expected
            d = ptype_lookup.get(ctx + (slot,))
            if d:
                for T, pT in d.items():
                    kk = (pid, sid, T)
                    expected_type_acc[kk] = expected_type_acc.get(kk, 0) + pT
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

    # v6 — GK shot-stopping DefR. The outfield "respond to opp pass/carry"
    # model doesn't fit keepers (they barely register). A GK's real
    # defensive responsibility is shot-stopping, and the DefR concept maps
    # cleanly: expected = post-shot xG of shots faced (goals an average
    # keeper concedes), actual prevention = expected − goals conceded.
    #   gk_goals_prevented = psxg_faced − goals_conceded   (+ = good keeper)
    # For GK rows we REPLACE defr_per90 with goals_prevented/90 so the
    # downstream position-normalization treats shot-stopping as their DefR.
    gk = ev[(ev['type.primary'] == 'shot')
              & (ev['shot.onTarget'] == True)
              & ev['shot.goalkeeper.id'].notna()].copy()
    if not gk.empty:
        gk['gk_id'] = gk['shot.goalkeeper.id'].astype(int)
        gk['psxg'] = pd.to_numeric(gk['shot.postShotXg'], errors='coerce').fillna(0)
        gk['goal'] = gk['shot.isGoal'].fillna(False).astype(int)
        gk['seasonId'] = pd.to_numeric(gk['seasonId'], errors='coerce').astype('Int64')
        gks = (gk.groupby(['gk_id', 'seasonId'])
                  .agg(gk_shots_faced=('goal', 'size'),
                        gk_psxg_faced=('psxg', 'sum'),
                        gk_goals_conceded=('goal', 'sum'))
                  .reset_index()
                  .rename(columns={'gk_id': 'playerId'}))
        gks['gk_goals_prevented'] = gks['gk_psxg_faced'] - gks['gk_goals_conceded']
        gks['playerId'] = gks['playerId'].astype('Int64')
        out = out.merge(gks, on=['playerId', 'seasonId'], how='left')
        # For GK rows: override defr_per90 with goals-prevented per 90.
        is_gk = (out['position'] == 'GK') & out['gk_goals_prevented'].notna()
        out.loc[is_gk, 'defr_per90'] = np.where(
            out.loc[is_gk, 'mins_played'].fillna(0) > 0,
            out.loc[is_gk, 'gk_goals_prevented'] / (out.loc[is_gk, 'mins_played'] / 90.0),
            np.nan)
        out['gk_gp_per90'] = np.where(
            (out['position'] == 'GK') & (out['mins_played'].fillna(0) > 0),
            out['gk_goals_prevented'] / (out['mins_played'] / 90.0), np.nan)

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
    valid = out[(out['mins_played'].fillna(0) >= 800) & out['defr_per90'].notna()]
    pos_median = valid.groupby('position')['defr_per90'].median().to_dict()
    out['defr_per90_vs_position'] = out.apply(
        lambda r: (r['defr_per90'] - pos_median.get(r['position'], 0))
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

    # ===== RELIABILITY FINDINGS (measured, not assumed) =====
    # Within-season split-half (odd vs even matches, each half ≥400 min,
    # n=3,587 player-seasons), Spearman-Brown corrected to full-season:
    #     defr_per90 (raw)        r = 0.871
    #     defr_per90_vs_position  r = 0.818   ← EXCELLENT within-season
    # Year-over-year (same-position consecutive pairs, n=383):
    #     defr_per90 (raw)        r = 0.75
    #     defr_per90_vs_position  r = 0.49
    #
    # The gap (0.82 within-season vs 0.49 YoY) is NOT measurement noise —
    # within a season the metric is highly reliable. The YoY drop is
    # GENUINE system-dependence: defensive responsibility changes with
    # the team's pressing scheme / block height / role. So DefR describes
    # "how this player performs defensive responsibility in his CURRENT
    # system", and is expected to shift if he moves to a very different one.
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
