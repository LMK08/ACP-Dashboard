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
import sys as _sys_lc
_sys_lc.path.insert(0, str(Path(__file__).resolve().parents[2]))  # league_config
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DASHBOARD_DIR = HERE.parent.parent

# v7.0 lookahead test hook (as in build_rating.py): ACP_MAX_YEAR drops every
# season after that year from the events, ACP_DEFR_OUT redirects the outputs.
import os as _os
_MAX_YR = int(_os.environ['ACP_MAX_YEAR']) if _os.environ.get('ACP_MAX_YEAR') else None
_OUT = Path(_os.environ.get('ACP_DEFR_OUT', str(HERE)))

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
    from league_config import SEASON_YEAR
    df['_yr'] = df['seasonId'].map(lambda v: SEASON_YEAR.get(int(v)) if pd.notna(v) else np.nan)
    if _MAX_YR is not None:
        df = df[df['_yr'] <= _MAX_YR].copy()
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
        # stable: same-time events keep Wyscout's file order. The default
        # quicksort orders ties differently on x86 CI and ARM laptops, and
        # the forward responder scan below depends on that order.
        mdf = mdf.sort_values('_t', kind='stable').reset_index(drop=True)
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
        id_arr = mdf['id'].to_numpy()                # event id (DefR-OBV value join)
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
                'event_id': id_arr[i],
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
                     # ties broken by slot name: an unstable sort let LATER
                     # seasons' rows decide a tied season (v7.0 lookahead check)
                     .sort_values(['n', 'slot'], ascending=[False, True], kind='stable')
                     .drop_duplicates(['player.id', 'seasonId'])
                     [['player.id', 'seasonId', 'slot']])
    psl = player_slot.set_index(['player.id', 'seasonId'])['slot'].to_dict()
    return player_slot, psl


def compute_match_slots(ev: pd.DataFrame, match_dates: dict | None = None) -> dict:
    """(player, match) -> the slot he played IN THAT MATCH: the most frequent
    slot of his events in it (ties by slot name) — his position in that
    match's line-up, known at kick-off. v7.1 NO LOOKAHEAD (2026-10-29): v10
    used his whole-SEASON modal slot, so an October event placed him with
    positions from later matches. A match where none of his events carries a
    position: the slot of his latest EARLIER match (kick-off order from
    match_dates), else none."""
    e = ev[['player.id', 'matchId', 'player.position']].dropna(subset=['player.id']).copy()
    e['slot'] = e['player.position'].map(slot_of)
    cnt = (e.dropna(subset=['slot']).groupby(['player.id', 'matchId', 'slot'])
             .size().reset_index(name='n')
             .sort_values(['n', 'slot'], ascending=[False, True], kind='stable')
             .drop_duplicates(['player.id', 'matchId']))
    msl = {(int(p), m): sl for p, m, sl in zip(cnt['player.id'], cnt['matchId'], cnt['slot'])}
    if match_dates:
        pm = e[['player.id', 'matchId']].drop_duplicates()
        pm = pm.assign(_d=pm['matchId'].map(match_dates)).dropna(subset=['_d'])
        for pid, g in pm.sort_values(['player.id', '_d', 'matchId'], kind='stable').groupby('player.id'):
            last = None
            for m in g['matchId']:
                k = (int(pid), m)
                if k in msl:
                    last = msl[k]
                elif last is not None:
                    msl[k] = last
    return msl


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


def build_presence_tables(resp, match_teams, def_team_players, pmt_lookup, msl):
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
            s = msl.get((pid, mid))
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


def dwae_win_rates(eng: pd.DataFrame) -> pd.Series:
    """Expected win rate of every engagement (aligned to eng.index) from
    engagements of EARLIER season-years only (v7.1); the first year: NaN.
    eng: etype, oppgrp, zx, phase, won, _yr."""
    out = pd.Series(np.nan, index=eng.index)
    for y in sorted(eng['_yr'].dropna().unique()):
        h = eng[eng['_yr'] < y]
        if h.empty:
            continue
        g1 = h.groupby('etype')['won'].agg(['sum', 'size'])
        p1 = (g1['sum'] / g1['size']).to_dict()
        g2 = h.groupby(['etype', 'oppgrp'])['won'].agg(['sum', 'size']).reset_index()
        g2['p1'] = g2['etype'].map(p1)
        g2['p2'] = (g2['sum'] + DWAE_K_MID * g2['p1']) / (g2['size'] + DWAE_K_MID)
        p2 = g2.set_index(['etype', 'oppgrp'])['p2'].to_dict()
        g4 = (h.groupby(['etype', 'oppgrp', 'zx', 'phase'])['won']
                 .agg(['sum', 'size']).reset_index())
        g4['p2'] = [p2[(e, o)] for e, o in zip(g4['etype'], g4['oppgrp'])]
        g4['p_win'] = (g4['sum'] + DWAE_K_FINE * g4['p2']) / (g4['size'] + DWAE_K_FINE)
        pw = g4.set_index(['etype', 'oppgrp', 'zx', 'phase'])['p_win'].to_dict()
        # a cell never seen before: its (etype, oppgrp) rate, then its etype rate
        rows = eng['_yr'] == y
        out.loc[rows] = [pw.get((e, o, a, c), p2.get((e, o), p1.get(e, np.nan))) for e, o, a, c in zip(
            eng.loc[rows, 'etype'], eng.loc[rows, 'oppgrp'],
            eng.loc[rows, 'zx'], eng.loc[rows, 'phase'])]
    return out


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
    # v7.1 NO LOOKAHEAD: the rates for a season-year come from engagements in
    # EARLIER years only (v7.0 counted the year itself: an engagement's own
    # result and the season's later ones). The first year has none: NaN.
    eng['p_win'] = dwae_win_rates(eng)
    eng = eng.rename(columns={'player.id': 'playerId'})
    return eng[['playerId', 'seasonId', 'matchId', 'etype', 'oppgrp', 'won', 'p_win']]


# ----- Step 3 — Per-player expected + actual ------------------------------
def compute_per_player(ev: pd.DataFrame,
                         resp: pd.DataFrame,
                         prob_table: pd.DataFrame,
                         ptype_lookup: dict,
                         player_slot: pd.DataFrame,
                         msl: dict,
                         match_teams: dict,
                         def_team_players: dict,
                         pmt_lookup: dict,
                         value_map: dict | None = None,
                         base_resp: pd.DataFrame | None = None) -> pd.DataFrame:
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
    # DefR-OBV (StatsBomb Hudl blog): distribute each opposition event's
    # on-ball value (GPA action_value — our OBV analog) across the on-pitch
    # defenders PROPORTIONALLY to their responsibility weights, so the
    # event's full value is conserved (Σ weights = 1 per event). Net values:
    # a failed opposition action carries negative value, crediting the
    # defenders responsible for that zone — suppression is rewarded.
    obv_acc = {}             # (pid, sid) → Σ w_i × value  (net conceded)
    obv_pos_acc = {}         # (pid, sid) → positive-value-only variant
    obv_szone_acc = {}       # (pid, sid) → Σ w_i × (value − E[value|ZONE]) —
                             #   zone-only adjustment (Lucas): normalizes the
                             #   danger of WHERE the domain is, keeps credit
                             #   for the rest.
    obv_supp_acc = {}        # (pid, sid) → Σ w_i × (value − E[value|ctx]):
                             #   SUPPRESSION vs expectation — negative = the
                             #   opposition generated LESS through this
                             #   player's domain than the same situations
                             #   yield league-wide (exposure-neutralized).
    _obv_events = 0          # coverage: events distributed
    _obv_total = 0.0         # total value distributed (calibration)

    print(f"  Processing {len(resp):,} opposition events…", flush=True)
    R_mid = resp['matchId'].to_numpy()
    R_opp = resp['opp_team'].to_numpy()
    R_zx = resp['zx'].to_numpy()
    R_zy = resp['zy'].to_numpy()
    R_type = resp['opp_type'].to_numpy()
    R_phase = resp['phase'].to_numpy()
    R_line = resp['line_state'].to_numpy()
    R_t = resp['t'].to_numpy(dtype=float)
    if value_map is not None and 'event_id' in resp.columns:
        R_val = resp['event_id'].map(value_map).fillna(0.0).to_numpy(dtype=float)
    else:
        R_val = np.zeros(len(resp))
    # context-mean opposition value E[V | zone x type x phase x line] over the
    # matched (nonzero) events — the exposure baseline for the suppression
    # variant. v7.1 NO LOOKAHEAD: from EARLIER seasons' events (base_resp), not
    # this season's (which held the event itself and everything after it).
    _b = base_resp if base_resp is not None else resp.iloc[0:0]
    if value_map is not None and 'event_id' in _b.columns:
        _bval = _b['event_id'].map(value_map).fillna(0.0).to_numpy(dtype=float)
    else:
        _bval = np.zeros(len(_b))
    _vmask = _bval != 0.0
    if _vmask.any():
        _ctxdf = pd.DataFrame({'zx': _b['zx'].to_numpy()[_vmask], 'zy': _b['zy'].to_numpy()[_vmask],
                                'ot': _b['opp_type'].to_numpy()[_vmask], 'ph': _b['phase'].to_numpy()[_vmask],
                                'ls': _b['line_state'].to_numpy()[_vmask], 'v': _bval[_vmask]})
        _ev_ctx = _ctxdf.groupby(['zx', 'zy', 'ot', 'ph', 'ls'])['v'].mean().to_dict()
        # zone-only baseline (Lucas): adjust for the AREA of responsibility
        # only — E[V | zone] — crediting the defender for everything else
        # (event type mix, phase, line state) that happens in his domain.
        _ev_zone = _ctxdf.groupby(['zx', 'zy'])['v'].mean().to_dict()
        _ev_global = float(_ctxdf['v'].mean())
    else:
        _ev_ctx, _ev_zone, _ev_global = {}, {}, 0.0
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
            slot = msl.get((pid, mid))
            if slot is None:
                continue
            onpitch.append((pid, sid, slot))
        if not onpitch:
            continue
        slot_cnt = {}
        for _, _, s in onpitch:
            slot_cnt[s] = slot_cnt.get(s, 0) + 1
        _resp_ws = []          # (pid, sid, responsibility weight) this event
        for pid, sid, slot in onpitch:
            share = slot_cnt[slot]
            p = prob_lookup.get(ctx + (slot,), 0.0)
            if p > 0:
                expected_acc[(pid, sid)] = expected_acc.get((pid, sid), 0) + p / share
                _resp_ws.append((pid, sid, p / share))
            d = ptype_lookup.get(ctx + (slot,))
            if d:
                for T, pT in d.items():
                    kk = (pid, sid, T)
                    expected_type_acc[kk] = expected_type_acc.get(kk, 0) + pT / share
            n_opp_acc[(pid, sid)] = n_opp_acc.get((pid, sid), 0) + 1
        # DefR-OBV distribution: normalize responsibilities to sum 1, then
        # hand each defender his share of this event's on-ball value.
        v = R_val[k]
        if v != 0.0 and _resp_ws:
            tot_w = sum(w for _, _, w in _resp_ws)
            if tot_w > 0:
                _obv_events += 1
                _obv_total += v
                ev_c = _ev_ctx.get((R_zx[k], R_zy[k], R_type[k],
                                     R_phase[k], R_line[k]), _ev_global)
                ev_z = _ev_zone.get((R_zx[k], R_zy[k]), _ev_global)
                for pid, sid, w in _resp_ws:
                    kk = (pid, sid)
                    wn = w / tot_w
                    obv_acc[kk] = obv_acc.get(kk, 0.0) + wn * v
                    obv_supp_acc[kk] = obv_supp_acc.get(kk, 0.0) + wn * (v - ev_c)
                    obv_szone_acc[kk] = obv_szone_acc.get(kk, 0.0) + wn * (v - ev_z)
                    if v > 0:
                        obv_pos_acc[kk] = obv_pos_acc.get(kk, 0.0) + wn * v

    # Actual defensive actions per (player, season) — v7 precomputed mask
    # (interception | clearance | defensive duel | recovery).
    actual = (ev_with_slot[ev_with_slot['_is_def']]
                .groupby(['player.id', 'seasonId']).size()
                .reset_index(name='actual_def_actions')
                .rename(columns={'player.id': 'playerId'}))

    # Build output
    if _obv_events:
        _dist = sum(obv_acc.values())
        print(f"  DefR-OBV: distributed {_obv_events:,} valued opposition events; "
               f"Σ value {_obv_total:+.1f} vs Σ player-conceded {_dist:+.1f} "
               f"(conservation Δ {abs(_obv_total-_dist):.4f})", flush=True)
    rows = []
    for (pid, sid), exp_v in expected_acc.items():
        rows.append({
            'playerId': pid,
            'seasonId': sid,
            'expected_def_actions': exp_v,
            'n_opp_events': n_opp_acc.get((pid, sid), 0),
            'obv_conceded': obv_acc.get((pid, sid), 0.0),
            'obv_conceded_pos': obv_pos_acc.get((pid, sid), 0.0),
            'obv_suppression': obv_supp_acc.get((pid, sid), 0.0),
            'obv_supp_zone': obv_szone_acc.get((pid, sid), 0.0),
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
        # DefR-OBV per 90 (responsibility-weighted opposition value through
        # the player's defensive domain; LOWER = more suppressive)
        out['obv_conceded_p90'] = np.where(_mp90 > 0, out['obv_conceded'] / _mp90, np.nan)
        out['obv_conceded_pos_p90'] = np.where(_mp90 > 0, out['obv_conceded_pos'] / _mp90, np.nan)
        out['obv_suppression_p90'] = np.where(_mp90 > 0, out['obv_suppression'] / _mp90, np.nan)
        out['obv_supp_zone_p90'] = np.where(_mp90 > 0, out['obv_supp_zone'] / _mp90, np.nan)
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

    return out


def finalize_defr(out: pd.DataFrame) -> pd.DataFrame:
    """Position-fair DefR (defr_per90_vs_position, defr_adj) and the career
    aggregate. v7.0 NO LOOKAHEAD: a season-year's position medians come from
    rows of that year and earlier only, and defr_career / n_seasons are the
    career AS OF that season (v6.9 pooled the medians over every season and
    merged one all-seasons career value onto every row)."""
    from league_config import SEASON_YEAR  # every season, incl. the current one
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
    out['_yr'] = out['seasonId'].map(
        lambda s: SEASON_YEAR.get(int(s)) if pd.notna(s) else np.nan)
    out['defr_per90_vs_position'] = np.nan
    for y in sorted(out['_yr'].dropna().unique()):
        hist = out[out['_yr'] <= y]
        valid = hist[(hist['mins_played'].fillna(0) >= 800) & hist['defr_per90'].notna()]
        med_lg = valid.groupby(['position', 'league'])['defr_per90'].median().to_dict()
        n_lg = valid.groupby(['position', 'league']).size().to_dict()
        med_pooled = valid.groupby('position')['defr_per90'].median().to_dict()

        def _pos_median(pos, lg):
            if n_lg.get((pos, lg), 0) >= 8:
                return med_lg[(pos, lg)]
            return med_pooled.get(pos, 0)

        rows = out['_yr'] == y
        out.loc[rows, 'defr_per90_vs_position'] = [
            (v - _pos_median(p, lg)) if pd.notna(v) else np.nan
            for v, p, lg in zip(out.loc[rows, 'defr_per90'], out.loc[rows, 'position'],
                                out.loc[rows, 'league'])]
    # Minutes-shrunk position-fair DefR — single-season display metric.
    _ok = out['defr_per90_vs_position'].notna() & out['mins_played'].notna()
    out['defr_adj'] = np.where(
        _ok, out['defr_per90_vs_position'] * (out['mins_played'] / (out['mins_played'] + SHRINK_K)),
        np.nan)

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
    qual = out[(out['mins_played'].fillna(0) >= 500)
                 & out['defr_per90_vs_position'].notna()].dropna(subset=['_yr'])
    hist = {pid: g for pid, g in qual.groupby('playerId')}
    car, nse = [], []
    for pid, y in zip(out['playerId'], out['_yr']):
        g = hist.get(pid)
        g = g[g['_yr'] <= y] if g is not None and y == y else None
        if g is None or g.empty:
            car.append(np.nan); nse.append(np.nan); continue
        w = g['mins_played'].to_numpy() * (CAREER_DECAY ** (y - g['_yr'].to_numpy()))
        car.append(float((g['defr_per90_vs_position'].to_numpy() * w).sum() / w.sum())
                   if w.sum() > 0 else np.nan)
        nse.append(int(g['seasonId'].nunique()))
    out['defr_career'] = car
    out['n_seasons'] = nse
    return out.drop(columns=['_yr'])


def asof_tables(resp, pres_by_season, years):
    """{y: (prob table, per-type lookup)} for each season-year y from the
    responses and presence counts of EARLIER years only; None when there is
    no earlier year (v7.1, 2026-10-29)."""
    from league_config import SEASON_YEAR
    out = {}
    for y in years:
        fine, coarse = {}, {}
        for sid, (f_s, c_s) in pres_by_season.items():
            if SEASON_YEAR.get(int(sid), 9999) < y:
                for k, v in f_s.items():
                    fine[k] = fine.get(k, 0) + v
                for k, v in c_s.items():
                    coarse[k] = coarse.get(k, 0) + v
        r_y = resp[resp['_yr'] < y]
        out[y] = (None if r_y.empty else
                  (build_prob_table(r_y, fine, coarse), build_prob_table_by_type(r_y, fine, coarse)))
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
    # the season's modal slot stays the row's descriptive position (and the
    # position medians' key); events use the slot of THEIR match (v7.1)
    player_slot, psl = compute_modal_slots(ev)
    _msum = pd.read_parquet(DASHBOARD_DIR / 'matches_summary.parquet', columns=['matchId', 'dateutc'])
    match_dates = dict(zip(_msum['matchId'], pd.to_datetime(_msum['dateutc'], errors='coerce')))
    msl = compute_match_slots(ev, match_dates)
    match_teams, def_team_players, pmt_lookup = match_lookups(ev)
    # v10 — numerators use the responder's MODAL slot (the event-moment
    # position disagrees with it on 33% of matched rows); v7.1: his modal
    # slot IN THAT MATCH, not over the whole season.
    resp['responder_modal'] = [
        msl.get((int(p), m)) if pd.notna(p) else None
        for p, m in zip(resp['responder_pid'], resp['matchId'])]
    # v7.1 NO LOOKAHEAD: P(slot responds | ctx) for a season-year is
    # estimated from EARLIER years only (v7.0 counted the year itself, so an
    # October event's expectation held that season's later events and its
    # own response). The first year has nothing earlier: no DefR for it.
    # Presence counts are additive, so they are built once per season and
    # summed per year.
    from league_config import SEASON_YEAR
    resp['_yr'] = resp['seasonId'].map(
        lambda v: SEASON_YEAR.get(int(v)) if pd.notna(v) else np.nan)
    pres_by_season = {}
    for sid, rs in resp.groupby('seasonId'):
        pres_by_season[sid] = build_presence_tables(
            rs, match_teams, def_team_players, pmt_lookup, msl)
    print(f"  Presence tables built for {len(pres_by_season)} seasons.")

    print("\n[4/5] Estimating presence-conditioned P(slot responds | ctx), "
          "as of each season-year…", flush=True)
    years = sorted(resp['_yr'].dropna().unique())
    tables = asof_tables(resp, pres_by_season, years)
    for y in years:
        print(f"  {int(y)}: " + ("no earlier season — no DefR" if tables[y] is None
                                 else f"{len(tables[y][0]):,} bucket × slot combinations."))

    print("\n[5/5] Computing per-(player, season) expected + actual…", flush=True)
    # DefR-OBV value source: per-event GPA action values (our OBV analog).
    # Join by event id; only opposition pass/carry/shot events in `resp`
    # will hit the map.
    print("  Loading GPA action values for DefR-OBV…", flush=True)
    try:
        _av = pd.read_parquet(DASHBOARD_DIR.parent.parent / 'GPA Model Project v2'
                               / 'parquet_data' / 'events_with_action_values.parquet',
                               columns=['id', 'action_value'])
        value_map = dict(zip(_av['id'], pd.to_numeric(_av['action_value'],
                                                        errors='coerce').fillna(0.0)))
        print(f"  {len(value_map):,} event values loaded.", flush=True)
    except Exception as _ve:
        print(f"  [warn] action values unavailable ({_ve}) — DefR-OBV will be 0.")
        value_map = None
    parts = []
    for y in years:
        if tables[y] is None:
            continue
        prob, ptype_lookup = tables[y]
        parts.append(compute_per_player(
            ev[ev['_yr'] == y], resp[resp['_yr'] == y], prob, ptype_lookup,
            player_slot, msl, match_teams, def_team_players, pmt_lookup,
            value_map=value_map, base_resp=resp[resp['_yr'] < y]))
    out = finalize_defr(pd.concat(parts, ignore_index=True))
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
    eng.to_parquet(_OUT / 'defensive_engagements.parquet')
    eng = eng.dropna(subset=['p_win'])          # the first year: no earlier rates
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

    out_path = _OUT / 'defr_per_player_season.parquet'
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
