#!/usr/bin/env python3
"""Player tendencies — the PRIMITIVE stylistic layer under styles v3.

Futi's framing: "tendencies help sort players into styles". Roles say WHERE a
player operates; TENDENCIES say, of the things he attempts there, which way he
leans; STYLES (build_styles.py) are archetypes derived from this tendency space.

Every tendency is a BIPOLAR pair (pole_low <-> pole_high) reduced to a single
number in [0, 1] (a share of attempts) or a map-derived scalar, then expressed
as a WITHIN-ROLE PERCENTILE (0-100, 50 = role-typical). This is ATTEMPT
COMPOSITION, not quality: a high percentile means "does more of this than his
role peers", never "is better". Tendencies are DESCRIPTIVE ONLY and never enter
the ACP rating or projection.

Design decisions (2026-07-16):
  * SOURCE = the same GPA event store the role maps are built from
    (liga3/campeonato_portugal_events.parquet), so tendencies and roles can
    never disagree about what happened. The map channels and grid come from
    build_role_features' own helpers (flag_channels / grid_index / smooth_norm)
    — one definition of the maps, not two. main() asserts the map scalars this
    file derives reproduce role_features_season.parquet.
  * EVERYTHING IS KEYED BY `keys` rather than hardcoded to the season, so the
    identical code path produces the odd/even-match half tables that
    build_styles.py needs for split-half validation.
  * COHORT / REFERENCE = pooled across seasons AND leagues within each role.
    One reference distribution per (role, tendency) means a percentile means the
    same thing in every season, which is what makes the YoY stability check and
    the All-Seasons average in the UI legible. League differences in these
    leanings are treated as real signal, not noise to normalise away.
  * PER SEASON, not futi's rolling 12 months — matches our engine's structure
    (documented divergence, per the approved spec).
  * Percentile reference is the >=900' cohort; 300-900' players are still
    scored against it but flagged thin_sample; <300' are dropped.
  * GKs are excluded (they never enter role_assignments_season).

Output: tendencies_season.parquet — one row per (playerId, seasonId, role):
    t_<key>  raw ratio / scalar
    p_<key>  within-role percentile 0-100
  + playerId, seasonId, role, side, name, mins_played, thin_sample

Run from the Dashboard dir:  python models/roles/build_tendencies.py
Event store location may be overridden with $ACP_GPA_DATA (CI uses the sibling
repo layout, which is the default).
"""
from pathlib import Path
import os
import sys
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_DASH = _HERE.parent.parent
sys.path.insert(0, str(_HERE))
from build_role_features import (flag_channels, grid_index, smooth_norm,
                                 NX, NY, RESTART_PRIMARIES)

# Sibling-repo layout by default (what CI replicates); overridable so this can
# run from a git worktree, where the relative walk-up lands somewhere else.
_GPA_DATA = Path(os.environ.get(
    'ACP_GPA_DATA', _DASH.parent.parent / 'GPA Model Project v2' / 'parquet_data'))
_ROLES_DIR = Path(os.environ.get('ACP_ROLES_DIR', _HERE))
_DEFR = Path(os.environ.get(
    'ACP_DEFR', _DASH / 'models' / 'defr' / 'defr_per_player_season.parquet'))
_OUT = _HERE / 'tendencies_season.parquet'

MIN_MINS_COHORT = 900     # percentile reference cohort
MIN_MINS_SCORED = 300     # below this we don't score at all
MIN_DENOM = 10            # a ratio needs this many attempts or it's NaN
EXCURSION_M = 20.0        # metres forward of his own station (see build_excursion)

DEADBALL_PRIMARIES = {'corner', 'free_kick'}   # taken by this player

# ---------------------------------------------------------------------------
# TENDENCY METADATA
#   pole_low / pole_high — the two ends; the value is the share of pole_high,
#   so percentile 100 = maximally pole_high, 0 = maximally pole_low.
#   confidence — 'high' | 'low'; low-confidence pairs are hidden behind fine
#   print in the UI (thin event support).
#   yoy — MEASURED year-on-year rank stability of the within-role percentile
#   (Spearman, cohort players in consecutive seasons, n=500). Documentation,
#   not a coefficient: nothing reads it. Three sit under 0.30 and are read as
#   team-context descriptors rather than player traits — low_high (0.26) and
#   counterpress_retreat (0.20) are set by the coach's line height and pressing
#   scheme, so they reset when a player changes club, and near_far_post (0.22)
#   is simply thin. Re-measure if the definitions change.
# ---------------------------------------------------------------------------
TENDENCY_META = {
    'low_high': dict(
        pole_low='Low block', pole_high='High line', confidence='high', yoy=0.262,
        desc='Height of his defensive actions (OOP map x-centroid).'),
    'passive_active': dict(
        pole_low='Passive', pole_high='Active', confidence='high', yoy=0.374,
        desc='Defensive volume above positional expectation (DefR /90).'),
    'passive_active_buildup': dict(
        pole_low='Passive', pole_high='Active', confidence='high', yoy=0.441,
        desc='How much he demands the ball in build-up (own-half passes /90).'),
    'stationary_mobile': dict(
        pole_low='Stationary', pole_high='Mobile', confidence='high', yoy=0.556,
        desc='Spatial spread of his in-possession map (entropy).'),
    'stationary_adventurous': dict(
        pole_low='Stationary', pole_high='Adventurous', confidence='high', yoy=0.340,
        desc='How often he ventures forward of his own station — share of his '
             'in-possession actions 20m+ ahead of his own average position.'),
    'secure_progressive': dict(
        pole_low='Secure', pole_high='Progressive', confidence='high', yoy=0.484,
        desc='Progressive passes vs back/square passes.'),
    'create_arrive': dict(
        pole_low='Create', pole_high='Arrive', confidence='high', yoy=0.422,
        desc='Makes the chance for someone else vs arrives to take it.'),
    'buildup_stretch': dict(
        pole_low='Build-up', pole_high='Stretch lines', confidence='high', yoy=0.346,
        desc='Drops into build-up vs plays on the last line (IP x-centroid).'),
    'outside_between': dict(
        pole_low='Outside block', pole_high='Between lines', confidence='high', yoy=0.346,
        desc='Operates in front of the block vs steps between the lines '
             '(IP x-centroid). Same mechanic as Build-up<->Stretch lines under '
             'the CB-facing name futi uses; no panel shows both.'),
    'controlled_longball': dict(
        pole_low='Controlled', pole_high='Long ball', confidence='high', yoy=0.585,
        desc='Long passes vs short/medium passes.'),
    'return_circulate': dict(
        pole_low='Recycle', pole_high='Create', confidence='high', yoy=0.504,
        desc='Passes back to reset vs passes that feed the attack.'),
    'central_wide': dict(
        pole_low='Central', pole_high='Wide', confidence='high', yoy=0.417,
        desc='Lateral station of his in-possession actions.'),
    'combine_cross': dict(
        pole_low='Combine', pole_high='Cross', confidence='high', yoy=0.469,
        desc='In the final third: combines short vs crosses.'),
    'building_attacking': dict(
        pole_low='Building', pole_high='Attacking', confidence='high', yoy=0.404,
        desc='Own-half construction vs final-third attacking actions.'),
    'ground_aerial': dict(
        pole_low='Ground', pole_high='Aerial', confidence='high', yoy=0.532,
        desc='Aerial duels vs ground duels.'),
    'come_short_run_behind': dict(
        pole_low='Come short', pole_high='Run behind', confidence='high', yoy=0.445,
        desc='How the ball reaches him: to feet vs through/over the top '
             '(offsides count as run-behind evidence).'),
    'near_far_post': dict(
        pole_low='Near post', pole_high='Far post', confidence='low', yoy=0.220,
        desc='Box shots on the same side as the attack vs the far side. '
             'Thin shot samples — read with caution.'),
    'cut_inside_outside': dict(
        pole_low='Cut inside', pole_high='Cut outside', confidence='high', yoy=0.350,
        desc='From wide areas, carries that go infield vs down the line.'),
    'carry_pass': dict(
        pole_low='Carry', pole_high='Pass', confidence='high', yoy=0.592,
        desc='Moves the ball himself vs releases it.'),
    'counterpress_retreat': dict(
        pole_low='Counterpress', pole_high='Retreat', confidence='high', yoy=0.202,
        desc='Wins it back immediately vs recovers after a reset.'),
    'openplay_deadball': dict(
        pole_low='Open play', pole_high='Dead ball', confidence='high', yoy=0.592,
        desc='Share of his involvement that comes from set-piece deliveries.'),
    'poacher_longrange': dict(
        pole_low='Poacher', pole_high='Long range', confidence='high', yoy=0.373,
        desc='Shoots from inside the box vs from distance.'),
}

# Per-role pole renames — futi names the same mechanic differently per panel.
ROLE_POLE_OVERRIDES = {
    ('Deep Midfielder', 'return_circulate'): dict(pole_low='Return',
                                                  pole_high='Circulate'),
}

# ---------------------------------------------------------------------------
# ROLE MENUS — which tendencies each role's panel shows, in order.
# Mirrors futi's six panels (Forward / WA / CA / Midfielder / WD / CB), mapped
# onto our six observed roles, plus our house additions.
# ---------------------------------------------------------------------------
ROLE_TENDENCY_MENU = {
    # futi "Forward"
    'Striker': ['create_arrive', 'buildup_stretch', 'ground_aerial',
                'low_high', 'passive_active', 'near_far_post',
                'come_short_run_behind', 'stationary_mobile',
                # house
                'poacher_longrange', 'carry_pass'],
    # futi "Wide Attacker"
    'Wide Attacker': ['create_arrive', 'low_high', 'passive_active',
                      'stationary_mobile', 'central_wide', 'combine_cross',
                      'secure_progressive', 'cut_inside_outside',
                      # house
                      'carry_pass', 'counterpress_retreat', 'openplay_deadball'],
    # futi "Central Attacker" -> our Advanced Midfielder
    'Advanced Midfielder': ['create_arrive', 'buildup_stretch', 'low_high',
                            'passive_active', 'stationary_mobile',
                            'central_wide', 'combine_cross',
                            'secure_progressive',
                            # house
                            'carry_pass', 'counterpress_retreat',
                            'openplay_deadball', 'poacher_longrange'],
    # futi "Midfielder" -> our Deep Midfielder
    'Deep Midfielder': ['low_high', 'passive_active',
                        'passive_active_buildup', 'controlled_longball',
                        'stationary_adventurous', 'return_circulate',
                        # house
                        'carry_pass', 'counterpress_retreat'],
    # futi "Wide Defender". The spec left the Mobile-vs-Adventurous call to the
    # data; MOBILE wins on all three counts measured (2026-07-16): it is far
    # more stable (YoY 0.53 vs 0.27 — Adventurous is below the 0.30 bar for
    # fullbacks), it is orthogonal to Building<->Attacking which already sits on
    # this panel (rho -0.06 vs Adventurous's 0.48 — a fullback getting forward
    # is what Building<->Attacking already says), and it has usable spread
    # (IQR 0.10 vs 0.03).
    'Wide Defender': ['create_arrive', 'low_high', 'passive_active',
                      'stationary_mobile', 'combine_cross',
                      'secure_progressive', 'building_attacking',
                      # house
                      'carry_pass', 'openplay_deadball'],
    # futi "Centre Back"
    'Central Defender': ['low_high', 'passive_active',
                         'stationary_adventurous', 'secure_progressive',
                         'outside_between', 'passive_active_buildup',
                         'return_circulate', 'building_attacking',
                         # house
                         'ground_aerial', 'carry_pass'],
}

ALL_TENDENCIES = list(TENDENCY_META.keys())

# Tendencies that cannot be recomputed on a half-season (DefR is published per
# player-season only). build_styles' split-half check holds these constant and
# discloses it.
SEASON_ONLY_TENDENCIES = {'passive_active'}


def poles(role, key):
    """(pole_low, pole_high) for a tendency as shown on a given role's panel."""
    m = TENDENCY_META[key]
    o = ROLE_POLE_OVERRIDES.get((role, key), {})
    return o.get('pole_low', m['pole_low']), o.get('pole_high', m['pole_high'])


# ---------------------------------------------------------------------------
# events
# ---------------------------------------------------------------------------
_EV_COLS = ['matchId', 'seasonId', 'team.id', 'player.id',
            'type.primary', 'type.secondary', 'location.x', 'location.y',
            'pass.endLocation.x', 'pass.endLocation.y', 'pass.recipient.id',
            'carry.endLocation.x', 'carry.endLocation.y',
            'groundDuel.duelType', 'aerialDuel.firstTouch',
            'possession.attack.flank', 'competitionId']

_TAGS = ['long_pass', 'short_or_medium_pass', 'forward_pass', 'back_pass',
         'lateral_pass', 'progressive_pass', 'pass_to_final_third',
         'pass_to_penalty_area', 'cross', 'through_pass', 'smart_pass',
         'key_pass', 'shot_assist', 'carry', 'progressive_run', 'dribble',
         'touch_in_box', 'aerial_duel', 'ground_duel', 'recovery',
         'counterpressing_recovery']


def load_events():
    frames = []
    for f in ['liga3_portugal_events', 'campeonato_portugal_events']:
        part = pd.read_parquet(_GPA_DATA / f'{f}.parquet', columns=_EV_COLS)
        print(f"  {f}: {len(part):,} events", flush=True)
        frames.append(part)
    ev = pd.concat(frames, ignore_index=True)
    ev['seasonId'] = pd.to_numeric(ev['seasonId'], errors='coerce').astype('Int64')
    # Same drop as build_role_features.load_events — required for the map
    # scalars to reproduce role_features_season exactly (see main()'s assert).
    ev = ev.dropna(subset=['player.id', 'location.x', 'location.y'])
    ev['player.id'] = ev['player.id'].astype('int64')
    return ev


def prepare(ev):
    """Tag the event frame once: secondary-tag booleans, IP/OOP channels
    (via build_role_features.flag_channels, so the channel definition is
    shared with the role maps), and the derived geometry columns."""
    sets = [frozenset(t) if isinstance(t, (list, tuple, np.ndarray)) else frozenset()
            for t in ev['type.secondary'].tolist()]
    for t in _TAGS:
        ev[f'x_{t}'] = np.fromiter((t in s for s in sets), dtype=bool,
                                   count=len(sets))
    ev = flag_channels(ev)          # -> _restart, _rec, _aer, _ip, _op
    ev['playerId'] = ev['player.id']
    ev['parity'] = (ev['matchId'].astype('int64') % 2).astype(int)
    ev['_x'] = ev['location.x'].astype(float)
    ev['_y'] = ev['location.y'].astype(float)
    ev['_yf'] = (ev['_y'] - 50.0).abs()
    return ev


def build_counts(ev, keys):
    """Counts per `keys` for every ratio."""
    tp = ev['type.primary']
    x, yf = ev['_x'], ev['_yf']
    is_pass = (tp == 'pass') & ~ev['_restart']
    is_shot = (tp == 'shot') & ~ev['_restart']
    op = ~ev['_restart']

    c = {}
    c['n_pass'] = is_pass
    for t in ['long_pass', 'short_or_medium_pass', 'forward_pass', 'back_pass',
              'lateral_pass', 'progressive_pass', 'pass_to_final_third',
              'pass_to_penalty_area', 'cross', 'through_pass', 'key_pass',
              'shot_assist']:
        c[f'p_{t}'] = is_pass & ev[f'x_{t}']
    c['p_ownhalf'] = is_pass & (x < 50)
    c['p_short_final3'] = (is_pass & (x >= 66) & ev['x_short_or_medium_pass']
                           & ~ev['x_cross'])

    c['a_carry'] = op & ev['x_carry']
    c['a_dribble'] = op & ev['x_dribble']
    c['a_prog_run'] = op & ev['x_progressive_run']
    c['a_accel'] = (tp == 'acceleration') & op
    c['a_touch_box'] = op & ev['x_touch_in_box']

    c['n_shot'] = is_shot
    c['sh_in_box'] = is_shot & (x >= 84) & (yf <= 20)
    c['sh_out_box'] = is_shot & ~((x >= 84) & (yf <= 20))

    c['d_aerial'] = op & ev['x_aerial_duel']
    c['d_ground'] = op & ev['x_ground_duel']

    c['r_recovery'] = op & ev['x_recovery']
    # counterpressing_recovery is a strict subset of recovery (verified)
    c['r_counterpress'] = op & ev['x_counterpressing_recovery']

    c['db_taken'] = tp.isin(DEADBALL_PRIMARIES)
    c['n_offside'] = (tp == 'offside')

    c['n_ip'] = ev['_ip']
    c['n_op'] = ev['_op']

    for k, v in c.items():
        ev[f'c_{k}'] = v
    cols = [f'c_{k}' for k in c]
    g = ev.groupby(keys, observed=True)[cols].sum()
    g.columns = [k[2:] for k in g.columns]
    return g.reset_index()


def build_map_scalars(ev, keys):
    """x_ip / x_op / yf_ip / IP-map entropy, per `keys`.

    Uses build_role_features' grid + smoothing so the map is defined once.
    main() asserts this reproduces role_features_season.parquet at season
    level."""
    ip = ev[ev['_ip']]
    opp = ev[ev['_op']]
    out = ip.groupby(keys, observed=True).agg(x_ip=('_x', 'mean'),
                                              yf_ip=('_yf', 'mean'))
    out['x_op'] = opp.groupby(keys, observed=True)['_x'].mean()

    cell = grid_index(ip['_x'], ip['_y'])
    cnt = (ip.assign(_cell=cell).groupby(keys + ['_cell'], observed=True)
             .size().unstack('_cell', fill_value=0)
             .reindex(columns=range(NX * NY), fill_value=0))
    maps = np.vstack([smooth_norm(r) for r in cnt.to_numpy()])
    ent = pd.Series(map_entropy(maps), index=cnt.index, name='_entropy')
    out = out.join(ent)
    return out.reset_index()


def build_receptions(ev, keys):
    """Come short <-> Run behind: how the ball ARRIVES to him.

    Keyed on pass.recipient.id, so it is about passes he receives, not passes
    he makes. 'Run behind' = the ball is played into space ahead of him
    (through ball, or a pass that gains real ground); 'come short' = it arrives
    to feet without gaining ground. Offsides are counted as run-behind evidence
    (the classic marker of a player who attacks the space behind)."""
    is_pass = (ev['type.primary'] == 'pass') & ~ev['_restart']
    rid = pd.to_numeric(ev['pass.recipient.id'], errors='coerce')
    # recipient 0 is Wyscout's unattributed sentinel, not a player
    r = ev[is_pass & rid.notna() & (rid > 0)].copy()
    r['playerId'] = rid[r.index].astype('int64')
    gain = r['pass.endLocation.x'].astype(float) - r['_x']
    r['_behind'] = r['x_through_pass'] | (gain >= 15)
    g = r.groupby(keys, observed=True)['_behind'].agg(['sum', 'size'])
    out = pd.DataFrame({'rec_behind': g['sum'],
                        'rec_short': g['size'] - g['sum']}).reset_index()
    # offsides are run-behind evidence, credited to the offside player
    off = (ev[ev['type.primary'] == 'offside'].groupby(keys, observed=True)
             .size().rename('n_off').reset_index())
    out = out.merge(off, on=keys, how='outer')
    out['rec_behind'] = (out['rec_behind'].fillna(0)
                         + out['n_off'].fillna(0))
    out['rec_short'] = out['rec_short'].fillna(0)
    return out.drop(columns=['n_off'])


def build_carry_direction(ev, keys):
    """Cut inside <-> Cut outside, from carries that START in wide areas.

    Side-agnostic: compare |y-50| at the start and end of the carry, so a left
    winger cutting right and a right winger cutting left are the same thing."""
    m = (~ev['_restart']) & ev['carry.endLocation.y'].notna() & ev['_y'].notna()
    ca = ev[m & (ev['x_carry'] | (ev['type.primary'] == 'acceleration'))].copy()
    yf1 = (ca['carry.endLocation.y'].astype(float) - 50.0).abs()
    d = yf1 - ca['_yf']
    ca['_out'] = (ca['_yf'] > 15) & (d > 2)      # only from wide starts
    ca['_in'] = (ca['_yf'] > 15) & (d < -2)
    g = ca.groupby(keys, observed=True)[['_out', '_in']].sum()
    return g.rename(columns={'_out': 'cut_out', '_in': 'cut_in'}).reset_index()


def build_post_side(ev, keys):
    """Near post <-> Far post (LOW CONFIDENCE).

    For box shots, compare which side of the goal the shooter is on against the
    flank the attack came down (possession.attack.flank). Same side = near
    post, opposite = far post. Wyscout: low y = LEFT. Attacks flagged 'center'
    carry no near/far meaning and are dropped, as are central shots
    (|y-50| <= 6), which leaves thin samples — hence confidence='low'."""
    s = ev[(ev['type.primary'] == 'shot') & ~ev['_restart']
           & (ev['_x'] >= 84) & (ev['_yf'] <= 20) & (ev['_yf'] > 6)
           & ev['possession.attack.flank'].notna()].copy()
    if s.empty:
        return pd.DataFrame(columns=keys + ['post_near', 'post_far'])
    flank = s['possession.attack.flank'].astype(str).str.lower()
    left = s['_y'] < 50
    s['_near'] = ((flank == 'left') & left) | ((flank == 'right') & ~left)
    s['_far'] = ((flank == 'left') & ~left) | ((flank == 'right') & left)
    g = s.groupby(keys, observed=True)[['_near', '_far']].sum()
    return g.rename(columns={'_near': 'post_near', '_far': 'post_far'}).reset_index()


def build_excursion(ev, keys):
    """Stationary <-> Adventurous: forward excursions from his OWN station.

    Measured against the player's own average in-possession x, NOT a fixed
    'advanced band'. That matters: a fixed band on an L1-normalised map is,
    empirically, a monotone restatement of the map's x-centroid (measured
    rho 0.98-0.99 with x_ip in every role), which would have made Adventurous
    a duplicate of Build-up<->Stretch — and on the CB panel a duplicate of the
    Outside block<->Between lines slider sitting right next to it. Anchoring
    the tail to his own mean is what the spec's 'forward-excursion share from
    a deep station' actually asks for, and it decorrelates properly
    (rho 0.34 DM / 0.54 CB) while staying stable (YoY 0.54 DM / 0.48 CB).

    20m chosen over 15m (too collinear: rho 0.79 with x_ip for CBs) and 25m
    (rarer, noisier: CB YoY drops)."""
    e = ev[ev['_ip'] & ev['_x'].notna()][keys + ['_x']].copy()
    mean_x = (e.groupby(keys, observed=True)['_x'].mean()
                .rename('_mean_x').reset_index())
    e = e.merge(mean_x, on=keys)
    e['_exc'] = e['_x'] >= (e['_mean_x'] + EXCURSION_M)
    g = e.groupby(keys, observed=True)['_exc'].agg(['mean', 'size'])
    return g.rename(columns={'mean': 'excursion_share',
                             'size': 'n_ip_loc'}).reset_index()


# ---------------------------------------------------------------------------
# ratios
# ---------------------------------------------------------------------------
def _ratio(hi, lo, min_denom=MIN_DENOM):
    """share of the HIGH pole; NaN when the pair has too little support."""
    hi = pd.to_numeric(hi, errors='coerce').fillna(0.0)
    lo = pd.to_numeric(lo, errors='coerce').fillna(0.0)
    den = hi + lo
    out = hi / den.replace(0, np.nan)
    return out.where(den >= min_denom)


def map_entropy(maps):
    """Shannon entropy of the (already L1-normalised) IP map — spatial spread.
    High = his touches are scattered (Mobile); low = concentrated in a
    station."""
    p = np.clip(maps, 1e-12, None)
    p = p / p.sum(axis=1, keepdims=True)
    return -(p * np.log(p)).sum(axis=1)


def compute_tendencies(C, keys):
    """C = one row per `keys`: counts + map scalars + defr + minutes joined."""
    t = C[keys].copy()

    t['t_secure_progressive'] = _ratio(
        C['p_progressive_pass'], C['p_back_pass'] + C['p_lateral_pass'])
    t['t_create_arrive'] = _ratio(
        C['n_shot'] + C['a_touch_box'],
        C['p_pass_to_penalty_area'] + C['p_key_pass'] + C['p_shot_assist'])
    t['t_controlled_longball'] = _ratio(C['p_long_pass'],
                                        C['p_short_or_medium_pass'])
    t['t_return_circulate'] = _ratio(
        C['p_pass_to_final_third'] + C['p_pass_to_penalty_area']
        + C['p_key_pass'] + C['p_through_pass'], C['p_back_pass'])
    t['t_combine_cross'] = _ratio(C['p_cross'], C['p_short_final3'])
    t['t_building_attacking'] = _ratio(
        C['a_touch_box'] + C['p_cross'] + C['n_shot']
        + C['p_pass_to_penalty_area'], C['p_ownhalf'])
    t['t_ground_aerial'] = _ratio(C['d_aerial'], C['d_ground'])
    t['t_carry_pass'] = _ratio(
        C['n_pass'], C['a_carry'] + C['a_dribble'] + C['a_prog_run']
        + C['a_accel'])
    t['t_counterpress_retreat'] = _ratio(
        C['r_recovery'] - C['r_counterpress'], C['r_counterpress'])
    t['t_openplay_deadball'] = _ratio(C['db_taken'], C['n_ip'])
    t['t_poacher_longrange'] = _ratio(C['sh_out_box'], C['sh_in_box'])
    t['t_come_short_run_behind'] = _ratio(C['rec_behind'], C['rec_short'])
    t['t_cut_inside_outside'] = _ratio(C['cut_out'], C['cut_in'])
    t['t_near_far_post'] = _ratio(C['post_far'], C['post_near'], min_denom=5)

    # volume-anchored ("how much", not "which of two"; percentile does the rest)
    mins = pd.to_numeric(C['mins_played'], errors='coerce')
    t['t_passive_active_buildup'] = (C['p_ownhalf'] / mins.replace(0, np.nan)) * 90.0
    t['t_passive_active'] = C['defr_per90']

    # map-derived scalars
    t['t_low_high'] = C['x_op']
    t['t_buildup_stretch'] = C['x_ip']
    t['t_outside_between'] = C['x_ip']      # same mechanic, CB-facing name
    t['t_central_wide'] = C['yf_ip']
    t['t_stationary_mobile'] = C['_entropy']
    t['t_stationary_adventurous'] = C['excursion_share'].where(
        pd.to_numeric(C['n_ip_loc'], errors='coerce') >= 100)
    return t


def tendency_table(ev, keys, roles, defr):
    """Full tendency table for any grouping (`keys` must start with
    playerId/seasonId). Shared by the season build and build_styles' split-half
    check."""
    C = build_counts(ev, keys)
    for part in (build_map_scalars(ev, keys), build_receptions(ev, keys),
                 build_carry_direction(ev, keys), build_post_side(ev, keys),
                 build_excursion(ev, keys)):
        for k in keys:
            part[k] = part[k].astype(C[k].dtype)
        C = C.merge(part, on=keys, how='left')
    C['playerId'] = pd.to_numeric(C['playerId'], errors='coerce').astype('Int64')
    C['seasonId'] = pd.to_numeric(C['seasonId'], errors='coerce').astype('Int64')
    C = C.merge(roles, on=['playerId', 'seasonId'], how='inner')
    C = C.merge(defr, on=['playerId', 'seasonId'], how='left')
    return compute_tendencies(C, keys), C


def within_role_percentiles(T, roles_col='role', ref_mask=None):
    """ECDF percentile of each raw tendency within its role, referenced to the
    >=900' cohort. Thin players are scored against the same cohort (so a thin
    player and a full-season player at the same raw value get the same
    percentile) — they are flagged, not rescaled."""
    if ref_mask is None:
        ref_mask = T['mins_played'] >= MIN_MINS_COHORT
    for key in ALL_TENDENCIES:
        raw, pct = f't_{key}', f'p_{key}'
        T[pct] = np.nan
        for role, g in T.groupby(roles_col, observed=True):
            ref = g.loc[ref_mask.reindex(g.index, fill_value=False), raw].dropna()
            if len(ref) < 20:
                continue
            ref_v = np.sort(ref.to_numpy())
            v = g[raw].to_numpy(dtype=float)
            # midpoint ECDF so ties land mid-band rather than at an edge
            lo = np.searchsorted(ref_v, v, side='left')
            hi = np.searchsorted(ref_v, v, side='right')
            p = (lo + hi) / 2.0 / len(ref_v) * 100.0
            p[np.isnan(v)] = np.nan
            T.loc[g.index, pct] = p
    return T


def load_roles():
    R = pd.read_parquet(_ROLES_DIR / 'role_assignments_season.parquet')
    R = R[['playerId', 'seasonId', 'primary_role_name', 'side', 'mins_played',
           'name']].rename(columns={'primary_role_name': 'role'})
    R['playerId'] = pd.to_numeric(R['playerId'], errors='coerce').astype('Int64')
    R['seasonId'] = pd.to_numeric(R['seasonId'], errors='coerce').astype('Int64')
    return R[R['role'].notna() & (R['playerId'] > 0)]


def load_defr():
    D = pd.read_parquet(_DEFR, columns=['playerId', 'seasonId', 'defr_per90'])
    D['playerId'] = pd.to_numeric(D['playerId'], errors='coerce').astype('Int64')
    D['seasonId'] = pd.to_numeric(D['seasonId'], errors='coerce').astype('Int64')
    return D.drop_duplicates(['playerId', 'seasonId'])


def main():
    print("[1/4] events…", flush=True)
    ev = prepare(load_events())
    print(f"  {len(ev):,} player events  "
          f"(IP {ev['_ip'].sum():,} / OOP {ev['_op'].sum():,})", flush=True)

    print("[2/4] tendencies…", flush=True)
    keys = ['playerId', 'seasonId']
    T, C = tendency_table(ev, keys, load_roles(), load_defr())
    T = T.merge(load_roles(), on=keys, how='left')

    # the maps here must be the same maps the roles were clustered from
    F = pd.read_parquet(_ROLES_DIR / 'role_features_season.parquet',
                        columns=['playerId', 'seasonId', 'x_ip', 'x_op', 'yf_ip'])
    F['playerId'] = pd.to_numeric(F['playerId'], errors='coerce').astype('Int64')
    F['seasonId'] = pd.to_numeric(F['seasonId'], errors='coerce').astype('Int64')
    chk = C[keys + ['x_ip', 'x_op', 'yf_ip']].merge(
        F, on=keys, how='inner', suffixes=('', '_ref'))
    print(f"  map-scalar agreement with role_features_season "
          f"({len(chk):,} rows):")
    for col in ['x_ip', 'x_op', 'yf_ip']:
        d = (chk[col] - chk[f'{col}_ref']).abs()
        print(f"    {col:<6} exact {(d < 1e-9).mean() * 100:5.1f}%   "
              f"median |delta| {d.median():.4f}   max {d.max():.3f}")
        # A definitional break would move these by whole pitch-units; the
        # sub-0.1 residue is the known local-vs-CI event-snapshot drift (the
        # committed reference parquet was built by CI from a marginally
        # different snapshot). In CI both are rebuilt from one snapshot and
        # this comes out exact.
        assert d.median() < 1.0, (
            f"{col} disagrees with role_features structurally "
            f"(median |delta| {d.median():.3f}) — the map definition has "
            f"drifted, not just the event snapshot")

    T['mins_played'] = pd.to_numeric(T['mins_played'], errors='coerce')
    T = T[T['mins_played'] >= MIN_MINS_SCORED].copy()
    T['thin_sample'] = T['mins_played'] < MIN_MINS_COHORT

    print("[3/4] within-role percentiles…", flush=True)
    T = within_role_percentiles(T)

    print("[4/4] writing…", flush=True)
    cols = (['playerId', 'seasonId', 'role', 'side', 'name', 'mins_played',
             'thin_sample']
            + [f't_{k}' for k in ALL_TENDENCIES]
            + [f'p_{k}' for k in ALL_TENDENCIES])
    T = T[cols].sort_values(['seasonId', 'role', 'name'])
    T.to_parquet(_OUT, index=False)
    print(f"\nsaved {len(T):,} rows -> {_OUT.name}")
    print(f"  cohort (>= {MIN_MINS_COHORT}'): "
          f"{(~T['thin_sample']).sum():,}   thin: {T['thin_sample'].sum():,}")

    print("\ncoverage by role (non-null share of each tendency, cohort only):")
    ck = T[~T['thin_sample']]
    print(f"  {'tendency':<26}" + ''.join(f"{r[:9]:>10}" for r in ROLE_TENDENCY_MENU))
    for k in ALL_TENDENCIES:
        line = f"  {k:<26}"
        for r in ROLE_TENDENCY_MENU:
            g = ck[ck['role'] == r]
            line += f"{(g[f't_{k}'].notna().mean() * 100 if len(g) else 0):>9.0f}%"
        print(line + ('   [menu]' if any(k in m for m in ROLE_TENDENCY_MENU.values()) else ''))


if __name__ == '__main__':
    main()
