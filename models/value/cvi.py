"""CVI — the club's Composite Value Index and its projected-EUR mapping.

Pure pandas/numpy: performance quality x age curve x reliability x league
factor -> CVI, then a power curve + position multipliers -> projected EUR,
plus the career (decay-weighted prior) variant and the market-context
features used by the EUR v2 regression. Extracted verbatim from app.py in
2026-09 so it can be trained and tested without booting Streamlit; app.py
re-exports every name here (see __all__) so call sites are unchanged.

Ownership: models/value. Nothing in this module may import streamlit.
"""
import logging
import math

import numpy as np
import pandas as pd

from league_config import competition_for_season, all_season_id_map

SEASON_ID_MAP = all_season_id_map()
logger = logging.getLogger(__name__)


_RADAR_TEMPLATES = None


def _radar_templates():
    """(weights, position_groups) role templates from config.yaml — the CVI
    eligibility rule ('which templates may this position be scored on')
    reads them. app.py passes its own copies; this fallback lets the model
    run standalone (training, tests) from the same file."""
    global _RADAR_TEMPLATES
    if _RADAR_TEMPLATES is None:
        import os
        import yaml
        here = os.path.dirname(os.path.abspath(__file__))
        cfg_path = os.path.join(os.path.dirname(os.path.dirname(here)), 'config.yaml')
        with open(cfg_path, encoding='utf-8') as fh:
            cfg = yaml.safe_load(fh) or {}
        _RADAR_TEMPLATES = (cfg.get('weights', {}) or {}, cfg.get('position_groups', {}) or {})
    return _RADAR_TEMPLATES

__all__ = [
    '_ENGINE_ATTACK_ROLES',
    '_ENGINE_DEF_ROLES',
    '_ATTACK_TEMPLATE_ROLES',
    '_ENGINE_ROLE_ALIASES',
    '_canonical_engine_role',
    '_engine_role_is_attacking',
    'CVI_PERF_WEIGHTS',
    'CVI_AGE_VALUE_PARAMS',
    '_cvi_expected_perf_at',
    '_cvi_cum_remaining_career',
    '_CVI_MAX_CAREER_VALUE',
    'CVI_RELIAB_CEILING_BY_POS',
    'CVI_RELIAB_CEILING_DEFAULT',
    'CVI_RELIAB_MINS_TO_CEILING',
    'CVI_RELIAB_MINS_TO_CEILING_DEFAULT',
    'CVI_REPLACEMENT_PERF',
    'CVI_LEAGUE_MULTIPLIER',
    'CVI_LEAGUE_DEFAULT',
    'POSITION_EUR_MULTIPLIER',
    'CAMP_PROJECTED_EUR_PENALTY',
    'PROJECTED_EUR_COEF',
    'PROJECTED_EUR_EXP',
    'PROJECTED_EUR_CAP',
    'cvi_to_projected_eur',
    '_cvi_position_group',
    '_cvi_reliability_weight',
    '_cvi_age_value_multiplier',
    'compute_cvi_columns',
    'CVI_CAREER_DECAY',
    'CVI_CAREER_MAX_LOOKBACK',
    'CVI_CAREER_CURRENT_BONUS',
    '_build_player_season_perf_table',
    '_season_year',
    'compute_career_cvi',
    'build_player_priors_lookup',
    'most_recent_season_for_player',
    'compute_market_features',
]


# --- Player one-pager: engine-role helpers -----------------------------------
# The six canonical engine roles (player_engine.parquet's `role`). The PDF
# branches on these, not on the raw position or the config template names:
# Lucas asked for the split by engine role (Central Defender / Wide Defender /
# Advanced Midfielder / ...). The three "attacking" roles get shot + creation
# maps; the three others get the defensive heatmap.
_ENGINE_ATTACK_ROLES = {'Striker', 'Wide Attacker', 'Advanced Midfielder'}
_ENGINE_DEF_ROLES = {'Deep Midfielder', 'Wide Defender', 'Central Defender'}
# Config-template fallback for players with no engine row (keepers, mostly).
# An ALLOWLIST, not a denylist: only these template roles get the attacking
# layout (shot + creation maps); everything else — GK templates, centre-backs,
# full-backs, holding mids — gets the defensive heatmap, which is the safe
# default for any position. A denylist silently routed keepers to the
# attacking branch, because the GK templates were not in it.
_ATTACK_TEMPLATE_ROLES = {
    'Mobile Striker', 'Shadow Striker', 'Poacher', 'Target Man',
    'Pressing Forward',                                    # CF / SS
    'Advanced Playmaker', 'Wide Winger', 'Creative Winger',
    'Inside Forward',                                      # wingers / AM
}
# A handful of legacy rows carry dirty variants (STRIKER, WINGER, CB, CM);
# normalise them so the split is total.
_ENGINE_ROLE_ALIASES = {
    'STRIKER': 'Striker', 'WINGER': 'Wide Attacker',
    'CB': 'Central Defender', 'CM': 'Deep Midfielder',
}


def _canonical_engine_role(raw):
    """Map a raw engine `role` value to one of the six canonical roles, or
    None when it is missing/unrecognised (e.g. keepers, whom the engine
    does not rate)."""
    if raw is None:
        return None
    s = str(raw).strip()
    if s in _ENGINE_ATTACK_ROLES or s in _ENGINE_DEF_ROLES:
        return s
    return _ENGINE_ROLE_ALIASES.get(s.upper())


def _engine_role_is_attacking(role):
    """True for Striker / Wide Attacker / Advanced Midfielder. Keepers and
    anything unclassified fall through to the defensive layout, which is the
    safe default — a defensive heatmap is meaningful for every position, a
    shot map is not."""
    return _canonical_engine_role(role) in _ENGINE_ATTACK_ROLES


# Position-tuned blend of Role_Score percentile and Action V percentile
# for PerformanceQuality.
#
# v1.8 rebalance — the old 50-80% role weights underestimated the
# overlap between Role_Score and Action V. Role_Score's underlying
# metric weights already include heavy contributions from the same
# signals that build Total Value/90:
#   - npxG → Shooting Value           (r ≈ 0.95 per GPA explainer)
#   - xAOP → Passing/Receiving Value
#   - xTOP → Passing/Receiving Value
#   - Progressive Passes → Passing Value
#   - Dribbles successful → Dribbling Value
# So weighting Action V at 40-50% (the old ST/AM_WG weights) was
# largely double-counting the same chance-creation/progression signal.
# The new weights keep Action V as a sanity-check cross-validator but
# make Role_Score the dominant input — which is what role-fit
# scouting at the lower-division level actually rewards.
#
# GK still gets the most extreme tilt because the GPA explainer
# (Part VI) shows Action V is a particularly weak signal for keepers
# (Shot-Stopping Value within-pos r ≈ 0.03 within a single season).

CVI_PERF_WEIGHTS = {
    'GK':    (0.90, 0.10),   # (role, action_v)
    'CB':    (0.85, 0.15),
    'FB':    (0.80, 0.20),
    'CM':    (0.80, 0.20),
    'AM_WG': (0.75, 0.25),
    'ST':    (0.75, 0.25),
}

# AgeValueMultiplier(age, position) — NPV of remaining career value
# (v2.5).
#
# Model: a player's age multiplier = the sum of expected future
# performance years from their current age until career end. The
# multiplier strictly decreases with age because every year you age,
# you lose one year of remaining career.
#
# Captures the four mechanisms the user identified:
#   1. Projected rate of perf IMPROVEMENT  → youth_baseline → 1.0 by peak_age
#   2. Years REMAINING before decline      → peak_age → decline_start
#   3. Projected rate of perf DECLINE      → decline_start → career_end
#   4. Total career value                  → integral of the above
#
# Three-phase performance trajectory at any future age:
#   age < peak_age:        linear growth from youth_baseline at 16 to 1.0
#   peak_age ≤ age < decline_start: flat at 1.0
#   decline_start ≤ age < career_end: linear decline from 1.0 to 0
#   age ≥ career_end:      0
#
# Remaining career value at age A = ∫ perf(t) dt from t=A to career_end
# (approximated by sum across integer year boundaries, linearly
# interpolated for fractional ages).
#
# Final multiplier:
#   m(A) = old_floor + (max_mult − old_floor) × (rcv(A) / rcv(16))
#
# Anchored to rcv(16) so the multiplier hits max_mult at age 16 and
# old_floor at career_end. Strictly monotone non-increasing across
# all ages.
#
# Per-position trajectory parameters (best evidence from CIES +
# market analyses):
#   GK     peak 28, decline 33, end 39  — longest career, latest decline
#   CB     peak 27, decline 31, end 36
#   CM     peak 26, decline 30, end 35
#   FB     peak 25, decline 29, end 33
#   ST     peak 25, decline 28, end 33
#   AM_WG  peak 24, decline 27, end 32  — pace-dependent, earliest end
#
# Sample multipliers for ST (peak 25, decline 28, end 33):
#   16yo → 1.80  ~17 years of remaining perf; wonderkid premium
#   21yo → 1.38  Approaching peak, still 12 years
#   25yo → 0.92  At perf peak, 8 years left
#   28yo → 0.51  Decline starts now, 5 years
#   30yo → 0.26  Mid-decline, 3 years
#   33yo → 0.10  Floor
#
# Same perf=70:
#   16yo CVI 126 vs 25yo CVI 65   → 2× premium for the wonderkid
#   16yo CVI 126 vs 30yo CVI 18   → 7× premium
CVI_AGE_VALUE_PARAMS = {
    # v2.7 — further compressed. v2.6 still felt too age-heavy in
    # practice. Tightened the range from max~1.55 / floor~0.40
    # (~4× spread) to max~1.30 / floor~0.55 (~2.4× spread). Performance
    # now strongly dominates the final CVI; age is a meaningful but
    # subordinate modifier. Career-NPV shape unchanged.
    'GK':    {'peak_age': 28, 'decline_start': 33, 'career_end': 39,
              'max_mult': 1.20, 'old_floor': 0.60, 'youth_baseline': 0.55},
    'CB':    {'peak_age': 27, 'decline_start': 31, 'career_end': 36,
              'max_mult': 1.28, 'old_floor': 0.55, 'youth_baseline': 0.50},
    'CM':    {'peak_age': 26, 'decline_start': 30, 'career_end': 35,
              'max_mult': 1.28, 'old_floor': 0.55, 'youth_baseline': 0.50},
    'FB':    {'peak_age': 25, 'decline_start': 29, 'career_end': 33,
              'max_mult': 1.30, 'old_floor': 0.55, 'youth_baseline': 0.50},
    'ST':    {'peak_age': 25, 'decline_start': 28, 'career_end': 33,
              'max_mult': 1.32, 'old_floor': 0.55, 'youth_baseline': 0.50},
    'AM_WG': {'peak_age': 24, 'decline_start': 27, 'career_end': 32,
              'max_mult': 1.35, 'old_floor': 0.55, 'youth_baseline': 0.50},
}


def _cvi_expected_perf_at(age, params):
    """Expected normalized performance at given age (0..1).
    Three-phase: youth growth → peak plateau → linear decline → 0."""
    if age < 16:
        return params['youth_baseline']
    if age >= params['career_end']:
        return 0.0
    if age < params['peak_age']:
        yb = params['youth_baseline']
        return yb + (1.0 - yb) * (age - 16) / (params['peak_age'] - 16)
    if age < params['decline_start']:
        return 1.0
    decline_yrs = params['career_end'] - params['decline_start']
    if decline_yrs <= 0:
        return 0.0
    return max(1.0 - (age - params['decline_start']) / decline_yrs, 0.0)


def _cvi_cum_remaining_career(age_int, params):
    """Sum of expected perf from int(age) up to career_end−1
    (integer-year boundaries)."""
    ce = int(params['career_end'])
    if age_int >= ce:
        return 0.0
    start = max(int(age_int), 16)
    return sum(_cvi_expected_perf_at(t, params) for t in range(start, ce))


# Pre-compute max remaining career value at age 16 per position so
# the multiplier hits max_mult exactly at age 16. Computed at module
# load time; safe because CVI_AGE_VALUE_PARAMS is fixed.
_CVI_MAX_CAREER_VALUE = {
    pos: _cvi_cum_remaining_career(16, p)
    for pos, p in CVI_AGE_VALUE_PARAMS.items()
}

# ---- ReliabilityWeight ----
# Replaces the naive `min(mins/1800, 1.0)` ramp from CVI v1.
#
# Grounded in the empirical per-position stability table from the GPA
# v2 explainer (reports/gpa_explainer.pdf, Part VI; raw data at
# models/validation/stability_by_minutes.csv). The headline finding:
# within-position YoY r for Total Value differs by ~5× across positions
# at the same sample size:
#
#   pos      within-pos YoY r @ 900 min (Total Value)
#   ─────    ────────────────────────────────────────
#   CM       0.64    ← most stable outfield position
#   FB       0.41
#   STRIKER  0.32
#   AM_WG    ~0.20   (winger 0.13, attmid sparse sample)
#   CB       0.19
#   GK       0.12    ← V-metrics need ~2 seasons per explainer Part VII
#
# Two implications a single linear ramp can't capture:
#  1. ASYMPTOTIC CEILING differs by position. A CB rated on event-only V
#     metrics has a structural noise floor (defensive valuation is hard
#     per Visual 4). No amount of minutes makes CB Total Value as
#     reliable as CM Total Value. The ceiling reflects that.
#  2. TIME-TO-CEILING differs by position. GK needs ~2 full seasons
#     for shot-stopping / handling / sweeping to stabilize. Outfielders
#     reach near-ceiling around 1500-1800 min.
#
# These ceilings ARE NOT raw YoY r values — they're translated into
# 0-1 weights via the psychometric convention also used in the explainer:
#   r ≥ 0.7 ≈ trustworthy standalone (weight = 1.0)
#   r ≈ 0.5 ≈ useful as composite input (weight ≈ 0.7)
#   r < 0.3 ≈ noise floor (weight ≈ 0.2)
# Then bumped upward by the Spearman-Brown effect of CVI being a
# composite (Role_Score blends 15-25 weighted metrics; n_eff ≈ 3
# accounting for inter-metric correlation), which lifts a within-pos
# r=0.45 composite-input into an effective ~0.71 — putting a typical
# outfielder near ceiling=0.95 once they have full minutes.
CVI_RELIAB_CEILING_BY_POS = {
    'GK':    0.70,   # event-V can't fully measure shot-stopping; need 2 seasons
    'CB':    0.85,   # defensive valuation hard (explainer Visual 4 hybrid)
    'AM_WG': 0.90,
    'FB':    0.92,
    'ST':    0.92,
    'CM':    0.95,   # most metrics in the role composite are stable
}
CVI_RELIAB_CEILING_DEFAULT = 0.85

# Minutes at which we approximately reach the position's ceiling
# (≈95% there; the curve smoothly saturates above this).
CVI_RELIAB_MINS_TO_CEILING = {
    'GK':    3600,   # ~2 full seasons per explainer Part VII
    'CB':    2100,
    'AM_WG': 1800,
    'FB':    1800,
    'ST':    1800,
    'CM':    1500,
}
CVI_RELIAB_MINS_TO_CEILING_DEFAULT = 1800

# NOTE on the (now-removed) very-short-sample floor:
# v1 had a linear floor below 270 min so small-sample players didn't
# collapse to reliab=0 (which would have killed their CVI via
# multiplication). With v2.0's empirical-Bayes shrinkage the
# justification went away — even a 0-reliability player gets shrunk
# to a sensible prior (their career mean if known, 40 if not), never
# to 0. The floor was also creating a JUMP of ~25 percentage points
# at the 270-min boundary where the floor formula and the saturating
# curve didn't meet smoothly. Removed in v2.1.

# ---- Shrinkage prior ----
# When reliability is low, we don't shrink the rating to ZERO — we
# shrink it toward a "replacement-level" prior (the freely-available
# player a club could sign tomorrow). Statistically this is empirical
# Bayes: with low sample, weight the prior more; with high sample,
# weight the observation more.
#
#   shrunk_perf = reliab × raw_perf + (1 − reliab) × replacement_perf
#
# Why 40 on the 0-100 scale? PerformanceQuality is a within-position
# percentile blend, so "40" literally means "≈40th-percentile player
# within this position group" — a marginal starter / quality bench
# player at this tier. That's the conventional sabermetric definition
# of replacement level (the worst player a competitive team would
# field), borrowed from Baseball Prospectus's VORP and FanGraphs' WAR.
#
# Effect on the math:
#   • A 70-rated CB with 300 min (reliab=0.27):
#       shrunk = 0.27×70 + 0.73×40 = 48.1
#       (vs old: CVI_perf×reliab = 70×0.27 = 18.9 — punished too hard)
#   • A 70-rated CB with 1800 min (reliab=0.79):
#       shrunk = 0.79×70 + 0.21×40 = 63.7
#       (already close to the observed rating)
#   • A 25-rated player with 300 min (reliab=0.27):
#       shrunk = 0.27×25 + 0.73×40 = 35.9
#       (low ratings ALSO pulled toward replacement — we don't
#       overreact to a small sample of bad performances either)
CVI_REPLACEMENT_PERF = 40.0

# Pure tier-level league multipliers, anchored to the empirical
# mover-based cross-tier ratio from the cross-tier analysis (see
# git history around 2026-05: 41 Camp→L3 movers showed ~0.80×
# V/90 in L3 vs Camp; 271 L3→Camp movers showed ~0.89× L3/Camp;
# the all-movers median of 0.88 weighted by sample sizes lands
# at ~0.85 once selection bias on the upward-mover side is folded
# in). We deliberately do NOT layer team-strength-within-tier on
# top — GPA Total Value already encodes team context per action,
# so a team multiplier would double-count it.
#
# Reference frame: Liga 3 = 1.0 (baseline).
CVI_LEAGUE_MULTIPLIER = {
    43324: 1.00,    # Liga 3
    702:   0.85,    # Campeonato de Portugal
}
# Fallback for any competition not in the dict (won't normally fire
# since the dashboard's data scope is Liga 3 + Camp only).
CVI_LEAGUE_DEFAULT = 1.0


# v2.8 — position-specific multiplier applied to the CVI→EUR mapping
# (bio "Projected value" cell). Literature-grounded priors compressed
# toward 1.0 for Liga 3 reality.
#
# Sources informing the magnitudes:
#   • CIES (Poli, Besson & Ravenel 2022, Economies 10/1/4) — standardized
#     experience betas: forwards 0.934 > mid 0.793 > CB 0.749 > FB 0.606
#     > GK 0.407. Ratio GK/FW ≈ 0.44 in Big-5.
#   • Müller, Simons & Weinmann (2017, EJOR) — position random-effect SD
#     0.050 on log-MV (~±5% spread once age/perf/club/league controlled).
#   • Franceschi, Brocard, Follert & Gouguet (2024, JoES 38(3)) — review
#     of 29 papers / 111 specs: directional ordering ST > AM/WG > CM >
#     CB > FB > GK is robust; FB shows the most-negative coef vs CF.
#   • Frick (2007, SJPE) — GK pay penalty mechanism (low role flexibility).
#   • Garcia-del-Barrio & Pujol (2007, MDE) — attacker premium driven by
#     crowd-pulling capacity — a mechanism MUCH weaker in Liga 3, so we
#     deliberately compress the GK discount toward 0.70 (not Big-5's ~0.50).
POSITION_EUR_MULTIPLIER = {
    'ST':    1.30,
    'AM_WG': 1.25,
    'CM':    1.00,
    'CB':    0.90,
    'FB':    0.85,
    'GK':    0.70,
}

# v2.9 — extra Campeonato discount on the EUR side. CVI already uses
# league_factor 0.85 for Camp inside the score itself, but the user wants
# Camp projected prices nudged down further to reflect that even an
# "equivalent CVI" Camp player commands a lower real fee at sale (smaller
# scout footprint, less liquid market, lower buyer competition). Combined
# with the in-CVI 0.85, a Camp player at the same raw inputs as a Liga 3
# player ends up at ~0.85 × 0.85 ≈ 72% of the Liga 3 projected EUR.
CAMP_PROJECTED_EUR_PENALTY = 0.85

# v2.10 — steeper top + lower mid/bottom. User: "top line should stay
# similar while the middle and bottom drop off a little." Higher
# exponent (2.55 → 2.70) widens the spread between bottom and top;
# coefficient anchored (1.10) so CVI 110 stays at ~€355k (matches v2.9).
# Effect vs v2.9 (CM, Liga 3):
#   CVI 40   €27k → €23k   (−14%)
#   CVI 60   €76k → €69k   (−9%)
#   CVI 80   €159k → €151k (−5%)
#   CVI 100  €281k → €275k (−2%)
#   CVI 110  €358k → €355k (anchor)
#   CVI 120  €447k → €450k (+1%)
PROJECTED_EUR_COEF = 1.10
PROJECTED_EUR_EXP  = 2.70
PROJECTED_EUR_CAP  = None   # cap removed 2026-06-23 (Lucas) — was 500_000. Only
                            # 1 rated player was pinned at it (max uncapped ~€431k),
                            # so it was a near-inert safety rail. Set back to a number
                            # (e.g. 500_000) to re-enable the min() clamp.


def cvi_to_projected_eur(cvi, position_group=None, competition_id=None):
    """Convert a CVI score to a projected EUR figure: power curve +
    position multiplier + Camp penalty. Cap removed 2026-06-23 (the
    €500k clamp is now opt-in via PROJECTED_EUR_CAP). Returns None if
    cvi is None/<=0."""
    try:
        v = float(cvi)
    except (TypeError, ValueError):
        return None
    if not (v > 0):
        return None
    pos_mult = POSITION_EUR_MULTIPLIER.get(position_group, 1.00)
    camp_mult = (CAMP_PROJECTED_EUR_PENALTY
                  if competition_id is not None
                  and not (isinstance(competition_id, float) and pd.isna(competition_id))
                  and int(competition_id) == 702
                  else 1.00)
    val = (PROJECTED_EUR_COEF * (v ** PROJECTED_EUR_EXP)
            * pos_mult * camp_mult)
    return val if PROJECTED_EUR_CAP is None else min(val, PROJECTED_EUR_CAP)


def _cvi_position_group(primary_position):
    """Map Wyscout primaryPosition to a CVI position-group key
    (matches keys in CVI_PERF_WEIGHTS / CVI_AGE_VALUE_PARAMS)."""
    if primary_position is None:
        return None
    try:
        if pd.isna(primary_position):
            return None
    except (TypeError, ValueError):
        pass
    p = str(primary_position)
    if p == 'GK': return 'GK'
    if p in ('CB', 'LCB', 'RCB', 'LCB3', 'RCB3'): return 'CB'
    if p in ('LB', 'RB', 'LB5', 'RB5', 'LWB', 'RWB'): return 'FB'
    if p in ('CMF', 'LCMF', 'RCMF', 'LCMF3', 'RCMF3',
             'DMF', 'LDMF', 'RDMF'): return 'CM'
    if p in ('AMF', 'LAMF', 'RAMF', 'LMF', 'RMF',
             'LW', 'RW', 'LWF', 'RWF'): return 'AM_WG'
    if p in ('CF', 'SS'): return 'ST'
    return None


def _cvi_reliability_weight(mins, position_group):
    """Position-aware reliability weight grounded in within-position
    YoY r data from the GPA explainer (Part VI).

    Shape:
      mins < 270:        linear ramp 0 → FLOOR (0.15)
      mins ≥ 270:        ceiling(pos) × (1 − exp(−3 × mins / mins_to_ceiling(pos)))
                          which reaches ~95% of ceiling at mins_to_ceiling
                          and asymptotes toward ceiling above that

    Returns a tuple (weight, breakdown_dict) where breakdown_dict has the
    raw ceiling, sample_factor (the 0..1 saturating curve value before
    multiplying by ceiling), and mins_to_ceiling — for surfacing in the
    UI so users can audit why a player got 0.65 vs 1.0.
    """
    import math
    if mins is None:
        return 0.0, {'ceiling': None, 'sample_factor': 0.0, 'mins_to_ceiling': None}
    try:
        if pd.isna(mins):
            return 0.0, {'ceiling': None, 'sample_factor': 0.0, 'mins_to_ceiling': None}
        mins = float(mins)
    except Exception:
        return 0.0, {'ceiling': None, 'sample_factor': 0.0, 'mins_to_ceiling': None}
    if mins <= 0:
        return 0.0, {'ceiling': None, 'sample_factor': 0.0, 'mins_to_ceiling': None}
    ceiling = CVI_RELIAB_CEILING_BY_POS.get(position_group, CVI_RELIAB_CEILING_DEFAULT)
    mins_full = CVI_RELIAB_MINS_TO_CEILING.get(position_group,
                                                  CVI_RELIAB_MINS_TO_CEILING_DEFAULT)
    # Smooth saturating curve from 0 — no discontinuity. Combined with
    # the v2.0 empirical-Bayes prior, a very-low-sample player no longer
    # collapses to 0; they're shrunk toward their career prior (or 40
    # for a debutant). At mins=100 the weight is ~0.17, at 270 it's
    # ~0.40, at mins_full it's ~0.95 — continuous everywhere.
    sample_factor = 1.0 - math.exp(-3.0 * mins / mins_full)
    weight = ceiling * sample_factor
    return weight, {'ceiling': ceiling, 'sample_factor': sample_factor,
                     'mins_to_ceiling': mins_full}


def _cvi_age_value_multiplier(age, position_group):
    """NPV-of-remaining-career age multiplier. Returns 1.0 if inputs
    can't be evaluated (so missing age doesn't tank the CVI).

    Sums the player's expected future performance from current age to
    career_end, normalizes against the value at age 16. Result is
    strictly non-increasing in age — same raw_perf, younger always
    wins, with the magnitude reflecting how many productive years
    they have left.

    Captures: rate of perf improvement (youth → peak), years before
    decline, rate of perf decline, and total career horizon. See
    CVI_AGE_VALUE_PARAMS docstring for parameters.

    For fractional ages, linearly interpolates between integer-year
    cumulative values so the curve is smooth (no step jumps).
    """
    import math
    if age is None or position_group not in CVI_AGE_VALUE_PARAMS:
        return 1.0
    try:
        a = float(age)
        if pd.isna(a):
            return 1.0
    except (TypeError, ValueError):
        return 1.0
    p = CVI_AGE_VALUE_PARAMS[position_group]
    if a >= p['career_end']:
        return p['old_floor']
    lo = int(math.floor(a))
    hi = lo + 1
    f = a - lo
    rcv_lo = _cvi_cum_remaining_career(lo, p)
    rcv_hi = _cvi_cum_remaining_career(hi, p)
    rcv = rcv_lo * (1.0 - f) + rcv_hi * f
    max_rcv = _CVI_MAX_CAREER_VALUE.get(position_group, 1.0)
    if max_rcv <= 0:
        return p['old_floor']
    norm = max(0.0, min(rcv / max_rcv, 1.0))
    return p['old_floor'] + (p['max_mult'] - p['old_floor']) * norm


def compute_cvi_columns(player_stats_df, *, age_lookup,
                         comp_id_lookup=None,
                         opta_team_strength_lookup=None,   # deprecated, kept for compat
                         team_col='teamName',
                         prior_lookup=None,
                         weights=None, position_groups=None):
    """Compute CVI + its components for every row in player_stats_df.

    Args:
        player_stats_df: DataFrame from calculate_player_percentiles_and_scores
        weights / position_groups: the radar role templates (config.yaml
            'weights' / 'position_groups'). app.py passes its loaded copies;
            None reads config.yaml via _radar_templates() (standalone use).
            (must have primaryPosition, totalMinutes, all {role}_Score
            columns, and 'Total Value' from GPA merge).
        age_lookup: callable playerId -> age in years (or None).
        comp_id_lookup: callable playerId -> competitionId (43324 or 702).
            If None, falls back to the player_stats_df's competitionId
            column if present, else assumes Liga 3 (1.0×).
        opta_team_strength_lookup: deprecated. Earlier versions used team
            Opta strength to scale LeagueMultiplier within a tier; we
            dropped that to avoid double-counting team context which is
            already encoded in GPA Total Value per action. Argument
            kept so call sites don't break.
        team_col: column in player_stats_df with the team name (still
            used for joining / display, but not for CVI math anymore).

    Returns:
        DataFrame with the same index as input plus columns:
            _CVI            — final composite (0–~150 typical)
            _CVI_perf       — raw PerformanceQuality (0-100)
            _CVI_perf_shrunk — shrunk toward replacement-level prior
                              (this is what actually feeds into CVI)
            _CVI_age        — AgeValueMultiplier (0.4-1.6)
            _CVI_reliab     — ReliabilityWeight (0-1), position-aware;
                              this is the *shrinkage weight*, not a
                              multiplier on perf anymore
            _CVI_reliab_ceiling          — asymptotic max for this pos
            _CVI_reliab_sample_factor    — 0..1 sample-driven curve value
            _CVI_reliab_mins_to_ceiling  — min count for ~95% of ceiling
            _CVI_league     — LeagueMultiplier (0.85 or 1.0)
            _CVI_trajectory — shrunk_perf - same-age-position-median
                              (a "+30 flag" surfaced separately, NOT
                              applied to CVI)
    """
    if weights is None or position_groups is None:
        _w, _pg = _radar_templates()
        weights = weights if weights is not None else _w
        position_groups = position_groups if position_groups is not None else _pg
    if player_stats_df is None or player_stats_df.empty:
        return pd.DataFrame()

    df = player_stats_df.copy()

    # ---- BULLETPROOF DTYPE COERCION AT ENTRY ----
    # The dashboard's upstream pipelines (especially the cross-tier Liga 3
    # + Campeonato merges) can leak object-dtype columns containing
    # sentinel strings like '—', '-', 'N/A', or even mixed int/str
    # values. Any of those would later blow up a sort/rank/between/clip
    # comparison with the cryptic "'>=' not supported between str and
    # float" TypeError. Coerce every column we will compare/sort here.
    #
    # primaryPosition: leave as object (string-keyed map below) but force
    # to string so int values from a broken merge don't trip _cvi_position_group.
    if 'primaryPosition' in df.columns:
        df['primaryPosition'] = df['primaryPosition'].apply(
            lambda v: str(v) if v is not None and not pd.isna(v) else None
        )
    # Total Value (the V/90 column we rank): force numeric.
    if 'Total Value' in df.columns:
        df['Total Value'] = pd.to_numeric(df['Total Value'], errors='coerce')
    # totalMinutes drives reliability — same defensive coercion.
    if 'totalMinutes' in df.columns:
        df['totalMinutes'] = pd.to_numeric(df['totalMinutes'], errors='coerce')
    # All <Role>_Score columns we read in _best_role_score.
    for _sc in [c for c in df.columns if c.endswith('_Score')]:
        df[_sc] = pd.to_numeric(df[_sc], errors='coerce')

    # Map position to CVI group + age. Coerce age to numeric — for some
    # Campeonato player-seasons the age lookup may return a string
    # (e.g. an unparsed birthDate) which would later blow up the
    # df['_cvi_age'].between(a-2, a+2) call inside _expected_perf with
    # "TypeError: '>=' not supported between str and float".
    df['_cvi_group'] = df['primaryPosition'].map(_cvi_position_group)
    df['_cvi_age'] = pd.to_numeric(df['playerId'].map(age_lookup),
                                     errors='coerce')

    # ---- PerformanceQuality (Role component) ----
    # v1.9 — versatility-aware aggregation. Pure max threw away the
    # signal that a player good across multiple eligible roles is
    # more flexible (and hence more valuable in the transfer market)
    # than a one-role specialist at the same peak.
    #
    # Formula:
    #   role_score = α × max(eligible_role_scores)
    #              + (1 − α) × mean(eligible_role_scores)
    #
    # α = 0.6 → 60% best role + 40% mean across all eligible roles.
    # Worked examples (a CF/SS eligible for 5 striker roles):
    #
    #   Player type            scores              old (max)  new (0.6/0.4)
    #   ────────────────────── ──────────────────  ─────────  ──────────────
    #   Specialist Poacher     [80, 30, 30, 30, 30]   80        64
    #   Versatile #9           [70, 65, 60, 50, 40]   70        64.8
    #   Compleat striker       [70, 70, 70, 70, 70]   70        70
    #
    # The compleat striker now wins — what scouts intuit. The
    # specialist takes a bigger hit because their non-Poacher numbers
    # really are weak (and a Mourinho would pay for an all-rounder
    # over a one-trick pony at the same headline). α=0.6 is a starting
    # point; can be tuned off the reported transfer fees.
    CVI_ROLE_VERSATILITY_ALPHA = 0.6  # weight on max vs mean

    def _role_score_blend(row):
        pos = row.get('primaryPosition')
        eligible = [r for r in weights if pos in position_groups.get(r, [])]
        vals = []
        for r in eligible:
            v = row.get(f"{r}_Score")
            try:
                if v is not None and not pd.isna(v):
                    vals.append(float(v))
            except Exception:
                pass
        if not vals:
            return None
        if len(vals) == 1:
            return vals[0]   # single-role case → no blending needed
        a = CVI_ROLE_VERSATILITY_ALPHA
        return a * max(vals) + (1.0 - a) * (sum(vals) / len(vals))

    df['_cvi_role_score'] = df.apply(_role_score_blend, axis=1)

    # Action V percentile within position group — rank Total Value
    # within same _cvi_group so a 0.05 V/90 striker isn't compared
    # against a 0.005 V/90 CB.
    # Defensive coercion: in cross-tier merges the 'Total Value' column
    # can come in as object dtype (string '—' for missing Camp rows
    # alongside floats for matched rows). rank() then raises
    # "'>=' not supported between str and float". Force numeric first.
    val_col = 'Total Value' if 'Total Value' in df.columns else None
    if val_col:
        df[val_col] = pd.to_numeric(df[val_col], errors='coerce')
        df['_cvi_av_pct'] = (df.groupby('_cvi_group')[val_col]
                              .rank(pct=True, method='average') * 100.0)
    else:
        df['_cvi_av_pct'] = None

    # Same risk for totalMinutes (some pipelines stash it as object).
    if 'totalMinutes' in df.columns:
        df['totalMinutes'] = pd.to_numeric(df['totalMinutes'], errors='coerce')

    def _perf_quality(row):
        g = row.get('_cvi_group')
        if g not in CVI_PERF_WEIGHTS:
            return None
        w_role, w_av = CVI_PERF_WEIGHTS[g]
        role = row.get('_cvi_role_score')
        av = row.get('_cvi_av_pct')
        # Coerce both to float (or None) up-front so we can't return a
        # weird type that breaks downstream sorts/comparisons.
        try:
            role_f = (float(role) if role is not None
                       and not pd.isna(role) else None)
        except Exception:
            role_f = None
        try:
            av_f = (float(av) if av is not None
                     and not pd.isna(av) else None)
        except Exception:
            av_f = None
        # Fall back gracefully when one side is missing — re-weight
        # so the score still uses the other side at full weight.
        if av_f is None and role_f is None:
            return None
        if av_f is None:
            return role_f
        if role_f is None:
            return av_f
        return w_role * role_f + w_av * av_f

    df['_CVI_perf'] = pd.to_numeric(df.apply(_perf_quality, axis=1),
                                      errors='coerce')

    # ---- AgeValueMultiplier ----
    df['_CVI_age'] = df.apply(
        lambda r: _cvi_age_value_multiplier(r.get('_cvi_age'), r.get('_cvi_group')),
        axis=1,
    )

    # ---- ReliabilityWeight ----
    # Position-aware empirical curve (see CVI_RELIAB_* constants above).
    if 'totalMinutes' in df.columns:
        _reliab_results = df.apply(
            lambda r: _cvi_reliability_weight(r.get('totalMinutes'), r.get('_cvi_group')),
            axis=1,
        )
        df['_CVI_reliab'] = _reliab_results.apply(lambda t: t[0])
        df['_CVI_reliab_ceiling'] = _reliab_results.apply(
            lambda t: t[1].get('ceiling'))
        df['_CVI_reliab_sample_factor'] = _reliab_results.apply(
            lambda t: t[1].get('sample_factor'))
        df['_CVI_reliab_mins_to_ceiling'] = _reliab_results.apply(
            lambda t: t[1].get('mins_to_ceiling'))
    else:
        df['_CVI_reliab'] = 1.0
        df['_CVI_reliab_ceiling'] = None
        df['_CVI_reliab_sample_factor'] = None
        df['_CVI_reliab_mins_to_ceiling'] = None

    # ---- LeagueMultiplier ----
    # Pure tier-level: 1.0 for Liga 3, 0.85 for Campeonato. The
    # comp_id_lookup callable wins if provided; otherwise we use
    # competitionId from the player_stats_df if present; otherwise
    # the conservative default of 1.0 (we'd rather not penalize a
    # player if we can't classify their tier).
    if comp_id_lookup is not None and 'playerId' in df.columns:
        comps = df['playerId'].map(comp_id_lookup)
    elif 'competitionId' in df.columns:
        comps = df['competitionId']
    else:
        comps = pd.Series([None] * len(df), index=df.index)
    df['_CVI_league'] = comps.map(
        lambda c: CVI_LEAGUE_MULTIPLIER.get(int(c), CVI_LEAGUE_DEFAULT)
                   if c is not None and not pd.isna(c) else CVI_LEAGUE_DEFAULT
    )

    # ---- Empirical-Bayes shrinkage toward player-specific prior ----
    # v2.0 — instead of always shrinking toward the generic
    # replacement-level (40), shrink toward THIS PLAYER's career prior
    # when we have rich prior-season data. A 1350-min season from a
    # player with 2400 effective prior minutes shouldn't be discounted
    # toward generic replacement — we know who he is.
    #
    # Formula:
    #   prior_strength  = min(prior_mins_eff / 1500, 1.0)
    #   effective_prior = prior_strength × player_career_perf
    #                     + (1 − prior_strength) × CVI_REPLACEMENT_PERF
    #   shrunk_perf     = season_reliability × raw_perf
    #                     + (1 − season_reliability) × effective_prior
    #
    # With no prior data: effective_prior = 40 (falls back to v1.7
    # behavior — debutants get the generic replacement target).
    # With strong prior data: effective_prior = player's own career
    # mean → the shrinkage just regresses toward what we already
    # believe about the player, not toward a generic floor.
    def _shrink_perf(raw_perf, reliab, prior_info):
        if raw_perf is None or pd.isna(raw_perf):
            return None
        if reliab is None or pd.isna(reliab):
            return float(raw_perf)
        # Resolve effective prior using player-specific info if present.
        if prior_info is None:
            effective_prior = CVI_REPLACEMENT_PERF
        else:
            p_perf = prior_info.get('prior_perf')
            p_strength = prior_info.get('prior_strength', 0.0) or 0.0
            if p_perf is None or pd.isna(p_perf):
                effective_prior = CVI_REPLACEMENT_PERF
            else:
                effective_prior = (p_strength * float(p_perf)
                                    + (1.0 - p_strength) * CVI_REPLACEMENT_PERF)
        w = float(reliab)
        return w * float(raw_perf) + (1.0 - w) * effective_prior

    df['_CVI_perf_shrunk'] = df.apply(
        lambda r: _shrink_perf(
            r.get('_CVI_perf'),
            r.get('_CVI_reliab'),
            prior_lookup(r.get('playerId')) if callable(prior_lookup) else None,
        ),
        axis=1,
    )
    # Expose the prior used so the UI can surface "shrunk toward 70"
    # vs "shrunk toward 40 (debutant)" instead of always saying 40.
    if callable(prior_lookup):
        _prior_resolved = df['playerId'].apply(
            lambda pid: prior_lookup(pid) if pid is not None else None
        )
        df['_CVI_prior_perf'] = _prior_resolved.apply(
            lambda x: x.get('prior_perf') if isinstance(x, dict) else None
        )
        df['_CVI_prior_strength'] = _prior_resolved.apply(
            lambda x: x.get('prior_strength') if isinstance(x, dict) else None
        )
        df['_CVI_prior_mins_eff'] = _prior_resolved.apply(
            lambda x: x.get('prior_mins_eff') if isinstance(x, dict) else None
        )
        # Effective shrinkage target = blended prior actually used.
        def _effective_prior_for_row(info):
            if not isinstance(info, dict) or info.get('prior_perf') is None:
                return CVI_REPLACEMENT_PERF
            s = info.get('prior_strength', 0.0) or 0.0
            return s * float(info['prior_perf']) + (1 - s) * CVI_REPLACEMENT_PERF
        df['_CVI_effective_prior'] = _prior_resolved.apply(_effective_prior_for_row)
    else:
        df['_CVI_prior_perf'] = None
        df['_CVI_prior_strength'] = None
        df['_CVI_prior_mins_eff'] = None
        df['_CVI_effective_prior'] = CVI_REPLACEMENT_PERF

    # ---- Final composite ----
    # Note: _CVI_reliab is now BAKED INTO _CVI_perf_shrunk (it's the
    # shrinkage weight); we no longer multiply by it again.
    df['_CVI'] = (df['_CVI_perf_shrunk']
                   * df['_CVI_age']
                   * df['_CVI_league'])

    # ---- Trajectory flag (separate, NOT multiplied into CVI) ----
    # Median shrunk PerformanceQuality among same-position-group
    # same-age-band players. Using shrunk perf (not raw) keeps the
    # comparison apples-to-apples — both numerator and denominator
    # reflect the same sample-discount treatment.
    def _expected_perf(row):
        g = row.get('_cvi_group')
        a = row.get('_cvi_age')
        if g is None or a is None or pd.isna(a):
            return None
        peers = df[
            (df['_cvi_group'] == g)
            & (df['_cvi_age'].between(a - 2, a + 2))
            & df['_CVI_perf_shrunk'].notna()
        ]
        if len(peers) < 10:
            return None
        return float(peers['_CVI_perf_shrunk'].median())

    df['_cvi_expected_perf'] = df.apply(_expected_perf, axis=1)
    df['_CVI_trajectory'] = df['_CVI_perf_shrunk'] - df['_cvi_expected_perf']

    return df[['_CVI', '_CVI_perf', '_CVI_perf_shrunk', '_CVI_age',
                '_CVI_reliab', '_CVI_reliab_ceiling',
                '_CVI_reliab_sample_factor', '_CVI_reliab_mins_to_ceiling',
                '_CVI_prior_perf', '_CVI_prior_strength',
                '_CVI_prior_mins_eff', '_CVI_effective_prior',
                '_CVI_league', '_CVI_trajectory']]


# ==============================================================================
# Career CVI (cross-season + cross-league aggregation)
# ------------------------------------------------------------------------------
# Single-season CVI answers "how good was this player in 2024/25?".
# Career CVI answers "what's the durable estimate combining everything we
# know about this player up to and including season X?".
#
# Aggregation rules (anchored to chosen season; never uses future seasons):
#   1. For each prior season i (counting backwards from anchor):
#        decay_factor_i = CVI_CAREER_DECAY ** seasons_back_i
#        league_factor_i = CVI_LEAGUE_MULTIPLIER[comp_at_season_i]
#                          (translates Camp perf to Liga 3 equivalent;
#                           anchor-season league is applied AT THE END)
#        weight_i        = decay_factor_i × mins_i
#        contribution_i  = perf_i × league_factor_i × weight_i
#   2. career_perf_raw_l3 = Σ contribution_i / Σ weight_i
#   3. effective_mins     = Σ weight_i        (drives reliability shrinkage)
#   4. shrunk_perf        = reliab × career_perf_raw_l3
#                            + (1 − reliab) × CVI_REPLACEMENT_PERF
#      (reliab computed at the anchor season's position group + effective_mins)
#   5. age_at_anchor      = player_age at anchor season's start (Aug of year)
#   6. career_CVI = shrunk_perf × AgeValueMultiplier(age_at_anchor, pos)
#                                × CVI_LEAGUE_MULTIPLIER[league_at_anchor]
#
# Why "anchor at anchor season's league"?
#   The career_perf is now in Liga-3-equivalent units (we translated each
#   season's contribution). To finish in the right scale, we re-apply the
#   anchor season's league multiplier. So a career CVI anchored to a Camp
#   season gets the 0.85 final discount; anchored to a Liga 3 season does
#   not. This keeps Current CVI commensurate with the player's current
#   league context.
#
# Anchored never INCLUDES future seasons (we don't peek). When called for
# "Current CVI", anchor = the player's most recent season; for "Season
# CVI" inside a historical season's view, anchor = that selected season.
CVI_CAREER_DECAY = 0.5           # weighting per season back (steeper than v2.7's 0.6 — current season counts relatively more)
CVI_CAREER_MAX_LOOKBACK = 4      # seasons back included (0..4 = up to 5 seasons)
# v2.8 — current season gets an explicit bonus multiplier on top of decay.
# User: "weight the current season a little bit more". With CURRENT_BONUS=1.5
# and DECAY=0.5, the current season's recency weight is 3× the prior season's
# (1.5 vs 0.5). The per-season MINUTES weighting (mins_played × recency)
# already keeps small-sample seasons from dragging the avg down — this just
# tilts further toward "what they're doing RIGHT NOW".
CVI_CAREER_CURRENT_BONUS = 1.5


def _build_player_season_perf_table(gpa_values_df, player_minutes_df=None):
    """One row per (playerId, seasonId, competitionId) with:
       playerId, seasonId, competitionId, position_group, mins_played, perf_pct

    perf_pct is the player's Total Value /90 percentile WITHIN the same
    (seasonId × position_group) cohort — a sensible historical proxy for
    PerformanceQuality that doesn't require re-running the full role-score
    pipeline for every season.

    Returns empty DataFrame if GPA data is unavailable.
    """
    if gpa_values_df is None or gpa_values_df.empty:
        return pd.DataFrame()
    g = gpa_values_df.copy()
    # Pick the per-90 Total Value column (name varies between snapshots)
    val_col = next((c for c in ('Total Value', 'total_v_per_90',
                                  'Total Value_per_90')
                     if c in g.columns), None)
    if val_col is None:
        return pd.DataFrame()
    # Map raw position to CVI position group
    pos_col = next((c for c in ('position', 'primaryPosition')
                     if c in g.columns), None)
    if pos_col is None:
        return pd.DataFrame()
    g['_cvi_group'] = g[pos_col].map(_cvi_position_group)
    # Defensive numeric coercion (Camp/L3 cross-tier merges sometimes
    # leave val_col as object dtype which breaks rank() with the
    # str/float comparison error).
    g[val_col] = pd.to_numeric(g[val_col], errors='coerce')
    g = g.dropna(subset=['_cvi_group', val_col, 'seasonId', 'playerId'])
    # Within (seasonId, position_group), percentile-rank Total Value/90
    g['_perf_pct'] = (g.groupby(['seasonId', '_cvi_group'])[val_col]
                        .rank(pct=True, method='average') * 100.0)
    mins_col = next((c for c in ('mins_played', 'totalMinutes', 'Minutes')
                      if c in g.columns), None)
    if mins_col is None:
        # Fall back to player_minutes_df if provided
        if player_minutes_df is not None and not player_minutes_df.empty:
            pm = player_minutes_df[['playerId', 'totalMinutes']].rename(
                columns={'totalMinutes': '_mins_filled'}
            )
            g = g.merge(pm, on='playerId', how='left')
            mins_col = '_mins_filled'
        else:
            return pd.DataFrame()
    out = g[['playerId', 'seasonId', '_cvi_group', mins_col, '_perf_pct']].copy()
    out = out.rename(columns={mins_col: 'mins_played',
                                '_cvi_group': 'position_group',
                                '_perf_pct': 'perf_pct'})
    if 'competitionId' in g.columns:
        out['competitionId'] = g['competitionId'].values
    else:
        out['competitionId'] = out['seasonId'].map(competition_for_season)
    return out


def _season_year(season_id):
    """Numeric chronology key from SEASON_ID_MAP labels like '2024/25' → 2024.
    Used to order seasons and compute 'seasons back' from an anchor.
    """
    label = SEASON_ID_MAP.get(int(season_id)) if season_id is not None else None
    if not label:
        return None
    try:
        return int(str(label).split('/')[0])
    except (ValueError, IndexError):
        return None


def compute_career_cvi(player_id, anchor_season_id, *,
                        perf_table, dob_lookup,
                        decay=CVI_CAREER_DECAY,
                        max_lookback=CVI_CAREER_MAX_LOOKBACK):
    """Career-aggregated CVI anchored to anchor_season_id, including that
    season + up to `max_lookback` prior seasons (whichever the player has
    data for). Never peeks at seasons AFTER the anchor.

    Returns dict with:
        career_cvi, career_perf_raw (L3-equivalent), career_perf_shrunk,
        reliability, effective_mins, age_at_anchor, league_at_anchor,
        position_group, n_seasons_used, breakdown (list of per-season dicts)

    Returns None if the player has no GPA seasons at-or-before the anchor.
    """
    if perf_table is None or perf_table.empty or anchor_season_id is None:
        return None
    anchor_year = _season_year(anchor_season_id)
    if anchor_year is None:
        return None

    rows = perf_table[perf_table['playerId'] == player_id].copy()
    if rows.empty:
        return None
    rows['_season_year'] = rows['seasonId'].map(_season_year)
    rows = rows.dropna(subset=['_season_year'])
    rows['_season_year'] = rows['_season_year'].astype(int)
    # Only the anchor season + prior seasons, up to max_lookback back
    rows = rows[(rows['_season_year'] <= anchor_year)
                 & (rows['_season_year'] >= anchor_year - max_lookback)]
    if rows.empty:
        return None
    rows = rows.copy()   # slice of perf_table — write on a copy
    rows['_seasons_back'] = anchor_year - rows['_season_year']
    rows['_decay'] = decay ** rows['_seasons_back']
    # v2.8 — current season (seasons_back==0) gets an explicit recency bonus
    # on top of the decay-to-the-zero (which is 1.0). Prior seasons unaffected.
    rows['_recency'] = rows['_decay'].copy()
    rows.loc[rows['_seasons_back'] == 0, '_recency'] *= CVI_CAREER_CURRENT_BONUS
    rows['_league_factor'] = rows['competitionId'].map(
        lambda c: (CVI_LEAGUE_MULTIPLIER.get(int(c), CVI_LEAGUE_DEFAULT)
                    if c is not None and not pd.isna(c) else CVI_LEAGUE_DEFAULT)
    )
    rows['_weight'] = rows['_recency'] * rows['mins_played'].fillna(0).clip(lower=0)
    rows['_contribution'] = rows['perf_pct'] * rows['_league_factor'] * rows['_weight']

    total_w = float(rows['_weight'].sum())
    if total_w <= 0:
        return None
    career_perf_raw_l3 = float(rows['_contribution'].sum() / total_w)
    effective_mins = total_w   # decay-weighted effective minutes

    # Resolve the anchor row to lock down position group + league for the
    # final shrinkage/multiplier step. Use the most-recent matching season
    # ≤ anchor (handles the case where the player skipped the anchor year).
    anchor_row_candidates = rows.sort_values('_season_year', ascending=False)
    anchor_row = anchor_row_candidates.iloc[0]
    pos_group = anchor_row['position_group']
    league_at_anchor = (CVI_LEAGUE_MULTIPLIER.get(int(anchor_row['competitionId']),
                                                     CVI_LEAGUE_DEFAULT)
                         if anchor_row.get('competitionId') is not None
                         and not pd.isna(anchor_row.get('competitionId'))
                         else CVI_LEAGUE_DEFAULT)

    # Reliability from effective_mins under the anchor-position curve
    reliab, reliab_breakdown = _cvi_reliability_weight(effective_mins, pos_group)

    # Shrinkage toward replacement-level
    shrunk_perf = (reliab * career_perf_raw_l3
                    + (1 - reliab) * CVI_REPLACEMENT_PERF)

    # Age at anchor season (Aug 1 of anchor_year used as a reference date
    # so a player born in March looks "the right age" for that season)
    age_at_anchor = None
    try:
        dob = dob_lookup(player_id) if callable(dob_lookup) else None
        if dob is not None and not pd.isna(dob):
            from datetime import date as _date_cls
            anchor_ref = _date_cls(anchor_year, 8, 1)
            if hasattr(dob, 'date'):
                dob_d = dob.date()
            else:
                dob_d = dob
            age_at_anchor = (anchor_ref - dob_d).days / 365.25
    except Exception:
        age_at_anchor = None
    age_mult = _cvi_age_value_multiplier(age_at_anchor, pos_group)

    career_cvi = shrunk_perf * age_mult * league_at_anchor

    breakdown = (rows.sort_values('_season_year', ascending=False)
                       [['seasonId', '_season_year', '_seasons_back',
                          'competitionId', 'position_group', 'mins_played',
                          'perf_pct', '_league_factor', '_decay', '_weight']]
                       .rename(columns={'_season_year': 'season_year',
                                          '_seasons_back': 'seasons_back',
                                          '_league_factor': 'league_factor',
                                          '_decay': 'decay_factor',
                                          '_weight': 'weight'})
                       .to_dict('records'))

    return {
        'career_cvi': career_cvi,
        'career_perf_raw_l3': career_perf_raw_l3,
        'career_perf_shrunk': shrunk_perf,
        'reliability': reliab,
        'reliability_ceiling': reliab_breakdown.get('ceiling'),
        'reliability_sample_factor': reliab_breakdown.get('sample_factor'),
        'effective_mins': effective_mins,
        'age_at_anchor': age_at_anchor,
        'age_multiplier': age_mult,
        'league_at_anchor': league_at_anchor,
        'position_group': pos_group,
        'anchor_season_id': int(anchor_season_id),
        'n_seasons_used': int(len(rows)),
        'breakdown': breakdown,
    }


def build_player_priors_lookup(perf_table, anchor_season_id,
                                  decay=CVI_CAREER_DECAY,
                                  max_lookback=CVI_CAREER_MAX_LOOKBACK,
                                  full_strength_mins=1500):
    """Pre-compute the empirical-Bayes prior for every player relative
    to an anchor season. Returns a dict:
        {playerId: {'prior_perf': float, 'prior_strength': float (0..1),
                     'prior_mins_eff': float}}

    Used by compute_cvi_columns to shrink each player's season perf
    toward THEIR OWN career mean (when we have enough prior data)
    rather than the generic replacement-level (40). Implements the
    empirical-Bayes pattern: with rich prior data the shrinkage
    target IS the player's career; with no prior data we fall back
    to the league-replacement default.

    Strictly uses seasons PRIOR to anchor_season_id (excludes the
    anchor season itself) to avoid leakage: when judging Caleb's
    2024/25 perf, the prior is built from his 2021/22 + 2022/23 +
    2023/24 data only — never from 2024/25 itself or anything later.
    """
    if perf_table is None or perf_table.empty or anchor_season_id is None:
        return {}
    anchor_year = _season_year(anchor_season_id)
    if anchor_year is None:
        return {}
    anchor_comp = competition_for_season(anchor_season_id)
    pt = perf_table.copy()
    pt['_year'] = pt['seasonId'].map(_season_year)
    pt = pt.dropna(subset=['_year'])
    pt['_year'] = pt['_year'].astype(int)
    # Eligible prior rows:
    #   - STRICTLY prior years (year < anchor_year), up to lookback
    #   - SAME year + DIFFERENT competition — cross-league concurrent
    #     play (e.g. Santi Guzman 23/24 played for Leça in Camp AND
    #     for Atlético CP in Liga 3; Dedé 24/25 Dezembro/Camp +
    #     Sintrense/Liga 3). When rating the Liga 3 portion, the
    #     concurrent Camp portion is real evidence about current
    #     level and should inform the prior.
    pt = pt[
        ((pt['_year'] < anchor_year) & (pt['_year'] >= anchor_year - max_lookback))
        | ((pt['_year'] == anchor_year)
            & (pt['competitionId'].fillna(-1).astype(int) != (anchor_comp or -1)))
    ]
    if pt.empty:
        return {}
    # seasons_back ≥ 0; same-year cross-league gets decay=1.0 (full
    # weight) since it's contemporary evidence.
    pt['_seasons_back'] = (anchor_year - pt['_year']).clip(lower=0)
    pt['_decay'] = decay ** pt['_seasons_back']
    pt['_league_factor'] = pt['competitionId'].map(
        lambda c: (CVI_LEAGUE_MULTIPLIER.get(int(c), CVI_LEAGUE_DEFAULT)
                    if c is not None and not pd.isna(c) else CVI_LEAGUE_DEFAULT)
    )
    pt['_weight'] = pt['_decay'] * pt['mins_played'].fillna(0).clip(lower=0)
    pt['_contrib'] = pt['perf_pct'] * pt['_league_factor'] * pt['_weight']
    grouped = pt.groupby('playerId').agg(
        _sum_w=('_weight', 'sum'),
        _sum_c=('_contrib', 'sum'),
    )
    out = {}
    for pid, r in grouped.iterrows():
        w = float(r['_sum_w'])
        if w <= 0:
            continue
        prior_perf = float(r['_sum_c'] / w)
        # Prior strength ramps linearly from 0 (no prior) to 1.0 (at or
        # above full_strength_mins of decay-weighted prior minutes).
        strength = min(w / float(full_strength_mins), 1.0)
        out[int(pid)] = {
            'prior_perf': prior_perf,
            'prior_strength': strength,
            'prior_mins_eff': w,
        }
    return out


def most_recent_season_for_player(perf_table, player_id):
    """Return the most recent seasonId this player has GPA data for,
    or None if they have none. Used to anchor 'Current CVI' in the
    bio row.

    Tiebreaker for players with two same-year league rows (e.g. Santi
    Guzman 23/24 Leça-Camp + Atlético-CP-Liga-3): pick the seasonId
    where the player logged MORE MINUTES. The other league's data
    still contributes via Career CVI's same-year cross-league
    aggregation; this choice only affects which league_at_anchor
    multiplier is applied to the final composite (so a player who
    played mostly in Liga 3 gets a Liga-3-framed Current CVI).
    """
    if perf_table is None or perf_table.empty:
        return None
    rows = perf_table[perf_table['playerId'] == player_id].copy()
    if rows.empty:
        return None
    rows['_y'] = rows['seasonId'].map(_season_year)
    rows = rows.dropna(subset=['_y'])
    if rows.empty:
        return None
    rows['_mins'] = pd.to_numeric(rows.get('mins_played', 0),
                                     errors='coerce').fillna(0)
    rows = rows.sort_values(['_y', '_mins'], ascending=[False, False])
    return int(rows.iloc[0]['seasonId'])


# ==============================================================================
# Market-context features (consumed by the v2 EUR regression)
# ==============================================================================
# These are NOT inputs to CVI — they're signals that shift how the MARKET
# prices a player at a given quality level. Nationality drives sell-on
# premiums, team success drives visibility, xG over/underperformance
# captures finishing skill the market rewards/discounts. Computed
# per-(player, season) and surfaced alongside CVI in the Player Profile.
def compute_market_features(player_id, season_id, *,
                              raw_events_df, matches_summary_df,
                              player_details_df,
                              player_minutes_data,
                              team_name=None,
                              opta_team_lookup=None):
    """Per-(player, season) bundle of market-context features for the
    v2 transfer-value regression.

    Returns dict with:
        xg_residual_season        goals - xG, non-pen, this season
        xg_residual_career        goals - xG, non-pen, all seasons
        xg_residual_per90_season  same /90
        ass_residual_season       assists - xA proxy
        ass_residual_career       same career-cumulative
        passport_nationality      str (e.g. 'Portugal', 'Brazil')
        birth_nationality         str
        team_opta_rating          float (current team strength)
        team_ppm_season           team's points-per-match this season
        team_league_position      1-N rank within the season (NaN if unknown)
        positions_played_career   count of distinct primaryPositions across career
        seasons_played            count of distinct seasons in our data
    """
    out = {
        'xg_residual_season': None, 'xg_residual_career': None,
        'xg_residual_per90_season': None,
        'ass_residual_season': None, 'ass_residual_career': None,
        'passport_nationality': None, 'birth_nationality': None,
        'team_opta_rating': None, 'team_ppm_season': None,
        'team_league_position': None,
        'positions_played_career': None, 'seasons_played': None,
    }

    # ---- xG over/under (non-penalty) ----
    try:
        ev = raw_events_df[
            (raw_events_df['player.id'] == player_id)
            & raw_events_df['shot.xg'].notna()
            & (raw_events_df['type.primary'] != 'penalty')
        ]
        if not ev.empty:
            goals_c = ev['shot.isGoal'].fillna(False).astype(bool).sum()
            xg_c = float(ev['shot.xg'].sum())
            out['xg_residual_career'] = float(goals_c) - xg_c
            ev_s = ev[ev['seasonId'] == season_id] if 'seasonId' in ev.columns else ev
            if not ev_s.empty:
                goals_s = ev_s['shot.isGoal'].fillna(False).astype(bool).sum()
                xg_s = float(ev_s['shot.xg'].sum())
                out['xg_residual_season'] = float(goals_s) - xg_s
                # Use the player's totalMinutes for the season for /90
                pm = player_minutes_data.get(season_id) if isinstance(player_minutes_data, dict) else None
                mins = None
                if pm is not None and 'playerId' in pm.columns:
                    sub = pm[pm['playerId'] == player_id]
                    if not sub.empty:
                        mins = float(sub['totalMinutes'].sum())
                if mins and mins > 0:
                    out['xg_residual_per90_season'] = (out['xg_residual_season']
                                                         / mins * 90.0)
    except Exception:
        pass

    # ---- xA proxy: count of 'shot_assist'-tagged passes by player → xG of the shot they assisted ----
    try:
        # Player's shot-assist events
        sa_mask = (raw_events_df['player.id'] == player_id) & (
            raw_events_df.get('type.secondary', pd.Series(dtype='object'))
                          .apply(lambda x: isinstance(x, (list, np.ndarray))
                                  and 'shot_assist' in x)
        )
        sa = raw_events_df[sa_mask]
        # Approximate: next event in same match with non-null shot.xg = shot they assisted
        if not sa.empty and 'matchId' in raw_events_df.columns:
            # Lookup shot xG of the next event in the same match for each assist event
            ev_sorted = (raw_events_df[['matchId', 'matchTimestamp',
                                          'shot.xg', 'shot.isGoal', 'player.id']]
                          .sort_values(['matchId', 'matchTimestamp'])
                          .reset_index(drop=True))
            ev_sorted['next_xg'] = ev_sorted.groupby('matchId')['shot.xg'].shift(-1)
            ev_sorted['next_goal'] = ev_sorted.groupby('matchId')['shot.isGoal'].shift(-1)
            joined = sa.reset_index().merge(
                ev_sorted[['matchId', 'matchTimestamp', 'next_xg', 'next_goal']],
                on=['matchId', 'matchTimestamp'], how='left',
            )
            xa_c = joined['next_xg'].dropna().sum()
            assists_c = joined['next_goal'].fillna(False).sum()
            out['ass_residual_career'] = float(assists_c) - float(xa_c)
            jl_s = joined[joined.get('seasonId') == season_id] if 'seasonId' in joined.columns else joined
            if not jl_s.empty:
                xa_s = jl_s['next_xg'].dropna().sum()
                assists_s = jl_s['next_goal'].fillna(False).sum()
                out['ass_residual_season'] = float(assists_s) - float(xa_s)
    except Exception:
        pass

    # ---- Nationality ----
    try:
        if (player_details_df is not None and not player_details_df.empty
                and player_id in player_details_df.index):
            row = player_details_df.loc[player_id]
            out['passport_nationality'] = row.get('passportArea')
            out['birth_nationality'] = row.get('birthArea')
    except Exception:
        pass

    # ---- Team Opta rating ----
    try:
        if opta_team_lookup and team_name:
            out['team_opta_rating'] = opta_team_lookup(team_name)
    except Exception:
        pass

    # ---- Team PPM + league position this season ----
    try:
        if team_name and matches_summary_df is not None and season_id is not None:
            sm = matches_summary_df[matches_summary_df['seasonId'] == season_id].copy()
            # Parse scores
            def _parse_score(s):
                try:
                    if pd.isna(s) or '-' not in str(s): return (None, None)
                    h, a = str(s).split('-')
                    return (int(h.strip()), int(a.strip()))
                except Exception:
                    return (None, None)
            sm[['h_g','a_g']] = sm['score'].apply(_parse_score).apply(pd.Series)
            sm = sm.dropna(subset=['h_g','a_g'])
            # Points per team
            from collections import defaultdict
            pts = defaultdict(int); games = defaultdict(int)
            for _, m in sm.iterrows():
                h, a = m['homeTeamName'], m['awayTeamName']
                hg, ag = m['h_g'], m['a_g']
                games[h] += 1; games[a] += 1
                if hg > ag: pts[h] += 3
                elif ag > hg: pts[a] += 3
                else: pts[h] += 1; pts[a] += 1
            if team_name in games and games[team_name] > 0:
                out['team_ppm_season'] = pts[team_name] / games[team_name]
                # League position
                ppm_all = {t: pts[t]/games[t] for t in games if games[t] > 0}
                ranked = sorted(ppm_all.items(), key=lambda kv: -kv[1])
                for rank, (t, _) in enumerate(ranked, start=1):
                    if t == team_name:
                        out['team_league_position'] = rank
                        break
    except Exception:
        pass

    # ---- Position versatility + seasons played (career) ----
    try:
        if isinstance(player_minutes_data, dict):
            pos_set = set(); seasons_set = set()
            for sid, _pm in player_minutes_data.items():
                if not isinstance(_pm, pd.DataFrame) or 'playerId' not in _pm.columns:
                    continue
                sub = _pm[_pm['playerId'] == player_id]
                if sub.empty: continue
                seasons_set.add(sid)
                if 'primaryPosition' in sub.columns:
                    pos_set.update(p for p in sub['primaryPosition'].dropna().unique())
            out['positions_played_career'] = len(pos_set) if pos_set else None
            out['seasons_played'] = len(seasons_set) if seasons_set else None
    except Exception:
        pass

    return out
