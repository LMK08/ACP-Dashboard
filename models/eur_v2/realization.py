"""Realization-rate layer on top of the EUR market-value model.

Most players at the Liga 3 / Campeonato tier don't sell at their
listed TM market value. Empirically (n=7,507 TM transfer records):
  • 35% leave on FREE TRANSFERS
  • 9% are end-of-loan returns
  • 9% are loan moves (mostly without fee)
  • ~44% have no recorded fee at all
  • Only ~3% of records have a non-zero recorded fee

Of the deals that DO record a fee (n=151 with both fee + market value
at transfer time), the fee/MV ratio breaks down sharply by tier:

  Fee bucket           n     Median fee/MV ratio
  ──────────────────  ───   ────────────────────
  €50k - €250k         20         0.25      ← Liga 3 tier reality
  €250k - €1M          24         0.55
  €1M+                106         1.27      ← signing premiums

So a €200k TM market value player at the Liga 3 tier realizes
~€50k when they actually sell, AND has a ~50% chance of leaving on
a free transfer instead.

This module turns the v2 EUR model's "market value" prediction into
two practical outputs:
  • expected_realized_fee — the EUR you'd actually receive IF a sale
                             happens (already discounted)
  • p_sells_for_fee       — probability the player sells at all
                             (vs free transfer / loan / contract end)

For scouting decisions:
  • "Buy this player" → use predicted_mv as ceiling cost
  • "Sell this player" → use expected_realized_fee × p_sells_for_fee
"""
from __future__ import annotations


# Realization ratio (fee / market value) — calibrated from a SMALL
# set of legitimate-tier observations (n=6 strict at-tier transfers
# with both fee and MV + 8 user-reported transfers, mostly without
# concurrent TM MV → can only inform fee level, not ratio).
#
# IMPORTANT DATA LIMITATION: the v1 calibration accidentally included
# 130+ transfers from later-career senior-team moves (Sporting CP,
# Vit. Guimarães senior, Real Valladolid, etc.) that share tokens
# with our B-teams. The strict tier_matcher fixed this but leaves us
# data-limited at the actual Liga 3 / Camp tier.
#
# The anchors below are therefore best-evidence priors rather than
# tight empirical fits. They'll tighten as more user-reported
# transfers accumulate.
#
# Anchor points (median realized fee/MV at each MV bucket):
#   MV €100k    → 0.20    (very small deals, near-free)
#   MV €250k    → 0.30    (Liga 3 modal — Joãozinho ratio ~0.20-0.25)
#   MV €500k    → 0.45
#   MV €1M      → 0.65
#   MV €2.5M    → 0.95
#   MV €5M+     → 1.10    (signing premium kicks in)
# v3 — steepened the upper tail. Top-tier players are scarce assets
# that competing clubs bid up well past TM's listed value (Bellingham,
# Mbappé, Endrick all cleared TM market by 30-90%). Holding the lower
# anchors fixed (Liga 3 modal player still realizes ~25% of MV) but
# bumping €1M+ tiers so 'top players are more valuable' reflects the
# real auction dynamics.
REALIZATION_ANCHORS = [
    (100_000,    0.20),
    (250_000,    0.30),
    (500_000,    0.50),    # was 0.45
    (1_000_000,  0.75),    # was 0.65
    (2_500_000,  1.10),    # was 0.95
    (5_000_000,  1.40),    # was 1.10
    (10_000_000, 1.60),    # NEW — premium auction territory
]


# Probability the player generates ANY transfer fee (vs free / loan /
# end-of-contract). Calibrated from the 7,507-record TM transfer
# history: about 3% of records have a non-zero fee, but FILTERED to
# players in the EUR range we care about (€50k-€5M), it's closer to
# 10-25% depending on age and MV tier.
#
# At Liga 3 / Camp tier most contracts are 1-3 years. Players in
# their first contract year are more likely to be sold (club still
# has leverage); players in their final contract year usually leave
# for free.
#
# Without contract-length data, we approximate with age + MV proxies:
#   Higher MV → higher P(sells_for_fee)   (clubs negotiate harder)
#   Younger  → higher P(sells_for_fee)    (more career runway → more buyer interest)
#   Older 28+ → lower P(sells_for_fee)    (contracts run out, free moves common)
P_SELLS_AGE_ADJUST = {
    # age range : multiplier on base probability
    (0, 20):   1.40,   # rare; clubs guard prospects tightly
    (20, 24):  1.20,   # peak sell-for-fee window
    (24, 28):  1.00,   # baseline
    (28, 31):  0.65,   # decline + contract expiry pressure
    (31, 35):  0.35,
    (35, 99):  0.15,
}

P_SELLS_MV_ANCHORS = [
    # v3 — top-tier players also have MUCH higher sale probability.
    # Big-money assets get sold; small-tier players walk on frees.
    (50_000,     0.05),    # tiny deals rarely formalized
    (150_000,    0.12),
    (250_000,    0.18),    # Liga 3 modal player
    (500_000,    0.30),
    (1_000_000,  0.50),    # was 0.45
    (2_500_000,  0.70),    # was 0.60
    (5_000_000,  0.85),    # was 0.70
    (10_000_000, 0.92),    # NEW — top assets nearly always sell
]


def _interp_anchors(x: float, anchors: list[tuple[float, float]]) -> float:
    """Piecewise-linear interpolation across (x, y) anchor points."""
    if x <= anchors[0][0]:
        return anchors[0][1]
    if x >= anchors[-1][0]:
        return anchors[-1][1]
    for (x_lo, y_lo), (x_hi, y_hi) in zip(anchors, anchors[1:]):
        if x_lo <= x <= x_hi:
            frac = (x - x_lo) / (x_hi - x_lo)
            return y_lo + frac * (y_hi - y_lo)
    return anchors[-1][1]


def realization_ratio(predicted_mv: float) -> float:
    """Median fee/MV ratio expected at this market-value tier.
    Returns the ratio (0..~1.3) by which to multiply predicted_mv to
    estimate the realized fee given that a sale happens."""
    if predicted_mv is None or predicted_mv <= 0:
        return 0.0
    return _interp_anchors(float(predicted_mv), REALIZATION_ANCHORS)


def p_sells_for_fee(predicted_mv: float, age: float | None) -> float:
    """Probability the player generates ANY transfer fee (vs leaving
    on a free transfer, end-of-loan, or end-of-contract). Strongly
    age-dependent because contract-expiry rates rise sharply past
    age ~28 at this tier."""
    if predicted_mv is None or predicted_mv <= 0:
        return 0.0
    base = _interp_anchors(float(predicted_mv), P_SELLS_MV_ANCHORS)
    if age is None:
        return base
    try:
        a = float(age)
    except (TypeError, ValueError):
        return base
    age_mult = 1.0
    for (lo, hi), m in P_SELLS_AGE_ADJUST.items():
        if lo <= a < hi:
            age_mult = m
            break
    return min(max(base * age_mult, 0.0), 1.0)


def elite_youth_perf_multiplier(perf_blend: float | None,
                                   age: float | None) -> float:
    """MV multiplier weighted HEAVILY on perf with a secondary youth bonus.

    v3.9 — user said "value performance even higher, expand the number
    of super high value players while not expanding the top line value."

    Perf dominates (0.4 + 0.6×youth factor inside the perf component) so
    elite-perf players get a strong boost even at age 24-25; youth amplifies.

    Calibration (multiplier values):
      Average player    (perf 55, age 25)  → 1.0  (no change)
      Strong young      (perf 75, age 22)  → ~2.9
      Elite young       (perf 90, age 20)  → ~4.5
      Peak              (perf 95, age 18)  → ~6.0 (cap)
      Elite older       (perf 95, age 30)  → 1.0  (no youth = no boost)

    Both perf AND age above threshold required — old or weak players get
    no multiplier. A hard MV cap (in expected_realized_fee) prevents the
    absolute top from running away — instead, MORE players cluster near
    the cap, which is the user's stated goal.
    """
    if perf_blend is None or age is None:
        return 1.0
    perf_excess = max(0.0, (float(perf_blend) - 55.0) / 40.0)   # 0 at perf 55, 1 at perf 95
    youth_excess = max(0.0, (24.0 - float(age)) / 6.0)            # 0 at age 24, 1 at age 18
    # Perf dominates; youth amplifies. Even a 25yo elite gets ~3x; an
    # 18yo elite gets ~6x. A weak 18yo gets nothing.
    combo = perf_excess * (0.4 + 0.6 * youth_excess)
    mult = 1.0 + 6.0 * combo
    return min(mult, 6.0)


# v3.9 — hard cap on MV. User asked to "expand the number of super
# high value players while not expanding the top line value." The cap
# is calibrated against the actual top of our reported_fees set
# (Yan Maranhão €450k summer 2025) plus a small headroom.
MV_CAP_EUR = 500_000


def expected_realized_fee(predicted_mv: float,
                            age: float | None = None,
                            perf_blend: float | None = None) -> dict:
    """Combined output: probability-weighted realized fee + components.

    Returns dict:
        realization_ratio  — fee/MV multiplier given sale (0..~1.3)
        elite_mult         — v3.8 young+perf exponential boost (1..6)
        expected_fee_if_sells — predicted_mv × realization_ratio × elite_mult
        p_sells_for_fee    — probability of a paid sale (vs free)
        expected_fee_overall — expected_fee_if_sells × p_sells_for_fee
                                (the right scouting figure when budgeting
                                 expected proceeds from a portfolio of
                                 players)

    The `perf_blend` arg is optional; if not passed, elite_mult is 1.0
    (preserves backwards-compatible behaviour for callers that don't
    have CVI perf available).
    """
    if predicted_mv is None or predicted_mv <= 0:
        return {'realization_ratio': 0, 'elite_mult': 1.0,
                 'expected_fee_if_sells': 0,
                 'p_sells_for_fee': 0, 'expected_fee_overall': 0}
    rr = realization_ratio(predicted_mv)
    em = elite_youth_perf_multiplier(perf_blend, age)
    p = p_sells_for_fee(predicted_mv, age)
    fee_if = predicted_mv * rr * em
    # v3.9 hard cap so the top doesn't run away as we steepen the perf
    # boost. More players cluster near the cap — that's the user's goal.
    fee_if = min(fee_if, MV_CAP_EUR)
    return {
        'realization_ratio': rr,
        'elite_mult': em,
        'expected_fee_if_sells': fee_if,
        'p_sells_for_fee': p,
        'expected_fee_overall': fee_if * p,
    }
