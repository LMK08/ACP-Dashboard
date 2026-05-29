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


# Realization ratio (fee / market value) — calibrated from 151 TM
# transfer records with both fee and MV known. Piecewise linear in
# log-MV space so the curve is smooth from €50k → €5M.
#
# Anchor points (median realized fee/MV at each MV bucket):
#   MV €100k    → 0.20    (very small deals, near-free)
#   MV €250k    → 0.30    (Liga 3 tier — your starting case)
#   MV €500k    → 0.45
#   MV €1M      → 0.65
#   MV €2.5M    → 0.95
#   MV €5M+     → 1.10    (signing premium kicks in)
REALIZATION_ANCHORS = [
    (100_000,   0.20),
    (250_000,   0.30),
    (500_000,   0.45),
    (1_000_000, 0.65),
    (2_500_000, 0.95),
    (5_000_000, 1.10),
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
    # (predicted MV in EUR, base P(sells_for_fee))
    (50_000,    0.05),   # tiny deals rarely formalized
    (150_000,   0.12),
    (250_000,   0.18),   # Liga 3 modal player
    (500_000,   0.30),
    (1_000_000, 0.45),
    (2_500_000, 0.60),
    (5_000_000, 0.70),
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


def expected_realized_fee(predicted_mv: float,
                            age: float | None = None) -> dict:
    """Combined output: probability-weighted realized fee + components.

    Returns dict:
        realization_ratio  — fee/MV multiplier given sale (0..~1.3)
        expected_fee_if_sells — predicted_mv × realization_ratio
        p_sells_for_fee    — probability of a paid sale (vs free)
        expected_fee_overall — expected_fee_if_sells × p_sells_for_fee
                                (the right scouting figure when budgeting
                                 expected proceeds from a portfolio of
                                 players)
    """
    if predicted_mv is None or predicted_mv <= 0:
        return {'realization_ratio': 0, 'expected_fee_if_sells': 0,
                 'p_sells_for_fee': 0, 'expected_fee_overall': 0}
    rr = realization_ratio(predicted_mv)
    p = p_sells_for_fee(predicted_mv, age)
    fee_if = predicted_mv * rr
    return {
        'realization_ratio': rr,
        'expected_fee_if_sells': fee_if,
        'p_sells_for_fee': p,
        'expected_fee_overall': fee_if * p,
    }
