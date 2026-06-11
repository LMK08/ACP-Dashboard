#!/usr/bin/env python3
"""Per-role PREDICTIVE weight of each GPA category (Lucas: "shooting gpa
and passing gpa have different predictive weight... maybe shooting value
is more predictive for CFs than CBs").

Cross-prediction test: current-season category percentile (within
role x league) vs NEXT-season overall rating and next-season off_pct,
per role, consecutive same-role pairs (n=589; per-role n 61-160,
SE ~0.08-0.13 — read patterns, not cells).

RESULTS 2026-06-11 (Spearman vs next-season RATING):
  role            n   Shoot  Pass  Recv  Drib  DeadB
  Adv Mid        61    .17   .14   .08   .03  -.07
  Central Def   160    .11   .03  -.14  -.04  -.05
  Deep Mid       61    .17   .07   .27   .02  -.01
  Striker        76   -.00  -.10   .00   .10   .04
  Wide Att       99    .10   .15   .09   .14  -.08
  Wide Def      132    .22   .09   .19   .28  -.28
  ALL           589    .13   .05   .05   .10  -.11

KEY FINDINGS (hypothesis REVERSED):
1. Striker shooting value predicts NOTHING (r=.00) — it is the most
   finishing-variance-dominated quantity in the system; striker futures
   are the hardest to forecast from any category.
2. Shooting predicts better for NON-forwards (WD .22, AM/DM .17,
   CB .11): low-volume shooting value reflects a persistent trait
   (arriving in the box, long-range threat), not finishing luck.
3. RECEIVING is the best predictor of future OFFENCE overall
   (ALL .13; DM .33, AM .26): where you get the ball persists and
   drives future output more than what you do with it.
4. Dribbling predicts for wide roles (WD .28/.32, WA .14/.20).
5. Dead-ball NEGATIVELY predicts future rating (ALL -.11, WD -.28) —
   third independent confirmation that set-piece reliance flags
   limited open-play value.

USE: descriptive rating unchanged (lambda self-reliability already
per-role). These are the per-role category weights for the future
acp_projection artifact. Display principle adopted: off is always
shown WITH its category components.
"""
print(__doc__)
