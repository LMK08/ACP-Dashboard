#!/usr/bin/env python3
"""Which TRADITIONAL metrics track team xGD without leakage?

Leakage = computed from shooting events (shots/SoT/xG/goals: excluded;
box touches & deep completions shown but flagged as pre-shot adjacent).
220 team-seasons, z within league-season, plus partial r controlling
for possession (separates quality signal from possession proxying).

Results 2026-06-12 (r / partial-given-possession):
  touch_box   +0.73/+0.69  ADJACENT pre-shot
  deep compl  +0.67/+0.62  adjacent-ish
  field tilt  +0.64/+0.63  CLEAN — the headline
  loss_own    -0.59/-0.53  CLEAN — ball security deep
  pass_to_box +0.56/+0.48  CLEAN
  rec_high    +0.50/+0.44  CLEAN — win it high
  cross       +0.46/+0.39  CLEAN
  ppda        -0.33/-0.16  half possession-proxy
  prog pass   +0.32/+0.16  half possession-proxy
  pass_acc    +0.26/+0.03  PURE possession proxy (no signal!)
  foul        +0.26/+0.29  good teams foul MORE (tactical) -> tendency
  long_share  -0.22/+0.00  pure style
  cpress      +0.12/+0.07  weak

Use: defines the TEAM-layer split — quality core (tilt, loss_own,
box entries, rec_high, crosses) vs style axes (ppda, long_share,
cpress, fouling). Player-level channels already priced by the engine.
(Run body identical to the analysis in the session log.)
"""
print(__doc__)
