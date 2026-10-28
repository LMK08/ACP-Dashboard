# ACP Rating v7.0 — no lookahead (branch `engine-v7`, not deployed)

**Rule:** a player's season-S rating never depends on anything that happened after season S.
Plan and proofs: `V7_PLAN.md`.

## What changed from v6.9.2

1. **Season-only Def Quality.**
   - Aerial and Ground Def Grade are the season's wins over expectation only.
   - The career duel ladders are gone. They were one rating per player for the whole career, applied to every season.
2. **Equal minutes shrink.** Def Quality, Def Volume and Off Duel Grade are pulled toward 50 under 2,500 minutes, exactly as Offensive Value already was.
3. **Off Duel Grade** is the season's take-on/shield wins over expectation per 90, damped by n/(n+60). The career ladder constant is gone.
4. **No later season reaches an earlier one:**
   - **GPA action values are walk-forward.** Season S is valued by models trained only on earlier years. 2021/22 has no earlier season, so it uses a 5-fold cross-fit by match within the season.
   - **DefR is as-of.** That covers the response table, DWAE rates, position medians and `defr_career`.
   - **RAPM** is solved per window on the minutes through that window.
   - **As-of in the builder:** the offence λ, the ground-duel expectations and the league gap.
   - **Roles** come from the frozen taxonomy, which is applied and never refit.
   - **Projection:** `n_seasons` is dropped. It was a whole-career count.
5. **Proof:** a build with every season after a cutoff removed gives identical values for the earlier seasons.
   - `tests/test_rating_no_lookahead.py` covers the rating builder at cutoffs 2023 and 2024.
   - `tests/test_engine_no_lookahead_upstream.py` covers DefR and RAPM at cutoff 2023. It is opt-in because it takes about 7 minutes locally (9–13 in CI). CI runs it on validate-only runs (`slow_checks=true`), not nightly.
   - **What the tests cannot see:**
     - GPA values and `pass_split`. These are leak-free by how they are trained (the manifest lists each season's training seasons), not by a truncation test.
     - `acp_rating_abs`. Its per-Camp-season gap was checked by review.
     - `build_player_engine`.

Both versions were built locally in the CI engine environment (pandas 3.0.3), from the same event data (dated 2026-10-20).
The headline numbers use completed seasons, 2021/22–2025/26. The local 2026/27 data has only 60 of about 70 matches.
The v7.0 column is the build with the stable-sort fixes (46b2cd0, c32b617; see V7_PLAN.md step B). They changed no number in this file except the projection row.

## Before / after

| | v6.9.2 | v7.0 |
|---|---|---|
| CB effective weight, offence : def quality (nominal 1.31) | 0.99 | 1.27 |
| WD (nominal 1.73) | 1.06 | 1.51 |
| DM (nominal 2.10) | 1.17 | 1.74 |
| Striker (nominal 3.00) | 1.86 | 2.69 |
| Rating YoY r (n=401) | 0.492 | 0.340 |
| Team test: prior-season ratings vs Liga 3 team xGD, r (n=57) | +0.45 | +0.38 |
| ↳ beyond last season's team xGD (n=50, `stages.py`) | +0.36 ± 0.15 | +0.29 ± 0.16 |
| Projection G1 Spearman: shipped 'full' / carry-rating baseline | 0.569 / 0.546 | 0.461 / 0.458 |
| League gap L3 − CAMP | 7.7 | 7.6 (as-of: 6.0 for Camp 23/24) |

Effective weight = weight × within-league-season spread, rows ≥ 500'.

### Where the validity change comes from (`stages.py`)

| stage | YoY r | team beta | minus v6.9.2 (95% paired bootstrap) |
|---|---|---|---|
| v6.9.2 | 0.492 | +0.36 | |
| design only (1–3 above, still leaky) | 0.406 | +0.39 | +0.03 [−0.05, +0.10] |
| + builder leak fixes | 0.358 | +0.32 | −0.04 [−0.13, +0.06] |
| + DefR/RAPM as-of + walk-forward GPA = v7.0 | 0.340 | +0.29 | −0.07 [−0.19, +0.03] |

- **Team test: no detectable change at any stage.** Every confidence interval includes 0.
  - Removing the career ladder (design only) raised the team beta slightly.
  - The later dips coincide with noisier early-season inputs (see below), so they are not evidence that v6.9.2 was "inflated".
- **YoY drop by period** (pairs y → y+1, n≈200 each):

  | stage | 21/22–22/23 | 23/24–24/25 |
  |---|---|---|
  | v6.9.2 | 0.573 | 0.383 |
  | design only | 0.498 | 0.284 |
  | + builder leak fixes | 0.423 | 0.276 |
  | v7.0 | 0.394 | 0.267 |

  - **Later seasons:** almost the whole drop comes from removing the career ladder. One constant repeated in both seasons raises YoY whatever the player does.
  - **Early seasons:** they also lose stability from the as-of inputs, which have little history there.

### What moves

- **Overall:** rank corr 0.890; median |change| 3.1 pts, p90 9.0.
- **By season:** the early seasons move most (21/22 and 22/23: rank r 0.80, against 0.89–0.95 later). Two causes, separated by build stage:
  - **Builder leak fixes, the larger cause.** These alone take 21/22 to rank r 0.85, with an offence-axis r of 0.86. 2021/22 has no earlier season pairs, so its offence category weights fall back to the flat 0.15 prior. Later seasons get fitted weights between 0.05 and 0.54.
  - **Walk-forward GPA.** This takes the 21/22 offence axis from 0.86 to 0.80, because those models are trained on one season or less.
  - **Walk-forward vs LOSO V/90 rank r** (outfield, within position group, ≥ 900', median over groups): 0.94 for 21/22, 0.95 for 22/23, 0.96–0.99 later. Interrupting is the least stable category early on (0.77–0.83, all positions).
- **Liga 3 25/26:** top-20 overlap 17/20.
- **Biggest movers:** players whose season duel results disagreed with their career ladder.
- **Atlético CP 25/26 ratings** move a few points.
  - César Sousa −9: Def Quality −23, Off Duel Grade −40 and Def Volume −15 percentile points.
  - Juan Herrera −8.5: Def Quality −27 and Def Volume −11 percentile points. With 1,574', both are now pulled toward 50.
  - Joãozinho +5.
- **Atlético CP projections** fall for Caleb, Caio Santana and César Sousa (−6 to −8). Their career-to-date rating fell once earlier seasons were valued without later data.

## Still tuned on all seasons (design level, disclosed rather than fixed)

- Role weights, `ROLE_CAT_RELEVANCE`, shrink constants and `Z_REPL = -0.4`.
- The offence λ fallback: the 0.15 prior and the 30-pair minimum.
- The DefR tunables.
- GPA hyperparameters and feature set.
- RAPM (decay, alpha), frozen at 0.7 / 3000.
- The role taxonomy.

These set how things are measured, not any player's value. Refitting them per year would change definitions from season to season.

## Known gaps

- **Old-season cards show the latest projection.** `card_scope` does this by Lucas's 2026-06-24 decision.
- **`acp_rating_career`** is the whole career as of today. The profile's all-seasons view uses it.
- **The DefR OBV columns** still use production GPA values. They are not part of the rating.
- **The projection ridge** loses to the baselines without the leaked inputs. The shipped 'full' model is unchanged, and the gate winner, marcel (0.486), also wins on v6.9.2.
- **`eur_interval_calibration.json`** drifts with the new projections. The engine rebuild regenerates it.
- **The committed parquets/json on this branch are a local build.** Adoption steps are in `V7_PLAN.md`.
- **Do not merge as is.** CI's engine rebuild does not train or read the walk-forward values. `build_pass_split.py` now stops with a clear message when the walk-forward files are missing, rather than mixing them with production values.
