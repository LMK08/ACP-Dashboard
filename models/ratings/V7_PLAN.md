# v7 no-lookahead plan (Lucas, 2026-10-27: "Full walk-forward")

**Rule:** a player's season-S rating never depends on anything after season S.
The role taxonomy stays frozen, because refitting it per year redefines the roles.

## Where the work happens

- **Dashboard:** worktree `~/acp_v7/Match_Reports_API/Dashboard` on branch `engine-v7-proto`.
  - Another session works on main in the Desktop checkout, so main and that checkout stay untouched.
  - `~/acp_v7/GPA Model Project v2` is a symlink to the real GPA repo, so the engine's relative paths still work.
- **GPA:** branch `walk-forward-v7` in its own checkout.
  - Nothing else is working there.
  - Outputs go to NEW paths (`parquet_data/walkforward/`, `models/walkforward/`). Production files are never overwritten.

## Limits

- **Disk:** 16 GB free on 2026-10-27. Keep this work under about 1 GB, because tonight's tracking batch downloads videos.
  - Save only the models needed to value the in-progress season.
- **Laptop:** long jobs run in `screen`, at `nice` with 2 XGBoost threads. They must end by 22:30 EDT, before batch 2 starts at 23:00.
- **Data:** local GPA events are dated 2026-10-20.
  - Completed seasons are identical to CI's.
  - 2026/27 has 60 of about 70 matches, so the headline comparison uses completed seasons only.

## Steps

- [x] **Builder fixes** (commit 731b6ba): no career ladders, as-of λ / duel expectations / league gap, main-row pairing.
  - `tests/test_rating_no_lookahead.py` passes at cutoffs 2023 and 2024.
- [x] **GPA walk-forward training** (`src/walkforward_train.py`), done 2026-10-27 15:24. Training seasons per season are in `models/walkforward/manifest.json`.
  - Season S is trained on seasons whose year is less than S's year.
  - 2021/22 uses a 5-fold match cross-fit within the season.
  - Early stopping uses a 10% match holdout inside the training seasons.
  - **Feasibility fold first:** 22/23 trained on 21/22, compared with LOSO and production.
    - Stop and report if the rank correlation of player-season V with LOSO is below 0.9. Result: 0.975, so it passed.
    - All seasons vs LOSO/production, V/90 rank r: 0.956 to 0.995.
- [x] **GPA assembly:** `loso_player_ratings.py` and `export_to_dashboard.py` read the walk-forward values when `GPA_VALUES=walkforward`, writing into the worktree.
- [x] **Dashboard `build_pass_split.py`:** Creating and Linking come from the walk-forward values.
- [x] **DefR:** response table, DWAE rates, position medians and `defr_career` are as-of; modal-slot ties are deterministic.
  - Proof: built with events cut off at 2023, every 2021-23 value is identical.
- [x] **RAPM:** own columns per window from minutes through T; (decay, alpha) frozen at 0.7 / 3000 (`RAPM_RETUNE=1` to re-pick).
  - Proof: built with data cut off at 2023, every 2021-23 coefficient is identical.
- [x] **Roles:** the frozen `role_model.joblib` is applied, never refit (`ROLE_REFIT=1`).
  - Reproduces committed assignments; differences are only the stale local 2026/27 data.
- [x] **Truncation checks for DefR and RAPM:** run on data cut off at 2023 and compare 2021-23. Both identical (see above).
- [x] **Rebuild the chain in the worktree** (pandas 3 venv). Compare v6.9.2 vs v7: what moves, validity, ACP players. Results are in `V7_PROTOTYPE.md`.
- [x] **Tests + fresh review.** Then report. Nothing is merged or pushed without Lucas's yes.
  - Full suite (base python, pandas 2.3.3): 148 passed, 2 skipped.
  - Review, 2026-10-27: no remaining lookahead found in the rating code. Its corrections to the write-up are in `V7_PROTOTYPE.md`. The CI wiring is below.

## Wiring v7 into CI and the nightly rebuild (plan, 2026-10-27)

Lucas asked for this plan "without deploying anything yet". A fresh review found nothing that deploys before step C. Its five fixes are folded in:
- the slow check skipping silently, and pytest missing from the engine job
- "frozen" being undefined
- validation runs in the shared deploy queue
- rollover training that saved nothing
- rebuilding the categories exactly

**Current state:**
- `engine_rebuild.yml` checks out GPA-Model-Engine `main`.
- Phase 1 runs `loso_player_ratings` and `export_to_dashboard` in LOSO mode.
- Nothing trains or reads walk-forward values.
- v7's `build_pass_split.py` needs them, so merging the branch alone breaks Phase 2.

### Design

- **One rule, applied everywhere: value a season of year Y with the newest walk-forward model set whose year ≤ Y.** Every such set was trained only on seasons before its own year, so it never sees season Y or later. A stale set is still leak-free, just older.
- **Completed seasons 2021–2025** use the frozen values already computed. They are committed to the GPA repo, as the LOSO files are today:
  - 7 files, about 180 MB, the largest 40 MB.
  - Same 23 columns as the LOSO files.
- **The season in progress (2026/27)** is predicted every night from the committed `wf_2026` models: 64 MB, never retrained in CI. These are the same models that produced today's local values.
- **CI never trains.** Training happens only at the yearly rollover, locally (about 35 min at 2 threads).
- **Production GPA values (`action_values.py`) keep running.** They still feed displays that are not part of the rating:
  - OBV pages, phases, viz aggregates
  - DefR OBV columns, GK values

### A. Code changes (local, on the two branches, no outward effect) — DONE 2026-10-27

Lucas, 2026-10-27, chose "Validate, then check in": do A and B, then show the CI result before C. Frozen values and models go in the GPA repo.

GPA repo, branch `walk-forward-v7`:
1. **`walkforward_train.py`:**
   - **Categories come from the model file.** XGBoost 3.1's `.ubj` keeps each feature's training categories, read with `get_categories(export_to_arrow=True)`. So no categories JSON is needed and nothing is re-derived.
     - Proof: with the training run's `wf_192831` moved aside, predict-only recreated it exactly (`assert_frame_equal`, `check_exact`).
   - **The manifest has `seasons` and `model_sets`** (old flat files are migrated on read). Season years come from the Dashboard's `league_config.SEASON_YEAR`, the one place a season is added.
   - **New `--train-for <year>` mode**, for rollover:
     - Trains on every season before that year, even when no season of that year has events yet.
     - Always saves the models.
     - Writes the `model_sets` entry.
     - Why it's needed: `--years 2027` would save nothing, because models are saved only for the newest year that has events.
2. **New `src/walkforward_predict.py`** (predict only):
   - **"Frozen" is explicit:** a season is frozen only if the manifest marks it `"frozen": true`. A file merely existing never counts.
     - At first, frozen means 2021/22–2025/26.
     - Every other season present in the events is re-predicted on every run, overwriting its file.
   - **Model choice:** use the newest model set with year ≤ the season's year, and record in the manifest which set valued it.
   - Warns, without failing, when a season's own-year set is missing (rollover pending).
   - Fails when no set qualifies at all.
3. **`git add -f` the committed set:**
   - the 7 frozen parquets
   - the `wf_2026` models
   - `manifest.json`
4. **`scripts/rebuild_all.sh`:**
   - Add `walkforward_predict.py` after `action_values.py`.
   - Export `GPA_VALUES=walkforward` for steps 7–8.

Dashboard: a new branch `engine-v7` cut from current `origin/main`:
5. **Apply the code-only diff** of `engine-v7-proto` against its base (9c2d471).
   - Leave out the local-build outputs, which would otherwise ship on the next deploy.
   - Do not cherry-pick 731b6ba / 6a02ff9, because that would revert the other session's comparison work now on main.
   - `main` has moved two commits, and they overlap only in `app.py` (different hunks).
6. **`engine_rebuild.yml`:**
   - Phase 1: run `walkforward_predict.py` after `action_values.py`. Prefix `loso_player_ratings.py` and `export_to_dashboard.py` with `GPA_VALUES=walkforward`.
   - Pin `xgboost==3.2.0`, the version CI already ran. The walk-forward models were trained with 3.1.3 but predict bit-identically under 3.2.0 + pandas 3. 3.1.3 itself breaks `xpass.py` on pandas 3: the first validation run, 40583311737, failed there. scikit-learn is already pinned to 1.9.0, which wrote `role_model.joblib`.
   - Install `pytest` in the engine job. Today only `deploy_to_hf` installs it.
   - After Phase 2: run `tests/test_rating_no_lookahead.py` (~25 s). This is a gate: a lookahead regression fails the run before anything is committed.
   - New dispatch inputs:
     - `gpa_ref` (default `main`), so a run can check out the GPA branch. On `workflow_run` it is empty, so the checkout falls back to the default branch.
     - `slow_checks` (default false). When true, run `test_engine_no_lookahead_upstream.py` with `ACP_SLOW_TESTS=1` (~7 min). Without that variable the test skips itself and reports green.
   - **Validate-only runs get their own concurrency group:**

     ```yaml
     group: ${{ (github.event_name == 'workflow_dispatch' && inputs.deploy != 'true') && format('engine-validate-{0}', github.run_id) || 'hf-deploy' }}
     ```

     - These runs never push to HF.
     - Holding `hf-deploy` for 1–2 h would let a newer pending run cancel a pending `scheduled_update`. The engine rebuild then skips, because its trigger run did not succeed.
   - On validate-only runs, upload the engine outputs as an artifact so they can be compared.
   - `glicko_ladders.py` is kept for now. v7 never reads `duel_ratings.parquet`, so it could be dropped later as a cleanup.
   - Checked on the clean `engine-v7` checkout with main's v6.9.2 data, which is what the C.2 smoke test sees: 172 passed, 2 skipped.
7. **`player_engine_meta.json`:** record which model set valued each season.

**Local check before anything leaves the laptop:**
- Run the CI-equivalent chain (predict instead of train) in the worktree.
- **Round-trip check:** move the training run's `wf_192831` out of the folder first (keep it in scratch). Predict-only must recreate it, and the new file must equal the saved copy exactly. This proves the saved models, best iteration and categories round-trip.
- Every completed-season value must equal today's v7 build.
- The full test suite must pass.

### B. Validate in CI without deploying (needs Lucas's yes: pushes two branches) — DONE 2026-10-27

Only a push to the Dashboard's `main` deploys (`deploy_to_hf.yml`). The GPA repo has no workflows.

1. Push `walk-forward-v7` (GPA) and `engine-v7` (Dashboard) as branches.
2. Run `gh workflow run engine_rebuild.yml --ref engine-v7 -f deploy=false -f gpa_ref=walk-forward-v7 -f slow_checks=true`.
   - This is validate-only: no commit, no deploy.
   - It runs in its own concurrency group (see A.6), so it never blocks or cancels a data update.
   - Confirm in the log that the slow test ran rather than skipped.
3. Download the artifact and compare it with the local v7 build.
   - Completed seasons must match exactly. CI has 8 more 2026/27 matches (68 vs 60), so an exact match is also a no-lookahead check on the whole chain.
   - 2026/27 differs because CI has more data.

**Results (2026-10-27).** Four validate-only runs. None committed or deployed.

| run | code | result |
|---|---|---|
| 40583311737 | 0210c80 | Failed in `xpass.py`: xgboost 3.1.3 can't fit on pandas 3 strings. Pinned 3.2.0 (94223da) after proving the walk-forward models predict bit-identically under it. |
| 40585321492 | 94223da | Passed. Completed seasons matched exactly in offence, quality, duel attack and RAPM, but not DefR (rating off by up to 0.4). |
| 40596437247 | 46b2cd0 | Passed. Ratings matched exactly; projections off by up to 0.026. |
| 40602698150 | c32b617 | **Passed. Every completed-season value matches the local build exactly:** rating, all five components, `acp_rating_abs`, projection (max 1.8e-13), projection gate G1. |

- Every run: all 4 lookahead tests passed in CI, the slow DefR/RAPM one included (it ran rather than skipped). 2026/27 was valued by the `wf_2026` models.
- **The "local/CI drift" was a bug, not data.** Single-key `sort_values` uses quicksort, which orders ties differently on CI's x86 runners and an ARM Mac.
  - **DefR:** `compute_defr` sorted each match's events by time, and its responder scan only looks forward, so tie order changed the matches: 563,676 responses on CI vs 571,128 locally, from identical events.
  - **Projection:** `build_projection` paired rows with the next season after `sort_values('yr')`. For winter movers (two rows in one year) the CPU chose the training row.
  - Fixed with `kind='stable'`, which keeps ties in Wyscout's file order: 46b2cd0 (DefR, two role-feature sorts) and c32b617 (projection, plus YoY diagnostics that are printed only).
  - v6.9.2 on main has the same bug. GitHub runners vary in CPU, so its DefR could move between nightlies with no new data.
  - **A fresh-context review scanned the whole CI chain, live-season-only paths included.** Two more were hardened:
    - GPA `exact_minutes.py`: duplicate-record tie at lines 72 and 138.
    - `cluster_roles` primary role: pandas 3 was already stable here; pandas 2.x was not.
    - Neither changes any output beyond float rounding (4 `primary_share` values at 1e-16).
    - A stable sort's result is fixed by its input, so the run-4 match still holds for CI.
- **Effect on v7's numbers:**
  - Ratings move by at most 0.25 and projections by at most 0.3.
  - Every headline in V7_PROTOTYPE.md is unchanged except the projection gate: 0.461 vs carry-rating 0.458, still a tie.
  - The suite still passes: 172 passed, 2 skipped.
- **Pre-existing, not v7:** Phase 3 logs 10 `[precompute] … Cannot hash argument` lines. The step passes, and main's last nightly (40343018781) logs the same 10.
- **Differ between CI and local by design:** the GK handling columns of `gpa_player_season_values`. They come from production GPA values, retrained every run on all data (see Decisions). So does `Total Value` on the 3 rows that include them. No rating input reads either; `rapm_v4` reads `Total Value` only in a printed check after its output is saved.

### C. Ship (a separate yes from Lucas; this IS the deploy) — DONE 2026-10-28

Lucas, 2026-10-27: "Yes, ship v7 now … Leave the display pages on production values for now."
- GPA `main` 75c484d: merge of `walk-forward-v7`.
- Dashboard `main` 4efd8a6: merge of `engine-v7` onto bcb4d44. `app.py` auto-merged; the suite passed on the merged tree (176 passed, 2 skipped). Its code deploy passed the smoke test.
- Engine rebuild 40622807464 (`deploy=true`, 50 min) committed 6bad65f: `rating_version` v7.0, 4,890 rows, data through 2026-10-25. The lookahead gate passed in the run.
- **All Seasons rating** (`acp_rating_career`) vs the v6.9.2 build on the same data, players with ≥ 900' career:
  - Median |change| 3.0 points, p90 8.0, 88 players move 10 or more points; rank correlation 0.940.
  - Players whose last season was 2021 or 2022 move most (p90 11–13); 2025/26 players least (p90 6.5).

Do this on a day without a data update: `scheduled_update` runs at 05:00 UTC on Sun/Mon and at 06:00 UTC on Wed/Thu. Check `gh run list` first to make sure no run is pending in `hf-deploy`.

1. **Merge `walk-forward-v7` into GPA `main`.** On its own this changes nothing live, because `GPA_VALUES` defaults to LOSO.
2. **Merge `engine-v7` into Dashboard `main`.** The push deploys the code, which is still on old data.
   - The radar skips a missing `woe_att_p90` axis.
   - If the new lookahead test fails in the smoke job on the old data, the deploy is blocked. That is fail-safe, and step 3 ships everything anyway.
   - **Don't run the local `rebuild_all.sh` between steps 1 and 2.** It would feed walk-forward values into v6.9 code in the Desktop checkout.
3. **Immediately run `gh workflow run engine_rebuild.yml -f deploy=true`.** This rebuilds and commits v7, then deploys through the smoke-tested path.
4. **Rollback:** revert the Dashboard merge, then rebuild with `deploy=true`. GPA `main` can stay as it is.

### Yearly rollover (add to the season-rollover checklist)

After the last 2026/27 match:
1. Refresh `raw_events` from the Space so the local data is complete, then run `walkforward_predict.py`. 2026/27 is not frozen yet, so it gets re-predicted on the full season.
2. Check its match count against the schedule. Then mark it `"frozen": true` in the manifest and `git add -f` the file.
3. Run `walkforward_train.py --train-for 2027` locally (all seasons ≤ 2026). Commit the models, categories and manifest, and drop the `wf_2026` models from the tree.
4. Until step 3 is done, CI values 2027/28 with `wf_2026` and logs a warning. That is leak-free, just older.

### Decisions for Lucas

- **Storage:** commit the frozen values and models to GPA-Model-Engine. That adds about 245 MB of git history, as the LOSO files did (recommended). The alternative is an HF dataset or B2.
- **Display pages:** OBV, GK values and DefR OBV still use production GPA values, which are in-sample for 2021–2024. Switch them to walk-forward later or not? They are not part of any rating.
