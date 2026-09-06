# Match_Reports_API/Dashboard — Streamlit club dashboard (HuggingFace Space)

Own git repo (deploys via git push to the HF Space). Python 3.11 venv with
strictly pinned deps — **pyarrow must stay 24.0.0** (25.0.0 segfaults); never
upgrade the pins.

Deploys: push to GitHub `main` triggers `.github/workflows/deploy_to_hf.yml`;
`scheduled_update.yml` / `engine_rebuild.yml` force-push single-commit deploys.
All three strip loose PNGs (HF's pre-receive hook rejects non-LFS binaries)
but **exempt `icons/`** — the team-crest PNGs the scatter plots load at
runtime, shipped via `git lfs track "icons/*.png"`. Keep any new runtime
image assets on that LFS+exempt path or the deploy will silently drop them.

## Where things live (2026-09 split, in progress)

- `models/value/cvi.py` — the CVI / projected-EUR model (pure pandas; no
  streamlit). app.py re-exports its `__all__` so call sites read unchanged;
  put model logic there, never back in app.py. Tests: `tests/test_cvi_model.py`.
- `theme.py` — every colour and figure convention. The four radar colours,
  the cream ground, the xG ramp and the GPA value ramp are read from here by
  app.py, opposition_report.py, obv_viz.py, pitch_visualizations.py and
  pitch_interactive.py. New plotters import `theme`; never re-type a hex.
  A colour change is a drawing change: bump `FIGURE_CACHE_VERSION`.
- `views/<page>.py` — one module per analysis type; app.py's chain is now
  nine one-line `views.<page>.render()` calls. A view reads its collaborators
  from the running app module (`sys.modules['__main__']`) in the binding
  block at the top of `render()` — that block IS the page's dependency list,
  keep it honest when you add a helper. Views never import app.py. The
  directory is deliberately NOT called `pages/` (Streamlit would treat that
  as an auto-discovered multipage app).
- `context_bar.py` — the League / Season choice, drawn ONCE at the top of the
  sidebar and read by every page through `league_selector()` /
  `season_selector()` (signatures unchanged; they no longer draw widgets).
  "All Seasons" is always offered (stable widget options — Streamlit resets a
  keyed widget whose options change); pages that can't use it resolve it to
  the default season and say so. One season key per league. To preset the bar
  from a view or bridge call `context_bar.set_context(league, season)` BEFORE
  the bar's widgets exist in that run (Home, app.py prelude, or right before
  `st.rerun()`), then `navigation.go_to(page)`.
- `navigation.py` — grouped sidebar navigation (Club / Opposition / Players /
  Recruitment). ONE piece of state, `st.session_state.current_page`; each
  group is a radio mirroring it. To send a user somewhere from a view use
  `navigation.go_to(page, **preset_keys)` (e.g. a season selector key or the
  opponent selectbox key) — never set radio keys by hand. Add a page by
  appending to `NAV_GROUPS` and the `views.<page>.render()` chain in app.py;
  the smoke test finds the right radio by page name.
- `views/home.py` — the club's Home page (position, form, promotion odds
  from season_simulation.pkl, last match, next opponent, squad ACP Index,
  data freshness). Every card deep-links into the detail page.
- `team_interactive.py` — Plotly versions of the team visuals both team pages
  show on screen: season shot maps, passing network, rolling xG (the season
  report dot plots were already Plotly). They share their DATA step with the
  matplotlib originals (`pitch_visualizations.compute_passing_network`,
  `team_interactive.season_shots` / `rolling_xg_frame`) so the PDF, which
  still embeds the matplotlib PNGs via `_pdf_png` in opposition_report.py,
  shows the same numbers. Click-through: a shot or rolling-xG point opens the
  match (`app.open_match_from_selection`), a network node opens the player
  profile (`app.open_profile_from_selection`), a dot-plot team opens that
  team's report (`on_team_select`). Add an interactive visual to BOTH pages.
  Shot maps (match / team / player) share ONE drawing: `pitch_interactive._pitch_layout`
  with true Wyscout proportions (`Y_PER_X` = 1.05/0.68 — drawn 1:1 they were
  squashed to half height), `team_interactive.SHOT_MARKER`, the xG ramp and
  the goal ring — the ONE builder is `pitch_interactive.shot_map_figure`; do not
  draw a shot map any other way. Pitch charts lock their aspect: never put
  one in a half-width column; stack for/against at full width. Their height
  comes from the sidebar "Chart size" control (`context_bar.pitch_height()`,
  Standard/Large/Huge = 760/1000/1300 px; a figure's height is fixed
  server-side while its width follows the browser, so wide monitors need
  Huge to fill the width). Pass it to every pitch chart.
- `models/scoreline/` — the Dixon-Coles scoreline model (`dixon_coles.py`,
  pure numpy/scipy). `build_dc.py` tunes time decay, shrinkage and the
  goals-vs-xG blend the rates are fitted on with a monthly walk-forward
  backtest, then writes `dc_params.json` (what the app loads) and
  `dc_backtest.json` (metrics vs base rate and vs the strength model,
  reliability). Runs in the engine rebuild. `scoreline_ui.py` renders it on
  the Match Predictor page; never quote its probabilities without the
  calibration expander next to them. Tests: `tests/test_dixon_coles.py`.
- `event_tags.py` — `TagIndex` / `has_tag`: vectorised membership tests on
  the `type.secondary` tag lists. NEVER write
  `sec.apply(lambda x: tag in x)` on an events frame again: those per-row
  lambdas cost the two team pages ~9 s of every cold render (2026-09
  profile: Team Analysis 16 s → 7 s, Opposition Report 29 s → 19 s after
  the switch). Several tags on one column → one TagIndex, `.has()` each.
  Tests: `tests/test_event_tags.py` (byte-identical to the old lambdas).
- Cold-render rules: the rest of a cold Opposition Report is matplotlib
  rendering its ~27 cached PNG figures (screen + PDF pair) — per opponent,
  once per day. The boot prewarm (`_prewarm_scope_caches` in app.py) warms
  each league's LANDING season (newest with ≥ MIN_MATCHES_FOR_DEFAULT_SEASON
  matches of events — `current_season` is fixtures-only early on and warmed
  nothing) including `compute_team_season_metrics`, whose cache key comes
  from `season_report_cache_key()` on both team pages and in the prewarm:
  one format, or the pages stop sharing the entry. The "shaking" on a page
  switch is Streamlit replacing the previous page's elements under the
  viewport as the new ones stream in; `navigation.scroll_to_top_on_page_change`
  scrolls to the top once per page change (hidden zero-height component
  iframe with a nonce — React keeps an iframe whose srcdoc is unchanged and
  never re-runs its script) so the new page builds downward from a stable
  header. Streamlit has no way to render a page all at once.
- `models/value/eur_intervals.py` — the ENGINE VALUE (the projected EUR
  the app shows for outfielders; it was a closure in app.py's
  load_player_engine and the calibration script re-typed it by hand) and
  its fee-calibrated LIKELY-FEE RANGE: a split-conformal prediction
  interval on log(fee ÷ value), calibrated on the real permanent sales in
  valuations/reported_fees.csv paired with the engine row of the
  pre-transfer season (valued at the age AT SALE — lapsed rows carry a
  forwarded age). `python models/value/eur_intervals.py` runs in the
  engine rebuild right after build_player_engine.py and writes
  `models/value/eur_interval_calibration.json` (committed; the JSON is
  what the app reads). Rules: the range is a pure function of the
  displayed point and one scalar per level (`projected_eur_interval`) —
  never add range columns to the stats frame (they would ride the
  percentiles disk cache); it is shown ONLY inside the calibration's
  support (`range_support_reason`: value ≥ €25k, minutes and evidence
  weight at least the sold players' minima) and NEVER for goalkeepers
  (different estimator, no keeper sales); a fee pairs only with a row at
  the SELLING club (a row at the destination club is post-transfer
  performance → excluded; January sales are flagged mid_season, not
  excluded); a level ships only when its order index k ≤ n−2 (80% needs
  14 sales, 90% needs 29); copy quotes COUNTS ('7 of 13 sales fell within
  ×/÷1.6'), never 'half' / '8 in 10'; leave-one-out coverage is a property
  of the construction, not a test — the prospective ledger (fees added
  after a calibration, scored against the band live at the time) is the
  only verification, so never present LOO as proof. Render
  through `eur_interval_ui.py` (range sentence as a CAPTION under the
  metric — a text delta on st.metric draws an arrow; the calibration panel
  behind a toggle on the Value tab). Synthetic (hand-estimated) fee rows
  and unaccepted offers are excluded by the script's own filter — the UI
  loader treats them like real fees. `PLAYER_ID_ALIASES` moved to
  league_config.py so the script applies the same remap as the app.
  Pages that need an engine column the stats merge does not carry (e.g.
  Player Analysis needs mins_played for the gate) read it from
  `app.engine_rows_for_scope(season_ids)` — the ONE row-pick rule the
  merge itself uses — never from a second pick of their own. After editing
  valuations/reported_fees.csv or a curve constant, run the script and
  commit the JSON in the same commit: tests/test_eur_intervals.py (run by
  CI's smoke job) hard-fails when the
  JSON's curve constants differ from the live ones and only WARNS on
  quantile drift (the engine rebuild regenerates the JSON; a hand-committed
  JSON that lands mid-rebuild can conflict its rebase — rerun the rebuild).
  Tests: `tests/test_eur_intervals.py` (incl. bit-for-bit parity with the
  old closure and the stale-artifact check against a fresh build).
- `scripts/config_migrations/` — spent one-shot config.yaml scripts (README).

## Team Analysis ↔ Opposition Report parity (RULE)

The Team Analysis page (`views/team_analysis.py`) and the
Opposition Report (`opposition_report.py`) are two views of the same team-level
visuals. **Any visual or feature added/changed on one page must be applied to
the other in the same change** (and to the PDF via `pdf_figures`/`pdf_texts`
when it's a team-level visual). Shared visuals as of 2026-09:

- 4 team radars (Offensive/Distribution/Defensive/Set Piece) — same titles,
  colors (`theme.RADAR_*`), metric lists (`OFFENSIVE_METRICS`
  etc. in opposition_report.py mirror the inline lists in Team Analysis), and
  %-formatting (Ball Possession, duel win %s, Short Corner/Long Throw/First
  Contact %). Both pass real league/season labels to `plot_radar_chart`.
- Season Report (7-dimension dot plots), On-Ball Value & Phases
  (obv_categories + phase_profile), formation XI graphic, season shot maps
  (for/against), rolling xG history, corner analysis (left/right), zone
  heatmaps (recovery/loss), passing network, defensive structure, average
  positions, shot assists & dribbles.

Intentionally page-specific: Team Analysis — squad roster, stage filter,
corner summary table; Opposition Report — projected XI/subs, key players,
strengths/weaknesses synopsis, set-piece table + scatters, takeaways, PDF.

Both pages share the actual plotters (app.py / pitch_visualizations / obv_viz),
so drawing-code changes propagate; what drifts is call-site config (metric
lists, colors, formatting, which sections exist). Check both call sites.
Bump `FIGURE_CACHE_VERSION` for drawing-code changes (cache keys describe data).

## Opposition Report PDF (generate_pdf.py) layout rule

Every image must go through `add_figure` / `add_figure_row`, which measure the
PNG's real aspect ratio (Pillow) and page-break or scale so nothing crosses the
bottom margin. **Never place an image with an assumed aspect ratio or a
hand-advanced `set_y`** — figures are saved with `bbox_inches='tight'`, so
proportions aren't knowable in advance, and a wrong guess silently clips the
chart. Geometry test: `tests/test_pdf_layout.py` spies on `FPDF.image` and
asserts every placement lies within x∈[10, 287], y+h ≤ 190 (it caught a
zero-size placement when the cursor sat on the bottom margin, 2026-09).

## Tests (run before pushing; CI gates the deploy on them)

`python -m pytest tests -v`. The Space pins Streamlit 1.50.0 (README
`sdk_version`, mirrored in deploy_to_hf.yml's smoke job) — the same version
as the local base env. It was 1.41 until 2026-09: 1.41 measured a Plotly
chart's container once at first render and never again, leaving every
chart ~20% narrower than the page, and it reset keyed widgets whose
parameters changed between runs. Keep seeding widget defaults via session
state rather than toggling `index=`. If the pin ever moves, run the suite
under the new version first (`python -m venv --system-site-packages
/tmp/venvX && /tmp/venvX/bin/pip install streamlit==X`, then that python).
`tests/test_smoke_pages.py` boots app.py
headlessly (streamlit.testing AppTest, needs the data files incl. the
HF-only `raw_events.parquet`; skips without them) and walks every analysis
type on the defaults plus the fixtures-only newest season; a page fails on
an uncaught exception or an `st.error`. `tests/test_pdf_layout.py` is
pure-Python. `deploy_to_hf.yml` runs both as a `smoke` job the deploy job
needs. Local runs delete stale `stats_cache/` parquets (fingerprint
mismatch) — `git checkout -- stats_cache/` afterwards.
