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

`python -m pytest tests -v` — run it under the PINNED Streamlit too before
pushing (`python -m venv --system-site-packages /tmp/venv141 && /tmp/venv141/bin/pip
install streamlit==1.41.0`, then that python): 1.41 treats a keyed widget
whose parameters change between runs as a new widget and resets it, which
the local 1.50 does not — seed defaults via session state, never toggle
`index=`. `tests/test_smoke_pages.py` boots app.py
headlessly (streamlit.testing AppTest, needs the data files incl. the
HF-only `raw_events.parquet`; skips without them) and walks every analysis
type on the defaults plus the fixtures-only newest season; a page fails on
an uncaught exception or an `st.error`. `tests/test_pdf_layout.py` is
pure-Python. `deploy_to_hf.yml` runs both as a `smoke` job the deploy job
needs. Local runs delete stale `stats_cache/` parquets (fingerprint
mismatch) — `git checkout -- stats_cache/` afterwards.
