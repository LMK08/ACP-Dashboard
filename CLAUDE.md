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
- `navigation.py` — grouped sidebar navigation (Club / Opposition / Coaches / Players /
  Recruitment). ONE piece of state, `st.session_state.current_page`; each
  group is a radio mirroring it. To send a user somewhere from a view use
  `navigation.go_to(page, **preset_keys)` (e.g. a season selector key or the
  opponent selectbox key) — never set radio keys by hand. Add a page by
  appending to `NAV_GROUPS` and the `views.<page>.render()` chain in app.py;
  the smoke test finds the right radio by page name.
- Page frame (RULE, 2026-11 UX pass). Every page opens with ONE
  `st.header(<page name>)` (the club name is in the sidebar, `app._sidebar_brand`;
  there is no app-wide `<h1>`), then the page's OWN pickers in the main area
  (team / stage, match, player, Player A / Template / Player B, Shadow Team's
  formation / save / load, Player Analysis's view row + "Filters" panel,
  open on arrival). Team Analysis keeps a "Team Report: X" subheader under
  the picker; Match Analysis carries its "Match Report" line in the scoreboard.
  Never put a page's picker in the sidebar again: at 900 px the player picker
  sat 1,100 px down it. The sidebar holds, in order: club block, League /
  Season (`context_bar.render`), navigation, Chart size
  (`context_bar.render_display`, only on `PAGES_WITH_PITCH_CHARTS`, default
  Large — Standard was tried and Lucas kept Large), the data date and the
  "About the data" panel
  (`app.about_data_panel`: version lines go there; a caveat a number depends
  on stays beside that number). The two team pages draw ONE section at a
  time — see "Team pages: sections" under the parity rule. CSS (app.py): the black button style names real buttons by
  data-testid (a bare `button {}` blanked the sidebar-reopen control and the
  table toolbars) — add a new button kind there; captions are full strength;
  the accent is `primaryColor` in `.streamlit/config.toml` (light tint under
  `[theme.sidebar]`). A `st.metric` delta used as a TEXT NOTE sits in a
  container whose key is listed in `theme.METRIC_NOTE_KEY_FRAGMENTS` (the
  arrow is hidden there only; real changes keep it). Every `@st.cache_*` in
  app.py passes `show_spinner=LOADING` (or False): the default prints the
  function's name to the visitor.
  The player radar (style switch, default, axis names) was left exactly as
  it was: Lucas wants any radar change discussed first (2026-11-13) —
  visuals may be upgraded, content may not change.
- Match Analysis (2026-11): a scoreboard under the picker
  (`views/match_analysis._scoreboard_html`; crests through
  `app.crest_data_uri`, xG = the sum of `shot.xg` in the match events, the
  same figure as the shot maps' subtitles; it carries the "Match Report"
  line), then ONE section drawn at a time (`MATCH_SECTIONS`, the radio
  labelled 'Match section': Summary / Team stats / Shots / Players /
  Tactics). The report button sits beside that row and must work from any
  section: the PDF renders its own figures (`_match_fig_or_none`), and the
  tables it prints (team / player stats, shot details) are prepared on every
  run, above the section branches — keep new PDF inputs there, never inside
  a section. A new block goes into a section, not below them.
- `views/home.py` — the club's Home page (position, form, promotion odds
  from season_simulation.pkl, last match, next opponent, squad ACP Index,
  data freshness). Every card deep-links into the detail page.
- `simulate_season.py` — the Monte Carlo season simulation behind Home's
  promotion tile and the Match Predictor's promotion / relegation tables
  (runs in scheduled_update.yml, writes season_simulation.pkl). Since
  2026-09 every remaining fixture is drawn as a full SCORELINE from the
  Dixon-Coles model (`models/scoreline/season_sim.py`: ScorelineSampler,
  sample_scores, apply_results) — never from the simple predictor, whose
  W/D/L draws with invented 2-1 / 1-1 goals it replaced. The refresh refits
  the model on today's matches with the hyperparameters build_dc.py chose
  and `--write-params` saves that fit to dc_params.json, so the tables,
  Home and the scoreline forecast read ONE fit (the engine rebuild re-runs
  the simulation with `--no-refit` right after build_dc.py, so a re-tune
  never leaves the pickle and the forecast on different parameters); the
  pickle records the fit under 'model' and both pages cite it. The
  league-format logic (séries, chained promotion + maintenance phases, FPF
  bonuses) stays in simulate_season.py; the format fixes the slot counts (8
  promotion-series places, 2 promotion slots, 4 relegation places) — printed
  as a check and asserted in `tests/test_season_sim.py`. Reserve / B teams
  are promotable and take a slot like anyone else (only the Primeira Liga
  bars them, and neither simulation reaches it): Sporting CP II went up from
  Liga 3 in 2024/25. Until 2026-11 the file skipped them and passed the slot
  down, so their promotion odds read 0%. Local runs:
  `--no-refit` draws from the committed dc_params.json (never commit a local
  refit), `--sims 500` for speed. Caveat: a side new to the data (relegated
  from Liga 2, say) starts AT the Liga 3 mean — the fit has no tier above
  Liga 3 to learn from; a side promoted from the Campeonato starts below it
  (league-mean prior, 2026-09-08).
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
  pure numpy/scipy). The penalty shrinks each team toward its OWN league's
  mean (`prior='league_mean'`, persisted in dc_params.json and read back by
  simulate_season's refit): with one shared zero centre an average
  Campeonato side equalled an average Liga 3 side, and clubs promoted from
  the Campeonato were over-rated by a third of a point per match in their
  first six Liga 3 matches (2024-26 walk-forward; 0.20 with the league
  means). The fitted tier gap and that promoted-side calibration live in
  dc_backtest.json and the Match Predictor's backtest toggle; keep the
  comparison there whenever the prior changes. An unseen team plays at its
  league's mean. `build_dc.py` tunes time decay, shrinkage and the
  goals-vs-xG blend the rates are fitted on with a monthly walk-forward
  backtest, then writes `dc_params.json` (what the app loads) and
  `dc_backtest.json` (metrics vs base rate and vs the strength model,
  reliability). Runs in the engine rebuild; the scheduled refresh also
  rewrites `dc_params.json` (simulate_season.py `--write-params`: same
  hyperparameters, today's matches). `season_sim.py` is the scoreline
  sampler the season simulation draws from. `scoreline_ui.py` renders the
  model on the Match Predictor page; never quote its probabilities without
  the calibration expander next to them. Tests: `tests/test_dixon_coles.py`,
  `tests/test_season_sim.py`.
- `event_tags.py` — `TagIndex` / `has_tag`: vectorised membership tests on
  the `type.secondary` tag lists. NEVER write
  `sec.apply(lambda x: tag in x)` on an events frame again: those per-row
  lambdas cost the two team pages ~9 s of every cold render (2026-09
  profile: Team Analysis 16 s → 7 s, Opposition Report 29 s → 19 s after
  the switch). Several tags on one column → one TagIndex, `.has()` each.
  Tests: `tests/test_event_tags.py` (byte-identical to the old lambdas).
- Cold-render rules: the rest of a cold Opposition Report is matplotlib
  rendering its ~23 cached PNG figures — per opponent, once per day; one
  200-dpi PNG serves screen and PDF, and the figures the screen shows as
  Plotly are drawn for the PDF only when it is generated (2026-10). The
  boot prewarm (`_prewarm_scope_caches` in app.py) warms each league's
  LANDING season (newest with ≥ MIN_MATCHES_FOR_DEFAULT_SEASON
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
- **Live / per-season engine value (2026-10-29).** Replay-tested first; see
  ACP_Official/live_value_plan_2026-10-29.md.
  - **build_projection** also writes `acp_projection_asof.parquet`: the
    shipped predictor applied to EVERY player-season.
    - It projects next season from everything known at the end of that
      season.
    - For the season in progress, it uses data up to the data date, at his
      age on that date.
    - `projection` (boards, cards) is unchanged.
  - **build_player_engine** merges it (asof_abs / asof_age /
    asof_w_evidence / asof_live). `live_ok` = his club has played ≥ 6
    league matches.
  - **eur_intervals.engine_values → (today, season)**, both ranked in
    today's projection pool, so the curve is unchanged.
    - `engine_value_eur` is TODAY's value: live once valued, else the
      summer projection.
    - Today's value is at his age ON `value_date()`, the Monday of the
      current week. Both the summer and the live age move to that date
      (Lucas: "age the summer value day by day"), so nothing steps down when
      live starts. Past-season values keep their as-of ages.
    - `engine_value_eur_season` is each row's value as of its season.
  - **The app follows the selected season:**
    - `engine_value_for_scope` (Analysis tables) and
      `engine_value_for_player` (profile, comparison) use the season value.
    - The live season falls back to today's value until it is valued.
    - All Seasons shows today's value.
  - **Fees** pair with the value as of the sale's season (no look-ahead).
  - The stats-cache stamp carries `value:asof2@<value_date>`, so cached
    frames refresh weekly. Bump the token when the rule changes.
  - Tests: `tests/test_live_value.py`.
- **Live forward projection on the boards (2026-11-12).** Lucas: the live
  projection means "how good a player will be moving forward (including for
  the rest of the current season)", not next season.
  - **build_projection** writes `projection_fwd(_abs)` on LIVE as-of rows:
    `projection_asof − (1 − FWD_YEARS) × age_delta`, FWD_YEARS = 0.5, i.e.
    the level over the 12 months after the data date (half a year of the
    age step instead of a full one; the level part's fit already looks ~7
    months ahead). A definition, not a fit — no rest-of-season ratings
    exist to fit or test it on. build_player_engine merges it as
    `fwd` / `fwd_abs`.
  - **`models/ratings/projection_now.attach`** (called in load_player_engine
    with the app's row pick): `projection_now(_abs)` on every row of a
    player = his live `fwd` once `live_ok` (the value's 6-match switch),
    else the summer projection; `band_now` NaN when live. An engine without
    `fwd` gets no such columns, so a code deploy ahead of the engine
    rebuild never labels a summer projection live.
  - **merge_engine_values_into_stats**: a scope holding the season in
    progress, or All Seasons, shows `projection_now` when the engine has it
    (`app.scope_shows_todays_projection` — the board's 'live / next 12
    months' caption reads the same helper); a past season keeps the
    projection made at its end. The € value is unchanged (next-season
    projection, fee-calibrated).
  - **The card** (`card_scope.card_row`, profile + comparison, and the
    profile's one-pager tile; Lucas: "make the profile card use the live
    projection"): once `projection_now_live`, the card shows the live
    forward projection in every scope — label 'Projection (next 12
    months)', ± `band_now`, Evidence = the live row's `asof_w_evidence`.
    Otherwise the summer 'Projection (next season)' as before.
  - **Career (as-of)** on the card: the engine's `career_asof` sits only on
    projection rows, so card_row fills every row from `asof_career`
    (build_player_engine: the as-of file's career, identical where both
    exist) — the live season and older seasons of players who kept playing
    were blank until 2026-11-12. All Seasons shows the career through his
    latest season by football year (`_latest_by_year`), never a
    minutes-weighted mean of careers-to-date.
  - **Band:** build_projection writes `band_fwd` on live rows = TODAY's
    role band (G2: the summer projection's role SD of next-season forecast
    error, on pairs that ended before the live season), so the ± doesn't
    jump when live starts (conservative for 12 months). build_player_engine
    carries it as `fwd_band`; attach puts it in `band_now`. (The as-of
    `band_sd` — all earlier pairs, in-sample — runs wider, median ~15.)
  - **Fan chart + comparison overlay** (`pitch_interactive.projection_series`,
    "switch the fan chart to the live projection"): live mode plots the
    forward projection at `x_last + 0.5` (the window's middle, tick 'next
    12 mo (proj)') with its band, age at the data date (`asof_age`) and the
    live evidence; before the switch, the summer projection at the next
    season as before.
  - **Headline rule** (`projection_now.projection_is_headline`): the role
    board and the Overview tables rank by projection only for All Seasons,
    the season in progress or a latest COMPLETED season (its summer
    projection covers every player). An older season's rows carry a
    projection only for players last seen then (35–45%), so those scopes
    rank by ACP Rating and the Overview drops the Proj column.
  - **Blank ≠ 0 (RULE):** `calculate_player_percentiles_and_scores` fills
    the scored frame with 0 EXCEPT `ENGINE_NAN_COLUMNS` (ACP rating /
    projection / band / evidence). The boards test `.notna()` to decide
    whether a scope has projections; the old fillna(0) made the 2026/27 role
    board rank every player on a tie of zero projections. The cache stamp
    carries `proj:fwd1` (bump it when this rule changes).
  - Readers that hide 0 must hide NaN too (the profile's stat expanders:
    `.notna() & (!= 0)`), or a blank prints as 'nan'.
  - Tests: `tests/test_projection_now.py` (pick, scopes, headline, merge,
    old engine, NaN kept). `projection_fwd(_abs)` sits in the no-look-ahead
    gate only as a guard: it is NaN up to the cutoffs (live rows only).
  - Plan: ACP_Official/live_projection_board_plan_2026-11-12.md.
- `models/value/availability.py` — the engine value's PLAYING-TIME FACTOR
  (`regular_factor`, 2026-10). The projection can't tell a cameo from a
  regular: a 95' player inherited a starter's value (Rafinha Monteiro, €218k).
  - **Share:** engine `mins_lineup` (exact API minutes) ÷ the minutes
    available to him, i.e. his club's league matches while he was there.
    These are counted MATCH BY MATCH (2026-10-29; it was event-match count
    × coverage ratio × median length):
    - **Schedule:** every match with status 'Played' in `matches_summary`
      (never a fixture). Its team names map to events `team.id`, ignoring
      accents and spacing ('Pevidém', 'Mortágua ').
    - **Excluded:** a per-match season's matches with no per-match API
      minutes (`match_minutes.excluded_match_ids`), because the numerator
      can't count them.
    - **Length, in order of preference:**
      - the API per-match minutes, i.e. the most anyone got (all of Liga 3);
      - the event clock, within 0.6' of the API (the Campeonato, whose
        per-match API minutes are gone);
      - the season median.
    - `len_api` records the API-exact fraction.
  - **Bench term:** `sub_frac` = sub appearances ÷ appearances.
  - **Factor:** a logistic chance of 900'+ the next season, ÷ the same at
    36% share; 1.0 from ~1,000'.
  - **Exempt:** reserve sides (`league_config.is_reserve_team`, '<club> II'),
    since few Liga 3 minutes there often mean first-team football we can't see.
  - **Windows:** from `models/value/transfers.parquet` (Wyscout transfer
    records), widened to every squad appearance, ≥ 6 matches, and NEVER more
    than one season (two clubs, or both leagues in one year).
  - **Club identity:** events `team.id`. NEVER the lineup file's team names,
    which are wrong in some matches.
  - **Nothing inferred from lineups:** no dated move means the whole season.
    A lineup January-arrival rule was tested and removed on 2026-10-29: it
    over-credited arrivals, 19% became regulars vs 35% predicted, with no
    gain. Lineups can't tell "left" from "dropped" either. A record from a
    club to itself (a renewal) is ignored.
  - **Not excused:** red-card suspensions and injury-like gaps were tested
    and didn't help.
  - **Pipeline:** runs in the engine rebuild right after
    build_player_engine.py and before eur_intervals.py. It writes
    `availability.parquet`; `availability.attach()` joins the factor on the
    RAW ids (before any alias remap) in `load_player_engine` and in the
    calibration's `load_inputs`, so the app and the calibration value
    identically.
  - **Display-only rows:** `availability.parquet` also holds rows with
    `in_engine` False and no factor. These are every other player-season with
    exact API minutes (GKs, sub-floor players). `attach()` joins engine rows
    only, so they never touch a value.
  - **Profile card:** the Player Profile header's "Available minutes played"
    card sits next to the value, GKs included. It reads the table via
    `app.load_minutes_share_table` / `get_minutes_share` →
    `availability.scoped_share`.
    - **Headline (descriptive):** minutes ÷ the minutes of the matches he was
      in the matchday squad for (`squad_mins`).
    - **Help:** "in squad N of M league matches at <club>" (`squad_clubs`),
      plus the all-club share the value uses.
    - **Not the value's measure:** squad-only as the value's denominator
      predicted worse (2026-10-29).
    - **Season:** the selected season (its year across both leagues), like
      everything else in the header; the career on All Seasons. The grey line
      shows the season and club (for a mid-season mover, the main club +N).
      Hidden when he has no minutes in the selected season.
    - **Counting:** each season year is counted once, across both leagues
      and all clubs. Available minutes are never below the minutes played.
    - **Marking:** "~" unless every match length came from the API.
  - **Rollover:** `COEF` is pinned; refit with `--fit` and regenerate the
    calibration JSON (its `point_model.availability` records the constants).
  - Plan + evidence: `ACP_Official/value_minutes_plan_2026-10-28.md`.
  - Tests: `tests/test_availability.py`.
- `models/value/fetch_transfers.py` — refreshes `transfers.parquet` in
  scheduled_update.yml (non-fatal).
  - Per TEAM via `/teams/{id}/transfers`. `/seasons/{id}/transfers` is 403
    on our subscription; date ranges are at most one year.
  - Per PLAYER for a team whose call fails.
  - Only active competitions; the Campeonato keeps the seeded records.
  - Wyscout returns a refusal as HTTP 400 with `{"error": {"code": 403}}`
    in the body; read the body.
  - Upserts on (playerId, startDate, fromTeamId, toTeamId, type) and never
    deletes.
  - `transfers.parquet` is in `engine_inputs_fp.py`, so a new move triggers
    the rebuild. Seeded 2026-10-28 from the loader cache (`--seed`).
- `models/similarity/similar_players.py` — the like-for-like search behind
  the profile's 'Similar Players' section (`similar_players_ui.py`) and
  the Shadow Team 'Find players like…' buttons. One vector per
  player-season from the per-season stats caches (36 style per-90 metrics
  + 8 value metrics at half weight + 6 pitch-occupancy scalars from
  models/roles/role_features_season.parquet), percentile-ranked WITHIN the
  position bucket over the whole pool (both leagues, every cached season,
  ≥ 500 min), weighted Euclidean inside the bucket; score = 100 − 50 ×
  d ÷ the bucket's median pair distance. The pool is built at runtime
  (~0.2 s, `app.load_similarity_pool` keyed on `similarity_cache_signature`
  = cache files' mtimes + SIM_VERSION — never a STATS_CACHE_VERSION bump,
  no artifact). Measured (`similarity_validation`, shown in the section's
  expander): self-match top-10 37% vs 1.5% chance, engine-role agreement
  0.80 vs 0.63, style 0.38 vs 0.19 — z-scores and other variants scored
  the same (scratchpad eval 2026-09-06), percentiles kept for robustness
  and the 'both high, top 20%' explanations (value words, never praise —
  'high' ball losses is the bad end). Rules: GKs are not covered (say so,
  never mix them in); the bucket is per season row (first position seen
  THAT season); filters apply AFTER the distance, and in 'latest season per
  player' mode the latest row is chosen BEFORE the filters so a current
  own-club player never resurfaces under an old club; a 0 rating / value in
  the caches means 'unrated' (NaN); bump SIM_VERSION when FEATURES/weights
  change; never present rank 1 as 'the replacement' — the section shows
  shared traits and the validation numbers; 'Compare on radar' only for a
  neighbour in the query's own league AND season (the radars are that
  scope's percentiles). A season below the floor (a new signing) is
  searched through `query_vector` / `neighbours_for_row` — the page's
  stats row placed on the bucket's percentile scale, with the spatial
  scalars from `pool.spatial` — always behind a 'partial profile' warning,
  and never added to the pool. 'Add to Shadow Team' queues
  `shadow_pending_adds`; views/shadow_team.py applies the queue BEFORE its
  multiselects draw, resolving playerIds against its own scope (an
  out-of-scope player stays queued with a warning — a multiselect value
  must be one of its options). Bridges:
  a row click → profile (nav_to_profile keys), 'Compare on radar' →
  `navigation.go_to('Player Comparison', compare_seed_a=, compare_seed_b=)`
  (the Comparison selectors are keyed `player_comparison_a/_template/_b`
  and seeded before they draw), Shadow Team → profile with
  `profile_active_tab='Similar Players'` (settable from another page only).
  Tests: `tests/test_similar_players.py`.
- `models/coaches/` — Coach Profile (`views/coach_profile.py`, Coaches
  group) and the Player Profile's 'Coached by' section
  (`coach_profile.render_coached_by`). `build_coaches.py` (scheduled
  refresh, incremental, `continue-on-error`) writes four committed
  parquets: who coached each team in each match (Wyscout match
  `teamsData.<id>.coachId`; `/matches/{id}/coaches` is 403), coach names
  (`/coaches/{id}` — refused with HTTP 400 for ~1 in 5, shown as
  'Coach #id' unless `coach_names.csv` names them by hand), one row per
  team per match (xPts from shot xG, style counts, game state, recorded
  substitution minutes, line-up changes) and coach × player (exact
  minutes, OBV). `coach_metrics.py` is every number on the page (pure
  pandas). RULES: pre-match ratings are walk-forward BY CALENDAR DAY
  (`add_ratings`: a 15:00 result never informs a 17:00 expectation) with a
  fixed step `RATING_K` — never tune it on the matches it judges; 'Change
  since arrival' needs ≥ 10 earlier rated matches of the team in the same
  competition, else it is blank; a player's baseline is only his matches
  BEFORE his first under that coach; style percentiles are within one
  competition among coaches with ≥ 10 event matches, and the copy says
  'how he plays, not how well'; 'Team level' is never called a coaching
  effect (squad, turnover and regression to the mean are inside it).
  Game-state rows use only matches whose event goals equal the final
  score (own goals are not shot events). Tests: `tests/test_coaches.py`
  (no-look-ahead on a prefix of the shipped table).
  Formations on the coach pages are OUR OWN (2026-11,
  `models/coaches/formations.py`; Lucas: use only ours, no Wyscout
  formation) — starters' mean open-play touch locations to first sub/red
  card, fitted to 7 reference shapes; `team.formation` and position labels
  must not be read in models/coaches (test enforces it). It is the shape
  WITH THE BALL (5-3-2 reads 3-5-2; a high full-back reads as a back
  three). `back3_share` is a stable coach trait (0.91 odd/even). The
  settled ↔ fluid score is formation variety alone (0.64); the change
  rates (0.49 / 0.43) are shown, not scored; an in-match change read was
  tested and dropped (noise). The Team Analysis and Opposition Report
  line-up graphics use the same read through `own_formations.py`
  (`line_up`: matches vote for the formation, each match's eleven is
  fitted to its slots from `models/coaches/team_match_places.parquet`).
  `team.formation` is no longer loaded or read anywhere.
  TRANSFERMARKT LINKS for teams, coaches and players (2026-11,
  `transfermarkt_links.py`).
  Steps `teams` → `managers` → `verify` → `coaches`, run by hand (not in
  a workflow): `valuations/tm_team_links.csv` (teamId → club id; set
  status `manual` to settle a `review` row, it is never overwritten),
  `tm_team_managers.parquet` (each club's manager stints),
  `tm_coach_links.csv` (coachId → Transfermarkt coach, matched by the
  DATES of his matches against those stints: ≥ 70 % and ≥ 8 matches, else
  `review`). `verify` checks each club link against coaches the provider
  did name and sends doubtful ones to `review`. `build_coaches.
  add_transfermarkt` names a coach Wyscout cannot name from an `ok` link
  (hand overrides still win; a stale name is withdrawn) and stamps
  `tm_coach_id`. zerozero.pt is NOT usable: it sits behind a human-
  verification check. Requests: plain client name, ~2 s apart, capped,
  stop at the first refusal — keep it that way.
  PLAYERS: steps `squads` → `players`. `squads` reads, per linked club
  and season, the squad page (ids, birth dates) and the season statistics
  page (appearances, minutes, all competitions) → `tm_squads.parquet`,
  `tm_player_club_seasons.parquet`. `players` links a player when his
  date of birth equals exactly ONE man's in the squad of the same club
  and season → `tm_player_links.csv` (80 % linked; 93 % Liga 3).
  Player Profile → Career (`views/player_career.py`) shows it, labelled
  as Transfermarkt's count (90 per full match, cups included) — never
  mix it into our exact minutes. Transfermarkt's per-player career page
  is NOT readable (it loads from an internal service; do not call it).
  `scrape_transfermarkt.py` (market values, profile, transfers) reads the
  VERIFIED profile from `tm_player_links.csv` when a player has one; the
  name search is only the fallback, and a name-search link whose profile
  has another date of birth is refused (`dob_mismatch`). Its old
  name-search links had 179 players on a namesake's profile
  (`tm_link_conflicts.csv`); the rows read from those profiles were
  removed 2026-11-12 (1,690 value, 1,448 transfer, 147 profile rows).
  Career lists EVERY club from the transfer record (`player_career.clubs`);
  the Team card shows Transfermarkt's club only when he joined it after
  his last match in our data (`team_now`; match date = matches_summary
  `dateutc`, NOT the events' matchTimestamp). Transfermarkt's per-season
  page is not readable as an ordinary page, so the full club list needs
  the transfer record: `python scrape_transfermarkt.py --verified` reads
  every linked player not yet read or read >120 days ago (saves every 25,
  safe to stop and re-run, ~4 h for all).
  FEES FROM TRANSFER RECORDS (2026-11-12): `valuations/build_tm_fees.py`
  writes `valuations/tm_fees.csv` (paid permanent sales by the club he
  last played for in our data, with data before the sale, mid-season or
  within 120 days; never market values). `eur_intervals.read_fee_rows`
  = hand file + that file, a hand row winning within 90 days; the
  calibration AND build_projection's sale-date projections read it.
  RESERVE SIDES ('<club> II') are out of the fee range: their sales
  (5, median 11x the value) are excluded as `reserve_side` and their
  players get no range (`team=` on range_support_reason and every UI
  call). CENTRE DRIFT: Lucas kept the curve again 2026-11-12 (18 sales,
  Liga 3 median 1.18x). The acknowledgement now holds while every
  acknowledged sale is still in the set and the Liga 3 median stays
  within 0.15 of 1.18 - added sales alone no longer re-open it. Where
  the hand file and Transfermarkt disagree on a fee, the hand file wins
  (Lucas 2026-11-12).
  SCHEDULED: `.github/workflows/transfermarkt_refresh.yml` runs the whole
  chain (new clubs, every club's coaches, new squads, player links, every
  linked player's profile + transfers) at 01:00 US Eastern on 1 Jan, 1 Feb,
  1 Jun, 1 Aug, 1 Sep; commits, deploys, and opens an issue if
  Transfermarkt refuses it. Not yet proven from GitHub's servers.
  `python scrape_transfermarkt.py --relink` re-reads those players from
  the verified profile; it calls Transfermarkt's internal data service,
  so Lucas runs it himself (Claude must not).
  NO NOISY PREDICTIVE METRICS on the coach pages (Lucas 2026-11). A
  measure read as a TRAIT (style, flexibility, tendencies: how he sets
  up, what he would do again) is shown only if a coach's odd and even
  matches agree at ≥ 0.40 on Liga 3 (`MIN_SHOWN_RELIABILITY`);
  `NOISY_HIDDEN` lists what that removed (opponent adaptation, style
  travel between clubs, reaction to defeat). Lucas kept three below the
  line: set-piece share of xG (Style), first-sub minute and subs before
  60' (In-game record). DESCRIPTIVE numbers — a record of what
  happened (points − expected, change since arrival, the In-game record:
  game states raw and vs expected, points after trailing, leads not won,
  late xGD, home-away gap; the player-development case tables) — stay
  whatever their agreement, worded as description. Decide which kind a
  new number is before adding it; measure odd/even agreement if predictive.
  Tendencies section (2026-11): `coach_metrics.TENDENCIES` / `tendencies()`
  — goal kicks short, short corners, centre attacks, left-right balance,
  fouls, cards, xGD after 75', home-away gap, changes after a defeat,
  reaction to defeat; each with league median, percentile and odd/even
  stability (0.79 … -0.20; the noisy ones are labelled, not hidden).
  Added 2026-11-12: tempo (passes per minute on the ball, 0.90),
  possession length (0.89; `poss_seconds` = each own possession's
  duration once) and fixed eleven (`core_xi_share`, share of starts taken
  by the 11 most-used starters per club-season with 8+ sheets, 0.82).
  Tested the same day and NOT built (Lucas: too noisy): substitutions by
  the score and style leading vs trailing (-0.23 … 0.17).
  COACH COMPARISON (`views/coach_comparison.py`, 2026-11-13, Coaches
  group): two coaches of ONE league, each on his whole record there,
  through the profile's own `summarise` / `league_comparison` /
  `coach_table` — add a measure to the profile and to the matching spec
  list there (RESULTS, SETUP, IN_GAME, SQUAD; style and tendencies read
  `STYLE_METRICS` / `TENDENCIES` directly). Coach B opens on the closest
  style; the Coach Profile's 'Compare with another coach' button seeds
  Coach A (`coach_compare_seed_a`) and the page follows him to his league.
  It is never a ranking: it says when the team-level intervals overlap.
  Coach RATING: feasibility tested 2026-11 and NOT built — after removing
  squad quality (starters' prior OBV/90, r 0.66 with team-season xGD) a
  coach's residual does not agree between odd and even matches of one
  spell (-0.05, 39 spells) and carries to the next club at +0.40
  [-0.18, +0.89] on 15 pairs; implied true sd ~0.14 xGD/match against
  0.18 of noise at 20+ matches. Scripts: scratchpad only.
  Flexibility + Young players sections (2026-11): `flexibility` =
  formation variety / structure change rate / back-line switch (see above),
  plus `opp_adapt` (noise-corrected slope of the context-adjusted style
  scores on opponent strength — shown, labelled noisy at 0.24, NOT in the
  score; before the correction it read 0.44 only because short records
  are noisier). Formation variety is inverse Simpson with the unbiased
  estimator, so it does not grow with match count; `z_<metric>` is fitted walk-forward by calendar
  month per competition (`add_context_adjusted`). `youth_trust` = share
  of minutes ÷ share of matchday-squad places (line-up sheet: starters +
  bench). RULES: every measure shown as a coach trait carries its
  odd/even-match reliability (`cm.reliability`, shown as stable /
  moderate / noisy) — match-to-match style distance (Hudl 2025) and the
  spread of the style scores scored ≈ 0 on Liga 3 and are deliberately
  NOT on the page; set-piece share of xG and xGD-when-trailing are also
  ≈ 0 and are captioned as 'what happened'. Player development (vs own
  earlier record, vs expected for age from spells that had already
  ENDED, first vs second half of minutes) is a list of cases, never a
  coach rating: random split-halves of a coach's players agree at only
  0.13–0.22. Exact minutes exist for Liga 3 only, so the trust and
  development views are blank for the Campeonato. 121 line-up sheets in
  match_lineups.pkl are filed under the OPPONENT's name —
  `verify_sheet_teams` re-labels a sheet to the team its starters acted
  for in the events; never read a sheet's team from the pkl key alone.
- Scrollbars (RULE): the theme styles `::-webkit-scrollbar` (thin 6 px
  bars), which makes Chrome's scrollbars layout-taking; any scrollbar that
  toggles then shifts the whole column sideways ('charts and text shift
  back and forth', 2026-09-07). `scrollbar-gutter: stable` on stMain and
  the sidebar (app.py CSS block) reserves the space — keep it whenever the
  scrollbar styling stays. Plotly charts take no `'responsive': True`
  config: Streamlit already sizes them to the container.
- `models/templates/percentiles.py` — the minutes floor behind the template
  percentiles (`minutes_floor`: half the current leader's minutes, capped at
  500' — rises smoothly; it no longer jumps to 500' when one player passes it) and `score_below_floor`,
  which places a sub-floor player into the qualifying sample ON THEIR OWN
  with the pipeline's own arithmetic. RULE: `calculate_player_percentiles_
  and_scores` leaves every player under the floor at 0 in all `_percentile`
  / `_Score` columns (deliberate for the leaderboards) — a radar of such a
  player must go through `score_below_floor` first (the bulk export does,
  and stamps a 'provisional' line), and any reader of a scored frame takes
  the floor from `minutes_floor`, never a literal 500 (the raw-mode
  population and the distribution panels did, and were empty for the first
  weeks of 2026/27). Tests: `tests/test_template_percentiles.py`.
- Player Comparison (`views/player_comparison.py`, 2026-10) mirrors the
  profile two-up: header (value + likely-fee range, role / style), ACP
  Index, career & projection overlay (`pitch_interactive.
  plotly_projection_overlay`, Season / Age x-axis — one data step with the
  profile's fan, `projection_series`; both players on ONE rating scale,
  L3-equivalent whenever the careers span two leagues), template radar
  (percentile / raw mean ± 2σ, DefR), ACP Index cards + tendencies,
  grouped stats (per 90 / totals). RULE: both players are
  ranked against ONE sample — the selected template's group at/above
  `minutes_floor` — via `template_pct.rank_in_template` (a qualifying
  in-group player keeps the pipeline's numbers; sub-floor / out-of-group
  players are placed and the page says so); the profile ranks a player in
  his own group. The ACP Index card row (scope pick, All-Seasons career
  aggregation, projection source) is `models/ratings/card_scope.card_row`
  (`app.engine_card_row`), shared by both pages — change it there, never
  inline. Keep the comparison's feature set in step when the profile gains
  a player-level section. The template dropdown offers the templates of
  every position Player A PLAYED in the scope (exact API minutes,
  `app.get_player_minutes_by_position`, >= 10% of his minutes) plus his
  stats-frame position; most-played position first, default = his best
  fit there via `rank_in_template`. Never the frame's position alone: a
  row keeps ONE position, so a career striker playing LAMF this season
  would be offered only winger templates in 2026/27 (Nicolás Souza,
  2026-10-27).
- Multi-season position (RULE, v16): an All-Seasons / multi-season
  minutes row takes the MOST-PLAYED position —
  `precompute_minutes.combine_season_minutes`: each season row's minutes
  credited to that season's primaryPosition (its most-played position by
  exact API minutes), ties to the earlier frame (newest season first
  within a league; the both-leagues scope lists every Liga 3 season before
  the Campeonato's), 'Unknown' / 0 never vote; name and team stay the
  first frame's. Until v16 it was groupby
  'first' — the newest season's position however few minutes: 207 Liga 3
  and 70 Campeonato players with 500'+ were ranked in the wrong template
  group (João Tarzan, 10,580' of 10,752' at CF, ranked as LAMF). Every
  live multi-season combine goes through that function (still 'first',
  and read by nothing: app.get_combined_career_minutes, no callers;
  process_data.py's all_time_player_minutes.pkl, overwritten by
  precompute_minutes right after).
- Widget seeding (RULE, 1.50 too): a keyed selectbox whose OPTIONS change
  between runs is a new widget and falls back to index 0 unless its key is
  set in THAT run — even when its value is still a valid option
  (tests/test_widget_reseed.py). A selector whose options follow the scope
  or another selector re-sets its session-state value every run before it
  draws (Player Comparison A / template / B; 'Paulinho vs Paulinho').
- `@st.cache_data` never hashes an underscore argument: a helper cached
  on `_events_df` serves the first scope computed for every scope. Key
  such helpers on a hashable scope (`app._scope_match_ids`) — the
  position-minutes helpers were wrong this way until 2026-10-27.
- Metric cards (RULE): `st.metric` text wraps at word breaks (app.py CSS),
  never '…'. A row of cards holding WORDS (team, nationality, role, style)
  is created inside `st.container(key=theme.metric_fit_key(name, *values))`
  — the key's bucket caps the value font so the row's longest word fits
  its card whole (`theme.metric_fit_css`, cqi units on the card; the width
  estimate is calibrated on measured font widths, tests/test_metric_fit.py).
  Put a caption under the row, not inside a column: a row whose columns
  each hold ONE element keeps its cards one height.
- `scripts/config_migrations/` — spent one-shot config.yaml scripts (README).
- Match Predictor "Team Strength Ratings" = the Dixon-Coles attack /
  defence parameters (`DixonColes.strength_table` → `scoreline_ui.
  render_strength_table`; Scores × / Concedes × / xGD vs a league-average
  side at a neutral venue, centred on the league's own sides), read from
  the same `dc_params.json` as the scoreline forecast (a team the fit has
  never seen is shown at the average and played at the league mean in the
  forecast — the caption says so). Since 2026-09 it replaced
  the per-match xG rates × schedule ratio as the rating: a joint fit with
  shrinkage and time decay is the only one of the two that knows four
  games are not thirty. The per-season rates below are context, behind a
  toggle — never rank by them in copy.
- `models/strength/sos.py` — the strength-of-schedule adjustment behind
  that toggle's per-season xG-rate table (pure pandas;
  `app.calculate_sos_adjusted_strength` is the cached wrapper). Direction
  RULE: Def Strength is goals/xG conceded (lower = better), so a tough
  schedule is credited — attack × (league conceded ÷ opponents' conceded),
  defence × (league scored ÷ opponents' scored), opponents' PRE-match
  strength, ≥ 3 such opponents, each factor clipped to `FACTOR_BOUNDS`
  (0.5–2.0). Until 2026-09 both factors were inverted
  (Caldas outranked Atlético and Mafra after MW4 for playing weak teams).
  The three season-keyed caches (`team_strength_`, `rolling_strength_`,
  `sos_strength_<SOS_CACHE_VERSION>_`) are dropped together for current
  seasons by the boot precompute; bump `SOS_CACHE_VERSION` when the
  formula changes. Tests: `tests/test_sos_direction.py`.

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

### Team pages: sections (2026-11)

Both pages draw ONE section at a time (`page_sections.py`): Team Analysis
shows `TEAM_SECTIONS` (Overview / Players / Set pieces / Form & shots /
Tactics), the Opposition Report `OPP_SECTIONS` = its own "Summary" in front
of the same five. The sections ARE the pages' old blocks, in their old order
(Overview = the four radars, unchanged, + season report + on-ball value &
phases). A shared visual goes into the SAME section on both pages, with one
standing exception: the formation graphic is under "Players" on Team
Analysis (primary formation + roster) and in the Opposition Report's
"Summary" (projected line-up: page-specific, and answer-first there).

- Team Analysis: `if sections.is_open(name):` — a closed block does not run,
  so anything a later block needs (starting XI, OBV frames, team id) is
  computed above the blocks.
- Opposition Report: `with sections.block(name):` — every block still RUNS,
  in the old order, because the takeaways read key_players / set_piece_df /
  form_results and the PDF needs every block's tables and figures. A closed
  block writes into a hidden container (`page_sections.SINK_KEY`, CSS in
  app.py). Inside it `sections.drawing` is False: `_show_opp_png` then
  QUEUES the figure (`_pdf_only`) instead of building it, and every
  `st.plotly_chart` (and the season report) sits under `if
  sections.drawing:`. "Generate PDF Report" sits beside the section row and
  builds whatever the run queued. RULE: a new figure goes through
  `_show_opp_png` / `_pdf_png` with a `pdf_key` if the PDF prints it; a new
  Plotly chart or keyed widget is guarded by `sections.drawing`; never put
  PDF inputs under `if sections.is_open(...)` on this page.
  `tests/test_smoke_pages.py::test_opposition_pdf_is_complete_from_any_section`
  compares the PDF's figures and texts across all six sections, with a
  stand-in renderer. LESSON (2026-11-13): never build the report's real
  figures all in one AppTest run — the PDF button renders about 26 in one
  run (+5 GB at the peak) and GitHub's runner was shut down mid-test, which
  blocked a deploy ("The runner has received a shutdown signal").
- Summary's three parts (takeaways, strengths & weaknesses, line-up) are
  drawn in that order through containers created at the top
  (`_takeaways_slot` / `_synopsis_slot` / `_lineup_slot`), although their
  code runs later.

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

`python -m pytest tests -v`. CI runs PYTHON 3.11 (the Space too) while
the local base env is 3.13: never nest an f-string's own quote inside its
braces (legal from 3.12 only) — `tests/test_py311_syntax.py` scans for it.
The Space pins Streamlit 1.50.0 (README
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
