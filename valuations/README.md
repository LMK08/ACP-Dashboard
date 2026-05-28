# Valuations infrastructure

Ground-truth market values for the CVI calibration loop. Sources land
in `valuations.parquet` keyed by `(playerId, source, as_of_date)`.

## Schema

| column | type | notes |
|---|---|---|
| `playerId` | int64 | Wyscout player id (matches `raw_events.player.id`) |
| `source` | str | one of `transfermarkt` / `zerozero` / `reported_fee` / `manual` |
| `value_eur` | float | market value or fee in euros |
| `as_of_date` | date | snapshot date for MV; event date for fees |
| `season_id` | int64 | season the value applies to (best-effort) |
| `source_url` | str | provenance link (if scraped) |
| `notes` | str | freeform — esp. for manual + reported_fee entries |

Multiple sources per player + season are kept side-by-side; the v2
calibration regression can either average them, weight by source
authority, or use `source` as a categorical feature.

## Sources

### `transfermarkt`
Public market values from transfermarkt.com. Crowd-edited but widely
cited. Scraper in `scrape_transfermarkt.py`. Matches Wyscout players
by name + DOB (fuzzy) — match log written to
`valuations/tm_match_log.csv` for audit.

### `zerozero`
Portuguese-focused market values from zerozero.pt. Different
methodology than TM — useful 2nd opinion. Scraper TBD.

### `reported_fee`
Hand-curated table of actual reported transfer fees (when published
by club/press). Highest authority but sparsest. Input via
`valuations/reported_fees.csv` (committed to repo).

### `manual`
Hand-entered estimates from club/agent conversations. Highest
authority for the specific player. Input via the dashboard's manual
entry UI (TBD) which writes to `valuations/manual_entries.csv`.

## Refresh cadence

Transfermarkt + ZeroZero are scraped on a separate (weekly?) cadence
— they don't need the nightly data refresh's tight loop. Manual +
reported entries are user-driven.
