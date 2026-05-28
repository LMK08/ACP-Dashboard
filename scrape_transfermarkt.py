#!/usr/bin/env python3
"""Scrape Transfermarkt market values for every player in our Wyscout
dataset and write to valuations/valuations.parquet (transfermarkt
rows).

Usage:
    python scrape_transfermarkt.py            # full pull, all players
    python scrape_transfermarkt.py --limit 50 # smoke-test 50 players
    python scrape_transfermarkt.py --player-id 12345  # single player

Behavior:
- Pulls player list from raw_events.parquet + matches_summary.parquet
  (same source the dashboard uses).
- For each unique playerId, searches Transfermarkt by name + DOB,
  picks the best match, fetches the market-value history page,
  appends rows to valuations/valuations.parquet keyed by
  (playerId, source='transfermarkt', as_of_date).
- Skips players already in the parquet with a recent as_of_date
  (so re-runs only refresh stale entries — controlled via
  --max-age-days, default 14).
- Writes a match audit log to valuations/tm_match_log.csv so we can
  spot-check fuzzy matches.

Implementation notes:
- Transfermarkt has no public API — we use the search page +
  player-page HTML scrape. Rate limit: ~1 req/sec to be polite.
- Use a long-lived requests.Session with a real user-agent.
- DOB is used as the primary disambiguator. Without it, we fall
  back to current-team match.

TODO (next session):
- Implement the actual scraping HTML parsers (search + market value
  history page). Currently this script is the schema + plumbing
  skeleton.
- Add ZeroZero scraper.
- Add manual-entry UI in the dashboard.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
VAL_DIR = HERE / 'valuations'
VAL_DIR.mkdir(exist_ok=True)
VALUATIONS_PATH = VAL_DIR / 'valuations.parquet'
MATCH_LOG_PATH  = VAL_DIR / 'tm_match_log.csv'

TM_BASE = 'https://www.transfermarkt.com'
TM_SEARCH = TM_BASE + '/schnellsuche/ergebnis/schnellsuche'
USER_AGENT = (
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
    'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0 Safari/537.36'
)
RATE_LIMIT_SEC = 1.2   # polite delay between requests

LOG = logging.getLogger('scrape_transfermarkt')


# ---------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------
VALUATION_SCHEMA = {
    'playerId':   'int64',
    'source':     'string',
    'value_eur':  'float64',
    'as_of_date': 'string',   # YYYY-MM-DD; keep as string for parquet portability
    'season_id':  'Int64',
    'source_url': 'string',
    'notes':      'string',
}


@dataclass
class TMMatch:
    """A single fuzzy-match candidate from a TM search result."""
    tm_id:      int
    tm_url:     str
    name:       str
    dob:        str | None
    club:       str | None
    score:      float = 0.0
    fields_used: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------
# Player list source (from raw_events + matches_summary)
# ---------------------------------------------------------------------
def load_player_universe() -> pd.DataFrame:
    """Return a DataFrame of unique players to scrape: playerId,
    playerName, most-recent team, most-recent season."""
    events_path = HERE / 'raw_events.parquet'
    matches_path = HERE / 'matches_summary.parquet'
    if not events_path.exists():
        raise SystemExit(f"Missing {events_path} — run process_data.py first.")

    ev = pd.read_parquet(events_path,
                          columns=['player.id', 'player.name',
                                    'team.name', 'matchId'])
    if matches_path.exists():
        ms = pd.read_parquet(matches_path, columns=['matchId', 'seasonId', 'dateutc'])
        ev = ev.merge(ms, on='matchId', how='left')
    ev = ev.dropna(subset=['player.id'])
    ev['player.id'] = ev['player.id'].astype('int64')

    # Most recent appearance per player, used as the lookup snapshot.
    if 'dateutc' in ev.columns:
        ev = ev.sort_values('dateutc')
    last = (ev.dropna(subset=['player.name'])
              .groupby('player.id', as_index=False)
              .agg(playerName=('player.name', 'last'),
                    teamName=('team.name', 'last'),
                    seasonId=('seasonId', 'last')))
    last = last.rename(columns={'player.id': 'playerId'})
    LOG.info(f"Loaded universe: {len(last):,} unique players")
    return last


# ---------------------------------------------------------------------
# Player birth-date enrichment (for DOB-based matching)
# ---------------------------------------------------------------------
def load_player_dob_map() -> dict:
    """Map playerId → birthDate (YYYY-MM-DD) from player_details.pkl
    where available. Used as the primary disambiguator when multiple
    TM search results share a name."""
    pd_path = HERE / 'player_details.pkl'
    if not pd_path.exists():
        return {}
    try:
        details = pd.read_pickle(pd_path)
    except Exception as e:
        LOG.warning(f"Could not read player_details.pkl: {e}")
        return {}
    if 'birthDate' not in details.columns:
        return {}
    return {int(pid): str(bd)[:10]
             for pid, bd in details['birthDate'].dropna().items()}


# ---------------------------------------------------------------------
# HTTP session + rate limiting
# ---------------------------------------------------------------------
def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({'User-Agent': USER_AGENT,
                       'Accept-Language': 'en-US,en;q=0.9'})
    return s


def polite_get(session: requests.Session, url: str, **kw) -> requests.Response | None:
    time.sleep(RATE_LIMIT_SEC)
    try:
        r = session.get(url, timeout=20, **kw)
        return r
    except requests.RequestException as e:
        LOG.warning(f"GET {url} failed: {e}")
        return None


# ---------------------------------------------------------------------
# Transfermarkt scrape — TO BE IMPLEMENTED in next session
# ---------------------------------------------------------------------
def search_transfermarkt(session, name: str, dob: str | None = None) -> list[TMMatch]:
    """Hit TM's quick-search endpoint, return ranked match candidates.

    TODO: parse the result HTML — extract /profil/spieler/<id> links
    plus the player's name, DOB, and current club from the result row.
    Score candidates: exact-name + DOB-match = highest; name-only =
    lower. Return sorted by score desc.
    """
    raise NotImplementedError("Implement in next session: TM search HTML parser")


def fetch_market_value_history(session, tm_id: int) -> list[dict]:
    """For a TM player ID, fetch their full market-value history.
    Returns a list of {as_of_date, value_eur, source_url} dicts.

    TODO: parse the data-table-marktwertentwicklung JSON embedded
    in the player's profile page. Each row = a TM revaluation.
    """
    raise NotImplementedError("Implement in next session: TM MV history parser")


# ---------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------
def load_existing_valuations() -> pd.DataFrame:
    """Read valuations.parquet if it exists; otherwise return empty
    DF with the canonical schema."""
    if not VALUATIONS_PATH.exists():
        return pd.DataFrame({c: pd.Series(dtype=t)
                              for c, t in VALUATION_SCHEMA.items()})
    return pd.read_parquet(VALUATIONS_PATH)


def append_rows(rows: list[dict]) -> None:
    """Append new rows and dedupe by (playerId, source, as_of_date)
    keeping the LAST entry."""
    if not rows:
        return
    existing = load_existing_valuations()
    new_df = pd.DataFrame(rows)
    for c, t in VALUATION_SCHEMA.items():
        if c not in new_df.columns:
            new_df[c] = pd.Series(dtype=t)
    combined = (pd.concat([existing, new_df], ignore_index=True)
                  .drop_duplicates(['playerId', 'source', 'as_of_date'],
                                    keep='last'))
    combined.to_parquet(VALUATIONS_PATH, index=False)
    LOG.info(f"Wrote {len(combined):,} rows to {VALUATIONS_PATH} "
             f"(added {len(new_df):,}, deduped to {len(combined):,})")


# ---------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------
def run(*, limit: int | None = None, player_id: int | None = None,
        max_age_days: int = 14) -> None:
    universe = load_player_universe()
    dob_map = load_player_dob_map()
    if player_id:
        universe = universe[universe['playerId'] == player_id]
    if limit:
        universe = universe.head(limit)
    LOG.info(f"Scraping TM for {len(universe):,} players")

    existing = load_existing_valuations()
    if not existing.empty:
        tm_existing = existing[existing['source'] == 'transfermarkt'].copy()
        # Re-pull anything whose newest entry is older than max_age_days
        tm_existing['_d'] = pd.to_datetime(tm_existing['as_of_date'], errors='coerce')
        newest = tm_existing.groupby('playerId')['_d'].max().to_dict()
    else:
        newest = {}

    session = make_session()
    fresh_rows: list[dict] = []
    match_audit: list[dict] = []
    today = date.today()

    for _, row in universe.iterrows():
        pid = int(row['playerId'])
        # Skip if scraped recently
        last = newest.get(pid)
        if pd.notna(last) and (today - last.date()).days < max_age_days:
            continue
        try:
            candidates = search_transfermarkt(session,
                                                row['playerName'],
                                                dob_map.get(pid))
        except NotImplementedError:
            LOG.warning("Scraper not implemented yet — exiting cleanly.")
            return
        if not candidates:
            match_audit.append({'playerId': pid, 'name': row['playerName'],
                                 'status': 'no_match'})
            continue
        best = candidates[0]
        match_audit.append({'playerId': pid, 'name': row['playerName'],
                             'status': 'matched', 'tm_id': best.tm_id,
                             'tm_name': best.name, 'score': best.score})
        history = fetch_market_value_history(session, best.tm_id)
        for h in history:
            fresh_rows.append({
                'playerId':   pid,
                'source':     'transfermarkt',
                'value_eur':  h.get('value_eur'),
                'as_of_date': h.get('as_of_date'),
                'season_id':  pd.NA,  # season inference happens at load
                'source_url': h.get('source_url'),
                'notes':      None,
            })
    append_rows(fresh_rows)
    if match_audit:
        pd.DataFrame(match_audit).to_csv(MATCH_LOG_PATH, index=False)
        LOG.info(f"Match audit written to {MATCH_LOG_PATH}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--player-id', type=int, default=None)
    ap.add_argument('--max-age-days', type=int, default=14)
    ap.add_argument('--verbose', '-v', action='store_true')
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s  %(levelname)-7s  %(message)s',
        datefmt='%H:%M:%S',
    )
    run(limit=args.limit, player_id=args.player_id, max_age_days=args.max_age_days)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
