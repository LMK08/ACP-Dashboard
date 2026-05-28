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
import re
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
    playerName (full first+last from player_details when available,
    falling back to the abbreviated Wyscout name), teamName,
    seasonId. Full names are critical — TM's quick search doesn't
    expand 'S. Iheanacho' → 'Stanley Iheanacho'."""
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

    if 'dateutc' in ev.columns:
        ev = ev.sort_values('dateutc')
    last = (ev.dropna(subset=['player.name'])
              .groupby('player.id', as_index=False)
              .agg(wy_name=('player.name', 'last'),
                    teamName=('team.name', 'last'),
                    seasonId=('seasonId', 'last')))
    last = last.rename(columns={'player.id': 'playerId'})

    # Enrich with full firstName + lastName from player_details.pkl
    pd_path = HERE / 'player_details.pkl'
    if pd_path.exists():
        try:
            details = pd.read_pickle(pd_path)
            if isinstance(details, list):
                details = pd.DataFrame(details)
            if 'playerId' in details.columns:
                details = details[['playerId', 'firstName', 'lastName']].copy()
                details['playerId'] = pd.to_numeric(details['playerId'],
                                                      errors='coerce').astype('Int64')
                details = details.dropna(subset=['playerId'])
                details['playerId'] = details['playerId'].astype('int64')
                last = last.merge(details, on='playerId', how='left')
                last['playerName'] = last.apply(
                    lambda r: (f"{r['firstName']} {r['lastName']}".strip()
                                if pd.notna(r.get('firstName'))
                                   and pd.notna(r.get('lastName'))
                                   and (r['firstName'] or r['lastName'])
                                else r['wy_name']),
                    axis=1,
                )
                last = last.drop(columns=['firstName', 'lastName'])
            else:
                last['playerName'] = last['wy_name']
        except Exception as e:
            LOG.warning(f"Could not enrich names from player_details.pkl: {e}")
            last['playerName'] = last['wy_name']
    else:
        last['playerName'] = last['wy_name']

    n_expanded = (last['playerName'] != last['wy_name']).sum()
    LOG.info(f"Loaded universe: {len(last):,} unique players "
             f"({n_expanded:,} with full names from player_details)")
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
    # player_details.pkl is a list[dict]; normalize.
    if isinstance(details, list):
        details = pd.DataFrame(details)
    if 'birthDate' not in details.columns or 'playerId' not in details.columns:
        return {}
    out = {}
    for _, row in details.iterrows():
        pid = row.get('playerId'); bd = row.get('birthDate')
        if pid is None or bd is None: continue
        try:
            out[int(pid)] = str(bd)[:10]
        except (TypeError, ValueError):
            continue
    return out


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
import unicodedata


def _ascii_fold(s: str) -> str:
    """'Atlético' → 'atletico'. Used for fuzzy name + club matching."""
    if s is None:
        return ''
    return ''.join(c for c in unicodedata.normalize('NFKD', str(s))
                    if not unicodedata.combining(c)).lower().strip()


def _name_similarity(a: str, b: str) -> float:
    """Cheap token-set Jaccard on ascii-folded names. 1.0 = identical,
    0.0 = no overlap. Avoids difflib dep + handles re-ordered names
    like 'Souza Nicolas' vs 'Nicolas Souza'."""
    ta = set(_ascii_fold(a).split())
    tb = set(_ascii_fold(b).split())
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


def search_transfermarkt(session, name: str, dob: str | None = None,
                          expected_age: int | None = None,
                          expected_clubs: list[str] | None = None) -> list[TMMatch]:
    """Hit TM's quick-search endpoint and return ranked match candidates.

    Scoring (higher = better):
      name similarity 0..1   * 100
      + age within ±1 yr       + 30
      + age within ±3 yr       + 10
      + club name overlap      + 25 per matching token
    """
    r = polite_get(session, TM_SEARCH, params={'query': name})
    if r is None or r.status_code != 200:
        return []
    soup = BeautifulSoup(r.text, 'html.parser')

    # Only consider the players-results table (the first .items table on
    # the page; subsequent ones are clubs/coaches/etc.).
    table = soup.find('table', class_='items')
    if table is None:
        return []

    candidates: list[TMMatch] = []
    for row in table.find_all('tr', class_=['odd', 'even']):
        link = row.find('a', href=re.compile(r'/profil/spieler/\d+'))
        if link is None:
            continue
        m = re.search(r'/profil/spieler/(\d+)', link['href'])
        if not m:
            continue
        tm_id = int(m.group(1))
        tm_url = TM_BASE + link['href'].split('?')[0]
        tm_name = link.get_text(strip=True)
        if not tm_name:
            continue
        cells = [c.get_text(' ', strip=True) for c in row.find_all('td')]
        # TM quick-search row schema (10 cells):
        #   [0] dumped text (incl name + club)
        #   [1] img · [2] name · [3] club · [4] position
        #   [5] flag · [6] age · [7] flag · [8] market value · [9] relatives
        club = cells[3] if len(cells) > 3 else None
        try:
            age = int(cells[6]) if len(cells) > 6 and cells[6].isdigit() else None
        except ValueError:
            age = None

        # ---- score the match ----
        score = _name_similarity(tm_name, name) * 100.0
        fields_used = ['name']
        if expected_age is not None and age is not None:
            diff = abs(age - expected_age)
            if diff <= 1:
                score += 30.0; fields_used.append('age±1')
            elif diff <= 3:
                score += 10.0; fields_used.append('age±3')
        if expected_clubs and club:
            club_a = _ascii_fold(club)
            for ec in expected_clubs:
                if not ec: continue
                # Token-set overlap so "Atlético CP" matches "Atletico CP"
                ec_tokens = set(_ascii_fold(ec).split())
                club_tokens = set(club_a.split())
                if ec_tokens & club_tokens:
                    score += 25.0 * len(ec_tokens & club_tokens)
                    fields_used.append(f'club:{ec[:20]}')
                    break
        candidates.append(TMMatch(
            tm_id=tm_id, tm_url=tm_url, name=tm_name,
            dob=None, club=club, score=score, fields_used=fields_used,
        ))

    candidates.sort(key=lambda c: -c.score)
    return candidates


def fetch_market_value_history(session, tm_id: int) -> list[dict]:
    """Pull the full TM market-value history via the ceapi JSON endpoint.

    Returns list of dicts: {as_of_date (YYYY-MM-DD), value_eur (int),
    club_at_time, source_url}. Returns [] if the player has no MV
    history (common for sub-tier players).
    """
    url = f"{TM_BASE}/ceapi/marketValueDevelopment/graph/{tm_id}"
    r = polite_get(session, url,
                    headers={'X-Requested-With': 'XMLHttpRequest',
                              'Referer': f'{TM_BASE}/'})
    if r is None or r.status_code != 200:
        return []
    try:
        payload = r.json()
    except ValueError:
        return []
    rows = payload.get('list') if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return []
    out = []
    for row in rows:
        try:
            y = row.get('y')
            x_ms = row.get('x')
            if y is None or x_ms is None:
                continue
            # TM dates come as DD/MM/YYYY in datum_mw or as epoch-ms in x.
            # The datum_mw string is more reliable across locales.
            dm = row.get('datum_mw')
            iso_date = None
            if isinstance(dm, str) and '/' in dm:
                try:
                    d, m, yr = dm.split('/')
                    iso_date = f"{int(yr):04d}-{int(m):02d}-{int(d):02d}"
                except (ValueError, AttributeError):
                    iso_date = None
            if iso_date is None:
                # Fall back to epoch-ms.
                from datetime import datetime, timezone
                iso_date = (datetime.fromtimestamp(int(x_ms) / 1000,
                                                     tz=timezone.utc)
                            .date().isoformat())
            out.append({
                'as_of_date': iso_date,
                'value_eur':  float(y),
                'club_at_time': row.get('verein'),
                'source_url': url,
            })
        except Exception:
            continue
    return out


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

    # Minimum score threshold to accept a match. Tuned conservatively:
    # 100 = identical name only (could be wrong); we want at least name +
    # some secondary signal (age or club).
    MIN_MATCH_SCORE = 120.0

    from tqdm import tqdm
    for _, row in tqdm(list(universe.iterrows()), total=len(universe),
                        desc="TM scrape"):
        pid = int(row['playerId'])
        # Skip if scraped recently
        last = newest.get(pid)
        if pd.notna(last) and (today - last.date()).days < max_age_days:
            continue

        # Compute expected age from DOB
        expected_age = None
        dob_str = dob_map.get(pid)
        if dob_str:
            try:
                bd = datetime.strptime(dob_str[:10], '%Y-%m-%d')
                expected_age = (today - bd.date()).days // 365
            except (ValueError, AttributeError):
                pass

        # Expected club = the player's most recent team in our data
        expected_clubs = [row['teamName']] if pd.notna(row.get('teamName')) else []

        try:
            candidates = search_transfermarkt(
                session, row['playerName'],
                dob=dob_str,
                expected_age=expected_age,
                expected_clubs=expected_clubs,
            )
        except NotImplementedError:
            LOG.warning("Scraper not implemented yet — exiting cleanly.")
            return

        if not candidates:
            match_audit.append({'playerId': pid, 'name': row['playerName'],
                                 'status': 'no_match'})
            continue

        best = candidates[0]
        if best.score < MIN_MATCH_SCORE:
            # Best candidate didn't clear our minimum confidence bar —
            # log for hand-review but don't pollute valuations.parquet.
            match_audit.append({
                'playerId': pid, 'name': row['playerName'],
                'status': 'low_confidence', 'tm_id': best.tm_id,
                'tm_name': best.name, 'score': round(best.score, 1),
                'fields': ','.join(best.fields_used),
                'tm_url': best.tm_url,
                'expected_club': (expected_clubs[0] if expected_clubs else ''),
                'tm_club': best.club or '',
            })
            continue

        match_audit.append({
            'playerId': pid, 'name': row['playerName'],
            'status': 'matched', 'tm_id': best.tm_id,
            'tm_name': best.name, 'score': round(best.score, 1),
            'fields': ','.join(best.fields_used),
            'tm_url': best.tm_url,
            'expected_club': (expected_clubs[0] if expected_clubs else ''),
            'tm_club': best.club or '',
        })

        history = fetch_market_value_history(session, best.tm_id)
        for h in history:
            fresh_rows.append({
                'playerId':   pid,
                'source':     'transfermarkt',
                'value_eur':  h.get('value_eur'),
                'as_of_date': h.get('as_of_date'),
                'season_id':  pd.NA,  # season inference happens at load
                'source_url': h.get('source_url'),
                'notes':      (f"tm_id={best.tm_id} club_at_time="
                                f"{h.get('club_at_time')}"
                                if h.get('club_at_time') else f"tm_id={best.tm_id}"),
            })
    append_rows(fresh_rows)
    if match_audit:
        pd.DataFrame(match_audit).to_csv(MATCH_LOG_PATH, index=False)
        LOG.info(f"Match audit written to {MATCH_LOG_PATH} "
                  f"({sum(1 for r in match_audit if r['status']=='matched')} matched, "
                  f"{sum(1 for r in match_audit if r['status']=='low_confidence')} low-confidence, "
                  f"{sum(1 for r in match_audit if r['status']=='no_match')} no-match)")


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
