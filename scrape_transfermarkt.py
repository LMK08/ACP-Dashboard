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
PLAYER_LINKS_PATH = VAL_DIR / 'tm_player_links.csv'      # transfermarkt_links.py
LINK_CONFLICTS_PATH = VAL_DIR / 'tm_link_conflicts.csv'  # transfermarkt_links.py

TM_BASE = 'https://www.transfermarkt.com'
TM_SEARCH = TM_BASE + '/schnellsuche/ergebnis/schnellsuche'
USER_AGENT = (
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
    'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0 Safari/537.36'
)
RATE_LIMIT_SEC = 1.2   # polite delay between requests
SAVE_EVERY = 25        # players between saves in a run
MAX_FAILS_IN_A_ROW = 8 # failed reads in a row before a run stops

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
    falling back to the abbreviated Wyscout name), teamName (most-
    recent for display), career_clubs (ALL distinct clubs the player
    has played for in our data — used for TM matching), seasonId.

    Full names are critical — TM's quick search doesn't expand
    'S. Iheanacho' → 'Stanley Iheanacho'.

    career_clubs (v2) — TM's quick search row shows a player's CURRENT
    club. If a player has retired or moved on, their TM-listed club
    won't equal our most-recent Wyscout team, so single-club matching
    misses the deal. We now pass every distinct team they've ever
    played for in our raw_events; the matcher takes the best-scoring
    club overlap across the list.
    """
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

    # Full career-clubs list per player — for robust TM matching.
    career_clubs = (ev.dropna(subset=['team.name'])
                       .groupby('player.id')['team.name']
                       .apply(lambda s: list(dict.fromkeys(s.tolist())))
                       .reset_index()
                       .rename(columns={'player.id': 'playerId',
                                          'team.name': 'career_clubs'}))
    last = last.merge(career_clubs, on='playerId', how='left')
    last['career_clubs'] = last['career_clubs'].apply(
        lambda x: x if isinstance(x, list) else []
    )

    # Per-(season, club) ranges per player. Each entry is a dict:
    #   {season_id, season_label, team, first_match, last_match}
    # Useful for matching against TM transfer history (their dates)
    # post-match and for v2 EUR regression features (movement churn).
    if 'seasonId' in ev.columns and 'dateutc' in ev.columns:
        try:
            # Lazy import to keep top of file clean
            from league_config import season_display_name
        except Exception:
            season_display_name = lambda s: str(s)
        per_ss = (ev.dropna(subset=['team.name', 'seasonId'])
                     .groupby(['player.id', 'seasonId', 'team.name'])
                     .agg(first_match=('dateutc', 'min'),
                           last_match=('dateutc', 'max'))
                     .reset_index()
                     .rename(columns={'player.id': 'playerId',
                                        'team.name': 'team'}))
        per_ss['season_label'] = per_ss['seasonId'].apply(
            lambda s: season_display_name(int(s))
                       if pd.notna(s) else None
        )
        per_ss['first_match'] = pd.to_datetime(per_ss['first_match'],
                                                  errors='coerce').dt.date.astype(str)
        per_ss['last_match'] = pd.to_datetime(per_ss['last_match'],
                                                 errors='coerce').dt.date.astype(str)
        cws = (per_ss.groupby('playerId')
                       .apply(lambda g: g[['seasonId', 'season_label',
                                            'team', 'first_match',
                                            'last_match']].to_dict('records'))
                       .reset_index(name='career_with_seasons'))
        last = last.merge(cws, on='playerId', how='left')
        last['career_with_seasons'] = last['career_with_seasons'].apply(
            lambda x: x if isinstance(x, list) else []
        )
    else:
        last['career_with_seasons'] = [[] for _ in range(len(last))]

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


class FetchFailed(Exception):
    """A page that should have data could not be read."""


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
      name similarity 0..1     * 100
      + age within ±1 yr         + 30
      + age within ±3 yr         + 10
      + club name overlap        + 25 per overlapping token
                                  (computed for EACH expected club,
                                   keep the BEST match)

    expected_clubs is now the player's full career-clubs list, not
    just their most-recent team. We try every club and take the best
    matching one — handles retired/loaned players whose TM-listed
    "current club" doesn't match their last Wyscout team.
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
            club_tokens = set(club_a.split())
            # Try every expected club, keep the BEST-scoring overlap.
            # This is critical for retired/loaned players whose
            # current TM club doesn't match their last Wyscout team —
            # we still match on a historical club.
            best_overlap = 0
            best_ec = None
            for ec in expected_clubs:
                if not ec: continue
                ec_tokens = set(_ascii_fold(ec).split())
                # Filter out very common noise tokens that would
                # otherwise inflate matches (e.g. 'FC', 'SC', 'CP',
                # 'CD' appear in many Portuguese club names).
                noise = {'fc', 'sc', 'cp', 'cd', 'sad', 'fca',
                          'sl', 'gd', 'ad', 'os', 'do', 'da', 'de'}
                ec_meaningful = ec_tokens - noise
                club_meaningful = club_tokens - noise
                if not ec_meaningful:
                    ec_meaningful = ec_tokens   # fallback if all-noise
                overlap = len(ec_meaningful & club_meaningful)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_ec = ec
            if best_overlap > 0:
                score += 25.0 * best_overlap
                fields_used.append(f'club:{(best_ec or "")[:20]}')
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
# Rich-metadata fetchers — pulled for every accepted match.
# These don't drive matching (we don't have them at search time)
# but they're useful for v2 EUR regression features, manual audit,
# and verifying that the chosen TM player's transfer history overlaps
# with our (club, season) data.
# ---------------------------------------------------------------------
def _clean_text(node) -> str | None:
    if node is None:
        return None
    t = node.get_text(' ', strip=True)
    return t if t else None


def _parse_eur(text: str | None) -> float | None:
    """'€500k' / '€2.50m' / '€1.20bn' / '-' → float EUR. Returns None
    if unparseable. TM uses lowercase k/m/bn suffixes."""
    if not text or not isinstance(text, str):
        return None
    t = text.strip().replace('\xa0', ' ')
    if t in ('-', '–', '?', ''):
        return None
    # Strip currency symbol + spaces
    t = re.sub(r'[€$£\s]', '', t)
    mult = 1.0
    if t.endswith('bn') or t.endswith('Bn'):
        mult = 1_000_000_000; t = t[:-2]
    elif t.endswith('m') or t.endswith('M'):
        mult = 1_000_000; t = t[:-1]
    elif t.endswith('k') or t.endswith('K'):
        mult = 1_000; t = t[:-1]
    t = t.replace(',', '.')
    try:
        return float(t) * mult
    except ValueError:
        return None


def _parse_tm_date(text: str | None) -> str | None:
    """TM dates: 'Mar 28, 2007' / '28/03/2007' / '2007-03-28' → ISO."""
    if not text or not isinstance(text, str):
        return None
    text = text.strip().rstrip('.,')
    # Try several formats
    for fmt in ('%b %d, %Y', '%B %d, %Y', '%d/%m/%Y', '%Y-%m-%d',
                  '%d.%m.%Y', '%d %b %Y', '%d %B %Y'):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def fetch_tm_player_profile(session: requests.Session,
                              tm_url: str) -> dict:
    """Pull rich profile metadata from /profil/spieler/{tm_id}.

    Returns dict with whichever fields could be parsed:
      full_name, dob, birthplace, citizenships (list of str),
      height_cm, foot, position_primary, position_secondary (list),
      current_club, joined_date, contract_until, agent,
      shirt_number, market_value_current_eur, on_loan_from
    """
    r = polite_get(session, tm_url)
    out: dict = {}
    if r is None or r.status_code != 200:
        return out
    soup = BeautifulSoup(r.text, 'html.parser')

    # h1 — full headline name (may contain shirt number prefix)
    h1 = soup.find('h1')
    if h1:
        shirt = h1.find('span', class_=re.compile(r'shirt'))
        if shirt:
            sh = _clean_text(shirt)
            if sh:
                out['shirt_number'] = re.sub(r'[^\d]', '', sh) or None
            shirt.extract()
        out['full_name'] = _clean_text(h1)

    # Modern TM (2024+) structure: an info-table with paired spans:
    #   <span class="info-table__content--regular">Label:</span>
    #   <span class="info-table__content--bold">Value</span>
    # Walking them in pairs is robust to label re-ordering.
    labels = soup.select('span.info-table__content--regular')
    values = soup.select('span.info-table__content--bold')
    for lab, val_node in zip(labels, values):
        label = (_clean_text(lab) or '').rstrip(':').lower().strip()
        val = _clean_text(val_node)
        if not val:
            continue
        if label in ('name in home country', 'full name'):
            # Prefer this over the headline h1 when present
            out['full_name'] = val
        elif label in ('date of birth/age', 'date of birth', 'born'):
            # "30/04/2007 (19)" or just "30/04/2007"
            m = re.match(r'(.+?)\s*\(\s*(\d+)\s*\)\s*$', val)
            if m:
                out['dob'] = _parse_tm_date(m.group(1))
                try: out['age'] = int(m.group(2))
                except ValueError: pass
            else:
                out['dob'] = _parse_tm_date(val)
        elif label == 'place of birth':
            out['birthplace'] = val
        elif label == 'height':
            m = re.search(r'(\d+[\.,]?\d*)\s*m', val)
            if m:
                try:
                    out['height_cm'] = int(round(
                        float(m.group(1).replace(',', '.')) * 100))
                except ValueError: pass
        elif label == 'citizenship':
            # TM joins with whitespace + flag emoji; split on multiple ws
            parts = [p.strip() for p in re.split(r'\s{2,}', val) if p.strip()]
            out['citizenships'] = parts if parts else [val]
        elif label == 'position':
            # "Attack - Right Winger" → primary "Right Winger" + group
            out['position_full'] = val
            if ' - ' in val:
                grp, role = val.split(' - ', 1)
                out['position_group'] = grp.strip()
                out['position_primary'] = role.strip()
            else:
                out['position_primary'] = val
        elif label == 'foot':
            out['foot'] = val.lower()
        elif label in ('player agent', 'agent'):
            out['agent'] = val
        elif label == 'outfitter':
            out['outfitter'] = val
        elif label == 'current club':
            out['current_club'] = val
        elif label == 'joined':
            out['joined_date'] = _parse_tm_date(val)
        elif label in ('contract expires', 'contract until'):
            out['contract_until'] = _parse_tm_date(val)
        elif label == 'last contract extension':
            out['last_contract_extension'] = _parse_tm_date(val)
        elif label == 'on loan from':
            out['on_loan_from'] = val
        elif label in ('contract option', 'option'):
            out['contract_option'] = val

    # Current market value — pulled directly from the header MV widget
    mv = soup.select_one('a.data-header__market-value-wrapper, '
                         'div.data-header__box--small a, '
                         '[class*="market-value"]')
    if mv:
        out['market_value_current_eur'] = _parse_eur(_clean_text(mv))

    return out


def fetch_tm_transfer_history(session: requests.Session,
                                tm_id: int) -> list[dict]:
    """Pull the player's full transfer history via TM's transfers
    JSON endpoint.

    Returns list of dicts (one per transfer event):
      {date, season, from_club, from_country, to_club, to_country,
       transfer_type, fee_eur, mv_at_transfer_eur, is_loan, source_url}
    """
    url = f"{TM_BASE}/ceapi/transferHistory/list/{tm_id}"
    r = polite_get(session, url,
                    headers={'X-Requested-With': 'XMLHttpRequest',
                              'Referer': f'{TM_BASE}/'})
    # a failed read is NOT 'no transfers': the caller must not record the
    # player as read
    if r is None or r.status_code != 200:
        raise FetchFailed(f"transfer history {tm_id}: "
                          f"{'no response' if r is None else 'HTTP ' + str(r.status_code)}")
    try:
        payload = r.json()
    except ValueError:
        raise FetchFailed(f"transfer history {tm_id}: not JSON")
    if not isinstance(payload, dict):
        return []
    rows = payload.get('transfers') or payload.get('list') or []
    if not isinstance(rows, list):
        return []
    out = []
    for row in rows:
        try:
            d = row.get('dateUnformatted') or row.get('date')
            iso = _parse_tm_date(d) if d else None
            old = row.get('from') or {}
            new = row.get('to') or {}
            fee_text = row.get('fee')
            mv_text = row.get('marketValue')
            out.append({
                'date': iso,
                'season': row.get('season'),
                'from_club': old.get('clubName') if isinstance(old, dict) else None,
                'from_country': old.get('countryName') if isinstance(old, dict) else None,
                'to_club': new.get('clubName') if isinstance(new, dict) else None,
                'to_country': new.get('countryName') if isinstance(new, dict) else None,
                'transfer_type': fee_text if isinstance(fee_text, str) and not _parse_eur(fee_text) else None,
                'fee_eur': _parse_eur(fee_text) if isinstance(fee_text, str) else None,
                'mv_at_transfer_eur': _parse_eur(mv_text) if isinstance(mv_text, str) else None,
                'is_loan': bool(row.get('isLoan')),
                'source_url': url,
            })
        except Exception:
            continue
    return out


METADATA_PATH = VAL_DIR / 'tm_player_metadata.parquet'
TRANSFERS_PATH = VAL_DIR / 'tm_transfer_history.parquet'


def append_metadata(rows: list[dict]) -> None:
    """One-row-per-player metadata. Last write wins per playerId."""
    if not rows:
        return
    new_df = pd.DataFrame(rows)
    if METADATA_PATH.exists():
        try:
            existing = pd.read_parquet(METADATA_PATH)
            combined = (pd.concat([existing, new_df], ignore_index=True)
                          .drop_duplicates(['playerId'], keep='last'))
        except Exception:
            combined = new_df
    else:
        combined = new_df
    # Cast list-of-strings columns to JSON strings for parquet portability
    for col in ('citizenships', 'position_secondary'):
        if col in combined.columns:
            combined[col] = combined[col].apply(
                lambda v: json.dumps(v) if isinstance(v, list) else v
            )
    combined.to_parquet(METADATA_PATH, index=False)
    LOG.info(f"Wrote {len(combined):,} rows to {METADATA_PATH}")


def append_transfers(rows: list[dict]) -> None:
    """Many-rows-per-player transfer history. Dedupe by
    (playerId, date, from_club, to_club)."""
    if not rows:
        return
    new_df = pd.DataFrame(rows)
    if TRANSFERS_PATH.exists():
        try:
            existing = pd.read_parquet(TRANSFERS_PATH)
            combined = (pd.concat([existing, new_df], ignore_index=True)
                          .drop_duplicates(
                              ['playerId', 'date', 'from_club', 'to_club'],
                              keep='last'))
        except Exception:
            combined = new_df
    else:
        combined = new_df
    combined.to_parquet(TRANSFERS_PATH, index=False)
    LOG.info(f"Wrote {len(combined):,} rows to {TRANSFERS_PATH}")


# ---------------------------------------------------------------------
# Verified links (transfermarkt_links.py) and wrong-profile clean-up
# ---------------------------------------------------------------------
def load_verified_links(path=None) -> dict:
    """playerId -> TMMatch for the links transfermarkt_links.py verified: same
    date of birth as exactly one man in the same club-season squad. These
    replace the name search, which put players on a namesake's profile."""
    path = Path(path) if path else PLAYER_LINKS_PATH
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    df = df[(df['status'] == 'ok') & df['tm_player_id'].notna()]
    out = {}
    for r in df.itertuples():
        tm_id = int(r.tm_player_id)
        slug = r.tm_player_slug if isinstance(r.tm_player_slug, str) and r.tm_player_slug else '-'
        out[int(r.playerId)] = TMMatch(
            tm_id=tm_id, tm_url=f"{TM_BASE}/{slug}/profil/spieler/{tm_id}",
            name=r.tm_name if isinstance(r.tm_name, str) else '',
            dob=r.tm_dob if isinstance(r.tm_dob, str) else None,
            club=None, score=float('nan'), fields_used=['squad', 'dob'])
    return out


def _row_tm_ids(df: pd.DataFrame) -> pd.Series:
    """Transfermarkt profile id each stored row was read from (valuations
    keep it inside `notes`)."""
    if 'tm_id' in df.columns:
        return pd.to_numeric(df['tm_id'], errors='coerce')
    return pd.to_numeric(df['notes'].astype(str).str.extract(r'tm_id=(\d+)')[0],
                         errors='coerce')


def wrong_profile_rows(df: pd.DataFrame, verified: dict, also=()) -> pd.Series:
    """True for rows read from a profile that is not the player's verified
    one, and for every row of the players in `also`."""
    want = df['playerId'].map({pid: m.tm_id for pid, m in verified.items()})
    have = _row_tm_ids(df)
    # a row with no profile id is not known to be wrong: keep it
    wrong = want.notna() & have.notna() & (have != want)
    if 'source' in df.columns:
        wrong &= df['source'] == 'transfermarkt'
    return wrong | df['playerId'].isin(list(also))


def purge_wrong_profiles(verified: dict, also=(), paths=None) -> dict:
    """Drop, from the three scraped files, rows that belong to another man's
    profile. Returns {file name: (rows dropped, players affected)}."""
    paths = paths or (VALUATIONS_PATH, METADATA_PATH, TRANSFERS_PATH)
    out = {}
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        wrong = wrong_profile_rows(df, verified, also)
        out[path.name] = (int(wrong.sum()), int(df.loc[wrong, 'playerId'].nunique()))
        if wrong.any():
            df[~wrong].to_parquet(path, index=False)
    return out


def players_to_relink(verified: dict) -> list:
    """Players the name search had on a namesake's profile (listed in
    tm_link_conflicts.csv) whose verified profile has not been read yet."""
    if not LINK_CONFLICTS_PATH.exists():
        return []
    pids = pd.read_csv(LINK_CONFLICTS_PATH)['playerId'].astype(int)
    have = {}
    if METADATA_PATH.exists():
        meta = pd.read_parquet(METADATA_PATH, columns=['playerId', 'tm_id'])
        have = dict(zip(meta['playerId'].astype(int), _row_tm_ids(meta)))
    return [p for p in pids if p in verified and have.get(p) != verified[p].tm_id]


def players_to_read_verified(verified: dict, stale_days: int = 120, today=None) -> list:
    """Verified players whose verified profile has not been read, or was last
    read more than `stale_days` ago (the club on a profile goes out of date)."""
    today = pd.Timestamp(today or date.today())
    read = {}
    if METADATA_PATH.exists():
        meta = pd.read_parquet(METADATA_PATH)
        when = pd.to_datetime(meta.get('fetched_at'), errors='coerce')
        read = {int(p): (t, w) for p, t, w in zip(meta['playerId'], _row_tm_ids(meta), when)}
    out = []
    for pid, m in verified.items():
        tm_id, when = read.get(pid, (None, pd.NaT))
        if tm_id != m.tm_id or pd.isna(when) or (today - when).days > stale_days:
            out.append(pid)
    return sorted(out)


def write_match_log(audit: list) -> None:
    """Players this run did not look at (recently read, or outside a
    partial run) keep their log rows."""
    new = pd.DataFrame(audit)
    if MATCH_LOG_PATH.exists():
        old = pd.read_csv(MATCH_LOG_PATH)
        new = pd.concat([old[~old['playerId'].isin(new['playerId'])], new],
                        ignore_index=True)
    new.to_csv(MATCH_LOG_PATH, index=False)


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
        max_age_days: int = 14, only=None) -> None:
    """`only`: playerIds to read now, whatever their last read date."""
    universe = load_player_universe()
    dob_map = load_player_dob_map()
    verified = load_verified_links()
    if player_id:
        universe = universe[universe['playerId'] == player_id]
    if only is not None:
        only = {int(p) for p in only}
        universe = universe[universe['playerId'].isin(only)]
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
    fresh_rows: list[dict] = []      # MV history (valuations.parquet)
    fresh_metadata: list[dict] = []  # Player profiles (tm_player_metadata.parquet)
    fresh_transfers: list[dict] = [] # Transfer history (tm_transfer_history.parquet)
    match_audit: list[dict] = []
    today = date.today()

    # Minimum score threshold to accept a match. v2 — dropped from
    # 120 to 100 because the new "best-of-career-clubs" matcher
    # supplies the secondary signal more reliably than the old
    # most-recent-team-only check. A score of 100 = perfect name match
    # alone (Jaccard=1.0 → 100); we still require name to be at least
    # close, but historical-team matches now compensate for cases
    # where TM's current-club listing diverges from our last team.
    MIN_MATCH_SCORE = 100.0

    def save() -> None:
        # Metadata last: --relink / --verified take a player's metadata row as
        # proof that his values and transfers were written too.
        append_rows(fresh_rows)
        append_transfers(fresh_transfers)
        append_metadata(fresh_metadata)
        if match_audit:
            write_match_log(match_audit)
        for rows in (fresh_rows, fresh_transfers, fresh_metadata, match_audit):
            rows.clear()

    counts = {'matched': 0, 'low_confidence': 0, 'no_match': 0, 'dob_mismatch': 0, 'fetch_failed': 0}
    failed_in_a_row = 0
    from tqdm import tqdm
    for n_seen, (_, row) in enumerate(tqdm(list(universe.iterrows()), total=len(universe),
                                           desc="TM scrape")):
        # a long run keeps what it has read: saved every SAVE_EVERY players
        if n_seen and n_seen % SAVE_EVERY == 0:
            for r in match_audit:
                counts[{'verified': 'matched'}.get(r['status'], r['status'])] += 1
            save()
        pid = int(row['playerId'])
        # Skip if scraped recently
        last = newest.get(pid)
        if (only is None and pd.notna(last)
                and (today - last.date()).days < max_age_days):
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

        # Expected clubs = FULL career-clubs list. The matcher will
        # try each one and keep the best-scoring overlap. This is
        # critical for retired/loaned players where TM's listed
        # "current club" might be a Spanish or Brazilian side they
        # moved to, while their Portuguese-tier career club is
        # buried deeper in their TM transfer history.
        career = row.get('career_clubs')
        if isinstance(career, list) and career:
            expected_clubs = [c for c in career if c]
        elif pd.notna(row.get('teamName')):
            expected_clubs = [row['teamName']]
        else:
            expected_clubs = []

        linked = verified.get(pid)
        try:
            candidates = [linked] if linked else search_transfermarkt(
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
        if not linked and best.score < MIN_MATCH_SCORE:
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

        # Compact (season → team) summary for the audit log — easier
        # to spot-check than the full career_with_seasons list.
        cws = row.get('career_with_seasons')
        season_summary = ''
        if isinstance(cws, list) and cws:
            try:
                # Group teams per season label, dedupe
                from collections import defaultdict
                by_ss = defaultdict(set)
                for r in cws:
                    lbl = r.get('season_label') or str(r.get('seasonId'))
                    if r.get('team'):
                        by_ss[lbl].add(r.get('team'))
                season_summary = '; '.join(
                    f"{lbl}: {'/'.join(sorted(ts))}"
                    for lbl, ts in sorted(by_ss.items())
                )
            except Exception:
                season_summary = ''
        match_audit.append({
            'playerId': pid, 'name': row['playerName'],
            'status': 'verified' if linked else 'matched', 'tm_id': best.tm_id,
            'tm_name': best.name,
            'score': '' if linked else round(best.score, 1),
            'fields': ','.join(best.fields_used),
            'tm_url': best.tm_url,
            'expected_club': (expected_clubs[0] if expected_clubs else ''),
            'tm_club': best.club or '',
            'our_season_clubs': season_summary,
        })

        # ---- Rich-metadata pull ----
        # For every accepted match, fetch full player profile + transfer
        # history. Stored separately from valuations.parquet but useful
        # for v2 EUR regression features and post-match verification.
        try:
            profile = fetch_tm_player_profile(session, best.tm_url)
            if profile:
                profile['playerId'] = pid
                profile['tm_id'] = best.tm_id
                profile['tm_url'] = best.tm_url
                profile['fetched_at'] = datetime.utcnow().isoformat(timespec='seconds')
                fresh_metadata.append(profile)
        except Exception as e:
            profile = None
            LOG.warning(f"Profile fetch failed for {pid}: {e}")
        # A name-search link whose profile has another date of birth is a
        # namesake: write nothing for it.
        tm_dob = str((profile or {}).get('dob') or '')[:10]
        if not linked and dob_str and tm_dob and tm_dob != dob_str[:10]:
            match_audit[-1]['status'] = 'dob_mismatch'
            if profile in fresh_metadata:
                fresh_metadata.remove(profile)
            continue

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

        try:
            xfers = fetch_tm_transfer_history(session, best.tm_id)
            for x in xfers:
                x['playerId'] = pid
                x['tm_id'] = best.tm_id
                fresh_transfers.append(x)
            failed_in_a_row = 0
        except Exception as e:
            # not read: drop his profile row too, so the next run reads him again
            LOG.warning(f"Transfer-history fetch failed for {pid}: {e}")
            fresh_metadata[:] = [m for m in fresh_metadata if m.get('playerId') != pid]
            fresh_rows[:] = [r for r in fresh_rows if r.get('playerId') != pid]
            match_audit[-1]['status'] = 'fetch_failed'
            failed_in_a_row += 1
            if failed_in_a_row >= MAX_FAILS_IN_A_ROW:
                LOG.error(f"{failed_in_a_row} players in a row could not be read: Transfermarkt is "
                          f"refusing. Stopping; what was read is saved. Run again later.")
                break

    for r in match_audit:
        counts[{'verified': 'matched'}.get(r['status'], r['status'])] += 1
    save()
    LOG.info(f"Done: {counts['matched']} read, {counts['low_confidence']} low-confidence, "
             f"{counts['no_match']} no-match, {counts['dob_mismatch']} refused on date of birth, "
             f"{counts['fetch_failed']} not read (run again)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--player-id', type=int, default=None)
    ap.add_argument('--max-age-days', type=int, default=14)
    ap.add_argument('--relink', action='store_true',
                    help='drop rows read from a profile that is not the verified one, '
                         'then read the verified profile of the players in '
                         'tm_link_conflicts.csv')
    ap.add_argument('--verified', action='store_true',
                    help='read every verified player whose profile has not been read or '
                         'was read more than 120 days ago; safe to stop and run again')
    ap.add_argument('--verbose', '-v', action='store_true')
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s  %(levelname)-7s  %(message)s',
        datefmt='%H:%M:%S',
    )
    only = None
    if args.relink:
        verified = load_verified_links()
        for name, (rows, players) in purge_wrong_profiles(verified).items():
            LOG.info(f"{name}: dropped {rows:,} rows of {players} players (wrong profile)")
        only = players_to_relink(verified)
        LOG.info(f"{len(only)} players to read from their verified profile")
        if not only:
            return 0
    elif args.verified:
        only = players_to_read_verified(load_verified_links())
        LOG.info(f"{len(only)} verified players to read")
        if not only:
            return 0
    run(limit=args.limit, player_id=args.player_id,
        max_age_days=args.max_age_days, only=only)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
