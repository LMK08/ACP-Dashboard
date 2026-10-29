#!/usr/bin/env python3
"""
Refresh the Wyscout transfer records behind the engine value's join/leave
windows (models/value/availability.py) into models/value/transfers.parquet.

Per TEAM, not per season: /seasons/{id}/transfers is 403 on our subscription
(probed 2026-10-28) while /teams/{id}/transfers works for any date range of at
most one year. Every team of the refreshed seasons (team ids from
raw_events.parquet) is pulled for [season start - 60 days, min(today, season
end + 30 days)], split into <= 1-year ranges. A team whose call fails falls
back to /players/{id}/transfers for that team's players in the season. NOTE:
Wyscout returns a refusal as HTTP 400 with {"error": {"code": 403}} in the
body — the body decides, not the status.

Rows are upserted on (playerId, startDate, fromTeamId, toTeamId, type) and
never deleted. Only ACTIVE competitions are refreshed (league_config): the
Campeonato is no longer fetched, its seasons rely on the seed.

  python models/value/fetch_transfers.py              # refresh (CI: scheduled_update.yml)
  python models/value/fetch_transfers.py --team 9706  # one team, for a check
  python models/value/fetch_transfers.py --seed DIR   # one-off: wyscout_loader/data/player_transfers
"""
import argparse
import glob
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DASH = HERE.parent.parent
sys.path.insert(0, str(DASH))
from league_config import COMPETITIONS, get_credentials, is_active  # noqa: E402

OUT = HERE / 'transfers.parquet'
BASE_URL = 'https://apirest.wyscout.com/v3'
KEY = ['playerId', 'startDate', 'fromTeamId', 'toTeamId', 'type']
COLS = KEY + ['fromTeamName', 'toTeamName', 'endDate', 'active', 'announceDate', 'source', 'fetched_at']
SEED_FROM = '2021-06-01'        # the first season in the data starts 2021/22
PAD_BEFORE, PAD_AFTER = 60, 30  # days around a season's matches
MAX_RANGE = 364                 # the team endpoint refuses ranges over one year


def _normalise(rows, source):
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=COLS)
    for c in ('playerId', 'fromTeamId', 'toTeamId'):
        df[c] = pd.to_numeric(df.get(c), errors='coerce').astype('Int64')
    for c in ('fromTeamName', 'toTeamName', 'endDate', 'active', 'announceDate', 'type', 'startDate'):
        if c not in df:
            df[c] = None
    df['type'] = df['type'].fillna('').astype(str)
    # an undated move can't cut a window ('0000-00-00', None): dropped
    sd = pd.to_datetime(df['startDate'], errors='coerce')
    df = df[sd.notna()].copy()
    df['startDate'] = sd[sd.notna()].dt.strftime('%Y-%m-%d')
    df['source'] = source
    df['fetched_at'] = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    df = df[df['playerId'].notna()]
    return df[COLS]


def upsert(old, new):
    """New rows replace old ones with the same key; nothing is deleted."""
    if old is None or old.empty:
        out = new
    else:
        out = pd.concat([old, new], ignore_index=True)
    out = out.drop_duplicates(KEY, keep='last')
    return out.sort_values(['playerId', 'startDate']).reset_index(drop=True)


def load():
    return pd.read_parquet(OUT) if OUT.exists() else pd.DataFrame(columns=COLS)


def seed(cache_dir):
    files = glob.glob(os.path.join(cache_dir, 'playerId=*', 'player_transfers.parquet'))
    parts = [pd.read_parquet(f) for f in files]
    raw = pd.concat([p for p in parts if len(p)], ignore_index=True)
    raw = raw[pd.to_datetime(raw['startDate'], errors='coerce') >= SEED_FROM]
    df = upsert(load(), _normalise(raw.to_dict('records'), 'loader_cache'))
    df.to_parquet(OUT, index=False)
    print(f'seeded {len(raw)} records ({len(files)} player files) -> {len(df)} rows in {OUT.name}')


def _get(session, auth, path, params=None):
    """(payload, None) or (None, reason). A 403 can arrive inside an HTTP 400."""
    r = session.get(BASE_URL + path, params=params or {}, auth=auth, timeout=30)
    try:
        body = r.json()
    except ValueError:
        return None, f'HTTP {r.status_code} (no JSON)'
    if isinstance(body, dict) and 'error' in body:
        e = body['error'] or {}
        return None, f"{e.get('code', r.status_code)} {e.get('message', '')}".strip()
    if r.status_code != 200:
        return None, f'HTTP {r.status_code}'
    return body, None


def _rows(body):
    """The transfer list: {'transfer': [...]} normally, a bare list when empty."""
    if isinstance(body, dict):
        return body.get('transfer') or []
    return body if isinstance(body, list) else []


def _ranges(start, end):
    while start <= end:
        stop = min(end, start + timedelta(days=MAX_RANGE))
        yield start, stop
        start = stop + timedelta(days=1)


def refresh(only_team=None, sleep=0.4):
    import requests
    from requests.auth import HTTPBasicAuth
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    try:
        from dotenv import load_dotenv
        load_dotenv(DASH / '.env')
    except ImportError:
        pass
    ms = pd.read_parquet(DASH / 'matches_summary.parquet', columns=['seasonId', 'dateutc'])
    ms['d'] = pd.to_datetime(ms['dateutc'], errors='coerce')
    span = ms.groupby('seasonId')['d'].agg(['min', 'max'])
    ev = pd.read_parquet(DASH / 'raw_events.parquet', columns=['seasonId', 'team.id', 'player.id']).dropna()
    ev = ev.astype({'seasonId': int, 'team.id': int, 'player.id': int}).drop_duplicates()

    session = requests.Session()
    session.mount('https://', HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1,
                                                           status_forcelist=[429, 500, 502, 503, 504])))
    today = date.today()
    new, stats = [], {'team_calls': 0, 'team_failed': 0, 'player_calls': 0, 'player_failed': 0}
    for cid, comp in COMPETITIONS.items():
        if not is_active(cid):
            continue
        user, pwd = get_credentials(cid)
        if not user or not pwd:
            print(f"no credentials for {comp['name']} — skipped")
            continue
        auth = HTTPBasicAuth(user, pwd)
        seasons = sorted(comp['seasons'])[-2:]          # the season in progress + the one before it
        for sid in seasons:
            if sid not in span.index or pd.isna(span.loc[sid, 'min']):
                continue
            lo = (span.loc[sid, 'min'] - timedelta(days=PAD_BEFORE)).date()
            hi = min(today, (span.loc[sid, 'max'] + timedelta(days=PAD_AFTER)).date())
            teams = sorted(ev.loc[ev['seasonId'] == sid, 'team.id'].unique())
            if only_team is not None:
                teams = [t for t in teams if t == only_team]
            for tid in teams:
                failed = None
                for a, b in _ranges(lo, hi):
                    stats['team_calls'] += 1
                    body, err = _get(session, auth, f'/teams/{tid}/transfers',
                                     {'fromDate': a.isoformat(), 'toDate': b.isoformat()})
                    time.sleep(sleep)
                    if err:
                        failed = err
                        break
                    new.extend(_rows(body))
                if failed is None:
                    continue
                stats['team_failed'] += 1
                players = sorted(ev.loc[(ev['seasonId'] == sid) & (ev['team.id'] == tid), 'player.id'].unique())
                print(f'  team {tid} ({sid}): {failed} — falling back to {len(players)} player calls')
                for pid in players:
                    stats['player_calls'] += 1
                    body, err = _get(session, auth, f'/players/{pid}/transfers')
                    time.sleep(sleep)
                    if err:
                        stats['player_failed'] += 1
                        continue
                    new.extend(_rows(body))
    before = load()
    df = upsert(before, _normalise(new, 'api'))
    df.to_parquet(OUT, index=False)
    print(f"transfers: {len(new)} fetched, {len(df) - len(before)} new rows, {len(df)} total | {stats}")
    if stats['team_calls'] and stats['team_failed'] == stats['team_calls'] and not new:
        sys.exit('every team call failed — check the credentials / subscription')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--seed', metavar='DIR', help='seed from the loader cache (player_transfers folder)')
    ap.add_argument('--team', type=int, help='refresh one team id only (a check)')
    a = ap.parse_args()
    if a.seed:
        seed(a.seed)
    else:
        refresh(only_team=a.team)
