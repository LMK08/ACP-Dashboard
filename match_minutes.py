"""
Exact per-match player minutes from the Wyscout API — the ONE minutes source.

RULE (Lucas, 2026-10-20): minutes are never estimated. No event-timestamp
spans, no lineup/substitution arithmetic, no hand-typed event-derived
overrides. Every minute the Dashboard shows comes from Wyscout's own
`minutesOnField`. A player the API has no minutes for has NO minutes row
(missing, not guessed).

Source: GET /v3/matches/{matchId}/advancedstats/players — one record per
player in the match squad with `total.minutesOnField` and `positions`
(position code + percent of his time there). Summed per season this equals
the season endpoint (/players/{pid}/advancedstats) exactly (checked on
Calegari 357' and Nicolás Souza 149', 2026-10-20), and unlike the season
endpoint it never 404s for players Wyscout later merged or moved — the
season endpoint 404'd for ~1,600 player-seasons, which is why the old
pipeline fell back to estimates.

Cache: match_player_minutes.parquet (committed; ships to HF with the other
root parquets). One row per (matchId, playerId) with minutes > 0. Matches
are fetched once; matches of the CURRENT seasons played in the last
REFRESH_DAYS are re-fetched each run because Wyscout revises recent ones.

    python match_minutes.py            # incremental (what process_data runs)
    python match_minutes.py --season 189147
"""
import argparse
import datetime as dt
import os
import time

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from requests.auth import HTTPBasicAuth
from urllib3.util.retry import Retry

from league_config import (competition_for_season, current_season_ids,
                           get_credentials)

CACHE_PATH = 'match_player_minutes.parquet'
BASE_URL = 'https://apirest.wyscout.com/v3'
REFRESH_DAYS = 14
COLUMNS = ['matchId', 'seasonId', 'competitionId', 'playerId', 'minutes',
           'started', 'positions']


def _session():
    s = requests.Session()
    s.mount('https://', HTTPAdapter(max_retries=Retry(
        total=5, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])))
    return s


def _positions_str(positions):
    """[{'position': {'code': 'cf'}, 'percent': 83}, ...] -> 'cf:83,lw:16'."""
    out = []
    for p in positions or []:
        code = ((p.get('position') or {}).get('code') or '').strip()
        if code:
            out.append(f"{code}:{int(p.get('percent') or 0)}")
    return ','.join(out)


def fetch_match(session, auth, match_id):
    """Rows for one match (players with minutesOnField > 0)."""
    # Wyscout answers a transient 400 now and then (match 5871525 400'd once
    # and returned 200 on the next try), which the urllib3 Retry won't cover.
    for attempt in range(4):
        r = session.get(f'{BASE_URL}/matches/{int(match_id)}/advancedstats/players',
                        auth=auth, timeout=30)
        if r.status_code != 400 or ('No statistical data available' in r.text
                                    or '"code":403' in r.text):
            break  # success, or a definitive answer (handled below)
        time.sleep(2 * (attempt + 1))
    if r.status_code == 400 and ('No statistical data available' in r.text
                                 or '"code":403' in r.text):
        # Wyscout has no player stats for this match (some 2021-23 matches),
        # or our accounts may not read them (403 on ~5% of Camp 23/24 for
        # BOTH club logins). Record a sentinel row (playerId 0, 0') so the
        # match counts as fetched and is not retried every run;
        # load_cache() drops it. Such a match has no minutes, so the engine
        # and the stats leave its events out too.
        return [{'matchId': int(match_id), 'seasonId': 0, 'competitionId': 0,
                 'playerId': 0, 'minutes': 0, 'started': False, 'positions': ''}]
    r.raise_for_status()
    rows = []
    for pl in r.json().get('players', []):
        tot = pl.get('total') or {}
        mins = int(tot.get('minutesOnField') or 0)
        if mins <= 0:
            continue
        rows.append({
            'matchId': int(match_id),
            'seasonId': int(pl.get('seasonId') or 0),
            'competitionId': int(pl.get('competitionId') or 0),
            'playerId': int(pl['playerId']),
            'minutes': mins,
            'started': bool(tot.get('matchesInStart')),
            'positions': _positions_str(pl.get('positions')),
        })
    return rows


def load_cache(path=CACHE_PATH, keep_sentinels=False):
    """The cache; sentinel rows (matches Wyscout has no stats for) are
    dropped unless keep_sentinels (update_cache needs them)."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_parquet(path)
    return df if keep_sentinels else df[df['playerId'] != 0].reset_index(drop=True)


PER_MATCH_MIN_COVERAGE = 0.5


def per_match_seasons(path=CACHE_PATH, matches_summary=None):
    """Seasons that run on per-match minutes: at least PER_MATCH_MIN_COVERAGE
    of their played matches have per-match rows (all Liga 3). The Campeonato
    has none — the club's Wyscout accounts lost all Campeonato access in
    2026-10 — so those seasons run on the exact SEASON totals stored earlier
    (player_minutes_and_positions.pkl); see precompute_minutes.py. A coverage
    threshold (not 'any row') keeps one stray match from flipping a season."""
    df = load_cache(path)
    if not len(df):
        return set()
    if matches_summary is None:
        matches_summary = pd.read_parquet('matches_summary.parquet', columns=['matchId', 'seasonId'])
    ms = matches_summary.dropna(subset=['matchId', 'seasonId'])
    played = ms.groupby(ms['seasonId'].astype('int64'))['matchId'].nunique()
    covered = df.groupby(df['seasonId'].astype('int64'))['matchId'].nunique()
    return {int(s) for s, n in covered.items() if n >= PER_MATCH_MIN_COVERAGE * played.get(s, n)}


def excluded_match_ids(matches_summary, path=CACHE_PATH):
    """Matches to leave out of per-90 work: matches of a per-match season
    with no per-match minutes — Wyscout said none (sentinel) OR the fetch has
    not succeeded yet. Seasons on stored season totals are kept."""
    pms = per_match_seasons(path, matches_summary)
    ms = matches_summary.dropna(subset=['matchId', 'seasonId'])
    have = set(load_cache(path)['matchId'].astype('int64'))
    return {int(m) for m, s in zip(ms['matchId'], ms['seasonId'])
            if int(s) in pms and int(m) not in have}


def matches_without_stats(path=CACHE_PATH):
    """matchIds Wyscout returned 'No statistical data available' for."""
    df = load_cache(path, keep_sentinels=True)
    return set(df.loc[df['playerId'] == 0, 'matchId'].astype('int64'))


def _save(df, path):
    df = df.astype({'matchId': 'int64', 'seasonId': 'int64',
                    'competitionId': 'int64', 'playerId': 'int64',
                    'minutes': 'int32', 'started': 'bool'})
    df = df.sort_values(['seasonId', 'matchId', 'playerId']).reset_index(drop=True)
    tmp = path + '.tmp'
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def update_cache(matches_summary, season_ids=None, path=CACHE_PATH,
                 refresh_days=REFRESH_DAYS, save_every=100):
    """Fetch every played match of `season_ids` (default: all in the summary)
    that is not cached yet, plus recent current-season matches. Returns the
    full cache. Saves every `save_every` matches so an interrupted backfill
    resumes where it stopped."""
    cache = load_cache(path, keep_sentinels=True)
    ms = matches_summary.dropna(subset=['matchId', 'seasonId']).copy()
    if 'status' in ms.columns:
        ms = ms[ms['status'].astype(str).str.lower() == 'played']
    if season_ids is not None:
        ms = ms[ms['seasonId'].isin(list(season_ids))]
    cached = set(cache['matchId'].astype('int64')) if len(cache) else set()

    recent = pd.Series(False, index=ms.index)
    if refresh_days and 'dateutc' in ms.columns:
        when = pd.to_datetime(ms['dateutc'], errors='coerce', utc=True)
        cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=refresh_days)
        recent = (when >= cutoff) & ms['seasonId'].isin(list(current_season_ids().values()))
    todo = ms[~ms['matchId'].astype('int64').isin(cached) | recent]
    if todo.empty:
        print('match minutes: cache up to date')
        return cache

    print(f'match minutes: fetching {len(todo)} matches '
          f'({int(recent.sum())} recent re-fetches)')
    session = _session()
    new_rows, done_ids, failed = [], [], []
    t0 = time.time()
    for i, (sid, mid) in enumerate(zip(todo['seasonId'], todo['matchId']), 1):
        comp = competition_for_season(int(sid))
        user, pwd = get_credentials(comp) if comp is not None else ('', '')
        if not user:
            failed.append(int(mid))
            continue
        try:
            new_rows.extend(fetch_match(session, HTTPBasicAuth(user, pwd), mid))
            done_ids.append(int(mid))
        except Exception as e:  # keep going; the match is retried next run
            print(f'  match {int(mid)}: {e}')
            failed.append(int(mid))
        if i % save_every == 0 or i == len(todo):
            if done_ids:
                keep = cache[~cache['matchId'].astype('int64').isin(done_ids)]
                cache = pd.concat([keep, pd.DataFrame(new_rows, columns=COLUMNS)],
                                  ignore_index=True)
                _save(cache, path)
                cache = load_cache(path, keep_sentinels=True)
                new_rows, done_ids = [], []
            print(f'  {i}/{len(todo)} matches, {time.time() - t0:.0f}s')
    if failed:
        print(f'match minutes: {len(failed)} matches failed (retried next run): '
              f'{failed[:10]}')
    return cache


if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument('--season', type=int, action='append',
                    help='limit to these season ids (repeatable)')
    args = ap.parse_args()
    update_cache(pd.read_parquet('matches_summary.parquet'), season_ids=args.season)
