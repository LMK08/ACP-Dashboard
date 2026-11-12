#!/usr/bin/env python3
"""Link our teams and coaches to their Transfermarkt profiles.

    python transfermarkt_links.py teams      [--limit N] [--team "AD Fafe"]
    python transfermarkt_links.py managers   [--limit N]
    python transfermarkt_links.py verify
    python transfermarkt_links.py coaches

  teams     our teamId → Transfermarkt club id, by searching the club name.
            Writes valuations/tm_team_links.csv. A row whose status you set
            to 'manual' (with the right tm_team_id) is never overwritten.
  managers  for every linked club, its manager history (name, date of
            birth, appointed, in charge until, matches).
            Writes valuations/tm_team_managers.parquet.
  verify    checks every club link against the coaches the event provider
            did name, and sends doubtful links to 'review'. No request.
  coaches   our coachId → Transfermarkt coach id, by matching the dates of
            the matches he took at each club against those manager stints.
            Writes valuations/tm_coach_links.csv. No request is made.

(Players are linked by scrape_transfermarkt.py → valuations/tm_match_log.csv.)

Rules this script keeps:
  * It identifies itself with a plain client name and sends no browser
    headers; Transfermarkt's robots.txt allows the pages read here.
  * One request every ~2 seconds, a cap per run, and it stops the whole run
    at the first refusal (403 / 429 / challenge page) instead of retrying.
  * Nothing is guessed: a link is 'ok' only above the stated thresholds,
    else 'review' with the candidates' numbers so a person can decide.
"""
import argparse
import random
import re
import sys
import time
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
VAL = HERE / 'valuations'
TEAM_LINKS = VAL / 'tm_team_links.csv'
MANAGERS = VAL / 'tm_team_managers.parquet'
COACH_LINKS = VAL / 'tm_coach_links.csv'
TEAM_MATCH = HERE / 'models' / 'coaches' / 'coach_team_match.parquet'

TM_BASE = 'https://www.transfermarkt.com'
CLIENT = 'club-data-research/1.0'
PAUSE = (1.8, 2.6)          # seconds between requests
MAX_REQUESTS = 400          # per run

SENIOR_LEAGUES = ('Liga Portugal', 'Liga 3', 'Campeonato de Portugal')     # incl. '… - Série B'


def senior(c):
    return str(c.get('league') or '').startswith(SENIOR_LEAGUES)
TEAM_OK_SCORE = 0.80        # name similarity for an automatic club link
TEAM_OK_MARGIN = 0.08       # ... and this far ahead of the next candidate
VERIFY_SHARE = 0.60         # a non-exact club link must agree with named coaches on this share …
VERIFY_MATCHES = 10         # … of at least this many matches
_STOP = {'de', 'da', 'do', 'dos', 'das', 'e', 'jr', 'junior'}
COACH_OK_SHARE = 0.70       # share of a coach's dated matches inside the stints of ONE Transfermarkt coach
COACH_OK_MATCHES = 8        # ... over at least this many matches
STINT_SLACK_DAYS = 3        # an appointment is often dated a few days after the first match


class Refused(Exception):
    pass


class Client:
    def __init__(self):
        import requests
        self.s = requests.Session()
        self.s.headers.update({'User-Agent': CLIENT, 'Accept-Language': 'en'})
        self.n = 0

    def get(self, path):
        if self.n >= MAX_REQUESTS:
            raise Refused(f'request cap of {MAX_REQUESTS} reached for this run')
        time.sleep(random.uniform(*PAUSE))
        self.n += 1
        import requests
        try:
            r = self.s.get(TM_BASE + path, timeout=30)
        except requests.RequestException:
            time.sleep(20)                                    # one retry for a dropped connection, never a loop
            try:
                r = self.s.get(TM_BASE + path, timeout=30)
            except requests.RequestException as exc:
                raise Refused(f'network error on {path}: {type(exc).__name__}')
        if r.status_code == 404:
            return None
        # a challenge or block page can come back as 200/202: anything that is
        # not Transfermarkt's own page stops the run
        if r.status_code != 200 or 'Just a moment' in r.text[:2000] or 'transfermarkt' not in r.text.lower():
            raise Refused(f'Transfermarkt refused {path} (HTTP {r.status_code})')
        return r.text


# ───────────────────────────── names ─────────────────────────────
_DROP = {'fc', 'cf', 'sc', 'cd', 'ad', 'gd', 'ud', 'sl', 'cs', 'ac', 'ca', 'clube', 'futebol', 'de', 'do', 'da',
         'dos', 'das', 'sad', 'desportivo', 'sport', 'sporting', 'uniao', 'associacao', 'grupo', 'club'}


_YOUTH = re.compile(r' (u ?\d\d|y ?\d\d|sub ?\d\d|under ?\d\d|youth ?\d*|juniores|juvenis|sf) ')


def fold(s):
    s = unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode().lower()
    return re.sub(r'[^a-z0-9 ]+', ' ', s).strip()


def name_key(s, drop=None):
    """Club name without the generic words, with reserve sides marked ' b'."""
    drop = _DROP if drop is None else drop
    f = ' ' + fold(s) + ' '
    reserve = bool(re.search(r' (ii|b|sub 23|u23|under 23) ', f))
    f = re.sub(r' (ii|b|sub 23|u23|under 23) ', ' ', f)
    words = [w for w in f.split() if w not in drop] or f.split()
    return ' '.join(words) + (' b' if reserve else '')


# words that carry no identity at all; 'União', 'Sporting', 'Desportivo' DO
# (União de Coimbra is not Académica Coimbra, Sporting de Tomar is not União de Tomar)
_ABBR = {'fc', 'cf', 'sc', 'cd', 'ad', 'gd', 'ud', 'sl', 'cs', 'ac', 'ca', 'clube', 'futebol', 'de', 'do', 'da',
         'dos', 'das', 'sad', 'club', 'os', 'o'}


def similarity(a, b):
    """1.0 = the same name once abbreviations are dropped. 0.95 = the same
    once 'União' / 'Sporting' / 'Desportivo' are dropped too (our 'União de
    Leiria', their 'UD Leiria'): likely, but it needs the data check in
    verify_team_links. 0.90 = one name contained in the other."""
    ka, kb = name_key(a), name_key(b)
    if (ka.endswith(' b')) != (kb.endswith(' b')):          # a first team never matches a reserve side
        return 0.0
    sa, sb = name_key(a, _ABBR), name_key(b, _ABBR)
    if sa == sb:
        return 1.0
    score = min(max(SequenceMatcher(None, sa, sb).ratio(), SequenceMatcher(None, fold(a), fold(b)).ratio()), 0.89)
    youth = bool(_YOUTH.search(' ' + fold(b) + ' '))
    if ka == kb and not youth:
        return 0.95
    wa, wb = set(sa.split()), set(sb.split())
    if wa and wb and (wa <= wb or wb <= wa) and not youth:
        score = 0.90
    return score


# ───────────────────────────── parsing ─────────────────────────────
def parse_club_search(html):
    """[{tm_team_id, tm_slug, tm_name, country}] from a quick-search page."""
    from bs4 import BeautifulSoup
    out = []
    for box in BeautifulSoup(html, 'html.parser').select('div.box'):
        h = box.select_one('h2')
        if not h or 'Clubs' not in h.get_text():
            continue
        for tr in box.select('table.items > tbody > tr'):
            a = tr.select_one('a[href*="/startseite/verein/"]')
            if a is None or not a.get_text(strip=True):
                a = next((x for x in tr.select('a[href*="/startseite/verein/"]') if x.get_text(strip=True)), None)
            if a is None:
                continue
            m = re.match(r'/([^/]+)/startseite/verein/(\d+)', a['href'])
            if not m:
                continue
            flag = tr.select_one('img.flaggenrahmen')
            lg = tr.select_one('a[href*="/startseite/wettbewerb/"]')
            out.append({'tm_team_id': int(m.group(2)), 'tm_slug': m.group(1), 'tm_name': a.get_text(strip=True),
                        'country': flag.get('title') if flag else None,
                        'league': lg.get_text(strip=True) if lg else None})
    return out


def _date(text):
    m = re.search(r'(\d{2})/(\d{2})/(\d{4})', text or '')
    return pd.Timestamp(int(m.group(3)), int(m.group(2)), int(m.group(1))) if m else pd.NaT


def parse_manager_history(html):
    """[{tm_coach_id, tm_coach_slug, name, dob, start, end, matches}] from a club's staff-history page
    (its default view is the managers). `end` is NaT for the man in charge."""
    from bs4 import BeautifulSoup
    out = []
    for tr in BeautifulSoup(html, 'html.parser').select('table.items > tbody > tr'):
        a = next((x for x in tr.select('a[href*="/profil/trainer/"]') if x.get_text(strip=True)), None)
        if a is None:
            continue
        m = re.match(r'/([^/]+)/profil/trainer/(\d+)', a['href'])
        tds = [td.get_text(' ', strip=True) for td in tr.find_all('td', recursive=False)]
        if not m or len(tds) < 6:
            continue
        games = pd.to_numeric(tds[5], errors='coerce')
        out.append({'tm_coach_id': int(m.group(2)), 'tm_coach_slug': m.group(1), 'name': a.get_text(strip=True),
                    'dob': _date(tds[0]), 'start': _date(tds[2]), 'end': _date(tds[3]),
                    'matches': int(games) if pd.notna(games) else None})
    return [r for r in out if pd.notna(r['start'])]


# ───────────────────────────── steps ─────────────────────────────
def our_teams():
    tm = pd.read_parquet(TEAM_MATCH, columns=['teamId', 'team', 'date'])
    g = tm.groupby('teamId').agg(team=('team', 'last'), matches=('date', 'size')).reset_index()
    return g.sort_values('matches', ascending=False)


def search_queries(team):
    """The name as we have it, then (only if that finds no sure match) a
    reserve side written the Transfermarkt way and the most distinctive word."""
    qs = [team]
    alt = re.sub(r'\bII\b', 'B', team)
    if alt != team:
        qs.append(alt)
        qs.append(' '.join(w for w in alt.split() if fold(w) not in _DROP or w == 'B'))
    words = [w for w in team.split() if fold(w) not in _DROP and len(w) >= 5 and w != 'II']
    if words and max(words, key=len) != team:
        qs.append(max(words, key=len))
    return list(dict.fromkeys(q for q in qs if q.strip()))[:3]


def pick_club(team, candidates):
    """(row or None, status, note). Portuguese clubs only."""
    c = [dict(x, score=similarity(team, x['tm_name'])) for x in candidates if x.get('country') == 'Portugal']
    # equal names (three clubs called Sanjoanense): the one playing in a
    # national senior league comes first
    c.sort(key=lambda x: (-x['score'], 0 if senior(x) else 1))
    if not c:
        return None, 'not_found', ''
    note = '; '.join(f"{x['tm_name']} {x['score']:.2f} ({x.get('league') or 'no league'})" for x in c[:3])
    exact = c[0]['score'] >= 0.999
    # an exact name may ignore namesakes outside the national leagues (three
    # clubs called Sanjoanense); a near or contained name ('Paredes' in 'USC
    # Paredes') must be clear of EVERY other candidate, because the club we
    # mean may no longer play in a national league
    rivals = [x for x in c[1:] if (senior(x) or not senior(c[0]))] if exact else c[1:]
    margin = c[0]['score'] - (rivals[0]['score'] if rivals else 0.0)
    ok = c[0]['score'] >= TEAM_OK_SCORE and margin >= TEAM_OK_MARGIN and (senior(c[0]) or exact)
    return c[0], ('ok' if ok else 'review'), note


def link_teams(limit=None, only=None, retry=False):
    from urllib.parse import quote_plus
    have = pd.read_csv(TEAM_LINKS) if TEAM_LINKS.exists() else pd.DataFrame(
        columns=['teamId', 'team', 'tm_team_id', 'tm_slug', 'tm_name', 'score', 'status', 'candidates'])
    # every team already searched is skipped, whatever came of it: 'review'
    # and 'not_found' rows are settled by hand (status 'manual'), or retried
    # on request with --team or --retry
    done = set(have.teamId) if not retry else set(have.loc[have.status.isin(['ok', 'manual']), 'teamId'])
    todo = our_teams()
    todo = todo[todo.team == only] if only else todo[~todo.teamId.isin(done)]
    if limit:
        todo = todo.head(limit)
    cl, rows = Client(), []
    try:
        for r in todo.itertuples():
            cands, seen = [], set()
            for q in search_queries(r.team):
                html = cl.get('/schnellsuche/ergebnis/schnellsuche?query=' + quote_plus(q))
                for x in parse_club_search(html or ''):
                    if x['tm_team_id'] not in seen:
                        seen.add(x['tm_team_id'])
                        cands.append(x)
                best, status, note = pick_club(r.team, cands)
                if status == 'ok':
                    break
            rows.append({'teamId': r.teamId, 'team': r.team,
                         'tm_team_id': best['tm_team_id'] if best else None,
                         'tm_slug': best['tm_slug'] if best else None, 'tm_name': best['tm_name'] if best else None,
                         'score': round(best['score'], 3) if best else None, 'status': status, 'candidates': note})
            print(f"  {r.team:28s} → {rows[-1]['tm_name']} [{status}] {note}", flush=True)
            _save_team_links(have, rows)
    except Refused as e:
        print(f'STOPPED: {e}. Saved what was read; wait before the next run.')
    return rows


def _save_team_links(have, rows):
    new = pd.DataFrame(rows)
    keep = have[~have.teamId.isin(new.teamId) | (have.status == 'manual')]
    new = new[~new.teamId.isin(keep.teamId)]
    pd.concat([d for d in (keep, new) if len(d)], ignore_index=True).sort_values('team').to_csv(TEAM_LINKS, index=False)


def fetch_managers(limit=None):
    links = pd.read_csv(TEAM_LINKS)
    if 'managers' not in links.columns:
        links['managers'] = pd.NA
    have = pd.read_parquet(MANAGERS) if MANAGERS.exists() else pd.DataFrame()
    got_before = have.groupby('teamId').size() if len(have) else pd.Series(dtype='int64')
    links['managers'] = links['managers'].where(links['managers'].notna(), links.teamId.map(got_before))
    # a club is fetched once: 'managers' records how many rows its page gave (0 included)
    todo = links[links.status.isin(['ok', 'manual']) & links.managers.isna()
                 & links.tm_team_id.notna() & links.tm_slug.notna()]
    if limit:
        todo = todo.head(limit)
    cl, rows = Client(), []
    try:
        for r in todo.itertuples():
            html = cl.get(f'/{r.tm_slug}/mitarbeiterhistorie/verein/{int(r.tm_team_id)}')
            got = parse_manager_history(html or '')
            rows += [dict(x, teamId=r.teamId, team=r.team, tm_team_id=int(r.tm_team_id)) for x in got]
            print(f'  {r.team:28s} {len(got)} managers', flush=True)
            links.loc[links.teamId == r.teamId, 'managers'] = len(got)
            links.to_csv(TEAM_LINKS, index=False)
            if rows:
                pd.concat([d for d in (have, pd.DataFrame(rows)) if len(d)], ignore_index=True).to_parquet(MANAGERS, index=False)
    except Refused as e:
        print(f'STOPPED: {e}. Saved what was read; wait before the next run.')
    return rows


def close_stints(mg):
    """Manager stints with every end filled except the club's latest: a
    stint with no end date that is NOT the club's most recent one ends the
    day before the next appointment (Transfermarkt leaves some old ends blank)."""
    mg = mg.sort_values(['teamId', 'start']).copy()
    nxt = mg.groupby('teamId').start.shift(-1)
    stale = mg.end.isna() & nxt.notna()
    mg.loc[stale, 'end'] = nxt[stale] - pd.Timedelta(days=1)
    return mg


def stint_hits(tm, mg):
    """One row per match that falls in a manager's stint at that club. A
    match strictly inside a stint belongs to it; only a match inside NO
    stint may fall back on the few days before an appointment date (a new
    man often takes his first match before the date Transfermarkt gives).
    Where stints overlap, the later appointment wins."""
    mg = close_stints(mg)
    far = pd.Timestamp('2100-01-01')
    m = tm.merge(mg[['teamId', 'tm_coach_id', 'name', 'dob', 'start', 'end']], on='teamId', how='inner')
    day = pd.to_datetime(m.date).dt.tz_localize(None).dt.normalize()
    end = m.end.fillna(far)
    strict = m[(day >= m.start) & (day <= end)]
    strict = strict.sort_values('start').drop_duplicates(['matchId', 'teamId'], keep='last')
    loose = m[(day >= m.start - pd.Timedelta(days=STINT_SLACK_DAYS)) & (day < m.start)]
    key = set(zip(strict.matchId, strict.teamId))
    import numpy as np
    loose = loose.loc[np.array([k not in key for k in zip(loose.matchId, loose.teamId)], dtype=bool)]
    loose = loose.sort_values('start').drop_duplicates(['matchId', 'teamId'], keep='first')
    return pd.concat([strict, loose], ignore_index=True)


def verify_team_links(team_match=None, managers=None, coaches=None, links=None):
    """Check every club link against coaches the event provider DID name:
    on the matches where both sources name a coach, how often do the names
    share a word? A link that is not an exact name match and agrees on fewer
    than VERIFY_SHARE of at least VERIFY_MATCHES matches is set to 'review'.
    Returns the links frame with `agree_share` / `agree_n`."""
    tm = pd.read_parquet(TEAM_MATCH, columns=['matchId', 'teamId', 'team', 'coachId', 'date']) \
        if team_match is None else team_match
    mg = pd.read_parquet(MANAGERS) if managers is None else managers
    co = pd.read_parquet(HERE / 'models' / 'coaches' / 'coaches.parquet') if coaches is None else coaches
    ln = pd.read_csv(TEAM_LINKS) if links is None else links.copy()
    named = co[co.resolved.astype(bool)].set_index('coachId')
    words = {int(c): set(fold(f'{r.name} {r.fullName}').replace('.', ' ').split()) - _STOP for c, r in named.iterrows()}
    h = stint_hits(tm[tm.coachId.isin(words)], mg)
    h['agree'] = [bool((set(fold(n).split()) - _STOP) & words[int(c)]) for n, c in zip(h.name, h.coachId)]
    g = h.groupby('teamId').agree.agg(['mean', 'size'])
    ln['agree_share'] = ln.teamId.map(g['mean']).round(3)
    ln['agree_n'] = ln.teamId.map(g['size'])
    weak = (ln.status == 'ok') & (ln.score < 0.999) & ~((ln.agree_n >= VERIFY_MATCHES) & (ln.agree_share >= VERIFY_SHARE))
    wrong = (ln.status == 'ok') & (ln.agree_n >= VERIFY_MATCHES) & (ln.agree_share < 0.25)
    ln.loc[weak | wrong, 'status'] = 'review'
    if links is None:
        ln.to_csv(TEAM_LINKS, index=False)
    return ln


def link_coaches(team_match=None, managers=None):
    """Our coachId → Transfermarkt coach, from match dates alone. For every
    match a coach took at a linked club, find the manager whose stint covers
    that date; a coach is linked when ONE Transfermarkt coach covers at least
    COACH_OK_SHARE of his dated matches (and COACH_OK_MATCHES of them)."""
    tm = pd.read_parquet(TEAM_MATCH, columns=['matchId', 'teamId', 'team', 'coachId', 'date']) \
        if team_match is None else team_match
    mg = pd.read_parquet(MANAGERS) if managers is None else managers
    tm = tm[tm.coachId > 0]
    if team_match is None and TEAM_LINKS.exists():                       # only clubs whose link is sure
        sure = pd.read_csv(TEAM_LINKS)
        mg = mg[mg.teamId.isin(set(sure.loc[sure.status.isin(['ok', 'manual']), 'teamId']))]
    hit = stint_hits(tm, mg)
    linked_teams = set(mg.teamId)
    rows = []
    for cid, g in tm.groupby('coachId'):
        in_linked = g[g.teamId.isin(linked_teams)]
        h = hit[hit.coachId == cid]
        row = {'coachId': int(cid), 'matches': len(g), 'matches_at_linked_clubs': len(in_linked),
               'tm_coach_id': None, 'tm_name': None, 'tm_dob': None, 'matches_covered': 0, 'share': 0.0,
               'status': 'no_linked_club' if in_linked.empty else 'no_stint', 'candidates': ''}
        if len(h):
            c = h.groupby(['tm_coach_id', 'name']).size().sort_values(ascending=False)
            (best_id, best_name), n = c.index[0], int(c.iloc[0])
            share = n / len(in_linked)
            row.update({'tm_coach_id': int(best_id), 'tm_name': best_name,
                        'tm_dob': h[h.tm_coach_id == best_id].dob.iloc[0], 'matches_covered': n,
                        'share': round(share, 3),
                        'status': 'ok' if share >= COACH_OK_SHARE and n >= COACH_OK_MATCHES else 'review',
                        'candidates': '; '.join(f'{nm} {k}' for (_, nm), k in c.head(3).items())})
        rows.append(row)
    out = pd.DataFrame(rows).sort_values('matches', ascending=False)
    if team_match is None:
        out.to_csv(COACH_LINKS, index=False)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('step', choices=['teams', 'managers', 'verify', 'coaches'])
    ap.add_argument('--retry', action='store_true', help="teams: search 'review' / 'not_found' teams again")
    ap.add_argument('--limit', type=int)
    ap.add_argument('--team')
    a = ap.parse_args()
    if a.step == 'teams':
        link_teams(a.limit, a.team, a.retry)
    elif a.step == 'managers':
        fetch_managers(a.limit)
    elif a.step == 'verify':
        out = verify_team_links()
        print(out.status.value_counts().to_string())
    else:
        out = link_coaches()
        print(out.status.value_counts().to_string())
    return 0


if __name__ == '__main__':
    sys.exit(main())
