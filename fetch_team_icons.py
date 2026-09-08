# fetch_team_icons.py
"""Download missing team crest PNGs into icons/ from the Wyscout API.

The team scatter plots (plot_team_strength / plot_custom_scatter) look up
`icons/<teamName>.png` and fall back to a text label when the file is missing —
so every team in the data should have a crest here. This script walks
matches_summary.parquet, finds teams without an icon file, fetches
/v3/teams/{wyId} for them (per-competition credentials from .env, same as the
other fetchers) and saves the crest from imageDataURL.

Run after new teams enter the data (new season, new competition):
    python fetch_team_icons.py
"""

import os
import sys

import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv

from league_config import COMPETITIONS, get_credentials

# Team ids come from raw_events (team.id/team.name): matches_summary's
# homeTeamId/awayTeamId columns are unpopulated (see commit e619e44).
EVENTS_FILE = 'raw_events.parquet'
ICONS_DIR = 'icons'
BASE_URL = 'https://apirest.wyscout.com/v3/teams'


def _norm_name(s):
    """Accent- and case-insensitive form for exact name comparison."""
    import unicodedata
    return ''.join(c for c in unicodedata.normalize('NFKD', s)
                   if not unicodedata.combining(c)).casefold().strip()


def _icon_path(team_name):
    """Mirror the plotters' lookup exactly (app.py plot_team_strength /
    plot_custom_scatter): icons/<name>.png with slashes replaced."""
    safe = team_name.replace('/', '_').replace('\\', '_')
    return os.path.join(ICONS_DIR, f'{safe}.png')


def collect_missing(teams_df):
    """{(comp_id, team_id): team_name} for teams without an icon file."""
    missing = {}
    for tid, name, comp in teams_df.itertuples(index=False):
        if os.path.exists(_icon_path(name)):
            continue
        missing.setdefault((int(comp), int(tid)), name)
    return missing


def _save_if_valid_png(content, team_name):
    """Write the crest only if the bytes are a real image (the CDN 404 page
    and error JSON both come back with 200-adjacent noise otherwise)."""
    import io
    from PIL import Image
    try:
        with Image.open(io.BytesIO(content)) as im:
            im.verify()
    except Exception:
        return False
    with open(_icon_path(team_name), 'wb') as f:
        f.write(content)
    return True


def fetch_crest(session, comp_id, team_id, team_name):
    """Fetch one team's crest; returns 'saved' | 'no-image' | 'error'.

    Tries /v3/teams/{id} first (works for teams inside the account's current
    license scope), then falls back to the public crest CDN with the legacy
    g{wyId} pattern — the API returns 400/403 for out-of-scope teams
    (historical seasons, other competitions), but many of their crests are
    still publicly served.
    """
    user, password = get_credentials(comp_id)
    api_error = None
    if user and password:
        try:
            r = session.get(f'{BASE_URL}/{team_id}',
                            auth=HTTPBasicAuth(user, password), timeout=15)
            if r.status_code not in (400, 403, 404):
                r.raise_for_status()
                image_url = r.json().get('imageDataURL')
                if image_url:
                    img = session.get(image_url, timeout=15)
                    img.raise_for_status()
                    if _save_if_valid_png(img.content, team_name):
                        return 'saved'
                # fall through to the CDN — the API can answer with an
                # empty imageDataURL for a team whose crest the CDN serves
        except requests.exceptions.RequestException as e:
            api_error = e

    # CDN fallback for out-of-scope teams
    for pattern in (f'g{team_id}', f'{team_id}'):
        try:
            img = session.get(
                f'https://cdn5.wyscout.com/photos/team/public/'
                f'{pattern}_120x120.png', timeout=15)
            if img.status_code == 200 and _save_if_valid_png(img.content,
                                                             team_name):
                return 'saved'
        except requests.exceptions.RequestException:
            pass

    if api_error is not None:
        print(f'  ! error fetching {team_name} ({team_id}): {api_error}')
        return 'error'
    return 'no-image'


def main(limit=None):
    load_dotenv()
    os.makedirs(ICONS_DIR, exist_ok=True)
    teams_df = (
        pd.read_parquet(EVENTS_FILE,
                        columns=['team.id', 'team.name', 'competitionId'])
        .dropna()
        .drop_duplicates()
    )
    missing = collect_missing(teams_df)

    # Teams that appear only in fixtures (no events yet — e.g. newly promoted/
    # relegated sides early in a season) have no team.id in raw_events.
    # Resolve their wyId via /v3/search instead.
    fixture_names = set()
    if os.path.exists('matches_summary.parquet'):
        ms = pd.read_parquet(
            'matches_summary.parquet',
            columns=['homeTeamName', 'awayTeamName', 'competitionId'])
        for side in ('homeTeamName', 'awayTeamName'):
            for name, comp in ms[[side, 'competitionId']].dropna()\
                    .drop_duplicates().itertuples(index=False):
                fixture_names.add((name, int(comp)))
    known_names = set(teams_df['team.name'])
    session_search = requests.Session()
    for name, comp in sorted(fixture_names):
        if name in known_names or os.path.exists(_icon_path(name)) \
                or comp not in COMPETITIONS:
            continue
        user, password = get_credentials(comp)
        if not user or not password:
            continue
        try:
            r = session_search.get(
                'https://apirest.wyscout.com/v3/search',
                params={'query': name, 'objType': 'team'},
                auth=HTTPBasicAuth(user, password), timeout=15)
            r.raise_for_status()
            # Only accept an EXACT name match (accent/case-insensitive):
            # Wyscout search is fuzzy, and a near-miss here would save the
            # wrong club's crest under this team's name — silently wrong on
            # every scatter plot, and never retried because the file exists.
            clubs = [c for c in r.json()
                     if c.get('type') == 'club' and c.get('wyId')
                     and _norm_name(name) in (_norm_name(c.get('name', '')),
                                              _norm_name(c.get('officialName',
                                                               '')))]
            if clubs:
                missing.setdefault((comp, int(clubs[0]['wyId'])), name)
                print(f'  resolved {name} -> wyId {clubs[0]["wyId"]} '
                      f'via search')
            else:
                print(f'  ? no exact search match for {name} — skipped')
        except requests.exceptions.RequestException as e:
            print(f'  ! search failed for {name}: {e}')
    known_comps = {c for c, _ in missing if c in COMPETITIONS}
    skipped = {name for (c, _), name in missing.items()
               if c not in COMPETITIONS}
    if skipped:
        print(f'{len(skipped)} teams in competitions without configured '
              f'credentials — skipped: {sorted(skipped)[:5]}...')
    todo = {(c, t): n for (c, t), n in missing.items() if c in known_comps}
    if limit:
        todo = dict(list(todo.items())[:limit])
    print(f'{len(todo)} teams missing icons')

    session = requests.Session()
    counts = {'saved': 0, 'no-image': 0, 'error': 0}
    for (comp_id, team_id), name in sorted(todo.items(), key=lambda kv: kv[1]):
        if os.path.exists(_icon_path(name)):
            continue  # saved under another (comp, id) pairing this run
        result = fetch_crest(session, comp_id, team_id, name)
        counts[result] += 1
        print(f'  {result:>8}  {name}')
    print(f"Done: {counts['saved']} saved, {counts['no-image']} without an "
          f"image, {counts['error']} errors")
    return 0 if counts['error'] == 0 else 1


if __name__ == '__main__':
    _limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    sys.exit(main(_limit))
