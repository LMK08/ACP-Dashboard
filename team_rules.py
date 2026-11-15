"""Rules that more than one page must share (pure pandas / numpy; no Streamlit).

Each of these existed in two or three slightly different copies (app.py's
team radars, the season report, process_data.py's match report), and the
copies had drifted: the same label showed different numbers on different
pages (2026-11 metric tracing). One definition lives here; the pages call it.

Coordinates are Wyscout's: 0-100 on both axes, always in the ACTING team's
own direction of play (x = 0 its own goal line, x = 100 the opponent's).
"""
import numpy as np
import pandas as pd

from event_tags import TagIndex

PITCH_LENGTH_M, PITCH_WIDTH_M = 105.0, 68.0

# Pressing zone for PPDA: the opponent's own 60 % of the pitch. In the
# opponent's coordinates that is x <= 60; in the pressing team's, x >= 40.
PRESS_PASS_MAX_X = 60
PRESS_ACTION_MIN_X = 40


def _tags(events):
    return TagIndex(events.get('type.secondary', pd.Series([None] * len(events), index=events.index)))


def dribble_attempts(events):
    """Boolean Series: the rows that are a player's OWN dribble attempt.

    Wyscout writes a dribble as a duel with two rows, the dribbler's (tagged
    'dribble') and the defender's (tagged 'defensive_duel'), and sets
    ``groundDuel.takeOn`` on BOTH. Counting the flag therefore counted every
    take-on a defender faced as a dribble of his own: a third of all
    "dribbles" in Liga 3 2026/27 (1,081 of 3,268). The tag is the dribbler's."""
    return _tags(events).has('dribble')


def correct_dribble_flags(events):
    """Bring the STORED dribble flags of an events frame up to the rule above.

    raw_events.parquet carries `is_dribble_attempt` / `is_custom_dribble_success`
    for every event, written when the event was first fetched; events stored
    before 2026-11-14 carry the old flag (defenders' rows included). A row's
    success conditions never depended on the attempt rule, so the corrected
    success is the stored success on a row that is still an attempt.
    Returns the frame (modified in place)."""
    attempt = dribble_attempts(events)
    if 'is_custom_dribble_success' in events.columns:
        events['is_custom_dribble_success'] = (events['is_custom_dribble_success'] == True) & attempt
    events['is_dribble_attempt'] = attempt
    return events


def ppda(team_events, opponent_events):
    """Passes per defensive action: the opponent's passes in ITS OWN 60 % of
    the pitch, divided by the team's defensive actions in that same zone
    (interceptions, clearances, defensive duels, sliding tackles). Lower =
    more pressing. None when the team made no such action.

    This is the season report's rule (app.compute_team_season_metrics). The
    match report and the events-based radar used to count the opponent's
    passes at x >= 40 of the OPPONENT's coordinates, i.e. in its attacking
    60 %, the wrong end of the pitch."""
    if team_events is None or opponent_events is None or len(team_events) == 0:
        return None
    opp_x = pd.to_numeric(opponent_events.get('location.x'), errors='coerce')
    passes = int(((opponent_events.get('type.primary') == 'pass') & (opp_x <= PRESS_PASS_MAX_X)).sum())
    primary = team_events.get('type.primary')
    tags = _tags(team_events)
    action = ((primary == 'interception') | (primary == 'clearance')
              | tags.has('defensive_duel') | tags.has('sliding_tackle'))
    own_x = pd.to_numeric(team_events.get('location.x'), errors='coerce')
    actions = int((action & (own_x >= PRESS_ACTION_MIN_X)).sum())
    return passes / actions if actions else None


def forward_pass_share(passes):
    """Directness: the share of passes played forward, in % (Wyscout's
    'forward_pass' tag; the team file's forward_passes / passes)."""
    if passes is None or len(passes) == 0:
        return 0.0
    return float(_tags(passes).has('forward_pass').mean() * 100.0)


def shot_distance_m(x, y):
    """Distance in METRES from (x, y) to the centre of the opponent's goal."""
    dx = (100.0 - pd.to_numeric(x, errors='coerce')) * PITCH_LENGTH_M / 100.0
    dy = (50.0 - pd.to_numeric(y, errors='coerce')) * PITCH_WIDTH_M / 100.0
    return np.sqrt(dx ** 2 + dy ** 2)


# --- One match, one spelling of each club -------------------------------------
# A match report picks each side's events BY NAME, with the names of the match
# list (matches_summary: read from Wyscout's match label, and stripped). The
# events carry their own spelling, and for two clubs the two differ:
# 'Pevidem' / 'Pevidém' (27 matches) and 'Mortágua' / 'Mortágua ' with a
# trailing space (48 matches). In those 75 matches one side's tables, shot map
# and figures were empty and the other side's PPDA read 0 (2026-11-14).
# The match list holds no team ids (homeTeamId / awayTeamId are empty for every
# match), so the sides cannot be matched on an id.
# Since 2026-11-15 the app gives its match list the events' spelling when it
# loads the data (match_list_in_events_spelling, app.load_data), so in the app
# the names agree before a page asks; the pipeline (process_data) still
# computes with the stored match list and needs align_match_team_names.

TEAM_NAME_COLUMNS = ('team.name', 'opponentTeam.name', 'possession.team.name')


def team_name_key(name):
    """A club's name without accents, capitals or stray spaces: 'Pevidém' and
    'Pevidem' share a key, 'Mortágua ' and 'Mortágua' too. (The same rule as
    models/value/availability._norm_name.)"""
    import unicodedata
    s = unicodedata.normalize('NFKD', str(name))
    return ' '.join(''.join(c for c in s if not unicodedata.combining(c)).casefold().split())


def match_team_name_map(event_names, home_team, away_team):
    """{spelling in the events: spelling in the match list}, for a side whose
    match-list name is not in the events. A side is mapped only when exactly
    ONE other name in the events has its key; anything else is left alone."""
    present = [n for n in event_names if isinstance(n, str)]
    listed = [n for n in (home_team, away_team) if isinstance(n, str)]
    out = {}
    for want in listed:
        if want in present:
            continue
        key = team_name_key(want)
        same = [n for n in present if n not in listed and team_name_key(n) == key]
        if len(same) == 1:
            out[same[0]] = want
    return out


def align_match_team_names(match_df, home_team, away_team):
    """ONE match's events, with each club named as the match list names it.

    Returns the frame itself when the names already agree (every match but
    the 75 above), else a copy whose team-name columns are rewritten. Only
    the spelling changes: no row is added, dropped or moved to another side."""
    if match_df is None or len(match_df) == 0 or 'team.name' not in match_df.columns:
        return match_df
    mapping = match_team_name_map(list(match_df['team.name'].dropna().unique()), home_team, away_team)
    if not mapping:
        return match_df
    out = match_df.copy()
    for col in TEAM_NAME_COLUMNS:
        if col in out.columns:
            # the app stores these columns as categories: rewrite them as text
            out[col] = out[col].astype(object).replace(mapping)
    return out


def match_list_in_events_spelling(matches_df, events_df):
    """The match list with each club named as the EVENTS name it.

    Every other file the dashboard reads (player stats, minutes, line-ups,
    the engine's tables, the phase profiles, Wyscout's team file) carries
    the events' spelling; the match list is the odd one out, so a SEASON
    page that picked a team from the match list found none of its events
    (2026-11-15: Team Analysis and the Opposition Report were empty for
    'Pevidem' 2021/22 and 'Mortágua').

    Returns (frame, renames). `renames` is {matchId: {old name: new name}}
    for the rows that changed; the frame is the one given when nothing did.
    A row is renamed when its match's events hold exactly ONE other name
    with the same key (match_team_name_map). A club that is not in its
    match's events at all (a match with no events, a fixture) follows its
    SEASON: renamed when that name is spelt ONE other way in that season's
    events and never as the match list spells it."""
    need = {'matchId', 'homeTeamName', 'awayTeamName'}
    if matches_df is None or len(matches_df) == 0 or not need.issubset(matches_df.columns):
        return matches_df, {}
    if events_df is None or len(events_df) == 0 or not {'matchId', 'team.name'}.issubset(events_df.columns):
        return matches_df, {}
    pairs = events_df[['matchId', 'team.name']].drop_duplicates().dropna()
    names_by_match = {}
    for mid, name in zip(pairs['matchId'].tolist(), pairs['team.name'].tolist()):
        names_by_match.setdefault(mid, []).append(name)

    has_season = 'seasonId' in matches_df.columns
    mids = matches_df['matchId'].tolist()
    homes, aways = matches_df['homeTeamName'].tolist(), matches_df['awayTeamName'].tolist()
    seasons = matches_df['seasonId'].tolist() if has_season else [None] * len(mids)

    per_match = {}            # row position -> {old: new}
    learned = {}              # (season, old) -> {new, ...}
    as_listed = set()         # (season, name) found in a match's events exactly as listed
    for i, (mid, home, away, season) in enumerate(zip(mids, homes, aways, seasons)):
        present = names_by_match.get(mid)
        if not present:
            continue
        for name in (home, away):
            if isinstance(name, str) and name in present:
                as_listed.add((season, name))
        to_list = match_team_name_map(present, home, away)      # {events: list}
        if to_list:
            per_match[i] = {old: new for new, old in to_list.items()}
            for new, old in to_list.items():
                learned.setdefault((season, old), set()).add(new)
    by_season = {k: next(iter(v)) for k, v in learned.items() if len(v) == 1 and k not in as_listed}

    renames, new_home, new_away = {}, list(homes), list(aways)
    for i, (mid, home, away, season) in enumerate(zip(mids, homes, aways, seasons)):
        row = dict(per_match.get(i, {}))
        present = names_by_match.get(mid) or ()
        for name in (home, away):
            # not in this match's events (it has none, or only the other
            # side's): the club follows its season
            if (isinstance(name, str) and name not in row and name not in present
                    and (season, name) in by_season):
                row[name] = by_season[(season, name)]
        if not row:
            continue
        new_home[i], new_away[i] = row.get(home, home), row.get(away, away)
        if new_home[i] == new_away[i]:                          # never two sides with one name
            new_home[i], new_away[i] = home, away
            continue
        renames[mid] = row
    if not renames:
        return matches_df, {}
    out = matches_df.copy()
    out['homeTeamName'], out['awayTeamName'] = new_home, new_away
    return out, renames


def relabel_match_tables(match_data, renames):
    """The stored match tables (all_match_data.pkl) are labelled with the
    names the match list had when they were computed. After
    match_list_in_events_spelling their columns take the same new names, so
    a page can go on reading `table[team]`. `match_data` is changed in place
    (each renamed table is a new frame); returns how many matches changed."""
    changed = 0
    for mid, mapping in (renames or {}).items():
        md = (match_data or {}).get(mid)
        tables = md.get('team_stats') if isinstance(md, dict) else None
        if not isinstance(tables, dict):
            continue
        hit = False
        for key, table in list(tables.items()):
            if isinstance(table, pd.DataFrame) and any(c in mapping for c in table.columns):
                tables[key] = table.rename(columns=mapping)
                hit = True
        changed += int(hit)
    return changed


def entry_for_team(by_team, team):
    """`by_team[team]` for a dict keyed by club name (a match's line-up
    sheets are keyed by the events' spelling), also when the two spellings
    differ. None when the club is not there."""
    if not by_team or not isinstance(team, str):
        return None
    if team in by_team:
        return by_team[team]
    key = team_name_key(team)
    same = [k for k in by_team if isinstance(k, str) and team_name_key(k) == key]
    return by_team[same[0]] if len(same) == 1 else None
