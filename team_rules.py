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
