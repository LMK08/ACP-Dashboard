"""Formation and line-up for the dashboard's graphics, from OUR OWN read of
touch locations (models/coaches/formations.py). No provider formation label
and no provider position code is used.

The coach build saves, for every side in every match it could read, the
formation and the eleven average places it was fitted on
(models/coaches/coach_team_match.parquet, team_match_places.parquet). Here
those become:

  line_up(team_id, weights, names)   the side's formation over a set of
        matches (weighted most common) and the player in each slot of it:
        every match's eleven are fitted to THAT formation's slots, each
        match votes with its weight, and slots are filled from the strongest
        vote down, one slot per player.

It is the shape with the ball - see formations.py for what that means.
"""
import os
from collections import Counter, defaultdict
from functools import lru_cache

import pandas as pd

from league_config import PLAYER_ID_ALIASES
from models.coaches import formations as fm

_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'coaches')
SHAPES = os.path.join(_DIR, 'coach_team_match.parquet')
PLACES = os.path.join(_DIR, 'team_match_places.parquet')


@lru_cache(maxsize=2)
def _tables(_shapes_mtime, _places_mtime):
    shapes = pd.read_parquet(SHAPES, columns=['matchId', 'teamId', 'formation']).dropna(subset=['formation'])
    places = pd.read_parquet(PLACES)
    form = {(int(m), int(t)): f for m, t, f in zip(shapes.matchId, shapes.teamId, shapes.formation)}
    by_match = {k: g for k, g in places.groupby(['matchId', 'teamId'])}
    return form, by_match


def tables():
    """(formation by (matchId, teamId), places frame by (matchId, teamId));
    two empty dicts when the coach tables are not built."""
    if not (os.path.exists(SHAPES) and os.path.exists(PLACES)):
        return {}, {}
    return _tables(os.path.getmtime(SHAPES), os.path.getmtime(PLACES))


def team_id(events_df, team_name):
    """The side's id from its events; None if it has none."""
    if 'team.id' not in events_df.columns:
        return None
    ids = pd.to_numeric(events_df.loc[events_df['team.name'] == team_name, 'team.id'], errors='coerce').dropna()
    return int(ids.iloc[0]) if len(ids) else None


def _canonical(pid):
    return int(PLAYER_ID_ALIASES.get(int(pid), int(pid)))


def read_matches(tid, match_ids):
    """The ids in match_ids (kept in order) that have a formation read."""
    form = tables()[0]
    return [int(m) for m in match_ids if tid is not None and (int(m), int(tid)) in form]


def formation_of(match_id, tid):
    return tables()[0].get((int(match_id), int(tid)))


def line_up(tid, weights, names, exclude=()):
    """(formation, {slot: {'id', 'name'}}) for one side over the matches in
    `weights` ({matchId: weight}); (None, {}) when none of them was read.
    `names` maps playerId → name; players named in `exclude` are left out."""
    form, by_match = tables()
    if tid is None:
        return None, {}
    read = {int(m): float(w) for m, w in weights.items() if (int(m), int(tid)) in form}
    if not read:
        return None, {}
    tally = Counter()
    for m, w in read.items():
        tally[form[(m, int(tid))]] += w
    top = max(tally.values())
    # a tie goes to the heaviest single match (the most recent, for the
    # recency-weighted callers), then to the name for a stable answer
    best = sorted((f for f, v in tally.items() if v == top),
                  key=lambda f: (-max(w for m, w in read.items() if form[(m, int(tid))] == f), f))[0]

    votes = defaultdict(Counter)
    for m, w in read.items():
        g = by_match.get((m, int(tid)))
        if g is None or len(g) != 11:
            continue
        gk = g[g.gk]
        out = g[~g.gk]
        if len(gk) != 1 or len(out) != 10:
            continue
        # the places carry raw event ids; the app's names are keyed by the
        # canonical id (league_config.PLAYER_ID_ALIASES), so a player with two
        # ids votes as one
        votes['GK'][_canonical(gk.playerId.iloc[0])] += w
        for slot, pid in fm.assign_slots(out.playerId.to_numpy(), out[['x', 'y']].to_numpy(), best).items():
            votes[slot][_canonical(pid)] += w

    banned = set(exclude)
    ranked = sorted(((v, slot, pid) for slot, c in votes.items() for pid, v in c.items()
                     if names.get(pid) and names[pid] not in banned),
                    key=lambda r: (-r[0], r[1], r[2]))
    xi, used = {}, set()
    for _, slot, pid in ranked:
        if slot in xi or pid in used:
            continue
        xi[slot] = {'id': pid, 'name': names[pid]}
        used.add(pid)
    order = ['GK'] + fm.SLOTS[best]
    return best, {s: xi[s] for s in order if s in xi}


def player_names(events_df, team_name):
    """playerId → name for one side, from its events."""
    te = events_df.loc[events_df['team.name'] == team_name, ['player.id', 'player.name']].dropna()
    te = te.drop_duplicates('player.id')
    ids = pd.to_numeric(te['player.id'], errors='coerce')
    return {int(i): str(n) for i, n in zip(ids, te['player.name']) if pd.notna(i)}
