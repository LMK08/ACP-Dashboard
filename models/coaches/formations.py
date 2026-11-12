"""Our own formation for each side in each match, from event locations only.
It is the only formation the dashboard shows (Lucas, 2026-11): no provider
formation label is used anywhere.

No provider formation label and no provider position label is read. A side's
shape is worked out from where its players touched the ball:

  1. The starters are the first eleven players of the side to appear in the
     events. The starting window runs from kick-off until the side's twelfth
     player first touches the ball (its first substitution, as the events
     show it) or it has a player sent off. If that leaves under 20 minutes
     (an early change), the stretch between that change and the next one is
     read instead, with the eleven then on the pitch.
  2. Each starter's place is the mean location of his open-play touches in
     that window: his own actions plus the passes he received. Restarts
     (corners, free kicks, throw-ins, goal kicks, penalties) and the passes
     received from them are left out (centre-backs go up for corners).
  3. The goalkeeper is the deepest player. The ten outfield places are put on
     a common scale (centred, each axis divided by its spread) and compared
     with a small set of reference shapes; each shape is fitted with the
     best one-to-one assignment of players to roles, and the closest wins.

What this can and cannot see: touches show a side's shape WITH the ball. A
back five that becomes a back three in possession reads as a back three, so
5-3-2 and 3-5-2 are one shape here, as are 4-1-4-1 / 4-3-3 and 4-4-1-1 /
4-2-3-1. The reference set keeps only shapes that touches can tell apart.

Tested and left out (2026-11): a second-half read, to say whether the shape
changed during a match. Second-half stretches between substitutions are
short, and the 'changed in-match' share that came out did not agree between
a coach's odd and even matches at all (0.02 on Liga 3) - noise, not a trait.

Checks on 6,008 team-matches (2026-11): a shape is read for 98.5 %; fitted
on two random halves of the same touches, the back line agrees in 89 % of
matches, the striker count in 83 %, the full name in 68 %.
"""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

# role places as (depth from own goal, width from the centre line); only the
# relative layout matters, each axis is standardised before fitting
_CB2 = [(0.20, -0.25), (0.20, 0.25)]
_CB3 = [(0.18, 0.0), (0.23, -0.48), (0.23, 0.48)]
_FB = [(0.42, -0.92), (0.42, 0.92)]
_WB = [(0.55, -0.95), (0.55, 0.95)]
TEMPLATES = {
    '4-4-2':   _CB2 + _FB + [(0.50, -0.25), (0.50, 0.25), (0.66, -0.90), (0.66, 0.90),
                             (0.86, -0.22), (0.86, 0.22)],
    '4-3-3':   _CB2 + _FB + [(0.42, 0.0), (0.58, -0.38), (0.58, 0.38), (0.76, -0.90), (0.76, 0.90),
                             (0.90, 0.0)],
    '4-2-3-1': _CB2 + _FB + [(0.45, -0.25), (0.45, 0.25), (0.70, 0.0), (0.73, -0.90), (0.73, 0.90),
                             (0.90, 0.0)],
    '4-3-1-2': _CB2 + [(0.50, -0.92), (0.50, 0.92)] + [(0.42, 0.0), (0.56, -0.45), (0.56, 0.45),
                                                       (0.72, 0.0), (0.88, -0.25), (0.88, 0.25)],
    '3-5-2':   _CB3 + _WB + [(0.45, 0.0), (0.60, -0.36), (0.60, 0.36), (0.88, -0.22), (0.88, 0.22)],
    '3-4-3':   _CB3 + _WB + [(0.50, -0.22), (0.50, 0.22), (0.76, -0.60), (0.76, 0.60), (0.90, 0.0)],
    '3-4-1-2': _CB3 + _WB + [(0.48, -0.22), (0.48, 0.22), (0.70, 0.0), (0.88, -0.25), (0.88, 0.25)],
}
STRIKERS = {'4-4-2': 2, '4-3-3': 1, '4-2-3-1': 1, '4-3-1-2': 2, '3-5-2': 2, '3-4-3': 1, '3-4-1-2': 2}

RESTARTS = ('corner', 'free_kick', 'throw_in', 'goal_kick', 'penalty')
MIN_WINDOW_MIN = 20.0     # shortest stretch a shape is read from
MIN_TOUCHES = 4           # per outfield player
EVENT_COLS = ['matchId', 'matchPeriod', 'minute', 'second', 'type.primary', 'team.id', 'player.id',
              'location.x', 'location.y', 'pass.accurate', 'pass.recipient.id',
              'pass.endLocation.x', 'pass.endLocation.y', 'infraction.redCard']


def _standardise(p):
    p = np.asarray(p, dtype=float)
    p = p - p.mean(axis=0)
    sd = p.std(axis=0)
    sd[sd == 0] = 1.0
    return p / sd


_T = {k: _standardise(v) for k, v in TEMPLATES.items()}


def back_line(formation):
    try:
        return int(str(formation).split('-')[0])
    except (ValueError, AttributeError):
        return np.nan


def fit_shape(xy):
    """Ten outfield places (depth, width) → (name, cost, margin to runner-up).
    Cost is the mean squared distance in standardised units."""
    p = _standardise(xy)
    costs = {}
    for name, t in _T.items():
        d = ((p[:, None, :] - t[None, :, :]) ** 2).sum(axis=2)
        r, c = linear_sum_assignment(d)
        costs[name] = d[r, c].mean()
    order = sorted(costs, key=costs.get)
    return order[0], float(costs[order[0]]), float(costs[order[1]] - costs[order[0]])


HALF_GAP = 15.0   # added to second-half minutes so first-half stoppage time never overlaps them


def _clock(ev):
    """Minutes on one running clock: the second half starts at 45 + HALF_GAP."""
    t = pd.to_numeric(ev['minute'], errors='coerce') + pd.to_numeric(ev['second'], errors='coerce').fillna(0) / 60.0
    return t + HALF_GAP * (ev['matchPeriod'].astype(str) == '2H')


def touches(ev):
    """Long table of touches: one row per action and one per open-play pass
    received. Columns matchId, teamId, playerId, t (minutes on one clock), x,
    y, restart (the action is a restart; dropped when places are taken) and
    received."""
    period = ev['matchPeriod'].astype(str)
    keep = period.isin(['1H', '2H']).to_numpy()
    t = _clock(ev)
    own = pd.DataFrame({'matchId': ev['matchId'], 'teamId': ev['team.id'], 'playerId': ev['player.id'],
                        't': t, 'x': ev['location.x'], 'y': ev['location.y'],
                        'restart': ev['type.primary'].isin(RESTARTS), 'received': False})[keep]
    got = (ev['pass.accurate'] == True) & ~ev['type.primary'].isin(RESTARTS)  # noqa: E712
    rec = pd.DataFrame({'matchId': ev['matchId'], 'teamId': ev['team.id'], 'playerId': ev['pass.recipient.id'],
                        't': t, 'x': ev['pass.endLocation.x'], 'y': ev['pass.endLocation.y'],
                        'restart': False, 'received': True})[keep & got.to_numpy()]
    out = pd.concat([own, rec], ignore_index=True)
    for c in ('playerId', 'teamId', 'x', 'y', 't'):
        out[c] = pd.to_numeric(out[c], errors='coerce')
    out = out.dropna(subset=['playerId', 'teamId', 'x', 'y', 't'])
    out = out[out.playerId > 0]
    return out.astype({'playerId': 'int64', 'teamId': 'int64', 'matchId': 'int64'})


def _places(g):
    """Mean open-play place per player: a frame indexed by playerId with x
    (depth), y (width) and n (touches)."""
    g = g[~g.restart]
    a = g.groupby('playerId').agg(x=('x', 'mean'), y=('y', 'mean'), n=('x', 'size'))
    return a


def _shape_of(g):
    """Fit one side over one stretch. None when it cannot be read."""
    a = _places(g)
    if len(a) != 11:
        return None
    gk = a.x.idxmin()
    out = a.drop(index=gk)
    if out.n.min() < MIN_TOUCHES:
        return None
    return fit_shape(out[['x', 'y']].to_numpy())


def team_match_shapes(tch, red_minutes=None):
    """One row per side per match: formation, fit cost, margin and the
    length of the window read. `tch` is touches(); `red_minutes` maps
    (matchId, teamId) → minute of the side's first red card."""
    red_minutes = red_minutes or {}
    rows = []
    for (mid, tid), g in tch.groupby(['matchId', 'teamId'], sort=False):
        first = g.groupby('playerId').t.min().sort_values()
        row = {'matchId': mid, 'teamId': tid, 'formation': None, 'formation_cost': np.nan,
               'formation_margin': np.nan, 'formation_window_min': np.nan}
        rows.append(row)
        if len(first) < 11:
            continue
        starters = list(first.index[:11])
        end = float(first.iloc[11]) if len(first) > 11 else float(g.t.max())
        red = red_minutes.get((mid, tid))
        if red is not None:
            end = min(end, float(red))
        start = 0.0
        if end < MIN_WINDOW_MIN and len(first) > 11 and (red is None or red > end):
            # an early substitution: read the next stretch, between that
            # change and the following one, with the eleven then on the pitch
            nxt = float(first.iloc[12]) if len(first) > 12 else float(g.t.max())
            if red is not None:
                nxt = min(nxt, float(red))
            last = g.groupby('playerId').t.max()
            on_now = [p for p in first.index[:12] if last[p] >= end]
            if len(on_now) == 11 and nxt - end >= MIN_WINDOW_MIN:
                start, end, starters = end, nxt, on_now
        row['formation_window_min'] = end - start
        if end - start < MIN_WINDOW_MIN:
            continue
        w = g[(g.t >= start) & (g.t < end) & g.playerId.isin(starters)]
        fit = _shape_of(w)
        if fit is None:
            continue
        row['formation'], row['formation_cost'], row['formation_margin'] = fit
    return pd.DataFrame(rows)


def red_card_minutes(ev):
    red = ev[(ev['infraction.redCard'] == True) & ev['team.id'].notna()]  # noqa: E712
    if red.empty:
        return {}
    t = _clock(red)
    s = pd.DataFrame({'m': red['matchId'].astype('int64'), 't': red['team.id'].astype('int64'), 'min': t})
    return s.groupby(['m', 't'])['min'].min().to_dict()


def formations_from_events(ev):
    """Events (EVENT_COLS) → one row per side per match."""
    return team_match_shapes(touches(ev), red_card_minutes(ev))
