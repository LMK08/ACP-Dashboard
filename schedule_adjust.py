"""Opponent-adjusted CURRENT-season totals once the second phase begins.

During Liga 3 / Campeonato second phases the schedule is unbalanced —
promotion-series teams face the strongest opposition, maintenance the
weakest — so raw season totals mis-state team strength in exactly the
months when standings pages and predictions matter most. This applies the
same 2-pass SOS correction used for priors (simulate_season
build_prior_strengths) to the current season's goal and xG totals.

Dormant during the first phase: with a single (balanced) stage the
function returns None and consumers keep raw totals.
"""
from __future__ import annotations

import numpy as np


def phase2_adjusted_totals(match_rows, xg_lookup):
    """match_rows: iterable of dicts with keys
        matchId, roundId, home, away, hg, ag
    xg_lookup: {(matchId, team_name): non-penalty xG}

    Returns {team: {'goals_for','goals_against','xG_for','xG_against'}}
    (season TOTALS, opponent-adjusted) or None while only one stage exists.
    """
    rows = list(match_rows)
    if not rows:
        return None
    round_counts = {}
    for r in rows:
        round_counts[r['roundId']] = round_counts.get(r['roundId'], 0) + 1
    if len(round_counts) < 2:
        return None  # first phase only — balanced schedule, nothing to adjust
    # require a real second stage, not a stray round id
    first_stage = max(round_counts, key=round_counts.get)
    if sum(n for rid, n in round_counts.items() if rid != first_stage) < 8:
        return None

    logs = {}
    for r in rows:
        mid = r['matchId']
        hx, ax = xg_lookup.get((mid, r['home'])), xg_lookup.get((mid, r['away']))
        logs.setdefault(r['home'], []).append((r['away'], r['hg'], r['ag'], hx, ax))
        logs.setdefault(r['away'], []).append((r['home'], r['ag'], r['hg'], ax, hx))

    raw = {}
    for t, lst in logs.items():
        n = len(lst)
        xs = [(x, xa) for _, _, _, x, xa in lst if x is not None and xa is not None]
        raw[t] = {
            'n': n,
            'gfpg': sum(g for _, g, _, _, _ in lst) / n,
            'gapg': sum(g for _, _, g, _, _ in lst) / n,
            'xfpg': (sum(x for x, _ in xs) / len(xs)) if xs else None,
            'xapg': (sum(x for _, x in xs) / len(xs)) if xs else None,
            'nx': len(xs),
        }
    lg_gf = float(np.mean([r['gfpg'] for r in raw.values()]))
    xf_vals = [r['xfpg'] for r in raw.values() if r['xfpg'] is not None]
    lg_xf = float(np.mean(xf_vals)) if xf_vals else None

    adj = {t: dict(r) for t, r in raw.items()}
    for _ in range(2):
        ref = {t: dict(r) for t, r in adj.items()}
        for t, lst in logs.items():
            n = len(lst)
            g_for = g_ag = x_for = x_ag = 0.0
            nx = 0
            for opp, gf, ga, xf, xa in lst:
                o = ref.get(opp)
                g_for += gf - ((o['gapg'] - lg_gf) if o else 0.0)
                g_ag += ga - ((o['gfpg'] - lg_gf) if o else 0.0)
                if xf is not None and xa is not None:
                    o_xa = o.get('xapg') if o else None
                    o_xf = o.get('xfpg') if o else None
                    x_for += xf - ((o_xa - lg_xf) if (o_xa is not None and lg_xf is not None) else 0.0)
                    x_ag += xa - ((o_xf - lg_xf) if (o_xf is not None and lg_xf is not None) else 0.0)
                    nx += 1
            adj[t]['gfpg'] = g_for / n
            adj[t]['gapg'] = g_ag / n
            if nx:
                adj[t]['xfpg'] = x_for / nx
                adj[t]['xapg'] = x_ag / nx

    out = {}
    for t, a in adj.items():
        n, nx = a['n'], a['nx']
        rec = {'goals_for': a['gfpg'] * n, 'goals_against': a['gapg'] * n}
        if nx and a['xfpg'] is not None:
            # scale per-game adjusted xG back to the team's xG-covered matches
            rec['xG_for'] = a['xfpg'] * nx
            rec['xG_against'] = a['xapg'] * nx
        out[t] = rec
    return out
