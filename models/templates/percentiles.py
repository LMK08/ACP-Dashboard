"""The minutes floor behind the template percentiles, and how to place a
player who is BELOW it onto the same scale.

Pure pandas — no streamlit. ``app.calculate_player_percentiles_and_scores``
ranks only the players at or above a minutes floor (500', or the
early-season clamp from :func:`minutes_floor` while nobody has 500') and
leaves everyone else at 0 for every ``<metric>_percentile`` and
``<role>_Score`` column. That is deliberate for the leaderboards; it is not
what a radar of such a player should show (an empty polygon, "Score 0.00,
Rank last" — the 2026/27 bulk export at matchweek 4, when the floor was
197' and 28% of the >= 90' players sat under it).

:func:`score_below_floor` gives a sub-floor player the numbers the pipeline
would have given them had they qualified, without moving anyone else: the
player is placed into the qualifying sample ON THEIR OWN and ranked exactly
as the pipeline ranks the sample. Sample players' cached numbers are never
touched, so the radars of everyone else stay as they are.
"""
import numpy as np
import pandas as pd


def minutes_floor(frame, min_minutes=500):
    """The floor ``calculate_player_percentiles_and_scores`` applies to
    ``frame``: ``min_minutes``, or half the current maximum while nobody
    has reached ``min_minutes`` (the early-season clamp). Readers of a
    scored frame must use THIS rather than a literal 500 or they disagree
    with the pipeline for the first weeks of a season.
    """
    if frame is None or 'totalMinutes' not in frame.columns:
        return int(min_minutes)
    mins = pd.to_numeric(frame['totalMinutes'], errors='coerce')
    _max = mins.max()
    if pd.notna(_max) and _max < min_minutes:
        return max(1, int(_max * 0.5))
    return int(min_minutes)


def _percent_rank_in_sample(sample_values, value):
    """Average-method percent rank of ``value`` inside ``sample_values``
    plus ``value`` itself — what ``pd.Series.rank(pct=True)`` returns for
    that element once it is appended to the sample."""
    lt = float((sample_values < value).sum())
    eq = float((sample_values == value).sum())
    return (lt + 1.0 + eq / 2.0) / (len(sample_values) + 1.0)


def score_below_floor(frame, scored, position_groups, weights, invert_metrics, floor):
    """Return a copy of ``frame`` in which every row with
    ``totalMinutes < floor`` carries percentiles, ``<role>_TotalScore`` and
    ``<role>_Score`` for each role whose group holds its position.

    ``scored`` is the pipeline's full output for the scope; its rows at or
    above ``floor`` are the sample. Rows of ``frame`` at or above the floor
    are returned unchanged.

    The arithmetic mirrors the pipeline pass for pass, so a player scored
    here gets the percentiles the pipeline would have ranked them at inside
    the sample, and a score on the sample's existing range
    (``tests/test_template_percentiles.py`` checks both against the
    pipeline's own output):

    * percentiles first, role by role in ``position_groups`` order, a metric
      shared by two roles keeping the LAST role's value (as the pipeline's
      column overwrite does); average-method percent rank within the
      role's sample plus the player, inverted metrics flipped;
    * then scores from those stored percentiles: sum of percentile x weight,
      min-max scaled onto the position group's ``_TotalScore`` range with
      the player's own total counted in that range.
    """
    out = frame.copy()
    if out.empty or 'totalMinutes' not in out.columns or 'primaryPosition' not in out.columns:
        return out

    mins = pd.to_numeric(out['totalMinutes'], errors='coerce').fillna(0)
    targets = out.index[mins < floor]
    if targets.empty:
        return out

    sample = scored[pd.to_numeric(scored['totalMinutes'], errors='coerce').fillna(0) >= floor]
    target_pos = out.loc[targets, 'primaryPosition'].astype(str)

    # Pass 1 — percentiles (last role to use a metric wins, as in the pipeline).
    for role, group in position_groups.items():
        idx = targets[target_pos.isin(group).values]
        if idx.empty:
            continue
        pop = sample[sample['primaryPosition'].isin(group)]
        if pop.empty:
            continue
        for metric in weights[role]:
            if metric not in scored.columns or metric not in out.columns:
                continue
            pop_vals = pd.to_numeric(pop[metric], errors='coerce').fillna(0).to_numpy()
            vals = pd.to_numeric(out.loc[idx, metric], errors='coerce').fillna(0)
            pcts = np.array([_percent_rank_in_sample(pop_vals, v) for v in vals], dtype=float)
            if metric in invert_metrics:
                pcts = 1.0 - pcts
            out.loc[idx, metric + '_percentile'] = pcts

    # Pass 2 — role scores from the stored percentiles.
    for role, group in position_groups.items():
        idx = targets[target_pos.isin(group).values]
        if idx.empty:
            continue
        total = pd.Series(0.0, index=idx)
        for metric, weight in weights[role].items():
            pct_col = metric + '_percentile'
            if pct_col in out.columns:
                total = total.add(
                    pd.to_numeric(out.loc[idx, pct_col], errors='coerce').fillna(0) * weight,
                    fill_value=0)
        out.loc[idx, role + '_TotalScore'] = total

        total_col = role + '_TotalScore'
        group_rows = scored.index[scored['primaryPosition'].isin(group)]
        for i in idx:
            others = group_rows.difference([i])
            if total_col in scored.columns and len(others):
                others_tot = pd.to_numeric(scored.loc[others, total_col], errors='coerce').dropna()
                lo = min(others_tot.min(), total[i]) if len(others_tot) else total[i]
                hi = max(others_tot.max(), total[i]) if len(others_tot) else total[i]
            else:
                lo = hi = total[i]
            out.loc[i, role + '_Score'] = (
                (total[i] - lo) / (hi - lo) * 100.0 if hi != lo else 0.0)
    return out
