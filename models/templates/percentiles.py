"""The minutes floor behind the template percentiles, and how to place a
player who is BELOW it onto the same scale.

Pure pandas — no streamlit. ``app.calculate_player_percentiles_and_scores``
ranks only the players at or above a minutes floor (half the leader's
minutes, capped at 500' — :func:`minutes_floor`) and
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
    ``frame``: half the current maximum, capped at ``min_minutes`` (the
    early-season clamp). It rises smoothly and reaches ``min_minutes``
    once the leader has twice that. Until 2026-10-21 the clamp switched
    off the moment ONE player passed ``min_minutes``: 2026/27 after MW6
    (leader 592') jumped from 296' to 500', leaving 88 of 448 players
    ranked (5 strikers) and everyone else at 0. Readers of a scored frame
    must use THIS rather than a literal 500 or they disagree with the
    pipeline for the first weeks of a season.
    """
    if frame is None or 'totalMinutes' not in frame.columns:
        return int(min_minutes)
    mins = pd.to_numeric(frame['totalMinutes'], errors='coerce')
    _max = mins.max()
    if pd.isna(_max):
        return int(min_minutes)
    return int(max(1, min(min_minutes, int(_max * 0.5))))


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


def rank_in_template(frame, scored, template, position_groups, weights,
                     invert_metrics, floor, extra_metrics=()):
    """Return ``(out, placed)``: a copy of ``frame`` in which every row is
    ranked against ONE sample — the template's position group at or above
    ``floor`` in ``scored`` — and the playerIds whose score had to be placed.

    This is what a radar that overlays two players needs. The pipeline ranks
    each player in their OWN position group (a metric shared by two roles
    keeps the last role's ranking, see :func:`score_below_floor`), so two
    players on one chart could be ranked against different samples, and a
    sub-floor or out-of-group player has no numbers for the template at all.

    * percentiles: every row, every template metric plus ``extra_metrics``
      (columns outside the template, e.g. the DefR axes), average-method percent rank within the sample WITHOUT the row's own
      player plus the row — so a qualifying in-group player gets exactly the
      pipeline's ranking for that group; inverted metrics flipped;
    * score: kept from the pipeline for a player who qualifies in the group
      (it is the leaderboards' number); otherwise placed like
      :func:`score_below_floor` — the weighted sum of the template
      percentiles, min-max scaled onto the group's ``_TotalScore`` range with
      the player's own total counted in that range.

    Players are matched on ``playerId`` (merges upstream reset indices).
    ``scored`` is never modified.
    """
    out = frame.copy()
    placed = []
    group = position_groups.get(template, [])
    if out.empty or not group or 'playerId' not in out.columns:
        return out, placed
    metrics = list(weights[template]) + [m for m in extra_metrics if m not in weights[template]]
    s_mins = pd.to_numeric(scored['totalMinutes'], errors='coerce').fillna(0)
    in_group = scored['primaryPosition'].isin(group)
    sample = scored[in_group & (s_mins >= floor)]
    sample_pids = set(pd.to_numeric(sample['playerId'], errors='coerce').dropna().astype(int))

    for i in out.index:
        pid = int(out.at[i, 'playerId'])
        own = sample[pd.to_numeric(sample['playerId'], errors='coerce') != pid]
        for metric in metrics:
            if metric not in own.columns or metric not in out.columns:
                continue
            pop_vals = pd.to_numeric(own[metric], errors='coerce').fillna(0).to_numpy()
            if len(pop_vals) == 0:
                continue
            v = pd.to_numeric(pd.Series([out.at[i, metric]]), errors='coerce').fillna(0).iloc[0]
            p = _percent_rank_in_sample(pop_vals, float(v))
            out.at[i, metric + '_percentile'] = 1.0 - p if metric in invert_metrics else p

        if pid in sample_pids:
            continue
        placed.append(pid)
        total = 0.0
        for metric, weight in weights[template].items():
            pct = out.at[i, metric + '_percentile'] if metric + '_percentile' in out.columns else 0.0
            total += (0.0 if pd.isna(pct) else float(pct)) * weight
        out.at[i, template + '_TotalScore'] = total
        total_col = template + '_TotalScore'
        others = scored[in_group & (pd.to_numeric(scored['playerId'], errors='coerce') != pid)]
        if total_col in scored.columns and len(others):
            others_tot = pd.to_numeric(others[total_col], errors='coerce').dropna()
            lo = min(others_tot.min(), total) if len(others_tot) else total
            hi = max(others_tot.max(), total) if len(others_tot) else total
        else:
            lo = hi = total
        out.at[i, template + '_Score'] = (total - lo) / (hi - lo) * 100.0 if hi != lo else 0.0
    return out, placed
