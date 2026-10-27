"""models/templates/percentiles.py — the minutes floor behind the template
percentiles and the sub-floor scoring the bulk radar export relies on.

The parity test re-implements the pipeline's arithmetic
(app.calculate_player_percentiles_and_scores: average-method percent rank
within the qualifying position group, inverted metrics flipped, sum of
percentile x weight, min-max onto the group's range) on a small frame —
app.py cannot be imported here (it boots the app). The same check was run
against the live pipeline's cached 2026/27 output for every one of its
214 qualifying players when the module was written (max |delta| 0.0).
"""
import numpy as np
import pandas as pd
import pytest

from models.templates import percentiles as tp

POSITION_GROUPS = {
    'Stopper': ['CB', 'LCB', 'RCB'],
    'Ball-Playing Centerback': ['CB', 'LCB', 'RCB'],
    'Full Back': ['LB', 'RB'],
}
WEIGHTS = {
    'Stopper': {'Aerial duels': 2.0, 'Interceptions': 1.0, 'Loss index': 1.0},
    'Ball-Playing Centerback': {'Passes': 2.0, 'Progressive Passes': 1.0, 'Loss index': 0.5},
    'Full Back': {'Passes': 1.0, 'Interceptions': 1.0},
}
INVERT = {'Loss index'}
METRICS = ['Aerial duels', 'Interceptions', 'Loss index', 'Passes', 'Progressive Passes']


def _pipeline(frame, min_minutes):
    """The pipeline's own arithmetic, pass for pass."""
    data = frame.copy()
    floor = tp.minutes_floor(data, min_minutes)
    qual = data['totalMinutes'] >= floor
    for role, group in POSITION_GROUPS.items():
        idx = data.index[data['primaryPosition'].isin(group)]
        q_idx = idx[qual.reindex(idx).values]
        for m in WEIGHTS[role]:
            if q_idx.empty:
                continue
            p = data.loc[q_idx, m].rank(pct=True)
            if m in INVERT:
                p = 1 - p.fillna(0.5)
            data.loc[q_idx, m + '_percentile'] = p
    for role, group in POSITION_GROUPS.items():
        idx = data.index[data['primaryPosition'].isin(group)]
        total = pd.Series(0.0, index=idx)
        for m, w in WEIGHTS[role].items():
            col = m + '_percentile'
            if col in data.columns:
                total = total.add(data.loc[idx, col].fillna(0) * w, fill_value=0)
        data.loc[idx, role + '_TotalScore'] = total
        lo, hi = total.min(), total.max()
        data.loc[idx, role + '_Score'] = ((total - lo) / (hi - lo) * 100) if hi != lo else 0.0
    return data.fillna(0)


def _frame(seed=0, n=40, max_minutes=900):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({m: rng.integers(0, 12, n).astype(float) for m in METRICS})
    df['playerId'] = np.arange(n)
    df['playerName'] = [f'P{i}' for i in range(n)]
    df['teamName'] = 'T'
    df['primaryPosition'] = rng.choice(['CB', 'RCB', 'LB', 'RB'], n)
    df['totalMinutes'] = rng.integers(30, max_minutes + 1, n).astype(float)
    df.loc[0, 'totalMinutes'] = max_minutes
    return df


# ---------------------------------------------------------------- floor ---

def test_floor_is_min_minutes_mid_season():
    df = pd.DataFrame({'totalMinutes': [40, 700, 1200]})
    assert tp.minutes_floor(df) == 500
    assert tp.minutes_floor(df, 300) == 300


def test_floor_clamps_to_half_the_max_early_season():
    # 2026/27 at matchweek 4: max 395' -> floor 197' (what left 28% of the
    # >= 90' players without percentiles in the bulk export).
    df = pd.DataFrame({'totalMinutes': [90, 164, 395]})
    assert tp.minutes_floor(df) == 197
    assert tp.minutes_floor(pd.DataFrame({'totalMinutes': [1.0]})) == 1
    assert tp.minutes_floor(pd.DataFrame({'x': [1]})) == 500
    assert tp.minutes_floor(None) == 500


def test_floor_rises_smoothly_past_min_minutes():
    # 2026/27 after MW6: leader 592'. The old clamp switched off at 500'
    # and jumped to a 500' floor (88 of 448 ranked); now half the leader.
    assert tp.minutes_floor(pd.DataFrame({'totalMinutes': [90, 400, 592]})) == 296
    assert tp.minutes_floor(pd.DataFrame({'totalMinutes': [90, 999]})) == 499
    assert tp.minutes_floor(pd.DataFrame({'totalMinutes': [90, 1000]})) == 500


# ---------------------------------------------------------------- parity --

@pytest.mark.parametrize('max_minutes', [900, 395])
def test_sub_floor_scoring_matches_the_pipeline(max_minutes):
    """Each qualifying player, dropped under the floor (row zeroed like the
    pipeline leaves sub-floor rows) and re-scored, gets the pipeline's own
    percentiles, totals and scores back — for every role their position
    belongs to, mid-season (500' floor) and early season (clamped)."""
    full = _pipeline(_frame(max_minutes=max_minutes), 500)
    floor = tp.minutes_floor(full)
    assert floor == min(500, max_minutes // 2)
    pct_cols = [c for c in full.columns if c.endswith('_percentile')]
    tot_cols = [c for c in full.columns if c.endswith('_TotalScore')]
    sc_cols = [c for c in full.columns if c.endswith('_Score')]
    qualifying = full.index[full['totalMinutes'] >= floor]
    assert len(qualifying) > 5
    for i in qualifying:
        mod = full.copy()
        mod.loc[i, pct_cols + tot_cols + sc_cols] = 0.0
        mod.loc[i, 'totalMinutes'] = floor - 1
        out = tp.score_below_floor(mod.loc[[i]], mod, POSITION_GROUPS, WEIGHTS, INVERT, floor)
        roles = [r for r, g in POSITION_GROUPS.items() if full.loc[i, 'primaryPosition'] in g]
        cols = ([m + '_percentile' for r in roles for m in WEIGHTS[r]]
                + [r + '_TotalScore' for r in roles] + [r + '_Score' for r in roles])
        np.testing.assert_allclose(out.loc[i, cols].astype(float).values,
                                   full.loc[i, cols].astype(float).values, atol=1e-12)


def test_rows_at_or_above_the_floor_are_untouched_and_others_never_move():
    full = _pipeline(_frame(seed=3), 500)
    floor = tp.minutes_floor(full)
    out = tp.score_below_floor(full, full, POSITION_GROUPS, WEIGHTS, INVERT, floor)
    above = full['totalMinutes'] >= floor
    pd.testing.assert_frame_equal(out[above], full[above])
    # the sample frame passed in is never modified
    pd.testing.assert_frame_equal(full, _pipeline(_frame(seed=3), 500))


def test_sub_floor_rows_get_real_numbers_not_zeros():
    full = _pipeline(_frame(seed=5), 500)
    floor = tp.minutes_floor(full)
    below = full.index[(full['totalMinutes'] < floor) & full['primaryPosition'].isin(['CB', 'RCB'])]
    assert len(below) > 0
    assert (full.loc[below, 'Stopper_Score'] == 0).all()          # the bug
    out = tp.score_below_floor(full.loc[below], full, POSITION_GROUPS, WEIGHTS, INVERT, floor)
    pct = out.loc[below, [m + '_percentile' for m in WEIGHTS['Stopper']]].astype(float)
    assert ((pct >= 0) & (pct <= 1)).all().all()
    assert (pct.sum(axis=1) > 0).all()   # an inverted metric CAN be 0; a whole row cannot
    assert out.loc[below, 'Stopper_Score'].between(0, 100).all()
    # a player better than everyone in the sample tops the range
    top = below[0]
    boosted = full.copy()
    boosted.loc[top, ['Aerial duels', 'Interceptions']] = 1e6
    boosted.loc[top, 'Loss index'] = -1e6
    out = tp.score_below_floor(boosted.loc[[top]], boosted, POSITION_GROUPS, WEIGHTS, INVERT, floor)
    assert out.loc[top, 'Stopper_Score'] == pytest.approx(100.0)


def test_percent_rank_matches_pandas_rank_with_ties():
    sample = np.array([1.0, 2.0, 2.0, 5.0, 7.0])
    for v in (0.0, 1.0, 2.0, 3.0, 7.0, 9.0):
        expected = pd.Series(np.append(sample, v)).rank(pct=True).iloc[-1]
        assert tp._percent_rank_in_sample(sample, v) == pytest.approx(expected)


# ------------------------------------------------- one-sample comparison ---

def test_rank_in_template_reproduces_the_pipeline_for_qualifying_group_players():
    """A qualifying player of the template's group gets the pipeline's own
    percentiles and score back — nothing is placed."""
    full = _pipeline(_frame(seed=7), 500)
    floor = tp.minutes_floor(full)
    rows = full[(full['totalMinutes'] >= floor) & full['primaryPosition'].isin(['LB', 'RB'])]
    assert len(rows) > 3
    out, placed = tp.rank_in_template(rows, full, 'Full Back', POSITION_GROUPS, WEIGHTS,
                                      INVERT, floor)
    assert placed == []
    cols = [m + '_percentile' for m in WEIGHTS['Full Back']] + ['Full Back_Score']
    np.testing.assert_allclose(out[cols].astype(float).values,
                               rows[cols].astype(float).values, atol=1e-12)


def test_rank_in_template_places_sub_floor_players_like_score_below_floor():
    full = _pipeline(_frame(seed=8), 500)
    floor = tp.minutes_floor(full)
    below = full[(full['totalMinutes'] < floor) & full['primaryPosition'].isin(['LB', 'RB'])]
    assert len(below) > 0
    out, placed = tp.rank_in_template(below, full, 'Full Back', POSITION_GROUPS, WEIGHTS,
                                      INVERT, floor)
    ref = tp.score_below_floor(below, full, POSITION_GROUPS, WEIGHTS, INVERT, floor)
    assert sorted(placed) == sorted(below['playerId'].tolist())
    cols = [m + '_percentile' for m in WEIGHTS['Full Back']] + ['Full Back_Score']
    np.testing.assert_allclose(out[cols].astype(float).values,
                               ref[cols].astype(float).values, atol=1e-12)


def test_rank_in_template_ranks_an_out_of_group_player_against_the_template_sample():
    """A centre-back on the Full Back template: ranked against the full-back
    sample (not his own), with a placed score on the full backs' range."""
    full = _pipeline(_frame(seed=9), 500)
    floor = tp.minutes_floor(full)
    cb = full[(full['totalMinutes'] >= floor) & full['primaryPosition'].isin(['CB', 'RCB'])].iloc[[0]]
    out, placed = tp.rank_in_template(cb, full, 'Full Back', POSITION_GROUPS, WEIGHTS,
                                      INVERT, floor)
    assert placed == [int(cb['playerId'].iloc[0])]
    fb = full[(full['totalMinutes'] >= floor) & full['primaryPosition'].isin(['LB', 'RB'])]
    v = float(cb['Passes'].iloc[0])
    expected = pd.Series(np.append(fb['Passes'].to_numpy(), v)).rank(pct=True).iloc[-1]
    assert out['Passes_percentile'].iloc[0] == pytest.approx(expected)
    assert 0.0 <= out['Full Back_Score'].iloc[0] <= 100.0
    # extra metrics (e.g. DefR axes) are ranked for everyone, scores untouched
    out2, _ = tp.rank_in_template(cb, full, 'Full Back', POSITION_GROUPS, WEIGHTS, INVERT,
                                  floor, extra_metrics=['Aerial duels'])
    assert 'Aerial duels_percentile' in out2.columns
    assert out2['Full Back_Score'].iloc[0] == pytest.approx(out['Full Back_Score'].iloc[0])
    pd.testing.assert_frame_equal(full, _pipeline(_frame(seed=9), 500))
