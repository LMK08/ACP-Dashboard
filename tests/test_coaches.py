"""Coach tables: no look-ahead in the ratings or the player baselines, and
the arithmetic of expected points, spells and the 'Coached by' shares."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.coaches import build_coaches as bc  # noqa: E402
from models.coaches import coach_metrics as cm  # noqa: E402


def _league(n_rounds=12, seed=0):
    """Six teams, a round a week; team 1 changes coach after round 6."""
    rng = np.random.default_rng(seed)
    rows, mid = [], 100
    teams = [1, 2, 3, 4, 5, 6]
    for rnd in range(n_rounds):
        order = list(rng.permutation(teams))
        for h, a in zip(order[:3], order[3:]):
            mid += 1
            xgd = float(rng.normal(0.2, 1.0))
            for t, o, side, sign in ((h, a, 'home', 1), (a, h, 'away', -1)):
                rows.append({'matchId': mid, 'teamId': int(t), 'oppId': int(o), 'side': side,
                             'competitionId': 1, 'seasonId': 1,
                             'date': pd.Timestamp('2025-01-01') + pd.Timedelta(days=7 * rnd),
                             'xgd': sign * xgd,
                             'coachId': 10 * int(t) + (1 if (t == 1 and rnd >= 6) else 0)})
    return pd.DataFrame(rows)


PRE = ['r_pre', 'r_opp_pre', 'hfa_pre', 'n_rated_pre']


def test_ratings_have_no_look_ahead():
    """A row's pre-match numbers must not move when later matches are removed
    or changed."""
    tm = _league()
    full = bc.add_ratings(tm).set_index(['matchId', 'teamId'])[PRE]
    cut = tm.date.sort_values().unique()[7]
    early = bc.add_ratings(tm[tm.date <= cut]).set_index(['matchId', 'teamId'])[PRE]
    pd.testing.assert_frame_equal(full.loc[early.index], early)
    # and changing the result of a match leaves its own pre-match row alone
    bumped = tm.copy()
    last = bumped.date == bumped.date.max()
    bumped.loc[last, 'xgd'] *= 5
    again = bc.add_ratings(bumped).set_index(['matchId', 'teamId'])[PRE]
    pd.testing.assert_frame_equal(full, again)


def test_first_match_starts_level():
    tm = bc.add_ratings(_league())
    first = tm[tm.date == tm.date.min()]
    assert (first.r_pre == 0).all() and (first.hfa_pre == 0).all() and (first.n_rated_pre == 0).all()


def test_spells_and_inherited_level():
    tm = bc.add_spells(bc.add_ratings(_league()))
    t1 = tm[tm.teamId == 1].sort_values('date')
    assert t1.spell.nunique() == 2
    second = t1[t1.coachId == 11]
    # 6 earlier matches < MIN_BASELINE_MATCHES → no baseline
    assert second.r_inherited.isna().all()
    tm = bc.add_spells(bc.add_ratings(_league(n_rounds=30)))
    t1 = tm[tm.teamId == 1].sort_values('date')
    late = t1.copy()
    late.loc[late.index[:15], 'coachId'] = 10
    late.loc[late.index[15:], 'coachId'] = 11
    other = tm[tm.teamId != 1]
    sp = bc.add_spells(pd.concat([other, late]).drop(columns=['spell', 'r_inherited']))
    new = sp[(sp.teamId == 1) & (sp.coachId == 11)].sort_values('date')
    assert new.r_inherited.notna().all()
    assert new.r_inherited.nunique() == 1 and new.r_inherited.iloc[0] == new.r_pre.iloc[0]


def test_expected_points():
    assert bc.expected_points([], []) == pytest.approx(1.0)
    assert bc.expected_points([0.99, 0.99], []) == pytest.approx(3.0, abs=1e-3)
    a, b = [0.3, 0.1, 0.45], [0.2, 0.6]
    pa, pb = bc.expected_points(a, b), bc.expected_points(b, a)
    # the two sides share 3 points for a decisive match and 2 for a draw
    assert 2.0 < pa + pb < 3.0
    assert bc.expected_points(a, a) == pytest.approx(bc.expected_points(a, a))


def test_coached_by_shares():
    cp = pd.DataFrame([{'coachId': 7, 'teamId': 1, 'playerId': 5, 'team': 'A',
                        'first_date': pd.Timestamp('2025-01-01'), 'last_date': pd.Timestamp('2025-03-01'),
                        'matches': 8, 'starts': 6, 'minutes': 540.0, 'obv_minutes': 540.0, 'obv': 0.6, 'points': 12,
                        'prior_minutes': 900.0, 'prior_obv': 0.5, 'team_matches_in_span': 10,
                        'named': 9, 'named_played': 8, 'team_minutes_in_span': 950.0}])
    r = cm.coached_by(cp, None, 5).iloc[0]
    assert r.start_share == pytest.approx(0.6) and r.played_share == pytest.approx(0.8)
    assert r.minutes_share == pytest.approx(540 / 950)
    # never above 100 %, and blank without a denominator
    assert cm.minutes_share(cp.assign(team_minutes_in_span=500.0)).iloc[0] == 1.0
    assert cm.minutes_share(cp.assign(team_minutes_in_span=0.0)).isna().all()
    assert r.obv90 == pytest.approx(0.1) and r.team_ppg_with == pytest.approx(1.5)
    dev, summ = cm.player_development(cp, 7)
    assert summ['n'] == 1 and dev.change.iloc[0] == pytest.approx(0.1 - 0.05)


BUILT = os.path.join(os.path.dirname(os.path.abspath(bc.__file__)), 'coach_team_match.parquet')


@pytest.mark.skipif(not os.path.exists(BUILT), reason='coach tables not built')
def test_built_table_is_consistent():
    tm = pd.read_parquet(BUILT)
    assert not tm.duplicated(['matchId', 'teamId']).any()
    ev = tm[tm.has_events]
    # the two sides of a match mirror each other
    both = ev.groupby('matchId').filter(lambda g: len(g) == 2)
    assert both.groupby('matchId').xgd.sum().abs().max() < 1e-9
    assert (both.groupby('matchId').xpts.sum().between(2.0, 3.0)).all()
    # the shipped pre-match columns are reproducible walk-forward from a prefix
    cut = tm.date.quantile(0.5)
    cols = ['matchId', 'teamId', 'competitionId', 'side', 'date', 'xgd']
    early = bc.add_ratings(tm.loc[tm.date <= cut, cols]).set_index(['matchId', 'teamId'])[PRE]
    shipped = tm.set_index(['matchId', 'teamId'])[PRE].loc[early.index]
    pd.testing.assert_frame_equal(shipped, early, check_exact=False, atol=1e-12)


def test_unknown_coach_does_not_split_a_spell():
    tm = bc.add_ratings(_league())
    t1 = tm[tm.teamId == 1].sort_values('date')
    tm.loc[t1.index[2], 'coachId'] = 0          # A, A, ?, A …
    sp = bc.add_spells(tm)
    first = sp[(sp.teamId == 1) & (sp.coachId == 10)]
    assert first.spell.nunique() == 1
    assert (sp.loc[sp.coachId == 0, 'spell'] == -1).all()


def _styled_league(n_rounds=140, seed=1):
    tm = bc.add_ratings(_league(n_rounds=n_rounds, seed=seed))
    rng = np.random.default_rng(seed)
    n = len(tm)
    tm['has_events'] = True
    for c in ('passes', 'opp_passes', 'passes_f3', 'opp_passes_f3', 'possessions', 'long_passes',
              'fwd_passes', 'def_actions', 'opp_press_passes', 'press_def'):
        tm[c] = rng.integers(50, 400, n).astype(float)
    tm['def_x_sum'] = tm.def_actions * rng.uniform(30, 40, n)
    return tm


def test_context_adjustment_has_no_look_ahead():
    """A match's z-scores must not move when later months are removed."""
    tm = _styled_league()
    full = bc.add_context_adjusted(tm)
    zc = ['z_' + m for m in bc.Z_METRICS]
    assert full[zc].notna().any().all(), 'the synthetic league never reached MIN_CONTEXT_ROWS'
    cut = tm.date.sort_values().iloc[int(len(tm) * 0.8)]
    early = bc.add_context_adjusted(tm[tm.date <= cut])
    a = full.set_index(['matchId', 'teamId']).loc[early.set_index(['matchId', 'teamId']).index, zc]
    pd.testing.assert_frame_equal(a, early.set_index(['matchId', 'teamId'])[zc])
    # the first month can never be adjusted
    first_month = full.date.dt.to_period('M') == full.date.dt.to_period('M').min()
    assert full.loc[first_month, zc].isna().all().all()


def test_flexibility_and_trust_arithmetic():
    rows = pd.DataFrame({
        'spell': [1] * 4, 'date': pd.date_range('2025-01-01', periods=4, freq='7D'),
        'formation': ['4-3-3', '4-3-3', '3-5-2', '4-3-3'],
        'u21_minutes': [99.0, 0, 99, 0], 'squad_minutes': [990.0] * 4, 'u21_named': [4, 4, 4, 4],
        'sheet_n': [20] * 4, 'u21_starts': [1, 0, 1, 0], 'xi_n': [11] * 4,
        'age_minutes_sum': [990 * 26.0] * 4, 'sheet_age_sum': [20 * 25.0] * 4})
    f = cm.flexibility(rows)
    assert f['formation_change_rate'] == pytest.approx(2 / 3)
    assert f['back_line_change_rate'] == pytest.approx(2 / 3)
    # 3 × 4-3-3 and 1 × 3-5-2: two matches share a shape in 3 of 6 pairs
    assert f['eff_formations'] == pytest.approx(2.0)
    y = cm.youth_trust(rows, 21)
    assert y['u21_minutes_share'] == pytest.approx(0.05) and y['u21_named_share'] == pytest.approx(0.2)
    assert y['u21_trust'] == pytest.approx(0.25)
    assert y['age_on_pitch'] == pytest.approx(26.0) and y['age_of_squad'] == pytest.approx(25.0)


def test_expected_for_age_uses_only_finished_spells():
    """Removing or changing spells that end later must not move an earlier
    pair's expected-for-age change."""
    rng = np.random.default_rng(3)
    n = 120
    start = pd.Timestamp('2022-01-01') + pd.to_timedelta(np.sort(rng.integers(0, 1200, n)), unit='D')
    cp = pd.DataFrame({'coachId': np.arange(n) % 7 + 1, 'playerId': np.arange(n) + 1, 'team': 'T',
                       'obv_minutes': 900.0, 'obv': rng.normal(0.5, 0.3, n), 'prior_minutes': 900.0,
                       'prior_obv': rng.normal(0.5, 0.3, n), 'age_first': 24.0,
                       'first_under_coach': start, 'last_date': start + pd.Timedelta(days=150)})
    full = cm.development_pairs(cp).set_index('playerId')
    assert full.expected.notna().sum() > 20
    cut = start[80]
    early = cm.development_pairs(cp[cp.last_date < cut]).set_index('playerId')
    pd.testing.assert_series_equal(full.loc[early.index, 'expected'], early.expected)
    assert full.sort_values('first_date').expected.iloc[:cm.MIN_PEERS].isna().all()
    # a coach's own earlier players are never his players' peers
    assert cm.development_pairs(cp.assign(coachId=1)).expected.isna().all()


def _state_league(n_rounds=140, seed=2):
    tm = bc.add_ratings(_league(n_rounds=n_rounds, seed=seed))
    rng = np.random.default_rng(seed)
    n = len(tm)
    cut = np.sort(rng.uniform(0, 90, (n, 2)), axis=1)
    tm['min_lead'], tm['min_draw'], tm['min_trail'] = cut[:, 0], cut[:, 1] - cut[:, 0], 90 - cut[:, 1]
    for s in bc.STATES:
        tm[f'xgd_{s}'] = rng.normal(0, 0.3, n)
    tm['state_ok'] = rng.uniform(size=n) > 0.1
    return tm


EXP = [f'exp_{k}_{s}' for k in ('min', 'xgd') for s in bc.STATES]


def test_state_expected_has_no_look_ahead():
    """A match's expected game-state numbers must not move when later months
    are removed or their results changed."""
    tm = _state_league()
    full = bc.add_state_expected(tm).set_index(['matchId', 'teamId'])
    assert full[EXP].notna().any().all(), 'the synthetic league never reached MIN_CONTEXT_ROWS'
    cut = tm.date.sort_values().iloc[int(len(tm) * 0.8)]
    early = bc.add_state_expected(tm[tm.date <= cut]).set_index(['matchId', 'teamId'])
    pd.testing.assert_frame_equal(full.loc[early.index, EXP], early[EXP])
    # a match's own result is not in its own expected minutes
    bumped = tm.copy()
    last = bumped.date.dt.to_period('M') == bumped.date.dt.to_period('M').max()
    bumped.loc[last, 'min_lead'], bumped.loc[last, 'min_trail'] = bumped.loc[last, 'min_trail'], bumped.loc[last, 'min_lead']
    again = bc.add_state_expected(bumped).set_index(['matchId', 'teamId'])
    cols = [f'exp_min_{s}' for s in bc.STATES]
    pd.testing.assert_frame_equal(full[cols], again[cols])
    # expected minutes add up to the match; rows without a clean timeline get none
    have = full[full.exp_min_lead.notna()]
    assert np.allclose(have[cols].sum(axis=1), have[['min_lead', 'min_draw', 'min_trail']].sum(axis=1))
    assert full.loc[~full.state_ok, EXP].isna().all().all()
    first_month = full.date.dt.to_period('M') == full.date.dt.to_period('M').min()
    assert full.loc[first_month, EXP].isna().all().all()


@pytest.mark.skipif(not os.path.exists(BUILT), reason='coach tables not built')
def test_state_vs_expected_arithmetic():
    tm = pd.read_parquet(BUILT)
    tm = tm[tm.coachId == tm[tm.coachId > 0].coachId.mode().iloc[0]].copy()
    tm['state_ok'] = True
    for s, m, x, em, ex in (('lead', 30.0, 0.3, 20.0, 0.1), ('draw', 45.0, 0.0, 50.0, 0.0), ('trail', 15.0, 0.2, 20.0, 0.3)):
        tm[f'min_{s}'], tm[f'xgd_{s}'], tm[f'exp_min_{s}'], tm[f'exp_xgd_{s}'] = m, x, em, ex
    s = cm.summarise(tm)
    assert s['exp_min_share_lead'] == pytest.approx(20 / 90) and s['min_share_vs_exp_lead'] == pytest.approx(10 / 90)
    assert s['exp_xgd90_lead'] == pytest.approx(0.1 * 90 / 30) and s['xgd90_vs_exp_lead'] == pytest.approx(0.2 * 90 / 30)
    assert s['xgd90_vs_exp_trail'] == pytest.approx(-0.1 * 90 / 15)


def test_league_comparison():
    table = pd.DataFrame({'coachId': range(1, 8), 'qualifies': [True] * 6 + [False],
                          'ppg': [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 9.0]})
    c = cm.league_comparison(table, {'ppg': 1.6, 'other': 3.0}, ['ppg', 'other', 'missing'])
    assert c[0]['n'] == 6 and c[0]['median'] == pytest.approx(1.5)
    assert c[0]['pct'] == pytest.approx((3 + 0.5) / 6 * 100)      # the non-qualifier is not in the pool
    assert pd.isna(c[1]['pct']) and pd.isna(c[2]['pct']) and c[1]['n'] == 0
    # too few coaches: no percentile
    assert pd.isna(cm.league_comparison(table.head(3), {'ppg': 1.2}, ['ppg'])[0]['pct'])


# ───────────────────────── our own formations ─────────────────────────
from models.coaches import formations as fm  # noqa: E402


def _side(shape, rng, team=1, match=1, n=30, sub_at=None, noise=4.0):
    """Touch rows for one side laid out in `shape`; an optional substitute
    appears at `sub_at` minutes and stands somewhere misleading."""
    rows = [{'playerId': 100 * team, 'x': 8.0, 'y': 50.0}]
    for i, (d, w) in enumerate(fm.TEMPLATES[shape]):
        rows.append({'playerId': 100 * team + i + 1, 'x': 100 * d * 0.8, 'y': 50 + 40 * w})
    out = []
    for r in rows:
        for k in range(n):
            out.append({'matchId': match, 'teamId': team, 'playerId': r['playerId'],
                        't': 1 + k * (60 / n) + rng.uniform(0, 1),
                        'x': r['x'] + rng.normal(0, noise), 'y': r['y'] + rng.normal(0, noise),
                        'restart': False, 'received': False})
    if sub_at is not None:
        for k in range(n):
            out.append({'matchId': match, 'teamId': team, 'playerId': 100 * team + 50,
                        't': sub_at + k, 'x': 95.0, 'y': 5.0,
                        'restart': False, 'received': False})
    return pd.DataFrame(out)


def test_formation_recovers_each_reference_shape():
    rng = np.random.default_rng(5)
    for shape in fm.TEMPLATES:
        got = fm.team_match_shapes(_side(shape, rng)).iloc[0]
        assert got.formation == shape, (shape, got.formation)


def test_formation_reads_only_the_starting_window():
    """A substitute's touches, and the starters' touches after he came on,
    must not move the shape."""
    rng = np.random.default_rng(6)
    g = _side('4-4-2', rng, sub_at=40.0)
    # after the change the shape itself changes: the strikers drop onto the
    # centre-backs and the wide midfielders come inside
    late = g.t >= 40
    for p_ in (109, 110):
        g.loc[late & (g.playerId == p_), 'x'] = 0.0
    for p_ in (107, 108):
        g.loc[late & (g.playerId == p_), ['x', 'y']] = [20.0, 50.0]
    g = pd.concat([g] + [g[g.t >= 40]] * 5, ignore_index=True)          # most touches come after it
    whole = g[g.playerId != 150].assign(t=lambda d: d.t.where(d.t < 40, 30.0))   # same touches, no change
    assert fm.team_match_shapes(whole).iloc[0].formation != '4-4-2'      # the swap does change the read
    got = fm.team_match_shapes(g).iloc[0]
    assert got.formation == '4-4-2' and got.formation_window_min == pytest.approx(40.0)
    # a red card ends the window too
    red = fm.team_match_shapes(_side('3-5-2', rng), {(1, 1): 10.0}).iloc[0]
    assert red.formation is None                               # under MIN_WINDOW_MIN


def test_formation_reads_no_provider_label():
    banned = ('formation', 'position')
    assert not [c for c in fm.EVENT_COLS if any(b in c for b in banned)]
    assert not [c for c in bc.EVENT_COLS if any(b in c for b in banned)]
    src = open(fm.__file__).read() + open(bc.__file__).read() + open(cm.__file__).read()
    assert "team.formation" not in src and "player.position" not in src


def test_touches_from_events_and_restarts():
    ev = pd.DataFrame({
        'matchId': [1, 1, 1], 'matchPeriod': ['1H', '2H', '1H'], 'minute': [10, 50, 12], 'second': [0, 30, 0],
        'type.primary': ['pass', 'pass', 'throw_in'], 'team.id': [7, 7, 7], 'player.id': [1, 2, 3],
        'location.x': [30.0, 40.0, 100.0], 'location.y': [50.0, 20.0, 0.0],
        'pass.accurate': [True, False, True], 'pass.recipient.id': [2, 3, 4],
        'pass.endLocation.x': [60.0, 70.0, 90.0], 'pass.endLocation.y': [55.0, 25.0, 50.0],
        'infraction.redCard': [None, None, None]})
    t = fm.touches(ev)
    # 3 own actions + 1 reception (the inaccurate pass and the throw-in give none)
    assert len(t) == 4 and t.received.sum() == 1
    assert t[t.received].iloc[0].playerId == 2 and t[t.received].iloc[0].x == 60.0
    assert t.loc[t.playerId == 3, 'restart'].all()
    assert t.loc[(t.playerId == 2) & ~t.received, 't'].iloc[0] == pytest.approx(50.5 + fm.HALF_GAP)


# ───────────────────── line-ups from our own formations ─────────────────────
def _own_tables(monkeypatch, shapes_by_match):
    """Fake coach tables: one side (team 1) laid out in the given shape per match."""
    import own_formations as of
    form, by_match = {}, {}
    for mid, (shape, ids) in shapes_by_match.items():
        rows = [{'matchId': mid, 'teamId': 1, 'playerId': ids[0], 'x': 8.0, 'y': 50.0, 'n': 20, 'gk': True}]
        for pid, (d, w) in zip(ids[1:], fm.TEMPLATES[shape]):
            rows.append({'matchId': mid, 'teamId': 1, 'playerId': pid, 'x': 80 * d, 'y': 50 + 40 * w, 'n': 20, 'gk': False})
        form[(mid, 1)] = shape
        by_match[(mid, 1)] = pd.DataFrame(rows)
    monkeypatch.setattr(of, 'tables', lambda: (form, by_match))
    return of


def test_line_up_votes_and_slots(monkeypatch):
    a = list(range(100, 111))
    b = [100] + list(range(201, 211))               # same keeper, ten other outfielders
    of = _own_tables(monkeypatch, {1: ('4-4-2', a), 2: ('3-5-2', b), 3: ('4-4-2', a)})
    names = {p: f'P{p}' for p in a + b}
    f, xi = of.line_up(1, {1: 1.0, 2: 1.0, 3: 1.0}, names)
    assert f == '4-4-2' and list(xi) == ['GK'] + fm.SLOTS['4-4-2']
    assert xi['GK']['id'] == 100
    # the two matches in that shape put each of their players in his own slot
    assert [xi[s]['id'] for s in fm.SLOTS['4-4-2']] == a[1:]
    # recency weights: 3 for the 3-5-2 match beats 1 + 1; a 3-3 tie goes to the heaviest match
    assert of.line_up(1, {1: 1.0, 2: 3.0, 3: 1.0}, names)[0] == '3-5-2'
    assert of.line_up(1, {1: 1.0, 2: 2.0, 3: 1.0}, names)[0] == '3-5-2'
    # one player per slot, excluded names never appear, unread matches are ignored
    f, xi = of.line_up(1, {1: 3.0, 2: 2.0, 9: 5.0}, names, exclude={'P101'})
    ids = [v['id'] for v in xi.values()]
    assert f == '4-4-2' and len(ids) == len(set(ids)) and 101 not in ids
    assert of.line_up(1, {9: 1.0}, names) == (None, {})
    assert of.line_up(None, {1: 1.0}, names) == (None, {})


def test_assign_slots_sides_and_any_shape():
    """Low y is the left side; a side can be fitted to a shape it was not read as."""
    ids = np.arange(1, 11)
    xy = np.array([(80 * d, 50 + 40 * w) for d, w in fm.TEMPLATES['4-3-3']])
    s = fm.assign_slots(ids, xy, '4-3-3')
    assert s == dict(zip(fm.SLOTS['4-3-3'], ids.tolist()))
    assert xy[s['LB'] - 1][1] < 50 < xy[s['RB'] - 1][1]
    other = fm.assign_slots(ids, xy, '3-5-2')
    assert sorted(other) == sorted(fm.SLOTS['3-5-2']) and sorted(other.values()) == ids.tolist()
    for shape, slots in fm.SLOTS.items():
        assert len(slots) == len(fm.TEMPLATES[shape]) == 10 and len(set(slots)) == 10


def test_line_up_merges_aliased_ids(monkeypatch):
    """A player whose events sit under two ids votes as one, under the
    canonical id the app's names use."""
    import own_formations as of
    a = list(range(100, 111))
    b = a[:1] + [901] + a[2:]                      # match 2 has player 101 under his other id
    of = _own_tables(monkeypatch, {1: ('4-4-2', a), 2: ('4-4-2', b)})
    monkeypatch.setattr(of, 'PLAYER_ID_ALIASES', {901: 101})
    names = {p: f'P{p}' for p in a}                # the app only knows the canonical id
    f, xi = of.line_up(1, {1: 1.0, 2: 3.0}, names)
    assert [v['id'] for v in xi.values()] == a and len(xi) == 11
    assert of.read_matches(1, [5, 2, 1]) == [2, 1]


def test_slots_exist_in_both_drawing_tables():
    import re
    import pitch_visualizations as pv
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'app.py')).read()
    for shape, slots in fm.SLOTS.items():
        assert set(pv.FORMATION_COORDS[shape]['positions']) == {'GK', *slots}, shape
        m = re.search(r"    '%s': \{\s*'positions': \[([^\]]*)\]" % re.escape(shape), src)
        assert m, f'{shape} missing from app.FORMATION_COORDS'
        assert {p.strip().strip("'") for p in m.group(1).split(',')} == {'GK', *slots}, shape


PLACES_BUILT = os.path.join(os.path.dirname(os.path.abspath(bc.__file__)), 'team_match_places.parquet')


@pytest.mark.skipif(not (os.path.exists(BUILT) and os.path.exists(PLACES_BUILT)), reason='coach tables not built')
def test_line_up_from_the_built_tables():
    """The real parquet round trip: keys, the gk flag and a full eleven."""
    import own_formations as of
    form, by_match = of.tables()
    # (places also exist for a few matches the team-match table leaves out;
    # line_up only ever uses matches that have a formation)
    assert form and len(set(by_match) & set(form)) > 0.9 * len(form)
    (mid, tid), g = next((k, v) for k, v in by_match.items() if k in form)
    assert len(g) == 11 and int(g.gk.sum()) == 1
    names = {int(p): f'P{int(p)}' for p in g.playerId}
    names.update({of._canonical(p): f'P{int(p)}' for p in g.playerId})
    f, xi = of.line_up(tid, {mid: 1.0}, names)
    assert f == form[(mid, tid)] and len(xi) == 11 and len({v['id'] for v in xi.values()}) == 11
