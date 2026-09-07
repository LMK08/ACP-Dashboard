"""models/similarity/similar_players — pure checks on a synthetic pool plus
data-backed checks on the committed caches (skipped without them)."""
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

DASH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, DASH)

from models.similarity import similar_players as sp  # noqa: E402


def _frame(rows, season):
    """A minimal scored-stats frame: every feature present, values given
    per row as a dict subset (missing -> 1.0)."""
    feats = sp.STYLE_FEATURES + sp.QUALITY_FEATURES
    out = []
    for r in rows:
        base = {f: 1.0 for f in feats}
        base.update(r.get('vals', {}))
        if 'rating' in r:
            base['ACP Rating (abs)'] = r['rating']
        base.update({'playerId': r['pid'], 'playerName': r.get('name', f"P{r['pid']}"),
                     'teamName': r.get('team', 'Club'), 'primaryPosition': r.get('pos', 'CF'),
                     'totalMinutes': r.get('mins', 1200), 'competitionId': r.get('comp', 43324)})
        out.append(base)
    return pd.DataFrame(out)


def _pool():
    # two strikers who look alike (A, B), one very different striker (C), a
    # centre-back (D) and a low-minutes striker (E), across two seasons
    s1 = _frame([
        {'pid': 1, 'name': 'A', 'vals': {'Shots': 5, 'npxG': 0.6, 'Aerial duels': 8}},
        {'pid': 2, 'name': 'B', 'vals': {'Shots': 4.8, 'npxG': 0.55, 'Aerial duels': 7.5}, 'team': 'Other', 'comp': 702, 'rating': 40.0},
        {'pid': 3, 'name': 'C', 'vals': {'Shots': 0.5, 'npxG': 0.05, 'Aerial duels': 1, 'Passes': 60}},
        {'pid': 4, 'name': 'D', 'pos': 'CB', 'vals': {'Clearances': 9}},
        {'pid': 5, 'name': 'E', 'mins': 200, 'vals': {'Shots': 5}},
        {'pid': 6, 'name': 'F', 'vals': {'Shots': 3, 'npxG': 0.3}, 'rating': 55.0},
    ], 191782)
    s2 = _frame([
        {'pid': 1, 'name': 'A', 'vals': {'Shots': 5.2, 'npxG': 0.62, 'Aerial duels': 8.2}},
        {'pid': 3, 'name': 'C', 'vals': {'Shots': 0.4, 'npxG': 0.04, 'Aerial duels': 1.2, 'Passes': 58}},
        {'pid': 7, 'name': 'G', 'vals': {'Shots': 2, 'npxG': 0.2}},
    ], 190090)
    return sp.build_pool({191782: s1, 190090: s2}, role_features=None)


def test_module_is_streamlit_free():
    code = ("import sys; sys.path.insert(0, %r); import models.similarity.similar_players as m; "
            "assert 'streamlit' not in sys.modules; print(len(m.STYLE_FEATURES))" % DASH)
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-400:]
    assert int(out.stdout.strip()) == 36


def test_pool_buckets_filters_minutes_and_keys():
    pool = _pool()
    assert len(pool.meta) == 8           # E (200 min) dropped
    assert set(pool.rows) == {'ST', 'CB'}
    assert len(pool.rows['CB']) == 1 and len(pool.rows['ST']) == 7
    assert pool.index_of(1) == pool.index_of(1, 191782)          # latest season = 2025/26
    assert pool.index_of(1, 190090) is not None and pool.index_of(5) is None
    assert pool.X['ST'].shape == (7, len(pool.features))
    assert len(pool.features) == 44                               # no spatial columns given


def test_neighbours_rank_the_lookalike_first_and_never_the_query():
    pool = _pool()
    nb = sp.neighbours(pool, 1, season_id=191782, k=10)
    assert nb['playerName'].iloc[0] == 'B'
    assert 1 not in nb['playerId'].tolist()                        # never the query player (any season)
    assert 'D' not in nb['playerName'].tolist()                    # other bucket never appears
    assert nb['similarity'].between(0, 100).all()
    assert nb['distance'].is_monotonic_increasing
    assert nb['playerId'].is_unique                                # profile='latest' dedupes C's two seasons
    assert sp.neighbours(pool, 1, profile='any', k=10)['playerId'].tolist().count(3) == 2


def test_neighbours_filters_apply_after_ranking():
    pool = _pool()
    assert 'B' not in sp.neighbours(pool, 1, league=43324)['playerName'].tolist()   # B is Campeonato
    assert 'B' not in sp.neighbours(pool, 1, exclude_team='Other')['playerName'].tolist()
    ages = {2: 31.0, 3: 22.0}
    assert 'B' not in sp.neighbours(pool, 1, max_age=25, ages=ages)['playerName'].tolist()
    assert 'C' in sp.neighbours(pool, 1, max_age=25, ages=ages)['playerName'].tolist()   # unknown-age F/G pass too
    assert sp.neighbours(pool, 1, min_year=2025)['seasonId'].eq(191782).all()
    assert sp.neighbours(pool, 99).empty                            # unknown player
    assert 'B' not in sp.neighbours(pool, 1, exclude_player_ids=[2])['playerName'].tolist()


def test_distance_is_symmetric_and_self_zero():
    pool = _pool()
    X = pool.X['ST']
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    assert np.allclose(D, D.T) and np.allclose(np.diag(D), 0)


def test_explain_names_the_shared_and_differing_traits():
    pool = _pool()
    a, b = pool.index_of(1, 191782), pool.index_of(2, 191782)
    ex = sp.explain(pool, a, b)
    assert any('shot volume' in s or 'aerial volume' in s or 'npxG' in s for s in ex['alike'])
    assert all(('high, top' in s) or ('low, bottom' in s) for s in ex['alike'])   # value words, never praise
    c = pool.index_of(3, 191782)
    ex2 = sp.explain(pool, a, c)
    assert ex2['differs'] and any('shot volume' in s for s in ex2['differs'])
    d = pool.index_of(4, 191782)
    assert sp.explain(pool, a, d) == {'alike': [], 'differs': []}


def test_validation_reports_self_match_and_label_agreement():
    pool = _pool()
    labels = pd.DataFrame({'playerId': [1, 2, 3, 1, 3, 6, 7], 'seasonId': [191782, 191782, 191782, 190090, 190090, 191782, 190090],
                           'role': ['Striker', 'Striker', 'Deep', 'Striker', 'Deep', 'Striker', 'Striker'],
                           'style': ['Poacher'] * 7})
    v = sp.validation(pool, labels, k=2)
    assert v['n_rows'] == 8 and v['self_match']['n'] >= 1
    assert v['role_agreement']['top_k_share'] is not None
    assert 0 <= v['role_agreement']['top_k_share'] <= 1


def test_missing_feature_column_fails_loudly_even_in_one_season():
    f = _frame([{'pid': 1}], 191782).drop(columns=['Shots'])
    with pytest.raises(KeyError):
        sp.build_pool({191782: f})
    ok = _frame([{'pid': 2}], 190090)
    with pytest.raises(KeyError, match='191782'):
        sp.build_pool({191782: f, 190090: ok})


def test_latest_is_chosen_before_the_filters():
    """A player whose LATEST season is at our club must not resurface under
    an older season at a previous club when 'Exclude own club' is on."""
    s1 = _frame([{'pid': 1, 'name': 'A', 'vals': {'Shots': 5}},
                 {'pid': 2, 'name': 'B', 'team': 'Atlético CP', 'vals': {'Shots': 4.9}}], 191782)
    s2 = _frame([{'pid': 2, 'name': 'B', 'team': 'Old Club', 'vals': {'Shots': 4.9}},
                 {'pid': 3, 'name': 'C', 'vals': {'Shots': 2}}], 190090)
    pool = sp.build_pool({191782: s1, 190090: s2})
    latest = sp.neighbours(pool, 1, exclude_team='Atlético CP')
    assert 'B' not in latest['playerName'].tolist()
    anyseason = sp.neighbours(pool, 1, profile='any', exclude_team='Atlético CP')
    assert 'B' in anyseason['playerName'].tolist()            # explicit 'any season' still may


def test_unknown_competition_and_zero_ratings_are_normalised():
    f = _frame([{'pid': 1, 'rating': 0.0}, {'pid': 2, 'rating': 52.0}], 191782)
    f.loc[0, 'competitionId'] = 0
    pool = sp.build_pool({191782: f})
    assert pool.meta['competitionId'].tolist() == [43324, 43324]      # backfilled from the season
    assert pd.isna(pool.meta.loc[0, 'ACP Rating (abs)']) and pool.meta.loc[1, 'ACP Rating (abs)'] == 52.0


# --- data-backed ---------------------------------------------------------------
_CACHES = [os.path.join(DASH, 'stats_cache', f'player_percentiles_v14_{sid}.parquet')
           for sid in (190090, 191782)]
_HAS_DATA = all(os.path.exists(p) for p in _CACHES)


@pytest.mark.skipif(not _HAS_DATA, reason='per-season stats caches not present')
def test_real_caches_build_and_self_match_beats_chance():
    frames = {sid: pd.read_parquet(p) for sid, p in zip((190090, 191782), _CACHES)}
    rf_path = os.path.join(DASH, 'models', 'roles', 'role_features_season.parquet')
    rf = pd.read_parquet(rf_path) if os.path.exists(rf_path) else None
    pool = sp.build_pool(frames, rf)
    assert set(pool.rows) <= set(sp.BUCKETS) and len(pool.meta) > 500
    v = sp.validation(pool, None, k=10)
    # a player's other season should land in his own top-10 far above chance
    assert v['self_match']['top_k_rate'] > 5 * v['self_match']['chance_top_k']
    pid = int(pool.meta['playerId'].iloc[0])
    nb = sp.neighbours(pool, pid, k=5)
    assert len(nb) == 5 and nb['similarity'].notna().all()


def test_rating_filter_uses_the_absolute_rating_and_drops_unrated_rows():
    pool = _pool()
    assert 'ACP Rating (abs)' in pool.meta.columns
    at_least_50 = sp.neighbours(pool, 1, rating_abs_min=50)
    assert at_least_50['playerName'].tolist() == ['F']          # B (40) out, unrated rows out
    window = sp.neighbours(pool, 1, rating_abs_min=35, rating_abs_max=45)
    assert window['playerName'].tolist() == ['B']
    assert 'B' in sp.neighbours(pool, 1)['playerName'].tolist()  # no filter: everyone back


# --- UI bridge helpers (no Streamlit session needed) --------------------------------
def test_ui_bridge_helpers_carry_league_and_gate_compare():
    import similar_players_ui as ui
    rec = {'playerId': 7, 'seasonId': 191779, 'competitionId': 702}
    state = ui.profile_bridge_state(rec)
    assert state == {'selected_player_id': 7, 'nav_to_profile': True, 'nav_season_id': 191779,
                     'nav_has_season': True, 'nav_league': 'Campeonato'}
    assert ui.profile_bridge_state({'playerId': 7, 'seasonId': 191782, 'competitionId': float('nan')})['nav_league'] is None
    q = {'competitionId': 43324, 'seasonId': 191782}
    assert ui.can_compare(q, {'competitionId': 43324, 'seasonId': 191782})
    assert not ui.can_compare(q, {'competitionId': 702, 'seasonId': 191779})
    assert not ui.can_compare(q, {'competitionId': 43324, 'seasonId': 190090})
    assert not ui.can_compare(q, {'competitionId': None, 'seasonId': 191782})
