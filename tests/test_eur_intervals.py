"""models/value/eur_intervals — engine value maths, conformal quantiles,
interval helpers and the calibration build.

Pure-Python tests run everywhere; the data-backed ones skip without the
engine parquet / fee CSV (HF-only style checkout).
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

DASH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, DASH)

from models.value import eur_intervals as ei  # noqa: E402
from models.value.cvi import _cvi_age_value_multiplier, cvi_to_projected_eur  # noqa: E402

# A FIXTURE of 14 nonconformity scores for hand-computed checks (the first
# 2026-09-06 measurement, before sales were valued at the age at sale). It
# is deliberately NOT the shipped calibration's scores — see the data-backed
# test at the bottom for those.
SCORES_14 = [0.04, 0.06, 0.12, 0.19, 0.24, 0.30, 0.34, 0.57, 0.79, 0.97, 1.03, 1.44, 1.49, 1.63]


# --- conformal maths ---------------------------------------------------------
def test_conformal_k_matches_split_conformal_formula():
    assert ei.conformal_k(14, 0.5) == 8
    assert ei.conformal_k(14, 0.8) == 12
    assert ei.conformal_k(14, 0.9) == 14


def test_conformal_quantile_is_kth_order_statistic_with_shipping_gate():
    q50, k50, ship50 = ei.conformal_quantile(SCORES_14, 0.5)
    q80, k80, ship80 = ei.conformal_quantile(SCORES_14, 0.8)
    q90, k90, ship90 = ei.conformal_quantile(SCORES_14, 0.9)
    assert (q50, k50, ship50) == (0.57, 8, True)
    assert (q80, k80, ship80) == (1.44, 12, True)
    assert (q90, k90, ship90) == (1.63, 14, False)   # the maximum: not pinnable
    # 90% unlocks once k <= n - 2, i.e. at n = 29
    _, k, shipped = ei.conformal_quantile(list(np.linspace(0.1, 2.0, 29)), 0.9)
    assert (k, shipped) == (27, True)
    # unordered input and NaNs are tolerated
    q, _, _ = ei.conformal_quantile([1.44, float('nan'), 0.04, 0.57] + SCORES_14[1:7] + SCORES_14[8:], 0.5)
    assert q == 0.57
    assert ei.conformal_quantile([], 0.5) == (None, 1, False)


def test_loo_coverage_reproduces_measured_hits():
    assert ei.loo_coverage(SCORES_14, 0.5) == (7, 14)
    assert ei.loo_coverage(SCORES_14, 0.8) == (12, 14)


# --- interval helpers --------------------------------------------------------
CALIB = {'levels': {'0.50': {'q': 0.572, 'shipped': True},
                    '0.80': {'q': 1.436, 'shipped': True},
                    '0.90': {'q': 1.63, 'shipped': False}}}


def test_interval_is_log_symmetric_and_monotone_in_level():
    lo, hi = ei.projected_eur_interval(150_000, 0.5, CALIB)
    assert lo < 150_000 < hi
    assert hi / 150_000 == pytest.approx(150_000 / lo)
    assert hi / 150_000 == pytest.approx(math.exp(0.572))
    lo80, hi80 = ei.projected_eur_interval(150_000, 0.8, CALIB)
    assert lo80 < lo and hi80 > hi
    # scales linearly with the point
    lo2, hi2 = ei.projected_eur_interval(300_000, 0.5, CALIB)
    assert (lo2, hi2) == pytest.approx((2 * lo, 2 * hi))


def test_interval_degrades_to_none():
    assert ei.projected_eur_interval(150_000, 0.9, CALIB) == (None, None)   # not shipped
    assert ei.projected_eur_interval(None, 0.5, CALIB) == (None, None)
    assert ei.projected_eur_interval(0, 0.5, CALIB) == (None, None)
    assert ei.projected_eur_interval(float('nan'), 0.5, CALIB) == (None, None)
    assert ei.projected_eur_interval(150_000, 0.5, {}) == (None, None)      # no artifact
    assert ei.load_calibration('/nonexistent/eur.json') == {}


def test_format_eur_short_two_significant_figures():
    assert ei.format_eur_short(850) == '€850'
    assert ei.format_eur_short(2_650) == '€2.7k'
    assert ei.format_eur_short(85_000) == '€85k'
    assert ei.format_eur_short(182_345) == '€180k'
    assert ei.format_eur_short(1_234_567) == '€1.2M'
    assert ei.format_eur_short(12_345_678) == '€12M'
    assert ei.format_eur_short(None) == '—'
    assert ei.format_eur_range(85_000, 265_000) == '€85k – €270k'
    assert ei.format_eur_range(None, 1) is None


# --- engine value ------------------------------------------------------------
def test_engine_perf_percentile_matches_closure_rule_with_ties():
    pool = np.sort(np.array([1.0, 2.0, 2.0, 3.0, 5.0]))
    # closure: (pool < pa).mean() + 0.5 * (pool == pa).mean()
    for pa in (0.5, 1.0, 2.0, 2.5, 5.0, 9.0):
        expect = ((pool < pa).mean() + 0.5 * (pool == pa).mean()) * 100.0
        assert ei.engine_perf_percentile(pa, pool) == pytest.approx(expect)
    assert ei.engine_perf_percentile(None, pool) is None
    assert ei.engine_perf_percentile(2.0, np.array([])) is None


def test_engine_value_eur_is_curve_times_temper():
    pool = np.sort(np.array([10.0, 20.0, 30.0, 40.0, 50.0]))
    v = ei.engine_value_eur(40.0, pool, 'Striker', 24.0, 191782)
    perf = 70.0  # 3 below + half of 1 equal, of 5
    expect = cvi_to_projected_eur(perf * _cvi_age_value_multiplier(24.0, 'ST'),
                                  position_group='ST', competition_id=43324) * 0.8
    assert v == pytest.approx(expect)
    camp = ei.engine_value_eur(40.0, pool, 'Striker', 24.0, 191779)
    assert camp == pytest.approx(expect * 0.85)   # Camp penalty via seasonId
    assert ei.engine_value_eur(None, pool, 'Striker', 24.0, 191782) is None


def test_engine_value_eur_frame_aligns_to_index():
    df = pd.DataFrame({'projection_abs': [40.0, None, 10.0],
                       'role': ['Striker', 'Striker', 'Central Defender'],
                       'age': [24.0, 24.0, 30.0], 'seasonId': [191782, 191782, 191779]},
                      index=[7, 3, 9])
    out = ei.engine_value_eur_frame(df)
    assert list(out.index) == [7, 3, 9]
    assert out[3] is None or pd.isna(out[3])
    assert out[7] > out[9] > 0


# --- calibration build ---------------------------------------------------------
def _pairs(scores, league='L3'):
    rows = []
    for i, s in enumerate(scores):
        value = 100_000.0
        rows.append({'playerId': 1000 + i, 'player_name': f'P{i}', 'season_id': 191782,
                     'league': league, 'role': 'Striker', 'role_group': 'ST',
                     'transfer_type': 'permanent', 'synthetic_flag': 0, 'as_of_date': '2026-01-01',
                     'fee_eur': value * math.exp(s if i % 2 else -s), 'value_eur': value,
                     'ratio': math.exp(s if i % 2 else -s), 'log_error': (s if i % 2 else -s),
                     'score': s, 'seasons_ago': 0, 'age_at_sale': 24.0,
                     'mins_played': 900.0 + i, 'w_evidence': 0.5, 'mid_season': (i == 0),
                     'included': True, 'exclusion': None})
    rows.append({'playerId': 5, 'player_name': 'Syn', 'season_id': 191782, 'league': 'L3',
                 'role': 'Striker', 'role_group': 'ST', 'transfer_type': 'permanent',
                 'synthetic_flag': 1, 'as_of_date': None, 'fee_eur': 50_000.0,
                 'value_eur': 100_000.0, 'ratio': None, 'log_error': None, 'score': None,
                 'seasons_ago': 0, 'age_at_sale': 24.0, 'mins_played': 900.0, 'w_evidence': 0.5,
                 'mid_season': False, 'included': False, 'exclusion': 'synthetic'})
    rows.append({'playerId': 6, 'player_name': 'Stayed', 'season_id': 189147, 'league': 'L3',
                 'role': None, 'role_group': None, 'transfer_type': 'permanent',
                 'synthetic_flag': 0, 'as_of_date': None, 'fee_eur': 90_000.0,
                 'value_eur': None, 'ratio': None, 'log_error': None, 'score': None,
                 'seasons_ago': 0, 'age_at_sale': None, 'mins_played': None, 'w_evidence': None,
                 'mid_season': False, 'included': False, 'exclusion': 'no_engine_row'})
    return pd.DataFrame(rows)


def test_build_calibration_reproduces_hand_computed_quantiles_and_ledger():
    calib = ei.build_calibration(_pairs(SCORES_14), engine_meta={'rating_version': 'x'}, today='2026-09-06')
    assert calib['n_calibration'] == 14 and calib['excluded'] == {'synthetic': 1, 'no_engine_row': 1}
    assert calib['excluded_rows'] == [{'player_name': 'Stayed', 'season_id': 189147, 'exclusion': 'no_engine_row'}]
    assert calib['support'] == {'min_value_eur': 25_000, 'min_mins_played': 900.0,
                                'min_w_evidence': 0.5, 'value_range_eur': [100_000.0, 100_000.0]}
    assert calib['centre_drift']['flag'] is False
    assert calib['n_mid_season'] == 1
    lv = calib['levels']
    assert lv['0.50']['q'] == pytest.approx(0.57) and lv['0.50']['shipped']
    assert lv['0.50']['loo_hits'] == 7 and lv['0.80']['loo_hits'] == 12
    assert lv['0.90']['shipped'] is False
    assert calib['coverage_ok'] is True
    assert calib['headline_level'] == '0.50'
    assert calib['sensitivity']['n_with_synthetic'] == 15
    assert any(s['kind'] == 'league' and s['n'] == 14 for s in calib['strata'])
    assert calib['prospective']['n'] == 0
    # a fee added later is scored against the PREVIOUS quantiles, once
    later = _pairs(SCORES_14 + [0.40])
    calib2 = ei.build_calibration(later, previous=calib, today='2026-10-01')
    p = calib2['prospective']
    assert p['n'] == 1 and p['hits_0.50'] == 1 and p['hits_0.80'] == 1
    assert p['history'][0]['scored_against_built'] == '2026-09-06'
    calib3 = ei.build_calibration(later, previous=calib2, today='2026-11-01')
    assert calib3['prospective']['n'] == 1   # not double-counted
    json.dumps(calib3)  # serialisable


def test_pair_fees_to_engine_filters_and_values():
    eng = pd.DataFrame({'playerId': [1, 1, 2], 'seasonId': [191782, 190090, 191782],
                        'projection_abs': [40.0, None, 20.0], 'role': ['Striker'] * 3,
                        'age': [26.0, 23.0, 30.0], 'mins_played': [900, 800, 700],
                        'seasons_ago': [2, 0, 0], 'w_evidence': [0.6, 0.5, 0.4],
                        'team': ['Club A', 'Club A', 'Atlético CP']})
    fees = pd.DataFrame({
        'playerId': [1, 1, 2, 2, 3, 2], 'player_name': list('abcdef'),
        'season_id': [191782, 190090, 191782, 191782, 191782, 191782],
        'transfer_type': ['permanent', 'permanent', 'offer', 'permanent', 'permanent', 'permanent'],
        'synthetic_flag': [0, 0, 0, 1, None, 0], 'fee_eur': [100_000, 50_000, 1, 2, 3, 25_000],
        'from_team': ['Club A', 'Club A', 'x', 'x', 'x', 'Penafiel'],
        'to_team': ['Club B', 'Club C', 'x', 'x', 'x', 'Atlético CP'],
        'as_of_date': ['2026-01-01', '2025-07-15', '2026-01-01', '2026-01-01', '2026-01-01', '2026-01-15']})
    pairs = ei.pair_fees_to_engine(fees, eng)
    assert pairs['included'].tolist() == [True, False, False, False, False, False]
    assert pairs['exclusion'].tolist() == [None, 'no_engine_row', 'offer', 'synthetic',
                                           'null_synthetic_flag', 'post_transfer_row']
    # a row AT the destination club is post-transfer performance, never a pre-sale value
    assert pairs.loc[5, 'engine_team'] == 'Atlético CP' and pairs.loc[5, 'to_team'] == 'Atlético CP'
    assert pairs['mid_season'].tolist() == [True, False, True, True, True, True]
    # valued at the age AT SALE (engine age 26 forwarded by seasons_ago 2 -> 24)
    assert pairs.loc[0, 'age_at_sale'] == 24.0 and pairs.loc[0, 'seasons_ago'] == 2
    assert pairs.loc[0, 'value_eur'] == pytest.approx(
        ei.engine_value_eur(40.0, np.array([20.0, 40.0]), 'Striker', 24.0, 191782))
    assert pairs.loc[0, 'value_eur'] > ei.engine_value_eur(40.0, np.array([20.0, 40.0]), 'Striker', 26.0, 191782)
    assert (pairs.loc[0, 'mins_played'], pairs.loc[0, 'w_evidence']) == (900.0, 0.6)
    assert pairs.loc[0, 'score'] == pytest.approx(abs(math.log(100_000 / pairs.loc[0, 'value_eur'])))
    # aliases apply on both sides of the join
    pairs2 = ei.pair_fees_to_engine(fees.assign(playerId=[11, 1, 2, 2, 3, 2]), eng, aliases={11: 1})
    assert pairs2.loc[0, 'included']


# --- data-backed ---------------------------------------------------------------
_HAS_DATA = os.path.exists(ei.ENGINE_PATH) and os.path.exists(ei.FEES_PATH)


@pytest.mark.skipif(not _HAS_DATA, reason='engine parquet / fee CSV not present')
def test_shipped_calibration_matches_a_fresh_build():
    # built in-process (NOT via main(): its --check mode exits the interpreter)
    fees, eng, meta, aliases = ei.load_inputs()
    calib = ei.build_calibration(ei.pair_fees_to_engine(fees, eng, aliases=aliases), engine_meta=meta)
    assert not any(r['engine_team'] and r['to_team'] and ei._same_club(r['engine_team'], r['to_team'])
                   for r in calib['calibration']), 'a post-transfer row leaked into the calibration set'
    assert calib['n_calibration'] >= 10
    q = [calib['levels'][k]['q'] for k in ('0.50', '0.80', '0.90')]
    assert q[0] < q[1] <= q[2]
    assert calib['levels']['0.50']['shipped']
    for row in calib['calibration']:
        assert row['score'] == pytest.approx(abs(math.log(row['fee_eur'] / row['value_eur'])))
    shipped = ei.load_calibration()
    assert shipped, 'models/value/eur_interval_calibration.json missing'
    assert shipped['levels']['0.50']['shipped'] and shipped['support']['min_value_eur'] == ei.SUPPORT_VALUE_FLOOR
    # HARD: the artifact must describe the live curve constants — a retune
    # without regenerating the JSON is a code change that forgot a step.
    assert shipped['point_model'] == calib['point_model'], (
        'curve constants changed without regenerating the calibration — run models/value/eur_intervals.py')
    # SOFT: quantile drift (a fee added, an engine refresh) is a warning, not
    # a deploy blocker — the engine rebuild regenerates the JSON.
    import warnings
    for k in ('0.50', '0.80'):
        if abs(shipped['levels'][k]['q'] - calib['levels'][k]['q']) > 1e-6:
            warnings.warn(f"shipped calibration q{k} {shipped['levels'][k]['q']:.3f} != fresh "
                          f"{calib['levels'][k]['q']:.3f}: rerun models/value/eur_intervals.py")
    if shipped.get('centre_drift', {}).get('flag'):
        warnings.warn(shipped['centre_drift']['note'])


# --- support gate ----------------------------------------------------------------
SUPPORTED = dict(CALIB, support={'min_value_eur': 25_000, 'min_mins_played': 550.0,
                                 'min_w_evidence': 0.27})


def test_range_only_inside_calibration_support():
    ok = (150_000, 0.5, SUPPORTED)
    assert ei.projected_eur_interval(*ok)[0] is not None
    assert ei.projected_eur_interval(*ok, w_evidence=0.5, mins=900) [0] is not None
    assert ei.projected_eur_interval(20_000, 0.5, SUPPORTED) == (None, None)
    assert ei.projected_eur_interval(*ok, mins=400) == (None, None)
    assert ei.projected_eur_interval(*ok, w_evidence=0.1) == (None, None)
    assert ei.range_support_reason(20_000, SUPPORTED) == 'value below €25k'
    assert ei.range_support_reason(150_000, SUPPORTED, mins=400) == 'too few minutes'
    assert ei.range_support_reason(150_000, SUPPORTED, w_evidence=0.1) == 'too little evidence behind the projection'
    assert ei.range_support_reason(150_000, SUPPORTED, w_evidence=0.27, mins=550) is None
    # unknown minutes / evidence are not held against the row
    assert ei.range_support_reason(150_000, SUPPORTED, w_evidence=None, mins=float('nan')) is None
    # a calibration without a support block gates on nothing but the value
    assert ei.range_support_reason(150_000, CALIB, w_evidence=0.01, mins=1) is None


# --- parity with the app.py closure this module replaced ---------------------------
def _old_closure_value(r, pool):
    """Verbatim logic of the _eng_eur closure that lived in app.py
    load_player_engine() until 2026-09."""
    _ROLE2CVI = {'Striker': 'ST', 'Wide Attacker': 'AM_WG',
                 'Advanced Midfielder': 'AM_WG', 'Deep Midfielder': 'CM',
                 'Wide Defender': 'FB', 'Central Defender': 'CB'}
    _CAMP_SEASON_IDS = {190230, 191779, 192925}
    pa = r.get('projection_abs')
    if pa is None or pd.isna(pa) or len(pool) == 0:
        return None
    perf = float((pool < float(pa)).mean() + 0.5 * (pool == float(pa)).mean()) * 100.0
    grp = _ROLE2CVI.get(r.get('role'))
    am = _cvi_age_value_multiplier(r.get('age'), grp)
    try:
        _comp = (702 if int(r.get('seasonId')) in _CAMP_SEASON_IDS else 43324)
    except (TypeError, ValueError):
        _comp = None
    v = cvi_to_projected_eur(perf * am, position_group=grp, competition_id=_comp)
    return None if v is None else v * 0.8


def test_engine_value_frame_matches_old_closure_bit_for_bit():
    df = pd.DataFrame({
        'projection_abs': [55.0, 40.0, 40.0, 61.5, 33.0, 48.0, 70.0, None, 52.0],
        'role': ['Striker', 'Wide Attacker', 'Advanced Midfielder', 'Deep Midfielder',
                 'Wide Defender', 'Central Defender', 'Goalkeeper', 'Striker', 'Striker'],
        'age': [24.0, 19.3, 31.0, None, 27.5, 22.0, 30.0, 25.0, 35.5],
        'seasonId': [191782, 191779, 190090, 191782, 190230, 192925, 191782, 191782, None],
    })
    pool = df['projection_abs'].dropna()
    expect = [_old_closure_value(r, pool) for _, r in df.iterrows()]
    got = ei.engine_value_eur_frame(df).tolist()
    for e, g in zip(expect, got):
        if e is None:
            assert g is None or pd.isna(g)
        else:
            assert g == e, 'engine value must be BIT-identical to the closure it replaced'


# --- render helpers (no Streamlit session needed) ---------------------------------
def test_ui_helpers_respect_support_and_goalkeepers():
    import eur_interval_ui as ui
    calib = dict(SUPPORTED, n_calibration=14, headline_level='0.50',
                 support=dict(SUPPORTED['support'], value_range_eur=[28_758.0, 482_741.0]))
    calib['levels']['0.50']['k'] = 8
    calib['levels']['0.50']['factor'] = math.exp(0.572)
    sent = ui.range_sentence(185_965, calib, w_evidence=0.5, mins=2400)
    assert sent.startswith('Likely fee if sold **€100k – €330k**') and '8 of 14 real sales' in sent
    assert ui.headline_factor(calib) == pytest.approx(math.exp(0.572))
    assert ui.headline_factor(dict(calib, levels={'0.50': {'q': 0.5, 'factor': 1.6, 'shipped': False}})) is None
    assert ui.range_sentence(185_965, calib, gk=True) is None
    assert ui.range_sentence(185_965, calib, mins=400) == (
        "No likely-fee range — too few minutes (outside the fee calibration's support).")
    assert ui.range_sentence(185_965, {}) is None
    tile = ui.pdf_range_text(185_965, calib, w_evidence=0.5, mins=2400)
    assert tile == '100k-330k' and len(tile) <= 10 and '€' not in tile
    assert ui.pdf_range_text(185_965, calib, mins=400) is None
    assert 'Campeonato' in ui.curve_text() and str(ei.ENGINE_VALUE_TEMPER) in ui.curve_text()
    assert 'No likely-fee range' in ui.help_text(calib, gk=True)
