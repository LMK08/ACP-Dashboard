"""Engine projected value (EUR) and its fee-calibrated prediction interval.

Two things live here, both pure pandas/numpy (no streamlit):

1. The ENGINE VALUE — the EUR figure the dashboard shows for outfielders.
   It used to be a closure inside app.py's load_player_engine(); it moved
   here so the calibration script and the app run the SAME code.
     perf  = percentile of projection_abs within the projection pool
     value = cvi_to_projected_eur(perf x age multiplier, role group, comp)
             x ENGINE_VALUE_TEMPER
   (see engine_value_eur / engine_value_eur_frame).

2. The INTERVAL — a split-conformal prediction interval for the realised
   permanent-transfer fee around that point, calibrated on the real sales
   in valuations/reported_fees.csv paired with the player's engine row for
   the pre-transfer season:
     s_i = |ln(fee_i / value_i)|                      (nonconformity score)
     q   = the k-th smallest s_i, k = ceil((n + 1) x level)
     interval = [value x exp(-q), value x exp(+q)]
   Symmetric in log space (so right-skewed in EUR, like fees), a pure
   function of the displayed point and ONE scalar per level: every page
   that shows the same point shows the same range, with no new cached
   columns. Finite-sample guarantee for an exchangeable future sale:
   coverage >= level. A level ships only when its order index k <= n - 2
   (the quantile must not be one of the two largest residuals).

   build_calibration() writes models/value/eur_interval_calibration.json
   in the engine rebuild (after build_player_engine.py); the app reads it
   through load_calibration(). Coverage is reported three ways: leave-one-
   out on the calibration set, a prospective ledger (every fee added after
   a calibration was written is scored against the quantiles that were
   live at the time — the only truly out-of-sample number), and stratum
   diagnostics (league / role group / value band) that are shown but never
   used to condition the width until a stratum reaches STRATUM_MIN_N.
"""
from __future__ import annotations

import json
import math
import os
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

if __name__ == '__main__':  # `python models/value/eur_intervals.py` from the Dashboard dir
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.value.cvi import (
    CAMP_PROJECTED_EUR_PENALTY,
    POSITION_EUR_MULTIPLIER,
    PROJECTED_EUR_COEF,
    PROJECTED_EUR_EXP,
    _cvi_age_value_multiplier,
    cvi_to_projected_eur,
)

__all__ = [
    'ENGINE_ROLE2CVI', 'ENGINE_CAMP_SEASON_IDS', 'ENGINE_VALUE_TEMPER',
    'CALIBRATION_PATH', 'LEVELS', 'HEADLINE_LEVEL',
    'engine_perf_percentile', 'engine_value_eur', 'engine_value_eur_frame',
    'conformal_k', 'conformal_quantile', 'projected_eur_interval',
    'load_calibration', 'format_eur_short', 'format_eur_range',
    'pair_fees_to_engine', 'build_calibration',
]

HERE = Path(__file__).resolve().parent
DASH = HERE.parent.parent
CALIBRATION_PATH = HERE / 'eur_interval_calibration.json'
ENGINE_PATH = DASH / 'models' / 'ratings' / 'player_engine.parquet'
ENGINE_META_PATH = DASH / 'models' / 'ratings' / 'player_engine_meta.json'
FEES_PATH = DASH / 'valuations' / 'reported_fees.csv'

# --- engine value -----------------------------------------------------------
ENGINE_ROLE2CVI = {'Striker': 'ST', 'Wide Attacker': 'AM_WG',
                   'Advanced Midfielder': 'AM_WG', 'Deep Midfielder': 'CM',
                   'Wide Defender': 'FB', 'Central Defender': 'CB'}
# Camp membership derived from seasonId (no competitionId in the engine
# parquet). Fee calibration (2026-07-26): L3 sales realise at a median
# ~1.1x engine value, Camp at ~0.7x — the existing Camp penalty closes it.
ENGINE_CAMP_SEASON_IDS = {190230, 191779, 192925}
# Global price temper (2026-06-12): the engine values read a touch rich
# for this market — the whole curve is scaled down 20%.
ENGINE_VALUE_TEMPER = 0.8

# --- interval ---------------------------------------------------------------
LEVELS = (0.5, 0.8, 0.9)
HEADLINE_LEVEL = 0.5
STRATUM_MIN_N = 15          # a stratum may earn its own width only past this
VALUE_BANDS = ((0, 100_000, '< EUR 100k'), (100_000, 300_000, 'EUR 100k-300k'),
               (300_000, float('inf'), '> EUR 300k'))
VERSION = 'eur_interval_v1'
# Ranges are only evidenced INSIDE the calibration set's own support: fees
# under EUR 25k are noise at this tier (the EUR trainer's floor), and the
# sold players all had real minutes / evidence behind their projection. A
# row outside that support gets no range (range_support_reason).
SUPPORT_VALUE_FLOOR = 25_000


def competition_for_engine_season(season_id):
    try:
        return 702 if int(season_id) in ENGINE_CAMP_SEASON_IDS else 43324
    except (TypeError, ValueError):
        return None


def engine_perf_percentile(projection_abs, pool_sorted):
    """Percentile (0-100) of one projection_abs within the sorted pool:
    share strictly below + half the share equal — the closure's rule."""
    n = len(pool_sorted)
    if n == 0 or projection_abs is None or pd.isna(projection_abs):
        return None
    pa = float(projection_abs)
    lo = int(np.searchsorted(pool_sorted, pa, side='left'))
    hi = int(np.searchsorted(pool_sorted, pa, side='right'))
    # Same floating-point order as the closure it replaced
    # ((pool < pa).mean() + 0.5 * (pool == pa).mean()) * 100 — each mean is
    # count/n — so the result is bit-identical, not merely close.
    return float(lo / n + 0.5 * ((hi - lo) / n)) * 100.0


def engine_value_eur(projection_abs, pool_sorted, role, age, season_id):
    """The dashboard's projected value (EUR) for one engine row, or None."""
    perf = engine_perf_percentile(projection_abs, pool_sorted)
    if perf is None:
        return None
    grp = ENGINE_ROLE2CVI.get(role)
    am = _cvi_age_value_multiplier(age, grp)
    v = cvi_to_projected_eur(perf * am, position_group=grp,
                             competition_id=competition_for_engine_season(season_id))
    return None if v is None else v * ENGINE_VALUE_TEMPER


def sorted_pool(engine_df):
    return np.sort(pd.to_numeric(engine_df['projection_abs'], errors='coerce')
                   .dropna().to_numpy(dtype=float))


def engine_value_eur_frame(engine_df):
    """engine_value_eur for every row of the engine frame (Series aligned
    to engine_df.index; None where there is no projection)."""
    pool = sorted_pool(engine_df)
    return engine_df.apply(
        lambda r: engine_value_eur(r.get('projection_abs'), pool, r.get('role'),
                                   r.get('age'), r.get('seasonId')), axis=1)


def conformal_k(n, level):
    """Order index of the split-conformal quantile for coverage `level`."""
    return int(math.ceil((n + 1) * float(level)))


def conformal_quantile(scores, level):
    """(q, k, shipped) for nonconformity scores at `level`. q is the k-th
    smallest score; shipped is False when k is off the end or one of the
    two largest scores (too few residuals to pin that tail)."""
    s = np.sort(np.asarray([x for x in scores if x is not None and not pd.isna(x)],
                           dtype=float))
    n = len(s)
    k = conformal_k(n, level)
    if n == 0 or k > n:
        return None, k, False
    return float(s[k - 1]), k, bool(k <= n - 2)


def loo_coverage(scores, level):
    """Leave-one-out hits: each score tested against the quantile computed
    from the other n-1 scores (with k for n-1)."""
    s = np.asarray(scores, dtype=float)
    hits = 0
    for i in range(len(s)):
        rest = np.delete(s, i)
        q, _, _ = conformal_quantile(rest, level)
        if q is not None and s[i] <= q:
            hits += 1
    return hits, len(s)


def range_support_reason(point, calib=None, w_evidence=None, mins=None):
    """None when the calibration supports a range for this row, else a
    short reason. The band is evidenced only inside the calibration set's
    own value / minutes / evidence-weight range (stored under 'support');
    pass w_evidence and mins when the row has them (engine columns)."""
    calib = load_calibration() if calib is None else calib
    sup = (calib or {}).get('support') or {}
    if point is None or pd.isna(point) or float(point) <= 0:
        return 'no value'
    floor = sup.get('min_value_eur')
    if floor and float(point) < float(floor):
        return f"value below {format_eur_short(floor)}"
    mm = sup.get('min_mins_played')
    if mm and mins is not None and not pd.isna(mins) and float(mins) < float(mm):
        return 'too few minutes'
    mw = sup.get('min_w_evidence')
    if mw and w_evidence is not None and not pd.isna(w_evidence) and float(w_evidence) < float(mw):
        return 'too little evidence behind the projection'
    return None


def projected_eur_interval(point, level=HEADLINE_LEVEL, calib=None,
                           w_evidence=None, mins=None):
    """(lo, hi) around a displayed point value, or (None, None) when there
    is no calibration, the level did not ship, the point is unusable, or
    the row sits outside the calibration's support."""
    if calib is None:
        calib = load_calibration()
    if not calib or point is None or pd.isna(point) or float(point) <= 0:
        return None, None
    if range_support_reason(point, calib, w_evidence=w_evidence, mins=mins):
        return None, None
    lv = (calib.get('levels') or {}).get(_level_key(level))
    if not lv or not lv.get('shipped') or lv.get('q') is None:
        return None, None
    q = float(lv['q'])
    p = float(point)
    return p * math.exp(-q), p * math.exp(q)


def _level_key(level):
    return f"{float(level):.2f}"


_CALIB_CACHE = {'path': None, 'mtime': None, 'data': {}}


def load_calibration(path=None):
    """The shipped calibration dict, {} if the file is missing. Cached on
    (path, mtime) so a rebuilt file is picked up without a restart."""
    p = Path(path) if path else CALIBRATION_PATH
    try:
        mtime = os.path.getmtime(p)
    except OSError:
        return {}
    if _CALIB_CACHE['path'] == str(p) and _CALIB_CACHE['mtime'] == mtime:
        return _CALIB_CACHE['data']
    try:
        with open(p) as f:
            data = json.load(f)
    except Exception:
        data = {}
    _CALIB_CACHE.update(path=str(p), mtime=mtime, data=data)
    return data


def _round_sf(v, sf=2):
    """Round half UP to `sf` significant figures (2650 -> 2700, not
    Python's banker's 2600)."""
    if v == 0:
        return 0.0
    d = sf - int(math.floor(math.log10(abs(v)))) - 1
    scale = 10.0 ** d
    return math.copysign(math.floor(abs(v) * scale + 0.5) / scale, v)


def format_eur_short(v):
    """Two significant figures: EUR 850 / 2.7k / 85k / 180k / 1.2M / 12M."""
    if v is None or pd.isna(v):
        return '—'
    r = _round_sf(float(v))
    sign = '-' if r < 0 else ''
    r = abs(r)
    if r < 1_000:
        return f"{sign}€{r:,.0f}"
    if r < 1_000_000:
        k = r / 1_000
        return f"{sign}€{k:.0f}k" if k >= 10 else f"{sign}€{k:.1f}k"
    m = r / 1_000_000
    return f"{sign}€{m:.0f}M" if m >= 10 else f"{sign}€{m:.1f}M"


def format_eur_range(lo, hi):
    if lo is None or hi is None:
        return None
    return f"{format_eur_short(lo)} – {format_eur_short(hi)}"


# --- calibration build ------------------------------------------------------
def _league_label(season_id):
    return 'Camp' if competition_for_engine_season(season_id) == 702 else 'L3'


def _same_club(a, b):
    """Loose club-name match ('Atlético CP' == 'Atlético CP'; 'Sporting CP B'
    ~ 'Sporting CP II' is NOT attempted — only used to catch a row at the
    DESTINATION club)."""
    a = ' '.join(str(a or '').casefold().split())
    b = ' '.join(str(b or '').casefold().split())
    return bool(a and b and (a == b or a in b or b in a))


def pair_fees_to_engine(fees_df, engine_df, aliases=None):
    """One row per fee with the engine value at the fee's pre-transfer
    season. `included` marks the calibration set (real permanent sales
    with an engine value); `exclusion` says why the others are out."""
    eng = engine_df.copy()
    eng['playerId'] = pd.to_numeric(eng['playerId'], errors='coerce').astype('Int64')
    eng['seasonId'] = pd.to_numeric(eng['seasonId'], errors='coerce').astype('Int64')
    if aliases:
        eng['playerId'] = eng['playerId'].map(
            lambda p: aliases.get(int(p), int(p)) if pd.notna(p) else p).astype('Int64')
    pool = sorted_pool(eng)
    fees = fees_df.copy()
    fees['playerId'] = pd.to_numeric(fees['playerId'], errors='coerce')
    if aliases:
        fees['playerId'] = fees['playerId'].map(
            lambda p: aliases.get(int(p), int(p)) if pd.notna(p) else p)
    rows = []
    for _, f in fees.iterrows():
        flag = f.get('synthetic_flag')
        ttype = str(f.get('transfer_type', '')).strip().lower()
        fee = pd.to_numeric(f.get('fee_eur'), errors='coerce')
        try:
            pid, sid = int(f['playerId']), int(f['season_id'])
        except (TypeError, ValueError):
            pid, sid = None, None
        er = (eng[(eng['playerId'] == pid) & (eng['seasonId'] == sid)]
              if pid is not None else eng.iloc[0:0])
        er = er.dropna(subset=['projection_abs']) if len(er) else er
        value = role = mins = w_ev = age_at_sale = eng_team = None
        seasons_ago = 0
        from_team = str(f.get('from_team') or '').strip()
        to_team = str(f.get('to_team') or '').strip()
        as_of = str(f.get('as_of_date') or '')
        # A January sale's season row also contains matches AFTER the sale
        # (one engine row per player-season); flagged, not excluded.
        mid_season = len(as_of) >= 7 and as_of[5:7] in ('01', '02', '03')
        if len(er):
            r = er.sort_values('mins_played').iloc[-1]
            role = r.get('role')
            eng_team = str(r.get('team') or '').strip() or None
            # build_projection forwards a lapsed row's age to the current
            # season (age + seasons_ago). The sale happened in the row's own
            # season, so value it at the age back then — otherwise the older
            # sales (5 of 14 in 2026-09) are valued rich by the age curve.
            # Only the value curve's age effect is back-dated: the row's
            # projection is still today's (later-horizon) projection ranked
            # in today's pool. That residual look-ahead is documented in the
            # panel, not corrected.
            sa = r.get('seasons_ago')
            seasons_ago = int(sa) if sa is not None and not pd.isna(sa) else 0
            age_row = r.get('age')
            age_at_sale = (float(age_row) - seasons_ago
                           if age_row is not None and not pd.isna(age_row) else age_row)
            value = engine_value_eur(r.get('projection_abs'), pool, role,
                                     age_at_sale, r.get('seasonId'))
            mp, we = r.get('mins_played'), r.get('w_evidence')
            mins = float(mp) if mp is not None and not pd.isna(mp) else None
            w_ev = float(we) if we is not None and not pd.isna(we) else None
        if flag is None or pd.isna(flag):
            excl = 'null_synthetic_flag'
        elif int(flag) != 0:
            excl = 'synthetic'
        elif ttype != 'permanent':
            excl = ttype or 'non_permanent'
        elif fee is None or pd.isna(fee) or fee <= 0:
            excl = 'no_fee'
        elif value is None:
            excl = 'no_engine_row'
        elif _same_club(eng_team, to_team):
            # The row is at the DESTINATION club: the player arrived from
            # outside the dataset, so this is post-transfer performance, not
            # the value a buyer saw (Hélder Suker, Penafiel -> ACP, 2026-01).
            excl = 'post_transfer_row'
        else:
            excl = None
        ratio = (float(fee) / value) if (excl is None) else None
        rows.append({
            'playerId': pid, 'player_name': f.get('player_name'), 'season_id': sid,
            'league': _league_label(sid), 'role': role,
            'role_group': ENGINE_ROLE2CVI.get(role),
            'transfer_type': ttype, 'synthetic_flag': None if pd.isna(flag) else int(flag),
            'as_of_date': f.get('as_of_date'),
            'fee_eur': None if pd.isna(fee) else float(fee),
            'value_eur': value, 'ratio': ratio,
            'seasons_ago': seasons_ago, 'age_at_sale': age_at_sale,
            'mins_played': mins, 'w_evidence': w_ev,
            'from_team': from_team or None, 'to_team': to_team or None,
            'engine_team': eng_team, 'mid_season': bool(mid_season),
            'log_error': (math.log(ratio) if ratio else None),
            'score': (abs(math.log(ratio)) if ratio else None),
            'included': excl is None, 'exclusion': excl,
        })
    return pd.DataFrame(rows)


def _value_band(v):
    for lo, hi, label in VALUE_BANDS:
        if lo <= v < hi:
            return label
    return VALUE_BANDS[-1][2]


def _strata(cal, loo_hits_by_level):
    out = []
    for kind, col in (('league', 'league'), ('role_group', 'role_group'), ('value_band', '_band')):
        for name, grp in cal.groupby(col, dropna=False):
            idx = grp.index.to_numpy()
            entry = {'kind': kind, 'name': str(name), 'n': int(len(grp)),
                     'median_ratio': float(grp['ratio'].median()),
                     'conditioned': False,
                     'eligible_at_n': STRATUM_MIN_N}
            for lk, hits in loo_hits_by_level.items():
                h = int(sum(hits[i] for i in idx))
                entry[f'loo_hits_{lk}'] = h
                entry[f'loo_coverage_{lk}'] = round(h / len(grp), 3)
            out.append(entry)
    return out


def build_calibration(pairs, engine_meta=None, previous=None, today=None):
    """Calibration dict from pair_fees_to_engine() output."""
    today = today or date.today().isoformat()
    cal = pairs[pairs['included']].copy().reset_index(drop=True)
    cal['_band'] = cal['value_eur'].map(_value_band)
    scores = cal['score'].to_numpy(dtype=float)
    n = len(cal)

    levels = {}
    loo_hits_by_level = {}
    for level in LEVELS:
        q, k, shipped = conformal_quantile(scores, level)
        hits, _ = loo_coverage(scores, level) if n else (0, 0)
        # per-row LOO hits for the strata table
        row_hits = []
        for i in range(n):
            qi, _, _ = conformal_quantile(np.delete(scores, i), level)
            row_hits.append(bool(qi is not None and scores[i] <= qi))
        lk = _level_key(level)
        loo_hits_by_level[lk] = row_hits
        tol = 1.0 / (n + 1) if n else 1.0
        levels[lk] = {
            'level': level, 'k': k, 'q': q,
            'factor': (math.exp(q) if q is not None else None),
            'shipped': shipped,
            # NOTE: for a split-conformal order statistic, leave-one-out
            # coverage is an identity of the construction (k-1 or k of n hit,
            # depending on how ceil(n*level) compares with k) — a property,
            # not a test. The prospective ledger is the only out-of-sample check.
            'loo_hits': hits, 'loo_n': n,
            'loo_coverage': (round(hits / n, 3) if n else None),
            'coverage_ok': (bool(hits / n >= level - tol) if n else False),
        }

    def _median(d):
        return float(d['ratio'].median()) if len(d) else None
    bias = {
        'pooled': {'n': n, 'median_ratio': _median(cal)},
        'L3': {'n': int((cal['league'] == 'L3').sum()), 'median_ratio': _median(cal[cal['league'] == 'L3'])},
        'Camp': {'n': int((cal['league'] == 'Camp').sum()), 'median_ratio': _median(cal[cal['league'] == 'Camp'])},
    }

    # sensitivity: synthetic rows included; asymmetric two-sided band
    syn = pairs[(pairs['exclusion'] == 'synthetic') & pairs['value_eur'].notna()
                & pairs['fee_eur'].notna()]
    syn_scores = np.abs(np.log(syn['fee_eur'] / syn['value_eur'])).to_numpy(dtype=float)
    with_syn = np.concatenate([scores, syn_scores]) if len(syn_scores) else scores
    sens = {'n_with_synthetic': int(len(with_syn))}
    for level in LEVELS:
        q, _, _ = conformal_quantile(with_syn, level)
        sens[f'q_with_synthetic_{_level_key(level)}'] = q
    if n:
        r = np.sort(cal['log_error'].to_numpy(dtype=float))
        k2 = int(math.ceil((n + 1) * (1 - HEADLINE_LEVEL) / 2))
        if 1 <= k2 <= n:
            sens['asymmetric_headline_band'] = {
                'm_lo': float(math.exp(r[k2 - 1])), 'm_hi': float(math.exp(r[n - k2]))}

    # prospective ledger: fees not in the previous calibration set, scored
    # against the quantiles that were live then
    prev = previous or {}
    history = list((prev.get('prospective') or {}).get('history') or [])
    prev_keys = {(int(c['playerId']), int(c['season_id']))
                 for c in (prev.get('calibration') or [])}
    seen = {(int(h['playerId']), int(h['season_id'])) for h in history}
    prev_levels = prev.get('levels') or {}
    if prev_keys:
        for _, c in cal.iterrows():
            key = (int(c['playerId']), int(c['season_id']))
            if key in prev_keys or key in seen:
                continue
            entry = {'playerId': key[0], 'player_name': c['player_name'],
                     'season_id': key[1], 'fee_eur': c['fee_eur'],
                     'value_eur': c['value_eur'], 'score': c['score'],
                     'scored_against_built': prev.get('built')}
            for lk, lv in prev_levels.items():
                if lv.get('shipped') and lv.get('q') is not None:
                    entry[f'hit_{lk}'] = bool(c['score'] <= float(lv['q']))
            history.append(entry)
    prospective = {'history': history, 'n': len(history)}
    for level in LEVELS:
        lk = _level_key(level)
        scored = [h for h in history if f'hit_{lk}' in h]
        prospective[f'n_{lk}'] = len(scored)
        prospective[f'hits_{lk}'] = int(sum(h[f'hit_{lk}'] for h in scored))

    excl_counts = pairs.loc[~pairs['included'], 'exclusion'].value_counts().to_dict()
    # Real sales that could not be paired (only a player's latest season
    # carries a projection, so within-tier sales of players who stayed are
    # dropped) — listed so the growing exclusion is visible, not silent.
    excluded_rows = [
        {'player_name': r['player_name'], 'season_id': r['season_id'], 'exclusion': r['exclusion']}
        for r in pairs[pairs['exclusion'].isin(['no_engine_row', 'post_transfer_row'])].to_dict('records')]
    support = {
        'min_value_eur': SUPPORT_VALUE_FLOOR,
        'min_mins_played': (float(cal['mins_played'].min()) if n and cal['mins_played'].notna().any() else None),
        'min_w_evidence': (float(cal['w_evidence'].min()) if n and cal['w_evidence'].notna().any() else None),
        'value_range_eur': ([float(cal['value_eur'].min()), float(cal['value_eur'].max())] if n else None),
    }
    l3_med = bias['L3']['median_ratio']
    centre_drift = {
        'L3_median_ratio': l3_med,
        'flag': bool(l3_med is not None and abs(l3_med - 1.0) > 0.15),
        'note': ('L3 median fee/value has moved more than 0.15 from 1.0: the point-'
                 'value constants and the band no longer share a centre — retune the '
                 'curve (with sign-off) before trusting the range' if
                 (l3_med is not None and abs(l3_med - 1.0) > 0.15) else 'centred'),
    }
    calib = {
        'version': VERSION, 'built': today, 'method': 'split-conformal-log',
        'engine_meta': engine_meta or {},
        'point_model': {
            'coef': PROJECTED_EUR_COEF, 'exp': PROJECTED_EUR_EXP,
            'temper': ENGINE_VALUE_TEMPER, 'camp_penalty': CAMP_PROJECTED_EUR_PENALTY,
            'position_multipliers': dict(POSITION_EUR_MULTIPLIER),
        },
        'n_fee_rows': int(len(pairs)), 'n_calibration': n,
        'n_mid_season': (int(cal['mid_season'].sum()) if n and 'mid_season' in cal.columns else 0),
        'excluded': {str(k): int(v) for k, v in excl_counts.items()},
        'excluded_rows': excluded_rows,
        'support': support,
        'centre_drift': centre_drift,
        'headline_level': _level_key(HEADLINE_LEVEL),
        'levels': levels,
        'coverage_ok': all(lv['coverage_ok'] for lv in levels.values() if lv['shipped']),
        'bias': bias,
        'strata': _strata(cal, loo_hits_by_level) if n else [],
        'sensitivity': sens,
        'prospective': prospective,
        'calibration': [
            {k: (None if (isinstance(v, float) and math.isnan(v)) else v)
             for k, v in row.items() if k != '_band'}
            for row in cal.drop(columns=['included', 'exclusion']).to_dict('records')
        ],
    }
    return calib


def _print_report(pairs, calib):
    t = pairs.copy()
    print("=== fee vs engine value (dashboard Projected value) ===")
    print(t.sort_values('ratio').to_string(index=False, columns=[
        'player_name', 'transfer_type', 'synthetic_flag', 'season_id', 'league',
        'role', 'fee_eur', 'value_eur', 'ratio', 'exclusion'],
        formatters={'fee_eur': lambda v: f"{v:,.0f}" if pd.notna(v) else '—',
                    'value_eur': lambda v: f"{v:,.0f}" if pd.notna(v) else '—',
                    'ratio': lambda v: f"{v:.2f}" if pd.notna(v) else '—'}))
    print(f"\ncalibration set: n={calib['n_calibration']} of {calib['n_fee_rows']} fee rows; "
          f"excluded {calib['excluded']}")
    b = calib['bias']
    print("median fee/value: " + ", ".join(
        f"{k} {v['median_ratio']:.2f} (n={v['n']})" for k, v in b.items() if v['median_ratio'] is not None))
    print("sorted scores |ln(fee/value)|:",
          ' '.join(f"{s:.2f}" for s in sorted(c['score'] for c in calib['calibration'])))
    for lk, lv in calib['levels'].items():
        print(f"level {lk}: k={lv['k']} q={lv['q']:.3f} factor x/÷{lv['factor']:.2f} "
              f"shipped={lv['shipped']} LOO {lv['loo_hits']}/{lv['loo_n']} "
              f"= {lv['loo_coverage']} ok={lv['coverage_ok']}"
              if lv['q'] is not None else f"level {lk}: no quantile (n too small)")
    p = calib['prospective']
    print(f"prospective ledger: {p['n']} fee(s) scored against earlier quantiles")
    sup = calib['support']
    print(f"support: value >= {sup['min_value_eur']:,}, minutes >= {sup['min_mins_played']}, "
          f"evidence weight >= {sup['min_w_evidence']}")
    if calib['centre_drift']['flag']:
        print("WARNING:", calib['centre_drift']['note'])
    if calib['excluded_rows']:
        print("unpaired real sales:", ', '.join(f"{r['player_name']} ({r['exclusion']})" for r in calib['excluded_rows']))
    print(f"mid-season (Jan-Mar) sales in the set: {calib['n_mid_season']} of {calib['n_calibration']}")
    others = pairs[~pairs['included'] & ~pairs['exclusion'].isin(['synthetic', 'no_engine_row', 'post_transfer_row'])]
    if len(others):
        print("other exclusions:", ', '.join(f"{r['player_name']} ({r['exclusion']})" for r in others.to_dict('records')))


def load_inputs():
    """(fees, engine, engine_meta, aliases) exactly as the build uses them —
    shared with the test that rebuilds the calibration in-process."""
    fees = pd.read_csv(FEES_PATH, comment='#')
    eng = pd.read_parquet(ENGINE_PATH)
    meta = {}
    if ENGINE_META_PATH.exists():
        with open(ENGINE_META_PATH) as f:
            meta = json.load(f)
    try:
        from league_config import PLAYER_ID_ALIASES
    except Exception as e:  # pragma: no cover
        print(f"WARNING: PLAYER_ID_ALIASES not importable ({e}) — pairing WITHOUT the alias remap")
        PLAYER_ID_ALIASES = {}
    return fees, eng, meta, PLAYER_ID_ALIASES


def main(write=True, check=False):
    """Build (and by default write) the calibration. check=True instead
    compares a fresh build with the shipped JSON and exits 2 when they
    differ — the `--check` CLI mode; never raised for library callers."""
    fees, eng, meta, PLAYER_ID_ALIASES = load_inputs()
    pairs = pair_fees_to_engine(fees, eng, aliases=PLAYER_ID_ALIASES)
    previous = {}
    if CALIBRATION_PATH.exists():
        with open(CALIBRATION_PATH) as f:
            previous = json.load(f)
    calib = build_calibration(pairs, engine_meta={
        k: meta.get(k) for k in ('data_through', 'rating_version', 'projection_version', 'built_at')},
        previous=previous)
    _print_report(pairs, calib)
    if write:
        with open(CALIBRATION_PATH, 'w') as f:
            json.dump(calib, f, indent=1, default=lambda o: None if pd.isna(o) else o)
        print(f"wrote {CALIBRATION_PATH}")
    elif check and previous:
        # --check: is the shipped artifact what a fresh build would write?
        stale = [k for k in ('0.50', '0.80')
                 if abs((previous.get('levels', {}).get(k, {}).get('q') or 0)
                        - (calib['levels'][k]['q'] or 0)) > 1e-6]
        if previous.get('point_model') != calib['point_model']:
            stale.append('point_model')
        if stale:
            print(f"STALE: shipped calibration differs from a fresh build in {stale} — "
                  f"rerun without --check and commit the JSON")
            raise SystemExit(2)
        print("shipped calibration is current")
    return calib


if __name__ == '__main__':
    import sys
    os.chdir(DASH)
    main(write='--check' not in sys.argv, check='--check' in sys.argv)
