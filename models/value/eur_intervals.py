"""Engine projected value (EUR) and its fee-calibrated prediction interval.

Two things live here, both pure pandas/numpy (no streamlit):

1. The ENGINE VALUE — the EUR figure the dashboard shows for outfielders.
   It used to be a closure inside app.py's load_player_engine(); it moved
   here so the calibration script and the app run the SAME code.
     perf  = percentile of projection_abs within the projection pool
     value = cvi_to_projected_eur(perf x age multiplier, role group, comp)
             x ENGINE_VALUE_TEMPER x regular_factor
   regular_factor (models/value/availability.py, 2026-10) scales the value by
   how likely a player with his playing time is to become a regular — a 95'
   cameo is no longer valued like a starter; 1.0 for a starter from
   ~1,000', for reserve sides, and wherever the engine row has no factor.
   (see engine_value_eur / engine_value_eur_frame).

2. The INTERVAL — a split-conformal prediction interval for the realised
   permanent-transfer fee around that point, calibrated on the real sales
   in valuations/reported_fees.csv paired with the player's engine value AS
   OF THE SALE DATE (pair_fees_to_engine — only what had been played before
   it):
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

from models.value import availability as _availability
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
SALES_PATH = DASH / 'models' / 'ratings' / 'acp_projection_sales.parquet'   # build_projection
POOLS_PATH = DASH / 'models' / 'ratings' / 'acp_projection_pools.parquet'   # build_projection

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

# Centre drift ACKNOWLEDGED (Lucas 2026-10-30, option A: keep the curve, ship
# the wider band). With every sale paired leak-free (as of its sale date) the
# Liga 3 sales split in two: five sold at 0.4-1.2x the value, six at 3-6x —
# mostly cheap players (EUR 13-55k) who still fetched EUR 60-250k, plus two
# big-buyer premiums — so the L3 median lands between them (3.12) and the
# 50% band is wider (x/÷3.12). Re-centring on that median was rejected (one
# sale decides it); re-shaping (fee ~ 496 x value^0.51, LOO band x/÷2.16) is
# on file in ACP_Official/leakage_audit_2026-10-29.md. The acknowledgement
# holds for THIS calibration set only: a new sale, or a different set, brings
# the warning back for a fresh decision.
CENTRE_DRIFT_ACK = {
    'date': '2026-10-30', 'decision': 'keep the curve (option A)',
    'l3_median_ratio': 3.12,
    'sales': [(445341, 191782), (489426, 190090), (575377, 191779), (650811, 190090),
              (693095, 191779), (709015, 189147), (776749, 190090), (807307, 189147),
              (846557, 190090), (852792, 189147), (873635, 191782), (915436, 191782),
              (1038698, 191779), (1039596, 191782)],
}

# --- interval ---------------------------------------------------------------
LEVELS = (0.5, 0.8, 0.9)
HEADLINE_LEVEL = 0.5
STRATUM_MIN_N = 15          # a stratum may earn its own width only past this
VALUE_BANDS = ((0, 100_000, '< EUR 100k'), (100_000, 300_000, 'EUR 100k-300k'),
               (300_000, float('inf'), '> EUR 300k'))
VERSION = 'eur_interval_v1'
# Ranges are only evidenced INSIDE the calibration set's own support: fees
# under EUR 25k are noise at this tier (the EUR trainer's floor), and the
# sold players all had real career evidence behind their projection. A row
# outside that support gets no range (range_support_reason).
# Career, not season (2026-10-29): the evidence is the time-weighted career
# minutes behind the projection (models/ratings/career.py; evidence weight =
# evidence / (evidence + 1500)). The season's own minutes are no longer a
# bound, so a regular with a long career keeps his range early in a season.
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


def engine_value_eur(projection_abs, pool_sorted, role, age, season_id, regular_factor=1.0):
    """The dashboard's projected value (EUR) for one engine row, or None.
    regular_factor: the row's playing-time factor (availability.py); None /
    NaN count as 1.0."""
    perf = engine_perf_percentile(projection_abs, pool_sorted)
    if perf is None:
        return None
    grp = ENGINE_ROLE2CVI.get(role)
    am = _cvi_age_value_multiplier(age, grp)
    v = cvi_to_projected_eur(perf * am, position_group=grp,
                             competition_id=competition_for_engine_season(season_id))
    if v is None:
        return None
    f = 1.0 if regular_factor is None or pd.isna(regular_factor) else float(regular_factor)
    return v * ENGINE_VALUE_TEMPER * f


def sorted_pool(engine_df):
    return np.sort(pd.to_numeric(engine_df['projection_abs'], errors='coerce')
                   .dropna().to_numpy(dtype=float))


def _role(r, col):
    """The role a value reads: the career role in `col` (value_role on a
    projection row, asof_role on an as-of value — 2026-10-29), else the
    season's own role (an engine file from before the career role)."""
    v = r.get(col)
    return v if isinstance(v, str) and v else r.get('role')


def engine_value_eur_frame(engine_df):
    """engine_value_eur for every row of the engine frame (Series aligned
    to engine_df.index; None where there is no projection), at the player's
    career role and TODAY's playing-time factor (regular_factor_now)."""
    pool = sorted_pool(engine_df)
    return engine_df.apply(
        lambda r: engine_value_eur(r.get('projection_abs'), pool, _role(r, 'value_role'),
                                   r.get('age'), r.get('seasonId'),
                                   r.get('regular_factor_now', r.get('regular_factor', 1.0))), axis=1)


def load_pools(path=None):
    """{season-year: sorted projection_abs} from build_projection's as-of
    pools (acp_projection_pools.parquet), or None when the file is absent."""
    path = Path(path or POOLS_PATH)
    if not path.exists():
        return None
    p = pd.read_parquet(path, columns=['yr', 'projection_abs'])
    return {int(y): np.sort(g['projection_abs'].to_numpy(dtype=float)) for y, g in p.groupby('yr')}


def asof_pool(engine_df, y, yr=None, pools=None):
    """The projection pool as it stood at the end of season-year y. From
    `pools` (load_pools: built like today's pool, as of y) when it has y;
    else every player's latest as-of projection (asof_abs) from a completed
    season up to y, from the engine frame. Nothing later than y either way."""
    if pools is not None and int(y) in pools:
        return pools[int(y)]
    from league_config import SEASON_YEAR
    if yr is None:
        yr = engine_df['seasonId'].map(lambda s: SEASON_YEAR.get(int(s)) if pd.notna(s) else None)
    live = (engine_df['asof_live'].fillna(False).astype(bool) if 'asof_live' in engine_df.columns
            else pd.Series(False, index=engine_df.index))
    m = engine_df['asof_abs'].notna() & ~live & (pd.to_numeric(yr, errors='coerce') <= y)
    last = (engine_df[m].assign(_y=yr[m]).sort_values(['_y', 'mins_played'], kind='stable')
            .drop_duplicates('playerId', keep='last'))
    return np.sort(pd.to_numeric(last['asof_abs'], errors='coerce').dropna().to_numpy(dtype=float))


def value_date(today=None):
    """The date TODAY's values are aged to: the Monday of the current ISO week
    (UTC). Values move week by week (~0.3%), and the stats caches — stamped
    with this date (app._engine_build_stamp) — refresh at the same rhythm."""
    t = pd.Timestamp.now(tz='UTC').tz_localize(None) if today is None else pd.Timestamp(today)
    t = t.normalize()
    return t - pd.Timedelta(days=t.weekday())


def engine_values(engine_df, as_of=None, pools='file'):
    """(now, season): two EUR Series aligned to engine_df.index (2026-10-29).

    season — each row's value as of its own season: the as-of projection
      (asof_abs, build_player_engine) at the as-of age and the row's own
      playing-time factor. A live row only once live_ok (its club has played
      enough league matches); NaN before.
    now — the player's value TODAY (as_of, default value_date()), on the row
      engine_value_eur_frame values (the one carrying `projection`): his live
      value when one of his live rows is live_ok, else today's projection. A
      player with no projection row carries it on that live row.
    Today's value takes his age ON as_of (Lucas 2026-10-29: "age the summer
    value day by day"): the summer projection's age (at its base season's
    midpoint, 1 Jan) and a live row's (at its data date) both move to as_of,
    so a value never steps down when the live value starts. Past-season
    values keep their as-of ages.
    `now` and a live row rank in today's projection pool (sorted_pool). A past
    season's value ranks in THAT year's pool (2026-10-29, no look-ahead):
    every player's latest as-of projection up to that season-year — the pool
    as it stood then, not today's (asof_pool). Without as-of columns (an
    older engine file): season is NaN and now is the summer projection's
    value (aged to as_of)."""
    from league_config import SEASON_YEAR
    as_of = value_date() if as_of is None else pd.Timestamp(as_of).normalize()
    cur = engine_df['projection_abs'].notna()
    if 'seasons_ago' in engine_df.columns:
        cur &= pd.to_numeric(engine_df['seasons_ago'], errors='coerce').eq(0)
    yrs = engine_df.loc[cur, 'seasonId'].map(lambda s: SEASON_YEAR.get(int(s)) if pd.notna(s) else None).dropna()
    base_mid = pd.Timestamp(int(yrs.max()) + 1, 1, 1) if len(yrs) else as_of
    shift = (as_of - base_mid).days / 365.25        # every projection row's age sits at base_mid
    now = pd.to_numeric(engine_value_eur_frame(
        engine_df.assign(age=pd.to_numeric(engine_df['age'], errors='coerce') + shift)), errors='coerce')
    season = pd.Series(np.nan, index=engine_df.index)
    if 'asof_abs' not in engine_df.columns:
        return now, season
    pool = sorted_pool(engine_df)
    live = engine_df['asof_live'].fillna(False).astype(bool)
    ok = engine_df['asof_abs'].notna() & (~live | engine_df['live_ok'].fillna(False).astype(bool))
    x = pd.to_numeric(engine_df['asof_abs'], errors='coerce')
    yr = engine_df['seasonId'].map(lambda s: SEASON_YEAR.get(int(s)) if pd.notna(s) else None)
    built = load_pools() if isinstance(pools, str) and pools == 'file' else pools
    pools = {}
    # the live season is today's: its rows age from their data date to as_of
    age = pd.to_numeric(engine_df['asof_age'], errors='coerce')
    if live.any() and 'asof_date' in engine_df.columns:
        d = pd.to_datetime(engine_df['asof_date'], errors='coerce')
        age = age.where(~live, age + (as_of - d).dt.days / 365.25)
    # NaN, not None, where a row can't be valued (below the pool's floor):
    # pandas 3 refuses an object into a float Series
    def _pool(i):
        if live.loc[i] or pd.isna(yr.loc[i]):
            return pool
        y = int(yr.loc[i])
        if y not in pools:
            pools[y] = asof_pool(engine_df, y, yr, built)
        return pools[y]
    vals = [engine_value_eur(xi, _pool(i), _role(r, 'asof_role'), ai, r.get('seasonId'), r.get('regular_factor', 1.0))
            for xi, ai, (i, r) in zip(x[ok], age[ok], engine_df[ok].iterrows())]
    season[ok] = np.array([np.nan if v is None else float(v) for v in vals], dtype=float)
    season = pd.to_numeric(season, errors='coerce')
    lv = engine_df[ok & live].assign(_v=season[ok & live])
    lv = lv.dropna(subset=['_v']).sort_values('mins_played').drop_duplicates('playerId', keep='last')
    has_proj = engine_df['projection_abs'].notna()
    for pid, v, i in zip(lv['playerId'], lv['_v'], lv.index):
        rows = engine_df.index[(engine_df['playerId'] == pid) & has_proj]
        now.loc[rows if len(rows) else [i]] = v
    return now, season


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
    own value / evidence-weight range (stored under 'support'); pass
    w_evidence when the row has it (engine columns). mins is checked only
    against a calibration that bounds season minutes (built before
    2026-10-29)."""
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


def season_ends(path=None):
    """season-year -> date of its last played match (both leagues)."""
    from league_config import SEASON_YEAR
    path = path or (DASH / 'matches_summary.parquet')
    if not Path(path).exists():
        return {}
    ms = pd.read_parquet(path, columns=['seasonId', 'dateutc', 'status'])
    ms = ms[ms['status'].eq('Played')]
    d = pd.to_datetime(ms['dateutc'], errors='coerce', utc=True).dt.tz_convert(None).dt.normalize()
    return d.groupby(ms['seasonId'].map(SEASON_YEAR)).max().to_dict()


def pair_fees_to_engine(fees_df, engine_df, aliases=None, ends=None, sales=None, avail=None, pools='file'):
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
    built = load_pools() if isinstance(pools, str) and pools == 'file' else pools
    season = engine_values(eng, pools=built)[1] if 'asof_abs' in eng.columns else None
    # AS OF THE SALE DATE (2026-10-29, no look-ahead): a sale meets the value
    # from everything played BEFORE it, at his age ON the sale date, ranked in
    # the pool as it stood then (asof_pool):
    #  - a summer sale (after its season's last match): that season's as-of
    #    value;
    #  - a mid-season sale: build_projection's sale-date projection
    #    (acp_projection_sales: his earlier seasons + the pre-sale part of the
    #    sale season when it can be told apart, the fits known then), with the
    #    playing-time coefficients and the pool of the previous season-year.
    from league_config import SEASON_YEAR
    ends = season_ends() if ends is None else ends
    eyr = eng['seasonId'].map(lambda v: SEASON_YEAR.get(int(v)) if pd.notna(v) else None)
    pools = {}
    if sales is None and SALES_PATH.exists():
        sales = pd.read_parquet(SALES_PATH)
    sale_idx = ({(int(p), str(d)): r for p, d, r in zip(sales['playerId'], sales['as_of_date'],
                                                       sales.to_dict('records'))}
                if sales is not None else None)
    if avail is None and _availability.OUT.exists():
        avail = pd.read_parquet(_availability.OUT, columns=['playerId', 'seasonId', 'career_share',
                                                           'career_sub_frac', 'reserve_team'])
    av_idx = ({(int(p), int(s)): (sh, sf, rt) for p, s, sh, sf, rt in zip(
        avail['playerId'], avail['seasonId'], avail['career_share'], avail['career_sub_frac'],
        avail['reserve_team'])} if avail is not None else {})
    fees = fees_df.copy()
    fees['playerId'] = pd.to_numeric(fees['playerId'], errors='coerce')
    fees['_raw_pid'] = fees['playerId']
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
        from_team = str(f.get('from_team') or '').strip()
        to_team = str(f.get('to_team') or '').strip()
        as_of = str(f.get('as_of_date') or '')
        sale = pd.to_datetime(as_of, errors='coerce')
        y_sale = SEASON_YEAR.get(sid) if sid is not None else None
        if pd.notna(sale) and y_sale is not None and ends.get(y_sale) is not None:
            mid_season = bool(sale < ends[y_sale])
        else:   # no match dates: a January-March sale is mid-season
            mid_season = len(as_of) >= 7 and as_of[5:7] in ('01', '02', '03')
        sr = None
        if season is not None and mid_season and sale_idx is not None and pd.notna(f['_raw_pid']):
            sr = sale_idx.get((int(f['_raw_pid']), as_of))
        if pid is None or (season is not None and mid_season and sale_idx is not None):
            er = eng.iloc[0:0]          # mid-season: the sale-date projection below
        elif season is not None and mid_season and y_sale is not None:
            er = eng[(eng['playerId'] == pid) & (eyr == y_sale - 1)]
        else:
            er = eng[(eng['playerId'] == pid) & (eng['seasonId'] == sid)]
        if len(er):
            if season is not None:
                live = (er['asof_live'].fillna(False).astype(bool) if 'asof_live' in er.columns
                        else pd.Series(False, index=er.index))
                er = er[er['asof_abs'].notna() & ~live]
            else:
                er = er.dropna(subset=['projection_abs'])
        value = role = mins = w_ev = ev = age_at_sale = eng_team = paired_sid = None
        seasons_ago = 0
        if len(er):
            r = er.sort_values('mins_played').iloc[-1]
            paired_sid = int(r['seasonId'])
            role = _role(r, 'asof_role' if season is not None else 'value_role')
            eng_team = str(r.get('team') or '').strip() or None
            # As-of engine (2026-10-29): the row's own as-of value — the
            # projection made from everything known at the end of the sale's
            # season, at the age then. No look-ahead.
            # Older engine files (fallback below): build_projection forwards a
            # lapsed row's age to the current season (age + seasons_ago), so it
            # is back-dated to the sale's season; the projection itself stays
            # today's (later-horizon) one — a residual look-ahead.
            if season is not None:
                # the row's as-of projection, at his age on the sale date (a
                # completed row's as-of age sits at 1 Jan of its winter)
                ry = int(eyr.loc[r.name])
                age_at_sale = r.get('asof_age')
                if pd.notna(sale) and age_at_sale is not None and pd.notna(age_at_sale):
                    age_at_sale = float(age_at_sale) + (sale - pd.Timestamp(ry + 1, 1, 1)).days / 365.25
                if ry not in pools:
                    pools[ry] = asof_pool(eng, ry, eyr, built)
                value = engine_value_eur(r.get('asof_abs'), pools[ry], role, age_at_sale,
                                         r.get('seasonId'), r.get('regular_factor', 1.0))
                we = r.get('asof_w_evidence')
            else:
                sa = r.get('seasons_ago')
                seasons_ago = int(sa) if sa is not None and not pd.isna(sa) else 0
                age_row = r.get('age')
                age_at_sale = (float(age_row) - seasons_ago
                               if age_row is not None and not pd.isna(age_row) else age_row)
                value = engine_value_eur(r.get('projection_abs'), pool, role,
                                         age_at_sale, r.get('seasonId'),
                                         r.get('regular_factor', 1.0))
                we = r.get('w_evidence')
            mp = r.get('mins_played')
            mins = float(mp) if mp is not None and not pd.isna(mp) else None
            w_ev = float(we) if we is not None and not pd.isna(we) else None
            e = r.get('asof_evidence') if season is not None else None
            ev = float(e) if e is not None and not pd.isna(e) else None
        elif sr is not None:
            paired_sid = int(sr['base_seasonId'])
            role, age_at_sale = sr['value_role'], float(sr['age'])
            py = y_sale - 1
            if py not in pools:
                pools[py] = asof_pool(eng, py, eyr, built)
            sh, sf, rt = av_idx.get((int(f['_raw_pid']), paired_sid), (np.nan, np.nan, False))
            c = _availability.coef_for(py)
            factor = (1.0 if bool(rt) or c is None or pd.isna(sh) or pd.isna(sf)
                      else _availability.regular_factor(sh, sf, c))
            value = engine_value_eur(sr['projection_sale_abs'], pools[py], role, age_at_sale,
                                     paired_sid, factor)
            b = eng[(eng['playerId'] == pid) & (eng['seasonId'] == paired_sid)]
            if len(b):
                bb = b.sort_values('mins_played').iloc[-1]
                mins = float(bb['mins_played']) if pd.notna(bb.get('mins_played')) else None
                eng_team = str(bb.get('team') or '').strip() or None
            w_ev, ev = float(sr['w_evidence']), float(sr['evidence'])
        if flag is None or pd.isna(flag):
            excl = 'null_synthetic_flag'
        elif int(flag) != 0:
            excl = 'synthetic'
        elif ttype != 'permanent':
            excl = ttype or 'non_permanent'
        elif fee is None or pd.isna(fee) or fee <= 0:
            excl = 'no_fee'
        elif value is None:
            excl = 'no_data_before_sale' if mid_season and season is not None else 'no_engine_row'
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
            'paired_season_id': paired_sid,
            'league': _league_label(sid), 'role': role,
            'role_group': ENGINE_ROLE2CVI.get(role),
            'transfer_type': ttype, 'synthetic_flag': None if pd.isna(flag) else int(flag),
            'as_of_date': f.get('as_of_date'),
            'fee_eur': None if pd.isna(fee) else float(fee),
            'value_eur': value, 'ratio': ratio,
            'seasons_ago': seasons_ago, 'age_at_sale': age_at_sale,
            'mins_played': mins, 'w_evidence': w_ev, 'evidence': ev,
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
        # the bound is the evidence weight; min_evidence is the same floor in
        # career minutes, for the caption
        'min_w_evidence': (float(cal['w_evidence'].min()) if n and cal['w_evidence'].notna().any() else None),
        'min_evidence': (float(cal['evidence'].min()) if n and 'evidence' in cal.columns
                         and cal['evidence'].notna().any() else None),
        'value_range_eur': ([float(cal['value_eur'].min()), float(cal['value_eur'].max())] if n else None),
    }
    l3_med = bias['L3']['median_ratio']
    drift = bool(l3_med is not None and abs(l3_med - 1.0) > 0.15)
    sales = (sorted((int(p), int(s)) for p, s in zip(cal['playerId'], cal['season_id']))
             if n else [])
    ack = drift and sorted(map(tuple, CENTRE_DRIFT_ACK['sales'])) == sales
    centre_drift = {
        'L3_median_ratio': l3_med,
        'flag': drift,
        'acknowledged': ack,
        'note': (f"Liga 3 sales sit at a median {l3_med:.2f}x the model value: five sold near "
                 "it, six well above (mostly cheap players who still fetched EUR 60-250k, plus "
                 "two big-buyer premiums). Kept as is by decision "
                 f"({CENTRE_DRIFT_ACK['date']}, {CENTRE_DRIFT_ACK['decision']}): the range is "
                 "wider rather than the curve re-centred on two split groups." if ack else
                 'L3 median fee/value has moved more than 0.15 from 1.0: the point-'
                 'value constants and the band no longer share a centre — retune the '
                 'curve (with sign-off) before trusting the range' if drift else 'centred'),
    }
    calib = {
        'version': VERSION, 'built': today, 'method': 'split-conformal-log',
        'engine_meta': engine_meta or {},
        'point_model': {
            'coef': PROJECTED_EUR_COEF, 'exp': PROJECTED_EUR_EXP,
            'temper': ENGINE_VALUE_TEMPER, 'camp_penalty': CAMP_PROJECTED_EUR_PENALTY,
            'position_multipliers': dict(POSITION_EUR_MULTIPLIER),
            'availability': {'coef': list(_availability.COEF), 'anchor_share': _availability.ANCHOR_SHARE,
                             'share_eps': _availability.SHARE_EPS, 'min_window': _availability.MIN_WINDOW,
                             'edge_days': _availability.EDGE_DAYS},
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
    print(f"support: value >= {sup['min_value_eur']:,}, evidence weight >= {sup['min_w_evidence']} "
          f"(career evidence >= {sup.get('min_evidence')}')")
    if calib['centre_drift']['flag']:
        print("ACKNOWLEDGED:" if calib['centre_drift'].get('acknowledged') else "WARNING:",
              calib['centre_drift']['note'])
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
    # the playing-time factor, joined on the raw ids (pairing applies the aliases)
    eng = _availability.attach(pd.read_parquet(ENGINE_PATH))
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
