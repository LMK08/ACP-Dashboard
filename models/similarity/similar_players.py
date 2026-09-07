"""Similar-player search — like-for-like neighbours for a scout.

WHAT IS COMPARED. One vector per (player, season) built from the per-season
player-stats caches (stats_cache/player_percentiles_<ver>_<seasonId>.parquet,
the frame every player page reads), joined to the roles layer's spatial
scalars (models/roles/role_features_season.parquet). Three blocks:
  STYLE   36 raw per-90 / rate metrics (what he does: volumes, passing
          shape, duels, pressing, dribbling...) — weight 1.0 each
  QUALITY 8 value/DefR per-90 metrics (how well)        — weight 0.5 each
  SPATIAL 6 pitch-occupancy scalars (where he does it)   — weight 1.0 each
Every column is turned into a PERCENTILE WITHIN THE POSITION BUCKET over the
whole pool (both leagues, every cached season with >= POOL_MIN_MINUTES), so
each dimension is on the same 0-1 scale, unbounded ratios (Loss index) and
per-90 magnitudes cannot dominate, and "both top 20% for pressing" is a
sentence a scout can read. Distance is weighted Euclidean inside the bucket.

WHY THIS SPACE. Measured 2026-09-06 on 3,825 outfield player-seasons
(scratchpad/sim_eval.py): a player's other season lands in his own top-10
37% of the time (chance 1.5%; median rank 23), the top-10 share the query's
engine role 0.80 vs 0.63 by chance and his style label 0.38 vs 0.19.
Within-bucket z-scores with collinearity pruning scored the same (0.37 /
0.81 / 0.38); the spatial scalars add ~+0.02 role agreement; a quality
weight of 1.0 lost self-match. Percentiles were kept for robustness and
explainability.

SCORE SHOWN. similarity = 100 - 50 x d / median_bucket_distance, clipped to
[0, 100]: an identical profile scores 100, a typical pair of that bucket
50, a far pair 0 — comparable across buckets and scopes because the pool
is fixed. The table also carries 'closer than X% of <bucket> pairs' from
the bucket's empirical distance distribution.

BUCKETS. models.value.cvi._cvi_position_group of the row's primaryPosition
(CB / FB / CM / AM_WG / ST), assigned PER SEASON ROW: primaryPosition is
the first position a player was seen in THAT season, so a converted player
can sit in different buckets in different seasons; a query only sees the
seasons that share its bucket (the self-match check counts only those).
Goalkeepers are NOT covered in v1 (their metric set is different). The UI
shows the bucket so a scout can tell.

CAVEAT the UI must carry: neighbour lists are descriptive and noisy at this
sample size — the same player's top-10 set overlaps little across seasons
even though he himself is found — so scores and shared traits are shown,
never "the replacement".
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from league_config import all_season_id_map, competition_for_season
from models.value.cvi import _cvi_position_group

__all__ = ['SIM_VERSION', 'POOL_MIN_MINUTES', 'STYLE_FEATURES', 'QUALITY_FEATURES',
           'SPATIAL_FEATURES', 'FEATURE_LABELS', 'Pool', 'build_pool', 'neighbours',
           'explain', 'validation', 'bucket_of', 'season_year']

SIM_VERSION = 'sim_v1'
POOL_MIN_MINUTES = 500
QUALITY_WEIGHT = 0.5
BUCKETS = ('CB', 'FB', 'CM', 'AM_WG', 'ST')
ECDF_SAMPLE = 400            # rows sampled per bucket for the distance CDF
ECDF_QUANTILES = 200

STYLE_FEATURES = [
    'npxG', 'Shots', 'xG per Shot', 'xAOP', 'xTOP', 'Touches in penalty area',
    'Passes', 'Passes successful %', 'Progressive Passes', 'Deep Completions',
    'Passes to final third successful', 'Long passes', 'Crosses', 'Through passes',
    'Passes to penalty area', 'Forward passes', 'Back passes', 'Dribbles',
    'Dribbles successful %', 'Progressive runs', 'Fouls suffered', 'Losses',
    'Defensive duels', 'Defensive duels successful %', 'Interceptions', 'Recoveries',
    'Recoveries Opp Half', 'Counterpressing Recoveries', 'Clearances', 'Sliding tackles',
    'Avg defensive action height', 'Aerial duels', 'Aerial duels successful %', 'Fouls',
    'Possessions won', 'Dribbled past % (proj)',
]
QUALITY_FEATURES = ['Total Value', 'Shooting Value', 'Passing Value', 'Receiving Value',
                    'Dribbling Value', 'Interrupting Value', 'DefR Total', 'Def Wins Above Exp']
SPATIAL_FEATURES = ['def_share', 'x_ip', 'x_op', 'yf_ip', 'yf_op', 'box_share_ip']
# 'ACP Rating (abs)' is league-ABSOLUTE; 'ACP Rating' is within league x role x season and
# must never be compared across the two-league pool.
DISPLAY_COLUMNS = ['ACP Rating (abs)', 'ACP Rating', 'ACP Projection', 'Evidence Weight', 'Engine Value EUR']

# plain-language names for the "why similar" text
FEATURE_LABELS = {
    'npxG': 'shot quality volume (npxG)', 'Shots': 'shot volume', 'xG per Shot': 'shot selection',
    'xAOP': 'open-play chance creation', 'xTOP': 'open-play threat', 'Touches in penalty area': 'box touches',
    'Passes': 'passing volume', 'Passes successful %': 'pass security', 'Progressive Passes': 'progressive passing',
    'Deep Completions': 'deep completions', 'Passes to final third successful': 'final-third entries',
    'Long passes': 'long passing', 'Crosses': 'crossing', 'Through passes': 'through balls',
    'Passes to penalty area': 'passes into the box', 'Forward passes': 'forward passing',
    'Back passes': 'back passing', 'Dribbles': 'dribbling volume', 'Dribbles successful %': 'dribble success',
    'Progressive runs': 'ball carrying', 'Fouls suffered': 'fouls drawn', 'Losses': 'ball losses',
    'Defensive duels': 'defensive duel volume', 'Defensive duels successful %': 'defensive duel success',
    'Interceptions': 'interceptions', 'Recoveries': 'recoveries', 'Recoveries Opp Half': 'high recoveries',
    'Counterpressing Recoveries': 'counter-pressing', 'Clearances': 'clearances', 'Sliding tackles': 'sliding tackles',
    'Avg defensive action height': 'height of defensive actions', 'Aerial duels': 'aerial volume',
    'Aerial duels successful %': 'aerial success', 'Fouls': 'fouls committed', 'Possessions won': 'possessions won',
    'Dribbled past % (proj)': 'getting dribbled past',
    'Total Value': 'total action value', 'Shooting Value': 'shooting value', 'Passing Value': 'passing value',
    'Receiving Value': 'receiving value', 'Dribbling Value': 'dribbling value',
    'Interrupting Value': 'interrupting value', 'DefR Total': 'defensive rating',
    'Def Wins Above Exp': 'duel wins above expectation',
    'def_share': 'share of actions that are defensive', 'x_ip': 'height of involvement in possession',
    'x_op': 'height of defensive positioning', 'yf_ip': 'width in possession',
    'yf_op': 'width out of possession', 'box_share_ip': 'share of actions in the box',
}


def bucket_of(primary_position):
    b = _cvi_position_group(primary_position)
    return b if b in BUCKETS else None


_SEASON_YEAR = {}


def season_year(season_id):
    """Sort key for seasons (label '2025/26' -> 2025); seasonId is NOT
    chronological across leagues."""
    if not _SEASON_YEAR:
        for sid, label in all_season_id_map().items():
            try:
                _SEASON_YEAR[int(sid)] = int(str(label)[:4])
            except (TypeError, ValueError):
                pass
    try:
        return _SEASON_YEAR.get(int(season_id), 0)
    except (TypeError, ValueError):
        return 0


@dataclass
class Pool:
    meta: pd.DataFrame                       # one row per (playerId, seasonId)
    X: dict = field(default_factory=dict)    # bucket -> weighted percentile matrix (rows = meta positions)
    rows: dict = field(default_factory=dict)  # bucket -> np.ndarray of meta row positions
    features: list = field(default_factory=list)
    weights: np.ndarray = None
    ecdf: dict = field(default_factory=dict)  # bucket -> sorted sample of pairwise distances
    median_d: dict = field(default_factory=dict)  # bucket -> median pairwise distance
    n_seasons: int = 0
    spatial_coverage: float = 0.0
    error: str = ''                            # set when the build failed (UI says so)

    def index_of(self, player_id, season_id=None):
        """Row position of (player, season); season None -> the player's
        latest qualifying season. None when absent."""
        m = self.meta
        rows = m.index[m['playerId'] == int(player_id)]
        if season_id is not None:
            rows = [r for r in rows if int(m.at[r, 'seasonId']) == int(season_id)]
            return int(rows[0]) if len(rows) else None
        if not len(rows):
            return None
        return int(max(rows, key=lambda r: (m.at[r, '_year'], m.at[r, 'totalMinutes'])))


def build_pool(frames_by_season, role_features=None, min_minutes=POOL_MIN_MINUTES):
    """frames_by_season: {seasonId: scored stats frame}. role_features: the
    roles layer's season parquet (playerId, seasonId, SPATIAL...) or None."""
    parts = []
    for sid, f in frames_by_season.items():
        if f is None or len(f) == 0 or 'playerId' not in f.columns:
            continue
        missing = [c for c in STYLE_FEATURES + QUALITY_FEATURES if c not in f.columns]
        if missing:
            raise KeyError(f"season {sid}: similarity features missing from the stats frame: {missing}")
        g = f[pd.to_numeric(f['totalMinutes'], errors='coerce').fillna(0) >= min_minutes].copy()
        g['seasonId'] = int(sid)
        # the caches fillna(0): a 0 / NaN competitionId is 'unknown', and every
        # row of a per-season cache belongs to that season's league
        comp = competition_for_season(int(sid))
        if 'competitionId' not in g.columns:
            g['competitionId'] = comp
        else:
            g['competitionId'] = (pd.to_numeric(g['competitionId'], errors='coerce')
                                  .replace(0, np.nan).fillna(comp))
        parts.append(g)
    if not parts:
        return Pool(meta=pd.DataFrame())
    df = pd.concat(parts, ignore_index=True)
    df['playerId'] = pd.to_numeric(df['playerId'], errors='coerce')
    df = df[df['playerId'].notna() & (df['playerId'] > 0)].copy()
    df['playerId'] = df['playerId'].astype(int)
    df['bucket'] = df['primaryPosition'].map(bucket_of)
    df = df[df['bucket'].notna()].drop_duplicates(['playerId', 'seasonId']).reset_index(drop=True)
    spatial_cov = 0.0
    if role_features is not None and len(role_features):
        rf = role_features[['playerId', 'seasonId'] + [c for c in SPATIAL_FEATURES if c in role_features.columns]].copy()
        rf['playerId'] = pd.to_numeric(rf['playerId'], errors='coerce')
        rf = rf[rf['playerId'].notna() & (rf['playerId'] > 0)]
        rf['playerId'] = rf['playerId'].astype(int)
        rf = rf.drop_duplicates(['playerId', 'seasonId'])
        df = df.merge(rf, on=['playerId', 'seasonId'], how='left')
        if SPATIAL_FEATURES[0] in df.columns:
            spatial_cov = float(df[SPATIAL_FEATURES[0]].notna().mean())
    features = [c for c in STYLE_FEATURES + QUALITY_FEATURES + SPATIAL_FEATURES if c in df.columns]
    weights = np.array([QUALITY_WEIGHT if c in QUALITY_FEATURES else 1.0 for c in features])
    df['_year'] = df['seasonId'].map(season_year)
    keep = ['playerId', 'seasonId', 'competitionId', 'playerName', 'teamName', 'primaryPosition',
            'bucket', 'totalMinutes', '_year'] + [c for c in DISPLAY_COLUMNS if c in df.columns]
    meta = df[keep].copy()
    # the caches store 0.0 (not NaN) for player-seasons without an engine row;
    # a rating / value of exactly 0 is never real, so it means 'unrated'
    for c in DISPLAY_COLUMNS:
        if c in meta.columns:
            meta[c] = pd.to_numeric(meta[c], errors='coerce').replace(0, np.nan)
    pool = Pool(meta=meta, features=features, weights=weights,
                n_seasons=int(df['seasonId'].nunique()), spatial_coverage=spatial_cov)
    rng = np.random.default_rng(0)
    for b, idx in df.groupby('bucket').indices.items():
        sub = df.iloc[idx][features].apply(pd.to_numeric, errors='coerce')
        P = sub.rank(pct=True).fillna(0.5)             # within-bucket percentiles
        X = P.values * np.sqrt(weights)
        pool.rows[b] = np.asarray(idx)
        pool.X[b] = X
        n = len(idx)
        if n >= 3:
            samp = rng.choice(n, size=min(n, ECDF_SAMPLE), replace=False)
            S = X[samp]
            D = np.sqrt(((S[:, None, :] - S[None, :, :]) ** 2).sum(-1))
            tri = np.sort(D[np.triu_indices(len(samp), k=1)])
            pool.ecdf[b] = tri
            pool.median_d[b] = float(np.median(tri))
    return pool


def _score(pool, bucket, d):
    """100 = identical, 50 = a typical pair of this bucket, 0 = far."""
    med = pool.median_d.get(bucket)
    if not med:
        return np.full_like(d, np.nan, dtype=float)
    return np.clip(100.0 - 50.0 * np.asarray(d, dtype=float) / med, 0.0, 100.0)


def _closer_than(pool, bucket, d):
    """Share of this bucket's pairs that are FURTHER apart than d (0-100)."""
    tri = pool.ecdf.get(bucket)
    if tri is None or not len(tri):
        return np.full_like(d, np.nan, dtype=float)
    return 100.0 * (1.0 - np.searchsorted(tri, np.asarray(d, dtype=float), side='right') / len(tri))


def neighbours(pool, player_id, season_id=None, k=10, profile='latest', league=None,
               min_minutes=None, min_year=None, exclude_team=None, max_age=None,
               ages=None, exclude_player_ids=(), rating_abs_min=None, rating_abs_max=None):
    """Ranked like-for-like table for one query row.

    profile: 'latest' -> each candidate player is represented by his latest
             qualifying season IN THIS BUCKET (a target list) — chosen BEFORE
             the filters, so a player whose current season fails a filter
             (own club, league, minutes...) drops out rather than resurfacing
             under an older season at another club; 'any' -> every player-season.
    Filters are applied AFTER the distance (the vectors never change with a
    filter): league (competitionId), min_minutes (row minutes), min_year
    (season start year), exclude_team (teamName), max_age with an
    {playerId: age} lookup (unknown ages pass), exclude_player_ids,
    rating_abs_min / rating_abs_max on 'ACP Rating (abs)' — the league-
    absolute rating (rows without one are dropped when the filter is on).
    Returns a DataFrame with rank, similarity (0-100), distance, and the
    meta columns; empty when the query has no qualifying row."""
    qi = pool.index_of(player_id, season_id)
    if qi is None or pool.meta.empty:
        return pd.DataFrame()
    b = pool.meta.at[qi, 'bucket']
    rows, X = pool.rows[b], pool.X[b]
    qpos = int(np.where(rows == qi)[0][0])
    d = np.sqrt(((X - X[qpos]) ** 2).sum(-1))
    cand = pool.meta.iloc[rows].copy()
    cand['distance'] = d
    cand['similarity'] = _score(pool, b, d)
    cand['closer_than_pct'] = _closer_than(pool, b, d)
    cand['_row'] = rows
    cand = cand[cand['playerId'] != int(player_id)]
    if profile == 'latest':
        cand = (cand.sort_values(['playerId', '_year', 'totalMinutes'])
                    .drop_duplicates('playerId', keep='last'))
    if exclude_player_ids:
        cand = cand[~cand['playerId'].isin([int(p) for p in exclude_player_ids])]
    if league is not None:
        cand = cand[pd.to_numeric(cand['competitionId'], errors='coerce') == int(league)]
    if min_minutes:
        cand = cand[cand['totalMinutes'] >= min_minutes]
    if min_year:
        cand = cand[cand['_year'] >= int(min_year)]
    if exclude_team:
        cand = cand[cand['teamName'].astype(str) != str(exclude_team)]
    if max_age is not None and ages:
        def _ok(pid):
            a = ages.get(int(pid))
            return True if a is None or (isinstance(a, float) and math.isnan(a)) else float(a) <= max_age
        cand = cand[cand['playerId'].map(_ok)]
    if (rating_abs_min is not None or rating_abs_max is not None):
        if 'ACP Rating (abs)' in cand.columns:
            r = pd.to_numeric(cand['ACP Rating (abs)'], errors='coerce')
            ok = r.notna()
            if rating_abs_min is not None:
                ok &= r >= float(rating_abs_min)
            if rating_abs_max is not None:
                ok &= r <= float(rating_abs_max)
            cand = cand[ok]
        else:
            cand = cand.iloc[0:0]
    cand = cand.sort_values(['distance', 'totalMinutes', 'playerId'],
                            ascending=[True, False, True], kind='stable').head(k)
    cand.insert(0, 'rank', range(1, len(cand) + 1))
    cand['query_row'] = qi
    return cand.reset_index(drop=True)


def explain(pool, row_a, row_b, n_alike=3, n_differ=2):
    """Plain-language shared traits and differences for two pool rows
    (meta positions). Percentiles are the within-bucket ones the distance
    used, so the sentence is literally what drove it."""
    b = pool.meta.at[row_a, 'bucket']
    if b != pool.meta.at[row_b, 'bucket']:
        return {'alike': [], 'differs': []}
    rows, X = pool.rows[b], pool.X[b]
    P = X / np.sqrt(pool.weights)     # back to raw percentiles
    pa = P[int(np.where(rows == row_a)[0][0])]
    pb = P[int(np.where(rows == row_b)[0][0])]
    alike, differs = [], []
    for j, f in enumerate(pool.features):
        a, c = float(pa[j]), float(pb[j])
        same_side = (a - 0.5) * (c - 0.5) > 0
        strength = min(abs(a - 0.5), abs(c - 0.5))
        if same_side and strength >= 0.15 and abs(a - c) <= 0.15:
            alike.append((f, a, c, strength * (1 - abs(a - c))))
        if abs(a - c) >= 0.35:
            differs.append((f, a, c, abs(a - c)))
    alike.sort(key=lambda t: -t[3]); differs.sort(key=lambda t: -t[3])

    def _pct(p):
        # 'high' / 'low' are about the raw value, never praise: for ball
        # losses or getting dribbled past, 'high' is the bad end
        return (f"high, top {max(1, round((1 - p) * 100))}%" if p >= 0.5
                else f"low, bottom {max(1, round(p * 100))}%")
    # the weaker of the two claims, so 'both …' is literally true for both
    out_alike = [f"{FEATURE_LABELS.get(f, f)} (both {_pct(min(a, c) if a >= 0.5 else max(a, c))})"
                 for f, a, c, _ in alike[:n_alike]]
    out_diff = [f"{FEATURE_LABELS.get(f, f)} ({_pct(a)} vs {_pct(c)})" for f, a, c, _ in differs[:n_differ]]
    return {'alike': out_alike, 'differs': out_diff}


def validation(pool, labels=None, k=10):
    """Label-free and label-based checks a scout can read:
    self_match: for every player-season with another qualifying season,
      rank of that other season among same-bucket rows of OTHER seasons —
      share found in the top-k and the median rank (chance ~ k / pool);
    role/style agreement: share of the top-k neighbours (other players)
      carrying the query's engine role / style label vs the pool share.
    labels: optional frame (playerId, seasonId, role, style)."""
    m = pool.meta
    lab = None
    if labels is not None and len(labels):
        lab = labels.drop_duplicates(['playerId', 'seasonId']).set_index(['playerId', 'seasonId'])
    hits, ranks, chance = [], [], []
    agree = {'role': [], 'style': []}; base = {'role': [], 'style': []}
    for b, rows in pool.rows.items():
        X = pool.X[b]
        pid = m.iloc[rows]['playerId'].values; sid = m.iloc[rows]['seasonId'].values
        D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
        lr = ls = None
        if lab is not None:
            keys = list(zip(pid, sid))
            lr = np.array([lab['role'].get(kk) if 'role' in lab.columns else None for kk in keys], dtype=object)
            ls = np.array([lab['style'].get(kk) if 'style' in lab.columns else None for kk in keys], dtype=object)
        for i in range(len(rows)):
            others = np.where((pid == pid[i]) & (sid != sid[i]))[0]
            mask = sid != sid[i]
            if len(others) and mask.sum() > k:
                order = np.argsort(D[i][mask], kind='stable'); cand = np.where(mask)[0][order]
                pos = np.where(np.isin(cand, others))[0]
                if len(pos):
                    hits.append(pos.min() < k); ranks.append(int(pos.min()) + 1); chance.append(k / mask.sum())
            mask2 = pid != pid[i]
            order = np.argsort(D[i][mask2], kind='stable'); top = np.where(mask2)[0][order][:k]
            for name, arr in (('role', lr), ('style', ls)):
                if arr is None or not isinstance(arr[i], str):
                    continue
                t = arr[top]; t = t[[isinstance(x, str) for x in t]]
                if len(t):
                    agree[name].append(float((t == arr[i]).mean()))
                    base[name].append(float((arr[mask2] == arr[i]).mean()))
    out = {'k': k, 'n_rows': int(len(m)), 'n_players': int(m['playerId'].nunique()),
           'n_seasons': pool.n_seasons, 'spatial_coverage': pool.spatial_coverage,
           'self_match': {'n': len(hits), 'top_k_rate': (float(np.mean(hits)) if hits else None),
                          'median_rank': (int(np.median(ranks)) if ranks else None),
                          'chance_top_k': (float(np.mean(chance)) if chance else None)}}
    for name in ('role', 'style'):
        out[f'{name}_agreement'] = {'n': len(agree[name]),
                                    'top_k_share': (float(np.mean(agree[name])) if agree[name] else None),
                                    'chance': (float(np.mean(base[name])) if base[name] else None)}
    return out
