"""A player's career as ONE continuous sample, for prediction (Lucas
2026-10-29: "when we are predicting something in the future for a player we
should use all of their data … as a continuous entity"; seasons stay the unit
for DESCRIPTION).

Each player-season row is a BLOCK placed at its date — the minute-weighted
date of the matches in it — and weighted by how long before the as-of time it
was played, not by how many season boundaries lie in between:

  w(block)        = minutes x 0.5 ** ((t_ref - centroid) / H)
  career rating   = sum(w x rating) / sum(w)                        H = H_FORM
  evidence        = sum(minutes x 0.5 ** ((t_ref - centroid) / H_EVID))

H_FORM (1 year) and H_EVID (6.6 years) are the old per-season decays (0.5 and
0.9 per season) made continuous — fixed in advance, not tuned on the replays
(pre-registered 2026-10-29). A block uses only seasons up to the row's own and
matches up to t_ref (no look-ahead).

Times are float years since 1970-01-01 (to_years). Campeonato seasons have no
per-match minutes (exact season totals only — Lucas's exact-minutes rule), so
a Campeonato block sits at its season's median match date.
"""
import math

import numpy as np
import pandas as pd

H_FORM = 1.0
H_EVID = math.log(0.5) / math.log(0.9)      # 6.58 years
_EPOCH = pd.Timestamp('1970-01-01')


def to_years(ts):
    """Timestamp / datetime Series -> float years since 1970."""
    if isinstance(ts, pd.Series):
        return (pd.to_datetime(ts) - _EPOCH).dt.days / 365.25
    return (pd.Timestamp(ts) - _EPOCH).days / 365.25


def match_dates(matches_summary):
    """matchId -> float years, for played matches."""
    ms = matches_summary
    if 'status' in ms.columns:
        ms = ms[ms['status'].eq('Played')]
    d = pd.to_datetime(ms['dateutc'], errors='coerce')
    if getattr(d.dt, 'tz', None) is not None:
        d = d.dt.tz_localize(None)
    return pd.Series(to_years(d).to_numpy(), index=ms['matchId'].astype(int).to_numpy())


def block_centroids(match_minutes, matches_summary, upto=None):
    """(playerId, seasonId) -> minute-weighted date (float years) of his
    matches, from per-match minutes (playerId, matchId, minutes), matches up
    to `upto` (float years) only; plus seasonId -> median match date, the
    fallback for seasons without per-match minutes."""
    md = match_dates(matches_summary)
    ms = matches_summary.assign(t=matches_summary['matchId'].astype(int).map(md))
    season_mid = ms.groupby('seasonId')['t'].median()
    mm = match_minutes[pd.to_numeric(match_minutes['minutes'], errors='coerce') > 0]
    mm = mm.assign(t=mm['matchId'].astype(int).map(md)).dropna(subset=['t'])
    mm = mm.merge(ms[['matchId', 'seasonId']].astype({'matchId': int}), on='matchId', how='inner')
    if upto is not None:
        mm = mm[mm['t'] <= upto + 1e-9]
    g = mm.assign(wt=mm['t'] * mm['minutes']).groupby(['playerId', 'seasonId'])[['wt', 'minutes']].sum()
    return (g['wt'] / g['minutes']).rename('centroid'), season_mid


def career_features(rows, centroid, season_mid, t_ref, h_form=H_FORM, h_evid=H_EVID):
    """Career rating and evidence for every row as of its own t_ref.

    rows: DataFrame with playerId, seasonId, yr, mins_played, acp_rating (one
    row per player-season block); t_ref: Series aligned to rows (float years).
    Returns DataFrame(career_asof, evidence) aligned to rows."""
    c = [centroid.get((int(p), int(s)), np.nan) for p, s in zip(rows['playerId'], rows['seasonId'])]
    b = rows.assign(_c=pd.Series(c, index=rows.index).fillna(rows['seasonId'].map(season_mid)),
                    _t=t_ref)
    out_car = pd.Series(np.nan, index=rows.index)
    out_ev = pd.Series(np.nan, index=rows.index)
    for _, g in b.groupby('playerId', sort=False):
        yr, cen = g['yr'].to_numpy(), g['_c'].to_numpy()
        mins, rat = g['mins_played'].to_numpy(float), g['acp_rating'].to_numpy(float)
        for i, (y_i, t_i) in zip(g.index, zip(yr, g['_t'].to_numpy())):
            use = (yr <= y_i) & np.isfinite(rat) & np.isfinite(cen)
            if not use.any():
                continue
            age = np.clip(t_i - cen[use], 0.0, None)
            wf = mins[use] * 0.5 ** (age / h_form)
            if wf.sum() > 0:
                out_car[i] = float((wf * rat[use]).sum() / wf.sum())
            out_ev[i] = float((mins[use] * 0.5 ** (age / h_evid)).sum())
    return pd.DataFrame({'career_asof': out_car, 'evidence': out_ev})



def time_weighted(rows, values, centroid, season_mid, t_ref, h=H_FORM):
    """Minute- and time-weighted mean of every column of `values` (aligned to
    rows) over each row's career blocks up to its own t_ref: career_asof's
    blocks and weights, for any per-block quantity (role shares, the role
    played). A row with no usable block gets NaN."""
    c = [centroid.get((int(p), int(s)), np.nan) for p, s in zip(rows['playerId'], rows['seasonId'])]
    b = rows.assign(_c=pd.Series(c, index=rows.index).fillna(rows['seasonId'].map(season_mid)),
                    _t=t_ref)
    V = values.to_numpy(dtype=float)
    out = np.full(V.shape, np.nan)
    pos = pd.Series(np.arange(len(rows)), index=rows.index)
    for _, g in b.groupby('playerId', sort=False):
        k = pos[g.index].to_numpy()
        yr, cen = g['yr'].to_numpy(), g['_c'].to_numpy()
        mins, v = g['mins_played'].to_numpy(float), V[k]
        ok = np.isfinite(cen) & np.isfinite(v).all(axis=1)
        for j, (y_i, t_i) in enumerate(zip(yr, g['_t'].to_numpy())):
            use = (yr <= y_i) & ok
            if not use.any():
                continue
            w = mins[use] * 0.5 ** (np.clip(t_i - cen[use], 0.0, None) / h)
            if w.sum() > 0:
                out[k[j]] = (w[:, None] * v[use]).sum(axis=0) / w.sum()
    return pd.DataFrame(out, index=rows.index, columns=values.columns)


def career_role(rows, centroid, season_mid, t_ref, h=H_FORM):
    """The role a player has played most up to each row's t_ref: the largest
    time-weighted minutes over his career blocks, each block counting for its
    own season's role (a one-season career: that season's role; ties go to
    the alphabetically first role)."""
    roles = sorted(rows['role'].dropna().unique())
    ind = pd.DataFrame({r: (rows['role'] == r).astype(float) for r in roles}, index=rows.index)
    tw = time_weighted(rows, ind, centroid, season_mid, t_ref, h)
    has = tw.notna().any(axis=1)
    best = pd.Series(rows['role'], index=rows.index, dtype=object)
    best[has] = tw[has].idxmax(axis=1)
    return best


def season_end_times(matches_summary, season_year):
    """season-year -> float years of the last played match in it (both
    leagues): the as-of time of a completed season's rows."""
    md = match_dates(matches_summary)
    ms = matches_summary.assign(t=matches_summary['matchId'].astype(int).map(md),
                                yr=matches_summary['seasonId'].map(season_year))
    return ms.dropna(subset=['t', 'yr']).groupby('yr')['t'].max()
