"""Load + harmonize the valuations.parquet sources into a clean
per-(player, season) view for the CVI calibration model.

Designed so future model code can do:

    from valuations.load_valuations import latest_valuation_per_player
    df = latest_valuation_per_player()
    # → columns: playerId, value_eur, source_weighted_avg, n_sources, ...

This module deliberately makes NO assumptions about which source is
authoritative — the calibration regression decides that, either by
weighting sources or by using source as a feature.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

VAL_DIR = Path(__file__).resolve().parent
VALUATIONS_PATH    = VAL_DIR / 'valuations.parquet'
REPORTED_FEES_PATH = VAL_DIR / 'reported_fees.csv'
MANUAL_ENTRIES_PATH = VAL_DIR / 'manual_entries.csv'

# Source authority weights — used when computing the
# `source_weighted_avg` convenience column. The model may override.
DEFAULT_SOURCE_WEIGHTS = {
    'manual':        4.0,  # club/agent-level info — highest authority
    'reported_fee':  3.0,  # actual paid fee
    'transfermarkt': 1.0,  # crowd-edited proxy
    'zerozero':      1.0,  # crowd-edited proxy
}


def load_all_valuations() -> pd.DataFrame:
    """Combine all sources into a single long-format DataFrame.

    Columns: playerId, source, value_eur, as_of_date, season_id,
    source_url, notes.
    """
    frames = []
    # 1) Parquet (TM + ZZ from scrapers)
    if VALUATIONS_PATH.exists():
        try:
            frames.append(pd.read_parquet(VALUATIONS_PATH))
        except Exception:
            pass
    # 2) Reported fees CSV
    if REPORTED_FEES_PATH.exists():
        try:
            r = pd.read_csv(REPORTED_FEES_PATH, comment='#')
            r = r.rename(columns={'fee_eur': 'value_eur'})
            r['source'] = 'reported_fee'
            r['source_url'] = r.get('source_url', None)
            r = r[r['value_eur'].notna() & (r['value_eur'] > 0)]
            frames.append(r[['playerId', 'source', 'value_eur',
                              'as_of_date', 'season_id',
                              'source_url', 'notes']])
        except Exception:
            pass
    # 3) Manual entries CSV
    if MANUAL_ENTRIES_PATH.exists():
        try:
            m = pd.read_csv(MANUAL_ENTRIES_PATH, comment='#')
            m['source'] = 'manual'
            frames.append(m)
        except Exception:
            pass
    if not frames:
        return pd.DataFrame(columns=['playerId', 'source', 'value_eur',
                                      'as_of_date', 'season_id',
                                      'source_url', 'notes'])
    out = pd.concat(frames, ignore_index=True, sort=False)
    # Normalize types
    out['playerId'] = pd.to_numeric(out['playerId'], errors='coerce').astype('Int64')
    out['value_eur'] = pd.to_numeric(out['value_eur'], errors='coerce')
    out['season_id'] = pd.to_numeric(out.get('season_id'), errors='coerce').astype('Int64')
    out = out.dropna(subset=['playerId', 'value_eur'])
    return out


def latest_valuation_per_player(
    source_weights: dict | None = None,
) -> pd.DataFrame:
    """Collapse to one row per playerId: weighted avg of most-recent
    per-source values. Useful when the model wants a single Y-target."""
    src_w = source_weights or DEFAULT_SOURCE_WEIGHTS
    long = load_all_valuations()
    if long.empty:
        return pd.DataFrame()
    # Most-recent per (player, source)
    long['_d'] = pd.to_datetime(long['as_of_date'], errors='coerce')
    long = long.sort_values('_d')
    latest_per_source = (long.dropna(subset=['_d'])
                              .groupby(['playerId', 'source'], as_index=False)
                              .tail(1))
    latest_per_source['_w'] = latest_per_source['source'].map(src_w).fillna(1.0)
    grouped = latest_per_source.groupby('playerId')
    out = grouped.apply(
        lambda g: pd.Series({
            'value_eur_weighted_avg':
                (g['value_eur'] * g['_w']).sum() / g['_w'].sum(),
            'n_sources': len(g),
            'sources': ','.join(sorted(g['source'].unique())),
            'most_recent_date': g['_d'].max().date().isoformat()
                                  if g['_d'].notna().any() else None,
        })
    ).reset_index()
    return out
