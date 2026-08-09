#!/usr/bin/env python3
"""Train the v2 EUR transfer-value regression.

Pipeline:
  1. Load TM market-value snapshots (valuations/valuations.parquet)
     + TM player metadata + the per-(player, season) GPA values.
  2. For each (playerId, seasonId) where player is matched to TM,
     find the TM snapshot closest to the season's MIDPOINT and use
     log(value_eur) as the target.
  3. Compute features for that (player, season):
       - log(CVI) — the on-pitch quality kernel from CVI v2.7
       - position_group ONE-HOT (GK, CB, FB, CM, AM_WG, ST)
            → lets the model learn cross-position price premiums
              (wingers > CBs at the same CVI is what scouts see)
       - age_at_season
       - league_factor (1.0 L3, 0.85 Camp)
       - log(career_mins_to_date)
       - xG / xA per-90 residuals (career-cumulative)
       - n_seasons_played (cardinality, not position cardinality)
       - passport_PT (binary: Portuguese passport vs other)
       - team_opta_rating (current team strength, when available)
       - season_year (calendar year — captures macro market drift)
  4. Standardize numeric features, fit Ridge regression with
     cross-validated alpha. Ridge handles small-sample + correlated
     features gracefully.
  5. Save model.pkl + coefficient table + diagnostics to models/eur_v2/

Usage:
  python train_eur_v2.py            # train + save
  python train_eur_v2.py --dry      # just print diagnostics
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Import the CVI pipeline + helpers from the dashboard's app.py
# We import the bits we need by reading the module in a controlled way.
# This avoids spinning up Streamlit just to call the functions.


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--dry', action='store_true',
                     help='Print diagnostics without writing model')
    ap.add_argument('--min-mins', type=int, default=500,
                     help='Min minutes per season to include in training')
    ap.add_argument('--eur-min', type=int, default=25_000,
                     help='Min EUR to include (filters noise-floor entries)')
    ap.add_argument('--eur-max', type=int, default=5_000_000,
                     help='Max EUR to include (filters Sporting/Benfica academy '
                           'outliers that distort the model for Liga 3/Camp use)')
    args = ap.parse_args()

    print("=" * 60)
    print("v2 EUR REGRESSION — TRAINING PIPELINE")
    print("=" * 60)

    # --- Load data sources ---
    # Use load_all_valuations() so we pick up BOTH the TM scrape AND
    # the hand-curated reported_fees.csv (and manual_entries.csv if
    # present). Previously the trainer read valuations.parquet
    # directly and missed the 8 user transfers entirely.
    sys.path.insert(0, str(HERE))
    from valuations.load_valuations import load_all_valuations
    vals = load_all_valuations()
    print(f"\n[data] All valuation sources: {len(vals):,} rows")
    print(f"  by source: {vals['source'].value_counts().to_dict()}")
    # Coerce types — load_all_valuations may keep playerId as Int64
    vals['playerId'] = pd.to_numeric(vals['playerId'], errors='coerce').astype('int64')
    vals['value_eur'] = pd.to_numeric(vals['value_eur'], errors='coerce')

    gpa = pd.read_parquet(HERE / 'gpa_player_season_values.parquet')
    print(f"[data] GPA per-(player, season): {len(gpa):,} rows for "
           f"{gpa['playerId'].nunique()} players")

    # Player details for age/passport/birth area
    details = pd.read_pickle(HERE / 'player_details.pkl')
    if isinstance(details, list):
        details = pd.DataFrame(details)
    print(f"[data] player_details: {len(details):,} players")

    # TM metadata for richer features when available
    tm_meta_path = HERE / 'valuations' / 'tm_player_metadata.parquet'
    if tm_meta_path.exists():
        tm_meta = pd.read_parquet(tm_meta_path)
        print(f"[data] TM metadata: {len(tm_meta):,} players")
    else:
        tm_meta = pd.DataFrame()
        print(f"[data] TM metadata: not found")

    # Opta team strength
    opta_path = HERE / 'opta_ratings.parquet'
    if opta_path.exists():
        opta = pd.read_parquet(opta_path)
        print(f"[data] Opta team ratings: {len(opta):,} clubs")
    else:
        opta = pd.DataFrame()

    # --- Tier matcher (handles B-teams + distinctive-token logic) ---
    from tier_matcher import build_tier_matcher, at_our_tier as _at_our_tier
    build_tier_matcher()
    print(f"[tier] Strict tier matcher loaded")
    import re as _re

    # --- Map seasons → dates ---
    SEASON_MIDPOINT = {
        # Approximate Dec 15 of the year covering the bulk of matches
        188221: '2021-12-15',
        188222: '2022-12-15',
        189147: '2023-12-15',
        190090: '2024-12-15',
        191782: '2025-12-15',
        192831: '2026-12-15',
        # Campeonato
        190230: '2023-12-15',
        191779: '2025-12-15',
        192925: '2026-12-15',
    }
    SEASON_LEAGUE_FACTOR = {
        188221: 1.00, 188222: 1.00, 189147: 1.00,
        190090: 1.00, 191782: 1.00, 192831: 1.00,  # Liga 3
        190230: 0.85, 191779: 0.85, 192925: 0.85,  # Campeonato
    }
    SEASON_YEAR = {
        188221: 2021, 188222: 2022, 189147: 2023,
        190090: 2024, 191782: 2025, 192831: 2026,
        190230: 2023, 191779: 2025, 192925: 2026,
    }
    # v3.3 — season date windows. The TIGHT job of "only count value
    # while at Liga 3" is done by the per-season-club match below
    # (snapshot's club_at_time must match the player's actual GPA-
    # season club). We allow a generous ±9-month window around the
    # season midpoint so we keep snapshots from late-pre-season and
    # early-post-season, both of which still reflect that tenure's
    # market value. Reported_fee/manual bypass this window — they're
    # trusted as-tier transfers regardless of date relative to a
    # specific GPA season.
    SEASON_WINDOW = {
        188221: ('2021-03-01', '2022-12-31'),
        188222: ('2022-03-01', '2023-12-31'),
        189147: ('2023-03-01', '2024-12-31'),
        190090: ('2024-03-01', '2025-12-31'),
        191782: ('2025-03-01', '2026-12-31'),
        192831: ('2026-03-01', '2027-12-31'),
        190230: ('2023-03-01', '2024-12-31'),
        191779: ('2025-03-01', '2026-12-31'),
        192925: ('2026-03-01', '2027-12-31'),
    }

    # --- Build training rows ---
    print(f"\n[build] Joining GPA seasons to in-season TM snapshots...")
    matched_pids = set(vals['playerId'].unique())
    gpa = gpa[gpa['playerId'].isin(matched_pids)].copy()
    print(f"[build] GPA rows for TM-matched players: {len(gpa):,}")

    # v3.3 — derive (player, season) → actual tier-club names from
    # match data. We use this to verify the TM snapshot's
    # club_at_time matches the player's real club for THAT season,
    # not just any tier club they were ever at.
    print(f"[build] Building (player × season) club map from raw events...")
    try:
        ev_cols = ['matchId', 'team.name', 'player.id']
        ev = pd.read_parquet(HERE / 'raw_events.parquet', columns=ev_cols)
        ms = pd.read_parquet(HERE / 'matches_summary.parquet',
                                columns=['matchId', 'seasonId'])
        ev = ev.merge(ms, on='matchId', how='left')
        ev = ev.dropna(subset=['team.name', 'player.id', 'seasonId'])
        ev['player.id'] = ev['player.id'].astype(int)
        ev['seasonId'] = ev['seasonId'].astype(int)
        # one row per (player, season, team) with appearance count
        ps_clubs = (ev.groupby(['player.id', 'seasonId', 'team.name'])
                      .size().reset_index(name='n')
                      .rename(columns={'player.id': 'playerId'}))
        # Build dict: (pid, sid) → list of clubs they actually appeared
        # for that season (sorted by appearances)
        ps_club_map = {}
        for (pid, sid), grp in ps_clubs.groupby(['playerId', 'seasonId']):
            clubs = grp.sort_values('n', ascending=False)['team.name'].tolist()
            ps_club_map[(pid, sid)] = clubs
        print(f"[build] (player × season) club map: {len(ps_club_map):,} entries")
    except Exception as e:
        print(f"[build] WARN: couldn't build player-season club map ({e}); "
               f"falling back to tier-only filter")
        ps_club_map = {}

    # pull the normalize helper directly off tier_matcher
    from tier_matcher import _normalize as _normalize_club

    def _club_matches(snapshot_club, season_clubs):
        """True if the snapshot's club_at_time is one of the clubs the
        player actually appeared for during this GPA season. Uses the
        tier_matcher's normalization for tolerant matching."""
        if not snapshot_club or not season_clubs:
            return False
        snap_id, snap_b = _normalize_club(snapshot_club)
        for sc in season_clubs:
            sc_id, sc_b = _normalize_club(sc)
            if snap_b != sc_b:
                continue
            if not snap_id or not sc_id:
                continue
            # Exact identity OR bidirectional subset (matches
            # 'Vit. Setúbal' ↔ 'Vitória Setúbal FC' style abbreviations)
            if snap_id == sc_id:
                return True
            if snap_id.issubset(sc_id) or sc_id.issubset(snap_id):
                return True
        return False

    # Filter to minimum minutes
    mins_col = next((c for c in ('mins_played', 'totalMinutes', 'Minutes')
                      if c in gpa.columns), None)
    if mins_col:
        gpa = gpa[gpa[mins_col] >= args.min_mins].copy()
        print(f"[build] After {args.min_mins}-min filter: {len(gpa):,}")

    # For each (player, season), find closest TM snapshot.
    # CRITICAL: only consider snapshots where the player was AT a
    # Liga 3 / Camp club at the time — otherwise we'd pull in
    # later-career snapshots (player moved to Primeira Liga or
    # abroad) that reflect a different market tier.
    vals['_d'] = pd.to_datetime(vals['as_of_date'], errors='coerce')
    def _extract_club(notes):
        if not isinstance(notes, str): return None
        m = _re.search(r'club_at_time=(.+?)$', notes)
        return m.group(1).strip() if m else None
    vals['_club_at_time'] = vals['notes'].apply(_extract_club)
    # TM snapshots: require club_at_time to be at our tier.
    # Reported_fee + manual: trust the user, always include (they
    # explicitly listed Liga 3 / Camp transfers).
    vals['_at_our_tier'] = vals.apply(
        lambda r: (True if r['source'] in ('reported_fee', 'manual')
                    else _at_our_tier(r['_club_at_time'])),
        axis=1,
    )
    n_tier = int(vals['_at_our_tier'].sum())
    print(f"[tier] At-our-tier valuations (TM by club + all user-entered): "
           f"{n_tier:,} of {len(vals):,} ({n_tier/len(vals)*100:.0f}%)")
    print(f"  by source: "
           f"{vals[vals['_at_our_tier']]['source'].value_counts().to_dict()}")
    vals_tier = vals[vals['_at_our_tier']].copy()
    # v3.6 — main loop ONLY pairs TM snapshots to seasons (nearest-by-
    # midpoint). reported_fees + manual entries are excluded here and
    # added below via force-include using their CSV-supplied season_id.
    # This guarantees each transfer is paired with its PRE-transfer
    # season — fixes the post-transfer-contamination bug where Juan
    # Perea's Jul 2024 fee was paired with 24-25 GPA (post-Bulgaria
    # move) instead of his 23-24 Académica season.
    vals_tm_only = vals_tier[
        ~vals_tier['source'].isin(['reported_fee', 'manual'])
    ].copy()
    rows = []
    n_no_snaps = 0
    for _, r in gpa.iterrows():
        pid = int(r['playerId'])
        sid = int(r['seasonId'])
        mid_str = SEASON_MIDPOINT.get(sid)
        if mid_str is None:
            continue
        mid = pd.to_datetime(mid_str)
        # v3.4 — use ALL tier-tagged TM snapshots, not just those in
        # the exact season window. The strict v3.3 filter cut the
        # training set to 127 rows and over-shrank the regression
        # (chosen alpha pinned at 100). With 25x sample weight on
        # reported_fees we lean on the 8 paid transfers instead of
        # needing the snapshot to be temporally precise. Keep a
        # generous 18-month nearest-snapshot cap so we don't pair
        # a snapshot from 3 years before the GPA season.
        snaps = vals_tm_only[(vals_tm_only['playerId'] == pid)
                            & (vals_tm_only['value_eur'].notna())
                            & (vals_tm_only['value_eur'] > 0)].copy()
        if snaps.empty:
            n_no_snaps += 1
            continue
        snaps['_diff'] = (snaps['_d'] - mid).abs()
        snaps = snaps.sort_values('_diff')
        # Reject if no snapshot within 18 months — too stale
        if snaps.iloc[0]['_diff'] > pd.Timedelta(days=540):
            n_no_snaps += 1
            continue
        best = snaps.iloc[0]
        rows.append({
            'playerId': pid,
            'seasonId': sid,
            'mins_played': r.get(mins_col, 0),
            'primaryPosition': r.get('position', r.get('primaryPosition')),
            'Total Value': r.get('Total Value', r.get('total_v_per_90')),
            'value_eur': float(best['value_eur']),
            'mv_snapshot_date': best['_d'].date().isoformat(),
            # Date of the actual valuation/transfer — used to compute
            # age correctly (vs old behaviour of using season midpoint).
            '_snap_date': best['_d'],
            'value_source': best.get('source', 'transfermarkt'),
            'season_year': SEASON_YEAR.get(sid, 2024),
            'league_factor': SEASON_LEAGUE_FACTOR.get(sid, 1.0),
            'days_to_snapshot': int(best['_diff'].days),
        })

    # ALSO ensure every reported_fee / manual transfer gets at least one
    # training row, even if the player's nearest-snapshot logic picked a
    # different TM record or the player is below the 500-min filter for
    # the season the transfer landed in. These are highest-authority data.
    user_rows = []
    user_vals = vals_tier[vals_tier['source'].isin(['reported_fee', 'manual'])].copy()
    if not user_vals.empty:
        gpa_all = pd.read_parquet(HERE / 'gpa_player_season_values.parquet')
        for _, uv in user_vals.iterrows():
            pid = int(uv['playerId'])
            xfer_date = uv['_d']
            # v3.6 — pre-transfer perf pairing.
            # The CSV's season_id field is treated as authoritative —
            # the user knows whether the transfer's pre-transfer perf
            # comes from the season just finished (summer move) or the
            # season currently in progress at transfer time (winter
            # move, where post-transfer data was at a different tier
            # and isn't in our GPA anyway).
            # If season_id is not set in the CSV, we fall back to:
            #   "latest season whose midpoint is BEFORE the transfer date"
            # which gives correct behaviour for summer transfers
            # (was previously broken — would pick the upcoming season
            # instead of the one just finished).
            gpa_pl = gpa_all[gpa_all['playerId'] == pid]
            if gpa_pl.empty:
                print(f"  [user-row skip] pid {pid}: no GPA rows")
                continue
            best_gpa = None
            csv_sid = uv.get('season_id')
            if csv_sid is not None and pd.notna(csv_sid):
                csv_sid = int(csv_sid)
                match = gpa_pl[gpa_pl['seasonId'].astype(int) == csv_sid]
                if not match.empty:
                    best_gpa = match.iloc[0]
                else:
                    print(f"  [user-row skip] pid {pid}: CSV season_id "
                           f"{csv_sid} has no GPA row")
                    continue
            else:
                # Auto-pick: latest season midpoint BEFORE transfer date
                for _, g in gpa_pl.iterrows():
                    sid = int(g['seasonId'])
                    if sid not in SEASON_MIDPOINT: continue
                    mid = pd.to_datetime(SEASON_MIDPOINT[sid])
                    if mid > xfer_date:
                        continue
                    if best_gpa is None or mid > pd.to_datetime(SEASON_MIDPOINT[int(best_gpa['seasonId'])]):
                        best_gpa = g
            if best_gpa is None:
                print(f"  [user-row skip] pid {pid}: no GPA season "
                       f"with midpoint before transfer date {xfer_date.date()}")
                continue
            sid_int = int(best_gpa['seasonId'])
            # Skip if main loop already added this (pid, seasonId)
            already_present = any(
                r['playerId'] == pid and r['seasonId'] == sid_int
                for r in rows
            )
            if already_present: continue
            best_dist = pd.Timedelta(days=0)
            user_rows.append({
                'playerId': pid,
                'seasonId': sid_int,
                'mins_played': best_gpa.get(mins_col, 0),
                'primaryPosition': best_gpa.get('position', best_gpa.get('primaryPosition')),
                'Total Value': best_gpa.get('Total Value',
                                              best_gpa.get('total_v_per_90')),
                'value_eur': float(uv['value_eur']),
                'mv_snapshot_date': xfer_date.date().isoformat(),
                '_snap_date': xfer_date,
                'value_source': uv['source'],
                'season_year': SEASON_YEAR.get(sid_int, 2024),
                'league_factor': SEASON_LEAGUE_FACTOR.get(sid_int, 1.0),
                'days_to_snapshot': int(best_dist.days),
            })
        if user_rows:
            print(f"[build] Force-included {len(user_rows)} user-reported "
                   f"transfer rows that the nearest-snapshot join missed")
    rows.extend(user_rows)
    train = pd.DataFrame(rows)
    # v3.3 — dedupe: when overlapping season windows pair the SAME
    # snapshot with two adjacent seasons, keep only the row whose
    # season midpoint is closest to the snapshot date. This avoids
    # double-counting Juan Perea's €100k fee across seasons 189147
    # AND 190090 just because his Jul 2024 transfer falls in both
    # windows.
    if len(train):
        train['_mid'] = train['seasonId'].map(
            {k: pd.to_datetime(v) for k, v in SEASON_MIDPOINT.items()})
        train['_dist_to_mid'] = (train['_snap_date'] - train['_mid']).abs()
        before = len(train)
        train = (train.sort_values('_dist_to_mid')
                       .drop_duplicates(subset=['playerId', '_snap_date',
                                                 'value_eur'], keep='first')
                       .drop(columns=['_mid', '_dist_to_mid']))
        if before > len(train):
            print(f"[v3.3 dedupe] dropped {before - len(train)} duplicate "
                   f"(player × snapshot) rows from overlapping windows")
    print(f"[v3.4 filter] dropped {n_no_snaps:,} GPA rows with no tier-tagged snapshot within 18 months")
    print(f"[build] Total training rows: {len(train):,} "
           f"(incl {(train['value_source'] != 'transfermarkt').sum()} user-entered)")

    # Filter to a realistic EUR range. Liga 3 / Camp transfers cluster
    # in €25k - €2M; values outside that are noise (random €5k entries)
    # or academy-pipeline outliers (Sporting/Benfica loanees with
    # €10M+ TM valuations) that distort log-space variance.
    pre = len(train)
    train = train[(train['value_eur'] >= args.eur_min)
                    & (train['value_eur'] <= args.eur_max)].copy()
    print(f"[build] After EUR €{args.eur_min:,}-€{args.eur_max:,} filter: "
           f"{len(train):,}  (dropped {pre - len(train)})")

    # --- Map positions to CVI groups ---
    POS_GROUP = {
        'GK': 'GK',
        'CB': 'CB', 'LCB': 'CB', 'RCB': 'CB', 'LCB3': 'CB', 'RCB3': 'CB',
        'LB': 'FB', 'RB': 'FB', 'LB5': 'FB', 'RB5': 'FB', 'LWB': 'FB', 'RWB': 'FB',
        'CMF': 'CM', 'LCMF': 'CM', 'RCMF': 'CM', 'LCMF3': 'CM', 'RCMF3': 'CM',
        'DMF': 'CM', 'LDMF': 'CM', 'RDMF': 'CM',
        'AMF': 'AM_WG', 'LAMF': 'AM_WG', 'RAMF': 'AM_WG',
        'LMF': 'AM_WG', 'RMF': 'AM_WG',
        'LW': 'AM_WG', 'RW': 'AM_WG', 'LWF': 'AM_WG', 'RWF': 'AM_WG',
        'CF': 'ST', 'SS': 'ST',
    }
    train['position_group'] = train['primaryPosition'].map(POS_GROUP)
    train = train.dropna(subset=['position_group'])
    print(f"[build] After position-group mapping: {len(train):,}")

    # --- Compute age (player_details FIRST, fall back to TM metadata) ---
    dob_map = {}
    for _, r in details.iterrows():
        try:
            pid = int(r['playerId'])
            bd = r.get('birthDate')
            if bd and not pd.isna(bd):
                dob_map[pid] = pd.to_datetime(bd)
        except Exception:
            pass
    # Fallback: pull DOBs from TM metadata for the ~600 players missing
    # from player_details (e.g. cross-tier merges where the source DB
    # only carried current-league rosters).
    if not tm_meta.empty and 'dob' in tm_meta.columns:
        filled_from_tm = 0
        for _, r in tm_meta.iterrows():
            try:
                pid = int(r['playerId'])
                if pid in dob_map:
                    continue
                bd = r.get('dob')
                if bd and pd.notna(bd):
                    dob_map[pid] = pd.to_datetime(bd)
                    filled_from_tm += 1
            except Exception:
                pass
        print(f"[build] DOBs from player_details: "
               f"{len(dob_map) - filled_from_tm}, from TM metadata: {filled_from_tm}")
    # v3.3 — user-supplied DOBs for players missing from both
    # player_details and TM metadata. Without these the transfer
    # rows for these players get dropped at the age filter below.
    MANUAL_DOB_OVERRIDES = {
        709015: '2000-11-15',   # Tâmble Monteiro
        807307: '1999-08-21',   # Juan Perea
    }
    for pid, ds in MANUAL_DOB_OVERRIDES.items():
        if pid not in dob_map:
            dob_map[pid] = pd.to_datetime(ds)
            print(f"[build] manual DOB override: pid {pid} → {ds}")
    # Age at the ACTUAL valuation/transfer date — not at season
    # midpoint. For a TM snapshot in Sept or a Jan-window transfer
    # this can be off by 6+ months vs the old behaviour.
    train['age'] = train.apply(
        lambda r: ((r['_snap_date'] - dob_map[int(r['playerId'])]).days / 365.25
                     if int(r['playerId']) in dob_map else np.nan),
        axis=1,
    )
    train = train.dropna(subset=['age'])
    # v3.6 — age centered at peak-market age 24, then squared. This
    # lets the regression fit a U-shape (slight discount young AND old,
    # peak at prime) rather than the linear "younger = always better"
    # the model was learning before. Centering at 24 means age_dev_sq
    # is 0 at 24 and grows symmetrically — the model can decide how
    # steep the decline is on either side.
    train['age_dev_sq'] = (train['age'] - 24.0) ** 2
    print(f"[build] After age filter (age at snapshot date): {len(train):,}")

    # --- Passport nationality flag ---
    pt_pids = set()
    for _, r in details.iterrows():
        try:
            pid = int(r['playerId'])
            pp = r.get('passportArea')
            if isinstance(pp, dict):
                if (pp.get('name', '') or '').lower() == 'portugal':
                    pt_pids.add(pid)
            ba = r.get('birthArea')
            if isinstance(ba, dict):
                if (ba.get('name', '') or '').lower() == 'portugal':
                    pt_pids.add(pid)
        except Exception:
            pass
    train['passport_pt'] = train['playerId'].apply(
        lambda p: 1 if int(p) in pt_pids else 0)

    # --- Career mins to date (career-cumulative up to season) ---
    gpa_career = (pd.read_parquet(HERE / 'gpa_player_season_values.parquet')
                     [['playerId', 'seasonId', mins_col]])
    season_orders = {sid: SEASON_YEAR.get(sid, 2024) for sid in SEASON_YEAR}
    gpa_career['_year'] = gpa_career['seasonId'].map(season_orders)
    career_mins_map = {}
    for pid, group in gpa_career.dropna(subset=['_year']).groupby('playerId'):
        sorted_g = group.sort_values('_year')
        cum = sorted_g[mins_col].cumsum().tolist()
        years = sorted_g['_year'].tolist()
        sids = sorted_g['seasonId'].tolist()
        for sid, c in zip(sids, cum):
            career_mins_map[(int(pid), int(sid))] = float(c)
    train['career_mins_to_date'] = train.apply(
        lambda r: career_mins_map.get((int(r['playerId']), int(r['seasonId'])),
                                         r['mins_played']),
        axis=1,
    )

    # --- Goals / assists / xG residuals (career + per-season) ---
    # Per-season + career-cumulative goals and assists are durable
    # output signals that markets price into transfer fees.
    try:
        ev = pd.read_parquet(HERE / 'raw_events.parquet',
                              columns=['player.id', 'matchId', 'type.primary',
                                        'type.secondary',
                                        'shot.xg', 'shot.isGoal'])
        ms_for_xg = pd.read_parquet(HERE / 'matches_summary.parquet',
                                       columns=['matchId', 'seasonId'])
        ev = ev.merge(ms_for_xg, on='matchId', how='left').dropna(subset=['player.id'])
        ev['player.id'] = ev['player.id'].astype('int64')
        sid_order = pd.DataFrame({'seasonId': list(SEASON_YEAR.keys()),
                                     '_yr': list(SEASON_YEAR.values())})

        # GOALS — from shots with isGoal
        shots = ev[ev['type.primary'].isin(['shot'])].copy()
        shots['xg'] = pd.to_numeric(shots.get('shot.xg'), errors='coerce').fillna(0)
        shots['goal'] = shots.get('shot.isGoal', False).fillna(False).astype(int)

        # ASSISTS — from passes with 'assist' tag in type.secondary
        passes = ev[ev['type.primary'] == 'pass'].copy()
        passes['is_assist'] = passes['type.secondary'].apply(
            lambda v: 1 if (isinstance(v, np.ndarray)
                              and 'assist' in v)
                       else (1 if isinstance(v, list) and 'assist' in v else 0)
        )

        per_ss_g = (shots.groupby(['player.id', 'seasonId'])
                          .agg(xg_sum=('xg', 'sum'),
                                goals=('goal', 'sum'))
                          .reset_index()
                          .rename(columns={'player.id': 'playerId'}))
        per_ss_a = (passes.groupby(['player.id', 'seasonId'])
                          .agg(assists=('is_assist', 'sum'))
                          .reset_index()
                          .rename(columns={'player.id': 'playerId'}))
        per_ss = per_ss_g.merge(per_ss_a, on=['playerId', 'seasonId'], how='outer').fillna(0)
        per_ss = per_ss.merge(sid_order, on='seasonId', how='left')
        per_ss = per_ss.sort_values(['playerId', '_yr'])
        per_ss['xg_career'] = per_ss.groupby('playerId')['xg_sum'].cumsum()
        per_ss['goals_career'] = per_ss.groupby('playerId')['goals'].cumsum()
        per_ss['assists_career'] = per_ss.groupby('playerId')['assists'].cumsum()
        per_ss['xg_residual_career'] = per_ss['goals_career'] - per_ss['xg_career']

        # Per-(player, season) feature maps
        xg_resid_map = {(int(r['playerId']), int(r['seasonId'])): float(r['xg_residual_career'])
                          for _, r in per_ss.iterrows()}
        goals_career_map = {(int(r['playerId']), int(r['seasonId'])): float(r['goals_career'])
                              for _, r in per_ss.iterrows()}
        goals_season_map = {(int(r['playerId']), int(r['seasonId'])): float(r['goals'])
                              for _, r in per_ss.iterrows()}
        assists_season_map = {(int(r['playerId']), int(r['seasonId'])): float(r['assists'])
                                for _, r in per_ss.iterrows()}
        assists_career_map = {(int(r['playerId']), int(r['seasonId'])): float(r['assists_career'])
                                for _, r in per_ss.iterrows()}

        train['xg_residual_career'] = train.apply(
            lambda r: xg_resid_map.get((int(r['playerId']), int(r['seasonId'])), 0.0),
            axis=1)
        train['goals_career'] = train.apply(
            lambda r: goals_career_map.get((int(r['playerId']), int(r['seasonId'])), 0.0),
            axis=1)
        train['goals_season'] = train.apply(
            lambda r: goals_season_map.get((int(r['playerId']), int(r['seasonId'])), 0.0),
            axis=1)
        train['assists_season'] = train.apply(
            lambda r: assists_season_map.get((int(r['playerId']), int(r['seasonId'])), 0.0),
            axis=1)
        train['assists_career'] = train.apply(
            lambda r: assists_career_map.get((int(r['playerId']), int(r['seasonId'])), 0.0),
            axis=1)
        print(f"[build] Goals/assists/xG computed for "
               f"{len(goals_career_map):,} player-seasons")
    except Exception as e:
        print(f"[build] Goals/assists computation failed: {e}; setting to 0")
        for c in ('xg_residual_career', 'goals_career', 'goals_season',
                   'assists_season', 'assists_career'):
            train[c] = 0.0

    # --- v3.7 — counting-stat features (passing %, clean sheets, saves) ---
    # These were shown to have meaningful per-position correlation with
    # log(fee) beyond what perf_blend captures. We add them MV-side only;
    # TV stays CVI-only as the "pure quality" anchor so the MV-TV gap
    # measures market-vs-quality including market's counting-stat premium.
    try:
        ev_cs = pd.read_parquet(HERE / 'raw_events.parquet',
                                 columns=['matchId', 'team.id', 'player.id',
                                            'type.primary', 'pass.accurate',
                                            'shot.isGoal', 'shot.onTarget',
                                            'shot.goalkeeper.id', 'seasonId'])
        ev_cs = ev_cs.dropna(subset=['player.id'])
        ev_cs['player.id'] = ev_cs['player.id'].astype(int)
        ev_cs['seasonId'] = pd.to_numeric(ev_cs['seasonId'], errors='coerce')
        ev_cs = ev_cs.dropna(subset=['seasonId'])
        ev_cs['seasonId'] = ev_cs['seasonId'].astype(int)

        # Passes accurate per (player, season)
        passes_ev = ev_cs[ev_cs['type.primary'] == 'pass']
        pa = (passes_ev.groupby(['player.id', 'seasonId'])
                  ['pass.accurate']
                  .apply(lambda s: int(s.fillna(False).sum()))
                  .reset_index(name='passes_accurate')
                  .rename(columns={'player.id': 'playerId'}))

        # Saves per (gk, season) — shot on target, not a goal, GK identified
        sv_ev = ev_cs[(ev_cs['type.primary'] == 'shot')
                        & (ev_cs['shot.onTarget'].fillna(False))
                        & (~ev_cs['shot.isGoal'].fillna(False))
                        & ev_cs['shot.goalkeeper.id'].notna()].copy()
        sv_ev['gk_id'] = sv_ev['shot.goalkeeper.id'].astype(int)
        sv = (sv_ev.groupby(['gk_id', 'seasonId']).size()
                  .reset_index(name='saves')
                  .rename(columns={'gk_id': 'playerId'}))

        # Per-match goals against each team (proxy for clean sheets)
        goals_per_match_team = (ev_cs[(ev_cs['type.primary'] == 'shot')
                                          & (ev_cs['shot.isGoal'].fillna(False))]
                                    .groupby(['matchId', 'team.id'])
                                    .size().reset_index(name='goals_for'))
        teams_in_match = ev_cs[['matchId', 'team.id']].drop_duplicates()
        opp = teams_in_match.merge(teams_in_match, on='matchId',
                                       suffixes=('', '_opp'))
        opp = opp[opp['team.id'] != opp['team.id_opp']]
        conceded = opp.merge(
            goals_per_match_team.rename(
                columns={'team.id': 'team.id_opp', 'goals_for': 'goals_against'}),
            on=['matchId', 'team.id_opp'], how='left')
        conceded['goals_against'] = conceded['goals_against'].fillna(0)
        # Each player's team per match
        pl_team = ev_cs[['matchId', 'player.id', 'team.id', 'seasonId']].drop_duplicates()
        pl_team = pl_team.merge(
            conceded[['matchId', 'team.id', 'goals_against']].drop_duplicates(),
            on=['matchId', 'team.id'], how='left')
        pl_team['clean_sheet'] = (pl_team['goals_against'].fillna(99) == 0).astype(int)
        cs = (pl_team.groupby(['player.id', 'seasonId'])
                  .agg(matches_played=('matchId', 'nunique'),
                        clean_sheets=('clean_sheet', 'sum'),
                        goals_conceded_total=('goals_against', 'sum'))
                  .reset_index()
                  .rename(columns={'player.id': 'playerId'}))

        # Merge all into a per (player, season) stats table
        stats_all = cs.merge(pa, on=['playerId', 'seasonId'], how='outer')
        stats_all = stats_all.merge(sv, on=['playerId', 'seasonId'], how='outer')
        stats_all = stats_all.fillna(0)
        # Drop the "team" rows where player.id == 0 (Wyscout artifact)
        stats_all = stats_all[stats_all['playerId'] > 0]

        # v3.7 — also bake per-season goals + assists into the shipped
        # artifact so the dashboard doesn't need to recompute them.
        # `per_ss` was built above (goals/assists block) and has
        # ['playerId','seasonId','goals','assists'].
        try:
            stats_all = stats_all.merge(
                per_ss[['playerId', 'seasonId', 'goals', 'assists']],
                on=['playerId', 'seasonId'], how='left')
            stats_all[['goals', 'assists']] = stats_all[['goals', 'assists']].fillna(0)
        except Exception as _merr:
            print(f"[counting] couldn't merge goals/assists: {_merr}")
            stats_all['goals'] = 0
            stats_all['assists'] = 0
        # Also include mins_played so the dashboard can compute per-90s
        # straight from the parquet without re-reading GPA.
        try:
            gpa_mins = pd.read_parquet(
                HERE / 'gpa_player_season_values.parquet',
                columns=['playerId', 'seasonId', 'mins_played'])
            stats_all = stats_all.merge(gpa_mins, on=['playerId', 'seasonId'],
                                            how='left')
            stats_all['mins_played'] = stats_all['mins_played'].fillna(0)
        except Exception as _merr:
            print(f"[counting] couldn't merge mins_played: {_merr}")
            stats_all['mins_played'] = 0

        # Save shipping artifact so predict.py / app.py can look up per
        # (player, season) — same dir as the model bundles.
        stats_out = (HERE / 'models' / 'eur_v2' / 'counting_stats.parquet')
        stats_out.parent.mkdir(parents=True, exist_ok=True)
        stats_all.to_parquet(stats_out)
        print(f"[counting] wrote {len(stats_all):,} (player x season) "
               f"counting-stat rows to {stats_out.name}")

        # Don't merge mins_played / goals / assists back into train —
        # train already has these from upstream. Strip them before the
        # merge to avoid _x/_y column collisions.
        stats_for_train = stats_all.drop(
            columns=[c for c in ('mins_played', 'goals', 'assists')
                       if c in stats_all.columns], errors='ignore')
        # Attach to training rows + compute per-90 / percentage versions
        train = train.merge(stats_for_train, on=['playerId', 'seasonId'], how='left')
        train[['passes_accurate', 'saves', 'matches_played',
               'clean_sheets', 'goals_conceded_total']] = train[[
            'passes_accurate', 'saves', 'matches_played',
            'clean_sheets', 'goals_conceded_total']].fillna(0)
        mins_played = train[mins_col].astype(float).clip(lower=1)
        train['mins_90'] = mins_played / 90.0
        train['ga_per90'] = (train['goals_season'] + train['assists_season']) / train['mins_90']
        train['passes_accurate_per90'] = train['passes_accurate'] / train['mins_90']
        train['cs_pct'] = np.where(train['matches_played'] > 0,
                                       train['clean_sheets'] / train['matches_played'], 0)
        # Save % = saves / (saves + goals_conceded_total) — only meaningful
        # for GKs. For non-GK we expect ~0 saves so this will be 0 for them.
        train['save_pct'] = np.where(
            (train['saves'] + train['goals_conceded_total']) > 0,
            train['saves'] / (train['saves'] + train['goals_conceded_total']), 0)
    except Exception as e:
        print(f"[counting] computation failed: {e}; setting to 0")
        for c in ('ga_per90', 'passes_accurate_per90', 'cs_pct', 'save_pct'):
            train[c] = 0.0

    # --- n_seasons_played (in our data, up to and including this season) ---
    seasons_to_date = {}
    for pid, group in gpa_career.dropna(subset=['_year']).groupby('playerId'):
        sorted_g = group.sort_values('_year')
        years = sorted_g['_year'].tolist()
        sids = sorted_g['seasonId'].tolist()
        for i, sid in enumerate(sids):
            seasons_to_date[(int(pid), int(sid))] = i + 1
    train['n_seasons_played'] = train.apply(
        lambda r: seasons_to_date.get((int(r['playerId']), int(r['seasonId'])), 1),
        axis=1,
    )

    # --- log target + features ---
    train['log_value_eur'] = np.log(train['value_eur'].clip(lower=1000))
    # log Total Value — use a small offset to handle 0/negative
    train['Total Value'] = pd.to_numeric(train['Total Value'], errors='coerce')
    train['signed_log_tv'] = train['Total Value'].apply(
        lambda v: np.sign(v) * np.log1p(abs(v) * 100) if pd.notna(v) else 0)
    # v3 — squared (sign-preserving) version so models can give convex
    # weight to elite performances.
    train['signed_log_tv_sq'] = train['signed_log_tv'] * train['signed_log_tv'].abs()

    # ====== v3.2 — CVI versatility blend: max + mean across templates ======
    # Mirrors the dashboard's CVI engine exactly:
    #   Role_Score = 0.6 × max(role_template_pcts) + 0.4 × mean(...)
    #   perf_blend = w_role × Role_Score + w_av × Action_V_pct
    # Position-tuned weights for the inner blend match CVI_PERF_WEIGHTS.
    #
    # Multiple role templates per position (each one a V/90 weighted
    # composite). The max captures specialist value; the mean rewards
    # versatility. Templates designed to mirror the dashboard's role
    # score templates by intent.
    ROLE_TEMPLATES = {
        'GK': [
            {'GK Total Value': 1.00},
        ],
        'CB': [
            # Stopper
            {'Total Value': 0.40, 'Interrupting Value': 0.60},
            # Build-up CB
            {'Total Value': 0.30, 'Passing Value': 0.50,
              'Interrupting Value': 0.20},
            # All-rounder CB
            {'Total Value': 0.40, 'Passing Value': 0.30,
              'Interrupting Value': 0.30},
        ],
        'FB': [
            # Defensive FB
            {'Total Value': 0.40, 'Interrupting Value': 0.40,
              'Passing Value': 0.20},
            # Wingback (attacking)
            {'Total Value': 0.30, 'Receiving Value': 0.30,
              'Dribbling Value': 0.30, 'Passing Value': 0.10},
            # Inverted FB
            {'Total Value': 0.30, 'Passing Value': 0.40,
              'Dribbling Value': 0.20, 'Receiving Value': 0.10},
        ],
        'CM': [
            # Box-to-Box
            {'Total Value': 0.30, 'Passing Value': 0.25,
              'Receiving Value': 0.20, 'Dribbling Value': 0.15,
              'Interrupting Value': 0.10},
            # Deep-lying Playmaker
            {'Total Value': 0.15, 'Passing Value': 0.60,
              'Dribbling Value': 0.15, 'Interrupting Value': 0.10},
            # Ball-winning Mid
            {'Total Value': 0.30, 'Interrupting Value': 0.50,
              'Passing Value': 0.20},
            # Holding Mid
            {'Total Value': 0.40, 'Passing Value': 0.30,
              'Interrupting Value': 0.30},
        ],
        'AM_WG': [
            # Inside Forward
            {'Total Value': 0.15, 'Receiving Value': 0.25,
              'Dribbling Value': 0.25, 'Shooting Value': 0.25,
              'Passing Value': 0.10},
            # Wide Winger
            {'Total Value': 0.15, 'Receiving Value': 0.30,
              'Dribbling Value': 0.40, 'Passing Value': 0.15},
            # Advanced Playmaker
            {'Total Value': 0.10, 'Passing Value': 0.55,
              'Receiving Value': 0.20, 'Dribbling Value': 0.15},
            # Shadow Striker
            {'Total Value': 0.15, 'Shooting Value': 0.40,
              'Receiving Value': 0.30, 'Dribbling Value': 0.15},
        ],
        'ST': [
            # Poacher
            {'Total Value': 0.10, 'Shooting Value': 0.70,
              'Receiving Value': 0.20},
            # Target Forward
            {'Total Value': 0.25, 'Shooting Value': 0.30,
              'Receiving Value': 0.45},
            # Mobile Striker
            {'Total Value': 0.15, 'Shooting Value': 0.40,
              'Receiving Value': 0.25, 'Dribbling Value': 0.20},
            # Pressing Forward (high work-rate ST)
            {'Total Value': 0.30, 'Shooting Value': 0.30,
              'Receiving Value': 0.20, 'Interrupting Value': 0.20},
        ],
    }
    CVI_ROLE_VERSATILITY_ALPHA = 0.6   # 0.6 × max + 0.4 × mean
    # CVI position weights: (role_weight, action_v_weight)
    CVI_PERF_WEIGHTS = {
        'GK': (0.80, 0.20), 'CB': (0.75, 0.25), 'FB': (0.65, 0.35),
        'CM': (0.60, 0.40), 'AM_WG': (0.55, 0.45), 'ST': (0.50, 0.50),
    }

    # Load FULL GPA per-season with V-category breakdown
    gpa_full = pd.read_parquet(HERE / 'gpa_player_season_values.parquet')
    # Coerce all V columns to numeric
    _v_cols = ['Total Value', 'Passing Value', 'Receiving Value',
                'Dribbling Value', 'Shooting Value', 'Interrupting Value',
                'GK Total Value']
    for c in _v_cols:
        if c in gpa_full.columns:
            gpa_full[c] = pd.to_numeric(gpa_full[c], errors='coerce').fillna(0)
    # Compute role-V composite per row (depends on position group)
    POS_GROUP_MAP = {
        'GK': 'GK',
        'CB': 'CB', 'LCB': 'CB', 'RCB': 'CB', 'LCB3': 'CB', 'RCB3': 'CB',
        'LB': 'FB', 'RB': 'FB', 'LB5': 'FB', 'RB5': 'FB',
        'LWB': 'FB', 'RWB': 'FB',
        'CMF': 'CM', 'LCMF': 'CM', 'RCMF': 'CM',
        'LCMF3': 'CM', 'RCMF3': 'CM',
        'DMF': 'CM', 'LDMF': 'CM', 'RDMF': 'CM',
        'AMF': 'AM_WG', 'LAMF': 'AM_WG', 'RAMF': 'AM_WG',
        'LMF': 'AM_WG', 'RMF': 'AM_WG',
        'LW': 'AM_WG', 'RW': 'AM_WG', 'LWF': 'AM_WG', 'RWF': 'AM_WG',
        'CF': 'ST', 'SS': 'ST',
    }
    gpa_full['_pos_group'] = gpa_full['position'].map(POS_GROUP_MAP)
    gpa_full = gpa_full.dropna(subset=['_pos_group'])

    # Compute role_v per template; percentile-rank each within
    # (season × position_group); take 0.6 × max + 0.4 × mean.
    # Action V percentile (Total Value /90) is also computed.
    gpa_full['_av_pct'] = (gpa_full.groupby(['seasonId', '_pos_group'])
                            ['Total Value']
                            .rank(pct=True, method='average') * 100.0)
    role_pct_cols = []
    for pos, templates in ROLE_TEMPLATES.items():
        sub = gpa_full[gpa_full['_pos_group'] == pos].copy()
        for t_idx, t_weights in enumerate(templates):
            col = f'_role_v_{pos}_{t_idx}'
            sub[col] = sum(sub.get(c, 0) * w for c, w in t_weights.items())
            sub[col + '_pct'] = (sub.groupby('seasonId')[col]
                                  .rank(pct=True, method='average') * 100.0)
            gpa_full.loc[sub.index, col + '_pct'] = sub[col + '_pct']
            role_pct_cols.append(col + '_pct')
    # For each row, collect its position-specific role pct cols + blend
    def _role_score_blend(row):
        pos = row['_pos_group']
        n_templates = len(ROLE_TEMPLATES.get(pos, []))
        pcts = []
        for t_idx in range(n_templates):
            v = row.get(f'_role_v_{pos}_{t_idx}_pct')
            if v is not None and not pd.isna(v):
                pcts.append(float(v))
        if not pcts:
            return 50.0
        if len(pcts) == 1:
            return pcts[0]
        a = CVI_ROLE_VERSATILITY_ALPHA
        return a * max(pcts) + (1.0 - a) * (sum(pcts) / len(pcts))
    gpa_full['_role_pct_blend'] = gpa_full.apply(_role_score_blend, axis=1)
    # CVI position-weighted blend
    gpa_full['_perf_blend'] = gpa_full.apply(
        lambda r: (
            CVI_PERF_WEIGHTS[r['_pos_group']][0] * r['_role_pct_blend']
            + CVI_PERF_WEIGHTS[r['_pos_group']][1] * r['_av_pct']
        ),
        axis=1,
    )
    perf_map = {(int(r['playerId']), int(r['seasonId'])):
                 (r['_perf_blend'], r['_av_pct'], r['_role_pct_blend'])
                 for _, r in gpa_full.iterrows()}
    train['perf_blend'] = train.apply(
        lambda r: perf_map.get((int(r['playerId']), int(r['seasonId'])),
                                  (50.0, 50.0, 50.0))[0],
        axis=1,
    )
    train['perf_blend_sq'] = train['perf_blend'] * train['perf_blend'] / 100.0
    # Boost factor — multiply the standardized perf signal so it
    # contributes more in the regression. v3.1 default = 2.0×.
    PERF_BOOST = 2.0
    train['perf_blend'] = train['perf_blend'] * PERF_BOOST
    train['perf_blend_sq'] = train['perf_blend_sq'] * PERF_BOOST
    print(f"[perf] CVI-style perf_blend computed for {len(perf_map):,} "
           f"(player, season) cohorts, boost factor {PERF_BOOST}×")

    # ====== END v3.1 perf blend ======

    train['log_career_mins'] = np.log(
        train['career_mins_to_date'].clip(lower=1))
    train['log_mins_season'] = np.log(train['mins_played'].clip(lower=1))
    # Raw career mins (in thousands so coef stays interpretable;
    # complements log_career_mins which captures concavity)
    train['career_mins_k'] = train['career_mins_to_date'] / 1000.0
    # Raw career goals + log(career goals) so the model can pick up
    # both 'has scored at all' (log lifts 0→1) and 'has scored a lot'.
    train['goals_career'] = pd.to_numeric(train['goals_career'],
                                              errors='coerce').fillna(0)
    train['log_goals_career'] = np.log1p(train['goals_career'])

    # --- Position-group one-hot (drop CM as reference) ---
    pos_dummies = pd.get_dummies(train['position_group'], prefix='pos',
                                     drop_first=False).astype(int)
    if 'pos_CM' in pos_dummies.columns:
        pos_dummies = pos_dummies.drop(columns=['pos_CM'])
    train = pd.concat([train, pos_dummies], axis=1)

    # --- Final feature list ---
    # v3.1 — slimmer feature set.
    # Dropped: signed_log_tv (replaced by perf_blend), goals_season +
    # assists_season (collinearity artifacts), assists_career (extreme
    # at n=201), n_seasons_played + passport_pt (zero impact),
    # career_mins_k (collinear with log_career_mins).
    # Kept: age (still strongest signal), perf_blend + sq (CVI-style),
    # league, log mins (sample size), goals_career + xg_residual
    # (finishing track), position dummies, season_year (market drift).
    num_features = ['perf_blend', 'perf_blend_sq',
                     'age', 'age_dev_sq', 'league_factor',
                     'log_career_mins', 'log_mins_season',
                     'season_year',
                     'xg_residual_career', 'goals_career',
                     # v3.11 — only G+A and clean sheets in MV counting
                     # stats (dropped passes_accurate_per90 which was
                     # inflating CBs via B-team-prospect training noise,
                     # and save_pct which had near-zero coefficient).
                     'ga_per90', 'cs_pct']
    cat_features = [c for c in train.columns if c.startswith('pos_')]
    features = num_features + cat_features
    print(f"\n[features] {len(features)} features: {features}")

    X = train[features].astype(float).values
    y = train['log_value_eur'].values
    print(f"[features] X shape: {X.shape}, y shape: {y.shape}")

    # v3.4 — heavier sample weighting on actual paid transfers.
    # User asked us to lean MUCH harder on the real transfer fees
    # vs the noisier TM snapshots. With ~8 reported_fees vs ~200 TM
    # rows, a 25x weight makes the reported fees count for the
    # equivalent of 200 effective rows — so the regression's
    # gradient comes about 50/50 from real money vs TM crowd-estimates.
    REPORTED_FEE_WEIGHT = 25.0
    sw = np.where(train['value_source'] == 'reported_fee',
                    REPORTED_FEE_WEIGHT, 1.0).astype(float)
    n_rep = int((train['value_source'] == 'reported_fee').sum())
    print(f"[weights] {n_rep} reported_fee rows × {REPORTED_FEE_WEIGHT:.0f}, "
           f"{len(sw) - n_rep} TM rows × 1.0  "
           f"(effective N = {sw.sum():.0f})")

    # --- Cross-validate Ridge ---
    print(f"\n[model] Fitting Ridge with 5-fold CV...")
    model = Pipeline([
        ('scale', StandardScaler()),
        ('ridge', RidgeCV(alphas=[0.01, 0.1, 1, 10, 100], cv=5)),
    ])
    model.fit(X, y, ridge__sample_weight=sw)
    chosen_alpha = model.named_steps['ridge'].alpha_
    print(f"[model] Chosen alpha: {chosen_alpha}")

    # Coefficients (back-mapped from standardized space)
    scaler = model.named_steps['scale']
    coefs = model.named_steps['ridge'].coef_
    intercept = model.named_steps['ridge'].intercept_

    print(f"\n[coefficients] Standardized (relative importance):")
    coef_df = pd.DataFrame({
        'feature': features,
        'coef_std': coefs,
        'abs_coef': np.abs(coefs),
    }).sort_values('abs_coef', ascending=False)
    print(coef_df[['feature', 'coef_std']].to_string(index=False))

    # --- Out-of-fold predictions for honest R² + MAE ---
    print(f"\n[validation] Computing 5-fold out-of-sample predictions...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    for fold_i, (tr, te) in enumerate(kf.split(X)):
        fm = Pipeline([
            ('scale', StandardScaler()),
            ('ridge', RidgeCV(alphas=[0.01, 0.1, 1, 10, 100], cv=3)),
        ])
        fm.fit(X[tr], y[tr], ridge__sample_weight=sw[tr])
        oof[te] = fm.predict(X[te])
    oof_r2 = r2_score(y, oof)
    oof_mae_log = mean_absolute_error(y, oof)
    # Back-transform to EUR for an interpretable MAE
    pred_eur = np.exp(oof)
    actual_eur = np.exp(y)
    # Median absolute % error is robust to outliers
    mape = np.median(np.abs(pred_eur - actual_eur) / actual_eur) * 100
    # Spearman rank correlation — captures "ordering correctness" which
    # matters more than absolute EUR accuracy for scouting purposes.
    from scipy.stats import spearmanr
    spearman_rho, _ = spearmanr(y, oof)
    # Within-position Spearman (so position one-hots don't drive the rank)
    pos_spearman = []
    train_idx = train.reset_index(drop=True)
    for pos in train_idx['position_group'].unique():
        mask = train_idx['position_group'] == pos
        if mask.sum() >= 10:
            r, _ = spearmanr(y[mask.values], oof[mask.values])
            pos_spearman.append((pos, int(mask.sum()), r))
    print(f"  Out-of-fold R²:           {oof_r2:.3f}")
    print(f"  Out-of-fold MAE (log):    {oof_mae_log:.3f}")
    print(f"  Out-of-fold MAPE:         {mape:.1f}%")
    print(f"  Out-of-fold MAE (EUR):    €{np.mean(np.abs(pred_eur - actual_eur)):,.0f}")
    print(f"  Out-of-fold Spearman ρ:   {spearman_rho:.3f}  (all positions pooled)")
    print(f"  Out-of-fold Spearman by position:")
    for pos, n, r in sorted(pos_spearman, key=lambda x: -x[2]):
        print(f"    {pos:<8}  n={n:>3}  ρ={r:+.3f}")

    # Top predictions vs actual
    train['predicted_eur'] = np.exp(model.predict(X))
    train['oof_predicted_eur'] = pred_eur
    train['log_actual'] = y
    train['oof_residual'] = y - oof
    print(f"\n[spot-check] 5 random rows with residuals:")
    sample = train.sample(min(5, len(train)), random_state=1)
    for _, r in sample.iterrows():
        print(f"  pid={r['playerId']:<8}  s={r['seasonId']}  pos={r['position_group']:<6} "
              f"age={r['age']:.1f}  actual=€{r['value_eur']:>10,.0f}  "
              f"pred=€{r['oof_predicted_eur']:>10,.0f}  residual={r['oof_residual']:+.2f}")

    if args.dry:
        print(f"\n[done] Dry run — no model saved")
        return 0

    # --- Train SECOND model: True Value (CVI-equivalent features only,
    # MV-scale target) ---
    # The user-facing "True Value" should be directly comparable to MV
    # (realized fee), NOT to TMV (theoretical market value). So we train
    # the True Value model with target = log(MV) where
    # MV = value_eur × realization_ratio(value_eur). This puts True
    # Value on the same scale as the realized-fee number — making
    # the MV − TV gap interpretable as "market overpaying/underpaying
    # vs pure-quality fair value".
    #
    # Features = CVI-equivalent only (signed_log_tv + age + league +
    # position_group). Strips market-noise features (goals/assists/
    # xG residuals/passport/career mins/season year).
    from models.eur_v2.realization import realization_ratio
    realized_target_eur = train['value_eur'] * train['value_eur'].apply(
        realization_ratio)
    # Clip lower bound so log is well-defined
    realized_target_eur = realized_target_eur.clip(lower=1000)
    y_mv = np.log(realized_target_eur.values)
    # v3 — added log_mins_season so small samples get explicitly
    # discounted. Also added signed_log_tv_sq (squared transform of the
    # performance signal) to let the model give more weight to top
    # performers — convex perf response means a 95th-percentile V/90
    # carries proportionally more EUR weight than a 50th-percentile.
    # v3.1 — True Value uses the CVI-style position-weighted perf
    # blend (matches what CVI's PerformanceQuality computes).
    true_value_features = ['perf_blend', 'perf_blend_sq',
                            'age', 'age_dev_sq',
                            'league_factor', 'log_mins_season',
                            'pos_AM_WG', 'pos_CB', 'pos_FB', 'pos_GK', 'pos_ST']
    X_tv = train[true_value_features].astype(float).values
    print(f"\n[true_value] Fitting on MV-scale target "
           f"(value_eur × realization_ratio), {len(true_value_features)} "
           f"pure-CVI features...")
    print(f"  Median target MV: €{np.exp(y_mv).median() if hasattr(np.exp(y_mv), 'median') else float(np.median(np.exp(y_mv))):,.0f}  "
           f"(vs median TMV €{np.exp(y).mean():,.0f})")
    tv_model = Pipeline([
        ('scale', StandardScaler()),
        ('ridge', RidgeCV(alphas=[0.01, 0.1, 1, 10, 100], cv=5)),
    ])
    tv_model.fit(X_tv, y_mv, ridge__sample_weight=sw)
    tv_alpha = tv_model.named_steps['ridge'].alpha_
    tv_coefs = tv_model.named_steps['ridge'].coef_

    # OOF on the SAME MV-target (so we can compare TV predictions
    # against the realized-MV target directly)
    tv_oof = np.zeros(len(y_mv))
    for tr, te in kf.split(X_tv):
        fm = Pipeline([
            ('scale', StandardScaler()),
            ('ridge', RidgeCV(alphas=[0.01, 0.1, 1, 10, 100], cv=3)),
        ])
        fm.fit(X_tv[tr], y_mv[tr], ridge__sample_weight=sw[tr])
        tv_oof[te] = fm.predict(X_tv[te])
    tv_r2 = r2_score(y_mv, tv_oof)
    tv_mae_log = mean_absolute_error(y_mv, tv_oof)
    tv_spearman, _ = spearmanr(y_mv, tv_oof)
    print(f"[true_value] chosen alpha: {tv_alpha}")
    print(f"[true_value] OOF R² (vs MV target):  {tv_r2:.3f}")
    print(f"[true_value] OOF Spearman:           {tv_spearman:.3f}")
    print(f"[true_value] coefficients (standardized):")
    tv_coef_df = pd.DataFrame({
        'feature': true_value_features,
        'coef_std': tv_coefs,
    }).sort_values('coef_std', key=abs, ascending=False)
    print(tv_coef_df.to_string(index=False))

    # --- Save ---
    out_dir = HERE / 'models' / 'eur_v2'
    out_dir.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump({
        'model': model,
        'features': features,
        'chosen_alpha': float(chosen_alpha),
        'n_training_rows': len(train),
        'oof_r2': float(oof_r2),
        'oof_mae_log': float(oof_mae_log),
        'oof_mape_pct': float(mape),
    }, out_dir / 'eur_v2_ridge.joblib')
    joblib.dump({
        'model': tv_model,
        'features': true_value_features,
        'chosen_alpha': float(tv_alpha),
        'n_training_rows': len(train),
        'oof_r2': float(tv_r2),
        'oof_mae_log': float(tv_mae_log),
        'target_scale': 'MV',  # log(value_eur × realization_ratio) — not raw TMV
        'description': ('CVI-only True Value model trained on MV target '
                          '(realized fee scale = TMV × realization_ratio). '
                          'Features: signed_log_tv + age + league + position. '
                          'Output is directly comparable to MV in the dashboard; '
                          'MV − TV is the market-vs-quality gap signal.'),
    }, out_dir / 'true_value_ridge.joblib')
    tv_coef_df.to_csv(out_dir / 'true_value_coefficients.csv', index=False)
    coef_df.to_csv(out_dir / 'coefficients.csv', index=False)
    train[['playerId', 'seasonId', 'position_group', 'age',
            'value_eur', 'predicted_eur', 'oof_predicted_eur']].to_csv(
        out_dir / 'training_set.csv', index=False)
    with open(out_dir / 'meta.json', 'w') as f:
        json.dump({
            'features': features,
            'chosen_alpha': float(chosen_alpha),
            'n_training_rows': int(len(train)),
            'oof_r2': float(oof_r2),
            'oof_mae_log': float(oof_mae_log),
            'oof_mape_pct': float(mape),
            'pos_groups_one_hot': [c for c in features if c.startswith('pos_')],
            'reference_position_group': 'CM',
            'version': 'v3.11',
            'changes': [
                'Added counting-stat features to MV (TV stays CVI-only): ga_per90, passes_accurate_per90, cs_pct, save_pct',
                'CB and CM Spearman improved (CB 0.65 -> 0.75, CM 0.53 -> 0.65); GK 0.45 -> 0.52',
                'Counting stats computed from raw_events at train time and shipped as models/eur_v2/counting_stats.parquet for dashboard lookup',
                'Inherits v3.6: pre-transfer perf pairing, 22 reported_fee transfers, age_dev_sq',
            ],
        }, f, indent=2)

    # --- v3 — also write JSON fallback bundles (no sklearn / joblib
    # needed at runtime). The HF Space loads these when joblib is
    # missing or sklearn version mismatches. KEEP IN SYNC with the
    # joblib bundles above. ---
    def _dump_json_bundle(pipeline, feature_list, target_path, extra_meta=None):
        s = pipeline.named_steps['scale']
        r = pipeline.named_steps['ridge']
        payload = {
            'scaler_mean': s.mean_.tolist(),
            'scaler_scale': s.scale_.tolist(),
            'ridge_coef': r.coef_.tolist(),
            'ridge_intercept': float(r.intercept_),
            'features': list(feature_list),
            'meta': extra_meta or {},
        }
        with open(target_path, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f"[json bundle] wrote {target_path.name}")
    _dump_json_bundle(model, features, out_dir / 'eur_v2_ridge.json',
                       extra_meta={'chosen_alpha': float(chosen_alpha),
                                    'oof_r2': float(oof_r2),
                                    'version': 'v3.11'})
    _dump_json_bundle(tv_model, true_value_features,
                       out_dir / 'true_value_ridge.json',
                       extra_meta={'chosen_alpha': float(tv_alpha),
                                    'oof_r2': float(tv_r2),
                                    'target_scale': 'MV',
                                    'version': 'v3.11'})
    print(f"\n[done] Model + diagnostics saved to {out_dir}/")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
