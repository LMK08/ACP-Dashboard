#!/usr/bin/env python3
"""Player roles — Phase 0/1 feature builder.

Builds each player's SPATIAL SIGNATURE from where their on-ball events
actually happen (not the lineup-card position), at two granularities:

  - per (player, match)  -> the atomic unit, so role changes across games
                              are observable (a season role is a
                              DISTRIBUTION over match roles, not one label)
  - per (player, season) -> dense maps used to learn the role taxonomy

Each signature is a two-channel smoothed heatmap on a side-folded grid:
  IP  (in possession):     pass / touch / acceleration / shot
                            + offensive ground duels & dribbles
  OOP (out of possession): interception / clearance / defensive duel
                            / ground recovery / own-half aerial duel

Side-folding (yf = |y-50|) makes "left winger" and "right winger" one
role family; the signed mean y is kept as a separate side attribute.
Direct restart events (corner / free kick / throw-in / goal kick /
penalty) are excluded so corner-takers don't "operate" at the flag.

Outputs (models/roles/):
  role_features_season.parquet  — one row per (playerId, seasonId)
  role_features_match.parquet   — one row per (playerId, matchId)
Both carry flattened map columns ip_00..ip_39 / op_00..op_39 + scalars.

Run from the Dashboard dir: python models/roles/build_role_features.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

_HERE = Path(__file__).resolve().parent
_DASH = _HERE.parent.parent
_GPA_DATA = _DASH.parent.parent / 'GPA Model Project v2' / 'parquet_data'

# ---- grid spec -------------------------------------------------------------
NX, NY = 8, 5            # x: own goal -> opp goal (8 bands); yf: centre -> wide (5 bands)
SMOOTH_SIGMA = 0.8       # cells
MIN_EV_SEASON = 60       # minimum located events for a season map
MIN_EV_MATCH = 12        # minimum located events for a match map

RESTART_PRIMARIES = {'corner', 'free_kick', 'throw_in', 'goal_kick',
                       'penalty', 'goalkeeper_exit', 'game_interruption',
                       'offside', 'fairplay', 'postmatch_penalty'}
IP_PRIMARIES = {'pass', 'touch', 'acceleration', 'shot'}
_RECOVERY_TAGS = {'recovery', 'counterpressing_recovery'}


def _is_recovery(v):
    if isinstance(v, (list, tuple, np.ndarray)):
        return any(t in _RECOVERY_TAGS for t in v)
    return v in _RECOVERY_TAGS


def load_events():
    cols = ['matchId', 'seasonId', 'team.id', 'player.id', 'player.position',
             'type.primary', 'type.secondary', 'location.x', 'location.y',
             'groundDuel.duelType', 'aerialDuel.firstTouch', 'competitionId']
    frames = []
    for f in ['liga3_portugal_events', 'campeonato_portugal_events']:
        part = pd.read_parquet(_GPA_DATA / f'{f}.parquet', columns=cols)
        frames.append(part)
        print(f"  {f}: {len(part):,} events")
    ev = pd.concat(frames, ignore_index=True)
    ev = ev.dropna(subset=['player.id', 'location.x', 'location.y'])
    ev['player.id'] = ev['player.id'].astype(int)
    ev['seasonId'] = pd.to_numeric(ev['seasonId'], errors='coerce').astype('Int64')
    return ev


def flag_channels(ev):
    tp = ev['type.primary']
    ev['_restart'] = tp.isin(RESTART_PRIMARIES)
    ev['_rec'] = [_is_recovery(t) for t in ev['type.secondary'].tolist()]
    ev['_aer'] = ev['aerialDuel.firstTouch'].notna()
    gd = ev['groundDuel.duelType']
    # In possession: open-play ball involvement
    ev['_ip'] = (~ev['_restart']) & (
        tp.isin(IP_PRIMARIES)
        | gd.isin(['offensive_duel', 'dribble']))
    # Out of possession: the DefR defensive-action family (own-half aerials
    # as the defensive-aerial proxy — fine for maps)
    ev['_op'] = (~ev['_restart']) & (
        (tp == 'interception') | (tp == 'clearance')
        | (gd == 'defensive_duel')
        | (ev['_rec'] & ~ev['_aer'])
        | (ev['_aer'] & (ev['location.x'] <= 50)))
    return ev


def grid_index(x, y):
    """(x, y) -> flattened side-folded cell index, NX*NY grid."""
    gx = np.clip((np.asarray(x, dtype=float) / (100.0 / NX)).astype(int), 0, NX - 1)
    yf = np.abs(np.asarray(y, dtype=float) - 50.0)            # 0 centre .. 50 wide
    gy = np.clip((yf / (50.0 / NY)).astype(int), 0, NY - 1)
    return gx * NY + gy


def smooth_norm(counts):
    """counts (NX*NY,) -> smoothed, L1-normalized map."""
    m = counts.reshape(NX, NY).astype(float)
    m = gaussian_filter(m, sigma=SMOOTH_SIGMA, mode='nearest')
    s = m.sum()
    return (m / s).ravel() if s > 0 else m.ravel()


def build_level(ev, keys, min_ev, label):
    """Aggregate channel maps + scalars per `keys` (list of columns)."""
    print(f"  building {label} signatures…", flush=True)
    sub = ev[ev['_ip'] | ev['_op']].copy()
    sub['_cell'] = grid_index(sub['location.x'], sub['location.y'])
    sub['_ysig'] = sub['location.y'] - 50.0
    sub['_yf'] = sub['_ysig'].abs()

    rows = []
    ncells = NX * NY
    for key_vals, g in sub.groupby(keys):
        ip = g[g['_ip']]; op = g[g['_op']]
        n_ip, n_op = len(ip), len(op)
        if n_ip + n_op < min_ev:
            continue
        ip_counts = np.bincount(ip['_cell'], minlength=ncells)
        op_counts = np.bincount(op['_cell'], minlength=ncells)
        rec = dict(zip(keys, key_vals if isinstance(key_vals, tuple) else (key_vals,)))
        rec.update({
            'n_ip': n_ip, 'n_op': n_op,
            'def_share': n_op / (n_ip + n_op),
            'x_ip': ip['location.x'].mean() if n_ip else np.nan,
            'x_op': op['location.x'].mean() if n_op else np.nan,
            'yf_ip': ip['_yf'].mean() if n_ip else np.nan,
            'yf_op': op['_yf'].mean() if n_op else np.nan,
            'y_signed': g['_ysig'].mean(),
            'box_share_ip': float(((ip['location.x'] >= 84)
                                     & (ip['_yf'] <= 20)).mean()) if n_ip else np.nan,
        })
        for i, v in enumerate(smooth_norm(ip_counts)):
            rec[f'ip_{i:02d}'] = v
        for i, v in enumerate(smooth_norm(op_counts)):
            rec[f'op_{i:02d}'] = v
        rows.append(rec)
    return pd.DataFrame(rows)


def rank_features(ev, gk):
    """v2 — within-team-match percentile ranks (teammate-RELATIVE position).
    For each (player, match): where does the player's mean action location
    rank among his own team's outfielders that match?
      rank_x_ip  — depth rank in possession   (0 = deepest, 1 = highest)
      rank_x_op  — depth rank out of possession
      rank_yf_ip — width rank in possession   (0 = most central, 1 = widest)
    Scale-free, so it sharpens orderings (deepest pivot, wide-CB vs WB)
    without re-importing team context. GKs excluded from the rank pool.
    Returns (per-match df, per-season df aggregated event-weighted)."""
    gkdf = pd.DataFrame(list(gk), columns=['player.id', 'seasonId'])
    gkdf['_gk'] = 1
    out_m = None
    for ch, val, name in [('_ip', 'location.x', 'rank_x_ip'),
                            ('_op', 'location.x', 'rank_x_op'),
                            ('_ip', '_yfv', 'rank_yf_ip')]:
        sub = ev[ev[ch]].copy()
        if val == '_yfv':
            sub['_yfv'] = (sub['location.y'] - 50.0).abs()
        g = (sub.groupby(['matchId', 'team.id', 'seasonId', 'player.id'])[val]
                .agg(['mean', 'size']).reset_index())
        g = g.merge(gkdf, on=['player.id', 'seasonId'], how='left')
        g = g[g['_gk'].isna()].drop(columns=['_gk'])
        g[name] = g.groupby(['matchId', 'team.id'])['mean'].rank(pct=True)
        g = g.rename(columns={'player.id': 'playerId', 'size': f'_n_{name}'})
        # A handful of players carry events under both team.ids in one
        # match (duel bookkeeping) — keep the dominant team's row so the
        # (player, match) key stays unique and the channel merge doesn't
        # cartesian-expand.
        g = (g.sort_values(f'_n_{name}', ascending=False)
                .drop_duplicates(['playerId', 'matchId', 'seasonId']))
        g = g[['playerId', 'matchId', 'seasonId', name, f'_n_{name}']]
        out_m = g if out_m is None else out_m.merge(
            g, on=['playerId', 'matchId', 'seasonId'], how='outer')
    # season aggregate: event-weighted mean of match ranks
    out_s = out_m.copy()
    rows = {'playerId': out_s['playerId'], 'seasonId': out_s['seasonId']}
    sea_parts = []
    for name in ['rank_x_ip', 'rank_x_op', 'rank_yf_ip']:
        w = out_s[f'_n_{name}'].fillna(0)
        v = out_s[name]
        t = pd.DataFrame({'playerId': out_s['playerId'],
                            'seasonId': out_s['seasonId'],
                            '_wv': v * w, '_w': w.where(v.notna(), 0)})
        a = t.groupby(['playerId', 'seasonId'])[['_wv', '_w']].sum()
        sea_parts.append((a['_wv'] / a['_w'].replace(0, np.nan)).rename(name))
    out_sea = pd.concat(sea_parts, axis=1).reset_index()
    out_m = out_m[['playerId', 'matchId', 'seasonId',
                     'rank_x_ip', 'rank_x_op', 'rank_yf_ip']]
    return out_m, out_sea


def main():
    print("[1/4] Loading events…", flush=True)
    ev = load_events()
    print(f"  {len(ev):,} located events")

    print("[2/4] Flagging channels…", flush=True)
    ev = flag_channels(ev)
    print(f"  IP events: {ev['_ip'].sum():,}   OOP events: {ev['_op'].sum():,}")

    # Modal raw position per (player, season) — for validation only, the
    # model never uses it as a feature.
    modal = (ev.groupby(['player.id', 'seasonId', 'player.position']).size()
               .reset_index(name='n').sort_values('n', ascending=False)
               .drop_duplicates(['player.id', 'seasonId'])
               .rename(columns={'player.id': 'playerId',
                                  'player.position': 'raw_pos'})
               [['playerId', 'seasonId', 'raw_pos']])
    gk = set(modal[modal['raw_pos'] == 'GK'][['playerId', 'seasonId']]
               .itertuples(index=False, name=None))
    print(f"  GK player-seasons excluded: {len(gk):,}")

    print("  computing within-team rank features…", flush=True)
    rank_m, rank_s = rank_features(ev, gk)

    print("[3/4] Season-level signatures…", flush=True)
    sea = build_level(ev.rename(columns={'player.id': 'playerId'}),
                        ['playerId', 'seasonId'], MIN_EV_SEASON, 'season')
    sea = sea.merge(modal, on=['playerId', 'seasonId'], how='left')
    sea = sea.merge(rank_s, on=['playerId', 'seasonId'], how='left')
    sea = sea[~sea.apply(lambda r: (r['playerId'], r['seasonId']) in gk, axis=1)]
    # attach minutes for thresholds downstream
    try:
        gpa = pd.read_parquet(_DASH / 'gpa_player_season_values.parquet',
                                columns=['playerId', 'seasonId', 'mins_played', 'name'])
        gpa['playerId'] = pd.to_numeric(gpa['playerId'], errors='coerce').astype('Int64')
        gpa['seasonId'] = pd.to_numeric(gpa['seasonId'], errors='coerce').astype('Int64')
        sea['playerId'] = sea['playerId'].astype('Int64')
        sea['seasonId'] = sea['seasonId'].astype('Int64')
        sea = sea.merge(gpa, on=['playerId', 'seasonId'], how='left')
    except Exception as e:
        print(f"  [warn] no GPA minutes: {e}")
    sea.to_parquet(_HERE / 'role_features_season.parquet')
    print(f"  {len(sea):,} season signatures -> role_features_season.parquet")

    print("[4/4] Match-level signatures…", flush=True)
    mat = build_level(ev.rename(columns={'player.id': 'playerId'}),
                        ['playerId', 'matchId', 'seasonId'], MIN_EV_MATCH, 'match')
    mat['playerId'] = mat['playerId'].astype('Int64')
    mat = mat[~mat.apply(lambda r: (r['playerId'], r['seasonId']) in gk, axis=1)]
    rank_m['playerId'] = rank_m['playerId'].astype('Int64')
    mat = mat.merge(rank_m, on=['playerId', 'matchId', 'seasonId'], how='left')
    mat.to_parquet(_HERE / 'role_features_match.parquet')
    print(f"  {len(mat):,} match signatures -> role_features_match.parquet")
    print("done")


if __name__ == '__main__':
    main()
