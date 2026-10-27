#!/usr/bin/env python3
"""League conversion: how many ACP rating points is Campeonato worth
relative to Liga 3? Triangulated from independent anchors.

  A1 MOVERS    — same player, consecutive seasons, league switch,
                  benchmarked against a STAYER shrinkage model
                  (next_dev = r x cur_dev + age-bin intercepts) so
                  regression-to-the-mean from selection-on-extremes is
                  removed (Lucas's catch: movers up are selected HIGH,
                  movers down LOW — both inflate the naive gap; the
                  naive +2.23 was ~37% RTM artifact, corrected +1.40).
  A2 TEAMS     — same club appearing in both leagues (promotion/releg):
                  xGD/90 shift / within-league slope of team xGD on
                  team mean rating. Direction check; magnitude inflated
                  by squad turnover.
  (Transfermarkt anchor REMOVED — Lucas directive: TM values are
   near-nonexistent at L3/Camp level; ignore moving forward. Duel-
   ladder mover residuals corroborate direction: L3 > Camp.)

Output: league_conversion.json {delta_pts: Camp rating - delta = L3-
equivalent}, and acp_rating_abs / projection_abs columns appended to
the rating and projection parquets (L3 = reference scale).

Run from the Dashboard dir: python models/ratings/league_conversion.py
"""
from pathlib import Path
import sys as _sys_lc
_sys_lc.path.insert(0, str(Path(__file__).resolve().parents[2]))
from league_config import SEASON_YEAR, CAMP_SEASON_IDS, IN_PROGRESS_SEASON_IDS  # noqa: E402
import json
import pickle
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_DASH = _HERE.parent.parent
_GPA_DATA = _DASH.parent.parent / 'GPA Model Project v2' / 'parquet_data'
SEASON_YR = SEASON_YEAR   # from league_config (every season, incl. the current one)

o_all = pd.read_parquet(_HERE / 'acp_rating_per_player_season.parquet')
# Estimate from COMPLETED seasons only (a season in progress would move the
# deltas every matchday); the _abs columns below are still written for every
# row, the live season included.
o_all = o_all[~o_all['seasonId'].isin(IN_PROGRESS_SEASON_IDS)].copy()
o_all['yr'] = o_all['seasonId'].map(SEASON_YR)

# age expectation (same curve construction as build_projection, condensed)
with open(_DASH / 'player_details.pkl', 'rb') as f:
    det = pd.DataFrame(pickle.load(f))
det['birthDate'] = pd.to_datetime(det['birthDate'], errors='coerce')
bd = (det.dropna(subset=['birthDate']).drop_duplicates('playerId')
         .set_index('playerId')['birthDate'])
o_all['age'] = o_all['yr'] + 1.0 - (o_all['playerId'].map(bd).dt.year
                                      + o_all['playerId'].map(bd).dt.dayofyear / 365)
AGE_BINS = [(15, 20.5), (20.5, 23.5), (23.5, 26.5), (26.5, 29.5), (29.5, 42)]
PRIOR = {0: 1.2, 1: 0.6, 2: 0.0, 3: -0.6, 4: -1.4}


def bin_of(a):
    for i, (lo, hi) in enumerate(AGE_BINS):
        if lo <= a < hi:
            return i
    return 4


# A2 inputs (team xGD per club-season, player -> club), built once and
# truncated per fit below
ev = pd.concat([pd.read_parquet(_GPA_DATA / f'{f}.parquet',
    columns=['matchId', 'seasonId', 'team.name', 'shot.xg', 'player.id'])
    for f in ['liga3_portugal_events', 'campeonato_portugal_events']],
    ignore_index=True)
sh = ev.dropna(subset=['shot.xg'])
xg = sh.groupby(['matchId', 'team.name'])['shot.xg'].sum().reset_index()
sn = ev.dropna(subset=['team.name']).groupby('matchId')['seasonId'].first()
rows = []
for mid, g in xg.groupby('matchId'):
    if len(g) != 2:
        continue
    a, b = g.iloc[0], g.iloc[1]
    rows.append({'team': a['team.name'], 'seasonId': sn[mid],
                   'xgd': a['shot.xg'] - b['shot.xg'], 'm': 1})
    rows.append({'team': b['team.name'], 'seasonId': sn[mid],
                   'xgd': b['shot.xg'] - a['shot.xg'], 'm': 1})
tg_all = pd.DataFrame(rows).groupby(['team', 'seasonId']).sum().reset_index()
tg_all = tg_all[~tg_all['seasonId'].isin(IN_PROGRESS_SEASON_IDS)]   # completed seasons only
tg_all['xgd90'] = tg_all['xgd'] / tg_all['m']
tg_all['yr'] = tg_all['seasonId'].map(SEASON_YR)
tg_all['lg'] = np.where(tg_all['seasonId'].isin(CAMP_SEASON_IDS), 'CAMP', 'L3')
evp = ev.dropna(subset=['player.id', 'team.name']).copy()
evp['player.id'] = evp['player.id'].astype(int)
ptm = (evp.groupby(['player.id', 'seasonId'])['team.name']
          .agg(lambda x: x.mode().iloc[0]).reset_index())
ptm.columns = ['playerId', 'seasonId', 'team']


def fit_delta(max_yr, verbose=False):
    """The L3 - Camp gap from seasons <= max_yr ONLY. v7.0 NO LOOKAHEAD: a
    Camp season's _abs rating uses the gap as it could be measured by the
    end of that season (v6.9 applied one gap, fitted on every completed
    season, to all Camp rows — Camp 23/24 used 24/25-25/26 movers). A
    direction with no movers yet (Camp -> L3 before any Camp season ended)
    is left out instead of counted as 0."""
    say = print if verbose else (lambda *a, **k: None)
    o = o_all[o_all['yr'] <= max_yr].copy()
    # season pairs use each player-year's MAIN row (most minutes): a winter
    # mover has one row per league that year, and sort order alone used to
    # decide whether he counted as a mover or a stayer
    om = (o.sort_values(['yr', 'mins_played', 'seasonId'], kind='stable')
            .groupby(['playerId', 'yr']).tail(1))
    tg = tg_all[tg_all['yr'] <= max_yr]
    # within-league deltas only (league switches excluded -> no circularity)
    dl = []
    for pid, g in om.dropna(subset=['age']).sort_values('yr', kind='stable').groupby('playerId'):
        rr = g.to_dict('records')
        for a, b in zip(rr, rr[1:]):
            if (b['yr'] - a['yr'] == 1 and a['league'] == b['league']
                    and a['mins_played'] >= 500 and b['mins_played'] >= 500):
                dl.append((bin_of(a['age']), b['acp_rating'] - a['acp_rating']))
    DL = pd.DataFrame(dl, columns=['ab', 'd'])
    AGE_D = {b: (DL[DL['ab'] == b]['d'].sum() + PRIOR[b] * 60)
                / (len(DL[DL['ab'] == b]) + 60) for b in range(5)}

    say("=== A1: MOVERS (vs STAYER shrinkage model — RTM-corrected) ===")
    # Regression-to-mean correction (Lucas): movers up are selected on HIGH
    # observed ratings (lucky noise included) and would regress down anyway;
    # movers down the reverse — both inflate the naive gap. Benchmark every
    # mover against what an identical STAYER would do: fit on within-league
    # pairs  next_dev = r * cur_dev + c[age_bin], dev = rating minus the
    # league-season mean of rated players; mover residual = actual - that.
    # mean over the historic >=500' cohort (v6.4 admits sub-floor rows)
    o['_lgmean'] = (o['acp_rating'].where(o['mins_played'] >= 500)
                      .groupby([o['league'], o['seasonId']]).transform('mean'))
    o['_dev'] = o['acp_rating'] - o['_lgmean']
    om = om.merge(o[['playerId', 'seasonId', '_dev']], on=['playerId', 'seasonId'], how='left')
    stay, mv = [], []
    for pid, g in om.dropna(subset=['age']).sort_values('yr', kind='stable').groupby('playerId'):
        rr = g.to_dict('records')
        for a, b in zip(rr, rr[1:]):
            # estimators pinned to the historic >=500' regime
            if (b['yr'] - a['yr'] != 1 or a['mins_played'] < 500
                    or b['mins_played'] < 500):
                continue
            row = {'cur_dev': a['_dev'], 'nxt_dev': b['_dev'],
                     'ab': bin_of(a['age']),
                     'w': min(a['mins_played'], b['mins_played'])}
            if a['league'] == b['league']:
                stay.append(row)
            else:
                mv.append(row | {'dir': f"{a['league']}->{b['league']}"})
    ST = pd.DataFrame(stay)
    Xs = np.column_stack([ST['cur_dev'].values] +
                           [(ST['ab'] == k).astype(float).values for k in range(5)])
    beta_s, *_ = np.linalg.lstsq(Xs, ST['nxt_dev'].values, rcond=None)
    r_shrink = beta_s[0]
    say(f"  stayer shrinkage r = {r_shrink:.2f} (n={len(ST)}; a +10 dev player "
          f"regresses to +{10*r_shrink:.1f} in a year)")
    res = {}
    MV = pd.DataFrame(mv)
    if len(MV):
        exp_dev = (MV['cur_dev'] * r_shrink
                     + MV['ab'].map({k: beta_s[1 + k] for k in range(5)}))
        MV['resid'] = MV['nxt_dev'] - exp_dev
        say("  mover selection profile (Lucas's question — how selected are they?):")
        for d, g in MV.groupby('dir'):
            say(f"    {d}: mean prior dev {np.average(g['cur_dev'], weights=g['w']):+.1f} pts"
                  f" vs own-league average")
        for d, g in MV.groupby('dir'):
            m = np.average(g['resid'], weights=g['w'])
            se = g['resid'].std() / np.sqrt(len(g))
            res[d] = (m, se, len(g))
            say(f"  {d}: mean resid {m:+.2f} ±{se:.2f} (n={len(g)})")
    # L3 stronger by delta => moving UP costs delta, moving DOWN gains delta
    dirs = [(-res['CAMP->L3'][0], res['CAMP->L3'][1])] if 'CAMP->L3' in res else []
    dirs += [(res['L3->CAMP'][0], res['L3->CAMP'][1])] if 'L3->CAMP' in res else []
    a1 = float(np.mean([v for v, _ in dirs])) if dirs else np.nan
    a1_se = (float(np.sqrt(sum(se ** 2 for _, se in dirs)) / len(dirs))
             if dirs else np.nan)
    say(f"  A1 delta = {a1:+.2f} ±{a1_se:.2f} rating pts (L3 minus Camp)")

    say("\n=== A2: CROSS-LEAGUE TEAMS (xGD shift -> rating pts) ===")
    # slope: team xGD/90 per team-mean-rating point (within league-season)
    ot = o.merge(ptm, on=['playerId', 'seasonId'])
    tr = (ot.groupby(['team', 'seasonId'])
             .apply(lambda g: np.average(g['acp_rating'], weights=g['mins_played']),
                     include_groups=False).rename('mrate').reset_index())
    TT = tg.merge(tr, on=['team', 'seasonId'])
    TT['xgd_c'] = TT['xgd90'] - TT.groupby(['lg', 'seasonId'])['xgd90'].transform('mean')
    TT['mr_c'] = TT['mrate'] - TT.groupby(['lg', 'seasonId'])['mrate'].transform('mean')
    slope = (TT['xgd_c'] * TT['mr_c']).sum() / (TT['mr_c'] ** 2).sum()
    say(f"  slope: {slope:.4f} xGD/90 per rating pt")
    # same club consecutive seasons across leagues
    tm_pairs = []
    for team, g in tg.sort_values('yr').groupby('team'):
        rr = g.to_dict('records')
        for a, b in zip(rr, rr[1:]):
            if b['yr'] - a['yr'] == 1 and a['lg'] != b['lg']:
                # xGD vs the league it ENTERS: shift = xgd_b - xgd_a
                tm_pairs.append({'dir': f"{a['lg']}->{b['lg']}",
                                   'shift': b['xgd90'] - a['xgd90']})
    TP = pd.DataFrame(tm_pairs)
    say(f"  cross-league club season-pairs: {len(TP)} "
          f"({TP.groupby('dir').size().to_dict() if len(TP) else {}})")
    if len(TP) >= 6:
        s_up = -TP[TP['dir'] == 'CAMP->L3']['shift'].mean()
        s_dn = TP[TP['dir'] == 'L3->CAMP']['shift'].mean()
        a2_xgd = np.nanmean([s_up, s_dn])
        a2 = a2_xgd / slope
        a2_se = TP['shift'].std() / np.sqrt(len(TP)) / abs(slope)
        say(f"  A2 delta = {a2:+.2f} ±{a2_se:.2f} rating pts "
              f"(xGD shift {a2_xgd:+.2f} / slope)")
    else:
        a2, a2_se = np.nan, np.nan
        say("  A2: too few club pairs — skipped")

    # A3 MARKET VALUES — REMOVED (Lucas directive 2026-06-11): Transfermarkt
    # coverage at Liga 3 / Campeonato level is near-nonexistent (9 Camp
    # player-seasons) — ignore TM values moving forward.

    say("\n=== SYNTHESIS ===")
    anchors = {'A1 movers': (a1, a1_se), 'A2 teams': (a2, a2_se)}
    ws, vs = [], []
    for k, (v, se) in anchors.items():
        if v == v and se and se > 0:
            ws.append(1 / se ** 2)
            vs.append(v)
            say(f"  {k:<11} {v:+.2f} ±{se:.2f}")
    delta = float(np.average(vs, weights=ws)) if ws else np.nan
    delta_se = float(1 / np.sqrt(sum(ws))) if ws else np.nan
    say(f"  COMBINED: Liga 3 minus Campeonato = {delta:+.2f} ±{delta_se:.2f} "
          f"rating pts\n  (a Camp rating R ~ L3 rating R - {delta:.1f})")
    # RECRUITMENT conversion (Lucas 2026-06-11: Camp projections still too
    # high): the both-directions average is right for DESCRIPTION, but the
    # up-mover direction (+0.10) is contaminated by elite selection — those
    # players held their level BECAUSE they were the chosen few. For
    # projecting a typical Camp player into L3 terms (the recruitment
    # counterfactual), use the L3->Camp direction estimate, which is free
    # of up-selection (its own bias runs the other way, making it the
    # conservative bound).
    rec = res.get('L3->CAMP', (np.nan, np.nan, 0))
    say(f"  RECRUIT delta (down-mover bound): {rec[0]:+.2f} "
          f"±{rec[1]:.2f} — applied to projections")
    return {'delta': delta, 'delta_se': delta_se, 'delta_recruit': rec[0],
            'anchors': anchors}


LATEST_YR = int(o_all['yr'].max())
fit = fit_delta(LATEST_YR, verbose=True)
delta, delta_se, delta_recruit = fit['delta'], fit['delta_se'], fit['delta_recruit']
# per Camp season: the gap measurable by the end of THAT season (a season at
# or after the latest completed one simply gets today's fit)
r_seasons = pd.read_parquet(_HERE / 'acp_rating_per_player_season.parquet',
                            columns=['seasonId', 'league'])
camp_sids = sorted(set(r_seasons.loc[r_seasons['league'] == 'CAMP', 'seasonId']))
camp_delta = {}
for sid in camp_sids:
    y = SEASON_YR[int(sid)]
    camp_delta[int(sid)] = delta if y >= LATEST_YR else fit_delta(y)['delta']
    print(f"  Camp {sid} ({y}/{(y + 1) % 100:02d}) _abs gap: {camp_delta[int(sid)]:+.2f}"
          + ("" if y >= LATEST_YR else f"  (seasons <= {y} only)"))
json.dump({'reference': 'L3',
             'delta_pts_L3_minus_CAMP_descriptive': delta,
             'delta_se': delta_se,
             'delta_pts_recruit': delta_recruit,
             'delta_pts_descriptive_by_camp_season': {str(k): round(v, 3)
                                                      for k, v in camp_delta.items()},
             'anchors': {k: {'delta': (None if v != v else round(v, 3)),
                              'se': (None if se != se else round(se, 3))}
                          for k, (v, se) in fit['anchors'].items()}},
            open(_HERE / 'league_conversion.json', 'w'), indent=2)

# absolute columns: descriptive delta for the rating, recruit delta for
# the projection (L3 reference)
# the projection is TODAY's forecast, so it takes today's recruit gap; the
# rating takes its own Camp season's as-of gap
for fname, col, dd in [('acp_rating_per_player_season.parquet', 'acp_rating', None),
                          ('acp_projection.parquet', 'projection', delta_recruit)]:
    df = pd.read_parquet(_HERE / fname)
    gap = (df['seasonId'].map(camp_delta).fillna(delta) if dd is None
           else pd.Series(dd, index=df.index))
    df[col + '_abs'] = df[col] - np.where(df['league'] == 'CAMP', gap, 0.0)
    df.to_parquet(_HERE / fname)
    print(f"  {fname}: wrote {col}_abs")
