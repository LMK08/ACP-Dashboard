#!/usr/bin/env python3
"""Are the dashboard's territorial / zone-denial metrics rating-grade?

Faithful multi-season replication of app.py's Defensive Area pipeline
(68% covariance ellipse of open-play defensive actions; opponent xT
moves flipped into the defender's frame; OE = normalized by Expected xT
at the zone's pitch location), then the standard battery:
  - split-half (odd/even matches, ellipse rebuilt per half — fully
    independent measurements)
  - YoY (same role, consecutive seasons)
  - overlap with existing rating axes (resp/qual/rating)
  - TEAM-share: how much of the player metric is just his team's
    defensive system (R^2 on team-season means)

Run from the Dashboard dir: python models/ratings/eval_territorial.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, chi2

_HERE = Path(__file__).resolve().parent
_DASH = _HERE.parent.parent
_GPA_DATA = _DASH.parent.parent / 'GPA Model Project v2' / 'parquet_data'
CHI2_68 = chi2(2).ppf(0.68)
MIN_DEF = 5
XT = np.array([[0.01,0.01,0.01,0.01,0.01,0.01,0.02,0.02,0.03,0.03,0.04,0.04],
                [0.01,0.01,0.01,0.01,0.01,0.02,0.02,0.02,0.03,0.04,0.05,0.05],
                [0.01,0.01,0.01,0.01,0.01,0.02,0.02,0.02,0.03,0.05,0.06,0.06],
                [0.01,0.01,0.01,0.01,0.01,0.02,0.02,0.02,0.04,0.11,0.26,0.26],
                [0.01,0.01,0.01,0.01,0.01,0.02,0.02,0.02,0.04,0.11,0.26,0.26],
                [0.01,0.01,0.01,0.01,0.01,0.02,0.02,0.02,0.03,0.05,0.06,0.06],
                [0.01,0.01,0.01,0.01,0.01,0.02,0.02,0.02,0.03,0.04,0.05,0.05],
                [0.01,0.01,0.01,0.01,0.01,0.01,0.02,0.02,0.03,0.03,0.04,0.04]])
XR, XC = XT.shape
SP_TAGS = {'corner', 'free_kick', 'goal_kick', 'throw_in', 'penalty'}

print("[1/4] events…", flush=True)
cols = ['matchId', 'seasonId', 'player.id', 'team.name', 'player.position',
         'type.primary', 'type.secondary', 'location.x', 'location.y',
         'pass.accurate', 'pass.endLocation.x', 'pass.endLocation.y',
         'carry.endLocation.x', 'carry.endLocation.y',
         'possession.types', 'possession.eventIndex']
ev = pd.concat([pd.read_parquet(_GPA_DATA / f'{f}.parquet', columns=cols)
                  for f in ['liga3_portugal_events', 'campeonato_portugal_events']],
                 ignore_index=True)
sec = ev['type.secondary'].apply(
    lambda x: set(x) if isinstance(x, (list, np.ndarray)) else set())
in_sp = ev['possession.types'].apply(
    lambda x: bool(set(x) & SP_TAGS) if isinstance(x, (list, np.ndarray)) else False)
open_play = ~(in_sp & (ev['possession.eventIndex'].fillna(99) <= 5))

def_mask = (ev['type.primary'].isin(['interception', 'clearance'])
              | sec.apply(lambda s: bool(s & {'defensive_duel', 'sliding_tackle',
                                                'aerial_duel', 'recovery'})))
dfa = ev[def_mask & open_play & ev['location.x'].notna()
           & ev['location.y'].notna() & ev['player.id'].notna()].copy()
dfa['player.id'] = dfa['player.id'].astype(int)
dfa['x_m'] = dfa['location.x'] * 1.05
dfa['y_m'] = dfa['location.y'] * 0.68

mv = ev[ev['type.primary'].isin(['pass', 'touch', 'acceleration']) & open_play].copy()
ok = ((mv['type.primary'] == 'pass') & (mv['pass.accurate'] == True)) \
       | mv['type.primary'].isin(['touch', 'acceleration'])
mv = mv[ok]
mv['end_x'] = np.where(mv['type.primary'] == 'pass',
                          mv['pass.endLocation.x'], mv['carry.endLocation.x'])
mv['end_y'] = np.where(mv['type.primary'] == 'pass',
                          mv['pass.endLocation.y'], mv['carry.endLocation.y'])
mv = mv.dropna(subset=['end_x', 'end_y', 'location.x', 'location.y'])
for a, b in [('s', 'location.x'), ('e', 'end_x')]:
    mv[a + 'c'] = np.clip((mv[b] / 100 * XC).astype(int), 0, XC - 1)
for a, b in [('s', 'location.y'), ('e', 'end_y')]:
    mv[a + 'r'] = np.clip((mv[b] / 100 * XR).astype(int), 0, XR - 1)
mv['xt'] = XT[mv['er'], mv['ec']] - XT[mv['sr'], mv['sc']]
mv['ex_m'] = (100 - mv['end_x']) * 1.05
mv['ey_m'] = (100 - mv['end_y']) * 0.68
mv['sx_m'] = (100 - mv['location.x']) * 1.05
mv['sy_m'] = (100 - mv['location.y']) * 0.68

pp = ev[(ev['type.primary'] == 'pass') & open_play].dropna(
    subset=['pass.endLocation.x', 'pass.endLocation.y']).copy()
pp['ex_m'] = (100 - pp['pass.endLocation.x']) * 1.05
pp['ey_m'] = (100 - pp['pass.endLocation.y']) * 0.68
pp['acc'] = (pp['pass.accurate'] == True).astype(float)
print(f"  def actions {len(dfa):,}; xT moves {len(mv):,}; passes {len(pp):,}")

rat = pd.read_parquet(_HERE / 'acp_rating_per_player_season.parquet')
eligible = set(zip(rat['playerId'], rat['seasonId']))
mins = rat.set_index(['playerId', 'seasonId'])['mins_played'].to_dict()

print("[2/4] per (player, season, half) territorial metrics…", flush=True)
mv_g = {k: v for k, v in mv.groupby('matchId')}
pp_g = {k: v for k, v in pp.groupby('matchId')}
rows = []
for (season, half), dsub in dfa.groupby([dfa['seasonId'],
                                            dfa['matchId'].astype('int64') % 2]):
    # ellipses per player from THIS half's actions (primary position only)
    epars = {}
    pteam = {}
    pmatches = {}
    for pid, g in dsub.groupby('player.id'):
        if (pid, season) not in eligible:
            continue
        pos = g['player.position'].dropna()
        if not pos.empty:
            g = g[g['player.position'] == pos.value_counts().index[0]]
        if len(g) < MIN_DEF:
            continue
        C = g[['x_m', 'y_m']].values
        m = C.mean(axis=0)
        cv = np.cov(C.T)
        det = np.linalg.det(cv)
        if det <= 1e-10:
            continue
        epars[pid] = (m, np.linalg.inv(cv), np.pi * np.sqrt(det) * CHI2_68)
        pteam[pid] = g['team.name'].mode().iloc[0]
        pmatches[pid] = set(g['matchId'].unique())
    if not epars:
        continue
    # expected xT at zone (opponent frame sampling of grid over ellipse)
    exp_xt = {}
    for pid, (m, cvi, _) in epars.items():
        xs = np.linspace(m[0] - 30, m[0] + 30, 31)
        ys = np.linspace(m[1] - 30, m[1] + 30, 31)
        gx, gy = np.meshgrid(xs, ys)
        P = np.column_stack([gx.ravel(), gy.ravel()])
        d = P - m
        ins = np.sum(d @ cvi * d, axis=1) < CHI2_68
        Pi = P[ins]
        if len(Pi) == 0:
            exp_xt[pid] = np.nan
            continue
        ox = 100 - Pi[:, 0] / 1.05
        oy = 100 - Pi[:, 1] / 0.68
        r = np.clip((oy / 100 * XR).astype(int), 0, XR - 1)
        c = np.clip((ox / 100 * XC).astype(int), 0, XC - 1)
        exp_xt[pid] = XT[r, c].mean()
    acc = {pid: np.zeros(5) for pid in epars}   # xt_into, xt_from, p_tot, p_acc, n_match
    by_match = {}
    for pid, ms in pmatches.items():
        for mid in ms:
            by_match.setdefault(mid, []).append(pid)
    for mid, pids in by_match.items():
        M = mv_g.get(mid)
        P = pp_g.get(mid)
        for pid in pids:
            m, cvi, _ = epars[pid]
            tn = pteam[pid]
            if M is not None:
                om = M[M['team.name'] != tn]
                if len(om):
                    E = om[['ex_m', 'ey_m']].values - m
                    S = om[['sx_m', 'sy_m']].values - m
                    xt = om['xt'].values
                    acc[pid][0] += xt[np.sum(E @ cvi * E, axis=1) < CHI2_68].sum()
                    acc[pid][1] += xt[np.sum(S @ cvi * S, axis=1) < CHI2_68].sum()
            if P is not None:
                op = P[P['team.name'] != tn]
                if len(op):
                    E = op[['ex_m', 'ey_m']].values - m
                    inp = np.sum(E @ cvi * E, axis=1) < CHI2_68
                    acc[pid][2] += inp.sum()
                    acc[pid][3] += op['acc'].values[inp].sum()
            acc[pid][4] += 1
    for pid, v in acc.items():
        mn = mins.get((pid, season), np.nan)
        if not mn or mn != mn or v[4] < 5:
            continue
        m90 = (mn / 2) / 90.0          # half the matches -> half the minutes
        rows.append({
            'playerId': pid, 'seasonId': season, 'half': half,
            'area': epars[pid][2], 'exp_xt': exp_xt[pid],
            'xt_into90': v[0] / m90, 'xt_from90': v[1] / m90,
            'pass_into_pct': v[3] / v[2] if v[2] > 20 else np.nan,
            'n_matches': v[4]})
T = pd.DataFrame(rows)
T['xt_into_oe'] = T['xt_into90'] / T['exp_xt']
T['xt_from_oe'] = T['xt_from90'] / T['exp_xt']
T['terr_dom_oe'] = T['xt_into90'] / (T['area'] * T['exp_xt']) * 10000
T.to_parquet(_HERE / 'territorial_halves.parquet')
print(f"  {len(T):,} player-season-half rows -> territorial_halves.parquet")

print("[3/4] battery…", flush=True)
METS = ['area', 'xt_into90', 'xt_into_oe', 'pass_into_pct',
         'xt_from90', 'xt_from_oe', 'terr_dom_oe']
meta = rat[['playerId', 'seasonId', 'role', 'league', 'position_group',
              'defr_pct', 'qual_pct', 'acp_rating']].copy()
SEASON_YR = {188221: 2021, 188222: 2022, 189147: 2023, 190090: 2024,
              191782: 2025, 190230: 2023, 191779: 2025}
meta['yr'] = meta['seasonId'].map(SEASON_YR)
H = T.merge(meta, on=['playerId', 'seasonId'])
DEFROLES = {'Central Defender', 'Wide Defender'}
print(f"{'metric':<14}{'SH all':>8}{'SH def':>8}{'YoY def':>9}"
       f"{'r resp':>8}{'r qual':>8}{'r rtg':>7}{'team R2':>9}")
for met in METS:
    a = H[H['half'] == 0][['playerId', 'seasonId', 'role', met]]
    b = H[H['half'] == 1][['playerId', 'seasonId', met]]
    m2 = a.merge(b, on=['playerId', 'seasonId'], suffixes=('_a', '_b')).dropna()
    sh_all = pearsonr(m2[met + '_a'], m2[met + '_b'])[0] if len(m2) > 50 else np.nan
    md = m2[m2['role'].isin(DEFROLES)]
    sh_def = pearsonr(md[met + '_a'], md[met + '_b'])[0] if len(md) > 50 else np.nan
    # season-level value = mean of halves
    S = (H.groupby(['playerId', 'seasonId', 'role', 'league', 'yr'])[met]
           .mean().reset_index().dropna())
    P = []
    for pid, g in S[S['role'].isin(DEFROLES)].sort_values('yr').groupby('playerId'):
        rr = g.to_dict('records')
        for x, y in zip(rr, rr[1:]):
            if y['yr'] - x['yr'] == 1 and x['role'] == y['role']:
                P.append((x[met], y[met]))
    P = pd.DataFrame(P)
    yoy = pearsonr(P[0], P[1])[0] if len(P) > 40 else np.nan
    Sm = S.merge(meta, on=['playerId', 'seasonId', 'role', 'league', 'yr'])
    Smd = Sm[Sm['role'].isin(DEFROLES)]
    rresp = spearmanr(Smd[met], Smd['defr_pct'])[0]
    rqual = spearmanr(Smd[met], Smd['qual_pct'])[0]
    rrtg = spearmanr(Smd[met], Smd['acp_rating'])[0]
    # team-share: R2 of player value on (team x season) mean — need team
    tm = (dfa[dfa['player.id'].isin(Smd['playerId'])]
            .groupby(['player.id', 'seasonId'])['team.name']
            .agg(lambda x: x.mode().iloc[0]).rename('team').reset_index())
    tm.columns = ['playerId', 'seasonId', 'team']
    Smt = Smd.merge(tm, on=['playerId', 'seasonId'], how='left')
    grp = Smt.groupby(['team', 'seasonId'])[met].transform('mean')
    ssr = ((Smt[met] - grp) ** 2).sum()
    sst = ((Smt[met] - Smt[met].mean()) ** 2).sum()
    team_r2 = 1 - ssr / sst if sst > 0 else np.nan
    print(f"{met:<14}{sh_all:>8.2f}{sh_def:>8.2f}{yoy:>9.2f}"
           f"{rresp:>8.2f}{rqual:>8.2f}{rrtg:>7.2f}{team_r2:>9.2f}")
print("\n[4/4] reference: resp YoY 0.41 | qual split-half ~0.86 (DWAE) | "
       "GPA Interrupting per-pos 0.04-0.19")
