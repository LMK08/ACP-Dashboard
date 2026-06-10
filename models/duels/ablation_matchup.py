#!/usr/bin/env python3
"""Ablation: is the POSITIONAL matchup factor needed, or can situational
context replace it?

Configs (offsets enter the Glicko expectation; ratings learn the rest):
  A none        — pure Glicko (ratings absorb everything persistent)
  B positional  — position-group pair offsets (current production)
  C situational — ground: att_kind + zone + phase; aerial: height-diff
                   bins. NO positional terms — ratings stay ABSOLUTE.
  D situ+pos    — both.

Each config: tau tuned on the 24/25 window (INIT_RD fixed at 100), then
prequential evaluation on untouched 25/26 (log-loss/AUC vs the running
position-pair bucket baseline) + split-half rating stability.

Run from the Dashboard dir: python models/duels/ablation_matchup.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from math import log, sqrt, pi
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

_HERE = Path(__file__).resolve().parent
Q = log(10) / 400.0
INIT_R, IRD, RD_MIN, RD_MAX = 1500.0, 100.0, 30.0, 350.0
C2 = (150.0**2 - 50.0**2) / 365.0
EVAL = {191782, 191779}
VAL = {190090}
MIN_PRIOR = 10
GRP = {'GK': 'GK', 'CB': 'CB', 'LCB': 'CB', 'RCB': 'CB', 'LCB3': 'CB', 'RCB3': 'CB',
        'LB': 'FB', 'RB': 'FB', 'LB5': 'FB', 'RB5': 'FB', 'LWB': 'FB', 'RWB': 'FB',
        'CMF': 'CM', 'LCMF': 'CM', 'RCMF': 'CM', 'LCMF3': 'CM', 'RCMF3': 'CM',
        'DMF': 'CM', 'LDMF': 'CM', 'RDMF': 'CM', 'AMF': 'AM', 'LAMF': 'AM',
        'RAMF': 'AM', 'LMF': 'AM', 'RMF': 'AM', 'LW': 'AM', 'RW': 'AM',
        'LWF': 'AM', 'RWF': 'AM', 'CF': 'ST', 'SS': 'ST'}

con = pd.read_parquet(_HERE / 'contests.parquet')
con['date'] = pd.to_datetime(con['date'])
con = con.sort_values(['date', 'matchId', 't']).reset_index(drop=True)
con['grpA'] = con['posA'].map(GRP).fillna('UNK')
con['grpB'] = con['posB'].map(GRP).fillna('UNK')
con['hdiff'] = (pd.to_numeric(con['heightA'], errors='coerce')
                  - pd.to_numeric(con['heightB'], errors='coerce'))
con['hbin'] = pd.cut(con['hdiff'], [-99, -8, -3, 3, 8, 99],
                       labels=['<<', '<', '=', '>', '>>']).astype(str)
con['zx'] = con['zx'].fillna(1).astype(int)
con['phase'] = con['phase'].fillna('settled')
g_of = lambda rd: 1.0 / sqrt(1.0 + 3.0 * Q * Q * rd * rd / (pi * pi))
tr = con[(~con['seasonId'].isin(EVAL)) & (con['scoreA'] != 0.5)]


def logit_offsets(df, keys):
    """Smoothed per-bucket logit offsets (rating points) from train data."""
    out = {}
    for vals, g in df.groupby(keys):
        p = (g['scoreA'].sum() + 5.0) / (len(g) + 10.0)
        out[vals if isinstance(vals, tuple) else (vals,)] = log(p / (1 - p)) / Q
    return out


# --- offset builders per config ---
pos_off = logit_offsets(tr, ['ladder', 'grpA', 'grpB'])
for (lad, ga, gb) in list(pos_off):
    if lad == 'aerial':
        m = (pos_off[(lad, ga, gb)] - pos_off.get((lad, gb, ga), 0.0)) / 2.0
        pos_off[(lad, ga, gb)] = m; pos_off[(lad, gb, ga)] = -m

situ_g = logit_offsets(tr[tr['ladder'] == 'ground'],
                         ['att_kind', 'zx', 'phase'])
situ_a = logit_offsets(tr[tr['ladder'] == 'aerial'], ['hbin'])
_REV = {'<<': '>>', '<': '>', '=': '=', '>': '<', '>>': '<<'}
for k in list(situ_a):       # antisymmetrize height bins (A/B arbitrary)
    rev = _REV.get(k[0])
    if rev is None:           # 'nan' bin (missing height) -> neutral
        situ_a[k] = 0.0
        continue
    m = (situ_a[k] - situ_a.get((rev,), 0.0)) / 2.0
    situ_a[k] = m; situ_a[(rev,)] = -m


def M_of(config, row):
    lad, ga, gb, ak, zx, ph, hb = row
    m = 0.0
    if config in ('B', 'D'):
        m += pos_off.get((lad, ga, gb), 0.0)
    if config in ('C', 'D'):
        if lad == 'ground':
            m += situ_g.get((ak, zx, ph), 0.0)
        else:
            m += situ_a.get((hb,), 0.0)
    return m


def run(sub, config, tau, eval_seasons, halfpar=None):
    if halfpar is not None:
        sub = sub[sub['matchId'].astype('int64') % 2 == halfpar]
    R, RD, N, LAST = {}, {}, {}, {}
    ev_rows = []
    arr = sub[['ladder', 'playerA', 'playerB', 'scoreA', 'seasonId',
                 'grpA', 'grpB', 'att_kind', 'zx', 'phase', 'hbin']].to_numpy()
    days = (sub['date'] - sub['date'].min()).dt.days.to_numpy()
    bn, bs = {}, {}
    for i in range(len(arr)):
        lad, pA, pB, s, season, gA, gB, ak, zx, ph, hb = arr[i]
        d = days[i]
        tA = ('tackle', pA) if lad == 'ground' else ('aerial', pA)
        tB = ('carry', pB) if lad == 'ground' else ('aerial', pB)
        for k in (tA, tB):
            if k not in R:
                R[k], RD[k], N[k], LAST[k] = INIT_R, IRD, 0, d
            idle = max(0, d - LAST[k])
            if idle:
                RD[k] = min(RD_MAX, sqrt(RD[k]**2 + C2 * idle))
            LAST[k] = d
        M = M_of(config, (lad, gA, gB, ak, zx, ph, hb))
        eP = 1.0 / (1.0 + 10 ** (-(tau * g_of(RD[tB]) * (R[tA] - R[tB]) + M) / 400.0))
        if season in eval_seasons and s != 0.5 \
                and N[tA] >= MIN_PRIOR and N[tB] >= MIN_PRIOR:
            bb = (lad, gA, gB)
            pb = (bs.get(bb, 0.0) + 2.5) / (bn.get(bb, 0) + 5.0)
            ev_rows.append((lad, eP, pb, s))
        if s != 0.5:
            bb = (lad, gA, gB)
            bn[bb] = bn.get(bb, 0) + 1; bs[bb] = bs.get(bb, 0.0) + s
        for (k, r, rd, ro, rdo, sc, mm) in ((tA, R[tA], RD[tA], R[tB], RD[tB], s, M),
                                              (tB, R[tB], RD[tB], R[tA], RD[tA], 1 - s, -M)):
            gg = g_of(rdo)
            e = 1.0 / (1.0 + 10 ** (-(gg * (r - ro) + mm) / 400.0))
            d2 = 1.0 / (Q * Q * gg * gg * e * (1 - e))
            den = 1.0 / rd**2 + 1.0 / d2
            R[k] = r + (Q / den) * gg * (sc - e)
            RD[k] = max(RD_MIN, sqrt(1.0 / den))
            N[k] += 1
    return (R, N), ev_rows


def lloss(p, s):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(s * np.log(p) + (1 - s) * np.log(1 - p)).mean())


print(f"{'config':<14}{'tau*':>5} | {'aer LL':>8}{'aer AUC':>8} | "
       f"{'grd LL':>8}{'grd AUC':>8} | split-half aer/tck/cry")
print('-' * 92)
results = {}
for config in ['A', 'B', 'C', 'D']:
    best = None
    for tau in (0.3, 0.45, 0.6, 0.8, 1.0):
        _, evr = run(con[~con['seasonId'].isin(EVAL)], config, tau, VAL)
        e = pd.DataFrame(evr, columns=['ladder', 'g', 'b', 's'])
        ll = lloss(e['g'], e['s'])
        if best is None or ll < best[1]:
            best = (tau, ll)
    tau = best[0]
    _, evr = run(con, config, tau, EVAL)
    e = pd.DataFrame(evr, columns=['ladder', 'g', 'b', 's'])
    stats = {}
    for lad in ('aerial', 'ground'):
        gs = e[e['ladder'] == lad]
        stats[lad] = (lloss(gs['g'], gs['s']), roc_auc_score(gs['s'], gs['g']))
    # split-half stability
    halves = []
    for par in (0, 1):
        (R, N), _ = run(con, config, tau, set(), halfpar=par)
        halves.append((R, N))
    sh = {}
    for trait in ('aerial', 'tackle', 'carry'):
        (R0, N0), (R1, N1) = halves
        common = [k for k in R0 if k in R1 and k[0] == trait
                    and N0[k] >= 40 and N1[k] >= 40]
        sh[trait] = pearsonr([R0[k] for k in common],
                               [R1[k] for k in common])[0]
    results[config] = (tau, stats, sh)
    print(f"{config:<14}{tau:>5.2f} | {stats['aerial'][0]:>8.4f}{stats['aerial'][1]:>8.3f} | "
           f"{stats['ground'][0]:>8.4f}{stats['ground'][1]:>8.3f} | "
           f"{sh['aerial']:.3f}/{sh['tackle']:.3f}/{sh['carry']:.3f}")
# bucket reference on the same eval set (from config B's eval rows)
_, evr = run(con, 'B', results['B'][0], EVAL)
e = pd.DataFrame(evr, columns=['ladder', 'g', 'b', 's'])
for lad in ('aerial', 'ground'):
    gs = e[e['ladder'] == lad]
    print(f"bucket baseline {lad}: LL {lloss(gs['b'], gs['s']):.4f}  "
           f"AUC {roc_auc_score(gs['s'], gs['b']):.3f}")
