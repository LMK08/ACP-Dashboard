#!/usr/bin/env python3
"""Duel grading system — Phase 2: Glicko ladders (HOPS-style, extended).

Three latent traits across two contest types, processed chronologically
with Glicko-1 updates (rating + RD uncertainty; idle-time RD inflation):

  aerial : symmetric — both players rate on the 'aerial' ladder
  ground : asymmetric — defender's 'tackle' rating vs attacker's 'carry'
            rating (offensive duels AND dribble take-ons)

Draws (stalemates, no-clean-first-touch aerials) update at s = 0.5.

PRE-REGISTERED GATES (prequential — every prediction is made BEFORE the
update, so the 25/26 evaluation window is honestly out-of-sample):
  G1 predictive : Glicko win-prob must beat (a) running base rate and
                   (b) running position-group-pair bucket rates on
                   log-loss AND AUC (decisive contests, both players
                   >=10 prior contests).
  G2 calibration: decile reliability table — predicted ~ actual.
  G3 stability  : two independent runs on odd/even matches; final-rating
                   corr >= 0.5 for players with >=40 contests per half.
  G4 face       : top-10 per ladder are football-credible.

Outputs:
  duel_ratings.parquet            (playerId, trait, rating, rd, n, wins, xwins)
  duel_ratings_by_season.parquet  (end-of-season snapshots + season xwins)

Run from the Dashboard dir: python models/duels/glicko_ladders.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from math import log, sqrt, pi
from sklearn.metrics import roc_auc_score

_HERE = Path(__file__).resolve().parent
_DASH = _HERE.parent.parent

Q = log(10) / 400.0
INIT_R, INIT_RD, RD_MIN, RD_MAX = 1500.0, 350.0, 30.0, 350.0
C2_PER_DAY = (150.0**2 - 50.0**2) / 365.0     # idle inflation: 50->150 over a year
EVAL_SEASONS = {191782, 191779}
MIN_PRIOR = 10

GRP = {'GK': 'GK', 'CB': 'CB', 'LCB': 'CB', 'RCB': 'CB', 'LCB3': 'CB', 'RCB3': 'CB',
        'LB': 'FB', 'RB': 'FB', 'LB5': 'FB', 'RB5': 'FB', 'LWB': 'FB', 'RWB': 'FB',
        'CMF': 'CM', 'LCMF': 'CM', 'RCMF': 'CM', 'LCMF3': 'CM', 'RCMF3': 'CM',
        'DMF': 'CM', 'LDMF': 'CM', 'RDMF': 'CM',
        'AMF': 'AM', 'LAMF': 'AM', 'RAMF': 'AM', 'LMF': 'AM', 'RMF': 'AM',
        'LW': 'AM', 'RW': 'AM', 'LWF': 'AM', 'RWF': 'AM', 'CF': 'ST', 'SS': 'ST'}

con = pd.read_parquet(_HERE / 'contests.parquet')
con['date'] = pd.to_datetime(con['date'])
con = con.sort_values(['date', 'matchId', 't']).reset_index(drop=True)
con['grpA'] = con['posA'].map(GRP).fillna('UNK')
con['grpB'] = con['posB'].map(GRP).fillna('UNK')
print(f"{len(con):,} contests, {con['date'].min().date()} -> {con['date'].max().date()}")

# ---- v2: MATCHUP OFFSETS inside the Glicko expectation ----------------------
# v1 finding: pure Glicko lost to the position-pair bucket baseline — the
# bucket knows structure (CB-vs-ST base rates 50-69%) that ratings must
# learn slowly, and matchup mix polluted the ratings. Fix: estimate
# matchup offsets (in rating points) from the TRAIN window and add them to
# the expectation, so E = f(g·Δrating + M[gA,gB]) — ratings then measure
# skill BEYOND matchup, and the model nests the bucket baseline.
_tr = con[(~con['seasonId'].isin(EVAL_SEASONS)) & (con['scoreA'] != 0.5)]
_off = {}
for (lad, ga, gb), gsub in _tr.groupby(['ladder', 'grpA', 'grpB']):
    n = len(gsub)
    p = (gsub['scoreA'].sum() + 10 * 0.5) / (n + 10)      # shrink to 0.5
    _off[(lad, ga, gb)] = log(p / (1 - p)) / Q              # logit -> rating pts
# antisymmetrize the symmetric aerial ladder (A/B side is arbitrary)
for (lad, ga, gb) in list(_off.keys()):
    if lad == 'aerial':
        rev = _off.get((lad, gb, ga), 0.0)
        m = (_off[(lad, ga, gb)] - rev) / 2.0
        _off[(lad, ga, gb)] = m
        _off[(lad, gb, ga)] = -m
print(f"matchup offsets: {len(_off)} cells "
       f"(e.g. ground CB-vs-ST {_off.get(('ground','CB','ST'), 0):+.0f} pts)")


def g_of(rd):
    return 1.0 / sqrt(1.0 + 3.0 * Q * Q * rd * rd / (pi * pi))


def run(sub, collect_eval=False, eval_seasons=EVAL_SEASONS, init_rd=INIT_RD, tau=1.0):
    """Sequential Glicko over `sub`. Returns (state, eval_records, xw_season)."""
    R, RD, N, W, LAST = {}, {}, {}, {}, {}
    xw = {}
    ev_rows = []
    arr = sub[['ladder', 'playerA', 'playerB', 'scoreA', 'seasonId',
                 'grpA', 'grpB']].to_numpy()
    days = (sub['date'] - sub['date'].min()).dt.days.to_numpy()
    # prequential baselines
    base_n, base_s = {}, {}
    buck_n, buck_s = {}, {}
    for i in range(len(arr)):
        ladder, pA, pB, s, season, gA, gB = arr[i]
        d = days[i]
        tA = ('tackle', pA) if ladder == 'ground' else ('aerial', pA)
        tB = ('carry', pB) if ladder == 'ground' else ('aerial', pB)
        for key in (tA, tB):
            if key not in R:
                R[key], RD[key], N[key], W[key], LAST[key] = INIT_R, init_rd, 0, 0.0, d
            idle = max(0, d - LAST[key])
            if idle > 0:
                RD[key] = min(RD_MAX, sqrt(RD[key]**2 + C2_PER_DAY * idle))
            LAST[key] = d
        rA, rdA = R[tA], RD[tA]
        rB, rdB = R[tB], RD[tB]
        M = _off.get((ladder, gA, gB), 0.0)
        eA = 1.0 / (1.0 + 10 ** (-(g_of(rdB) * (rA - rB) + M) / 400.0))
        # prediction-side calibration: compress the skill term by tau —
        # the Glicko scale (chess-calibrated) overstates how much a rating
        # gap moves duel win probability. Updates keep tau=1 dynamics.
        eA_pred = 1.0 / (1.0 + 10 ** (-(tau * g_of(rdB) * (rA - rB) + M) / 400.0))
        # prequential eval bookkeeping (decisive only, warmed-up players)
        if collect_eval and season in eval_seasons and s != 0.5 \
                and N[tA] >= MIN_PRIOR and N[tB] >= MIN_PRIOR:
            bk = (ladder,)
            bb = (ladder, gA, gB)
            p_base = (base_s.get(bk, 0.0) + 1.0) / (base_n.get(bk, 0) + 2.0)
            p_buck = ((buck_s.get(bb, 0.0) + 5.0 * p_base)
                        / (buck_n.get(bb, 0) + 5.0))
            ev_rows.append((ladder, eA_pred, p_base, p_buck, s))
        if s != 0.5:
            bk = (ladder,); bb = (ladder, gA, gB)
            base_n[bk] = base_n.get(bk, 0) + 1; base_s[bk] = base_s.get(bk, 0.0) + s
            buck_n[bb] = buck_n.get(bb, 0) + 1; buck_s[bb] = buck_s.get(bb, 0.0) + s
        # xwins per (player, trait, season)
        kA = (tA[1], tA[0], season); kB = (tB[1], tB[0], season)
        xw[kA] = xw.get(kA, np.zeros(3)) + np.array([eA_pred, s, 1.0])
        xw[kB] = xw.get(kB, np.zeros(3)) + np.array([1.0 - eA_pred, 1.0 - s, 1.0])
        # Glicko-1 updates (simultaneous, from pre-update values); the
        # matchup offset enters each side's expectation with its sign.
        for (key, r, rd, r_o, rd_o, sc, mm) in (
                (tA, rA, rdA, rB, rdB, s, M),
                (tB, rB, rdB, rA, rdA, 1.0 - s, -M)):
            gg = g_of(rd_o)
            e = 1.0 / (1.0 + 10 ** (-(gg * (r - r_o) + mm) / 400.0))
            d2 = 1.0 / (Q * Q * gg * gg * e * (1.0 - e))
            denom = 1.0 / rd**2 + 1.0 / d2
            R[key] = r + (Q / denom) * gg * (sc - e)
            RD[key] = max(RD_MIN, sqrt(1.0 / denom))
            N[key] += 1
            W[key] += sc
    return (R, RD, N, W), ev_rows, xw


print("\n[0/3] tune INIT_RD on 24/25 validation window (25/26 untouched)…",
        flush=True)
VAL = {190090}
tune = {}
for ird in (60.0, 100.0):
    for tau in (0.3, 0.45, 0.6, 0.8, 1.0):
        _, evr, _ = run(con[~con['seasonId'].isin(EVAL_SEASONS)],
                          collect_eval=True, eval_seasons=VAL,
                          init_rd=ird, tau=tau)
        e = pd.DataFrame(evr, columns=['ladder', 'glicko', 'base', 'bucket', 's'])
        ll = -(e['s'] * np.log(np.clip(e['glicko'], 1e-6, 1)) +
                (1 - e['s']) * np.log(np.clip(1 - e['glicko'], 1e-6, 1))).mean()
        tune[(ird, tau)] = ll
        print(f"  INIT_RD={ird:>4.0f} tau={tau:.2f}  24/25 log-loss={ll:.4f}")
BEST_RD, BEST_TAU = min(tune, key=tune.get)
print(f"  -> INIT_RD={BEST_RD:.0f}, tau={BEST_TAU:.2f}")

print("\n[1/3] full chronological run + prequential gates…", flush=True)
(R, RD, N, W), ev_rows, xw = run(con, collect_eval=True, init_rd=BEST_RD, tau=BEST_TAU)
ev = pd.DataFrame(ev_rows, columns=['ladder', 'glicko', 'base', 'bucket', 's'])
print(f"  eval contests (25/26, decisive, warmed-up): {len(ev):,}")


def lloss(p, s):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(s * np.log(p) + (1 - s) * np.log(1 - p)).mean())


print(f"\n  G1 predictive ({'model':<8}{'log-loss':>10}{'AUC':>8}):")
for lad, gsub in ev.groupby('ladder'):
    print(f"  -- {lad} (n={len(gsub):,}, base rate {gsub['s'].mean():.3f})")
    for mname in ['base', 'bucket', 'glicko']:
        auc = roc_auc_score(gsub['s'], gsub[mname])
        print(f"     {mname:<8}{lloss(gsub[mname], gsub['s']):>10.4f}{auc:>8.3f}")

print("\n  G2 calibration (glicko, deciles of predicted p):")
ev['bin'] = pd.qcut(ev['glicko'], 10, duplicates='drop')
cal = ev.groupby('bin', observed=True).agg(pred=('glicko', 'mean'),
                                              actual=('s', 'mean'),
                                              n=('s', 'size'))
print((cal.round(3)).to_string())

print("\n[2/3] G3 split-half (independent odd/even-match runs)…", flush=True)
half_states = []
for par in (0, 1):
    st, _, _ = run(con[con['matchId'].astype('int64') % 2 == par], init_rd=BEST_RD, tau=BEST_TAU)
    half_states.append(st)
rows = []
for trait in ['aerial', 'tackle', 'carry']:
    (R0, _, N0, _), (R1, _, N1, _) = half_states
    common = [k for k in R0 if k in R1 and k[0] == trait
                and N0[k] >= 40 and N1[k] >= 40]
    a = np.array([R0[k] for k in common]); b = np.array([R1[k] for k in common])
    r = float(np.corrcoef(a, b)[0, 1])
    rows.append((trait, r, len(common)))
    print(f"  {trait:<8} split-half r = {r:.3f} (n={len(common)})")

print("\n[3/3] outputs + face validity…", flush=True)
out = pd.DataFrame([{'playerId': k[1], 'trait': k[0], 'rating': R[k],
                       'rd': RD[k], 'n': N[k], 'wins': W[k]} for k in R])
xws = pd.DataFrame([{'playerId': p, 'trait': t, 'seasonId': s,
                       'xwins': v[0], 'wins': v[1], 'n': v[2]}
                      for (p, t, s), v in xw.items()])
out.to_parquet(_HERE / 'duel_ratings.parquet')
xws.to_parquet(_HERE / 'duel_ratings_by_season.parquet')
print(f"  duel_ratings.parquet ({len(out):,} player-traits), "
       f"duel_ratings_by_season.parquet ({len(xws):,})")

gpa = pd.read_parquet(_DASH / 'gpa_player_season_values.parquet',
                        columns=['playerId', 'name'])
nm = gpa.groupby('playerId')['name'].first().to_dict()
for trait in ['aerial', 'tackle', 'carry']:
    top = (out[(out['trait'] == trait) & (out['n'] >= 150)]
             .nlargest(10, 'rating'))
    names = [f"{str(nm.get(int(p), p))} ({int(n)} duels, {r:.0f}±{rd:.0f})"
              for p, n, r, rd in zip(top['playerId'], top['n'],
                                       top['rating'], top['rd'])]
    print(f"\n  TOP {trait.upper()} (>=150 contests):")
    for s in names:
        print(f"    {s}")
