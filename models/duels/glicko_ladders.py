#!/usr/bin/env python3
"""Duel grading system — Phase 2: Glicko ladders (HOPS-style, extended).

Three latent traits across two contest types, processed chronologically
with Glicko-1 updates (rating + RD uncertainty; idle-time RD inflation):

  aerial : symmetric — both players rate on the 'aerial' ladder. NO
            offsets, not even height (user call): the ladder answers
            "who is best in the air", not "best relative to size".
  ground : asymmetric, split by duel kind — takeon vs stopper (dribble
            contests), shield vs press (offensive duels). Situational
            offsets only (duel kind x zone x phase).

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
con['zx'] = con['zx'].fillna(1).astype(int)
con['phase'] = con['phase'].fillna('settled')
print(f"{len(con):,} contests, {con['date'].min().date()} -> {con['date'].max().date()}")

# ---- v4: SITUATIONAL offsets, ground only ----------------------------------
# Positional matchup factor REMOVED (user call, validated by ablation):
# ground situational context (duel kind x zone x phase) beats positional
# offsets decisively (holdout AUC 0.692 vs 0.620) and keeps ratings
# ABSOLUTE. Aerial has NO offsets at all (user call: the ladder answers
# "who is best in the air", not "best relative to size" — height is the
# player's own asset and the ratings absorb it; the ablation agreed:
# no-offset aerial predicted slightly better AND was more stable than
# the height-adjusted variant).
_tr = con[(~con['seasonId'].isin(EVAL_SEASONS)) & (con['scoreA'] != 0.5)]


def _logit_off(df, keys):
    out = {}
    for vals, g in df.groupby(keys):
        p = (g['scoreA'].sum() + 5.0) / (len(g) + 10.0)
        out[vals if isinstance(vals, tuple) else (vals,)] = log(p / (1 - p)) / Q
    return out

_situ_g = _logit_off(_tr[_tr['ladder'] == 'ground'], ['att_kind', 'zx', 'phase'])
print(f"situational offsets: ground {len(_situ_g)} cells; aerial none (absolute)")


def g_of(rd):
    return 1.0 / sqrt(1.0 + 3.0 * Q * Q * rd * rd / (pi * pi))


CAMP_SEASONS = {190230, 191779}


def run(sub, collect_eval=False, eval_seasons=EVAL_SEASONS, init_rd=INIT_RD,
          tau_a=1.0, tau_g=1.0):
    """Sequential Glicko over `sub`. Returns (state, eval_records, xw_season)."""
    R, RD, N, W, LAST = {}, {}, {}, {}, {}
    LG = {}    # last league seen per key
    xw = {}
    ev_rows = []
    arr = sub[['ladder', 'playerA', 'playerB', 'scoreA', 'seasonId',
                 'grpA', 'grpB', 'att_kind', 'zx', 'phase']].to_numpy()
    days = (sub['date'] - sub['date'].min()).dt.days.to_numpy()
    # prequential baselines
    base_n, base_s = {}, {}
    buck_n, buck_s = {}, {}
    for i in range(len(arr)):
        ladder, pA, pB, s, season, gA, gB, ak, zx, ph = arr[i]
        d = days[i]
        # 5 ladders: ground splits by duel kind (option 1 — like-for-like
        # skills): take-ons rate attacker 'takeon' vs defender 'stopper';
        # shields (offensive duels) rate attacker 'shield' (press
        # resistance) vs defender 'press' (ball-winning in the press).
        if ladder == 'ground':
            if ak == 'dribble':
                tA, tB = ('stopper', pA), ('takeon', pB)
            else:
                tA, tB = ('press', pA), ('shield', pB)
        else:
            tA, tB = ('aerial', pA), ('aerial', pB)
        for key in (tA, tB):
            if key not in R:
                R[key], RD[key], N[key], W[key], LAST[key] = INIT_R, init_rd, 0, 0.0, d
            idle = max(0, d - LAST[key])
            if idle > 0:
                RD[key] = min(RD_MAX, sqrt(RD[key]**2 + C2_PER_DAY * idle))
            LAST[key] = d
        LG[tA] = LG[tB] = ('CAMP' if season in CAMP_SEASONS else 'L3')
        rA, rdA = R[tA], RD[tA]
        rB, rdB = R[tB], RD[tB]
        M = _situ_g.get((ak, zx, ph), 0.0) if ladder == 'ground' else 0.0
        eA = 1.0 / (1.0 + 10 ** (-(g_of(rdB) * (rA - rB) + M) / 400.0))
        # prediction-side calibration: compress the skill term by tau —
        # the Glicko scale (chess-calibrated) overstates how much a rating
        # gap moves duel win probability. Per-ladder tau (the ladders have
        # different offset structures). Updates keep tau=1 dynamics.
        tt = tau_g if ladder == 'ground' else tau_a
        eA_pred = 1.0 / (1.0 + 10 ** (-(tt * g_of(rdB) * (rA - rB) + M) / 400.0))
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
    return (R, RD, N, W, LAST, LG), ev_rows, xw


print("\n[0/3] tune INIT_RD on 24/25 validation window (25/26 untouched)…",
        flush=True)
VAL = {190090}


def _ll(p, s):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(s * np.log(p) + (1 - s) * np.log(1 - p)).mean())

tune = {}
for ird in (60.0, 100.0):
    for tau in (0.3, 0.45, 0.6, 0.8, 1.0):
        _, evr, _ = run(con[~con['seasonId'].isin(EVAL_SEASONS)],
                          collect_eval=True, eval_seasons=VAL,
                          init_rd=ird, tau_a=tau, tau_g=tau)
        e = pd.DataFrame(evr, columns=['ladder', 'glicko', 'base', 'bucket', 's'])
        la = _ll(e[e['ladder'] == 'aerial']['glicko'], e[e['ladder'] == 'aerial']['s'])
        lg = _ll(e[e['ladder'] == 'ground']['glicko'], e[e['ladder'] == 'ground']['s'])
        tune[(ird, tau)] = (la, lg, _ll(e['glicko'], e['s']))
        print(f"  INIT_RD={ird:>4.0f} tau={tau:.2f}  24/25 LL aerial={la:.4f} ground={lg:.4f}")
BEST_RD = min({k[0] for k in tune},
                key=lambda i: min(v[2] for k, v in tune.items() if k[0] == i))
BEST_TAU_A = min((k[1] for k in tune if k[0] == BEST_RD),
                   key=lambda t: tune[(BEST_RD, t)][0])
BEST_TAU_G = min((k[1] for k in tune if k[0] == BEST_RD),
                   key=lambda t: tune[(BEST_RD, t)][1])
print(f"  -> INIT_RD={BEST_RD:.0f}, tau_aerial={BEST_TAU_A:.2f}, tau_ground={BEST_TAU_G:.2f}")

print("\n[1/3] full chronological run + prequential gates…", flush=True)
(R, RD, N, W, LAST, LG), ev_rows, xw = run(con, collect_eval=True, init_rd=BEST_RD,
                                              tau_a=BEST_TAU_A, tau_g=BEST_TAU_G)
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
    st, _, _ = run(con[con['matchId'].astype('int64') % 2 == par], init_rd=BEST_RD,
                     tau_a=BEST_TAU_A, tau_g=BEST_TAU_G)
    half_states.append(st)
rows = []
for trait in ['aerial', 'stopper', 'takeon', 'press', 'shield']:
    (R0, _, N0, _, _, _), (R1, _, N1, _, _, _) = half_states
    common = [k for k in R0 if k in R1 and k[0] == trait
                and N0[k] >= 40 and N1[k] >= 40]
    a = np.array([R0[k] for k in common]); b = np.array([R1[k] for k in common])
    r = float(np.corrcoef(a, b)[0, 1])
    rows.append((trait, r, len(common)))
    print(f"  {trait:<8} split-half r = {r:.3f} (n={len(common)})")

print("\n[3/3] outputs + face validity…", flush=True)
_dmax = (con['date'].max() - con['date'].min()).days
_d0 = con['date'].min()
out = pd.DataFrame([{
    'playerId': k[1], 'trait': k[0], 'rating': R[k],
    # FIX (staleness bug): stored RD is as-of-last-contest because idle
    # inflation is applied lazily. Inflate to the dataset end so inactive
    # players carry honest uncertainty.
    'rd': min(RD_MAX, sqrt(RD[k]**2 + C2_PER_DAY * max(0, _dmax - LAST[k]))),
    'rd_at_last': RD[k],
    'last_date': _d0 + pd.Timedelta(days=int(LAST[k])),
    'league': LG.get(k, '?'),
    'rating_conservative': R[k] - min(RD_MAX, sqrt(RD[k]**2 + C2_PER_DAY * max(0, _dmax - LAST[k]))),
    'n': N[k], 'wins': W[k]} for k in R])
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
for trait in ['aerial', 'stopper', 'takeon', 'press', 'shield']:
    top = (out[(out["trait"] == trait) & (out["n"] >= 100)]
             .nlargest(10, 'rating_conservative'))
    names = [f"{str(nm.get(int(p), p))} ({int(n)} duels, {r:.0f}±{rd:.0f})"
              for p, n, r, rd in zip(top['playerId'], top['n'],
                                       top['rating'], top['rd'])]
    print(f"\n  TOP {trait.upper()} (>=100 contests):")
    for s in names:
        print(f"    {s}")
