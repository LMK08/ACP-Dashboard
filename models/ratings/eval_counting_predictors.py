#!/usr/bin/env python3
"""Do traditional/counting metrics add PREDICTIVE power to the
projection beyond current rating + career?

Battery (train pairs 2021-2023 only; 2024 test untouched until the
final augmented-ridge gate):
  1. ~17 counting candidates per-90, role x league x season percentiles
  2. YoY stability (same-role consecutive pairs)
  3. PARTIAL predictive correlation with next-season rating,
     controlling for current rating + career_asof (both residualized)
  4. survivors -> augmented ridge -> the SAME out-of-time gate as
     build_projection (must beat career carry-forward on Spearman)

Run from the Dashboard dir: python models/ratings/eval_counting_predictors.py
"""
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

_HERE = Path(__file__).resolve().parent
_DASH = _HERE.parent.parent
_GPA_DATA = _DASH.parent.parent / 'GPA Model Project v2' / 'parquet_data'
SEASON_YR = {188221: 2021, 188222: 2022, 189147: 2023, 190090: 2024,
              191782: 2025, 190230: 2023, 191779: 2025}

TAGS = ['progressive_pass', 'pass_to_final_third', 'pass_to_penalty_area',
         'cross', 'through_pass', 'deep_completion', 'shot_assist', 'carry',
         'progressive_run', 'dribble', 'touch_in_box', 'loss', 'recovery',
         'counterpressing_recovery', 'foul_suffered', 'linkup_play',
         'aerial_duel']

print("[1/4] counting metrics from events…", flush=True)
ev = pd.concat([pd.read_parquet(_GPA_DATA / f'{f}.parquet',
    columns=['seasonId', 'player.id', 'type.primary', 'type.secondary'])
    for f in ['liga3_portugal_events', 'campeonato_portugal_events']],
    ignore_index=True).dropna(subset=['player.id'])
ev['player.id'] = ev['player.id'].astype(int)
sec = ev['type.secondary'].apply(
    lambda x: set(x) if isinstance(x, (list, np.ndarray)) else set())
agg = {}
for t in TAGS:
    ev[f'c_{t}'] = sec.apply(lambda s, t=t: t in s).astype(int)
ev['c_shots'] = (ev['type.primary'] == 'shot').astype(int)
ev['c_passes'] = (ev['type.primary'] == 'pass').astype(int)
ev['c_touches'] = 1
CANDS = [f'c_{t}' for t in TAGS] + ['c_shots', 'c_passes', 'c_touches']
cnt = ev.groupby(['player.id', 'seasonId'])[CANDS].sum().reset_index()
cnt.columns = ['playerId', 'seasonId'] + CANDS

print("[2/4] panel…", flush=True)
o = pd.read_parquet(_HERE / 'acp_rating_per_player_season.parquet')
o['yr'] = o['seasonId'].map(SEASON_YR)
o = o.merge(cnt, on=['playerId', 'seasonId'], how='left')
for c in CANDS:
    o[c + '90'] = o[c].fillna(0) / o['mins_played'] * 90
    o['p' + c] = o.groupby(['role', 'league', 'seasonId'])[c + '90'].rank(pct=True)
# leak-free career
o = o.sort_values(['playerId', 'yr'])
car = []
for pid, g in o.groupby('playerId', sort=False):
    num = den = 0.0
    for _, r in g.iterrows():
        num = num * 0.5 + r['mins_played'] * r['acp_rating']
        den = den * 0.5 + r['mins_played']
        car.append(num / den)
o['career_asof'] = car
pairs = []
for pid, g in o.sort_values('yr').groupby('playerId'):
    rr = g.to_dict('records')
    for a, b in zip(rr, rr[1:]):
        if b['yr'] - a['yr'] == 1 and a['role'] == b['role']:
            pairs.append({**{('p' + c): a['p' + c] for c in CANDS},
                            'rating': a['acp_rating'], 'career': a['career_asof'],
                            'role': a['role'], 'start_yr': a['yr'],
                            'nxt': b['acp_rating']})
P = pd.DataFrame(pairs).dropna()
tr = P[P['start_yr'] <= 2023]
te = P[P['start_yr'] == 2024]
print(f"  pairs: train {len(tr)}, test {len(te)} (held out)")

print("[3/4] battery (train only)…", flush=True)
# residualize next rating on (rating, career) once
Xb = np.column_stack([tr['rating'], tr['career'], np.ones(len(tr))])
bb, *_ = np.linalg.lstsq(Xb, tr['nxt'].values, rcond=None)
nxt_res = tr['nxt'].values - Xb @ bb


def yoy(col):
    a = []
    for pid_g in [P]:
        pass
    # YoY across all train pairs of the percentile itself
    return spearmanr(tr[col], tr.groupby(level=0)[col].shift(0))[0]

print(f"{'metric':<28}{'YoY':>6}{'partial-r':>10}")
results = []
for c in CANDS:
    col = 'p' + c
    # YoY: same-player consecutive percentiles — reuse pairs frame by
    # matching current with next via a second panel pass
    bc, *_ = np.linalg.lstsq(Xb, tr[col].values, rcond=None)
    c_res = tr[col].values - Xb @ bc
    pr = spearmanr(c_res, nxt_res)[0]
    results.append((c, pr))
# YoY computed separately (needs next-season value of the same metric)
pairs2 = []
for pid, g in o.sort_values('yr').groupby('playerId'):
    rr = g.to_dict('records')
    for a, b in zip(rr, rr[1:]):
        if b['yr'] - a['yr'] == 1 and a['role'] == b['role'] and a['yr'] <= 2023:
            pairs2.append({c: (a['p' + c], b['p' + c]) for c in CANDS})
yoys = {}
for c in CANDS:
    arr = pd.DataFrame([(r[c][0], r[c][1]) for r in pairs2]).dropna()
    yoys[c] = spearmanr(arr[0], arr[1])[0]
for c, pr in sorted(results, key=lambda x: -abs(x[1])):
    print(f"{c:<28}{yoys[c]:>6.2f}{pr:>10.3f}")

print("[4/4] augmented ridge -> OOT gate…", flush=True)
keep = [('p' + c) for c, pr in results if abs(pr) >= 0.08]
print(f"  survivors (|partial| >= .08): {[k[3:] for k in keep] or 'NONE'}")
base_sp = spearmanr(te['career'], te['nxt'])[0]
if keep:
    FE = ['rating', 'career'] + keep
    mu, sd = tr[FE].mean(), tr[FE].std()
    m = Ridge(alpha=30.0).fit((tr[FE] - mu) / sd, tr['nxt'])
    pred = m.predict((te[FE] - mu) / sd)
    sp = spearmanr(pred, te['nxt'])[0]
    mae = np.abs(pred - te['nxt']).mean()
    mae_c = np.abs(te['career'] - te['nxt']).mean()
    print(f"  GATE: augmented ridge Spearman {sp:.3f} vs career carry {base_sp:.3f}"
           f"  -> {'PASS' if sp > base_sp else 'FAIL'}")
    print(f"        MAE {mae:.2f} vs career {mae_c:.2f}")
    print("  coefs:", {f: round(float(c), 2)
                         for f, c in zip(FE, m.coef_) if abs(c) > 0.1})
else:
    print(f"  nothing to add; career carry Spearman {base_sp:.3f} stands")
