"""Strength-metric horse race: which current-season signal at matchweek m
best predicts REST-OF-SEASON points? Candidates: ppg, GD/game, xGD (incl
pens), NPxGD, OBV-diff/game. Also: does adding ppg on top of the best
metric add anything (bivariate OLS R^2)?"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

DASH = "/Users/lucaskimball/Desktop/ACP_Official/Match_Reports_API/Dashboard"
PAIRS = [(188222, 189147), (189147, 190090), (190090, 191782)]
SIDS = sorted({s for p in PAIRS for s in p})
MIN_MATCHES = 16

ms = pd.read_parquet(f"{DASH}/matches_summary.parquet")
ms = ms[ms['status'] == 'Played'].copy()
sc = ms['score'].astype(str).str.extract(r'(\d+)\s*-\s*(\d+)')
ms['hg'] = pd.to_numeric(sc[0], errors='coerce')
ms['ag'] = pd.to_numeric(sc[1], errors='coerce')
ms = ms.dropna(subset=['hg', 'ag'])

# shots incl. penalties, flagged
ev = pd.read_parquet(f"{DASH}/raw_events.parquet",
                     columns=['matchId', 'seasonId', 'team.id', 'team.name', 'type.primary', 'shot.xg'],
                     filters=[('seasonId', 'in', SIDS),
                              ('type.primary', 'in', ['shot', 'penalty'])])
ev = ev.dropna(subset=['shot.xg', 'team.name'])
ev['is_pen'] = ev['type.primary'] == 'penalty'
xg_all = ev.groupby(['matchId', 'team.name'])['shot.xg'].sum().to_dict()
xg_np = ev[~ev['is_pen']].groupby(['matchId', 'team.name'])['shot.xg'].sum().to_dict()

# per-match team OBV with names (teamId -> name via events)
obv = pd.read_parquet(f"{DASH}/obv_match_minute.parquet")
obv_m = obv.groupby(['matchId', 'teamId'])['obv'].sum().reset_index()
idname = (ev.dropna(subset=['team.id'])
          .drop_duplicates(['matchId', 'team.id'])[['matchId', 'team.id', 'team.name']])
idname['team.id'] = idname['team.id'].astype('int64')
obv_m = obv_m.merge(idname, left_on=['matchId', 'teamId'],
                    right_on=['matchId', 'team.id'], how='inner')
obv_by = obv_m.set_index(['matchId', 'team.name'])['obv'].to_dict()


def first_stage(season):
    f = ms[ms['seasonId'] == season]
    rid = f.groupby('roundId').size().idxmax()
    return f[f['roundId'] == rid].sort_values('dateutc')


def team_rows(f):
    rows = {}
    for _, r in f.iterrows():
        mid = r['matchId']
        for team, opp, gfor, gag in [(r['homeTeamName'], r['awayTeamName'], r['hg'], r['ag']),
                                     (r['awayTeamName'], r['homeTeamName'], r['ag'], r['hg'])]:
            pts = 3 if gfor > gag else (1 if gfor == gag else 0)
            def diff(dic):
                a, b = dic.get((mid, team)), dic.get((mid, opp))
                return (a - b) if (a is not None and b is not None) else None
            rows.setdefault(team, []).append({
                'pts': pts, 'gd': gfor - gag,
                'xgd': diff(xg_all), 'npxgd': diff(xg_np), 'obvd': diff(obv_by),
            })
    return rows


CANDS = ['ppg', 'gd', 'xgd', 'npxgd', 'obvd']
samples = []
for prior_s, target_s in PAIRS:
    for team, lst in team_rows(first_stage(target_s)).items():
        if len(lst) < MIN_MATCHES:
            continue
        n = len(lst)
        for m in range(2, n - 3):
            cur, fut = lst[:m], lst[m:]
            row = {'m': m, 'F_ppg': np.mean([x['pts'] for x in fut])}
            row['ppg'] = np.mean([x['pts'] for x in cur])
            row['gd'] = np.mean([x['gd'] for x in cur])
            for k in ('xgd', 'npxgd', 'obvd'):
                vals = [x[k] for x in cur if x[k] is not None]
                row[k] = np.mean(vals) if len(vals) == m else None
            samples.append(row)

df = pd.DataFrame(samples)
print(f"samples: {len(df)}  | with OBV: {df['obvd'].notna().sum()}  with npxg: {df['npxgd'].notna().sum()}")

print(f"\ncorr(current metric at m, future ppg) — pooled, complete rows only")
print(f"{'m':>2} {'n':>3} | " + " ".join(f"{c:>6}" for c in CANDS))
for m in sorted(df['m'].unique()):
    if m > 12:
        break
    d = df[df['m'] == m].dropna(subset=CANDS)
    if len(d) < 20:
        continue
    cs = [np.corrcoef(d[c], d['F_ppg'])[0, 1] for c in CANDS]
    print(f"{m:>2} {len(d):>3} | " + " ".join(f"{c:>+6.3f}" for c in cs))

# early window pooled (m=2..6): rank + incremental value of ppg over best
d = df[(df['m'] >= 2) & (df['m'] <= 6)].dropna(subset=CANDS)
print(f"\npooled m=2-6 (n={len(d)}):")
for c in CANDS:
    r = np.corrcoef(d[c], d['F_ppg'])[0, 1]
    print(f"  corr({c:>6}) = {r:+.3f}   R2={r*r:.3f}")

def ols_r2(X, y):
    X = np.column_stack([X, np.ones(len(y))])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    ss_res = ((y - pred) ** 2).sum(); ss_tot = ((y - y.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot

y = d['F_ppg'].values
for base in ('npxgd', 'xgd', 'obvd'):
    r2b = ols_r2(d[[base]].values, y)
    r2bp = ols_r2(d[[base, 'ppg']].values, y)
    r2bo = ols_r2(d[[base, 'obvd']].values, y) if base != 'obvd' else None
    extra = f"  +obvd -> {r2bo:.3f}" if r2bo is not None else ""
    print(f"  R2({base})={r2b:.3f}   +ppg -> {r2bp:.3f}{extra}")
r2c = ols_r2(d[['npxgd', 'obvd', 'ppg']].values, y)
print(f"  R2(npxgd+obvd+ppg) = {r2c:.3f}")

# prior-season versions: which metric's PRIOR predicts next-season future ppg best?
print("\nprior-season metric -> next season future ppg (m=3 snapshots):")
prior_rates = {}
for prior_s, target_s in PAIRS:
    pr = {}
    for team, lst in team_rows(first_stage(prior_s)).items():
        if len(lst) < MIN_MATCHES:
            continue
        rec = {'ppg': np.mean([x['pts'] for x in lst])}
        for k in ('xgd', 'npxgd', 'obvd'):
            vals = [x[k] for x in lst if x[k] is not None]
            rec[k] = np.mean(vals) if len(vals) >= MIN_MATCHES - 4 else None
        pr[team] = rec
    prior_rates[target_s] = pr

rows = []
for prior_s, target_s in PAIRS:
    for team, lst in team_rows(first_stage(target_s)).items():
        if len(lst) < MIN_MATCHES or team not in prior_rates[target_s]:
            continue
        fut = lst[3:]
        rec = dict(prior_rates[target_s][team])
        rec['F_ppg'] = np.mean([x['pts'] for x in fut])
        rows.append(rec)
pdf = pd.DataFrame(rows).dropna()
print(f"  n={len(pdf)} returning teams")
for c in ('ppg', 'xgd', 'npxgd', 'obvd'):
    r = np.corrcoef(pdf[c], pdf['F_ppg'])[0, 1]
    print(f"  corr(prior {c:>6}) = {r:+.3f}")
