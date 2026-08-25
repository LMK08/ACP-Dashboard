"""Empirical blend/decay backtest: prior-season vs current-season strength.

For each Liga 3 season pair (prior -> target), same-tier returning teams only:
  - S_prior: full prior-season rate (ppg / xGD per game)
  - C_m: current-season rate through each team's first m matches
  - F_m: FUTURE rate over the team's remaining first-stage matches
For each matchweek m and blend weight w: pred = w*C_m + (1-w)*S_prior.
Pooled across teams and season pairs, find w*(m) minimizing MSE vs F_m,
then fit exp-decay lambda (current weight = 1 - exp(-lambda*m)) and the
shrinkage-equivalent k (w = m/(m+k)).
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

DASH = "/Users/lucaskimball/Desktop/ACP_Official/Match_Reports_API/Dashboard"
PAIRS = [(188222, 189147), (189147, 190090), (190090, 191782)]  # Liga 3 22/23->23/24->24/25->25/26
MIN_MATCHES = 16

ms = pd.read_parquet(f"{DASH}/matches_summary.parquet")
ms = ms[ms['status'] == 'Played'].copy()
sc = ms['score'].astype(str).str.extract(r'(\d+)\s*-\s*(\d+)')
ms['hg'] = pd.to_numeric(sc[0], errors='coerce')
ms['ag'] = pd.to_numeric(sc[1], errors='coerce')
ms = ms.dropna(subset=['hg', 'ag'])

sids = sorted({s for p in PAIRS for s in p})
ev = pd.read_parquet(f"{DASH}/raw_events.parquet",
                     columns=['matchId', 'seasonId', 'team.name', 'type.primary', 'shot.xg'],
                     filters=[('seasonId', 'in', sids), ('type.primary', '==', 'shot')])
xg_by = (ev.dropna(subset=['shot.xg', 'team.name'])
         .groupby(['matchId', 'team.name'])['shot.xg'].sum().to_dict())


def first_stage(season):
    f = ms[ms['seasonId'] == season]
    rid = f.groupby('roundId').size().idxmax()
    return f[f['roundId'] == rid].sort_values('dateutc')


def team_match_rows(f):
    """Per team: chronological list of (pts, gd, xgd) per match."""
    rows = {}
    for _, r in f.iterrows():
        mid = r['matchId']
        for team, opp, gfor, gag in [(r['homeTeamName'], r['awayTeamName'], r['hg'], r['ag']),
                                     (r['awayTeamName'], r['homeTeamName'], r['ag'], r['hg'])]:
            pts = 3 if gfor > gag else (1 if gfor == gag else 0)
            xf = xg_by.get((mid, team))
            xa = xg_by.get((mid, opp))
            xgd = (xf - xa) if (xf is not None and xa is not None) else None
            rows.setdefault(team, []).append((pts, gfor - gag, xgd))
    return rows


def season_rates(rows):
    out = {}
    for t, lst in rows.items():
        if len(lst) < MIN_MATCHES:
            continue
        pts = [x[0] for x in lst]
        xgd = [x[2] for x in lst if x[2] is not None]
        out[t] = {'ppg': np.mean(pts), 'xgd': np.mean(xgd) if len(xgd) >= MIN_MATCHES - 4 else None}
    return out


samples = []  # (m, S_prior_ppg, S_prior_xgd, C_ppg, C_xgd, F_ppg)
for prior_s, target_s in PAIRS:
    prior_rates = season_rates(team_match_rows(first_stage(prior_s)))
    cur_rows = team_match_rows(first_stage(target_s))
    for team, lst in cur_rows.items():
        if len(lst) < MIN_MATCHES or team not in prior_rates:
            continue
        pr = prior_rates[team]
        n = len(lst)
        for m in range(1, n - 3):  # need >=4 future matches
            cur = lst[:m]
            fut = lst[m:]
            c_ppg = np.mean([x[0] for x in cur])
            f_ppg = np.mean([x[0] for x in fut])
            cx = [x[2] for x in cur if x[2] is not None]
            c_xgd = np.mean(cx) if len(cx) == m else None
            samples.append((m, pr['ppg'], pr['xgd'], c_ppg, c_xgd, f_ppg))

df = pd.DataFrame(samples, columns=['m', 'S_ppg', 'S_xgd', 'C_ppg', 'C_xgd', 'F_ppg'])
n_teams = df.groupby('m').size()
print(f"pooled team-matchweek samples: {len(df)} "
      f"({df['m'].nunique()} matchweeks, ~{int(n_teams.mean())} teams each)")

# xGD -> future-ppg needs a scale: fit linear map on full data once per predictor
def fit_scale(x, y):
    A = np.vstack([x, np.ones(len(x))]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef  # slope, intercept


ws = np.round(np.arange(0, 1.0001, 0.05), 2)
print(f"\n{'m':>2} {'n':>3} | {'w* ppg':>7} {'MSE*':>6} {'MSE cur':>7} {'MSE prior':>9} | {'w* xgd':>7}")
wstar_ppg, wstar_xgd = {}, {}
for m in sorted(df['m'].unique()):
    if m > 14:
        break
    d = df[df['m'] == m]
    # ppg blend
    errs = {w: ((w * d['C_ppg'] + (1 - w) * d['S_ppg'] - d['F_ppg']) ** 2).mean() for w in ws}
    w_p = min(errs, key=errs.get)
    wstar_ppg[m] = w_p
    # xgd blend (rows with xg both sides)
    dx = d.dropna(subset=['C_xgd', 'S_xgd'])
    w_x = None
    if len(dx) >= 15:
        sl, ic = fit_scale(np.concatenate([dx['C_xgd'], dx['S_xgd']]),
                           np.concatenate([dx['F_ppg'], dx['F_ppg']]))
        errx = {w: (((w * dx['C_xgd'] + (1 - w) * dx['S_xgd']) * sl + ic - dx['F_ppg']) ** 2).mean()
                for w in ws}
        w_x = min(errx, key=errx.get)
        wstar_xgd[m] = w_x
    print(f"{m:>2} {len(d):>3} | {w_p:>7.2f} {errs[w_p]:>6.3f} {errs[1.0]:>7.3f} {errs[0.0]:>9.3f} | "
          f"{'-' if w_x is None else f'{w_x:.2f}':>7}")

# fit lambda: current weight = 1 - exp(-lam m); and k: w = m/(m+k)
def fit_curve(wd):
    ms_ = np.array(list(wd.keys()), dtype=float)
    w_ = np.array(list(wd.values()), dtype=float)
    lams = np.arange(0.01, 1.01, 0.01)
    lam = min(lams, key=lambda l: ((1 - np.exp(-l * ms_) - w_) ** 2).sum())
    ks = np.arange(0.5, 30.1, 0.5)
    k = min(ks, key=lambda kk: ((ms_ / (ms_ + kk) - w_) ** 2).sum())
    return lam, k

lam_p, k_p = fit_curve(wstar_ppg)
print(f"\nppg blend:  fitted lambda={lam_p:.2f} (current 0.30), shrinkage k≈{k_p:.1f} games")
if wstar_xgd:
    lam_x, k_x = fit_curve(wstar_xgd)
    print(f"xGD blend:  fitted lambda={lam_x:.2f}, shrinkage k≈{k_x:.1f} games")

# which single metric predicts future ppg better at m=3 and m=5 (today's regime)?
for m in (3, 5):
    d = df[(df['m'] == m)].dropna(subset=['C_xgd', 'S_xgd'])
    if len(d) < 15:
        continue
    for name, col in [('current ppg', 'C_ppg'), ('prior ppg', 'S_ppg'),
                      ('current xGD', 'C_xgd'), ('prior xGD', 'S_xgd')]:
        r = np.corrcoef(d[col], d['F_ppg'])[0, 1]
        print(f"m={m}: corr({name:>12}, future ppg) = {r:+.3f}  (n={len(d)})")
