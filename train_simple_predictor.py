"""Train the interpretable match predictor (simple_strength_v1).

Team strength = 0.7*NPxGD/game + 0.3*GD/game, where each rate is the
production blend of current-season form with tier-aware priors (imported
from simulate_season: calculate_prediction_features with the calibrated
Bayes shrinkage k's). Single feature (home strength - away strength) into
a multinomial logistic regression; the class intercepts carry home
advantage.

Evidence: on the held-out 2025/26 season this beats the 35-feature
RandomForest on every metric (log-loss 1.078 vs 1.113, Brier 0.653 vs
0.675, accuracy 41.3% vs 36.3%) — small data favors the simple model.
See calibrate_strength_metrics.py / calibrate_blend_decay.py.

Writes match_predictor_model.pkl with feature_mode='simple_strength_v1';
update_model.py keeps refreshing team_stats nightly as before. Override
the output path with MODEL_OUT for experiments.
"""
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

import simulate_season as ss

STRENGTH_MIX = 0.7          # NPxGD share (CV optimum was 0.8; 0.7 within noise)
HOLDOUT_SEASON = 191782     # reported metrics only; final fit uses ALL seasons
MODEL_OUT = os.environ.get('MODEL_OUT', 'match_predictor_model.pkl')
MIN_PRIOR_MATCHES = 16

print("Loading data...")
ms = pd.read_parquet('historical_matches.parquet')
lab = ms['label'].astype(str).str.extract(r'^(.*?)\s*-\s*(.*?),\s*(\d+)\s*-\s*(\d+)')
ms['home'], ms['away'] = lab[0].str.strip(), lab[1].str.strip()
ms['hg'] = pd.to_numeric(lab[2], errors='coerce')
ms['ag'] = pd.to_numeric(lab[3], errors='coerce')
ms = ms.dropna(subset=['home', 'away', 'hg', 'ag']).sort_values('dateutc')
SEASONS = sorted(ms['seasonId'].unique())
print(f"{len(ms)} matches, seasons {SEASONS}")

ev = pd.read_parquet('raw_events.parquet',
                     columns=['matchId', 'seasonId', 'team.name', 'type.primary', 'shot.xg'],
                     filters=[('seasonId', 'in', [int(s) for s in SEASONS]),
                              ('type.primary', '==', 'shot')])
npxg = (ev.dropna(subset=['shot.xg', 'team.name'])
        .groupby(['matchId', 'team.name'])['shot.xg'].sum().to_dict())


def empty_stats():
    return {'matches': 0, 'points': 0, 'wins': 0, 'draws': 0, 'losses': 0,
            'goals_for': 0, 'goals_against': 0, 'xG_for': 0.0, 'xG_against': 0.0,
            'shots_for': 0, 'sot_for': 0, 'clean_sheets': 0,
            'home_matches': 0, 'home_wins': 0, 'home_goals': 0,
            'away_matches': 0, 'away_wins': 0, 'away_goals': 0,
            'last_5_results': [], 'last_5_xG': []}


def season_prior_rates(season):
    """Full-season per-game rates for the prior side of the blend."""
    f = ms[ms['seasonId'] == season]
    teams = set(f['home']) | set(f['away'])
    out = {}
    for t in teams:
        home = f[f['home'] == t]
        away = f[f['away'] == t]
        n = len(home) + len(away)
        if n < MIN_PRIOR_MATCHES:
            continue
        gf = home['hg'].sum() + away['ag'].sum()
        ga = home['ag'].sum() + away['hg'].sum()
        wins = (home['hg'] > home['ag']).sum() + (away['ag'] > away['hg']).sum()
        draws = (home['hg'] == home['ag']).sum() + (away['ag'] == away['hg']).sum()
        cs = (home['ag'] == 0).sum() + (away['hg'] == 0).sum()
        xf = xa = 0.0
        for _, r in pd.concat([home, away]).iterrows():
            mine = npxg.get((r['matchId'], t), 0.0)
            opp = r['away'] if r['home'] == t else r['home']
            theirs = npxg.get((r['matchId'], opp), 0.0)
            xf += mine; xa += theirs
        out[t] = {'per_game': {'ppg': (3 * wins + draws) / n, 'gpg': gf / n,
                               'gapg': ga / n, 'xgpg': xf / n, 'xgapg': xa / n,
                               'winrate': wins / n, 'csrate': cs / n},
                  'source': 'same_tier'}
    return out


league_avg = {'ppg': 1.35, 'gpg': 1.19, 'gapg': 1.19,
              'xgpg': 1.1, 'xgapg': 1.1, 'csrate': 0.3,
              'shot_conv': 0.1, 'sot_rate': 0.35}

X, y, season_of = [], [], []
for i, season in enumerate(SEASONS):
    priors = season_prior_rates(SEASONS[i - 1]) if i > 0 else {}
    cum = {}
    f = ms[ms['seasonId'] == season]
    for _, r in f.iterrows():
        h, a = r['home'], r['away']
        hs = cum.setdefault(h, empty_stats())
        as_ = cum.setdefault(a, empty_stats())
        hf = ss.calculate_prediction_features(hs, priors.get(h), league_avg, is_home=True)
        af = ss.calculate_prediction_features(as_, priors.get(a), league_avg, is_home=False)
        Sh = STRENGTH_MIX * (hf['xgpg'] - hf['xgapg']) + (1 - STRENGTH_MIX) * (hf['gpg'] - hf['gapg'])
        Sa = STRENGTH_MIX * (af['xgpg'] - af['xgapg']) + (1 - STRENGTH_MIX) * (af['gpg'] - af['gapg'])
        X.append([Sh - Sa])
        y.append(1 if r['hg'] > r['ag'] else (2 if r['ag'] > r['hg'] else 0))
        season_of.append(season)
        # accumulate AFTER featurizing
        mid = r['matchId']
        hx, ax_ = npxg.get((mid, h), 0.0), npxg.get((mid, a), 0.0)
        for st, gf, ga, xf, xa, is_home, won in (
                (hs, r['hg'], r['ag'], hx, ax_, True, r['hg'] > r['ag']),
                (as_, r['ag'], r['hg'], ax_, hx, False, r['ag'] > r['hg'])):
            st['matches'] += 1
            st['goals_for'] += gf; st['goals_against'] += ga
            st['xG_for'] += xf; st['xG_against'] += xa
            st['last_5_xG'] = (st['last_5_xG'] + [xf])[-5:]
            if ga == 0:
                st['clean_sheets'] += 1
            vk = 'home' if is_home else 'away'
            st[f'{vk}_matches'] += 1; st[f'{vk}_goals'] += gf
            if won:
                st['wins'] += 1; st['points'] += 3
                st[f'{vk}_wins'] += 1
                st['last_5_results'] = (st['last_5_results'] + [3])[-5:]
            elif gf == ga:
                st['draws'] += 1; st['points'] += 1
                st['last_5_results'] = (st['last_5_results'] + [1])[-5:]
            else:
                st['losses'] += 1
                st['last_5_results'] = (st['last_5_results'] + [0])[-5:]

X = np.array(X); y = np.array(y); season_of = np.array(season_of)

# held-out report
tr = season_of != HOLDOUT_SEASON
te = ~tr
scaler = StandardScaler().fit(X[tr])
lr = LogisticRegression(max_iter=1000, C=10.0).fit(scaler.transform(X[tr]), y[tr])
p = lr.predict_proba(scaler.transform(X[te]))[:, np.argsort(lr.classes_)]
ll = log_loss(y[te], p, labels=[0, 1, 2])
acc = (np.array([0, 1, 2])[p.argmax(1)] == y[te]).mean()
brier = float(np.mean(np.sum((p - np.eye(3)[y[te]]) ** 2, axis=1)))
print(f"\nHELD-OUT {HOLDOUT_SEASON} (n={te.sum()}): "
      f"log-loss {ll:.4f} | Brier {brier:.4f} | accuracy {acc:.1%}")
print("  (previous RF: log-loss 1.1125 | Brier 0.6751 | accuracy 36.3%)")

# final fit on everything
scaler = StandardScaler().fit(X)
lr = LogisticRegression(max_iter=1000, C=10.0).fit(scaler.transform(X), y)
print(f"\nFinal fit on all {len(X)} matches "
      f"(coef {lr.coef_.ravel().round(3).tolist()}, "
      f"intercepts {lr.intercept_.round(3).tolist()})")

# keep nightly-updated operational fields from the existing artifact
model_data = {}
if os.path.exists('match_predictor_model.pkl'):
    with open('match_predictor_model.pkl', 'rb') as fh:
        old = pickle.load(fh)
    for k in ('team_stats', 'team_ratings', 'prior_season_stats', 'league_avg_stats'):
        if k in old:
            model_data[k] = old[k]

model_data.update({
    'model': lr, 'scaler': scaler,
    'feature_mode': 'simple_strength_v1', 'strength_mix': STRENGTH_MIX,
    'holdout_log_loss': ll, 'holdout_brier': brier,
    'test_accuracy': float(acc), 'cv_accuracy': float(acc),
    'accuracy': float(acc),
})
model_data.setdefault('league_avg_stats', league_avg)
model_data.setdefault('prior_season_stats', {})

with open(MODEL_OUT, 'wb') as fh:
    pickle.dump(model_data, fh)
print(f"Saved {MODEL_OUT} (feature_mode=simple_strength_v1, mix={STRENGTH_MIX})")
