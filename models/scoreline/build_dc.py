"""Fit the Dixon-Coles scoreline model and back-test it.

    python models/scoreline/build_dc.py            # from the Dashboard dir

1. Tunes (xi, l2, mix) — time decay, shrinkage on team parameters, and the
   goals-vs-xG blend the Poisson rates are fitted on — on a walk-forward
   backtest (monthly refits on everything before each window) over every
   season after the first. xG comes from raw_events.parquet when present
   (HF-only file; the build falls back to goals-only without it).
2. Refits on everything up to today with the chosen settings and writes
   models/scoreline/dc_params.json (what the app loads).
3. Writes models/scoreline/dc_backtest.json: metrics per season and league,
   the base-rate comparison, the reliability table, and the like-for-like
   number against the simple predictor (its reported 2025/26 holdout log
   loss is 1.078 — see train_simple_predictor.py).
4. Records the promoted-side calibration — expected vs actual points of
   sides in their first Liga 3 season after promotion from the Campeonato —
   for the chosen fit and for the same settings shrunk toward zero. That
   comparison is why the penalty shrinks toward each league's own mean
   (PRIOR below; see the dixon_coles.py module docstring).

Pure Python; a full run is well under a minute.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DASH = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, DASH)

from models.scoreline.dixon_coles import (DixonColes, attach_xg, base_rate_metrics, matches_from_summary,  # noqa: E402
                                          metrics, outcome_index, promoted_side_bias, reliability, walk_forward)

XI_GRID = (0.0015, 0.0025, 0.004)
L2_GRID = (0.5, 1.0, 2.0)
MIX_GRID = (0.0, 0.1, 0.25, 1.0)   # 1.0 = goals only (the classic model), kept for the record
# What the L2 penalty shrinks each team toward. 'league_mean' (the team's own
# league's mean, fitted freely) is a design choice, not a grid axis: the
# zero-centred fit had a calibration fault the aggregate log loss barely
# sees — promoted sides over-rated by a third of a point per match early
# in their first Liga 3 season (2024-26 walk-forward) — so the choice
# rests on the promoted-side calibration recorded in dc_backtest.json.
PRIOR = 'league_mean'
SIMPLE_PREDICTOR_HOLDOUT = {'season_id': 191782, 'log_loss': 1.078, 'brier': 0.653, 'accuracy': 0.413}


def main():
    t0 = time.time()
    ms = pd.read_parquet(os.path.join(DASH, 'matches_summary.parquet'))
    matches = matches_from_summary(ms)
    events_path = os.path.join(DASH, 'raw_events.parquet')
    has_xg = os.path.exists(events_path)
    if has_xg:
        matches = attach_xg(matches, events_path)
    print(f'{len(matches)} played matches, {matches["home"].nunique()} home teams, '
          f'{matches["date"].min().date()} -> {matches["date"].max().date()}, '
          f'xG coverage {matches["xg_h"].notna().mean():.0%}' if has_xg else 'no raw_events.parquet: goals only')

    # Backtest window: everything from the start of the second season onward
    seasons = matches.groupby('season_id')['date'].min().sort_values()
    start = seasons.iloc[1] if len(seasons) > 1 else matches['date'].iloc[300]
    results = {}
    for xi in XI_GRID:
        for l2 in L2_GRID:
            for mix in (MIX_GRID if has_xg else (1.0,)):
                pred = walk_forward(matches, xi=xi, start=start, l2=l2, mix=mix, prior=PRIOR)
                results[(xi, l2, mix)] = (metrics(pred), pred)
                r = results[(xi, l2, mix)][0]
                print(f'  xi={xi:<7} l2={l2:<4} mix={mix:<5} n={r["n"]:5d} log_loss={r["log_loss"]:.4f} '
                      f'brier={r["brier"]:.4f} acc={r["accuracy"]:.3f}')
    best = min(results, key=lambda k: results[k][0]['log_loss'])
    best_xi, best_l2, best_mix = best
    pred = results[best][1]
    print(f'chosen xi={best_xi} l2={best_l2} mix={best_mix}')

    # Base-rate comparison uses the pre-backtest matches' W/D/L frequencies
    train = matches[matches['date'] < start]
    o = outcome_index(train['hg'].values, train['ag'].values)
    base_rates = [float((o == k).mean()) for k in range(3)]

    per_season = {}
    for sid, g in pred.groupby('season_id'):
        per_season[str(int(sid))] = {**metrics(g), 'base_rate': base_rate_metrics(g, base_rates),
                                     'leagues': {str(int(l)): metrics(gg) for l, gg in g.groupby('league')}}
    holdout = pred[(pred['season_id'] == SIMPLE_PREDICTOR_HOLDOUT['season_id'])
                   & (pred['league'] == 43324)]
    like_for_like = {'season_id': SIMPLE_PREDICTOR_HOLDOUT['season_id'], 'league': 43324,
                     'dixon_coles': metrics(holdout),
                     'simple_predictor_reported': SIMPLE_PREDICTOR_HOLDOUT,
                     'base_rate': base_rate_metrics(holdout, base_rates) if not holdout.empty else {}}

    final = DixonColes.fit(matches, xi=best_xi, l2=best_l2, mix=best_mix, prior=PRIOR)
    final.save(os.path.join(HERE, 'dc_params.json'))

    # Promoted-side calibration for the chosen fit and for the same settings
    # shrunk toward zero (the comparison the Match Predictor's backtest shows).
    pred_zero = walk_forward(matches, xi=best_xi, start=start, l2=best_l2, mix=best_mix, prior='zero')
    promoted = {
        'prior': PRIOR,
        'chosen': promoted_side_bias(pred, matches),
        'zero_prior': promoted_side_bias(pred_zero, matches),
        'liga3_log_loss': {'chosen': metrics(pred[pred['league'] == 43324]).get('log_loss'),
                           'zero_prior': metrics(pred_zero[pred_zero['league'] == 43324]).get('log_loss')},
    }
    backtest = {
        'built': pd.Timestamp.today().strftime('%Y-%m-%d'),
        'grid': [{'xi': k[0], 'l2': k[1], 'mix': k[2], **v[0]} for k, v in results.items()],
        'xi': best_xi, 'l2': best_l2, 'mix': best_mix, 'prior': PRIOR, 'xg_available': has_xg,
        'overall': metrics(pred),
        'base_rate_overall': base_rate_metrics(pred, base_rates),
        'base_rates': base_rates,
        'per_season': per_season,
        'like_for_like_2025_26_liga3': like_for_like,
        'reliability': reliability(pred).to_dict(orient='records'),
        'n_predicted': int(len(pred)),
        'promoted_side_calibration': promoted,
        'fit': {'asof': final.asof, 'n_matches': final.n_matches, 'home_adv': final.home_adv,
                'rho': final.rho, 'prior': final.prior,
                'leagues': {str(l): float(b) for l, b in zip(final.leagues, final.base)},
                'league_means': {str(l): {'att': m[0], 'def': m[1]} for l, m in final.league_means.items()},
                'tier_gap_cdp_vs_liga3': final.tier_gap(43324, 702)},
    }
    with open(os.path.join(HERE, 'dc_backtest.json'), 'w', encoding='utf-8') as fh:
        json.dump(backtest, fh, indent=1, ensure_ascii=False)
    print(f'final fit: {final.n_matches} matches, home_adv={final.home_adv:+.3f}, rho={final.rho:+.3f}, '
          f'base={dict(zip(final.leagues, np.round(final.base, 3)))}, prior={final.prior}')
    gap = final.tier_gap(43324, 702)
    if gap:
        print(f'tier gap: an average Campeonato side scores x{gap[0]:.2f} and concedes x{gap[1]:.2f} vs an average Liga 3 side')
    for label, rows in (('chosen', promoted['chosen']), ('zero prior', promoted['zero_prior'])):
        if rows:
            r = rows[0]
            print(f'promoted sides, matches {r["matches"]} ({label}): expected {r["exp_pts"]:.2f} pts/match, actual {r["act_pts"]:.2f}')
    print(f'like-for-like 2025/26 Liga 3: DC log_loss={like_for_like["dixon_coles"].get("log_loss", float("nan")):.4f} '
          f'vs simple 1.078 vs base {like_for_like["base_rate"].get("log_loss", float("nan")):.4f}')
    print(f'wrote dc_params.json + dc_backtest.json in {time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
