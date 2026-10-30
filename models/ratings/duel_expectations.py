"""Duel expectations as of each contest's kick-off (v7.1, 2026-10-29).

A contest is judged only against contests from matches that kicked off
BEFORE its own match: never its own outcome, its own match or anything
later (Lucas 2026-10-29: no data leakage, even where it buys no accuracy).
Until v7.1 the strengths and bucket rates were cumulative through the
contest's season-year, so an October duel was judged with expectations
built from that season's later contests and from its own result.

  strength     a player's shrunk win share over every earlier contest of
               his career (K0 = 40 pseudo-contests at 0.5)
  bucket rate  the defender win rate of kind x zone x phase x league over
               earlier contests, shrunk (K = 20) toward the all-bucket rate
               of earlier contests, itself shrunk (K = 20) toward 0.5
  centring     ground strengths are centred on the mean strength over the
               season-year's contests from earlier kick-offs (a year's
               first kick-off: every earlier contest)

expectations(ct, camp_ids, season_year) -> (aerial, ground), the contest
rows with the expected wins build_rating turns into wins over expectation:
  aerial  pA, pB    expected win of playerA / playerB
  ground  phat      expected defender (playerA) win vs THIS attacker
          pdef      expected defender win for an average attacker vs THIS
                    defender (the attacker side's expectation is 1 - pdef)
"""
import numpy as np
import pandas as pd

K0 = 40.0          # strength shrink: pseudo-contests at 0.5
K_BUCKET = 20.0    # bucket-rate shrink toward the earlier all-bucket rate


def _before(p, ts, w, k0=K0):
    """Shrunk win share of player p from contests that kicked off strictly
    before ts (one value per input row)."""
    d = pd.DataFrame({'p': np.asarray(p), 'ts': np.asarray(ts), 'w': np.asarray(w, dtype=float)})
    g = (d.groupby(['p', 'ts'])['w'].agg(['sum', 'size']).reset_index()
           .sort_values(['p', 'ts'], kind='stable'))
    cum = g.groupby('p')[['sum', 'size']].cumsum()
    g['W'] = cum['sum'] - g['sum']
    g['N'] = cum['size'] - g['size']
    s = pd.Series(((g['W'] + k0 / 2) / (g['N'] + k0)).to_numpy(),
                  index=pd.MultiIndex.from_frame(g[['p', 'ts']]))
    return s.reindex(pd.MultiIndex.from_frame(d[['p', 'ts']])).to_numpy()


def _bucket_rate(g):
    """Defender win rate of each contest's bucket from earlier kick-offs."""
    b = (g.groupby(['bucket', 'ts'])['scoreA'].agg(['sum', 'size']).reset_index()
           .sort_values(['bucket', 'ts'], kind='stable'))
    cum = b.groupby('bucket')[['sum', 'size']].cumsum()
    b['S'] = cum['sum'] - b['sum']
    b['N'] = cum['size'] - b['size']
    a = g.groupby('ts')['scoreA'].agg(['sum', 'size']).sort_index()
    gs, gn = a['sum'].cumsum() - a['sum'], a['size'].cumsum() - a['size']
    prior = (gs + K_BUCKET * 0.5) / (gn + K_BUCKET)   # itself shrunk: the data's first kick-offs
    b['prior'] = b['ts'].map(prior)
    b['mu_b'] = (b['S'] + K_BUCKET * b['prior']) / (b['N'] + K_BUCKET)
    return g.merge(b[['bucket', 'ts', 'mu_b']], on=['bucket', 'ts'], how='left')['mu_b'].to_numpy()


def _running_mean(g, col):
    """Mean of `col` over the season-year's contests from EARLIER kick-offs;
    a year's first kick-off takes the mean over every earlier contest (0.5
    before any)."""
    m = (g.groupby(['cyr', 'ts'])[col].agg(['sum', 'count']).reset_index()
           .sort_values(['cyr', 'ts'], kind='stable'))
    cum = m.groupby('cyr')[['sum', 'count']].cumsum()
    n = cum['count'] - m['count']
    m['mean'] = (cum['sum'] - m['sum']) / n.where(n > 0)
    t = g.groupby('ts')[col].agg(['sum', 'count']).sort_index()
    gn = t['count'].cumsum() - t['count']
    every = ((t['sum'].cumsum() - t['sum']) / gn.where(gn > 0)).fillna(0.5)
    m['mean'] = m['mean'].fillna(m['ts'].map(every))
    return g.merge(m[['cyr', 'ts', 'mean']], on=['cyr', 'ts'], how='left')['mean'].to_numpy()


def expectations(ct, camp_ids, season_year):
    """ct: contests (ladder, seasonId, date, playerA, playerB, scoreA,
    att_kind, zx, phase). Returns (aerial, ground) copies with expectations."""
    ct = ct.copy()
    ct['ts'] = pd.to_datetime(ct['date'])
    ct['league'] = np.where(ct['seasonId'].isin(set(camp_ids)), 'CAMP', 'L3')
    ct['cyr'] = ct['seasonId'].map(season_year)

    a = ct[ct['ladder'] == 'aerial'].reset_index(drop=True)
    both = pd.DataFrame({'p': np.concatenate([a['playerA'].to_numpy(), a['playerB'].to_numpy()]),
                         'ts': np.concatenate([a['ts'].to_numpy(), a['ts'].to_numpy()]),
                         'w': np.concatenate([a['scoreA'].to_numpy(), 1 - a['scoreA'].to_numpy()])})
    s = _before(both['p'], both['ts'], both['w'])
    a['_sA'], a['_sB'] = s[:len(a)], s[len(a):]
    a['pA'] = (1 - a['_sB']).clip(0.08, 0.92)
    a['pB'] = (1 - a['_sA']).clip(0.08, 0.92)

    g = ct[ct['ladder'] == 'ground'].reset_index(drop=True)
    g['bucket'] = (g['att_kind'].astype(str) + '|' + g['zx'].astype(str)
                   + '|' + g['phase'].astype(str) + '|' + g['league'])
    g['mu_b'] = _bucket_rate(g)
    # the attacker's strength counts his contests as the attacker, the
    # defender's his contests as the defender (as before v7.1)
    g['s_opp'] = _before(g['playerB'], g['ts'], 1 - g['scoreA'])
    g['s_def'] = _before(g['playerA'], g['ts'], g['scoreA'])
    g['phat'] = (g['mu_b'] - (g['s_opp'] - _running_mean(g, 's_opp'))).clip(0.05, 0.95)
    g['pdef'] = (g['mu_b'] + (g['s_def'] - _running_mean(g, 's_def'))).clip(0.05, 0.95)
    return a, g
