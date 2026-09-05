"""Dixon-Coles scoreline model (Dixon & Coles 1997) with time decay.

Goals in a match are Poisson with rates

    home:  lambda = exp(base[league] + home_adv + att[home] - def[away])
    away:  mu     = exp(base[league]            + att[away] - def[home])

and the joint probability of a scoreline is the product of the two Poisson
pmfs times the Dixon-Coles correction tau(x, y) for the four low scores
(0-0, 1-0, 0-1, 1-1), whose strength rho captures the excess of draws that
independent Poissons miss. Matches are weighted by exp(-xi * days_ago), so
recent form counts more; xi is chosen by walk-forward backtest (build_dc.py).

One model covers both leagues: team attack/defence parameters are shared
across competitions (the twenty clubs that have played in both connect
them), each league has its own scoring baseline, and a light L2 penalty on
team parameters keeps the fit identifiable and pulls a club with little
data toward league average — a newly promoted side starts as an average
team rather than an unknown.

Pure numpy/scipy; no Streamlit. predict() gives the score matrix and
everything derived from it (W/D/L, expected goals, clean sheets, over 2.5,
most likely scorelines).
"""
import json
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

MAX_GOALS = 8          # score matrix is (MAX_GOALS+1)^2; mass beyond is negligible at these rates
DEFAULT_XI = 0.0025    # per day: half-life ~ 9 months
L2_PENALTY = 0.02      # on att/def; ~ prior sd 5 on the log-rate scale, mild


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def matches_from_summary(matches_summary_df):
    """Played matches from matches_summary.parquet as the model's frame:
    columns home, away, hg, ag, date, league (competitionId int), season_id."""
    m = matches_summary_df.copy()
    score = m['score'].astype(str).str.extract(r'^\s*(\d+)\s*-\s*(\d+)\s*$')
    m['hg'] = pd.to_numeric(score[0], errors='coerce')
    m['ag'] = pd.to_numeric(score[1], errors='coerce')
    m = m.dropna(subset=['hg', 'ag', 'homeTeamName', 'awayTeamName', 'competitionId'])
    out = pd.DataFrame({
        'match_id': m['matchId'].values,
        'home': m['homeTeamName'].astype(str).values,
        'away': m['awayTeamName'].astype(str).values,
        'hg': m['hg'].astype(int).values,
        'ag': m['ag'].astype(int).values,
        'date': pd.to_datetime(m['dateutc'], errors='coerce').values,
        'league': pd.to_numeric(m['competitionId'], errors='coerce').astype(int).values,
        'season_id': pd.to_numeric(m['seasonId'], errors='coerce').astype('Int64').values,
    })
    return out.dropna(subset=['date']).sort_values('date').reset_index(drop=True)


def attach_xg(matches, raw_events_path):
    """Add per-match non-penalty xG for both sides (xg_h, xg_a) from the
    events parquet; matches without events get NaN and fall back to goals."""
    import pyarrow.parquet as pq
    ev = pq.read_table(raw_events_path, columns=['matchId', 'team.name', 'type.primary', 'shot.xg'],
                       filters=[('type.primary', '==', 'shot')]).to_pandas()
    ev = ev.dropna(subset=['shot.xg', 'team.name'])
    xg = ev.groupby(['matchId', 'team.name'], observed=True)['shot.xg'].sum()
    out = matches.copy()
    out['xg_h'] = [xg.get((m, h), np.nan) for m, h in zip(out['match_id'], out['home'])]
    out['xg_a'] = [xg.get((m, a), np.nan) for m, a in zip(out['match_id'], out['away'])]
    return out


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def _tau_and_grads(x, y, lam, mu, rho):
    """Dixon-Coles correction tau for arrays of scores, plus d log tau wrt
    lam, mu and rho (zeros where tau == 1)."""
    tau = np.ones_like(lam)
    dl = np.zeros_like(lam)
    dm = np.zeros_like(lam)
    dr = np.zeros_like(lam)
    m00 = (x == 0) & (y == 0)
    m10 = (x == 1) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m11 = (x == 1) & (y == 1)
    tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
    tau[m10] = 1.0 + mu[m10] * rho
    tau[m01] = 1.0 + lam[m01] * rho
    tau[m11] = 1.0 - rho
    tau = np.clip(tau, 1e-9, None)
    dl[m00] = -mu[m00] * rho / tau[m00]
    dm[m00] = -lam[m00] * rho / tau[m00]
    dr[m00] = -lam[m00] * mu[m00] / tau[m00]
    dm[m10] = rho / tau[m10]
    dr[m10] = mu[m10] / tau[m10]
    dl[m01] = rho / tau[m01]
    dr[m01] = lam[m01] / tau[m01]
    dr[m11] = -1.0 / tau[m11]
    return tau, dl, dm, dr


@dataclass
class DixonColes:
    teams: list = field(default_factory=list)
    leagues: list = field(default_factory=list)
    att: np.ndarray = None
    dfn: np.ndarray = None
    base: np.ndarray = None
    home_adv: float = 0.0
    rho: float = 0.0
    xi: float = DEFAULT_XI
    l2: float = L2_PENALTY
    mix: float = 1.0
    asof: str = None
    n_matches: int = 0
    weight_sum: float = 0.0

    # ---- fitting ---------------------------------------------------------
    @classmethod
    def fit(cls, matches, asof=None, xi=DEFAULT_XI, l2=L2_PENALTY, teams=None, mix=1.0):
        """Fit on matches with date < asof (all when None), weights
        exp(-xi * days before asof). mix < 1 fits the Poisson rates on a
        blend mix*goals + (1-mix)*xG (quasi-likelihood; xG columns xg_h/xg_a
        from attach_xg, goals where missing) — the Dixon-Coles low-score
        correction always uses the actual goals."""
        df = matches
        asof_ts = pd.Timestamp(asof) if asof is not None else df['date'].max() + pd.Timedelta(days=1)
        df = df[df['date'] < asof_ts]
        if df.empty:
            raise ValueError('no matches to fit on')
        teams = sorted(set(df['home']) | set(df['away']) | set(teams or []))
        leagues = sorted(set(int(l) for l in df['league']))
        t_idx = {t: i for i, t in enumerate(teams)}
        l_idx = {l: i for i, l in enumerate(leagues)}
        hi = df['home'].map(t_idx).values
        ai = df['away'].map(t_idx).values
        li = df['league'].map(l_idx).values
        x = df['hg'].values.astype(float)
        y = df['ag'].values.astype(float)
        xt, yt = x, y
        if mix < 1.0 and 'xg_h' in df.columns:
            xg_h = df['xg_h'].fillna(df['hg']).values.astype(float)
            xg_a = df['xg_a'].fillna(df['ag']).values.astype(float)
            xt = mix * x + (1 - mix) * xg_h
            yt = mix * y + (1 - mix) * xg_a
        days = (asof_ts - df['date']).dt.days.values.astype(float)
        w = np.exp(-xi * days)
        n, L = len(teams), len(leagues)

        def unpack(theta):
            att = theta[:n]
            dfn = theta[n:2 * n]
            home = theta[2 * n]
            rho = theta[2 * n + 1]
            base = theta[2 * n + 2:2 * n + 2 + L]
            return att, dfn, home, rho, base

        def negloglik(theta):
            att, dfn, home, rho, base = unpack(theta)
            loglam = base[li] + home + att[hi] - dfn[ai]
            logmu = base[li] + att[ai] - dfn[hi]
            lam, mu = np.exp(loglam), np.exp(logmu)
            tau, dl, dm, dr = _tau_and_grads(x, y, lam, mu, rho)
            ll = w * (-lam + xt * loglam - mu + yt * logmu + np.log(tau))
            pen = l2 * (np.sum(att ** 2) + np.sum(dfn ** 2))
            # gradients wrt log-rates
            g_loglam = w * ((xt - lam) + dl * lam)
            g_logmu = w * ((yt - mu) + dm * mu)
            g_rho = np.sum(w * dr)
            g_att = np.zeros(n); g_dfn = np.zeros(n); g_base = np.zeros(L)
            np.add.at(g_att, hi, g_loglam); np.add.at(g_att, ai, g_logmu)
            np.add.at(g_dfn, ai, -g_loglam); np.add.at(g_dfn, hi, -g_logmu)
            np.add.at(g_base, li, g_loglam + g_logmu)
            g_home = np.sum(g_loglam)
            grad = np.concatenate([g_att - 2 * l2 * att, g_dfn - 2 * l2 * dfn,
                                   [g_home, g_rho], g_base])
            return -(np.sum(ll) - pen), -grad

        theta0 = np.zeros(2 * n + 2 + L)
        theta0[2 * n + 2:] = np.log(max(xt.mean(), 0.1))
        bounds = [(None, None)] * (2 * n + 1) + [(-0.9, 0.9)] + [(None, None)] * L
        res = minimize(negloglik, theta0, jac=True, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 500})
        att, dfn, home, rho, base = unpack(res.x)
        return cls(teams=teams, leagues=leagues, att=att, dfn=dfn, base=base,
                   home_adv=float(home), rho=float(rho), xi=xi, l2=l2, mix=mix,
                   asof=str(asof_ts.date()), n_matches=int(len(df)), weight_sum=float(w.sum()))

    # ---- prediction ------------------------------------------------------
    def _team_params(self, team):
        if team in self.teams:
            i = self.teams.index(team)
            return float(self.att[i]), float(self.dfn[i]), True
        return 0.0, 0.0, False

    def rates(self, home, away, league):
        """(lambda, mu, known_home, known_away) for a fixture."""
        ah, dh, kh = self._team_params(home)
        aa, da, ka = self._team_params(away)
        li = self.leagues.index(int(league)) if int(league) in self.leagues else 0
        b = float(self.base[li])
        lam = math.exp(b + self.home_adv + ah - da)
        mu = math.exp(b + aa - dh)
        return lam, mu, kh, ka

    def score_matrix(self, lam, mu, max_goals=MAX_GOALS):
        """P(home goals = i, away goals = j) for i, j in 0..max_goals, with
        the Dixon-Coles correction applied to the four low scores; rows sum
        to slightly under 1 only through truncation."""
        k = np.arange(max_goals + 1)
        ph = np.exp(-lam + k * math.log(lam) - gammaln(k + 1))
        pa = np.exp(-mu + k * math.log(mu) - gammaln(mu * 0 + k + 1))
        P = np.outer(ph, pa)
        P[0, 0] *= 1 - lam * mu * self.rho
        P[1, 0] *= 1 + mu * self.rho
        P[0, 1] *= 1 + lam * self.rho
        P[1, 1] *= 1 - self.rho
        P = np.clip(P, 0, None)
        return P / P.sum()

    def predict(self, home, away, league, max_goals=MAX_GOALS):
        lam, mu, kh, ka = self.rates(home, away, league)
        P = self.score_matrix(lam, mu, max_goals)
        i, j = np.indices(P.shape)
        p_home = float(P[i > j].sum())
        p_draw = float(np.trace(P))
        p_away = float(P[i < j].sum())
        flat = sorted(((float(P[a, b]), int(a), int(b)) for a in range(P.shape[0]) for b in range(P.shape[1])),
                      reverse=True)
        return {
            'home': home, 'away': away, 'league': int(league),
            'lambda': lam, 'mu': mu,
            'known_home': kh, 'known_away': ka,
            'p_home': p_home, 'p_draw': p_draw, 'p_away': p_away,
            'over_2_5': float(P[(i + j) > 2].sum()),
            'btts': float(P[(i > 0) & (j > 0)].sum()),
            'clean_sheet_home': float(P[:, 0].sum()),
            'clean_sheet_away': float(P[0, :].sum()),
            'top_scores': [(f'{a}-{b}', p) for p, a, b in flat[:6]],
            'matrix': P,
        }

    # ---- persistence -----------------------------------------------------
    def to_dict(self):
        return {
            'model': 'dixon_coles_v1', 'asof': self.asof, 'xi': self.xi, 'l2': self.l2, 'mix': self.mix,
            'home_adv': self.home_adv, 'rho': self.rho,
            'n_matches': self.n_matches, 'weight_sum': self.weight_sum,
            'leagues': {str(l): float(b) for l, b in zip(self.leagues, self.base)},
            'teams': {t: {'att': float(a), 'def': float(d)} for t, a, d in zip(self.teams, self.att, self.dfn)},
        }

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(self.to_dict(), fh, indent=1, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, encoding='utf-8') as fh:
            d = json.load(fh)
        teams = list(d['teams'])
        leagues = [int(l) for l in d['leagues']]
        return cls(teams=teams, leagues=leagues,
                   att=np.array([d['teams'][t]['att'] for t in teams]),
                   dfn=np.array([d['teams'][t]['def'] for t in teams]),
                   base=np.array([d['leagues'][str(l)] for l in leagues]),
                   home_adv=float(d['home_adv']), rho=float(d['rho']), xi=float(d['xi']),
                   l2=float(d.get('l2', L2_PENALTY)), mix=float(d.get('mix', 1.0)),
                   asof=d.get('asof'), n_matches=int(d.get('n_matches', 0)),
                   weight_sum=float(d.get('weight_sum', 0.0)))

    def strength_table(self):
        """Per-team attack/defence on the log-rate scale plus a single
        'rating' (att + def) for display, sorted best first."""
        rows = [{'team': t, 'attack': float(a), 'defence': float(d), 'rating': float(a + d)}
                for t, a, d in zip(self.teams, self.att, self.dfn)]
        return pd.DataFrame(rows).sort_values('rating', ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def outcome_index(hg, ag):
    """0 = home win, 1 = draw, 2 = away win."""
    return np.where(hg > ag, 0, np.where(hg == ag, 1, 2))


def walk_forward(matches, xi=DEFAULT_XI, start=None, step_days=30, min_train=300, l2=L2_PENALTY, mix=1.0):
    """Refit every `step_days` from `start` (default: after min_train matches)
    on everything before the window, predict the window. Returns one row per
    predicted match with p_home/p_draw/p_away, lambda/mu and the outcome."""
    df = matches.sort_values('date').reset_index(drop=True)
    if start is None:
        start = df['date'].iloc[min(min_train, len(df) - 1)]
    start = pd.Timestamp(start).normalize()
    end = df['date'].max()
    rows = []
    t = start
    while t <= end:
        t_next = t + pd.Timedelta(days=step_days)
        window = df[(df['date'] >= t) & (df['date'] < t_next)]
        if not window.empty:
            model = DixonColes.fit(df, asof=t, xi=xi, l2=l2, mix=mix)
            for _, r in window.iterrows():
                pr = model.predict(r['home'], r['away'], r['league'])
                rows.append({'match_id': r['match_id'], 'date': r['date'], 'season_id': r['season_id'],
                             'league': r['league'], 'home': r['home'], 'away': r['away'],
                             'hg': r['hg'], 'ag': r['ag'], 'lambda': pr['lambda'], 'mu': pr['mu'],
                             'p_home': pr['p_home'], 'p_draw': pr['p_draw'], 'p_away': pr['p_away'],
                             'known': pr['known_home'] and pr['known_away'], 'asof': t})
        t = t_next
    return pd.DataFrame(rows)


def metrics(pred):
    """Multinomial log loss, Brier and accuracy over W/D/L for a walk_forward frame."""
    if pred.empty:
        return {'n': 0}
    P = pred[['p_home', 'p_draw', 'p_away']].values
    P = np.clip(P, 1e-9, 1)
    P = P / P.sum(axis=1, keepdims=True)
    o = outcome_index(pred['hg'].values, pred['ag'].values)
    ll = -np.mean(np.log(P[np.arange(len(o)), o]))
    onehot = np.eye(3)[o]
    brier = float(np.mean(np.sum((P - onehot) ** 2, axis=1)))
    acc = float(np.mean(P.argmax(axis=1) == o))
    return {'n': int(len(o)), 'log_loss': float(ll), 'brier': brier, 'accuracy': acc}


def base_rate_metrics(pred, train_rates):
    """Same metrics for a constant W/D/L forecast (the training base rates)."""
    P = np.tile(np.asarray(train_rates, dtype=float), (len(pred), 1))
    o = outcome_index(pred['hg'].values, pred['ag'].values)
    ll = -np.mean(np.log(np.clip(P[np.arange(len(o)), o], 1e-9, 1)))
    onehot = np.eye(3)[o]
    return {'n': int(len(o)), 'log_loss': float(ll),
            'brier': float(np.mean(np.sum((P - onehot) ** 2, axis=1))),
            'accuracy': float(np.mean(P.argmax(axis=1) == o))}


def reliability(pred, bins=(0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0)):
    """Calibration table per outcome: predicted-probability bin -> mean
    predicted, observed frequency, count."""
    o = outcome_index(pred['hg'].values, pred['ag'].values)
    out = []
    for k, col in enumerate(['p_home', 'p_draw', 'p_away']):
        p = pred[col].values
        hit = (o == k).astype(float)
        idx = np.digitize(p, bins[1:-1])
        for b in range(len(bins) - 1):
            m = idx == b
            if m.sum() == 0:
                continue
            out.append({'outcome': ['home', 'draw', 'away'][k], 'bin_lo': bins[b], 'bin_hi': bins[b + 1],
                        'n': int(m.sum()), 'predicted': float(p[m].mean()), 'observed': float(hit[m].mean())})
    return pd.DataFrame(out)
