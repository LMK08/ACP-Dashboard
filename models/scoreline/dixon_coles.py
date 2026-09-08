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
across competitions (the twenty-odd clubs that have played in both connect
them), each league has its own scoring baseline, and an L2 penalty on the
team parameters keeps the fit identifiable and pulls a club with little
data toward an average side.

Which average matters. Shrinking every team toward ZERO (prior='zero', the
2026-09-07 model) gives both leagues the same centre, which forces an
average Campeonato side and an average Liga 3 side to coincide; the tier
gap is then carried only by the movers and shrunk away with them, and on
the 2024-26 walk-forward the sides promoted from the Campeonato were
expected to take 1.52 points per match over their first six Liga 3 matches
and took 1.17. With prior='league_mean' each team is shrunk toward the mean
of its OWN league (the league of its most recent match), the two means are
fitted freely, and the gap between them is identified by the clubs that
have played in both leagues: the fit puts an average Campeonato side about
a quarter of a goal behind in attack and half a goal in defence, the early
over-rating of promoted sides falls to 0.19 points per match, and the Liga 3
log loss improves (1.0509 -> 1.0489; the figures live in dc_backtest.json
under promoted_side_calibration). An unseen team plays at its league's mean
under either prior.

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


MEAN_PENALTY = 0.01   # ridge on the league means under prior='league_mean': pins the flat
                      # direction (a common shift of every attack, defence and mean changes no
                      # rate) without touching the gap between the leagues


def _objective(theta, hi, ai, li, tl, x, y, xt, yt, w, n, L, l2, league_mean):
    """Penalised negative weighted log-likelihood and its gradient.

    theta = [att (n), def (n), home_adv, rho, base (L)] and, with
    ``league_mean``, [m_att (L), m_def (L)] — the means the penalty shrinks
    each team toward, one pair per league. ``tl`` maps each team to the
    league index of its most recent match; ``xt`` / ``yt`` are the fitted
    targets (goals or the goals-xG blend), ``x`` / ``y`` the actual goals
    for the Dixon-Coles low-score correction."""
    att = theta[:n]
    dfn = theta[n:2 * n]
    home = theta[2 * n]
    rho = theta[2 * n + 1]
    base = theta[2 * n + 2:2 * n + 2 + L]
    loglam = base[li] + home + att[hi] - dfn[ai]
    logmu = base[li] + att[ai] - dfn[hi]
    lam, mu = np.exp(loglam), np.exp(logmu)
    tau, dl, dm, dr = _tau_and_grads(x, y, lam, mu, rho)
    ll = w * (-lam + xt * loglam - mu + yt * logmu + np.log(tau))
    # gradients wrt log-rates
    g_loglam = w * ((xt - lam) + dl * lam)
    g_logmu = w * ((yt - mu) + dm * mu)
    g_att = np.zeros(n); g_dfn = np.zeros(n); g_base = np.zeros(L)
    np.add.at(g_att, hi, g_loglam); np.add.at(g_att, ai, g_logmu)
    np.add.at(g_dfn, ai, -g_loglam); np.add.at(g_dfn, hi, -g_logmu)
    np.add.at(g_base, li, g_loglam + g_logmu)
    g_home = np.sum(g_loglam)
    g_rho = np.sum(w * dr)
    if league_mean:
        m_att = theta[2 * n + 2 + L:2 * n + 2 + 2 * L]
        m_def = theta[2 * n + 2 + 2 * L:2 * n + 2 + 3 * L]
        ra, rd = att - m_att[tl], dfn - m_def[tl]
        pen = (l2 * (np.sum(ra ** 2) + np.sum(rd ** 2))
               + MEAN_PENALTY * (np.sum(m_att ** 2) + np.sum(m_def ** 2)))
        g_matt = np.zeros(L); g_mdef = np.zeros(L)
        np.add.at(g_matt, tl, 2 * l2 * ra); np.add.at(g_mdef, tl, 2 * l2 * rd)
        g_matt -= 2 * MEAN_PENALTY * m_att; g_mdef -= 2 * MEAN_PENALTY * m_def
        grad = np.concatenate([g_att - 2 * l2 * ra, g_dfn - 2 * l2 * rd,
                               [g_home, g_rho], g_base, g_matt, g_mdef])
    else:
        pen = l2 * (np.sum(att ** 2) + np.sum(dfn ** 2))
        grad = np.concatenate([g_att - 2 * l2 * att, g_dfn - 2 * l2 * dfn,
                               [g_home, g_rho], g_base])
    return -(np.sum(ll) - pen), -grad


def team_leagues(df):
    """Series team -> league of the team's most recent match in ``df``."""
    both = pd.concat([df[['date', 'league', 'home']].rename(columns={'home': 'team'}),
                      df[['date', 'league', 'away']].rename(columns={'away': 'team'})])
    return both.sort_values('date', kind='stable').groupby('team')['league'].last()


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
    prior: str = 'zero'              # 'zero' | 'league_mean' — what the L2 penalty shrinks toward
    league_means: dict = None        # league -> (mean attack, mean defence); zeros under 'zero'

    # ---- fitting ---------------------------------------------------------
    @classmethod
    def fit(cls, matches, asof=None, xi=DEFAULT_XI, l2=L2_PENALTY, teams=None, mix=1.0, prior='zero'):
        """Fit on matches with date < asof (all when None), weights
        exp(-xi * days before asof). mix < 1 fits the Poisson rates on a
        blend mix*goals + (1-mix)*xG (quasi-likelihood; xG columns xg_h/xg_a
        from attach_xg, goals where missing) — the Dixon-Coles low-score
        correction always uses the actual goals. ``prior`` is what the L2
        penalty shrinks each team toward: 'zero' (one centre for both
        leagues) or 'league_mean' (the mean of the team's own league — the
        league of its most recent match — with the means fitted freely; see
        the module docstring)."""
        if prior not in ('zero', 'league_mean'):
            raise ValueError(f"prior must be 'zero' or 'league_mean', not {prior!r}")
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
        league_mean = prior == 'league_mean'
        tl = np.zeros(n, dtype=int)
        if league_mean:
            last = team_leagues(df)
            extras = [t for t in teams if t not in last.index]
            if extras:   # a team with no matches has no league whose mean could hold it
                raise ValueError(f"prior='league_mean' cannot place teams without matches: {extras}")
            tl = np.array([l_idx[int(last[t])] for t in teams])
        k = 2 * n + 2 + L + (2 * L if league_mean else 0)
        theta0 = np.zeros(k)
        theta0[2 * n + 2:2 * n + 2 + L] = np.log(max(xt.mean(), 0.1))
        bounds = [(None, None)] * (2 * n + 1) + [(-0.9, 0.9)] + [(None, None)] * (k - 2 * n - 2)
        res = minimize(_objective, theta0, args=(hi, ai, li, tl, x, y, xt, yt, w, n, L, l2, league_mean),
                       jac=True, method='L-BFGS-B', bounds=bounds, options={'maxiter': 500})
        th = res.x
        means = {l: (0.0, 0.0) for l in leagues}
        if league_mean:
            means = {l: (float(th[2 * n + 2 + L + j]), float(th[2 * n + 2 + 2 * L + j]))
                     for j, l in enumerate(leagues)}
        return cls(teams=teams, leagues=leagues, att=th[:n], dfn=th[n:2 * n],
                   base=th[2 * n + 2:2 * n + 2 + L],
                   home_adv=float(th[2 * n]), rho=float(th[2 * n + 1]), xi=xi, l2=l2, mix=mix,
                   asof=str(asof_ts.date()), n_matches=int(len(df)), weight_sum=float(w.sum()),
                   prior=prior, league_means=means)

    # ---- prediction ------------------------------------------------------
    def league_mean(self, league=None):
        """(mean attack, mean defence) the penalty shrinks a league's teams
        toward — what a team the fit has never seen plays at. (0, 0) under
        the zero prior; the first league's pair when ``league`` is unknown."""
        if not self.league_means:
            return 0.0, 0.0
        if league is not None and int(league) in self.league_means:
            m = self.league_means[int(league)]
        else:
            m = self.league_means.get(self.leagues[0] if self.leagues else None, (0.0, 0.0))
        return float(m[0]), float(m[1])

    def _team_params(self, team, league=None):
        if team in self.teams:
            i = self.teams.index(team)
            return float(self.att[i]), float(self.dfn[i]), True
        m_att, m_def = self.league_mean(league)
        return m_att, m_def, False

    def rates(self, home, away, league):
        """(lambda, mu, known_home, known_away) for a fixture; an unseen
        team plays at the league's mean parameters."""
        ah, dh, kh = self._team_params(home, league)
        aa, da, ka = self._team_params(away, league)
        li = self.leagues.index(int(league)) if int(league) in self.leagues else 0
        b = float(self.base[li])
        lam = math.exp(b + self.home_adv + ah - da)
        mu = math.exp(b + aa - dh)
        return lam, mu, kh, ka

    def tier_gap(self, league_a, league_b):
        """(scores x, concedes x) of an average ``league_b`` side relative to
        an average ``league_a`` side against the same opposition, from the
        fitted league means — e.g. tier_gap(43324, 702) = (0.73, 1.55): an
        average Campeonato side scores 27% fewer and concedes 55% more than
        an average Liga 3 side. None under the zero prior."""
        if self.prior != 'league_mean' or not self.league_means:
            return None
        a = self.league_means.get(int(league_a))
        b = self.league_means.get(int(league_b))
        if a is None or b is None:
            return None
        return math.exp(b[0] - a[0]), math.exp(a[1] - b[1])

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
            'prior': self.prior,
            'league_means': {str(l): {'att': float(m[0]), 'def': float(m[1])}
                             for l, m in (self.league_means or {}).items()},
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
                   weight_sum=float(d.get('weight_sum', 0.0)),
                   prior=d.get('prior', 'zero'),
                   league_means={int(l): (float(v['att']), float(v['def']))
                                 for l, v in d.get('league_means', {}).items()} or {l: (0.0, 0.0) for l in leagues})

    def strength_table(self, teams=None, league=None, centre=True):
        """Per-team strength for display, best first.

        Columns: ``attack`` / ``defence`` (the raw log-rate parameters; a
        HIGHER defence concedes less), ``rating`` (attack + defence),
        ``scores_x`` / ``concedes_x`` — goals scored / conceded relative to
        the AVERAGE SIDE OF THE TEAMS SHOWN (1.0), ``xgd_vs_avg`` — expected
        goal difference against that average side at a neutral venue — and
        ``known`` (False for a team the fit has never seen; it is shown AT
        the average: 1.0 / 1.0 / 0).

        With ``centre`` (default) the parameters are centred on the mean of
        the known teams shown, so a league's table reads against its own
        average rather than against a zero-parameter side (the fit spans
        both leagues, so zero sits nearer the cross-league average and a
        whole league can come out 'above average'). ``teams`` restricts (or,
        for unseen names, extends) the rows; ``league`` picks the base rate
        that scales ``xgd_vs_avg`` (first league when None). Sorted by
        ``xgd_vs_avg`` descending.
        """
        names = list(self.teams) if teams is None else list(teams)
        li = (self.leagues.index(int(league))
              if league is not None and int(league) in self.leagues else 0)
        base = float(self.base[li]) if len(self.base) else 0.0
        params = {t: self._team_params(t, league) for t in names}
        known_a = [a for a, _, k in params.values() if k]
        known_d = [d for _, d, k in params.values() if k]
        a_bar = float(np.mean(known_a)) if (centre and known_a) else 0.0
        d_bar = float(np.mean(known_d)) if (centre and known_d) else 0.0
        # Goals per side per match between two average sides at a neutral venue.
        avg_goals = math.exp(base + a_bar - d_bar)
        rows = []
        for t in names:
            a, d, known = params[t]
            rel_a = (a - a_bar) if known else 0.0
            rel_d = (d - d_bar) if known else 0.0
            scores_x, concedes_x = math.exp(rel_a), math.exp(-rel_d)
            rows.append({'team': t, 'attack': a, 'defence': d, 'rating': a + d,
                         'scores_x': scores_x, 'concedes_x': concedes_x,
                         'xgd_vs_avg': avg_goals * (scores_x - concedes_x), 'known': known})
        cols = ['team', 'attack', 'defence', 'rating', 'scores_x', 'concedes_x', 'xgd_vs_avg', 'known']
        if not rows:
            return pd.DataFrame(columns=cols)
        return (pd.DataFrame(rows, columns=cols)
                .sort_values('xgd_vs_avg', ascending=False).reset_index(drop=True))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def outcome_index(hg, ag):
    """0 = home win, 1 = draw, 2 = away win."""
    return np.where(hg > ag, 0, np.where(hg == ag, 1, 2))


def walk_forward(matches, xi=DEFAULT_XI, start=None, step_days=30, min_train=300, l2=L2_PENALTY, mix=1.0,
                 prior='zero'):
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
            model = DixonColes.fit(df, asof=t, xi=xi, l2=l2, mix=mix, prior=prior)
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


def team_season_categories(matches):
    """Category of every (team, season_id) from the data's own history —
    'promoted from CdP' when the team's previous season in the data was in
    the Campeonato (702) and this one is Liga 3 (43324), 'relegated to CdP'
    for the reverse, 'stayed' for the same league, 'new to the data' with no
    earlier season. Seasons are ordered by their first match."""
    start = matches.groupby('season_id')['date'].min()
    ts = pd.concat([matches[['season_id', 'league', 'home']].rename(columns={'home': 'team'}),
                    matches[['season_id', 'league', 'away']].rename(columns={'away': 'team'})]).drop_duplicates()
    ts['start'] = ts['season_id'].map(start)
    out, hist = {}, {}
    for r in ts.sort_values('start', kind='stable').itertuples():
        prev = hist.get(r.team)
        league = int(r.league)
        if not prev:
            cat = 'new to the data'
        elif prev[-1] == league:
            cat = 'stayed'
        elif prev[-1] == 702 and league == 43324:
            cat = 'promoted from CdP'
        elif prev[-1] == 43324 and league == 702:
            cat = 'relegated to CdP'
        else:
            cat = 'moved'
        out[(r.team, int(r.season_id))] = cat
        hist.setdefault(r.team, []).append(league)
    return out


def promoted_side_bias(pred, matches, league=43324, buckets=((1, 6), (7, 12), (13, 18), (19, 99))):
    """How a walk_forward forecast treats sides in their first season after
    promotion from the Campeonato: expected vs actual points (and goal
    difference) per match, by the side's match number within the season.
    One row per bucket: matches, n (side-matches), exp_pts, act_pts,
    exp_gd, act_gd. Expected points = 3 P(win) + P(draw)."""
    cats = team_season_categories(matches)
    mm = pd.concat([matches[['match_id', 'date', 'season_id', 'home']].rename(columns={'home': 'team'}),
                    matches[['match_id', 'date', 'season_id', 'away']].rename(columns={'away': 'team'})],
                   ignore_index=True)
    mm['match_no'] = mm.sort_values('date', kind='stable').groupby(['team', 'season_id']).cumcount() + 1
    number = {(r.match_id, r.team): int(r.match_no) for r in mm.itertuples()}
    p = pred[pred['league'] == league]
    sides = []
    for side, own, opp, p_win, sign in (('home', 'hg', 'ag', 'p_home', 1.0), ('away', 'ag', 'hg', 'p_away', -1.0)):
        sides.append(pd.DataFrame({
            'match_id': p['match_id'].values, 'team': p[side].values, 'season_id': p['season_id'].values,
            'exp_pts': (3 * p[p_win] + p['p_draw']).values,
            'act_pts': np.where(p[own] > p[opp], 3, np.where(p[own] == p[opp], 1, 0)),
            'exp_gd': sign * (p['lambda'] - p['mu']).values,
            'act_gd': (p[own] - p[opp]).values}))
    s = pd.concat(sides, ignore_index=True)
    s['cat'] = [cats.get((t, int(sid))) for t, sid in zip(s['team'], s['season_id'])]
    s['match_no'] = [number.get((mid, t)) for mid, t in zip(s['match_id'], s['team'])]
    s = s[s['cat'] == 'promoted from CdP'].dropna(subset=['match_no'])
    rows = []
    for lo, hi in buckets:
        g = s[(s['match_no'] >= lo) & (s['match_no'] <= hi)]
        if g.empty:
            continue
        rows.append({'matches': f'{lo}-{hi}' if hi < 99 else f'{lo}+', 'n': int(len(g)),
                     'exp_pts': float(g['exp_pts'].mean()), 'act_pts': float(g['act_pts'].mean()),
                     'exp_gd': float(g['exp_gd'].mean()), 'act_gd': float(g['act_gd'].mean())})
    return rows
