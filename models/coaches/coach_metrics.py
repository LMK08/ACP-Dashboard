"""Coach metrics: pure pandas over the tables build_coaches.py writes
(no streamlit). One row of `coach_team_match.parquet` is one team in one match.

What is measured, and where the idea comes from:

  Results vs chances      points − expected points (each shot an independent
                          chance): the standard luck filter.
  Team level              xG difference per match against the opponent's
                          PRE-match rating and the home edge → what the side
                          would do against an average team on neutral ground.
  Change since arrival    the same, against the level the team had in the
                          coach's first match (Analytics FC's 'performance
                          impact'; regression to the mean and squad turnover
                          are NOT removed — the page says so).
  Style                   possession, field tilt, PPDA, defensive-action
                          height, high / counter-press recoveries, long-ball
                          and forward share, passes per possession, crossing,
                          counters, set-piece share of xG — the Coach ID family
                          of tendencies, as percentiles among coaches of the
                          same competition.
  In-game                 substitution timing, game-state xG, points after
                          trailing / leading.
  Squad                   line-up changes, formations, players used, under-21
                          minutes, players' on-ball value against their own
                          earlier record.
"""
import numpy as np
import pandas as pd

MIN_MATCHES = 10          # a coach needs this many event matches to be ranked
MIN_PLAYER_MINUTES = 450  # each side of a before / under comparison

# key → (label, higher value means…, format)
STYLE_METRICS = {
    'possession':        ('Possession (pass share)', 'more of the ball', '{:.1%}'),
    'field_tilt':        ('Field tilt', 'more final-third passing than the opponent', '{:.1%}'),
    'passes_per_poss':   ('Passes per possession', 'longer spells on the ball', '{:.2f}'),
    'long_share':        ('Long-pass share', 'more long balls', '{:.1%}'),
    'forward_share':     ('Forward-pass share', 'more direct', '{:.1%}'),
    'cross_share':       ('Crosses per final-third pass', 'more crossing', '{:.1%}'),
    'counters':          ('Counter-attacks per match', 'more counters', '{:.2f}'),
    'set_piece_share':   ('Set-piece share of xG', 'more reliant on set pieces', '{:.1%}'),
    'press_intensity':   ('Pressing (PPDA, lower = more)', 'more pressing', '{:.1f}'),
    'def_height':        ('Defensive-action height', 'defends higher up', '{:.1f}'),
    'high_recoveries':   ('Final-third recoveries per match', 'wins it high', '{:.1f}'),
    'counterpress':      ('Counter-press recoveries per match', 'more counter-pressing', '{:.1f}'),
}
# shown value differs from the ranked value for PPDA (ranked inverted)
_SUM_COLS = ['passes', 'opp_passes', 'passes_f3', 'opp_passes_f3', 'long_passes', 'fwd_passes',
             'crosses', 'possessions', 'counterattacks', 'xg', 'xg_set_piece', 'opp_press_passes',
             'press_def', 'def_x_sum', 'def_actions', 'high_recoveries', 'counterpress', 'shots',
             'opp_xg', 'opp_shots']


def _div(a, b):
    return float(a) / float(b) if b and not pd.isna(b) and b != 0 else np.nan


def style_values(rows):
    """Style numbers for a set of team-match rows (ratios of sums)."""
    r = rows[rows.has_events]
    n = len(r)
    if n == 0:
        return {k: np.nan for k in STYLE_METRICS} | {'ppda': np.nan}
    s = r[_SUM_COLS].sum()
    ppda = _div(s.opp_press_passes, s.press_def)
    return {
        'possession': _div(s.passes, s.passes + s.opp_passes),
        'field_tilt': _div(s.passes_f3, s.passes_f3 + s.opp_passes_f3),
        'passes_per_poss': _div(s.passes, s.possessions),
        'long_share': _div(s.long_passes, s.passes),
        'forward_share': _div(s.fwd_passes, s.passes),
        'cross_share': _div(s.crosses, s.passes_f3),
        'counters': _div(s.counterattacks, n),
        'set_piece_share': _div(s.xg_set_piece, s.xg),
        'press_intensity': -ppda if not pd.isna(ppda) else np.nan,  # ranked: higher = more pressing
        'ppda': ppda,
        'def_height': _div(s.def_x_sum, s.def_actions),
        'high_recoveries': _div(s.high_recoveries, n),
        'counterpress': _div(s.counterpress, n),
    }


def _mean_ci(x):
    x = pd.Series(x).dropna()
    n = len(x)
    if n == 0:
        return np.nan, np.nan, 0
    half = 1.96 * x.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return float(x.mean()), float(half), n


def summarise(rows):
    """Everything the profile shows for a set of team-match rows."""
    r = rows
    n = len(r)
    ev = r[r.has_events]
    out = {'matches': n, 'event_matches': len(ev),
           'wins': int((r.points == 3).sum()), 'draws': int((r.points == 1).sum()),
           'losses': int((r.points == 0).sum()),
           'ppg': _div(r.points.sum(), n), 'gf': _div(r.gf.sum(), n), 'ga': _div(r.ga.sum(), n),
           'xg': _div(ev.xg.sum(), len(ev)), 'xga': _div(ev.opp_xg.sum(), len(ev)),
           'xgd': _div(ev.xgd.sum(), len(ev)),
           'xpts_pg': _div(ev.xpts.sum(), len(ev)),
           'pts_minus_xpts': float((ev.points - ev.xpts).sum()) if len(ev) else np.nan,
           'ppg_events': _div(ev.points.sum(), len(ev))}
    home = np.where(ev.side == 'home', 1.0, -1.0)
    # vs an average side at a neutral venue: remove the opponent and the venue
    adj = ev.xgd + ev.r_opp_pre - home * ev.hfa_pre
    out['adj_xgd'], out['adj_xgd_ci'], _ = _mean_ci(adj)
    # vs the level the team had when the coach arrived
    base = ev[ev.r_inherited.notna()]
    hb = np.where(base.side == 'home', 1.0, -1.0)
    imp = base.xgd - (base.r_inherited - base.r_opp_pre + hb * base.hfa_pre)
    out['impact'], out['impact_ci'], out['impact_n'] = _mean_ci(imp)
    out.update(style_values(r))

    st = r[r.state_ok.eq(True)]
    out['state_matches'] = len(st)
    for s in ('lead', 'draw', 'trail'):
        mins = st[f'min_{s}'].sum()
        out[f'min_share_{s}'] = _div(mins, st[['min_lead', 'min_draw', 'min_trail']].sum().sum())
        out[f'xgd90_{s}'] = _div(st[f'xgd_{s}'].sum() * 90, mins)
    trailed = st[st.ever_trailed.eq(True)]
    led = st[st.ever_led.eq(True)]
    out['trailed_n'] = len(trailed)
    out['ppg_after_trailing'] = _div(trailed.points.sum(), len(trailed))
    out['led_n'] = len(led)
    out['ppg_after_leading'] = _div(led.points.sum(), len(led))
    out['led_not_won'] = int((led.points < 3).sum())

    sub = r[r.subs_used.notna()]
    out['sub_matches'] = len(sub)
    out['subs_used'] = _div(sub.subs_used.sum(), len(sub))
    out['first_sub_min'] = float(sub.first_sub_min.mean()) if sub.first_sub_min.notna().any() else np.nan
    out['subs_before_60_share'] = _div(sub.subs_before_60.sum(), sub.subs_used.sum())
    out['xi_changes'] = float(r.xi_changes.mean()) if r.xi_changes.notna().any() else np.nan
    f = r.formation.dropna()
    out['formations'] = f.value_counts().to_dict()
    out['main_formation'] = f.mode().iloc[0] if len(f) else None
    out['main_formation_share'] = _div((f == out['main_formation']).sum(), len(f)) if len(f) else np.nan
    out['in_match_switch_share'] = _div((r.formations_in_match > 1).sum(), r.formations_in_match.notna().sum())
    players = set()
    for xi, subs in zip(r.xi, r.sub_in_ids):
        players.update(int(p) for p in xi)
        players.update(int(p) for p in subs)
    out['players_used'] = len(players)
    if 'u21_minutes' in r.columns:
        out['u21_share'] = _div(r.u21_minutes.sum(), r.squad_minutes.sum())
    return out


def spells(tm, coach_id):
    """One row per unbroken run of matches at a team, newest first."""
    r = tm[tm.coachId == coach_id]
    rows = []
    for spell, g in r.groupby('spell'):
        s = summarise(g)
        rows.append({'spell': spell, 'team': g.team.iloc[0], 'teamId': int(g.teamId.iloc[0]),
                     'competitionId': int(g.competitionId.iloc[0]),
                     'from': g.date.min(), 'to': g.date.max(), **s})
    return pd.DataFrame(rows).sort_values('from', ascending=False).reset_index(drop=True) if rows else pd.DataFrame()


def coach_table(tm, competition_id=None, min_matches=MIN_MATCHES):
    """One row per coach (his whole record in the competition) with the style
    metrics and their percentile among coaches with ≥ min_matches there."""
    r = tm[tm.coachId > 0]
    if competition_id is not None:
        r = r[r.competitionId == competition_id]
    rows = []
    for cid, g in r.groupby('coachId'):
        s = summarise(g)
        last = g.sort_values('date').iloc[-1]
        rows.append({'coachId': int(cid), 'last_team': last.team, 'last_match': last.date,
                     **{k: v for k, v in s.items() if not isinstance(v, dict)}})
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    q = t.event_matches >= min_matches
    t['qualifies'] = q
    for k in list(STYLE_METRICS) + ['adj_xgd', 'ppg', 'xgd', 'xpts_pg']:
        pool = t.loc[q, k].dropna().to_numpy()
        t[k + '_pct'] = [float((pool < v).mean() + 0.5 * (pool == v).mean()) * 100
                         if len(pool) and not pd.isna(v) else np.nan for v in t[k]]
    return t


def similar_coaches(table, coach_id, n=5):
    """Nearest coaches on the style percentiles (qualifying coaches only)."""
    keys = [k + '_pct' for k in STYLE_METRICS]
    pool = table[table.qualifies].dropna(subset=keys)
    me = table[table.coachId == coach_id]
    if me.empty or me[keys].isna().any(axis=1).iloc[0] or len(pool) < 3:
        return pd.DataFrame()
    d = np.sqrt(((pool[keys].to_numpy() - me[keys].to_numpy()) ** 2).mean(axis=1))
    out = pool.assign(distance=d)
    out = out[out.coachId != coach_id].sort_values('distance').head(n)
    return out[['coachId', 'last_team', 'event_matches', 'distance']]


def player_development(cp, coach_id, min_minutes=MIN_PLAYER_MINUTES):
    """Players' on-ball value per 90 under the coach against their OWN record
    in earlier matches (any team). Age and team context are not removed."""
    r = cp[cp.coachId == coach_id]
    if r.empty:
        return r, {'n': 0}
    # one row per player: his clubs under this coach together
    r = r.sort_values('minutes', ascending=False).groupby('playerId').agg(
        team=('team', 'first'), minutes=('obv_minutes', 'sum'), obv=('obv', 'sum'),
        prior_minutes=('prior_minutes', 'first'), prior_obv=('prior_obv', 'first')).reset_index()
    r = r[(r.minutes >= min_minutes) & (r.prior_minutes >= min_minutes)].copy()
    if r.empty:
        return r, {'n': 0}
    r['obv90'] = r.obv / r.minutes * 90
    r['prior_obv90'] = r.prior_obv / r.prior_minutes * 90
    r['change'] = r.obv90 - r.prior_obv90
    summary = {'n': len(r), 'improved': int((r.change > 0).sum()),
               'mean_change': float(np.average(r.change, weights=r.minutes))}
    return r.sort_values('change', ascending=False), summary


def coached_by(cp, tm, player_id):
    """The coaches a player has played under: exact minutes, share of the
    coach's matches he played in while he was there, OBV per 90."""
    r = cp[cp.playerId == player_id].copy()
    if r.empty:
        return r
    r['obv90'] = np.where(r.obv_minutes > 0, r.obv / r.obv_minutes.replace(0, np.nan) * 90, np.nan)
    r['start_share'] = r.starts / r.team_matches_in_span
    r['played_share'] = r.matches / r.team_matches_in_span
    r['minutes_share'] = r.minutes / (r.team_matches_in_span * 90.0)
    r['team_ppg_with'] = r.points / r.matches
    return r.sort_values('last_date', ascending=False).reset_index(drop=True)
