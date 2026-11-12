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

from models.coaches.formations import STRIKERS

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
           'pts_minus_xpts_pg': float((ev.points - ev.xpts).mean()) if len(ev) else np.nan,
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
    # against expected (build_coaches.add_state_expected): what a side of
    # that strength, against those opponents, typically does in each state.
    # Actual and expected are both taken over the matches that have one.
    ex = st[st['exp_min_lead'].notna()] if 'exp_min_lead' in st.columns else st.iloc[0:0]
    out['state_exp_matches'] = len(ex)
    tot = ex[['min_lead', 'min_draw', 'min_trail']].sum().sum()
    for s in ('lead', 'draw', 'trail'):
        mins = ex[f'min_{s}'].sum() if len(ex) else 0.0
        out[f'exp_min_share_{s}'] = _div(ex[f'exp_min_{s}'].sum(), tot) if len(ex) else np.nan
        out[f'min_share_vs_exp_{s}'] = (_div(mins, tot) - out[f'exp_min_share_{s}']) if len(ex) else np.nan
        out[f'exp_xgd90_{s}'] = _div(ex[f'exp_xgd_{s}'].sum() * 90, mins) if len(ex) else np.nan
        out[f'xgd90_vs_exp_{s}'] = (_div((ex[f'xgd_{s}'] - ex[f'exp_xgd_{s}']).sum() * 90, mins)
                                    if len(ex) else np.nan)
    trailed = st[st.ever_trailed.eq(True)]
    led = st[st.ever_led.eq(True)]
    out['trailed_n'] = len(trailed)
    out['ppg_after_trailing'] = _div(trailed.points.sum(), len(trailed))
    out['led_n'] = len(led)
    out['ppg_after_leading'] = _div(led.points.sum(), len(led))
    out['led_not_won'] = int((led.points < 3).sum())
    out['led_not_won_share'] = _div(out['led_not_won'], len(led))

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
    # share of matches read as three at the back (our formations are the
    # shape with the ball, so this is how often he builds with a back three)
    out['back3_share'] = float((f.map(_back_line) == 3).mean()) if len(f) else np.nan
    players = set()
    for xi, subs in zip(r.xi, r.sub_in_ids):
        players.update(int(p) for p in xi)
        players.update(int(p) for p in subs)
    out['players_used'] = len(players)
    if 'u21_minutes' in r.columns:
        out['u21_share'] = _div(r.u21_minutes.sum(), r.squad_minutes.sum())
    flex = flexibility(r)
    out.update(flex)
    for cut in AGE_CUTS:
        out.update(youth_trust(r, cut))
    out['age_gap'] = out.get('age_on_pitch', np.nan) - out.get('age_of_squad', np.nan)
    return out


Z_COLS = ['z_possession', 'z_field_tilt', 'z_passes_per_poss', 'z_long_share',
          'z_forward_share', 'z_def_height', 'z_log_ppda']
AGE_CUTS = (21, 23)
MIN_ADAPT_MATCHES = 12    # context-adjusted matches needed for the opponent-adaptation slope
FLEX_KEYS = ('eff_formations', 'formation_change_rate', 'back_line_change_rate', 'opp_adapt')
# the settled ↔ fluid score: formation variety alone, the steadiest of the
# three formation measures with our own formations (formations.py, 2026-11:
# 0.64 odd/even on Liga 3, against 0.49 and 0.43 for the two change rates,
# which are shown, labelled, but not scored; opp_adapt likewise at 0.26). A
# 'changed shape during the match' measure was removed: from touches it is noise.
FLUIDITY_KEYS = FLEX_KEYS[:1]
FLEX_LABELS = {
    'eff_formations': ('Formation variety', 'effective number of starting shapes'),
    'formation_change_rate': ('Changes structure between matches',
                              'share of matches starting with a different back line or number of strikers'),
    'back_line_change_rate': ('Switches back three ↔ four', 'share of matches changing the back line'),
    'opp_adapt': ('Adapts style to the opponent', 'style shift per unit of opponent strength, beyond the league norm'),
}


def _strikers(formation):
    """Strikers in one of our shapes (formations.STRIKERS); None if unknown."""
    return STRIKERS.get(str(formation))


def _back_line(formation):
    try:
        return int(str(formation).split('-')[0])
    except (ValueError, IndexError):
        return None


def flexibility(rows):
    """How much a coach varies what he does.

    eff_formations         effective number of starting shapes (inverse Simpson,
                           unbiased in the number of matches): 1 = always the
                           same shape, 3 = three shapes used equally.
    formation_change_rate  share of matches starting with a different
                           structure (back line or number of strikers) from
                           the previous match of the same spell. The full
                           shape name is not compared: close names (4-3-3 /
                           4-2-3-1) flip on detection noise.
    back_line_change_rate  the same for a back three ↔ back four alone.
    opp_adapt              how strongly his context-adjusted style still moves
                           with opponent strength (RMS slope over the style
                           scores): the league-typical shift is already
                           removed, so 0 = adapts like everyone else.

    Tested and left out (2026-11, odd/even-match reliability ≈ 0 on Liga 3):
    match-to-match style distance (Hooker / Hudl 2025) and the spread of the
    style scores — on this data they are match noise, not a coach trait.
    """
    out = {k: np.nan for k in FLEX_KEYS}
    out.update({'style_matches': 0, 'opp_slopes': {}})
    r = rows.sort_values(['spell', 'date'])
    f = r.formation.dropna()
    if len(f):
        # inverse Simpson with the unbiased estimator: the chance two of his
        # matches share a shape, so the number does not grow with match count
        n_i = f.value_counts().to_numpy(dtype=float)
        n = n_i.sum()
        if n >= 2:
            same = (n_i * (n_i - 1)).sum() / (n * (n - 1))
            out['eff_formations'] = float(1 / same) if same > 0 else float(n)
    ff = r.dropna(subset=['formation'])
    same_spell = ff.spell == ff.spell.shift()
    if same_spell.sum():
        back = ff.formation.map(_back_line)
        strikers = ff.formation.map(_strikers)
        new_back = back != back.shift()
        out['formation_change_rate'] = float((new_back | (strikers != strikers.shift()))[same_spell].mean())
        out['back_line_change_rate'] = float(new_back[same_spell].mean())
    if not set(Z_COLS) <= set(r.columns):
        return out
    z = r.dropna(subset=Z_COLS)
    out['style_matches'] = len(z)
    gap = (z.r_pre - z.r_opp_pre).to_numpy(dtype=float)
    if len(z) >= MIN_ADAPT_MATCHES and np.nanstd(gap) > 0:
        # RMS slope with each slope's own sampling variance taken off, so a
        # short record is not 'more adaptive' just for being noisier
        slopes, excess = {}, []
        sxx = float(((gap - gap.mean()) ** 2).sum())
        for c in Z_COLS:
            y = z[c].to_numpy(dtype=float)
            b, a = np.polyfit(gap, y, 1)
            resid = y - (a + b * gap)
            se2 = float((resid ** 2).sum() / (len(y) - 2) / sxx)
            slopes[c] = float(b)
            excess.append(b * b - se2)
        out['opp_adapt'] = float(np.sqrt(max(float(np.mean(excess)), 0.0)))
        out['opp_slopes'] = slopes
    return out


def youth_trust(rows, cut=21):
    """Does he play the young players he has? Minutes are exact (Liga 3 only);
    the matchday squad is the line-up sheet. trust = share of minutes given to
    under-`cut` players ÷ their share of matchday-squad places (1.0 = used in
    proportion to their places)."""
    k = f'u{cut}'
    need = [f'{k}_minutes', 'squad_minutes', f'{k}_named', 'sheet_n']
    if not set(need) <= set(rows.columns):
        return {}
    r = rows.dropna(subset=need)
    min_share = _div(r[f'{k}_minutes'].sum(), r.squad_minutes.sum())
    named_share = _div(r[f'{k}_named'].sum(), r.sheet_n.sum())
    return {
        f'{k}_matches': len(r),
        f'{k}_minutes_share': min_share,
        f'{k}_named_share': named_share,
        f'{k}_trust': _div(min_share, named_share) if named_share and named_share >= 0.03 else np.nan,
        f'{k}_start_share': _div(r[f'{k}_starts'].sum(), r.xi_n.sum()),
        f'{k}_named_per_match': _div(r[f'{k}_named'].sum(), len(r)),
        'age_on_pitch': _div(r.age_minutes_sum.sum(), r.squad_minutes.sum()),
        'age_of_squad': _div(r.sheet_age_sum.sum(), r.sheet_n.sum()),
    }


def club_style_vectors(tm, min_matches=MIN_MATCHES):
    """Mean context-adjusted style per coach × club (≥ min_matches)."""
    r = tm[(tm.coachId > 0)].dropna(subset=Z_COLS)
    g = r.groupby(['competitionId', 'coachId', 'teamId'])
    v = g[Z_COLS].mean()
    v['n'] = g.size()
    v['team'] = g.team.first()
    return v[v.n >= min_matches].reset_index()


def _pair_distances(v):
    a = v[Z_COLS].to_numpy()
    d = np.sqrt(((a[:, None, :] - a[None, :, :]) ** 2).mean(axis=2))
    i, j = np.triu_indices(len(v), 1)
    same_coach = v.coachId.to_numpy()[i] == v.coachId.to_numpy()[j]
    same_club = v.teamId.to_numpy()[i] == v.teamId.to_numpy()[j]
    return d[i, j], same_coach, same_club


def style_persistence(vectors):
    """Is style the coach's or the club's? Median style distance between
    (same coach, different clubs), (same club, different coaches) and
    (different coach and club), within one competition's vectors."""
    if len(vectors) < 3:
        return {}
    d, same_coach, same_club = _pair_distances(vectors)
    def med(mask):
        return (float(np.median(d[mask])) if mask.sum() else np.nan, int(mask.sum()))
    return {'same_coach_other_club': med(same_coach & ~same_club),
            'same_club_other_coach': med(~same_coach & same_club),
            'unrelated': med(~same_coach & ~same_club)}


def coach_club_distances(vectors, coach_id):
    """Pairwise style distance between one coach's clubs."""
    v = vectors[vectors.coachId == coach_id].reset_index(drop=True)
    rows = []
    for i in range(len(v)):
        for j in range(i + 1, len(v)):
            dist = float(np.sqrt(((v.loc[i, Z_COLS] - v.loc[j, Z_COLS]).astype(float) ** 2).mean()))
            rows.append({'Club A': v.team[i], 'Club B': v.team[j], 'Matches A': int(v.n[i]),
                         'Matches B': int(v.n[j]), 'Style distance': dist})
    return pd.DataFrame(rows)


AGE_BANDS = (0, 21, 23, 26, 30, 99)
MIN_PEERS = 15


def development_pairs(cp, min_minutes=MIN_PLAYER_MINUTES):
    """One row per coach × player with enough minutes before and under him:
    the change in on-ball value per 90, and the change EXPECTED FOR HIS AGE —
    the average change of OTHER same-age-band players whose spells under another
    coach had already ENDED when this one began (walk-forward; blank until
    MIN_PEERS such cases exist)."""
    g = cp.groupby(['coachId', 'playerId'])
    r = g.agg(team=('team', 'first'), minutes=('obv_minutes', 'sum'), obv=('obv', 'sum'),
              prior_minutes=('prior_minutes', 'first'), prior_obv=('prior_obv', 'first'),
              age_first=('age_first', 'first'), first_date=('first_under_coach', 'first'),
              last_date=('last_date', 'max')).reset_index()
    r = r[(r.minutes >= min_minutes) & (r.prior_minutes >= min_minutes) & r.age_first.notna()].copy()
    r['obv90'] = r.obv / r.minutes * 90
    r['prior_obv90'] = r.prior_obv / r.prior_minutes * 90
    r['change'] = r.obv90 - r.prior_obv90
    r['band'] = pd.cut(r.age_first, AGE_BANDS, right=False, labels=False)
    # peers: same age band, a DIFFERENT coach and a different player, spell
    # already over when this one began
    band = r.band.to_numpy()
    coach = r.coachId.to_numpy()
    player = r.playerId.to_numpy()
    start = r.first_date.to_numpy()
    end = r.last_date.to_numpy()
    change = r.change.to_numpy()
    exp, peers = np.full(len(r), np.nan), np.zeros(len(r), dtype=int)
    for i in range(len(r)):
        m = (band == band[i]) & (end < start[i]) & (coach != coach[i]) & (player != player[i])
        peers[i] = int(m.sum())
        if peers[i] >= MIN_PEERS:
            exp[i] = change[m].mean()
    r['expected'] = exp
    r['peers'] = peers
    r['vs_expected'] = r.change - r.expected
    return r


def youth_development(pairs, coach_id, cut=23):
    """Young players (under `cut` when they first played for him) against
    their own earlier record and against what their age predicted."""
    r = pairs[(pairs.coachId == coach_id) & (pairs.age_first < cut)].sort_values('change', ascending=False)
    if r.empty:
        return r, {'n': 0}
    e = r.dropna(subset=['vs_expected'])
    return r, {'n': len(r), 'improved': int((r.change > 0).sum()),
               'mean_change': float(np.average(r.change, weights=r.minutes)),
               'n_expected': len(e), 'beat_expected': int((e.vs_expected > 0).sum()),
               'mean_vs_expected': float(np.average(e.vs_expected, weights=e.minutes)) if len(e) else np.nan}


MIN_PROGRESS_MINUTES = 900   # under the coach; each half then holds 300+


def youth_progress(cp, coach_id, cut=23):
    """Young players' on-ball value per 90 in the second half of their minutes
    under the coach against the first half (by date). Needs no earlier
    record, so it covers players who arrived with no history."""
    r = cp[(cp.coachId == coach_id) & (cp.age_first < cut)]
    if r.empty or 'early_minutes' not in r.columns:
        return pd.DataFrame()
    g = r.sort_values('minutes', ascending=False).groupby('playerId').agg(
        team=('team', 'first'), age_first=('age_first', 'first'), minutes=('obv_minutes', 'sum'),
        obv=('obv', 'sum'), early_minutes=('early_minutes', 'sum'), early_obv=('early_obv', 'sum')).reset_index()
    late = g.minutes - g.early_minutes
    g = g[(g.minutes >= MIN_PROGRESS_MINUTES) & (g.early_minutes >= 300) & (late >= 300)].copy()
    if g.empty:
        return g
    g['first_half'] = g.early_obv / g.early_minutes * 90
    g['second_half'] = (g.obv - g.early_obv) / (g.minutes - g.early_minutes) * 90
    g['progress'] = g.second_half - g.first_half
    return g.sort_values('progress', ascending=False)


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
    extra = list(FLEX_KEYS) + ['age_on_pitch', 'age_gap'] + \
        [f'u{c}_{m}' for c in AGE_CUTS for m in ('trust', 'minutes_share', 'start_share')]
    extra = [k for k in extra if k in t.columns]
    for k in list(STYLE_METRICS) + ['adj_xgd', 'ppg', 'xgd', 'xpts_pg'] + extra:
        pool = t.loc[q, k].dropna().to_numpy()
        t[k + '_pct'] = [float((pool < v).mean() + 0.5 * (pool == v).mean()) * 100
                         if len(pool) and not pd.isna(v) else np.nan for v in t[k]]
    # fluid ↔ settled: the percentile(s) of FLUIDITY_KEYS
    cols = [k + '_pct' for k in FLUIDITY_KEYS if k + '_pct' in t.columns]
    if cols:
        t['fluidity'] = t[cols].mean(axis=1).where(t[cols].notna().all(axis=1))
    return t


def league_comparison(table, s, keys):
    """Where the numbers in summary `s` stand among the league's qualifying
    coaches (each on his whole record there): one dict per key with the
    value, the league median, the percentile (share of those coaches with a
    lower number, ties counted half) and the pool size. `s` may be any scope
    (one spell, one season) - the pool is always whole records."""
    out = []
    pool_rows = table[table.qualifies] if len(table) and 'qualifies' in table.columns else table.iloc[0:0]
    for k in keys:
        v = s.get(k, np.nan)
        pool = pd.to_numeric(pool_rows[k], errors='coerce').dropna().to_numpy() if k in pool_rows.columns else np.array([])
        ok = len(pool) >= 5 and v is not None and not pd.isna(v)
        out.append({'key': k, 'value': v, 'n': int(len(pool)),
                    'median': float(np.median(pool)) if len(pool) >= 5 else np.nan,
                    'pct': float((pool < v).mean() + 0.5 * (pool == v).mean()) * 100 if ok else np.nan})
    return out


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
    r['played_when_named'] = r.named_played / r.named
    r['minutes_share'] = r.minutes / (r.team_matches_in_span * 90.0)
    r['team_ppg_with'] = r.points / r.matches
    return r.sort_values('last_date', ascending=False).reset_index(drop=True)


def reliability(tm, competition_id, keys, min_matches=30):
    """Odd/even-match agreement of each metric among coaches with at least
    min_matches event matches in the competition: the correlation between a
    coach's number on his odd matches and on his even matches. Near 1 = a
    stable trait of the coach; near 0 = match noise."""
    r = tm[(tm.competitionId == competition_id) & (tm.coachId > 0)].sort_values(['date', 'matchId'])
    r = r.assign(_half=r.groupby('coachId').cumcount() % 2)
    a, b = [], []
    for _, g in r.groupby('coachId'):
        if g.has_events.sum() < min_matches:
            continue
        a.append(summarise(g[g._half == 0]))
        b.append(summarise(g[g._half == 1]))
    out = {}
    for k in keys:
        x = pd.DataFrame({'a': [d.get(k, np.nan) for d in a], 'b': [d.get(k, np.nan) for d in b]}).dropna()
        out[k] = (float(x.a.corr(x.b)), len(x)) if len(x) >= 10 else (np.nan, len(x))
    return out


def reliability_word(r):
    if r is None or pd.isna(r):
        return 'not measured'
    return 'stable' if r >= 0.7 else 'moderate' if r >= 0.4 else 'noisy'
