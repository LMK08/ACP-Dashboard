"""Coach Profile view, and the 'Coached by' block the Player Profile shows.

Data: the four tables models/coaches/build_coaches.py writes; every number is
computed in models/coaches/coach_metrics.py. Collaborators are read from the
running app module in the binding block at the top of render().
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import context_bar
import navigation
import theme
from models.coaches import coach_metrics as cm

_DIR = Path(__file__).resolve().parents[1] / 'models' / 'coaches'
_FILES = ('coach_team_match.parquet', 'coaches.parquet', 'coach_player.parquet')
_CFG = {'displayModeBar': False}
PAGE = 'Coach Profile'
SEED_KEY = 'coach_profile_seed'


def _app():
    return sys.modules['__main__']


def tables_signature():
    return tuple((f, (_DIR / f).stat().st_mtime_ns if (_DIR / f).exists() else None) for f in _FILES)


@st.cache_data(show_spinner=False)
def load_tables(signature):
    if any(m is None for _, m in signature):
        return None
    tm = pd.read_parquet(_DIR / 'coach_team_match.parquet')
    coaches = pd.read_parquet(_DIR / 'coaches.parquet').set_index('coachId')
    cp = pd.read_parquet(_DIR / 'coach_player.parquet')
    return tm, coaches, cp


@st.cache_data(show_spinner=False)
def league_table(signature, comp_id):
    tm, _, _ = load_tables(signature)
    return cm.coach_table(tm, comp_id)


RELIABILITY_KEYS = list(cm.FLEX_KEYS) + ['xgd90_vs_exp_lead', 'xgd90_vs_exp_draw', 'xgd90_vs_exp_trail'] + ['u21_trust', 'u23_trust', 'u21_minutes_share', 'u23_minutes_share',
                                           'age_gap', 'first_sub_min', 'subs_before_60_share',
                                           'ppg_after_trailing', 'xgd90_trail', 'xgd90_lead', 'xi_changes',
                                           'set_piece_share']


@st.cache_data(show_spinner=False)
def league_reliability(signature, comp_id):
    tm, _, _ = load_tables(signature)
    return cm.reliability(tm, comp_id, RELIABILITY_KEYS)


@st.cache_data(show_spinner=False)
def league_club_vectors(signature, comp_id):
    tm, _, _ = load_tables(signature)
    v = cm.club_style_vectors(tm[tm.competitionId == comp_id])
    return v, cm.style_persistence(v)


@st.cache_data(show_spinner=False)
def league_development_pairs(signature):
    _, _, cp = load_tables(signature)
    return cm.development_pairs(cp)


def _stable(rel, key):
    r = rel.get(key, (np.nan, 0))[0]
    return cm.reliability_word(r) + ('' if pd.isna(r) else f' ({r:.2f})')


def coach_name(coaches, coach_id):
    try:
        return str(coaches.at[int(coach_id), 'name'])
    except KeyError:
        return f'Coach #{int(coach_id)}'


def _fmt(v, spec='{:.2f}', dash='–'):
    return dash if v is None or pd.isna(v) else spec.format(v)


def _signed(v, spec='{:+.2f}'):
    return _fmt(v, spec)


def _pm(v, half, spec='{:+.2f}'):
    if pd.isna(v):
        return '–'
    return spec.format(v) + (f' ± {half:.2f}' if not pd.isna(half) else '')


def _layout(fig, height=320, **kw):
    fig.update_layout(height=height, paper_bgcolor=theme.FIGURE_BG, plot_bgcolor=theme.FIGURE_BG,
                      font=dict(family=theme.FONT_STACK, color=theme.FIGURE_INK, size=12),
                      margin=dict(l=10, r=10, t=36, b=10), **kw)
    fig.update_xaxes(gridcolor=theme.GRID, zeroline=False)
    fig.update_yaxes(gridcolor=theme.GRID, zeroline=False)
    return fig


# ─────────────────────────────── sections ───────────────────────────────
def _vs_league(table, s, spec, league_name):
    """Under a row of cards: the same numbers against the league's coaches.
    spec = [(summary key, label, formatter)]."""
    comp = cm.league_comparison(table, s, [k for k, _, _ in spec])
    if not comp or max(c['n'] for c in comp) < 5:
        st.caption(f'No league comparison: fewer than 5 {league_name} coaches have {cm.MIN_MATCHES}+ matches.')
        return
    rows = [{'Measure': label, 'His number': f(c['value']), 'League median': f(c['median']),
             'Percentile': c['pct']} for (_, label, f), c in zip(spec, comp)]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                 column_config={'Percentile': st.column_config.ProgressColumn(format='%.0f', min_value=0, max_value=100)})
    st.caption(f"Against the {max(c['n'] for c in comp)} {league_name} coaches with {cm.MIN_MATCHES}+ matches, "
               "each on his whole record there. Percentile = share of those coaches with a lower number: "
               "it says high or low, not good or bad.")


def _overview(rows, s, league_row, league_name, coaches, table):
    c = st.columns(3) + st.columns(3)
    c[0].metric('Points per match', _fmt(s['ppg']))
    c[1].metric('Expected points per match', _fmt(s['xpts_pg']),
                help='From the shots of both sides: every shot is an independent chance worth its xG.')
    c[2].metric('Points − expected', _signed(s['pts_minus_xpts'], '{:+.1f}'),
                help='Total over the matches with event data. Positive = results ran ahead of the chances.')
    c[3].metric('xG difference per match', _signed(s['xgd']))
    c[4].metric('Team level', _pm(s['adj_xgd'], s['adj_xgd_ci']),
                help='xG difference per match after removing each opponent\'s pre-match rating and the '
                     'home edge: what the side would do against an average team of the league on '
                     'neutral ground. ± is a 95% interval on the mean.')
    c[5].metric('Change since arrival', _pm(s['impact'], s['impact_ci']),
                help='xG difference per match against what the team\'s level on the day he arrived '
                     'predicted for the same fixtures.')
    _vs_league(table, s, [
        ('ppg', 'Points per match', _fmt),
        ('xpts_pg', 'Expected points per match', _fmt),
        ('pts_minus_xpts_pg', 'Points − expected, per match', _signed),
        ('xgd', 'xG difference per match', _signed),
        ('adj_xgd', 'Team level', _signed),
        ('impact', 'Change since arrival', _signed)], league_name)
    notes = [f"{s['event_matches']} of {s['matches']} matches have event data."]
    if league_row is not None and league_row.get('qualifies'):
        notes.append(f"Among {league_name} coaches with {cm.MIN_MATCHES}+ matches: team level "
                     f"{league_row['adj_xgd_pct']:.0f}th percentile, points per match "
                     f"{league_row['ppg_pct']:.0f}th.")
    if s['impact_n']:
        notes.append(f"Change since arrival covers {s['impact_n']} matches; it needs "
                     f"{10}+ earlier rated matches of the team in the same competition.")
    else:
        notes.append('No change-since-arrival figure: the team had too few earlier matches in the data '
                     'when he took over.')
    st.caption(' '.join(notes))
    st.caption('Read with care: the squad he was given, summer turnover and plain regression to the mean '
               'all sit inside these numbers. Team level describes the team under him; it is not a '
               'pure coaching effect.')

    ev = rows[rows.has_events].sort_values('date')
    if len(ev) >= 3:
        x = np.arange(1, len(ev) + 1)
        hover = [f"{d:%d %b %Y} · {t} v {o} ({'H' if sd == 'home' else 'A'}) {int(gf)}-{int(ga)}"
                 for d, t, o, sd, gf, ga in zip(ev.date, ev.team, ev.opponent, ev.side, ev.gf, ev.ga)]
        left, right = st.columns(2)
        fig = go.Figure()
        fig.add_scatter(x=x, y=ev.xpts.cumsum(), name='Expected points', mode='lines',
                        line=dict(color=theme.CONTEXT, width=2, dash='dot'), text=hover,
                        hovertemplate='%{text}<br>expected %{y:.1f}<extra></extra>')
        fig.add_scatter(x=x, y=ev.points.cumsum(), name='Points', mode='lines',
                        line=dict(color=theme.FOCUS, width=2.5), text=hover,
                        hovertemplate='%{text}<br>points %{y:.0f}<extra></extra>')
        _layout(fig, title='Points against expected points', xaxis_title='Match',
                legend=dict(orientation='h', y=1.12, x=0.45))
        left.plotly_chart(fig, use_container_width=True, config=_CFG)

        roll = ev.xgd.rolling(8, min_periods=4).mean()
        fig2 = go.Figure()
        fig2.add_bar(x=x, y=ev.xgd, name='Match xG difference', text=hover,
                     marker_color=np.where(ev.xgd >= 0, theme.FOCUS, theme.OPPONENT), opacity=0.35,
                     hovertemplate='%{text}<br>xGD %{y:+.2f}<extra></extra>')
        fig2.add_scatter(x=x, y=roll, name='8-match average', mode='lines',
                         line=dict(color=theme.FIGURE_INK, width=2.5), hoverinfo='skip')
        _layout(fig2, title='xG difference by match', xaxis_title='Match', showlegend=False)
        right.plotly_chart(fig2, use_container_width=True, config=_CFG)


def _spell_table(sp):
    if sp.empty:
        return
    t = pd.DataFrame({
        'Team': sp.team, 'From': sp['from'].dt.strftime('%b %Y'), 'To': sp['to'].dt.strftime('%b %Y'),
        'Matches': sp.matches, 'W-D-L': [f'{w}-{d}-{l}' for w, d, l in zip(sp.wins, sp.draws, sp.losses)],
        'Pts/match': sp.ppg.round(2), 'xPts/match': sp.xpts_pg.round(2),
        'xGD/match': sp.xgd.round(2), 'Team level': sp.adj_xgd.round(2),
        'Change since arrival': sp.impact.round(2),
    })
    st.dataframe(t, hide_index=True, use_container_width=True)


def _style(s, league_row, table, coaches, coach_id, league_name):
    if league_row is None or not league_row.get('qualifies'):
        st.info(f'Style percentiles need {cm.MIN_MATCHES}+ matches with event data in {league_name}; '
                'the raw numbers are below.')
    else:
        keys = list(cm.STYLE_METRICS)[::-1]
        pct = [league_row[k + '_pct'] for k in keys]
        shown = []
        for k in keys:
            v = league_row['ppda'] if k == 'press_intensity' else league_row[k]
            shown.append(_fmt(v, cm.STYLE_METRICS[k][2]))
        fig = go.Figure(go.Bar(
            x=pct, y=[cm.STYLE_METRICS[k][0] for k in keys], orientation='h',
            marker_color=theme.FOCUS, text=[f'{v}  ·  {p:.0f}' for v, p in zip(shown, pct)],
            textposition='outside', cliponaxis=False,
            customdata=[cm.STYLE_METRICS[k][1] for k in keys],
            hovertemplate='%{y}<br>percentile %{x:.0f} — higher = %{customdata}<extra></extra>'))
        fig.add_vline(x=50, line_color=theme.CONTEXT, line_dash='dot')
        _layout(fig, height=440, title=f'Style against {league_name} coaches (percentile)')
        fig.update_xaxes(range=[0, 118], tickvals=[0, 25, 50, 75, 100])
        st.plotly_chart(fig, use_container_width=True, config=_CFG)
        st.caption(f"His whole record in {league_name} against the {int(table.qualifies.sum())} coaches with "
                   f"{cm.MIN_MATCHES}+ matches there. A bar says how he plays, not how well. Style follows the "
                   "scoreline and the squad: a side that leads often sits deeper whatever the coach wants. "
                   "Set-piece share of xG does not agree between a coach's odd and even matches — read it as "
                   "what happened, not as a habit.")
        sim = cm.similar_coaches(table, coach_id)
        if not sim.empty:
            st.markdown('**Closest styles**')
            cols = st.columns(len(sim))
            for col, (_, r) in zip(cols, sim.iterrows()):
                if col.button(f"{coach_name(coaches, r.coachId)}\n\n{r.last_team} · {int(r.event_matches)} matches",
                              key=f'coach_sim_{int(r.coachId)}', use_container_width=True):
                    st.session_state[SEED_KEY] = int(r.coachId)
                    st.rerun()
    f = s['formations']
    if f:
        total = sum(f.values())
        top = sorted(f.items(), key=lambda kv: -kv[1])[:6]
        st.markdown('**Starting formations** — ' + ' · '.join(f'{k} {v / total:.0%}' for k, v in top))


def _in_game(s, rel, table, league_name):
    c = st.columns(4)
    c[0].metric('Substitutions per match', _fmt(s['subs_used'], '{:.1f}'))
    c[1].metric('First substitution', _fmt(s['first_sub_min'], "{:.0f}'"))
    c[2].metric('Substitutions before 60\'', _fmt(s['subs_before_60_share'], '{:.0%}'))
    c[3].metric('Formation changed in-match', _fmt(s['in_match_switch_share'], '{:.0%}'),
                help='Share of matches in which the side used more than one formation.')
    st.caption(f"Substitution minutes are the recorded ones ({s['sub_matches']} matches with a line-up sheet).")
    _vs_league(table, s, [
        ('subs_used', 'Substitutions per match', lambda v: _fmt(v, '{:.1f}')),
        ('first_sub_min', 'First substitution (minute)', lambda v: _fmt(v, '{:.0f}')),
        ('subs_before_60_share', "Substitutions before 60'", lambda v: _fmt(v, '{:.0%}')),
        ('in_match_switch_share', 'Formation changed in-match', lambda v: _fmt(v, '{:.0%}')),
        ('ppg_after_trailing', 'Points per match after trailing', _fmt),
        ('led_not_won_share', 'Leads not won', lambda v: _fmt(v, '{:.0%}')),
        ('xgd90_vs_exp_lead', 'xG difference vs expected, leading', _signed),
        ('xgd90_vs_exp_draw', 'xG difference vs expected, level', _signed),
        ('xgd90_vs_exp_trail', 'xG difference vs expected, trailing', _signed)], league_name)
    if not s['state_matches']:
        st.info('No matches with a clean goal timeline in this scope.')
        return
    keys = ('lead', 'draw', 'trail')
    t = pd.DataFrame({
        'Game state': ['Leading', 'Level', 'Trailing'],
        'Share of minutes': [_fmt(s[f'min_share_{k}'], '{:.0%}') for k in keys],
        'Expected share': [_fmt(s.get(f'exp_min_share_{k}'), '{:.0%}') for k in keys],
        'xG difference per 90': [_signed(s[f'xgd90_{k}']) for k in keys],
        'Expected': [_signed(s.get(f'exp_xgd90_{k}')) for k in keys],
        'vs expected': [_signed(s.get(f'xgd90_vs_exp_{k}')) for k in keys],
    })
    left, right = st.columns([3, 2])
    left.dataframe(t, hide_index=True, use_container_width=True)
    right.metric('Points per match after trailing', _fmt(s['ppg_after_trailing']),
                 help=f"{s['trailed_n']} matches in which the side was behind at some point.")
    right.metric('Leads not won', f"{s['led_not_won']} of {s['led_n']}")
    st.caption(f"Game state over {s['state_matches']} matches whose event goals match the final score "
               "(own goals are not shot events, so those matches are left out). Every side does better "
               "on xG when trailing, so read 'vs expected', not the raw number.")
    if s.get('state_exp_matches'):
        st.caption(f"Expected = what a side of that strength, against those opponents and at those venues, "
                   f"typically does in that game state: a straight-line fit on the pre-match rating gap and "
                   f"home/away, fitted only on the league's earlier months. The two expected columns and "
                   f"'vs expected' cover the {s['state_exp_matches']} of {s['state_matches']} matches that "
                   "have an expectation; the two actual columns cover all of them, so they are not exactly "
                   "like for like when those counts differ.")
    else:
        st.caption('No expected numbers yet: they need about 150 earlier matches with a clean goal timeline in the league.')
    st.caption("How stable these are as coach traits (odd vs even matches): first-substitution minute "
               f"{_stable(rel, 'first_sub_min')}, substitutions before 60' {_stable(rel, 'subs_before_60_share')}, "
               f"points after trailing {_stable(rel, 'ppg_after_trailing')}, xG difference when trailing "
               f"{_stable(rel, 'xgd90_trail')}; xG difference vs expected when leading "
               f"{_stable(rel, 'xgd90_vs_exp_lead')}, level {_stable(rel, 'xgd90_vs_exp_draw')}, trailing "
               f"{_stable(rel, 'xgd90_vs_exp_trail')}. Read the noisy ones as what happened, not as how he manages.")


def _squad(app, s, cp, coach_id, table, league_name):
    c = st.columns(4)
    c[0].metric('Line-up changes per match', _fmt(s['xi_changes'], '{:.1f}'),
                help='Starters changed from the team\'s previous match of the same season.')
    c[1].metric('Players used', str(s['players_used']))
    c[2].metric('Under-21 share of minutes', _fmt(s.get('u21_share'), '{:.0%}'))
    c[3].metric('Most-used formation', (s['main_formation'] or '–') + _fmt(s['main_formation_share'], ' · {:.0%}', ''),
                help='Share of matches started in it.')
    _vs_league(table, s, [
        ('xi_changes', 'Line-up changes per match', lambda v: _fmt(v, '{:.1f}')),
        ('u21_share', 'Under-21 share of minutes', lambda v: _fmt(v, '{:.0%}')),
        ('main_formation_share', 'Share of matches in his most-used formation', lambda v: _fmt(v, '{:.0%}'))],
        league_name)
    st.caption('Players used is not compared: it grows with the number of matches.')
    details = app.load_player_details()

    def _name(pid):
        try:
            return str(details.at[int(pid), 'shortName'])
        except Exception:
            return f'#{int(pid)}'

    dev, summ = cm.player_development(cp, coach_id)
    st.markdown('**Players against their own earlier record**')
    if not summ['n']:
        st.caption(f'No player has {cm.MIN_PLAYER_MINUTES}+ minutes both under him and before him.')
    else:
        st.caption(f"{summ['improved']} of {summ['n']} players with {cm.MIN_PLAYER_MINUTES}+ minutes on each side "
                   f"produced more on-ball value per 90 under him than in their earlier matches "
                   f"(minutes-weighted change {summ['mean_change']:+.3f}). The earlier record is only matches "
                   "played before their first game under him. Age, role and team strength are not removed.")
        t = pd.DataFrame({'Player': dev.playerId.map(_name), 'Team': dev.team,
                          'Minutes under him': dev.minutes.round(0).astype(int),
                          'OBV/90 under him': dev.obv90.round(3), 'OBV/90 before': dev.prior_obv90.round(3),
                          'Change': dev.change.round(3),
                          'Minutes before': dev.prior_minutes.round(0).astype(int)})
        st.dataframe(t, hide_index=True, use_container_width=True, height=min(38 * (len(t) + 1), 420))
    used = cp[cp.coachId == coach_id].sort_values('minutes', ascending=False).head(15)
    if not used.empty:
        st.markdown('**Most-used players**')
        t = pd.DataFrame({'Player': used.playerId.map(_name), 'Team': used.team,
                          'Minutes': used.minutes.round(0).astype(int), 'Matches': used.matches,
                          'Starts': used.starts.astype(int),
                          'Matches played while there': (used.matches / used.team_matches_in_span).map('{:.0%}'.format)})
        st.dataframe(t, hide_index=True, use_container_width=True)


def _matches(rows):
    r = rows.sort_values('date', ascending=False)
    t = pd.DataFrame({
        'Date': r.date.dt.strftime('%Y-%m-%d'), 'Team': r.team,
        'Opponent': r.opponent + np.where(r.side == 'home', ' (H)', ' (A)'),
        'Score': [f'{int(a)}-{int(b)}' for a, b in zip(r.gf, r.ga)],
        'xG': r.xg.round(2), 'xG against': r.opp_xg.round(2), 'xPts': r.xpts.round(2),
        'Formation': r.formation, 'First sub': r.first_sub_min, 'XI changes': r.xi_changes,
    })
    st.dataframe(t, hide_index=True, use_container_width=True, height=520)


def _flexibility(s, league_row, table, rel, vectors, persistence, coach_id, league_name):
    if league_row is None or not league_row.get('qualifies'):
        st.info(f'Needs {cm.MIN_MATCHES}+ matches with event data in {league_name}.')
        return
    fl = league_row.get('fluidity')
    word = '–' if pd.isna(fl) else 'Settled' if fl < 33 else 'Fluid' if fl > 67 else 'In between'
    c = st.columns(3)
    c[0].metric('Settled ↔ fluid', word,
                help='From the mean of the four formation percentiles below. Opponent adaptation is shown '
                     'but left out of the score: it is too noisy to be a trait.')
    c[1].metric('Fluidity percentile', _fmt(fl, '{:.0f}'))
    c[2].metric('Most-used formation', (s['main_formation'] or '–') + _fmt(s['main_formation_share'], ' · {:.0%}', ''),
                help='Share of matches started in it.')
    fmt = {'eff_formations': '{:.1f}', 'opp_adapt': '{:.2f}'}
    rows = []
    for k in cm.FLEX_KEYS:
        label, what = cm.FLEX_LABELS[k]
        rows.append({'Measure': label, 'What it counts': what,
                     'His number': _fmt(league_row.get(k), fmt.get(k, '{:.0%}')),
                     'Percentile': league_row.get(k + '_pct'),
                     'How stable the measure is': _stable(rel, k)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                 column_config={'Percentile': st.column_config.ProgressColumn(format='%.0f', min_value=0, max_value=100)})
    st.caption(f"Percentiles are among {league_name} coaches with {cm.MIN_MATCHES}+ matches; high = more fluid. "
               "'How stable' is the agreement between a coach's odd and even matches (1 = a fixed trait, "
               "0 = noise). Wyscout names close shapes separately (4-2-3-1, 4-4-1-1, 4-3-3), so formation "
               "variety runs high for everyone — the back-line switch is the cleaner signal. Injuries and "
               "suspensions also change shapes; this cannot tell choice from necessity.")
    st.caption("Tested and left out: how much his style numbers move from one match to the next "
               "(the measure in Hudl's 2025 'Manager Bounce' study). In this league it does not agree "
               "between a coach's odd and even matches at all, so it is noise here, not a trait.")
    st.markdown('**Does his style travel between clubs?**')
    d = cm.coach_club_distances(vectors, coach_id)
    ref = persistence or {}
    def _ref(key):
        v = ref.get(key)
        return '–' if not v or pd.isna(v[0]) else f'{v[0]:.2f} ({v[1]} pairs)'
    if d.empty:
        st.caption(f'He has one club with {cm.MIN_MATCHES}+ matches in {league_name}, so there is nothing to compare.')
    else:
        d['Style distance'] = d['Style distance'].round(2)
        st.dataframe(d, hide_index=True, use_container_width=True)
    st.caption(f"Style distance: 0 = identical context-adjusted style. {league_name} medians — the same coach at "
               f"two clubs {_ref('same_coach_other_club')}; two coaches at the same club "
               f"{_ref('same_club_other_coach')}; unrelated coach and club {_ref('unrelated')}. "
               "A coach's style carries to his next club slightly more than a club's style survives a "
               "change of coach, and both are only a little closer than strangers.")
    f = s['formations']
    if f:
        total = sum(f.values())
        top = sorted(f.items(), key=lambda kv: -kv[1])[:8]
        st.markdown('**Starting formations** — ' + ' · '.join(f'{k} {v / total:.0%}' for k, v in top))


def _young(app, rows, s, league_row, rel, cp, pairs, coach_id, league_name, table):
    cut = st.radio('Young means', [21, 23], format_func=lambda c: f'Under {c}', horizontal=True,
                   key='coach_profile_age_cut')
    k = f'u{cut}'
    if not s.get(f'{k}_matches'):
        st.info('No exact minutes and line-up sheets for these matches (exact minutes exist for Liga 3 only).')
        return
    st.markdown('**Does he trust them?**')
    c = st.columns(5)
    c[0].metric('Share of minutes', _fmt(s[f'{k}_minutes_share'], '{:.0%}'))
    c[1].metric('Share of matchday squad', _fmt(s[f'{k}_named_share'], '{:.0%}'),
                help='Places among starters + bench on the line-up sheet.')
    c[2].metric('Trust ratio', _fmt(s[f'{k}_trust'], '{:.2f}'),
                help='Share of minutes ÷ share of matchday-squad places. 1.00 = young players play in '
                     'proportion to their places; below 1 = they are named but used less.')
    c[3].metric('Share of starts', _fmt(s[f'{k}_start_share'], '{:.0%}'))
    c[4].metric('Age on pitch vs squad', _signed(s.get('age_gap'), '{:+.1f} y'),
                help=f"Minutes-weighted age on the pitch {_fmt(s.get('age_on_pitch'), '{:.1f}')} against the "
                     f"average age of the matchday squad {_fmt(s.get('age_of_squad'), '{:.1f}')}. "
                     'Negative = he plays younger than the squad he names.')
    _vs_league(table, s, [
        (f'{k}_minutes_share', 'Share of minutes', lambda v: _fmt(v, '{:.0%}')),
        (f'{k}_named_share', 'Share of matchday squad', lambda v: _fmt(v, '{:.0%}')),
        (f'{k}_trust', 'Trust ratio', _fmt),
        (f'{k}_start_share', 'Share of starts', lambda v: _fmt(v, '{:.0%}')),
        ('age_gap', 'Age on pitch vs squad (years)', lambda v: _signed(v, '{:+.1f}'))], league_name)
    note = [f"{s[f'{k}_matches']} matches with exact minutes and a line-up sheet; "
            f"{_fmt(s[f'{k}_named_per_match'], '{:.1f}')} under-{cut} players named per match."]
    if league_row is not None and league_row.get('qualifies') and not pd.isna(league_row.get(f'{k}_trust_pct', np.nan)):
        note.append(f"Whole record in {league_name}: trust ratio {league_row[f'{k}_trust_pct']:.0f}th percentile, "
                    f"share of minutes {league_row[f'{k}_minutes_share_pct']:.0f}th. "
                    f"The trust ratio is a {_stable(rel, f'{k}_trust')} measure.")
    note.append("The ratio removes how many young players the club gave him, which the raw share does not: "
                "a B team is all young players (ratio ≈ 1 by construction). It still cannot see quality — "
                "a young player may sit because the senior ahead of him is better.")
    st.caption(' '.join(note))

    details = app.load_player_details()

    def _name(pid):
        try:
            return str(details.at[int(pid), 'shortName'])
        except Exception:
            return f'#{int(pid)}'

    spells_here = set(zip(rows.coachId, rows.teamId))
    mine = cp[(cp.coachId == coach_id) & (cp.age_first < cut)]
    mine = mine[[(c_, t_) in spells_here for c_, t_ in zip(mine.coachId, mine.teamId)]]
    mine = mine.sort_values('minutes', ascending=False)
    if not mine.empty:
        debut = (mine.prior_all_minutes == 0) & (mine.first_under_coach >= pd.Timestamp('2022-07-01'))
        t = pd.DataFrame({
            'Player': mine.playerId.map(_name), 'Team': mine.team,
            'Age at first match': mine.age_first.round(1),
            'Minutes': mine.minutes.round(0).astype(int), 'Matches': mine.matches,
            'Starts': mine.starts.astype(int), 'Named in squad': mine.named,
            'Played when named': mine.named_played / mine.named,
            'First Liga 3 match in the data': np.where(debut, 'yes', ''),
        })
        st.dataframe(t, hide_index=True, use_container_width=True, height=min(38 * (len(t) + 1), 400),
                     column_config={'Played when named': st.column_config.ProgressColumn(
                         format='percent', min_value=0, max_value=1)})
        st.caption(f"{int(debut.sum())} of these players made their first appearance in our data under him "
                   "(counted from 2022/23, after one season of history). A young player named but never used "
                   "is in the squad share above, not in this table.")

    st.markdown('**Do they get better under him?**')
    dev, summ = cm.youth_development(pairs, coach_id, cut)
    if not summ['n']:
        st.caption(f'No under-{cut} player has {cm.MIN_PLAYER_MINUTES}+ minutes both before him and under him.')
    else:
        line = (f"{summ['improved']} of {summ['n']} produced more on-ball value per 90 under him than in their "
                f"earlier matches (minutes-weighted change {summ['mean_change']:+.3f}).")
        if summ['n_expected']:
            line += (f" Against what other players of their age band did under other coaches before: "
                     f"{summ['beat_expected']} of {summ['n_expected']} beat it "
                     f"(average {summ['mean_vs_expected']:+.3f}).")
        st.caption(line)
        t = pd.DataFrame({'Player': dev.playerId.map(_name), 'Team': dev.team,
                          'Age at first match': dev.age_first.round(1),
                          'Minutes under him': dev.minutes.round(0).astype(int),
                          'OBV/90 before': dev.prior_obv90.round(3), 'OBV/90 under him': dev.obv90.round(3),
                          'Change': dev.change.round(3), 'Expected for age': dev.expected.round(3),
                          'Against expected': dev.vs_expected.round(3)})
        st.dataframe(t, hide_index=True, use_container_width=True)
    # halves are split over ALL his minutes under this coach, so the table
    # always uses the coach's whole record for the players in this scope
    prog = cm.youth_progress(cp, coach_id, cut)
    if not prog.empty:
        prog = prog[prog.playerId.isin(set(mine.playerId))]
    if not prog.empty:
        st.caption(f"While playing for him — first half against second half of each player's minutes under him "
                   f"({cm.MIN_PROGRESS_MINUTES}+ minutes, all his clubs under this coach together): "
                   f"{int((prog.progress > 0).sum())} of {len(prog)} did more "
                   "in the second half. Across the league the average change is about zero at every age.")
        t = pd.DataFrame({'Player': prog.playerId.map(_name), 'Team': prog.team,
                          'Age at first match': prog.age_first.round(1),
                          'Minutes under him': prog.minutes.round(0).astype(int),
                          'OBV/90 first half': prog.first_half.round(3),
                          'OBV/90 second half': prog.second_half.round(3),
                          'Change': prog.progress.round(3)})
        st.dataframe(t, hide_index=True, use_container_width=True)
    st.warning("Treat this as a list of cases, not a rating of the coach. Across the league the typical coach "
               f"has two under-23 players with enough minutes on both sides, and when coaches' players "
               "(all ages) are split at random into two halves, the two halves barely agree (correlation "
               "0.13 over 57 coaches, 2026-11; 0.22 for the first-half / second-half view). On-ball value "
               "also moves with role and team strength. "
               "'Before' is only matches played before his first game under this coach; 'expected for age' "
               "uses only other coaches' spells that had already ended.")


# ──────────────────────────────── page ────────────────────────────────
def render():
    app = _app()
    COMPETITIONS = app.COMPETITIONS
    league_selector = app.league_selector

    st.header('Coach Profile')
    sig = tables_signature()
    data = load_tables(sig)
    if data is None:
        st.info('Coach data has not been built yet (`python models/coaches/build_coaches.py`).')
        return
    tm, coaches, cp = data
    comp_ids = [int(c) for c in league_selector('coach_profile')]
    comp_id = comp_ids[0]
    # A coach opened from another page may only have matches in the other
    # league. The league bar is already drawn, so the page follows the coach
    # (and says so) until the bar is changed.
    seed = st.session_state.pop(SEED_KEY, None)
    if seed is not None:
        seed_comps = tm.loc[tm.coachId == int(seed)].sort_values('date').competitionId
        if len(seed_comps) and comp_id not in set(seed_comps):
            st.session_state['coach_profile_follow'] = (int(seed_comps.iloc[-1]), comp_id)
        else:
            st.session_state.pop('coach_profile_follow', None)
    follow = st.session_state.get('coach_profile_follow')
    if follow and follow[1] == comp_id and follow[0] in COMPETITIONS:
        comp_id = follow[0]
        st.info(f"Showing {COMPETITIONS[comp_id]['name']}: this coach has no matches in the league "
                "selected in the sidebar.")
    else:
        st.session_state.pop('coach_profile_follow', None)
    league_name = COMPETITIONS[comp_id]['name']
    scope = tm[(tm.competitionId == comp_id) & (tm.coachId > 0)]
    if scope.empty:
        st.info(f'No coach data for {league_name}.')
        return
    table = league_table(sig, comp_id)

    last = scope.sort_values('date').groupby('coachId').agg(team=('team', 'last'), last=('date', 'max'),
                                                           n=('matchId', 'size'))
    last = last.sort_values(['last', 'n'], ascending=False)
    ids = [int(i) for i in last.index]
    labels = {i: f"{coach_name(coaches, i)} ({last.at[i, 'team']}, {int(last.at[i, 'n'])} matches)" for i in ids}
    if seed is not None and int(seed) in labels:
        st.session_state['coach_profile_selector'] = int(seed)
    elif st.session_state.get('coach_profile_selector') not in labels:
        # default: our own coach when he is in the list
        ours = scope[scope.team == app.OUR_TEAM].sort_values('date')
        st.session_state['coach_profile_selector'] = int(ours.coachId.iloc[-1]) if len(ours) else ids[0]
    coach_id = st.selectbox('Coach', ids, format_func=labels.get, key='coach_profile_selector',
                            help='Most recent match first.')

    sp = cm.spells(scope, coach_id)
    options = ['Whole record'] + [f"{r.team} · {r['from']:%b %Y} – {r['to']:%b %Y}" for _, r in sp.iterrows()]
    if st.session_state.get('coach_profile_scope') not in options:
        st.session_state['coach_profile_scope'] = options[0]
    choice = st.radio('Scope', options, horizontal=True, key='coach_profile_scope',
                      label_visibility='collapsed') if len(options) > 2 else options[0]
    rows = scope[scope.coachId == coach_id]
    if choice != options[0]:
        rows = rows[rows.spell == sp.iloc[options.index(choice) - 1].spell]
    s = cm.summarise(rows)

    info = coaches.loc[coach_id] if coach_id in coaches.index else None
    born = pd.to_datetime(info.birthDate, errors='coerce') if info is not None else pd.NaT
    age = int((pd.Timestamp.today() - born).days // 365.25) if not pd.isna(born) else None
    latest = rows.sort_values('date').iloc[-1]
    with st.container(key=theme.metric_fit_key('coach_hdr', latest.team,
                                               (info.nationality if info is not None else '') or '–')):
        h = st.columns(5)
        h[0].metric('Latest team', latest.team)
        h[1].metric('Age', str(age) if age else '–')
        h[2].metric('Nationality', (info.nationality if info is not None else None) or '–')
        h[3].metric('Matches', str(s['matches']))
        h[4].metric('W-D-L', f"{s['wins']}-{s['draws']}-{s['losses']}")
    st.caption(f"{league_name} matches in the data, {rows.date.min():%b %Y} to {rows.date.max():%b %Y}. "
               "Matches he coached in other leagues are not here.")

    lr = table[table.coachId == coach_id]
    league_row = lr.iloc[0].to_dict() if len(lr) else None
    sections = ['Overview', 'Style', 'Flexibility', 'In-game', 'Young players', 'Squad & players', 'Matches']
    active = st.radio('Coach section', sections, horizontal=True, label_visibility='collapsed',
                      key='coach_profile_section')
    if active == 'Overview':
        _overview(rows, s, league_row, league_name, coaches, table)
        st.markdown('**Spells**')
        _spell_table(sp)
    elif active == 'Style':
        _style(s, league_row, table, coaches, coach_id, league_name)
    elif active == 'Flexibility':
        vectors, persistence = league_club_vectors(sig, comp_id)
        _flexibility(s, league_row, table, league_reliability(sig, comp_id), vectors, persistence,
                     coach_id, league_name)
    elif active == 'In-game':
        _in_game(s, league_reliability(sig, comp_id), table, league_name)
    elif active == 'Young players':
        _young(app, rows, s, league_row, league_reliability(sig, comp_id), cp,
               league_development_pairs(sig), coach_id, league_name, table)
    elif active == 'Squad & players':
        _squad(app, s, cp, coach_id, table, league_name)
    else:
        _matches(rows)


# ───────────────────── 'Coached by' (Player Profile) ─────────────────────
def render_coached_by(player_id):
    """Every coach the player has played under in the data, newest first."""
    data = load_tables(tables_signature())
    st.subheader('Coached by')
    if data is None:
        st.info('Coach data has not been built yet.')
        return
    tm, coaches, cp = data
    r = cm.coached_by(cp, tm, int(player_id))
    if r.empty:
        st.info('No matches with a known coach for this player.')
        return
    t = pd.DataFrame({
        'Coach': [coach_name(coaches, c) for c in r.coachId], 'Team': r.team,
        'From': r.first_date.dt.strftime('%b %Y'), 'To': r.last_date.dt.strftime('%b %Y'),
        'Matches': r.matches, 'Starts': r.starts.astype(int),
        'Minutes': r.minutes.round(0).astype(int),
        'Started (of coach\'s matches)': r.start_share,
        'Played in': r.played_share,
        'OBV/90': r.obv90.round(3),
        'Team pts/match with him': r.team_ppg_with.round(2),
    })
    pct = st.column_config.ProgressColumn(format='percent', min_value=0, max_value=1)
    ev = st.dataframe(t, hide_index=True, use_container_width=True, on_select='rerun',
                      selection_mode='single-row', key=f'coached_by_{int(player_id)}',
                      column_config={'Started (of coach\'s matches)': pct, 'Played in': pct})
    st.caption("Exact minutes. 'Started' and 'Played in' count the coach's matches between the player's "
               "first and last appearance under him — how much that coach relied on him while he was "
               "available. OBV/90 is on-ball value per 90 in those matches; under 450 minutes it is noise. "
               "Select a row to open the coach.")
    sel = getattr(getattr(ev, 'selection', None), 'rows', None) or []
    if sel:
        row = r.iloc[sel[0]]
        navigation.go_to(PAGE, **{SEED_KEY: int(row.coachId)})
