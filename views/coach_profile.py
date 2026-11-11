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
def _overview(rows, s, league_row, league_name, coaches):
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
                   "scoreline and the squad: a side that leads often sits deeper whatever the coach wants.")
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


def _in_game(s):
    c = st.columns(4)
    c[0].metric('Substitutions per match', _fmt(s['subs_used'], '{:.1f}'))
    c[1].metric('First substitution', _fmt(s['first_sub_min'], "{:.0f}'"))
    c[2].metric('Substitutions before 60\'', _fmt(s['subs_before_60_share'], '{:.0%}'))
    c[3].metric('Formation changed in-match', _fmt(s['in_match_switch_share'], '{:.0%}'),
                help='Share of matches in which the side used more than one formation.')
    st.caption(f"Substitution minutes are the recorded ones ({s['sub_matches']} matches with a line-up sheet).")
    if not s['state_matches']:
        st.info('No matches with a clean goal timeline in this scope.')
        return
    t = pd.DataFrame({
        'Game state': ['Leading', 'Level', 'Trailing'],
        'Share of minutes': [_fmt(s[f'min_share_{k}'], '{:.0%}') for k in ('lead', 'draw', 'trail')],
        'xG difference per 90': [_signed(s[f'xgd90_{k}']) for k in ('lead', 'draw', 'trail')],
    })
    left, right = st.columns([3, 2])
    left.dataframe(t, hide_index=True, use_container_width=True)
    right.metric('Points per match after trailing', _fmt(s['ppg_after_trailing']),
                 help=f"{s['trailed_n']} matches in which the side was behind at some point.")
    right.metric('Leads not won', f"{s['led_not_won']} of {s['led_n']}")
    st.caption(f"Game state over {s['state_matches']} matches whose event goals match the final score "
               "(own goals are not shot events, so those matches are left out). Every side does better "
               "on xG when trailing — compare coaches on the same row, not rows with each other.")


def _squad(app, s, cp, coach_id):
    c = st.columns(4)
    c[0].metric('Line-up changes per match', _fmt(s['xi_changes'], '{:.1f}'),
                help='Starters changed from the team\'s previous match of the same season.')
    c[1].metric('Players used', str(s['players_used']))
    c[2].metric('Under-21 share of minutes', _fmt(s.get('u21_share'), '{:.0%}'))
    c[3].metric('Most-used formation', s['main_formation'] or '–',
                delta=_fmt(s['main_formation_share'], '{:.0%} of matches'), delta_color='off')
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
    sections = ['Overview', 'Style', 'In-game', 'Squad & players', 'Matches']
    active = st.radio('Coach section', sections, horizontal=True, label_visibility='collapsed',
                      key='coach_profile_section')
    if active == 'Overview':
        _overview(rows, s, league_row, league_name, coaches)
        st.markdown('**Spells**')
        _spell_table(sp)
    elif active == 'Style':
        _style(s, league_row, table, coaches, coach_id, league_name)
    elif active == 'In-game':
        _in_game(s)
    elif active == 'Squad & players':
        _squad(app, s, cp, coach_id)
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
