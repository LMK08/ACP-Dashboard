"""Coach Comparison: two coaches of one league side by side.

Every number is the one the Coach Profile shows (models/coaches/
coach_metrics.py: `summarise` on each coach's whole record in the league,
`league_comparison` for the median and the percentile), so the two pages
cannot disagree. The same rules apply: a percentile says high or low, not
good or bad; results sit on the squad he was given; measures read as a
habit are shown only when they repeat (the Tendencies list); the in-game
record describes what happened and does not predict.
"""
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import navigation
import theme
from models.coaches import coach_metrics as cm
from views import coach_profile as cp_view

PAGE = 'Coach Comparison'
SEED_A, SEED_B = 'coach_compare_seed_a', 'coach_compare_seed_b'
KEY_A, KEY_B = 'coach_comparison_a', 'coach_comparison_b'
SECTIONS = ['Results', 'Style', 'Set-up & tendencies', 'In-game record', 'Squad & young players']

_fmt, _signed = cp_view._fmt, cp_view._signed
_pct1 = lambda v: _fmt(v, '{:.0%}')        # noqa: E731
_one = lambda v: _fmt(v, '{:.1f}')         # noqa: E731

RESULTS = [
    ('ppg', 'Points per match', _fmt),
    ('xpts_pg', 'Expected points per match', _fmt),
    ('pts_minus_xpts_pg', 'Points − expected, per match', _signed),
    ('xgd', 'xG difference per match', _signed),
    ('adj_xgd', 'Team level', _signed),
    ('impact', 'Change since arrival', _signed)]
SETUP = [
    ('back3_share', 'Matches started with a back three', _pct1),
    ('main_formation_share', 'Share of matches in his most-used formation', _pct1),
    ('eff_formations', 'Formation variety (effective number)', _one),
    ('formation_change_rate', 'Formation changed from the previous match', _pct1),
    ('back_line_change_rate', 'Back line changed from the previous match', _pct1)]
IN_GAME = [
    ('first_sub_min', 'First substitution (minute)', lambda v: _fmt(v, '{:.0f}')),
    ('subs_before_60_share', "Substitutions before 60'", _pct1),
    ('ppg_after_trailing', 'Points per match after trailing', _fmt),
    ('led_not_won_share', 'Leads not won', _pct1),
    ('late_xgd', "xG difference after 75', per match", _signed),
    ('home_minus_away', 'xG difference per match, home minus away', _signed),
    ('xgd90_vs_exp_lead', 'xG difference vs expected, leading', _signed),
    ('xgd90_vs_exp_draw', 'xG difference vs expected, level', _signed),
    ('xgd90_vs_exp_trail', 'xG difference vs expected, trailing', _signed)]
SQUAD = [
    ('xi_changes', 'Line-up changes per match', _one),
    ('core_xi_share', 'Fixed eleven', _pct1),
    ('u21_share', 'Under-21 share of minutes', _pct1),
    ('u21_trust', 'Under-21 trust ratio', _fmt),
    ('u23_trust', 'Under-23 trust ratio', _fmt)]


def _app():
    return sys.modules['__main__']


def comparison_rows(table, sa, sb, spec, names):
    """One row per measure: both coaches' numbers, the league median and each
    coach's percentile among the league's qualifying coaches."""
    keys = [k for k, _, _ in spec]
    ca, cb = cm.league_comparison(table, sa, keys), cm.league_comparison(table, sb, keys)
    rows = []
    for (_, label, f), a, b in zip(spec, ca, cb):
        rows.append({'Measure': label, names[0]: f(a['value']), names[1]: f(b['value']),
                     'League median': f(a['median']),
                     f'{names[0]} percentile': a['pct'], f'{names[1]} percentile': b['pct']})
    n = max([c['n'] for c in ca] or [0])
    return pd.DataFrame(rows), n


def _table(table, sa, sb, spec, names, league_name, extra=None):
    df, n = comparison_rows(table, sa, sb, spec, names)
    if extra is not None:
        df[extra[0]] = extra[1]
    bar = st.column_config.ProgressColumn(format='%.0f', min_value=0, max_value=100)
    st.dataframe(df, hide_index=True, use_container_width=True,
                 column_config={f'{names[0]} percentile': bar, f'{names[1]} percentile': bar})
    if n < 5:
        st.caption(f'No league median or percentile: fewer than 5 {league_name} coaches have '
                   f'{cm.MIN_MATCHES}+ matches.')
    return n


def _pool_note(n, league_name):
    if n < 5:
        return ''
    return (f"Against the {n} {league_name} coaches with {cm.MIN_MATCHES}+ matches, each on his whole record "
            "there. Percentile = share of those coaches with a lower number: it says high or low, not good "
            "or bad.")


def _results(table, sa, sb, names, league_name):
    n = _table(table, sa, sb, RESULTS, names, league_name)
    st.caption(_pool_note(n, league_name))
    for name, s in zip(names, (sa, sb)):
        half = s.get('adj_xgd_ci')
        st.caption(f"{name}: team level {cp_view._pm(s['adj_xgd'], half)} (95% interval on the mean), "
                   f"{s['event_matches']} of {s['matches']} matches with event data"
                   + (f", change since arrival over {s['impact_n']} matches." if s.get('impact_n') else
                      ', no change-since-arrival figure (too few earlier matches of the team).'))
    lo = [s['adj_xgd'] - (s.get('adj_xgd_ci') or np.nan) for s in (sa, sb)]
    hi = [s['adj_xgd'] + (s.get('adj_xgd_ci') or np.nan) for s in (sa, sb)]
    if not any(pd.isna(v) for v in lo + hi) and max(lo) <= min(hi):
        st.caption('The two team-level intervals overlap: on this evidence the two sides cannot be told apart.')
    st.caption('Read with care: the squad each was given, summer turnover and plain regression to the mean '
               'all sit inside these numbers. Team level describes the team under him; it is not a pure '
               'coaching effect, and this table is not a ranking of the two coaches.')


def style_figure(row_a, row_b, names, league_name):
    """Both coaches' style percentiles on one chart (None when either lacks them)."""
    if row_a is None or row_b is None or not row_a.get('qualifies') or not row_b.get('qualifies'):
        return None
    keys = list(cm.STYLE_METRICS)[::-1]
    labels = [cm.STYLE_METRICS[k][0] for k in keys]
    fig = go.Figure()
    for row, name, colour in ((row_b, names[1], theme.COMPARE_B), (row_a, names[0], theme.COMPARE_A)):
        fig.add_bar(x=[row[k + '_pct'] for k in keys], y=labels, orientation='h', name=name,
                    marker_color=colour, customdata=[cm.STYLE_METRICS[k][1] for k in keys],
                    hovertemplate=name + '<br>%{y}<br>percentile %{x:.0f} — higher = %{customdata}<extra></extra>')
    fig.add_vline(x=50, line_color=theme.CONTEXT, line_dash='dot')
    cp_view._layout(fig, height=560, title=f'Style against {league_name} coaches (percentile)',
                    barmode='group', legend=dict(orientation='h', y=-0.06, x=0, traceorder='reversed'))
    fig.update_xaxes(range=[0, 100], tickvals=[0, 25, 50, 75, 100])
    return fig


def _style(table, sa, sb, row_a, row_b, names, league_name):
    fig = style_figure(row_a, row_b, names, league_name)
    if fig is None:
        st.info(f'The style chart needs {cm.MIN_MATCHES}+ matches with event data in {league_name} for both '
                'coaches; the raw numbers are below.')
    else:
        st.plotly_chart(fig, use_container_width=True, config=cp_view._CFG)
        gaps = sorted(((abs(row_a[k + '_pct'] - row_b[k + '_pct']), k) for k in cm.STYLE_METRICS), reverse=True)
        big = [f"{cm.STYLE_METRICS[k][0].lower()} ({row_a[k + '_pct']:.0f} v {row_b[k + '_pct']:.0f})"
               for g, k in gaps[:3] if g >= 20]
        if big:
            st.caption('Largest differences in percentile: ' + '; '.join(big) + '.')
    spec = [(k, v[0], (lambda x, f=v[2]: _fmt(x, f))) for k, v in cm.STYLE_METRICS.items()]
    df, n = comparison_rows(table, sa, sb, spec, names)
    # pressing: the PPDA number with the pressing percentile, exactly as the profile and the chart show it
    ppda, _ = comparison_rows(table, sa, sb, [('ppda', '', lambda x: _fmt(x, cm.STYLE_METRICS['press_intensity'][2]))], names)
    row = df.index[df['Measure'] == cm.STYLE_METRICS['press_intensity'][0]]
    for col in (names[0], names[1], 'League median'):
        df.loc[row, col] = ppda.iloc[0][col]
    bar = st.column_config.ProgressColumn(format='%.0f', min_value=0, max_value=100)
    st.dataframe(df, hide_index=True, use_container_width=True,
                 column_config={f'{names[0]} percentile': bar, f'{names[1]} percentile': bar})
    st.caption(_pool_note(n, league_name) + ' A bar says how he plays, not how well. Style follows the '
               'scoreline and the squad: a side that leads often sits deeper whatever the coach wants. '
               "Pressing is shown as PPDA (lower = more pressing) with the percentile of pressing "
               "intensity, so a high percentile means more pressing. Set-piece share of xG does not repeat between a coach's "
               "odd and even matches: read it as what happened, not as a habit.")


def _setup(table, sa, sb, names, league_name, rel):
    c = st.columns(2)
    for col, name, s in zip(c, names, (sa, sb)):
        f = s.get('formations') or {}
        total = sum(f.values())
        top = sorted(f.items(), key=lambda kv: -kv[1])[:4]
        col.markdown(f'**{name}** — ' + (' · '.join(f'{k} {v / total:.0%}' for k, v in top) if total else
                                         'no formation read'))
    st.markdown('**Set-up**')
    n = _table(table, sa, sb, SETUP, names, league_name,
               extra=('How stable the measure is', [cp_view._stable(rel, k) for k, _, _ in SETUP]))
    st.caption('Formations are our own read of where the starters touched the ball in open play, so this is '
               'the shape with the ball: a side with a high full-back can read as a back three.')
    st.markdown('**Tendencies**')
    spec = [(k, v[0], (lambda x, f=v[2]: _fmt(x, f))) for k, v in cm.TENDENCIES.items()]
    _table(table, sa, sb, spec, names, league_name,
           extra=('How stable the measure is', [cp_view._stable(rel, k) for k in cm.TENDENCIES]))
    st.caption(_pool_note(n, league_name) + " 'How stable' is the agreement between a coach's odd and even "
               'matches (1 = a fixed trait, 0 = noise).')


def _in_game(table, sa, sb, names, league_name):
    st.caption('A record of what happened in their matches. These numbers do not repeat from one half of a '
               "coach's matches to the other, so they describe; they do not predict.")
    n = _table(table, sa, sb, IN_GAME, names, league_name)
    st.caption(_pool_note(n, league_name))
    keys = ('lead', 'draw', 'trail')
    rows = []
    for label, k in zip(('Leading', 'Level', 'Trailing'), keys):
        row = {'Game state': label}
        for name, s in zip(names, (sa, sb)):
            row[f'{name}: share of minutes'] = _fmt(s.get(f'min_share_{k}'), '{:.0%}')
            row[f'{name}: xG difference per 90'] = _signed(s.get(f'xgd90_{k}'))
            row[f'{name}: vs expected'] = _signed(s.get(f'xgd90_vs_exp_{k}'))
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption(f"Game state over the matches whose event goals match the final score ({names[0]}: "
               f"{sa['state_matches']}, {names[1]}: {sb['state_matches']}). Every side does better on xG "
               "when trailing, so read 'vs expected', not the raw number.")


def _squad(table, sa, sb, names, league_name):
    c = st.columns(2)
    for col, name, s in zip(c, names, (sa, sb)):
        col.metric(f'{name}: players used', str(s['players_used']))
    n = _table(table, sa, sb, SQUAD, names, league_name)
    st.caption(_pool_note(n, league_name) + ' Players used is not compared: it grows with the number of '
               'matches. Fixed eleven is the share of all starts taken by his 11 most-used starters. Minutes '
               'shares and the trust ratios need exact minutes, which exist for Liga 3 only. '
               'A trust ratio above 1 means young players got more of the minutes than of the matchday-squad '
               'places.')


def render():
    app = _app()
    COMPETITIONS = app.COMPETITIONS
    st.header('Coach Comparison')
    sig = cp_view.tables_signature()
    data = cp_view.load_tables(sig)
    if data is None:
        st.info('Coach data has not been built yet (`python models/coaches/build_coaches.py`).')
        return
    tm, coaches, _ = data
    comp_id = [int(c) for c in app.league_selector('coach_comparison')][0]
    seed_a, seed_b = st.session_state.pop(SEED_A, None), st.session_state.pop(SEED_B, None)
    # A coach sent here from his profile may only have matches in the other
    # league. The league bar is already drawn, so the page follows him (and
    # says so) until the bar is changed - as the Coach Profile does.
    if seed_a is not None:
        seed_comps = tm.loc[tm.coachId == int(seed_a)].sort_values('date').competitionId
        if len(seed_comps) and comp_id not in set(seed_comps):
            st.session_state['coach_comparison_follow'] = (int(seed_comps.iloc[-1]), comp_id)
        else:
            st.session_state.pop('coach_comparison_follow', None)
    follow = st.session_state.get('coach_comparison_follow')
    if follow and follow[1] == comp_id and follow[0] in COMPETITIONS:
        comp_id = follow[0]
        st.info(f"Showing {COMPETITIONS[comp_id]['name']}: the coach you came from has no matches in the "
                "league selected in the sidebar.")
    else:
        st.session_state.pop('coach_comparison_follow', None)
    league_name = COMPETITIONS[comp_id]['name']
    scope = tm[(tm.competitionId == comp_id) & (tm.coachId > 0)]
    if scope.empty:
        st.info(f'No coach data for {league_name}.')
        return
    table = cp_view.league_table(sig, comp_id)

    last = scope.sort_values('date').groupby('coachId').agg(team=('team', 'last'), last=('date', 'max'),
                                                           n=('matchId', 'size'))
    last = last.sort_values(['last', 'n'], ascending=False)
    ids = [int(i) for i in last.index]
    if len(ids) < 2:
        st.info(f'{league_name} has fewer than two coaches in the data.')
        return
    labels = {i: f"{cp_view.coach_name(coaches, i)} ({last.at[i, 'team']}, {int(last.at[i, 'n'])} matches)"
              for i in ids}

    # Seed both selectors before they draw (a keyed selectbox whose options
    # change falls back to its first option unless its key is set that run).
    ours = scope[scope.team == app.OUR_TEAM].sort_values('date')
    a = seed_a if seed_a in labels else st.session_state.get(KEY_A)
    if a not in labels:
        a = int(ours.coachId.iloc[-1]) if len(ours) else ids[0]
    b = seed_b if seed_b in labels else st.session_state.get(KEY_B)
    # Coach B is chosen for the user only when he has none (first visit, a new
    # league) or a coach sent here as A is the one already in B. When the user
    # himself picks the same coach twice, the page says so below.
    if b not in labels or (b == a and seed_a is not None):
        sim = cm.similar_coaches(table, a)          # closest style first, else the next coach in the list
        near = [int(x) for x in sim.coachId if int(x) in labels and int(x) != a] if len(sim) else []
        b = near[0] if near else next(i for i in ids if i != a)
    st.session_state[KEY_A], st.session_state[KEY_B] = int(a), int(b)
    c = st.columns(2)
    a = c[0].selectbox('Coach A', ids, format_func=labels.get, key=KEY_A, help='Most recent match first.')
    b = c[1].selectbox('Coach B', ids, format_func=labels.get, key=KEY_B,
                       help="Opens on the coach whose style is closest to Coach A's.")
    if a == b:
        st.info('Pick two different coaches.')
        return

    names = [cp_view.coach_name(coaches, a), cp_view.coach_name(coaches, b)]
    if names[0] == names[1]:
        names = [f"{names[0]} ({last.at[a, 'team']})", f"{names[1]} ({last.at[b, 'team']})"]
    rows_a, rows_b = scope[scope.coachId == a], scope[scope.coachId == b]
    sa, sb = cm.summarise(rows_a), cm.summarise(rows_b)

    for col, cid, name, rows, s in zip(st.columns(2), (a, b), names, (rows_a, rows_b), (sa, sb)):
        with col:
            latest = rows.sort_values('date').iloc[-1]
            st.markdown(f'**{name}**')
            with st.container(key=theme.metric_fit_key(f'coach_cmp_{cid}', latest.team)):
                h = st.columns(3)
                h[0].metric('Latest team', latest.team)
                h[1].metric('Matches', str(s['matches']))
                h[2].metric('W-D-L', f"{s['wins']}-{s['draws']}-{s['losses']}")
            st.caption(f'{rows.date.min():%b %Y} to {rows.date.max():%b %Y}')
            if st.button('Open his profile', key=f'coach_cmp_open_{cid}'):
                navigation.go_to(cp_view.PAGE, **{cp_view.SEED_KEY: int(cid)})
    st.caption(f"Each coach on his whole record in {league_name}; matches coached in other leagues are not "
               "here. The two records can cover different seasons and squads.")
    short = [n for n, s in zip(names, (sa, sb)) if s['event_matches'] < cm.MIN_MATCHES]
    if short:
        st.warning(f"{' and '.join(short)} {'has' if len(short) == 1 else 'have'} fewer than {cm.MIN_MATCHES} "
                   "matches with event data here: too few to read much into the numbers.")

    def league_row(cid):
        lr = table[table.coachId == cid]
        return lr.iloc[0].to_dict() if len(lr) else None

    active = st.radio('Comparison section', SECTIONS, horizontal=True, label_visibility='collapsed',
                      key='coach_comparison_section')
    if active == 'Results':
        _results(table, sa, sb, names, league_name)
    elif active == 'Style':
        _style(table, sa, sb, league_row(a), league_row(b), names, league_name)
    elif active == 'Set-up & tendencies':
        _setup(table, sa, sb, names, league_name, cp_view.league_reliability(sig, comp_id))
    elif active == 'In-game record':
        _in_game(table, sa, sb, names, league_name)
    else:
        _squad(table, sa, sb, names, league_name)
