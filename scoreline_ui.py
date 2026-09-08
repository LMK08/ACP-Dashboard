"""Streamlit section for the Dixon-Coles scoreline model (Match Predictor page).

Loads models/scoreline/dc_params.json (fitted by models/scoreline/build_dc.py)
and dc_backtest.json, and renders for one fixture: W/D/L, expected goals, the
score matrix as a heatmap, the most likely scorelines, over 2.5 / both to
score / clean sheets — plus a calibration expander that shows the
walk-forward backtest honestly (log loss against the base rate and against
the simple predictor, per season, and the reliability table).
"""
import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import theme
from models.scoreline.dixon_coles import DixonColes

_HERE = os.path.dirname(os.path.abspath(__file__))
PARAMS_PATH = os.path.join(_HERE, 'models', 'scoreline', 'dc_params.json')
BACKTEST_PATH = os.path.join(_HERE, 'models', 'scoreline', 'dc_backtest.json')


@st.cache_resource(show_spinner=False)
def load_model(mtime):
    """mtime is part of the key so a rebuilt params file is picked up."""
    return DixonColes.load(PARAMS_PATH)


@st.cache_data(show_spinner=False)
def load_backtest(mtime):
    with open(BACKTEST_PATH, encoding='utf-8') as fh:
        return json.load(fh)


def _mtime(path):
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def _score_heatmap(P, home, away, max_show=6):
    n = min(max_show, P.shape[0] - 1)
    M = P[:n + 1, :n + 1]
    fig = go.Figure(go.Heatmap(
        z=M, x=[str(j) for j in range(n + 1)], y=[str(i) for i in range(n + 1)],
        colorscale=[[0, theme.FIGURE_BG], [1, theme.FOCUS]], zmin=0, zmax=float(M.max()),
        text=[[f'{v:.1%}' if v >= 0.005 else '' for v in row] for row in M],
        texttemplate='%{text}', textfont=dict(size=13),
        hovertemplate=f'{home} %{{y}} – {away} %{{x}}<br>%{{z:.1%}}<extra></extra>',
        showscale=False))
    fig.update_layout(
        title=dict(text=f'<b>Scoreline probabilities</b>', font=dict(size=14, color=theme.FIGURE_INK), x=0.01),
        xaxis=dict(title=f'{away} goals', side='top', fixedrange=True),
        yaxis=dict(title=f'{home} goals', autorange='reversed', fixedrange=True),
        paper_bgcolor=theme.FIGURE_BG, plot_bgcolor=theme.FIGURE_BG, height=500,
        margin=dict(l=60, r=10, t=70, b=10), font=dict(color=theme.FIGURE_INK))
    return fig


def _reliability_fig(rel_rows):
    rel = pd.DataFrame(rel_rows)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(color='#9a948d', dash='dot'),
                             hoverinfo='skip', showlegend=False))
    colours = {'home': theme.HOME_BLUE, 'draw': theme.CONTEXT, 'away': theme.AWAY_RED}
    for outcome, g in rel.groupby('outcome'):
        g = g[g['n'] >= 20]
        fig.add_trace(go.Scatter(x=g['predicted'], y=g['observed'], mode='lines+markers', name=outcome,
                                 marker=dict(size=np.clip(g['n'] / 25, 6, 22), color=colours.get(outcome)),
                                 line=dict(color=colours.get(outcome)),
                                 hovertemplate='%{customdata} matches<br>predicted %{x:.0%} · observed %{y:.0%}<extra>' + outcome + '</extra>',
                                 customdata=g['n']))
    fig.update_layout(title=dict(text='<b>Reliability</b> · predicted vs observed frequency (bins with 20+ matches)',
                                 font=dict(size=13, color=theme.FIGURE_INK), x=0.01),
                      xaxis=dict(title='predicted probability', range=[0, 1], fixedrange=True),
                      yaxis=dict(title='observed frequency', range=[0, 1], fixedrange=True),
                      paper_bgcolor=theme.FIGURE_BG, plot_bgcolor=theme.FIGURE_BG, height=440,
                      legend=dict(orientation='h', x=0, y=1.0, xanchor='left', yanchor='bottom'),
                      margin=dict(l=50, r=10, t=70, b=40), font=dict(color=theme.FIGURE_INK))
    return fig


def render_strength_table(league_id, teams, played=None, league_name=None, season_label=None,
                          key='dc_strength'):
    """The Team Strength Ratings table: the Dixon-Coles attack / defence
    parameters for ``teams`` (a league's current-season sides), best first.

    Every team's attack and defence are fitted JOINTLY on all matches with
    time decay and shrinkage toward the league average, so opponent quality
    is inside the fit and a handful of early-season games moves a rating
    only as far as the evidence warrants — the reason this replaced the
    per-match xG rates + schedule ratio in 2026-09. ``played`` (Series
    team -> matches played this season) is shown for context only: it is
    NOT the sample the fit rests on.
    """
    mt = _mtime(PARAMS_PATH)
    if mt is None:
        st.info('Scoreline model not built yet — run `python models/scoreline/build_dc.py`.')
        return None
    model = load_model(mt)
    tbl = model.strength_table(teams=teams, league=league_id)
    if tbl.empty:
        st.info('No teams to rate for this league.')
        return None

    unseen = tbl.loc[~tbl['known'], 'team'].tolist()
    show = pd.DataFrame({
        'Rank': range(1, len(tbl) + 1),
        'Team': [f'{t} *' if t in unseen else t for t in tbl['team']],
        'Scores ×': tbl['scores_x'].round(2),
        'Concedes ×': tbl['concedes_x'].round(2),
        'xGD vs avg': tbl['xgd_vs_avg'].round(2),
    })
    if played is not None:
        show['P'] = [int(played.get(t, 0)) for t in tbl['team']]
    st.dataframe(
        show, hide_index=True, use_container_width=True,
        column_config={
            'Rank': st.column_config.NumberColumn('Rank', width='small'),
            'Scores ×': st.column_config.NumberColumn('Scores ×', format='%.2f',
                                                      help='Goals scored relative to the average of the sides shown (1.00)'),
            'Concedes ×': st.column_config.NumberColumn('Concedes ×', format='%.2f',
                                                        help='Goals conceded relative to the average of the sides shown (1.00); lower is better'),
            'xGD vs avg': st.column_config.NumberColumn('xGD vs avg', format='%+.2f',
                                                        help='Expected goal difference against an average side of this league at a neutral venue'),
            'P': st.column_config.NumberColumn('P', width='small', help='Matches played this season (context only)'),
        },
        key=f'{key}_table')
    mix_txt = ('goals' if model.mix >= 1 else f'{int(round((1 - model.mix) * 100))}% xG / {int(round(model.mix * 100))}% goals')
    half_life = int(round(np.log(2) / model.xi)) if model.xi > 0 else None
    scope = ' · '.join(x for x in (league_name, season_label) if x)
    st.caption(
        (f'{scope}. ' if scope else '')
        + 'Dixon-Coles attack / defence parameters: every team\'s attack and defence are fitted jointly, so '
          'opponent quality is inside the fit, with shrinkage toward the league average and '
        + (f'a {half_life}-day half-life on older matches' if half_life else 'no time decay')
        + f'; rates fitted on {mix_txt}, {model.n_matches:,} matches through {model.asof}. '
          'Scores × / Concedes × = goals scored / conceded relative to the average of the sides shown; '
          'xGD vs avg = expected goal difference against that average side at a neutral venue; '
          'the table is ranked by it. P = matches played this season (context only). '
          'Same model as the scoreline forecast below; its walk-forward backtest is there.'
        + (f" * {', '.join(unseen)}: not seen by the fit yet — shown at the average here; the forecast "
           f"below plays them at zero parameters." if unseen else ''))
    return tbl


def render_scoreline_section(home, away, league_id, season_id_map, key='dc'):
    """One fixture's scoreline forecast plus the model's calibration."""
    st.subheader('Scoreline model')
    mt = _mtime(PARAMS_PATH)
    if mt is None:
        st.info('Scoreline model not built yet — run `python models/scoreline/build_dc.py`.')
        return
    model = load_model(mt)
    pr = model.predict(home, away, league_id)

    unknown = [t for t, k in ((home, pr['known_home']), (away, pr['known_away'])) if not k]
    mix_txt = ('goals' if model.mix >= 1 else f'{int(round((1 - model.mix) * 100))}% xG / {int(round(model.mix * 100))}% goals')
    half_life = int(round(np.log(2) / model.xi)) if model.xi > 0 else None
    st.caption(f"Dixon-Coles, rates fitted on {mix_txt}, "
               f"{'half-life ' + str(half_life) + ' days' if half_life else 'no time decay'}, "
               f"{model.n_matches:,} matches through {model.asof} · home advantage {np.exp(model.home_adv) - 1:+.0%} goals"
               + (f" · **{' and '.join(unknown)} not seen yet: treated as league average**" if unknown else ''))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f'{home} win', f"{pr['p_home']:.0%}")
    c2.metric('Draw', f"{pr['p_draw']:.0%}")
    c3.metric(f'{away} win', f"{pr['p_away']:.0%}")
    c4.metric('Expected goals', f"{pr['lambda']:.2f} – {pr['mu']:.2f}")

    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(_score_heatmap(pr['matrix'], home, away), use_container_width=True,
                        config={'displayModeBar': False}, theme=None, key=f'{key}_heat')
    with right:
        st.markdown('**Most likely scorelines**')
        top = pd.DataFrame([{'Score': s, 'Probability': round(100 * p, 1)} for s, p in pr['top_scores']])
        st.dataframe(top, hide_index=True, use_container_width=True,
                     column_config={'Probability': st.column_config.ProgressColumn(
                         'Probability', min_value=0.0, max_value=float(top['Probability'].max()),
                         format='%.1f%%')})
        st.markdown(f"Over 2.5 goals **{pr['over_2_5']:.0%}** · both score **{pr['btts']:.0%}**  \n"
                    f"Clean sheet: {home} **{pr['clean_sheet_home']:.0%}** · {away} **{pr['clean_sheet_away']:.0%}**")

    bt_mt = _mtime(BACKTEST_PATH)
    if bt_mt is None:
        return
    bt = load_backtest(bt_mt)
    # A toggle, not an expander: a Plotly chart first drawn inside a collapsed
    # expander keeps a collapsed height when opened.
    if st.toggle('How good is it? Show the walk-forward backtest', value=False, key=f'{key}_show_bt'):
        ov, base = bt['overall'], bt['base_rate_overall']
        lfl = bt.get('like_for_like_2025_26_liga3', {})
        dcm, sp = lfl.get('dixon_coles', {}), lfl.get('simple_predictor_reported', {})
        st.markdown(
            f"Refit every month on everything before that month, {ov['n']:,} matches predicted "
            f"({bt.get('n_predicted', ov['n']):,}). Lower log loss is better; a constant forecast at the league's "
            f"base rates scores **{base['log_loss']:.3f}**.")
        m1, m2, m3 = st.columns(3)
        m1.metric('Log loss (all)', f"{ov['log_loss']:.3f}", f"{ov['log_loss'] - base['log_loss']:+.3f} vs base rate",
                  delta_color='inverse')
        m2.metric('Accuracy (all)', f"{ov['accuracy']:.1%}", f"{ov['accuracy'] - base['accuracy']:+.1%} vs base rate")
        if dcm and sp:
            m3.metric('Log loss, Liga 3 2025/26', f"{dcm['log_loss']:.3f}",
                      f"{dcm['log_loss'] - sp['log_loss']:+.3f} vs the strength model ({sp['log_loss']:.3f})",
                      delta_color='inverse')
        league_names = {'43324': 'Liga 3', '702': 'Campeonato'}
        rows = []
        for sid, r in sorted(bt['per_season'].items(), key=lambda kv: str(season_id_map.get(int(kv[0]), kv[0]))):
            leagues = ' + '.join(league_names.get(l, l) for l in r.get('leagues', {}))
            rows.append({'Season': f"{season_id_map.get(int(sid), sid)} {leagues}".strip(), 'Matches': r['n'],
                         'Log loss': round(r['log_loss'], 3), 'Base rate': round(r['base_rate']['log_loss'], 3),
                         'Brier': round(r['brier'], 3), 'Accuracy': f"{r['accuracy']:.0%}"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.plotly_chart(_reliability_fig(bt['reliability']), use_container_width=True,
                        config={'displayModeBar': False}, theme=None, key=f'{key}_rel')
        st.caption(f"Settings chosen on this backtest: decay xi={bt['xi']}, shrinkage l2={bt['l2']}, "
                   f"goals share of the fitted target={bt['mix']}. Built {bt.get('built', '?')}.")
