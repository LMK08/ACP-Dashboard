"""Interactive (Plotly) versions of the team-level visuals both the Team
Analysis page and the Opposition Report show on screen:

    plotly_season_shot_map   shots for / conceded, hover per shot, click -> match
    plotly_passing_network   same data as pitch_visualizations.compute_passing_network,
                             hover on nodes and edges, click a player -> profile
    plotly_rolling_xg        5-game rolling xG for/against, hover per match, click -> match

The matplotlib originals stay in app.py / pitch_visualizations.py and are
what the Opposition Report PDF embeds; the two renderers share their data
steps so a number on screen is the number in the PDF. Colours come from
theme.py. `selected_customdata` turns a st.plotly_chart selection event into
the customdata rows the pages act on.
"""
import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import theme
from pitch_interactive import _pitch_layout  # portrait attacking half, Wyscout dims

LINE = theme.PITCH_LINE_PLOTLY
NODE = '#1d3557'
EDGE_PLAIN = '#457b9d'
EDGE_NEG = '#b0b0a8'
EDGE_RAMP = ['#c9d6c4', theme.FOCUS]


# ---------------------------------------------------------------------------
# Selection plumbing
# ---------------------------------------------------------------------------
def selected_customdata(event):
    """customdata rows of the points in a st.plotly_chart(on_select='rerun')
    event, or [] — tolerant of the dict / object forms across versions."""
    try:
        sel = event['selection'] if isinstance(event, dict) else event.selection
        points = sel['points'] if isinstance(sel, dict) else sel.points
    except (KeyError, AttributeError, TypeError):
        return []
    rows = []
    for p in points or []:
        cd = p.get('customdata') if isinstance(p, dict) else getattr(p, 'customdata', None)
        if cd is not None:
            rows.append(cd)
    return rows


def match_id_from_rows(rows):
    """First positive matchId carried by selected customdata rows: shots put
    it at index 6 (before the team name), rolling-xG points at the end."""
    for row in rows:
        cells = list(row) if not isinstance(row, (str, bytes)) else [row]
        for cell in ([cells[6]] if len(cells) > 6 else []) + ([cells[-1]] if cells else []):
            try:
                mid = int(cell)
            except (TypeError, ValueError):
                continue
            if mid > 0:
                return mid
    return None


def player_id_from_rows(rows):
    """playerId from a passing-network node selection (customdata = [id, name])."""
    for row in rows:
        try:
            return int(row[0])
        except (TypeError, ValueError, IndexError):
            continue
    return None


def _match_lookup(matches_df):
    """matchId -> (date str, home, away, score) for hover text."""
    if matches_df is None or matches_df.empty:
        return {}
    out = {}
    for _, m in matches_df.iterrows():
        d = pd.to_datetime(m.get('dateutc'), errors='coerce')
        out[m['matchId']] = (d.strftime('%d %b %Y') if pd.notna(d) else '?',
                             str(m.get('homeTeamName', '?')), str(m.get('awayTeamName', '?')),
                             str(m.get('score', '')))
    return out


# ---------------------------------------------------------------------------
# 1. Season shot maps (for / against)
# ---------------------------------------------------------------------------
def season_shots(season_events_df, matches_summary_df, team, mode='for'):
    """The shots the matplotlib maps draw: our shots (mode='for') or every
    opponent shot in our matches (mode='against'). Non-penalty is implied by
    the events frame the pages pass (they already drop penalties upstream)."""
    ev = season_events_df
    if mode == 'for':
        shots = ev[(ev.get('team.name') == team) & (ev.get('type.primary') == 'shot')]
    else:
        ids = matches_summary_df[(matches_summary_df.get('homeTeamName') == team)
                                 | (matches_summary_df.get('awayTeamName') == team)]['matchId'].unique()
        rel = ev[ev['matchId'].isin(ids)]
        shots = rel[(rel.get('type.primary') == 'shot') & (rel.get('team.name') != team)]
    shots = shots.copy()
    for c in ('location.x', 'location.y', 'shot.xg'):
        shots[c] = pd.to_numeric(shots.get(c), errors='coerce')
    return shots.dropna(subset=['location.x', 'location.y', 'shot.xg']).reset_index(drop=True)


def plotly_season_shot_map(season_events_df, matches_summary_df, team, mode='for', height=560):
    shots = season_shots(season_events_df, matches_summary_df, team, mode)
    title = f"{team} — Shots {'For' if mode == 'for' else 'Conceded'} (Non-Pen)"
    fig = go.Figure()
    if shots.empty:
        fig.add_annotation(text='No shots in this scope', xref='paper', yref='paper',
                           x=0.5, y=0.5, showarrow=False, font=dict(color='#6b7570'))
        _pitch_layout(fig, height=height)
        return fig

    lookup = _match_lookup(matches_summary_df)
    is_goal = (shots.get('shot.isGoal') == True)  # noqa: E712
    meta = []
    for _, s in shots.iterrows():
        date, home, away, score = lookup.get(s['matchId'], ('?', '?', '?', ''))
        opponent = away if home == team else home
        meta.append([str(s.get('player.name', '—')), date, opponent, score,
                     f"{s['shot.xg']:.2f}", 'Goal' if bool(s.get('shot.isGoal') == True) else 'No goal',  # noqa: E712
                     int(s['matchId']), str(s.get('team.name', ''))])
    custom = np.array(meta, dtype=object)
    fig.add_trace(go.Scatter(
        x=shots['location.y'], y=shots['location.x'], mode='markers',
        marker=dict(size=16, opacity=0.75, color=shots['shot.xg'], coloraxis='coloraxis',
                    line=dict(color=np.where(is_goal, '#1a7a2e', 'rgba(0,0,0,0.45)'),
                              width=np.where(is_goal, 3, 1))),
        customdata=custom, showlegend=False,
        hovertemplate=('<b>%{customdata[0]}</b> (%{customdata[7]})<br>'
                       '%{customdata[1]} vs %{customdata[2]} %{customdata[3]}<br>'
                       'xG %{customdata[4]} · %{customdata[5]}<extra></extra>')))
    total_xg = float(shots['shot.xg'].sum())
    goals = int(is_goal.sum())
    fig.add_annotation(text=f'<b>{title}</b>', xref='paper', yref='paper', x=0.01, y=1.10,
                       showarrow=False, font=dict(size=14, color=theme.FIGURE_INK), xanchor='left')
    fig.add_annotation(text=(f'{len(shots)} shots · {goals} goals · {total_xg:.2f} '
                             f'{"xG" if mode == "for" else "xGA"} · colour = xG · ring = goal · '
                             'click a shot to open that match'),
                       xref='paper', yref='paper', x=0.01, y=1.055, showarrow=False,
                       font=dict(size=11, color='#5a5a5a'), xanchor='left')
    _pitch_layout(fig, height=height)
    fig.update_layout(coloraxis=dict(colorscale=theme.XG_COLORSCALE, cmin=0, cmax=theme.XG_MAX,
                                     colorbar=dict(title='xG', thickness=12, len=0.5, y=0.5)),
                      margin=dict(l=10, r=10, t=58, b=10), clickmode='event+select')
    return fig


# ---------------------------------------------------------------------------
# 2. Passing network
# ---------------------------------------------------------------------------
def _full_pitch_shapes():
    """Landscape Wyscout pitch, attacking left -> right; y=0 is the TOP
    (mplsoccer's wyscout convention), so the layout reverses the y axis."""
    ln = dict(color=LINE, width=1.5)
    return [
        dict(type='rect', x0=0, y0=0, x1=100, y1=100, line=ln),
        dict(type='line', x0=50, y0=0, x1=50, y1=100, line=ln),
        dict(type='circle', x0=41.3, y0=36.5, x1=58.7, y1=63.5, line=ln),
        dict(type='rect', x0=0, y0=19, x1=16, y1=81, line=ln),
        dict(type='rect', x0=84, y0=19, x1=100, y1=81, line=ln),
        dict(type='rect', x0=0, y0=37, x1=6, y1=63, line=ln),
        dict(type='rect', x0=94, y0=37, x1=100, y1=63, line=ln),
        dict(type='rect', x0=-2, y0=44.8, x1=0, y1=55.2, line=ln),
        dict(type='rect', x0=100, y0=44.8, x1=102, y1=55.2, line=ln),
        dict(type='path', path='M 16,41.5 Q 22.8,50 16,58.5', line=ln),
        dict(type='path', path='M 84,41.5 Q 77.2,50 84,58.5', line=ln),
    ]


def _full_pitch_layout(fig, height=560):
    fig.update_layout(
        shapes=_full_pitch_shapes(),
        xaxis=dict(range=[-4, 104], visible=False, fixedrange=True),
        yaxis=dict(range=[104, -8], visible=False, fixedrange=True, scaleanchor='x', scaleratio=0.68),
        plot_bgcolor=theme.FIGURE_BG, paper_bgcolor=theme.FIGURE_BG,
        margin=dict(l=10, r=10, t=54, b=10), height=height, showlegend=False,
        dragmode=False, clickmode='event+select')
    return fig


def _ramp(v, vmax):
    """Hex colour on the cream -> focus-green ramp for a non-negative value."""
    t = 0.0 if vmax <= 0 else min(max(v, 0.0) / vmax, 1.0)
    a, b = (int(EDGE_RAMP[0][i:i + 2], 16) for i in (1, 3, 5)), (int(EDGE_RAMP[1][i:i + 2], 16) for i in (1, 3, 5))
    return '#%02x%02x%02x' % tuple(int(x + (y - x) * t) for x, y in zip(a, b))


def plotly_passing_network(net, team_name, title=None, height=560):
    """`net` is pitch_visualizations.compute_passing_network(...)."""
    nodes, edges = net['nodes'], net['edges']
    fig = go.Figure()
    if nodes.empty:
        fig.add_annotation(text='Not enough passes for a network', xref='paper', yref='paper',
                           x=0.5, y=0.5, showarrow=False, font=dict(color='#6b7570'))
        return _full_pitch_layout(fig, height)

    pos = {int(r['player.id']): (float(r['x']), float(r['y'])) for _, r in nodes.iterrows()}
    names = {int(r['player.id']): str(r['name']) for _, r in nodes.iterrows()}
    max_cnt = float(edges['count'].max()) if not edges.empty else 1.0
    has_obv = net['has_obv']
    vmax = float(max((abs(v) for v in edges.get('obv', pd.Series(dtype=float))), default=0.0)) or 1.0

    mid_x, mid_y, mid_text = [], [], []
    for _, e in edges.iterrows():
        p, r = int(e['passer']), int(e['receiver'])
        if p not in pos or r not in pos:
            continue
        (sx, sy), (ex, ey) = pos[p], pos[r]
        width = 0.8 + 5.5 * (e['count'] / max_cnt)
        if has_obv:
            v = float(e.get('obv', 0.0))
            colour = _ramp(v, vmax) if v >= 0 else EDGE_NEG
            opacity = 0.85
        else:
            colour, opacity = EDGE_PLAIN, 0.6
        fig.add_trace(go.Scatter(x=[sx, ex], y=[sy, ey], mode='lines',
                                 line=dict(color=colour, width=width), opacity=opacity,
                                 hoverinfo='skip', showlegend=False))
        mid_x.append((sx + ex) / 2); mid_y.append((sy + ey) / 2)
        mid_text.append(f"<b>{names[p]} → {names[r]}</b><br>{int(e['count'])} passes"
                        + (f"<br>on-ball value {float(e.get('obv', 0.0)):+.2f}" if has_obv else ''))
    # invisible edge midpoints carry the hover (plotly lines hover poorly)
    fig.add_trace(go.Scatter(x=mid_x, y=mid_y, mode='markers', marker=dict(size=14, opacity=0.0),
                             hovertext=mid_text, hoverinfo='text', showlegend=False))

    metric = nodes['node_metric'].astype(float)
    mmax = float(metric.max()) or 1.0
    sizes = 16 + 26 * np.sqrt(metric / mmax)
    hover = [(f"<b>{r['name']}</b> · {r['position']}<br>{int(r['pass_count'])} passes made"
              + (f"<br>passing value {float(r['node_metric']):+.2f}" if has_obv else f"<br>{int(r['count'])} touches"))
             for _, r in nodes.iterrows()]
    fig.add_trace(go.Scatter(
        x=nodes['x'], y=nodes['y'], mode='markers+text',
        marker=dict(size=sizes, color=NODE, line=dict(color='white', width=2)),
        text=nodes['pass_count'].astype(int).astype(str), textfont=dict(color='white', size=9),
        textposition='middle center',
        customdata=np.stack([nodes['player.id'].astype(int), nodes['name'].astype(str)], axis=1),
        hovertext=hover, hoverinfo='text', showlegend=False))
    fig.add_trace(go.Scatter(
        x=nodes['x'], y=nodes['y'] + 4.5, mode='text',
        text=[_short(n) for n in nodes['name']], textposition='bottom center',
        textfont=dict(size=10, color=NODE), hoverinfo='skip', showlegend=False))

    legend = ('line width = pass volume · line colour = on-ball value added (grey = negative) · '
              'circle size = passing value' if has_obv else
              'line width = pass volume · circle size = involvement')
    fig.add_annotation(text=f'<b>{title or f"{team_name} — Passing Network"}</b>',
                       xref='paper', yref='paper', x=0.01, y=1.09, showarrow=False,
                       font=dict(size=14, color=theme.FIGURE_INK), xanchor='left')
    fig.add_annotation(text=legend + ' · click a player to open the profile',
                       xref='paper', yref='paper', x=0.01, y=1.045, showarrow=False,
                       font=dict(size=10.5, color='#5a5a5a'), xanchor='left')
    return _full_pitch_layout(fig, height)


def _short(name):
    parts = str(name).split()
    return parts[-1] if len(parts) > 1 else str(name)


# ---------------------------------------------------------------------------
# 3. Rolling xG
# ---------------------------------------------------------------------------
def rolling_xg_frame(all_matches_df, team, window=5):
    """Same steps as app.plot_match_xg_history: team rows, last 365 days,
    ordinal match axis, rolling means. Returns the frame or an empty one."""
    df = all_matches_df[all_matches_df['teamName'] == team].copy()
    if df.empty:
        return df
    today = pd.to_datetime(datetime.date.today())
    df = df[(df['date'] >= today - pd.DateOffset(years=1)) & (df['date'] <= today)]
    df = df.sort_values('date').dropna(subset=['xG_For', 'xG_Against']).reset_index(drop=True)
    df['match_seq'] = df.index
    df['xG_For_Roll'] = df['xG_For'].rolling(window=window, min_periods=1).mean()
    df['xG_Against_Roll'] = df['xG_Against'].rolling(window=window, min_periods=1).mean()
    return df


def plotly_rolling_xg(all_matches_df, team, matches_summary_df=None, window=5, height=420):
    df = rolling_xg_frame(all_matches_df, team, window)
    fig = go.Figure()
    if df.empty:
        fig.add_annotation(text='No matches in the last year', xref='paper', yref='paper',
                           x=0.5, y=0.5, showarrow=False, font=dict(color='#6b7570'))
        fig.update_layout(height=height, paper_bgcolor=theme.FIGURE_BG, plot_bgcolor=theme.FIGURE_BG)
        return fig

    lookup = _match_lookup(matches_summary_df) if 'matchId' in df.columns else {}
    opp, score, mids = [], [], []
    for _, r in df.iterrows():
        d, home, away, sc = lookup.get(r.get('matchId'), ('?', '?', '?', ''))
        opp.append(away if home == team else home if home != '?' else '—')
        score.append(sc)
        mids.append(int(r['matchId']) if 'matchId' in df.columns and pd.notna(r.get('matchId')) else -1)
    custom = np.array([[d.strftime('%d %b %Y'), o, s, f"{f:.2f}", f"{a:.2f}", m]
                       for d, o, s, f, a, m in zip(df['date'], opp, score, df['xG_For'], df['xG_Against'], mids)],
                      dtype=object)
    x = df['match_seq']
    for_r, ag_r = df['xG_For_Roll'], df['xG_Against_Roll']
    # sign-coloured band between the rolling lines
    fig.add_trace(go.Scatter(x=x, y=ag_r, mode='lines', line=dict(width=0), hoverinfo='skip', showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=np.where(for_r >= ag_r, for_r, ag_r), mode='lines', line=dict(width=0),
                             fill='tonexty', fillcolor='rgba(0,119,182,0.18)', hoverinfo='skip', showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=ag_r, mode='lines', line=dict(width=0), hoverinfo='skip', showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=np.where(for_r < ag_r, for_r, ag_r), mode='lines', line=dict(width=0),
                             fill='tonexty', fillcolor='rgba(230,57,70,0.18)', hoverinfo='skip', showlegend=False))
    hover = ('<b>%{customdata[0]}</b> vs %{customdata[1]} %{customdata[2]}<br>'
             'match xG %{customdata[3]} for · %{customdata[4]} against<br>'
             f'{window}-game rolling: ' + '%{y:.2f}<extra>%{fullData.name}</extra>')
    fig.add_trace(go.Scatter(x=x, y=for_r, mode='lines+markers', name='xG for',
                             line=dict(color=theme.HOME_BLUE, width=2.5), marker=dict(size=6),
                             customdata=custom, hovertemplate=hover))
    fig.add_trace(go.Scatter(x=x, y=ag_r, mode='lines+markers', name='xG against',
                             line=dict(color=theme.AWAY_RED, width=2.5), marker=dict(size=6),
                             customdata=custom, hovertemplate=hover))
    # season / second-stage separators, as in the matplotlib version
    if 'seasonId' in df.columns:
        for idx in df.index[df['seasonId'].diff().fillna(0) != 0]:
            fig.add_vline(x=idx - 0.5, line=dict(color='gray', dash='dot', width=1.2),
                          annotation_text='New season', annotation_position='top left',
                          annotation_font=dict(size=10, color='gray'))
    if 'roundId' in df.columns and 'seasonId' in df.columns:
        for sid in df['seasonId'].unique():
            season_all = all_matches_df[all_matches_df['seasonId'] == sid]
            if season_all.empty:
                continue
            first_round = season_all.groupby('roundId').size().idxmax()
            slice_ = df[(df['seasonId'] == sid) & (df['roundId'] != first_round)]
            if not slice_.empty:
                fig.add_vline(x=slice_.index[0] - 0.5, line=dict(color='#6a0dad', dash='dash', width=1.2),
                              annotation_text='Second stage', annotation_position='top left',
                              annotation_font=dict(size=10, color='#6a0dad'))
    step = max(1, len(df) // 10)
    fig.update_layout(
        title=dict(text=f'<b>{team} — {window}-game rolling xG</b><span style="font-size:11px;color:#5a5a5a">'
                        '  · click a match to open its report</span>', x=0.01, xanchor='left',
                   font=dict(color=theme.FIGURE_INK)),
        font=dict(color=theme.FIGURE_INK),
        xaxis=dict(tickmode='array', tickvals=list(x[::step]), ticktext=list(df['date'].dt.strftime('%d/%m')[::step]),
                   showgrid=False),
        yaxis=dict(title=f'{window}-game rolling xG', rangemode='tozero', gridcolor='rgba(0,0,0,0.08)'),
        legend=dict(orientation='h', x=0, y=1.0, xanchor='left', yanchor='bottom'),
        paper_bgcolor=theme.FIGURE_BG, plot_bgcolor=theme.FIGURE_BG, height=height,
        margin=dict(l=50, r=20, t=70, b=40), hovermode='closest', clickmode='event+select')
    return fig
