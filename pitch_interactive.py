"""Interactive (Plotly) pitch visuals for the Player Profile:

- plotly_shot_map        — hoverable upgrade of the static shot map
- plotly_box_passes_map  — every pass into the attacking box, arrows
                            colored by GPA pass value
- mpl_box_passes_map     — static matplotlib twin for PDF export

Coordinates are Wyscout 100x100 (x toward opponent goal). Both plotly
charts render a portrait half-pitch, goal at the top:
    plot_x = event y   ·   plot_y = event x
"""
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

PITCH_BG = '#f5f1e9'
LINE_C = '#5a5a5a'

# Same anchors as the static shot map (create_player_shotmap)
XG_MAX = 0.8
XG_COLORSCALE = [[0.0, '#03045e'], [0.125, '#ade8f4'], [0.25, '#fff3b0'],
                 [0.5, '#ff8c00'], [0.75, '#e63946'], [1.0, '#800f2f']]

# Diverging pass-value scale with a warm-gray midpoint (visible on cream):
# cold blue = value conceded/negative, warm red = value created
VAL_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'gpa_div', ['#2166ac', '#b8b2a7', '#b2182b'])
VAL_COLORSCALE = [[0.0, '#2166ac'], [0.5, '#b8b2a7'], [1.0, '#b2182b']]


def _half_pitch_shapes():
    """Plotly shapes for a portrait attacking half (Wyscout dims)."""
    ln = dict(color=LINE_C, width=1.5)
    return [
        # outer boundary (half pitch) + goal line emphasized by boundary
        dict(type='rect', x0=0, y0=50, x1=100, y1=100, line=ln),
        # penalty area (depth 16 units, y 19-81)
        dict(type='rect', x0=19, y0=84, x1=81, y1=100, line=ln),
        # six-yard box
        dict(type='rect', x0=37, y0=94, x1=63, y1=100, line=ln),
        # goal mouth
        dict(type='rect', x0=44.8, y0=100, x1=55.2, y1=102, line=ln),
        # penalty spot
        dict(type='circle', x0=49.4, y0=89.4, x1=50.6, y1=90.6,
             line=dict(color=LINE_C, width=1), fillcolor=LINE_C),
        # the "D" — quadratic approximation of the arc outside the box
        dict(type='path', path='M 41.5,84 Q 50,77.2 58.5,84', line=ln),
        # center circle sliver at the halfway line
        dict(type='circle', x0=41.3, y0=41.3, x1=58.7, y1=58.7, line=ln),
    ]


def _pitch_layout(fig, height=620):
    fig.update_layout(
        shapes=_half_pitch_shapes(),
        xaxis=dict(range=[-3, 103], visible=False, fixedrange=True),
        yaxis=dict(range=[49.7, 104.5], visible=False, fixedrange=True,
                   scaleanchor='x', scaleratio=1),
        plot_bgcolor=PITCH_BG, paper_bgcolor=PITCH_BG,
        margin=dict(l=10, r=10, t=54, b=10), height=height,
        showlegend=False, dragmode=False,
    )
    return fig


def plotly_shot_map(shot_log: pd.DataFrame, player_name: str,
                    height: int = 460) -> go.Figure:
    """Interactive shot map. Expects the processed shot_log from the
    Shots section (Shot Number, Date, Opponent, Result, xG, Body Part,
    Phase, location.x/y, shot.isGoal)."""
    df = shot_log.dropna(subset=['location.x', 'location.y']).copy()
    df['xG'] = pd.to_numeric(df.get('xG', df.get('shot.xg')),
                             errors='coerce').fillna(0.0)
    is_goal = df.get('shot.isGoal') == True  # noqa: E712

    fig = go.Figure()
    custom = np.stack([
        df.get('Date', pd.Series('—', index=df.index)).astype(str),
        df.get('Opponent', pd.Series('—', index=df.index)).astype(str),
        df.get('minute', pd.Series(np.nan, index=df.index)).astype(str),
        df.get('Result', pd.Series('—', index=df.index)).astype(str),
        df['xG'].round(3).astype(str),
        df.get('Body Part', pd.Series('—', index=df.index)).astype(str),
        df.get('Phase', pd.Series('—', index=df.index)).astype(str),
    ], axis=1)
    fig.add_trace(go.Scatter(
        x=df['location.y'], y=df['location.x'],
        mode='markers+text',
        text=df.get('Shot Number', pd.Series('', index=df.index)).astype(str),
        textfont=dict(color='white', size=9),
        marker=dict(
            size=12 + (df['xG'] / XG_MAX).clip(0, 1) * 26,
            color=df['xG'], colorscale=XG_COLORSCALE, cmin=0, cmax=XG_MAX,
            opacity=0.92,
            line=dict(
                color=np.where(is_goal, '#1a7a2e', 'rgba(0,0,0,0.45)'),
                width=np.where(is_goal, 3, 1)),
            colorbar=dict(title='xG', thickness=12, len=0.55, y=0.45),
        ),
        customdata=custom,
        hovertemplate=('<b>#%{text} · %{customdata[3]}</b><br>'
                       '%{customdata[0]} vs %{customdata[1]} · '
                       "%{customdata[2]}'<br>"
                       'xG %{customdata[4]} · %{customdata[5]}<br>'
                       '%{customdata[6]}<extra></extra>'),
    ))
    goals = int(is_goal.sum())
    fig.add_annotation(
        text=(f'<b>{player_name}</b> — Season Shot Map (non-penalty)<br>'
              f'{len(df)} shots · {goals} goals · '
              f'{df["xG"].sum():.2f} xG · ring = goal'),
        xref='paper', yref='paper', x=0.5, y=1.085, showarrow=False,
        font=dict(size=13), align='center')
    return _pitch_layout(fig, height=height)


def _robust_vmax(vals: pd.Series) -> float:
    v = pd.to_numeric(vals, errors='coerce').dropna()
    if v.empty:
        return 0.02
    return float(max(abs(v.quantile(0.05)), abs(v.quantile(0.95)), 0.02))


def _pass_hover(df: pd.DataFrame) -> np.ndarray:
    tags = []
    for _, r in df.iterrows():
        t = [k.replace('_', ' ') for k in
             ('cross', 'through_pass', 'smart_pass', 'key_pass', 'assist')
             if r.get(k)]
        tags.append(' · '.join(t) if t else 'entry pass')
    date = pd.to_datetime(df['dateutc'], errors='coerce').dt.strftime('%Y-%m-%d')
    return np.stack([
        date.fillna('—'),
        df['opponentTeam.name'].fillna('—').astype(str),
        df['minute'].fillna(0).astype(int).astype(str),
        pd.to_numeric(df['action_value'], errors='coerce').fillna(0)
          .map(lambda v: f'{v:+.4f}'),
        np.where(df['pass.accurate'] == True, 'completed', 'incomplete'),  # noqa: E712
        df['pass.recipient.name'].fillna('—').astype(str),
        np.array(tags, dtype=object),
    ], axis=1)


def plotly_box_passes_map(passes: pd.DataFrame, player_name: str,
                          max_arrows: int = 400) -> go.Figure:
    """Every pass into the attacking box, arrow-colored by GPA pass value."""
    df = passes.dropna(subset=['location.x', 'location.y',
                               'pass.endLocation.x',
                               'pass.endLocation.y']).copy()
    df = df.sort_values('dateutc')
    clipped = 0
    if len(df) > max_arrows:
        clipped = len(df) - max_arrows
        df = df.tail(max_arrows)

    vmax = _robust_vmax(df['action_value'])
    norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
    vals = pd.to_numeric(df['action_value'], errors='coerce').fillna(0)

    fig = go.Figure()
    annotations = []
    for (_, r), v in zip(df.iterrows(), vals):
        annotations.append(dict(
            x=r['pass.endLocation.y'], y=r['pass.endLocation.x'],
            ax=r['location.y'], ay=r['location.x'],
            xref='x', yref='y', axref='x', ayref='y',
            showarrow=True, arrowhead=3, arrowsize=0.9,
            arrowwidth=1.1 + 1.6 * min(abs(v) / vmax, 1.0),
            arrowcolor=mcolors.to_hex(VAL_CMAP(norm(v))),
            opacity=0.42 + 0.5 * min(abs(v) / vmax, 1.0),
            text='',
        ))

    fig.add_trace(go.Scatter(
        x=df['pass.endLocation.y'], y=df['pass.endLocation.x'],
        mode='markers',
        marker=dict(size=8, color=vals, colorscale=VAL_COLORSCALE,
                    cmin=-vmax, cmax=vmax,
                    line=dict(color='white', width=1),
                    colorbar=dict(title='Pass value<br>(GPA)', thickness=12,
                                  len=0.55, y=0.45)),
        customdata=_pass_hover(df),
        hovertemplate=('<b>%{customdata[6]}</b> · %{customdata[4]}<br>'
                       "%{customdata[0]} vs %{customdata[1]} · "
                       "%{customdata[2]}'<br>"
                       'value %{customdata[3]}<br>'
                       'to %{customdata[5]}<extra></extra>'),
    ))

    acc = (df['pass.accurate'] == True).mean() if len(df) else 0  # noqa: E712
    total_v = vals.sum()
    note = f' · showing latest {max_arrows}' if clipped else ''
    fig.add_annotation(
        text=(f'<b>{player_name}</b> — Passes into the Box<br>'
              f'{len(df)} passes · {acc:.0%} completed · '
              f'{total_v:+.3f} total value{note}'),
        xref='paper', yref='paper', x=0.5, y=1.085, showarrow=False,
        font=dict(size=13), align='center')
    fig.update_layout(annotations=fig.layout.annotations + tuple(annotations))
    return _pitch_layout(fig)


# Keep in sync with ROLE_PEAK in models/ratings/build_projection.py —
# display-only copy (shading the typical peak window on the fan chart).
ROLE_PEAK_DISPLAY = {
    'Striker': 25.0, 'Wide Attacker': 25.0, 'Advanced Midfielder': 26.0,
    'Deep Midfielder': 26.0, 'Wide Defender': 26.0, 'Central Defender': 27.5,
}


def _season_start_year(season_id, season_labels):
    """'2025/26' or '25-26' → 2025. None when the label doesn't parse."""
    import re
    lbl = str(season_labels.get(int(season_id), ''))
    m = re.match(r'^(\d{4})', lbl)
    if m:
        return int(m.group(1))
    m = re.match(r'^(\d{2})', lbl)
    if m:
        return 2000 + int(m.group(1))
    return None


def plotly_projection_fan(erows: pd.DataFrame, season_labels: dict,
                          player_name: str) -> "go.Figure | None":
    """Career ratings by season flowing into next season's projection with
    a ±1 SD uncertainty fan; the role's typical peak window is shaded
    (translated from peak AGE into calendar seasons for this player —
    the engine's `age` column is as-of, not per-season, so the x-axis is
    season years).

    erows: this player's rows from player_engine.parquet. Returns None if
    there isn't enough to draw.
    """
    df = erows.dropna(subset=['acp_rating']).copy()
    if df.empty:
        return None
    df['yr'] = df['seasonId'].map(
        lambda s: _season_start_year(s, season_labels))
    df = df.dropna(subset=['yr'])
    if df.empty:
        return None
    # one point per football year — keep the biggest-minutes row (a player
    # can have two same-year rows after a mid-season league switch)
    df = (df.sort_values('mins_played')
            .groupby('yr', as_index=False).tail(1)
            .sort_values('yr'))
    proj_rows = erows[erows.get('projection').notna()] \
        if 'projection' in erows.columns else pd.DataFrame()
    if proj_rows.empty:
        return None
    pr = proj_rows.iloc[-1]
    proj, band = float(pr['projection']), float(pr.get('band_sd', 5.0) or 5.0)
    role = str(pr.get('role', ''))
    last = df.iloc[-1]
    x_last, y_last = float(last['yr']), float(last['acp_rating'])
    x_proj = x_last + 1

    fig = go.Figure()
    # typical peak window for the role, mapped from age to seasons
    peak = ROLE_PEAK_DISPLAY.get(role)
    age_now = pd.to_numeric(pr.get('age'), errors='coerce')
    x_hi = x_proj
    if peak is not None and pd.notna(age_now):
        yr_at_peak = x_last + (peak - float(age_now))
        if df['yr'].min() - 1.5 <= yr_at_peak <= x_proj + 3:
            fig.add_vrect(x0=yr_at_peak - 1.0, x1=yr_at_peak + 1.0,
                          fillcolor='#2aa876', opacity=0.10, line_width=0,
                          annotation_text=f'typical {role} peak',
                          annotation_position='top right',
                          annotation_font=dict(size=10, color='#2aa876'))
            x_hi = max(x_hi, yr_at_peak + 1.0)
    # league-average reference
    fig.add_hline(y=50, line_dash='dot', line_color='rgba(128,128,128,0.6)',
                  annotation_text='league avg (50)',
                  annotation_font=dict(size=10, color='gray'))
    # uncertainty fan (±1 SD)
    fig.add_trace(go.Scatter(
        x=[x_last, x_proj, x_proj, x_last],
        y=[y_last, proj + band, proj - band, y_last],
        mode='none', fill='toself', fillcolor='rgba(42,168,118,0.16)',
        hoverinfo='skip', showlegend=False))
    # career history
    hovers = [f"{season_labels.get(int(r['seasonId']), r['seasonId'])} · "
              f"{r.get('league', '')}<br>rating {r['acp_rating']:.1f} · "
              f"{int(r['mins_played'])}'" for _, r in df.iterrows()]
    fig.add_trace(go.Scatter(
        x=df['yr'], y=df['acp_rating'], mode='lines+markers',
        line=dict(color='#3987e5', width=2.5), marker=dict(size=9),
        text=hovers, hovertemplate='%{text}<extra></extra>',
        name='seasons'))
    # dashed connector + projection diamond
    fig.add_trace(go.Scatter(
        x=[x_last, x_proj], y=[y_last, proj], mode='lines',
        line=dict(color='#2aa876', width=2, dash='dash'),
        hoverinfo='skip', showlegend=False))
    fig.add_trace(go.Scatter(
        x=[x_proj], y=[proj], mode='markers+text',
        marker=dict(symbol='diamond', size=14, color='#2aa876',
                    line=dict(color='white', width=1.5)),
        text=[f'{proj:.0f} ± {band:.0f}'],
        textposition=('bottom center' if y_last > proj else 'top center'),
        textfont=dict(size=12, color='#2aa876'),
        hovertemplate=(f'projection {proj:.1f} ± {band:.1f}'
                       '<extra></extra>'),
        name='projection'))

    # season labels on the x ticks ('25-26' style), projection tick marked
    tickvals = list(df['yr']) + [x_proj]
    _sample = str(season_labels.get(int(last['seasonId']), ''))
    _proj_lbl = (f'{int(x_proj)}/{(int(x_proj) + 1) % 100:02d} (proj)'
                 if '/' in _sample else
                 f'{int(x_proj) % 100:02d}-{(int(x_proj) + 1) % 100:02d} (proj)')
    ticktext = ([str(season_labels.get(int(r['seasonId']), int(r['yr'])))
                 for _, r in df.iterrows()] + [_proj_lbl])
    ys = list(df['acp_rating']) + [proj + band, proj - band, 50]
    fig.update_layout(
        height=340, showlegend=False,
        margin=dict(l=10, r=30, t=32, b=10),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(tickvals=tickvals, ticktext=ticktext, showgrid=False,
                   zeroline=False, fixedrange=True,
                   range=[df['yr'].min() - 0.4, x_hi + 0.6]),
        yaxis=dict(title='ACP rating', range=[min(ys) - 4, max(ys) + 4],
                   gridcolor='rgba(128,128,128,0.15)', zeroline=False,
                   fixedrange=True),
        title=dict(text=f'{player_name} — projection outlook',
                   font=dict(size=14)),
    )
    return fig


def mpl_box_passes_map(passes: pd.DataFrame, player_name: str):
    """Static matplotlib twin of plotly_box_passes_map (PDF export)."""
    from mplsoccer import Pitch

    df = passes.dropna(subset=['location.x', 'location.y',
                               'pass.endLocation.x',
                               'pass.endLocation.y']).copy()
    fig = plt.figure(figsize=(12, 8))
    fig.set_facecolor(PITCH_BG)
    pitch = Pitch(pitch_type='wyscout', pitch_color=PITCH_BG,
                  line_color='black', line_zorder=2, half=True)
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.82])
    pitch.draw(ax=ax)
    if df.empty:
        ax.text(75, 50, 'No passes into the box', ha='center', fontsize=12)
        return fig

    vmax = _robust_vmax(df['action_value'])
    norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
    vals = pd.to_numeric(df['action_value'], errors='coerce').fillna(0)
    pitch.arrows(df['location.x'], df['location.y'],
                 df['pass.endLocation.x'], df['pass.endLocation.y'],
                 color=VAL_CMAP(norm(vals)), width=1.6,
                 headwidth=6, headlength=6, alpha=0.8, ax=ax, zorder=3)

    sm = plt.cm.ScalarMappable(cmap=VAL_CMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.04,
                        pad=0.04, shrink=0.6)
    cbar.set_label('Pass value (GPA)', fontsize=11)
    cbar.outline.set_visible(False)
    acc = (df['pass.accurate'] == True).mean()  # noqa: E712
    ax.set_title(f'{player_name} — Passes into the Box\n'
                 f'{len(df)} passes · {acc:.0%} completed · '
                 f'{vals.sum():+.3f} total value',
                 fontsize=15, weight='bold', pad=15)
    return fig
