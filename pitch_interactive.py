"""Interactive (Plotly) pitch visuals for the Player Profile:

- plotly_shot_map        — hoverable upgrade of the static shot map
- plotly_box_passes_map  — every pass into the attacking box, arrows
                            colored by GPA pass value
- mpl_box_passes_map     — static matplotlib twin for PDF export
- plotly_projection_fan  — career ratings flowing into next season's
                            projection, with a ±1 SD fan
- mpl_projection_fan     — static matplotlib twin for PDF export

Coordinates are Wyscout 100x100 (x toward opponent goal). Both plotly
charts render a portrait half-pitch, goal at the top:
    plot_x = event y   ·   plot_y = event x
"""
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import theme

PITCH_BG = theme.FIGURE_BG
LINE_C = theme.PITCH_LINE_PLOTLY

# Same anchors as the static shot maps (theme.XG_COLORS / XG_NODES)
XG_MAX = theme.XG_MAX
XG_COLORSCALE = theme.XG_COLORSCALE

# StatsBomb-style shape encoding: marker symbol = the shot-creating
# action. Labels must match label_sca() in app.py's Shots section;
# anything unlisted falls back to 'Other'.
SCA_SYMBOLS = [
    ('Cross', 'triangle-up'),
    ('Through Pass', 'star'),
    ('Deep Completion', 'diamond'),
    ('Pass', 'circle'),
    ('Dribble/Duel', 'square'),
    ('Carry', 'hexagon'),
    ('Clearance', 'triangle-down'),
    ('Interception', 'x'),
    ('Recovery/None', 'cross'),
    ('Other', 'pentagon'),
]
_SCA_SYMBOL_MAP = dict(SCA_SYMBOLS)

# Diverging pass-value scale with a warm-gray midpoint (visible on cream):
# cold blue = value conceded/negative, warm red = value created
VAL_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'gpa_div', [theme.VALUE_NEG, theme.VALUE_MID, theme.VALUE_POS])
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
                    height: int = 660) -> go.Figure:
    """Interactive StatsBomb-style shot map: marker SHAPE = creating
    action, color = xG, green ring = goal. Uniform marker size, slightly
    translucent so overlapping shots stay readable; shape legend beneath
    the pitch. Expects the processed shot_log from the Shots section
    (Shot Number, Date, Opponent, Result, xG, Body Part, Phase, SCA,
    location.x/y, shot.isGoal)."""
    df = shot_log.dropna(subset=['location.x', 'location.y']).copy()
    df['xG'] = pd.to_numeric(df.get('xG', df.get('shot.xg')),
                             errors='coerce').fillna(0.0)
    if 'SCA' in df.columns:
        df['_sca'] = df['SCA'].fillna('Recovery/None').astype(str)
        df.loc[~df['_sca'].isin(_SCA_SYMBOL_MAP), '_sca'] = 'Other'
    else:
        df['_sca'] = 'Other'

    fig = go.Figure()
    present = [lbl for lbl, _ in SCA_SYMBOLS if (df['_sca'] == lbl).any()]

    # neutral legend swatches — the data traces carry per-point xG colors,
    # which would render legend symbols in arbitrary colors
    for lbl in present:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(symbol=_SCA_SYMBOL_MAP[lbl], size=11,
                        color='#7d8a80'),
            name=lbl, showlegend=True, hoverinfo='skip'))

    for lbl in present:
        sub = df[df['_sca'] == lbl]
        is_goal = sub.get('shot.isGoal') == True  # noqa: E712
        custom = np.stack([
            sub.get('Date', pd.Series('—', index=sub.index)).astype(str),
            sub.get('Opponent', pd.Series('—', index=sub.index)).astype(str),
            sub.get('minute', pd.Series(np.nan, index=sub.index)).astype(str),
            sub.get('Result', pd.Series('—', index=sub.index)).astype(str),
            sub['xG'].round(3).astype(str),
            sub.get('Body Part', pd.Series('—', index=sub.index)).astype(str),
            sub.get('Phase', pd.Series('—', index=sub.index)).astype(str),
        ], axis=1)
        fig.add_trace(go.Scatter(
            x=sub['location.y'], y=sub['location.x'],
            mode='markers+text',
            text=sub.get('Shot Number',
                         pd.Series('', index=sub.index)).astype(str),
            textfont=dict(color='white', size=8.5),
            marker=dict(
                symbol=_SCA_SYMBOL_MAP[lbl], size=20, opacity=0.75,
                color=sub['xG'], coloraxis='coloraxis',
                line=dict(
                    color=np.where(is_goal, '#1a7a2e', 'rgba(0,0,0,0.45)'),
                    width=np.where(is_goal, 3, 1)),
            ),
            customdata=custom, showlegend=False,
            hovertemplate=('<b>#%{text} · %{customdata[3]}</b><br>'
                           '%{customdata[0]} vs %{customdata[1]} · '
                           "%{customdata[2]}'<br>"
                           'xG %{customdata[4]} · %{customdata[5]}<br>'
                           f'created by: {lbl} · ' + '%{customdata[6]}'
                           '<extra></extra>'),
        ))

    goals = int((df.get('shot.isGoal') == True).sum())  # noqa: E712
    # StatsBomb-style left-aligned two-line header
    fig.add_annotation(
        text=f'<b>{player_name} — Season Shot Map</b>',
        xref='paper', yref='paper', x=0.01, y=1.10, showarrow=False,
        font=dict(size=15), align='left', xanchor='left')
    fig.add_annotation(
        text=(f'{len(df)} non-penalty shots · {goals} goals · '
              f'{df["xG"].sum():.2f} xG · shape = creating action · '
              f'color = xG · ring = goal'),
        xref='paper', yref='paper', x=0.01, y=1.055, showarrow=False,
        font=dict(size=11.5, color='#5a5a5a'), align='left',
        xanchor='left')
    _pitch_layout(fig, height=height)
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation='h', x=0.5, xanchor='center',
                    y=-0.01, yanchor='top', font=dict(size=11),
                    itemclick=False, itemdoubleclick=False),
        coloraxis=dict(colorscale=XG_COLORSCALE, cmin=0, cmax=XG_MAX,
                       colorbar=dict(title='xG', thickness=12, len=0.5,
                                     y=0.5)),
        margin=dict(l=10, r=10, t=58, b=42))
    return fig


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
    band = float(pr.get('band_sd', 5.0) or 5.0)
    role = str(pr.get('role', ''))
    # cross-league careers plot on the L3-equivalent (abs) scale so the
    # line doesn't jump between league baselines; CAMP seasons are
    # discounted by the league-conversion delta
    use_abs = (df['league'].astype(str).nunique() > 1
               and 'acp_rating_abs' in df.columns)
    if use_abs:
        df['_val'] = pd.to_numeric(df['acp_rating_abs'],
                                   errors='coerce').fillna(df['acp_rating'])
        proj = (float(pr['projection_abs'])
                if pd.notna(pr.get('projection_abs'))
                else float(pr['projection']))
    else:
        df['_val'] = df['acp_rating']
        proj = float(pr['projection'])
    last = df.iloc[-1]
    x_last, y_last = float(last['yr']), float(last['_val'])
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
                  annotation_position='bottom left',
                  annotation_font=dict(size=10, color='gray'))
    # uncertainty fan (±1 SD)
    fig.add_trace(go.Scatter(
        x=[x_last, x_proj, x_proj, x_last],
        y=[y_last, proj + band, proj - band, y_last],
        mode='none', fill='toself', fillcolor='rgba(42,168,118,0.16)',
        hoverinfo='skip', showlegend=False))
    # career history — solid segments between consecutive seasons, FADED
    # across gaps (dashed is reserved for the projection connector; with
    # every season shown on the axis, a missing dot + faded bridge reads
    # as "no data those years")
    _yrs = df['yr'].tolist()
    _vals = df['_val'].tolist()
    for _i in range(len(_yrs) - 1):
        _gap = _yrs[_i + 1] - _yrs[_i]
        fig.add_trace(go.Scatter(
            x=[_yrs[_i], _yrs[_i + 1]], y=[_vals[_i], _vals[_i + 1]],
            mode='lines',
            line=dict(color=('rgba(57,135,229,0.30)' if _gap > 1
                             else '#3987e5'),
                      width=2 if _gap > 1 else 2.5),
            hoverinfo='skip', showlegend=False))

    def _hover(r):
        lbl = (f"{season_labels.get(int(r['seasonId']), r['seasonId'])} · "
               f"{r.get('league', '')}<br>")
        native = float(r['acp_rating'])
        val = float(r['_val'])
        if abs(native - val) > 0.05:
            lbl += f"rating {native:.1f} → L3-eq {val:.1f}"
        else:
            lbl += f"rating {val:.1f}"
        return lbl + f" · {int(r['mins_played'])}'"

    hovers = [_hover(r) for _, r in df.iterrows()]
    fig.add_trace(go.Scatter(
        x=df['yr'], y=df['_val'], mode='markers',
        marker=dict(size=9, color='#3987e5'),
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
    # evidence % — the data weight behind the projection's starting point
    _w_ev = pd.to_numeric(pr.get('w_evidence'), errors='coerce')
    if pd.notna(_w_ev):
        fig.add_annotation(
            x=x_proj, y=proj,
            yshift=(-34 if y_last > proj else 34),
            text=f'evidence {float(_w_ev):.0%}', showarrow=False,
            font=dict(size=10.5, color='rgba(128,128,128,0.9)'))

    # x ticks: EVERY season from first career year to the projection —
    # including years the player has no data for — with league next to
    # played seasons and the player's age under each tick (engine age is
    # as-of the latest season, so per-season age = age_now - years back)
    _sample = str(season_labels.get(int(last['seasonId']), ''))
    _slash = '/' in _sample

    def _season_lbl(y):
        y = int(y)
        return (f'{y}/{(y + 1) % 100:02d}' if _slash
                else f'{y % 100:02d}-{(y + 1) % 100:02d}')

    def _with_age(lbl, yr):
        if pd.isna(age_now):
            return lbl
        return f'{lbl}<br>age {float(age_now) - (x_last - float(yr)):.1f}'

    _row_by_yr = {int(r['yr']): r for _, r in df.iterrows()}
    _years = list(range(int(df['yr'].min()), int(x_last) + 1))

    def _tick_lbl(y):
        r = _row_by_yr.get(int(y))
        if r is not None:
            base = str(season_labels.get(int(r['seasonId']), _season_lbl(y)))
            base += f" · {r.get('league', '')}"
        else:
            base = _season_lbl(y)
        return _with_age(base, y)

    _proj_lbl = _season_lbl(x_proj) + ' (proj)'
    tickvals = _years + [x_proj]
    ticktext = [_tick_lbl(y) for y in _years] + [_with_age(_proj_lbl, x_proj)]
    ys = list(df['_val']) + [proj + band, proj - band, 50]
    fig.update_layout(
        height=340, showlegend=False,
        margin=dict(l=10, r=30, t=32, b=10),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(tickvals=tickvals, ticktext=ticktext, showgrid=False,
                   zeroline=False, fixedrange=True,
                   range=[df['yr'].min() - 0.4, x_hi + 0.6]),
        yaxis=dict(title=('ACP rating (L3-equivalent)' if use_abs
                          else 'ACP rating'),
                   range=[min(ys) - 4, max(ys) + 4],
                   gridcolor='rgba(128,128,128,0.15)', zeroline=False,
                   fixedrange=True),
        title=dict(text=f'{player_name} — projection outlook',
                   font=dict(size=14)),
    )
    return fig


def mpl_projection_fan(erows: pd.DataFrame, season_labels: dict,
                       player_name: str):
    """Static matplotlib twin of plotly_projection_fan (PDF export).

    Deliberately mirrors that function's decisions rather than making its
    own: the same L3-equivalent scale for cross-league careers, the same
    biggest-minutes row per football year, the same faded bridge across
    seasons with no data, the same peak-age window. The two are meant to
    show the same picture — if you change one, change the other.

    Returns None when there is not enough to draw (no rated seasons, no
    parseable season labels, or no projection), matching the plotly twin.
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
    proj_rows = (erows[erows.get('projection').notna()]
                 if 'projection' in erows.columns else pd.DataFrame())
    if proj_rows.empty:
        return None
    pr = proj_rows.iloc[-1]
    band = float(pr.get('band_sd', 5.0) or 5.0)
    role = str(pr.get('role', ''))
    # cross-league careers plot on the L3-equivalent (abs) scale so the line
    # does not jump between league baselines
    use_abs = (df['league'].astype(str).nunique() > 1
               and 'acp_rating_abs' in df.columns)
    if use_abs:
        df['_val'] = pd.to_numeric(df['acp_rating_abs'],
                                   errors='coerce').fillna(df['acp_rating'])
        proj = (float(pr['projection_abs'])
                if pd.notna(pr.get('projection_abs'))
                else float(pr['projection']))
    else:
        df['_val'] = df['acp_rating']
        proj = float(pr['projection'])
    last = df.iloc[-1]
    x_last, y_last = float(last['yr']), float(last['_val'])
    x_proj = x_last + 1

    fig = plt.figure(figsize=(12, 5))
    fig.set_facecolor(PITCH_BG)
    ax = fig.add_subplot(111)
    ax.set_facecolor(PITCH_BG)

    # typical peak window for the role, mapped from age to seasons
    peak = ROLE_PEAK_DISPLAY.get(role)
    age_now = pd.to_numeric(pr.get('age'), errors='coerce')
    x_hi = x_proj
    if peak is not None and pd.notna(age_now):
        yr_at_peak = x_last + (peak - float(age_now))
        if df['yr'].min() - 1.5 <= yr_at_peak <= x_proj + 3:
            ax.axvspan(yr_at_peak - 1.0, yr_at_peak + 1.0,
                       color='#2aa876', alpha=0.10, lw=0, zorder=0)
            ax.text(yr_at_peak + 1.0, 0.985, f'typical {role} peak',
                    transform=ax.get_xaxis_transform(), ha='right',
                    va='top', fontsize=9, color='#2aa876')
            x_hi = max(x_hi, yr_at_peak + 1.0)

    x_lo = df['yr'].min() - 0.4
    # league-average reference
    ax.axhline(50, ls=':', color='gray', alpha=0.6, lw=1, zorder=1)
    ax.text(x_lo, 50, ' league avg (50)', ha='left', va='top',
            fontsize=9, color='gray')
    # uncertainty fan (±1 SD)
    ax.fill([x_last, x_proj, x_proj, x_last],
            [y_last, proj + band, proj - band, y_last],
            color='#2aa876', alpha=0.16, lw=0, zorder=1)
    # career history — solid between consecutive seasons, FADED across gaps
    # (dashed is reserved for the projection connector; a missing dot plus a
    # faded bridge reads as "no data those years")
    _yrs = df['yr'].tolist()
    _vals = df['_val'].tolist()
    for _i in range(len(_yrs) - 1):
        _gap = _yrs[_i + 1] - _yrs[_i]
        ax.plot([_yrs[_i], _yrs[_i + 1]], [_vals[_i], _vals[_i + 1]],
                color='#3987e5', alpha=0.30 if _gap > 1 else 1.0,
                lw=2 if _gap > 1 else 2.5, solid_capstyle='round', zorder=2)
    ax.scatter(df['yr'], df['_val'], s=55, color='#3987e5', zorder=3)
    # dashed connector + projection diamond
    ax.plot([x_last, x_proj], [y_last, proj], ls='--', color='#2aa876',
            lw=2, zorder=2)
    ax.scatter([x_proj], [proj], marker='D', s=130, color='#2aa876',
               edgecolors='white', linewidths=1.5, zorder=4)
    _below = y_last > proj
    ax.annotate(f'{proj:.0f} ± {band:.0f}', (x_proj, proj),
                textcoords='offset points',
                xytext=(0, -20 if _below else 14), ha='center',
                fontsize=11, weight='bold', color='#2aa876')
    # evidence % — the data weight behind the projection's starting point
    _w_ev = pd.to_numeric(pr.get('w_evidence'), errors='coerce')
    if pd.notna(_w_ev):
        ax.annotate(f'evidence {float(_w_ev):.0%}', (x_proj, proj),
                    textcoords='offset points',
                    xytext=(0, -34 if _below else 30), ha='center',
                    fontsize=9, color='gray')

    # x ticks: EVERY season from the first career year to the projection,
    # including years with no data, with league next to played seasons and
    # the player's age under each tick (engine age is as-of the latest
    # season, so per-season age = age_now - years back)
    _sample = str(season_labels.get(int(last['seasonId']), ''))
    _slash = '/' in _sample

    def _season_lbl(y):
        y = int(y)
        return (f'{y}/{(y + 1) % 100:02d}' if _slash
                else f'{y % 100:02d}-{(y + 1) % 100:02d}')

    def _with_age(lbl, yr):
        if pd.isna(age_now):
            return lbl
        return f'{lbl}\nage {float(age_now) - (x_last - float(yr)):.1f}'

    _row_by_yr = {int(r['yr']): r for _, r in df.iterrows()}
    _years = list(range(int(df['yr'].min()), int(x_last) + 1))

    def _tick_lbl(y):
        r = _row_by_yr.get(int(y))
        if r is not None:
            base = str(season_labels.get(int(r['seasonId']), _season_lbl(y)))
            base += f" · {r.get('league', '')}"
        else:
            base = _season_lbl(y)
        return _with_age(base, y)

    ax.set_xticks(_years + [x_proj])
    ax.set_xticklabels([_tick_lbl(y) for y in _years]
                       + [_with_age(_season_lbl(x_proj) + ' (proj)', x_proj)],
                       fontsize=8)
    ys = list(df['_val']) + [proj + band, proj - band, 50]
    ax.set_xlim(x_lo, x_hi + 0.6)
    ax.set_ylim(min(ys) - 4, max(ys) + 4)
    ax.set_ylabel('ACP rating (L3-equivalent)' if use_abs else 'ACP rating',
                  fontsize=10)
    ax.set_title(f'{player_name} — projection outlook',
                 fontsize=14, weight='bold', pad=12)
    ax.grid(axis='y', color='gray', alpha=0.15)
    ax.set_axisbelow(True)
    for _s in ('top', 'right'):
        ax.spines[_s].set_visible(False)
    fig.tight_layout()
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
