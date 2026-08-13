"""OBV-powered visualizations (engine action values).

Charts built on the GPA engine's exported aggregates:
  obv_match_minute.parquet   -> plot_obv_momentum
  team_phase_profile.parquet -> plot_phase_profile
  obv_team_season.parquet    -> plot_team_obv_categories

House style: cream field, one green accent for the focus team, muted grey
league context, direct labels over legends.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BG = '#f5f1e9'
INK = '#1c2321'
ACCENT = '#1a472a'      # focus team green
OPP = '#a63a46'         # opponent / against
GREY = '#8d968e'        # league context
GRID = '#ddd8c9'

PHASE_LABELS = {
    'buildup': 'Buildup',
    'progression': 'Progression',
    'finishing': 'Finishing',
    'fast_break': 'Fast break',
    'set_piece': 'Set piece',
}
PHASE_ORDER = ['buildup', 'progression', 'finishing', 'fast_break', 'set_piece']

OBV_CATEGORIES = {
    'Passing': ['pass'],
    'Shooting': ['shot'],
    'Carrying': ['acceleration', 'touch'],
    'Set pieces': ['corner', 'free_kick', 'throw_in', 'goal_kick', 'penalty'],
    'Defending': ['duel', 'interception', 'clearance'],
    'Goalkeeping': ['goalkeeper_exit'],
    'Discipline': ['infraction', 'offside', 'own_goal'],
}


def _style_axes(ax):
    ax.set_facecolor(BG)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=8)
    ax.grid(axis='x', color=GRID, linewidth=0.6, alpha=0.7)


# ---------------------------------------------------------------------------
# 1. Momentum
# ---------------------------------------------------------------------------
def plot_obv_momentum(minute_df, home_id, away_id, home_name, away_name,
                      goals=None, smooth=3):
    """Diverging per-minute OBV momentum chart for one match.

    minute_df: rows for ONE match from obv_match_minute.parquet
    goals: optional list of dicts {minute, teamId} for goal markers
    """
    if minute_df is None or minute_df.empty:
        return None
    df = minute_df.copy()
    df['minute'] = pd.to_numeric(df['minute'], errors='coerce')
    df = df.dropna(subset=['minute'])
    max_min = int(max(90, df['minute'].max()))

    def team_series(tid):
        s = (df[df['teamId'] == tid].groupby('minute')['obv'].sum()
             .reindex(range(max_min + 1), fill_value=0.0))
        if smooth and smooth > 1:
            s = s.rolling(smooth, center=True, min_periods=1).mean()
        return s.clip(lower=0)  # each side shows its own positive threat

    home_s, away_s = team_series(home_id), team_series(away_id)
    top = float(max(home_s.max(), away_s.max(), 1e-3)) * 1.15

    fig, ax = plt.subplots(figsize=(10, 4.4), facecolor=BG)
    _style_axes(ax)
    ax.grid(False)
    mins = np.arange(max_min + 1)
    ax.bar(mins, home_s.values, width=0.85, color=ACCENT, zorder=3)
    ax.bar(mins, -away_s.values, width=0.85, color=OPP, alpha=0.85, zorder=3)
    ax.axhline(0, color=INK, linewidth=0.8)

    ht = 45.5 if max_min >= 90 else max_min / 2
    ax.axvline(ht, color=GREY, linewidth=0.8, linestyle='--', alpha=0.8)
    ax.text(ht + 0.8, top * 0.88, 'HT', ha='left', va='center', fontsize=8, color=GREY)

    for g in (goals or []):
        gm = g.get('minute')
        if gm is None:
            continue
        is_home = g.get('teamId') == home_id
        y = top * 0.88 if is_home else -top * 0.88
        ax.scatter([gm], [y], s=60, color=ACCENT if is_home else OPP,
                   edgecolors='white', linewidths=1.4, zorder=6)
        ax.annotate(f"{int(gm)}'", (gm, y), textcoords='offset points',
                    xytext=(0, 9 if is_home else -13), ha='center', fontsize=7.5,
                    fontweight='bold', color=ACCENT if is_home else OPP, zorder=7)

    ax.text(0.005, 0.98, home_name, transform=ax.transAxes, ha='left', va='top',
            fontsize=10, fontweight='bold', color=ACCENT)
    ax.text(0.005, 0.02, away_name, transform=ax.transAxes, ha='left', va='bottom',
            fontsize=10, fontweight='bold', color=OPP)
    ax.set_ylim(-top, top)
    ax.set_xlim(-1, max_min + 1)
    ax.set_yticks([])
    ax.set_xticks([0, 15, 30, 45, 60, 75, 90][:max(2, max_min // 15 + 1)])
    ax.set_xlabel('Match minute', fontsize=8, color=GREY)
    ax.set_title('Momentum — on-ball value created per minute (engine OBV)',
                 fontsize=11, fontweight='bold', color=INK, loc='left', pad=10)
    fig.text(0.99, 0.01, 'Data: Wyscout · Model: ACP engine',
             ha='right', fontsize=6.5, color=GREY)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Phase profile (futi-style)
# ---------------------------------------------------------------------------
def plot_phase_profile(profile_df, team_name):
    """Team phases vs the league: volume, success, value.

    profile_df: team_phase_profile rows for ONE season+competition (all teams).
    """
    if profile_df is None or profile_df.empty:
        return None
    df = profile_df[profile_df['phase'].isin(PHASE_ORDER)].copy()
    if team_name not in set(df['teamName']):
        return None

    def _vol_fmt(v):
        return f'{v:.1f}' if v < 10 else f'{v:.0f}'

    metrics = [('n_per90', 'Volume  (per 90)', _vol_fmt),
               ('success_rate', 'Success rate', '{:.0%}'.format),
               ('obv_per_phase', 'OBV per phase', '{:+.3f}'.format)]

    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2), facecolor=BG, sharey=True)
    ys = {ph: i for i, ph in enumerate(reversed(PHASE_ORDER))}

    for ax, (col, label, fmt) in zip(axes, metrics):
        _style_axes(ax)
        for ph in PHASE_ORDER:
            y = ys[ph]
            league = df[df['phase'] == ph]
            vals = league[col].dropna()
            if vals.empty:
                continue
            ax.plot([vals.min(), vals.max()], [y, y], color=GRID,
                    linewidth=5, solid_capstyle='round', zorder=1)
            ax.scatter(vals, [y] * len(vals), s=14, color=GREY, alpha=0.55, zorder=2)
            team_val = league.loc[league['teamName'] == team_name, col]
            if not team_val.empty:
                v = float(team_val.iloc[0])
                ax.scatter([v], [y], s=90, color=ACCENT, edgecolors='white',
                           linewidths=1.2, zorder=4)
                ax.annotate(fmt(v), (v, y), textcoords='offset points',
                            xytext=(0, 9), ha='center', fontsize=8,
                            fontweight='bold', color=ACCENT, zorder=5)
        ax.set_title(label, fontsize=9.5, fontweight='bold', color=INK, loc='left')
        ax.set_yticks(list(ys.values()))
        ax.set_yticklabels([PHASE_LABELS[ph] for ph in reversed(PHASE_ORDER)],
                           fontsize=9, color=INK)
        ax.set_ylim(-0.6, len(PHASE_ORDER) - 0.4)

    fig.suptitle(f'{team_name} — phases of play vs the league',
                 fontsize=12, fontweight='bold', color=INK, x=0.01, ha='left')
    fig.text(0.01, 0.925,
             'Grey dots = every league team. Phases: possession segments by pitch third; '
             'fast breaks & set pieces from Wyscout possession tags.',
             fontsize=7.5, color=GREY)
    fig.text(0.99, 0.01, 'Data: Wyscout · Model: ACP engine phases v1',
             ha='right', fontsize=6.5, color=GREY)
    fig.tight_layout(rect=[0, 0.02, 1, 0.90])
    return fig


# ---------------------------------------------------------------------------
# 3. Team OBV total + by category
# ---------------------------------------------------------------------------
def plot_team_obv_categories(team_season_df, team_id, team_name):
    """Team season OBV per 90 by action category, vs league average.

    team_season_df: obv_team_season rows for ONE season+competition (all teams).
    """
    if team_season_df is None or team_season_df.empty:
        return None
    df = team_season_df.copy()
    cat_map = {t: cat for cat, types in OBV_CATEGORIES.items() for t in types}
    df['category'] = df['actionType'].map(cat_map).fillna('Other')

    matches = df.groupby('teamId')['matches'].max()
    cat = (df.groupby(['teamId', 'category'], observed=True)['obv'].sum()
           .reset_index()
           .join(matches.rename('matches'), on='teamId'))
    cat['obv_per90'] = cat['obv'] / cat['matches']

    team_rows = cat[cat['teamId'] == team_id]
    if team_rows.empty:
        return None
    league_avg = cat.groupby('category', observed=True)['obv_per90'].mean()

    order = team_rows.set_index('category')['obv_per90'].sort_values()
    cats = list(order.index)
    ys = np.arange(len(cats))

    fig, ax = plt.subplots(figsize=(9, 4.6), facecolor=BG)
    _style_axes(ax)
    colors = [ACCENT if v >= 0 else OPP for v in order.values]
    ax.barh(ys, order.values, height=0.62, color=colors, zorder=3)
    ax.scatter(league_avg.reindex(cats).values, ys, marker='|', s=260,
               color=INK, linewidths=1.6, zorder=4)
    for y, v in zip(ys, order.values):
        ax.annotate(f'{v:+.2f}', (v, y), textcoords='offset points',
                    xytext=(6 if v >= 0 else -6, 0),
                    ha='left' if v >= 0 else 'right', va='center',
                    fontsize=8.5, fontweight='bold',
                    color=ACCENT if v >= 0 else OPP)
    ax.axvline(0, color=INK, linewidth=0.8)
    ax.set_yticks(ys)
    ax.set_yticklabels(cats, fontsize=9.5, color=INK)

    total90 = float(team_rows['obv'].sum() / team_rows['matches'].max())
    league_total90 = float(cat.groupby('teamId')
                           .apply(lambda g: g['obv'].sum() / g['matches'].max(),
                                  include_groups=False).mean())
    ax.set_title(
        f'{team_name} — on-ball value by category  '
        f'(total {total90:+.2f} OBV/90, league avg {league_total90:+.2f})',
        fontsize=11, fontweight='bold', color=INK, loc='left', pad=16)
    fig.text(0.99, 0.955, 'Black tick = league average per category',
             ha='right', fontsize=7.5, color=GREY)
    fig.text(0.99, 0.01, 'Data: Wyscout · Model: ACP engine',
             ha='right', fontsize=6.5, color=GREY)
    ax.set_xlabel('OBV per 90', fontsize=8, color=GREY)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig
