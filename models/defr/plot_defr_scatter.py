#!/usr/bin/env python3
"""StatsBomb-style DefR scatterplots, faceted by position group.

Each outfield panel plots a player at (expected def actions/90, actual/90).
The four quadrants are the StatsBomb archetypes (Front-Foot Aggressor,
System Absorber, Passive Line-Holder, Selective "Van Dijk zone"); the
dashed line is the position-typical actual/expected ratio (above = the
player over-performs the defensive demand placed on them).

The GK panel is different — keepers use the shot-stopping DefR
(goals prevented from post-shot xG), so it plots shot-stopping workload
(post-shot xG faced /90) vs goals prevented /90; above the y=0 line =
better than an average keeper.

Run from the Dashboard dir: python models/defr/plot_defr_scatter.py
Output: models/defr/plots/defr_scatter_2526.png
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
try:
    from adjustText import adjust_text
    _HAS_ADJUST = True
except Exception:
    _HAS_ADJUST = False

_HERE = Path(__file__).resolve().parent
_DASH = _HERE.parent.parent

# ---- palette ----
C_L3 = '#2E6FB0'      # Liga 3 blue
C_CP = '#E08A2B'      # Campeonato amber
C_LINE = '#9aa0a6'    # reference line
C_GRID = '#e8eaed'
C_TXT = '#202124'
C_QUAD = '#bdc1c6'    # quadrant labels (subtle)
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.edgecolor': '#dadce0',
    'axes.linewidth': 0.8,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

# ---- data ----
df = pd.read_parquet(_HERE / 'defr_per_player_season.parquet')
details = pd.read_pickle(_DASH / 'player_details.pkl')
nm = {int(p['playerId']): (f"{p.get('firstName','')} {p.get('lastName','')}").strip()
        or p.get('shortName', '?') for p in details}
gpa = pd.read_parquet(_DASH / 'gpa_player_season_values.parquet')
gn = gpa.groupby('playerId')['name'].first().to_dict()


def short_name(pid):
    n = nm.get(int(pid), '')
    if not n or n == '?':
        n = gn.get(int(pid), f'pid {pid}')
    parts = n.split()
    if len(parts) >= 2:
        # first initial + last name, e.g. "T. Castro"
        return f"{parts[0][0]}. {parts[-1]}"
    return n


GROUP = {'GK': 'GK', 'CB': 'CB', 'LCB': 'CB', 'RCB': 'CB', 'LB': 'FB', 'RB': 'FB',
          'LCM': 'CM', 'RCM': 'CM', 'DMF': 'CM', 'AM': 'AM/W', 'LAM': 'AM/W',
          'RAM': 'AM/W', 'LW': 'AM/W', 'RW': 'AM/W', 'ST': 'ST'}
df['grp'] = df['position'].map(GROUP)
df['exp90'] = df['expected_def_actions'] / (df['mins_played'] / 90)
df['act90'] = df['actual_def_actions'] / (df['mins_played'] / 90)
df['psxg90'] = df['gk_psxg_faced'] / (df['mins_played'] / 90)

SEASON_LBL = {191782: 'Liga 3', 191779: 'Camp'}
cur = df[df['seasonId'].isin([191782, 191779]) & (df['mins_played'] >= 700)].copy()
cur['name'] = cur['playerId'].apply(short_name)
cur['league'] = cur['seasonId'].map(SEASON_LBL)
cur['color'] = cur['league'].map({'Liga 3': C_L3, 'Camp': C_CP})


def style_axes(ax):
    ax.grid(True, color=C_GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8, colors='#5f6368', length=0)


def outfield_panel(ax, g):
    sub = cur[cur['grp'] == g].copy()
    if sub.empty:
        ax.set_visible(False)
        return
    x, y = sub['exp90'].values, sub['act90'].values
    mx, my = np.median(x), np.median(y)
    lim_x = np.percentile(x, 98) * 1.12
    # extra top headroom so the highest points + their labels clear the title
    lim_y = max(y.max(), np.percentile(y, 98)) * 1.30
    # quadrant crosshairs
    ax.axvline(mx, color='#f1f3f4', lw=8, zorder=0)
    ax.axhline(my, color='#f1f3f4', lw=8, zorder=0)
    # ratio reference line (position-typical actual/expected)
    slope = (my / mx) if mx > 0 else 1.0
    ax.plot([0, lim_x], [0, slope * lim_x], '--', color=C_LINE, lw=1.1, zorder=1)
    # points
    ax.scatter(x, y, c=sub['color'], s=34, alpha=0.55, edgecolors='white',
                linewidths=0.5, zorder=3)
    # quadrant archetype labels — placed in the true corners, subtle
    ax.text(0.985, 0.97, 'Front-foot\naggressor', transform=ax.transAxes,
             ha='right', va='top', fontsize=7.5, color=C_QUAD, style='italic', zorder=2)
    ax.text(0.985, 0.03, 'System\nabsorber', transform=ax.transAxes,
             ha='right', va='bottom', fontsize=7.5, color=C_QUAD, style='italic', zorder=2)
    ax.text(0.015, 0.03, 'Passive\nline-holder', transform=ax.transAxes,
             ha='left', va='bottom', fontsize=7.5, color=C_QUAD, style='italic', zorder=2)
    ax.text(0.015, 0.97, 'Selective', transform=ax.transAxes,
             ha='left', va='top', fontsize=7.5, color=C_QUAD, style='italic', zorder=2)
    # label the top over-performers (highest defr_adj), highlighted
    top = sub.nlargest(5, 'defr_adj')
    ax.scatter(top['exp90'], top['act90'], s=58, facecolors='none',
                edgecolors='#202124', linewidths=1.1, zorder=4)
    texts = []
    for _, r in top.iterrows():
        texts.append(ax.text(r['exp90'], r['act90'], r['name'], fontsize=8,
                              fontweight='bold', color=C_TXT, zorder=5))
    if _HAS_ADJUST and texts:
        adjust_text(texts, ax=ax, only_move={'text': 'xy'},
                     arrowprops=dict(arrowstyle='-', color='#9aa0a6', lw=0.6),
                     expand=(1.4, 1.6))
    ax.set_xlim(0, lim_x)
    ax.set_ylim(0, lim_y)
    ax.set_title(g, fontsize=13, fontweight='bold', color=C_TXT, pad=8)
    ax.set_xlabel('Expected defensive actions / 90', fontsize=8.5, color='#5f6368')
    ax.set_ylabel('Actual defensive actions / 90', fontsize=8.5, color='#5f6368')
    style_axes(ax)


def gk_panel(ax):
    sub = cur[(cur['grp'] == 'GK') & cur['psxg90'].notna()
                & cur['gk_gp_per90'].notna()].copy()
    if sub.empty:
        ax.set_visible(False)
        return
    x = sub['psxg90'].values           # shot-stopping workload
    y = sub['gk_gp_per90'].values      # goals prevented / 90
    lim_x = np.percentile(x, 98) * 1.15
    yabs = max(abs(y.min()), abs(y.max())) * 1.32
    # zero reference (average keeper)
    ax.axhline(0, color=C_LINE, lw=1.1, ls='--', zorder=1)
    ax.scatter(x, y, c=sub['color'], s=34, alpha=0.6, edgecolors='white',
                linewidths=0.5, zorder=3)
    ax.text(0.985, 0.97, 'Over-performs\n(saves shots)', transform=ax.transAxes,
             ha='right', va='top', fontsize=7.5, color=C_QUAD, style='italic')
    ax.text(0.985, 0.03, 'Under-performs', transform=ax.transAxes,
             ha='right', va='bottom', fontsize=7.5, color=C_QUAD, style='italic')
    top = sub.nlargest(5, 'gk_gp_per90')
    ax.scatter(top['psxg90'], top['gk_gp_per90'], s=58, facecolors='none',
                edgecolors='#202124', linewidths=1.1, zorder=4)
    texts = [ax.text(r['psxg90'], r['gk_gp_per90'], r['name'], fontsize=8,
                      fontweight='bold', color=C_TXT, zorder=5)
              for _, r in top.iterrows()]
    if _HAS_ADJUST and texts:
        adjust_text(texts, ax=ax, only_move={'text': 'xy'},
                     arrowprops=dict(arrowstyle='-', color='#9aa0a6', lw=0.6),
                     expand=(1.4, 1.6))
    ax.set_xlim(0, lim_x)
    ax.set_ylim(-yabs, yabs)
    ax.set_title('GK  (shot-stopping)', fontsize=13, fontweight='bold',
                  color=C_TXT, pad=8)
    ax.set_xlabel('Post-shot xG faced / 90  (workload)', fontsize=8.5, color='#5f6368')
    ax.set_ylabel('Goals prevented / 90', fontsize=8.5, color='#5f6368')
    style_axes(ax)


# ---- figure ----
fig, axes = plt.subplots(2, 3, figsize=(18, 11.5))
fig.patch.set_facecolor('white')
fig.suptitle('Defensive Responsibility (DefR) — 25/26 Liga 3 + Campeonato',
              fontsize=18, fontweight='bold', color=C_TXT, x=0.5, y=0.985)
fig.text(0.5, 0.945,
          'Outfield: expected vs actual defensive actions per 90 (≥700 min). '
          'Above the dashed line = does more defending than the role demands. '
          'GK: shot-stopping (goals prevented).',
          ha='center', fontsize=10.5, color='#5f6368')

for ax, g in zip(axes.flat[:5], ['ST', 'AM/W', 'CM', 'FB', 'CB']):
    outfield_panel(ax, g)
gk_panel(axes.flat[5])

handles = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=C_L3, markersize=9, label='Liga 3'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=C_CP, markersize=9, label='Campeonato'),
    Line2D([0], [0], marker='o', color='w', markeredgecolor='#202124',
            markerfacecolor='none', markersize=9, markeredgewidth=1.1,
            label='Top over-performer (labelled)'),
    Line2D([0], [0], ls='--', color=C_LINE, label='Position-typical level'),
]
fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=10,
            frameon=False, bbox_to_anchor=(0.5, 0.005))

plt.tight_layout(rect=[0, 0.04, 1, 0.92])
plt.subplots_adjust(hspace=0.42, wspace=0.24)
out = _HERE / 'plots' / 'defr_scatter_2526.png'
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=130, bbox_inches='tight', facecolor='white')
print(f"Saved {out}")
