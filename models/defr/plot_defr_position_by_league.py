#!/usr/bin/env python3
"""Per-league DefR scatter charts (StatsBomb-article style) for the
remaining outfield position groups — DM, CM, Attacking Mids & Wingers,
and Strikers — one chart per (group × league). Axes zoom to the data
range; labels the most interesting players (outliers + extremes).

CB and FB have their own dedicated scripts (plot_defr_centerback.py,
plot_defr_cb_by_league.py, plot_defr_fb_by_league.py).

Run from the Dashboard dir:
    python models/defr/plot_defr_position_by_league.py
Outputs: plots/defr_<grp>_<league>_2526.png
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    from adjustText import adjust_text; HAS = True
except Exception:
    HAS = False

_HERE = Path(__file__).resolve().parent
_DASH = _HERE.parent.parent

df = pd.read_parquet(_HERE / 'defr_per_player_season.parquet')
details = pd.read_pickle(_DASH / 'player_details.pkl')
nm = {int(p['playerId']): (f"{p.get('firstName','')} {p.get('lastName','')}").strip()
        or p.get('shortName', '?') for p in details}
gpa = pd.read_parquet(_DASH / 'gpa_player_season_values.parquet')
gn = gpa.groupby('playerId')['name'].first().to_dict()


def short(pid):
    n = nm.get(int(pid), '')
    if not n or n == '?':
        n = gn.get(int(pid), str(pid))
    parts = n.split()
    return f"{parts[0][0]}. {parts[-1]}" if len(parts) >= 2 else n


C_TXT='#202124'; C_LINE='#9aa0a6'; C_QUAD='#b0b4b8'; C_GRID='#e8eaed'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.edgecolor': '#dadce0',
                       'figure.facecolor': 'white', 'axes.facecolor': 'white'})

# group key, slot codes, pretty label, n_labels, optional caveat
GROUPS = [
    ('dm',  ['DMF'],                      'Defensive Midfielders', 24,
       "Note: the single-pivot 'DMF' slot has artificially low expected "
       "(model dilution) — read the RANKING, not the absolute expected."),
    ('cm',  ['LCM', 'RCM'],               'Central Midfielders',   30, None),
    ('amw', ['AM', 'LAM', 'RAM', 'LW', 'RW'], 'Attacking Mids & Wingers', 30, None),
    ('st',  ['ST'],                       'Strikers',              28, None),
]
LEAGUES = [(191782, 'Liga 3', '#2E6FB0'), (191779, 'Campeonato', '#E08A2B')]


def make(slots, pretty, n_lab, caveat, sid, league, color, fname):
    cur = df[df['position'].isin(slots) & (df['seasonId'] == sid)
               & (df['mins_played'] >= 700)].copy()
    if len(cur) < 8:
        print(f"  skip {fname}: only {len(cur)} players")
        return
    cur['exp90'] = cur['expected_def_actions'] / (cur['mins_played'] / 90)
    cur['act90'] = cur['actual_def_actions'] / (cur['mins_played'] / 90)
    cur['name'] = cur['playerId'].apply(short)
    fig, ax = plt.subplots(figsize=(14, 11.5))
    x, y = cur['exp90'].values, cur['act90'].values
    mx, my = np.median(x), np.median(y)
    padx = (x.max() - x.min()) * 0.08 or 1.0
    pady = (y.max() - y.min()) * 0.08 or 1.0
    xlo, xhi = x.min() - padx, x.max() + padx
    ylo, yhi = y.min() - pady, y.max() + pady
    ax.axvline(mx, color='#f1f3f4', lw=10, zorder=0)
    ax.axhline(my, color='#f1f3f4', lw=10, zorder=0)
    ax.grid(True, color=C_GRID, lw=0.7, zorder=0); ax.set_axisbelow(True)
    slope = my / mx if mx > 0 else 1
    ax.plot([0, xhi * 1.3], [0, slope * xhi * 1.3], '--', color=C_LINE, lw=1.3, zorder=1)
    ax.scatter(x, y, c=color, s=110, alpha=0.65, edgecolors='white', linewidths=1.0, zorder=3)
    ax.text(0.985, 0.975, 'OVER-DELIVERS\nhigh demand · does more', transform=ax.transAxes,
             ha='right', va='top', fontsize=10, color=C_QUAD, style='italic', linespacing=1.4)
    ax.text(0.985, 0.025, 'UNDER-DELIVERS\nhigh demand · does less', transform=ax.transAxes,
             ha='right', va='bottom', fontsize=10, color=C_QUAD, style='italic', linespacing=1.4)
    ax.text(0.015, 0.025, 'LOW INVOLVEMENT\nlow demand · does little', transform=ax.transAxes,
             ha='left', va='bottom', fontsize=10, color=C_QUAD, style='italic', linespacing=1.4)
    ax.text(0.015, 0.975, 'SELECTIVE\nlow demand · responds when needed', transform=ax.transAxes,
             ha='left', va='top', fontsize=10, color=C_QUAD, style='italic', linespacing=1.4)
    cur['defr90'] = cur['act90'] - cur['exp90']
    cur['_dist'] = np.sqrt(((cur['exp90'] - mx) / max(x.std(), 1e-9))**2
                             + ((cur['act90'] - my) / max(y.std(), 1e-9))**2)
    N = min(n_lab, len(cur))
    lab = pd.concat([cur.nlargest(N, '_dist'), cur.nlargest(5, 'defr90'),
                       cur.nsmallest(5, 'defr90'), cur.nlargest(4, 'exp90'),
                       cur.nlargest(4, 'act90')]).drop_duplicates('playerId')
    ax.scatter(lab['exp90'], lab['act90'], s=135, facecolors='none',
                edgecolors='#202124', linewidths=1.2, zorder=4)
    texts = [ax.text(r['exp90'], r['act90'], r['name'], fontsize=9, fontweight='bold',
                      color=C_TXT, zorder=5) for _, r in lab.iterrows()]
    if HAS:
        adjust_text(texts, ax=ax, only_move={'text': 'xy'},
                     arrowprops=dict(arrowstyle='-', color='#9aa0a6', lw=0.6),
                     expand=(1.25, 1.45))
    ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=10, colors='#5f6368', length=0)
    ax.set_xlabel('Expected defensive actions per 90  (how much the role/situation demands)',
                    fontsize=12, color='#3c4043', labelpad=10)
    ax.set_ylabel('Actual defensive actions per 90  (what the player delivers)',
                    fontsize=12, color='#3c4043', labelpad=10)
    ax.set_title(f'{pretty} — Defensive Responsibility — {league} 25/26',
                  fontsize=16, fontweight='bold', color=C_TXT, pad=14)
    sub = (f'{slots} · ≥700 min ({len(cur)} players, {len(lab)} labelled). '
            'Above the dashed line = more than typical for the role.')
    if caveat:
        sub += '\n' + caveat
    fig.text(0.5, 0.915, sub, ha='center', fontsize=10, color='#5f6368')
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out = str(_HERE / 'plots' / fname)
    plt.savefig(out, dpi=140, bbox_inches='tight', facecolor='white'); plt.close()
    print(f"Saved {out}  ({len(cur)} players, {len(lab)} labelled)")


for gkey, slots, pretty, n_lab, caveat in GROUPS:
    for sid, league, color in LEAGUES:
        lkey = 'liga3' if league == 'Liga 3' else 'camp'
        make(slots, pretty, n_lab, caveat, sid, league, color,
              f'defr_{gkey}_{lkey}_2526.png')
