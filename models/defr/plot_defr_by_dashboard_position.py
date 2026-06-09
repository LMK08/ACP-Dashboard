#!/usr/bin/env python3
"""Per-league DefR scatter charts grouped by the DASHBOARD position groups
(_cvi_position_group): CB, FB, CM, AM_WG, ST — one chart per
(group x league). Uses each player's RAW Wyscout primary position
(re-derived from events) so EVERY granular code is included
(LDMF, RAMF, LB5, LWB, CMF, LCMF3, CF, SS, ...).

Axes zoom to the data range; labels the most interesting players.

Run from the Dashboard dir:
    python models/defr/plot_defr_by_dashboard_position.py
Outputs: plots/defr_<group>_<league>_2526.png
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
_GPA_DATA = _DASH.parent.parent / 'GPA Model Project v2' / 'parquet_data'


def cvi_position_group(p):
    """Exact copy of the dashboard's _cvi_position_group (GK dropped)."""
    if p is None or (isinstance(p, float) and pd.isna(p)):
        return None
    p = str(p)
    if p in ('CB', 'LCB', 'RCB', 'LCB3', 'RCB3'): return 'CB'
    if p in ('LB', 'RB', 'LB5', 'RB5', 'LWB', 'RWB'): return 'FB'
    if p in ('CMF', 'LCMF', 'RCMF', 'LCMF3', 'RCMF3', 'DMF', 'LDMF', 'RDMF'): return 'CM'
    if p in ('AMF', 'LAMF', 'RAMF', 'LMF', 'RMF', 'LW', 'RW', 'LWF', 'RWF'): return 'AM_WG'
    if p in ('CF', 'SS'): return 'ST'
    return None


GROUP_TITLE = {'CB': 'Centre-Backs', 'FB': 'Full-Backs', 'CM': 'Central / Defensive Mids',
                'AM_WG': 'Attacking Mids & Wingers', 'ST': 'Strikers'}
N_LAB = {'CB': 32, 'FB': 30, 'CM': 32, 'AM_WG': 34, 'ST': 28}

# ---- names ----
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


# ---- raw primary position per (player, season) from events ----
print("Deriving raw primary position from events…", flush=True)
ecols = ['seasonId', 'player.id', 'player.position']
ev = pd.concat([pd.read_parquet(_GPA_DATA / f'{f}.parquet', columns=ecols)
                  for f in ['liga3_portugal_events', 'campeonato_portugal_events']],
                 ignore_index=True).dropna(subset=['player.id', 'player.position'])
ev['player.id'] = ev['player.id'].astype(int)
ev['seasonId'] = pd.to_numeric(ev['seasonId'], errors='coerce').astype('Int64')
raw_pos = (ev.groupby(['player.id', 'seasonId', 'player.position']).size()
             .reset_index(name='n').sort_values('n', ascending=False)
             .drop_duplicates(['player.id', 'seasonId'])
             .rename(columns={'player.id': 'playerId', 'player.position': 'raw_pos'}))

df = pd.read_parquet(_HERE / 'defr_per_player_season.parquet')
df = df.merge(raw_pos[['playerId', 'seasonId', 'raw_pos']], on=['playerId', 'seasonId'], how='left')
df['grp'] = df['raw_pos'].map(cvi_position_group)
df['exp90'] = df['expected_def_actions'] / (df['mins_played'] / 90)
df['act90'] = df['actual_def_actions'] / (df['mins_played'] / 90)
df['name'] = df['playerId'].apply(short)

C_TXT='#202124'; C_LINE='#9aa0a6'; C_QUAD='#b0b4b8'; C_GRID='#e8eaed'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.edgecolor': '#dadce0',
                       'figure.facecolor': 'white', 'axes.facecolor': 'white'})
LEAGUES = [(191782, 'Liga 3', 'liga3', '#2E6FB0'), (191779, 'Campeonato', 'camp', '#E08A2B')]


def make(grp, sid, league, lkey, color):
    cur = df[(df['grp'] == grp) & (df['seasonId'] == sid) & (df['mins_played'] >= 700)].copy()
    if len(cur) < 8:
        print(f"  skip {grp}/{lkey}: only {len(cur)} players")
        return
    fig, ax = plt.subplots(figsize=(14, 11.5))
    x, y = cur['exp90'].values, cur['act90'].values
    mx, my = np.median(x), np.median(y)
    padx = (x.max() - x.min()) * 0.08 or 1.0
    pady = (y.max() - y.min()) * 0.08 or 1.0
    xlo, xhi = x.min() - padx, x.max() + padx
    ylo, yhi = y.min() - pady, y.max() + pady
    ax.axvline(mx, color='#f1f3f4', lw=10, zorder=0); ax.axhline(my, color='#f1f3f4', lw=10, zorder=0)
    ax.grid(True, color=C_GRID, lw=0.7, zorder=0); ax.set_axisbelow(True)
    slope = my / mx if mx > 0 else 1
    ax.plot([0, xhi * 1.3], [0, slope * xhi * 1.3], '--', color=C_LINE, lw=1.3, zorder=1)
    ax.scatter(x, y, c=color, s=110, alpha=0.65, edgecolors='white', linewidths=1.0, zorder=3)
    ax.text(0.985, 0.975, 'OVER-DELIVERS\nhigh demand · does more', transform=ax.transAxes, ha='right', va='top', fontsize=10, color=C_QUAD, style='italic', linespacing=1.4)
    ax.text(0.985, 0.025, 'UNDER-DELIVERS\nhigh demand · does less', transform=ax.transAxes, ha='right', va='bottom', fontsize=10, color=C_QUAD, style='italic', linespacing=1.4)
    ax.text(0.015, 0.025, 'LOW INVOLVEMENT\nlow demand · does little', transform=ax.transAxes, ha='left', va='bottom', fontsize=10, color=C_QUAD, style='italic', linespacing=1.4)
    ax.text(0.015, 0.975, 'SELECTIVE\nlow demand · responds when needed', transform=ax.transAxes, ha='left', va='top', fontsize=10, color=C_QUAD, style='italic', linespacing=1.4)
    cur['defr90'] = cur['act90'] - cur['exp90']
    cur['_dist'] = np.sqrt(((cur['exp90'] - mx) / max(x.std(), 1e-9))**2 + ((cur['act90'] - my) / max(y.std(), 1e-9))**2)
    N = min(N_LAB[grp], len(cur))
    lab = pd.concat([cur.nlargest(N, '_dist'), cur.nlargest(5, 'defr90'),
                       cur.nsmallest(5, 'defr90'), cur.nlargest(4, 'exp90'),
                       cur.nlargest(4, 'act90')]).drop_duplicates('playerId')
    ax.scatter(lab['exp90'], lab['act90'], s=135, facecolors='none', edgecolors='#202124', linewidths=1.2, zorder=4)
    texts = [ax.text(r['exp90'], r['act90'], r['name'], fontsize=9, fontweight='bold', color=C_TXT, zorder=5) for _, r in lab.iterrows()]
    if HAS:
        adjust_text(texts, ax=ax, only_move={'text': 'xy'}, arrowprops=dict(arrowstyle='-', color='#9aa0a6', lw=0.6), expand=(1.25, 1.45))
    ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=10, colors='#5f6368', length=0)
    ax.set_xlabel('Expected defensive actions per 90  (how much the role/situation demands)', fontsize=12, color='#3c4043', labelpad=10)
    ax.set_ylabel('Actual defensive actions per 90  (what the player delivers)', fontsize=12, color='#3c4043', labelpad=10)
    ax.set_title(f'{GROUP_TITLE[grp]} — Defensive Responsibility — {league} 25/26', fontsize=16, fontweight='bold', color=C_TXT, pad=14)
    codes = sorted(cur['raw_pos'].dropna().unique().tolist())
    sub = (f'Dashboard group "{grp}" — positions: {", ".join(codes)}.  '
            f'≥700 min ({len(cur)} players, {len(lab)} labelled). '
            'Above the dashed line = more than typical for the group.')
    fig.text(0.5, 0.912, sub, ha='center', fontsize=9.5, color='#5f6368')
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out = str(_HERE / 'plots' / f'defr_{grp.lower()}_{lkey}_2526.png')
    plt.savefig(out, dpi=140, bbox_inches='tight', facecolor='white'); plt.close()
    print(f"Saved {out}  ({len(cur)} players, {len(lab)} labelled)")


for grp in ['CB', 'FB', 'CM', 'AM_WG', 'ST']:
    for sid, league, lkey, color in LEAGUES:
        make(grp, sid, league, lkey, color)
