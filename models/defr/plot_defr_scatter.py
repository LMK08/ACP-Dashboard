#!/usr/bin/env python3
"""StatsBomb-style DefR scatterplots, faceted by position group.
Run from the Dashboard dir: python models/defr/plot_defr_scatter.py
Output: models/defr/plots/defr_scatter_2526.png"""
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_DASH = _HERE.parent.parent
df = pd.read_parquet(_HERE / 'defr_per_player_season.parquet')
details = pd.read_pickle(_DASH / 'player_details.pkl')
nm = {int(p['playerId']):(f"{p.get('firstName','')} {p.get('lastName','')}").strip() or p.get('shortName','?') for p in details}
gpa = pd.read_parquet(_DASH / 'gpa_player_season_values.parquet'); gn = gpa.groupby('playerId')['name'].first().to_dict()
def name(pid):
    n=nm.get(int(pid),''); 
    if not n or n=='?': n=gn.get(int(pid),f'pid {pid}')
    # shorten to last name or short form
    parts=n.split()
    return n if len(n)<=16 else (parts[-1] if parts else n)

# Slot -> broad position group
GROUP = {'GK':'GK','CB':'CB','LCB':'CB','RCB':'CB','LB':'FB','RB':'FB',
         'LCM':'CM','RCM':'CM','DMF':'CM','AM':'AM/W','LAM':'AM/W','RAM':'AM/W',
         'LW':'AM/W','RW':'AM/W','ST':'ST'}
df['grp'] = df['position'].map(GROUP)
df['exp90'] = df['expected_def_actions']/(df['mins_played']/90)
df['act90'] = df['actual_def_actions']/(df['mins_played']/90)

# Current season, both leagues
SEASON_LBL={191782:'Liga 3',191779:'Camp'}
cur = df[df['seasonId'].isin([191782,191779]) & (df['mins_played']>=700)].copy()
cur['name']=cur['playerId'].apply(name)
cur['league']=cur['seasonId'].map(SEASON_LBL)

groups=['ST','AM/W','CM','FB','CB','GK']
fig, axes = plt.subplots(2,3, figsize=(20,13))
fig.suptitle('Defensive Responsibility (DefR) — Expected vs Actual defensive actions per 90\n'
             '25/26 Liga 3 + Campeonato (≥700 min). Above the dashed line = over-performs the defensive demand placed on them.',
             fontsize=15, fontweight='bold')

for ax, g in zip(axes.flat, groups):
    sub = cur[cur['grp']==g].copy()
    if sub.empty:
        ax.set_visible(False); continue
    x, y = sub['exp90'].values, sub['act90'].values
    mx_med, my_med = np.median(x), np.median(y)
    # color L3 vs Camp
    colors = sub['league'].map({'Liga 3':'#1f77b4','Camp':'#ff7f0e'})
    ax.scatter(x, y, c=colors, s=42, alpha=0.6, edgecolors='white', linewidths=0.5, zorder=3)
    lim = max(x.max(), y.max())*1.05
    # "Meets expectation" line = position's typical actual/expected ratio
    # through the origin. expected & actual are on slightly different
    # bases (expected = matched responses; actual = all def actions), so
    # the raw y=x diagonal isn't the neutral line. Above this ratio line
    # = over-performs the typical conversion for the position.
    slope = (my_med / mx_med) if mx_med > 0 else 1.0
    ax.plot([0, lim], [0, slope*lim], '--', color='gray', lw=1.2, zorder=1)
    # median crosshairs (quadrant split)
    ax.axvline(mx_med, color='#cccccc', lw=0.8, zorder=1)
    ax.axhline(my_med, color='#cccccc', lw=0.8, zorder=1)
    # quadrant archetype labels (StatsBomb)
    ax.text(0.97,0.97,'Front-Foot\nAggressor',transform=ax.transAxes,ha='right',va='top',fontsize=8,color='#888',style='italic')
    ax.text(0.97,0.03,'System\nAbsorber',transform=ax.transAxes,ha='right',va='bottom',fontsize=8,color='#888',style='italic')
    ax.text(0.03,0.03,'Passive\nLine-Holder',transform=ax.transAxes,ha='left',va='bottom',fontsize=8,color='#888',style='italic')
    ax.text(0.03,0.97,'Selective\n(Van Dijk zone)',transform=ax.transAxes,ha='left',va='top',fontsize=8,color='#888',style='italic')
    # label the top over-performers (highest defr_adj) + a couple notable
    sub['_over']=sub['act90']-sub['exp90']
    for _,r in sub.nlargest(4,'defr_adj').iterrows():
        ax.annotate(r['name'], (r['exp90'],r['act90']), fontsize=8, fontweight='bold',
                     xytext=(4,4), textcoords='offset points', zorder=4)
    ax.set_title(f"{g}  (n={len(sub)})", fontsize=13, fontweight='bold')
    ax.set_xlabel('Expected defensive actions / 90', fontsize=10)
    ax.set_ylabel('Actual defensive actions / 90', fontsize=10)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.grid(alpha=0.2, zorder=0)

# legend
from matplotlib.lines import Line2D
handles=[Line2D([0],[0],marker='o',color='w',markerfacecolor='#1f77b4',markersize=9,label='Liga 3'),
         Line2D([0],[0],marker='o',color='w',markerfacecolor='#ff7f0e',markersize=9,label='Campeonato'),
         Line2D([0],[0],ls='--',color='gray',label='Position-typical actual/expected ratio')]
fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=11, bbox_to_anchor=(0.5,-0.01))
plt.tight_layout(rect=[0,0.03,1,0.95])
out = _HERE / 'plots' / 'defr_scatter_2526.png'
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=110, bbox_inches='tight')
print(f"Saved {out}")
