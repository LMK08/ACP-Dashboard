#!/usr/bin/env python3
"""Dedicated centre-back DefR chart (StatsBomb-article style).
Single panel, back-four CBs (LCB+RCB) — the 'CB' code is a back-3 central
role with a different defensive demand and is excluded.
Run from the Dashboard dir: python models/defr/plot_defr_centerback.py
Output: models/defr/plots/defr_centerback_2526.png"""
from pathlib import Path
_HERE=Path(__file__).resolve().parent; _DASH=_HERE.parent.parent
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
try:
    from adjustText import adjust_text; HAS=True
except Exception: HAS=False

df=pd.read_parquet(_HERE/'defr_per_player_season.parquet')
details=pd.read_pickle(_DASH/'player_details.pkl')
nm={int(p['playerId']):(f"{p.get('firstName','')} {p.get('lastName','')}").strip() or p.get('shortName','?') for p in details}
gpa=pd.read_parquet(_DASH/'gpa_player_season_values.parquet'); gn=gpa.groupby('playerId')['name'].first().to_dict()
def short(pid):
    n=nm.get(int(pid),'')
    if not n or n=='?': n=gn.get(int(pid),str(pid))
    parts=n.split()
    return f"{parts[0][0]}. {parts[-1]}" if len(parts)>=2 else n

CB={'LCB','RCB'}  # back-four centre-backs (the 'CB' code is back-3 central, a different role)
cur=df[df['position'].isin(CB) & df['seasonId'].isin([191782,191779]) & (df['mins_played']>=900)].copy()
cur['exp90']=cur['expected_def_actions']/(cur['mins_played']/90)
cur['act90']=cur['actual_def_actions']/(cur['mins_played']/90)
cur['name']=cur['playerId'].apply(short)
cur['league']=cur['seasonId'].map({191782:'Liga 3',191779:'Campeonato'})
cur['color']=cur['league'].map({'Liga 3':'#2E6FB0','Campeonato':'#E08A2B'})

C_TXT='#202124'; C_LINE='#9aa0a6'; C_QUAD='#b0b4b8'; C_GRID='#e8eaed'
plt.rcParams.update({'font.family':'DejaVu Sans','axes.edgecolor':'#dadce0','figure.facecolor':'white','axes.facecolor':'white'})
fig,ax=plt.subplots(figsize=(13.5,11))
x,y=cur['exp90'].values,cur['act90'].values
mx,my=np.median(x),np.median(y)
lim_x=np.percentile(x,99)*1.10; lim_y=max(y.max(),np.percentile(y,99))*1.12

# quadrant crosshairs (median)
ax.axvline(mx,color='#f1f3f4',lw=10,zorder=0)
ax.axhline(my,color='#f1f3f4',lw=10,zorder=0)
ax.grid(True,color=C_GRID,lw=0.7,zorder=0); ax.set_axisbelow(True)
# ratio reference line (position-typical actual/expected)
slope=my/mx if mx>0 else 1
ax.plot([0,lim_x],[0,slope*lim_x],'--',color=C_LINE,lw=1.3,zorder=1,label='Typical CB level')

ax.scatter(x,y,c=cur['color'],s=120,alpha=0.7,edgecolors='white',linewidths=1.0,zorder=3)

# quadrant archetype labels
ax.text(0.985,0.975,'FRONT-FOOT AGGRESSOR\nhigh demand · over-delivers',transform=ax.transAxes,ha='right',va='top',fontsize=10,color=C_QUAD,style='italic',linespacing=1.4)
ax.text(0.985,0.025,'SYSTEM ABSORBER\nhigh demand · falls short',transform=ax.transAxes,ha='right',va='bottom',fontsize=10,color=C_QUAD,style='italic',linespacing=1.4)
ax.text(0.015,0.025,'PASSIVE LINE-HOLDER\nlow demand · does little',transform=ax.transAxes,ha='left',va='bottom',fontsize=10,color=C_QUAD,style='italic',linespacing=1.4)
ax.text(0.015,0.975,'SELECTIVE ("Van Dijk")\nlow demand · responds when needed',transform=ax.transAxes,ha='left',va='top',fontsize=10,color=C_QUAD,style='italic',linespacing=1.4)

# label the most extreme + notable: top 8 by DefR, bottom 4, plus quadrant extremes
cur['_defr']=cur['act90']-cur['exp90']
to_label=pd.concat([cur.nlargest(9,'defr_per90'),cur.nsmallest(4,'defr_per90'),
                    cur.nlargest(3,'exp90'),cur.nlargest(3,'act90')]).drop_duplicates('playerId')
ax.scatter(to_label['exp90'],to_label['act90'],s=150,facecolors='none',edgecolors='#202124',linewidths=1.3,zorder=4)
texts=[ax.text(r['exp90'],r['act90'],r['name'],fontsize=10,fontweight='bold',color=C_TXT,zorder=5) for _,r in to_label.iterrows()]
if HAS: adjust_text(texts,ax=ax,only_move={'text':'xy'},arrowprops=dict(arrowstyle='-',color='#9aa0a6',lw=0.7),expand=(1.3,1.5))

ax.set_xlim(0,lim_x); ax.set_ylim(0,lim_y)
for s in ('top','right'): ax.spines[s].set_visible(False)
ax.tick_params(labelsize=10,colors='#5f6368',length=0)
ax.set_xlabel('Expected defensive actions per 90  (how much the role/situation demands)',fontsize=12,color='#3c4043',labelpad=10)
ax.set_ylabel('Actual defensive actions per 90  (what the player delivers)',fontsize=12,color='#3c4043',labelpad=10)
ax.set_title('Centre-Back (back-four) Defensive Responsibility — 25/26 Liga 3 + Campeonato',fontsize=16,fontweight='bold',color=C_TXT,pad=14)
fig.text(0.5,0.915,'Above the dashed line = does more defending than a typical CB in the same situation.  ≥900 min.',ha='center',fontsize=10.5,color='#5f6368')
handles=[Line2D([0],[0],marker='o',color='w',markerfacecolor='#2E6FB0',markersize=11,label='Liga 3'),
         Line2D([0],[0],marker='o',color='w',markerfacecolor='#E08A2B',markersize=11,label='Campeonato'),
         Line2D([0],[0],ls='--',color=C_LINE,label='Typical CB level')]
ax.legend(handles=handles,loc='lower right',fontsize=10,frameon=True,framealpha=0.9,bbox_to_anchor=(0.998,0.10))
plt.tight_layout(rect=[0,0,1,0.93])
out=str(_HERE/'plots'/'defr_centerback_2526.png')
plt.savefig(out,dpi=140,bbox_inches='tight',facecolor='white')
print(f"Saved {out}  ({len(cur)} CBs)")
