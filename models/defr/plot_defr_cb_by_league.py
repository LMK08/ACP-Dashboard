#!/usr/bin/env python3
"""Per-league centre-back DefR charts (StatsBomb-article style), one each
for Liga 3 and Campeonato. Back-four CBs (LCB/RCB), >=700 min. Labels the
~45 most interesting players (outliers + DefR/demand/volume extremes) so
the dense centre stays readable.
Run from the Dashboard dir: python models/defr/plot_defr_cb_by_league.py
Outputs: plots/defr_cb_liga3_2526.png, plots/defr_cb_camp_2526.png"""
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

C_TXT='#202124'; C_LINE='#9aa0a6'; C_QUAD='#b0b4b8'; C_GRID='#e8eaed'
plt.rcParams.update({'font.family':'DejaVu Sans','axes.edgecolor':'#dadce0','figure.facecolor':'white','axes.facecolor':'white'})

def make(sid, label, color, fname):
    cur=df[df['position'].isin(['LCB','RCB']) & (df['seasonId']==sid) & (df['mins_played']>=700)].copy()
    cur['exp90']=cur['expected_def_actions']/(cur['mins_played']/90)
    cur['act90']=cur['actual_def_actions']/(cur['mins_played']/90)
    cur['name']=cur['playerId'].apply(short)
    fig,ax=plt.subplots(figsize=(14,11.5))
    x,y=cur['exp90'].values,cur['act90'].values
    mx,my=np.median(x),np.median(y)
    # zoom to the data range (with padding) so the points fill the chart
    padx=(x.max()-x.min())*0.08; pady=(y.max()-y.min())*0.08
    xlo,xhi=x.min()-padx,x.max()+padx; ylo,yhi=y.min()-pady,y.max()+pady
    ax.axvline(mx,color='#f1f3f4',lw=10,zorder=0); ax.axhline(my,color='#f1f3f4',lw=10,zorder=0)
    ax.grid(True,color=C_GRID,lw=0.7,zorder=0); ax.set_axisbelow(True)
    slope=my/mx if mx>0 else 1
    ax.plot([0,xhi*1.3],[0,slope*xhi*1.3],'--',color=C_LINE,lw=1.3,zorder=1)  # ratio line (clipped to view)
    ax.scatter(x,y,c=color,s=110,alpha=0.65,edgecolors='white',linewidths=1.0,zorder=3)
    ax.text(0.985,0.975,'FRONT-FOOT AGGRESSOR\nhigh demand · over-delivers',transform=ax.transAxes,ha='right',va='top',fontsize=10,color=C_QUAD,style='italic',linespacing=1.4)
    ax.text(0.985,0.025,'SYSTEM ABSORBER\nhigh demand · falls short',transform=ax.transAxes,ha='right',va='bottom',fontsize=10,color=C_QUAD,style='italic',linespacing=1.4)
    ax.text(0.015,0.025,'PASSIVE LINE-HOLDER\nlow demand · does little',transform=ax.transAxes,ha='left',va='bottom',fontsize=10,color=C_QUAD,style='italic',linespacing=1.4)
    ax.text(0.015,0.975,'SELECTIVE ("Van Dijk")\nlow demand · responds when needed',transform=ax.transAxes,ha='left',va='top',fontsize=10,color=C_QUAD,style='italic',linespacing=1.4)
    # label the most INTERESTING players — the outliers furthest from the
    # average cloud (so the dense centre of "average" CBs stays clean),
    # plus the DefR / demand / volume extremes. Cap for readability.
    cur['defr90']=cur['act90']-cur['exp90']
    cur['_dist']=np.sqrt(((cur['exp90']-mx)/max(x.std(),1e-9))**2 + ((cur['act90']-my)/max(y.std(),1e-9))**2)
    N=min(45,len(cur))
    lab=pd.concat([cur.nlargest(N,'_dist'),cur.nlargest(8,'defr90'),
                    cur.nsmallest(8,'defr90'),cur.nlargest(5,'exp90'),
                    cur.nlargest(5,'act90')]).drop_duplicates('playerId')
    ax.scatter(lab['exp90'],lab['act90'],s=135,facecolors='none',edgecolors='#202124',linewidths=1.2,zorder=4)
    texts=[ax.text(r['exp90'],r['act90'],r['name'],fontsize=9,fontweight='bold',color=C_TXT,zorder=5) for _,r in lab.iterrows()]
    if HAS: adjust_text(texts,ax=ax,only_move={'text':'xy'},arrowprops=dict(arrowstyle='-',color='#9aa0a6',lw=0.6),expand=(1.25,1.45))
    ax.set_xlim(xlo,xhi); ax.set_ylim(ylo,yhi)
    for s in ('top','right'): ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=10,colors='#5f6368',length=0)
    ax.set_xlabel('Expected defensive actions per 90  (how much the role/situation demands)',fontsize=12,color='#3c4043',labelpad=10)
    ax.set_ylabel('Actual defensive actions per 90  (what the player delivers)',fontsize=12,color='#3c4043',labelpad=10)
    ax.set_title(f'Centre-Back Defensive Responsibility — {label} 25/26',fontsize=16,fontweight='bold',color=C_TXT,pad=14)
    fig.text(0.5,0.915,f'Back-four CBs (LCB/RCB), ≥700 min ({len(cur)} players, {len(lab)} labelled). Above the dashed line = more than a typical CB in the same situation.',ha='center',fontsize=10.5,color='#5f6368')
    plt.tight_layout(rect=[0,0,1,0.93])
    out=str(_HERE/'plots'/fname)
    plt.savefig(out,dpi=140,bbox_inches='tight',facecolor='white'); plt.close()
    print(f"Saved {out}  ({len(cur)} CBs, {len(lab)} labelled)")

make(191782,'Liga 3','#2E6FB0','defr_cb_liga3_2526.png')
make(191779,'Campeonato','#E08A2B','defr_cb_camp_2526.png')
