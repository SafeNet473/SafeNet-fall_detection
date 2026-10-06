"""Plot saved Pareto points only; no fitting or data selection."""
import sys,os,csv
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/causal_feature_expansion'
os.environ['MPLCONFIGDIR']=str(OUT/'matplotlib_cache')
sys.path.insert(0,str(ROOT/'outputs/deps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with (OUT/'pareto_frontiers.csv').open(newline='') as f:rows=list(csv.DictReader(f))
fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
for ax,limit,title in zip(axes,(25,160),('Priority false-alarm budgets','Cost of exceeding 98% sensitivity')):
    for name,label,color in [('frozen3','Frozen 3 features','#2367a4'),('expanded5','5 features (C=0.01)','#c05a21')]:
        group=[r for r in rows if r['feature_set']==name]
        ax.step([float(r['false_alarms_per_adl_hour']) for r in group],
                [100*float(r['sensitivity']) for r in group],where='post',label=label,color=color,lw=2)
    ax.axhline(98,color='#7b394c',ls='--',lw=1,label='Target requires >98%')
    ax.axhline(100*370/375,color='#555555',ls=':',lw=1,label='Trigger ceiling 98.67%')
    for x in (1,5):ax.axvline(x,color='#aaaaaa',lw=.8,ls=':')
    ax.set(xlim=(0,limit),ylim=(80,100),xlabel='ADL false alarms per hour',ylabel='Fall-event sensitivity (%)',title=title)
    ax.grid(alpha=.2)
axes[0].legend(loc='lower right',fontsize=8)
fig.suptitle('Validation only: fixed trigger, window and logistic family')
fig.savefig(OUT/'frontier_comparison.png',dpi=160)
fig.savefig(OUT/'frontier_comparison.svg')
plt.close(fig)
