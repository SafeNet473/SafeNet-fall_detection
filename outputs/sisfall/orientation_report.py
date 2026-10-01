"""Render existing orientation-study tables; no threshold selection."""
import argparse
import csv
from pathlib import Path
import orientation_features as f
import numpy as np


def read(path):
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def render(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary = [r for r in read(out/'rotation_summary.csv') if r['split']=='test']
    metrics = read(out/'metrics.csv')
    activities = read(out/'target_activity_summary.csv')
    chosen = ['C2','C8','C9','mag_std','mag_peak','gyro_std','ga_C2','ga_C8']
    by_name = {r['feature']:r for r in summary}
    fig,axes=plt.subplots(1,3,figsize=(15,5),sharex=True)
    for ax,key,title in zip(axes,('sensitivity','specificity','balanced_accuracy'),('Sensitivity','Specificity','Balanced accuracy')):
        base=np.array([float(by_name[name][key+'_original']) for name in chosen])*100
        mean=np.array([float(by_name[name][key+'_mean']) for name in chosen])*100
        low=np.array([float(by_name[name][key+'_min']) for name in chosen])*100
        high=np.array([float(by_name[name][key+'_max']) for name in chosen])*100
        x=np.arange(len(chosen))
        ax.scatter(x-.12,base,label='Original waist coordinates',marker='o',color='#176c9c')
        ax.errorbar(x+.12,mean,yerr=np.maximum(np.array([mean-low,high-mean]),0),fmt='s',capsize=3,
                    color='#c46518',label='Random rotations: mean and range')
        ax.set_xticks(x,chosen,rotation=55,ha='right')
        ax.set(title=title,ylabel='Percent',ylim=(0,103))
        ax.grid(axis='y',alpha=.25)
    axes[0].legend(fontsize=7,loc='lower left')
    fig.suptitle('Fixed test subjects and thresholds — coordinate rotations, not wrist recordings')
    fig.tight_layout()
    fig.savefig(out/'orientation_metrics.png',dpi=160)
    fig.savefig(out/'orientation_metrics.svg')
    plt.close(fig)
    codes=['D03','D04','D06','D18','D19','F13','F14','F15']
    names=['C2','C8','mag_std','gyro_std','ga_C2','ga_C8']
    matrix=np.array([[100*(float(next(r for r in activities if r['split']=='test' and r['feature']==name and r['activity_code']==code)['random_rate_mean'])
                         -float(next(r for r in activities if r['split']=='test' and r['feature']==name and r['activity_code']==code)['original_rate']))
                        for code in codes] for name in names])
    matrix[np.abs(matrix)<1e-12]=0
    fig,ax=plt.subplots(figsize=(11,4.5))
    scale=max(1,float(np.max(np.abs(matrix))))
    img=ax.imshow(matrix,cmap='RdBu_r',vmin=-scale,vmax=scale,aspect='auto')
    ax.set_xticks(range(len(codes)),codes)
    ax.set_yticks(range(len(names)),names)
    for i in range(len(names)):
        for j in range(len(codes)):
            ax.text(j,i,f'{matrix[i,j]:+.1f}',ha='center',va='center',color='white' if abs(matrix[i,j])>.65*scale else 'black')
    ax.set_title('Test error-rate change: random-rotation mean minus original (percentage points)')
    ax.set_xlabel('D codes: false-positive rate; F codes: false-negative rate')
    fig.colorbar(img,ax=ax,label='Error-rate change (pp)')
    fig.tight_layout()
    fig.savefig(out/'activity_rotation_changes.png',dpi=160)
    plt.close(fig)
    text=['# Orientation study findings',
          'The original paper results are preserved. This report describes fixed-threshold changes on waist recordings under synthetic coordinate rotations; it does not validate a wrist detector.',
          '## C2/C8 axis dependence']
    for name in ('C2','C8'):
        r=by_name[name]
        controls={s:next(v for v in metrics if v['feature']==name and v['split']=='test' and v['scenario']==s) for s in ('original','yaw_y_90','tilt_x_90')}
        text.append(f"- **{name}**: original test BA {float(r['balanced_accuracy_original']):.2%}; random rotations mean {float(r['balanced_accuracy_mean']):.2%} "
                    f"(range {float(r['balanced_accuracy_min']):.2%}–{float(r['balanced_accuracy_max']):.2%}). "
                    f"90° y-yaw BA {float(controls['yaw_y_90']['balanced_accuracy']):.2%}; 90° x-tilt BA {float(controls['tilt_x_90']['balanced_accuracy']):.2%}.")
    text += ['The yaw/tilt controls directly test the x/z assumption: rotation inside that plane leaves scores unchanged, while mixing in y changes scores and decisions. This isolates coordinate dependence without changing physical samples or retuning thresholds.',
             '## Robust scores and limits',
             'All magnitude features, gyro-magnitude features, C9, and gravity-aligned features passed numerical invariance checks. Their metrics and requested activity errors remain unchanged under the tested constant rotations. Rotation invariance does not imply good discrimination: consult the original-score table before considering a feature useful.',
             'C9 is already invariant, whereas C3 is not generally invariant. Gravity-aligned features depend on transforming gravity history together with the window, and may perform differently from the paper horizontal features because a 0.5 Hz estimate follows some motion and posture changes. No test-based feature selection is performed.',
             'Gravity state uses raw samples causally. Existing zero-phase signal filtering and label-aware event localization remain offline. Neither invariance nor the operation estimates establish MCU readiness or accuracy at the wrist.',
             '![Orientation metrics](orientation_metrics.png)',
             '![Activity changes](activity_rotation_changes.png)',
             'Complete tables: [REPORT.md](REPORT.md), [metrics.csv](metrics.csv), [target_activity_summary.csv](target_activity_summary.csv). Equations, causal-state initialization, operation counts and storage estimates: [ORIENTATION_METHODS.md](../sisfall/ORIENTATION_METHODS.md).']
    for name in ('mag_peak','gyro_std','ga_C2','ga_C8'):
        row=next(r for r in metrics if r['scenario']=='original' and r['split']=='test' and r['feature']==name)
        text.insert(-3,f"- **{name}**, unchanged by rotations: sensitivity {float(row['sensitivity']):.2%}, "
                    f"specificity {float(row['specificity']):.2%}, balanced accuracy {float(row['balanced_accuracy']):.2%}, "
                    f"precision {float(row['precision']):.2%}, F1 {float(row['f1']):.2%}, PR-AUC {float(row['pr_auc_trapezoidal']):.4f}.")
    text.append('Recreate these plots and this findings summary with `python outputs/sisfall/orientation_report.py outputs/orientation_study`. No evaluation is rerun and thresholds are not changed.')
    (out/'FINDINGS.md').write_text('\n\n'.join(text)+'\n',encoding='utf-8')
    # Export documented MCU table without importing optional spreadsheet dependencies.
    lines=Path(__file__).with_name('ORIENTATION_METHODS.md').read_text(encoding='utf-8').splitlines()
    start=next(i for i,line in enumerate(lines) if line.startswith('| Feature | A per window'))
    costs=[]
    for line in lines[start+2:]:
        if not line.startswith('|'):
            break
        values=[cell.strip() for cell in line.strip('|').split('|')]
        costs.append(dict(feature=values[0],additions=int(values[1]),multiplications=int(values[2]),
                          square_roots=int(values[3]),notes=values[4]))
    f.p.write_csv(out/'mcu_operation_estimates.csv',costs)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    args=parser.parse_args()
    output=args.out.resolve()
    if not output.is_relative_to((f.p.PROJECT/'outputs').resolve()):
        parser.error('Output must be inside project outputs/')
    render(output)
