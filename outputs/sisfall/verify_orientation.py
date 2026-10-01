"""Standard-library cross-check of saved orientation result consistency."""
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
out=ROOT/'outputs/orientation_study'
def read(name):
    with (out/name).open(newline='') as handle:
        return list(csv.DictReader(handle))

metrics=read('metrics.csv')
activities=read('activity_errors.csv')
thresholds=json.loads((out/'thresholds.json').read_text())['thresholds']
counts={}
for row in activities:
    key=(row['scenario'],row['split'],row['feature'],row['error_type'])
    errors,total=counts.get(key,(0,0))
    counts[key]=(errors+int(row['errors']),total+int(row['total']))
    assert abs(float(row['rate'])-int(row['errors'])/int(row['total']))<1e-14
for row in metrics:
    scenario,split,feature=row['scenario'],row['split'],row['feature']
    assert float(row['threshold'])==thresholds[feature]
    fp,negatives=counts[scenario,split,feature,'FP']
    fn,positives=counts[scenario,split,feature,'FN']
    assert fp==int(row['fp']) and fn==int(row['fn'])
    assert int(row['tp'])+fn==positives and int(row['tn'])+fp==negatives
    assert positives+negatives==int(row['n'])
assert (out/'paper_results_unchanged.csv').read_bytes()==(ROOT/'outputs/paper_feature_comparison/metrics.csv').read_bytes()
result=dict(metric_rows=len(metrics),activity_rows=len(activities),fixed_thresholds_verified=True,
            activity_confusion_counts_reconciled=True,paper_result_copy_exact=True)
(out/'verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
