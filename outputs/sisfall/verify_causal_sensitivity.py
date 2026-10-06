"""Independent direct counting checks for selected validation operating points."""
import sys
sys.dont_write_bytecode=True
import causal_sensitivity_sweep as s
import math
import json

def main():
    points=s.read(s.OUT/'operating_points.csv')
    trials=[r for r in s.read(s.CAUSAL/'validation_trial_results.csv')
            if r['key']=='g00' and r['design']=='pre100_post200']
    events=[r for r in s.read(s.MODELS/'event_predictions.csv')
            if r['split']=='validation' and r['model']=='logistic']
    trial_map={r['path']:r for r in trials}
    sweep=s.read(s.OUT/'threshold_sweep.csv')
    checks={r['threshold']:r for r in points}
    checks[sweep[-1]['threshold']]=sweep[-1]
    checks[sweep[len(sweep)//2]['threshold']]=sweep[len(sweep)//2]
    for threshold,row in checks.items():
        alarms=[e for e in events if float(e['score'])>=float(threshold)]
        detected={e['path'] for e in alarms if int(e['proxy_target'])}
        adl=[e for e in alarms if not int(trial_map[e['path']]['label'])]
        adl_trials=[t for t in trials if not int(t['label'])]
        alarmed_adl={e['path'] for e in adl}
        assert len(detected)==int(row['detected_fall_trials'])
        assert len(alarms)==int(row['emitted_positive_decisions'])
        assert len(adl)==int(row['adl_false_alarms'])
        specificity=1-len(alarmed_adl)/len(adl_trials)
        assert math.isclose(specificity,float(row['adl_trial_specificity']),abs_tol=1e-12)
        if alarms:assert math.isclose(len(detected)/len(alarms),float(row['event_precision']),abs_tol=1e-12)
        else:assert row['event_precision']==''
        for a in sorted({t['activity'] for t in trials}):
            group=[t for t in trials if t['activity']==a]
            if int(group[0]['label']):
                rate=sum(t['path'] in detected for t in group)/len(group)
                assert rate==float(row[a+'_sensitivity'])
            else:
                count=sum(trial_map[e['path']]['activity']==a for e in adl)
                rate=count/(sum(float(t['duration_s']) for t in group)/3600)
                assert math.isclose(rate,float(row[a+'_false_alarms_per_hour']),abs_tol=1e-12)
    for cap in (5,10,20):
        point=next(r for r in points if r['operating_point']==f'max_sensitivity_FA_le_{cap}')
        eligible=[r for r in sweep if float(r['adl_false_alarms_per_hour'])<=cap]
        best_count=max(int(r['detected_fall_trials']) for r in eligible)
        assert int(point['detected_fall_trials'])==best_count
        best_fa=min(float(r['adl_false_alarms_per_hour']) for r in eligible if int(r['detected_fall_trials'])==best_count)
        assert float(point['adl_false_alarms_per_hour'])==best_fa
    provenance=json.loads((s.OUT/'provenance.json').read_text())
    assert s.hashes([s.ROOT/p for p in provenance['input_sha256']])==provenance['input_sha256']
    s.dump('independent_verification.json',dict(points_directly_checked=len(checks),
                                               all_selected_activity_rates_verified=True,budget_optimality_verified=True,
                                               source_hashes_unchanged=True,no_test_data_used=True))
    print(f'Independent direct counting passed at {len(checks)} thresholds, including all selected points and no-alarm boundary.')

if __name__=='__main__':main()
