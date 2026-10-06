"""Artifact-only MCU v1 freeze. Never fit, tune, replay, or evaluate test data."""
import sys,os
sys.dont_write_bytecode=True
from pathlib import Path
import csv,json,hashlib,math,subprocess,platform
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'outputs/deps'))
import numpy as np
import scipy,sklearn,joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

OUT=ROOT/'outputs/mcu_v1_frozen'
MODEL=ROOT/'outputs/causal_trained_models_v1'
CAUSAL=ROOT/'outputs/causal_event_study'
SENS=ROOT/'outputs/causal_sensitivity_analysis'
STATEMENT='**MCU v1 is frozen for implementation. Any future change to preprocessing, trigger, window, feature set, model weights, or feature ordering must create MCU v2 or a new experimental branch.**'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def csvrows(p):
    with p.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def dump(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False,
                                   default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n',encoding='utf-8')

def main():
    if OUT.exists():raise RuntimeError('MCU v1 destination already exists; never overwrite a freeze.')
    paths=[MODEL/'model_parameters.json',MODEL/'logistic.joblib',MODEL/'event_predictions.csv',MODEL/'metrics.csv',
           MODEL/'experiment_plan.json',MODEL/'selection.json',MODEL/'verification.json',MODEL/'export_verification.json',
           CAUSAL/'experiment_plan.json',CAUSAL/'selection.json',CAUSAL/'regression_tests.json',
           SENS/'operating_points.csv',SENS/'provenance.json',SENS/'verification.json',
           SENS/'maximum_sensitivity_misses.csv',ROOT/'outputs/orientation_study/provenance.json',
           ROOT/'processed/baseline_v2/config.json',ROOT/'processed/baseline_v2/split_subjects.json',
           ROOT/'outputs/sisfall/pipeline.py',ROOT/'outputs/sisfall/causal_events.py',
           ROOT/'outputs/sisfall/orientation_features.py',ROOT/'outputs/sisfall/train_causal_models.py',
           ROOT/'outputs/sisfall/causal_sensitivity_sweep.py',Path(__file__)]
    before={str(p.relative_to(ROOT)):sha(p) for p in paths}
    exported=read(MODEL/'model_parameters.json')['logistic']
    obj=joblib.load(MODEL/'logistic.joblib');fitted=obj['model'];scaler=obj['scaler']
    plan=read(CAUSAL/'experiment_plan.json');selection=read(CAUSAL/'selection.json')
    orientation=read(ROOT/'outputs/orientation_study/provenance.json')
    config=read(ROOT/'processed/baseline_v2/config.json')
    assert exported['feature_names']==obj['feature_names']==['ga_C2','jerk_abs_mean','ga_parallel_peak']
    assert selection['design']=='pre100_post200' and selection['trigger']==[1.1,5.]
    for key,array in [('mean',scaler.mean_),('scale',scaler.scale_),('coefficients',fitted.coef_[0])]:
        np.testing.assert_array_equal(exported[key],array)
    assert exported['intercept']==float(fitted.intercept_[0])
    assert exported['threshold']==obj['threshold']==0.8850929456953477
    ops=csvrows(SENS/'operating_points.csv')
    ref=next(r for r in ops if r['operating_point']=='current_BA_reference')
    sensitivity=next(r for r in ops if r['operating_point']=='highest_threshold_sensitivity_ge_0.950000')
    assert float(sensitivity['threshold'])==0.7742031648776018
    assert orientation['gravity_alpha']==0.01558523664828626
    weights=[w/s for w,s in zip(exported['coefficients'],exported['scale'])]
    bias=exported['intercept']-sum(w*mu for w,mu in zip(weights,exported['mean']))
    metrics_keys=['fall_event_sensitivity','event_precision','adl_false_alarms_per_hour','adl_trial_specificity',
                  'trial_balanced_accuracy','detected_fall_trials','missed_fall_trials','emitted_positive_decisions','adl_false_alarms']
    points={}
    for name,row in [('balanced_reference',ref),('sensitivity_oriented_candidate',sensitivity)]:
        threshold=float(row['threshold'])
        points[name]=dict(score_threshold=threshold,comparison='>=',status='documented operating point; not final hardware deployment threshold',
                          derived_logit_threshold=math.log(threshold/(1-threshold)),
                          validation_metrics={k:float(row[k]) if k not in ('detected_fall_trials','missed_fall_trials','emitted_positive_decisions','adl_false_alarms') else int(row[k]) for k in metrics_keys})
    manifest=dict(schema_version=1,baseline_id='MCU_v1',status='algorithm frozen for implementation; hardware parity unresolved',
                  final_hardware_operating_threshold=None,final_hardware_operating_threshold_status='NOT FROZEN',
                  excluded_feature_expansions=[5,7],input=dict(sensor='ADXL345',columns_one_based=[1,2,3],axes=['x','y','z'],
                      accelerometer_only=True,gyroscope_used=False,sampling_rate_hz=200,units='g',
                      raw_counts_to_g=config['counts_to_g'],range_plus_minus_g=16,resolution_bits=13,placement='SisFall waist'),
                  acceleration_filter=dict(family='Butterworth',order=4,cutoff_hz=5,sampling_rate_hz=200,
                      realization='two cascaded direct-form-II-transposed SOS per axis',sos_column_order=['b0','b1','b2','a0','a1','a2'],
                      sos=plan['sos'],initialization='sosfilt_zi(SOS)[:,:,None] * first_raw_acceleration_g[None,None,:]',
                      state_reset='only at independent trial start; never per trigger/window',state_dtype='float64',
                      filtered_sample_interface='cast float32 then float64 before derived scalar calculations',
                      section_update=['y=b0*x+z1','z1_new=b1*x-a1*y+z2','z2_new=b2*x-a2*y']),
                  gravity=dict(input='raw unfiltered acceleration in g',alpha=orientation['gravity_alpha'],nominal_cutoff_hz=.5,
                      alpha_definition='1-exp(-2*pi*0.5/200)',equation='G[t]=alpha*r[t]+(1-alpha)*G[t-1]',
                      initialization='G[-1]=r[0]',state_reset='trial start only',state_dtype='float64',normalization_floor_g=1e-8,
                      unit_equation='u=G/max(sqrt(G dot G),1e-8)'),
                  derived_streams=dict(m='sqrt(a dot a)',p='a dot u (signed)',b='a-p*u',h='sqrt(b dot b)',dtype='float64'),
                  trigger=dict(magnitude_g=selection['trigger'][0],absolute_magnitude_jerk_g_per_s=selection['trigger'][1],
                      jerk='j[t]=200*abs(m[t]-m[t-1]); j[0]=0',condition='H[t]=(m[t]>=1.1) OR (j[t]>=5.0)',
                      edge='H[t] AND NOT H[t-1]',initial_high=False,refractory_samples=300,refractory_seconds=1.5,
                      accepted_when='rising edge AND t-last_accepted_trigger>=300',initial_last_accepted_trigger=-300,
                      refractory_edges='discard, do not queue; staying high does not retrigger',
                      incomplete_window_trigger='accepted trigger still updates refractory state'),
                  window=dict(name='pre100_post200',pre_samples=100,trigger_and_post_samples=200,total_samples=300,
                      nominal_duration_seconds=1.5,start_offset=-100,end_offset_exclusive=200,
                      sample_offsets_included='-100 through +199',decision_offset_samples=200,post_trigger_delay_seconds=1.,
                      decision_sample_excluded=True,reference_ring_rows=301,
                      reference_ring_channels=['m','h','abs(p)'],boundary_rule='require k>=100 and k+200<trial_length; otherwise discard; no padding/shifting'),
                  features=dict(order=exported['feature_names'],definitions={
                      'ga_C2':dict(equation='max(h[i]), i=0..299',units='g'),
                      'jerk_abs_mean':dict(equation='mean(abs(200*(m[i]-m[i-1]))), i=1..299',differences=299,units='g/s'),
                      'ga_parallel_peak':dict(equation='max(abs(p[i])), i=0..299',units='g')},dtype='float64'),
                  scaler=dict(class_name='StandardScaler',mean=exported['mean'],scale=exported['scale'],
                      variance=scaler.var_.tolist(),with_mean=scaler.with_mean,with_std=scaler.with_std,ddof=0,
                      n_features_in=int(scaler.n_features_in_),n_samples_seen=int(scaler.n_samples_seen_),fit_partition='train only'),
                  logistic=dict(class_name='LogisticRegression',feature_order=exported['feature_names'],
                      coefficients=exported['coefficients'],intercept=exported['intercept'],classes=fitted.classes_.tolist(),
                      n_features_in=int(fitted.n_features_in_),hyperparameters=fitted.get_params(deep=False),
                      equation='z=intercept+sum_i(coefficients[i]*(x[i]-mean[i])/scale[i]); score=1/(1+exp(-z))',
                      score_interpretation='classifier score, not calibrated real-world fall probability',fit_partition='train only'),
                  fused_raw_feature_equation=dict(status='algebraic derivation only; not firmware or bitwise-parity certified',
                      raw_feature_means='unstandardized extracted features, not raw accelerometer counts',
                      coefficients=weights,intercept=bias,derivation='w_raw=w/scale; b_raw=intercept-sum(w_raw*mean)',
                      binary_rule='dot(w_raw,x)+b_raw >= log(tau/(1-tau)); sigmoid unnecessary in real arithmetic'),
                  documented_operating_points=points,
                  evaluation=dict(matching='fall-trial trigger within +/-200 samples of raw-magnitude argmax proxy; complete window required',
                      matching_tolerance_samples=200,one_credited_detection_per_fall_trial=True,
                      duplicate_unmatched_alarms='count against event precision, including fall-recording alarms',
                      precision='detected fall trials / all positive candidate decisions',
                      trial_balanced_accuracy='0.5*(fall-event sensitivity + fraction of ADL trials without alarm)',
                      validation_fall_trials=375,validation_adl_trials=572,validation_adl_hours=3.0679416666666666,
                      validation_candidate_recall=370/375,unavoidable_no_matched_trigger_falls=5),
                  subject_partitions=read(ROOT/'processed/baseline_v2/split_subjects.json'),
                  unresolved=['final hardware operating threshold','float32 firmware parity','wrist-domain validation','runtime','RAM high-water mark','power'])
    # Verification reconstructs objects by assigning saved fitted attributes, never fit().
    reconstructed=LogisticRegression(**manifest['logistic']['hyperparameters'])
    reconstructed.coef_=np.array([manifest['logistic']['coefficients']],dtype=np.float64)
    reconstructed.intercept_=np.array([manifest['logistic']['intercept']],dtype=np.float64)
    reconstructed.classes_=np.array(manifest['logistic']['classes'])
    reconstructed.n_features_in_=manifest['logistic']['n_features_in']
    sc=StandardScaler(with_mean=manifest['scaler']['with_mean'],with_std=manifest['scaler']['with_std'])
    sc.mean_=np.array(manifest['scaler']['mean']);sc.scale_=np.array(manifest['scaler']['scale'])
    sc.var_=np.array(manifest['scaler']['variance']);sc.n_features_in_=3;sc.n_samples_seen_=manifest['scaler']['n_samples_seen']
    rows=[r for r in csvrows(MODEL/'event_predictions.csv') if r['model']=='logistic' and r['split']=='validation']
    assert len(rows)==4387
    X=np.array([[float(r[n]) for n in manifest['features']['order']] for r in rows],dtype=np.float64)
    saved=np.array([float(r['score']) for r in rows]);authoritative=fitted.predict_proba(scaler.transform(X))[:,1]
    scores=reconstructed.predict_proba(sc.transform(X))[:,1]
    np.testing.assert_array_equal(scores,authoritative);np.testing.assert_array_equal(scores,saved)
    assert scores.tobytes()==authoritative.tobytes()==saved.tobytes()
    pred=(scores>=points['balanced_reference']['score_threshold']).astype(np.uint8)
    np.testing.assert_array_equal(pred,[int(r['prediction']) for r in rows])
    threshold_checks={}
    for name,point in points.items():
        expected=(authoritative>=point['score_threshold']).astype(np.uint8)
        actual=(scores>=point['score_threshold']).astype(np.uint8)
        assert actual.tobytes()==expected.tobytes()
        threshold_checks[name]=dict(prediction_mismatches=0,uint8_bytes_identical=True,positive_decisions=int(actual.sum()))
        assert int(actual.sum())==point['validation_metrics']['emitted_positive_decisions']
    # Initialization constants are derived only from the existing stored SOS.
    from scipy.signal import sosfilt_zi
    manifest['acceleration_filter']['unit_constant_input_zi_per_section']=sosfilt_zi(np.array(plan['sos'])).tolist()
    OUT.mkdir(parents=True)
    dump('model_manifest.json',manifest)
    # Check the actual on-disk serialized manifest retains all inference parameters.
    frozen=read(OUT/'model_manifest.json');assert frozen==manifest
    after={str(p.relative_to(ROOT)):sha(p) for p in paths};assert before==after
    commit=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip()
    dump('provenance.json',dict(baseline_id='MCU_v1',source_paths_relative_to='workspace root',source_sha256=before,
                               code_git_head=commit,code_version_note='Git HEAD is contextual; individual file hashes are authoritative for uncommitted files.',
                               creation_method='source parameter copy plus algebraic coefficient/logit derivation; no fit/tune/replay',
                               creator_code_path=str(Path(__file__).relative_to(ROOT)),
                               libraries=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__),
                               model_manifest_sha256=sha(OUT/'model_manifest.json'),test_data_evaluated=False))
    dump('verification.json',dict(status='passed',feature_order_exact=True,model_parameters_exact=True,hyperparameters_exact=True,
                                 validation_candidates=4387,validation_score_mismatches_saved_csv=0,
                                 validation_score_mismatches_authoritative_joblib=0,maximum_absolute_score_error=0.,
                                 float64_score_bytes_identical=True,reference_prediction_mismatches_saved_csv=0,
                                 operating_point_checks=threshold_checks,serialized_manifest_round_trip_exact=True,
                                 source_artifacts_modified=[],all_source_hashes_unchanged=before==after,
                                 no_training_or_retuning=True,no_test_evaluation=True,
                                 verification_method='Assign saved fitted attributes to sklearn containers; use identical transform/predict_proba operations on saved validation features; no fit method called.',
                                 fused_float32_parity_verified=False,final_hardware_threshold_frozen=False))
    parameter_table='\n'.join('| '+n+' | '+repr(mu)+' | '+repr(scale)+' | '+repr(w)+' |' for n,mu,scale,w in zip(exported['feature_names'],exported['mean'],exported['scale'],exported['coefficients']))
    metric_table='\n'.join('| '+name+' | '+repr(point['score_threshold'])+' | '+str(point['validation_metrics']['fall_event_sensitivity'])+' | '+str(point['validation_metrics']['adl_false_alarms_per_hour'])+' | '+str(point['validation_metrics']['event_precision'])+' | '+str(point['validation_metrics']['adl_trial_specificity'])+' | '+str(point['validation_metrics']['trial_balanced_accuracy'])+' |' for name,point in points.items())
    doc=[ '# MCU v1 model freeze',STATEMENT,
          '**The logistic weights are frozen. The final hardware operating threshold is NOT frozen.** Neither documented operating point below is silently promoted to deployment. Five-feature and prospective seven-feature expansions are excluded and preserved only as comparison artifacts.',
          '## Complete pipeline',
          'Raw ADXL345 XYZ counts -> counts/256 in g -> causal two-biquad 5 Hz acceleration filter; in parallel raw g -> gravity EMA -> magnitude and gravity-aligned streams -> magnitude/jerk rising-edge candidate trigger -> circular prehistory and delayed 300-sample window -> three features -> training StandardScaler -> fixed logistic score -> explicitly chosen operating threshold -> FALL/non-fall. Filter/gravity/trigger/history run continuously; window aggregation and classifier inference run for completed candidates. Gyroscope is not used.',
          '## Exact preprocessing',
          'Sampling is 200 Hz. The causal filter is fourth-order Butterworth with 5 Hz nominal cutoff, two cascaded direct-form-II-transposed biquads per axis. SOS rows `[b0,b1,b2,a0,a1,a2]` are:\n\n```json\n'+json.dumps(plan['sos'],indent=2)+'\n```',
          'Per section: `y=b0*x+z1; z1_new=b1*x-a1*y+z2; z2_new=b2*x-a2*y`. Initialize `zi=sosfilt_zi(SOS)*first_raw_sample_g` for each axis; keep states across triggers/windows and reset only per independent trial. The manifest includes unit-input zi values as a reproducibility aid. There is no backward pass or future-dependent phase compensation.',
          f"Gravity uses raw, unfiltered acceleration in g: `G[t]=alpha*r[t]+(1-alpha)*G[t-1]`, `alpha=1-exp(-2*pi*0.5/200)={orientation['gravity_alpha']}`, `G[-1]=r[0]`. Preserve history across windows. Normalize with `u=G/max(norm(G),1e-8)`; the denominator floor is part of the frozen behavior.",
          'Numeric reference: float64 filter/gravity; filtered samples cast float32 then back to float64 before derived calculations; float64 features, scaler and logistic inference. An all-float32 port is not yet certified equivalent.',
          '## Trigger and event window',
          '`m[t]=norm(a[t])`; `j[t]=200*abs(m[t]-m[t-1])`, with `j[0]=0`. Combined condition `H[t]=(m[t]>=1.1 g) OR (j[t]>=5.0 g/s)`. Accept only `H[t] AND NOT H[t-1]` and at least 300 samples (1.5 s) since the previous accepted trigger. Start with previous-high=false and last-trigger=-300. Crossings in refractory are discarded; remaining high does not retrigger. Even an accepted trigger with an incomplete window updates refractory state.',
          '`pre100_post200` means `[k-100,k+200)` for trigger k: 100 pre-trigger samples, then 200 samples including k through k+199, total 300 (nominal 1.5 s). Decision occurs at k+200, exactly 1 s after trigger, and that current sample is excluded. Require k>=100 and k+200<trial_length; discard incomplete windows, without padding/shifting. The reference ring has 301 rows of m, h, abs(p). Post-trigger samples are causal because inference waits until they arrive.',
          '## Exact feature definitions and order',
          'For each window sample, use filtered `a`, raw-driven gravity `G`, `u=G/max(norm(G),1e-8)`, signed `p=a dot u`, residual `b=a-p*u`, and `h=norm(b)`. The ordered vector is:\n\n1. **ga_C2:** `max_i h[i]`, i=0..299, in g.\n2. **jerk_abs_mean:** `sum_(i=1..299) abs(200*(m[i]-m[i-1])) / 299`, in g/s. No cross-window difference is included.\n3. **ga_parallel_peak:** `max_i abs(p[i])`, i=0..299, in g.',
          'The jerk is the derivative of scalar magnitude, not vector jerk magnitude. Parallel acceleration retains gravity; no 1 g subtraction is performed. Epsilon normalization need not produce a unit vector at near-zero gravity.',
          '## Frozen logistic model',
          'Train-only StandardScaler, population standard deviation. Feature order is the table order. L2 logistic regression, C=10.0, class_weight=balanced, solver=liblinear, max_iter=5000, tol=1e-8, random_state=473. The manifest copies every exported estimator hyperparameter, including defaults, from the authoritative joblib object. No refit occurs.',
          '| Feature | Mean | Scale | Coefficient |\n|---|---:|---:|---:|\n'+parameter_table,
          f"Intercept: `{exported['intercept']}`. `z=intercept+sum(w_i*(x_i-mean_i)/scale_i)`; `score=1/(1+exp(-z))`; predict FALL when `score>=tau`.",
          f"Algebraic unstandardized-feature form: `w_raw={weights}`, `b_raw={bias}`. These are derived as `w/scale` and `intercept-sum(w_raw*mean)`, not refitted. Compare `dot(w_raw,x)+b_raw >= log(tau/(1-tau))` to omit runtime sigmoid. Fusion/float32 parity has not been verified, and the standardized equation remains the verified reference.",
          '## Documented operating points and validation',
          '| Operating point | Exact threshold | Sensitivity | ADL FA/hour | Event precision | ADL trial specificity | Trial BA |\n|---|---:|---:|---:|---:|---:|---:|\n'+metric_table,
          'Balanced reference: 350 detected / 25 missed falls, 376 positive decisions and 15 ADL alarms. Sensitivity candidate: 357 detected / 18 missed, 410 positive decisions and 39 ADL alarms. Neither is a final hardware threshold.',
          'Validation comprises 375 fall trials and 572 ADL trials, 3.0679416666666666 ADL hours. Matching uses trigger within +/-1 s of a fall trial’s raw-magnitude argmax proxy and a complete window. Credit at most one detected fall per trial. Precision divides credited detections by all positive candidate decisions, counting unmatched/duplicate fall-recording alarms as false positives. Trial BA averages fall-event sensitivity and the fraction of ADL trials with no alarm; it is not event specificity. Candidate recall is 370/375=98.6667%, an end-to-end ceiling; five no-matched-trigger misses cannot be recovered by classifier threshold changes.',
          '## Verification and provenance',
          'The serialized manifest preserves exact feature ordering, scaler/model parameters and estimator hyperparameters. Reconstructed inference produces bit-for-bit identical float64 scores for all 4,387 validation candidates against both authoritative logistic.joblib and saved CSV scores, with zero reference prediction mismatches. Both documented thresholds have zero prediction mismatches against the authoritative model. Reconstruction only assigns saved fitted attributes and calls transform/predict_proba; it never calls fit. All authoritative source hashes remained unchanged. See verification.json and provenance.json.',
          'Key sources: ../causal_trained_models_v1/model_parameters.json and logistic.joblib; ../causal_event_study/experiment_plan.json and selection.json; ../causal_sensitivity_analysis/operating_points.csv; ../sisfall/causal_events.py and orientation_features.py. provenance.json provides workspace-relative paths and SHA-256 hashes of every authoritative input and code file used, including the freeze script. Git HEAD is supplementary because individual files may be uncommitted.',
          '## Known limitations and unresolved work',
          'SisFall is waist-mounted, simulated-fall data, not wrist-domain validation. Raw-magnitude peak matching is proxy supervision, not annotated onset/impact. Validation has been reused for trigger/window/model/threshold development and is not an independent performance guarantee. Balanced logistic scores are not automatically calibrated probabilities. Short scripted ADL recordings do not establish real-world daily alarm rates. Desktop verification does not establish float32 firmware parity, actual nRF5340 execution time, RAM high-water mark, power, sensor timing, or live boot/shutdown behavior. No test optimization or new test evaluation was performed.',
          '| Frozen for MCU v1 | Unresolved |\n|---|---|\n| 200 Hz accelerometer units/conversion; saved causal SOS, initialization and numeric reference | Hardware acquisition/calibration and float32 parity |\n| Raw gravity EMA, alpha and normalization floor | Wrist gravity-estimation behavior |\n| Selected trigger, rising-edge/refractory semantics and pre100_post200 bounds | Real wrist performance and annotated timing validation |\n| Exactly three features and their order; saved scaler, logistic weights and hyperparameters | Final hardware operating threshold |\n| Two named documented thresholds retained as alternatives | Runtime, RAM, power measurements and firmware qualification |\n| Source paths/hashes and desktop verification | Any algorithm/model/feature change requires MCU v2 or an experimental branch |']
    (OUT/'MODEL_FREEZE.md').write_text('\n\n'.join(doc)+'\n',encoding='utf-8')
    (OUT/'README.md').write_text('# MCU v1 algorithm baseline\n\n'+STATEMENT+'\n\n'
        'This directory is an artifact freeze for implementation, not a new training run or a hardware-certified detector. The 5-feature comparison and prospective 7-feature expansion are excluded. Prior experiments remain untouched.\n\n'
        '**Final hardware operating threshold: NOT FROZEN.** Preserve the balanced reference `0.8850929456953477` and sensitivity candidate `0.7742031648776018` as documented alternatives. Neither is the implicit final deployment choice. Any eventual hardware-threshold decision must be recorded explicitly rather than silently editing this freeze.\n\n'
        '- [MODEL_FREEZE.md](MODEL_FREEZE.md): complete frozen pipeline and limitations.\n'
        '- [model_manifest.json](model_manifest.json): exact algorithm, scaler and model parameters.\n'
        '- [provenance.json](provenance.json): authoritative source hashes and code/version context.\n'
        '- [verification.json](verification.json): exact validation score/prediction parity and unchanged-source checks.\n\n'
        'Further feature/model tuning must create a new version, never edit v1. Float32 firmware parity, the final deployment threshold, real wrist validation, runtime, RAM and power measurements remain unresolved. No neural network, new feature expansion, refit, retuning or test-set optimization is part of this freeze.\n',encoding='utf-8')
    assert {str(p.relative_to(ROOT)):sha(p) for p in paths}==before
    print('Created: '+', '.join(p.name for p in sorted(OUT.iterdir())))
    print('4387 validation scores byte-identical; zero prediction mismatches; all authoritative sources unchanged.')

if __name__=='__main__':main()
