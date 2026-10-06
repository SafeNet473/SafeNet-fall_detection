"""Compile, replay raw counts, and compare C to unchanged causal Python functions.

Uses validation recordings only; no training, selection, or held-out evaluation.
Run: python tools/generate_golden_reference.py [--compiler path/to/zig.exe]
Existing project dependencies live in outputs/deps; compiler may be gcc/clang/zig.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT/'outputs/deps'))
sys.path.insert(0, str(ROOT/'outputs/sisfall'))
import numpy as np
import joblib
import causal_events as c
from export_model import export

OUT = ROOT/'outputs/portable_c_verification'
STREAM_NAMES = ('fx fy fz ax ay az gx gy gz ux uy uz bx by bz m h p jerk high edge trigger').split()
EVENT_NAMES = 'trigger start end decision ga_C2 jerk_abs_mean ga_parallel_peak z label'.split()

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def reference(raw, manifest, estimator, point):
    front = c.Frontend(raw[0])
    filtered = np.empty_like(raw)
    gravity = np.empty_like(raw)
    for i, sample in enumerate(raw):
        filtered[i], gravity[i] = front.push(sample)
    acc = filtered.astype(np.float32).astype(np.float64)
    unit = gravity/np.maximum(np.linalg.norm(gravity,axis=1)[:,None],c.of.GRAVITY_EPS)
    parallel = np.sum(acc*unit,axis=1)
    residual = acc-parallel[:,None]*unit
    values = c.derived(filtered,gravity)
    jerk = np.r_[0., np.abs(np.diff(values[:,0])*c.FS)]
    high = (values[:,0]>=1.1) | (jerk>=5.)
    edge = high & ~np.r_[False,high[:-1]]
    times = c.triggers(values,1.1,5.)
    accepted = np.zeros(len(raw)); accepted[times] = 1
    streams = np.column_stack((filtered,acc,gravity,unit,residual,
                               values[:,:2],parallel,jerk,high,edge,accepted))
    # ring_windows carries an obsolete tree prediction; disregard that field.
    # Use exactly its ring indexing and feature_row extraction, then frozen LR.
    events = c.ring_windows(values,{'frozen':times}, {'pre100_post200':(100,200)})
    features = np.array([[e['ga_C2'],e['jerk_abs_mean'],e['ga_parallel_peak']] for e in events]).reshape(-1,3)
    threshold = manifest['documented_operating_points'][point]['score_threshold']
    if len(features):
        standardized = estimator['scaler'].transform(features)
        z = estimator['model'].decision_function(standardized)
        probabilities = estimator['model'].predict_proba(standardized)[:,1]
        labels = probabilities >= threshold
        rows = [[e['trigger'],e['start'],e['end'],e['decision'],*f,score,int(label)]
                for e,f,score,label in zip(events,features,z,labels)]
    else:
        rows = []
    return streams, np.array(rows,dtype=float).reshape(-1,9)

def write_csv(path, names, values):
    np.savetxt(path,values,delimiter=',',header=','.join(names),comments='',fmt='%.17g')

def read_csv(path, columns):
    with path.open() as f:
        rows=list(csv.reader(f))[1:]
    return np.array(rows,dtype=float).reshape(-1,columns)

def compare(expected, actual, names, limits):
    result = {}
    if expected.shape != actual.shape:
        return {'passed':False,'shape_reference':list(expected.shape),'shape_c':list(actual.shape)}
    for i,name in enumerate(names):
        errors=np.abs(expected[:,i]-actual[:,i])
        maximum=float(errors.max()) if len(errors) else 0.
        mean=float(errors.mean()) if len(errors) else 0.
        limit=limits.get(name,1e-10)
        result[name]={'max_abs':maximum,'mean_abs':mean,'absolute_tolerance':limit,
                      'passed':bool(np.isfinite(actual[:,i]).all() and maximum<=limit)}
        if maximum>limit:
            result[name]['first_failed_row']=int(np.flatnonzero(errors>limit)[0])
    return {'passed':all(v['passed'] for v in result.values()),'columns':result}

def compile_replay(compiler, use_float):
    binary=OUT/('replay_float.exe' if use_float else 'replay_reference.exe')
    command=[compiler]
    if Path(compiler).stem.lower()=='zig': command+=['cc']
    command+=['-std=c99','-O2','-Wall','-Wextra','-Werror','-pedantic','-ffp-contract=off','-Iembedded']
    if use_float: command+=['-DFALL_USE_FLOAT']
    command += ['tests/replay_test.c', 'embedded/fall_detector.c','embedded/filters.c',
                'embedded/features.c','embedded/classifier.c','-o',str(binary)]
    if os.name!='nt': command+=['-lm']
    env=os.environ.copy()
    env['ZIG_GLOBAL_CACHE_DIR']=str(OUT/'compiler_cache')
    subprocess.run(command,cwd=ROOT,env=env,check=True,capture_output=True,text=True)
    return binary, command

def main():
    parser=argparse.ArgumentParser()
    local=ROOT/'outputs/host_toolchain/ziglang/zig.exe'
    parser.add_argument('--compiler',default=str(local) if local.exists() else shutil.which('gcc') or shutil.which('clang'))
    args=parser.parse_args()
    if not args.compiler: parser.error('Supply --compiler gcc/clang/zig path')
    OUT.mkdir(parents=True,exist_ok=True)
    authoritative=[ROOT/'outputs/mcu_v1_frozen/model_manifest.json',
        ROOT/'outputs/causal_trained_models_v1/logistic.joblib',
        ROOT/'outputs/sisfall/causal_events.py',ROOT/'outputs/sisfall/orientation_features.py',
        ROOT/'outputs/sisfall/pipeline.py',ROOT/'processed/baseline_v2/trials.csv',
        ROOT/'processed/baseline_v2/config.json']
    hashes={str(p.relative_to(ROOT)):digest(p) for p in authoritative}
    manifest=export()
    np.testing.assert_array_equal(c.SOS,manifest['acceleration_filter']['sos'])
    assert c.of.ALPHA==manifest['gravity']['alpha']
    estimator=joblib.load(authoritative[1])
    assert estimator['feature_names']==manifest['features']['order']
    np.testing.assert_array_equal(estimator['scaler'].mean_,manifest['scaler']['mean'])
    np.testing.assert_array_equal(estimator['scaler'].scale_,manifest['scaler']['scale'])
    np.testing.assert_array_equal(estimator['model'].coef_[0],manifest['logistic']['coefficients'])
    assert estimator['model'].intercept_[0]==manifest['logistic']['intercept']
    trials=c.read_csv(ROOT/'processed/baseline_v2/trials.csv')
    config=json.loads((ROOT/'processed/baseline_v2/config.json').read_text())
    cases=[]
    selected=[next(t for t in trials if t['split']=='validation' and t['activity_id']==activity)
              for activity in ('D01','D03','D04','D05','D06','D11','F01','F05','F13','F14','F15')]
    # Add an already documented detected walking fall and difficult missed
    # seated fall. This is regression case coverage, not parameter selection.
    saved_trials=c.read_csv(ROOT/'outputs/causal_trained_models_v1/trial_results.csv')
    for detected,activities in [('1',('F01',)),('0',('F13','F14','F15'))]:
        saved=next(t for t in saved_trials if t['model']=='logistic' and t['split']=='validation'
                   and t['activity'] in activities and t['detected']==detected)
        trial=next(t for t in trials if t['path']==saved['path'])
        if trial not in selected: selected.append(trial)
    for trial in selected:
        path=Path(config['data_root'])/trial['path']
        raw, sha=c.p.load_trial(path)
        assert sha==trial['sha256'] and len(raw)==int(trial['n_samples'])
        cases.append((Path(trial['path']).stem,path,raw,sha,'validation recording'))
    synth=OUT/'synthetic'; synth.mkdir(exist_ok=True)
    rng=np.random.default_rng(473)
    pulses=np.zeros((1500,3),dtype=int); pulses[:,2]=256
    for start in (100,160,400,460,700,760,1000,1060,1300):
        pulses[start:start+15,0]=1500
    arrays={'zero_gravity':np.zeros((800,3),dtype=int),
            'initial_high':np.tile([0,0,512],(800,1)),
            'pulses':pulses,
            'random_counts':rng.integers(-1000,1000,size=(2000,3))}
    # Obtain real causal trigger indices, then test EOF immediately before and
    # after the required decision sample. No precomputed features enter C.
    _,pulse_events=reference(pulses/256.,manifest,estimator,'balanced_reference')
    assert len(pulse_events)>0
    decision=int(pulse_events[0,3])
    arrays['eof_before_decision']=pulses[:decision]
    arrays['eof_at_decision']=pulses[:decision+1]
    pulse_streams,_=reference(pulses/256.,manifest,estimator,'balanced_reference')
    accepted=np.flatnonzero(pulse_streams[:,-1])
    assert np.any(np.diff(accepted)==300), 'Exercise exact refractory release'
    assert np.any((pulse_streams[:,-2]==1)&(pulse_streams[:,-1]==0)), 'Exercise discarded edges'
    semantics={'exact_300_sample_refractory_release':True,'refractory_edges_discarded':True}
    for name,counts in arrays.items():
        path=synth/(name+'.txt')
        np.savetxt(path,np.column_stack((counts,np.zeros((len(counts),6),dtype=int))),delimiter=',',fmt='%d')
        cases.append((name,path,counts.astype(float)/256.,digest(path),'synthetic boundary/causality'))
    report={'reference':'existing causal Frontend/derived/triggers/ring_windows + saved scaler/logistic',
            'deployment_threshold_selected':False,'authoritative_sha256':hashes,
            'boundary_checks':semantics,'builds':{}}
    # Explicit absolute tolerances chosen before replay. Exact logic/index/labels.
    # Default reference arithmetic preserves the original float32 interface.
    for use_float in (False,True):
        mode='experimental_float' if use_float else 'reference_precision'
        print('Building '+mode,flush=True)
        try:
            binary,command=compile_replay(args.compiler,use_float)
        except subprocess.CalledProcessError as error:
            print(error.stderr); raise
        build={'command':command,'state_bytes':int(subprocess.check_output([str(binary),'--size'],text=True)),
               'cases':[]}
        report['builds'][mode]=build
        for name,path,raw,sha,category in cases:
            for cli,point in [('balanced','balanced_reference'),('sensitivity','sensitivity_oriented_candidate')]:
                directory=OUT/mode/(name+'_'+cli); directory.mkdir(parents=True,exist_ok=True)
                expected_s,expected_e=reference(raw,manifest,estimator,point)
                write_csv(directory/'python_streams.csv',['index',*STREAM_NAMES],
                          np.column_stack((np.arange(len(raw)),expected_s)))
                write_csv(directory/'python_events.csv',EVENT_NAMES,expected_e)
                subprocess.run([str(binary),str(path),str(directory/'c_streams.csv'),
                                str(directory/'c_events.csv'),cli],check=True)
                actual_s=read_csv(directory/'c_streams.csv',23)
                actual_e=read_csv(directory/'c_events.csv',9)
                stream_limits={n:0 for n in ('high','edge','trigger')}
                for n in ('fx','fy','fz','ax','ay','az','gx','gy','gz','ux','uy','uz','bx','by','bz','m','h','p'):
                    stream_limits[n]=2e-10
                stream_limits['jerk']=5e-8
                event_limits={n:0 for n in ('trigger','start','end','decision','label')}
                event_limits.update(ga_C2=2e-10,jerk_abs_mean=2e-10,ga_parallel_peak=2e-10,z=5e-9)
                # Float build is evaluated against the SAME limits to make its
                # non-parity visible, never widened to force a passing result.
                streams=compare(expected_s,actual_s[:,1:],STREAM_NAMES,stream_limits)
                events=compare(expected_e,actual_e,EVENT_NAMES,event_limits)
                assert np.array_equal(actual_s[:,0],np.arange(len(raw)))
                case={'name':name,'category':category,'raw_sha256':sha,'samples':len(raw),
                      'operating_point':point,'accepted_triggers':int(expected_s[:,-1].sum()),
                      'events':len(expected_e),'positive_decisions':int(expected_e[:,-1].sum()),
                      'streams':streams,'event_comparison':events,'passed':streams['passed'] and events['passed']}
                build['cases'].append(case)
                print(f"{mode}: {name} {cli}: {len(expected_e)} events, passed={case['passed']}",flush=True)
        build['passed']=all(x['passed'] for x in build['cases'])
        before=OUT/mode/'eof_before_decision_balanced'
        after=OUT/mode/'eof_at_decision_balanced'
        full=OUT/mode/'pulses_balanced'
        np.testing.assert_array_equal(read_csv(before/'c_streams.csv',23),
                                      read_csv(full/'c_streams.csv',23)[:decision])
        np.testing.assert_array_equal(read_csv(after/'c_streams.csv',23),
                                      read_csv(full/'c_streams.csv',23)[:decision+1])
        assert len(read_csv(before/'c_events.csv',9))==0
        assert len(read_csv(after/'c_events.csv',9))==1
        np.testing.assert_array_equal(read_csv(after/'c_events.csv',9),read_csv(full/'c_events.csv',9)[:1])
        zero=read_csv(OUT/mode/'zero_gravity_balanced/c_streams.csv',23)
        assert np.isfinite(zero).all() and not zero[:,-1].any()
        initial=read_csv(OUT/mode/'initial_high_balanced/c_streams.csv',23)
        assert np.flatnonzero(initial[:,-1]).tolist()==[0]
        assert len(read_csv(OUT/mode/'initial_high_balanced/c_events.csv',9))==0
        build['prefix_and_eof_checks_passed']=True
    report['authoritative_inputs_unchanged']=all(digest(ROOT/p)==sha for p,sha in hashes.items())
    report['implementation_sha256']={str(p.relative_to(ROOT)):digest(p) for folder in ('embedded','tools','tests')
                                    for p in (ROOT/folder).glob('*') if p.suffix in ('.c','.h','.py')}
    report['compiler_version']=subprocess.check_output([args.compiler,'version'] if Path(args.compiler).stem=='zig'
                                                       else [args.compiler,'--version'],text=True).strip()
    (OUT/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    if not report['builds']['reference_precision']['passed'] or not report['authoritative_inputs_unchanged']:
        raise SystemExit('Reference precision verification FAILED; inspect comparison.json')
    print('Reference precision verified; experimental float results reported separately.',flush=True)

if __name__=='__main__':
    main()
