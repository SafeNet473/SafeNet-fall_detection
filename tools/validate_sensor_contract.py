"""Float32 replay + isolated LSM6DSOX input/rate experiments; frozen files read-only.

Run: python tools/validate_sensor_contract.py [--compiler PATH]
Uses exactly the recordings and golden CSVs of outputs/host_pc_validation.
No training, no threshold changes, no production resampling or MCU integration.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True
from generate_golden_reference import (
    ROOT, STREAM_NAMES, EVENT_NAMES, c, compare, digest, joblib, np,
    read_csv, reference, write_csv,
)
from scipy.signal import butter, resample_poly, sosfilt_zi

sys.path.insert(0, str(ROOT / 'experiments/sampling_rate'))
from linear_resampler import run as causal_linear_resample

OUT = ROOT / 'outputs/sensor_contract_validation'
EXPERIMENT = ROOT / 'experiments/sampling_rate/native208'
POINTS = [('balanced', 'balanced_reference'), ('sensitivity', 'sensitivity_oriented_candidate')]
STREAM_LIMITS = {name: 2e-10 for name in STREAM_NAMES}
STREAM_LIMITS.update(jerk=5e-8, high=0, edge=0, trigger=0)
EVENT_LIMITS = {name: 0 for name in EVENT_NAMES}
EVENT_LIMITS.update(ga_C2=2e-10, jerk_abs_mean=2e-10, ga_parallel_peak=2e-10, z=5e-9)


def build(compiler, name, source_dir=None, use_float=False, adapter=False):
    binary = OUT / (name + ('.exe' if os.name == 'nt' else ''))
    command = [compiler]
    if Path(compiler).stem.lower() == 'zig':
        command.append('cc')
    command += ['-std=c99', '-O2', '-UNDEBUG', '-Wall', '-Wextra', '-Werror', '-pedantic', '-ffp-contract=off']
    if adapter:
        command += ['-Iembedded', '-Isensor_adapters', 'tests/lsm6dsox_input_test.c',
                    'sensor_adapters/lsm6dsox_input.c']
    else:
        folder = source_dir or ROOT / 'embedded'
        runner = source_dir / 'host_validation.c' if source_dir else ROOT / 'tests/host_validation.c'
        command += ['-I' + str(folder)]
        if use_float:
            command.append('-DFALL_USE_FLOAT')
        command += [str(runner), *[str(folder / (name + '.c'))
                                   for name in ('filters', 'features', 'classifier', 'fall_detector')]]
    command += ['-o', str(binary)]
    if os.name != 'nt':
        command.append('-lm')
    env = os.environ.copy()
    env['ZIG_GLOBAL_CACHE_DIR'] = str(OUT / 'compiler_cache')
    result = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr)
    return binary, command


def run_replay(binary, path, directory, cli):
    directory.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(binary), 'replay', str(path), str(directory / 'c_streams.csv'),
                    str(directory / 'c_events.csv'), str(directory / 'c_history.csv'), cli], check=True)
    return read_csv(directory / 'c_streams.csv', 23)[:, 1:], read_csv(directory / 'c_events.csv', 11)


def array_error(expected, actual):
    if expected.shape != actual.shape:
        return dict(reference_shape=list(expected.shape), actual_shape=list(actual.shape))
    error = np.abs(expected - actual)
    return dict(max_abs=float(error.max()) if error.size else 0.,
                mean_abs=float(error.mean()) if error.size else 0.)


def logic_comparison(expected_s, actual_s, expected_e, actual_e):
    original_triggers = np.flatnonzero(expected_s[:, -1]).tolist()
    actual_triggers = np.flatnonzero(actual_s[:, -1]).tolist()
    expected_by_trigger = {int(row[0]): row for row in expected_e}
    actual_by_trigger = {int(row[0]): row for row in actual_e}
    shared = sorted(expected_by_trigger.keys() & actual_by_trigger.keys())
    unmatched = sorted(expected_by_trigger.keys() ^ actual_by_trigger.keys())
    indices = [key for key in shared
               if not np.array_equal(expected_by_trigger[key][:4], actual_by_trigger[key][:4])]
    labels = [key for key in shared if expected_by_trigger[key][8] != actual_by_trigger[key][8]]
    return dict(trigger_index_mismatches=len(set(original_triggers) ^ set(actual_triggers)),
                reference_trigger_indices=original_triggers, actual_trigger_indices=actual_triggers,
                event_index_mismatches=len(indices), event_index_mismatch_triggers=indices,
                unmatched_events=len(unmatched), unmatched_event_triggers=unmatched,
                classification_mismatches=len(labels), classification_mismatch_triggers=labels,
                matched_decisions=len(shared), reference_decisions=len(expected_e), actual_decisions=len(actual_e))


def behavior_diagnostics(expected_s, actual_s, expected_e, actual_e, threshold):
    mismatched = np.flatnonzero(np.any(expected_s[:, 19:22] != actual_s[:, 19:22], axis=1))
    if len(mismatched):
        index = int(mismatched[0])
        first_numeric = next((i for i in range(index + 1)
                              if np.any(expected_s[i, :19] != actual_s[i, :19])), None)
        return dict(first_behavior_sample=index, first_numerical_difference_sample=first_numeric,
                    stage='trigger condition/edge/refractory',
                    reference_magnitude=float(expected_s[index, 15]), actual_magnitude=float(actual_s[index, 15]),
                    reference_jerk=float(expected_s[index, 18]), actual_jerk=float(actual_s[index, 18]),
                    magnitude_threshold=1.1, jerk_threshold=5.)
    actual_by_trigger = {int(row[0]): row for row in actual_e}
    for row in expected_e:
        actual = actual_by_trigger.get(int(row[0]))
        if actual is not None and row[8] != actual[8]:
            return dict(stage='feature/score decision threshold', trigger=int(row[0]),
                        reference_features=row[4:7].tolist(), actual_features=actual[4:7].tolist(),
                        reference_score=float(row[7]), actual_score=float(actual[7]),
                        reference_logit_threshold=threshold, actual_logit_threshold=float(actual[9]))
    return None


def generate_native208():
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    sources = {}
    for source in sorted((ROOT / 'embedded').glob('*.[ch]')):
        content = source.read_text()
        sources[str(source.relative_to(ROOT))] = digest(source)
        if source.name in ('fall_detector.c', 'fall_detector.h', 'features.c', 'features.h'):
            content = re.sub(r'\b(100|200|300)\b',
                             lambda match: {'100': '104', '200': '208', '300': '312'}[match[0]], content)
            content = content.replace('299', '311')
        (EXPERIMENT / source.name).write_text('/* HOST EXPERIMENT ONLY: native 208 Hz, not MCU v1. */\n' + content)
    runner = (ROOT / 'tests/host_validation.c').read_text()
    runner = re.sub(r'\b(100|200|300)\b',
                    lambda match: {'100': '104', '200': '208', '300': '312'}[match[0]], runner)
    (EXPERIMENT / 'host_validation.c').write_text('/* HOST EXPERIMENT ONLY: 312-sample events. */\n' + runner)
    sos = butter(4, 5, fs=208, output='sos')
    zi = sosfilt_zi(sos)
    alpha = float(1 - np.exp(-2 * np.pi * .5 / 208))
    params = (EXPERIMENT / 'model_params.h').read_text()
    def matrix(values):
        return '{' + ',\n    '.join('{' + ', '.join(f'{x:.17g}' for x in row) + '}' for row in values) + '}'
    for name, value in [('FALL_SOS', matrix(sos)), ('FALL_ZI', matrix(zi))]:
        params = re.sub(r'(static const FallReal ' + name + r'\[.*?=\s*)\{.*?\};',
                        lambda match: match[1] + value + ';', params, flags=re.S)
    params = re.sub(r'(static const FallReal FALL_ALPHA = )[^;]+;',
                    lambda match: match[1] + f'{alpha:.17g};', params)
    (EXPERIMENT / 'model_params.h').write_text(params)
    config = dict(fs=208, pre=104, post=208, refractory=312, samples=312,
                  differences=311, sos=sos.tolist(), zi=zi.tolist(), alpha=alpha,
                  source_sha256=sources, model='same frozen weights/scaler/logits; no fit')
    (EXPERIMENT / 'experiment_manifest.json').write_text(json.dumps(config, indent=2))
    return config


def rate_reference(raw, config, estimator, tau):
    sos = np.asarray(config['sos'])
    state = np.asarray(config['zi'])[:, :, None] * raw[0][None, None, :]
    gravity_state = raw[0].copy()
    filtered = np.empty_like(raw)
    gravity = np.empty_like(raw)
    for index, sample in enumerate(raw):
        value = sample
        for section, (b0, b1, b2, _, a1, a2) in enumerate(sos):
            output = b0 * value + state[section, 0]
            z1 = b1 * value - a1 * output + state[section, 1]
            state[section, 1] = b2 * value - a2 * output
            state[section, 0] = z1
            value = output
        gravity_state = config['alpha'] * sample + (1 - config['alpha']) * gravity_state
        filtered[index] = value
        gravity[index] = gravity_state
    acceleration = filtered.astype(np.float32).astype(float)
    unit = gravity / np.maximum(np.linalg.norm(gravity, axis=1)[:, None], 1e-8)
    parallel = np.sum(acceleration * unit, axis=1)
    residual = acceleration - parallel[:, None] * unit
    scalars = c.derived(filtered, gravity)
    jerk = np.r_[0., np.abs(np.diff(scalars[:, 0]) * config['fs'])]
    high = (scalars[:, 0] >= 1.1) | (jerk >= 5.)
    edges = high & ~np.r_[False, high[:-1]]
    accepted = np.zeros(len(raw))
    last = -config['refractory']
    events = []
    for index in np.flatnonzero(edges):
        if index - last < config['refractory']:
            continue
        last = int(index)
        accepted[index] = 1
        if index >= config['pre'] and index + config['post'] < len(raw):
            start, end = int(index - config['pre']), int(index + config['post'])
            features = window_features(scalars[start:end], config['fs'])
            score = float(estimator['model'].decision_function(estimator['scaler'].transform(features[None]))[0])
            probability = float(estimator['model'].predict_proba(estimator['scaler'].transform(features[None]))[0, 1])
            events.append([index, start, end, end, *features, score, int(probability >= tau)])
    streams = np.column_stack((filtered, acceleration, gravity, unit, residual,
                               scalars[:, :2], parallel, jerk, high, edges, accepted))
    return streams, np.asarray(events).reshape(-1, 9)


def window_features(scalars, fs):
    return np.array([np.max(scalars[:, 1]), np.mean(np.abs(np.diff(scalars[:, 0]) * fs)),
                     np.max(scalars[:, 2])])


def features_from_streams(streams, trigger, pre, post, fs):
    window = streams[trigger - pre:trigger + post]
    scalars = np.column_stack((window[:, 15:17], np.abs(window[:, 17])))
    return window_features(scalars, fs)


def physical_candidate_comparison(original, changed, fs):
    used = set()
    matched_original = set()
    offsets = []
    bounds = []
    label_mismatches = 0
    for original_index, row in enumerate(original):
        possible = [(abs(candidate[0] / fs - row[0] / 200), index)
                    for index, candidate in enumerate(changed) if index not in used]
        if not possible:
            continue
        distance, index = min(possible)
        if distance > .025:
            continue
        used.add(index)
        matched_original.add(original_index)
        candidate = changed[index]
        offsets.append(float(candidate[0] / fs - row[0] / 200))
        bounds.append((candidate[:4] / fs - row[:4] / 200).tolist())
        label_mismatches += int(candidate[8] != row[8])
    return dict(matching_tolerance_seconds=.025, matched_events=len(used),
                unmatched_original=len(original) - len(used), unmatched_variant=len(changed) - len(used),
                matched_class_changes=label_mismatches, trigger_offsets_seconds=offsets,
                original_positive_decisions=int(original[:, 8].sum()),
                variant_positive_decisions=int(changed[:, 8].sum()),
                unmatched_original_events=[dict(trigger=int(row[0]), time_seconds=float(row[0] / 200), label=int(row[8]))
                                           for index, row in enumerate(original) if index not in matched_original],
                unmatched_variant_events=[dict(trigger=int(row[0]), time_seconds=float(row[0] / fs), label=int(row[8]))
                                          for index, row in enumerate(changed) if index not in used],
                max_abs_physical_bound_shift_seconds=float(np.max(np.abs(bounds))) if bounds else None)


def resampler_checks():
    times = np.arange(1041) / 208.
    ramp = np.column_stack((times, 2 * times - 1, np.full(len(times), 3.)))
    output, delays = causal_linear_resample(ramp)
    targets = np.arange(len(output)) / 200.
    expected = np.column_stack((targets, 2 * targets - 1, np.full(len(targets), 3.)))
    np.testing.assert_allclose(output, expected, rtol=0, atol=2e-15)
    assert len(output) == 1001 and np.min(delays) >= -1e-12
    assert np.max(delays) <= 24 / 25 / 208 + 1e-12
    prefix, _ = causal_linear_resample(ramp[:517])
    np.testing.assert_array_equal(prefix, output[:len(prefix)])
    sine = np.column_stack([np.sin(2 * np.pi * frequency * times) for frequency in (.5, 5., 50.)])
    sine_output, _ = causal_linear_resample(sine)
    sine_expected = np.column_stack([np.sin(2 * np.pi * frequency * targets) for frequency in (.5, 5., 50.)])
    error = np.abs(sine_output - sine_expected).max(axis=0)
    return dict(ramp_and_dc_passed=True, prefix_invariance_passed=True,
                sample_count_passed=True, no_future_extrapolation_passed=True,
                analytic_sine_max_abs_errors=dict(zip(['0.5_Hz', '5_Hz', '50_Hz'], error.tolist())))


def feature_distribution(pairs, estimator, manifest):
    baseline = np.array([row['baseline'] for row in pairs])
    variant = np.array([row['variant'] for row in pairs])
    delta = variant - baseline
    scales = estimator['scaler'].scale_
    result = dict(paired_windows=len(pairs), max_abs=(np.abs(delta).max(axis=0)).tolist(),
                  mean_abs=np.abs(delta).mean(axis=0).tolist(),
                  mean_signed_standardized_delta=(delta / scales).mean(axis=0).tolist(),
                  rms_standardized_delta=np.sqrt(np.mean((delta / scales) ** 2, axis=0)).tolist(),
                  max_abs_standardized_delta=np.abs(delta / scales).max(axis=0).tolist(), decisions={})
    for _, point in POINTS:
        tau = manifest['documented_operating_points'][point]['score_threshold']
        old_z = estimator['model'].decision_function(estimator['scaler'].transform(baseline))
        new_z = estimator['model'].decision_function(estimator['scaler'].transform(variant))
        old_labels = estimator['model'].predict_proba(estimator['scaler'].transform(baseline))[:, 1] >= tau
        new_labels = estimator['model'].predict_proba(estimator['scaler'].transform(variant))[:, 1] >= tau
        result['decisions'][point] = dict(max_abs_logit_shift=float(np.abs(new_z - old_z).max()),
                                         mean_abs_logit_shift=float(np.abs(new_z - old_z).mean()),
                                         class_changes=int(np.sum(new_labels != old_labels)),
                                         changed_windows=[pairs[i] for i in np.flatnonzero(new_labels != old_labels)])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compiler', default=str(ROOT / 'outputs/host_toolchain/ziglang/zig.exe'))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    prior = json.loads((ROOT / 'outputs/host_pc_validation/validation.json').read_text())
    assert prior['passed'], 'Previous mixed-precision suite must have passed'
    hashes = prior['protected_sha256']
    assert all(digest(ROOT / path) == sha for path, sha in hashes.items()), 'Validated inputs changed'
    manifest = json.loads((ROOT / 'outputs/mcu_v1_frozen/model_manifest.json').read_text())
    estimator = joblib.load(ROOT / 'outputs/causal_trained_models_v1/logistic.joblib')
    config208 = generate_native208()
    report = dict(protected_sha256=hashes, native208_configuration=config208, builds={}, float_cases=[],
                  strategy_cases=[], reconstruction='resample_poly 26/25; Kaiser beta=5; edge extension; trim timestamp span',
                  hardware_configured_range=None, hardware_configured_odr=None)
    report['linear_resampler_checks'] = resampler_checks()
    binaries = {}
    for name, folder, use_float, adapter in [('reference', None, False, False),
                                             ('float', None, True, False),
                                             ('native208', EXPERIMENT, False, False),
                                             ('adapter_test', None, False, True)]:
        print('Compiling ' + name, flush=True)
        binaries[name], command = build(args.compiler, name, folder, use_float, adapter)
        report['builds'][name] = command
    adapter_output = subprocess.check_output([str(binaries['adapter_test'])], text=True)
    report['adapter_tests_passed'] = True
    (OUT / 'adapter_test.txt').write_text(adapter_output)
    report['float_classifier_only'] = {}
    for cli, point in POINTS:
        classifier_path = OUT / ('float_classifier_' + cli + '.csv')
        subprocess.run([str(binaries['float']), 'classify',
                        str(ROOT / 'outputs/host_pc_validation/classifier_features.csv'),
                        str(classifier_path), cli], check=True)
        expected = read_csv(ROOT / 'outputs/host_pc_validation' / ('classifier_python_' + cli + '.csv'), 7)
        actual = read_csv(classifier_path, 7)
        report['float_classifier_only'][point] = dict(
            vectors=len(expected), score_error=array_error(expected[:, 4], actual[:, 4]),
            feature_cast_error=array_error(expected[:, 1:4], actual[:, 1:4]),
            threshold_error=array_error(expected[:, 5], actual[:, 5]),
            classification_mismatches=int(np.sum(expected[:, 6] != actual[:, 6])))
    trials = c.read_csv(ROOT / 'processed/baseline_v2/trials.csv')
    data_root = Path(json.loads((ROOT / 'processed/baseline_v2/config.json').read_text())['data_root'])
    names = list(dict.fromkeys(row['name'] for row in prior['replays']))
    assert len(names) == 7
    pairs = {'A_linear': [], 'B_native208': [], 'B_linear_reconstruction': []}
    native_parity = []
    latencies = []
    raw_range = []
    for name in names:
        trial = next(row for row in trials if Path(row['path']).stem == name)
        raw, sha = c.p.load_trial(data_root / trial['path'])
        assert sha == trial['sha256'] and trial['split'] == 'validation'
        original_input = ROOT / 'outputs/host_pc_validation' / name / 'input_g.csv'
        np.testing.assert_array_equal(np.loadtxt(original_input, delimiter=','), raw)
        raw_range.append(dict(name=name, peak_abs_axis_g=float(np.abs(raw).max()),
                              axis_samples_above_range={str(fs): int(np.sum(np.abs(raw) > fs)) for fs in (2, 4, 8, 16)}))
        # Offline bandlimited reconstruction, never called a causal sensor model.
        end_time = (len(raw) - 1) / 200.
        count208 = int(math.floor((len(raw) - 1) * 26 / 25)) + 1
        raw208 = resample_poly(raw, 26, 25, axis=0, window=('kaiser', 5.), padtype='edge')[:count208]
        raw208 = raw208.astype(np.float32).astype(float)
        raw208_linear = np.column_stack([np.interp(np.arange(count208) / 208., np.arange(len(raw)) / 200., raw[:, axis])
                                        for axis in range(3)]).astype(np.float32).astype(float)
        raw_a, delays = causal_linear_resample(raw208)
        raw_a = raw_a.astype(np.float32).astype(float)
        latencies.extend(delays.tolist())
        inputs = {}
        for key, values in [('native208', raw208), ('A_linear', raw_a), ('linear208', raw208_linear)]:
            path = OUT / name / (key + '_g.csv')
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(path, values, delimiter=',', fmt='%.17g')
            inputs[key] = path
        for cli, point in POINTS:
            baseline_dir = ROOT / 'outputs/host_pc_validation' / name / cli
            expected_s = read_csv(baseline_dir / 'python_streams.csv', 23)[:, 1:]
            expected_e = read_csv(baseline_dir / 'python_events.csv', 11)
            fresh_s, fresh_e = run_replay(binaries['reference'], original_input, OUT / name / ('reference_' + cli), cli)
            original_c_s = read_csv(baseline_dir / 'c_streams.csv', 23)[:, 1:]
            original_c_e = read_csv(baseline_dir / 'c_events.csv', 11)
            np.testing.assert_array_equal(fresh_s, original_c_s)
            np.testing.assert_array_equal(fresh_e, original_c_e)
            float_dir = OUT / name / ('float_' + cli)
            float_s, float_e = run_replay(binaries['float'], original_input, float_dir, cli)
            streams = compare(expected_s, float_s, STREAM_NAMES, STREAM_LIMITS)
            events = compare(expected_e[:, :9], float_e[:, :9], EVENT_NAMES, EVENT_LIMITS)
            logic = logic_comparison(expected_s, float_s, expected_e, float_e)
            threshold = float(expected_e[0, 9])
            comparison = dict(name=name, point=point, samples=len(raw), streams=streams,
                              events=events, logic=logic,
                              threshold_error=array_error(expected_e[:, 9], float_e[:, 9]),
                              minimum_reference_decision_margin=float(np.abs(expected_e[:, 7] - expected_e[:, 9]).min()),
                              vs_validated_c_streams={n: array_error(fresh_s[:, i], float_s[:, i])
                                                     for i, n in enumerate(STREAM_NAMES)},
                              behavior_diagnostic=behavior_diagnostics(expected_s, float_s, expected_e, float_e, threshold),
                              strict_numerical_parity=streams['passed'] and events['passed'])
            expected_history = read_csv(baseline_dir / 'python_history.csv', 5)
            actual_history = read_csv(float_dir / 'c_history.csv', 5)
            comparison['prehistory_indices_exact'] = np.array_equal(expected_history[:, :2], actual_history[:, :2])
            comparison['prehistory_values_error'] = array_error(expected_history[:, 2:], actual_history[:, 2:])
            report['float_cases'].append(comparison)
            print(f'Float {name}/{cli}: triggers={logic["trigger_index_mismatches"]}, '
                  f'events={logic["event_index_mismatches"]}, labels={logic["classification_mismatches"]}, '
                  f'strict numeric parity={comparison["strict_numerical_parity"]}', flush=True)
            tau = manifest['documented_operating_points'][point]['score_threshold']
            native_expected_s, native_expected_e = rate_reference(raw208, config208, estimator, tau)
            native_s, native_e = run_replay(binaries['native208'], inputs['native208'], OUT / name / ('native208_' + cli), cli)
            native_stream_comparison = compare(native_expected_s, native_s, STREAM_NAMES, STREAM_LIMITS)
            native_event_comparison = compare(native_expected_e, native_e[:, :9], EVENT_NAMES, EVENT_LIMITS)
            native_parity.append(dict(name=name, point=point, streams=native_stream_comparison,
                                      events=native_event_comparison,
                                      passed=native_stream_comparison['passed'] and native_event_comparison['passed']))
            write_csv(OUT / name / ('native208_' + cli) / 'python_events.csv', EVENT_NAMES, native_expected_e)
            a_expected_s, a_expected_e = reference(raw_a, manifest, estimator, point)
            a_s, a_e = run_replay(binaries['reference'], inputs['A_linear'], OUT / name / ('A_linear_' + cli), cli)
            assert compare(a_expected_s, a_s, STREAM_NAMES, STREAM_LIMITS)['passed']
            assert compare(a_expected_e, a_e[:, :9], EVENT_NAMES, EVENT_LIMITS)['passed']
            report['strategy_cases'].append(dict(name=name, point=point,
                original_events=len(expected_e), A_linear=physical_candidate_comparison(expected_e, a_e, 200),
                B_native208=physical_candidate_comparison(expected_e, native_e, 208)))
            if cli == 'balanced':
                # Fixed physical anchors isolate feature effects from candidate shifts.
                # Exclude 100 ms at both boundaries to limit offline FIR padding effects.
                alt_s, _ = rate_reference(raw208_linear, config208, estimator, tau)
                for event in expected_e:
                    trigger = int(event[0])
                    trigger208 = int(math.floor(trigger * 26 / 25 + .5))
                    if (event[1] / 200. < .1 or event[2] / 200. > end_time - .1
                            or trigger + 200 >= len(raw_a) or trigger208 + 208 >= len(raw208)):
                        continue
                    baseline_features = event[4:7].tolist()
                    for key, stream, anchor, pre, post, fs in [
                        ('A_linear', a_s, trigger, 100, 200, 200),
                        ('B_native208', native_s, trigger208, 104, 208, 208),
                        ('B_linear_reconstruction', alt_s, trigger208, 104, 208, 208)]:
                        variant = features_from_streams(stream, anchor, pre, post, fs)
                        pairs[key].append(dict(name=name, original_trigger=trigger,
                                               variant_trigger=anchor, baseline=baseline_features,
                                               variant=variant.tolist(), anchor_offset_seconds=anchor / fs - trigger / 200.))
        print('Completed raw recording ' + name, flush=True)
    # Repeat the successful suite's decision-exclusion and EOF checks in float.
    chosen = names[0]
    original_float_events = read_csv(OUT / chosen / 'float_balanced/c_events.csv', 11)
    original_float_streams = read_csv(OUT / chosen / 'float_balanced/c_streams.csv', 23)
    decision = int(original_float_events[-1, 3])
    input_g = np.loadtxt(ROOT / 'outputs/host_pc_validation' / chosen / 'input_g.csv', delimiter=',')
    mutation = input_g[:decision + 1].copy()
    mutation[decision] = [12., -12., 12.]
    np.savetxt(OUT / 'float_decision_mutated.csv', mutation, delimiter=',', fmt='%.17g')
    changed_s, changed_e = run_replay(binaries['float'], OUT / 'float_decision_mutated.csv',
                                     OUT / 'float_decision_mutation', 'balanced')
    np.testing.assert_array_equal(changed_e, original_float_events)
    assert np.any(changed_s[-1, :3] != original_float_streams[decision, 1:4])
    np.savetxt(OUT / 'float_before_decision.csv', input_g[:decision], delimiter=',', fmt='%.17g')
    _, before_e = run_replay(binaries['float'], OUT / 'float_before_decision.csv',
                             OUT / 'float_before_decision', 'balanced')
    np.testing.assert_array_equal(before_e, original_float_events[:-1])
    report['float_boundary_checks'] = dict(decision_sample_exclusion_passed=True, eof_check_passed=True,
                                           chronological_history_indices_exact=all(x['prehistory_indices_exact']
                                                                                   for x in report['float_cases']))
    report['native208_c_vs_python'] = native_parity
    report['fixed_window_feature_shift'] = {key: feature_distribution(value, estimator, manifest)
                                          for key, value in pairs.items()}
    (OUT / 'fixed_window_pairs.json').write_text(json.dumps(pairs, indent=2))
    report['interpolation_latency_seconds'] = dict(maximum=max(latencies), mean=float(np.mean(latencies)),
                                                  theoretical_maximum=24 / 25 / 208,
                                                  state='previous XYZ + input/output integer indices; current XYZ during push')
    report['raw_sensor_range_diagnostics'] = raw_range
    report['protected_inputs_unchanged'] = all(digest(ROOT / path) == sha for path, sha in hashes.items())
    report['mixed_reference_reproduced_bitwise'] = True
    def max_error(group, names):
        return max(case[group]['columns'][name]['max_abs'] for case in report['float_cases'] for name in names)
    report['float_summary'] = dict(recordings=len(names),
        decisions_tested=sum(row['logic']['reference_decisions'] for row in report['float_cases']),
        maximum_requested_stream_error=max_error('streams', ['fx', 'fy', 'fz', 'gx', 'gy', 'gz', 'm', 'h', 'p', 'jerk']),
        maximum_feature_error=max_error('events', ['ga_C2', 'jerk_abs_mean', 'ga_parallel_peak']),
        maximum_classifier_score_error=max_error('events', ['z']),
        trigger_index_mismatches=sum(row['logic']['trigger_index_mismatches'] for row in report['float_cases']),
        event_index_mismatches=sum(row['logic']['event_index_mismatches'] for row in report['float_cases']),
        unmatched_events=sum(row['logic']['unmatched_events'] for row in report['float_cases']),
        classification_mismatches=sum(row['logic']['classification_mismatches'] for row in report['float_cases']),
        strict_numerical_parity=all(row['strict_numerical_parity'] for row in report['float_cases']))
    report['experiment_validation_passed'] = (report['protected_inputs_unchanged']
                                              and all(x['passed'] for x in native_parity))
    (OUT / 'validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report['float_summary'], indent=2), flush=True)
    print('Protected reference unchanged:', report['protected_inputs_unchanged'], flush=True)
    print('Native 208 C/Python parity:', all(row['passed'] for row in native_parity), flush=True)
    if not report['experiment_validation_passed']:
        raise SystemExit('Experimental implementation/reference checks failed; inspect validation.json')


if __name__ == '__main__':
    main()
