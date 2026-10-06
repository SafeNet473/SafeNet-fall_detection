"""Validate timestamp-aware production boundary without touching the frozen core.

Reuses the seven recordings/208 Hz surrogates from the prior Strategy-A study.
Tests archived unquantized g parity separately from the complete int16 path.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
from generate_golden_reference import ROOT, STREAM_NAMES, EVENT_NAMES, compare, digest, joblib, np, read_csv, reference
sys.path.insert(0, str(ROOT / 'experiments/sampling_rate'))
from linear_resampler import run as strategy_a

OUT = ROOT / 'outputs/acquisition_validation'
EPOCH = 1000000000000
INPUT_PERIOD = 250000
OUTPUT_PERIOD = 260000
POINTS = [('balanced', 'balanced_reference'), ('sensitivity', 'sensitivity_oriented_candidate')]
STREAM_LIMITS = {name: 2e-10 for name in STREAM_NAMES}
STREAM_LIMITS.update(jerk=5e-8, high=0, edge=0, trigger=0)
EVENT_LIMITS = {name: 0 for name in EVENT_NAMES}
EVENT_LIMITS.update(ga_C2=2e-10, jerk_abs_mean=2e-10, ga_parallel_peak=2e-10, z=5e-9)


def build(compiler, source, name, float_mode, boundary=True):
    binary = OUT / (name + ('.exe' if os.name == 'nt' else ''))
    command = [compiler]
    if Path(compiler).stem.lower() == 'zig': command.append('cc')
    command += ['-std=c99', '-O2', '-UNDEBUG', '-Wall', '-Wextra', '-Werror', '-pedantic',
                '-ffp-contract=off', '-Iembedded', '-Isensor_adapters', '-Iacquisition']
    if float_mode: command.append('-DFALL_USE_FLOAT')
    command.append(source)
    if boundary:
        command += ['acquisition/timing_adapter.c', 'acquisition/acquisition.c', 'sensor_adapters/lsm6dsox_input.c']
    command += [f'embedded/{name}.c' for name in ('filters', 'features', 'classifier', 'fall_detector')]
    command += ['-o', str(binary)]
    if os.name != 'nt': command.append('-lm')
    env = os.environ.copy()
    env['ZIG_GLOBAL_CACHE_DIR'] = str(OUT / 'compiler_cache')
    result = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True)
    if result.returncode: raise RuntimeError(result.stderr)
    return binary, command


def mismatches(expected_s, actual_s, expected_e, actual_e):
    triggers_old = set(np.flatnonzero(expected_s[:, -1]).tolist())
    triggers_new = set(np.flatnonzero(actual_s[:, -1]).tolist())
    old = {int(row[0]): row for row in expected_e}
    new = {int(row[0]): row for row in actual_e}
    shared = old.keys() & new.keys()
    return dict(trigger_indices=len(triggers_old ^ triggers_new),
                event_indices=int(sum(not np.array_equal(old[k][:4], new[k][:4]) for k in shared)),
                unmatched_events=len(old.keys() ^ new.keys()),
                labels=int(sum(old[k][8] != new[k][8] for k in shared)),
                matched_decisions=len(shared))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compiler', default=str(ROOT / 'outputs/host_toolchain/ziglang/zig.exe'))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    prior = json.loads((ROOT / 'outputs/sensor_contract_validation/validation.json').read_text())
    hashes = prior['protected_sha256']
    assert all(digest(ROOT / path) == sha for path, sha in hashes.items())
    estimator = joblib.load(ROOT / 'outputs/causal_trained_models_v1/logistic.joblib')
    manifest = json.loads((ROOT / 'outputs/mcu_v1_frozen/model_manifest.json').read_text())
    report = dict(protected_sha256=hashes, timestamp_ticks_per_second=52000000,
                  grid_period_ticks=OUTPUT_PERIOD, nominal_sensor_period_ticks=INPUT_PERIOD,
                  max_gap_ticks=375000, builds={}, unit_tests={}, cases=[])
    names = list(dict.fromkeys(row['name'] for row in prior['float_cases']))
    inputs = {}
    for name in names:
        directory = OUT / name
        directory.mkdir(exist_ok=True)
        raw208 = np.loadtxt(ROOT / 'outputs/sensor_contract_validation' / name / 'native208_g.csv', delimiter=',')
        raw208 = raw208.astype(np.float32).astype(float)
        prior200 = np.loadtxt(ROOT / 'outputs/sensor_contract_validation' / name / 'A_linear_g.csv', delimiter=',')
        expected200, _ = strategy_a(raw208)
        expected200 = expected200.astype(np.float32).astype(float)
        np.testing.assert_array_equal(prior200, expected200)
        # Simulate the actual signed int16 representation, openly distinguishing
        # conversion/quantization changes from resampler implementation errors.
        ideal_counts = np.rint(raw208 / .000488)
        clipped = int(np.sum((ideal_counts < -32768) | (ideal_counts > 32767)))
        raw_counts = np.clip(ideal_counts, -32768, 32767).astype(np.int16)
        decoded = (raw_counts.astype(np.float32) * np.float32(.000488)).astype(float)
        decoded200, _ = strategy_a(decoded)
        decoded200 = decoded200.astype(np.float32).astype(float)
        with (directory / 'timestamped_g.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerows((EPOCH + index * INPUT_PERIOD, *row) for index, row in enumerate(raw208))
        with (directory / 'timestamped_counts.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerows((EPOCH + index * INPUT_PERIOD, *(int(x) for x in row))
                             for index, row in enumerate(raw_counts))
        np.savetxt(directory / 'strategy_a_g.csv', expected200, delimiter=',', fmt='%.17g')
        np.savetxt(directory / 'strategy_a_quantized_g.csv', decoded200, delimiter=',', fmt='%.17g')
        inputs[name] = dict(original=raw208, g=expected200, raw=decoded200, decoded=decoded,
                            clipped_axis_values=clipped,
                            quantization_max_g=float(np.max(np.abs(decoded - raw208))),
                            source_sha256=digest(ROOT / 'outputs/sensor_contract_validation' / name / 'native208_g.csv'))
    for float_mode in (False, True):
        mode = 'float' if float_mode else 'mixed'
        unit_binary, command = build(args.compiler, 'tests/timing_adapter_test.c', 'unit_' + mode, float_mode)
        report['builds']['unit_' + mode] = command
        unit_output = subprocess.check_output([str(unit_binary)], text=True)
        (OUT / ('unit_' + mode + '.txt')).write_text(unit_output)
        sizes = re.search(r'timing=(\d+) config=(\d+) detector=(\d+) acquisition=(\d+)', unit_output)
        assert sizes
        report['unit_tests'][mode] = dict(passed=True, state_bytes=dict(zip(
            ['timing', 'config', 'detector', 'acquisition'], map(int, sizes.groups()))))
        binary, command = build(args.compiler, 'tests/acquisition_replay.c', 'replay_' + mode, float_mode)
        report['builds']['replay_' + mode] = command
        oracle_binary, command = build(args.compiler, 'tests/host_validation.c', 'oracle_' + mode, float_mode, boundary=False)
        report['builds']['oracle_' + mode] = command
        print(f'{mode}: synthetic unit tests passed; state {report["unit_tests"][mode]["state_bytes"]}', flush=True)
        for name in names:
            for path in ('g', 'raw'):
                source = OUT / name / ('timestamped_g.csv' if path == 'g' else 'timestamped_counts.csv')
                expected200 = inputs[name][path]
                oracle_input = OUT / name / ('strategy_a_g.csv' if path == 'g' else 'strategy_a_quantized_g.csv')
                for cli, point in POINTS:
                    directory = OUT / name / f'{mode}_{path}_{cli}'
                    directory.mkdir(exist_ok=True)
                    samples_file, streams_file, events_file = [directory / f'{key}.csv' for key in ('samples', 'streams', 'events')]
                    subprocess.run([str(binary), path, str(source), str(samples_file), str(streams_file), str(events_file), cli], check=True)
                    samples = read_csv(samples_file, 6)
                    streams = read_csv(streams_file, 23)[:, 1:]
                    events = read_csv(events_file, 9)
                    indices = np.arange(len(expected200))
                    brackets = np.maximum(1, (indices * 26 + 24) // 25)
                    expected_samples = np.column_stack((indices, EPOCH + indices * OUTPUT_PERIOD,
                                                        EPOCH + brackets * INPUT_PERIOD, expected200))
                    sample_names = ['index', 'logical_ticks', 'bracket_ticks', 'ax', 'ay', 'az']
                    sample_limits = {key: 0 for key in sample_names}
                    prototype_comparison = compare(expected_samples, samples, sample_names, sample_limits)
                    # At exact coincidences the timestamp specification requires
                    # the original endpoint. The old Python experiment's general
                    # expression can cancel tiny surrogate components near zero.
                    # Compare the mathematical endpoint contract independently,
                    # while retaining the prototype's actual nonzero errors.
                    endpoint_expected = expected_samples.copy()
                    endpoints = expected_samples[:, 1] == expected_samples[:, 2]
                    source_samples = inputs[name]['original'] if path == 'g' else inputs[name]['decoded']
                    endpoint_expected[endpoints, 3:] = source_samples[brackets[endpoints]]
                    sample_comparison = compare(endpoint_expected, samples,
                                                ['index', 'logical_ticks', 'bracket_ticks', 'ax', 'ay', 'az'],
                                                sample_limits)
                    different = np.any(expected_samples[:, 3:] != samples[:, 3:], axis=1)
                    assert np.all(endpoints[different]), 'Any prototype discrepancy must be an exact endpoint'
                    subprocess.run([str(oracle_binary), 'replay', str(oracle_input), str(directory / 'oracle_streams.csv'),
                                    str(directory / 'oracle_events.csv'), str(directory / 'oracle_history.csv'), cli], check=True)
                    oracle_s = read_csv(directory / 'oracle_streams.csv', 23)[:, 1:]
                    oracle_e = read_csv(directory / 'oracle_events.csv', 11)[:, :9]
                    core_exact = np.array_equal(streams, oracle_s) and np.array_equal(events, oracle_e)
                    python_s, python_e = reference(expected200, manifest, estimator, point)
                    python_stream_comparison = compare(python_s, streams, STREAM_NAMES, STREAM_LIMITS)
                    python_event_comparison = compare(python_e, events, EVENT_NAMES, EVENT_LIMITS)
                    logic = mismatches(python_s, streams, python_e, events)
                    cached_exact = None
                    if path == 'g' and not float_mode:
                        cached = ROOT / 'outputs/sensor_contract_validation' / name / ('A_linear_' + cli)
                        cached_exact = (np.array_equal(streams, read_csv(cached / 'c_streams.csv', 23)[:, 1:])
                                        and np.array_equal(events, read_csv(cached / 'c_events.csv', 11)[:, :9]))
                    passed = sample_comparison['passed'] and core_exact and (cached_exact is not False)
                    if not float_mode:
                        passed = passed and python_stream_comparison['passed'] and python_event_comparison['passed']
                    else:
                        passed = passed and all(logic[key] == 0 for key in ('trigger_indices', 'event_indices', 'unmatched_events', 'labels'))
                    row = dict(name=name, path=path, precision=mode, point=point,
                               outputs=len(samples), decisions=len(events), generated_samples=sample_comparison,
                               archived_prototype_samples=prototype_comparison,
                               endpoint_rounding_different_outputs=int(different.sum()),
                               direct_200_c_bitwise_equal=core_exact, archived_strategy_a_bitwise_equal=cached_exact,
                               python_streams=python_stream_comparison, python_events=python_event_comparison,
                               logic=logic, passed=passed)
                    report['cases'].append(row)
                    print(f'{name} {mode}/{path}/{cli}: outputs={len(samples)} events={len(events)}, '
                          f'boundary exact={sample_comparison["passed"]}, core exact={core_exact}, passed={passed}', flush=True)
    report['quantization'] = []
    for name in names:
        data = inputs[name]
        points = {}
        for cli, point in POINTS:
            g_s = read_csv(OUT / name / f'mixed_g_{cli}/streams.csv', 23)[:, 1:]
            g_e = read_csv(OUT / name / f'mixed_g_{cli}/events.csv', 9)
            raw_s = read_csv(OUT / name / f'mixed_raw_{cli}/streams.csv', 23)[:, 1:]
            raw_e = read_csv(OUT / name / f'mixed_raw_{cli}/events.csv', 9)
            points[point] = mismatches(g_s, raw_s, g_e, raw_e)
        report['quantization'].append(dict(name=name, clipped_axis_values=data['clipped_axis_values'],
                                           maximum_decoded_g_difference=data['quantization_max_g'],
                                           source_sha256=data['source_sha256'], comparison_to_unquantized=points))
    report['frozen_inputs_unchanged'] = all(digest(ROOT / path) == sha for path, sha in hashes.items())
    report['passed'] = all(row['passed'] for row in report['cases']) and report['frozen_inputs_unchanged']
    report['implementation_sha256'] = {str(file.relative_to(ROOT)): digest(file)
        for directory in ('acquisition', 'sensor_adapters') for file in (ROOT / directory).glob('*.[ch]')}
    report['summary'] = dict(recordings=len(names), replay_comparisons=len(report['cases']),
                             mixed_decisions=sum(row['decisions'] for row in report['cases'] if row['precision'] == 'mixed'),
                             float_decisions=sum(row['decisions'] for row in report['cases'] if row['precision'] == 'float'),
                             generated_sample_mismatches=sum(not row['generated_samples']['passed'] for row in report['cases']),
                             detector_oracle_mismatches=sum(not row['direct_200_c_bitwise_equal'] for row in report['cases']),
                             trigger_mismatches=sum(row['logic']['trigger_indices'] for row in report['cases']),
                             event_index_mismatches=sum(row['logic']['event_indices'] for row in report['cases']),
                             classification_mismatches=sum(row['logic']['labels'] for row in report['cases']),
                             unmatched_events=sum(row['logic']['unmatched_events'] for row in report['cases']))
    (OUT / 'validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report['summary'], indent=2), flush=True)
    if not report['passed']:
        raise SystemExit('Acquisition validation FAILED. Inspect per-stage comparisons in validation.json.')


if __name__ == '__main__':
    main()
