# Portable frozen MCU v1 detector

`fall_detector_push_counts` accepts integer ADXL345 XYZ counts; it divides by 256.
`fall_detector_push_sample` accepts XYZ **in g**. Call once per chronological 200 Hz
sample. Use a separately initialized `FallDetector` for each independent stream.
Read `event` only when the return value is `FALL_ADL` or `FALL_DETECTED`.
`trace` contains current-sample diagnostics; score is the logistic **logit**.

`fall_detector_init` uses the documented balanced reference for host testing.
Use `fall_detector_init_operating_point` to choose either named documented point.
Neither is a frozen hardware deployment threshold. An explicitly selected custom
logit may be assigned to `logit_threshold` after initialization.

Core is C99 with standard `<math.h>`, `<stdint.h>`, and `<string.h>`; no heap,
I/O, OS, ML runtime, sensor driver, or platform dependencies. State is caller-owned.
Inputs must be finite and count conversion assumes the frozen ADXL345 scaling.
No samples may be dropped, duplicated, or reordered. Do not flush or pad at EOF:
an active incomplete event is discarded when the stream state is reset.

Default `FallReal=double` preserves frozen filter/gravity/feature arithmetic;
filtered acceleration is explicitly cast to float before derived streams.
`-DFALL_USE_FLOAT` is an experimental single-precision build, independently
reported by verification. Build without fast-math and with FP contraction disabled.

Run `python tools/export_model.py` to reproduce constants from the saved manifest.
Run `python tools/generate_golden_reference.py --compiler /path/to/zig.exe` to
compile both modes, replay validation recordings and synthetic boundary cases,
and write intermediate CSVs and machine-readable comparisons under
`outputs/portable_c_verification`. GCC and Clang paths also work. Existing Python
dependencies are read from `outputs/deps`; the C implementation requires none.
The harness never calls `fit` or adjusts thresholds/tolerances to obtain parity.

Example GCC/Clang build:

```sh
cc -std=c99 -O2 -Wall -Wextra -Werror -pedantic -ffp-contract=off -Iembedded \
  tests/replay_test.c embedded/fall_detector.c embedded/filters.c \
  embedded/features.c embedded/classifier.c -lm -o replay_test
./replay_test data/SA05/F01_SA05_R01.txt streams.csv events.csv balanced
```

Only the host replay program uses files. See the engineering report in
`outputs/portable_c_verification/REPORT.md` for formulas, indexing, memory,
precision results, and the remaining platform integration work.
