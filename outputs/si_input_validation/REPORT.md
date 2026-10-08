# Driver m/s² input validation

Added an explicit SI input path: `acquisition_push_lsm6dsox_ms2_sample()` converts
each axis using `g = acceleration_ms2 / 9.80665`, then shares the existing
timing/restart/detector path. The raw-count API remains supported. No files in
the frozen `embedded/` core were changed.

Host GCC builds used C99, `-O2 -UNDEBUG -Wall -Wextra -Werror -pedantic
-ffp-contract=off`, in both mixed and `FALL_USE_FLOAT` modes. Full compiler
arguments and test output are recorded in [results.json](results.json).

All six test executions passed:

- `si_acquisition_test`: known signed/fractional g conversions, zero, null
  output, NaN/infinity and unrepresentable values on every axis; conversion
  failure preserves output. Synthetic equivalent SI/raw-count replays produced
  2,884 logical samples and five matching completed decisions per operating
  point, for both existing operating points in each numeric build. Samples,
  timestamps, event indices, features, scores and labels matched exactly.
  Invalid SI input discarded an active event and restarted; duplicate and
  hardware-reset handling were also checked.
- Existing `timing_adapter_test`: interpolation, jitter, endpoint coincidence,
  discontinuities, configuration and one-hour zero-drift checks passed.
- Existing `lsm6dsox_input_test`: all four raw-count ranges passed unchanged.

These are host synthetic/regression checks, not a new replay of the complete
recorded-data corpus or live nRF5340 validation. The board application must
select the SI entry point for SI driver readings and supply acquisition
timestamps and confirmed configuration as before.
