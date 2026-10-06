# Isolated sampling-rate experiments

Nothing in this directory changes the frozen `embedded/` detector. Run
`python tools/validate_sensor_contract.py` to reproduce the float suite, sensor
adapter test, Strategy A experiment and generated native-208 Hz host variant.

`linear_resampler.py` is a chronological, two-sample interpolation prototype.
It emits a 200 Hz grid only once the required 208 Hz bracket has arrived. It is
an experiment, not an approved anti-aliasing filter or production adapter.

`native208/` is generated from the current C source with explicit changes:
208 Hz SOS and gravity alpha, jerk scaling 208, prehistory 104, post 208,
refractory 312, event length 312 and 311 within-window differences. Weights,
scaler folding and both operating thresholds are copied unchanged. The
modified host runner asserts these new experimental bounds. The directory
contains its own source headers, generated parameters and manifest; it never
overwrites or links against changed frozen preprocessing.

The offline 208 Hz surrogate is created from floating-point SisFall g samples
using polyphase resampling (26/25, Kaiser beta 5, edge extension), with output
trimmed to the original timestamp span. This offline reconstruction uses future
samples and models neither the LSM6DSOX electronics nor its noise/aliasing.
It is NOT the proposed real-time production resampler. Comparisons separately
measure changed candidate sets and fixed-physical-window feature shifts.
