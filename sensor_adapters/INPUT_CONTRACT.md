# Selected detector / LSM6DSOX input contract

The deployment architecture is **LSM6DSOX effective ±16 g, nominal 208 Hz ->
explicit counts-to-g conversion -> timestamp-aware linear timing adapter ->
uniform 200 Hz g-valued XYZ including gravity -> unchanged frozen MCU v1 core**.

The driver must configure/read back the effective range and ODR before calling
`acquisition_init`. Register access is outside the unit converter and detector.
The converter supports all four explicit ranges; the selected acquisition path
accepts only ±16 g and nominal 208 Hz, failing closed for other configurations.
Sensitivity at ±16 g is 0.000488 g/LSB. The range enum is physical g, not FS bits.
Never pass LSM6DSOX raw counts to the ADXL345 `/256` detector helper.

Timestamp units are explicit in AcquisitionConfig, with a tick rate divisible
by 200. The first measurement anchors the logical grid. Two measurements are
required before any output; exact endpoints are copied, and other XYZ values
share one interpolation weight. Logical timestamps are exactly 5 ms apart.
The later bracket measurement timestamp is not the ISR/processing timestamp.
Actual driver/FIFO timebase reconstruction and counter extension remain driver
responsibilities. No missing data is filled across a rejected gap.

Duplicate, reversed or excessive-gap timing resets detector/event history as
an independent stream. Sensor reset, ODR change and full-scale change require
explicit notification and block further samples until fresh configuration and
read-back. No output extrapolation or EOF flush is allowed.

Mixed precision remains the numerical reference. Float remains the deployment
candidate, with exact integrated-vs-standalone behavior on the host suite but
known numerical differences versus mixed/Python. The final hardware classifier
operating threshold remains an explicit unresolved choice.

Host validation passed both precision builds, all 56 replay comparisons and
1056 decisions, with the prototype endpoint and sensor-quantization differences
reported separately. No frozen detector/model/reference file was changed.

See `acquisition/README.md` for the target-facing API and restart policy,
`outputs/acquisition_validation/REPORT.md` for measured evidence, and
`input_contract.json` for selected settings versus unverified hardware read-back.
No MCU driver, Zephyr, interrupts, BLE or alarm implementation is included.
