# SafeNet — Wearable Fall Detection

SafeNet is a team project for **EECS 473: Advanced Embedded Systems at the University of Michigan**. We are developing a wearable prototype that detects falls by processing accelerometer data locally on an **nRF5340 running Zephyr RTOS**.

This repository contains the algorithm development, portable C detection pipeline, and host validation. Nordic firmware and broader system integration are maintained separately in private team repositories.

## Demo

**[Watch the prototype demo](https://drive.google.com/file/d/1_khGQGUkrOEyt7mjG0K-89TAA9hnMUFj/view?usp=drive_link)**

The video demonstrates a staged fall detected by the prototype. The detection result is relayed through an anchor node to a laptop, where an alert is displayed.

The demo illustrates prototype functionality. Detection performance across representative falls, daily activities, and wearing positions is still being evaluated.

## My Contributions — Richard Chang

My work focuses on the fall-detection algorithm and its integration into the embedded prototype. I contributed to:

- Evaluating acceleration features and lightweight classifiers using the SisFall dataset.
- Implementing the streaming detection pipeline in portable C.
- Validating the C implementation against the Python reference through replay tests.
- Integrating the detector with Nordic/Zephyr sensor acquisition and debugging the embedded data pipeline.
- Collaborating on the combined accelerometer/barometer detection logic and integration into the broader SafeNet system.
SafeNet is a collaborative project; these contributions describe my role within the team.

## System Architecture

The integrated prototype combines accelerometer and barometer measurements in its detection logic. The table below describes the acceleration-based pipeline contained in this repository. Barometer decision logic and Bluetooth Mesh communication are maintained in the private system integration repositories.

| Stage | Implementation |
|---|---|
| Acquisition | Accelerometer configured at nominal 208 Hz and ±16 g; hardware FIFO and a sample queue in the Nordic firmware |
| Input adaptation | Raw counts converted to g, followed by linear interpolation onto a uniform 200 Hz grid |
| Preprocessing | Causal fourth-order 5 Hz Butterworth filter and a separate gravity estimate |
| Event detection | Magnitude/jerk trigger with a 1.5-second refractory interval |
| Feature extraction | Streaming computation over 0.5 seconds before and 1 second following the trigger |
| Classification | Logistic regression score and threshold comparison |
| Diagnostics | UART classification results, features, acquisition statistics, and error counters |

The portable acquisition interface accepts timestamps. The current Nordic implementation reconstructs nominal sample times from FIFO sequence numbers; it does not measure or correct sensor clock drift.

Known sample loss or stream discontinuities reset the processing state and discard incomplete events.

## Features and Classifier

The acceleration-based classifier uses three features

- **`ga_C2`** — peak acceleration magnitude perpendicular to the estimated gravity direction.
- **`jerk_abs_mean`** — mean absolute rate of change of acceleration magnitude.
- **`ga_parallel_peak`** — peak absolute acceleration parallel to the estimated gravity direction.

Training-time feature scaling is folded into the classifier coefficients. Runtime inference requires a linear score and threshold comparison, with **no external ML runtime or dynamic memory allocation in the portable detection core**.

The core processes samples incrementally without retaining a complete event window. A completed event produces a decision approximately **one second after its trigger**, plus acquisition and processing delays.

## Validation and Current Status

### Host validation

Python-to-C replay validation covered **56,999 samples from seven recordings**. Across two documented operating points, all **264 classification decisions matched**, along with trigger and event indices.

The mixed/double-precision implementation serves as the numerical reference. The float32 build passed behavioral regression on the documented suite but did not achieve strict numerical parity with that reference.

These checks establish implementation consistency on the tested inputs. They do not establish real-world detection accuracy or validate timing and resource usage on the nRF5340.

### Hardware prototype

The integrated SafeNet prototype combines accelerometer and barometer measurements in its fall-detection logic. The system also includes Bluetooth Mesh communication, implemented by other team members.

My contributions focus on the detection pipeline, sensor integration, and system debugging. The public repository contains the portable acceleration-based core and host validation; the combined detection logic and communication integration are maintained in private team repositories.

The prototype is operational, while quantitative evaluation of detection accuracy, false alarms, and target resource usage is ongoing.

## Code and Technical Documentation

| Location | Contents |
|---|---|
| [`embedded/`](embedded/) | Portable C filters, streaming features, classifier, and event state machine |
| [`acquisition/`](acquisition/) | Input adaptation, timing, and restart handling |
| [`sensor_adapters/`](sensor_adapters/) | Sensor input conversion and input contract |
| [`tests/`](tests/) | Host replay and interface tests |
| [`tools/`](tools/) | Model export and validation utilities |
| [`outputs/`](outputs/) | Experiment results, model artifacts, and validation reports |

Further details:

- [Detection core API and build instructions](embedded/README.md)
- [Acquisition interface and timing semantics](acquisition/README.md)
- [Sensor input contract](sensor_adapters/INPUT_CONTRACT.md)
- [Frozen model specification](outputs/mcu_v1_frozen/MODEL_FREEZE.md)
- [Python-to-C replay validation](outputs/host_pc_validation/REPORT.md)
- [Portable C verification](outputs/portable_c_verification/REPORT.md)
- [Float32 and sensor-contract validation](outputs/sensor_contract_validation/REPORT.md)
- [Acquisition validation](outputs/acquisition_validation/REPORT.md)
