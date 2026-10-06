# MCU v1 algorithm baseline

**MCU v1 is frozen for implementation. Any future change to preprocessing, trigger, window, feature set, model weights, or feature ordering must create MCU v2 or a new experimental branch.**

This directory is an artifact freeze for implementation, not a new training run or a hardware-certified detector. The 5-feature comparison and prospective 7-feature expansion are excluded. Prior experiments remain untouched.

**Final hardware operating threshold: NOT FROZEN.** Preserve the balanced reference `0.8850929456953477` and sensitivity candidate `0.7742031648776018` as documented alternatives. Neither is the implicit final deployment choice. Any eventual hardware-threshold decision must be recorded explicitly rather than silently editing this freeze.

- [MODEL_FREEZE.md](MODEL_FREEZE.md): complete frozen pipeline and limitations.
- [model_manifest.json](model_manifest.json): exact algorithm, scaler and model parameters.
- [provenance.json](provenance.json): authoritative source hashes and code/version context.
- [verification.json](verification.json): exact validation score/prediction parity and unchanged-source checks.

Further feature/model tuning must create a new version, never edit v1. Float32 firmware parity, the final deployment threshold, real wrist validation, runtime, RAM and power measurements remain unresolved. No neural network, new feature expansion, refit, retuning or test-set optimization is part of this freeze.
