# Prototype evidence inventory

Raw JSON and CSV files are committed so results can be audited independently of the narrative report.

- `small-sustained-180s.json` — passing Small/512/4096 sustained run.
- `small-sustained-gpu.csv` — one-second system-wide Apple GPU counters during that process.
- `base-30s.json` — failing Base real-time run.
- `small-2048-buffer-failure.json` — deterministic buffer-capacity failure that forced the 4096 decision.
- `small-low-latency-30s.json` — failing Small/256/3840 experiment.
- `small-256x4096-30s.json` — passing but non-sustained Small/256/4096 experiment.
- `small-conditioning-sweep.json` and its spectrogram — passing 16-second continuous-control capture run.
- `small-control-step-10s.json` and `control-latency-analysis.json` — direct causal control-to-output measurement.
- `machine.txt` and `model-assets.sha256` — redacted machine/tool record and local asset identity.
- `local-app-smoke.json` — passing standalone GUI/CoreAudio/MRT2 interaction smoke test.
- `source-connected-app-smoke.json` — passing golden-track audio-prefill,
  parallel-transport, semantic-control, and Home-return test from the exact
  delivered app bundle.

Generated WAV files are deliberately ignored by Git. The conditioning capture SHA-256 is `d3b6334b5dd988cc5636b61cfc9305fb9ff694b0c5d0dc8bef92f1a1a00a81ec`; the differential-test WAV hashes are recorded in `control-latency-analysis.json`.
