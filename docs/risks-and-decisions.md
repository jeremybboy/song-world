# Risks and decisions

This log distinguishes verified decisions from open questions. Phase-specific evidence belongs in the feasibility report and benchmark artifacts.

## Decisions

### D-001 — Instrument architecture survives model failure

**Status:** accepted.

The product keeps deterministic transport and audio I/O independent from MRT2. A `MockGenerativeEngine` must keep the audio/timing path runnable when MRT2 cannot initialize or sustain generation.

### D-002 — Native audio host

**Status:** accepted for prototype.

Use JUCE for the macOS audio-device harness and later standalone host. Treat commercial distribution licensing as a later product decision rather than a Phase 0 gate.

### D-003 — MRT2 integration boundary

**Status:** accepted.

Integrate the official Google C++ core through an adapter and use its `RealtimeRunner`/lock-free read boundary where verified. No other module may depend directly on MRT2 types.

## Open risks

### R-001 — Actual reference hardware differs from stated target

Read-only system inspection must establish the machine identity. Model-size conclusions apply only to the measured machine and must not be relabeled as M4 Pro evidence.

### R-002 — Base model real-time viability

The Small and Base variants require separate sustained tests. A successful short render is not evidence of real-time stability.

### R-003 — Control-to-audible latency

Measure prompt/semantic changes through the actual control, inference, ring-buffer, and audio path. Do not substitute Google's published latency target for a local measurement.

### R-004 — Simultaneous audio I/O and inference

Offline generation alone does not pass Phase 0. The gate requires real 48 kHz stereo device callbacks alongside active inference, with xrun/underrun accounting.

### R-005 — Upstream churn

Pin MRT2 and JUCE revisions. Record exact revisions, asset hashes, build commands, and any local compatibility patches.
