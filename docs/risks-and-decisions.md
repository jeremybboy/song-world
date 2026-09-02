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

### D-004 — M3 Pro model policy

**Status:** accepted from measured evidence.

Use MRT2 Small by default. Base is disabled on this machine unless a future calibration proves a sustained real-time factor above 1.0 with zero underruns.

### D-005 — Stable buffer policy

**Status:** accepted for Phase 1 baseline.

Use a 512-sample CoreAudio buffer and 4,096-sample MRT2 ring on this machine. A 2,048 ring is structurally too small for consecutive 1,920-sample model frames; 256/4,096 remains experimental because it passed only 30 seconds.

### D-006 — Control latency claim

**Status:** accepted with boundary.

Claim 134.854 ms direct control-to-first-output-divergence on this test, not an instantaneous semantic-perception response. Keep the measured-component 94.329/114.783 ms p50/p95 values labeled as estimates.

## Open risks

### R-001 — Actual reference hardware differs from stated target

**Resolved:** the test machine is an M3 Pro with 36 GB unified memory, not an M4 Pro. Results are labeled accordingly.

### R-002 — Base model real-time viability

**Resolved no-go:** Base averaged 51.362 ms per 40 ms frame and produced 1,107 underruns in 30 seconds.

### R-003 — Control-to-audible latency

**Resolved for first-sample response:** deterministic differential capture measured 134.854 ms. Perceptual semantic-recognition latency remains intentionally unclaimed.

### R-004 — Simultaneous audio I/O and inference

**Resolved for Small:** 180.04 seconds and 16,879 callbacks completed with zero underruns or device errors at 512/4,096.

### R-005 — Upstream churn

**Mitigated:** source revisions and all local asset hashes are recorded. Upstream API and Xcode component behavior still require revalidation on upgrade.

### R-006 — Latency versus whole-frame buffering

**Open for Phase 1.** The robust ring queues roughly 61–85 ms. Partial-frame producer writes or a different handoff are required to reduce this without repeating measured underruns.

### R-007 — V0 source-song pipeline

**Open by scope.** Phase 0 did not test arbitrary-track analysis, separation, exact Home playback, Hold/Loop, or synchronization against golden tracks.
