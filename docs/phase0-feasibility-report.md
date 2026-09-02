# Phase 0 feasibility report

**Gate decision: conditional go for Phase 1 using MRT2 Small; no-go for Base on this Mac.**

## Test boundary

Tests ran locally on 2026-09-01 on an Apple M3 Pro MacBook Pro (`Mac15,6`), not an M4 Pro: 11 CPU cores, 14 GPU cores, 36 GB unified memory, macOS 26.6.2. The harness uses JUCE/CoreAudio at 48 kHz stereo, the official MRT2 C++ `RealtimeRunner`, MLX Metal inference, a non-blocking audio callback, and a separate deterministic two-clock transport.

Prompts were `minimal dub techno, dry drums, instrumental` and `warm disco funk, acoustic drums, instrumental`. The continuous test swept their official MusicCoCa blend weights A→B→A every eight seconds.

## Gate answers

| Question | Verified answer |
|---|---|
| Can MRT2 run locally? | Yes. Official v2.0.3 built and generated through MLX Metal while JUCE drove the built-in speakers. |
| Which model is viable? | Small only on this M3 Pro. Base loads but cannot sustain its 25 Hz frame deadline. |
| Is audio I/O stable beside inference? | Yes for Small at 512 device / 4096 ring samples: 16,879 callbacks over 180.04 seconds with zero underruns or device errors. |
| Can conditioning move continuously? | Yes at the engine boundary: 3,367 thread-safe updates covered 0.000→1.000 during uninterrupted sustained playback; a captured sweep completed with zero underruns. |
| Is control latency measured? | Yes. A deterministic differential test measured 134.9 ms from the recorded A→B control write to the first changed PCM sample. |
| Are timing primitives established? | Yes. Performance samples and source beats/bars are separate atomics; the sustained run produced 180.043 performance seconds, 450.107 source beats, and 112.527 source bars at a 1.25 tempo ratio. |
| Is the code reusable? | Yes. `GenerativeEngine` isolates MRT2, `WorldTransport` owns chronology, and the audio callback only performs lock-free reads/copies. |

## Measured performance

| Run | Result | Frame avg / p95 / p99 | Real-time factor | Underruns | Peak process RSS | Measured-component control estimate p50 / p95 |
|---|---|---:|---:|---:|---:|---:|
| Small, 180 s, 512/4096 | **Pass** | 20.531 / 26.142 / 28.767 ms | 1.948× | 0 / 16,879 callbacks | 1009.188 MB | 94.329 / 114.783 ms |
| Base, 30 s, 512/4096 | **Fail** | 51.362 / 65.226 / 67.752 ms | 0.779× | 1,107 / 2,816 callbacks | 3238.047 MB | not meaningful under underrun |
| Small, 30 s, 256/3840 | **Fail** | 21.157 / 27.322 / 36.948 ms | 1.891× | 15 / 5,633 callbacks | 1009.062 MB | 85.185 / 105.153 ms |
| Small, 30 s, 256/4096 | Pass, not sustained | 21.024 / 26.616 / 28.806 ms | 1.903× | 0 / 5,627 callbacks | 1008.734 MB | 91.520 / 111.577 ms |

The sustained Small run averaged 0.139% of its 10.667 ms audio-callback budget and peaked at 37.750 microseconds in the callback. Across 164 one-second, system-wide GPU samples, device utilization was 38–78% (72.27% average), renderer utilization was 0–100% (57.34% average), and reported GPU in-use system memory was 1050.39–1949.94 MiB (1684.73 MiB average); these Apple counters are system-wide, not process attribution.

## Direct control latency and conditioning evidence

Two independent constant-A WAV captures were bit-for-bit identical (`2e3198…f359`), establishing deterministic repeatability. In a third capture, the A→B setter executed at 4.026 seconds and the first differing PCM frame appeared at 4.160854 seconds: **134.854 ms direct control-to-output latency**, with ±0.5 ms timestamp-rounding uncertainty; divergence exceeded 10 PCM counts at 145.437 ms and 100 counts at 159.708 ms.

This proves that changing the semantic blend reaches audible output causally during playback. It does not prove that a listener recognizes a new genre within 135 ms; semantic perception is gradual and requires listening evaluation, not a fabricated numeric threshold.

The separate 16.053-second sweep capture sent 300 continuous blend updates with zero underruns. Its WAV is intentionally excluded from Git; SHA-256 is `d3b6334b5dd988cc5636b61cfc9305fb9ff694b0c5d0dc8bef92f1a1a00a81ec`, and the committed spectrogram is [here](evidence/small-conditioning-sweep-spectrogram.png).

## Failures and compromises

1. The machine is M3 Pro, not the requested M4 Pro. These numbers cannot be presented as M4 evidence.
2. Base misses the 40 ms model-frame deadline by 28% on average. More RAM does not repair insufficient GPU throughput.
3. MRT2 emits 1,920-sample frames. A 2,048-sample virtual ring cannot enqueue the next whole frame before the consumer drains the first, causing 469 underruns in 20 seconds despite fast Small inference; 4,096 samples fixes stability but contributes roughly 61–85 ms of queued audio.
4. The directly measured Small control-to-output latency is ~135 ms. Reducing it requires upstream ring-buffer/producer work, not simply a smaller CoreAudio buffer.
5. Xcode 26.4 required a separately downloaded 687.9 MB Metal component and explicit toolchain selection. This is a setup cost and packaging risk.
6. No arbitrary-track import, analysis, separation, source-master Home path, Hold/Loop, or polished app exists yet. Those are V0 obligations, not Phase 0 claims.

No thermal or performance warning was recorded after the sustained run.

## Phase 1 recommendation — not started

Proceed with a JUCE standalone application around the existing engine boundaries, default to Small and 512/4096 on this M3 Pro, expose 256/4096 only as experimental, and disable Base unless a machine-specific calibration passes. Phase 1 should first add the exact source-master player and song-position map, then golden-track import/analysis/separation, while investigating partial-frame ring writes to reduce control latency without underruns.

Phase 0 ends here. No Phase 1 implementation is present.
