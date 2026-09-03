# Playable Window + Single-Stream Lab

## Result

The signed arm64 standalone bundle passed the automated two-traversal runtime
check on the Apple M3 Pro test machine. The user separately confirmed that the
single-stream playable-window interaction is fun and worth preserving. This is
not evidence that every semantic destination preserves source identity or that
generated rhythm remains phase-locked.

## Prepared window

- Source tempo/grid: 123 BPM; known first source downbeat at 0.397 seconds.
- Playable window: 62.836024–90.153098 seconds (14 bars, 27.317073 seconds).
- Immediate temporal prefill: 34.836024–62.836024 seconds (28 seconds).
- Source audio landmark: 62.836024–70.640902 seconds, looped to the supported
  10-second MusicCoCa audio-prompt input.
- Audio inside World: one MRT2 Small stream only; the source master advances
  silently and returns at its corresponding timeline position.

## Verified runtime evidence

The final packaged-app smoke test performed the complete window twice, including
one checkpoint replay:

- Persistent runner instances: 1.
- Model prefills: 1.
- Runner starts: 2 (initial session plus the explicit inter-session replay).
- Semantic conditioning updates: 54.
- Generator resets during World navigation: 0.
- Entries / completed entries: 2 / 2.
- Automatic exits / completed exits: 2 / 2.
- Audio underruns / dropped reads: 0 / 0.
- Minimum generated buffer: 22,272 samples of 24,576.
- MRT2 inference: 16.256 ms/frame mean; 29.119 ms/frame maximum.
- Source/generated timeline drift: 0 samples.
- Home maximum sample error: 0.

The implementation reuses Collider's persistent `RealtimeRunner`, lock-free
producer/consumer buffer, audio-prompt slot, and live MusicCoCa blend-weight
updates. Replay restores the post-prefill model checkpoint only after the prior
window has ended; it is not a navigation reset.

## Honest boundary

The objective checks prove lifecycle, buffering, chronology, replay, and exact
Home behavior. They do not prove the perceived quality of the generated
continuation, semantic distinctness, harmonic compatibility, or exit phrasing;
those remain listening judgments. No commercial audio, model weights, generated
audio, secrets, or machine-local asset path is stored in Git.
