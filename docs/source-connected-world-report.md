# Source-connected World milestone

## Outcome

The standalone app now uses MRT2's real SpectroStream audio-prefill path. It
decodes the user-provided golden track locally, resamples it to the engine's
48 kHz timeline, encodes the first 28 seconds, and checkpoints 650 trimmed
source-token frames into MRT2 Small. The master and generated continuation then
advance as parallel timelines while playback is active.

Entering World crossfades to the source-seeded model stream. The three semantic
controls continuously steer MusicCoCa prompt weights. Home crossfades to the
decoded/resampled master at its continuously advancing source position instead
of restarting or returning to the handoff point.

## Delivered-bundle verification

The exact signed delivery bundle completed the automated GUI/CoreAudio/MRT2 test:

- source prefill: **10,939.484 ms** for **1,344,000 stereo frames**;
- effective prefill tokens: **650 frames** after the upstream 1-second head and
  tail trims;
- source anchor: **27.000 seconds**;
- synchronized playback exercised: **15.179 seconds**;
- generated audio reads: **1,423**;
- real-time underruns: **0**;
- peak generated level: **0.490204**;
- safety limiting: **1 sample**;
- semantic selections exercised: **4**;
- final source position after returning Home: **42.179 seconds**.

The raw report is `docs/evidence/source-connected-app-smoke.json`.

## Claim boundary

This proves that the model state is seeded from actual source audio and that the
source and World transports remain synchronized through the tested interaction.
It does **not** prove perceptual similarity, stable key/tempo over long World
sessions, or recognizable preservation of vocals, bass lines, or hooks. Those
are listening and analysis gates for the user-provided track.

## Known compromise

The current upstream prefill encoder uses a fixed 28-second input and trims one
second at each edge. The app therefore starts Home at 27 seconds and the prefill
has one second of non-causal lookahead beyond that anchor. Startup also waits
about 11 seconds for prefill on this M3 Pro. Arbitrary handoff points, causal
rolling prefill, beat/bar analysis, source separation, and long-run drift
correction are not implemented.

The commercial golden track is embedded only in the local delivered `.app`; it
is not staged or committed to Git.
