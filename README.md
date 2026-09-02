# Song World

Song World is a local-first macOS musical instrument prototype: a listener moves continuously through semantic musical territories while deterministic transport preserves the identity and chronology of the source song.

![Song World architecture](docs/assets/repository-overview.svg)

## Current phase

**Phase 0 — feasibility gate.** The current work proves that Apple Silicon can sustain 48 kHz stereo audio I/O beside Magenta RealTime 2 inference, accept continuous semantic conditioning, expose buffer/latency measurements, and preserve reusable timing boundaries.

Phase 0 deliberately does not build the polished standalone interface. Passing the gate authorizes, but does not automatically begin, Phase 1.

## Non-negotiable boundaries

- JUCE owns native macOS audio-device integration.
- A deterministic transport owns source beats, bars, and performance time.
- `GenerativeEngine` isolates model-specific behavior.
- Google MRT2 inference runs outside the real-time audio callback.
- The eventual exact Home state is the untouched source master.
- Model assets and test music remain local and outside Git.

## Repository workflow

The bootstrap commit is the only direct `main` change. All implementation work follows branch, focused commits, pull request, and user review/merge; agents never merge.

## Status

No Phase 0 claim is valid until the feasibility report records the exact machine, dependency revisions, runnable commands, sustained measurements, and failures.
