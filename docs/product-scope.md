# Product scope

## Thesis

Turn a linear recorded song into a playable, persistent musical world. The user should be able to move continuously among style destinations, preserve selected source-identity anchors, alter tempo, hold a musical moment, and return deterministically to the untouched source recording at the coherent source-musical position.

## Two-clock rule

Song World maintains separate clocks:

1. **Performance time** — elapsed real time in the user's session.
2. **Source musical position** — deterministic beat, bar, and section position in the composition.

Tempo changes and Hold/Loop may make the clocks diverge.

## V0 commitment

The committed V0 deliverable is a runnable standalone Apple Silicon macOS application. Phase 0 is only the engineering gate; it must produce reusable engine code rather than a disposable research notebook.

## V0 non-goals

- streaming-service integration;
- cloud accounts or mandatory cloud inference;
- remix or DAW export;
- mobile or Raspberry Pi targets;
- custom model training;
- a full sequencer or large genre ontology.
