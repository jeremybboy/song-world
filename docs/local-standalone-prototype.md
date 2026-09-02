# Local standalone prototype

## Purpose

This milestone packages the proven Phase 0 engine as the smallest usable macOS
application. It is intentionally local-only and does not add song import, source
separation, polished design, or any GitHub publishing workflow.

## Experience

- **Play / Pause** controls a deterministic, project-owned 16-second test track.
- **Enter World** crossfades from that track into Magenta RealTime 2 Small.
- **Deep Dub**, **Warm Disco**, and **Airy Ambient** continuously interpolate the
  model's semantic conditioning while audio remains active.
- **Home / Original** crossfades back to the original track at its continuously
  advancing transport position.
- The status area exposes model frame time, buffered audio, and underruns.

## Runtime assets

The application expects the already-installed Phase 0 assets at:

```text
~/Documents/Magenta/magenta-rt-v2/models/mrt2_small/
~/Documents/Magenta/magenta-rt-v2/resources/
```

No additional downloads or system-level changes were made for this milestone.
The generated home track is synthesized in the application from project-owned
code, so it does not introduce a third-party media asset.

## Verification

The delivered arm64 bundle was exercised through its built-in automated smoke
mode on the target Mac. The test launched the real JUCE GUI and CoreAudio path,
loaded MRT2 Small, played the home track, entered the generated world, changed
semantic styles twice, returned home, and exited successfully. The machine-readable
result is stored in `docs/evidence/local-app-smoke.json`.

## Honest limitation

The generated world is not conditioned on the musical content of the home track.
It is a live semantic MRT2 stream crossfaded against a purpose-built reference
track. Track analysis, continuity-aware seeding, arbitrary import, and separation
remain later-product work and were deliberately excluded from this milestone.
