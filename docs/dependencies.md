# Phase 0 dependencies

Captured 2026-09-01. Dependencies are pinned by `cmake/Dependencies.cmake`; model assets remain outside Git.

## Installed or downloaded

| Item | Exact version | Change and size |
|---|---:|---|
| Apple Metal Toolchain | component `17E188`; compiler `32023.883` | Optional Xcode component installed system-wide; Apple reported a 687.9 MB download. Xcode requires `TOOLCHAINS=com.apple.dt.toolchain.Metal.32023.883` on this machine. |
| CMake | `3.27.9` | Installed only in workspace Python 3.13.1 virtual environment; system CMake 4.3.2 was not replaced. |
| Build dependency cache | pinned sources and objects | Approximately 2.5 GB in the workspace after the native build. |
| Metal exported bundle | `17E188` | A temporary 666 MB workspace copy remains under `work/metal-export`; it is not part of the repository. |

No Homebrew package, kernel extension, audio driver, global Python package, or model weight was installed. FFmpeg 7.1 was already present and was used only to render the evidence spectrogram.

## Source pins

| Dependency | Tag | Commit | Role |
|---|---|---|---|
| [Magenta RealTime](https://github.com/magenta/magenta-realtime) | `v2.0.3` | `694a545e4ba0b88bf1150137b129582166d3e07f` | Official C++ runner and model integration |
| [JUCE](https://github.com/juce-framework/JUCE) | `8.0.13` | `7c9d3783b127263d72bb65fe0a7e2dc8a02a7ac2` | CoreAudio device host and later standalone-app path |
| [MLX](https://github.com/ml-explore/mlx) | `v0.31.1` | `ce45c52505c8158ea48d2a54e8caae05efd86bfe` | Apple GPU inference runtime |
| [TensorFlow](https://github.com/tensorflow/tensorflow) | `v2.21.0` | `a481b10260dfdf833a1b16007eead49c1d7febf3` | TensorFlow Lite MusicCoCa assets |
| [SentencePiece](https://github.com/google/sentencepiece) | `v0.2.0` | `17d7580d6407802f85855d2cc9190634e2c95624` | Prompt tokenization dependency |

The MLX shell-script compatibility replacement in `cmake/Dependencies.cmake` mirrors the patch carried by MRT2 v2.0.3 for its pinned MLX version.

## Model assets

The local asset directory is 4.3 GB. Key model files are 455,654,550 bytes for Small and 2,771,414,746 bytes for Base; complete sizes and SHA-256 values are in [model-assets.sha256](evidence/model-assets.sha256).

The official MRT2 repository states Apache-2.0 for code and CC BY 4.0 for released weights. JUCE 8 is dual-licensed under AGPLv3/commercial terms; per the prototype-only decision this was not treated as a Phase 0 blocker, but it must be revisited before distribution.
