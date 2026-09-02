include(FetchContent)

# Phase 0 pins exact upstream revisions so measurements remain reproducible.
set(FETCHCONTENT_QUIET OFF)

FetchContent_Declare(
    JUCE
    GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
    GIT_TAG 8.0.13
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(JUCE)

if(NOT SONG_WORLD_ENABLE_MRT2)
    return()
endif()

set(MLX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_PYTHON_STUBS OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_GGUF OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_CUDA OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    mlx
    GIT_REPOSITORY https://github.com/ml-explore/mlx.git
    GIT_TAG v0.31.1
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(mlx)

# Upstream MRT2 v2.0.3 carries this compatibility patch pending MLX #3607.
file(READ "${mlx_SOURCE_DIR}/mlx/backend/metal/make_compiled_preamble.sh" MLX_PREAMBLE_SCRIPT)
string(REPLACE
    "declare -a HDRS_LIST=($HDRS)"
    "declare -a HDRS_LIST=(); while read -r dots path; do [ -n \"$dots\" ] && HDRS_LIST+=(\"$dots\" \"$path\"); done <<< \"$HDRS\""
    MLX_PREAMBLE_SCRIPT
    "${MLX_PREAMBLE_SCRIPT}"
)
file(WRITE "${mlx_SOURCE_DIR}/mlx/backend/metal/make_compiled_preamble.sh" "${MLX_PREAMBLE_SCRIPT}")

set(SPM_ENABLE_SHARED OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    sentencepiece
    GIT_REPOSITORY https://github.com/google/sentencepiece.git
    GIT_TAG v0.2.0
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(sentencepiece)

set(TFLITE_ENABLE_XNNPACK OFF CACHE BOOL "" FORCE)
set(TFLITE_ENABLE_GPU OFF CACHE BOOL "" FORCE)
set(TFLITE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    tensorflow-lite
    GIT_REPOSITORY https://github.com/tensorflow/tensorflow.git
    GIT_TAG v2.21.0
    GIT_SHALLOW TRUE
    SOURCE_SUBDIR tensorflow/lite
)
set(FETCHCONTENT_SOURCE_DIR_TENSORFLOW
    "${CMAKE_BINARY_DIR}/_deps/tensorflow-lite-src"
    CACHE PATH "Reuse the pinned TensorFlow source for TFLite internals" FORCE
)
FetchContent_MakeAvailable(tensorflow-lite)

FetchContent_Declare(
    magenta_realtime
    GIT_REPOSITORY https://github.com/magenta/magenta-realtime.git
    GIT_TAG v2.0.3
    GIT_SHALLOW TRUE
    SOURCE_SUBDIR core
)
FetchContent_MakeAvailable(magenta_realtime)
