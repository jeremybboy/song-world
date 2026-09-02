cmake_minimum_required(VERSION 3.27)

set(CMAKE_OSX_DEPLOYMENT_TARGET "14.0" CACHE STRING "Minimum macOS deployment version" FORCE)
project(SongWorldPhase0 VERSION 0.0.1 LANGUAGES C CXX OBJC OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(SONG_WORLD_ENABLE_MRT2 "Build the Magenta RealTime 2 adapter" ON)
option(SONG_WORLD_BUILD_TESTS "Build deterministic Phase 0 tests" ON)

include(cmake/Dependencies.cmake)

set(song_world_engine_sources
    engine/src/MockGenerativeEngine.cpp
    engine/src/WorldTransport.cpp
)

if(SONG_WORLD_ENABLE_MRT2)
    list(APPEND song_world_engine_sources engine/src/MRT2GenerativeEngine.cpp)
endif()

add_library(song_world_engine STATIC ${song_world_engine_sources})
target_include_directories(song_world_engine PUBLIC engine/include)
target_compile_features(song_world_engine PUBLIC cxx_std_20)
target_compile_definitions(song_world_engine PUBLIC
    SONG_WORLD_ENABLE_MRT2=$<BOOL:${SONG_WORLD_ENABLE_MRT2}>
)
target_link_libraries(song_world_engine PUBLIC juce::juce_audio_basics)

if(SONG_WORLD_ENABLE_MRT2)
    target_link_libraries(song_world_engine PRIVATE magentart::core)
endif()

juce_add_console_app(song_world_phase0 PRODUCT_NAME "Song World Phase 0")
target_sources(song_world_phase0 PRIVATE apps/phase0/main.cpp)
target_link_libraries(song_world_phase0 PRIVATE
    song_world_engine
    juce::juce_audio_devices
    juce::juce_events
)
target_compile_definitions(song_world_phase0 PRIVATE
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
)

if(SONG_WORLD_ENABLE_MRT2)
    add_custom_command(TARGET song_world_phase0 POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${mlx_BINARY_DIR}/mlx/backend/metal/kernels/mlx.metallib"
            "$<TARGET_FILE_DIR:song_world_phase0>/mlx.metallib"
        COMMENT "Copying mlx.metallib beside the Phase 0 harness"
    )
endif()

if(SONG_WORLD_BUILD_TESTS)
    enable_testing()
    add_executable(song_world_engine_tests tests/engine_tests.cpp)
    target_link_libraries(song_world_engine_tests PRIVATE song_world_engine)
    add_test(NAME song_world_engine_tests COMMAND song_world_engine_tests)
endif()
