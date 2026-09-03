cmake_minimum_required(VERSION 3.27)

set(CMAKE_OSX_DEPLOYMENT_TARGET "14.0" CACHE STRING "Minimum macOS deployment version" FORCE)
project(SongWorldPhase0 VERSION 0.0.1 LANGUAGES C CXX OBJC OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(SONG_WORLD_ENABLE_MRT2 "Build the Magenta RealTime 2 adapter" ON)
option(SONG_WORLD_BUILD_TESTS "Build deterministic Phase 0 tests" ON)
set(SONG_WORLD_GOLDEN_TRACK "" CACHE FILEPATH
    "Local test track to bundle in standalone lab apps; never committed")

include(cmake/Dependencies.cmake)

set(song_world_engine_sources
    engine/src/MockGenerativeEngine.cpp
    engine/src/SourceAnchorMixer.cpp
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

if(SONG_WORLD_ENABLE_MRT2)
    juce_add_gui_app(song_world_app
        PRODUCT_NAME "Song World Hybrid Anchors"
        COMPANY_NAME "Song World"
        BUNDLE_ID "com.songworld.hybridanchors"
        VERSION "0.3.0"
        NEEDS_MICROPHONE_PERMISSION FALSE
        NEEDS_CAMERA_PERMISSION FALSE
        HARDENED_RUNTIME_ENABLED FALSE
    )
    target_sources(song_world_app PRIVATE frontend/mac/Main.cpp)
    target_link_libraries(song_world_app PRIVATE
        song_world_engine
        juce::juce_audio_utils
        juce::juce_gui_extra
    )
    target_compile_definitions(song_world_app PRIVATE
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0
    )
    add_custom_command(TARGET song_world_app POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${mlx_BINARY_DIR}/mlx/backend/metal/kernels/mlx.metallib"
            "$<TARGET_FILE_DIR:song_world_app>/mlx.metallib"
        COMMENT "Copying mlx.metallib into the standalone app bundle"
    )

    juce_add_gui_app(song_world_source_gravity
        PRODUCT_NAME "Song World Generator Clock Lab"
        COMPANY_NAME "Song World"
        BUNDLE_ID "com.songworld.generatorclocklab"
        VERSION "0.7.0"
        NEEDS_MICROPHONE_PERMISSION FALSE
        NEEDS_CAMERA_PERMISSION FALSE
        HARDENED_RUNTIME_ENABLED FALSE
    )
    target_sources(song_world_source_gravity PRIVATE frontend/mac/SourceGravityMain.cpp)
    target_link_libraries(song_world_source_gravity PRIVATE
        song_world_engine
        juce::juce_audio_utils
        juce::juce_gui_extra
    )
    target_compile_definitions(song_world_source_gravity PRIVATE
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0
    )
    add_custom_command(TARGET song_world_source_gravity POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${mlx_BINARY_DIR}/mlx/backend/metal/kernels/mlx.metallib"
            "$<TARGET_FILE_DIR:song_world_source_gravity>/mlx.metallib"
        COMMENT "Copying mlx.metallib into the Generator + Clock Lab app bundle"
    )

    juce_add_gui_app(song_world_playable_window
        PRODUCT_NAME "Song World Playable Window Lab"
        COMPANY_NAME "Song World"
        BUNDLE_ID "com.songworld.playablewindowlab"
        VERSION "0.8.0"
        NEEDS_MICROPHONE_PERMISSION FALSE
        NEEDS_CAMERA_PERMISSION FALSE
        HARDENED_RUNTIME_ENABLED FALSE
    )
    target_sources(song_world_playable_window PRIVATE
        frontend/mac/PlayableWindowMain.cpp
    )
    target_link_libraries(song_world_playable_window PRIVATE
        song_world_engine
        juce::juce_audio_utils
        juce::juce_gui_extra
    )
    target_compile_definitions(song_world_playable_window PRIVATE
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0
    )
    add_custom_command(TARGET song_world_playable_window POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${mlx_BINARY_DIR}/mlx/backend/metal/kernels/mlx.metallib"
            "$<TARGET_FILE_DIR:song_world_playable_window>/mlx.metallib"
        COMMAND ${CMAKE_COMMAND} -E make_directory
            "$<TARGET_BUNDLE_DIR:song_world_playable_window>/Contents/Resources"
        COMMENT "Bundling MRT2 runtime"
        VERBATIM
    )
    if(SONG_WORLD_GOLDEN_TRACK)
        if(NOT EXISTS "${SONG_WORLD_GOLDEN_TRACK}")
            message(FATAL_ERROR
                "SONG_WORLD_GOLDEN_TRACK does not exist: ${SONG_WORLD_GOLDEN_TRACK}")
        endif()
        add_custom_command(TARGET song_world_playable_window POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${SONG_WORLD_GOLDEN_TRACK}"
                "$<TARGET_BUNDLE_DIR:song_world_playable_window>/Contents/Resources/golden-track.mp3"
            COMMENT "Bundling user-supplied local test track"
            VERBATIM
        )
    else()
        message(STATUS
            "Playable Window Lab will build without audio; set SONG_WORLD_GOLDEN_TRACK locally to bundle a test track")
    endif()
endif()

if(SONG_WORLD_BUILD_TESTS)
    enable_testing()
    add_executable(song_world_engine_tests tests/engine_tests.cpp)
    target_link_libraries(song_world_engine_tests PRIVATE song_world_engine)
    add_test(NAME song_world_engine_tests COMMAND song_world_engine_tests)
endif()
