#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace song_world {

struct EngineConfig {
    std::string model_path;
    std::string resource_directory;
    std::string prompt_a{"minimal dub techno, dry drums, instrumental"};
    std::string prompt_b{"warm disco funk, acoustic drums, instrumental"};
    std::string prompt_c{"airy ambient electronica, soft percussion, instrumental"};
    // MRT2 emits 1,920-sample frames. Two frames are required so the producer
    // can enqueue the next frame before the audio callback drains the current one.
    std::size_t ring_buffer_samples{4096};
    float output_gain_db{-12.0F};
};

struct ConditioningState {
    float x{0.0F};
    float y{0.0F};
    float style_a{1.0F};
    float style_b{0.0F};
    float style_c{0.0F};
    std::uint64_t sequence{0};
};

struct EngineTelemetry {
    std::string backend;
    bool ready{false};
    float transformer_ms{0.0F};
    float frame_total_ms{0.0F};
    std::size_t buffer_available_samples{0};
    std::size_t buffer_capacity_samples{0};
    std::uint64_t dropped_audio_reads{0};
    std::uint64_t conditioning_sequence{0};
    double prepare_ms{0.0};
};

class GenerativeEngine {
public:
    virtual ~GenerativeEngine() = default;

    virtual bool prepare(const EngineConfig& config, std::string& error) = 0;
    virtual void start() = 0;
    virtual void stop() = 0;
    virtual void set_conditioning(const ConditioningState& state) noexcept = 0;

    // Single audio-thread consumer. Implementations must not allocate or block.
    virtual bool pull_audio(float* left, float* right, std::size_t samples) noexcept = 0;
    virtual EngineTelemetry telemetry() const = 0;
};

std::unique_ptr<GenerativeEngine> make_mock_generative_engine();

#if SONG_WORLD_ENABLE_MRT2
std::unique_ptr<GenerativeEngine> make_mrt2_generative_engine();
#endif

}  // namespace song_world
