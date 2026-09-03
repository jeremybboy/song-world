#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace song_world {

inline constexpr std::size_t kSemanticPromptSlots = 6;

struct EngineConfig {
    std::string model_path;
    std::string resource_directory;
    std::string prefill_model_path;
    std::string prompt_a{"minimal dub techno, dry drums, instrumental"};
    std::string prompt_b{"warm disco funk, acoustic drums, instrumental"};
    std::string prompt_c{"airy ambient electronica, soft percussion, instrumental"};
    std::string prompt_d{"dark club music, tense harmony, dry drums"};
    std::string prompt_e{"funk groove, syncopated bass, bright rhythm guitar"};
    std::string prompt_f{"drum and bass, fast breakbeats, deep bass"};
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
    float style_d{0.0F};
    float style_e{0.0F};
    float style_f{0.0F};
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
    bool source_prefilled{false};
    std::size_t source_prefill_frames{0};
    double source_prefill_ms{0.0};
    int prompt_encoder_status{0};
    int prompt_quantizer_status{0};
    std::array<int, kSemanticPromptSlots> audio_prompt_statuses{};
    int audio_reference_status{0};
};

class GenerativeEngine {
public:
    virtual ~GenerativeEngine() = default;

    virtual bool prepare(const EngineConfig& config, std::string& error) = 0;
    // Controller-thread only. Seeds the model from interleaved 48 kHz stereo
    // audio; implementations may block while encoding and rebuilding state.
    virtual bool prefill_source(const float* interleaved_stereo,
                                std::size_t frames,
                                std::string& error) = 0;
    virtual void start() = 0;
    virtual void stop() = 0;
    // Controller-thread only. Restores the checkpoint captured by the most
    // recent source prefill and restarts generation from that exact state.
    virtual bool restart_from_prefill(std::string& error) = 0;
    // Controller-thread only. MRT2 encodes changed text asynchronously while
    // generation continues from the current model state.
    virtual void update_semantic_prompts(
        const std::array<std::string, kSemanticPromptSlots>& prompts) = 0;
    // Controller-thread only. MRT2 expects up to ten seconds of mono 16 kHz
    // audio. Audio prompts replace the text embedding in the selected slot.
    virtual bool set_audio_prompt(std::size_t slot,
                                  const float* mono_16khz,
                                  std::size_t samples,
                                  const std::string& filename,
                                  std::string& error) = 0;
    bool set_audio_style_reference(const float* mono_16khz,
                                   std::size_t samples,
                                   const std::string& filename,
                                   std::string& error) {
        return set_audio_prompt(kSemanticPromptSlots - 1, mono_16khz, samples,
                                filename, error);
    }
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
