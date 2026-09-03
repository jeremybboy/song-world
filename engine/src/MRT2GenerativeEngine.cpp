#include <song_world/GenerativeEngine.h>

#include <magentart/realtime_runner.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <limits>
#include <thread>
#include <vector>

namespace song_world {
namespace {

class MRT2GenerativeEngine final : public GenerativeEngine {
public:
    ~MRT2GenerativeEngine() override { stop(); }

    bool prepare(const EngineConfig& config, std::string& error) override {
        using clock = std::chrono::steady_clock;
        const auto started = clock::now();

        if (!std::filesystem::exists(config.model_path)) {
            error = "MRT2 model path does not exist: " + config.model_path;
            return false;
        }
        if (!std::filesystem::is_directory(config.resource_directory)) {
            error = "MRT2 resource directory does not exist: " + config.resource_directory;
            return false;
        }

        runner_.set_buffer_size(config.ring_buffer_samples);
        runner_.set_volume_db(config.output_gain_db);
        runner_.set_latency_comp(false);

        if (!runner_.init_assets(config.resource_directory.c_str())) {
            error = "MRT2 failed to initialize MusicCoCa assets";
            return false;
        }

        const std::vector<std::string> prompts{
            config.prompt_a, config.prompt_b, config.prompt_c,
            config.prompt_d, config.prompt_e, config.prompt_f};
        const std::vector<float> initial_weights{1.0F, 0.0F, 0.0F,
                                                 0.0F, 0.0F, 0.0F};
        runner_.set_text_prompts(prompts, initial_weights);

        constexpr auto prompt_timeout = std::chrono::seconds(60);
        const auto prompt_started = clock::now();
        while (runner_.get_text_encoder_status() == 1 ||
               runner_.get_quantizer_status() == 1) {
            if (clock::now() - prompt_started > prompt_timeout) {
                error = "MRT2 prompt encoding exceeded 60 seconds";
                return false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        if (runner_.get_text_encoder_status() == 3 ||
            runner_.get_quantizer_status() == 3) {
            error = "MRT2 prompt encoding failed";
            return false;
        }

        runner_.set_blend_weights(initial_weights.data(),
                                  static_cast<int>(initial_weights.size()));
        if (!runner_.load_model(config.model_path.c_str())) {
            error = "MRT2 failed to load the exported MLX model";
            return false;
        }
        if (!config.prefill_model_path.empty()) {
            if (!std::filesystem::exists(config.prefill_model_path)) {
                error = "MRT2 prefill model path does not exist: " +
                        config.prefill_model_path;
                return false;
            }
            if (!runner_.load_prefill_model(config.prefill_model_path.c_str(), nullptr)) {
                error = "MRT2 failed to load the SpectroStream prefill encoder";
                return false;
            }
            prefill_available_.store(true, std::memory_order_release);
        }

        prepare_ms_ = std::chrono::duration<double, std::milli>(clock::now() - started).count();
        ready_.store(true, std::memory_order_release);
        running_.store(true, std::memory_order_release);
        return true;
    }

    bool prefill_source(const float* interleaved_stereo, std::size_t frames,
                        std::string& error) override {
        using clock = std::chrono::steady_clock;
        if (!ready_.load(std::memory_order_acquire)) {
            error = "MRT2 must be prepared before source prefill";
            return false;
        }
        if (!prefill_available_.load(std::memory_order_acquire)) {
            error = "MRT2 SpectroStream prefill encoder is not loaded";
            return false;
        }
        if (interleaved_stereo == nullptr || frames == 0 ||
            frames > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            error = "MRT2 source prefill requires a valid frame count";
            return false;
        }

        std::vector<std::string> logs;
        const auto started = clock::now();
        const bool success = runner_.prefill_state(
            interleaved_stereo, static_cast<int>(frames),
            [&logs](const std::string& message) { logs.push_back(message); });
        prefill_ms_ = std::chrono::duration<double, std::milli>(
                          clock::now() - started)
                          .count();
        if (!success) {
            error = logs.empty() ? "MRT2 source prefill failed" : logs.back();
            return false;
        }
        prefill_frames_.store(frames, std::memory_order_relaxed);
        source_prefilled_.store(true, std::memory_order_release);
        return true;
    }

    void start() override {
        if (!ready_.load(std::memory_order_acquire)) return;
        if (!running_.exchange(true, std::memory_order_acq_rel)) runner_.start();
    }

    void stop() override {
        if (running_.exchange(false, std::memory_order_acq_rel)) runner_.stop();
    }

    bool restart_from_prefill(std::string& error) override {
        if (!ready_.load(std::memory_order_acquire) ||
            !source_prefilled_.load(std::memory_order_acquire)) {
            error = "MRT2 cannot replay before source prefill is complete";
            return false;
        }
        runner_.start();
        running_.store(true, std::memory_order_release);
        return true;
    }

    void update_semantic_prompts(
        const std::array<std::string, kSemanticPromptSlots>& prompts) override {
        std::vector<std::string> prompt_vector(prompts.begin(), prompts.end());
        std::vector<float> weights(kSemanticPromptSlots, 0.0F);
        for (std::size_t index = 0; index < kSemanticPromptSlots; ++index) {
            weights[index] = conditioning_weights_[index].load(
                std::memory_order_relaxed);
        }
        runner_.set_text_prompts(prompt_vector, weights);
    }

    bool set_audio_prompt(std::size_t slot,
                          const float* mono_16khz,
                          std::size_t samples,
                          const std::string& filename,
                          std::string& error) override {
        if (!ready_.load(std::memory_order_acquire)) {
            error = "MRT2 must be prepared before loading an audio reference";
            return false;
        }
        if (mono_16khz == nullptr || samples == 0) {
            error = "Audio reference must contain mono 16 kHz samples";
            return false;
        }
        if (slot >= kSemanticPromptSlots) {
            error = "Audio prompt slot is out of range";
            return false;
        }
        runner_.set_audio_prompt_samples(
            static_cast<int>(slot), filename, mono_16khz, samples);
        return true;
    }

    void set_conditioning(const ConditioningState& state) noexcept override {
        float weights[kSemanticPromptSlots]{
            std::max(0.0F, state.style_a), std::max(0.0F, state.style_b),
            std::max(0.0F, state.style_c), std::max(0.0F, state.style_d),
            std::max(0.0F, state.style_e), std::max(0.0F, state.style_f)};
        float total = 0.0F;
        for (const auto weight : weights) total += weight;
        if (total > 0.0F) {
            for (auto& weight : weights) weight /= total;
        } else {
            weights[0] = 1.0F;
        }
        for (std::size_t index = 0; index < kSemanticPromptSlots; ++index) {
            conditioning_weights_[index].store(weights[index],
                                               std::memory_order_relaxed);
        }
        runner_.set_blend_weights(weights, static_cast<int>(kSemanticPromptSlots));
        sequence_.store(state.sequence, std::memory_order_relaxed);
    }

    bool pull_audio(float* left, float* right, std::size_t samples) noexcept override {
        return runner_.read_audio_stereo(left, right, samples, false);
    }

    EngineTelemetry telemetry() const override {
        const auto metrics = runner_.get_metrics();
        std::array<int, kSemanticPromptSlots> prompt_statuses{};
        for (std::size_t slot = 0; slot < kSemanticPromptSlots; ++slot) {
            prompt_statuses[slot] = runner_.get_prompt_status(
                static_cast<int>(slot));
        }
        return {
            .backend = "mrt2",
            .ready = ready_.load(std::memory_order_acquire),
            .transformer_ms = metrics.transformer_ms,
            .frame_total_ms = metrics.total_ms,
            .buffer_available_samples = metrics.buffer_available,
            .buffer_capacity_samples = metrics.buffer_capacity,
            .dropped_audio_reads = metrics.dropped_frames,
            .conditioning_sequence = sequence_.load(std::memory_order_relaxed),
            .prepare_ms = prepare_ms_,
            .source_prefilled = source_prefilled_.load(std::memory_order_acquire),
            .source_prefill_frames = prefill_frames_.load(std::memory_order_relaxed),
            .source_prefill_ms = prefill_ms_,
            .prompt_encoder_status = runner_.get_text_encoder_status(),
            .prompt_quantizer_status = runner_.get_quantizer_status(),
            .audio_prompt_statuses = prompt_statuses,
            .audio_reference_status = prompt_statuses.back(),
        };
    }

private:
    magentart::core::RealtimeRunner runner_;
    std::atomic<bool> ready_{false};
    std::atomic<bool> running_{false};
    std::atomic<std::uint64_t> sequence_{0};
    std::atomic<bool> prefill_available_{false};
    std::atomic<bool> source_prefilled_{false};
    std::atomic<std::size_t> prefill_frames_{0};
    std::array<std::atomic<float>, kSemanticPromptSlots> conditioning_weights_{
        1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    double prepare_ms_{0.0};
    double prefill_ms_{0.0};
};

}  // namespace

std::unique_ptr<GenerativeEngine> make_mrt2_generative_engine() {
    return std::make_unique<MRT2GenerativeEngine>();
}

}  // namespace song_world
