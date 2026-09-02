#include <song_world/GenerativeEngine.h>

#include <magentart/realtime_runner.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
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
            config.prompt_a, config.prompt_b, config.prompt_c};
        const std::vector<float> initial_weights{1.0F, 0.0F, 0.0F};
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

        prepare_ms_ = std::chrono::duration<double, std::milli>(clock::now() - started).count();
        ready_.store(true, std::memory_order_release);
        running_.store(true, std::memory_order_release);
        return true;
    }

    void start() override {
        if (!ready_.load(std::memory_order_acquire)) return;
        if (!running_.exchange(true, std::memory_order_acq_rel)) runner_.start();
    }

    void stop() override {
        if (running_.exchange(false, std::memory_order_acq_rel)) runner_.stop();
    }

    void set_conditioning(const ConditioningState& state) noexcept override {
        float weights[3]{std::max(0.0F, state.style_a),
                         std::max(0.0F, state.style_b),
                         std::max(0.0F, state.style_c)};
        const float total = weights[0] + weights[1] + weights[2];
        if (total > 0.0F) {
            for (auto& weight : weights) weight /= total;
        } else {
            weights[0] = 1.0F;
        }
        runner_.set_blend_weights(weights, 3);
        sequence_.store(state.sequence, std::memory_order_relaxed);
    }

    bool pull_audio(float* left, float* right, std::size_t samples) noexcept override {
        return runner_.read_audio_stereo(left, right, samples, false);
    }

    EngineTelemetry telemetry() const override {
        const auto metrics = runner_.get_metrics();
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
        };
    }

private:
    magentart::core::RealtimeRunner runner_;
    std::atomic<bool> ready_{false};
    std::atomic<bool> running_{false};
    std::atomic<std::uint64_t> sequence_{0};
    double prepare_ms_{0.0};
};

}  // namespace

std::unique_ptr<GenerativeEngine> make_mrt2_generative_engine() {
    return std::make_unique<MRT2GenerativeEngine>();
}

}  // namespace song_world
