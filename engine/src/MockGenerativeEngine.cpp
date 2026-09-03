#include <song_world/GenerativeEngine.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <numbers>

namespace song_world {
namespace {

class MockGenerativeEngine final : public GenerativeEngine {
public:
    bool prepare(const EngineConfig&, std::string&) override {
        ready_.store(true, std::memory_order_release);
        return true;
    }

    bool prefill_source(const float* interleaved_stereo, std::size_t frames,
                        std::string& error) override {
        if (interleaved_stereo == nullptr || frames == 0) {
            error = "Mock source prefill requires non-empty stereo audio";
            return false;
        }
        source_prefill_frames_.store(frames, std::memory_order_relaxed);
        source_prefilled_.store(true, std::memory_order_release);
        return true;
    }

    void start() override { running_.store(true, std::memory_order_release); }
    void stop() override { running_.store(false, std::memory_order_release); }

    bool restart_from_prefill(std::string& error) override {
        if (!source_prefilled_.load(std::memory_order_acquire)) {
            error = "Mock cannot replay before source prefill";
            return false;
        }
        phase_ = 0.0;
        pulse_phase_ = 0.0;
        running_.store(true, std::memory_order_release);
        return true;
    }

    void update_semantic_prompts(
        const std::array<std::string, kSemanticPromptSlots>&) override {}

    bool set_audio_prompt(std::size_t slot,
                          const float* mono_16khz,
                          std::size_t samples,
                          const std::string&,
                          std::string& error) override {
        if (mono_16khz == nullptr || samples == 0) {
            error = "Mock audio reference requires samples";
            return false;
        }
        if (slot >= kSemanticPromptSlots) {
            error = "Mock audio prompt slot is out of range";
            return false;
        }
        audio_prompt_ready_[slot].store(true, std::memory_order_release);
        return true;
    }

    void set_conditioning(const ConditioningState& state) noexcept override {
        morph_.store(std::clamp(state.x, 0.0F, 1.0F), std::memory_order_relaxed);
        sequence_.store(state.sequence, std::memory_order_relaxed);
    }

    bool pull_audio(float* left, float* right, std::size_t samples) noexcept override {
        if (!running_.load(std::memory_order_acquire)) {
            std::fill_n(left, samples, 0.0F);
            std::fill_n(right, samples, 0.0F);
            return false;
        }

        const double morph = morph_.load(std::memory_order_relaxed);
        const double frequency = 110.0 + 110.0 * morph;
        const double increment = 2.0 * std::numbers::pi * frequency / 48000.0;
        const double pulse_increment = 2.0 * std::numbers::pi * 2.0 / 48000.0;

        for (std::size_t i = 0; i < samples; ++i) {
            const float tonal = static_cast<float>(std::sin(phase_));
            const float pulse = std::sin(pulse_phase_) > 0.82 ? 1.0F : -0.15F;
            const float sample = 0.035F * tonal + static_cast<float>(0.012 * morph) * pulse;
            left[i] = sample;
            right[i] = 0.92F * sample;

            phase_ += increment;
            pulse_phase_ += pulse_increment;
            if (phase_ >= 2.0 * std::numbers::pi) phase_ -= 2.0 * std::numbers::pi;
            if (pulse_phase_ >= 2.0 * std::numbers::pi) pulse_phase_ -= 2.0 * std::numbers::pi;
        }
        return true;
    }

    EngineTelemetry telemetry() const override {
        std::array<int, kSemanticPromptSlots> prompt_statuses{};
        for (std::size_t slot = 0; slot < kSemanticPromptSlots; ++slot) {
            prompt_statuses[slot] = audio_prompt_ready_[slot].load(
                std::memory_order_acquire) ? 2 : 0;
        }
        return {
            .backend = "mock",
            .ready = ready_.load(std::memory_order_acquire),
            .conditioning_sequence = sequence_.load(std::memory_order_relaxed),
            .source_prefilled = source_prefilled_.load(std::memory_order_acquire),
            .source_prefill_frames = source_prefill_frames_.load(std::memory_order_relaxed),
            .audio_prompt_statuses = prompt_statuses,
            .audio_reference_status = prompt_statuses.back(),
        };
    }

private:
    std::atomic<bool> ready_{false};
    std::atomic<bool> running_{false};
    std::atomic<float> morph_{0.0F};
    std::atomic<std::uint64_t> sequence_{0};
    std::atomic<bool> source_prefilled_{false};
    std::atomic<std::size_t> source_prefill_frames_{0};
    std::array<std::atomic<bool>, kSemanticPromptSlots> audio_prompt_ready_{};
    double phase_{0.0};
    double pulse_phase_{0.0};
};

}  // namespace

std::unique_ptr<GenerativeEngine> make_mock_generative_engine() {
    return std::make_unique<MockGenerativeEngine>();
}

}  // namespace song_world
