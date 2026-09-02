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

    void start() override { running_.store(true, std::memory_order_release); }
    void stop() override { running_.store(false, std::memory_order_release); }

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
        return {
            .backend = "mock",
            .ready = ready_.load(std::memory_order_acquire),
            .conditioning_sequence = sequence_.load(std::memory_order_relaxed),
        };
    }

private:
    std::atomic<bool> ready_{false};
    std::atomic<bool> running_{false};
    std::atomic<float> morph_{0.0F};
    std::atomic<std::uint64_t> sequence_{0};
    double phase_{0.0};
    double pulse_phase_{0.0};
};

}  // namespace

std::unique_ptr<GenerativeEngine> make_mock_generative_engine() {
    return std::make_unique<MockGenerativeEngine>();
}

}  // namespace song_world
