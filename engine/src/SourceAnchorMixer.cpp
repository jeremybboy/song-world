#include <song_world/SourceAnchorMixer.h>

#include <algorithm>
#include <cmath>

namespace song_world {
namespace {

constexpr double kPi = 3.14159265358979323846;

float one_pole_coefficient(double cutoff_hz, double sample_rate) noexcept {
    return static_cast<float>(1.0 - std::exp(-2.0 * kPi * cutoff_hz / sample_rate));
}

}  // namespace

SourceAnchorMixer::SourceAnchorMixer(double sample_rate) noexcept {
    set_sample_rate(sample_rate);
}

void SourceAnchorMixer::set_sample_rate(double sample_rate) noexcept {
    sample_rate_ = std::max(1.0, sample_rate);
    update_coefficients();
    reset();
}

void SourceAnchorMixer::reset() noexcept {
    bass_left_ = 0.0F;
    bass_right_ = 0.0F;
    hook_lowpass_ = 0.0F;
    side_lowpass_ = 0.0F;
}

SourceAnchorBands SourceAnchorMixer::analyse(float left, float right) noexcept {
    bass_left_ += bass_coefficient_ * (left - bass_left_);
    bass_right_ += bass_coefficient_ * (right - bass_right_);

    const float mid = 0.5F * (left + right);
    hook_lowpass_ += hook_coefficient_ * (mid - hook_lowpass_);
    const float bass_mid = 0.5F * (bass_left_ + bass_right_);
    const float hook = hook_lowpass_ - bass_mid;

    const float side = 0.5F * (left - right);
    side_lowpass_ += air_coefficient_ * (side - side_lowpass_);
    const float air = side - side_lowpass_;

    return {
        .bass_left = bass_left_,
        .bass_right = bass_right_,
        .hook_mid = hook,
        .air_side = air,
    };
}

SourceAnchorFrame SourceAnchorMixer::render(
    const SourceAnchorBands& bands, const SourceAnchorGains& gains) noexcept {
    const float bass_gain = std::clamp(gains.bass, 0.0F, 1.0F);
    const float hook_gain = std::clamp(gains.hook, 0.0F, 1.0F);
    const float air_gain = std::clamp(gains.air, 0.0F, 1.0F);
    return {
        .left = bands.bass_left * bass_gain + bands.hook_mid * hook_gain +
                bands.air_side * air_gain,
        .right = bands.bass_right * bass_gain + bands.hook_mid * hook_gain -
                 bands.air_side * air_gain,
    };
}

SourceAnchorFrame SourceAnchorMixer::process(
    float left, float right, const SourceAnchorGains& gains) noexcept {
    return render(analyse(left, right), gains);
}

void SourceAnchorMixer::update_coefficients() noexcept {
    bass_coefficient_ = one_pole_coefficient(190.0, sample_rate_);
    hook_coefficient_ = one_pole_coefficient(5200.0, sample_rate_);
    air_coefficient_ = one_pole_coefficient(1800.0, sample_rate_);
}

}  // namespace song_world
