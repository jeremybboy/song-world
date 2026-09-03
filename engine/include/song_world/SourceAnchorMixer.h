#pragma once

namespace song_world {

struct SourceAnchorGains {
    float bass{0.0F};
    float hook{0.0F};
    float air{0.0F};
};

struct SourceAnchorFrame {
    float left{0.0F};
    float right{0.0F};
};

struct SourceAnchorBands {
    float bass_left{0.0F};
    float bass_right{0.0F};
    float hook_mid{0.0F};
    float air_side{0.0F};
};

// Lightweight real-time proxy for the future prepared-stem mixer. It derives
// three deliberately broad source bands without allocating or blocking:
// bass/pulse, centre mid/hook, and stereo high-frequency air.
class SourceAnchorMixer {
public:
    explicit SourceAnchorMixer(double sample_rate = 48000.0) noexcept;

    void set_sample_rate(double sample_rate) noexcept;
    void reset() noexcept;
    [[nodiscard]] SourceAnchorBands analyse(float left, float right) noexcept;
    [[nodiscard]] static SourceAnchorFrame render(
        const SourceAnchorBands& bands, const SourceAnchorGains& gains) noexcept;
    [[nodiscard]] SourceAnchorFrame process(float left, float right,
                                            const SourceAnchorGains& gains) noexcept;

private:
    void update_coefficients() noexcept;

    double sample_rate_{48000.0};
    float bass_coefficient_{0.0F};
    float hook_coefficient_{0.0F};
    float air_coefficient_{0.0F};
    float bass_left_{0.0F};
    float bass_right_{0.0F};
    float hook_lowpass_{0.0F};
    float side_lowpass_{0.0F};
};

}  // namespace song_world
