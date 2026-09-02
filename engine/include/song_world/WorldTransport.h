#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace song_world {

struct TransportSnapshot {
    double performance_seconds{0.0};
    double source_beats{0.0};
    double source_bars{0.0};
    double source_bpm{120.0};
    double tempo_ratio{1.0};
    std::uint64_t rendered_samples{0};
};

class WorldTransport {
public:
    explicit WorldTransport(double sample_rate = 48000.0, double source_bpm = 120.0);

    void set_tempo_ratio(double ratio) noexcept;
    void reset() noexcept;

    // Audio-thread writer; readers use snapshot() from non-audio threads.
    void advance(std::size_t samples) noexcept;
    [[nodiscard]] TransportSnapshot snapshot() const noexcept;

private:
    double sample_rate_;
    double source_bpm_;
    std::atomic<double> tempo_ratio_{1.0};
    std::atomic<std::uint64_t> rendered_samples_{0};
    std::atomic<double> source_beats_{0.0};
};

}  // namespace song_world
