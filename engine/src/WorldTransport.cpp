#include <song_world/WorldTransport.h>

#include <algorithm>

namespace song_world {

WorldTransport::WorldTransport(double sample_rate, double source_bpm)
    : sample_rate_(sample_rate), source_bpm_(source_bpm) {}

void WorldTransport::set_tempo_ratio(double ratio) noexcept {
    tempo_ratio_.store(std::clamp(ratio, 0.25, 4.0), std::memory_order_relaxed);
}

void WorldTransport::reset() noexcept {
    rendered_samples_.store(0, std::memory_order_relaxed);
    source_beats_.store(0.0, std::memory_order_relaxed);
}

void WorldTransport::advance(std::size_t samples) noexcept {
    const double seconds = static_cast<double>(samples) / sample_rate_;
    const double beats = seconds * (source_bpm_ / 60.0) *
                         tempo_ratio_.load(std::memory_order_relaxed);
    rendered_samples_.fetch_add(samples, std::memory_order_relaxed);
    source_beats_.fetch_add(beats, std::memory_order_relaxed);
}

TransportSnapshot WorldTransport::snapshot() const noexcept {
    const auto samples = rendered_samples_.load(std::memory_order_relaxed);
    const double beats = source_beats_.load(std::memory_order_relaxed);
    return {
        .performance_seconds = static_cast<double>(samples) / sample_rate_,
        .source_beats = beats,
        .source_bars = beats / 4.0,
        .source_bpm = source_bpm_,
        .tempo_ratio = tempo_ratio_.load(std::memory_order_relaxed),
        .rendered_samples = samples,
    };
}

}  // namespace song_world
