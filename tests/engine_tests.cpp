#include <song_world/GenerativeEngine.h>
#include <song_world/SourceAnchorMixer.h>
#include <song_world/WorldTransport.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

bool nearly_equal(double a, double b, double tolerance = 1.0e-9) {
    return std::abs(a - b) <= tolerance;
}

int fail(const char* message) {
    std::cerr << "FAIL: " << message << '\n';
    return EXIT_FAILURE;
}

}  // namespace

int main() {
    song_world::SourceAnchorMixer anchor_mixer(48000.0);
    float bass_energy = 0.0F;
    float hook_energy = 0.0F;
    float air_energy = 0.0F;
    for (int sample = 0; sample < 48000; ++sample) {
        const float time = static_cast<float>(sample) / 48000.0F;
        const float left = 0.5F * std::sin(2.0F * 3.14159265F * 90.0F * time) +
                           0.2F * std::sin(2.0F * 3.14159265F * 900.0F * time) +
                           0.1F * std::sin(2.0F * 3.14159265F * 7000.0F * time);
        const float right = 0.5F * std::sin(2.0F * 3.14159265F * 90.0F * time) +
                            0.2F * std::sin(2.0F * 3.14159265F * 900.0F * time) -
                            0.1F * std::sin(2.0F * 3.14159265F * 7000.0F * time);
        const auto bass = anchor_mixer.process(left, right, {1.0F, 0.0F, 0.0F});
        bass_energy += std::abs(bass.left) + std::abs(bass.right);
    }
    anchor_mixer.reset();
    for (int sample = 0; sample < 48000; ++sample) {
        const float time = static_cast<float>(sample) / 48000.0F;
        const float mid = 0.2F * std::sin(2.0F * 3.14159265F * 900.0F * time);
        const auto hook = anchor_mixer.process(mid, mid, {0.0F, 1.0F, 0.0F});
        hook_energy += std::abs(hook.left) + std::abs(hook.right);
    }
    anchor_mixer.reset();
    for (int sample = 0; sample < 48000; ++sample) {
        const float time = static_cast<float>(sample) / 48000.0F;
        const float side = 0.1F * std::sin(2.0F * 3.14159265F * 7000.0F * time);
        const auto air = anchor_mixer.process(side, -side, {0.0F, 0.0F, 1.0F});
        air_energy += std::abs(air.left) + std::abs(air.right);
    }
    if (bass_energy < 100.0F || hook_energy < 100.0F || air_energy < 100.0F) {
        return fail("source anchor bands must produce finite nonzero output");
    }

    song_world::WorldTransport transport(48000.0, 120.0);
    transport.set_tempo_ratio(1.25);
    transport.advance(48000);
    const auto snapshot = transport.snapshot();
    if (!nearly_equal(snapshot.performance_seconds, 1.0)) {
        return fail("performance clock must follow rendered samples");
    }
    if (!nearly_equal(snapshot.source_beats, 2.5)) {
        return fail("source beats must include tempo ratio");
    }
    if (!nearly_equal(snapshot.source_bars, 0.625)) {
        return fail("bar position must derive from source beats");
    }

    auto mock = song_world::make_mock_generative_engine();
    song_world::EngineConfig config;
    std::string error;
    if (!mock->prepare(config, error)) return fail("mock prepare failed");
    std::vector<float> source_audio(48000 * 2, 0.1F);
    if (!mock->prefill_source(source_audio.data(), 48000, error)) {
        return fail("mock source prefill failed");
    }
    if (!mock->telemetry().source_prefilled ||
        mock->telemetry().source_prefill_frames != 48000) {
        return fail("source prefill telemetry must reach the engine boundary");
    }
    mock->start();
    mock->set_conditioning({
        .x = 0.75F,
        .y = 0.25F,
        .style_a = 0.25F,
        .style_b = 0.75F,
        .style_c = 0.0F,
        .sequence = 7,
    });

    std::vector<float> left(512);
    std::vector<float> right(512);
    if (!mock->pull_audio(left.data(), right.data(), left.size())) {
        return fail("mock audio pull failed");
    }
    bool nonzero = false;
    for (std::size_t i = 0; i < left.size(); ++i) {
        if (!std::isfinite(left[i]) || !std::isfinite(right[i])) {
            return fail("mock output must remain finite");
        }
        nonzero = nonzero || std::abs(left[i]) > 1.0e-6F;
    }
    if (!nonzero) return fail("mock output must be audible/nonzero");
    if (mock->telemetry().conditioning_sequence != 7) {
        return fail("conditioning sequence must reach the engine");
    }
    const float first_sample = left.front();
    if (!mock->restart_from_prefill(error)) {
        return fail("mock replay from prefill failed");
    }
    if (!mock->pull_audio(left.data(), right.data(), left.size()) ||
        !nearly_equal(left.front(), first_sample, 1.0e-7)) {
        return fail("replay must restore deterministic post-prefill generation");
    }
    mock->stop();

    std::cout << "PASS: source anchors, deterministic transport, and mock engine\n";
    return EXIT_SUCCESS;
}
