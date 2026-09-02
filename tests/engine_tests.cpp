#include <song_world/GenerativeEngine.h>
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
    mock->stop();

    std::cout << "PASS: deterministic transport and mock engine\n";
    return EXIT_SUCCESS;
}
