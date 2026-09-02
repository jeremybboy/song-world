#include <song_world/GenerativeEngine.h>
#include <song_world/WorldTransport.h>

#include <juce_audio_devices/juce_audio_devices.h>
#include <juce_events/juce_events.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include <sys/resource.h>

namespace {

using clock_type = std::chrono::steady_clock;

struct Options {
    std::string engine{"mock"};
    std::string model_path;
    std::string resources;
    std::string device_name;
    std::string report_path;
    std::string capture_path;
    std::string control_mode{"sweep"};
    std::string prompt_a{"minimal dub techno, dry drums, instrumental"};
    std::string prompt_b{"warm disco funk, acoustic drums, instrumental"};
    double duration_seconds{30.0};
    double control_period_seconds{8.0};
    double control_step_seconds{4.0};
    double constant_x{0.0};
    double tempo_ratio{1.25};
    int device_buffer_samples{512};
    int ring_buffer_samples{4096};
};

struct RunMeasurements {
    std::vector<double> frame_ms;
    std::vector<double> transformer_ms;
    std::vector<double> buffer_ms;
    std::vector<double> control_pipeline_estimate_ms;
    std::uint64_t conditioning_updates{0};
    double min_condition_x{1.0};
    double max_condition_x{0.0};
    double first_control_change_seconds{-1.0};
};

double timeval_seconds(const timeval& value) {
    return static_cast<double>(value.tv_sec) + static_cast<double>(value.tv_usec) / 1.0e6;
}

double process_cpu_seconds() {
    rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
    return timeval_seconds(usage.ru_utime) + timeval_seconds(usage.ru_stime);
}

double process_peak_rss_mb() {
    rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
    return static_cast<double>(usage.ru_maxrss) / (1024.0 * 1024.0);
}

double percentile(std::vector<double> values, double p) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const double index = p * static_cast<double>(values.size() - 1);
    const auto lower = static_cast<std::size_t>(std::floor(index));
    const auto upper = static_cast<std::size_t>(std::ceil(index));
    if (lower == upper) return values[lower];
    const double fraction = index - static_cast<double>(lower);
    return values[lower] + (values[upper] - values[lower]) * fraction;
}

double average(const std::vector<double>& values) {
    if (values.empty()) return 0.0;
    return std::accumulate(values.begin(), values.end(), 0.0) /
           static_cast<double>(values.size());
}

std::string json_escape(const std::string& value) {
    std::string result;
    result.reserve(value.size());
    for (char character : value) {
        switch (character) {
            case '\\': result += "\\\\"; break;
            case '"': result += "\\\""; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            default: result += character; break;
        }
    }
    return result;
}

void print_usage(const char* executable) {
    std::cout
        << "Usage: " << executable << " [options]\n"
        << "  --engine mock|mrt2\n"
        << "  --model PATH_TO_MLXFN       required for mrt2\n"
        << "  --resources DIRECTORY       required for mrt2\n"
        << "  --duration SECONDS          default 30\n"
        << "  --device NAME               optional JUCE output device\n"
        << "  --device-buffer SAMPLES     default 512\n"
        << "  --ring-buffer SAMPLES       default 4096\n"
        << "  --control-mode MODE         sweep|constant|step, default sweep\n"
        << "  --control-period SECONDS    A->B->A sweep period, default 8\n"
        << "  --control-step SECONDS      A->B step time, default 4\n"
        << "  --constant-x VALUE          constant blend position, default 0\n"
        << "  --tempo-ratio RATIO         timing primitive, default 1.25\n"
        << "  --prompt-a TEXT\n"
        << "  --prompt-b TEXT\n"
        << "  --capture PATH              write 16-bit stereo WAV evidence\n"
        << "  --report PATH               write JSON evidence\n";
}

bool parse_options(int argc, char** argv, Options& options, std::string& error) {
    auto require_value = [&](int& index, const std::string& flag) -> const char* {
        if (index + 1 >= argc) {
            error = flag + " requires a value";
            return nullptr;
        }
        return argv[++index];
    };

    for (int index = 1; index < argc; ++index) {
        const std::string flag = argv[index];
        const char* value = nullptr;
        if (flag == "--help" || flag == "-h") {
            print_usage(argv[0]);
            std::exit(EXIT_SUCCESS);
        } else if (flag == "--engine") {
            if (!(value = require_value(index, flag))) return false;
            options.engine = value;
        } else if (flag == "--model") {
            if (!(value = require_value(index, flag))) return false;
            options.model_path = value;
        } else if (flag == "--resources") {
            if (!(value = require_value(index, flag))) return false;
            options.resources = value;
        } else if (flag == "--duration") {
            if (!(value = require_value(index, flag))) return false;
            options.duration_seconds = std::stod(value);
        } else if (flag == "--device") {
            if (!(value = require_value(index, flag))) return false;
            options.device_name = value;
        } else if (flag == "--device-buffer") {
            if (!(value = require_value(index, flag))) return false;
            options.device_buffer_samples = std::stoi(value);
        } else if (flag == "--ring-buffer") {
            if (!(value = require_value(index, flag))) return false;
            options.ring_buffer_samples = std::stoi(value);
        } else if (flag == "--control-mode") {
            if (!(value = require_value(index, flag))) return false;
            options.control_mode = value;
        } else if (flag == "--control-period") {
            if (!(value = require_value(index, flag))) return false;
            options.control_period_seconds = std::stod(value);
        } else if (flag == "--control-step") {
            if (!(value = require_value(index, flag))) return false;
            options.control_step_seconds = std::stod(value);
        } else if (flag == "--constant-x") {
            if (!(value = require_value(index, flag))) return false;
            options.constant_x = std::stod(value);
        } else if (flag == "--tempo-ratio") {
            if (!(value = require_value(index, flag))) return false;
            options.tempo_ratio = std::stod(value);
        } else if (flag == "--prompt-a") {
            if (!(value = require_value(index, flag))) return false;
            options.prompt_a = value;
        } else if (flag == "--prompt-b") {
            if (!(value = require_value(index, flag))) return false;
            options.prompt_b = value;
        } else if (flag == "--capture") {
            if (!(value = require_value(index, flag))) return false;
            options.capture_path = value;
        } else if (flag == "--report") {
            if (!(value = require_value(index, flag))) return false;
            options.report_path = value;
        } else {
            error = "unknown option: " + flag;
            return false;
        }
    }

    if (options.engine != "mock" && options.engine != "mrt2") {
        error = "--engine must be mock or mrt2";
        return false;
    }
    if (options.control_mode != "sweep" && options.control_mode != "constant" &&
        options.control_mode != "step") {
        error = "--control-mode must be sweep, constant, or step";
        return false;
    }
    if (options.constant_x < 0.0 || options.constant_x > 1.0) {
        error = "--constant-x must be in [0, 1]";
        return false;
    }
    if (options.control_mode == "step" &&
        (options.control_step_seconds <= 0.0 ||
         options.control_step_seconds >= options.duration_seconds)) {
        error = "--control-step must occur within the run duration";
        return false;
    }
    if (options.engine == "mrt2" &&
        (options.model_path.empty() || options.resources.empty())) {
        error = "--model and --resources are required for mrt2";
        return false;
    }
    if (options.duration_seconds <= 0.0 || options.control_period_seconds <= 0.0) {
        error = "duration and control period must be positive";
        return false;
    }
    if (options.ring_buffer_samples <= 0 || options.ring_buffer_samples > 8192) {
        error = "ring buffer must be between 1 and MRT2's 8192-sample capacity";
        return false;
    }
    if (options.engine == "mrt2" && options.ring_buffer_samples < 3840) {
        error = "MRT2 ring buffer must hold at least two 1920-sample model frames (3840 samples)";
        return false;
    }
    return true;
}

class AudioHarness final : public juce::AudioIODeviceCallback {
public:
    AudioHarness(song_world::GenerativeEngine& engine,
                 song_world::WorldTransport& transport,
                 std::size_t capture_capacity_samples)
        : engine_(engine), transport_(transport),
          capture_left_(capture_capacity_samples),
          capture_right_(capture_capacity_samples) {}

    void audioDeviceIOCallbackWithContext(
        const float* const*, int,
        float* const* output_channels, int output_channel_count,
        int sample_count, const juce::AudioIODeviceCallbackContext&) override {
        const auto started = clock_type::now();

        if (output_channel_count < 2 || output_channels[0] == nullptr ||
            output_channels[1] == nullptr) {
            callback_failures_.fetch_add(1, std::memory_order_relaxed);
            return;
        }

        const bool complete = engine_.pull_audio(output_channels[0], output_channels[1],
                                                 static_cast<std::size_t>(sample_count));
        if (!complete) callback_failures_.fetch_add(1, std::memory_order_relaxed);

        const auto capture_offset = capture_samples_.load(std::memory_order_relaxed);
        if (capture_offset < capture_left_.size()) {
            const auto copy_count = std::min<std::size_t>(
                static_cast<std::size_t>(sample_count),
                capture_left_.size() - capture_offset);
            std::copy_n(output_channels[0], copy_count,
                        capture_left_.data() + capture_offset);
            std::copy_n(output_channels[1], copy_count,
                        capture_right_.data() + capture_offset);
            capture_samples_.store(capture_offset + copy_count,
                                   std::memory_order_relaxed);
        }
        for (int channel = 2; channel < output_channel_count; ++channel) {
            if (output_channels[channel]) {
                std::fill_n(output_channels[channel], sample_count, 0.0F);
            }
        }

        transport_.advance(static_cast<std::size_t>(sample_count));
        callback_count_.fetch_add(1, std::memory_order_relaxed);
        rendered_samples_.fetch_add(static_cast<std::uint64_t>(sample_count),
                                    std::memory_order_relaxed);

        const double micros = std::chrono::duration<double, std::micro>(
                                  clock_type::now() - started)
                                  .count();
        callback_micros_sum_.fetch_add(micros, std::memory_order_relaxed);
        double previous = max_callback_micros_.load(std::memory_order_relaxed);
        while (micros > previous &&
               !max_callback_micros_.compare_exchange_weak(
                   previous, micros, std::memory_order_relaxed)) {}
    }

    void audioDeviceAboutToStart(juce::AudioIODevice* device) override {
        sample_rate_.store(device->getCurrentSampleRate(), std::memory_order_relaxed);
        device_buffer_samples_.store(device->getCurrentBufferSizeSamples(),
                                     std::memory_order_relaxed);
    }

    void audioDeviceStopped() override {}

    void audioDeviceError(const juce::String& message) override {
        std::lock_guard lock(error_mutex_);
        device_error_ = message.toStdString();
    }

    [[nodiscard]] std::uint64_t callback_count() const {
        return callback_count_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] std::uint64_t callback_failures() const {
        return callback_failures_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] std::uint64_t rendered_samples() const {
        return rendered_samples_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] double sample_rate() const {
        return sample_rate_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] int device_buffer_samples() const {
        return device_buffer_samples_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] double average_callback_micros() const {
        const auto count = callback_count();
        return count == 0 ? 0.0 :
            callback_micros_sum_.load(std::memory_order_relaxed) /
                static_cast<double>(count);
    }
    [[nodiscard]] double max_callback_micros() const {
        return max_callback_micros_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] std::string device_error() const {
        std::lock_guard lock(error_mutex_);
        return device_error_;
    }
    [[nodiscard]] std::size_t capture_samples() const {
        return capture_samples_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] const std::vector<float>& capture_left() const { return capture_left_; }
    [[nodiscard]] const std::vector<float>& capture_right() const { return capture_right_; }

private:
    song_world::GenerativeEngine& engine_;
    song_world::WorldTransport& transport_;
    std::atomic<std::uint64_t> callback_count_{0};
    std::atomic<std::uint64_t> callback_failures_{0};
    std::atomic<std::uint64_t> rendered_samples_{0};
    std::atomic<double> callback_micros_sum_{0.0};
    std::atomic<double> max_callback_micros_{0.0};
    std::atomic<double> sample_rate_{0.0};
    std::atomic<int> device_buffer_samples_{0};
    std::vector<float> capture_left_;
    std::vector<float> capture_right_;
    std::atomic<std::size_t> capture_samples_{0};
    mutable std::mutex error_mutex_;
    std::string device_error_;
};

void write_little_endian_u16(std::ofstream& output, std::uint16_t value) {
    const char bytes[2]{static_cast<char>(value & 0xff),
                        static_cast<char>((value >> 8) & 0xff)};
    output.write(bytes, sizeof(bytes));
}

void write_little_endian_u32(std::ofstream& output, std::uint32_t value) {
    const char bytes[4]{static_cast<char>(value & 0xff),
                        static_cast<char>((value >> 8) & 0xff),
                        static_cast<char>((value >> 16) & 0xff),
                        static_cast<char>((value >> 24) & 0xff)};
    output.write(bytes, sizeof(bytes));
}

bool write_capture_wav(const std::filesystem::path& path,
                       const AudioHarness& audio,
                       std::string& error) {
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        error = "could not create capture: " + path.string();
        return false;
    }

    const auto sample_count = audio.capture_samples();
    const auto data_bytes = static_cast<std::uint32_t>(sample_count * 2 * sizeof(std::int16_t));
    output.write("RIFF", 4);
    write_little_endian_u32(output, 36 + data_bytes);
    output.write("WAVEfmt ", 8);
    write_little_endian_u32(output, 16);
    write_little_endian_u16(output, 1);
    write_little_endian_u16(output, 2);
    write_little_endian_u32(output, static_cast<std::uint32_t>(audio.sample_rate()));
    write_little_endian_u32(output,
                            static_cast<std::uint32_t>(audio.sample_rate()) * 4);
    write_little_endian_u16(output, 4);
    write_little_endian_u16(output, 16);
    output.write("data", 4);
    write_little_endian_u32(output, data_bytes);

    for (std::size_t index = 0; index < sample_count; ++index) {
        const auto encode = [](float sample) {
            return static_cast<std::int16_t>(
                std::lrint(std::clamp(sample, -1.0F, 1.0F) * 32767.0F));
        };
        write_little_endian_u16(
            output, static_cast<std::uint16_t>(encode(audio.capture_left()[index])));
        write_little_endian_u16(
            output, static_cast<std::uint16_t>(encode(audio.capture_right()[index])));
    }
    if (!output) {
        error = "capture write failed: " + path.string();
        return false;
    }
    return true;
}

std::string build_report(const Options& options,
                         const std::string& model_name,
                         const std::string& device_name,
                         const AudioHarness& audio,
                         const RunMeasurements& measurements,
                         const song_world::EngineTelemetry& engine,
                         const song_world::TransportSnapshot& transport,
                         double wall_seconds,
                         double cpu_percent,
                         double peak_rss_mb) {
    const double frame_average = average(measurements.frame_ms);
    const double frame_p95 = percentile(measurements.frame_ms, 0.95);
    const double callback_budget_us = audio.sample_rate() > 0.0
        ? static_cast<double>(audio.device_buffer_samples()) / audio.sample_rate() * 1.0e6
        : 0.0;
    const double callback_load_percent = callback_budget_us > 0.0
        ? audio.average_callback_micros() / callback_budget_us * 100.0
        : 0.0;

    std::ostringstream output;
    output << std::fixed << std::setprecision(3);
    output << "{\n"
           << "  \"schema_version\": 1,\n"
           << "  \"engine\": \"" << json_escape(options.engine) << "\",\n"
           << "  \"model\": \"" << json_escape(model_name) << "\",\n"
           << "  \"device\": \"" << json_escape(device_name) << "\",\n"
           << "  \"sample_rate_hz\": " << audio.sample_rate() << ",\n"
           << "  \"device_buffer_samples\": " << audio.device_buffer_samples() << ",\n"
           << "  \"ring_buffer_samples\": " << options.ring_buffer_samples << ",\n"
           << "  \"control_mode\": \"" << json_escape(options.control_mode) << "\",\n"
           << "  \"requested_control_step_seconds\": "
           << options.control_step_seconds << ",\n"
           << "  \"first_control_change_seconds\": "
           << measurements.first_control_change_seconds << ",\n"
           << "  \"wall_duration_seconds\": " << wall_seconds << ",\n"
           << "  \"rendered_audio_seconds\": "
           << (audio.sample_rate() > 0.0 ? audio.rendered_samples() / audio.sample_rate() : 0.0)
           << ",\n"
           << "  \"audio_callbacks\": " << audio.callback_count() << ",\n"
           << "  \"callback_failures\": " << audio.callback_failures() << ",\n"
           << "  \"engine_dropped_audio_reads\": " << engine.dropped_audio_reads << ",\n"
           << "  \"callback_average_us\": " << audio.average_callback_micros() << ",\n"
           << "  \"callback_max_us\": " << audio.max_callback_micros() << ",\n"
           << "  \"callback_average_budget_percent\": " << callback_load_percent << ",\n"
           << "  \"inference_frame_average_ms\": " << frame_average << ",\n"
           << "  \"inference_frame_p50_ms\": " << percentile(measurements.frame_ms, 0.50) << ",\n"
           << "  \"inference_frame_p95_ms\": " << frame_p95 << ",\n"
           << "  \"inference_frame_p99_ms\": " << percentile(measurements.frame_ms, 0.99) << ",\n"
           << "  \"generation_realtime_factor_from_average\": "
           << (frame_average > 0.0 ? 40.0 / frame_average : 0.0) << ",\n"
           << "  \"buffer_fill_p50_ms\": " << percentile(measurements.buffer_ms, 0.50) << ",\n"
           << "  \"buffer_fill_p95_ms\": " << percentile(measurements.buffer_ms, 0.95) << ",\n"
           << "  \"control_pipeline_estimate_p50_ms\": "
           << percentile(measurements.control_pipeline_estimate_ms, 0.50) << ",\n"
           << "  \"control_pipeline_estimate_p95_ms\": "
           << percentile(measurements.control_pipeline_estimate_ms, 0.95) << ",\n"
           << "  \"conditioning_updates_sent\": " << measurements.conditioning_updates << ",\n"
           << "  \"conditioning_x_min\": " << measurements.min_condition_x << ",\n"
           << "  \"conditioning_x_max\": " << measurements.max_condition_x << ",\n"
           << "  \"last_conditioning_sequence_observed\": " << engine.conditioning_sequence << ",\n"
           << "  \"engine_prepare_ms\": " << engine.prepare_ms << ",\n"
           << "  \"process_cpu_percent_of_one_core\": " << cpu_percent << ",\n"
           << "  \"process_peak_rss_mb\": " << peak_rss_mb << ",\n"
           << "  \"performance_seconds\": " << transport.performance_seconds << ",\n"
           << "  \"source_beats\": " << transport.source_beats << ",\n"
           << "  \"source_bars\": " << transport.source_bars << ",\n"
           << "  \"tempo_ratio\": " << transport.tempo_ratio << ",\n"
           << "  \"captured_samples\": " << audio.capture_samples() << ",\n"
           << "  \"capture_path\": \"" << json_escape(options.capture_path) << "\",\n"
           << "  \"device_error\": \"" << json_escape(audio.device_error()) << "\"\n"
           << "}\n";
    return output.str();
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    std::string error;
    try {
        if (!parse_options(argc, argv, options, error)) {
            std::cerr << "Error: " << error << "\n\n";
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
    } catch (const std::exception& exception) {
        std::cerr << "Error parsing options: " << exception.what() << '\n';
        return EXIT_FAILURE;
    }

    juce::ScopedJuceInitialiser_GUI juce_runtime;

    std::unique_ptr<song_world::GenerativeEngine> engine;
    if (options.engine == "mock") {
        engine = song_world::make_mock_generative_engine();
    } else {
#if SONG_WORLD_ENABLE_MRT2
        engine = song_world::make_mrt2_generative_engine();
#else
        std::cerr << "This build does not include MRT2\n";
        return EXIT_FAILURE;
#endif
    }

    song_world::EngineConfig engine_config{
        .model_path = options.model_path,
        .resource_directory = options.resources,
        .prompt_a = options.prompt_a,
        .prompt_b = options.prompt_b,
        .ring_buffer_samples = static_cast<std::size_t>(options.ring_buffer_samples),
        .output_gain_db = -12.0F,
    };

    std::cout << "Preparing " << options.engine << " engine...\n";
    if (!engine->prepare(engine_config, error)) {
        std::cerr << "Engine preparation failed: " << error << '\n';
        return EXIT_FAILURE;
    }
    engine->start();

    song_world::WorldTransport transport(48000.0, 120.0);
    transport.set_tempo_ratio(options.tempo_ratio);
    const auto capture_capacity = options.capture_path.empty()
        ? std::size_t{0}
        : static_cast<std::size_t>(std::ceil(options.duration_seconds * 48000.0)) + 8192;
    AudioHarness audio_harness(*engine, transport, capture_capacity);
    juce::AudioDeviceManager device_manager;

    auto initialise_error = device_manager.initialise(0, 2, nullptr, true);
    if (initialise_error.isNotEmpty()) {
        std::cerr << "Audio initialization failed: " << initialise_error << '\n';
        engine->stop();
        return EXIT_FAILURE;
    }

    auto setup = device_manager.getAudioDeviceSetup();
    setup.sampleRate = 48000.0;
    setup.bufferSize = options.device_buffer_samples;
    if (!options.device_name.empty()) setup.outputDeviceName = options.device_name;
    const auto setup_error = device_manager.setAudioDeviceSetup(setup, true);
    if (setup_error.isNotEmpty()) {
        std::cerr << "Audio device setup failed: " << setup_error << '\n';
        engine->stop();
        return EXIT_FAILURE;
    }

    auto* device = device_manager.getCurrentAudioDevice();
    if (device == nullptr) {
        std::cerr << "No JUCE output device is active\n";
        engine->stop();
        return EXIT_FAILURE;
    }
    if (std::abs(device->getCurrentSampleRate() - 48000.0) > 0.5) {
        std::cerr << "Phase 0 requires 48 kHz; device opened at "
                  << device->getCurrentSampleRate() << " Hz\n";
        engine->stop();
        return EXIT_FAILURE;
    }

    const std::string device_name = device->getName().toStdString();
    std::cout << "Audio device: " << device_name << ", "
              << device->getCurrentSampleRate() << " Hz, "
              << device->getCurrentBufferSizeSamples() << " samples\n";

    const double cpu_started = process_cpu_seconds();
    const auto run_started = clock_type::now();
    device_manager.addAudioCallback(&audio_harness);

    RunMeasurements measurements;
    std::uint64_t conditioning_sequence = 0;
    while (true) {
        const double elapsed = std::chrono::duration<double>(clock_type::now() - run_started).count();
        if (elapsed >= options.duration_seconds) break;

        double control_x = options.constant_x;
        if (options.control_mode == "sweep") {
            const double normalized = std::fmod(elapsed, options.control_period_seconds) /
                                      options.control_period_seconds;
            control_x = normalized < 0.5
                ? normalized * 2.0
                : (1.0 - normalized) * 2.0;
        } else if (options.control_mode == "step") {
            control_x = elapsed < options.control_step_seconds ? 0.0 : 1.0;
            if (control_x > 0.0 && measurements.first_control_change_seconds < 0.0) {
                measurements.first_control_change_seconds = elapsed;
            }
        }
        const float x = static_cast<float>(control_x);
        engine->set_conditioning({.x = x, .y = 0.5F, .sequence = ++conditioning_sequence});
        measurements.conditioning_updates += 1;
        measurements.min_condition_x = std::min(measurements.min_condition_x, control_x);
        measurements.max_condition_x = std::max(measurements.max_condition_x, control_x);

        const auto telemetry = engine->telemetry();
        if (telemetry.frame_total_ms > 0.0F) {
            measurements.frame_ms.push_back(telemetry.frame_total_ms);
            measurements.transformer_ms.push_back(telemetry.transformer_ms);
            const double buffered_ms = static_cast<double>(telemetry.buffer_available_samples) /
                                       48000.0 * 1000.0;
            measurements.buffer_ms.push_back(buffered_ms);
            const double device_ms = static_cast<double>(device->getCurrentBufferSizeSamples()) /
                                     48000.0 * 1000.0;
            measurements.control_pipeline_estimate_ms.push_back(
                buffered_ms + device_ms + telemetry.frame_total_ms);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    device_manager.removeAudioCallback(&audio_harness);
    const double wall_seconds = std::chrono::duration<double>(clock_type::now() - run_started).count();
    const double cpu_seconds = process_cpu_seconds() - cpu_started;
    engine->stop();

    const auto final_telemetry = engine->telemetry();
    const auto final_transport = transport.snapshot();
    const std::string model_name = options.model_path.empty()
        ? "none"
        : std::filesystem::path(options.model_path).filename().string();
    const auto report = build_report(
        options, model_name, device_name, audio_harness, measurements,
        final_telemetry, final_transport, wall_seconds,
        wall_seconds > 0.0 ? cpu_seconds / wall_seconds * 100.0 : 0.0,
        process_peak_rss_mb());

    std::cout << report;
    if (!options.report_path.empty()) {
        const std::filesystem::path report_path(options.report_path);
        if (report_path.has_parent_path()) {
            std::filesystem::create_directories(report_path.parent_path());
        }
        std::ofstream output(report_path);
        if (!output) {
            std::cerr << "Could not write report: " << options.report_path << '\n';
            return EXIT_FAILURE;
        }
        output << report;
    }
    if (!options.capture_path.empty() &&
        !write_capture_wav(options.capture_path, audio_harness, error)) {
        std::cerr << error << '\n';
        return EXIT_FAILURE;
    }

    const bool passed = audio_harness.callback_count() > 0 &&
                        audio_harness.callback_failures() == 0 &&
                        audio_harness.device_error().empty();
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
