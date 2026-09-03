#include <song_world/GenerativeEngine.h>

#include <juce_audio_utils/juce_audio_utils.h>
#include <juce_gui_extra/juce_gui_extra.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr double kSampleRate = 48000.0;
constexpr int kDeviceBufferSamples = 512;
constexpr double kGroovejetBpm = 123.0;
constexpr double kGroovejetDownbeatSeconds = 0.397;
constexpr int kBeatsPerBar = 4;
constexpr double kBeatSeconds = 60.0 / kGroovejetBpm;
constexpr double kBarSeconds = kBeatSeconds * kBeatsPerBar;
constexpr int kWindowStartBar = 32;
constexpr int kWindowBars = 14;
constexpr double kWindowStartSeconds =
    kGroovejetDownbeatSeconds + kWindowStartBar * kBarSeconds;
constexpr double kWindowEndSeconds =
    kWindowStartSeconds + kWindowBars * kBarSeconds;
constexpr double kWindowDurationSeconds = kWindowEndSeconds - kWindowStartSeconds;
constexpr double kPrefillDurationSeconds = 28.0;
constexpr double kPrefillStartSeconds =
    kWindowStartSeconds - kPrefillDurationSeconds;
constexpr double kLandmarkStartSeconds = kWindowStartSeconds;
constexpr double kLandmarkDurationSeconds = 4.0 * kBarSeconds;
constexpr double kUserApproachSeconds = 12.0;
constexpr double kSmokeApproachSeconds = 2.0;
constexpr double kEntryFadeSeconds = 0.25;
constexpr double kExitFadeSeconds = 0.30;
constexpr int kAudioPromptSampleRate = 16000;
constexpr int kAudioPromptSamples = 10 * kAudioPromptSampleRate;
constexpr std::size_t kGeneratedRingSamples = 24576;
constexpr std::size_t kPrimeBufferSamples = 23040;
constexpr std::size_t kSlots = song_world::kSemanticPromptSlots;
constexpr std::size_t kStyles = kSlots - 1;

constexpr std::array<const char*, kSlots> kPrompts{
    "source audio landmark",
    "123 BPM micro house, precise minimal groove, clipped samples, deep bass, instrumental",
    "123 BPM disco, four on the floor drums, octave bass, bright rhythm guitar, strings, instrumental",
    "123 BPM funk, syncopated drums, elastic bass, wah rhythm guitar, instrumental",
    "123 BPM afro house, polyrhythmic hand percussion, deep kick, warm marimba, instrumental",
    "123 BPM acid house, resonant 303 bassline, four on the floor drums, instrumental"};

constexpr std::array<const char*, kStyles> kStyleNames{
    "Micro House", "Disco", "Funk", "Afro House", "Acid"};

constexpr std::array<juce::uint32, kStyles> kStyleColours{
    0xff62d9bb, 0xffff7959, 0xffd8ed56, 0xff9f83ff, 0xffffbd59};

std::uint64_t sample_at(double seconds) {
    return static_cast<std::uint64_t>(std::llround(seconds * kSampleRate));
}

const std::uint64_t kWindowStartSample = sample_at(kWindowStartSeconds);
const std::uint64_t kWindowEndSample = sample_at(kWindowEndSeconds);
const std::uint64_t kEntryFadeSamples = sample_at(kEntryFadeSeconds);
const std::uint64_t kExitFadeSamples = sample_at(kExitFadeSeconds);

enum class WindowState : int { home = 0, entering = 1, world = 2, exiting = 3, past = 4 };

const char* state_name(WindowState state) {
    switch (state) {
        case WindowState::home: return "HOME";
        case WindowState::entering: return "ENTERING";
        case WindowState::world: return "WORLD";
        case WindowState::exiting: return "EXITING";
        case WindowState::past: return "HOME / WINDOW PASSED";
    }
    return "HOME";
}

void update_max(std::atomic<float>& destination, float value) {
    auto previous = destination.load(std::memory_order_relaxed);
    while (value > previous &&
           !destination.compare_exchange_weak(previous, value,
                                              std::memory_order_relaxed)) {}
}

void update_max(std::atomic<std::uint64_t>& destination, std::uint64_t value) {
    auto previous = destination.load(std::memory_order_relaxed);
    while (value > previous &&
           !destination.compare_exchange_weak(previous, value,
                                              std::memory_order_relaxed)) {}
}

void update_min(std::atomic<std::size_t>& destination, std::size_t value) {
    auto previous = destination.load(std::memory_order_relaxed);
    while (value < previous &&
           !destination.compare_exchange_weak(previous, value,
                                              std::memory_order_relaxed)) {}
}

class GoldenTrack {
public:
    bool load(const juce::File& file, std::string& error) {
        juce::AudioFormatManager formats;
        formats.registerBasicFormats();
        std::unique_ptr<juce::AudioFormatReader> reader(formats.createReaderFor(file));
        if (!reader || reader->sampleRate <= 0.0 || reader->numChannels == 0 ||
            reader->lengthInSamples <= 0 ||
            reader->lengthInSamples > std::numeric_limits<int>::max()) {
            error = "Cannot decode Groovejet test track: " +
                    file.getFullPathName().toStdString();
            return false;
        }

        const int input_samples = static_cast<int>(reader->lengthInSamples);
        juce::AudioBuffer<float> decoded(2, input_samples + 8);
        decoded.clear();
        if (!reader->read(&decoded, 0, input_samples, 0, true,
                          reader->numChannels > 1)) {
            error = "Groovejet decode failed";
            return false;
        }
        if (reader->numChannels == 1)
            decoded.copyFrom(1, 0, decoded, 0, 0, input_samples);

        const double ratio = reader->sampleRate / kSampleRate;
        const int output_samples = static_cast<int>(
            std::floor(static_cast<double>(input_samples) / ratio));
        if (output_samples <= static_cast<int>(kWindowEndSeconds * kSampleRate)) {
            error = "Groovejet test track is shorter than the playable window";
            return false;
        }
        audio_.setSize(2, output_samples, false, true, false);
        for (int channel = 0; channel < 2; ++channel) {
            juce::LagrangeInterpolator interpolator;
            interpolator.process(ratio, decoded.getReadPointer(channel),
                                 audio_.getWritePointer(channel), output_samples);
        }
        return true;
    }

    float sample(int channel, std::uint64_t position) const noexcept {
        if (position >= static_cast<std::uint64_t>(audio_.getNumSamples())) return 0.0F;
        return audio_.getSample(channel, static_cast<int>(position));
    }

    std::vector<float> interleaved_segment(double start_seconds,
                                           double end_seconds) const {
        const auto begin = sample_at(start_seconds);
        const auto end = std::min<std::uint64_t>(
            sample_at(end_seconds), static_cast<std::uint64_t>(audio_.getNumSamples()));
        std::vector<float> result(static_cast<std::size_t>(end - begin) * 2);
        for (std::uint64_t position = begin; position < end; ++position) {
            const auto output = static_cast<std::size_t>(position - begin) * 2;
            result[output] = sample(0, position);
            result[output + 1] = sample(1, position);
        }
        return result;
    }

    std::vector<float> landmark_prompt() const {
        const auto begin = static_cast<double>(sample_at(kLandmarkStartSeconds));
        const auto excerpt_frames = static_cast<double>(sample_at(kLandmarkDurationSeconds));
        std::vector<float> output(kAudioPromptSamples, 0.0F);
        for (int index = 0; index < kAudioPromptSamples; ++index) {
            double relative_source_frame =
                static_cast<double>(index) * kSampleRate / kAudioPromptSampleRate;
            relative_source_frame = std::fmod(relative_source_frame, excerpt_frames);
            const double source_frame = begin + relative_source_frame;
            const auto a = static_cast<std::uint64_t>(std::floor(source_frame));
            const auto b = a + 1;
            const float fraction = static_cast<float>(source_frame - std::floor(source_frame));
            const float mono_a = 0.5F * (sample(0, a) + sample(1, a));
            const float mono_b = 0.5F * (sample(0, b) + sample(1, b));
            output[static_cast<std::size_t>(index)] =
                mono_a + (mono_b - mono_a) * fraction;
        }
        return output;
    }

private:
    juce::AudioBuffer<float> audio_;
};

struct TempoEstimate {
    bool valid{false};
    double normalized_bpm{0.0};
    double first_half_bpm{0.0};
    double second_half_bpm{0.0};
    double drift_bpm{0.0};
    double correlation{0.0};
};

struct TempoPeak {
    bool valid{false};
    double bpm{0.0};
    double score{0.0};
};

std::vector<double> onset_envelope(const std::vector<float>& audio,
                                   std::size_t begin,
                                   std::size_t end) {
    begin = std::min(begin, audio.size());
    end = std::min(end, audio.size());
    constexpr int hop = 240;
    std::vector<double> onset;
    if (end <= begin + hop) return onset;
    onset.reserve((end - begin) / hop);
    double previous_energy = 0.0;
    float previous_sample = audio[begin];
    for (std::size_t frame = begin; frame + hop <= end; frame += hop) {
        double transient = 0.0;
        for (int index = 0; index < hop; ++index) {
            const float current = audio[frame + static_cast<std::size_t>(index)];
            const double difference = current - previous_sample;
            transient += difference * difference;
            previous_sample = current;
        }
        const double energy = std::log1p(5000.0 * transient / hop);
        onset.push_back(std::max(0.0, energy - previous_energy));
        previous_energy = energy;
    }
    return onset;
}

TempoPeak estimate_tempo_peak(const std::vector<float>& audio,
                              std::size_t begin,
                              std::size_t end) {
    auto onset = onset_envelope(audio, begin, end);
    if (onset.size() < 600) return {};
    const double mean = std::accumulate(onset.begin(), onset.end(), 0.0) /
                        static_cast<double>(onset.size());
    for (auto& value : onset) value -= mean;
    constexpr double envelope_rate = kSampleRate / 240.0;
    const int minimum_lag = static_cast<int>(std::floor(envelope_rate * 60.0 / 150.0));
    const int maximum_lag = static_cast<int>(std::ceil(envelope_rate * 60.0 / 100.0));
    double best_score = -1.0;
    int best_lag = 0;
    for (int lag = minimum_lag; lag <= maximum_lag; ++lag) {
        double cross = 0.0;
        double energy_a = 0.0;
        double energy_b = 0.0;
        for (std::size_t index = static_cast<std::size_t>(lag);
             index < onset.size(); ++index) {
            const double a = onset[index];
            const double b = onset[index - static_cast<std::size_t>(lag)];
            cross += a * b;
            energy_a += a * a;
            energy_b += b * b;
        }
        const double score = cross /
            std::sqrt(std::max(1.0e-18, energy_a * energy_b));
        if (score > best_score) {
            best_score = score;
            best_lag = lag;
        }
    }
    if (best_lag <= 0 || best_score <= 0.0) return {};
    return {.valid = true,
            .bpm = envelope_rate * 60.0 / static_cast<double>(best_lag),
            .score = best_score};
}

TempoEstimate estimate_tempo(const std::vector<float>& audio) {
    const auto whole = estimate_tempo_peak(audio, 0, audio.size());
    if (!whole.valid) return {};
    const auto middle = audio.size() / 2;
    const auto first = estimate_tempo_peak(audio, 0, middle);
    const auto second = estimate_tempo_peak(audio, middle, audio.size());
    return {.valid = true,
            .normalized_bpm = whole.bpm,
            .first_half_bpm = first.valid ? first.bpm : 0.0,
            .second_half_bpm = second.valid ? second.bpm : 0.0,
            .drift_bpm = first.valid && second.valid
                ? std::abs(first.bpm - second.bpm) : 0.0,
            .correlation = whole.score};
}

struct PhaseEstimate {
    bool valid{false};
    double offset_ms{0.0};
    double first_half_ms{0.0};
    double second_half_ms{0.0};
    double drift_ms{0.0};
    double confidence{0.0};
};

struct PhasePeak {
    bool valid{false};
    double offset_ms{0.0};
    double confidence{0.0};
};

PhasePeak estimate_phase_peak(const std::vector<float>& audio,
                              std::size_t begin,
                              std::size_t end) {
    auto onset = onset_envelope(audio, begin, end);
    if (onset.size() < 300) return {};
    const double mean = std::accumulate(onset.begin(), onset.end(), 0.0) /
                        static_cast<double>(onset.size());
    double variance = 0.0;
    for (const auto value : onset) variance += (value - mean) * (value - mean);
    const double threshold = mean +
        std::sqrt(variance / static_cast<double>(onset.size()));
    constexpr std::size_t bins = 24;
    constexpr double envelope_rate = kSampleRate / 240.0;
    std::array<double, bins> histogram{};
    int events = 0;
    std::size_t last_event = 0;
    const std::size_t refractory = static_cast<std::size_t>(envelope_rate * 0.14);
    for (std::size_t index = 1; index + 1 < onset.size(); ++index) {
        if (onset[index] <= threshold || onset[index] < onset[index - 1] ||
            onset[index] < onset[index + 1] ||
            (events > 0 && index < last_event + refractory)) continue;
        const double absolute_seconds =
            (static_cast<double>(begin) / kSampleRate) +
            (static_cast<double>(index) / envelope_rate);
        double phase = std::fmod(absolute_seconds / kBeatSeconds, 1.0);
        if (phase < 0.0) phase += 1.0;
        const auto bin = std::min<std::size_t>(
            bins - 1, static_cast<std::size_t>(phase * bins));
        histogram[bin] += onset[index];
        last_event = index;
        ++events;
    }
    if (events < 8) return {};
    const double total = std::accumulate(histogram.begin(), histogram.end(), 0.0);
    const auto strongest = std::max_element(histogram.begin(), histogram.end());
    const auto strongest_index = static_cast<std::size_t>(
        std::distance(histogram.begin(), strongest));
    double phase = (static_cast<double>(strongest_index) + 0.5) / bins;
    if (phase > 0.5) phase -= 1.0;
    return {.valid = true,
            .offset_ms = phase * kBeatSeconds * 1000.0,
            .confidence = total > 0.0 ? *strongest / total : 0.0};
}

PhaseEstimate estimate_phase(const std::vector<float>& audio) {
    const auto whole = estimate_phase_peak(audio, 0, audio.size());
    if (!whole.valid) return {};
    const auto middle = audio.size() / 2;
    const auto first = estimate_phase_peak(audio, 0, middle);
    const auto second = estimate_phase_peak(audio, middle, audio.size());
    double drift = 0.0;
    if (first.valid && second.valid) {
        drift = second.offset_ms - first.offset_ms;
        const double beat_ms = kBeatSeconds * 1000.0;
        while (drift > beat_ms * 0.5) drift -= beat_ms;
        while (drift < -beat_ms * 0.5) drift += beat_ms;
    }
    return {.valid = true,
            .offset_ms = whole.offset_ms,
            .first_half_ms = first.valid ? first.offset_ms : 0.0,
            .second_half_ms = second.valid ? second.offset_ms : 0.0,
            .drift_ms = drift,
            .confidence = whole.confidence};
}

class MainComponent final : public juce::AudioAppComponent,
                            private juce::Timer {
public:
    MainComponent(bool smoke_test, juce::File smoke_report, juce::File track_file)
        : smoke_test_(smoke_test), smoke_report_(std::move(smoke_report)),
          track_file_(std::move(track_file)) {
        setOpaque(true);
        configure_label(title_, "SONG WORLD", 30.0F, juce::Font::bold);
        configure_label(subtitle_, "GROOVEJET | PLAYABLE WINDOW + SINGLE-STREAM LAB",
                        14.0F, juce::Font::plain, juce::Colour(0xff9da8b8));
        configure_label(mode_, "HOME | UNTOUCHED MASTER", 16.0F, juce::Font::bold);
        configure_label(portal_, "Preparing MRT2 Small...", 14.0F, juce::Font::bold,
                        juce::Colour(0xffffbd59));
        configure_label(status_, "One persistent stream | no resets | onset follower bypassed",
                        12.0F, juce::Font::plain, juce::Colour(0xffaeb9c7));
        configure_label(technical_, "Loading track and source-conditioned model...",
                        12.0F, juce::Font::plain, juce::Colour(0xff7f8c9d));
        configure_label(weights_, "SOURCE 35% | MICRO HOUSE 65%",
                        12.0F, juce::Font::bold, juce::Colour(0xff62d9bb));
        for (auto* label : {&title_, &subtitle_, &mode_, &portal_, &status_,
                            &technical_, &weights_}) addAndMakeVisible(*label);

        configure_button(play_, "Play", juce::Colour(0xff283545));
        configure_button(enter_, "Enter World", juce::Colour(0xffff6d4a));
        configure_button(home_, "Home", juce::Colour(0xff283545));
        play_.setEnabled(false);
        enter_.setEnabled(false);
        home_.setEnabled(false);
        play_.onClick = [this] {
            const bool next = !playing_.load(std::memory_order_relaxed);
            playing_.store(next, std::memory_order_relaxed);
            play_.setButtonText(next ? "Pause" : "Play");
        };
        enter_.onClick = [this] {
            const auto source = source_position_.load(std::memory_order_relaxed);
            if (source >= kWindowStartSample && source < kWindowEndSample &&
                engine_ready_.load(std::memory_order_acquire)) {
                const double source_seconds = static_cast<double>(source) /
                                              kSampleRate;
                const double beat = std::ceil(
                    (source_seconds - kGroovejetDownbeatSeconds) / kBeatSeconds);
                const auto target = sample_at(kGroovejetDownbeatSeconds +
                                              beat * kBeatSeconds);
                entry_target_sample_.store(
                    std::clamp(target, kWindowStartSample,
                               kWindowEndSample - 1),
                    std::memory_order_relaxed);
                enter_requested_.store(true, std::memory_order_release);
            }
        };
        home_.onClick = [this] {
            const auto state = static_cast<WindowState>(
                state_.load(std::memory_order_relaxed));
            if (state == WindowState::past) {
                replay_window(smoke_test_);
            } else {
                home_requested_.store(true, std::memory_order_release);
            }
        };

        configure_slider(depth_, 0.0, 100.0, 65.0, "%");
        depth_.setColour(juce::Slider::trackColourId, juce::Colour(0xffff6d4a));
        addAndMakeVisible(depth_);
        for (std::size_t index = 0; index < kStyles; ++index) {
            style_labels_[index] = std::make_unique<juce::Label>();
            configure_label(*style_labels_[index], kStyleNames[index], 12.0F,
                            juce::Font::bold, juce::Colour(kStyleColours[index]));
            addAndMakeVisible(*style_labels_[index]);
            style_sliders_[index] = std::make_unique<juce::Slider>();
            configure_slider(*style_sliders_[index], 0.0, 100.0,
                             index == 0 ? 100.0 : 0.0, "%");
            style_sliders_[index]->setColour(juce::Slider::trackColourId,
                                             juce::Colour(kStyleColours[index]));
            addAndMakeVisible(*style_sliders_[index]);
        }

        generated_audio_.setSize(2, 8192, false, true, false);
        if (smoke_test_) generated_capture_.resize(
            static_cast<std::size_t>(std::ceil(kWindowDurationSeconds * kSampleRate)) +
            kDeviceBufferSamples, 0.0F);
        setSize(1040, 680);
        setAudioChannels(0, 2);
        auto setup = deviceManager.getAudioDeviceSetup();
        setup.sampleRate = kSampleRate;
        setup.bufferSize = kDeviceBufferSamples;
        const auto setup_error = deviceManager.setAudioDeviceSetup(setup, true);
        if (setup_error.isNotEmpty()) set_error(
            "Audio setup failed: " + setup_error.toStdString());

        loader_ = std::thread([this] { load_experience(); });
        smoke_started_ms_ = juce::Time::getMillisecondCounterHiRes();
        startTimerHz(20);
    }

    ~MainComponent() override {
        stopTimer();
        shutdownAudio();
        if (loader_.joinable()) loader_.join();
        if (engine_) engine_->stop();
    }

    void prepareToPlay(int, double sample_rate) override {
        actual_sample_rate_.store(sample_rate, std::memory_order_relaxed);
    }

    void getNextAudioBlock(const juce::AudioSourceChannelInfo& output) override {
        if (output.buffer == nullptr || output.buffer->getNumChannels() < 2 ||
            output.numSamples <= 0) return;
        output.clearActiveBufferRegion();
        if (!playing_.load(std::memory_order_relaxed)) return;

        auto source_position = source_position_.load(std::memory_order_relaxed);
        auto state = static_cast<WindowState>(
            state_.load(std::memory_order_relaxed));
        if (replay_transport_requested_.exchange(false,
                                                 std::memory_order_acq_rel)) {
            source_position = sample_at(kWindowStartSeconds -
                                        kUserApproachSeconds);
            state = WindowState::home;
            source_position_.store(source_position, std::memory_order_relaxed);
            state_.store(static_cast<int>(state), std::memory_order_relaxed);
            fade_position_.store(0, std::memory_order_relaxed);
            previous_output_left_ = 0.0F;
            previous_output_right_ = 0.0F;
            replay_transport_applied_count_.fetch_add(1,
                                                      std::memory_order_relaxed);
        }
        const auto block_end = source_position +
            static_cast<std::uint64_t>(output.numSamples);
        int generated_begin = output.numSamples;
        int generated_end = 0;
        // Advance the generated continuation for the entire prepared window,
        // even while Home remains audible, so a later manual entry reaches the
        // same elapsed point rather than replaying generation from window start.
        if (block_end > kWindowStartSample && source_position < kWindowEndSample) {
            generated_begin = source_position < kWindowStartSample
                ? static_cast<int>(kWindowStartSample - source_position) : 0;
            generated_end = block_end > kWindowEndSample
                ? static_cast<int>(kWindowEndSample - source_position)
                : output.numSamples;
            generated_begin = std::clamp(generated_begin, 0, output.numSamples);
            generated_end = std::clamp(generated_end, generated_begin,
                                       output.numSamples);
        }

        const int generated_samples = generated_end - generated_begin;
        bool generated_ok = generated_samples == 0;
        generated_audio_.clear();
        auto* engine = engine_ptr_.load(std::memory_order_acquire);
        if (generated_samples > 0 && engine != nullptr) {
            generated_ok = engine->pull_audio(
                generated_audio_.getWritePointer(0),
                generated_audio_.getWritePointer(1),
                static_cast<std::size_t>(generated_samples));
            generated_reads_.fetch_add(1, std::memory_order_relaxed);
            generated_requested_frames_.fetch_add(
                static_cast<std::uint64_t>(generated_samples),
                std::memory_order_relaxed);
            source_generated_span_frames_.fetch_add(
                static_cast<std::uint64_t>(generated_samples),
                std::memory_order_relaxed);
            if (!generated_ok) underruns_.fetch_add(1, std::memory_order_relaxed);
            const auto requested = generated_requested_frames_.load(
                std::memory_order_relaxed);
            const auto source_span = source_generated_span_frames_.load(
                std::memory_order_relaxed);
            update_max(max_timeline_drift_samples_,
                       requested > source_span ? requested - source_span
                                               : source_span - requested);
        }

        auto* left = output.buffer->getWritePointer(0, output.startSample);
        auto* right = output.buffer->getWritePointer(1, output.startSample);
        const auto* generated_left = generated_audio_.getReadPointer(0);
        const auto* generated_right = generated_audio_.getReadPointer(1);
        auto fade_position = fade_position_.load(std::memory_order_relaxed);
        float block_peak = 0.0F;
        float transition_step = 0.0F;
        float previous_left = previous_output_left_;
        float previous_right = previous_output_right_;
        std::uint64_t transition_frames = 0;
        std::uint64_t exact_home_frames = 0;
        double source_energy = 0.0;
        double generated_energy = 0.0;

        for (int index = 0; index < output.numSamples; ++index) {
            const auto absolute = source_position + static_cast<std::uint64_t>(index);
            if (state == WindowState::home &&
                enter_requested_.load(std::memory_order_acquire) &&
                absolute >= entry_target_sample_.load(std::memory_order_relaxed) &&
                absolute < kWindowEndSample) {
                state = WindowState::entering;
                fade_position = 0;
                enter_requested_.store(false, std::memory_order_release);
                entry_count_.fetch_add(1, std::memory_order_relaxed);
                first_entry_offset_samples_.compare_exchange_strong(
                    first_entry_unset_, absolute - kWindowStartSample,
                    std::memory_order_relaxed);
            }
            if ((state == WindowState::entering || state == WindowState::world) &&
                home_requested_.exchange(false, std::memory_order_acq_rel)) {
                state = WindowState::exiting;
                fade_position = 0;
                manual_home_count_.fetch_add(1, std::memory_order_relaxed);
            }
            if ((state == WindowState::entering || state == WindowState::world) &&
                absolute >= kWindowEndSample - kExitFadeSamples) {
                state = WindowState::exiting;
                fade_position = absolute - (kWindowEndSample - kExitFadeSamples);
                automatic_exit_count_.fetch_add(1, std::memory_order_relaxed);
            }
            if (absolute >= kWindowEndSample) {
                if (state == WindowState::entering || state == WindowState::world ||
                    state == WindowState::exiting) {
                    exit_complete_count_.fetch_add(1, std::memory_order_relaxed);
                }
                state = WindowState::past;
                fade_position = 0;
                enter_requested_.store(false, std::memory_order_release);
                home_requested_.store(false, std::memory_order_release);
            }

            const float original_left = track_.sample(0, absolute);
            const float original_right = track_.sample(1, absolute);
            float generated_l = 0.0F;
            float generated_r = 0.0F;
            if (index >= generated_begin && index < generated_end) {
                const int generated_index = index - generated_begin;
                generated_l = generated_left[generated_index];
                generated_r = generated_right[generated_index];
                if (smoke_test_) {
                    auto capture = capture_position_.load(std::memory_order_relaxed);
                    if (capture < generated_capture_.size()) {
                        generated_capture_[capture] = 0.5F * (generated_l + generated_r);
                        capture_position_.store(capture + 1, std::memory_order_relaxed);
                    }
                }
            }

            float mixed_left = original_left;
            float mixed_right = original_right;
            if (state == WindowState::entering) {
                const float amount = std::clamp(
                    static_cast<float>(fade_position) /
                        static_cast<float>(std::max<std::uint64_t>(1, kEntryFadeSamples)),
                    0.0F, 1.0F);
                mixed_left = original_left * (1.0F - amount) + generated_l * amount;
                mixed_right = original_right * (1.0F - amount) + generated_r * amount;
                ++fade_position;
                ++transition_frames;
                if (fade_position >= kEntryFadeSamples) {
                    state = WindowState::world;
                    fade_position = 0;
                    entry_complete_count_.fetch_add(1, std::memory_order_relaxed);
                }
            } else if (state == WindowState::world) {
                mixed_left = generated_l;
                mixed_right = generated_r;
            } else if (state == WindowState::exiting) {
                const float amount = std::clamp(
                    static_cast<float>(fade_position + 1) /
                        static_cast<float>(std::max<std::uint64_t>(1, kExitFadeSamples)),
                    0.0F, 1.0F);
                mixed_left = generated_l * (1.0F - amount) + original_left * amount;
                mixed_right = generated_r * (1.0F - amount) + original_right * amount;
                ++fade_position;
                ++transition_frames;
                if (fade_position >= kExitFadeSamples) {
                    state = absolute >= kWindowEndSample ? WindowState::past
                                                        : WindowState::home;
                    fade_position = 0;
                    exit_complete_count_.fetch_add(1, std::memory_order_relaxed);
                }
            }

            left[index] = mixed_left;
            right[index] = mixed_right;
            source_energy += static_cast<double>(original_left) * original_left +
                             static_cast<double>(original_right) * original_right;
            generated_energy += static_cast<double>(generated_l) * generated_l +
                                static_cast<double>(generated_r) * generated_r;
            block_peak = std::max(block_peak,
                std::max(std::abs(mixed_left), std::abs(mixed_right)));
            if (state == WindowState::entering || state == WindowState::exiting) {
                transition_step = std::max(transition_step,
                    std::max(std::abs(mixed_left - previous_left),
                             std::abs(mixed_right - previous_right)));
            }
            previous_left = mixed_left;
            previous_right = mixed_right;
            if (state == WindowState::home || state == WindowState::past) {
                const float error = std::max(std::abs(mixed_left - original_left),
                                             std::abs(mixed_right - original_right));
                update_max(max_home_error_, error);
                ++exact_home_frames;
            }
        }

        source_position += static_cast<std::uint64_t>(output.numSamples);
        source_position_.store(source_position, std::memory_order_relaxed);
        state_.store(static_cast<int>(state), std::memory_order_relaxed);
        fade_position_.store(fade_position, std::memory_order_relaxed);
        previous_output_left_ = previous_left;
        previous_output_right_ = previous_right;
        transition_frames_.fetch_add(transition_frames, std::memory_order_relaxed);
        exact_home_frames_.fetch_add(exact_home_frames, std::memory_order_relaxed);
        update_max(max_output_peak_, block_peak);
        update_max(max_transition_step_, transition_step);
        const double denominator = std::max(1, output.numSamples * 2);
        source_level_.store(static_cast<float>(
            std::sqrt(source_energy / denominator)), std::memory_order_relaxed);
        generated_level_.store(static_cast<float>(
            std::sqrt(generated_energy / denominator)), std::memory_order_relaxed);
        audio_callbacks_.fetch_add(1, std::memory_order_relaxed);
    }

    void releaseResources() override {}

    void paint(juce::Graphics& graphics) override {
        juce::ColourGradient background(juce::Colour(0xff0a1017), 0.0F, 0.0F,
                                        juce::Colour(0xff152131), 0.0F,
                                        static_cast<float>(getHeight()), false);
        graphics.setGradientFill(background);
        graphics.fillAll();

        graphics.setColour(juce::Colour(0xff121c28));
        graphics.fillRoundedRectangle(field_bounds_, 22.0F);
        graphics.setColour(juce::Colour(0xff283646));
        graphics.drawRoundedRectangle(field_bounds_, 22.0F, 1.0F);

        auto timeline = juce::Rectangle<float>(48.0F, 174.0F, 944.0F, 62.0F);
        graphics.setColour(juce::Colour(0xff1c2937));
        graphics.fillRoundedRectangle(timeline, 10.0F);
        const float start_x = timeline.getX() + timeline.getWidth() * 0.30F;
        const float end_x = timeline.getX() + timeline.getWidth() * 0.85F;
        graphics.setColour(juce::Colour(0x3362d9bb));
        graphics.fillRect(start_x, timeline.getY(), end_x - start_x,
                          timeline.getHeight());
        graphics.setColour(juce::Colour(0xff62d9bb));
        graphics.drawText("PLAYABLE WORLD WINDOW", static_cast<int>(start_x + 10),
                          static_cast<int>(timeline.getY() + 19),
                          static_cast<int>(end_x - start_x - 20), 24,
                          juce::Justification::centred);
        const double display_start = kWindowStartSeconds - kUserApproachSeconds;
        const double display_end = kWindowEndSeconds + 4.0;
        const double source_seconds = source_position_.load(std::memory_order_relaxed) /
                                      kSampleRate;
        const float position = static_cast<float>(std::clamp(
            (source_seconds - display_start) / (display_end - display_start), 0.0, 1.0));
        const float marker_x = timeline.getX() + position * timeline.getWidth();
        graphics.setColour(juce::Colour(0xffff7959));
        graphics.fillRoundedRectangle(marker_x - 2.0F, timeline.getY() - 5.0F,
                                      4.0F, timeline.getHeight() + 10.0F, 2.0F);

        graphics.setColour(juce::Colour(0xff91a0b2));
        graphics.setFont(juce::Font(juce::FontOptions(11.0F, juce::Font::bold)));
        graphics.drawText("WORLD DEPTH | SOURCE LANDMARK TO STYLE",
                          700, 256, 294, 22,
                          juce::Justification::centredLeft);
        graphics.drawText("STYLE MIX | NORMALIZED",
                          700, 348, 294, 20,
                          juce::Justification::centredLeft);

        const auto centre = field_bounds_.getCentre();
        graphics.setColour(juce::Colour(0xffeaf1f8));
        graphics.fillEllipse(centre.x - 32.0F, centre.y - 32.0F, 64.0F, 64.0F);
        graphics.setColour(juce::Colour(0xff14202d));
        graphics.setFont(juce::Font(juce::FontOptions(11.0F, juce::Font::bold)));
        graphics.drawText("SOURCE", static_cast<int>(centre.x - 31),
                          static_cast<int>(centre.y - 10), 62, 20,
                          juce::Justification::centred);
        const std::array<juce::Point<float>, kStyles> positions{
            juce::Point<float>{centre.x - 250.0F, centre.y - 80.0F},
            {centre.x - 125.0F, centre.y + 105.0F},
            {centre.x + 120.0F, centre.y + 110.0F},
            {centre.x + 250.0F, centre.y - 70.0F},
            {centre.x, centre.y - 130.0F}};
        for (std::size_t index = 0; index < kStyles; ++index) {
            const float weight = applied_weights_[index + 1];
            graphics.setColour(juce::Colour(kStyleColours[index]).withAlpha(
                0.25F + 0.75F * weight));
            graphics.drawLine(centre.x, centre.y, positions[index].x,
                              positions[index].y, 1.0F + 7.0F * weight);
            const float radius = 24.0F + 28.0F * weight;
            graphics.fillEllipse(positions[index].x - radius,
                                 positions[index].y - radius,
                                 radius * 2.0F, radius * 2.0F);
            graphics.setColour(juce::Colour(0xff081018));
            graphics.setFont(juce::Font(juce::FontOptions(11.0F, juce::Font::bold)));
            graphics.drawText(kStyleNames[index],
                              static_cast<int>(positions[index].x - radius),
                              static_cast<int>(positions[index].y - 10),
                              static_cast<int>(radius * 2.0F), 20,
                              juce::Justification::centred);
        }
    }

    void resized() override {
        title_.setBounds(40, 24, 260, 40);
        subtitle_.setBounds(40, 64, 600, 22);
        play_.setBounds(40, 106, 118, 44);
        enter_.setBounds(170, 106, 164, 44);
        home_.setBounds(346, 106, 118, 44);
        mode_.setBounds(488, 110, 280, 32);
        portal_.setBounds(772, 110, 230, 32);
        field_bounds_ = juce::Rectangle<float>(40.0F, 258.0F, 610.0F, 356.0F);
        depth_.setBounds(700, 282, 294, 40);
        weights_.setBounds(700, 326, 294, 24);
        int y = 370;
        for (std::size_t index = 0; index < kStyles; ++index) {
            style_labels_[index]->setBounds(700, y, 96, 30);
            style_sliders_[index]->setBounds(798, y, 196, 32);
            y += 42;
        }
        status_.setBounds(48, 628, 680, 20);
        technical_.setBounds(48, 650, 944, 20);
    }

private:
    static void configure_label(juce::Label& label, const juce::String& text,
                                float size, int style,
                                juce::Colour colour = juce::Colour(0xffeff4fa)) {
        label.setText(text, juce::dontSendNotification);
        label.setFont(juce::Font(juce::FontOptions(size, style)));
        label.setColour(juce::Label::textColourId, colour);
        label.setInterceptsMouseClicks(false, false);
    }

    void configure_button(juce::TextButton& button, const juce::String& text,
                          juce::Colour colour) {
        button.setButtonText(text);
        button.setColour(juce::TextButton::buttonColourId, colour);
        button.setColour(juce::TextButton::buttonOnColourId,
                         colour.brighter(0.15F));
        button.setColour(juce::TextButton::textColourOffId,
                         juce::Colour(0xfff4f7fb));
        addAndMakeVisible(button);
    }

    static void configure_slider(juce::Slider& slider, double minimum,
                                 double maximum, double value,
                                 const juce::String& suffix) {
        slider.setSliderStyle(juce::Slider::LinearHorizontal);
        slider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 62, 26);
        slider.setRange(minimum, maximum, 1.0);
        slider.setValue(value, juce::dontSendNotification);
        slider.setTextValueSuffix(suffix);
        slider.setColour(juce::Slider::backgroundColourId,
                         juce::Colour(0xff263242));
        slider.setColour(juce::Slider::thumbColourId,
                         juce::Colour(0xffeff4fa));
        slider.setColour(juce::Slider::textBoxTextColourId,
                         juce::Colour(0xffeff4fa));
        slider.setColour(juce::Slider::textBoxBackgroundColourId,
                         juce::Colour(0xff111923));
        slider.setColour(juce::Slider::textBoxOutlineColourId,
                         juce::Colours::transparentBlack);
    }

    void set_error(std::string error) {
        std::lock_guard lock(error_mutex_);
        error_ = std::move(error);
        engine_failed_.store(true, std::memory_order_release);
    }

    std::string error() const {
        std::lock_guard lock(error_mutex_);
        return error_;
    }

    song_world::ConditioningState conditioning(
        const std::array<float, kSlots>& weights, std::uint64_t sequence) const {
        return {.style_a = weights[0], .style_b = weights[1],
                .style_c = weights[2], .style_d = weights[3],
                .style_e = weights[4], .style_f = weights[5],
                .sequence = sequence};
    }

    void load_experience() {
        std::string load_error;
        if (!track_file_.existsAsFile() || !track_.load(track_file_, load_error)) {
            set_error(load_error.empty() ? "Bundled Groovejet test track is missing"
                                         : load_error);
            return;
        }

        const auto root = juce::File::getSpecialLocation(
                              juce::File::userDocumentsDirectory)
                              .getChildFile("Magenta")
                              .getChildFile("magenta-rt-v2");
        const auto model = root.getChildFile("models")
                              .getChildFile("mrt2_small")
                              .getChildFile("mrt2_small.mlxfn");
        const auto resources = root.getChildFile("resources");
        const auto prefill_model = resources.getChildFile("spectrostream")
                                       .getChildFile("spectrostream_encoder.mlxfn");
        if (!model.existsAsFile() || !resources.isDirectory() ||
            !prefill_model.existsAsFile()) {
            set_error("MRT2 Small assets are missing under " +
                      root.getFullPathName().toStdString());
            return;
        }

        auto candidate = song_world::make_mrt2_generative_engine();
        song_world::EngineConfig config{
            .model_path = model.getFullPathName().toStdString(),
            .resource_directory = resources.getFullPathName().toStdString(),
            .prefill_model_path = prefill_model.getFullPathName().toStdString(),
            .prompt_a = kPrompts[0], .prompt_b = kPrompts[1],
            .prompt_c = kPrompts[2], .prompt_d = kPrompts[3],
            .prompt_e = kPrompts[4], .prompt_f = kPrompts[5],
            .ring_buffer_samples = kGeneratedRingSamples,
            .output_gain_db = -4.0F};
        if (!candidate->prepare(config, load_error)) {
            set_error(load_error);
            return;
        }
        runner_instances_.store(1, std::memory_order_relaxed);

        auto landmark = track_.landmark_prompt();
        if (!candidate->set_audio_prompt(0, landmark.data(), landmark.size(),
                                         "groovejet-window-landmark.wav",
                                         load_error)) {
            set_error(load_error);
            return;
        }
        const auto prompt_started = std::chrono::steady_clock::now();
        while (candidate->telemetry().audio_prompt_statuses[0] != 2) {
            const auto telemetry = candidate->telemetry();
            if (telemetry.audio_prompt_statuses[0] == 3) {
                set_error("MRT2 source landmark encoding failed");
                return;
            }
            if (std::chrono::steady_clock::now() - prompt_started >
                std::chrono::seconds(60)) {
                set_error("MRT2 source landmark encoding exceeded 60 seconds");
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        applied_weights_ = {0.35F, 0.65F, 0.0F, 0.0F, 0.0F, 0.0F};
        conditioning_sequence_ = 1;
        candidate->set_conditioning(conditioning(applied_weights_,
                                                  conditioning_sequence_));
        conditioning_updates_.store(1, std::memory_order_relaxed);

        auto prefill = track_.interleaved_segment(kPrefillStartSeconds,
                                                  kWindowStartSeconds);
        const auto prefill_frames = prefill.size() / 2;
        if (prefill_frames != sample_at(kPrefillDurationSeconds) ||
            !candidate->prefill_source(prefill.data(), prefill_frames,
                                       load_error)) {
            set_error(load_error.empty() ? "Prefill interval was not exactly 28 seconds"
                                         : load_error);
            return;
        }
        prefill_count_.store(1, std::memory_order_relaxed);
        candidate->start();
        runner_start_count_.store(1, std::memory_order_relaxed);

        bool primed = false;
        for (int attempt = 0; attempt < 1600; ++attempt) {
            if (candidate->telemetry().buffer_available_samples >=
                kPrimeBufferSamples) {
                primed = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        if (!primed) {
            set_error("MRT2 Small did not prime 6400 generated samples");
            candidate->stop();
            return;
        }

        const double approach = smoke_test_ ? kSmokeApproachSeconds
                                            : kUserApproachSeconds;
        source_position_.store(sample_at(kWindowStartSeconds - approach),
                               std::memory_order_relaxed);
        engine_ = std::move(candidate);
        engine_ptr_.store(engine_.get(), std::memory_order_release);
        engine_ready_.store(true, std::memory_order_release);
    }

    std::array<float, kSlots> requested_weights() const {
        std::array<float, kSlots> weights{};
        const float depth = static_cast<float>(depth_.getValue() / 100.0);
        float style_total = 0.0F;
        for (const auto& slider : style_sliders_)
            style_total += static_cast<float>(slider->getValue());
        weights[0] = 1.0F - depth;
        if (style_total <= 0.0F) {
            weights[1] = depth;
        } else {
            for (std::size_t index = 0; index < kStyles; ++index) {
                weights[index + 1] = depth *
                    static_cast<float>(style_sliders_[index]->getValue()) /
                    style_total;
            }
        }
        return weights;
    }

    void replay_window(bool arm_entry) {
        if (!engine_ready_.load(std::memory_order_acquire) || !engine_ ||
            replay_in_progress_.exchange(true, std::memory_order_acq_rel)) {
            return;
        }
        playing_.store(false, std::memory_order_release);
        play_.setButtonText("Preparing...");
        home_.setEnabled(false);
        std::string restart_error;
        if (!engine_->restart_from_prefill(restart_error)) {
            replay_in_progress_.store(false, std::memory_order_release);
            set_error(restart_error);
            return;
        }
        runner_start_count_.fetch_add(1, std::memory_order_relaxed);
        replay_count_.fetch_add(1, std::memory_order_relaxed);
        entry_target_sample_.store(kWindowStartSample,
                                   std::memory_order_relaxed);
        enter_requested_.store(arm_entry, std::memory_order_release);
        home_requested_.store(false, std::memory_order_release);
        replay_transport_requested_.store(true, std::memory_order_release);
        playing_.store(true, std::memory_order_release);
        play_.setButtonText("Pause");
        replay_in_progress_.store(false, std::memory_order_release);
    }

    void update_conditioning(const song_world::EngineTelemetry& telemetry) {
        if (!engine_ready_.load(std::memory_order_acquire)) return;
        const double now = juce::Time::getMillisecondCounterHiRes();
        if (now - last_conditioning_update_ms_ < 400.0 ||
            telemetry.buffer_available_samples < kPrimeBufferSamples) return;
        const auto requested = requested_weights();
        float maximum_delta = 0.0F;
        for (std::size_t index = 0; index < kSlots; ++index) {
            const float delta = requested[index] - applied_weights_[index];
            applied_weights_[index] += delta * 0.45F;
            maximum_delta = std::max(maximum_delta, std::abs(delta));
        }
        if (maximum_delta < 0.004F) {
            return;
        }
        if (auto* engine = engine_ptr_.load(std::memory_order_acquire)) {
            engine->set_conditioning(conditioning(applied_weights_,
                                                   ++conditioning_sequence_));
            conditioning_updates_.fetch_add(1, std::memory_order_relaxed);
            last_conditioning_update_ms_ = now;
        }
    }

    void set_smoke_style(std::size_t style, double depth) {
        depth_.setValue(depth, juce::dontSendNotification);
        for (std::size_t index = 0; index < kStyles; ++index)
            style_sliders_[index]->setValue(index == style ? 100.0 : 0.0,
                                            juce::dontSendNotification);
        semantic_transitions_.fetch_add(1, std::memory_order_relaxed);
    }

    void update_labels(const song_world::EngineTelemetry& telemetry) {
        const double source_seconds = source_position_.load(std::memory_order_relaxed) /
                                      kSampleRate;
        const auto state = static_cast<WindowState>(
            state_.load(std::memory_order_relaxed));
        if (state == WindowState::world || state == WindowState::entering ||
            state == WindowState::exiting) {
            mode_.setText("WORLD | GENERATED STREAM ONLY",
                          juce::dontSendNotification);
        } else {
            mode_.setText("HOME | UNTOUCHED MASTER", juce::dontSendNotification);
        }
        if (!engine_ready_.load(std::memory_order_acquire)) {
            portal_.setText(engine_failed_.load(std::memory_order_acquire)
                                ? "SETUP FAILED" : "PREPARING...",
                            juce::dontSendNotification);
        } else if (source_seconds < kWindowStartSeconds) {
            portal_.setText("OPENS IN " +
                juce::String(kWindowStartSeconds - source_seconds, 1) + " s",
                juce::dontSendNotification);
        } else if (enter_requested_.load(std::memory_order_acquire)) {
            portal_.setText("ENTRY ON NEXT BEAT", juce::dontSendNotification);
        } else if (source_seconds < kWindowEndSeconds) {
            portal_.setText("PORTAL AVAILABLE", juce::dontSendNotification);
        } else {
            portal_.setText("PORTAL CLOSED", juce::dontSendNotification);
        }

        double beat_position = (source_seconds - kGroovejetDownbeatSeconds) /
                               kBeatSeconds;
        const int beat_integer = static_cast<int>(std::floor(beat_position));
        const int bar = std::max(0, beat_integer / kBeatsPerBar) + 1;
        const int beat = ((beat_integer % kBeatsPerBar) + kBeatsPerBar) %
                         kBeatsPerBar + 1;
        status_.setText(
            juce::String(state_name(state)) + " | 123 BPM | bar " +
            juce::String(bar) + " beat " + juce::String(beat) +
            " | prefill 28.000 s | replays " +
            juce::String(replay_count_.load()) + " | navigation resets 0",
            juce::dontSendNotification);
        technical_.setText(
            "source " + juce::String(source_seconds, 3) + " s | buffer " +
            juce::String(telemetry.buffer_available_samples) + "/" +
            juce::String(telemetry.buffer_capacity_samples) +
            " | underruns " + juce::String(underruns_.load()) +
            " | MRT2 " + juce::String(telemetry.frame_total_ms, 1) +
            " ms/frame | source level " + juce::String(source_level_.load(), 3) +
            " | gen level " + juce::String(generated_level_.load(), 3),
            juce::dontSendNotification);
        juce::String weight_text = "SOURCE " +
            juce::String(applied_weights_[0] * 100.0F, 0) + "%";
        for (std::size_t index = 0; index < kStyles; ++index) {
            if (applied_weights_[index + 1] >= 0.01F) {
                weight_text += " | " + juce::String(kStyleNames[index]).toUpperCase() +
                    " " + juce::String(applied_weights_[index + 1] * 100.0F, 0) + "%";
            }
        }
        weights_.setText(weight_text, juce::dontSendNotification);

        play_.setEnabled(engine_ready_.load(std::memory_order_acquire));
        enter_.setEnabled(engine_ready_.load(std::memory_order_acquire) &&
                          source_seconds >= kWindowStartSeconds &&
                          source_seconds < kWindowEndSeconds &&
                          state == WindowState::home &&
                          !enter_requested_.load(std::memory_order_acquire));
        home_.setButtonText(state == WindowState::past ? "Replay Window"
                                                       : "Home");
        home_.setEnabled(!replay_in_progress_.load(std::memory_order_acquire) &&
                         (state == WindowState::entering ||
                          state == WindowState::world ||
                          state == WindowState::past));
    }

    void sample_telemetry(const song_world::EngineTelemetry& telemetry) {
        const auto state = static_cast<WindowState>(
            state_.load(std::memory_order_relaxed));
        if (state != WindowState::entering && state != WindowState::world &&
            state != WindowState::exiting) return;
        update_min(min_buffer_available_, telemetry.buffer_available_samples);
        if (telemetry.buffer_available_samples == 0) {
            ++consecutive_zero_buffer_;
            zero_buffer_samples_.fetch_add(1, std::memory_order_relaxed);
            update_max(max_consecutive_zero_buffer_, consecutive_zero_buffer_);
        } else {
            consecutive_zero_buffer_ = 0;
        }
        frame_ms_sum_ += telemetry.frame_total_ms;
        ++frame_ms_samples_;
        max_frame_ms_ = std::max(max_frame_ms_,
                                 static_cast<double>(telemetry.frame_total_ms));
    }

    void advance_smoke_test() {
        const double now = juce::Time::getMillisecondCounterHiRes();
        if (smoke_finished_) return;
        if (now - smoke_started_ms_ > 360000.0 ||
            engine_failed_.load(std::memory_order_acquire)) {
            finish_smoke_test(false);
            return;
        }
        if (!engine_ready_.load(std::memory_order_acquire)) return;
        if (smoke_stage_ == 0) {
            set_smoke_style(0, 65.0);
            enter_requested_.store(true, std::memory_order_release);
            playing_.store(true, std::memory_order_relaxed);
            play_.setButtonText("Pause");
            smoke_stage_ = 1;
        }

        const double source_seconds = source_position_.load(std::memory_order_relaxed) /
                                      kSampleRate;
        const double world_seconds = source_seconds - kWindowStartSeconds;
        if (smoke_stage_ == 1 && world_seconds >= 3.5) {
            set_smoke_style(1, 70.0);
            smoke_stage_ = 2;
        } else if (smoke_stage_ == 2 && world_seconds >= 7.5) {
            set_smoke_style(2, 80.0);
            smoke_stage_ = 3;
        } else if (smoke_stage_ == 3 && world_seconds >= 11.5) {
            set_smoke_style(3, 90.0);
            smoke_stage_ = 4;
        } else if (smoke_stage_ == 4 && world_seconds >= 15.5) {
            set_smoke_style(4, 85.0);
            smoke_stage_ = 5;
        } else if (smoke_stage_ == 5 && world_seconds >= 19.5) {
            depth_.setValue(60.0, juce::dontSendNotification);
            style_sliders_[1]->setValue(70.0, juce::dontSendNotification);
            style_sliders_[4]->setValue(30.0, juce::dontSendNotification);
            semantic_transitions_.fetch_add(1, std::memory_order_relaxed);
            smoke_stage_ = 6;
        } else if (smoke_stage_ == 6 && world_seconds >= 23.0) {
            set_smoke_style(0, 45.0);
            smoke_stage_ = 7;
        } else if (smoke_stage_ == 7 &&
                   source_seconds >= kWindowEndSeconds + 1.0) {
            replay_window(true);
            smoke_stage_ = 8;
        } else if (smoke_stage_ == 8 &&
                   replay_transport_applied_count_.load() == 1 &&
                   source_seconds >= kWindowEndSeconds + 1.0) {
            finish_smoke_test(true);
        }
    }

    void finish_smoke_test(bool reached_end) {
        if (smoke_finished_) return;
        smoke_finished_ = true;
        playing_.store(false, std::memory_order_relaxed);
        const auto telemetry = engine_ ? engine_->telemetry()
                                       : song_world::EngineTelemetry{};
        generated_capture_.resize(std::min(capture_position_.load(),
                                           generated_capture_.size()));
        const auto tempo = estimate_tempo(generated_capture_);
        const auto phase = estimate_phase(generated_capture_);
        const bool passed = reached_end &&
            std::abs(actual_sample_rate_.load() - kSampleRate) < 1.0 &&
            runner_instances_.load() == 1 && runner_start_count_.load() == 2 &&
            prefill_count_.load() == 1 && navigation_reset_count_.load() == 0 &&
            replay_count_.load() == 1 &&
            replay_transport_applied_count_.load() == 1 &&
            entry_count_.load() == 2 && entry_complete_count_.load() == 2 &&
            automatic_exit_count_.load() == 2 &&
            exit_complete_count_.load() >= 2 &&
            first_entry_offset_samples_.load() == 0 &&
            underruns_.load() == 0 && telemetry.dropped_audio_reads == 0 &&
            max_consecutive_zero_buffer_.load() < 3 &&
            generated_requested_frames_.load() >=
                sample_at(2.0 * kWindowDurationSeconds) &&
            max_timeline_drift_samples_.load() == 0 &&
            max_home_error_.load() == 0.0F &&
            semantic_transitions_.load() >= 6 &&
            static_cast<WindowState>(state_.load()) == WindowState::past;

        const auto boolean = [](bool value) { return value ? "true" : "false"; };
        const double mean_frame_ms = frame_ms_samples_ > 0
            ? frame_ms_sum_ / static_cast<double>(frame_ms_samples_) : 0.0;
        const juce::String report =
            "{\n"
            "  \"pass\": " + juce::String(boolean(passed)) + ",\n" +
            "  \"objective_scope\": \"automated transport, stream lifecycle, buffer, transition, and passive timing checks; musical quality requires listening\",\n"
            "  \"model\": \"MRT2 Small\",\n"
            "  \"source_bpm\": 123.000,\n"
            "  \"source_downbeat_seconds\": 0.397,\n"
            "  \"window_start_seconds\": " + juce::String(kWindowStartSeconds, 6) + ",\n" +
            "  \"window_end_seconds\": " + juce::String(kWindowEndSeconds, 6) + ",\n" +
            "  \"window_duration_seconds\": " + juce::String(kWindowDurationSeconds, 6) + ",\n" +
            "  \"window_bars\": 14,\n"
            "  \"prefill_start_seconds\": " + juce::String(kPrefillStartSeconds, 6) + ",\n" +
            "  \"prefill_end_seconds\": " + juce::String(kWindowStartSeconds, 6) + ",\n" +
            "  \"prefill_duration_seconds\": 28.000000,\n"
            "  \"landmark_excerpt_start_seconds\": " + juce::String(kLandmarkStartSeconds, 6) + ",\n" +
            "  \"landmark_excerpt_end_seconds\": " + juce::String(kLandmarkStartSeconds + kLandmarkDurationSeconds, 6) + ",\n" +
            "  \"landmark_prompt_seconds\": 10.000000,\n"
            "  \"landmark_looped_to_ten_seconds\": true,\n"
            "  \"persistent_runner_instances\": " + juce::String(runner_instances_.load()) + ",\n" +
            "  \"runner_start_count\": " + juce::String(runner_start_count_.load()) + ",\n" +
            "  \"prefill_count\": " + juce::String(prefill_count_.load()) + ",\n" +
            "  \"replay_count\": " + juce::String(replay_count_.load()) + ",\n" +
            "  \"replay_transport_applied_count\": " +
                juce::String(replay_transport_applied_count_.load()) + ",\n" +
            "  \"navigation_reset_count\": " + juce::String(navigation_reset_count_.load()) + ",\n" +
            "  \"conditioning_updates\": " + juce::String(conditioning_updates_.load()) + ",\n" +
            "  \"semantic_transitions\": " + juce::String(semantic_transitions_.load()) + ",\n" +
            "  \"entry_count\": " + juce::String(entry_count_.load()) + ",\n" +
            "  \"entry_complete_count\": " + juce::String(entry_complete_count_.load()) + ",\n" +
            "  \"automatic_exit_count\": " + juce::String(automatic_exit_count_.load()) + ",\n" +
            "  \"exit_complete_count\": " + juce::String(exit_complete_count_.load()) + ",\n" +
            "  \"first_entry_offset_samples\": " + juce::String(first_entry_offset_samples_.load()) + ",\n" +
            "  \"entry_crossfade_seconds\": " + juce::String(kEntryFadeSeconds, 3) + ",\n" +
            "  \"exit_crossfade_seconds\": " + juce::String(kExitFadeSeconds, 3) + ",\n" +
            "  \"transition_frames\": " + juce::String(transition_frames_.load()) + ",\n" +
            "  \"max_transition_sample_step\": " + juce::String(max_transition_step_.load(), 7) + ",\n" +
            "  \"generated_requested_frames\": " + juce::String(generated_requested_frames_.load()) + ",\n" +
            "  \"source_generated_span_frames\": " + juce::String(source_generated_span_frames_.load()) + ",\n" +
            "  \"timeline_drift_samples\": " + juce::String(max_timeline_drift_samples_.load()) + ",\n" +
            "  \"underruns\": " + juce::String(underruns_.load()) + ",\n" +
            "  \"runner_dropped_audio_reads\": " + juce::String(telemetry.dropped_audio_reads) + ",\n" +
            "  \"minimum_sampled_buffer_frames\": " + juce::String(min_buffer_available_.load()) + ",\n" +
            "  \"zero_buffer_samples\": " + juce::String(zero_buffer_samples_.load()) + ",\n" +
            "  \"maximum_consecutive_zero_buffer_samples\": " + juce::String(max_consecutive_zero_buffer_.load()) + ",\n" +
            "  \"mean_mrt2_ms_per_frame\": " + juce::String(mean_frame_ms, 3) + ",\n" +
            "  \"max_mrt2_ms_per_frame\": " + juce::String(max_frame_ms_, 3) + ",\n" +
            "  \"home_exact_max_sample_error\": " + juce::String(max_home_error_.load(), 9) + ",\n" +
            "  \"exact_home_frames_checked\": " + juce::String(exact_home_frames_.load()) + ",\n" +
            "  \"max_output_peak\": " + juce::String(max_output_peak_.load(), 7) + ",\n" +
            "  \"generated_tempo_valid\": " + juce::String(boolean(tempo.valid)) + ",\n" +
            "  \"generated_tempo_bpm\": " + juce::String(tempo.normalized_bpm, 3) + ",\n" +
            "  \"generated_first_half_bpm\": " + juce::String(tempo.first_half_bpm, 3) + ",\n" +
            "  \"generated_second_half_bpm\": " + juce::String(tempo.second_half_bpm, 3) + ",\n" +
            "  \"generated_tempo_drift_bpm\": " + juce::String(tempo.drift_bpm, 3) + ",\n" +
            "  \"generated_tempo_autocorrelation\": " + juce::String(tempo.correlation, 6) + ",\n" +
            "  \"passive_phase_valid\": " + juce::String(boolean(phase.valid)) + ",\n" +
            "  \"passive_phase_offset_ms\": " + juce::String(phase.offset_ms, 3) + ",\n" +
            "  \"passive_phase_first_half_ms\": " + juce::String(phase.first_half_ms, 3) + ",\n" +
            "  \"passive_phase_second_half_ms\": " + juce::String(phase.second_half_ms, 3) + ",\n" +
            "  \"passive_phase_drift_ms\": " + juce::String(phase.drift_ms, 3) + ",\n" +
            "  \"passive_phase_confidence\": " + juce::String(phase.confidence, 6) + ",\n" +
            "  \"phase_analysis_note\": \"post-run measurement only; no onset follower or correction exists in the audio path\",\n"
            "  \"collider_patterns_reused\": [\"one RealtimeRunner owner\", \"continuous read_audio_stereo consumer\", \"set_blend_weights reblends MusicCoCa tokens without reset\", \"set_audio_prompt_samples for slot 0 source landmark\", \"ring-buffer producer-consumer\"],\n"
            "  \"collider_reset_paths_used\": false,\n"
            "  \"source_audible_inside_world\": false,\n"
            "  \"onset_follower_enabled\": false,\n"
            "  \"listening_judgment_required\": true,\n"
            "  \"error\": \"" + juce::String(error()).replace("\"", "'") + "\"\n"
            "}\n";
        smoke_report_.getParentDirectory().createDirectory();
        smoke_report_.replaceWithText(report);
        juce::JUCEApplicationBase::getInstance()->setApplicationReturnValue(
            passed ? 0 : 1);
        juce::MessageManager::callAsync([] {
            if (auto* app = juce::JUCEApplicationBase::getInstance()) app->quit();
        });
    }

    void timerCallback() override {
        song_world::EngineTelemetry telemetry;
        if (engine_ready_.load(std::memory_order_acquire) && engine_)
            telemetry = engine_->telemetry();
        update_conditioning(telemetry);
        sample_telemetry(telemetry);
        update_labels(telemetry);
        if (smoke_test_) advance_smoke_test();
        repaint();
    }

    const bool smoke_test_;
    const juce::File smoke_report_;
    const juce::File track_file_;
    GoldenTrack track_;
    std::unique_ptr<song_world::GenerativeEngine> engine_;
    std::atomic<song_world::GenerativeEngine*> engine_ptr_{nullptr};
    std::thread loader_;
    juce::AudioBuffer<float> generated_audio_;

    juce::Label title_, subtitle_, mode_, portal_, status_, technical_, weights_;
    juce::TextButton play_, enter_, home_;
    juce::Slider depth_;
    std::array<std::unique_ptr<juce::Label>, kStyles> style_labels_;
    std::array<std::unique_ptr<juce::Slider>, kStyles> style_sliders_;
    juce::Rectangle<float> field_bounds_;

    std::atomic<bool> engine_ready_{false};
    std::atomic<bool> engine_failed_{false};
    std::atomic<bool> playing_{false};
    std::atomic<bool> enter_requested_{false};
    std::atomic<std::uint64_t> entry_target_sample_{kWindowStartSample};
    std::atomic<bool> home_requested_{false};
    std::atomic<bool> replay_in_progress_{false};
    std::atomic<bool> replay_transport_requested_{false};
    std::atomic<int> state_{static_cast<int>(WindowState::home)};
    std::atomic<std::uint64_t> source_position_{0};
    std::atomic<std::uint64_t> fade_position_{0};
    std::atomic<double> actual_sample_rate_{0.0};
    mutable std::mutex error_mutex_;
    std::string error_;

    std::array<float, kSlots> applied_weights_{0.35F, 0.65F, 0.0F,
                                               0.0F, 0.0F, 0.0F};
    std::uint64_t conditioning_sequence_{1};
    std::atomic<std::uint64_t> runner_instances_{0};
    std::atomic<std::uint64_t> runner_start_count_{0};
    std::atomic<std::uint64_t> prefill_count_{0};
    std::atomic<std::uint64_t> replay_count_{0};
    std::atomic<std::uint64_t> replay_transport_applied_count_{0};
    std::atomic<std::uint64_t> navigation_reset_count_{0};
    std::atomic<std::uint64_t> conditioning_updates_{0};
    std::atomic<std::uint64_t> semantic_transitions_{0};
    std::atomic<std::uint64_t> entry_count_{0};
    std::atomic<std::uint64_t> entry_complete_count_{0};
    std::atomic<std::uint64_t> automatic_exit_count_{0};
    std::atomic<std::uint64_t> manual_home_count_{0};
    std::atomic<std::uint64_t> exit_complete_count_{0};
    std::atomic<std::uint64_t> first_entry_offset_samples_{
        std::numeric_limits<std::uint64_t>::max()};
    std::uint64_t first_entry_unset_{std::numeric_limits<std::uint64_t>::max()};
    std::atomic<std::uint64_t> generated_reads_{0};
    std::atomic<std::uint64_t> generated_requested_frames_{0};
    std::atomic<std::uint64_t> source_generated_span_frames_{0};
    std::atomic<std::uint64_t> max_timeline_drift_samples_{0};
    std::atomic<std::uint64_t> underruns_{0};
    std::atomic<std::uint64_t> transition_frames_{0};
    std::atomic<std::uint64_t> exact_home_frames_{0};
    std::atomic<std::uint64_t> audio_callbacks_{0};
    std::atomic<std::size_t> min_buffer_available_{
        std::numeric_limits<std::size_t>::max()};
    std::atomic<std::uint64_t> zero_buffer_samples_{0};
    std::uint64_t consecutive_zero_buffer_{0};
    std::atomic<std::uint64_t> max_consecutive_zero_buffer_{0};
    std::atomic<float> max_home_error_{0.0F};
    std::atomic<float> max_output_peak_{0.0F};
    std::atomic<float> max_transition_step_{0.0F};
    std::atomic<float> source_level_{0.0F};
    std::atomic<float> generated_level_{0.0F};
    float previous_output_left_{0.0F};
    float previous_output_right_{0.0F};
    double frame_ms_sum_{0.0};
    std::uint64_t frame_ms_samples_{0};
    double max_frame_ms_{0.0};
    double last_conditioning_update_ms_{0.0};

    std::vector<float> generated_capture_;
    std::atomic<std::size_t> capture_position_{0};
    double smoke_started_ms_{0.0};
    int smoke_stage_{0};
    bool smoke_finished_{false};
};

class MainWindow final : public juce::DocumentWindow {
public:
    MainWindow(bool smoke_test, const juce::File& smoke_report,
               const juce::File& track_file)
        : DocumentWindow("Song World Playable Window Lab",
                         juce::Colour(0xff0a1017),
                         juce::DocumentWindow::allButtons) {
        setUsingNativeTitleBar(true);
        setResizable(false, false);
        setContentOwned(new MainComponent(smoke_test, smoke_report, track_file), true);
        centreWithSize(getWidth(), getHeight());
        setVisible(true);
    }
    void closeButtonPressed() override {
        if (auto* app = juce::JUCEApplicationBase::getInstance())
            app->systemRequestedQuit();
    }
};

class SongWorldApplication final : public juce::JUCEApplication {
public:
    const juce::String getApplicationName() override {
        return "Song World Playable Window Lab";
    }
    const juce::String getApplicationVersion() override {
        return "0.8.0-experiment";
    }
    bool moreThanOneInstanceAllowed() override { return false; }

    void initialise(const juce::String&) override {
        bool smoke_test = false;
        auto smoke_report = juce::File::getSpecialLocation(
                                juce::File::tempDirectory)
                                .getChildFile("song-world-playable-window-smoke.json");
        const auto executable = juce::File::getSpecialLocation(
            juce::File::currentExecutableFile);
        auto track_file = executable.getParentDirectory().getParentDirectory()
                              .getChildFile("Resources")
                              .getChildFile("golden-track.mp3");
        for (const auto& argument : getCommandLineParameterArray()) {
            if (argument == "--smoke-test") smoke_test = true;
            if (argument.startsWith("--smoke-report=")) {
                smoke_report = juce::File(
                    argument.fromFirstOccurrenceOf("=", false, false));
            }
            if (argument.startsWith("--track=")) {
                track_file = juce::File(
                    argument.fromFirstOccurrenceOf("=", false, false));
            }
        }
        window_ = std::make_unique<MainWindow>(smoke_test, smoke_report,
                                               track_file);
    }
    void shutdown() override { window_.reset(); }
    void systemRequestedQuit() override { quit(); }

private:
    std::unique_ptr<MainWindow> window_;
};

}  // namespace

START_JUCE_APPLICATION(SongWorldApplication)
