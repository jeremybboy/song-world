#include <song_world/GenerativeEngine.h>
#include <song_world/SourceAnchorMixer.h>

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
constexpr int kPrefillSeconds = 28;
constexpr int kPrefillTailTrimSeconds = 1;
constexpr std::size_t kSlots = song_world::kSemanticPromptSlots;
constexpr std::size_t kStyleSlots = kSlots - 1;
constexpr double kGroovejetBpm = 123.0;
constexpr double kGroovejetDownbeatSeconds = 0.397;
constexpr int kBeatsPerBar = 4;
constexpr int kAudioReferenceSampleRate = 16000;
constexpr int kAudioReferenceSamples = 10 * kAudioReferenceSampleRate;
constexpr std::size_t kGeneratedRingSamples = 8192;
constexpr std::size_t kPrimeBufferSamples = 6400;
constexpr int kTempoCaptureSeconds = 12;
constexpr std::size_t kTempoConditions = 6;
constexpr std::size_t kTempoCaptureSamples =
    static_cast<std::size_t>(kTempoCaptureSeconds * kSampleRate);

constexpr std::array<const char*, kSlots> kDefaultPrompts{
    "123 BPM disco, four on the floor drums, octave bass, bright rhythm guitar, strings, instrumental",
    "123 BPM micro house, minimal shuffled percussion, clipped samples, deep sub bass, instrumental",
    "123 BPM funk, syncopated live drums, slap bass, wah rhythm guitar, instrumental",
    "123 BPM afro house, polyrhythmic hand percussion, deep kick, warm marimba, instrumental",
    "123 BPM acid house, 303 bassline, four on the floor drums, resonant synth, instrumental",
    "audio style reference"};

constexpr std::array<const char*, kStyleSlots> kStyleNames{
    "Disco", "Micro House", "Funk", "Afro House", "Acid"};

constexpr std::array<std::size_t, kTempoConditions> kTempoStyleSlots{
    1, 0, 2, 3, 4, 1};

constexpr std::array<const char*, kTempoConditions> kTempoConditionNames{
    "micro_house", "disco", "funk", "afro_house", "acid",
    "micro_house_audio_reference"};

constexpr std::array<juce::uint32, kSlots> kNodeColours{
    0xffff7959, 0xff62d9bb, 0xffd8ed56, 0xff9f83ff, 0xffffbd59, 0xff5aa9ff};

enum class MonitorMode : int {
    original_solo = 0,
    generated_solo = 1,
    hybrid = 2,
};

enum class GeneratorVariant : int { small = 0, base = 1 };

struct MixProfile {
    float bass;
    float hook;
    float air;
    float generation;
};

struct TempoEstimate {
    bool valid{false};
    double raw_peak_bpm{0.0};
    double normalized_bpm{0.0};
    double half_time_candidate_bpm{0.0};
    double double_time_candidate_bpm{0.0};
    double autocorrelation{0.0};
    double first_half_bpm{0.0};
    double second_half_bpm{0.0};
    double drift_bpm{0.0};
};

struct TempoPeak {
    bool valid{false};
    double raw_bpm{0.0};
    double normalized_bpm{0.0};
    double autocorrelation{0.0};
};

TempoPeak estimate_tempo_peak(const std::vector<float>& samples,
                              double sample_rate,
                              std::size_t begin,
                              std::size_t end) {
    begin = std::min(begin, samples.size());
    end = std::min(end, samples.size());
    if (end <= begin || end - begin < static_cast<std::size_t>(sample_rate * 4.0))
        return {};

    const int hop = std::max(1, static_cast<int>(std::round(sample_rate / 200.0)));
    std::vector<double> onset;
    onset.reserve((end - begin) / static_cast<std::size_t>(hop));
    double previous_energy = 0.0;
    float previous_sample = samples[begin];
    for (std::size_t frame = begin; frame + static_cast<std::size_t>(hop) <= end;
         frame += static_cast<std::size_t>(hop)) {
        double transient_energy = 0.0;
        for (int index = 0; index < hop; ++index) {
            const float current = samples[frame + static_cast<std::size_t>(index)];
            const double difference = static_cast<double>(current - previous_sample);
            transient_energy += difference * difference;
            previous_sample = current;
        }
        const double energy = std::log1p(
            5000.0 * transient_energy / static_cast<double>(hop));
        onset.push_back(std::max(0.0, energy - previous_energy));
        previous_energy = energy;
    }
    if (onset.size() < 400) return {};

    const double mean = std::accumulate(onset.begin(), onset.end(), 0.0) /
                        static_cast<double>(onset.size());
    for (auto& value : onset) value -= mean;

    const double envelope_rate = sample_rate / static_cast<double>(hop);
    const int minimum_lag = static_cast<int>(
        std::floor(envelope_rate * 60.0 / 260.0));
    const int maximum_lag = static_cast<int>(
        std::ceil(envelope_rate * 60.0 / 70.0));
    double best_score = -1.0;
    int best_lag = 0;
    double best_plausible_score = -1.0;
    int best_plausible_lag = 0;
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
        const double bpm = envelope_rate * 60.0 / static_cast<double>(lag);
        if (bpm >= 100.0 && bpm <= 150.0 && score > best_plausible_score) {
            best_plausible_score = score;
            best_plausible_lag = lag;
        }
    }
    if (best_lag <= 0 || best_score <= 0.0 || best_plausible_lag <= 0 ||
        best_plausible_score <= 0.0) return {};

    const double raw = envelope_rate * 60.0 / static_cast<double>(best_lag);
    const double normalized = envelope_rate * 60.0 /
                              static_cast<double>(best_plausible_lag);
    return {.valid = true, .raw_bpm = raw, .normalized_bpm = normalized,
            .autocorrelation = best_plausible_score};
}

TempoEstimate estimate_tempo(const std::vector<float>& samples,
                             double sample_rate) {
    const auto whole = estimate_tempo_peak(samples, sample_rate, 0, samples.size());
    if (!whole.valid) return {};
    const auto middle = samples.size() / 2;
    const auto first = estimate_tempo_peak(samples, sample_rate, 0, middle);
    const auto second = estimate_tempo_peak(samples, sample_rate, middle,
                                            samples.size());
    return {
        .valid = true,
        .raw_peak_bpm = whole.raw_bpm,
        .normalized_bpm = whole.normalized_bpm,
        .half_time_candidate_bpm = whole.normalized_bpm * 0.5,
        .double_time_candidate_bpm = whole.normalized_bpm * 2.0,
        .autocorrelation = whole.autocorrelation,
        .first_half_bpm = first.valid ? first.normalized_bpm : 0.0,
        .second_half_bpm = second.valid ? second.normalized_bpm : 0.0,
        .drift_bpm = first.valid && second.valid
            ? std::abs(first.normalized_bpm - second.normalized_bpm) : 0.0,
    };
}

juce::String tempo_estimate_json(const TempoEstimate& estimate) {
    if (!estimate.valid) return "{\"valid\":false}";
    return "{\"valid\":true,\"raw_peak_bpm\":" +
        juce::String(estimate.raw_peak_bpm, 3) +
        ",\"normalized_bpm\":" + juce::String(estimate.normalized_bpm, 3) +
        ",\"half_time_candidate_bpm\":" +
        juce::String(estimate.half_time_candidate_bpm, 3) +
        ",\"double_time_candidate_bpm\":" +
        juce::String(estimate.double_time_candidate_bpm, 3) +
        ",\"autocorrelation\":" + juce::String(estimate.autocorrelation, 6) +
        ",\"first_half_bpm\":" + juce::String(estimate.first_half_bpm, 3) +
        ",\"second_half_bpm\":" + juce::String(estimate.second_half_bpm, 3) +
        ",\"drift_bpm\":" + juce::String(estimate.drift_bpm, 3) +
        ",\"stable_within_3_bpm\":" +
        juce::String(estimate.drift_bpm <= 3.0 ? "true" : "false") + "}";
}

constexpr std::array<MixProfile, kSlots> kMixProfiles{{
    {0.76F, 0.84F, 0.30F, 0.48F},
    {0.66F, 0.52F, 0.40F, 0.54F},
    {0.82F, 0.72F, 0.28F, 0.49F},
    {0.70F, 0.38F, 0.34F, 0.58F},
    {0.54F, 0.43F, 0.48F, 0.57F},
    {0.48F, 0.38F, 0.46F, 0.60F},
}};

void update_atomic_max(std::atomic<float>& destination, float value) {
    float previous = destination.load(std::memory_order_relaxed);
    while (value > previous &&
           !destination.compare_exchange_weak(previous, value,
                                              std::memory_order_relaxed)) {}
}

class GoldenTrack {
public:
    bool load(const juce::File& file, std::string& error) {
        juce::AudioFormatManager formats;
        formats.registerBasicFormats();
        std::unique_ptr<juce::AudioFormatReader> reader(formats.createReaderFor(file));
        if (!reader) {
            error = "Cannot decode golden track: " + file.getFullPathName().toStdString();
            return false;
        }
        if (reader->lengthInSamples <= 0 ||
            reader->lengthInSamples > std::numeric_limits<int>::max() ||
            reader->sampleRate <= 0.0 || reader->numChannels == 0) {
            error = "Golden track has unsupported audio metadata";
            return false;
        }

        const int input_samples = static_cast<int>(reader->lengthInSamples);
        juce::AudioBuffer<float> decoded(2, input_samples + 8);
        decoded.clear();
        if (!reader->read(&decoded, 0, input_samples, 0, true,
                          reader->numChannels > 1)) {
            error = "Golden track decode failed";
            return false;
        }
        if (reader->numChannels == 1) {
            decoded.copyFrom(1, 0, decoded, 0, 0, input_samples);
        }

        const double speed_ratio = reader->sampleRate / kSampleRate;
        const int output_samples = static_cast<int>(
            std::floor(static_cast<double>(input_samples) / speed_ratio));
        if (output_samples <= static_cast<int>(kSampleRate * 4.0)) {
            error = "Golden track must be longer than four seconds";
            return false;
        }
        audio_.setSize(2, output_samples, false, true, false);
        for (int channel = 0; channel < 2; ++channel) {
            juce::LagrangeInterpolator interpolator;
            interpolator.process(speed_ratio, decoded.getReadPointer(channel),
                                 audio_.getWritePointer(channel), output_samples);
        }
        name_ = file.getFileNameWithoutExtension();
        return true;
    }

    [[nodiscard]] int length() const noexcept { return audio_.getNumSamples(); }
    [[nodiscard]] const juce::String& name() const noexcept { return name_; }
    [[nodiscard]] float sample(int channel, std::uint64_t position) const noexcept {
        if (position >= static_cast<std::uint64_t>(length())) return 0.0F;
        return audio_.getSample(channel, static_cast<int>(position));
    }
    [[nodiscard]] std::size_t prefill_frames() const noexcept {
        return std::min<std::size_t>(static_cast<std::size_t>(length()),
                                     static_cast<std::size_t>(kPrefillSeconds * kSampleRate));
    }
    [[nodiscard]] std::uint64_t handoff_anchor() const noexcept {
        const auto frames = prefill_frames();
        const auto trim = static_cast<std::size_t>(kPrefillTailTrimSeconds * kSampleRate);
        return frames > trim ? frames - trim : 0;
    }
    [[nodiscard]] std::uint64_t prefill_end() const noexcept {
        return static_cast<std::uint64_t>(prefill_frames());
    }
    [[nodiscard]] std::vector<float> interleaved_prefill() const {
        const auto frames = prefill_frames();
        std::vector<float> result(frames * 2);
        for (std::size_t frame = 0; frame < frames; ++frame) {
            result[frame * 2] = audio_.getSample(0, static_cast<int>(frame));
            result[frame * 2 + 1] = audio_.getSample(1, static_cast<int>(frame));
        }
        return result;
    }
    [[nodiscard]] std::vector<float> mono_excerpt(double start_seconds,
                                                   double duration_seconds) const {
        const auto begin = static_cast<std::size_t>(std::clamp(
            start_seconds * kSampleRate, 0.0, static_cast<double>(length())));
        const auto requested_end = begin + static_cast<std::size_t>(
            std::max(0.0, duration_seconds) * kSampleRate);
        const auto end = std::min<std::size_t>(requested_end,
                                               static_cast<std::size_t>(length()));
        std::vector<float> result(end - begin);
        for (std::size_t index = begin; index < end; ++index) {
            result[index - begin] = 0.5F *
                (audio_.getSample(0, static_cast<int>(index)) +
                 audio_.getSample(1, static_cast<int>(index)));
        }
        return result;
    }

private:
    juce::AudioBuffer<float> audio_;
    juce::String name_;
};

class MainComponent final : public juce::AudioAppComponent,
                            private juce::Timer {
public:
    MainComponent(bool smoke_test, juce::File smoke_report, juce::File track_file)
        : smoke_test_(smoke_test), smoke_report_(std::move(smoke_report)),
          track_file_(std::move(track_file)) {
        setOpaque(true);

        configure_label(title_, "SONG WORLD", 30.0F, juce::Font::bold);
        configure_label(subtitle_, "Groovejet | GENERATOR + CLOCK LAB", 14.0F,
                        juce::Font::plain, juce::Colour(0xff9da8b8));
        configure_label(mode_, "HOME | UNTOUCHED MASTER", 16.0F, juce::Font::bold);
        configure_label(status_primary_, "Loading Groovejet, MRT2 Small, and MRT2 Base...",
                        13.0F, juce::Font::plain, juce::Colour(0xffcbd4df));
        configure_label(status_secondary_, "Prefill: waiting | semantic encoder: waiting",
                        12.0F, juce::Font::plain, juce::Colour(0xff8794a6));
        configure_label(status_technical_, "48 kHz | 512 samples | underruns 0",
                        12.0F, juce::Font::plain, juce::Colour(0xff6f7c8c));
        for (auto* label : {&title_, &subtitle_, &mode_, &status_primary_,
                            &status_secondary_, &status_technical_}) {
            addAndMakeVisible(*label);
        }

        configure_button(play_, "Play", juce::Colour(0xff283545));
        configure_button(enter_world_, "Enter World", juce::Colour(0xffff6d4a));
        configure_button(home_, "Home", juce::Colour(0xff283545));
        configure_button(add_prompt_, "+ Add destination", juce::Colour(0xff202b38));
        configure_button(current_mode_, "ORIGINAL SOLO", juce::Colour(0xff283545));
        configure_button(source_owner_, "GENERATED SOLO", juce::Colour(0xff283545));
        configure_button(mrt2_owner_, "HYBRID", juce::Colour(0xffff6d4a));
        for (auto* button : {&current_mode_, &source_owner_, &mrt2_owner_}) {
            button->setClickingTogglesState(true);
            button->setRadioGroupId(1001);
        }
        mrt2_owner_.setToggleState(true, juce::dontSendNotification);
        configure_button(small_model_, "MRT2 SMALL", juce::Colour(0xffff6d4a));
        configure_button(base_model_, "MRT2 BASE (DIAG)", juce::Colour(0xff283545));
        for (auto* button : {&small_model_, &base_model_}) {
            button->setClickingTogglesState(true);
            button->setRadioGroupId(1002);
        }
        small_model_.setToggleState(true, juce::dontSendNotification);
        configure_button(text_only_, "TEXT ONLY", juce::Colour(0xffff6d4a));
        configure_button(text_reference_, "TEXT + REF", juce::Colour(0xff283545));
        for (auto* button : {&text_only_, &text_reference_}) {
            button->setClickingTogglesState(true);
            button->setRadioGroupId(1003);
        }
        text_only_.setToggleState(true, juce::dontSendNotification);
        configure_button(load_reference_, "LOAD REF", juce::Colour(0xff202b38));
        text_reference_.setEnabled(false);
        small_model_.setEnabled(false);
        base_model_.setEnabled(false);
        play_.setEnabled(false);
        enter_world_.setEnabled(false);
        home_.setEnabled(false);

        play_.onClick = [this] {
            const bool next = !playing_.load(std::memory_order_relaxed);
            playing_.store(next, std::memory_order_relaxed);
            play_.setButtonText(next ? "Pause" : "Play");
        };
        enter_world_.onClick = [this] {
            if (semantic_sum() <= 0.0F) set_semantic_value(0, 100.0);
            set_world_depth(25.0);
        };
        home_.onClick = [this] { return_home(); };
        add_prompt_.onClick = [this] { add_destination(); };
        current_mode_.onClick = [this] { set_monitor_mode(MonitorMode::original_solo); };
        source_owner_.onClick = [this] { set_monitor_mode(MonitorMode::generated_solo); };
        mrt2_owner_.onClick = [this] { set_monitor_mode(MonitorMode::hybrid); };
        small_model_.onClick = [this] { switch_generator(GeneratorVariant::small); };
        base_model_.onClick = [this] { switch_generator(GeneratorVariant::base); };
        text_only_.onClick = [this] { set_reference_conditioning(false); };
        text_reference_.onClick = [this] { set_reference_conditioning(true); };
        load_reference_.onClick = [this] { choose_audio_reference(); };

        world_depth_slider_.setSliderStyle(juce::Slider::LinearHorizontal);
        world_depth_slider_.setTextBoxStyle(juce::Slider::TextBoxRight, false, 64, 28);
        world_depth_slider_.setRange(0.0, 100.0, 1.0);
        world_depth_slider_.setTextValueSuffix("%");
        world_depth_slider_.setValue(0.0, juce::dontSendNotification);
        world_depth_slider_.setColour(juce::Slider::trackColourId,
                                      juce::Colour(0xffff6d4a));
        world_depth_slider_.setColour(juce::Slider::backgroundColourId,
                                      juce::Colour(0xff263242));
        world_depth_slider_.setColour(juce::Slider::thumbColourId,
                                      juce::Colour(0xffff6d4a));
        world_depth_slider_.setColour(juce::Slider::textBoxTextColourId,
                                      juce::Colour(0xffeff4fa));
        world_depth_slider_.setColour(juce::Slider::textBoxBackgroundColourId,
                                      juce::Colour(0xff111923));
        world_depth_slider_.setColour(juce::Slider::textBoxOutlineColourId,
                                      juce::Colours::transparentBlack);
        world_depth_slider_.onValueChange = [this] {
            update_world_depth();
        };
        addAndMakeVisible(world_depth_slider_);

        for (std::size_t index = 0; index < kSlots; ++index) {
            prompt_active_[index] = index < kStyleSlots;
            prompt_texts_[index] = index < kStyleSlots
                ? kStyleNames[index] : juce::String("Audio Reference");

            prompt_editors_[index] = std::make_unique<juce::TextEditor>();
            auto& editor = *prompt_editors_[index];
            editor.setText(index < kStyleSlots ? prompt_texts_[index]
                                               : juce::String("Audio Reference"), false);
            editor.setFont(juce::Font(juce::FontOptions(14.0F, juce::Font::bold)));
            editor.setColour(juce::TextEditor::backgroundColourId,
                             juce::Colour(0xff151e29));
            editor.setColour(juce::TextEditor::outlineColourId,
                             juce::Colour(0xff2a3747));
            editor.setColour(juce::TextEditor::focusedOutlineColourId,
                             juce::Colour(kNodeColours[index]));
            editor.setColour(juce::TextEditor::textColourId,
                             juce::Colour(0xffeff4fa));
            editor.onReturnKey = [this, index] { commit_prompt(index); };
            editor.onFocusLost = [this, index] { commit_prompt(index); };
            addAndMakeVisible(editor);

            semantic_sliders_[index] = std::make_unique<juce::Slider>();
            auto& slider = *semantic_sliders_[index];
            slider.setSliderStyle(juce::Slider::LinearHorizontal);
            slider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 58, 24);
            slider.setRange(0.0, 100.0, 1.0);
            slider.setTextValueSuffix("%");
            slider.setValue(index == 0 ? 100.0 : 0.0,
                            juce::dontSendNotification);
            slider.setColour(juce::Slider::trackColourId,
                             juce::Colour(kNodeColours[index]));
            slider.setColour(juce::Slider::backgroundColourId,
                             juce::Colour(0xff263242));
            slider.setColour(juce::Slider::thumbColourId,
                             juce::Colour(kNodeColours[index]));
            slider.setColour(juce::Slider::textBoxTextColourId,
                             juce::Colour(0xffeff4fa));
            slider.setColour(juce::Slider::textBoxBackgroundColourId,
                             juce::Colour(0xff111923));
            slider.setColour(juce::Slider::textBoxOutlineColourId,
                             juce::Colours::transparentBlack);
            slider.onValueChange = [this] { update_semantic_targets(); };
            addAndMakeVisible(slider);

            remove_buttons_[index] = std::make_unique<juce::TextButton>("X");
            auto& remove = *remove_buttons_[index];
            remove.setTooltip("Remove this semantic destination");
            remove.setColour(juce::TextButton::buttonColourId,
                             juce::Colour(0xff1a2430));
            remove.setColour(juce::TextButton::textColourOffId,
                             juce::Colour(0xff8290a2));
            remove.onClick = [this, index] { remove_destination(index); };
            addAndMakeVisible(remove);
            if (index >= kStyleSlots) {
                editor.setVisible(false);
                slider.setVisible(false);
                remove.setVisible(false);
            }
        }
        add_prompt_.setVisible(false);
        setSize(1180, 760);

        if (smoke_test_) {
            for (auto& model_captures : tempo_captures_)
                for (auto& capture : model_captures)
                    capture.resize(kTempoCaptureSamples, 0.0F);
        }
        small_audio_.setSize(2, 8192, false, true, false);
        base_audio_.setSize(2, 8192, false, true, false);
        setAudioChannels(0, 2);
        auto setup = deviceManager.getAudioDeviceSetup();
        setup.sampleRate = kSampleRate;
        setup.bufferSize = kDeviceBufferSamples;
        const auto device_error = deviceManager.setAudioDeviceSetup(setup, true);
        if (device_error.isNotEmpty()) {
            set_engine_error("Audio setup failed: " + device_error.toStdString());
        }

        update_semantic_targets();
        model_loader_ = std::thread([this] { load_experience(); });
        smoke_started_ms_ = juce::Time::getMillisecondCounterHiRes();
        startTimerHz(20);
    }

    ~MainComponent() override {
        stopTimer();
        shutdownAudio();
        if (model_loader_.joinable()) model_loader_.join();
        if (small_engine_) small_engine_->stop();
        if (base_engine_) base_engine_->stop();
    }

    void prepareToPlay(int, double sample_rate) override {
        actual_sample_rate_.store(sample_rate, std::memory_order_relaxed);
        source_analyser_.set_sample_rate(sample_rate);
        generated_analyser_.set_sample_rate(sample_rate);
    }

    void getNextAudioBlock(const juce::AudioSourceChannelInfo& output) override {
        if (output.buffer == nullptr || output.buffer->getNumChannels() < 2 ||
            output.numSamples <= 0) return;

        output.clearActiveBufferRegion();
        small_audio_.clear();
        base_audio_.clear();

        const bool playing = playing_.load(std::memory_order_relaxed);
        bool world_ok = false;
        const int requested_correction = sync_correction_enabled_.load(
            std::memory_order_relaxed)
            ? timing_correction_samples_.load(std::memory_order_relaxed) : 0;
        const int generated_input_samples = std::clamp(
            output.numSamples + requested_correction, 2,
            small_audio_.getNumSamples());
        auto* active_engine = active_engine_.load(std::memory_order_acquire);
        bool small_ok = false;
        bool base_ok = false;
        if (playing && active_engine != nullptr &&
            generated_input_samples <= small_audio_.getNumSamples()) {
            const bool use_base_engine = active_generator_.load(
                std::memory_order_relaxed) ==
                static_cast<int>(GeneratorVariant::base);
            if (!use_base_engine && small_engine_ &&
                small_engine_ready_.load(std::memory_order_acquire)) {
                small_ok = small_engine_->pull_audio(
                    small_audio_.getWritePointer(0), small_audio_.getWritePointer(1),
                    static_cast<std::size_t>(generated_input_samples));
            }
            if (use_base_engine && base_engine_ &&
                base_engine_ready_.load(std::memory_order_acquire)) {
                base_ok = base_engine_->pull_audio(
                    base_audio_.getWritePointer(0), base_audio_.getWritePointer(1),
                    static_cast<std::size_t>(generated_input_samples));
            }
            world_ok = use_base_engine ? base_ok : small_ok;
            world_reads_.fetch_add(1, std::memory_order_relaxed);
            if (!world_ok) world_underruns_.fetch_add(1, std::memory_order_relaxed);
            if (!use_base_engine && !small_ok &&
                small_engine_ready_.load(std::memory_order_relaxed))
                small_underruns_.fetch_add(1, std::memory_order_relaxed);
            if (use_base_engine && !base_ok &&
                base_engine_ready_.load(std::memory_order_relaxed))
                base_underruns_.fetch_add(1, std::memory_order_relaxed);
            if (world_ok) {
                generated_frames_consumed_.fetch_add(
                    static_cast<std::uint64_t>(generated_input_samples),
                    std::memory_order_relaxed);
                timing_correction_total_.fetch_add(requested_correction,
                                                   std::memory_order_relaxed);
            }
        }

        auto* left = output.buffer->getWritePointer(0, output.startSample);
        auto* right = output.buffer->getWritePointer(1, output.startSample);
        const bool use_base = active_generator_.load(std::memory_order_relaxed) ==
                              static_cast<int>(GeneratorVariant::base);
        const auto* small_left = small_audio_.getReadPointer(0);
        const auto* small_right = small_audio_.getReadPointer(1);
        const auto* base_left = base_audio_.getReadPointer(0);
        const auto* base_right = base_audio_.getReadPointer(1);
        const int capture_condition = tempo_capture_condition_.load(
            std::memory_order_acquire);
        auto capture_position = tempo_capture_position_.load(
            std::memory_order_relaxed);
        const bool capture_this_block = capture_condition >= 0 &&
            capture_condition < static_cast<int>(kTempoConditions) &&
            world_ok && capture_position < kTempoCaptureSamples;
        auto source_position = source_position_.load(std::memory_order_relaxed);
        const double sample_rate = std::max(
            1.0, actual_sample_rate_.load(std::memory_order_relaxed));
        const float depth_alpha = static_cast<float>(
            1.0 - std::exp(-1.0 / (0.24 * sample_rate)));
        const float gain_alpha = static_cast<float>(
            1.0 - std::exp(-1.0 / (0.12 * sample_rate)));
        const float envelope_attack = static_cast<float>(
            1.0 - std::exp(-1.0 / (0.012 * sample_rate)));
        const float envelope_release = static_cast<float>(
            1.0 - std::exp(-1.0 / (0.18 * sample_rate)));
        const float availability_alpha = static_cast<float>(
            1.0 - std::exp(-1.0 / (0.045 * sample_rate)));

        const float depth_target = engine_ready_.load(std::memory_order_relaxed)
            ? target_world_depth_.load(std::memory_order_relaxed) : 0.0F;
        float depth = current_world_depth_.load(std::memory_order_relaxed);
        float bass_gain = current_bass_gain_.load(std::memory_order_relaxed);
        float hook_gain = current_hook_gain_.load(std::memory_order_relaxed);
        float air_gain = current_air_gain_.load(std::memory_order_relaxed);
        float generation_gain = current_generation_gain_.load(std::memory_order_relaxed);
        float generated_availability = generated_availability_;
        const float bass_target = target_bass_gain_.load(std::memory_order_relaxed);
        const float hook_target = target_hook_gain_.load(std::memory_order_relaxed);
        const float air_target = target_air_gain_.load(std::memory_order_relaxed);
        const float generation_target = target_generation_gain_.load(std::memory_order_relaxed);

        float output_peak = 0.0F;
        float generated_peak = 0.0F;
        float anchor_peak = 0.0F;
        double source_square_sum = 0.0;
        double generated_square_sum = 0.0;
        std::uint64_t limited = 0;
        std::uint64_t depth_frames = 0;
        std::array<std::uint64_t, 3> mode_frames{};
        const auto monitor_mode = static_cast<MonitorMode>(
            monitor_mode_.load(std::memory_order_relaxed));
        for (int sample = 0; sample < output.numSamples; ++sample) {
            depth += (depth_target - depth) * depth_alpha;
            if (depth_target == 0.0F && depth < 1.0e-5F) depth = 0.0F;
            bass_gain += (bass_target - bass_gain) * gain_alpha;
            hook_gain += (hook_target - hook_gain) * gain_alpha;
            air_gain += (air_target - air_gain) * gain_alpha;
            generation_gain += (generation_target - generation_gain) * gain_alpha;
            generated_availability += ((world_ok ? 1.0F : 0.0F) -
                                       generated_availability) * availability_alpha;

            if (playing) {
                const float original_left = golden_track_.sample(0, source_position);
                const float original_right = golden_track_.sample(1, source_position);
                float small_gen_left = 0.0F;
                float small_gen_right = 0.0F;
                float base_gen_left = 0.0F;
                float base_gen_right = 0.0F;
                if (small_ok || base_ok) {
                    const double read_position = output.numSamples > 1
                        ? static_cast<double>(sample) *
                              static_cast<double>(generated_input_samples - 1) /
                              static_cast<double>(output.numSamples - 1)
                        : 0.0;
                    const int index_a = std::clamp(
                        static_cast<int>(read_position), 0,
                        generated_input_samples - 1);
                    const int index_b = std::min(index_a + 1,
                                                 generated_input_samples - 1);
                    const float fraction = static_cast<float>(
                        read_position - static_cast<double>(index_a));
                    if (small_ok) {
                        small_gen_left = small_left[index_a] +
                            (small_left[index_b] - small_left[index_a]) * fraction;
                        small_gen_right = small_right[index_a] +
                            (small_right[index_b] - small_right[index_a]) * fraction;
                    }
                    if (base_ok) {
                        base_gen_left = base_left[index_a] +
                            (base_left[index_b] - base_left[index_a]) * fraction;
                        base_gen_right = base_right[index_a] +
                            (base_right[index_b] - base_right[index_a]) * fraction;
                    }
                }
                const float gen_left = use_base ? base_gen_left : small_gen_left;
                const float gen_right = use_base ? base_gen_right : small_gen_right;
                if (capture_this_block && capture_position < kTempoCaptureSamples) {
                    const auto condition = static_cast<std::size_t>(capture_condition);
                    tempo_captures_[use_base ? 1 : 0][condition][capture_position] =
                        0.5F * (gen_left + gen_right);
                    ++capture_position;
                }
                source_square_sum += static_cast<double>(original_left) * original_left +
                                     static_cast<double>(original_right) * original_right;
                generated_square_sum += static_cast<double>(gen_left) * gen_left +
                                        static_cast<double>(gen_right) * gen_right;
                const auto source_bands = source_analyser_.analyse(
                    original_left, original_right);
                const auto generated_bands = generated_analyser_.analyse(
                    gen_left, gen_right);
                observe_generated_phase(generated_bands, source_position, sample_rate);

                update_envelope(generated_bass_envelope_,
                                std::max(std::abs(generated_bands.bass_left),
                                         std::abs(generated_bands.bass_right)),
                                envelope_attack, envelope_release);
                update_envelope(generated_hook_envelope_,
                                std::abs(generated_bands.hook_mid),
                                envelope_attack, envelope_release);
                update_envelope(generated_air_envelope_,
                                std::abs(generated_bands.air_side),
                                envelope_attack, envelope_release);

                const float bass_pressure = std::clamp(
                    generated_bass_envelope_ * 3.2F, 0.0F, 1.0F);
                const float hook_pressure = std::clamp(
                    generated_hook_envelope_ * 2.6F, 0.0F, 1.0F);
                const float air_pressure = std::clamp(
                    generated_air_envelope_ * 3.0F, 0.0F, 1.0F);
                const song_world::SourceAnchorGains dynamic_source_gains{
                    .bass = depth * bass_gain * (1.0F - depth * 0.72F * bass_pressure),
                    .hook = depth * hook_gain * (1.0F - depth * 0.52F * hook_pressure),
                    .air = depth * air_gain * (1.0F - depth * 0.42F * air_pressure),
                };
                const auto anchor = song_world::SourceAnchorMixer::render(
                    source_bands, dynamic_source_gains);
                const float dry_gain = 1.0F - std::sqrt(depth) * 0.90F;
                const float generated_level = std::sqrt(depth) *
                    (1.02F + generation_gain * 0.26F);
                const float hybrid_left = original_left * dry_gain +
                    anchor.left * 0.46F + gen_left * generated_level;
                const float hybrid_right = original_right * dry_gain +
                    anchor.right * 0.46F + gen_right * generated_level;

                float mixed_left = original_left;
                float mixed_right = original_right;
                if (depth > 0.0F) {
                    if (monitor_mode == MonitorMode::generated_solo) {
                        mixed_left = gen_left * 1.12F * generated_availability;
                        mixed_right = gen_right * 1.12F * generated_availability;
                    } else if (monitor_mode == MonitorMode::hybrid) {
                        mixed_left = original_left * (1.0F - generated_availability) +
                            hybrid_left * generated_availability;
                        mixed_right = original_right * (1.0F - generated_availability) +
                            hybrid_right * generated_availability;
                    }
                }

                if (depth == 0.0F) {
                    left[sample] = mixed_left;
                    right[sample] = mixed_right;
                } else {
                    limited += std::abs(mixed_left) > 1.0F ? 1 : 0;
                    limited += std::abs(mixed_right) > 1.0F ? 1 : 0;
                    left[sample] = std::clamp(mixed_left, -1.0F, 1.0F);
                    right[sample] = std::clamp(mixed_right, -1.0F, 1.0F);
                }
                output_peak = std::max(output_peak,
                    std::max(std::abs(left[sample]), std::abs(right[sample])));
                generated_peak = std::max(generated_peak,
                    std::max(std::abs(gen_left), std::abs(gen_right)));
                anchor_peak = std::max(anchor_peak,
                    std::max(std::abs(anchor.left), std::abs(anchor.right)));
                ++mode_frames[static_cast<std::size_t>(monitor_mode)];
                if (depth > 0.01F) ++depth_frames;
                ++source_position;
            }
        }

        source_position_.store(source_position, std::memory_order_relaxed);
        current_world_depth_.store(depth, std::memory_order_relaxed);
        current_bass_gain_.store(bass_gain, std::memory_order_relaxed);
        current_hook_gain_.store(hook_gain, std::memory_order_relaxed);
        current_air_gain_.store(air_gain, std::memory_order_relaxed);
        current_generation_gain_.store(generation_gain, std::memory_order_relaxed);
        generated_availability_ = generated_availability;
        if (capture_this_block) {
            tempo_capture_position_.store(capture_position,
                                          std::memory_order_release);
        }
        update_atomic_max(max_world_depth_, depth);
        update_atomic_max(peak_output_, output_peak);
        update_atomic_max(peak_generated_, generated_peak);
        update_atomic_max(peak_anchor_, anchor_peak);
        limited_samples_.fetch_add(limited, std::memory_order_relaxed);
        world_depth_active_frames_.fetch_add(depth_frames, std::memory_order_relaxed);
        if (playing && output.numSamples > 0) {
            const double denominator = static_cast<double>(output.numSamples) * 2.0;
            const float source_rms = static_cast<float>(
                std::sqrt(source_square_sum / denominator));
            const float generated_rms = static_cast<float>(
                std::sqrt(generated_square_sum / denominator));
            source_level_.store(source_level_.load(std::memory_order_relaxed) * 0.78F +
                                source_rms * 0.22F, std::memory_order_relaxed);
            generated_level_.store(
                generated_level_.load(std::memory_order_relaxed) * 0.78F +
                generated_rms * 0.22F, std::memory_order_relaxed);
            source_square_total_.fetch_add(source_square_sum,
                                           std::memory_order_relaxed);
            generated_square_total_.fetch_add(generated_square_sum,
                                              std::memory_order_relaxed);
            level_samples_.fetch_add(static_cast<std::uint64_t>(output.numSamples) * 2,
                                     std::memory_order_relaxed);
        }
        for (std::size_t index = 0; index < mode_frames.size(); ++index) {
            monitor_mode_frames_[index].fetch_add(mode_frames[index],
                                                  std::memory_order_relaxed);
        }
        audio_callbacks_.fetch_add(1, std::memory_order_relaxed);
    }

    void releaseResources() override {}

    void paint(juce::Graphics& graphics) override {
        juce::ColourGradient background(juce::Colour(0xff0b1017), 0.0F, 0.0F,
                                        juce::Colour(0xff111a25),
                                        0.0F, static_cast<float>(getHeight()), false);
        graphics.setGradientFill(background);
        graphics.fillAll();

        graphics.setColour(juce::Colour(0xff121b26));
        graphics.fillRoundedRectangle(world_bounds_, 24.0F);
        graphics.setColour(juce::Colour(0xff202d3b));
        graphics.drawRoundedRectangle(world_bounds_, 24.0F, 1.0F);

        graphics.setColour(juce::Colour(0xff121b26));
        graphics.fillRoundedRectangle(control_bounds_, 20.0F);
        graphics.setColour(juce::Colour(0xff202d3b));
        graphics.drawRoundedRectangle(control_bounds_, 20.0F, 1.0F);

        graphics.setColour(juce::Colour(0xff7e8c9e));
        graphics.setFont(juce::Font(juce::FontOptions(11.0F, juce::Font::bold)));
        graphics.drawText("SEMANTIC FIELD", world_bounds_.toNearestInt().reduced(22, 16)
                              .removeFromTop(22), juce::Justification::centredLeft);
        graphics.drawText("STYLE MIX | NORMALIZED INTERNALLY",
                          control_bounds_.toNearestInt().reduced(20, 16)
                              .removeFromTop(22), juce::Justification::centredLeft);
        graphics.drawText("WORLD DEPTH | DISTANCE FROM HOME",
                          52, 558, 400, 20, juce::Justification::centredLeft);

        draw_world(graphics);
        draw_level_meters(graphics);
    }

    void resized() override {
        title_.setBounds(42, 28, 300, 40);
        subtitle_.setBounds(42, 66, 560, 24);
        play_.setBounds(42, 108, 120, 44);
        enter_world_.setBounds(174, 108, 170, 44);
        home_.setBounds(356, 108, 120, 44);
        mode_.setBounds(500, 112, 210, 34);
        current_mode_.setBounds(725, 108, 130, 44);
        source_owner_.setBounds(865, 108, 142, 44);
        mrt2_owner_.setBounds(1017, 108, 120, 44);

        world_bounds_ = juce::Rectangle<float>(32.0F, 174.0F, 675.0F, 544.0F);
        control_bounds_ = juce::Rectangle<float>(725.0F, 174.0F, 423.0F, 544.0F);

        int y = 220;
        for (std::size_t index = 0; index < kStyleSlots; ++index) {
            prompt_editors_[index]->setBounds(748, y, 154, 32);
            semantic_sliders_[index]->setBounds(910, y - 1, 190, 34);
            remove_buttons_[index]->setBounds(1104, y, 26, 30);
            y += 54;
        }
        small_model_.setBounds(748, 500, 178, 34);
        base_model_.setBounds(938, 500, 178, 34);
        text_only_.setBounds(748, 542, 118, 34);
        text_reference_.setBounds(874, 542, 126, 34);
        load_reference_.setBounds(1008, 542, 108, 34);

        world_depth_slider_.setBounds(72, 582, 590, 42);

        status_primary_.setBounds(52, 652, 610, 20);
        status_secondary_.setBounds(52, 674, 610, 20);
        status_technical_.setBounds(52, 696, 610, 20);
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
        button.setColour(juce::TextButton::buttonOnColourId, colour.brighter(0.15F));
        button.setColour(juce::TextButton::textColourOffId, juce::Colour(0xfff4f7fb));
        addAndMakeVisible(button);
    }

    static void update_envelope(float& envelope, float input, float attack,
                                float release) noexcept {
        envelope += (input - envelope) * (input > envelope ? attack : release);
    }

    void reset_phase_tracker() noexcept {
        generated_onset_fast_ = 0.0F;
        generated_onset_slow_ = 0.0F;
        previous_generated_onset_ = 0.0F;
        generated_onset_peak_ = 0.001F;
        last_generated_onset_sample_ = 0;
        generated_onset_count_.store(0, std::memory_order_relaxed);
        phase_confidence_.store(0.0F, std::memory_order_relaxed);
        phase_offset_ms_.store(0.0F, std::memory_order_relaxed);
        timing_correction_samples_.store(0, std::memory_order_relaxed);
        generated_phase_histogram_.fill(0.0F);
    }

    void observe_generated_phase(const song_world::SourceAnchorBands& bands,
                                 std::uint64_t source_sample,
                                 double sample_rate) noexcept {
        const float bass = 0.5F * (std::abs(bands.bass_left) +
                                    std::abs(bands.bass_right));
        const float fast_alpha = static_cast<float>(
            1.0 - std::exp(-1.0 / (0.004 * sample_rate)));
        const float slow_alpha = static_cast<float>(
            1.0 - std::exp(-1.0 / (0.070 * sample_rate)));
        generated_onset_fast_ += (bass - generated_onset_fast_) * fast_alpha;
        generated_onset_slow_ += (bass - generated_onset_slow_) * slow_alpha;
        const float onset = std::max(0.0F, generated_onset_fast_ -
                                             generated_onset_slow_);
        generated_onset_peak_ = std::max(onset, generated_onset_peak_ * 0.99997F);
        const float threshold = std::max(0.0020F, generated_onset_peak_ * 0.30F);
        const auto beat_samples = sample_rate * 60.0 / kGroovejetBpm;
        const auto refractory = static_cast<std::uint64_t>(beat_samples * 0.28);
        if (previous_generated_onset_ > onset &&
            previous_generated_onset_ > threshold &&
            source_sample > last_generated_onset_sample_ + refractory) {
            last_generated_onset_sample_ = source_sample;
            for (auto& value : generated_phase_histogram_) value *= 0.975F;
            const double downbeat_sample = kGroovejetDownbeatSeconds * sample_rate;
            double phase = std::fmod((static_cast<double>(source_sample) -
                                      downbeat_sample) / beat_samples, 1.0);
            if (phase < 0.0) phase += 1.0;
            const auto bin = std::min<std::size_t>(
                generated_phase_histogram_.size() - 1,
                static_cast<std::size_t>(phase *
                    static_cast<double>(generated_phase_histogram_.size())));
            generated_phase_histogram_[bin] += std::max(previous_generated_onset_,
                                                        threshold);
            generated_onset_count_.fetch_add(1, std::memory_order_relaxed);
            total_generated_onsets_.fetch_add(1, std::memory_order_relaxed);

            float total = 0.0F;
            float strongest = 0.0F;
            std::size_t strongest_bin = 0;
            for (std::size_t index = 0; index < generated_phase_histogram_.size();
                 ++index) {
                total += generated_phase_histogram_[index];
                if (generated_phase_histogram_[index] > strongest) {
                    strongest = generated_phase_histogram_[index];
                    strongest_bin = index;
                }
            }
            const float confidence = strongest / std::max(total, 1.0e-6F);
            double signed_phase =
                (static_cast<double>(strongest_bin) + 0.5) /
                static_cast<double>(generated_phase_histogram_.size());
            if (signed_phase > 0.5) signed_phase -= 1.0;
            const float offset_ms = static_cast<float>(
                signed_phase * (60.0 / kGroovejetBpm) * 1000.0);
            phase_confidence_.store(confidence, std::memory_order_relaxed);
            phase_offset_ms_.store(offset_ms, std::memory_order_relaxed);

            int correction = 0;
            if (generated_onset_count_.load(std::memory_order_relaxed) >= 8 &&
                confidence >= 0.30F && std::abs(signed_phase) < 0.24) {
                const float magnitude = std::abs(offset_ms);
                const int step = magnitude > 35.0F ? 4 :
                                 (magnitude > 18.0F ? 2 : 1);
                if (offset_ms > 4.0F) correction = step;
                if (offset_ms < -4.0F) correction = -step;
            }
            timing_correction_samples_.store(correction,
                                             std::memory_order_relaxed);
        }
        previous_generated_onset_ = onset;
    }

    void update_min_buffer(std::size_t value) noexcept {
        auto previous = min_buffer_available_.load(std::memory_order_relaxed);
        while (value < previous &&
               !min_buffer_available_.compare_exchange_weak(
                   previous, value, std::memory_order_relaxed)) {}
    }

    static void update_min_buffer(std::atomic<std::size_t>& destination,
                                  std::size_t value) noexcept {
        auto previous = destination.load(std::memory_order_relaxed);
        while (value < previous &&
               !destination.compare_exchange_weak(
                   previous, value, std::memory_order_relaxed)) {}
    }

    float semantic_sum() const {
        float sum = 0.0F;
        for (std::size_t index = 0; index < kSlots; ++index) {
            if (prompt_active_[index]) {
                sum += static_cast<float>(semantic_sliders_[index]->getValue() / 100.0);
            }
        }
        return sum;
    }

    void update_semantic_targets() {
        std::array<float, kSlots> raw{};
        float sum = 0.0F;
        for (std::size_t index = 0; index < kSlots; ++index) {
            raw[index] = prompt_active_[index]
                ? static_cast<float>(semantic_sliders_[index]->getValue() / 100.0)
                : 0.0F;
            sum += raw[index];
        }
        float bass = kMixProfiles[0].bass;
        float hook = kMixProfiles[0].hook;
        float air = kMixProfiles[0].air;
        float generation = kMixProfiles[0].generation;
        if (sum > 0.0F) {
            bass = hook = air = generation = 0.0F;
            for (std::size_t index = 0; index < kSlots; ++index) {
                requested_semantic_weights_[index] = raw[index] / sum;
                bass += kMixProfiles[index].bass * requested_semantic_weights_[index];
                hook += kMixProfiles[index].hook * requested_semantic_weights_[index];
                air += kMixProfiles[index].air * requested_semantic_weights_[index];
                generation += kMixProfiles[index].generation *
                              requested_semantic_weights_[index];
            }
        } else {
            requested_semantic_weights_.fill(0.0F);
            requested_semantic_weights_[0] = 1.0F;
        }
        requested_mix_profile_ = {bass, hook, air, generation};
        if (!engine_ready_.load(std::memory_order_acquire) ||
            !playing_.load(std::memory_order_relaxed)) {
            apply_requested_style();
        } else {
            pending_style_change_ = true;
            schedule_quantized_controls();
        }
        update_mode_label();
        ++semantic_slider_updates_;
        repaint();
    }

    void update_world_depth() {
        const float depth = static_cast<float>(
            world_depth_slider_.getValue() / 100.0);
        requested_world_depth_ = depth;
        if (depth == 0.0F || !playing_.load(std::memory_order_relaxed) ||
            !engine_ready_.load(std::memory_order_acquire)) {
            target_world_depth_.store(depth, std::memory_order_relaxed);
            pending_depth_change_ = false;
        } else {
            pending_depth_change_ = true;
            schedule_quantized_controls();
        }
        update_mode_label();
        ++world_depth_updates_;
        repaint();
    }

    MonitorMode active_monitor_mode() const {
        return static_cast<MonitorMode>(
            monitor_mode_.load(std::memory_order_relaxed));
    }

    static const char* monitor_mode_name(MonitorMode mode) {
        switch (mode) {
            case MonitorMode::original_solo: return "ORIGINAL SOLO";
            case MonitorMode::generated_solo: return "GENERATED SOLO";
            case MonitorMode::hybrid: return "HYBRID";
        }
        return "HYBRID";
    }

    void set_monitor_mode(MonitorMode mode) {
        monitor_mode_.store(static_cast<int>(mode), std::memory_order_relaxed);
        repaint();
    }

    const char* active_generator_name() const noexcept {
        return active_generator_.load(std::memory_order_relaxed) ==
                       static_cast<int>(GeneratorVariant::base)
            ? "MRT2 Base" : "MRT2 Small";
    }

    void switch_generator(GeneratorVariant variant) {
        if (active_engine_.load(std::memory_order_acquire) != nullptr &&
            active_generator_.load(std::memory_order_relaxed) ==
                static_cast<int>(variant)) return;
        if (variant == GeneratorVariant::base &&
            !base_model_available_.load(std::memory_order_acquire)) return;

        const bool resume_playback = playing_.exchange(false,
                                                        std::memory_order_relaxed);
        active_engine_.store(nullptr, std::memory_order_release);
        engine_ready_.store(false, std::memory_order_release);
        engine_state_.store(0, std::memory_order_release);
        std::this_thread::sleep_for(std::chrono::milliseconds(40));
        if (audio_reference_requested_.load(std::memory_order_acquire))
            audio_reference_ready_.store(false, std::memory_order_release);
        std::string error;
        if (!load_variant(variant, error)) {
            set_engine_error(error);
            return;
        }
        source_position_.store(golden_track_.prefill_end(),
                               std::memory_order_relaxed);
        generated_frames_consumed_.store(0, std::memory_order_relaxed);
        playing_.store(resume_playback, std::memory_order_relaxed);
        reset_phase_tracker();
        small_model_.setToggleState(variant == GeneratorVariant::small,
                                    juce::dontSendNotification);
        base_model_.setToggleState(variant == GeneratorVariant::base,
                                   juce::dontSendNotification);
    }

    void set_reference_conditioning(bool enabled) {
        if (enabled && !audio_reference_ready_.load(std::memory_order_acquire)) {
            text_only_.setToggleState(true, juce::dontSendNotification);
            return;
        }
        requested_reference_conditioning_ = enabled;
        if (!playing_.load(std::memory_order_relaxed)) {
            reference_conditioning_.store(enabled, std::memory_order_relaxed);
        } else {
            pending_reference_change_ = true;
            schedule_quantized_controls();
        }
    }

    song_world::ConditioningState make_conditioning_state() {
        std::array<float, kSlots> weights = current_semantic_weights_;
        if (reference_conditioning_.load(std::memory_order_relaxed) &&
            audio_reference_ready_.load(std::memory_order_acquire)) {
            const float micro_house = weights[1];
            weights[1] = micro_house * 0.35F;
            weights[kSlots - 1] = micro_house * 0.65F;
        } else {
            weights[kSlots - 1] = 0.0F;
        }
        return {
            .style_a = weights[0], .style_b = weights[1],
            .style_c = weights[2], .style_d = weights[3],
            .style_e = weights[4], .style_f = weights[5],
            .sequence = ++conditioning_sequence_,
        };
    }

    bool decode_audio_reference(const juce::File& file,
                                std::vector<float>& output,
                                std::string& error) {
        juce::AudioFormatManager formats;
        formats.registerBasicFormats();
        std::unique_ptr<juce::AudioFormatReader> reader(
            formats.createReaderFor(file));
        if (!reader || reader->sampleRate <= 0.0 || reader->lengthInSamples <= 0 ||
            reader->numChannels == 0) {
            error = "Cannot decode audio style reference";
            return false;
        }
        const auto input_count = static_cast<int>(std::min<juce::int64>(
            reader->lengthInSamples,
            static_cast<juce::int64>(std::ceil(reader->sampleRate * 10.0))));
        juce::AudioBuffer<float> decoded(
            static_cast<int>(std::min<unsigned int>(2, reader->numChannels)),
            input_count + 8);
        decoded.clear();
        if (!reader->read(&decoded, 0, input_count, 0, true,
                          reader->numChannels > 1)) {
            error = "Audio style reference decode failed";
            return false;
        }
        std::vector<float> mono(static_cast<std::size_t>(input_count + 8), 0.0F);
        for (int sample = 0; sample < input_count; ++sample) {
            const float left = decoded.getSample(0, sample);
            const float right = decoded.getNumChannels() > 1
                ? decoded.getSample(1, sample) : left;
            mono[static_cast<std::size_t>(sample)] = 0.5F * (left + right);
        }
        const double ratio = reader->sampleRate /
                             static_cast<double>(kAudioReferenceSampleRate);
        const int converted_count = std::max(1, std::min(
            kAudioReferenceSamples,
            static_cast<int>(std::floor(static_cast<double>(input_count) / ratio))));
        output.assign(kAudioReferenceSamples, 0.0F);
        juce::LagrangeInterpolator interpolator;
        interpolator.process(ratio, mono.data(), output.data(), converted_count);
        for (int sample = converted_count; sample < kAudioReferenceSamples; ++sample)
            output[static_cast<std::size_t>(sample)] =
                output[static_cast<std::size_t>(sample % converted_count)];
        return true;
    }

    void load_audio_reference(const juce::File& file) {
        std::vector<float> samples;
        std::string error;
        if (!decode_audio_reference(file, samples, error)) {
            set_reference_error(error);
            return;
        }
        auto* engine = active_engine_.load(std::memory_order_acquire);
        if (engine == nullptr || !engine->set_audio_style_reference(
                samples.data(), samples.size(), file.getFileName().toStdString(), error)) {
            set_reference_error(error.empty() ? "MRT2 generator is not ready" : error);
            return;
        }
        audio_reference_samples_ = std::move(samples);
        audio_reference_filename_ = file.getFileName();
        audio_reference_name_ = file.getFileNameWithoutExtension();
        audio_reference_requested_.store(true, std::memory_order_release);
        audio_reference_ready_.store(false, std::memory_order_release);
    }

    void choose_audio_reference() {
        reference_chooser_ = std::make_unique<juce::FileChooser>(
            "Choose a short audio style reference", juce::File{},
            "*.wav;*.aif;*.aiff;*.mp3;*.m4a");
        reference_chooser_->launchAsync(
            juce::FileBrowserComponent::openMode |
                juce::FileBrowserComponent::canSelectFiles,
            [this](const juce::FileChooser& chooser) {
                const auto file = chooser.getResult();
                if (file.existsAsFile()) load_audio_reference(file);
            });
    }

    void set_reference_error(std::string message) {
        std::lock_guard lock(reference_error_mutex_);
        reference_error_ = std::move(message);
    }

    std::uint64_t next_bar_sample(std::uint64_t source_sample) const {
        const double seconds = static_cast<double>(source_sample) / kSampleRate;
        const double beats = (seconds - kGroovejetDownbeatSeconds) /
            (60.0 / kGroovejetBpm);
        const double next_bar_beat =
            (std::floor(beats / kBeatsPerBar) + 1.0) * kBeatsPerBar;
        const double boundary_seconds = kGroovejetDownbeatSeconds +
            next_bar_beat * (60.0 / kGroovejetBpm);
        return static_cast<std::uint64_t>(
            std::max(0.0, std::round(boundary_seconds * kSampleRate)));
    }

    void schedule_quantized_controls() {
        const auto position = source_position_.load(std::memory_order_relaxed);
        if (pending_control_sample_ <= position) {
            pending_control_sample_ = next_bar_sample(position);
            ++quantized_controls_scheduled_;
        }
    }

    void apply_requested_style() {
        target_semantic_weights_ = requested_semantic_weights_;
        target_bass_gain_.store(requested_mix_profile_.bass,
                                std::memory_order_relaxed);
        target_hook_gain_.store(requested_mix_profile_.hook,
                                std::memory_order_relaxed);
        target_air_gain_.store(requested_mix_profile_.air,
                               std::memory_order_relaxed);
        target_generation_gain_.store(requested_mix_profile_.generation,
                                      std::memory_order_relaxed);
    }

    void apply_quantized_controls_if_due() {
        if ((!pending_style_change_ && !pending_depth_change_ &&
             !pending_reference_change_) ||
            !playing_.load(std::memory_order_relaxed)) return;
        const auto position = source_position_.load(std::memory_order_relaxed);
        if (position < pending_control_sample_) return;
        if (pending_style_change_) apply_requested_style();
        if (pending_depth_change_) {
            target_world_depth_.store(requested_world_depth_,
                                      std::memory_order_relaxed);
        }
        if (pending_reference_change_) {
            reference_conditioning_.store(requested_reference_conditioning_,
                                          std::memory_order_relaxed);
        }
        pending_style_change_ = false;
        pending_depth_change_ = false;
        pending_reference_change_ = false;
        const auto late_frames = position - pending_control_sample_;
        max_quantization_late_frames_ = std::max(max_quantization_late_frames_,
                                                late_frames);
        ++quantized_controls_applied_;
        update_mode_label();
    }

    void update_mode_label() {
        const float depth = target_world_depth_.load(std::memory_order_relaxed);
        mode_.setText(depth > 0.0F
                          ? "WORLD DEPTH " + juce::String(depth * 100.0F, 0) + "%"
                          : "HOME | UNTOUCHED MASTER",
                      juce::dontSendNotification);
    }

    void set_world_depth(double percent) {
        world_depth_slider_.setValue(percent, juce::sendNotificationSync);
    }

    void set_semantic_value(std::size_t index, double percent) {
        if (index >= kSlots || !prompt_active_[index]) return;
        semantic_sliders_[index]->setValue(percent, juce::sendNotificationSync);
    }

    void set_style_one_hot(std::size_t selected) {
        if (selected >= kStyleSlots || !prompt_active_[selected]) return;
        for (std::size_t index = 0; index < kSlots; ++index) {
            semantic_sliders_[index]->setValue(index == selected ? 100.0 : 0.0,
                                               juce::dontSendNotification);
        }
        update_semantic_targets();
    }

    void return_home() {
        world_depth_slider_.setValue(0.0, juce::dontSendNotification);
        requested_world_depth_ = 0.0F;
        target_world_depth_.store(0.0F, std::memory_order_relaxed);
        pending_depth_change_ = false;
        update_mode_label();
        ++world_depth_updates_;
    }

    void commit_prompt(std::size_t index) {
        if (index >= kSlots || !prompt_active_[index]) return;
        auto text = prompt_editors_[index]->getText().trim();
        if (text.isEmpty()) text = "Untitled World";
        prompt_editors_[index]->setText(text, false);
        prompt_texts_[index] = text;
        if (!engine_ready_.load(std::memory_order_acquire)) return;
        std::array<std::string, kSlots> prompts;
        for (std::size_t slot = 0; slot < kStyleSlots; ++slot) {
            prompts[slot] = prompt_active_[slot]
                ? ("123 BPM " + prompt_texts_[slot].toStdString() +
                   ", instrumental") : "silence";
        }
        prompts[kSlots - 1] = "audio style reference";
        if (auto* engine = active_engine_.load(std::memory_order_acquire))
            engine->update_semantic_prompts(prompts);
        ++prompt_updates_;
    }

    void remove_destination(std::size_t index) {
        if (index >= kSlots) return;
        prompt_active_[index] = false;
        semantic_sliders_[index]->setValue(0.0, juce::dontSendNotification);
        prompt_editors_[index]->setEnabled(false);
        semantic_sliders_[index]->setEnabled(false);
        remove_buttons_[index]->setEnabled(false);
        prompt_editors_[index]->setText("Empty slot", false);
        add_prompt_.setEnabled(true);
        update_semantic_targets();
        commit_all_prompts();
    }

    void add_destination() {
        for (std::size_t index = 0; index < kSlots; ++index) {
            if (!prompt_active_[index]) {
                prompt_active_[index] = true;
                prompt_texts_[index] = "New World";
                prompt_editors_[index]->setText(prompt_texts_[index], false);
                prompt_editors_[index]->setEnabled(true);
                semantic_sliders_[index]->setEnabled(true);
                remove_buttons_[index]->setEnabled(true);
                prompt_editors_[index]->grabKeyboardFocus();
                break;
            }
        }
        add_prompt_.setEnabled(std::any_of(
            prompt_active_.begin(), prompt_active_.end(), [](bool active) { return !active; }));
        commit_all_prompts();
    }

    void commit_all_prompts() {
        if (!engine_ready_.load(std::memory_order_acquire)) return;
        std::array<std::string, kSlots> prompts;
        for (std::size_t index = 0; index < kStyleSlots; ++index) {
            prompts[index] = prompt_active_[index]
                ? ("123 BPM " + prompt_texts_[index].toStdString() +
                   ", instrumental") : "silence";
        }
        prompts[kSlots - 1] = "audio style reference";
        if (auto* engine = active_engine_.load(std::memory_order_acquire))
            engine->update_semantic_prompts(prompts);
        ++prompt_updates_;
    }

    bool prepare_variant(song_world::GenerativeEngine& engine,
                         const juce::File& model,
                         const juce::File& resources,
                         const juce::File& prefill_model,
                         std::string& error) {
        song_world::EngineConfig config{
            .model_path = model.getFullPathName().toStdString(),
            .resource_directory = resources.getFullPathName().toStdString(),
            .prefill_model_path = prefill_model.getFullPathName().toStdString(),
            .prompt_a = kDefaultPrompts[0], .prompt_b = kDefaultPrompts[1],
            .prompt_c = kDefaultPrompts[2], .prompt_d = kDefaultPrompts[3],
            .prompt_e = kDefaultPrompts[4], .prompt_f = kDefaultPrompts[5],
            .ring_buffer_samples = kGeneratedRingSamples,
            .output_gain_db = -4.0F,
        };
        if (!engine.prepare(config, error)) return false;
        engine.set_conditioning({.style_a = 1.0F, .sequence = 1});
        auto prefill = golden_track_.interleaved_prefill();
        if (!engine.prefill_source(prefill.data(), golden_track_.prefill_frames(),
                                   error)) return false;
        return true;
    }

    bool load_variant(GeneratorVariant variant, std::string& error) {
        const auto root = juce::File::getSpecialLocation(
                              juce::File::userDocumentsDirectory)
                              .getChildFile("Magenta")
                              .getChildFile("magenta-rt-v2");
        const auto small_model = root.getChildFile("models")
                                     .getChildFile("mrt2_small")
                                     .getChildFile("mrt2_small.mlxfn");
        const auto base_model = root.getChildFile("models")
                                    .getChildFile("mrt2_base")
                                    .getChildFile("mrt2_base.mlxfn");
        const auto resources = root.getChildFile("resources");
        const auto prefill_model = resources.getChildFile("spectrostream")
                                       .getChildFile("spectrostream_encoder.mlxfn");
        base_model_available_.store(base_model.existsAsFile(),
                                    std::memory_order_release);
        const auto model = variant == GeneratorVariant::base ? base_model
                                                              : small_model;
        if (!model.existsAsFile() || !resources.isDirectory() ||
            !prefill_model.existsAsFile()) {
            error = std::string("Missing MRT2 ") +
                (variant == GeneratorVariant::base ? "Base" : "Small") +
                " assets under " + root.getFullPathName().toStdString();
            return false;
        }

        if (small_engine_) {
            small_telemetry_snapshot_ = small_engine_->telemetry();
            small_engine_->stop();
            small_engine_.reset();
        }
        if (base_engine_) {
            base_telemetry_snapshot_ = base_engine_->telemetry();
            base_engine_->stop();
            base_engine_.reset();
        }
        small_engine_ready_.store(false, std::memory_order_release);
        base_engine_ready_.store(false, std::memory_order_release);

        auto candidate = song_world::make_mrt2_generative_engine();
        if (!prepare_variant(*candidate, model, resources, prefill_model, error))
            return false;
        if (!audio_reference_samples_.empty() &&
            !candidate->set_audio_style_reference(
                audio_reference_samples_.data(), audio_reference_samples_.size(),
                audio_reference_filename_.toStdString(), error)) {
            return false;
        }
        candidate->start();
        bool primed = false;
        for (int attempt = 0; attempt < 1200; ++attempt) {
            if (candidate->telemetry().buffer_available_samples >=
                kPrimeBufferSamples) {
                primed = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        if (!primed) {
            candidate->stop();
            error = "MRT2 output buffer did not prime to 6400 samples within 12 seconds";
            return false;
        }

        if (variant == GeneratorVariant::base) {
            base_engine_ = std::move(candidate);
            base_engine_ready_.store(true, std::memory_order_release);
            active_engine_.store(base_engine_.get(), std::memory_order_release);
        } else {
            small_engine_ = std::move(candidate);
            small_engine_ready_.store(true, std::memory_order_release);
            active_engine_.store(small_engine_.get(), std::memory_order_release);
        }
        active_generator_.store(static_cast<int>(variant),
                                std::memory_order_relaxed);
        engine_ready_.store(true, std::memory_order_release);
        engine_state_.store(1, std::memory_order_release);
        return true;
    }

    void load_experience() {
        std::string error;
        if (!track_file_.existsAsFile()) {
            set_engine_error("Golden track is missing: " +
                             track_file_.getFullPathName().toStdString());
            return;
        }
        if (!golden_track_.load(track_file_, error)) {
            set_engine_error(error);
            return;
        }
        track_ready_.store(true, std::memory_order_release);
        source_position_.store(golden_track_.prefill_end(),
                               std::memory_order_relaxed);

        if (!load_variant(GeneratorVariant::small, error)) {
            set_engine_error(error);
            return;
        }
        const auto executable = juce::File::getSpecialLocation(
            juce::File::currentExecutableFile);
        const auto bundled_reference = executable.getParentDirectory()
            .getParentDirectory().getChildFile("Resources")
            .getChildFile("micro-house-reference.wav");
        if (bundled_reference.existsAsFile()) load_audio_reference(bundled_reference);
    }

    void set_engine_error(std::string message) {
        {
            std::lock_guard lock(error_mutex_);
            engine_error_ = std::move(message);
        }
        engine_state_.store(-1, std::memory_order_release);
    }

    std::string engine_error() const {
        std::lock_guard lock(error_mutex_);
        return engine_error_;
    }

    void set_base_error(std::string message) {
        std::lock_guard lock(base_error_mutex_);
        base_error_ = std::move(message);
    }

    std::string base_error() const {
        std::lock_guard lock(base_error_mutex_);
        return base_error_;
    }

    void timerCallback() override {
        const int state = engine_state_.load(std::memory_order_acquire);
        if (state == 1) {
            play_.setEnabled(true);
            enter_world_.setEnabled(true);
            home_.setEnabled(true);
            if (!ready_ui_initialized_) {
                ready_ui_initialized_ = true;
                subtitle_.setText(golden_track_.name() +
                                      " | GENERATOR + CLOCK LAB",
                                  juce::dontSendNotification);
            }
            small_model_.setEnabled(true);
            base_model_.setEnabled(
                base_model_available_.load(std::memory_order_acquire));

            apply_quantized_controls_if_due();
            const bool use_base = active_generator_.load(std::memory_order_relaxed) ==
                                  static_cast<int>(GeneratorVariant::base);
            auto live_telemetry = use_base ? base_telemetry_snapshot_
                                           : small_telemetry_snapshot_;
            if (auto* engine = active_engine_.load(std::memory_order_acquire)) {
                live_telemetry = engine->telemetry();
                if (use_base) base_telemetry_snapshot_ = live_telemetry;
                else small_telemetry_snapshot_ = live_telemetry;
            }
            update_min_buffer(live_telemetry.buffer_available_samples);
            update_min_buffer(use_base ? min_base_buffer_available_
                                       : min_small_buffer_available_,
                              live_telemetry.buffer_available_samples);

            if (audio_reference_requested_.load(std::memory_order_acquire) &&
                live_telemetry.audio_reference_status == 2) {
                audio_reference_ready_.store(true, std::memory_order_release);
                text_reference_.setEnabled(true);
                load_reference_.setButtonText("REF: " + audio_reference_name_);
            }

            for (std::size_t index = 0; index < kStyleSlots; ++index) {
                current_semantic_weights_[index] +=
                    (target_semantic_weights_[index] -
                     current_semantic_weights_[index]) * 0.16F;
            }
            const auto conditioning = make_conditioning_state();
            if (auto* engine = active_engine_.load(std::memory_order_acquire))
                engine->set_conditioning(conditioning);

            if (++status_tick_ >= 5) {
                status_tick_ = 0;
                const auto telemetry = live_telemetry;
                const double source_seconds = static_cast<double>(
                    source_position_.load(std::memory_order_relaxed)) / kSampleRate;
                const double absolute_beat = (source_seconds -
                    kGroovejetDownbeatSeconds) / (60.0 / kGroovejetBpm);
                const auto beat_index = static_cast<std::int64_t>(
                    std::floor(std::max(0.0, absolute_beat)));
                const auto bar = beat_index / kBeatsPerBar + 1;
                const auto beat = beat_index % kBeatsPerBar + 1;
                const double generated_seconds = static_cast<double>(
                    golden_track_.prefill_end() +
                    generated_frames_consumed_.load(std::memory_order_relaxed)) /
                    kSampleRate;
                const double generated_absolute_beat = (generated_seconds -
                    kGroovejetDownbeatSeconds) / (60.0 / kGroovejetBpm);
                const auto generated_beat_index = static_cast<std::int64_t>(
                    std::floor(std::max(0.0, generated_absolute_beat)));
                const auto generated_bar = generated_beat_index / kBeatsPerBar + 1;
                const auto generated_beat = generated_beat_index % kBeatsPerBar + 1;
                const double buffer_ms = static_cast<double>(
                    live_telemetry.buffer_available_samples) * 1000.0 / kSampleRate;
                status_primary_.setText(
                    juce::String(monitor_mode_name(active_monitor_mode())) + " | " +
                    active_generator_name() + " | " +
                    juce::String(kGroovejetBpm, 1) + " BPM | SOURCE " +
                    juce::String(bar) + " BEAT " + juce::String(beat),
                    juce::dontSendNotification);
                status_secondary_.setText(
                    "GEN " + juce::String(generated_bar) + ":" +
                    juce::String(generated_beat) + " | " +
                    juce::String(live_telemetry.frame_total_ms, 1) +
                    " ms/frame | buffer " + juce::String(buffer_ms, 0) +
                    " ms | underruns " +
                    juce::String(world_underruns_.load(std::memory_order_relaxed)),
                    juce::dontSendNotification);
                status_technical_.setText(
                    "Phase " + juce::String(
                        phase_offset_ms_.load(std::memory_order_relaxed), 1) +
                    " ms (confidence " + juce::String(
                        phase_confidence_.load(std::memory_order_relaxed) * 100.0F, 0) +
                    "%) | correction " + juce::String(
                        timing_correction_samples_.load(std::memory_order_relaxed)) +
                    "/block" +
                    (pending_style_change_ || pending_depth_change_ ||
                     pending_reference_change_
                         ? " | control pending next bar"
                         : " | controls aligned"),
                    juce::dontSendNotification);
            }
        } else if (state < 0) {
            play_.setEnabled(track_ready_.load(std::memory_order_acquire));
            status_primary_.setText("MRT2 unavailable: " + juce::String(engine_error()),
                                    juce::dontSendNotification);
            status_secondary_.setText("Home remains available; World is disabled.",
                                      juce::dontSendNotification);
        }

        if (smoke_test_) advance_smoke_test();
        repaint();
    }

    static juce::String format_time(double seconds) {
        const int total = std::max(0, static_cast<int>(seconds));
        return juce::String(total / 60).paddedLeft('0', 2) + ":" +
               juce::String(total % 60).paddedLeft('0', 2);
    }

    void draw_world(juce::Graphics& graphics) {
        const auto centre = juce::Point<float>(world_bounds_.getCentreX(),
                                               world_bounds_.getCentreY() - 48.0F);
        juce::Point<float> weighted_destination{};
        for (float radius : {82.0F, 138.0F, 194.0F}) {
            graphics.setColour(juce::Colour(0xff263445).withAlpha(0.55F));
            graphics.drawEllipse(centre.x - radius, centre.y - radius,
                                 radius * 2.0F, radius * 2.0F, 1.0F);
        }

        for (std::size_t index = 0; index < kStyleSlots; ++index) {
            const float angle = -juce::MathConstants<float>::halfPi +
                juce::MathConstants<float>::twoPi *
                    static_cast<float>(index) / static_cast<float>(kStyleSlots);
            const float radius = 168.0F;
            const auto point = centre + juce::Point<float>(
                std::cos(angle) * radius, std::sin(angle) * radius);
            const float weight = prompt_active_[index]
                ? target_semantic_weights_[index] : 0.0F;
            weighted_destination += juce::Point<float>(std::cos(angle),
                                                       std::sin(angle)) * weight;
            const auto colour = juce::Colour(kNodeColours[index]);
            if (weight > 0.0F) {
                graphics.setColour(colour.withAlpha(0.18F + weight * 0.42F));
                graphics.drawLine(centre.x, centre.y, point.x, point.y,
                                  1.0F + weight * 5.0F);
            }
            const float node_radius = 17.0F + weight * 14.0F;
            graphics.setColour(colour.withAlpha(prompt_active_[index] ?
                                                    0.55F + weight * 0.45F : 0.15F));
            graphics.fillEllipse(point.x - node_radius, point.y - node_radius,
                                 node_radius * 2.0F, node_radius * 2.0F);
            graphics.setColour(juce::Colour(0xfff5f8fc));
            graphics.setFont(juce::Font(juce::FontOptions(10.5F, juce::Font::bold)));
            const auto label = prompt_active_[index]
                ? prompt_texts_[index].upToFirstOccurrenceOf(" ", false, false)
                : juce::String("EMPTY");
            graphics.drawText(label.toUpperCase(),
                              juce::Rectangle<int>(static_cast<int>(point.x - 50.0F),
                                                   static_cast<int>(point.y - 8.0F),
                                                   100, 16),
                              juce::Justification::centred);
        }

        const float depth = current_world_depth_.load(std::memory_order_relaxed);
        const float home_radius = 42.0F;
        graphics.setColour(juce::Colour(0xffeff4fa));
        graphics.fillEllipse(centre.x - home_radius, centre.y - home_radius,
                             home_radius * 2.0F, home_radius * 2.0F);
        graphics.setColour(juce::Colour(0xff101721));
        graphics.setFont(juce::Font(juce::FontOptions(13.0F, juce::Font::bold)));
        graphics.drawText("HOME", static_cast<int>(centre.x - 45.0F),
                          static_cast<int>(centre.y - 9.0F), 90, 18,
                          juce::Justification::centred);

        const float puck_radius = 7.0F;
        const auto puck = centre + weighted_destination * (depth * 116.0F);
        graphics.setColour(juce::Colour(0xffff6d4a).withAlpha(0.55F));
        graphics.drawLine(centre.x, centre.y, puck.x, puck.y, 2.0F);
        graphics.setColour(juce::Colour(0xffff6d4a));
        graphics.fillEllipse(puck.x - puck_radius, puck.y - puck_radius,
                             puck_radius * 2.0F, puck_radius * 2.0F);
    }

    static float level_to_db(float level) {
        return juce::Decibels::gainToDecibels(std::max(level, 1.0e-5F), -100.0F);
    }

    void draw_level_meters(juce::Graphics& graphics) {
        const auto draw_meter = [&graphics](juce::Rectangle<float> bounds,
                                             const juce::String& name,
                                             float level,
                                             juce::Colour colour) {
            const float db = level_to_db(level);
            const float normalized = juce::jlimit(0.0F, 1.0F, (db + 60.0F) / 60.0F);
            graphics.setColour(juce::Colour(0xff202d3b));
            graphics.fillRoundedRectangle(bounds, 5.0F);
            auto fill = bounds;
            fill.setWidth(fill.getWidth() * normalized);
            graphics.setColour(colour);
            graphics.fillRoundedRectangle(fill, 5.0F);
            graphics.setColour(juce::Colour(0xffdfe6ef));
            graphics.setFont(juce::Font(juce::FontOptions(10.5F, juce::Font::bold)));
            graphics.drawText(name + "  " + juce::String(db, 1) + " dBFS",
                              bounds.toNearestInt().translated(0, -21),
                              juce::Justification::centredLeft);
        };
        draw_meter({748.0F, 630.0F, 178.0F, 12.0F}, "SOURCE",
                   source_level_.load(std::memory_order_relaxed),
                   juce::Colour(0xffeff4fa));
        draw_meter({948.0F, 630.0F, 178.0F, 12.0F}, "GENERATED",
                   generated_level_.load(std::memory_order_relaxed),
                   juce::Colour(0xffff6d4a));
        const float depth = current_world_depth_.load(std::memory_order_relaxed);
        const float source_master_gain = active_monitor_mode() ==
                                                  MonitorMode::generated_solo
            ? 0.0F : (active_monitor_mode() == MonitorMode::original_solo
                          ? 1.0F : 1.0F - std::sqrt(depth) * 0.90F);
        graphics.setColour(juce::Colour(0xff7e8c9e));
        graphics.setFont(juce::Font(juce::FontOptions(10.5F, juce::Font::plain)));
        graphics.drawText("Full-source path: " +
                              juce::String(source_master_gain * 100.0F, 0) + "%",
                          748, 657, 378, 18, juce::Justification::centredLeft);
    }

    void configure_smoke_tempo_condition(std::size_t condition) {
        if (condition >= kTempoConditions) return;
        reference_conditioning_.store(condition == kTempoConditions - 1,
                                      std::memory_order_relaxed);
        requested_reference_conditioning_ = condition == kTempoConditions - 1;
        text_only_.setToggleState(condition != kTempoConditions - 1,
                                  juce::dontSendNotification);
        text_reference_.setToggleState(condition == kTempoConditions - 1,
                                       juce::dontSendNotification);
        set_style_one_hot(kTempoStyleSlots[condition]);
    }

    void advance_smoke_test() {
        const double now = juce::Time::getMillisecondCounterHiRes();
        if (now - smoke_started_ms_ > 420000.0 ||
            engine_state_.load(std::memory_order_acquire) < 0) {
            finish_smoke_test(false);
            return;
        }
        if (smoke_stage_ == 0 && engine_ready_.load(std::memory_order_acquire)) {
            playing_.store(true, std::memory_order_relaxed);
            play_.setButtonText("Pause");
            set_smoke_monitor(MonitorMode::hybrid);
            switch_generator(GeneratorVariant::small);
            return_home();
            set_style_one_hot(1);
            sync_correction_enabled_.store(false, std::memory_order_relaxed);
            reset_phase_tracker();
            smoke_stage_ = 1;
            smoke_stage_ms_ = now;
        } else if (smoke_stage_ == 1 && now - smoke_stage_ms_ > 2000.0) {
            set_world_depth(65.0);
            advance_smoke_stage(now);
        } else if (smoke_stage_ == 2 && now - smoke_stage_ms_ > 10000.0) {
            uncorrected_phase_ms_ = std::abs(
                phase_offset_ms_.load(std::memory_order_relaxed));
            sync_correction_enabled_.store(true, std::memory_order_relaxed);
            reset_phase_tracker();
            advance_smoke_stage(now);
        } else if (smoke_stage_ == 3 && now - smoke_stage_ms_ > 12000.0) {
            corrected_phase_ms_ = std::abs(
                phase_offset_ms_.load(std::memory_order_relaxed));
            set_smoke_monitor(MonitorMode::generated_solo);
            advance_smoke_stage(now);
        } else if (smoke_stage_ == 4) {
            if (smoke_tempo_condition_ >= kTempoConditions) {
                if (smoke_tempo_model_ == 0) {
                    small_variant_tested_ = true;
                    switch_generator(GeneratorVariant::base);
                    smoke_tempo_model_ = 1;
                    smoke_tempo_condition_ = 0;
                    smoke_tempo_phase_ = 0;
                    smoke_stage_ms_ = now;
                } else {
                    base_variant_tested_ = true;
                    advance_smoke_stage(now);
                }
            } else if (smoke_tempo_phase_ == 0) {
                configure_smoke_tempo_condition(smoke_tempo_condition_);
                smoke_tempo_phase_ = 1;
                smoke_stage_ms_ = now;
            } else if (smoke_tempo_phase_ == 1 &&
                       now - smoke_stage_ms_ > 3000.0) {
                tempo_capture_position_.store(0, std::memory_order_relaxed);
                tempo_capture_condition_.store(
                    static_cast<int>(smoke_tempo_condition_),
                    std::memory_order_release);
                smoke_tempo_phase_ = 2;
                smoke_stage_ms_ = now;
            } else if (smoke_tempo_phase_ == 2 &&
                       tempo_capture_position_.load(std::memory_order_acquire) >=
                           kTempoCaptureSamples) {
                tempo_capture_condition_.store(-1, std::memory_order_release);
                tempo_capture_lengths_[smoke_tempo_model_][smoke_tempo_condition_] =
                    tempo_capture_position_.load(std::memory_order_relaxed);
                ++smoke_tempo_condition_;
                smoke_tempo_phase_ = 0;
                smoke_stage_ms_ = now;
                ++diagnostic_steps_;
            }
        } else if (smoke_stage_ == 5 && now - smoke_stage_ms_ > 4000.0) {
            set_smoke_monitor(MonitorMode::hybrid);
            set_style_one_hot(1);
            advance_smoke_stage(now);
        } else if (smoke_stage_ == 6 && now - smoke_stage_ms_ > 4000.0) {
            set_smoke_monitor(MonitorMode::original_solo);
            advance_smoke_stage(now);
        } else if (smoke_stage_ == 7 && now - smoke_stage_ms_ > 3000.0) {
            return_home();
            advance_smoke_stage(now);
        } else if (smoke_stage_ == 8 && now - smoke_stage_ms_ > 3000.0) {
            if (auto* engine = active_engine_.load(std::memory_order_acquire))
                base_telemetry_snapshot_ = engine->telemetry();
            const auto small_telemetry = small_telemetry_snapshot_;
            const auto base_telemetry = base_telemetry_snapshot_;
            const bool passed = engine_ready_.load() && track_ready_.load() &&
                small_variant_tested_ && base_variant_tested_ &&
                small_telemetry.source_prefilled && base_telemetry.source_prefilled &&
                small_telemetry.prompt_encoder_status == 2 &&
                base_telemetry.prompt_encoder_status == 2 &&
                audio_reference_ready_.load() &&
                audio_callbacks_.load() > 0 && world_reads_.load() > 0 &&
                small_underruns_.load() == 0 && peak_generated_.load() > 0.001F &&
                peak_anchor_.load() > 0.001F && max_world_depth_.load() > 0.60F &&
                current_world_depth_.load() == 0.0F &&
                world_depth_active_frames_.load() > 0 &&
                monitor_mode_frames_[0].load() > 2 * kSampleRate &&
                monitor_mode_frames_[1].load() > 2 * kSampleRate &&
                monitor_mode_frames_[2].load() > 2 * kSampleRate &&
                min_small_buffer_available_.load() > 0 &&
                quantized_controls_scheduled_ >= 4 &&
                quantized_controls_applied_ >= 4 &&
                max_quantization_late_frames_ < 4000 &&
                source_position_.load() > golden_track_.prefill_end() +
                    static_cast<std::uint64_t>(100.0 * kSampleRate) &&
                semantic_slider_updates_ >= 8 &&
                std::all_of(tempo_capture_lengths_.begin(),
                            tempo_capture_lengths_.end(),
                            [](const auto& model_lengths) {
                                return std::all_of(
                                    model_lengths.begin(), model_lengths.end(),
                                    [](std::size_t length) {
                                        return length >= kTempoCaptureSamples;
                                    });
                            }) &&
                total_generated_onsets_.load() >= 8;
            finish_smoke_test(passed);
        }
    }

    void set_smoke_monitor(MonitorMode mode) {
        set_monitor_mode(mode);
        current_mode_.setToggleState(mode == MonitorMode::original_solo,
                                     juce::dontSendNotification);
        source_owner_.setToggleState(mode == MonitorMode::generated_solo,
                                     juce::dontSendNotification);
        mrt2_owner_.setToggleState(mode == MonitorMode::hybrid,
                                   juce::dontSendNotification);
    }

    void advance_smoke_stage(double now) {
        ++smoke_stage_;
        smoke_stage_ms_ = now;
        ++diagnostic_steps_;
    }

    void finish_smoke_test(bool passed) {
        if (smoke_finished_) return;
        smoke_finished_ = true;
        if (auto* engine = active_engine_.load(std::memory_order_acquire)) {
            if (active_generator_.load(std::memory_order_relaxed) ==
                static_cast<int>(GeneratorVariant::base))
                base_telemetry_snapshot_ = engine->telemetry();
            else
                small_telemetry_snapshot_ = engine->telemetry();
        }
        const auto small = small_telemetry_snapshot_;
        const auto base = base_telemetry_snapshot_;
        const double level_samples = static_cast<double>(std::max<std::uint64_t>(
            1, level_samples_.load(std::memory_order_relaxed)));
        const double source_rms = std::sqrt(
            source_square_total_.load(std::memory_order_relaxed) / level_samples);
        const double generated_rms = std::sqrt(
            generated_square_total_.load(std::memory_order_relaxed) / level_samples);
        const bool sync_materially_improved =
            (uncorrected_phase_ms_ <= 10.0F && corrected_phase_ms_ <= 10.0F) ||
            (corrected_phase_ms_ + 5.0F < uncorrected_phase_ms_);
        const auto source_tempo = estimate_tempo(
            golden_track_.mono_excerpt(30.0, 60.0), kSampleRate);
        std::array<std::array<TempoEstimate, kTempoConditions>, 2> tempos;
        std::array<int, 2> stable_condition_counts{};
        for (std::size_t model = 0; model < tempos.size(); ++model) {
            for (std::size_t condition = 0; condition < kTempoConditions;
                 ++condition) {
                tempos[model][condition] = estimate_tempo(
                    tempo_captures_[model][condition], kSampleRate);
                if (tempos[model][condition].valid &&
                    tempos[model][condition].drift_bpm <= 3.0) {
                    ++stable_condition_counts[model];
                }
            }
        }
        juce::String generated_tempos = "{\n";
        for (std::size_t model = 0; model < tempos.size(); ++model) {
            generated_tempos += model == 0 ? "    \"mrt2_small\": {" :
                                             "    \"mrt2_base\": {";
            for (std::size_t condition = 0; condition < kTempoConditions;
                 ++condition) {
                if (condition > 0) generated_tempos += ",";
                generated_tempos += "\n      \"" +
                    juce::String(kTempoConditionNames[condition]) + "\": " +
                    tempo_estimate_json(tempos[model][condition]);
            }
            generated_tempos += "\n    }";
            if (model + 1 < tempos.size()) generated_tempos += ",";
            generated_tempos += "\n";
        }
        generated_tempos += "  }";
        const auto reference_pull_json = [&tempos](std::size_t model) {
            const auto& text = tempos[model][0];
            const auto& reference = tempos[model][kTempoConditions - 1];
            const bool valid = text.valid && reference.valid;
            const double text_distance = valid
                ? std::abs(text.normalized_bpm - 120.0) : 0.0;
            const double reference_distance = valid
                ? std::abs(reference.normalized_bpm - 120.0) : 0.0;
            return "{\"valid\":" + juce::String(valid ? "true" : "false") +
                ",\"text_only_distance_from_120_bpm\":" +
                juce::String(text_distance, 3) +
                ",\"text_plus_reference_distance_from_120_bpm\":" +
                juce::String(reference_distance, 3) +
                ",\"appears_to_pull_toward_reference_tempo\":" +
                juce::String(valid && reference_distance + 0.5 < text_distance
                                 ? "true" : "false") + "}";
        };
        const juce::String report =
            "{\n"
            "  \"passed\": " + juce::String(passed ? "true" : "false") + ",\n" +
            "  \"engine_error\": \"" + juce::String(engine_error()) + "\",\n" +
            "  \"base_error\": \"" + juce::String(base_error()) + "\",\n" +
            "  \"small_source_prefilled\": " + juce::String(small.source_prefilled ? "true" : "false") + ",\n" +
            "  \"base_source_prefilled\": " + juce::String(base.source_prefilled ? "true" : "false") + ",\n" +
            "  \"small_prefill_ms\": " + juce::String(small.source_prefill_ms, 3) + ",\n" +
            "  \"base_prefill_ms\": " + juce::String(base.source_prefill_ms, 3) + ",\n" +
            "  \"small_frame_ms\": " + juce::String(small.frame_total_ms, 3) + ",\n" +
            "  \"base_frame_ms\": " + juce::String(base.frame_total_ms, 3) + ",\n" +
            "  \"small_buffer_samples\": " + juce::String(static_cast<juce::int64>(small.buffer_available_samples)) + ",\n" +
            "  \"base_buffer_samples\": " + juce::String(static_cast<juce::int64>(base.buffer_available_samples)) + ",\n" +
            "  \"min_observed_buffer_samples\": " + juce::String(static_cast<juce::int64>(min_buffer_available_.load())) + ",\n" +
            "  \"small_min_observed_buffer_samples\": " +
                juce::String(static_cast<juce::int64>(min_small_buffer_available_.load())) + ",\n" +
            "  \"base_min_observed_buffer_samples\": " +
                juce::String(static_cast<juce::int64>(min_base_buffer_available_.load())) + ",\n" +
            "  \"device_buffer_samples\": " + juce::String(kDeviceBufferSamples) + ",\n" +
            "  \"source_bpm\": " + juce::String(kGroovejetBpm, 3) + ",\n" +
            "  \"source_tempo_analysis\": " + tempo_estimate_json(source_tempo) + ",\n" +
            "  \"source_tempo_selected_bpm\": 123.000,\n"
            "  \"source_tempo_half_double_ambiguity_checked\": true,\n"
            "  \"source_downbeat_seconds\": " + juce::String(kGroovejetDownbeatSeconds, 3) + ",\n" +
            "  \"generated_origin_source_seconds\": " + juce::String(static_cast<double>(golden_track_.prefill_end()) / kSampleRate, 3) + ",\n" +
            "  \"final_source_seconds\": " + juce::String(static_cast<double>(source_position_.load()) / kSampleRate, 3) + ",\n" +
            "  \"audio_callbacks\": " + juce::String(audio_callbacks_.load()) + ",\n" +
            "  \"world_reads\": " + juce::String(world_reads_.load()) + ",\n" +
            "  \"world_underruns\": " + juce::String(world_underruns_.load()) + ",\n" +
            "  \"small_underruns\": " + juce::String(small_underruns_.load()) + ",\n" +
            "  \"base_underruns\": " + juce::String(base_underruns_.load()) + ",\n" +
            "  \"small_realtime_viable\": " +
                juce::String(small_underruns_.load() == 0 ? "true" : "false") + ",\n" +
            "  \"base_realtime_viable\": " +
                juce::String(base_underruns_.load() == 0 ? "true" : "false") + ",\n" +
            "  \"default_realtime_variant\": \"mrt2_small\",\n"
            "  \"model_switch_strategy\": \"serialized unload-load-prefill; source resets to the common 28-second A/B anchor\",\n"
            "  \"dual_resident_model_trial\": \"failed with SIGSEGV while loading the second MLX engine\",\n"
            "  \"peak_output\": " + juce::String(peak_output_.load(), 6) + ",\n" +
            "  \"peak_generated\": " + juce::String(peak_generated_.load(), 6) + ",\n" +
            "  \"peak_source_anchor\": " + juce::String(peak_anchor_.load(), 6) + ",\n" +
            "  \"source_rms_dbfs\": " + juce::String(level_to_db(static_cast<float>(source_rms)), 3) + ",\n" +
            "  \"generated_rms_dbfs\": " + juce::String(level_to_db(static_cast<float>(generated_rms)), 3) + ",\n" +
            "  \"limited_samples\": " + juce::String(limited_samples_.load()) + ",\n" +
            "  \"max_world_depth\": " + juce::String(max_world_depth_.load(), 6) + ",\n" +
            "  \"final_world_depth\": " + juce::String(current_world_depth_.load(), 6) + ",\n" +
            "  \"world_depth_updates\": " + juce::String(world_depth_updates_) + ",\n" +
            "  \"diagnostic_steps\": " + juce::String(diagnostic_steps_) + ",\n" +
            "  \"semantic_slider_updates\": " + juce::String(semantic_slider_updates_) + ",\n" +
            "  \"conditioning_sequence\": " + juce::String(conditioning_sequence_) + ",\n" +
            "  \"monitor_frames\": {\"original\":" +
                juce::String(monitor_mode_frames_[0].load()) + ",\"generated\":" +
                juce::String(monitor_mode_frames_[1].load()) + ",\"hybrid\":" +
                juce::String(monitor_mode_frames_[2].load()) + "},\n" +
            "  \"quantized_controls_scheduled\": " +
                juce::String(quantized_controls_scheduled_) + ",\n" +
            "  \"quantized_controls_applied\": " +
                juce::String(quantized_controls_applied_) + ",\n" +
            "  \"max_quantization_late_frames\": " +
                juce::String(max_quantization_late_frames_) + ",\n" +
            "  \"generated_onsets_observed\": " + juce::String(total_generated_onsets_.load()) + ",\n" +
            "  \"uncorrected_phase_abs_ms\": " + juce::String(uncorrected_phase_ms_, 3) + ",\n" +
            "  \"corrected_phase_abs_ms\": " + juce::String(corrected_phase_ms_, 3) + ",\n" +
            "  \"phase_confidence\": " + juce::String(phase_confidence_.load(), 6) + ",\n" +
            "  \"timing_correction_total_samples\": " + juce::String(timing_correction_total_.load()) + ",\n" +
            "  \"sync_materially_improved\": " + juce::String(sync_materially_improved ? "true" : "false") + ",\n" +
            "  \"audio_reference_requested\": " + juce::String(audio_reference_requested_.load() ? "true" : "false") + ",\n" +
            "  \"audio_reference_ready\": " + juce::String(audio_reference_ready_.load() ? "true" : "false") + ",\n" +
            "  \"small_audio_reference_status\": " + juce::String(small.audio_reference_status) + ",\n" +
            "  \"base_audio_reference_status\": " + juce::String(base.audio_reference_status) + ",\n" +
            "  \"audio_reference_declared_bpm\": 120.000,\n"
            "  \"audio_reference_tempo_pull\": {\"mrt2_small\":" +
                reference_pull_json(0) + ",\"mrt2_base\":" +
                reference_pull_json(1) + "},\n" +
            "  \"generated_tempos\": " + generated_tempos + ",\n" +
            "  \"tempo_capture_seconds_per_condition\": " +
                juce::String(kTempoCaptureSeconds) + ",\n" +
            "  \"tempo_capabilities\": {\n"
            "    \"mrt2_small\": {\"explicit_bpm_api\":false,\"bpm_via_text_prompt_only\":true,\"continuous_bpm_api\":false,\"model_phase_control\":false,\"stable_conditions_out_of_6\":" +
                juce::String(stable_condition_counts[0]) + "},\n" +
            "    \"mrt2_base\": {\"explicit_bpm_api\":false,\"bpm_via_text_prompt_only\":true,\"continuous_bpm_api\":false,\"model_phase_control\":false,\"stable_conditions_out_of_6\":" +
                juce::String(stable_condition_counts[1]) + "},\n" +
            "    \"lyria_realtime\": {\"tested\":false,\"reason\":\"no local gcloud or Gemini credentials\",\"explicit_bpm_api\":true,\"documented_bpm_range\":\"60-200\",\"smooth_continuous_bpm_changes\":false,\"bpm_change_requires_context_reset\":true,\"stable_tempo_over_time\":\"not locally measured\",\"beat_or_downbeat_phase_api\":false}\n"
            "  },\n"
            "  \"mrt2_explicit_tempo_api\": false,\n"
            "  \"tempo_strategy\": \"123 BPM text prompt plus generated onset phase follower\",\n"
            "  \"timing_strategy\": \"bar quantization plus at most four-sample-per-device-block linear resampling\",\n"
            "  \"home_output\": \"untouched_master_at_current_source_position\"\n"
            "}\n";
        smoke_report_.getParentDirectory().createDirectory();
        smoke_report_.replaceWithText(report);
        juce::JUCEApplicationBase::getInstance()->setApplicationReturnValue(passed ? 0 : 1);
        juce::MessageManager::callAsync([] {
            if (auto* app = juce::JUCEApplicationBase::getInstance())
                app->systemRequestedQuit();
        });
    }

    juce::Label title_;
    juce::Label subtitle_;
    juce::Label mode_;
    juce::Label status_primary_;
    juce::Label status_secondary_;
    juce::Label status_technical_;
    juce::TextButton play_;
    juce::TextButton enter_world_;
    juce::TextButton home_;
    juce::TextButton add_prompt_;
    juce::TextButton current_mode_;
    juce::TextButton source_owner_;
    juce::TextButton mrt2_owner_;
    juce::TextButton small_model_;
    juce::TextButton base_model_;
    juce::TextButton text_only_;
    juce::TextButton text_reference_;
    juce::TextButton load_reference_;
    juce::Slider world_depth_slider_;
    std::array<std::unique_ptr<juce::TextEditor>, kSlots> prompt_editors_;
    std::array<std::unique_ptr<juce::Slider>, kSlots> semantic_sliders_;
    std::array<std::unique_ptr<juce::TextButton>, kSlots> remove_buttons_;
    std::array<bool, kSlots> prompt_active_{};
    std::array<juce::String, kSlots> prompt_texts_{};

    juce::Rectangle<float> world_bounds_;
    juce::Rectangle<float> control_bounds_;
    const bool smoke_test_;
    const juce::File smoke_report_;
    const juce::File track_file_;
    GoldenTrack golden_track_;
    juce::AudioBuffer<float> small_audio_;
    juce::AudioBuffer<float> base_audio_;
    song_world::SourceAnchorMixer source_analyser_{kSampleRate};
    song_world::SourceAnchorMixer generated_analyser_{kSampleRate};
    std::unique_ptr<song_world::GenerativeEngine> small_engine_;
    std::unique_ptr<song_world::GenerativeEngine> base_engine_;
    std::atomic<song_world::GenerativeEngine*> active_engine_{nullptr};
    std::thread model_loader_;
    std::unique_ptr<juce::FileChooser> reference_chooser_;

    std::atomic<bool> engine_ready_{false};
    std::atomic<bool> small_engine_ready_{false};
    std::atomic<bool> base_engine_ready_{false};
    std::atomic<bool> base_model_available_{false};
    std::atomic<bool> track_ready_{false};
    std::atomic<int> engine_state_{0};
    std::atomic<bool> playing_{false};
    std::atomic<double> actual_sample_rate_{kSampleRate};
    std::atomic<std::uint64_t> source_position_{0};
    std::atomic<std::uint64_t> generated_frames_consumed_{0};
    std::atomic<int> monitor_mode_{static_cast<int>(MonitorMode::hybrid)};
    std::atomic<int> active_generator_{static_cast<int>(GeneratorVariant::small)};
    std::atomic<float> target_world_depth_{0.0F};
    std::atomic<float> current_world_depth_{0.0F};
    std::atomic<float> max_world_depth_{0.0F};
    std::atomic<float> target_bass_gain_{kMixProfiles[0].bass};
    std::atomic<float> target_hook_gain_{kMixProfiles[0].hook};
    std::atomic<float> target_air_gain_{kMixProfiles[0].air};
    std::atomic<float> target_generation_gain_{kMixProfiles[0].generation};
    std::atomic<float> current_bass_gain_{kMixProfiles[0].bass};
    std::atomic<float> current_hook_gain_{kMixProfiles[0].hook};
    std::atomic<float> current_air_gain_{kMixProfiles[0].air};
    std::atomic<float> current_generation_gain_{kMixProfiles[0].generation};
    std::atomic<float> peak_output_{0.0F};
    std::atomic<float> peak_generated_{0.0F};
    std::atomic<float> peak_anchor_{0.0F};
    std::atomic<float> source_level_{0.0F};
    std::atomic<float> generated_level_{0.0F};
    std::atomic<double> source_square_total_{0.0};
    std::atomic<double> generated_square_total_{0.0};
    std::atomic<std::uint64_t> level_samples_{0};
    std::atomic<std::uint64_t> audio_callbacks_{0};
    std::atomic<std::uint64_t> world_reads_{0};
    std::atomic<std::uint64_t> world_underruns_{0};
    std::atomic<std::uint64_t> small_underruns_{0};
    std::atomic<std::uint64_t> base_underruns_{0};
    std::atomic<std::uint64_t> limited_samples_{0};
    std::atomic<std::uint64_t> world_depth_active_frames_{0};
    std::atomic<std::size_t> min_buffer_available_{
        std::numeric_limits<std::size_t>::max()};
    std::atomic<std::size_t> min_small_buffer_available_{
        std::numeric_limits<std::size_t>::max()};
    std::atomic<std::size_t> min_base_buffer_available_{
        std::numeric_limits<std::size_t>::max()};
    std::array<std::atomic<std::uint64_t>, 3> monitor_mode_frames_{};

    std::array<float, kSlots> target_semantic_weights_{1.0F, 0.0F, 0.0F,
                                                       0.0F, 0.0F, 0.0F};
    std::array<float, kSlots> current_semantic_weights_{1.0F, 0.0F, 0.0F,
                                                        0.0F, 0.0F, 0.0F};
    std::array<float, kSlots> requested_semantic_weights_{1.0F, 0.0F, 0.0F,
                                                          0.0F, 0.0F, 0.0F};
    MixProfile requested_mix_profile_{kMixProfiles[0]};
    float requested_world_depth_{0.0F};
    std::uint64_t pending_control_sample_{0};
    std::uint64_t max_quantization_late_frames_{0};
    bool pending_style_change_{false};
    bool pending_depth_change_{false};
    bool pending_reference_change_{false};
    bool requested_reference_conditioning_{false};
    std::atomic<bool> reference_conditioning_{false};
    std::atomic<bool> audio_reference_requested_{false};
    std::atomic<bool> audio_reference_ready_{false};
    juce::String audio_reference_name_;
    juce::String audio_reference_filename_;
    std::vector<float> audio_reference_samples_;
    song_world::EngineTelemetry small_telemetry_snapshot_{};
    song_world::EngineTelemetry base_telemetry_snapshot_{};
    float generated_bass_envelope_{0.0F};
    float generated_hook_envelope_{0.0F};
    float generated_air_envelope_{0.0F};
    int status_tick_{0};
    int semantic_slider_updates_{0};
    int prompt_updates_{0};
    int world_depth_updates_{0};
    int diagnostic_steps_{0};
    int quantized_controls_scheduled_{0};
    int quantized_controls_applied_{0};
    std::uint64_t conditioning_sequence_{1};
    bool ready_ui_initialized_{false};

    std::array<float, 32> generated_phase_histogram_{};
    float generated_onset_fast_{0.0F};
    float generated_onset_slow_{0.0F};
    float previous_generated_onset_{0.0F};
    float generated_onset_peak_{0.001F};
    std::uint64_t last_generated_onset_sample_{0};
    float generated_availability_{0.0F};
    std::atomic<std::uint64_t> generated_onset_count_{0};
    std::atomic<std::uint64_t> total_generated_onsets_{0};
    std::atomic<float> phase_offset_ms_{0.0F};
    std::atomic<float> phase_confidence_{0.0F};
    std::atomic<int> timing_correction_samples_{0};
    std::atomic<std::int64_t> timing_correction_total_{0};
    std::atomic<bool> sync_correction_enabled_{true};
    float uncorrected_phase_ms_{0.0F};
    float corrected_phase_ms_{0.0F};
    std::array<std::array<std::vector<float>, kTempoConditions>, 2> tempo_captures_;
    std::atomic<int> tempo_capture_condition_{-1};
    std::atomic<std::size_t> tempo_capture_position_{0};
    std::array<std::array<std::size_t, kTempoConditions>, 2>
        tempo_capture_lengths_{};
    std::size_t smoke_tempo_condition_{0};
    std::size_t smoke_tempo_model_{0};
    int smoke_tempo_phase_{0};
    bool small_variant_tested_{false};
    bool base_variant_tested_{false};

    mutable std::mutex error_mutex_;
    std::string engine_error_;
    mutable std::mutex base_error_mutex_;
    std::string base_error_;
    mutable std::mutex reference_error_mutex_;
    std::string reference_error_;
    double smoke_started_ms_{0.0};
    double smoke_stage_ms_{0.0};
    int smoke_stage_{0};
    bool smoke_finished_{false};
};

class MainWindow final : public juce::DocumentWindow {
public:
    MainWindow(bool smoke_test, const juce::File& smoke_report,
               const juce::File& track_file)
        : DocumentWindow("Song World Generator Clock Lab", juce::Colour(0xff0b1017),
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
        return "Song World Generator Clock Lab";
    }
    const juce::String getApplicationVersion() override {
        return "0.7.0-experiment";
    }
    bool moreThanOneInstanceAllowed() override { return false; }

    void initialise(const juce::String&) override {
        bool smoke_test = false;
        auto smoke_report = juce::File::getSpecialLocation(juce::File::tempDirectory)
                                .getChildFile("song-world-generator-clock-smoke.json");
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
        window_ = std::make_unique<MainWindow>(smoke_test, smoke_report, track_file);
    }
    void shutdown() override { window_.reset(); }
    void systemRequestedQuit() override { quit(); }

private:
    std::unique_ptr<MainWindow> window_;
};

}  // namespace

START_JUCE_APPLICATION(SongWorldApplication)
