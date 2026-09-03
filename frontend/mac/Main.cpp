#include <song_world/GenerativeEngine.h>
#include <song_world/SourceAnchorMixer.h>
#include <song_world/WorldTransport.h>

#include <juce_audio_utils/juce_audio_utils.h>
#include <juce_gui_extra/juce_gui_extra.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr double kSampleRate = 48000.0;
constexpr int kDeviceBufferSamples = 512;
constexpr int kPrefillSeconds = 28;
constexpr int kPrefillTailTrimSeconds = 1;

struct HybridProfile {
    song_world::SourceAnchorGains anchors;
    float generation{0.0F};
};

constexpr std::array<HybridProfile, 3> kHybridProfiles{{
    {{0.88F, 0.72F, 0.35F}, 0.28F},  // Source Orbit
    {{0.95F, 0.20F, 0.18F}, 0.58F},  // Dub Shadow
    {{0.22F, 0.32F, 0.75F}, 0.55F},  // Ambient Afterglow
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
            reader->lengthInSamples > std::numeric_limits<int>::max()) {
            error = "Golden track has an unsupported length";
            return false;
        }
        if (reader->sampleRate <= 0.0 || reader->numChannels == 0) {
            error = "Golden track has invalid audio metadata";
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
        source_sample_rate_ = reader->sampleRate;
        return true;
    }

    [[nodiscard]] int length() const noexcept { return audio_.getNumSamples(); }
    [[nodiscard]] double duration_seconds() const noexcept {
        return static_cast<double>(length()) / kSampleRate;
    }
    [[nodiscard]] const juce::String& name() const noexcept { return name_; }
    [[nodiscard]] double source_sample_rate() const noexcept { return source_sample_rate_; }

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
        const auto trimmed_tail = static_cast<std::size_t>(
            kPrefillTailTrimSeconds * kSampleRate);
        return frames > trimmed_tail ? frames - trimmed_tail : 0;
    }

    [[nodiscard]] std::vector<float> interleaved_prefill() const {
        const auto frames = prefill_frames();
        std::vector<float> interleaved(frames * 2);
        for (std::size_t frame = 0; frame < frames; ++frame) {
            interleaved[frame * 2] = audio_.getSample(0, static_cast<int>(frame));
            interleaved[frame * 2 + 1] = audio_.getSample(1, static_cast<int>(frame));
        }
        return interleaved;
    }

private:
    juce::AudioBuffer<float> audio_;
    juce::String name_;
    double source_sample_rate_{0.0};
};

class MainComponent final : public juce::AudioAppComponent,
                            private juce::Timer {
public:
    MainComponent(bool smoke_test, juce::File smoke_report, juce::File track_file)
        : smoke_test_(smoke_test), smoke_report_(std::move(smoke_report)),
          track_file_(std::move(track_file)),
          engine_(song_world::make_mrt2_generative_engine()) {
        setOpaque(true);
        setSize(920, 680);

        configure_label(title_, "SONG WORLD", 30.0F, juce::Font::bold);
        configure_label(subtitle_, "Loading golden track and source-conditioned World…",
                        15.0F, juce::Font::plain);
        subtitle_.setColour(juce::Label::textColourId, juce::Colour(0xff97a0ad));

        configure_button(play_, "Play");
        configure_button(enter_world_, "Enter World");
        configure_button(home_, "Home / Original");
        configure_button(ab_compare_, "A/B: Hybrid Anchors");
        play_.setEnabled(false);
        enter_world_.setEnabled(false);
        home_.setEnabled(false);
        ab_compare_.setEnabled(false);
        ab_compare_.setClickingTogglesState(true);
        ab_compare_.setToggleState(true, juce::dontSendNotification);

        configure_style_button(dub_, "Source Orbit", 0);
        configure_style_button(disco_, "Dub Shadow", 1);
        configure_style_button(ambient_, "Ambient Afterglow", 2);
        select_style(0);

        configure_anchor_button(bass_anchor_, "Bass / Pulse", anchor_bass_enabled_);
        configure_anchor_button(hook_anchor_, "Mid / Hook", anchor_hook_enabled_);
        configure_anchor_button(air_anchor_, "Stereo / Air", anchor_air_enabled_);

        configure_label(mode_, "HOME • ORIGINAL TRACK", 18.0F, juce::Font::bold);
        configure_label(status_, "Decoding source and loading MRT2 Small…", 14.0F,
                        juce::Font::plain);
        configure_label(debug_, "Audio: starting…", 13.0F, juce::Font::plain);
        status_.setColour(juce::Label::textColourId, juce::Colour(0xffb9c2ce));
        debug_.setColour(juce::Label::textColourId, juce::Colour(0xff788493));

        play_.onClick = [this] {
            const bool next = !playing_.load(std::memory_order_relaxed);
            playing_.store(next, std::memory_order_relaxed);
            play_.setButtonText(next ? "Pause" : "Play");
        };
        enter_world_.onClick = [this] { enter_world(); };
        home_.onClick = [this] { return_home(); };
        ab_compare_.onClick = [this] {
            set_hybrid_enabled(ab_compare_.getToggleState());
        };
        dub_.onClick = [this] { select_style(0); };
        disco_.onClick = [this] { select_style(1); };
        ambient_.onClick = [this] { select_style(2); };

        world_audio_.setSize(2, 8192, false, true, false);
        setAudioChannels(0, 2);
        auto setup = deviceManager.getAudioDeviceSetup();
        setup.sampleRate = kSampleRate;
        setup.bufferSize = kDeviceBufferSamples;
        const auto device_error = deviceManager.setAudioDeviceSetup(setup, true);
        if (device_error.isNotEmpty()) {
            set_engine_error("Audio setup failed: " + device_error.toStdString());
        }

        model_loader_ = std::thread([this] { load_experience(); });
        smoke_started_ms_ = juce::Time::getMillisecondCounterHiRes();
        startTimerHz(20);
    }

    ~MainComponent() override {
        stopTimer();
        shutdownAudio();
        if (model_loader_.joinable()) model_loader_.join();
        engine_->stop();
    }

    void prepareToPlay(int, double sample_rate) override {
        actual_sample_rate_.store(sample_rate, std::memory_order_relaxed);
        anchor_mixer_.set_sample_rate(sample_rate);
    }

    void getNextAudioBlock(const juce::AudioSourceChannelInfo& output) override {
        if (output.buffer == nullptr || output.buffer->getNumChannels() < 2 ||
            output.numSamples <= 0) {
            return;
        }

        output.clearActiveBufferRegion();
        world_audio_.clear(0, 0, output.numSamples);
        world_audio_.clear(1, 0, output.numSamples);

        const bool playing = playing_.load(std::memory_order_relaxed);
        bool world_ok = false;
        if (playing && engine_ready_.load(std::memory_order_acquire) &&
            output.numSamples <= world_audio_.getNumSamples()) {
            world_ok = engine_->pull_audio(world_audio_.getWritePointer(0),
                                           world_audio_.getWritePointer(1),
                                           static_cast<std::size_t>(output.numSamples));
            world_reads_.fetch_add(1, std::memory_order_relaxed);
            if (!world_ok) world_underruns_.fetch_add(1, std::memory_order_relaxed);
        }

        auto* left = output.buffer->getWritePointer(0, output.startSample);
        auto* right = output.buffer->getWritePointer(1, output.startSample);
        const auto* world_left = world_audio_.getReadPointer(0);
        const auto* world_right = world_audio_.getReadPointer(1);
        auto position = home_position_.load(std::memory_order_relaxed);
        float mix = current_world_mix_.load(std::memory_order_relaxed);
        const float target = engine_ready_.load(std::memory_order_relaxed)
            ? target_world_mix_.load(std::memory_order_relaxed)
            : 0.0F;
        const double sample_rate = std::max(1.0, actual_sample_rate_.load(std::memory_order_relaxed));
        const float smoothing = static_cast<float>(1.0 - std::exp(-1.0 / (0.45 * sample_rate)));
        const float anchor_smoothing = static_cast<float>(
            1.0 - std::exp(-1.0 / (0.12 * sample_rate)));
        const bool hybrid_enabled = hybrid_enabled_.load(std::memory_order_relaxed);
        const float target_bass = hybrid_enabled
            ? target_anchor_bass_.load(std::memory_order_relaxed) *
                  (anchor_bass_enabled_.load(std::memory_order_relaxed) ? 1.0F : 0.0F)
            : 0.0F;
        const float target_hook = hybrid_enabled
            ? target_anchor_hook_.load(std::memory_order_relaxed) *
                  (anchor_hook_enabled_.load(std::memory_order_relaxed) ? 1.0F : 0.0F)
            : 0.0F;
        const float target_air = hybrid_enabled
            ? target_anchor_air_.load(std::memory_order_relaxed) *
                  (anchor_air_enabled_.load(std::memory_order_relaxed) ? 1.0F : 0.0F)
            : 0.0F;
        const float target_generation = hybrid_enabled
            ? target_generation_gain_.load(std::memory_order_relaxed)
            : 1.0F;
        float bass_gain = current_anchor_bass_.load(std::memory_order_relaxed);
        float hook_gain = current_anchor_hook_.load(std::memory_order_relaxed);
        float air_gain = current_anchor_air_.load(std::memory_order_relaxed);
        float generation_gain = current_generation_gain_.load(std::memory_order_relaxed);
        float output_peak = 0.0F;
        float world_peak = 0.0F;
        float anchor_peak = 0.0F;
        std::uint64_t limited_samples = 0;
        std::uint64_t anchor_frames = 0;
        std::uint64_t hybrid_frames = 0;
        std::uint64_t prefill_only_frames = 0;
        for (int sample = 0; sample < output.numSamples; ++sample) {
            mix += (target - mix) * smoothing;
            bass_gain += (target_bass - bass_gain) * anchor_smoothing;
            hook_gain += (target_hook - hook_gain) * anchor_smoothing;
            air_gain += (target_air - air_gain) * anchor_smoothing;
            generation_gain +=
                (target_generation - generation_gain) * anchor_smoothing;
            const float generated_left = world_ok ? world_left[sample] : 0.0F;
            const float generated_right = world_ok ? world_right[sample] : 0.0F;
            world_peak = std::max(world_peak,
                                  std::max(std::abs(generated_left), std::abs(generated_right)));
            if (playing) {
                const float original_left = golden_track_.sample(0, position);
                const float original_right = golden_track_.sample(1, position);
                const auto anchor = anchor_mixer_.process(
                    original_left, original_right,
                    {.bass = bass_gain, .hook = hook_gain, .air = air_gain});
                anchor_peak = std::max(
                    anchor_peak, std::max(std::abs(anchor.left), std::abs(anchor.right)));
                if (mix > 0.5F &&
                    (std::abs(anchor.left) > 1.0e-5F ||
                     std::abs(anchor.right) > 1.0e-5F)) {
                    ++anchor_frames;
                }
                if (mix > 0.5F) {
                    if (hybrid_enabled) {
                        ++hybrid_frames;
                    } else {
                        ++prefill_only_frames;
                    }
                }
                const float hybrid_left =
                    anchor.left * 0.78F + generated_left * generation_gain;
                const float hybrid_right =
                    anchor.right * 0.78F + generated_right * generation_gain;
                const float mixed_left = original_left * (1.0F - mix) + hybrid_left * mix;
                const float mixed_right = original_right * (1.0F - mix) + hybrid_right * mix;
                limited_samples += std::abs(mixed_left) > 1.0F ? 1 : 0;
                limited_samples += std::abs(mixed_right) > 1.0F ? 1 : 0;
                left[sample] = std::clamp(mixed_left, -1.0F, 1.0F);
                right[sample] = std::clamp(mixed_right, -1.0F, 1.0F);
                output_peak = std::max(output_peak,
                                       std::max(std::abs(left[sample]), std::abs(right[sample])));
                ++position;
            }
        }

        home_position_.store(position, std::memory_order_relaxed);
        current_world_mix_.store(mix, std::memory_order_relaxed);
        current_anchor_bass_.store(bass_gain, std::memory_order_relaxed);
        current_anchor_hook_.store(hook_gain, std::memory_order_relaxed);
        current_anchor_air_.store(air_gain, std::memory_order_relaxed);
        current_generation_gain_.store(generation_gain, std::memory_order_relaxed);
        update_atomic_max(max_world_mix_, mix);
        update_atomic_max(peak_output_, output_peak);
        update_atomic_max(peak_world_, world_peak);
        update_atomic_max(peak_anchor_, anchor_peak);
        source_anchor_frames_.fetch_add(anchor_frames, std::memory_order_relaxed);
        hybrid_frames_.fetch_add(hybrid_frames, std::memory_order_relaxed);
        prefill_only_frames_.fetch_add(prefill_only_frames, std::memory_order_relaxed);
        limited_samples_.fetch_add(limited_samples, std::memory_order_relaxed);
        audio_callbacks_.fetch_add(1, std::memory_order_relaxed);
    }

    void releaseResources() override {}

    void paint(juce::Graphics& graphics) override {
        graphics.fillAll(juce::Colour(0xff0b0f14));
        auto body = getLocalBounds().toFloat().reduced(24.0F);
        graphics.setColour(juce::Colour(0xff151b23));
        graphics.fillRoundedRectangle(body.withTrimmedTop(96.0F), 18.0F);

        auto world_panel = juce::Rectangle<float>(body.getX() + 24.0F,
                                                  body.getY() + 245.0F,
                                                  body.getWidth() - 48.0F,
                                                  230.0F);
        graphics.setColour(juce::Colour(0xff10161e));
        graphics.fillRoundedRectangle(world_panel, 14.0F);
        graphics.setColour(juce::Colour(0xff273140));
        graphics.drawRoundedRectangle(world_panel, 14.0F, 1.0F);

        graphics.setColour(juce::Colour(0xff788493));
        graphics.setFont(juce::Font(juce::FontOptions(12.0F, juce::Font::bold)));
        graphics.drawText("SEMANTIC DESTINATIONS",
                          world_panel.toNearestInt().removeFromTop(30).reduced(18, 0),
                          juce::Justification::centredLeft);
        auto anchor_title = world_panel.toNearestInt().removeFromBottom(76).removeFromTop(24);
        graphics.drawText("SOURCE ANCHORS • REAL-TIME MULTIBAND PROXY",
                          anchor_title.reduced(18, 0),
                          juce::Justification::centredLeft);
    }

    void resized() override {
        auto area = getLocalBounds().reduced(48);
        title_.setBounds(area.removeFromTop(42));
        subtitle_.setBounds(area.removeFromTop(32));
        area.removeFromTop(34);

        auto transport = area.removeFromTop(58);
        play_.setBounds(transport.removeFromLeft(150).reduced(2));
        transport.removeFromLeft(12);
        enter_world_.setBounds(transport.removeFromLeft(190).reduced(2));
        transport.removeFromLeft(12);
        home_.setBounds(transport.removeFromLeft(190).reduced(2));
        transport.removeFromLeft(12);
        ab_compare_.setBounds(transport.removeFromLeft(220).reduced(2));

        area.removeFromTop(32);
        auto world_area = area.removeFromTop(230);
        auto style_area = world_area.removeFromTop(150).reduced(18, 40);
        const int node_width = (style_area.getWidth() - 24) / 3;
        dub_.setBounds(style_area.removeFromLeft(node_width));
        style_area.removeFromLeft(12);
        disco_.setBounds(style_area.removeFromLeft(node_width));
        style_area.removeFromLeft(12);
        ambient_.setBounds(style_area);

        auto anchor_area = world_area.removeFromTop(68).reduced(18, 9);
        const int anchor_width = (anchor_area.getWidth() - 24) / 3;
        bass_anchor_.setBounds(anchor_area.removeFromLeft(anchor_width));
        anchor_area.removeFromLeft(12);
        hook_anchor_.setBounds(anchor_area.removeFromLeft(anchor_width));
        anchor_area.removeFromLeft(12);
        air_anchor_.setBounds(anchor_area);

        area.removeFromTop(20);
        mode_.setBounds(area.removeFromTop(32));
        status_.setBounds(area.removeFromTop(27));
        debug_.setBounds(area.removeFromTop(25));
    }

private:
    static void configure_label(juce::Label& label, const juce::String& text,
                                float size, int style) {
        label.setText(text, juce::dontSendNotification);
        label.setFont(juce::Font(juce::FontOptions(size, style)));
        label.setColour(juce::Label::textColourId, juce::Colour(0xffeef2f7));
        label.setInterceptsMouseClicks(false, false);
    }

    void configure_button(juce::TextButton& button, const juce::String& text) {
        button.setButtonText(text);
        button.setColour(juce::TextButton::buttonColourId, juce::Colour(0xff26303d));
        button.setColour(juce::TextButton::buttonOnColourId, juce::Colour(0xfff05a3c));
        button.setColour(juce::TextButton::textColourOffId, juce::Colour(0xffeef2f7));
        addAndMakeVisible(button);
    }

    void configure_style_button(juce::TextButton& button, const juce::String& text,
                                int index) {
        configure_button(button, text);
        button.setClickingTogglesState(true);
        button.setRadioGroupId(1300);
        button.setTooltip(index == 0 ? "Closest semantic orbit around the source context"
                                    : index == 1 ? "Source-seeded continuation steered toward dub space"
                                                 : "Source-seeded continuation steered toward open ambience");
        button.setEnabled(false);
    }

    void configure_anchor_button(juce::TextButton& button, const juce::String& text,
                                 std::atomic<bool>& enabled_state) {
        configure_button(button, text);
        button.setClickingTogglesState(true);
        button.setToggleState(true, juce::dontSendNotification);
        button.setTooltip("Toggle this source-derived anchor inside World mode");
        auto* state = &enabled_state;
        button.onClick = [&button, state] {
            state->store(button.getToggleState(), std::memory_order_relaxed);
        };
        button.setEnabled(false);
    }

    void select_style(int index) {
        selected_style_ = std::clamp(index, 0, 2);
        dub_.setToggleState(selected_style_ == 0, juce::dontSendNotification);
        disco_.setToggleState(selected_style_ == 1, juce::dontSendNotification);
        ambient_.setToggleState(selected_style_ == 2, juce::dontSendNotification);
        target_style_weights_ = {0.0F, 0.0F, 0.0F};
        target_style_weights_[static_cast<std::size_t>(selected_style_)] = 1.0F;
        ++style_changes_;
    }

    void enter_world() {
        if (!engine_ready_.load(std::memory_order_acquire)) return;
        in_world_ = true;
        target_world_mix_.store(1.0F, std::memory_order_relaxed);
        enter_world_.setButtonText("In World");
        update_world_mode_label();
    }

    void return_home() {
        in_world_ = false;
        target_world_mix_.store(0.0F, std::memory_order_relaxed);
        enter_world_.setButtonText("Enter World");
        mode_.setText("HOME • ORIGINAL TRACK", juce::dontSendNotification);
    }

    void set_hybrid_enabled(bool enabled) {
        hybrid_enabled_.store(enabled, std::memory_order_relaxed);
        ab_compare_.setToggleState(enabled, juce::dontSendNotification);
        ab_compare_.setButtonText(enabled ? "A/B: Hybrid Anchors"
                                          : "A/B: Prefill Only");
        if (in_world_) update_world_mode_label();
    }

    void update_world_mode_label() {
        mode_.setText(hybrid_enabled_.load(std::memory_order_relaxed)
                          ? "WORLD • SOURCE ANCHORS + MRT2"
                          : "WORLD • PREFILL-ONLY MRT2",
                      juce::dontSendNotification);
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
        home_position_.store(golden_track_.handoff_anchor(),
                             std::memory_order_relaxed);

        const auto asset_root = juce::File::getSpecialLocation(
                                    juce::File::userDocumentsDirectory)
                                    .getChildFile("Magenta")
                                    .getChildFile("magenta-rt-v2");
        const auto model = asset_root.getChildFile("models")
                               .getChildFile("mrt2_small")
                               .getChildFile("mrt2_small.mlxfn");
        const auto resources = asset_root.getChildFile("resources");
        const auto prefill_model = resources.getChildFile("spectrostream")
                                       .getChildFile("spectrostream_encoder.mlxfn");
        if (!model.existsAsFile() || !resources.isDirectory() ||
            !prefill_model.existsAsFile()) {
            set_engine_error("Missing MRT2 assets under " + asset_root.getFullPathName().toStdString());
            return;
        }

        song_world::EngineConfig config{
            .model_path = model.getFullPathName().toStdString(),
            .resource_directory = resources.getFullPathName().toStdString(),
            .prefill_model_path = prefill_model.getFullPathName().toStdString(),
            .prompt_a = "disco house groove, melodic bass, bright strings, dance music",
            .prompt_b = "deep dub house, spacious echoes, restrained drums, dance music",
            .prompt_c = "airy ambient disco, soft percussion, luminous synthesizers",
            .ring_buffer_samples = 4096,
            .output_gain_db = -6.0F,
        };
        if (!engine_->prepare(config, error)) {
            set_engine_error(error);
            return;
        }
        engine_->set_conditioning({
            .x = 0.0F,
            .y = 0.5F,
            .style_a = 1.0F,
            .style_b = 0.0F,
            .style_c = 0.0F,
            .sequence = 1,
        });
        auto prefill_audio = golden_track_.interleaved_prefill();
        if (!engine_->prefill_source(prefill_audio.data(),
                                     golden_track_.prefill_frames(), error)) {
            set_engine_error(error);
            return;
        }
        home_position_.store(golden_track_.handoff_anchor(),
                             std::memory_order_relaxed);
        engine_->start();
        engine_ready_.store(true, std::memory_order_release);
        engine_state_.store(1, std::memory_order_release);
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

    void timerCallback() override {
        const int state = engine_state_.load(std::memory_order_acquire);
        if (state == 1) {
            play_.setEnabled(true);
            enter_world_.setEnabled(true);
            home_.setEnabled(true);
            ab_compare_.setEnabled(true);
            dub_.setEnabled(true);
            disco_.setEnabled(true);
            ambient_.setEnabled(true);
            bass_anchor_.setEnabled(true);
            hook_anchor_.setEnabled(true);
            air_anchor_.setEnabled(true);
            if (!ready_ui_initialized_) {
                ready_ui_initialized_ = true;
                subtitle_.setText(golden_track_.name() + " • source context → parallel World",
                                  juce::dontSendNotification);
            }
            for (std::size_t index = 0; index < current_style_weights_.size(); ++index) {
                current_style_weights_[index] +=
                    (target_style_weights_[index] - current_style_weights_[index]) * 0.16F;
            }
            song_world::SourceAnchorGains anchor_targets;
            float generation_target = 0.0F;
            for (std::size_t index = 0; index < current_style_weights_.size(); ++index) {
                const float weight = current_style_weights_[index];
                anchor_targets.bass += kHybridProfiles[index].anchors.bass * weight;
                anchor_targets.hook += kHybridProfiles[index].anchors.hook * weight;
                anchor_targets.air += kHybridProfiles[index].anchors.air * weight;
                generation_target += kHybridProfiles[index].generation * weight;
            }
            target_anchor_bass_.store(anchor_targets.bass, std::memory_order_relaxed);
            target_anchor_hook_.store(anchor_targets.hook, std::memory_order_relaxed);
            target_anchor_air_.store(anchor_targets.air, std::memory_order_relaxed);
            target_generation_gain_.store(generation_target, std::memory_order_relaxed);
            engine_->set_conditioning({
                .x = current_style_weights_[1],
                .y = current_style_weights_[2],
                .style_a = current_style_weights_[0],
                .style_b = current_style_weights_[1],
                .style_c = current_style_weights_[2],
                .sequence = ++conditioning_sequence_,
            });

            if (++status_tick_ >= 5) {
                status_tick_ = 0;
                const auto telemetry = engine_->telemetry();
                const double buffered_ms = telemetry.buffer_available_samples / kSampleRate * 1000.0;
                status_.setText("Source prefill " +
                                    juce::String(telemetry.source_prefill_ms / 1000.0, 1) +
                                    " s • MRT2 " +
                                    juce::String(telemetry.frame_total_ms, 1) +
                                    " ms/frame • " + juce::String(buffered_ms, 0) +
                                    " ms buffered",
                                juce::dontSendNotification);
                const double source_seconds =
                    static_cast<double>(home_position_.load(std::memory_order_relaxed)) /
                    kSampleRate;
                debug_.setText("Source " + format_time(source_seconds) +
                                   " • anchors " +
                                   juce::String(current_anchor_bass_.load(), 2) + "/" +
                                   juce::String(current_anchor_hook_.load(), 2) + "/" +
                                   juce::String(current_anchor_air_.load(), 2) +
                                   " • underruns " +
                                   juce::String(world_underruns_.load(std::memory_order_relaxed)),
                               juce::dontSendNotification);
            }
        } else if (state < 0) {
            const bool source_ready = track_ready_.load(std::memory_order_acquire);
            play_.setEnabled(source_ready);
            home_.setEnabled(source_ready);
            status_.setText("Engine unavailable: " + juce::String(engine_error()),
                            juce::dontSendNotification);
            debug_.setText("Home track remains usable; Enter World is disabled.",
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

    void advance_smoke_test() {
        const double now = juce::Time::getMillisecondCounterHiRes();
        if (now - smoke_started_ms_ > 180000.0) {
            finish_smoke_test(false);
            return;
        }
        if (engine_state_.load(std::memory_order_acquire) < 0) {
            finish_smoke_test(false);
            return;
        }
        if (smoke_stage_ == 0 && engine_ready_.load(std::memory_order_acquire)) {
            playing_.store(true, std::memory_order_relaxed);
            play_.setButtonText("Pause");
            smoke_stage_ = 1;
            smoke_stage_ms_ = now;
        } else if (smoke_stage_ == 1 && now - smoke_stage_ms_ > 2000.0) {
            select_style(0);
            enter_world();
            smoke_stage_ = 2;
            smoke_stage_ms_ = now;
        } else if (smoke_stage_ == 2 && now - smoke_stage_ms_ > 3000.0) {
            set_hybrid_enabled(false);
            select_style(1);
            smoke_stage_ = 3;
            smoke_stage_ms_ = now;
        } else if (smoke_stage_ == 3 && now - smoke_stage_ms_ > 3000.0) {
            set_hybrid_enabled(true);
            select_style(2);
            smoke_stage_ = 4;
            smoke_stage_ms_ = now;
        } else if (smoke_stage_ == 4 && now - smoke_stage_ms_ > 3000.0) {
            return_home();
            smoke_stage_ = 5;
            smoke_stage_ms_ = now;
        } else if (smoke_stage_ == 5 && now - smoke_stage_ms_ > 4000.0) {
            const auto telemetry = engine_->telemetry();
            const bool passed = engine_ready_.load(std::memory_order_relaxed) &&
                                track_ready_.load(std::memory_order_relaxed) &&
                                telemetry.source_prefilled &&
                                audio_callbacks_.load(std::memory_order_relaxed) > 0 &&
                                world_reads_.load(std::memory_order_relaxed) > 0 &&
                                world_underruns_.load(std::memory_order_relaxed) == 0 &&
                                peak_output_.load(std::memory_order_relaxed) > 0.01F &&
                                peak_world_.load(std::memory_order_relaxed) > 0.001F &&
                                peak_anchor_.load(std::memory_order_relaxed) > 0.001F &&
                                source_anchor_frames_.load(std::memory_order_relaxed) > 0 &&
                                hybrid_frames_.load(std::memory_order_relaxed) > 0 &&
                                prefill_only_frames_.load(std::memory_order_relaxed) > 0 &&
                                max_world_mix_.load(std::memory_order_relaxed) > 0.95F &&
                                current_world_mix_.load(std::memory_order_relaxed) < 0.05F &&
                                home_position_.load(std::memory_order_relaxed) >
                                    golden_track_.handoff_anchor() +
                                        static_cast<std::uint64_t>(12.0 * kSampleRate) &&
                                style_changes_ >= 3;
            finish_smoke_test(passed);
        }
    }

    void finish_smoke_test(bool passed) {
        if (smoke_finished_) return;
        smoke_finished_ = true;
        const auto telemetry = engine_->telemetry();
        const juce::String report =
            "{\n"
            "  \"passed\": " + juce::String(passed ? "true" : "false") + ",\n" +
            "  \"engine_ready\": " + juce::String(engine_ready_.load() ? "true" : "false") + ",\n" +
            "  \"track_ready\": " + juce::String(track_ready_.load() ? "true" : "false") + ",\n" +
            "  \"source_prefilled\": " + juce::String(telemetry.source_prefilled ? "true" : "false") + ",\n" +
            "  \"source_prefill_ms\": " + juce::String(telemetry.source_prefill_ms, 3) + ",\n" +
            "  \"source_prefill_frames\": " + juce::String(static_cast<juce::int64>(telemetry.source_prefill_frames)) + ",\n" +
            "  \"model_transformer_ms\": " + juce::String(telemetry.transformer_ms, 3) + ",\n" +
            "  \"model_frame_total_ms\": " + juce::String(telemetry.frame_total_ms, 3) + ",\n" +
            "  \"buffer_available_samples\": " + juce::String(static_cast<juce::int64>(telemetry.buffer_available_samples)) + ",\n" +
            "  \"device_buffer_samples\": " + juce::String(kDeviceBufferSamples) + ",\n" +
            "  \"handoff_anchor_seconds\": " + juce::String(static_cast<double>(golden_track_.handoff_anchor()) / kSampleRate, 3) + ",\n" +
            "  \"final_source_seconds\": " + juce::String(static_cast<double>(home_position_.load()) / kSampleRate, 3) + ",\n" +
            "  \"played_source_seconds\": " + juce::String(static_cast<double>(home_position_.load() - golden_track_.handoff_anchor()) / kSampleRate, 3) + ",\n" +
            "  \"audio_callbacks\": " + juce::String(audio_callbacks_.load()) + ",\n" +
            "  \"world_reads\": " + juce::String(world_reads_.load()) + ",\n" +
            "  \"world_underruns\": " + juce::String(world_underruns_.load()) + ",\n" +
            "  \"peak_output\": " + juce::String(peak_output_.load(), 6) + ",\n" +
            "  \"peak_world\": " + juce::String(peak_world_.load(), 6) + ",\n" +
            "  \"peak_source_anchor\": " + juce::String(peak_anchor_.load(), 6) + ",\n" +
            "  \"source_anchor_frames\": " + juce::String(source_anchor_frames_.load()) + ",\n" +
            "  \"hybrid_ab_frames\": " + juce::String(hybrid_frames_.load()) + ",\n" +
            "  \"prefill_only_ab_frames\": " + juce::String(prefill_only_frames_.load()) + ",\n" +
            "  \"source_anchor_method\": \"realtime_multiband_proxy\",\n" +
            "  \"limited_samples\": " + juce::String(limited_samples_.load()) + ",\n" +
            "  \"max_world_mix\": " + juce::String(max_world_mix_.load(), 6) + ",\n" +
            "  \"final_world_mix\": " + juce::String(current_world_mix_.load(), 6) + ",\n" +
            "  \"style_changes\": " + juce::String(style_changes_) + ",\n" +
            "  \"conditioning_sequence\": " + juce::String(conditioning_sequence_) + "\n"
            "}\n";
        smoke_report_.getParentDirectory().createDirectory();
        smoke_report_.replaceWithText(report);
        juce::JUCEApplicationBase::getInstance()->setApplicationReturnValue(passed ? 0 : 1);
        juce::MessageManager::callAsync([] {
            if (auto* app = juce::JUCEApplicationBase::getInstance()) app->systemRequestedQuit();
        });
    }

    juce::Label title_;
    juce::Label subtitle_;
    juce::Label mode_;
    juce::Label status_;
    juce::Label debug_;
    juce::TextButton play_;
    juce::TextButton enter_world_;
    juce::TextButton home_;
    juce::TextButton ab_compare_;
    juce::TextButton dub_;
    juce::TextButton disco_;
    juce::TextButton ambient_;
    juce::TextButton bass_anchor_;
    juce::TextButton hook_anchor_;
    juce::TextButton air_anchor_;

    const bool smoke_test_;
    const juce::File smoke_report_;
    const juce::File track_file_;
    GoldenTrack golden_track_;
    juce::AudioBuffer<float> world_audio_;
    song_world::SourceAnchorMixer anchor_mixer_{kSampleRate};
    std::unique_ptr<song_world::GenerativeEngine> engine_;
    std::thread model_loader_;

    std::atomic<bool> engine_ready_{false};
    std::atomic<bool> track_ready_{false};
    std::atomic<int> engine_state_{0};
    std::atomic<bool> playing_{false};
    std::atomic<float> target_world_mix_{0.0F};
    std::atomic<float> current_world_mix_{0.0F};
    std::atomic<float> max_world_mix_{0.0F};
    std::atomic<float> peak_output_{0.0F};
    std::atomic<float> peak_world_{0.0F};
    std::atomic<float> peak_anchor_{0.0F};
    std::atomic<float> target_anchor_bass_{kHybridProfiles[0].anchors.bass};
    std::atomic<float> target_anchor_hook_{kHybridProfiles[0].anchors.hook};
    std::atomic<float> target_anchor_air_{kHybridProfiles[0].anchors.air};
    std::atomic<float> target_generation_gain_{kHybridProfiles[0].generation};
    std::atomic<float> current_anchor_bass_{kHybridProfiles[0].anchors.bass};
    std::atomic<float> current_anchor_hook_{kHybridProfiles[0].anchors.hook};
    std::atomic<float> current_anchor_air_{kHybridProfiles[0].anchors.air};
    std::atomic<float> current_generation_gain_{kHybridProfiles[0].generation};
    std::atomic<bool> anchor_bass_enabled_{true};
    std::atomic<bool> anchor_hook_enabled_{true};
    std::atomic<bool> anchor_air_enabled_{true};
    std::atomic<bool> hybrid_enabled_{true};
    std::atomic<double> actual_sample_rate_{kSampleRate};
    std::atomic<std::uint64_t> home_position_{0};
    std::atomic<std::uint64_t> audio_callbacks_{0};
    std::atomic<std::uint64_t> world_reads_{0};
    std::atomic<std::uint64_t> world_underruns_{0};
    std::atomic<std::uint64_t> source_anchor_frames_{0};
    std::atomic<std::uint64_t> hybrid_frames_{0};
    std::atomic<std::uint64_t> prefill_only_frames_{0};
    std::atomic<std::uint64_t> limited_samples_{0};

    std::array<float, 3> current_style_weights_{1.0F, 0.0F, 0.0F};
    std::array<float, 3> target_style_weights_{1.0F, 0.0F, 0.0F};
    int selected_style_{0};
    int style_changes_{0};
    int status_tick_{0};
    std::uint64_t conditioning_sequence_{1};
    bool in_world_{false};
    bool ready_ui_initialized_{false};

    mutable std::mutex error_mutex_;
    std::string engine_error_;

    double smoke_started_ms_{0.0};
    double smoke_stage_ms_{0.0};
    int smoke_stage_{0};
    bool smoke_finished_{false};
};

class MainWindow final : public juce::DocumentWindow {
public:
    MainWindow(bool smoke_test, const juce::File& smoke_report,
               const juce::File& track_file)
        : DocumentWindow("Song World Hybrid Anchors",
                         juce::Colour(0xff0b0f14),
                         juce::DocumentWindow::allButtons) {
        setUsingNativeTitleBar(true);
        setResizable(true, true);
        setResizeLimits(760, 620, 1200, 900);
        setContentOwned(new MainComponent(smoke_test, smoke_report, track_file), true);
        centreWithSize(getWidth(), getHeight());
        setVisible(true);
    }

    void closeButtonPressed() override {
        if (auto* app = juce::JUCEApplicationBase::getInstance()) app->systemRequestedQuit();
    }
};

class SongWorldApplication final : public juce::JUCEApplication {
public:
    const juce::String getApplicationName() override { return "Song World Hybrid Anchors"; }
    const juce::String getApplicationVersion() override { return "0.3.0-experiment"; }
    bool moreThanOneInstanceAllowed() override { return false; }

    void initialise(const juce::String&) override {
        bool smoke_test = false;
        auto smoke_report = juce::File::getSpecialLocation(juce::File::tempDirectory)
                                .getChildFile("song-world-app-smoke.json");
        const auto executable = juce::File::getSpecialLocation(
            juce::File::currentExecutableFile);
        auto track_file = executable.getParentDirectory()
                              .getParentDirectory()
                              .getChildFile("Resources")
                              .getChildFile("golden-track.mp3");
        for (const auto& argument : getCommandLineParameterArray()) {
            if (argument == "--smoke-test") smoke_test = true;
            if (argument.startsWith("--smoke-report=")) {
                smoke_report = juce::File(argument.fromFirstOccurrenceOf("=", false, false));
            }
            if (argument.startsWith("--track=")) {
                track_file = juce::File(argument.fromFirstOccurrenceOf("=", false, false));
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
