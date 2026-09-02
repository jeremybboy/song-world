#include <song_world/GenerativeEngine.h>
#include <song_world/WorldTransport.h>

#include <juce_audio_utils/juce_audio_utils.h>
#include <juce_gui_extra/juce_gui_extra.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace {

constexpr double kSampleRate = 48000.0;
constexpr int kDeviceBufferSamples = 512;
constexpr double kPi = 3.14159265358979323846;

void update_atomic_max(std::atomic<float>& destination, float value) {
    float previous = destination.load(std::memory_order_relaxed);
    while (value > previous &&
           !destination.compare_exchange_weak(previous, value,
                                              std::memory_order_relaxed)) {}
}

class TestTrack {
public:
    TestTrack() : audio_(2, static_cast<int>(kSampleRate * 16.0)) { generate(); }

    [[nodiscard]] int length() const noexcept { return audio_.getNumSamples(); }

    [[nodiscard]] float sample(int channel, std::uint64_t position) const noexcept {
        const auto index = static_cast<int>(position % static_cast<std::uint64_t>(length()));
        return audio_.getSample(channel, index);
    }

private:
    static double midi_to_hz(int note) {
        return 440.0 * std::pow(2.0, (static_cast<double>(note) - 69.0) / 12.0);
    }

    void generate() {
        constexpr double bpm = 120.0;
        constexpr double beat_seconds = 60.0 / bpm;
        constexpr std::array<int, 4> bass_notes{38, 34, 41, 36};  // D, Bb, F, C
        constexpr int chords[4][4]{{50, 53, 57, 60},
                                   {46, 50, 53, 57},
                                   {53, 57, 60, 64},
                                   {48, 52, 55, 62}};

        std::uint32_t random_state = 0x51a7c0deU;
        float previous_noise = 0.0F;
        for (int index = 0; index < audio_.getNumSamples(); ++index) {
            random_state = random_state * 1664525U + 1013904223U;
            const float noise = static_cast<float>(
                (static_cast<double>(random_state) / 4294967295.0) * 2.0 - 1.0);
            const float bright_noise = noise - previous_noise * 0.82F;
            previous_noise = noise;

            const double seconds = static_cast<double>(index) / kSampleRate;
            const double beat = seconds / beat_seconds;
            const int beat_index = static_cast<int>(std::floor(beat));
            const double beat_phase_seconds =
                (beat - std::floor(beat)) * beat_seconds;
            const int beat_in_bar = beat_index % 4;
            const int chord_index = std::min(3, static_cast<int>(seconds / 4.0));
            const double chord_phase = std::fmod(seconds, 4.0);

            float kick = 0.0F;
            if (beat_phase_seconds < 0.24) {
                const double t = beat_phase_seconds;
                const double phase = 2.0 * kPi * (48.0 * t + 0.9 * (1.0 - std::exp(-18.0 * t)));
                kick = static_cast<float>(0.62 * std::sin(phase) * std::exp(-15.0 * t));
            }

            float snare = 0.0F;
            if ((beat_in_bar == 1 || beat_in_bar == 3) && beat_phase_seconds < 0.18) {
                const double t = beat_phase_seconds;
                snare = static_cast<float>(
                    (0.20 * noise + 0.08 * std::sin(2.0 * kPi * 185.0 * t)) *
                    std::exp(-18.0 * t));
            }

            const double half_beat = beat_seconds * 0.5;
            const double hat_phase = std::fmod(seconds, half_beat);
            const float hat = hat_phase < 0.055
                ? static_cast<float>(0.055 * bright_noise * std::exp(-65.0 * hat_phase))
                : 0.0F;

            const double bass_frequency = midi_to_hz(bass_notes[chord_index]);
            const double bass_envelope = std::exp(-2.6 * beat_phase_seconds);
            const float bass = static_cast<float>(
                0.22 * bass_envelope *
                (std::sin(2.0 * kPi * bass_frequency * beat_phase_seconds) +
                 0.22 * std::sin(4.0 * kPi * bass_frequency * beat_phase_seconds)));

            const double chord_edge = std::clamp(
                std::min(chord_phase, 4.0 - chord_phase) / 0.22, 0.0, 1.0);
            float pad_left = 0.0F;
            float pad_right = 0.0F;
            for (int voice = 0; voice < 4; ++voice) {
                const double frequency = midi_to_hz(chords[chord_index][voice]);
                const double drift = 1.0 + 0.0015 * (voice - 1.5);
                pad_left += static_cast<float>(
                    std::sin(2.0 * kPi * frequency * seconds) * 0.038 * chord_edge);
                pad_right += static_cast<float>(
                    std::sin(2.0 * kPi * frequency * drift * seconds + voice * 0.21) *
                    0.038 * chord_edge);
            }

            const double eighth = beat_seconds * 0.5;
            const double pluck_phase = std::fmod(seconds + eighth * 0.5, eighth);
            const double pluck_frequency = midi_to_hz(chords[chord_index][2] + 12);
            const float pluck = pluck_phase < 0.16
                ? static_cast<float>(0.07 * std::sin(2.0 * kPi * pluck_frequency * pluck_phase) *
                                     std::exp(-22.0 * pluck_phase))
                : 0.0F;

            const float centre = kick + snare + hat + bass + pluck;
            const float left = std::tanh((centre + pad_left) * 0.88F);
            const float right = std::tanh((centre + pad_right) * 0.88F);
            audio_.setSample(0, index, left);
            audio_.setSample(1, index, right);
        }
    }

    juce::AudioBuffer<float> audio_;
};

class MainComponent final : public juce::AudioAppComponent,
                            private juce::Timer {
public:
    MainComponent(bool smoke_test, juce::File smoke_report)
        : smoke_test_(smoke_test), smoke_report_(std::move(smoke_report)),
          engine_(song_world::make_mrt2_generative_engine()) {
        setOpaque(true);
        setSize(920, 590);

        configure_label(title_, "SONG WORLD", 30.0F, juce::Font::bold);
        configure_label(subtitle_, "Local experience prototype • project-owned 8-bar test track",
                        15.0F, juce::Font::plain);
        subtitle_.setColour(juce::Label::textColourId, juce::Colour(0xff97a0ad));

        configure_button(play_, "Play");
        configure_button(enter_world_, "Enter World");
        configure_button(home_, "Home / Original");
        enter_world_.setEnabled(false);

        configure_style_button(dub_, "Deep Dub", 0);
        configure_style_button(disco_, "Warm Disco", 1);
        configure_style_button(ambient_, "Airy Ambient", 2);
        select_style(0);

        configure_label(mode_, "HOME • ORIGINAL TRACK", 18.0F, juce::Font::bold);
        configure_label(status_, "Loading MRT2 Small…", 14.0F, juce::Font::plain);
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

        model_loader_ = std::thread([this] { load_model(); });
        smoke_started_ms_ = juce::Time::getMillisecondCounterHiRes();
        if (smoke_test_) {
            playing_.store(true, std::memory_order_relaxed);
            play_.setButtonText("Pause");
        }
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
        home_position_.store(0, std::memory_order_relaxed);
    }

    void getNextAudioBlock(const juce::AudioSourceChannelInfo& output) override {
        if (output.buffer == nullptr || output.buffer->getNumChannels() < 2 ||
            output.numSamples <= 0) {
            return;
        }

        output.clearActiveBufferRegion();
        world_audio_.clear(0, 0, output.numSamples);
        world_audio_.clear(1, 0, output.numSamples);

        bool world_ok = true;
        if (engine_ready_.load(std::memory_order_acquire) &&
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
        const float smoothing = static_cast<float>(1.0 - std::exp(-1.0 / (0.20 * sample_rate)));
        float output_peak = 0.0F;
        float world_peak = 0.0F;
        const bool playing = playing_.load(std::memory_order_relaxed);

        for (int sample = 0; sample < output.numSamples; ++sample) {
            mix += (target - mix) * smoothing;
            const float generated_left = world_ok ? world_left[sample] : 0.0F;
            const float generated_right = world_ok ? world_right[sample] : 0.0F;
            world_peak = std::max(world_peak,
                                  std::max(std::abs(generated_left), std::abs(generated_right)));
            if (playing) {
                const float original_left = test_track_.sample(0, position);
                const float original_right = test_track_.sample(1, position);
                left[sample] = original_left + (generated_left - original_left) * mix;
                right[sample] = original_right + (generated_right - original_right) * mix;
                output_peak = std::max(output_peak,
                                       std::max(std::abs(left[sample]), std::abs(right[sample])));
                ++position;
            }
        }

        home_position_.store(position, std::memory_order_relaxed);
        current_world_mix_.store(mix, std::memory_order_relaxed);
        update_atomic_max(max_world_mix_, mix);
        update_atomic_max(peak_output_, output_peak);
        update_atomic_max(peak_world_, world_peak);
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
                                                  150.0F);
        graphics.setColour(juce::Colour(0xff10161e));
        graphics.fillRoundedRectangle(world_panel, 14.0F);
        graphics.setColour(juce::Colour(0xff273140));
        graphics.drawRoundedRectangle(world_panel, 14.0F, 1.0F);

        graphics.setColour(juce::Colour(0xff788493));
        graphics.setFont(juce::Font(juce::FontOptions(12.0F, juce::Font::bold)));
        graphics.drawText("SEMANTIC DESTINATIONS",
                          world_panel.toNearestInt().removeFromTop(30).reduced(18, 0),
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

        area.removeFromTop(32);
        auto style_area = area.removeFromTop(150).reduced(18, 40);
        const int node_width = (style_area.getWidth() - 24) / 3;
        dub_.setBounds(style_area.removeFromLeft(node_width));
        style_area.removeFromLeft(12);
        disco_.setBounds(style_area.removeFromLeft(node_width));
        style_area.removeFromLeft(12);
        ambient_.setBounds(style_area);

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
        button.setTooltip(index == 0 ? "Sparse dub space and dry drums"
                                    : index == 1 ? "Warm groove and acoustic drums"
                                                 : "Soft, open ambient electronics");
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
        mode_.setText("WORLD • LIVE MRT2 SMALL", juce::dontSendNotification);
    }

    void return_home() {
        in_world_ = false;
        target_world_mix_.store(0.0F, std::memory_order_relaxed);
        enter_world_.setButtonText("Enter World");
        mode_.setText("HOME • ORIGINAL TRACK", juce::dontSendNotification);
    }

    void load_model() {
        const auto asset_root = juce::File::getSpecialLocation(
                                    juce::File::userDocumentsDirectory)
                                    .getChildFile("Magenta")
                                    .getChildFile("magenta-rt-v2");
        const auto model = asset_root.getChildFile("models")
                               .getChildFile("mrt2_small")
                               .getChildFile("mrt2_small.mlxfn");
        const auto resources = asset_root.getChildFile("resources");
        if (!model.existsAsFile() || !resources.isDirectory()) {
            set_engine_error("Missing MRT2 assets under " + asset_root.getFullPathName().toStdString());
            return;
        }

        song_world::EngineConfig config{
            .model_path = model.getFullPathName().toStdString(),
            .resource_directory = resources.getFullPathName().toStdString(),
            .prompt_a = "minimal dub techno, dry drums, instrumental",
            .prompt_b = "warm disco funk, acoustic drums, instrumental",
            .prompt_c = "airy ambient electronica, soft percussion, instrumental",
            .ring_buffer_samples = 4096,
            .output_gain_db = -12.0F,
        };
        std::string error;
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
            enter_world_.setEnabled(true);
            dub_.setEnabled(true);
            disco_.setEnabled(true);
            ambient_.setEnabled(true);
            for (std::size_t index = 0; index < current_style_weights_.size(); ++index) {
                current_style_weights_[index] +=
                    (target_style_weights_[index] - current_style_weights_[index]) * 0.16F;
            }
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
                status_.setText("MRT2 Small ready • " +
                                    juce::String(telemetry.frame_total_ms, 1) +
                                    " ms/frame • " + juce::String(buffered_ms, 0) +
                                    " ms buffered",
                                juce::dontSendNotification);
                debug_.setText("48 kHz • 512 device / 4096 ring • direct control response ≈135 ms • underruns " +
                                   juce::String(world_underruns_.load(std::memory_order_relaxed)),
                               juce::dontSendNotification);
            }
        } else if (state < 0) {
            status_.setText("Engine unavailable: " + juce::String(engine_error()),
                            juce::dontSendNotification);
            debug_.setText("Home track remains usable; Enter World is disabled.",
                           juce::dontSendNotification);
        }

        if (smoke_test_) advance_smoke_test();
        repaint();
    }

    void advance_smoke_test() {
        const double now = juce::Time::getMillisecondCounterHiRes();
        if (engine_state_.load(std::memory_order_acquire) < 0) {
            finish_smoke_test(false);
            return;
        }
        if (smoke_stage_ == 0 && engine_ready_.load(std::memory_order_acquire)) {
            select_style(1);
            enter_world();
            smoke_stage_ = 1;
            smoke_stage_ms_ = now;
        } else if (smoke_stage_ == 1 && now - smoke_stage_ms_ > 3000.0) {
            select_style(2);
            smoke_stage_ = 2;
            smoke_stage_ms_ = now;
        } else if (smoke_stage_ == 2 && now - smoke_stage_ms_ > 3000.0) {
            return_home();
            smoke_stage_ = 3;
            smoke_stage_ms_ = now;
        } else if (smoke_stage_ == 3 && now - smoke_stage_ms_ > 2500.0) {
            const bool passed = engine_ready_.load(std::memory_order_relaxed) &&
                                audio_callbacks_.load(std::memory_order_relaxed) > 0 &&
                                world_reads_.load(std::memory_order_relaxed) > 0 &&
                                world_underruns_.load(std::memory_order_relaxed) == 0 &&
                                peak_output_.load(std::memory_order_relaxed) > 0.01F &&
                                peak_world_.load(std::memory_order_relaxed) > 0.001F &&
                                max_world_mix_.load(std::memory_order_relaxed) > 0.95F &&
                                current_world_mix_.load(std::memory_order_relaxed) < 0.05F &&
                                style_changes_ >= 3;
            finish_smoke_test(passed);
        }
    }

    void finish_smoke_test(bool passed) {
        if (smoke_finished_) return;
        smoke_finished_ = true;
        const juce::String report =
            "{\n"
            "  \"passed\": " + juce::String(passed ? "true" : "false") + ",\n" +
            "  \"engine_ready\": " + juce::String(engine_ready_.load() ? "true" : "false") + ",\n" +
            "  \"audio_callbacks\": " + juce::String(audio_callbacks_.load()) + ",\n" +
            "  \"world_reads\": " + juce::String(world_reads_.load()) + ",\n" +
            "  \"world_underruns\": " + juce::String(world_underruns_.load()) + ",\n" +
            "  \"peak_output\": " + juce::String(peak_output_.load(), 6) + ",\n" +
            "  \"peak_world\": " + juce::String(peak_world_.load(), 6) + ",\n" +
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
    juce::TextButton dub_;
    juce::TextButton disco_;
    juce::TextButton ambient_;

    const bool smoke_test_;
    const juce::File smoke_report_;
    TestTrack test_track_;
    juce::AudioBuffer<float> world_audio_;
    std::unique_ptr<song_world::GenerativeEngine> engine_;
    std::thread model_loader_;

    std::atomic<bool> engine_ready_{false};
    std::atomic<int> engine_state_{0};
    std::atomic<bool> playing_{false};
    std::atomic<float> target_world_mix_{0.0F};
    std::atomic<float> current_world_mix_{0.0F};
    std::atomic<float> max_world_mix_{0.0F};
    std::atomic<float> peak_output_{0.0F};
    std::atomic<float> peak_world_{0.0F};
    std::atomic<double> actual_sample_rate_{kSampleRate};
    std::atomic<std::uint64_t> home_position_{0};
    std::atomic<std::uint64_t> audio_callbacks_{0};
    std::atomic<std::uint64_t> world_reads_{0};
    std::atomic<std::uint64_t> world_underruns_{0};

    std::array<float, 3> current_style_weights_{1.0F, 0.0F, 0.0F};
    std::array<float, 3> target_style_weights_{1.0F, 0.0F, 0.0F};
    int selected_style_{0};
    int style_changes_{0};
    int status_tick_{0};
    std::uint64_t conditioning_sequence_{1};
    bool in_world_{false};

    mutable std::mutex error_mutex_;
    std::string engine_error_;

    double smoke_started_ms_{0.0};
    double smoke_stage_ms_{0.0};
    int smoke_stage_{0};
    bool smoke_finished_{false};
};

class MainWindow final : public juce::DocumentWindow {
public:
    MainWindow(bool smoke_test, const juce::File& smoke_report)
        : DocumentWindow("Song World Prototype",
                         juce::Colour(0xff0b0f14),
                         juce::DocumentWindow::allButtons) {
        setUsingNativeTitleBar(true);
        setResizable(true, true);
        setResizeLimits(760, 520, 1200, 800);
        setContentOwned(new MainComponent(smoke_test, smoke_report), true);
        centreWithSize(getWidth(), getHeight());
        setVisible(true);
    }

    void closeButtonPressed() override {
        if (auto* app = juce::JUCEApplicationBase::getInstance()) app->systemRequestedQuit();
    }
};

class SongWorldApplication final : public juce::JUCEApplication {
public:
    const juce::String getApplicationName() override { return "Song World Prototype"; }
    const juce::String getApplicationVersion() override { return "0.1.0-local"; }
    bool moreThanOneInstanceAllowed() override { return false; }

    void initialise(const juce::String&) override {
        bool smoke_test = false;
        auto smoke_report = juce::File::getSpecialLocation(juce::File::tempDirectory)
                                .getChildFile("song-world-app-smoke.json");
        for (const auto& argument : getCommandLineParameterArray()) {
            if (argument == "--smoke-test") smoke_test = true;
            if (argument.startsWith("--smoke-report=")) {
                smoke_report = juce::File(argument.fromFirstOccurrenceOf("=", false, false));
            }
        }
        window_ = std::make_unique<MainWindow>(smoke_test, smoke_report);
    }

    void shutdown() override { window_.reset(); }

    void systemRequestedQuit() override { quit(); }

private:
    std::unique_ptr<MainWindow> window_;
};

}  // namespace

START_JUCE_APPLICATION(SongWorldApplication)
