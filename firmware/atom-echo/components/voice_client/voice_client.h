#pragma once

#include "esphome/core/component.h"
#include "esphome/components/microphone/microphone.h"
#include "esphome/components/speaker/speaker.h"
#include "esphome/components/light/light_state.h"

#include "esp_http_client.h"

namespace esphome {
namespace voice_client {

enum class State : uint8_t {
  IDLE,
  RECORDING,      // mic on, streaming audio via chunked HTTP
  PROCESSING,     // HTTP request sent, waiting for response
  PLAYING,        // streaming response to speaker
  ERROR,
};

static const size_t WAV_HEADER_SIZE = 44;
// Small ring buffer for mic callback → HTTP write handoff.
static const size_t MIC_CHUNK_SIZE = 4096;

class VoiceClient : public Component {
 public:
  void setup() override;
  void loop() override;
  float get_setup_priority() const override { return setup_priority::AFTER_WIFI; }

  void set_microphone(microphone::Microphone *mic) { this->mic_ = mic; }
  void set_speaker(speaker::Speaker *spk) { this->spk_ = spk; }
  void set_led(light::LightState *led) { this->led_ = led; }
  void set_api_url(const std::string &url) { this->api_url_ = url; }
  void set_api_token(const std::string &token) { this->api_token_ = token; }
  void set_max_record_seconds(int secs) { this->max_record_seconds_ = secs; }

  void start_recording();
  void stop_recording();

 protected:
  void set_state_(State state);
  void set_led_color_(float r, float g, float b, float brightness = 0.8f);
  void set_led_pulse_();
  void streaming_task_();
  void play_response_(esp_http_client_handle_t client);
  void build_wav_header_(uint8_t *buf, uint32_t sample_rate);

  microphone::Microphone *mic_{nullptr};
  speaker::Speaker *spk_{nullptr};
  light::LightState *led_{nullptr};

  std::string api_url_;
  std::string api_token_;
  int max_record_seconds_{10};

  State state_{State::IDLE};

  // Ring buffer for mic→HTTP streaming.
  uint8_t mic_buf_[MIC_CHUNK_SIZE];
  volatile size_t mic_buf_pos_{0};
  volatile bool mic_buf_ready_{false};  // chunk ready to send

  uint32_t record_start_{0};
  volatile bool stop_requested_{false};
  bool should_start_stream_{false};
  size_t total_bytes_sent_{0};
};

}  // namespace voice_client
}  // namespace esphome
