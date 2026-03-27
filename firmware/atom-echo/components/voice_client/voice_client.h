#pragma once

#include "esphome/core/component.h"
#include "esphome/components/microphone/microphone.h"
#include "esphome/components/speaker/speaker.h"
#include "esphome/components/light/light_state.h"

#include "esp_websocket_client.h"

namespace esphome {
namespace voice_client {

enum class State : uint8_t {
  IDLE,
  CONNECTING,     // WebSocket connecting
  RECORDING,      // mic on, streaming audio frames
  PROCESSING,     // stop sent, waiting for response
  PLAYING,        // streaming response to speaker
  ERROR,
};

static const size_t WAV_HEADER_SIZE = 44;
static const size_t WS_AUDIO_CHUNK = 2048;  // bytes per WebSocket binary frame

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

  static void ws_event_handler_(void *arg, esp_event_base_t base, int32_t event_id, void *event_data);
  void on_ws_connected_();
  void on_ws_data_(uint8_t *data, int len, int opcode);
  void on_ws_disconnected_();
  void on_ws_error_();

  microphone::Microphone *mic_{nullptr};
  speaker::Speaker *spk_{nullptr};
  light::LightState *led_{nullptr};

  std::string api_url_;
  std::string api_token_;
  int max_record_seconds_{10};

  State state_{State::IDLE};
  esp_websocket_client_handle_t ws_client_{nullptr};

  // Mic data buffer for sending — double buffer to avoid blocking callback.
  uint8_t mic_send_buf_[WS_AUDIO_CHUNK];
  volatile size_t mic_send_pos_{0};
  volatile bool mic_send_ready_{false};

  uint32_t record_start_{0};
  bool speaker_started_{false};
  bool wav_header_skipped_{false};
  size_t total_audio_received_{0};
  volatile bool should_disconnect_{false};
};

}  // namespace voice_client
}  // namespace esphome
