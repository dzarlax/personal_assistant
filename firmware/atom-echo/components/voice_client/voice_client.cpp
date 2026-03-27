#include "voice_client.h"
#include "esphome/core/log.h"
#include "esphome/core/application.h"

#include "esp_crt_bundle.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include <algorithm>
#include <cstring>

namespace esphome {
namespace voice_client {

static const char *const TAG = "voice_client";
static const uint32_t SAMPLE_RATE = 16000;
static const int MIC_GAIN = 4;

void VoiceClient::setup() {
  ESP_LOGI(TAG, "Voice client ready (url: %s, max_record: %ds)", api_url_.c_str(), max_record_seconds_);

  // Mic callback: fill send buffer, signal when full.
  this->mic_->add_data_callback([this](const std::vector<uint8_t> &data) {
    if (state_ != State::RECORDING) return;
    if (mic_send_ready_) return;  // previous chunk not sent yet

    const int16_t *src = reinterpret_cast<const int16_t *>(data.data());
    int16_t *dst = reinterpret_cast<int16_t *>(mic_send_buf_ + mic_send_pos_);
    size_t samples = std::min(data.size(), WS_AUDIO_CHUNK - mic_send_pos_) / 2;

    for (size_t i = 0; i < samples; i++) {
      int32_t amplified = static_cast<int32_t>(src[i]) * MIC_GAIN;
      if (amplified > 32767) amplified = 32767;
      if (amplified < -32768) amplified = -32768;
      dst[i] = static_cast<int16_t>(amplified);
    }
    mic_send_pos_ += samples * 2;

    if (mic_send_pos_ >= WS_AUDIO_CHUNK) {
      mic_send_ready_ = true;
      mic_send_pos_ = 0;
    }
  });

  set_state_(State::IDLE);
}

void VoiceClient::loop() {
  // Send mic chunks over WebSocket when ready.
  if (state_ == State::RECORDING && mic_send_ready_ && ws_client_) {
    esp_websocket_client_send_bin(ws_client_, (const char *)mic_send_buf_, WS_AUDIO_CHUNK, portMAX_DELAY);
    mic_send_ready_ = false;
  }

  // Clean up WebSocket from main loop (not from event callback).
  if (should_disconnect_) {
    should_disconnect_ = false;
    auto *client = ws_client_;
    ws_client_ = nullptr;
    if (client) {
      // Run cleanup in a separate task to avoid blocking loop / watchdog.
      xTaskCreate([](void *param) {
        auto *c = static_cast<esp_websocket_client_handle_t>(param);
        esp_websocket_client_stop(c);
        esp_websocket_client_destroy(c);
        ESP_LOGI("voice_client", "WebSocket cleaned up");
        vTaskDelete(nullptr);
      }, "ws_cleanup", 4096, client, 1, nullptr);
    }
    set_state_(State::IDLE);
  }

  // Check recording timeout.
  if (state_ == State::RECORDING) {
    uint32_t elapsed = millis() - record_start_;
    if (elapsed >= (uint32_t)(max_record_seconds_ * 1000)) {
      ESP_LOGW(TAG, "Max recording time reached (%ds)", max_record_seconds_);
      stop_recording();
    }
  }
}

void VoiceClient::start_recording() {
  if (state_ != State::IDLE) {
    ESP_LOGW(TAG, "Cannot record: state=%d", (int)state_);
    return;
  }

  mic_send_pos_ = 0;
  mic_send_ready_ = false;
  speaker_started_ = false;
  wav_header_skipped_ = false;
  total_audio_received_ = 0;

  // Build WebSocket URL (convert https:// to wss://, http:// to ws://).
  std::string ws_url = api_url_;
  if (ws_url.find("https://") == 0) {
    ws_url = "wss://" + ws_url.substr(8);
  } else if (ws_url.find("http://") == 0) {
    ws_url = "ws://" + ws_url.substr(7);
  }
  // Append /ws path if not already there.
  if (ws_url.find("/ws") == std::string::npos) {
    if (ws_url.back() != '/') ws_url += "/";
    ws_url += "ws";
  }

  // Add token as query parameter (WebSocket doesn't support custom headers in ESP-IDF).
  ws_url += "?token=" + api_token_;

  ESP_LOGI(TAG, "Connecting to %s", ws_url.c_str());
  set_state_(State::CONNECTING);

  esp_websocket_client_config_t ws_cfg = {};
  ws_cfg.uri = ws_url.c_str();
  ws_cfg.buffer_size = 4096;
  ws_cfg.crt_bundle_attach = esp_crt_bundle_attach;
  ws_cfg.task_stack = 6144;

  ws_client_ = esp_websocket_client_init(&ws_cfg);
  if (!ws_client_) {
    ESP_LOGE(TAG, "WS client init failed");
    set_state_(State::ERROR);
    return;
  }

  esp_websocket_register_events(ws_client_, WEBSOCKET_EVENT_ANY, ws_event_handler_, this);
  esp_websocket_client_start(ws_client_);
}

void VoiceClient::stop_recording() {
  if (state_ != State::RECORDING) return;

  this->mic_->stop();

  // Flush remaining mic data.
  if (mic_send_pos_ > 0 && ws_client_) {
    esp_websocket_client_send_bin(ws_client_, (const char *)mic_send_buf_, mic_send_pos_, portMAX_DELAY);
    mic_send_pos_ = 0;
  }

  // Send stop signal.
  const char *stop_msg = "{\"action\":\"stop\"}";
  if (ws_client_) {
    esp_websocket_client_send_text(ws_client_, stop_msg, strlen(stop_msg), portMAX_DELAY);
  }

  uint32_t elapsed = millis() - record_start_;
  ESP_LOGI(TAG, "Recording stopped (%u ms)", elapsed);
  set_state_(State::PROCESSING);
}

// --- WebSocket event handler (static, dispatches to instance) ---

void VoiceClient::ws_event_handler_(void *arg, esp_event_base_t base, int32_t event_id, void *event_data) {
  auto *self = static_cast<VoiceClient *>(arg);
  auto *data = static_cast<esp_websocket_event_data_t *>(event_data);

  switch (event_id) {
    case WEBSOCKET_EVENT_CONNECTED:
      self->on_ws_connected_();
      break;
    case WEBSOCKET_EVENT_DATA:
      if (data->data_ptr && data->data_len > 0) {
        self->on_ws_data_((uint8_t *)data->data_ptr, data->data_len, data->op_code);
      }
      break;
    case WEBSOCKET_EVENT_DISCONNECTED:
      self->on_ws_disconnected_();
      break;
    case WEBSOCKET_EVENT_ERROR:
      self->on_ws_error_();
      break;
    default:
      break;
  }
}

void VoiceClient::on_ws_connected_() {
  ESP_LOGI(TAG, "WebSocket connected, starting mic");
  record_start_ = millis();
  set_state_(State::RECORDING);
  this->mic_->start();
}

void VoiceClient::on_ws_data_(uint8_t *data, int len, int opcode) {
  if (opcode == 1) {
    // Text frame — status message from server.
    std::string msg(reinterpret_cast<char *>(data), len);
    ESP_LOGI(TAG, "Server: %s", msg.c_str());

    if (msg.find("\"done\"") != std::string::npos) {
      // Response complete — signal loop() to clean up.
      if (speaker_started_) {
        this->spk_->finish();
        speaker_started_ = false;
      }
      ESP_LOGI(TAG, "Playback complete (%u bytes received)", (unsigned)total_audio_received_);
      should_disconnect_ = true;
    }
  } else if (opcode == 2) {
    // Binary frame — audio data.
    if (!speaker_started_) {
      set_state_(State::PLAYING);
      this->spk_->start();
      speaker_started_ = true;
    }

    uint8_t *audio = data;
    size_t audio_len = len;

    // Skip WAV header in first binary frame.
    if (!wav_header_skipped_ && audio_len > 44) {
      if (memcmp(audio, "RIFF", 4) == 0) {
        audio += 44;
        audio_len -= 44;
        ESP_LOGI(TAG, "Skipped WAV header");
      }
      wav_header_skipped_ = true;
    }

    // Feed to speaker in small pieces.
    size_t offset = 0;
    while (offset < audio_len) {
      size_t to_write = std::min(audio_len - offset, (size_t)512);
      size_t written = this->spk_->play(audio + offset, to_write);
      if (written > 0) {
        offset += written;
      } else {
        vTaskDelay(pdMS_TO_TICKS(5));
      }
    }
    total_audio_received_ += len;
  }
}

void VoiceClient::on_ws_disconnected_() {
  ESP_LOGI(TAG, "WebSocket disconnected");
  if (state_ != State::IDLE) {
    if (speaker_started_) {
      this->spk_->finish();
      speaker_started_ = false;
    }
    should_disconnect_ = true;  // clean up in loop()
  }
}

void VoiceClient::on_ws_error_() {
  ESP_LOGE(TAG, "WebSocket error");
  this->mic_->stop();
  set_state_(State::ERROR);
  should_disconnect_ = true;  // clean up in loop()
}

// --- LED helpers ---

void VoiceClient::set_state_(State state) {
  state_ = state;
  switch (state) {
    case State::IDLE:
      set_led_color_(0, 0, 1.0f, 0.2f);
      break;
    case State::CONNECTING:
      set_led_color_(1.0f, 0.5f, 0, 0.5f);  // orange
      break;
    case State::RECORDING:
      set_led_color_(1.0f, 0, 0);
      break;
    case State::PROCESSING:
      set_led_pulse_();
      break;
    case State::PLAYING:
      set_led_color_(0, 1.0f, 0);
      break;
    case State::ERROR:
      for (int i = 0; i < 3; i++) {
        set_led_color_(1.0f, 0, 0, 1.0f);
        delay(150);
        set_led_color_(0, 0, 0, 0);
        delay(150);
      }
      set_state_(State::IDLE);
      break;
  }
}

void VoiceClient::set_led_color_(float r, float g, float b, float brightness) {
  if (!led_) return;
  auto call = led_->turn_on();
  call.set_rgb(r, g, b);
  call.set_brightness(brightness);
  call.set_transition_length(0);
  call.perform();
}

void VoiceClient::set_led_pulse_() {
  if (!led_) return;
  auto call = led_->turn_on();
  call.set_rgb(1.0f, 0.7f, 0);
  call.set_brightness(0.8f);
  call.set_effect("Pulse");
  call.perform();
}

}  // namespace voice_client
}  // namespace esphome
