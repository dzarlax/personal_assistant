#include "voice_client.h"
#include "esphome/core/log.h"
#include "esphome/core/application.h"

#include "esp_tls.h"
#include "esp_crt_bundle.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include <algorithm>
#include <cstring>

namespace esphome {
namespace voice_client {

static const char *const TAG = "voice_client";

static const uint32_t SAMPLE_RATE = 16000;
static const uint32_t BYTES_PER_SEC = SAMPLE_RATE * 2;
static const int MIC_GAIN = 4;

void VoiceClient::setup() {
  ESP_LOGI(TAG, "Voice client initialized (url: %s, max_record: %ds)", api_url_.c_str(), max_record_seconds_);

  // Mic callback: fill chunk buffer, signal when full.
  this->mic_->add_data_callback([this](const std::vector<uint8_t> &data) {
    if (state_ != State::RECORDING) return;
    if (mic_buf_ready_) return;  // previous chunk not consumed yet, drop

    const int16_t *src = reinterpret_cast<const int16_t *>(data.data());
    int16_t *dst = reinterpret_cast<int16_t *>(mic_buf_ + mic_buf_pos_);
    size_t samples = std::min(data.size(), MIC_CHUNK_SIZE - mic_buf_pos_) / 2;

    for (size_t i = 0; i < samples; i++) {
      int32_t amplified = static_cast<int32_t>(src[i]) * MIC_GAIN;
      if (amplified > 32767) amplified = 32767;
      if (amplified < -32768) amplified = -32768;
      dst[i] = static_cast<int16_t>(amplified);
    }
    mic_buf_pos_ += samples * 2;

    if (mic_buf_pos_ >= MIC_CHUNK_SIZE) {
      mic_buf_ready_ = true;
      mic_buf_pos_ = 0;
    }
  });

  set_state_(State::IDLE);
}

void VoiceClient::loop() {
  if (should_start_stream_) {
    should_start_stream_ = false;
    xTaskCreate([](void *param) {
      auto *self = static_cast<VoiceClient *>(param);
      self->streaming_task_();
      vTaskDelete(nullptr);
    }, "voice_stream", 8192, this, 5, nullptr);
  }
}

void VoiceClient::start_recording() {
  if (state_ != State::IDLE) {
    ESP_LOGW(TAG, "Cannot record: state=%d", (int)state_);
    return;
  }

  mic_buf_pos_ = 0;
  mic_buf_ready_ = false;
  stop_requested_ = false;
  total_bytes_sent_ = 0;
  record_start_ = millis();

  set_state_(State::RECORDING);
  this->mic_->start();
  should_start_stream_ = true;
  ESP_LOGI(TAG, "Recording started (streaming mode)");
}

void VoiceClient::stop_recording() {
  if (state_ != State::RECORDING) return;
  ESP_LOGI(TAG, "Stop requested");
  stop_requested_ = true;
  // The streaming task will handle mic stop and state transitions.
}

void VoiceClient::streaming_task_() {
  ESP_LOGI(TAG, "Opening HTTP connection...");

  std::string auth_header = "Bearer " + api_token_;

  esp_http_client_config_t config = {};
  config.url = api_url_.c_str();
  config.method = HTTP_METHOD_POST;
  config.timeout_ms = 60000;
  config.buffer_size = 2048;
  config.buffer_size_tx = 2048;
  config.crt_bundle_attach = esp_crt_bundle_attach;

  esp_http_client_handle_t client = esp_http_client_init(&config);
  if (!client) {
    ESP_LOGE(TAG, "HTTP client init failed");
    this->mic_->stop();
    set_state_(State::ERROR);
    return;
  }

  esp_http_client_set_header(client, "Content-Type", "audio/wav");
  esp_http_client_set_header(client, "Authorization", auth_header.c_str());
  esp_http_client_set_header(client, "Accept", "audio/wav");

  // Open connection without specifying content length (-1 = chunked).
  esp_err_t err = esp_http_client_open(client, -1);
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "HTTP open failed: %s", esp_err_to_name(err));
    this->mic_->stop();
    esp_http_client_cleanup(client);
    set_state_(State::ERROR);
    return;
  }

  // Send WAV header first (with placeholder data size = 0xFFFFFFFF).
  uint8_t wav_header[WAV_HEADER_SIZE];
  build_wav_header_(wav_header, SAMPLE_RATE);
  esp_http_client_write(client, (const char *)wav_header, WAV_HEADER_SIZE);
  total_bytes_sent_ += WAV_HEADER_SIZE;

  ESP_LOGI(TAG, "Streaming audio...");

  // Stream audio chunks as they come from the mic callback.
  uint32_t max_ms = max_record_seconds_ * 1000;
  while (!stop_requested_) {
    uint32_t elapsed = millis() - record_start_;
    if (elapsed >= max_ms) {
      ESP_LOGW(TAG, "Max recording time reached (%ds)", max_record_seconds_);
      break;
    }

    if (mic_buf_ready_) {
      int written = esp_http_client_write(client, (const char *)mic_buf_, MIC_CHUNK_SIZE);
      if (written < 0) {
        ESP_LOGE(TAG, "HTTP write failed");
        break;
      }
      total_bytes_sent_ += written;
      mic_buf_ready_ = false;
    } else {
      vTaskDelay(pdMS_TO_TICKS(10));
    }
  }

  // Flush any remaining partial chunk.
  this->mic_->stop();
  if (mic_buf_pos_ > 0) {
    esp_http_client_write(client, (const char *)mic_buf_, mic_buf_pos_);
    total_bytes_sent_ += mic_buf_pos_;
    mic_buf_pos_ = 0;
  }

  uint32_t elapsed = millis() - record_start_;
  size_t audio_bytes = total_bytes_sent_ - WAV_HEADER_SIZE;
  ESP_LOGI(TAG, "Recording done: %u bytes, %u ms", (unsigned)audio_bytes, elapsed);

  if (audio_bytes < BYTES_PER_SEC / 4) {
    ESP_LOGW(TAG, "Recording too short, ignoring");
    esp_http_client_close(client);
    esp_http_client_cleanup(client);
    set_state_(State::IDLE);
    return;
  }

  // Signal end of body and read response.
  set_state_(State::PROCESSING);
  set_led_pulse_();

  int content_length = esp_http_client_fetch_headers(client);
  int status = esp_http_client_get_status_code(client);
  ESP_LOGI(TAG, "HTTP response: status=%d, content_length=%d", status, content_length);

  if (status != 200) {
    ESP_LOGE(TAG, "API error: HTTP %d", status);
    esp_http_client_close(client);
    esp_http_client_cleanup(client);
    set_state_(State::ERROR);
    return;
  }

  play_response_(client);

  esp_http_client_close(client);
  esp_http_client_cleanup(client);
  set_state_(State::IDLE);
}

void VoiceClient::play_response_(esp_http_client_handle_t client) {
  set_state_(State::PLAYING);
  this->spk_->start();

  const size_t chunk_size = 2048;
  uint8_t chunk_buf[2048];
  size_t total_read = 0;
  bool header_skipped = false;

  while (true) {
    int read_len = esp_http_client_read(client, (char *)chunk_buf, chunk_size);
    if (read_len <= 0) break;
    total_read += read_len;

    uint8_t *data = chunk_buf;
    size_t data_len = read_len;

    if (!header_skipped) {
      header_skipped = true;
      if (data_len > 44 && memcmp(data, "RIFF", 4) == 0) {
        data += 44;
        data_len -= 44;
      }
    }

    size_t offset = 0;
    while (offset < data_len) {
      size_t to_write = std::min(data_len - offset, (size_t)512);
      size_t written = this->spk_->play(data + offset, to_write);
      if (written > 0) {
        offset += written;
      } else {
        vTaskDelay(pdMS_TO_TICKS(10));
      }
    }
  }

  ESP_LOGI(TAG, "Played %u bytes of audio", (unsigned)total_read);
  this->spk_->finish();
  vTaskDelay(pdMS_TO_TICKS(500));
}

void VoiceClient::build_wav_header_(uint8_t *buf, uint32_t sample_rate) {
  // WAV header with unknown data size (0xFFFFFFFF) — server will handle it.
  uint16_t num_channels = 1;
  uint16_t bits_per_sample = 16;
  uint32_t byte_rate = sample_rate * num_channels * bits_per_sample / 8;
  uint16_t block_align = num_channels * bits_per_sample / 8;

  auto write16 = [&](size_t offset, uint16_t v) {
    buf[offset] = v & 0xFF;
    buf[offset + 1] = (v >> 8) & 0xFF;
  };
  auto write32 = [&](size_t offset, uint32_t v) {
    buf[offset] = v & 0xFF;
    buf[offset + 1] = (v >> 8) & 0xFF;
    buf[offset + 2] = (v >> 16) & 0xFF;
    buf[offset + 3] = (v >> 24) & 0xFF;
  };

  memcpy(buf, "RIFF", 4);
  write32(4, 0xFFFFFFFF);  // unknown file size
  memcpy(buf + 8, "WAVE", 4);
  memcpy(buf + 12, "fmt ", 4);
  write32(16, 16);
  write16(20, 1);  // PCM
  write16(22, num_channels);
  write32(24, sample_rate);
  write32(28, byte_rate);
  write16(32, block_align);
  write16(34, bits_per_sample);
  memcpy(buf + 36, "data", 4);
  write32(40, 0xFFFFFFFF);  // unknown data size
}

void VoiceClient::set_state_(State state) {
  state_ = state;
  switch (state) {
    case State::IDLE:
      set_led_color_(0, 0, 1.0f, 0.2f);
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
