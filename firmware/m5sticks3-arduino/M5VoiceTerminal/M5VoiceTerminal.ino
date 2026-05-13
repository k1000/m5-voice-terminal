#include <M5Unified.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include "config.h"

// Arduino/M5Unified microphone uploader sketch for StickS3.
// Requires board: M5StickS3, libraries: M5Unified, M5GFX.
// Iteration target: press BtnA, record short WAV, POST to /voice-command.

static constexpr uint32_t SAMPLE_RATE = 16000;
static constexpr uint32_t RECORD_SECONDS = 3;
static constexpr size_t SAMPLE_COUNT = SAMPLE_RATE * RECORD_SECONDS;
static int16_t *samples = nullptr;
static const String SERVER_BASE_URL = String(VOICE_URL).substring(0, String(VOICE_URL).lastIndexOf('/'));
static constexpr uint32_t POLL_INTERVAL_MS = 1500;
static constexpr uint32_t POLL_TIMEOUT_MS = 120000;
static constexpr uint8_t SPEAKER_VOLUME = 230;  // Keep <= ~230 on battery to avoid brownouts.

static void drawWrapped(const String &text, int x, int y, int maxChars = 22) {
  int lineLen = 0;
  M5.Display.setCursor(x, y);
  for (size_t i = 0; i < text.length(); ++i) {
    char c = text[i];
    if (c == '\n' || lineLen >= maxChars) {
      y += 12;
      if (y > M5.Display.height() - 10) return;
      M5.Display.setCursor(x, y);
      lineLen = 0;
      if (c == '\n') continue;
    }
    M5.Display.print(c);
    lineLen++;
  }
}

static void drawStatus(const char *title, const String &line = "") {
  M5.Display.clear(BLACK);
  M5.Display.setTextColor(WHITE, BLACK);
  M5.Display.setTextSize(1);
  M5.Display.setCursor(4, 8);
  M5.Display.println(title);
  M5.Display.setCursor(4, 28);
  drawWrapped(line, 4, 28);
}

static uint16_t sentimentColor(const String &sentiment) {
  if (sentiment == "happy") return GREEN;
  if (sentiment == "sad") return RED;
  return YELLOW;
}

static void drawFace(const String &sentiment) {
  const int cx = M5.Display.width() / 2;
  const int cy = 50;
  const int r = 28;
  const uint16_t color = sentimentColor(sentiment);

  M5.Display.drawCircle(cx, cy, r, color);
  M5.Display.fillCircle(cx - 10, cy - 8, 3, color);
  M5.Display.fillCircle(cx + 10, cy - 8, 3, color);

  if (sentiment == "happy") {
    M5.Display.drawLine(cx - 14, cy + 8, cx - 7, cy + 15, color);
    M5.Display.drawLine(cx - 7, cy + 15, cx + 7, cy + 15, color);
    M5.Display.drawLine(cx + 7, cy + 15, cx + 14, cy + 8, color);
  } else if (sentiment == "sad") {
    M5.Display.drawLine(cx - 14, cy + 17, cx - 7, cy + 10, color);
    M5.Display.drawLine(cx - 7, cy + 10, cx + 7, cy + 10, color);
    M5.Display.drawLine(cx + 7, cy + 10, cx + 14, cy + 17, color);
  } else {
    M5.Display.drawLine(cx - 14, cy + 12, cx + 14, cy + 12, color);
  }
}

static void drawSentimentResponse(const String &sentimentInput, const String &line = "") {
  String sentiment = sentimentInput;
  if (sentiment != "happy" && sentiment != "neutral" && sentiment != "sad") sentiment = "neutral";
  M5.Display.clear(BLACK);
  M5.Display.setTextSize(1);
  M5.Display.setTextColor(sentimentColor(sentiment), BLACK);
  M5.Display.setCursor(4, 6);
  M5.Display.println("Face: " + sentiment);
  drawFace(sentiment);
  M5.Display.setTextColor(WHITE, BLACK);
  drawWrapped(line, 4, 88, 12);
}

static bool downloadAndDrawImage(const String &imageUrl, int width = 96, int height = 96) {
  if (!imageUrl.length()) return false;
  String url = imageUrl.startsWith("http") ? imageUrl : SERVER_BASE_URL + imageUrl;
  HTTPClient http;
  http.begin(url);
  int code = http.GET();
  if (code != 200) {
    http.end();
    return false;
  }
  int len = http.getSize();
  int expected = width * height * 2;
  if (len != expected) {
    http.end();
    return false;
  }
  uint8_t *rgb = (uint8_t *)ps_malloc(len);
  if (!rgb) {
    http.end();
    return false;
  }
  WiFiClient *stream = http.getStreamPtr();
  int readTotal = 0;
  uint32_t started = millis();
  while (http.connected() && readTotal < len && millis() - started < 10000) {
    size_t available = stream->available();
    if (available) {
      readTotal += stream->readBytes(rgb + readTotal, min((int)available, len - readTotal));
    } else {
      delay(1);
    }
  }
  http.end();
  if (readTotal == len) {
    M5.Display.pushImage((M5.Display.width() - width) / 2, 46, width, height, (uint16_t *)rgb);
    free(rgb);
    return true;
  }
  free(rgb);
  return false;
}

static void writeWavHeader(uint8_t *header, uint32_t sampleRate, uint32_t sampleCount) {
  const uint16_t channels = 1;
  const uint16_t bitsPerSample = 16;
  uint32_t dataSize = sampleCount * channels * bitsPerSample / 8;
  uint32_t chunkSize = 36 + dataSize;
  uint32_t byteRate = sampleRate * channels * bitsPerSample / 8;
  uint16_t blockAlign = channels * bitsPerSample / 8;

  memcpy(header + 0, "RIFF", 4);
  memcpy(header + 4, &chunkSize, 4);
  memcpy(header + 8, "WAVE", 4);
  memcpy(header + 12, "fmt ", 4);
  uint32_t subchunk1Size = 16;
  uint16_t audioFormat = 1;
  memcpy(header + 16, &subchunk1Size, 4);
  memcpy(header + 20, &audioFormat, 2);
  memcpy(header + 22, &channels, 2);
  memcpy(header + 24, &sampleRate, 4);
  memcpy(header + 28, &byteRate, 4);
  memcpy(header + 32, &blockAlign, 2);
  memcpy(header + 34, &bitsPerSample, 2);
  memcpy(header + 36, "data", 4);
  memcpy(header + 40, &dataSize, 4);
}

static bool connectWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  drawStatus("WiFi", "connecting...");
  for (int i = 0; i < 30 && WiFi.status() != WL_CONNECTED; ++i) {
    delay(500);
  }
  if (WiFi.status() == WL_CONNECTED) {
    drawStatus("WiFi OK", WiFi.localIP().toString());
    return true;
  }
  drawStatus("WiFi failed");
  return false;
}

static bool recordAudio() {
  drawStatus("Recording", "speak now");
  M5.Speaker.end(); // StickS3 cannot reliably use mic and speaker together.
  if (!M5.Mic.isEnabled()) {
    if (!M5.Mic.begin()) {
      drawStatus("Mic failed");
      return false;
    }
  }

  size_t offset = 0;
  static constexpr size_t CHUNK = 256;
  while (offset < SAMPLE_COUNT) {
    size_t n = min(CHUNK, SAMPLE_COUNT - offset);
    if (M5.Mic.record(samples + offset, n, SAMPLE_RATE)) {
      offset += n;
      M5.Display.fillRect(4, 52, (offset * 120) / SAMPLE_COUNT, 8, GREEN);
    } else {
      delay(1);
    }
    M5.update();
  }
  M5.Mic.end();
  return true;
}

static String extractJobId(const String &response) {
  JsonDocument doc;
  DeserializationError err = deserializeJson(doc, response);
  if (err) return "";
  const char *jobId = doc["meta"]["job_id"];
  return jobId ? String(jobId) : "";
}

static bool playAudioUrl(const String &audioUrl) {
  if (!audioUrl.length()) return false;
  String url = audioUrl.startsWith("http") ? audioUrl : SERVER_BASE_URL + audioUrl;
  drawStatus("Audio", "downloading...");

  HTTPClient http;
  http.begin(url);
  int code = http.GET();
  if (code != 200) {
    drawStatus("Audio HTTP", String(code));
    http.end();
    return false;
  }

  int len = http.getSize();
  if (len <= 44 || len > 4 * 1024 * 1024) {
    drawStatus("Audio size", String(len));
    http.end();
    return false;
  }

  uint8_t *wav = (uint8_t *)ps_malloc(len);
  if (!wav) {
    drawStatus("Audio OOM", String(len));
    http.end();
    return false;
  }

  WiFiClient *stream = http.getStreamPtr();
  int readTotal = 0;
  uint32_t started = millis();
  while (http.connected() && readTotal < len && millis() - started < 20000) {
    size_t available = stream->available();
    if (available) {
      int n = stream->readBytes(wav + readTotal, min((int)available, len - readTotal));
      readTotal += n;
      M5.Display.fillRect(4, 52, (readTotal * 120) / len, 8, BLUE);
    } else {
      delay(1);
    }
  }
  http.end();

  if (readTotal != len) {
    free(wav);
    drawStatus("Audio read", String(readTotal) + "/" + String(len));
    return false;
  }

  drawStatus("Audio", "playing...");
  M5.Mic.end();
  M5.Speaker.begin();
  M5.Speaker.setVolume(SPEAKER_VOLUME);
  bool ok = M5.Speaker.playWav(wav, len, 1, -1, true);
  if (ok) {
    while (M5.Speaker.isPlaying()) {
      M5.update();
      delay(10);
    }
  }
  M5.Speaker.end();
  free(wav);
  return ok;
}

static bool pollJobResult(const String &jobId) {
  const String url = SERVER_BASE_URL + "/agent/jobs/" + jobId;
  const char spin[] = {'|', '/', '-', '\\'};
  uint32_t start = millis();
  uint32_t i = 0;

  while (millis() - start < POLL_TIMEOUT_MS) {
    HTTPClient http;
    http.begin(url);
    int code = http.GET();
    String body = http.getString();
    http.end();

    if (code != 200) {
      drawStatus("Poll HTTP", String(code));
      delay(POLL_INTERVAL_MS);
      continue;
    }

    JsonDocument doc;
    DeserializationError err = deserializeJson(doc, body);
    if (err) {
      drawStatus("Poll JSON", err.c_str());
      delay(POLL_INTERVAL_MS);
      continue;
    }

    String status = doc["status"] | "?";
    if (status == "done") {
      String result = doc["result_text"] | "[done: no text]";
      String sentiment = doc["sentiment"] | "neutral";
      String audioUrl = doc["audio_url"] | "";
      String imageUrl = doc["image_url"] | "";
      drawSentimentResponse(sentiment, result);
      if (imageUrl.length()) {
        downloadAndDrawImage(imageUrl);
      }
      if (audioUrl.length()) {
        delay(700);
        playAudioUrl(audioUrl);
        drawSentimentResponse(sentiment, result);
        if (imageUrl.length()) {
          downloadAndDrawImage(imageUrl);
        }
      }
      return true;
    }
    if (status == "failed") {
      String error = doc["error"] | "agent failed";
      drawSentimentResponse("sad", error);
      return false;
    }

    drawStatus((String("Waiting ") + spin[i++ % 4]).c_str(), "Job " + jobId + "\n" + status);
    delay(POLL_INTERVAL_MS);
  }

  drawStatus("Timeout", "Job " + jobId);
  return false;
}

static String postAudio() {
  drawStatus("Uploading", "please wait");
  HTTPClient http;
  http.begin(VOICE_URL);

  const String boundary = "----m5stickS3Boundary";
  http.addHeader("Content-Type", "multipart/form-data; boundary=" + boundary);

  String head = "--" + boundary + "\r\n";
  head += "Content-Disposition: form-data; name=\"device\"\r\n\r\n" DEVICE_ID "\r\n";
  head += "--" + boundary + "\r\n";
  head += "Content-Disposition: form-data; name=\"event\"\r\n\r\naudio_upload\r\n";
  head += "--" + boundary + "\r\n";
  head += "Content-Disposition: form-data; name=\"audio\"; filename=\"command.wav\"\r\n";
  head += "Content-Type: audio/wav\r\n\r\n";
  String tail = "\r\n--" + boundary + "--\r\n";

  uint8_t wavHeader[44];
  writeWavHeader(wavHeader, SAMPLE_RATE, SAMPLE_COUNT);
  const size_t audioBytes = SAMPLE_COUNT * sizeof(int16_t);
  const size_t total = head.length() + sizeof(wavHeader) + audioBytes + tail.length();

  uint8_t *body = (uint8_t *)ps_malloc(total);
  if (!body) {
    http.end();
    drawStatus("OOM", "upload buffer");
    return "";
  }
  size_t pos = 0;
  memcpy(body + pos, head.c_str(), head.length()); pos += head.length();
  memcpy(body + pos, wavHeader, sizeof(wavHeader)); pos += sizeof(wavHeader);
  memcpy(body + pos, samples, audioBytes); pos += audioBytes;
  memcpy(body + pos, tail.c_str(), tail.length()); pos += tail.length();

  int code = http.POST(body, total);
  free(body);
  String response = http.getString();
  http.end();

  drawStatus(code == 200 ? "Uploaded" : "HTTP error", String(code));
  return response;
}

void setup() {
  auto cfg = M5.config();
  M5.begin(cfg);
  M5.Display.setRotation(0);
  samples = (int16_t *)ps_malloc(SAMPLE_COUNT * sizeof(int16_t));
  if (!samples) {
    drawStatus("OOM", "samples");
    while (true) delay(1000);
  }
  connectWiFi();
  drawStatus("Ready", "BtnA: record");
}

void loop() {
  M5.update();
  if (M5.BtnA.wasPressed()) {
    if (WiFi.status() != WL_CONNECTED && !connectWiFi()) {
      delay(1000);
      return;
    }
    if (recordAudio()) {
      String response = postAudio();
      String jobId = extractJobId(response);
      if (jobId.length()) {
        drawStatus("Queued", "Job " + jobId);
        pollJobResult(jobId);
        // Leave the final result/error on screen until the next button press.
      } else {
        drawStatus("Server", response.substring(0, 120));
      }
    }
    delay(1000);
  }
  delay(10);
}
