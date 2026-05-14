#include <M5Unified.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include "config.h"
#include "wolf_faces.h"

// Arduino/M5Unified microphone uploader sketch for StickS3.
// Requires board: M5StickS3, libraries: M5Unified, M5GFX.
// Iteration target: press BtnA, record short WAV, POST to /voice-command.

static constexpr uint32_t SAMPLE_RATE = 16000;
static constexpr uint32_t MAX_RECORD_SECONDS = 10;
static constexpr uint32_t MIN_RECORD_MS = 300;
static constexpr size_t MAX_SAMPLE_COUNT = SAMPLE_RATE * MAX_RECORD_SECONDS;
static int16_t *samples = nullptr;
static size_t recorded_samples = 0;
static const String SERVER_BASE_URL = String(VOICE_URL).substring(0, String(VOICE_URL).lastIndexOf('/'));
static constexpr uint32_t POLL_INTERVAL_MS = 1500;
static constexpr uint32_t POLL_TIMEOUT_MS = 120000;
static constexpr uint32_t SCREEN_SLEEP_MS = 20000;
static constexpr uint8_t SCREEN_BRIGHTNESS = 255;
static constexpr uint8_t SPEAKER_VOLUME = 230;  // Keep <= ~230 on battery to avoid brownouts.
static uint32_t last_screen_activity_ms = 0;
static bool screen_awake = true;

static void wakeScreen() {
  if (!screen_awake) {
    M5.Display.setBrightness(SCREEN_BRIGHTNESS);
    screen_awake = true;
  }
  last_screen_activity_ms = millis();
}

static void sleepScreen() {
  if (screen_awake && millis() - last_screen_activity_ms > SCREEN_SLEEP_MS) {
    M5.Display.setBrightness(0);
    screen_awake = false;
  }
}

static void drawWrapped(const String &text, int x, int y, int maxChars = 22, int lineHeight = 12) {
  int lineLen = 0;
  M5.Display.setCursor(x, y);
  for (size_t i = 0; i < text.length(); ++i) {
    char c = text[i];
    if (c == '\n' || lineLen >= maxChars) {
      y += lineHeight;
      if (y > M5.Display.height() - lineHeight) return;
      M5.Display.setCursor(x, y);
      lineLen = 0;
      if (c == '\n') continue;
    }
    M5.Display.print(c);
    lineLen++;
  }
}

static void drawStatus(const char *title, const String &line = "") {
  wakeScreen();
  M5.Display.clear(BLACK);
  M5.Display.setTextColor(WHITE, BLACK);
  M5.Display.setTextSize(1);
  M5.Display.setCursor(4, 8);
  M5.Display.println(title);
  M5.Display.setCursor(4, 28);
  drawWrapped(line, 4, 28);
}

static const uint16_t *faceDataFor(const String &state) {
  if (state == "recording") return WOLF_FACE_RECORDING;
  if (state == "waiting-left") return WOLF_FACE_WAITING_LEFT;
  if (state == "waiting-right") return WOLF_FACE_WAITING_RIGHT;
  if (state == "waiting") return WOLF_FACE_WAITING_RIGHT;
  if (state == "happy") return WOLF_FACE_HAPPY;
  if (state == "sad") return WOLF_FACE_SAD;
  return WOLF_FACE_NEUTRAL;
}

static void drawFaceImage(const String &state) {
  // RGB565 arrays are stored as normal 16-bit values in ESP32 little-endian memory.
  // ST7789 expects big-endian pixel bytes, so enable byte swap for correct colors.
  bool oldSwap = M5.Display.getSwapBytes();
  M5.Display.setSwapBytes(true);
  M5.Display.pushImage(0, 0, WOLF_FACE_WIDTH, WOLF_FACE_HEIGHT, faceDataFor(state));
  M5.Display.setSwapBytes(oldSwap);
}

static void drawFace(const String &sentiment) {
  drawFaceImage(sentiment);
}

static void drawReady() {
  wakeScreen();
  M5.Display.clear(BLACK);
  drawFaceImage("neutral");
  M5.Display.setTextSize(2);
  M5.Display.setTextColor(WHITE, BLACK);
  drawWrapped("Hit it", 4, 142, 10, 18);
}

static void drawSentimentResponse(const String &sentimentInput, const String &line = "") {
  String sentiment = sentimentInput;
  if (sentiment != "happy" && sentiment != "neutral" && sentiment != "sad") sentiment = "neutral";
  wakeScreen();
  M5.Display.clear(BLACK);
  drawFace(sentiment);
  M5.Display.setTextSize(2);
  M5.Display.setTextColor(WHITE, BLACK);
  drawWrapped(line, 4, 142, 10, 18);
}

static bool drawRemoteImageUrl(const String &imageUrl, const String &line = "") {
  if (!imageUrl.length()) return false;
  String url = imageUrl.startsWith("http") ? imageUrl : SERVER_BASE_URL + imageUrl;
  HTTPClient http;
  http.begin(url);
  int code = http.GET();
  if (code != 200) {
    http.end();
    return false;
  }
  const int expected = WOLF_FACE_WIDTH * WOLF_FACE_HEIGHT * 2;
  int len = http.getSize();
  if (len != expected) {
    http.end();
    return false;
  }
  uint8_t *bytes = (uint8_t *)ps_malloc(expected);
  if (!bytes) {
    http.end();
    return false;
  }
  WiFiClient *stream = http.getStreamPtr();
  int readTotal = 0;
  uint32_t started = millis();
  while (http.connected() && readTotal < expected && millis() - started < 10000) {
    size_t available = stream->available();
    if (available) {
      int n = stream->readBytes(bytes + readTotal, min((int)available, expected - readTotal));
      readTotal += n;
    } else {
      delay(1);
    }
  }
  http.end();
  if (readTotal != expected) {
    free(bytes);
    return false;
  }
  wakeScreen();
  M5.Display.clear(BLACK);
  bool oldSwap = M5.Display.getSwapBytes();
  M5.Display.setSwapBytes(true);
  M5.Display.pushImage(0, 0, WOLF_FACE_WIDTH, WOLF_FACE_HEIGHT, (uint16_t *)bytes);
  M5.Display.setSwapBytes(oldSwap);
  free(bytes);
  M5.Display.setTextSize(2);
  M5.Display.setTextColor(WHITE, BLACK);
  drawWrapped(line, 4, 142, 10, 18);
  return true;
}

static void drawJobResponse(const String &sentiment, const String &line, const String &imageUrl) {
  if (imageUrl.length() && drawRemoteImageUrl(imageUrl, line)) return;
  drawSentimentResponse(sentiment, line);
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

static bool recordAudioWhileHeld() {
  wakeScreen();
  M5.Display.clear(BLACK);
  drawFaceImage("recording");
  M5.Display.setTextSize(2);
  M5.Display.setTextColor(WHITE, BLACK);
  drawWrapped("Hold to talk\nRelease send", 4, 142, 10, 18);
  M5.Speaker.end(); // StickS3 cannot reliably use mic and speaker together.
  if (!M5.Mic.isEnabled()) {
    if (!M5.Mic.begin()) {
      drawStatus("Mic failed");
      return false;
    }
  }

  recorded_samples = 0;
  static constexpr size_t CHUNK = 256;
  uint32_t started = millis();
  while (recorded_samples < MAX_SAMPLE_COUNT) {
    M5.update();
    if (!M5.BtnA.isPressed() && millis() - started >= MIN_RECORD_MS) {
      break;
    }
    size_t n = min(CHUNK, MAX_SAMPLE_COUNT - recorded_samples);
    if (M5.Mic.record(samples + recorded_samples, n, SAMPLE_RATE)) {
      recorded_samples += n;
      M5.Display.fillRect(4, 62, (recorded_samples * 120) / MAX_SAMPLE_COUNT, 8, GREEN);
    } else {
      delay(1);
    }
  }
  M5.Mic.end();

  uint32_t duration = millis() - started;
  if (duration < MIN_RECORD_MS || recorded_samples < SAMPLE_RATE / 4) {
    drawStatus("Too short", "hold longer");
    delay(1000);
    return false;
  }
  drawStatus("Recorded", String(duration / 1000.0, 1) + " sec");
  return true;
}

static String extractJobId(const String &response) {
  JsonDocument doc;
  DeserializationError err = deserializeJson(doc, response);
  if (err) return "";
  const char *jobId = doc["meta"]["job_id"];
  return jobId ? String(jobId) : "";
}

static bool pollJobResult(const String &jobId);
static String postTextCommand(const String &text);

static void drawOptions(JsonArray options, int selected) {
  wakeScreen();
  M5.Display.fillRect(0, 158, M5.Display.width(), 82, BLACK);
  M5.Display.setTextSize(1);
  M5.Display.setTextColor(WHITE, BLACK);
  M5.Display.setCursor(4, 158);
  M5.Display.println("BtnB select  BtnA OK");
  for (int i = 0; i < (int)options.size() && i < 4; ++i) {
    String opt = options[i].as<String>();
    if (opt.length() > 10) opt = opt.substring(0, 10);
    uint16_t bg = (i == selected) ? BLUE : BLACK;
    uint16_t fg = (i == selected) ? WHITE : YELLOW;
    int y = 172 + i * 17;
    M5.Display.fillRect(2, y - 2, 131, 17, bg);
    M5.Display.setTextSize(2);
    M5.Display.setTextColor(fg, bg);
    M5.Display.setCursor(8, y);
    M5.Display.print(opt);
  }
  M5.Display.setTextSize(1);
}

static String chooseOption(JsonArray options) {
  if (options.size() == 0) return "";
  int selected = 0;
  drawOptions(options, selected);
  uint32_t started = millis();
  while (millis() - started < 60000) {
    M5.update();
    if (M5.BtnB.wasPressed()) {
      selected = (selected + 1) % min((int)options.size(), 4);
      drawOptions(options, selected);
    }
    if (M5.BtnA.wasPressed()) {
      return options[selected].as<String>();
    }
    delay(20);
  }
  return "";
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
      drawJobResponse(sentiment, result, imageUrl);
      if (audioUrl.length()) {
        delay(700);
        playAudioUrl(audioUrl);
        drawJobResponse(sentiment, result, imageUrl);
        // Re-draw response image/face after audio playback.
      }
      JsonArray options = doc["options"].as<JsonArray>();
      String selected = chooseOption(options);
      if (selected.length()) {
        if (selected == "New request") {
          drawReady();
          return true;
        }
        drawStatus("Selected", selected);
        String response = postTextCommand(selected);
        String nextJob = extractJobId(response);
        if (nextJob.length()) {
          pollJobResult(nextJob);
        }
      }
      return true;
    }
    if (status == "failed") {
      String error = doc["error"] | "agent failed";
      drawSentimentResponse("sad", error);
      return false;
    }

    wakeScreen();
    M5.Display.clear(BLACK);
    drawFaceImage(((millis() - start) / 3000) % 2 == 0 ? "waiting-left" : "waiting-right");
    M5.Display.setTextSize(2);
    M5.Display.setTextColor(YELLOW, BLACK);
    drawWrapped(String("Waiting ") + spin[i++ % 4], 4, 142, 10, 18);
    M5.Display.setTextSize(1);
    M5.Display.setTextColor(WHITE, BLACK);
    drawWrapped("Job " + jobId + "\n" + status, 4, 200, 18, 12);
    delay(POLL_INTERVAL_MS);
  }

  drawStatus("Timeout", "Job " + jobId);
  return false;
}

static String postTextCommand(const String &text) {
  drawStatus("Sending", text);
  HTTPClient http;
  http.begin(SERVER_BASE_URL + "/command");
  http.addHeader("Content-Type", "application/json");

  JsonDocument doc;
  doc["device"] = DEVICE_ID;
  doc["event"] = "option_select";
  doc["text"] = text;
  String payload;
  serializeJson(doc, payload);

  int code = http.POST(payload);
  String response = http.getString();
  http.end();
  if (code != 200) {
    drawStatus("Send HTTP", String(code));
  }
  return response;
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
  writeWavHeader(wavHeader, SAMPLE_RATE, recorded_samples);
  const size_t audioBytes = recorded_samples * sizeof(int16_t);
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
  M5.Display.setBrightness(SCREEN_BRIGHTNESS);
  last_screen_activity_ms = millis();
  samples = (int16_t *)ps_malloc(MAX_SAMPLE_COUNT * sizeof(int16_t));
  if (!samples) {
    drawStatus("OOM", "samples");
    while (true) delay(1000);
  }
  connectWiFi();
  drawReady();
}

void loop() {
  M5.update();
  if (!screen_awake && (M5.BtnA.wasPressed() || M5.BtnB.wasPressed())) {
    wakeScreen();
    drawReady();
    delay(250);
    return;
  }
  if (M5.BtnA.wasPressed()) {
    if (WiFi.status() != WL_CONNECTED && !connectWiFi()) {
      delay(1000);
      return;
    }
    // Debounce and let the user hold BtnA to talk.
    delay(80);
    if (recordAudioWhileHeld()) {
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
  sleepScreen();
  delay(10);
}
