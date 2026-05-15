#include <M5Unified.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <WebSocketsClient.h>
#include <ArduinoJson.h>
#include "config.h"
#include "wolf_faces.h"
#include "generated_faces.h"

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

// WebSocket job streaming (replaces polling).
static WebSocketsClient wsClient;
static StaticJsonDocument<3072> wsJobDoc;
static volatile bool wsJobDone = false;
static volatile bool wsJobSuccess = false;
static volatile bool wsJobConnected = false;
static uint32_t wsLastActivityMs = 0;
static bool wsBinaryDisplayed = false;  // set true when binary image pushed to display
static int wsImageW = WOLF_FACE_WIDTH;   // width for next binary frame
static int wsImageH = WOLF_FACE_HEIGHT;  // height for next binary frame
static constexpr uint32_t WS_TIMEOUT_MS = 120000;
static constexpr uint32_t WS_PING_INTERVAL_MS = 10000;
static constexpr uint32_t SCREEN_SLEEP_MS = 20000;
static constexpr uint8_t SCREEN_BRIGHTNESS = 255;
static constexpr uint8_t SPEAKER_VOLUME = 200;  // Conservative volume to avoid battery/amp brownout cutoffs.
static uint32_t last_screen_activity_ms = 0;
static bool screen_awake = true;

// ─── Debug logging ────────────────────────────────────────────────────────────
// When stickDebugMode is true, the Stick sends JSON log frames over the
// existing WebSocket connection so the server (and thus the agent) can see
// what is happening on-device.  Toggle with long-press of both BtnA + BtnB.
static bool stickDebugMode = false;
static String wsJobId = "";          // job_id for the current WS connection
static uint32_t stickLogSeq = 0;    // monotonically increasing log sequence
static uint32_t lastTelemetryMs = 0; // last time telemetry was sent (accessible from wsEventHandler)
static constexpr uint32_t TELEMETRY_INTERVAL_MS = 5000;  // emit every 5 seconds

// Send a debug log frame over WebSocket.  level: debug|info|warn|error
// tag: short source identifier (max 8 chars).  msg: message text.
// Only sends when stickDebugMode is true and wsClient is connected.
static void stickLog(const char *level, const char *tag, const String &msg) {
  if (!stickDebugMode || !wsJobConnected || !wsJobId.length()) return;
  JsonDocument doc;
  doc["_stick_log"] = true;
  doc["level"] = level;
  doc["tag"] = tag;
  doc["msg"] = msg;
  doc["seq"] = stickLogSeq++;
  doc["ms"] = millis();
  String out;
  serializeJson(doc, out);
  wsClient.sendTXT(out);
}

// ─── Telemetry: periodic health report ─────────────────────────────────────────
// Gathers and sends a compact telemetry snapshot over the WS channel.
// Sends a {"_stick_telemetry": true, ...} frame with health data.
// Call from loop() whenever stickDebugMode is true.
static void stickTelemetry() {
  if (!stickDebugMode || !wsJobConnected || !wsJobId.length()) return;
  uint32_t now = millis();
  if (now - lastTelemetryMs < TELEMETRY_INTERVAL_MS) return;
  lastTelemetryMs = now;

  int batteryPct = M5.Power.getBatteryLevel();  // -1 = no battery / USB only
  float batteryV = (batteryPct >= 0)
                       ? M5.Power.Axp2101.getBatteryVoltage() : -1.0f;
  uint32_t heapFree = ESP.getFreeHeap();
  uint32_t heapTotal = ESP.getHeapSize();
  size_t psramFree = 0, psramTotal = 0;
  if (ESP.getPsramSize() > 0) {
    psramFree = ESP.getFreePsram();
    psramTotal = ESP.getPsramSize();
  }
  float tempC = temperatureRead();                     // internal sensor, non-calibrated
  int wifiRssi = (WiFi.status() == WL_CONNECTED)
                      ? WiFi.RSSI() : -999;            // dBm, -999 = disconnected

  JsonDocument doc;
  doc["_stick_telemetry"] = true;
  doc["uptime_s"] = now / 1000;
  doc["battery_pct"] = batteryPct;
  doc["battery_v"] = batteryV;
  doc["heap_free"] = heapFree;
  doc["heap_total"] = heapTotal;
  doc["psram_free"] = psramFree;
  doc["psram_total"] = psramTotal;
  doc["temp_c"] = tempC;
  doc["wifi_rssi"] = wifiRssi;
  doc["ms"] = now;
  String out;
  serializeJson(doc, out);
  wsClient.sendTXT(out);
}

// Convenience macros — emit at key points in the flow.
#define STICK_LOG_DEBUG(tag, msg)   stickLog("debug", tag, String(msg))
#define STICK_LOG_INFO(tag, msg)    stickLog("info",  tag, String(msg))
#define STICK_LOG_WARN(tag, msg)    stickLog("warn",  tag, String(msg))
#define STICK_LOG_ERROR(tag, msg)   stickLog("error", tag, String(msg))

static void wakeScreen() {
  // Always restore brightness. Some subsystems/power transitions can leave the
  // panel dim/off while our local screen_awake flag is still true.
  M5.Display.setBrightness(SCREEN_BRIGHTNESS);
  screen_awake = true;
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
  // Custom generated face mapping.
  if (state == "recording") return FACE_SKEPTICAL;
  if (state == "waiting-left") return FACE_SHUSHING;
  if (state == "waiting-right") return FACE_SHUSHING;
  if (state == "waiting") return FACE_SHUSHING;
  if (state == "happy") return FACE_HAPPY;
  if (state == "sad") return FACE_WORRIED;
  return FACE_THINKING;
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
  if (stickDebugMode) {
    M5.Display.setTextSize(1);
    M5.Display.setTextColor(GREEN, BLACK);
    M5.Display.setCursor(M5.Display.width() - 28, 4);
    M5.Display.print("DBG");
  }
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

// ─── RLE Decompressor ─────────────────────────────────────────────────────────
// Decodes the RLE\x01 format produced by server/app.py rle_compress().
// Format: [0x52,0x4C,0x45,0x01] + encoded stream
//   [count, lo, hi]     count >= 2  → repeat 16-bit pixel count times
//   [0x00, N, lo, hi]  escape        → literal N copies of pixel
//
// Caller owns the returned pointer (ps_malloc'd).  Sets *outLen to byte count.
static uint8_t *rleDecode(const uint8_t *data, int dataLen, int expectedPixels, int *outLen) {
  *outLen = 0;
  if (!data || dataLen < 4) return nullptr;
  int offset = 0;
  if (data[0] == 0x52 && data[1] == 0x4C && data[2] == 0x45 && data[3] == 0x01) {
    offset = 4;  // skip RLE\x01 header
  }
  const int expectedBytes = expectedPixels * 2;
  uint8_t *decoded = (uint8_t *)ps_malloc(expectedBytes);
  if (!decoded) return nullptr;
  int di = 0;
  for (int i = offset; i < dataLen && di < expectedBytes; ) {
    uint8_t count = data[i++];
    if (count == 0x00) {
      // Escape: literal single pixel.
      count = data[i++];  // should be 1
      uint8_t lo = data[i++];
      uint8_t hi = data[i++];
      for (uint8_t k = 0; k < count && di < expectedBytes; k++) {
        decoded[di++] = lo;
        decoded[di++] = hi;
      }
    } else {
      // Repeat run.
      uint8_t lo = data[i++];
      uint8_t hi = data[i++];
      for (uint8_t k = 0; k < count && di < expectedBytes; k++) {
        decoded[di++] = lo;
        decoded[di++] = hi;
      }
    }
  }
  *outLen = di;
  return decoded;
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

  const int expected = WOLF_FACE_WIDTH * WOLF_FACE_HEIGHT * 2;  // 36450
  // Download into a buffer.  The compressed stream is smaller but we allocate
  // the full decoded size as a safe upper bound.
  uint8_t *raw = (uint8_t *)ps_malloc(expected);
  if (!raw) {
    http.end();
    return false;
  }

  WiFiClient *stream = http.getStreamPtr();
  int downloaded = 0;
  uint32_t started = millis();
  while (http.connected() && downloaded < expected && millis() - started < 10000) {
    size_t avail = stream->available();
    if (avail) {
      int n = stream->readBytes(raw + downloaded, min((int)avail, expected - downloaded));
      downloaded += n;
    } else {
      delay(1);
    }
  }
  http.end();

  // Decode via shared helper.
  int decodedLen = 0;
  uint8_t *pixels = rleDecode(raw, downloaded, expected, &decodedLen);
  free(raw);
  if (!pixels || decodedLen != expected) {
    if (pixels) free(pixels);
    return false;
  }

  wakeScreen();
  M5.Display.clear(BLACK);
  bool oldSwap = M5.Display.getSwapBytes();
  M5.Display.setSwapBytes(true);
  M5.Display.pushImage(0, 0, WOLF_FACE_WIDTH, WOLF_FACE_HEIGHT, (uint16_t *)pixels);
  M5.Display.setSwapBytes(oldSwap);
  free(pixels);
  M5.Display.setTextSize(2);
  M5.Display.setTextColor(WHITE, BLACK);
  drawWrapped(line, 4, 142, 10, 18);
  return true;
}

static void drawJobResponse(const String &sentiment, const String &line, const String &imageUrl) {
  if (imageUrl.length() && drawRemoteImageUrl(imageUrl, line)) return;
  // If image was already displayed via WS binary frame, just draw text on top.
  // Skip drawSentimentResponse which calls clear(BLACK) and would wipe the image.
  if (wsBinaryDisplayed) {
    wakeScreen();
    M5.Display.setTextSize(2);
    M5.Display.setTextColor(WHITE, BLACK);
    drawWrapped(line, 4, wsImageH + 4, 10, 18);
    return;
  }
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

static bool parseHostPort(const String &url, String &host, uint16_t &port) {
  int start = url.indexOf("://");
  if (start < 0) return false;
  start += 3;
  int colon = url.indexOf(':', start);
  int slash = url.indexOf('/', start);
  if (colon > start) {
    host = url.substring(start, colon);
    String portStr = url.substring(colon + 1, (slash > 0 ? slash : url.length()));
    port = portStr.toInt();
  } else {
    host = url.substring(start, (slash > 0 ? slash : url.length()));
    port = 80;
  }
  return host.length() > 0;
}

static void wsEventHandler(WStype_t type, uint8_t *payload, size_t length) {
  switch (type) {
    case WStype_CONNECTED:
      wsJobConnected = true;
      wsLastActivityMs = millis();
      STICK_LOG_INFO("WS", "connected " + wsJobId);
      lastTelemetryMs = 0;  // force telemetry snapshot on connect
      stickTelemetry();
      break;
    case WStype_DISCONNECTED:
      wsJobConnected = false;
      STICK_LOG_INFO("WS", "disconnected");
      if (!wsJobDone) {
        wsJobDone = true;
        wsJobSuccess = false;
      }
      break;
    case WStype_TEXT: {
      wsLastActivityMs = millis();
      wsJobDoc.clear();
      DeserializationError err = deserializeJson(wsJobDoc, payload, length);
      if (err) break;
      // Extract binary frame dimensions from metadata if present (server sends these
      // just before a binary frame so the Stick knows what size to expect).
      if (wsJobDoc.containsKey("image_w")) wsImageW = wsJobDoc["image_w"].as<int>();
      if (wsJobDoc.containsKey("image_h")) wsImageH = wsJobDoc["image_h"].as<int>();
      const char *status = wsJobDoc["status"] | "";
      if (strcmp(status, "done") == 0 || strcmp(status, "failed") == 0) {
        wsJobDone = true;
        wsJobSuccess = (strcmp(status, "done") == 0);
      }
      break;
    }
    case WStype_BIN: {
      // Binary frame: RLE-compressed or raw RGB565 image from server over WebSocket.
      // wsImageW/H were set from the JSON metadata sent just before this frame.
      wsLastActivityMs = millis();
      const int expectedBytes = wsImageW * wsImageH * 2;
      int decodedLen = 0;
      uint8_t *pixels = rleDecode(payload, length, wsImageW * wsImageH, &decodedLen);
      if (!pixels || decodedLen != expectedBytes) {
        if (pixels) free(pixels);
        break;
      }
      wsBinaryDisplayed = true;
      wakeScreen();
      M5.Display.clear(BLACK);
      bool oldSwap = M5.Display.getSwapBytes();
      M5.Display.setSwapBytes(true);
      M5.Display.pushImage(0, 0, wsImageW, wsImageH, (uint16_t *)pixels);
      M5.Display.setSwapBytes(oldSwap);
      free(pixels);
      if (wsJobDoc.containsKey("result_text")) {
        String result = wsJobDoc["result_text"].as<String>();
        M5.Display.setTextSize(2);
        M5.Display.setTextColor(WHITE, BLACK);
        drawWrapped(result, 4, wsImageH + 4, 10, 18);
      }
      break;
    }
    case WStype_PONG:
      wsLastActivityMs = millis();
      break;
    case WStype_ERROR:
      wsJobConnected = false;
      break;
    default:
      break;
  }
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
static bool streamJobResult(const String &jobId);
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

static uint32_t readLe32(const uint8_t *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint32_t wavDurationMs(const uint8_t *wav, int len) {
  if (!wav || len < 44 || memcmp(wav, "RIFF", 4) != 0 || memcmp(wav + 8, "WAVE", 4) != 0) return 0;
  uint32_t byteRate = readLe32(wav + 28);
  if (byteRate == 0) return 0;
  int pos = 12;
  while (pos + 8 <= len) {
    uint32_t chunkSize = readLe32(wav + pos + 4);
    if (memcmp(wav + pos, "data", 4) == 0) {
      return (uint32_t)(((uint64_t)chunkSize * 1000ULL) / byteRate);
    }
    pos += 8 + chunkSize + (chunkSize & 1);
  }
  return 0;
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

static void drawAudioOverlay(const String &line) {
  wakeScreen();
  M5.Display.fillRect(0, 222, M5.Display.width(), 18, BLACK);
  M5.Display.setTextSize(1);
  M5.Display.setTextColor(CYAN, BLACK);
  M5.Display.setCursor(4, 226);
  M5.Display.print(line.substring(0, 16));
}

static bool playAudioUrl(const String &audioUrl) {
  if (!audioUrl.length()) return false;
  String url = audioUrl.startsWith("http") ? audioUrl : SERVER_BASE_URL + audioUrl;
  drawAudioOverlay("Audio download");

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

  uint32_t durationMs = wavDurationMs(wav, len);
  drawAudioOverlay("Audio playing");
  M5.Mic.end();
  M5.Speaker.begin();
  M5.Speaker.setVolume(SPEAKER_VOLUME);
  bool ok = M5.Speaker.playWav(wav, len, 1, -1, true);
  if (ok) {
    // M5Unified playback is asynchronous and isPlaying() may briefly report
    // false before the decoder/DMA fully drains. Keep the WAV buffer alive for
    // the expected WAV duration so playback is not truncated by free()/end().
    uint32_t started = millis();
    uint32_t holdMs = durationMs ? durationMs + 500 : 3000;
    while (millis() - started < holdMs) {
      M5.update();
      delay(10);
    }
    // Small drain window if the speaker still reports active after duration.
    uint32_t drainStarted = millis();
    while (M5.Speaker.isPlaying() && millis() - drainStarted < 1000) {
      M5.update();
      delay(10);
    }
  }
  M5.Speaker.end();
  free(wav);
  return ok;
}

static bool streamJobResult(const String &jobId) {
  String host;
  uint16_t port = 8010;
  if (!parseHostPort(SERVER_BASE_URL, host, port)) {
    drawStatus("WS parse", SERVER_BASE_URL);
    STICK_LOG_ERROR("WS", "url parse failed: " + SERVER_BASE_URL);
    return false;
  }

  String path = "/ws/jobs/" + jobId;
  wsJobDone = false;
  wsJobSuccess = false;
  wsJobConnected = false;
  wsLastActivityMs = millis();
  wsBinaryDisplayed = false;
  wsJobDoc.clear();
  wsJobId = jobId;
  stickLogSeq = 0;
  STICK_LOG_INFO("WS", "connecting job=" + jobId);

  wsClient.disconnect();
  wsClient.onEvent(wsEventHandler);
  wsClient.begin(host, port, path);
  wsClient.setReconnectInterval(0);  // No auto-reconnect; we handle it.

  const char spin[] = {'|', '/', '-', '\\'};
  uint32_t i = 0;
  uint32_t lastPingMs = millis();
  uint32_t lastWaitDrawMs = 0;
  int lastWaitFaceFrame = -1;
  bool waitingScreenDrawn = false;

  while (!wsJobDone && millis() - wsLastActivityMs < WS_TIMEOUT_MS) {
    wsClient.loop();
    M5.update();

    if (wsJobConnected) {
      // Keep the TCP connection alive.
      if (millis() - lastPingMs >= WS_PING_INTERVAL_MS) {
        wsClient.sendTXT("ping");
        lastPingMs = millis();
      }

      // Waiting screen: do NOT full-clear repeatedly. Full clear + pushImage
      // causes visible blinking on ST7789. Draw once, then update only small
      // text rectangles and replace the face image in-place.
      if (!wsBinaryDisplayed) {
        uint32_t nowMs = millis();
        int faceFrame = (nowMs / 3000) % 2;
        if (!waitingScreenDrawn) {
          waitingScreenDrawn = true;
          wakeScreen();
          M5.Display.clear(BLACK);
          drawFaceImage(faceFrame == 0 ? "waiting-left" : "waiting-right");
          M5.Display.setTextSize(1);
          M5.Display.setTextColor(WHITE, BLACK);
          drawWrapped("Job " + jobId, 4, 200, 18, 12);
          lastWaitFaceFrame = faceFrame;
        }
        if (faceFrame != lastWaitFaceFrame) {
          lastWaitFaceFrame = faceFrame;
          drawFaceImage(faceFrame == 0 ? "waiting-left" : "waiting-right");
        }
        if (nowMs - lastWaitDrawMs >= 250) {
          lastWaitDrawMs = nowMs;
          M5.Display.fillRect(0, 142, M5.Display.width(), 42, BLACK);
          M5.Display.setTextSize(2);
          M5.Display.setTextColor(YELLOW, BLACK);
          drawWrapped(String("Thinking ") + spin[i++ % 4], 4, 142, 10, 18);
        }
      }
    }
    delay(20);
  }

  wsClient.disconnect();
  STICK_LOG_INFO("WS", "loop done wsJobDone");

  if (!wsJobDone) {
    STICK_LOG_ERROR("WS", "timeout after 120s");
    drawStatus("WS timeout", "Job " + jobId);
    wsJobId = "";
    return false;
  }

  if (!wsJobSuccess) {
    String error = wsJobDoc["error"] | "agent failed";
    STICK_LOG_ERROR("JOB", "failed: " + error);
    drawSentimentResponse("sad", error);
    wsJobId = "";
    return false;
  }

  // ---- Job succeeded ----
  // If binary image was received over WebSocket, it was already displayed by
  // wsEventHandler.  Pass empty imageUrl so we skip the HTTP GET path.
  String result = wsJobDoc["result_text"] | "[done: no text]";
  String sentiment = wsJobDoc["sentiment"] | "neutral";
  String audioUrl = wsJobDoc["audio_url"] | "";
  // imageUrl is deliberately omitted when wsBinaryDisplayed to avoid re-fetch.
  String displayImageUrl = wsBinaryDisplayed ? String("") : (wsJobDoc["image_url"] | String(""));

  STICK_LOG_INFO("JOB", "done sentiment=" + sentiment + " img=" + (wsBinaryDisplayed ? "ws" : displayImageUrl.length() ? "http" : "none"));

  drawJobResponse(sentiment, result, displayImageUrl);
  if (audioUrl.length()) {
    delay(700);
    STICK_LOG_INFO("AUDIO", "playing " + audioUrl);
    playAudioUrl(audioUrl);
    STICK_LOG_INFO("AUDIO", "done");
    drawJobResponse(sentiment, result, displayImageUrl);
  }

  wsJobId = "";

  JsonArray options = wsJobDoc["options"].as<JsonArray>();
  String selected = chooseOption(options);
  if (selected.length()) {
    if (selected == "New request") {
      drawReady();
      return true;
    }
    drawStatus("Selected", selected);
    STICK_LOG_INFO("MENU", "selected: " + selected);
    String response = postTextCommand(selected);
    String nextJob = extractJobId(response);
    if (nextJob.length()) {
      return streamJobResult(nextJob);
    }
  }
  return true;
}

static bool pollJobResult(const String &jobId) {
  // Polling fallback: used when WebSocket stream fails.  Set USE_POLLING_FALLBACK
  // to 1 in config.h to force polling instead of WebSocket.
#ifdef USE_POLLING_FALLBACK
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
#else
  (void)jobId;  // unused without polling
  return false;
#endif
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
  STICK_LOG_INFO("SYS", "setup done");
  drawReady();
}

// Toggle debug mode when both buttons are held for 3 seconds.
static void checkDebugToggle() {
  static uint32_t bothHeldSince = 0;
  if (M5.BtnA.isPressed() && M5.BtnB.isPressed()) {
    if (bothHeldSince == 0) bothHeldSince = millis();
    if (millis() - bothHeldSince >= 3000) {
      stickDebugMode = !stickDebugMode;
      bothHeldSince = 0;
      wakeScreen();
      M5.Display.clear(BLACK);
      M5.Display.setTextSize(2);
      M5.Display.setTextColor(stickDebugMode ? GREEN : RED, BLACK);
      M5.Display.setCursor(4, 40);
      M5.Display.print(stickDebugMode ? "DEBUG ON" : "DEBUG OFF");
      M5.Display.setTextSize(1);
      M5.Display.setTextColor(WHITE, BLACK);
      M5.Display.setCursor(4, 70);
      M5.Display.print("Hold both 3s to toggle");
      delay(1500);
      drawReady();
    }
  } else {
    bothHeldSince = 0;
  }
}

void loop() {
  M5.update();
  checkDebugToggle();
  stickTelemetry();  // sends periodic telemetry every TELEMETRY_INTERVAL_MS when debug mode is on

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
    // Emit telemetry snapshot right before recording starts.
    lastTelemetryMs = 0;  // force immediate telemetry on next loop
    stickTelemetry();
    if (recordAudioWhileHeld()) {
      STICK_LOG_INFO("MIC", "recorded " + String(recorded_samples) + " samples");
      String response = postAudio();
      String jobId = extractJobId(response);
      if (jobId.length()) {
        STICK_LOG_INFO("HTTP", "queued job=" + jobId);
        drawStatus("Queued", "Job " + jobId);
        if (!streamJobResult(jobId)) {
          STICK_LOG_WARN("WS", "falling back to polling");
          // Fall back to polling if WebSocket stream fails.
          pollJobResult(jobId);
        }
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
