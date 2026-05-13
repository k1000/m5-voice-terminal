import gc
import json
import time

import machine
import network
try:
    import urequests as requests
except ImportError:
    import requests

import m5pm1
from config import DEVICE_ID, SERVER_URL, WIFI_PASSWORD, WIFI_SSID
from st7789 import BLACK, BLUE, CYAN, GREEN, RED, WHITE, YELLOW, ST7789

BUTTON_A = 11
BUTTON_B = 12
POLL_INTERVAL_MS = 1500
POLL_TIMEOUT_MS = 120000


def server_base_url():
    if SERVER_URL.endswith("/command"):
        return SERVER_URL[:-8]
    if SERVER_URL.endswith("/voice-command"):
        return SERVER_URL[:-14]
    return SERVER_URL.rstrip("/")


def init_display():
    m5pm1.enable_l3b()
    m5pm1.speaker_amp(False)
    display = ST7789(rotation=0)
    display.fill(BLACK)
    return display


def draw_lines(display, title, lines, title_color=CYAN):
    display.fill(BLACK)
    display.text(title[:16], 4, 4, title_color, BLACK)
    y = 20
    for line in lines[:25]:
        display.text(line[:16], 4, y, WHITE, BLACK)
        y += 10
        if y > display.height - 10:
            break


def draw_face(display, sentiment):
    """Draw a simple Wolfenstein-inspired block face: happy, neutral, or sad."""
    color = GREEN if sentiment == "happy" else RED if sentiment == "sad" else YELLOW
    cell = 10
    x0 = (display.width - (7 * cell)) // 2
    y0 = 30
    patterns = {
        "happy": [
            ".......",
            ".#...#.",
            ".#...#.",
            ".......",
            ".#...#.",
            "..###..",
            ".......",
        ],
        "neutral": [
            ".......",
            ".#...#.",
            ".#...#.",
            ".......",
            ".#####.",
            ".......",
            ".......",
        ],
        "sad": [
            ".......",
            ".#...#.",
            ".#...#.",
            ".......",
            "..###..",
            ".#...#.",
            ".......",
        ],
    }
    for row, line in enumerate(patterns.get(sentiment, patterns["neutral"])):
        for col, pixel in enumerate(line):
            if pixel == "#":
                display.fill_rect(x0 + col * cell, y0 + row * cell, cell - 2, cell - 2, color)
    display.rect(x0 - 6, y0 - 6, 7 * cell + 10, 7 * cell + 10, color)


def draw_response(display, sentiment, text):
    sentiment = sentiment if sentiment in ("happy", "neutral", "sad") else "neutral"
    color = GREEN if sentiment == "happy" else RED if sentiment == "sad" else YELLOW
    display.fill(BLACK)
    display.text(("Face " + sentiment)[:16], 4, 4, color, BLACK)
    draw_face(display, sentiment)
    y = 116
    for line in wrap_text(text)[:11]:
        display.text(line[:16], 4, y, WHITE, BLACK)
        y += 10
        if y > display.height - 10:
            break


def wrap_text(text, width=16):
    words = str(text).replace("\n", " ").split(" ")
    lines = []
    line = ""
    for word in words:
        if not word:
            continue
        if len(word) > width:
            if line:
                lines.append(line)
                line = ""
            for i in range(0, len(word), width):
                lines.append(word[i:i + width])
        elif not line:
            line = word
        elif len(line) + 1 + len(word) <= width:
            line += " " + word
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines or [""]


def connect_wifi(display):
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    if not wlan.isconnected():
        draw_lines(display, "WiFi", ["Connecting", WIFI_SSID[:16]], YELLOW)
        wlan.connect(WIFI_SSID, WIFI_PASSWORD)
        for _ in range(30):
            if wlan.isconnected():
                break
            time.sleep(1)
    if wlan.isconnected():
        ip = wlan.ifconfig()[0]
        draw_lines(display, "WiFi OK", [ip, "Press button"], GREEN)
        return wlan
    draw_lines(display, "WiFi failed", ["Check config", "or signal"], RED)
    return wlan


def fetch_json(url):
    response = None
    try:
        response = requests.get(url)
        return response.json()
    finally:
        if response is not None:
            response.close()
        gc.collect()


def poll_job_result(display, job_id):
    base = server_base_url()
    url = base + "/agent/jobs/" + job_id
    started = time.ticks_ms()
    spin = ["|", "/", "-", "\\"]
    i = 0
    while time.ticks_diff(time.ticks_ms(), started) < POLL_TIMEOUT_MS:
        try:
            job = fetch_json(url)
            status = job.get("status", "?")
            if status == "done":
                text = job.get("result_text") or "[done: no text]"
                draw_response(display, job.get("sentiment") or "neutral", text)
                return
            if status == "failed":
                text = job.get("error") or "agent failed"
                draw_response(display, "sad", text)
                return
            draw_lines(display, "Waiting " + spin[i % 4], ["Job " + job_id, status], YELLOW)
            i += 1
        except Exception as exc:
            draw_lines(display, "Poll error", wrap_text(repr(exc)), RED)
            time.sleep_ms(POLL_INTERVAL_MS)
        time.sleep_ms(POLL_INTERVAL_MS)
    draw_lines(display, "Timeout", ["Job " + job_id, "check server"], RED)


def post_command(display):
    draw_lines(display, "Sending", ["Please wait..."], YELLOW)
    payload = {
        "device": DEVICE_ID,
        "event": "button_press",
        "text": "test command from StickS3",
    }
    response = None
    try:
        headers = {"Content-Type": "application/json"}
        response = requests.post(SERVER_URL, data=json.dumps(payload), headers=headers)
        body = response.json()
        meta = body.get("meta", {})
        job_id = meta.get("job_id")
        if job_id:
            draw_lines(display, "Queued", ["Job " + job_id, "Waiting agent"], YELLOW)
            poll_job_result(display, job_id)
        else:
            text = body.get("text", str(body))
            draw_response(display, body.get("sentiment") or "neutral", text)
    except Exception as exc:
        draw_lines(display, "HTTP error", wrap_text(repr(exc)), RED)
    finally:
        if response is not None:
            response.close()
        gc.collect()


def main():
    display = init_display()
    draw_lines(display, "Boot", ["M5 voice", "terminal MVP"], CYAN)
    wlan = connect_wifi(display)
    btn = machine.Pin(BUTTON_A, machine.Pin.IN, machine.Pin.PULL_UP)
    last = 1
    last_press = 0
    while True:
        value = btn.value()
        now = time.ticks_ms()
        if last == 1 and value == 0 and time.ticks_diff(now, last_press) > 800:
            last_press = now
            if not wlan.isconnected():
                wlan = connect_wifi(display)
            if wlan.isconnected():
                post_command(display)
        last = value
        time.sleep_ms(50)


main()
