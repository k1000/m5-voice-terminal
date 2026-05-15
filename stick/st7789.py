# Minimal pure-Python ST7789 driver for M5StickS3 MicroPython.
# Pins from M5Stack docs: MOSI=39, SCK=40, DC/RS=45, CS=41, RST=21, BL=38.

from machine import Pin, SPI
import time

try:
    import framebuf
except ImportError:
    framebuf = None


# ST7789 commands
_SWRESET = 0x01
_SLPOUT = 0x11
_NORON = 0x13
_INVOFF = 0x20
_INVON = 0x21
_DISPON = 0x29
_CASET = 0x2A
_RASET = 0x2B
_RAMWR = 0x2C
_MADCTL = 0x36
_COLMOD = 0x3A

# MADCTL bits
_MADCTL_MY = 0x80
_MADCTL_MX = 0x40
_MADCTL_MV = 0x20
_MADCTL_BGR = 0x08


def color565(r, g, b):
    """Return RGB888 as RGB565 integer."""
    return ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)


BLACK = color565(0, 0, 0)
WHITE = color565(255, 255, 255)
RED = color565(255, 0, 0)
GREEN = color565(0, 255, 0)
BLUE = color565(0, 0, 255)
YELLOW = color565(255, 255, 0)
CYAN = color565(0, 255, 255)
MAGENTA = color565(255, 0, 255)
ORANGE = color565(255, 128, 0)


class ST7789:
    """Small ST7789 LCD driver suitable for M5StickS3.

    Defaults target the StickS3 135x240 panel in portrait orientation.
    If the image appears offset/rotated, try set_rotation(1), set_rotation(2), etc.
    """

    def __init__(
        self,
        spi=None,
        width=135,
        height=240,
        sck=40,
        mosi=39,
        dc=45,
        cs=41,
        rst=21,
        bl=38,
        baudrate=40_000_000,
        rotation=0,
    ):
        self.spi = spi or SPI(1, baudrate=baudrate, polarity=0, phase=0, sck=Pin(sck), mosi=Pin(mosi))
        self.dc = Pin(dc, Pin.OUT, value=0)
        self.cs = Pin(cs, Pin.OUT, value=1)
        self.rst = Pin(rst, Pin.OUT, value=1) if rst is not None else None
        self.bl = Pin(bl, Pin.OUT, value=0) if bl is not None else None
        self.base_width = width
        self.base_height = height
        self.width = width
        self.height = height
        self.xstart = 52
        self.ystart = 40
        self.rotation = rotation
        self.init()

    def hard_reset(self):
        if self.rst is None:
            return
        self.rst.value(1)
        time.sleep_ms(50)
        self.rst.value(0)
        time.sleep_ms(50)
        self.rst.value(1)
        time.sleep_ms(150)

    def backlight(self, on=True):
        if self.bl is not None:
            self.bl.value(1 if on else 0)

    def write_cmd(self, cmd, data=None):
        self.cs.value(0)
        self.dc.value(0)
        self.spi.write(bytes([cmd]))
        if data is not None:
            self.dc.value(1)
            self.spi.write(data if isinstance(data, (bytes, bytearray)) else bytes(data))
        self.cs.value(1)

    def init(self):
        self.hard_reset()
        self.write_cmd(_SWRESET)
        time.sleep_ms(150)
        self.write_cmd(_SLPOUT)
        time.sleep_ms(120)
        self.write_cmd(_COLMOD, b"\x55")  # 16-bit RGB565
        time.sleep_ms(10)
        self.set_rotation(self.rotation)
        self.write_cmd(_INVON)  # common for ST7789 IPS panels
        self.write_cmd(_NORON)
        time.sleep_ms(10)
        self.write_cmd(_DISPON)
        time.sleep_ms(120)
        self.backlight(True)

    def set_rotation(self, rotation):
        self.rotation = rotation & 3
        if self.rotation == 0:
            madctl = _MADCTL_BGR
            self.width, self.height = self.base_width, self.base_height
            self.xstart, self.ystart = 52, 40
        elif self.rotation == 1:
            madctl = _MADCTL_MX | _MADCTL_MV | _MADCTL_BGR
            self.width, self.height = self.base_height, self.base_width
            self.xstart, self.ystart = 40, 52
        elif self.rotation == 2:
            madctl = _MADCTL_MX | _MADCTL_MY | _MADCTL_BGR
            self.width, self.height = self.base_width, self.base_height
            self.xstart, self.ystart = 53, 40
        else:
            madctl = _MADCTL_MY | _MADCTL_MV | _MADCTL_BGR
            self.width, self.height = self.base_height, self.base_width
            self.xstart, self.ystart = 40, 53
        self.write_cmd(_MADCTL, bytes([madctl]))

    def set_window(self, x0, y0, x1, y1):
        x0 += self.xstart
        x1 += self.xstart
        y0 += self.ystart
        y1 += self.ystart
        self.write_cmd(_CASET, bytes([x0 >> 8, x0 & 0xFF, x1 >> 8, x1 & 0xFF]))
        self.write_cmd(_RASET, bytes([y0 >> 8, y0 & 0xFF, y1 >> 8, y1 & 0xFF]))
        self.write_cmd(_RAMWR)

    def write_pixels(self, data):
        self.cs.value(0)
        self.dc.value(1)
        self.spi.write(data)
        self.cs.value(1)

    def fill(self, color=BLACK):
        self.fill_rect(0, 0, self.width, self.height, color)

    def fill_rect(self, x, y, w, h, color):
        if w <= 0 or h <= 0:
            return
        if x < 0:
            w += x
            x = 0
        if y < 0:
            h += y
            y = 0
        if x + w > self.width:
            w = self.width - x
        if y + h > self.height:
            h = self.height - y
        if w <= 0 or h <= 0:
            return
        self.set_window(x, y, x + w - 1, y + h - 1)
        hi = color >> 8
        lo = color & 0xFF
        chunk_pixels = min(w * h, 512)
        chunk = bytes([hi, lo]) * chunk_pixels
        remaining = w * h
        while remaining:
            n = min(remaining, chunk_pixels)
            self.write_pixels(chunk[: n * 2])
            remaining -= n

    def pixel(self, x, y, color):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.set_window(x, y, x, y)
            self.write_pixels(bytes([color >> 8, color & 0xFF]))

    def hline(self, x, y, w, color):
        self.fill_rect(x, y, w, 1, color)

    def vline(self, x, y, h, color):
        self.fill_rect(x, y, 1, h, color)

    def rect(self, x, y, w, h, color):
        self.hline(x, y, w, color)
        self.hline(x, y + h - 1, w, color)
        self.vline(x, y, h, color)
        self.vline(x + w - 1, y, h, color)

    def text(self, text, x, y, fg=WHITE, bg=None):
        """Draw 8x8 MicroPython framebuf text."""
        if framebuf is None:
            raise RuntimeError("framebuf module unavailable")
        text = str(text)
        w = max(8, len(text) * 8)
        h = 8
        mono = bytearray((w * h + 7) // 8)
        fb = framebuf.FrameBuffer(mono, w, h, framebuf.MONO_HLSB)
        fb.fill(0)
        fb.text(text, 0, 0, 1)

        # Build one RGB565 rectangle and send in a single transaction.
        out = bytearray(w * h * 2)
        for yy in range(h):
            for xx in range(w):
                pix_on = fb.pixel(xx, yy)
                if pix_on:
                    c = fg
                elif bg is not None:
                    c = bg
                else:
                    continue
                if bg is None and not pix_on:
                    continue
                i = (yy * w + xx) * 2
                out[i] = c >> 8
                out[i + 1] = c & 0xFF

        if bg is None:
            # Transparent mode: only plot foreground pixels.
            for yy in range(h):
                for xx in range(w):
                    if fb.pixel(xx, yy):
                        self.pixel(x + xx, y + yy, fg)
        else:
            self.set_window(x, y, min(x + w - 1, self.width - 1), min(y + h - 1, self.height - 1))
            self.write_pixels(out)


def sticks3_display(rotation=0, baudrate=40_000_000):
    """Construct an ST7789 with M5StickS3 pins."""
    return ST7789(rotation=rotation, baudrate=baudrate)


# ─── RLE Decompressor ────────────────────────────────────────────────────────
# Decodes the RLE\x01 format produced by server/app.py rle_compress().
# Format: b"RLE\x01" + encoded stream
#   [count, lo, hi]      count >= 2  → repeat 16-bit pixel (hi<<8|lo) count times
#   [0x00, 1, lo, hi]   escape       → literal single pixel

def rle_decompress(data, width, height):
    """Decompress RLE\x01 data into a flat bytearray of RGB565 pixels (little-endian).

    Args:
        data: bytes with b"RLE\x01" prefix followed by encoded stream
        width: image width in pixels
        height: image height in pixels

    Returns:
        bytearray of (width * height * 2) bytes in RGB565 LE order
    """
    expected_len = width * height * 2
    out = bytearray(expected_len)
    out_idx = 0

    # Skip RLE\x01 header if present.
    if data[:4] == b"RLE\x01":
        data = data[4:]

    i = 0
    data_len = len(data)
    while i < data_len and out_idx < expected_len:
        count = data[i]
        i += 1
        if count == 0:
            # Escape: literal single pixel. Next byte is count (always 1).
            literal_count = data[i]
            i += 1
            for _ in range(literal_count):
                if out_idx >= expected_len:
                    break
                lo = data[i]
                hi = data[i + 1]
                i += 2
                out[out_idx] = lo
                out[out_idx + 1] = hi
                out_idx += 2
        else:
            # Repeat run: next 2 bytes are the pixel (lo, hi).
            lo = data[i]
            hi = data[i + 1]
            i += 2
            for _ in range(count):
                if out_idx >= expected_len:
                    break
                out[out_idx] = lo
                out[out_idx + 1] = hi
                out_idx += 2

    return out[:out_idx]  # trim if overproduced


def blit_rle(display, data, x=0, y=0, width=135, height=135):
    """Decompress RLE data and blit it to the display at (x, y).

    Works for both RLE-compressed (b"RLE\x01" header) and raw RGB565 data.
    """
    pixels = rle_decompress(data, width, height) if data[:4] == b"RLE\x01" else bytearray(data[:width * height * 2])
    display.set_window(x, y, x + width - 1, y + height - 1)
    display.write_pixels(pixels)
