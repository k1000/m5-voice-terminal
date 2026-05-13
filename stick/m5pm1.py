# Minimal M5PM1 helper for M5StickS3 MicroPython.
# Enables the L3B power rail used by LCD backlight/MIC/SPK on StickS3.

from machine import I2C, Pin
import time

ADDR = 0x6E
SCL = 48
SDA = 47

REG_GPIO_MODE = 0x10
REG_GPIO_OUT = 0x11
REG_GPIO_DRV = 0x13
REG_I2C_CFG = 0x09
REG_GPIO_FUNC0 = 0x16

GPIO2_L3B_EN = 2
GPIO3_SPK_AMP = 3


def _i2c():
    return I2C(0, scl=Pin(SCL), sda=Pin(SDA), freq=100_000)


def _read_reg(i2c, reg):
    return i2c.readfrom_mem(ADDR, reg, 1)[0]


def _write_reg(i2c, reg, val):
    i2c.writeto_mem(ADDR, reg, bytes([val & 0xFF]))


def _set_bits(i2c, reg, mask, value):
    cur = _read_reg(i2c, reg)
    cur = (cur & ~mask) | (value & mask)
    _write_reg(i2c, reg, cur)


def _gpio_set_func(i2c, pin, func):
    # GPIO0-3 use GPIO_FUNC0, 2 bits each.
    if pin > 3:
        raise ValueError("only GPIO0-3 helper implemented")
    shift = pin * 2
    _set_bits(i2c, REG_GPIO_FUNC0, 0b11 << shift, (func & 0b11) << shift)


def _gpio_set_mode(i2c, pin, output):
    mask = 1 << pin
    _set_bits(i2c, REG_GPIO_MODE, mask, mask if output else 0)


def _gpio_set_drive_pushpull(i2c, pin):
    # 0 = push-pull, 1 = open-drain
    _set_bits(i2c, REG_GPIO_DRV, 1 << pin, 0)


def _gpio_set_output(i2c, pin, high):
    mask = 1 << pin
    _set_bits(i2c, REG_GPIO_OUT, mask, mask if high else 0)


def enable_l3b():
    """Enable StickS3 L3B rail: LCD backlight, microphone, speaker domain.

    M5GFX source shows PYG2 / M5PM1 GPIO2 controls LCD power and should be set HIGH.
    """
    i2c = _i2c()
    _gpio_set_func(i2c, GPIO2_L3B_EN, 0)       # GPIO function
    _gpio_set_mode(i2c, GPIO2_L3B_EN, True)    # output
    _gpio_set_drive_pushpull(i2c, GPIO2_L3B_EN)
    _gpio_set_output(i2c, GPIO2_L3B_EN, True)  # LCD/L3B power on per M5GFX StickS3 init
    _write_reg(i2c, REG_I2C_CFG, 0x00)          # disable PMIC I2C idle sleep
    time.sleep_ms(100)
    return True


def speaker_amp(enable=False):
    """Control speaker amplifier via M5PM1 GPIO3. Default off."""
    i2c = _i2c()
    _gpio_set_func(i2c, GPIO3_SPK_AMP, 0)
    _gpio_set_mode(i2c, GPIO3_SPK_AMP, True)
    _gpio_set_drive_pushpull(i2c, GPIO3_SPK_AMP)
    _gpio_set_output(i2c, GPIO3_SPK_AMP, bool(enable))
    return True
