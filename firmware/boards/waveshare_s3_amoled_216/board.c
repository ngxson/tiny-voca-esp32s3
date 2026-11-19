// Waveshare ESP32-S3-Touch-AMOLED-2.16: ES8311 DAC over I2S, speaker amplifier enabled by GPIO 46.
// Pins (Waveshare BSP esp32_s3_touch_amoled_2_16, schematic): I2C SDA 15, SCL 14; I2S MCLK 42,
// BCLK 9, WS 45, DOUT 8; PA 46. The AXP2101 power-on defaults already feed the codec.
// The AMOLED screen is not used yet.
#include "board.h"
#include <stdio.h>
#include "driver/gpio.h"
#include "driver/i2c_master.h"
#include "es8311.h"
#include "i2s_player.h"

#define ES8311_ADDR 0x18
#define PA_PIN 46

static i2c_master_bus_handle_t bus;
static i2c_master_dev_handle_t codec;

const char *board_name(void) { return "waveshare_s3_amoled_216"; }

int board_init(int volume) {
    i2c_master_bus_config_t bc = {.i2c_port = 0, .sda_io_num = 15, .scl_io_num = 14, .clk_source = I2C_CLK_SRC_DEFAULT,
                                  .glitch_ignore_cnt = 7, .flags.enable_internal_pullup = true};
    if (i2c_new_master_bus(&bc, &bus)) return -1;
    i2c_device_config_t dc = {.dev_addr_length = I2C_ADDR_BIT_LEN_7, .device_address = ES8311_ADDR, .scl_speed_hz = 100000};
    if (i2c_master_bus_add_device(bus, &dc, &codec)) return -1;
    i2s_player_pins pins = {.mclk = 42, .bclk = 9, .ws = 45, .dout = 8, .sample_rate = 24000};
    if (i2s_player_start(&pins)) return -2;                 // MCLK must run before the codec is set up
    int rc = es8311_start(codec, volume);
    if (rc) { printf("audio: ES8311 init failed (%d)\n", rc); codec = NULL; return -3; }
    gpio_reset_pin(PA_PIN);
    gpio_set_direction(PA_PIN, GPIO_MODE_OUTPUT);
    gpio_set_level(PA_PIN, 1);
    printf("audio: ready (24 kHz, volume %d)\n", volume);
    return 0;
}

void board_play(int16_t *pcm, int n) { i2s_player_play(pcm, n); }
int board_set_volume(int volume) { return codec ? es8311_set_volume(codec, volume) : -1; }
void board_beep(int hz, int ms) { i2s_player_beep(hz, ms); }
void board_show_text(const char *text) { (void)text; }   // screen not driven yet
