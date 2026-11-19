// ES8311 setup for playback (register values from Espressif's es8311 driver in esp-bsp).
#include "es8311.h"
#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static esp_err_t wr(i2c_master_dev_handle_t dev, uint8_t reg, uint8_t value) {
    uint8_t buf[2] = {reg, value};
    return i2c_master_transmit(dev, buf, 2, 100);
}

static esp_err_t rd(i2c_master_dev_handle_t dev, uint8_t reg, uint8_t *value) {
    return i2c_master_transmit_receive(dev, &reg, 1, value, 1, 100);
}

// reg = (reg & keep) | set, written only when the read worked (else a garbage value would be).
static esp_err_t update(i2c_master_dev_handle_t dev, uint8_t reg, uint8_t keep, uint8_t set) {
    uint8_t v;
    esp_err_t err = rd(dev, reg, &v);
    return err ? err : wr(dev, reg, (uint8_t)((v & keep) | set));
}

int es8311_set_volume(i2c_master_dev_handle_t dev, int volume) {
    if (volume < 0) volume = 0;
    if (volume > 100) volume = 100;
    return wr(dev, 0x32, volume == 0 ? 0 : volume * 256 / 100 - 1) ? -1 : 0;
}

int es8311_start(i2c_master_dev_handle_t dev, int volume) {
    uint8_t id1 = 0, id2 = 0;
    if (rd(dev, 0xFD, &id1) || rd(dev, 0xFE, &id2)) return -1;
    printf("audio: ES8311 chip id %02x%02x\n", id1, id2);
    if (id1 != 0x83 || id2 != 0x11) return -2;
    esp_err_t err = wr(dev, 0x00, 0x1F);
    vTaskDelay(pdMS_TO_TICKS(20));
    err |= wr(dev, 0x00, 0x00);
    err |= wr(dev, 0x00, 0x80);                           // power on
    err |= wr(dev, 0x01, 0x3F);                           // all clocks on, MCLK from pin
    err |= update(dev, 0x06, 0xDF, 0x00);                 // BCLK not inverted
    // 6.144 MHz / 24 kHz: pre_div 1, mult 1x, adc/dac div 1, single speed, lrck 0x00ff, osr 0x10
    err |= update(dev, 0x02, 0x07, 0x00);
    err |= wr(dev, 0x03, 0x10);
    err |= wr(dev, 0x04, 0x10);
    err |= wr(dev, 0x05, 0x00);
    err |= update(dev, 0x06, 0xE0, 4 - 1);
    err |= update(dev, 0x07, 0xC0, 0x00);
    err |= wr(dev, 0x08, 0xFF);
    err |= update(dev, 0x00, 0xBF, 0x00);                 // slave serial port
    err |= wr(dev, 0x09, 0x0C);                           // SDP in: I2S, 16-bit
    err |= wr(dev, 0x0A, 0x0C);                           // SDP out: I2S, 16-bit
    err |= wr(dev, 0x0D, 0x01);                           // analog power up
    err |= wr(dev, 0x0E, 0x02);
    err |= wr(dev, 0x12, 0x00);                           // DAC power up
    err |= wr(dev, 0x13, 0x10);                           // output to HP drive
    err |= wr(dev, 0x1C, 0x6A);
    err |= wr(dev, 0x37, 0x08);                           // bypass DAC equalizer
    err |= es8311_set_volume(dev, volume) ? ESP_FAIL : ESP_OK;
    err |= update(dev, 0x31, 0x9F, 0x00);                 // unmute
    return err ? -3 : 0;
}
