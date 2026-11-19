// ES8311 mono codec, DAC path only: I2S slave, MCLK = 256 x fs from the MCLK pin, 16-bit samples.
#pragma once
#include "driver/i2c_master.h"
// dev: the codec on an I2C bus (address 0x18 on Waveshare boards). MCLK must already run.
// Returns 0, -1 (no answer), -2 (not an ES8311) or -3 (a register access failed). Sample rate fixed at 24 kHz from 6.144 MHz MCLK.
int es8311_start(i2c_master_dev_handle_t dev, int volume);
int es8311_set_volume(i2c_master_dev_handle_t dev, int volume);     // 0..100
