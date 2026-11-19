// I2S output (standard mode, 16-bit mono, MCLK = 256 x fs) fed by a queue and a player task.
#pragma once
#include <stdint.h>
typedef struct {
    int mclk, bclk, ws, dout;   // GPIO numbers
    int sample_rate;
} i2s_player_pins;

int i2s_player_start(const i2s_player_pins *pins);     // 0 on success (MCLK runs afterwards), else < 0
void i2s_player_play(int16_t *pcm, int n);             // queued; takes ownership (tv_alloc'd)
int i2s_player_pending(void);                          // chunks waiting
void i2s_player_beep(int hz, int ms);
// Called from the player task every 10 ms with the RMS (0..1) of the samples being heard then
// (delayed by the DMA depth; 0 during silence). NULL to stop.
void i2s_player_set_meter(void (*meter)(float rms));
