#include "ws2812.h"
#include <stdlib.h>
#include <string.h>
#include "driver/rmt_tx.h"

struct ws2812 {
    rmt_channel_handle_t channel;
    rmt_encoder_handle_t encoder;
    int count;
    uint8_t *pixels, *sending;      // GRB; `sending` must not change while the RMT reads it
};

ws2812 *ws2812_new(int gpio, int count) {
    ws2812 *s = calloc(1, sizeof *s);
    if (!s) return NULL;
    s->count = count;
    s->pixels = calloc(count, 3);
    s->sending = calloc(count, 3);
    rmt_tx_channel_config_t tc = {.gpio_num = gpio, .clk_src = RMT_CLK_SRC_DEFAULT, .resolution_hz = 10000000,
                                  .mem_block_symbols = 64, .trans_queue_depth = 2};
    // 10 MHz ticks: 0 = 0.3 us high + 0.9 us low, 1 = 0.9 us high + 0.3 us low
    rmt_bytes_encoder_config_t ec = {.bit0 = {.level0 = 1, .duration0 = 3, .level1 = 0, .duration1 = 9},
                                     .bit1 = {.level0 = 1, .duration0 = 9, .level1 = 0, .duration1 = 3},
                                     .flags.msb_first = 1};
    if (!s->pixels || !s->sending || rmt_new_tx_channel(&tc, &s->channel) || rmt_new_bytes_encoder(&ec, &s->encoder) ||
        rmt_enable(s->channel)) {
        if (s->encoder) rmt_del_encoder(s->encoder);
        if (s->channel) rmt_del_channel(s->channel);
        free(s->pixels); free(s->sending); free(s);
        return NULL;
    }
    return s;
}

void ws2812_set(ws2812 *s, int i, uint8_t r, uint8_t g, uint8_t b) {
    if (!s || i < 0 || i >= s->count) return;
    s->pixels[3 * i] = g;
    s->pixels[3 * i + 1] = r;
    s->pixels[3 * i + 2] = b;
}

void ws2812_show(ws2812 *s) {
    if (!s) return;
    rmt_tx_wait_all_done(s->channel, 20);               // the previous frame, and the >50 us reset gap
    memcpy(s->sending, s->pixels, (size_t)s->count * 3);
    rmt_transmit_config_t cfg = {.loop_count = 0};
    rmt_transmit(s->channel, s->encoder, s->sending, (size_t)s->count * 3, &cfg);
}
