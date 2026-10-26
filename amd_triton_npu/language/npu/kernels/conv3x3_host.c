// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Host side of tl.extra.npu.conv3x3: the int8 layout changes around a launch,
// threaded with OpenMP (the strided permutes in torch run at a fraction of
// memory bandwidth). Layouts as in conv3x3.py:
//   input   x [ci][h][w] -> [2 halves][2 strips][bands][h+2][ci/16][WPS][8];
//           strip s of band b reads image columns 64b + 32s - 1 + px, px < WPS,
//           zero outside the image and in each band's top and bottom row
//   output  [bands][h+2][n/32][2 strips][4][OWS][8] -> y [n][h][w]; the last
//           2 rows of a band are not output rows

#include <stdint.h>
#include <string.h>

#define OWS 32
#define WPS 40
#define BAND 64

// 8 rows of 8 bytes (one per channel) -> 8 pixels of 8 channels
static inline void transpose8x8(const uint64_t in[8], uint64_t out[8]) {
  uint64_t a[8];
  for (int i = 0; i < 8; ++i)
    a[i] = in[i];
  for (int i = 0; i < 8; ++i) {
    uint64_t v = 0;
    for (int j = 0; j < 8; ++j)
      v |= ((a[j] >> (8 * i)) & 0xff) << (8 * j);
    out[i] = v;
  }
}

void conv3x3_pack(const int8_t *x, int ci, int h, int w, int8_t *out,
                  int threads) {
  const int cg = ci / 16, bands = w / BAND, hp = h + 2;
  const long plane = (long)h * w, rowb = (long)cg * WPS * 8;
#pragma omp parallel for collapse(3) schedule(static) num_threads(threads)
  for (int hs = 0; hs < 4; ++hs)
    for (int b = 0; b < bands; ++b)
      for (int r = 0; r < hp; ++r) {
        int8_t *o = out + (((long)hs * bands + b) * hp + r) * rowb;
        if (r == 0 || r == h + 1) {
          memset(o, 0, rowb);
          continue;
        }
        const int half = hs >> 1, s = hs & 1;
        const int c0 = b * BAND + s * OWS - 1;      // image column of px 0
        const int lo = c0 < 0 ? -c0 : 0;            // first px inside the image
        const int hi = w - c0 < WPS ? w - c0 : WPS; // one past the last
        for (int g = 0; g < cg; ++g) {
          const int8_t *src =
              x + (long)(half * cg + g) * 8 * plane + (long)(r - 1) * w + c0;
          int8_t *og = o + (long)g * WPS * 8;
          memset(og, 0, (size_t)lo * 8);
          memset(og + hi * 8, 0, (size_t)(WPS - hi) * 8);
          int px = lo;
          for (; px + 8 <= hi; px += 8) { // 8 pixels x 8 channels at a time
            uint64_t in[8], t[8];
            for (int c = 0; c < 8; ++c)
              memcpy(&in[c], src + c * plane + px, 8);
            transpose8x8(in, t);
            memcpy(og + px * 8, t, 64);
          }
          for (; px < hi; ++px)
            for (int c = 0; c < 8; ++c)
              og[px * 8 + c] = src[c * plane + px];
        }
      }
}

void conv3x3_unpack(const int8_t *y, int n, int h, int w, int8_t *out,
                    int threads) {
  const int ncol = n / 32, bands = w / BAND, hp = h + 2;
  const long plane = (long)h * w, rowb = (long)n * BAND;
#pragma omp parallel for collapse(2) schedule(static) num_threads(threads)
  for (int b = 0; b < bands; ++b)
    for (int r = 0; r < h; ++r) {
      const int8_t *yr = y + ((long)b * hp + r) * rowb;
      for (int c = 0; c < ncol; ++c)
        for (int g = 0; g < 4; ++g) {
          uint64_t t[8][8]; // [channel][8-pixel chunk]: whole 64-byte lines
          for (int s = 0; s < 2; ++s) {
            const int8_t *blk = yr + ((c * 2 + s) * 4 + g) * OWS * 8;
            for (int k = 0; k < OWS / 8; ++k) {
              uint64_t in[8], tt[8];
              memcpy(in, blk + k * 64, 64);
              transpose8x8(in, tt);
              for (int c8 = 0; c8 < 8; ++c8)
                t[c8][s * 4 + k] = tt[c8];
            }
          }
          int8_t *o =
              out + (long)(c * 32 + g * 8) * plane + (long)r * w + b * BAND;
          for (int c8 = 0; c8 < 8; ++c8)
            memcpy(o + c8 * plane, t[c8], 64);
        }
    }
}
