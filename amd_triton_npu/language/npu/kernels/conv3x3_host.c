// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Host side of tl.extra.npu.conv3x3: the int8 layout changes around a launch,
// threaded with OpenMP (the strided permutes in torch run at a fraction of
// memory bandwidth). Layouts as in conv3x3.py:
//   input   x [n][ci][h][w] -> [L][2 strips][rows+2][R][CG][WPS][8]
//   output  [rows][8 columns][cpc][2*OGP][OWS][8] -> y [n][co][h][w]
// A plan places the images on the streams. Piece i (9 ints)
//   n r s o line0 nrows row0 col0 width
// covers image n, rows row0 .. row0+nrows-1, columns col0 .. col0+width-1, at
// pixel offset o of partition r, strip s: input lines line0 .. line0+nrows+1
// hold image rows row0-1 .. row0+nrows (zero outside the image), input pixel
// o+t holds image column col0-1+t; output line line0+t, pixel o+u is image
// row row0+t, column col0+u. For the output the plan is also given per image
// row: rowmap[n*h + row][tile] = (line, r, s, o), tiles of OWS columns.

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define OWS 32
#define WPS 40

// 8 rows of 8 bytes -> 8 columns of 8 bytes (pixels <-> channels)
static inline void transpose8x8(const uint64_t in[8], uint64_t out[8]) {
  for (int i = 0; i < 8; ++i) {
    uint64_t v = 0;
    for (int j = 0; j < 8; ++j)
      v |= ((in[j] >> (8 * i)) & 0xff) << (8 * j);
    out[i] = v;
  }
}

void conv3x3_pack(const int8_t *x, int ci, int h, int w, int8_t *out,
                  const int32_t *pieces, int npieces, int L, int CG, int R,
                  int rows, int threads) {
  const long plane = (long)h * w, line = (long)CG * WPS * 8;
  // (piece, input line) work items, so the threads share the lines of a piece
  long count = 0;
  for (int i = 0; i < npieces; ++i)
    count += pieces[9 * i + 5] + 2;
  int32_t *items = malloc(sizeof(int32_t) * 2 * (count ? count : 1));
  for (long i = 0, k = 0; i < npieces; ++i)
    for (int t = 0; t < pieces[9 * i + 5] + 2; ++t, ++k) {
      items[2 * k] = (int32_t)i;
      items[2 * k + 1] = t;
    }
#pragma omp parallel for schedule(static) num_threads(threads)
  for (long k = 0; k < count; ++k) {
    const int32_t *pc = pieces + 9 * items[2 * k];
    const int t = items[2 * k + 1];
    const int n = pc[0], r = pc[1], s = pc[2], o = pc[3], l = pc[4] + t;
    const int ir = pc[6] - 1 + t, c0 = pc[7] - 1, pw = pc[8] + 2;
    // input pixels [lo, hi) are inside the image
    const int lo = c0 < 0 ? -c0 : 0;
    const int hi = w - c0 < pw ? w - c0 : pw;
    for (int p = 0; p < L; ++p)
      for (int g = 0; g < CG; ++g) {
        int8_t *dst = out +
                      (((long)(p * 2 + s) * (rows + 2) + l) * R + r) * line +
                      ((long)g * WPS + o) * 8;
        const int ch0 = (p * CG + g) * 8;
        const int nch = ci - ch0 >= 8 ? 8 : (ci - ch0 > 0 ? ci - ch0 : 0);
        memset(dst, 0, (size_t)pw * 8);
        if (ir < 0 || ir >= h || nch == 0)
          continue;
        const int8_t *src =
            x + ((long)n * ci + ch0) * plane + (long)ir * w + c0;
        int tt = lo;
        if (nch == 8)
          for (; tt + 8 <= hi; tt += 8) { // 8 pixels x 8 channels at a time
            uint64_t in[8], tr[8];
            for (int c = 0; c < 8; ++c)
              memcpy(&in[c], src + c * plane + tt, 8);
            transpose8x8(in, tr);
            memcpy(dst + tt * 8, tr, 64);
          }
        for (; tt < hi; ++tt)
          for (int c = 0; c < nch; ++c)
            dst[tt * 8 + c] = src[c * plane + tt];
      }
  }
  free(items);
}

// Output channels co0 .. co0 + G*16*OGP - 1 of a slab (those below co) to y.
// One work item per image row, so every channel's row is written whole.
void conv3x3_unpack(const int8_t *y, int nimg, int co, int co0, int h, int w,
                    int8_t *out, const int32_t *rowmap, int L, int G, int OGP,
                    int threads) {
  const long plane = (long)h * w;
  const int cpc = 4 / L, ochain = 2 * OGP * OWS * 8, ntiles = (w + 31) / 32;
  const long rowb = 8L * cpc * ochain;
#pragma omp parallel num_threads(threads)
  {
    uint64_t *buf = malloc(sizeof(uint64_t) * ntiles * 4 * 8);
#pragma omp for schedule(static)
    for (long k = 0; k < (long)nimg * h; ++k) {
      const int n = (int)(k / h), ir = (int)(k % h);
      const int32_t *map = rowmap + k * ntiles * 4;
      for (int g = 0; g < G; ++g)
        for (int og8 = 0; og8 < 2 * OGP; ++og8) {
          const int ch = co0 + (g * 2 * OGP + og8) * 8;
          if (ch >= co)
            continue;
          const int nch = co - ch < 8 ? co - ch : 8;
          // gather the row's tiles: buf[c][chunk] = 8 pixels of channel c
          int8_t *row = (int8_t *)buf;
          for (int tl = 0; tl < ntiles; ++tl) {
            const int l = map[4 * tl], r = map[4 * tl + 1];
            const int s = map[4 * tl + 2], o = map[4 * tl + 3];
            const int chain =
                L == 2 ? (r * G + g) * 2 + s : (r * 2 + s) * G + g;
            const int8_t *src =
                y + l * rowb + (long)chain * ochain + ((long)og8 * OWS + o) * 8;
            const int pw = w - tl * 32 < 32 ? w - tl * 32 : 32;
            for (int u = 0; u < pw; u += 8) {
              uint64_t in[8], tr[8];
              memcpy(in, src + u * 8, 64);
              transpose8x8(in, tr);
              for (int c = 0; c < 8; ++c)
                memcpy(row + (long)c * ntiles * 32 + tl * 32 + u, &tr[c], 8);
            }
          }
          int8_t *dst = out + ((long)n * co + ch) * plane + (long)ir * w;
          for (int c = 0; c < nch; ++c)
            memcpy(dst + c * plane, row + (long)c * ntiles * 32, w);
        }
    }
    free(buf);
  }
}
