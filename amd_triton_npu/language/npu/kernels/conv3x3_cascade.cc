// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// 3x3 stride-1 int8 convolution, one output line per call, for a chain of
// AIE2P cores that splits the input channels and passes partial sums down the
// cascade (tl.extra.npu.conv3x3).
//
// Every core of a chain holds the weights of its input-channel slice for the
// chain's output channels and sees the same three input lines (its channel
// slice). Per block of 16 output pixels x 16 output channels it reduces 3 ky x
// CG channel groups x 3 kx into four 8x8 int32 accumulators, which never leave
// the registers:
//   head  starts the block with mul, sends the accumulators down the cascade
//   mid   starts from the accumulators coming in on the cascade, sends them on
//   tail  starts from the cascade, shift-round-saturates to int8 (signed, or
//         unsigned = fused ReLU)
//
// Layouts are 8-channel interleaved, so an 8-pixel x 8-channel operand is 64
// contiguous bytes:
//   line*  [CG][WP][8] int8, the input line already padded: output pixel x
//          reads input pixels x .. x+2
//   W      per output-channel pair, in consumption order
//          [OGP][3 ky][CG][3 kx][2 og][8 ci][8 co], then 64 bytes whose first
//          int32 is the tail's requantization shift (runtime data, so one
//          build serves every shift)
//   O      [2*OGP][OW][8] int8 (tail only)
// The kx = 1, 2 windows are built from aligned loads with VSHIFT, which issues
// beside the VMACs.
//
// Built with Peano (llvm-aie) for aie2p; CONV_CG, CONV_WP, CONV_OW and
// CONV_OGP set the geometry.

#define UNROLL_FULL _Pragma("clang loop unroll(full)")
#define PRAGMA_STR(x) #x
#define MIN_ITERS(x) _Pragma(PRAGMA_STR(clang loop min_iteration_count(x)))

enum { HEAD = 0, MID = 1, TAIL = 2 };

template <int CG, int WP, int OW, int OGP, int ROLE, bool RELU>
static inline void
conv3x3_line(const v64int8 *__restrict line0, const v64int8 *__restrict line1,
             const v64int8 *__restrict line2, const v64int8 *__restrict W,
             v64int8 *__restrict O) {
  static_assert(WP % 8 == 0 && OW % 16 == 0 && WP >= OW + 8, "line geometry");
  constexpr int NB = OW / 16; // 16-pixel blocks per line
  constexpr int LV = WP / 8;  // 64-byte vectors per channel group in a line
  // weight vectors per output-channel pair
  constexpr int WSTEP = 3 * CG * 3 * 2;
  const int shift = *(const int *)(W + OGP * WSTEP);
  if constexpr (ROLE == TAIL) {
    // Saturate and round half up, whatever the caller left. crSat is control
    // register 0 on AIE2P; mlir-aie's core entry writes AIE2's index (9), so
    // saturation is not otherwise on.
    set_sat();
    set_rnd(9);
  }
  const v64int8 *__restrict lines[3] = {line0, line1, line2};
  const v64int8 *__restrict wog = W;
  int pb = 0, og = 0;
  MIN_ITERS(2)
  for (int it = 0; it < OGP * NB; ++it) {
    const v64int8 *__restrict wb = wog;
    v64acc32 c00, c01, c10, c11;  // [pixel group][output-channel group]
    if constexpr (ROLE != HEAD) { // partial sums from the core above
      c00 = get_scd_v64acc32();
      c01 = get_scd_v64acc32();
      c10 = get_scd_v64acc32();
      c11 = get_scd_v64acc32();
    }
    UNROLL_FULL
    for (int ky = 0; ky < 3; ++ky) {
      const v64int8 *__restrict xa = lines[ky] + pb * 2;
      UNROLL_FULL
      for (int cg = 0; cg < CG; ++cg) {
        const v64int8 v0 = xa[0], v1 = xa[1], v2 = xa[2];
        xa += LV;
        // pixel group 0 = v0 (+1, +2 pixel shifts), group 1 = v1 (+1, +2)
        const v64int8 p0[3] = {v0, shift_bytes(v0, v1, 8),
                               shift_bytes(v0, v1, 16)};
        const v64int8 p1[3] = {v1, shift_bytes(v1, v2, 8),
                               shift_bytes(v1, v2, 16)};
        UNROLL_FULL
        for (int kx = 0; kx < 3; ++kx) {
          const v64int8 w0 = wb[0], w1 = wb[1];
          wb += 2;
          if (ROLE == HEAD && ky == 0 && cg == 0 && kx == 0) {
            c00 = mul_8x8_8x8(p0[kx], w0);
            c01 = mul_8x8_8x8(p0[kx], w1);
            c10 = mul_8x8_8x8(p1[kx], w0);
            c11 = mul_8x8_8x8(p1[kx], w1);
          } else {
            c00 = mac_8x8_8x8(p0[kx], w0, c00);
            c01 = mac_8x8_8x8(p0[kx], w1, c01);
            c10 = mac_8x8_8x8(p1[kx], w0, c10);
            c11 = mac_8x8_8x8(p1[kx], w1, c11);
          }
        }
      }
    }
    if constexpr (ROLE == TAIL) {
      // output channel group 2*og + j, pixels 16*pb .. +15
      v64int8 *__restrict o0 = O + (2 * og) * (OW / 8) + pb * 2;
      v64int8 *__restrict o1 = o0 + OW / 8;
      if constexpr (RELU) { // sign 0: unsigned saturation to [0, 255]
        o0[0] = (v64int8)to_v64uint8(c00, shift, 0);
        o0[1] = (v64int8)to_v64uint8(c10, shift, 0);
        o1[0] = (v64int8)to_v64uint8(c01, shift, 0);
        o1[1] = (v64int8)to_v64uint8(c11, shift, 0);
      } else {
        o0[0] = to_v64int8(c00, shift, 1);
        o0[1] = to_v64int8(c10, shift, 1);
        o1[0] = to_v64int8(c01, shift, 1);
        o1[1] = to_v64int8(c11, shift, 1);
      }
    } else { // on to the core below
      put_mcd(extract_v16acc32(c00, 0), 1);
      put_mcd(extract_v16acc32(c00, 1), 1);
      put_mcd(extract_v16acc32(c00, 2), 1);
      put_mcd(extract_v16acc32(c00, 3), 1);
      put_mcd(extract_v16acc32(c01, 0), 1);
      put_mcd(extract_v16acc32(c01, 1), 1);
      put_mcd(extract_v16acc32(c01, 2), 1);
      put_mcd(extract_v16acc32(c01, 3), 1);
      put_mcd(extract_v16acc32(c10, 0), 1);
      put_mcd(extract_v16acc32(c10, 1), 1);
      put_mcd(extract_v16acc32(c10, 2), 1);
      put_mcd(extract_v16acc32(c10, 3), 1);
      put_mcd(extract_v16acc32(c11, 0), 1);
      put_mcd(extract_v16acc32(c11, 1), 1);
      put_mcd(extract_v16acc32(c11, 2), 1);
      put_mcd(extract_v16acc32(c11, 3), 1);
    }
    // Next block: 16 pixels on, or the next output-channel pair. The weight
    // pointer moving with the loop variable keeps LLVM from hoisting the pair's
    // weight loads out of the loop (they do not fit the registers and would be
    // spilled to the stack).
    const bool wrap = pb == NB - 1;
    pb = wrap ? 0 : pb + 1;
    og = wrap ? og + 1 : og;
    wog = wrap ? wog + WSTEP : wog;
  }
}

#ifndef CONV_CG
#define CONV_CG 16 // input channels of this core's slice / 8
#endif
#ifndef CONV_WP
#define CONV_WP 40 // padded input line width in pixels
#endif
#ifndef CONV_OW
#define CONV_OW 32 // output line width
#endif
#ifndef CONV_OGP
#define CONV_OGP 2 // output channels / 16
#endif

#define CONV_ARGS                                                              \
  const v64int8 *l0, const v64int8 *l1, const v64int8 *l2, const v64int8 *w,   \
      v64int8 *o

extern "C" {
void conv3x3_head(CONV_ARGS) {
  conv3x3_line<CONV_CG, CONV_WP, CONV_OW, CONV_OGP, HEAD, false>(l0, l1, l2, w,
                                                                 o);
}
void conv3x3_mid(CONV_ARGS) {
  conv3x3_line<CONV_CG, CONV_WP, CONV_OW, CONV_OGP, MID, false>(l0, l1, l2, w,
                                                                o);
}
void conv3x3_tail(CONV_ARGS) {
  conv3x3_line<CONV_CG, CONV_WP, CONV_OW, CONV_OGP, TAIL, false>(l0, l1, l2, w,
                                                                 o);
}
void conv3x3_tail_relu(CONV_ARGS) {
  conv3x3_line<CONV_CG, CONV_WP, CONV_OW, CONV_OGP, TAIL, true>(l0, l1, l2, w,
                                                                o);
}
}
