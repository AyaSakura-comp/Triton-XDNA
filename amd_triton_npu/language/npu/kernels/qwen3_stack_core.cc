// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// tl.extra.npu.qwen3_stack on the AIE2P array: the per-core functions of one
// launch through a Qwen3-0.6B-sized stack (hidden 1024, 16 query and 8 kv
// heads of 128, MLP 3072) and its linear head.
//
// Every core runs the same static schedule (per layer QKV GEMM, attention,
// O GEMM, gate/up GEMM, down GEMM, then the head); the IRON design
// (qwen3_stack_design.py) calls these between its object FIFO acquires.
// Operands are bfp16 (EBS8), accumulation fp32.
//
// Rows come in groups of 128 (positions 64g..64g+63 of sequence 0, then the
// same positions of sequence 1);
// each GEMM phase runs once per group (its weights stream once per group) or,
// gate/up and the heads, once per pair of groups (the cores hold both groups'
// A slices and use each weight object for both); the attention runs once per
// layer over every group's keys.
//
// QKV post-pass (between the QKV GEMM and the attention, every core): q and
// k heads get their q/k-norm, gamma and RoPE, V tiles are transposed into
// V^T tiles; tiles come back from DDR, one row block per core.
//
// GEMMs: a row of 8 cores is a cascade chain (west to east) over the K
// slices; core c holds slice c of its 32 activation rows (abuf) and gets that
// slice of the weights one group of 32 output columns at a time. The tail
// (column 7) finishes each group with the phase's epilogue and writes 128
// output columns ("a unit") per output tile.
//
// Tiles (every FIFO object, 4608 bytes): 32 rows x 128 columns of bfp16 as
//   [row pair p][column block kb (16)][row block of the pair j][72]
// (a 72-byte block is 8 rows of [exponent, 8 mantissas], 8 values of a row
// sharing the exponent). V^T tiles (attention values) are
//   [d block (16)][key block (4)][72]   rows d, columns keys.
// Weight objects are [column pair q (2)][k block (16)][2][72] (rows = output
// columns n, 8 values of k per exponent).
//
// Program memory is 16 KB per core and every core runs attention, so the
// code is kept small: no unrolling outside the matrix loops, reductions on
// the matrix unit (a block times a bfp16 block of ones sums each row into all
// eight of its lanes).

#include <aie_api/aie.hpp>
#include <stdint.h>

// Built twice: STACK_TAIL for the tails (GEMM tail + epilogues, symbols
// suffixed _t), without it for the other cores (GEMM head/mid + attention).
#ifdef STACK_TAIL
#define SYM(x) x##_t
#else
#define SYM(x) x
#endif

#define NOUNROLL _Pragma("clang loop unroll(disable)")
#define UNROLL8 _Pragma("clang loop unroll_count(8)")
#define UNROLL2 _Pragma("clang loop unroll_count(2)")
#define UNROLLF _Pragma("clang loop unroll(full)")
#define PIPE4 _Pragma("clang loop pipeline_initiation_interval(4)")
#define NOINLINE __attribute__((noinline))
#define INLINE __attribute__((always_inline)) inline

constexpr int BLK = 72;
constexpr int PAIR = 2 * BLK;
constexpr int TILE = 4608;
constexpr int TILE_PAIR = 16 * PAIR; // one row pair of a tile
constexpr int KBMAX = 48;            // k blocks of a K slice at K = 3072
constexpr int ABUF_PAIR = KBMAX * PAIR;
constexpr float EPS = 1e-6f;
constexpr float LOG2E = 1.4426950408889634f;
constexpr float QK_SCALE = LOG2E / 11.313708498984761f; // log2(e) / sqrt(128)
constexpr float NEG = -3.0e38f;

using BV = aie::block_vector<bfp16ebs8, 64>;
using ACC = aie::accum<accfloat, 64>;
using BS = aie::block_vector_input_buffer_stream<bfp16ebs8, 64>;
using BO = aie::block_vector_output_buffer_stream<bfp16ebs8, 64>;
using V64 = aie::vector<float, 64>;
using VB64 = aie::vector<bfloat16, 64>;
using V8 = aie::vector<float, 8>;

// ------------------------------------------------------------------ state
// One L1 buffer per core; the attention fields overlay the GEMM ones.
constexpr int OFF_KVLEN = 0; // int32 [2]
constexpr int OFF_EPI = 8;   // int32: the GEMM phase's epilogue
constexpr int OFF_PC = 16;   // int32 [3]: phase counters (group, kind, layer)
constexpr int OFF_ABUF = 64; // GEMM A slice (13824)
constexpr int OFF_ACC =
    OFF_ABUF + 2 * ABUF_PAIR; // fp32 [4 rb][4 nb][64] (4096)
constexpr int OFF_RSTD =
    OFF_ACC + 4096; // bf16 [2 groups][4 rb][64]: row norms in all lanes
constexpr int ST_BODY = OFF_RSTD + 1024;
// tail only: the x tile of the residual (O, down), the second group's output
// tile of a pair
constexpr int OFF_X = ST_BODY;
constexpr int OFF_X2 = OFF_X + TILE;
constexpr int ST_TAIL = OFF_X2 + TILE;
// the QKV post-pass overlays [OFF_ABUF ...): the RoPE tile of its row block
// (4608), gamma q/k bf16 [256] (512), a row pair's results bf16 [2][16 cb][64]
constexpr int OFF_QR = OFF_ABUF;
constexpr int OFF_QG = OFF_QR + TILE;
constexpr int OFF_QT = OFF_QG + 512;
// attention overlays [OFF_ABUF ...): the q tile, o fp32 [4 qb][16 db][64],
// the scores of a chunk's key blocks bf16 [4 qb][8 key blocks of 8][64], P
// bfp16 [4 qb][8][72], and per row (in all 8 lanes of its block row) the
// running max (bf16), the last rescale factor (bf16) and the running sum (fp32)
constexpr int NKB = 2;           // key blocks of 32 per chunk (one row group)
constexpr int OFF_AQ = OFF_ABUF; // 4608
constexpr int OFF_AO = OFF_AQ + TILE;                 // 16384
constexpr int OFF_AS = OFF_AO + 16384;                // 4096
constexpr int OFF_AP = OFF_AS + 4 * 4 * NKB * 64 * 2; // 2304
constexpr int OFF_AM = OFF_AP + 4 * 4 * NKB * BLK;    // 512
constexpr int OFF_AF = OFF_AM + 512;                  // 512
constexpr int OFF_AL = OFF_AF + 512;                  // 1024
constexpr int ST_ATTN = OFF_AL + 1024;
static_assert(ST_ATTN == 29504 && ST_TAIL <= ST_ATTN &&
                  OFF_QT + 4096 <= ST_ATTN,
              "the state buffer of qwen3_stack_design.py");

// bfp16 blocks of the identity and of ones (exponent 127, mantissa 64 = 1.0)
alignas(64) static const uint8_t EYE[BLK] = {
    127, 64, 0, 0,  0, 0,  0, 0,  0, 127, 0, 64, 0, 0,  0, 0,  0, 0,
    127, 0,  0, 64, 0, 0,  0, 0,  0, 127, 0, 0,  0, 64, 0, 0,  0, 0,
    127, 0,  0, 0,  0, 64, 0, 0,  0, 127, 0, 0,  0, 0,  0, 64, 0, 0,
    127, 0,  0, 0,  0, 0,  0, 64, 0, 127, 0, 0,  0, 0,  0, 0,  0, 64};
alignas(64) static const uint8_t ONES[BLK] = {
    127, 64, 64, 64, 64, 64, 64, 64, 64, 127, 64, 64, 64, 64, 64, 64, 64, 64,
    127, 64, 64, 64, 64, 64, 64, 64, 64, 127, 64, 64, 64, 64, 64, 64, 64, 64,
    127, 64, 64, 64, 64, 64, 64, 64, 64, 127, 64, 64, 64, 64, 64, 64, 64, 64,
    127, 64, 64, 64, 64, 64, 64, 64, 64, 127, 64, 64, 64, 64, 64, 64, 64, 64};

static inline ACC mac(ACC c, BV a, BV b) {
  return ACC(::mac_8x8_8x8T(a, b, c));
}
static inline ACC mul(BV a, BV b) { return ACC(::mul_8x8_8x8T(a, b)); }

static inline BV load_blk(const uint8_t *p) {
  BS s((const bfp16ebs8 *)p);
  BV v;
  s >> v;
  return v;
}

static inline ACC load_acc(const float *p) {
  ACC a;
  a.from_vector(aie::load_v<64>(p));
  return a;
}

INLINE static BV to_bfp(V64 v) {
  ACC a;
  a.from_vector(v);
  return a.template to_vector<bfp16ebs8>();
}

static inline VB64 to_bf(V64 v) {
  ACC a;
  a.from_vector(v);
  return a.template to_vector<bfloat16>();
}

static inline V64 from_bf(VB64 v) {
  ACC a;
  a.from_vector(v);
  return a.template to_vector<float>();
}

// exact bfp16 -> fp32: x times the identity
INLINE static ACC decode(const uint8_t *blk) {
  return mul(load_blk(blk), load_blk(EYE));
}

// each row's sum (of the bfp16 rounding of v) in all eight of its lanes
INLINE static ACC row_sums(ACC a) {
  return mul(a.template to_vector<bfp16ebs8>(), load_blk(ONES));
}

// lane r*8+c of the result is v[r]
static inline V64 rows8(V8 v) {
  auto a = aie::interleave_zip(v, v, 1);
  aie::vector<float, 16> b = aie::concat(a.first, a.second);
  auto c = aie::interleave_zip(b, b, 2);
  aie::vector<float, 32> d = aie::concat(c.first, c.second);
  auto e = aie::interleave_zip(d, d, 4);
  return aie::concat(e.first, e.second);
}

// lane r*8+c of the result is v[c]
static inline VB64 cols8(aie::vector<bfloat16, 8> v) {
  aie::vector<bfloat16, 16> b = aie::concat(v, v);
  aie::vector<bfloat16, 32> d = aie::concat(b, b);
  return aie::concat(d, d);
}

static inline VB64 rows8b(aie::vector<bfloat16, 8> v) {
  auto a = aie::interleave_zip(v, v, 1);
  aie::vector<bfloat16, 16> b = aie::concat(a.first, a.second);
  auto c = aie::interleave_zip(b, b, 2);
  aie::vector<bfloat16, 32> d = aie::concat(c.first, c.second);
  auto e = aie::interleave_zip(d, d, 4);
  return aie::concat(e.first, e.second);
}

// the max of each row (8 lanes) of an 8x8 block, at bf16 precision
static inline V8 row_max(V64 v) {
  VB64 t = aie::transpose(to_bf(v), 8, 8);
  aie::vector<bfloat16, 32> a = aie::max(t.extract<32>(0), t.extract<32>(1));
  aie::vector<bfloat16, 16> b = aie::max(a.extract<16>(0), a.extract<16>(1));
  aie::vector<bfloat16, 8> c = aie::max(b.extract<8>(0), b.extract<8>(1));
  aie::accum<accfloat, 16> f;
  f.from_vector(aie::concat(c, c));
  return f.template to_vector<float>().extract<8>(0);
}

// 2^a for an accumulator (the cores' bf16 exp2), 16 lanes at a time
static inline VB64 exp2a(ACC a) {
  VB64 r;
  UNROLLF for (int i = 0; i < 4; i++)
      r.insert(i, aie::vector<bfloat16, 16>(
                      ::exp2((v16accfloat)a.template extract<16>(i))));
  return r;
}

// lanes r*8+c with c < n (n = valid keys of an 8-key block)
static inline aie::mask<64> first_cols(int n) {
  uint32_t row = n >= 8 ? 0xFFu : n <= 0 ? 0u : ((1u << n) - 1);
  row |= row << 8;
  row |= row << 16;
  return aie::mask<64>::from_uint64(((uint64_t)row << 32) | row);
}

static inline BV bfp(ACC a) { return a.template to_vector<bfp16ebs8>(); }

static inline BV bfp(VB64 v) {
  ACC a;
  a.from_vector(v);
  return a.template to_vector<bfp16ebs8>();
}

// block (rb, kb) of a tile
static inline const uint8_t *tile_blk(const uint8_t *t, int rb, int kb) {
  return t + (((rb >> 1) * 16 + kb) * 2 + (rb & 1)) * BLK;
}

NOINLINE static void copy_tile(const uint8_t *__restrict s,
                               uint8_t *__restrict d) {
  const v64int8 *__restrict a = (const v64int8 *)s;
  v64int8 *__restrict b = (v64int8 *)d;
  UNROLL8 for (int i = 0; i < TILE / 64; i++) b[i] = a[i];
}

extern "C" {

// ------------------------------------------------------------------ phases
// The step's GEMMs, in order: per layer QKV for every row group, O for every
// group, gate/up for every pair of groups, down for every group; then the
// heads for every pair. ctl (read by the design's loops): chunks of the K
// slice, output units, groups (of 32 columns) per unit, RMSNorm, row groups,
// x tiles per unit (tails), QKV post-pass + attention after the phase, A tiles
// per chunk, row groups - 1.
enum { EPI_RES = 1, EPI_SCALE = 2, EPI_LOGITS = 3 };
enum { CTL_NGI = 4, CTL_ATT = 6, CTL_PAIR = 8 };

// The head's output units of 64 columns, set by the build
#ifndef HEAD_U
#error "build with -DHEAD_U=<the head's output units of 64 columns>"
#endif

void SYM(og_phase)(int32_t *ctl, uint8_t *st, int32_t nl, int32_t ng) {
  static const int32_t T[5][8] = {
      // nch units gpu norm paired x attn parts
      {1, 32, 4, 1, 0, 0, 0, 1}, // QKV, normed (q/k-norm, RoPE: the post-pass)
      {2, 8, 4, 0, 0, 1, 0, 1},  // O (+ residual)
      {1, 48, 4, 1, 1, 0, 0,
       1}, // gate / up, normed, as tiles (gate unit, up unit, ...)
      {3, 8, 4, 0, 0, 1, 0,
       2}, // down (+ residual); A = silu(gate) * up, made on load
      {1, HEAD_U, 2, 1, 1, 0, 0, 1}}; // the head (64 columns per object)
  static const int32_t EPI[5] = {EPI_SCALE, EPI_RES, EPI_SCALE, EPI_RES,
                                 EPI_LOGITS};
  int32_t *pc = (int32_t *)(st + OFF_PC);
  const int g = pc[0], kind = pc[2] < nl ? pc[1] : 4;
  NOUNROLL for (int f = 0; f < 8; f++) ctl[f] = T[kind][f];
  const int paired = T[kind][CTL_NGI];
  const int nph = paired ? (ng + 1) >> 1 : ng;
  ctl[CTL_NGI] = paired && 2 * g + 1 < ng ? 2 : 1;
  ctl[CTL_PAIR] = ctl[CTL_NGI] - 1;
  ctl[CTL_ATT] = kind == 0 && g == ng - 1;
  *(int32_t *)(st + OFF_EPI) = EPI[kind];
  if (++pc[0] == nph) {
    pc[0] = 0;
    if (++pc[1] == 4) {
      pc[1] = 0;
      ++pc[2];
    }
  }
}

// ------------------------------------------------------------------ setup
// The step's parameters (first object of the step): int32 kvlen[2].
void SYM(oc_params)(const uint8_t *obj, uint8_t *st) {
  ((int32_t *)(st + OFF_KVLEN))[0] = ((const int32_t *)obj)[0];
  ((int32_t *)(st + OFF_KVLEN))[1] = ((const int32_t *)obj)[1];
  int32_t *pc = (int32_t *)(st + OFF_PC);
  pc[0] = pc[1] = pc[2] = 0;
}

#ifdef STACK_TAIL
void SYM(oc_take_x)(const uint8_t *obj, uint8_t *st) {
  copy_tile(obj, st + OFF_X);
}
#endif

// ------------------------------------------------------------------ GEMM
// silu(g) = g * sigmoid(g) = h (1 + tanh h) with h = g / 2
static inline ACC silu(VB64 g) {
  ACC h = aie::mul(g, bfloat16(0.5f));
  VB64 hb = h.template to_vector<bfloat16>();
  VB64 t;
  UNROLLF for (int i = 0; i < 4; i++)
      t.insert(i, aie::vector<bfloat16, 16>(
                      ::tanh((v16accfloat)h.template extract<16>(i))));
  return aie::mac(h, hb, t);
}

// Chunk ch of the core's A slice into abuf: one tile, or (parts = 2, the down
// GEMM) a gate tile then an up tile, from which the chunk is silu(gate) * up.
// The gate tile waits in the region past the GEMM state (attention scratch;
// on the tails the x tile, taken only after the loads).
constexpr int OFF_GTMP = ST_BODY;
void SYM(og_load)(const uint8_t *tile, uint8_t *st, int32_t ch, int32_t part,
                  int32_t parts, int32_t gi) {
  ch += gi; // a pair's single-chunk slices sit in chunk slots 0, 1
  if (parts == 2 && part == 0) {
    copy_tile(tile, st + OFF_GTMP);
    return;
  }
  if (parts == 1) {
    NOUNROLL for (int p = 0; p < 2; p++) {
      const v64int8 *__restrict s = (const v64int8 *)(tile + p * TILE_PAIR);
      v64int8 *__restrict d =
          (v64int8 *)(st + OFF_ABUF + p * ABUF_PAIR + ch * TILE_PAIR);
      UNROLL8 for (int i = 0; i < TILE_PAIR / 64; i++) d[i] = s[i];
    }
    return;
  }
  NOUNROLL for (int p = 0; p < 2; p++) {
    BO o((bfp16ebs8 *)(st + OFF_ABUF + p * ABUF_PAIR + ch * TILE_PAIR));
    NOUNROLL for (int i = 0; i < 32; i++) {
      const int off = p * TILE_PAIR + i * BLK;
      VB64 g = decode(st + OFF_GTMP + off).template to_vector<bfloat16>();
      VB64 u = decode(tile + off).template to_vector<bfloat16>();
      o << aie::mul(silu(g).template to_vector<bfloat16>(), u)
               .template to_vector<bfp16ebs8>();
    }
  }
}

// Sums of squares of the slice's 32 rows (nch chunks, from chunk slot gi) as
// [4 rb][8]: the diagonal of A A^T per row block.
NOINLINE static void ssq_own(const uint8_t *st, int nch, int gi, float *out) {
  NOUNROLL for (int rb = 0; rb < 4; rb++) {
    const uint8_t *base =
        st + OFF_ABUF + (rb >> 1) * ABUF_PAIR + gi * TILE_PAIR + (rb & 1) * BLK;
    ACC g = aie::zeros<accfloat, 64>();
    NOUNROLL for (int kb = 0; kb < nch * 16; kb++) {
      BV a = load_blk(base + kb * PAIR);
      g = mac(g, a, a);
    }
    V64 v = g.template to_vector<float>();
    NOUNROLL for (int i = 0; i < 8; i++) out[rb * 8 + i] = v[i * 9];
  }
}

#ifndef STACK_TAIL
void og_ssq_head(uint8_t *st, int32_t nch, int32_t gi) {
  alignas(64) float s[32];
  ssq_own(st, nch, gi, s);
  put_mcd(*(v16float *)s);
  put_mcd(*(v16float *)(s + 16));
}

void og_ssq_mid(uint8_t *st, int32_t nch, int32_t gi) {
  alignas(64) float s[32];
  ssq_own(st, nch, gi, s);
  aie::vector<float, 16> a = (v16float)get_scd_v16accfloat(1);
  aie::vector<float, 16> b = (v16float)get_scd_v16accfloat(1);
  put_mcd((v16float)aie::add(a, aie::load_v<16>(s)));
  put_mcd((v16float)aie::add(b, aie::load_v<16>(s + 16)));
}

#else
// tail: rstd = 1 / sqrt(mean of squares + eps) over the whole row, kept as
// [4 rb][64] (each row in all 8 lanes of its row of the block). The normed
// GEMMs all have K = D = 1024: rstd = 32 / sqrt(sum + 1024 eps), the 32 an
// exact bf16 product (the scalar unit has no float multiply).
void SYM(og_ssq_tail)(uint8_t *st, int32_t nch, int32_t gi) {
  alignas(64) float s[32];
  ssq_own(st, nch, gi, s);
  aie::vector<float, 16> a = (v16float)get_scd_v16accfloat(1);
  aie::vector<float, 16> b = (v16float)get_scd_v16accfloat(1);
  aie::store_v(s, aie::add(a, aie::load_v<16>(s)));
  aie::store_v(s + 16, aie::add(b, aie::load_v<16>(s + 16)));
  NOUNROLL for (int i = 0; i < 32; i++) s[i] = invsqrt(s[i] + 1024 * EPS);
  bfloat16 *r = (bfloat16 *)(st + OFF_RSTD) + gi * 256;
  NOUNROLL for (int rb = 0; rb < 4; rb++) aie::store_v(
      r + rb * 64,
      aie::mul(to_bf(rows8(aie::load_v<8>(s + rb * 8))), bfloat16(32.0f))
          .template to_vector<bfloat16>());
}

#endif

// acc [4 rb][4 nb] += A[:, chunk ch] x W^T for one weight object
// Partial sums travel the cascade in accumulator registers: the first chunk of
// a group starts from the core to the west (or zero at the head), the last one
// sends its sums east (or, at the tail, applies the phase's epilogue to them);
// only groups of several chunks keep them in acc between chunks.
enum { ROLE_HEAD = 0, ROLE_MID = 1, ROLE_TAIL = 2 };

static inline ACC cas_get() { return ACC(::get_scd_v64accfloat(1)); }

static inline void cas_put(ACC a) {
  v64accfloat v = a;
  put_mcd(::extract_v16accfloat(v, 0));
  put_mcd(::extract_v16accfloat(v, 1));
  put_mcd(::extract_v16accfloat(v, 2));
  put_mcd(::extract_v16accfloat(v, 3));
}

#ifdef STACK_TAIL
// ------------------------------------------------------------------ tail
// epilogues AIE2P has no fp32 vector multiply (it is emulated with bf16
// pieces), so the epilogues multiply bf16 operands into fp32 accumulators and
// add there. They run on a group's sums as the last chunk's mac leaves them in
// registers.
static inline VB64 rstd_bf(const uint8_t *st, int gi, int rb) {
  return aie::load_v<64>((const bfloat16 *)(st + OFF_RSTD) +
                         (gi * 4 + rb) * 64);
}

// The epilogue of a quarter of group g's sums (row blocks 2p, 2p + 1 x column
// blocks 2q, 2q + 1 of the group), as the last chunk's mac leaves them in
// registers; o is the output tile (the unit's object, or for the second row
// group of a pair a tile in the state that og_copy moves later):
//   gate/up       y = a * rstd, bfp16 blocks of the output tile
//   O, down       y = x + a (x: this unit's tile of the residual), the same
//   heads         y = a * rstd, bf16 8x8 blocks [g % 2][4 rb][4 nb][64] (the
//                 unit's object carries both groups, its first 4 KB)
static inline void epi_pq(uint8_t *st, uint8_t *o, int e, int gi, int g, int p,
                          int q, ACC a00, ACC a01, ACC a10, ACC a11) {
  const int rb = 2 * p, nb = 2 * q, cb = 4 * (g & 3) + nb;
  if (e == EPI_RES) {
    const uint8_t *x = st + OFF_X;
    BO out((bfp16ebs8 *)(o + (p * 16 + cb) * 2 * BLK));
    out << bfp(aie::add(a00, decode(tile_blk(x, rb, cb))))
        << bfp(aie::add(a10, decode(tile_blk(x, rb + 1, cb))))
        << bfp(aie::add(a01, decode(tile_blk(x, rb, cb + 1))))
        << bfp(aie::add(a11, decode(tile_blk(x, rb + 1, cb + 1))));
    return;
  }
  const VB64 r0 = rstd_bf(st, gi, rb), r1 = rstd_bf(st, gi, rb + 1);
  const ACC y00 = aie::mul(a00.template to_vector<bfloat16>(), r0),
            y01 = aie::mul(a01.template to_vector<bfloat16>(), r0);
  const ACC y10 = aie::mul(a10.template to_vector<bfloat16>(), r1),
            y11 = aie::mul(a11.template to_vector<bfloat16>(), r1);
  if (e == EPI_LOGITS) {
    bfloat16 *d = (bfloat16 *)o + (g & 1) * 1024 + (rb * 4 + nb) * 64;
    aie::store_v(d, y00.template to_vector<bfloat16>());
    aie::store_v(d + 64, y01.template to_vector<bfloat16>());
    aie::store_v(d + 256, y10.template to_vector<bfloat16>());
    aie::store_v(d + 320, y11.template to_vector<bfloat16>());
    return;
  }
  BO out((bfp16ebs8 *)(o + (p * 16 + cb) * 2 * BLK));
  out << bfp(y00) << bfp(y10) << bfp(y01) << bfp(y11);
}

// The second row group's output tile of a pair into its object
void SYM(og_copy)(uint8_t *o, uint8_t *st) { copy_tile(st + OFF_X2, o); }
#endif

// one group's (g) chunk ch of nch: acc [4 rb][4 nb] += A[:, chunk ch] x W^T,
// for each of the ngi row groups of the phase (pairs: nch = 1)
#ifdef STACK_TAIL
void SYM(og_mac)(const uint8_t *b, uint8_t *st, uint8_t *o, int32_t ch,
                 int32_t nch, int32_t role, int32_t g, int32_t ngi) {
#else
void SYM(og_mac)(const uint8_t *b, uint8_t *st, int32_t ch, int32_t nch,
                 int32_t role, int32_t g, int32_t ngi) {
#endif
#ifdef SKIP_MAC
  return;
#endif
  const bool first = ch == 0, last = ch == nch - 1;
  float *acc = (float *)(st + OFF_ACC);
#ifdef STACK_TAIL
  const int e = *(const int32_t *)(st + OFF_EPI);
#endif
  NOUNROLL for (int gq = 0; gq < 4 * ngi; gq++) {
    const int gi = gq >> 2, pq = gq & 3, p = pq >> 1, q = pq & 1;
    float *c00 = acc + ((2 * p) * 4 + 2 * q) * 64;
    float *c01 = c00 + 64;
    float *c10 = c00 + 4 * 64;
    float *c11 = c10 + 64;
    ACC a00, a01, a10, a11;
    if (!first) {
      a00 = load_acc(c00), a01 = load_acc(c01), a10 = load_acc(c10),
      a11 = load_acc(c11);
    } else if (role == ROLE_HEAD) {
      a00 = a01 = a10 = a11 = aie::zeros<accfloat, 64>();
    } else {
      a00 = cas_get(), a01 = cas_get(), a10 = cas_get(), a11 = cas_get();
    }
    BS as((const bfp16ebs8 *)(st + OFF_ABUF + p * ABUF_PAIR +
                              (ch + gi) * TILE_PAIR));
    BS bs((const bfp16ebs8 *)(b + q * TILE_PAIR));
    // software pipelined (5 bundles per 4 MACs, load-bound; 7 without)
    PIPE4 for (int kb = 0; kb < 16; kb++) {
      BV x0, x1, y0, y1;
      as >> x0 >> x1;
      bs >> y0 >> y1;
      a00 = mac(a00, x0, y0);
      a01 = mac(a01, x0, y1);
      a10 = mac(a10, x1, y0);
      a11 = mac(a11, x1, y1);
    }
#ifdef STACK_TAIL
    if (last) {
      epi_pq(st, gi ? st + OFF_X2 : o, e, gi, g, p, q, a00, a01, a10, a11);
      continue;
    }
#else
    if (last) {
      cas_put(a00), cas_put(a01), cas_put(a10), cas_put(a11);
      continue;
    }
#endif
    aie::store_v(c00, a00.to_vector<float>());
    aie::store_v(c01, a01.to_vector<float>());
    aie::store_v(c10, a10.to_vector<float>());
    aie::store_v(c11, a11.to_vector<float>());
  }
}

#ifdef STACK_TAIL
#endif // STACK_TAIL

// ------------------------------------------------------------------ QKV
// post-pass Per layer: gamma q | k (bf16 [256], times sqrt(128): the q/k-norm
// below is 1 / sqrt(sum + 128 eps)), then per row group the RoPE tile of the
// core's row block and its tiles.
void SYM(qp_gamma)(const uint8_t *obj, uint8_t *st) {
  const v32bfloat16 *s = (const v32bfloat16 *)obj;
  v32bfloat16 *d = (v32bfloat16 *)(st + OFF_QG);
  NOUNROLL for (int i = 0; i < 8; i++) d[i] = s[i];
}

void SYM(qp_rope)(const uint8_t *obj, uint8_t *st) {
  copy_tile(obj, st + OFF_QR);
}

// A q (k = 0) or k (k = 1) head tile: per row 1 / rms (the diagonal of Y Y^T
// over the 16 column blocks), gamma, RoPE on column block pairs (cb, cb + 8),
// a row pair at a time into OFF_QT, then the pair's 32 blocks in tile order.
void SYM(qp_qk)(const uint8_t *in, uint8_t *out, uint8_t *st, int32_t k) {
  const bfloat16 *gamma = (const bfloat16 *)(st + OFF_QG) + (k ? 128 : 0);
  const uint8_t *rope = st + OFF_QR;
  bfloat16 *tmp = (bfloat16 *)(st + OFF_QT);
  BO o((bfp16ebs8 *)out);
  NOUNROLL for (int p = 0; p < 2; p++) {
    NOUNROLL for (int j = 0; j < 2; j++) {
      const int rb = 2 * p + j;
      ACC d = aie::zeros<accfloat, 64>();
      NOUNROLL for (int cb = 0; cb < 16; cb++) {
        const BV y = load_blk(tile_blk(in, rb, cb));
        d = mac(d, y, y);
      }
      const V64 v = d.template to_vector<float>();
      alignas(64) float rr[8];
      NOUNROLL for (int i = 0; i < 8; i++) rr[i] =
          invsqrt(v[i * 9] + 128 * EPS);
      const VB64 r = to_bf(rows8(aie::load_v<8>(rr)));
      bfloat16 *t = tmp + j * 16 * 64;
      NOUNROLL for (int cb = 0; cb < 8; cb++) {
        const VB64 a =
            aie::mul(
                decode(tile_blk(in, rb, cb)).template to_vector<bfloat16>(),
                cols8(aie::load_v<8>(gamma + cb * 8)))
                .template to_vector<bfloat16>();
        const VB64 b =
            aie::mul(
                decode(tile_blk(in, rb, cb + 8)).template to_vector<bfloat16>(),
                cols8(aie::load_v<8>(gamma + 64 + cb * 8)))
                .template to_vector<bfloat16>();
        const VB64 c =
            decode(tile_blk(rope, rb, cb)).template to_vector<bfloat16>();
        const VB64 sn =
            decode(tile_blk(rope, rb, cb + 8)).template to_vector<bfloat16>();
        const VB64 ra =
            aie::msc(aie::mul(a, c), b, sn).template to_vector<bfloat16>();
        const VB64 rbv =
            aie::mac(aie::mul(b, c), a, sn).template to_vector<bfloat16>();
        aie::store_v(t + cb * 64,
                     aie::mul(ra, r).template to_vector<bfloat16>());
        aie::store_v(t + (cb + 8) * 64,
                     aie::mul(rbv, r).template to_vector<bfloat16>());
      }
    }
    NOUNROLL for (int i = 0; i < 32; i++) o
        << bfp(aie::load_v<64>(tmp + ((i & 1) * 16 + (i >> 1)) * 64));
  }
}

// A V tile (rows = keys) as a V^T tile [16 db][4 kb][72] (rows d, columns keys)
void SYM(qp_v)(const uint8_t *in, uint8_t *out, uint8_t *st) {
  BO o((bfp16ebs8 *)out);
  NOUNROLL for (int i = 0; i < 64; i++) o << bfp(aie::transpose(
      decode(tile_blk(in, i & 3, i >> 2)).template to_vector<bfloat16>(), 8,
      8));
}

// ------------------------------------------------------------------ attention
// One work item: 32 query rows of one head against the keys of their
// sequence, chunk by chunk (a chunk = the NKB key blocks of one row group),
// keys >= kvlen masked: per chunk the scores, then a softmax step against the
// running row max (o and l rescaled when it grows), then P V. Every core runs
// it, the tails too.
constexpr float M0 = -1.0e4f; // the running max before the first chunk

void SYM(oa_q)(const uint8_t *qt, uint8_t *st) {
  copy_tile(qt, st + OFF_AQ);
  float *o = (float *)(st + OFF_AO);
  NOUNROLL for (int i = 0; i < 64; i++)
      aie::store_v(o + 64 * i, aie::zeros<float, 64>());
  bfloat16 *m = (bfloat16 *)(st + OFF_AM);
  float *l = (float *)(st + OFF_AL);
  NOUNROLL for (int i = 0; i < 4; i++) {
    aie::store_v(m + 64 * i, aie::broadcast<bfloat16, 64>(bfloat16(M0)));
    aie::store_v(l + 64 * i, aie::zeros<float, 64>());
  }
}

// S of key block kb of chunk j (bf16, masked) for all 4 query row blocks
void SYM(oa_k)(const uint8_t *kt, uint8_t *st, int32_t kb, int32_t s,
               int32_t j) {
#ifdef SKIP_ATT
  return;
#endif
  const int kvlen =
      ((const int32_t *)(st + OFF_KVLEN))[s] - j * 32 * NKB - kb * 32;
  bfloat16 *sc = (bfloat16 *)(st + OFF_AS);
  // as og_mac: a query row pair x a key row pair per pass, 2 x 2 blocks
  NOUNROLL for (int pq = 0; pq < 4; pq++) {
    const int p = pq >> 1, kp = pq & 1;
    BS qs((const bfp16ebs8 *)(st + OFF_AQ + p * TILE_PAIR));
    BS ks((const bfp16ebs8 *)(kt + kp * TILE_PAIR));
    ACC c00 = aie::zeros<accfloat, 64>(), c01 = c00, c10 = c00, c11 = c00;
    PIPE4 for (int db = 0; db < 16; db++) {
      BV q0, q1, k0, k1;
      qs >> q0 >> q1;
      ks >> k0 >> k1;
      c00 = mac(c00, q0, k0);
      c01 = mac(c01, q0, k1);
      c10 = mac(c10, q1, k0);
      c11 = mac(c11, q1, k1);
    }
    // keys past kvlen do not exist (branch-free: a branch here trips Peano's
    // early if-conversion)
    const VB64 neg = aie::broadcast<bfloat16, 64>(bfloat16(NEG));
    const aie::mask<64> m0 = first_cols(kvlen - 16 * kp),
                        m1 = first_cols(kvlen - 16 * kp - 8);
    bfloat16 *d = sc + ((2 * p) * 4 * NKB + kb * 4 + 2 * kp) * 64;
    aie::store_v(d, aie::select(neg, c00.template to_vector<bfloat16>(), m0));
    aie::store_v(d + 64,
                 aie::select(neg, c01.template to_vector<bfloat16>(), m1));
    aie::store_v(d + 4 * NKB * 64,
                 aie::select(neg, c10.template to_vector<bfloat16>(), m0));
    aie::store_v(d + 4 * NKB * 64 + 64,
                 aie::select(neg, c11.template to_vector<bfloat16>(), m1));
  }
}

// an fp32 accumulator times a bf16 factor, to ~16 bits: a = hi + lo, both bf16
static inline ACC scale_acc(ACC a, VB64 f) {
  const VB64 hi = a.template to_vector<bfloat16>();
  ACC h;
  h.from_vector(hi);
  const VB64 lo = aie::sub(a, h).template to_vector<bfloat16>();
  return aie::mac(aie::mul(hi, f), lo, f);
}

// softmax step over chunk j: the new row max m', P = 2^((s - m') scale)
// (bfp16, masked keys 0), f = 2^((m - m') scale) for o (applied by oa_v) and
// l = l f + sum P. f and the P are bf16, l keeps fp32; o / l is unaffected by
// f's rounding since o and l take the same f.
void SYM(oa_soft)(uint8_t *st, int32_t s, int32_t j) {
#ifdef SKIP_ATT
  return;
#endif
  const int kvlen = ((const int32_t *)(st + OFF_KVLEN))[s] - j * 32 * NKB;
  const bfloat16 *sc = (const bfloat16 *)(st + OFF_AS);
  BO po((bfp16ebs8 *)(st + OFF_AP));
  bfloat16 *mp = (bfloat16 *)(st + OFF_AM);
  bfloat16 *fp = (bfloat16 *)(st + OFF_AF);
  float *lp = (float *)(st + OFF_AL);
  const BV ones = load_blk(ONES);
  const bfloat16 scale = bfloat16(QK_SCALE);
  const VB64 vscale = aie::broadcast<bfloat16, 64>(scale);
  NOUNROLL for (int qb = 0; qb < 4; qb++) {
    VB64 mx = aie::load_v<64>(sc + qb * 4 * NKB * 64);
    NOUNROLL for (int k = 1; k < 4 * NKB; k++) mx =
        aie::max(mx, aie::load_v<64>(sc + (qb * 4 * NKB + k) * 64));
    VB64 t = aie::transpose(mx, 8, 8);
    aie::vector<bfloat16, 32> a = aie::max(t.extract<32>(0), t.extract<32>(1));
    aie::vector<bfloat16, 16> b = aie::max(a.extract<16>(0), a.extract<16>(1));
    const VB64 m0 = aie::load_v<64>(mp + qb * 64);
    const VB64 m =
        aie::max(m0, rows8b(aie::max(b.extract<8>(0), b.extract<8>(1))));
    ACC L = aie::zeros<accfloat, 64>();
    NOUNROLL for (int k = 0; k < 4 * NKB; k++) {
      // 2^((s - m) * scale), masked keys to 0
      ACC e = aie::msc(
          aie::mul(aie::load_v<64>(sc + (qb * 4 * NKB + k) * 64), scale), m,
          vscale);
      VB64 p = aie::select(aie::zeros<bfloat16, 64>(), exp2a(e),
                           first_cols(kvlen - k * 8));
      ACC pa;
      pa.from_vector(p);
      BV pb = pa.template to_vector<bfp16ebs8>();
      L = mac(L, pb, ones);
      po << pb;
    }
    const VB64 f = exp2a(aie::msc(aie::mul(m0, scale), m, vscale));
    aie::store_v(lp + qb * 64, aie::add(L, scale_acc(load_acc(lp + qb * 64), f))
                                   .template to_vector<float>());
    aie::store_v(fp + qb * 64, f);
    aie::store_v(mp + qb * 64, m);
  }
}

// o += P V for key block kb (V^T tile [16 db][4 kb][72]); o is first rescaled
// by f when a later chunk starts (kb == 0, j > 0)
void SYM(oa_v)(const uint8_t *vt, uint8_t *st, int32_t kb, int32_t j) {
#ifdef SKIP_ATT
  return;
#endif
  const uint8_t *pb = st + OFF_AP;
  const bool rescale = kb == 0 && j > 0;
  NOUNROLL for (int qb = 0; qb < 4; qb++) {
    const uint8_t *pq = pb + (qb * 4 * NKB + kb * 4) * BLK;
    BV p0 = load_blk(pq), p1 = load_blk(pq + BLK), p2 = load_blk(pq + 2 * BLK),
       p3 = load_blk(pq + 3 * BLK);
    const VB64 f = aie::load_v<64>((const bfloat16 *)(st + OFF_AF) + qb * 64);
    float *op = (float *)(st + OFF_AO) + qb * 16 * 64;
    BS vs((const bfp16ebs8 *)vt);
    // two d blocks per pass, their chains interleaved (one chain alone waits on
    // the MAC latency), loads before stores
    NOUNROLL for (int d2 = 0; d2 < 8; d2++) {
      ACC c0 = load_acc(op), c1 = load_acc(op + 64);
      if (rescale) {
        c0 = scale_acc(c0, f);
        c1 = scale_acc(c1, f);
      }
      BV a0, a1, a2, a3, b0, b1, b2, b3;
      vs >> a0 >> a1 >> a2 >> a3 >> b0 >> b1 >> b2 >> b3;
      c0 = mac(c0, p0, a0);
      c1 = mac(c1, p0, b0);
      c0 = mac(c0, p1, a1);
      c1 = mac(c1, p1, b1);
      c0 = mac(c0, p2, a2);
      c1 = mac(c1, p2, b2);
      c0 = mac(c0, p3, a3);
      c1 = mac(c1, p3, b3);
      aie::store_v(op, c0.template to_vector<float>());
      aie::store_v(op + 64, c1.template to_vector<float>());
      op += 128;
    }
  }
}

// o / l as a bfp16 tile
void SYM(oa_out)(uint8_t *out, uint8_t *st) {
#ifdef SKIP_ATT
  return;
#endif
  const float *o = (const float *)(st + OFF_AO);
  const float *lp = (const float *)(st + OFF_AL);
  bfloat16 *il = (bfloat16 *)(st + OFF_AF);
  NOUNROLL for (int qb = 0; qb < 4; qb++) {
    V64 lv = aie::load_v<64>(lp + qb * 64);
    alignas(64) float r[8];
    NOUNROLL for (int i = 0; i < 8; i++) r[i] = inv(lv[i * 8]);
    aie::store_v(il + qb * 64, to_bf(rows8(aie::load_v<8>(r))));
  }
  BO po((bfp16ebs8 *)out);
  NOUNROLL for (int i = 0; i < 64; i++) {
    const int p = i >> 5, db = (i >> 1) & 15, j = i & 1, qb = 2 * p + j;
    ACC y = aie::mul(to_bf(aie::load_v<64>(o + (qb * 16 + db) * 64)),
                     aie::load_v<64>(il + qb * 64));
    po << y.template to_vector<bfp16ebs8>();
  }
}
} // extern "C"
