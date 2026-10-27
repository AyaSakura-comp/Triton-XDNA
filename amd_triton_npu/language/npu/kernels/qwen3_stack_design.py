# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""The AIE design of ``tl.extra.npu.qwen3_stack`` (IRON): a Qwen3-0.6B-sized stack of NL
layers, its final norm and a linear head over two sequences, as ONE launch.

Run as a script in its own process (mlir-aie's MLIR bindings and mlir-air's cannot share
an interpreter); imports nothing from Triton:

    python qwen3_stack_design.py '{"NL": .., "NG": .., "HU": .., "obj_body": .., "obj_tail": ..}' OUTDIR

writes OUTDIR/final.xclbin and OUTDIR/insts.bin.

Every core of the 8 x 4 array runs a static schedule (qwen3_stack_core.cc); the runtime
sequence moves 4608-byte tiles (32 rows x 128 columns of bfp16) between DDR and the cores:

  GEMMs     a row of 8 cores is a cascade chain over K slices (core c: slice c of its row
            block's 32 activation rows, loaded once per GEMM); the weights stream through
            memtile c to column c's 4 cores, one contiguous run per column per GEMM and row
            group; the tails (column 7) apply the epilogue and write one output tile per 128
            columns:
              QKV      RMSNorm (sums of squares passed along the chain)
              O, down  + the residual
              gate/up  RMSNorm; gate and up as separate units (h = silu(gate) * up is made
                       by the down GEMM's cores as they load their A slice)
              head     final norm, logits
  QKV post  every core: q/k-norm + gamma + RoPE of its row block of q heads 2c, 2c + 1 and of
            kv head c's K (rows 0, 1) or the V -> V^T transpose (rows 2, 3), back to DDR
  attention column c runs kv head c for both sequences; its 4 cores take the 2 query heads x
            2 query blocks of a row group at a time; q arrives on A, the shared K / V^T of
            every row group on the weights' broadcast (B), an online softmax runs over them

Rows: NG row groups of 128 = sequence 0 (row blocks 0, 1: positions 64g..64g+63) + sequence
1 (row blocks 2, 3: the same positions); QKV, O and down run once per group (their weights
stream NG times), gate/up and the head once per pair of groups (ceil(NG / 2) times: the
cores hold both groups' A slices); keys past each sequence's length are masked.

DDR (one activation buffer, tiles of 4608 bytes): PARAMS 1 | GAMMA NL | then per row group
  ATT 64 | X 32 | GU 192 | X1 32 | QR 160
  layouts [unit][row block] (GU: [24][gate, up][row block]); QR, per column c: RoPE [row
  block] 4 | q heads 2c, 2c + 1 [row block] 8 | kv head c [sequence][K, V^T][key block] 8,
  so a column's post-pass input is one run per group; each tail's A run of a phase is
  contiguous (ATT heads 14-15 + X, GU block 21-23 + X1). Weights per column, per layer: QKV
  128, O 64, GU 192, down 96 objects, then the head's 2 HU; logits: [group][unit HU][row
  block 4] objects, bf16 [32][64] each.
"""

import json
import os
import sys

import numpy as np

NCOL, NROW, MT = 8, 4, 32
TILE = 4608
HALF = TILE // 2
D, NH, NKV, DH, INTER = 1024, 16, 8, 128, 3072
GROUP = (("ATT", 64), ("X", 32), ("GU", 192), ("X1", 32), ("QR", 160))
QR_COL, QR_Q, QR_KV = 20, 4, 12  # QR: tiles per column, offsets of q and of K / V^T


class Layout:
    """Tile offsets of the activation buffer's regions: L["X", g] is row group g's X."""

    def __init__(self, NL, NG):
        self.r = {"PARAMS": 0, "GAMMA": 1}
        o = 0
        for name, n in GROUP:
            self.r[name] = o
            o += n
        self.gs = o  # tiles per row group
        self.g0 = 1 + NL  # first group's offset
        self.ntiles = self.g0 + NG * self.gs
        self.NL, self.NG = NL, NG

    def __getitem__(self, k):
        if isinstance(k, tuple):
            name, g = k
            return self.g0 + g * self.gs + self.r[name]
        return self.r[k]


W_QKV = 32 * 4  # weight objects of the QKV GEMM per column
W_O, W_GU, W_DOWN = 8 * 4 * 2, 48 * 4 * 1, 8 * 4 * 3


def w_objects_per_layer():
    return W_QKV + W_O + W_GU + W_DOWN  # 480


def tap(tensor_len, offset_tile, dims):
    """A shim transfer of tiles: dims = [(count, stride_in_tiles), ...] outer to inner (at
    most 3). A contiguous innermost run moves as 2 * count halves of 2304 bytes (a shim BD's
    innermost dimension wraps at 1023 words), any other tile as its two halves."""
    from aie.helpers.taplib import TensorAccessPattern

    dims = [(n, st * TILE) for n, st in dims]
    if dims and dims[-1][1] == TILE:
        dims[-1] = (2 * dims[-1][0], HALF)
    else:
        dims.append((2, HALF))
    dims.append((HALF, 1))
    while len(dims) < 4:
        dims.insert(0, (1, 0))
    assert len(dims) == 4, dims
    sizes, strides = zip(*dims)
    assert sizes[0] <= 63 and sizes[1] <= 1023 and sizes[2] <= 1023, sizes
    return TensorAccessPattern(
        (tensor_len,),
        offset=offset_tile * TILE,
        sizes=list(sizes),
        strides=list(strides),
    )


def run_tap(tensor_len, offset_tile, ntiles):
    """ntiles contiguous tiles."""
    if ntiles <= 511:
        return tap(tensor_len, offset_tile, [(ntiles, 1)])
    a = next(d for d in range(2, 64) if ntiles % d == 0 and ntiles // d <= 511)
    return tap(tensor_len, offset_tile, [(a, ntiles // a), (ntiles // a, 1)])


def design(ACT, W, LOG, *, NL, NG, HU, obj_body, obj_tail, trace=0):
    """HU: the head's output units of 64 columns (its weights: 2 HU objects per column)."""
    import aie.iron as iron
    from aie.iron import (
        Buffer,
        CascadeFlow,
        Kernel,
        ObjectFifo,
        Program,
        Runtime,
        Worker,
    )
    from aie.iron.controlflow import range_
    from aie.iron.device import Tile
    from aie.iron.runtime import TaskGroup

    L = Layout(NL, NG)
    NP = (NG + 1) // 2  # row group pairs (gate/up, head)
    HEAD_W = 2 * HU
    # attention: column c runs kv head c for both sequences. A traced build leaves
    # column 6 out (kv head 6 is not computed): its 4 output streams would otherwise
    # fill every channel from row 2 down, leaving none for trace packets
    att = [c != 6 or not trace for c in range(NCOL)]
    has_o = att
    u8 = np.dtype[np.uint8]
    tile_ty = np.ndarray[(TILE,), u8]
    quad_ty = np.ndarray[(NROW * TILE,), u8]
    # per-core state (qwen3_stack_core.cc: ST_ATTN, ST_TAIL)
    st_ty = {
        False: np.ndarray[(29504,), u8],
        True: np.ndarray[(29504,), u8],
    }  # ST_ATTN (>= ST_TAIL)
    ctl_ty = np.ndarray[(16,), np.dtype[np.int32]]
    i32 = np.int32
    tb, tt = st_ty[False], st_ty[True]
    kb_ = lambda name, *a: Kernel(name, obj_body, list(a))  # noqa: E731
    kt_ = lambda name, *a: Kernel(name + "_t", obj_tail, list(a))  # noqa: E731
    k = {
        False: dict(
            params=kb_("oc_params", tile_ty, tb),
            phase=kb_("og_phase", ctl_ty, tb, i32, i32),
            load=kb_("og_load", tile_ty, tb, i32, i32, i32, i32),
            ssq_head=kb_("og_ssq_head", tb, i32, i32),
            ssq_mid=kb_("og_ssq_mid", tb, i32, i32),
            mac=kb_("og_mac", tile_ty, tb, i32, i32, i32, i32, i32),
            oa_q=kb_("oa_q", tile_ty, tb),
            oa_k=kb_("oa_k", tile_ty, tb, i32, i32, i32),
            oa_soft=kb_("oa_soft", tb, i32, i32),
            oa_v=kb_("oa_v", tile_ty, tb, i32, i32),
            oa_out=kb_("oa_out", tile_ty, tb),
            qp_gamma=kb_("qp_gamma", tile_ty, tb),
            qp_rope=kb_("qp_rope", tile_ty, tb),
            qp_qk=kb_("qp_qk", tile_ty, tile_ty, tb, i32),
            qp_v=kb_("qp_v", tile_ty, tile_ty, tb),
        ),
        True: dict(
            params=kt_("oc_params", tile_ty, tt),
            phase=kt_("og_phase", ctl_ty, tt, i32, i32),
            take_x=kt_("oc_take_x", tile_ty, tt),
            load=kt_("og_load", tile_ty, tt, i32, i32, i32, i32),
            ssq_tail=kt_("og_ssq_tail", tt, i32, i32),
            copy=kt_("og_copy", tile_ty, tt),
            mac=kt_("og_mac", tile_ty, tt, tile_ty, i32, i32, i32, i32, i32),
            qp_gamma=kt_("qp_gamma", tile_ty, tt),
            qp_rope=kt_("qp_rope", tile_ty, tt),
            qp_qk=kt_("qp_qk", tile_ty, tile_ty, tt, i32),
            qp_v=kt_("qp_v", tile_ty, tile_ty, tt),
            oa_q=kt_("oa_q", tile_ty, tt),
            oa_k=kt_("oa_k", tile_ty, tt, i32, i32, i32),
            oa_soft=kt_("oa_soft", tt, i32, i32),
            oa_v=kt_("oa_v", tile_ty, tt, i32, i32),
            oa_out=kt_("oa_out", tile_ty, tt),
        ),
    }

    # A: column c through memtile c, split into the 4 row blocks
    a3 = [ObjectFifo(quad_ty, depth=1, name=f"a3_{c}") for c in range(NCOL)]
    a1 = [
        a3[c]
        .cons()
        .split(
            [r * TILE for r in range(NROW)],
            obj_types=[tile_ty] * NROW,
            names=[f"a1_{c}_{r}" for r in range(NROW)],
            depths=[1 if c == NCOL - 1 else 2] * NROW,
            tile=Tile(c, 1),
        )
        for c in range(NCOL)
    ]
    # W: column c's slice through memtile c, broadcast to the column's 4 cores
    b3 = [ObjectFifo(tile_ty, depth=2, name=f"b3_{c}") for c in range(NCOL)]
    b1 = [
        b3[c].cons().forward(obj_type=tile_ty, depth=2, name=f"b1_{c}", tile=Tile(c, 1))
        for c in range(NCOL)
    ]
    # outputs: the column's 4 cores joined in memtile c (GEMM tails: column 7; attention: 0-6)
    o3 = [
        ObjectFifo(quad_ty, depth=1, name=f"o3_{c}") if has_o[c] else None
        for c in range(NCOL)
    ]
    o1 = [
        (
            o3[c]
            .prod()
            .join(
                [r * TILE for r in range(NROW)],
                obj_types=[tile_ty] * NROW,
                names=[f"o1_{c}_{r}" for r in range(NROW)],
                depths=[2] * NROW,
                tile=Tile(c, 1),
            )
            if has_o[c]
            else None
        )
        for c in range(NCOL)
    ]

    def gemm(role, A, B, O, st, ctl, kk):
        """One GEMM phase; its counts come from ctl (og_phase), so every phase of the step
        runs this one piece of code."""
        nch, ngi = ctl[0], ctl[4]
        for gi in range_(ngi):
            for ch in range_(nch):
                for part in range_(ctl[7]):
                    x = A.acquire(1)
                    kk["load"](x, st, ch, part, ctl[7], gi)
                    A.release(1)
        for gi in range_(ngi):
            for _ in range_(ctl[3]):
                if role == "head":
                    kk["ssq_head"](st, nch, gi)
                elif role == "mid":
                    kk["ssq_mid"](st, nch, gi)
                else:
                    kk["ssq_tail"](st, nch, gi)
        for u in range_(ctl[1]):
            if role == "tail":
                # the output object first: the epilogues write into it
                o = O.acquire(1)
                for _ in range_(ctl[5]):
                    xt = A.acquire(1)
                    kk["take_x"](xt, st)
                    A.release(1)
            rcode = {"head": 0, "mid": 1, "tail": 2}[role]
            for g in range_(ctl[2]):
                # the cascade's get / put ride in the first / last chunk's mac, so
                # does the tails' epilogue
                for ch in range_(nch):
                    w = B.acquire(1)
                    if role == "tail":
                        kk["mac"](w, st, o, ch, nch, rcode, g, ngi)
                    else:
                        kk["mac"](w, st, ch, nch, rcode, g, ngi)
                    B.release(1)
            if role == "tail":
                O.release(1)
                # a pair's second row group: its tile waited in the state
                for _ in range_(ctl[8]):
                    o2 = O.acquire(1)
                    kk["copy"](o2, st)
                    O.release(1)

    def post(A, O, st, kk, r):
        """QKV post-pass of column c, core r (row block r): gamma, then per row group the
        RoPE tile, q heads 2c and 2c + 1, and two of kv head c's K / V tiles (rows 0, 1:
        K of sequences 0, 1 -> q/k-norm + RoPE; rows 2, 3: V -> V^T)."""
        g_ = A.acquire(1)
        kk["qp_gamma"](g_, st)
        A.release(1)
        for _ in range_(NG):
            rt = A.acquire(1)
            kk["qp_rope"](rt, st)
            A.release(1)
            for i in range(4):
                x = A.acquire(1)
                y = O.acquire(1)
                if i < 2 or r < 2:
                    kk["qp_qk"](x, y, st, 0 if i < 2 else 1)
                else:
                    kk["qp_v"](x, y, st)
                A.release(1)
                O.release(1)

    def attention(A, B, O, st, kk):
        """Column c: kv head c, both sequences; per row group (round), core r: query head
        2c + (r >> 1), query block r & 1 of the group. q comes on A, the K and V^T tiles of
        every group (chunk j), shared by the column, on B."""
        for s in range_(2):
            for _ in range_(NG):
                q = A.acquire(1)
                kk["oa_q"](q, st)
                A.release(1)
                for j in range_(NG):
                    for kb in range_(2):
                        kt = B.acquire(1)
                        kk["oa_k"](kt, st, kb, s, j)
                        B.release(1)
                    kk["oa_soft"](st, s, j)
                    for kb in range_(2):
                        vt = B.acquire(1)
                        kk["oa_v"](vt, st, kb, j)
                        B.release(1)
                o = O.acquire(1)
                kk["oa_out"](o, st)
                O.release(1)

    def make_body(c, r):
        role = "head" if c == 0 else "tail" if c == NCOL - 1 else "mid"

        def body(A, B, *rest):
            O, st, ctl, kargs = (
                (rest[0], rest[1], rest[2], rest[3:])
                if has_o[c]
                else (None, rest[0], rest[1], rest[2:])
            )
            kk = dict(zip(sorted(k[role == "tail"]), kargs))
            p = A.acquire(1)
            kk["params"](p, st)
            A.release(1)
            for _ in range_(NL * (3 * NG + NP) + NP):
                kk["phase"](ctl, st, NL, NG)
                gemm(role, A, B, O, st, ctl, kk)
                if att[c]:
                    for _ in range_(ctl[6]):
                        post(A, O, st, kk, r)
                        attention(A, B, O, st, kk)

        return body

    workers = []
    for r in range(NROW):
        row = []
        for c in range(NCOL):
            tail = c == NCOL - 1
            st = Buffer(st_ty[tail], name=f"st_{c}_{r}")
            ctl = Buffer(ctl_ty, name=f"ctl_{c}_{r}")
            kk = k[tail]
            args = (
                [a1[c][r].cons(), b1[c].cons()]
                + ([o1[c][r].prod()] if has_o[c] else [])
                + [st, ctl]
                + [kk[n] for n in sorted(kk)]
            )
            # dynamic FIFO lowering keeps the loops (the static one unrolls them by the
            # FIFO depths, which does not fit the tails' 16 KB of program memory); the
            # stack is set by hand: aiecc's measurement falls back to 1 KB for the tails,
            # whose output epilogue alone has a 3.4 KB frame
            row.append(
                Worker(
                    make_body(c, r),
                    args,
                    tile=Tile(c, 2 + r),
                    stack_size=0x1000,
                    dynamic_objfifo_lowering=True,
                )
            )
        for w0, w1 in zip(row, row[1:]):
            CascadeFlow(w0, w1)
        workers += row

    act_len = L.ntiles * TILE
    w_len = NCOL * (NL * w_objects_per_layer() + HEAD_W) * TILE
    log_len = NG * HU * NROW * TILE
    wl = w_objects_per_layer()
    GS = L.gs

    def sequence(Ai, Wi, Lo, *hd):
        ap, bp = hd[:NCOL], hd[NCOL : 2 * NCOL]
        it = iter(hd[2 * NCOL :])
        oc = [next(it) if has_o[c] else None for c in range(NCOL)]
        col_w = NL * wl + HEAD_W  # objects of one column's weight stream

        def weights(c, off, n, group, reps=NG):
            """n objects of column c's stream from off, once per row group (or pair)."""
            bp[c].fill(
                Wi, tap(w_len, c * col_w + off, [(reps, 0), (n, 1)]), group=group
            )

        def pair_drains(dst, ten_len, base, units, gstride, group):
            """The tails' output of a paired phase: per pair, unit by unit, each group's tiles."""
            per = -(-units // -(-units // 63))  # units per transfer (outer dim <= 63)
            for p_ in range(NP):
                ngi = min(2, NG - 2 * p_)
                for u0 in range(0, units, per):
                    n = min(per, units - u0)
                    oc[7].drain(
                        dst,
                        tap(
                            ten_len,
                            base + 2 * p_ * gstride + 4 * u0,
                            [(n, 4), (ngi, gstride), (4, 1)],
                        ),
                        wait=p_ == NP - 1 and u0 + n == units,
                        group=group,
                    )

        def per_group(name, off, n):
            """n tiles at off of region name, in every row group."""
            return tap(act_len, L[name, 0] + off, [(NG, GS), (n, 1)])

        # the step's parameters to every core; the first layer's QKV weights
        tg = TaskGroup()
        for c in range(NCOL):
            ap[c].fill(Ai, tap(act_len, L["PARAMS"], [(NROW, 0)]), group=tg)
        wq = TaskGroup()
        for c in range(NCOL):
            weights(c, 0, W_QKV, wq)
        for layer in range(NL):
            base = layer * wl
            # QKV (per group): chunk = X unit c. Out per group: q heads (column c's pair),
            # K heads, V heads into QR
            for c in range(NCOL):
                ap[c].fill(Ai, per_group("X", 4 * c, 4), group=tg)
            prev = None
            for g in range(NG):
                dg = TaskGroup()
                oc[7].drain(
                    Ai,
                    tap(act_len, L["QR", g] + QR_Q, [(NCOL, QR_COL), (8, 1)]),
                    group=dg,
                )
                for v in range(2):
                    oc[7].drain(
                        Ai,
                        tap(
                            act_len,
                            L["QR", g] + QR_KV + 2 * v,
                            [(NCOL, QR_COL), (2, 4), (2, 1)],
                        ),
                        wait=v == 1,
                        group=dg,
                    )
                if prev is not None:
                    prev.finish()
                prev = dg
            prev.finish()
            tg.finish()
            wq.finish()
            # QKV post-pass: per column the layer's gamma, then per group its QR run (RoPE,
            # q heads, K / V); back to the same places
            tg = TaskGroup()
            for c in range(NCOL):
                if att[c]:
                    ap[c].fill(
                        Ai, tap(act_len, L["GAMMA"] + layer, [(NROW, 0)]), group=tg
                    )
                    ap[c].fill(Ai, per_group("QR", QR_COL * c, QR_COL), group=tg)
            for c in range(NCOL):
                if att[c]:
                    oc[c].drain(
                        Ai,
                        per_group("QR", QR_COL * c + QR_Q, QR_COL - QR_Q),
                        wait=True,
                        group=tg,
                    )
            tg.finish()
            # attention, column c = kv head c, per sequence s: per round (row group) q of
            # heads 2c, 2c+1 x query blocks 2s, 2s+1 on A; per round all groups' K, V^T
            # tiles on B. Then the drains, and only then the rest of the layer's weights
            # (a full task queue makes the command processor wait)
            tg = TaskGroup()
            for s_ in range(2):
                for c in range(NCOL):
                    if att[c]:
                        ap[c].fill(
                            Ai,
                            tap(
                                act_len,
                                L["QR", 0] + QR_COL * c + QR_Q + 2 * s_,
                                [(NG, GS), (2, 4), (2, 1)],
                            ),
                            group=tg,
                        )
                for c in range(NCOL):
                    if att[c]:
                        bp[c].fill(
                            Ai,
                            tap(
                                act_len,
                                L["QR", 0] + QR_COL * c + QR_KV + 4 * s_,
                                [(NG, 0), (NG, GS), (4, 1)],
                            ),
                            group=tg,
                        )
            for s_ in range(2):
                for c in range(NCOL):
                    if att[c]:
                        oc[c].drain(
                            Ai,
                            tap(
                                act_len,
                                L["ATT", 0] + 2 * c * 4 + 2 * s_,
                                [(NG, GS), (2, 4), (2, 1)],
                            ),
                            wait=s_ == 1,
                            group=tg,
                        )
            wr = TaskGroup()
            for c in range(NCOL):
                weights(c, base + W_QKV, W_O, wr)
                weights(c, base + W_QKV + W_O, W_GU, wr, NP)
                weights(c, base + W_QKV + W_O + W_GU, W_DOWN, wr)
            wq = TaskGroup()
            for c in range(NCOL):
                if layer + 1 < NL:
                    weights(c, base + wl, W_QKV, wq)
                else:
                    weights(c, NL * wl, HEAD_W, wq, NP)
            tg.finish()
            # O: chunks = ATT heads 2c, 2c+1; tails: heads 14, 15 + the residual x
            tg = TaskGroup()
            for c in range(NCOL):
                ap[c].fill(
                    Ai, per_group("ATT", 8 * c, 8 if c < NCOL - 1 else 40), group=tg
                )
            oc[7].drain(Ai, per_group("X1", 0, 32), wait=True, group=tg)
            tg.finish()
            # gate/up, per pair of groups: units alternate gate i, up i
            tg = TaskGroup()
            for c in range(NCOL):
                ap[c].fill(Ai, per_group("X1", 4 * c, 4), group=tg)
            pair_drains(Ai, act_len, L["GU", 0], 48, GS, tg)
            tg.finish()
            # down: chunk = intermediate block 3c + ch, as its gate then its up tiles;
            # tails: blocks 21-23 + the residual x1
            tg = TaskGroup()
            for c in range(NCOL):
                ap[c].fill(
                    Ai, per_group("GU", 24 * c, 24 if c < NCOL - 1 else 56), group=tg
                )
            oc[7].drain(Ai, per_group("X", 0, 32), wait=True, group=tg)
            tg.finish()
            wr.finish()
            tg = TaskGroup()
        # the head, per pair of groups: chunk = X unit c
        for c in range(NCOL):
            ap[c].fill(Ai, per_group("X", 4 * c, 4), group=tg)
        pair_drains(Lo, log_len, 0, HU, HU * NROW, tg)
        tg.finish()
        wq.finish()

    # each column's streams through its own shim (left to the placer, column 0's weights
    # came in through shim 7 and column 7's through shim 0, across the whole array)
    hd = (
        [f.prod(tile=Tile(c, 0)) for c, f in enumerate(a3)]
        + [f.prod(tile=Tile(c, 0)) for c, f in enumerate(b3)]
        + [f.cons(tile=Tile(c, 0)) for c, f in enumerate(o3) if f is not None]
    )
    rt = Runtime(
        sequence,
        [
            np.ndarray[(act_len,), u8],
            np.ndarray[(w_len,), u8],
            np.ndarray[(log_len,), u8],
        ]
        + hd,
    )
    prog = Program(iron.get_current_device(), rt, workers=workers)
    if trace:
        # row 0's head, a middle core and the tail; the trace buffer becomes the
        # sequence's last argument
        from aie.dialects.aie import CoreEventAIE2P as E

        prog.enable_trace(
            trace_size=trace,
            workers=[workers[5], workers[7]],
            egress_shim_col=6,
            coretile_events=[
                E.INSTR_CALL,
                E.INSTR_RETURN,
                E.LOCK_STALL,
                E.CASCADE_STALL,
                E.STREAM_STALL,
                E.MEMORY_STALL,
                E.INSTR_VECTOR,
                E.ACTIVE,
            ],
        )
    return prog.resolve_program()


def build(params, out_dir):
    import aie.iron as iron
    from aie.iron import CompileTime, In, Out

    def fn(
        ACT: In,
        W: In,
        LOG: Out,
        *,
        NL: CompileTime[int],
        NG: CompileTime[int],
        HU: CompileTime[int],
        obj_body: CompileTime[str],
        obj_tail: CompileTime[str],
        trace: CompileTime[int] = 0,
    ):
        return design(
            ACT,
            W,
            LOG,
            NL=NL,
            NG=NG,
            HU=HU,
            obj_body=obj_body,
            obj_tail=obj_tail,
            trace=trace,
        )

    jit = iron.jit(fn, aiecc_flags=["--reclaim-runtime-bds"])
    os.makedirs(out_dir, exist_ok=True)
    jit.specialize(**params).compile(
        xclbin_path=os.path.join(out_dir, "final.xclbin"),
        inst_path=os.path.join(out_dir, "insts.bin"),
    )


if __name__ == "__main__":
    build(json.loads(sys.argv[1]), sys.argv[2])
