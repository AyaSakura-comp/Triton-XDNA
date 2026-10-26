# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""The AIE design of ``tl.extra.npu.conv3x3`` (IRON), compiled to an xclbin and an
instruction stream.

Run as a script in its own process: mlir-aie's MLIR bindings and mlir-air's (which the
Triton backend has loaded) cannot share an interpreter. Imports nothing from Triton.

    python conv3x3_design.py \
        '{"ROWS": .., "CG": .., "NCOL": .., "relu": .., "obj": ..}' OUTDIR

Layout of the design (see conv3x3.py): NCOL columns x 2 chains (one per 32-pixel strip)
x 2 cores (head: input-channel half 0, tail: half 1, joined over the cascade). The
requantization shift is not part of the design: it rides at the end of each core's
weights.
"""

import json
import os
import sys

import numpy as np

OWS, WPS, BAND, OGP = 32, 40, 64, 2
WTRAILER = 64  # bytes after each core's weights; the first int32 is the shift


def conv3x3_cascade(x_in, w_in, y_out, *, ROWS, CG, NCOL, obj, relu=False):
    import aie.iron as iron
    from aie.helpers.taplib import TensorAccessPattern
    from aie.iron import CascadeFlow, Kernel, ObjectFifo, Program, Runtime, Worker
    from aie.iron.controlflow import range_
    from aie.iron.device import Tile

    S = BAND // OWS
    line = CG * WPS * 8
    wsl = OGP * 3 * CG * 3 * 2 * 64 + WTRAILER
    ostrip = 2 * OGP * OWS * 8
    i8 = np.dtype[np.int8]
    line_ty = np.ndarray[(line,), i8]
    w_ty = np.ndarray[(wsl,), i8]
    wcol_ty = np.ndarray[(2 * wsl,), i8]
    o_ty = np.ndarray[(ostrip,), i8]
    ocol_ty = np.ndarray[(S * ostrip,), i8]

    kargs = [line_ty, line_ty, line_ty, w_ty, line_ty]
    k_head = Kernel("conv3x3_head", obj, kargs, stack_size_override=1024)
    k_tail = Kernel(
        "conv3x3_tail_relu" if relu else "conv3x3_tail",
        obj,
        [line_ty, line_ty, line_ty, w_ty, o_ty],
        stack_size_override=1024,
    )

    # input half h of strip s: memtile 2h+s, broadcast to that chain position of
    # every column
    x_l3 = {
        (h, s): ObjectFifo(line_ty, depth=2, name=f"x_l3_{h}{s}")
        for h in range(2)
        for s in range(S)
    }
    x_l1 = {
        (h, s): x_l3[h, s]
        .cons()
        .forward(
            obj_type=line_ty, depth=4, name=f"x_l1_{h}{s}", tile=Tile(2 * h + s, 1)
        )
        for h in range(2)
        for s in range(S)
    }
    # column c's two weight slices through memtile c, each to both of its chains
    w_l3 = [ObjectFifo(wcol_ty, depth=1, name=f"w_l3_{c}") for c in range(NCOL)]
    w_l1 = [
        w_l3[c]
        .cons()
        .split(
            [0, wsl],
            tile=Tile(c, 1),
            depths=[1, 1],
            obj_types=[w_ty, w_ty],
            names=[f"w_l1_{c}_0", f"w_l1_{c}_1"],
        )
        for c in range(NCOL)
    ]
    # column c's two tails joined in memtile c, one stream to the host
    y_l3 = [ObjectFifo(ocol_ty, depth=2, name=f"y_l3_{c}") for c in range(NCOL)]
    y_l1 = [
        y_l3[c]
        .prod()
        .join(
            [s * ostrip for s in range(S)],
            tile=Tile(c, 1),
            depths=[4] * S,
            obj_types=[o_ty] * S,
            names=[f"y_l1_{c}{s}" for s in range(S)],
        )
        for c in range(NCOL)
    ]

    def head_body(x, w, kern):
        wl = w.acquire(1)
        for _ in range_(ROWS):
            ls = x.acquire(3)
            kern(ls[0], ls[1], ls[2], wl, ls[0])
            x.release(1)
        x.release(2)
        w.release(1)

    def tail_body(x, w, y, kern):
        wl = w.acquire(1)
        for _ in range_(ROWS):
            ls = x.acquire(3)
            yo = y.acquire(1)
            kern(ls[0], ls[1], ls[2], wl, yo)
            y.release(1)
            x.release(1)
        x.release(2)
        w.release(1)

    workers = []
    for c in range(NCOL):
        for s in range(S):
            head = Worker(
                head_body,
                [x_l1[0, s].cons(), w_l1[c][0].cons(), k_head],
                tile=Tile(c, 5 - 2 * s),
                stack_size=0x800,
            )
            tail = Worker(
                tail_body,
                [x_l1[1, s].cons(), w_l1[c][1].cons(), y_l1[c][s].prod(), k_tail],
                tile=Tile(c, 4 - 2 * s),
                stack_size=0x800,
            )
            CascadeFlow(head, tail)
            workers += [head, tail]

    # DDR layouts are stream-major, so every transfer is one contiguous run per row:
    #   x_in  [2 halves][S strips][ROWS+2][CG][WPS][8]
    #   y_out [ROWS][NCOL][S strips][4 groups][32][8]
    rows_in = ROWS + 2
    xs_b = rows_in * line
    orow_b = NCOL * S * ostrip
    xkeys = [(h, s) for h in range(2) for s in range(S)]
    x_taps = {
        k: TensorAccessPattern(
            (2 * S, xs_b),
            offset=(k[0] * S + k[1]) * xs_b,
            sizes=[1, 1, 1, xs_b],
            strides=[0, 0, 0, 1],
        )
        for k in xkeys
    }
    w_taps = [
        TensorAccessPattern(
            (NCOL, 2 * wsl),
            offset=c * 2 * wsl,
            sizes=[1, 1, 1, 2 * wsl],
            strides=[0, 0, 0, 1],
        )
        for c in range(NCOL)
    ]
    y_taps = [
        TensorAccessPattern(
            (ROWS, orow_b),
            offset=c * S * ostrip,
            sizes=[1, 1, ROWS, S * ostrip],
            strides=[0, 0, orow_b, 1],
        )
        for c in range(NCOL)
    ]

    def sequence(xi, wi, yo, *hd):
        nx = len(xkeys)
        xps, wps, ycs = hd[:nx], hd[nx : nx + NCOL], hd[nx + NCOL :]
        for c in range(NCOL):
            wps[c].fill(wi, w_taps[c])
        for i, k in enumerate(xkeys):
            xps[i].fill(xi, x_taps[k])
        for c in range(NCOL):
            ycs[c].drain(yo, y_taps[c], wait=True)

    x_ty = np.ndarray[(2 * S * xs_b,), i8]
    wall_ty = np.ndarray[(NCOL * 2 * wsl,), i8]
    y_ty = np.ndarray[(ROWS * orow_b,), i8]
    hd = (
        [x_l3[k].prod() for k in xkeys]
        + [w_l3[c].prod() for c in range(NCOL)]
        + [y_l3[c].cons() for c in range(NCOL)]
    )
    rt = Runtime(sequence, [x_ty, wall_ty, y_ty] + hd)
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def build(params, out_dir):
    import aie.iron as iron
    from aie.iron import CompileTime, In, Out

    def design(
        x_in: In,
        w_in: In,
        y_out: Out,
        *,
        ROWS: CompileTime[int],
        CG: CompileTime[int],
        NCOL: CompileTime[int],
        obj: CompileTime[str],
        relu: CompileTime[bool] = False,
    ):
        return conv3x3_cascade(
            x_in, w_in, y_out, ROWS=ROWS, CG=CG, NCOL=NCOL, obj=obj, relu=relu
        )

    # several bands make long runs, which the shim splits into more BDs than its 16
    fn = iron.jit(design, aiecc_flags=["--reclaim-runtime-bds"])
    os.makedirs(out_dir, exist_ok=True)
    fn.specialize(**params).compile(
        xclbin_path=os.path.join(out_dir, "final.xclbin"),
        inst_path=os.path.join(out_dir, "insts.bin"),
    )


if __name__ == "__main__":
    build(json.loads(sys.argv[1]), sys.argv[2])
