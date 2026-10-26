# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""The AIE design of ``tl.extra.npu.conv3x3`` (IRON), compiled to a full ELF (or an
xclbin and an instruction stream), and the merge of several into one ELF.

Run as a script in its own process: mlir-aie's MLIR bindings and mlir-air's (which the
Triton backend has loaded) cannot share an interpreter. Imports nothing from Triton.

    python conv3x3_design.py \\
        '{"ROWS": .., "CG": .., "L": .., "OGP": .., "G": .., "R": .., "NSLAB": ..,
          "relu": .., "obj": .., "format": "elf"}' OUTDIR
    python conv3x3_design.py --merge OUTDIR design1/design.mlir ...

A chain of L cores (L = 2 or 4, one per part of the input channels, joined over the
cascade) computes 16*OGP output channels of one 32-pixel strip, one output line per
call. The array holds R partitions x 2 strips x G channel groups of chains, with
R * G = 8 for L = 2 (two chains per column) and 4 for L = 4 (one chain per column).
Partitions are independent sets of lines (other rows or images); the G chains of a
partition and strip read the same input lines. One launch runs NSLAB slabs of output
channels one after the other: every core reloads its weights and the input lines are
streamed again for each.

Data movement:
  input   stream (p, s) carries one line of part p, strip s for every partition;
          memtile 2p+s splits it and broadcasts each partition's line to its G chains
  weights group g's L slices through one memtile (g, or 4+g when L = 2 and the
          input memtiles already split R > 1 ways), slice p broadcast to the 2R chains
          of group g at position p
  output  column c's tails through memtile c to the host, one stream per column
The requantization shift is not part of the design: it rides at the end of each core's
weights.
"""

import json
import os
import sys

import numpy as np

OWS, WPS = 32, 40
WTRAILER = 64  # bytes after each core's weights; the first int32 is the shift


def geometry(CG, OGP):
    """Bytes of one input line, one weight slice and one chain's output line."""
    return CG * WPS * 8, OGP * 3 * CG * 3 * 2 * 64 + WTRAILER, 2 * OGP * OWS * 8


def conv3x3_cascade(
    x_in, w_in, y_out, *, ROWS, CG, L, OGP, G, R, NSLAB, obj, relu=False
):
    import aie.iron as iron
    from aie.helpers.taplib import TensorAccessPattern
    from aie.iron import CascadeFlow, Kernel, ObjectFifo, Program, Runtime, Worker
    from aie.iron.controlflow import range_
    from aie.iron.device import Tile

    assert L in (2, 4) and R * G == 16 // L, (L, R, G)
    cpc = 4 // L  # chains per column
    line, wsl, ochain = geometry(CG, OGP)
    i8 = np.dtype[np.int8]
    line_ty = np.ndarray[(line,), i8]
    xs_ty = np.ndarray[(R * line,), i8]
    w_ty = np.ndarray[(wsl,), i8]
    wg_ty = np.ndarray[(L * wsl,), i8]
    o_ty = np.ndarray[(ochain,), i8]
    ocol_ty = np.ndarray[(cpc * ochain,), i8]

    tail = "conv3x3_tail_relu" if relu else "conv3x3_tail"
    args = [line_ty, line_ty, line_ty, w_ty]
    kern = {
        "head": Kernel("conv3x3_head", obj, args + [line_ty], stack_size_override=1024),
        "mid": Kernel("conv3x3_mid", obj, args + [line_ty], stack_size_override=1024),
        "tail": Kernel(tail, obj, args + [o_ty], stack_size_override=1024),
    }

    def column(r, s, g):
        return r * G + g if L == 2 else (r * 2 + s) * G + g

    def core_row(s, p):
        return 5 - 2 * s - p if L == 2 else 5 - p

    # input: stream (p, s) through memtile 2p+s, one line of every partition per object
    x_l3, x_l1 = {}, {}
    for p in range(L):
        for s in range(2):
            m = 2 * p + s
            x_l3[p, s] = ObjectFifo(xs_ty, depth=2, name=f"x_l3_{p}{s}")
            names = [f"x_l1_{p}{s}{r}" for r in range(R)]
            if R == 1:
                x_l1[p, s, 0] = (
                    x_l3[p, s]
                    .cons()
                    .forward(obj_type=line_ty, depth=4, name=names[0], tile=Tile(m, 1))
                )
                continue
            parts = (
                x_l3[p, s]
                .cons()
                .split(
                    [r * line for r in range(R)],
                    tile=Tile(m, 1),
                    depths=[4] * R,
                    obj_types=[line_ty] * R,
                    names=names,
                )
            )
            for r in range(R):
                x_l1[p, s, r] = parts[r]
    # weights: group g's L slices through memtile g, or 4+g where the input memtiles
    # 0..3 already spend R > 1 of their 6 output channels on the input
    w_mem = [g if L == 4 or R == 1 else 4 + g for g in range(G)]
    w_l3, w_l1 = {}, {}
    for g in range(G):
        w_l3[g] = ObjectFifo(wg_ty, depth=1, name=f"w_l3_{g}")
        parts = (
            w_l3[g]
            .cons()
            .split(
                [p * wsl for p in range(L)],
                tile=Tile(w_mem[g], 1),
                depths=[1] * L,
                obj_types=[w_ty] * L,
                names=[f"w_l1_{g}_{p}" for p in range(L)],
            )
        )
        for p in range(L):
            w_l1[g, p] = parts[p]
    # output: column c's tails through memtile c, one stream per column
    y_l3 = [ObjectFifo(ocol_ty, depth=2, name=f"y_l3_{c}") for c in range(8)]
    y_l1 = [
        y_l3[c]
        .prod()
        .join(
            [i * ochain for i in range(cpc)],
            tile=Tile(c, 1),
            depths=[4] * cpc,
            obj_types=[o_ty] * cpc,
            names=[f"y_l1_{c}{i}" for i in range(cpc)],
        )
        for c in range(8)
    ]

    def head_body(x, w, k):
        for _ in range_(NSLAB):
            wl = w.acquire(1)
            for _ in range_(ROWS):
                ls = x.acquire(3)
                k(ls[0], ls[1], ls[2], wl, ls[0])
                x.release(1)
            x.release(2)
            w.release(1)

    def tail_body(x, w, y, k):
        for _ in range_(NSLAB):
            wl = w.acquire(1)
            for _ in range_(ROWS):
                ls = x.acquire(3)
                yo = y.acquire(1)
                k(ls[0], ls[1], ls[2], wl, yo)
                y.release(1)
                x.release(1)
            x.release(2)
            w.release(1)

    workers = []
    for r in range(R):
        for s in range(2):
            for g in range(G):
                c = column(r, s, g)
                chain = []
                for p in range(L):
                    x, w = x_l1[p, s, r].cons(), w_l1[g, p].cons()
                    tile = Tile(c, core_row(s, p))
                    if p < L - 1:
                        k = kern["head" if p == 0 else "mid"]
                        body, wargs = head_body, [x, w, k]
                    else:
                        y = y_l1[c][s if L == 2 else 0].prod()
                        body, wargs = tail_body, [x, w, y, kern["tail"]]
                    chain.append(Worker(body, wargs, tile=tile, stack_size=0x800))
                for a, b in zip(chain, chain[1:]):
                    CascadeFlow(a, b)
                workers += chain

    # DDR layouts are stream-major, so every transfer is one contiguous run (the
    # input's, once per slab):
    #   x_in  [L parts][2 strips][ROWS+2][R][CG][WPS][8]
    #   w_in  [NSLAB][G][L][slice]
    #   y_out [NSLAB][ROWS][8 columns][cpc][2*OGP][OWS][8]
    xs_b = (ROWS + 2) * R * line
    orow_b = 8 * cpc * ochain
    xkeys = [(p, s) for p in range(L) for s in range(2)]
    x_taps = {
        k: TensorAccessPattern(
            (2 * L, xs_b),
            offset=(k[0] * 2 + k[1]) * xs_b,
            sizes=[NSLAB, 1, 1, xs_b],
            strides=[0, 0, 0, 1],
        )
        for k in xkeys
    }
    w_taps = [
        TensorAccessPattern(
            (NSLAB, G * L * wsl),
            offset=g * L * wsl,
            sizes=[1, NSLAB, 1, L * wsl],
            strides=[0, G * L * wsl, 0, 1],
        )
        for g in range(G)
    ]
    y_taps = [
        TensorAccessPattern(
            (NSLAB * ROWS, orow_b),
            offset=c * cpc * ochain,
            sizes=[1, 1, NSLAB * ROWS, cpc * ochain],
            strides=[0, 0, orow_b, 1],
        )
        for c in range(8)
    ]

    def sequence(xi, wi, yo, *hd):
        nx = len(xkeys)
        xps, wps, ycs = hd[:nx], hd[nx : nx + G], hd[nx + G :]
        for g in range(G):
            wps[g].fill(wi, w_taps[g])
        for i, k in enumerate(xkeys):
            xps[i].fill(xi, x_taps[k])
        for c in range(8):
            ycs[c].drain(yo, y_taps[c], wait=True)

    x_ty = np.ndarray[(2 * L * xs_b,), i8]
    wall_ty = np.ndarray[(NSLAB * G * L * wsl,), i8]
    y_ty = np.ndarray[(NSLAB * ROWS * orow_b,), i8]
    hd = (
        [x_l3[k].prod() for k in xkeys]
        + [w_l3[g].prod() for g in range(G)]
        + [y_l3[c].cons() for c in range(8)]
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
        L: CompileTime[int],
        OGP: CompileTime[int],
        G: CompileTime[int],
        R: CompileTime[int],
        NSLAB: CompileTime[int],
        obj: CompileTime[str],
        relu: CompileTime[bool] = False,
    ):
        return conv3x3_cascade(
            x_in,
            w_in,
            y_out,
            ROWS=ROWS,
            CG=CG,
            L=L,
            OGP=OGP,
            G=G,
            R=R,
            NSLAB=NSLAB,
            obj=obj,
            relu=relu,
        )

    # long runs are split by the shim into more BDs than its 16
    fn = iron.jit(design, aiecc_flags=["--reclaim-runtime-bds"])
    os.makedirs(out_dir, exist_ok=True)
    fmt = params.pop("format", "xclbin")
    if fmt == "elf":
        # the runtime sequence configures the device itself (load_pdi), so designs can
        # also be merged into one ELF (`merge`); the module is kept for that, since
        # IRON may copy a cached build without lowering it here
        d = fn.specialize(full_elf=True, **params)
        d.compile(full_elf_path=os.path.join(out_dir, "conv.elf"))
        with open(os.path.join(out_dir, "kernel.txt"), "w") as f:
            f.write(d.compilable._full_elf_kernel_name)
        with open(os.path.join(out_dir, "design.mlir"), "w") as f:
            f.write(d.as_mlir(None, None, None))
    else:
        fn.specialize(**params).compile(
            xclbin_path=os.path.join(out_dir, "final.xclbin"),
            inst_path=os.path.join(out_dir, "insts.bin"),
        )


def merge(out_dir, mlir_paths):
    """One full ELF whose kernel ``conv<k>:sequence`` is design ``mlir_paths[k]``, so
    the layers of a model share one hardware context.

    Each design's runtime sequence configures its device itself (load_pdi, expanded
    into explicit writes: a reloaded PDI does not undo another device's routes and
    locks), so any kernel can follow any other.
    """
    import subprocess

    import aie.dialects.aie  # noqa: F401  (registers the dialects)
    import aie.dialects.aiex  # noqa: F401
    from aie.ir import Context, FlatSymbolRefAttr, Location, Module, StringAttr

    devices = []
    with Context(), Location.unknown():
        for k, path in enumerate(mlir_paths):
            with open(path) as f:
                module = Module.parse(f.read())
            for dev in module.body.operations:
                if dev.operation.name != "aie.device":
                    continue
                name = f"conv{k}"
                dev.operation.attributes["sym_name"] = StringAttr.get(name)
                for op in dev.regions[0].blocks[0].operations:
                    if op.operation.name != "aie.runtime_sequence":
                        continue
                    for inner in op.regions[0].blocks[0].operations:
                        if inner.operation.name == "aiex.npu.load_pdi":
                            ref = FlatSymbolRefAttr.get(name)
                            inner.operation.attributes["device_ref"] = ref
                devices.append(dev.operation.get_asm(enable_debug_info=False))
    os.makedirs(out_dir, exist_ok=True)
    mlir = os.path.join(out_dir, "model.mlir")
    with open(mlir, "w") as f:
        f.write("module {\n" + "\n".join(devices) + "\n}\n")
    cmd = [
        "aiecc",
        "--unified",
        f"--peano={os.environ['PEANO_INSTALL_DIR']}",
        "--get-full-elf",
        f"--full-elf-name={os.path.join(out_dir, 'model.elf')}",
        f"--tmpdir={os.path.join(out_dir, 'model.prj')}",
        "--reclaim-runtime-bds",
        "--expand-load-pdis",
        mlir,
    ]
    r = subprocess.run(cmd, cwd=out_dir, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"aiecc failed:\n{(r.stdout + r.stderr)[-6000:]}")


if __name__ == "__main__":
    if sys.argv[1] == "--merge":
        merge(sys.argv[2], sys.argv[3:])
    else:
        build(json.loads(sys.argv[1]), sys.argv[2])
