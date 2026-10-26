# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""tl.extra.npu.conv3x3: a 3x3 int8 convolution over the whole array.

Every core keeps its weights for the whole image; the input-channel reduction is split
across a chain of cores whose partial sums meet over the AIE cascade (see
amd_triton_npu/language/npu/conv3x3.py). Checks shapes that exercise every way the op
splits a layer (chains of 2 and 4 cores, 1-4 partitions, narrow tiles side by side,
padded channels, batches) against the CPU -- the same integers, bit for bit -- the
first at a second shift set without a rebuild; then all of them again, interleaved, as
the kernels of one shared ELF (`share_context`, how a model's layers run); and with
--bench times large ones: NPU time alone, and per call with the host layouts.

usage: python conv3x3_cascade.py [--bench] [--iters N] [--quick]
Exit status: 0 pass, 1 failure.
"""

import argparse
import time

import numpy as np
import torch

from triton.language.extra.npu import conv3x3

SHAPES = [  # (N, Ci, Co, H, W, relu)
    (1, 256, 256, 64, 64, False),
    (2, 3, 64, 24, 40, False),
    (1, 64, 128, 20, 28, True),
    (1, 512, 512, 14, 14, False),
    (3, 384, 96, 9, 7, False),
    (1, 128, 256, 16, 100, True),
]


def verdict(op, x, label):
    y = op(x)
    want = op.reference(x)
    same = torch.equal(y, want)
    diff = "" if same else f" ({int((y != want).sum())} of {y.numel()} differ)"
    print(f"{label}: {'PASS' if same else 'FAIL'}{diff}")
    return same


def check(n, ci, co, h, w, relu=False, shifts=(12,)):
    rng = np.random.default_rng(ci * 7 + co + h + w + n)
    x = torch.as_tensor(rng.integers(-60, 60, (n, ci, h, w), dtype=np.int8))
    wt = torch.as_tensor(rng.integers(-40, 40, (co, ci, 3, 3), dtype=np.int8))
    op = conv3x3.Conv3x3(wt, height=h, width=w, shift=shifts[0], relu=relu, batch=n)
    cfg = f"L={op.chain} CG={op.cg} OGP={op.ogp} G={op.groups} R={op.parts}"
    ok = True
    for s in shifts:
        if s != op.shift:
            op.set_shift(s)
        ok &= verdict(op, x, f"{n}x{ci}x{h}x{w} -> {co} relu={relu} shift={s} [{cfg}]")
    return ok, op, x


def median_ms(fn, iters):
    fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return sorted(ts)[len(ts) // 2] * 1e3


def bench(n, ci, co, h, w, iters):
    rng = np.random.default_rng(0)
    x = torch.as_tensor(rng.integers(-60, 60, (n, ci, h, w), dtype=np.int8))
    wt = torch.as_tensor(rng.integers(-40, 40, (co, ci, 3, 3), dtype=np.int8))
    op = conv3x3.Conv3x3(wt, height=h, width=w, shift=12, batch=n)
    npu = median_ms(op.launch, iters)
    call = median_ms(lambda: op(x), iters)
    gop = 2 * n * ci * co * h * w * 9 / 1e9
    print(
        f"{n}x{ci}x{h}x{w} -> {co}: NPU {npu:.3f} ms = {gop / npu:.1f} TOPS; "
        f"per call with host layouts {call:.3f} ms = {gop / call:.1f} TOPS"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--quick", action="store_true", help="the first shape only")
    a = ap.parse_args()
    shapes = SHAPES[:1] if a.quick else SHAPES
    runs = [check(*s[:5], relu=s[5], shifts=(12, 10) if i == 0 else (12,))
            for i, s in enumerate(shapes)]  # fmt: skip
    ok = all(r[0] for r in runs)
    if len(runs) > 1 and conv3x3.program_format() == "elf":
        conv3x3.share_context([op for _, op, _ in runs])
        for rep in range(2):  # every launch follows another design's
            for _, op, x in runs:
                ok &= verdict(op, x, f"shared context, pass {rep}, {tuple(x.shape)}")
    if a.bench:
        bench(1, 256, 256, 1536, 64, a.iters)
        bench(1, 256, 256, 256, 256, a.iters)
    raise SystemExit(0 if ok else 1)
