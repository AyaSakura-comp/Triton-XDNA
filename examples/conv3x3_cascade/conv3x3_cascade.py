# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""tl.extra.npu.conv3x3: a 3x3 int8 convolution over the whole array.

Every core keeps its weights for the whole image; the input-channel reduction is split
across two cores whose partial sums meet over the AIE cascade (see
amd_triton_npu/language/npu/conv3x3.py). Checks five shapes against the CPU (the same
integers, bit for bit), each at two shifts (the second one set without a rebuild),
and with --bench times large ones: NPU time alone, and per call with the host layouts.

usage: python conv3x3_cascade.py [--bench] [--iters N]
Exit status: 0 pass, 1 failure.
"""

import argparse
import time

import numpy as np
import torch

from triton.language.extra.npu import conv3x3


def check(ci, co, h, w, relu=False, shift=12):
    rng = np.random.default_rng(ci * 7 + co + h + w)
    x = torch.as_tensor(rng.integers(-60, 60, (ci, h, w), dtype=np.int8))
    wt = torch.as_tensor(rng.integers(-40, 40, (co, ci, 3, 3), dtype=np.int8))
    op = conv3x3.Conv3x3(wt, height=h, width=w, shift=shift, relu=relu)
    ok = True
    for s in (shift, shift - 2):
        if s != shift:
            op.set_shift(s)
        y = op(x)
        want = op.reference(x)
        same = torch.equal(y, want)
        diff = "" if same else f" ({int((y != want).sum())} of {y.numel()} differ)"
        verdict = "PASS" if same else "FAIL"
        print(f"Ci={ci} Co={co} H={h} W={w} relu={relu} shift={s}: {verdict}{diff}")
        ok &= same
    return ok


def median_ms(fn, iters):
    fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return sorted(ts)[len(ts) // 2] * 1e3


def bench(ci, co, h, w, iters):
    rng = np.random.default_rng(0)
    x = torch.as_tensor(rng.integers(-60, 60, (ci, h, w), dtype=np.int8))
    wt = torch.as_tensor(rng.integers(-40, 40, (co, ci, 3, 3), dtype=np.int8))
    op = conv3x3.Conv3x3(wt, height=h, width=w, shift=12)

    def launches():
        for _, _, prog, (xbo, _), (wbo, _), (ybo, _) in op.slabs:
            prog.run(xbo, wbo, ybo)

    npu = median_ms(launches, iters)
    call = median_ms(lambda: op(x), iters)
    gop = 2 * ci * co * h * w * 9 / 1e9
    print(
        f"Ci={ci} Co={co} {h}x{w}: NPU {npu:.3f} ms = {gop / npu:.1f} TOPS; "
        f"per call with host layouts {call:.3f} ms = {gop / call:.1f} TOPS"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--iters", type=int, default=10)
    a = ap.parse_args()
    ok = all(
        [
            check(256, 256, 64, 64),
            check(128, 256, 32, 128),
            check(64, 512, 16, 64),
            check(256, 64, 8, 64),
            check(256, 256, 24, 64, relu=True),
        ]
    )
    if a.bench:
        bench(256, 256, 1536, 64, a.iters)
        bench(256, 256, 256, 256, a.iters)
    raise SystemExit(0 if ok else 1)
