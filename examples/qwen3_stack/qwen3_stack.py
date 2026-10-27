# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""tl.extra.npu.qwen3_stack: a Qwen3-0.6B-sized stack, its final norm and a head in one launch.

Makes a stack of random weights (Qwen3-0.6B's widths, a few layers, a head that is not a
multiple of 64 wide), runs pairs of sequences whose lengths take every build (1 to 6 row
groups of 64 positions) and checks
  * each sequence against the same function in fp32 on the CPU (`qwen3_stack.reference`):
    every GEMM's operands are bfp16, ~1.5% per stage, ~5% at two random layers' logits
    (OmniVoice's 28 trained layers: 1.3-2.3%), so the bound is 8% -- what it catches is
    a wrong layout, mask or rotation, which costs far more;
  * that the sequences do not see each other: sequence 1's output stays the same, bit for
    bit, when sequence 0 changes.
With --bench, times a launch of each build.

usage: python qwen3_stack.py [--layers 2] [--out 1000] [--bench] [--iters N]
Exit status: 0 pass, 1 failure.
"""

import argparse
import time

import torch

from triton.language.extra.npu import qwen3_stack as q3

# (n0, n1): every row-group count, uneven pairs, the 1-position and 384-position edges
PAIRS = [
    (1, 1),
    (58, 43),
    (64, 64),
    (65, 30),
    (113, 92),
    (150, 190),
    (200, 129),
    (264, 222),
    (384, 300),
]
TOL = 0.08


def random_stack(layers, out, seed=0):
    g = torch.Generator().manual_seed(seed)

    def r(*shape, std=0.02):
        return torch.randn(*shape, generator=g) * std

    def layer():
        # k-norm gains with a few large channels, as trained Qwen3 models have (OmniVoice's
        # reach 35): attention as peaked as a trained model's, not flat
        k_norm = 1 + r(128, std=0.1)
        k_norm[torch.randperm(128, generator=g)[:4]] *= 8
        return dict(
            q=r(2048, 1024),
            k=r(1024, 1024),
            v=r(1024, 1024),
            o=r(1024, 2048),
            gate=r(3072, 1024),
            up=r(3072, 1024),
            down=r(1024, 3072),
            input_norm=1 + r(1024, std=0.1),
            post_norm=1 + r(1024, std=0.1),
            q_norm=1 + r(128, std=0.1),
            k_norm=k_norm,
        )

    inv_freq = 1.0 / (1e6 ** (torch.arange(0, 128, 2).float() / 128))
    return (
        [layer() for _ in range(layers)],
        1 + r(1024, std=0.1),
        r(out, 1024),
        inv_freq,
    )


def rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--out", type=int, default=1000)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--iters", type=int, default=20)
    a = ap.parse_args()
    layers, norm, head, inv_freq = random_stack(a.layers, a.out)
    t = time.time()
    op = q3.Qwen3Stack(layers, norm=norm, head=head, inv_freq=inv_freq)
    print(
        f"{a.layers} layers, head {a.out}: weights packed in {time.time() - t:.1f} s",
        flush=True,
    )
    g = torch.Generator().manual_seed(1)
    ok = True
    for n0, n1 in PAIRS:
        x0, x1 = torch.randn(n0, 1024, generator=g), torch.randn(n1, 1024, generator=g)
        t = time.time()
        y0, y1 = op(x0, x1)
        dt = time.time() - t
        e0 = rel(y0, q3.reference(layers, norm, head, inv_freq, x0))
        e1 = rel(y1, q3.reference(layers, norm, head, inv_freq, x1))
        y1b = y1.clone()
        apart = torch.equal(op(torch.randn(n0, 1024, generator=g), x1)[1], y1b)
        good = (
            max(e0, e1) < TOL
            and apart
            and tuple(y0.shape) == (n0, a.out)
            and tuple(y1.shape) == (n1, a.out)
        )
        ok &= good
        print(
            f"{'PASS' if good else 'FAIL'} {n0:3d} + {n1:3d} positions ({op.groups_for(n0, n1)} row groups): "
            f"rel err vs fp32 {e0:.4f} / {e1:.4f}, sequences apart {apart} ({dt:.2f} s incl. a first build)",
            flush=True,
        )
    if a.bench:
        for groups in range(1, q3.MAX_GROUPS + 1):
            n = 64 * groups
            x = torch.randn(n, 1024)
            op(x, x)
            ts = []
            for _ in range(a.iters):
                t = time.perf_counter()
                op(x, x)
                ts.append(time.perf_counter() - t)
            ts.sort()
            print(
                f"{groups} row groups ({n} + {n} positions): {1e3 * ts[len(ts) // 2]:.2f} ms per call"
            )
    print("PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
