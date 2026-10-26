# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""W8A8 GEMM (int8 x int8 -> int32) for LLM linears, with a K loop and resident weights.

The kernel is Triton's tutorial GEMM: each program loops over K in BLOCK_K
steps, so K only has to be a multiple of BLOCK_K (Gemma's 1536 / 6144 / 12288
are not powers of two). The schedule is generated for that loop (k_loop,
l2_k=BLOCK_K) and spreads one BLOCK_M x BLOCK_N output block over the whole
array:

  --block-m 512: 512x512 block, 64x128 per core (herd 8x4, 32 cores)
  --block-m 256: 256x512 block, 64x128 per core (herd 4x4)
  --block-m 128: 128x512 block, 32x128 per core (herd 4x4)

The stock 256x256 / 64x64 schedule reaches ~4 TOPS on these shapes; the 512x512
one 3.4-9.4 TOPS (9+ on Gemma 4 E2B's gate_up / down). Smaller row blocks waste less padding when a launch has few
rows (a diffusion TTS step, a short prompt) but every launch also pays a fixed
~0.3 ms: the runtime sequence starts with aiex.configure, i.e. the array is
reconfigured per dispatch (XRT's own no-op latency is ~50 us).

Weights are static resident tensors (synced once); activations and results
are resident too, so a launch copies nothing. Results are checked bit-exact
against torch on the CPU.

One schedule per process: AIR_TRANSFORM_TILING_SCRIPT is read when a kernel
is compiled.

usage: python matmul_i8_k_loop.py [--block-m 512|256|128] [--rows M]
Exit status: 0 pass, 1 failure.
"""

import argparse
import os
import sys
import time

import torch
import triton
import triton.language as tl

sys.path.append(os.path.abspath(".."))
import benchmark

HERE = os.path.dirname(os.path.abspath(__file__))
BLOCK_N = 512
BLOCK_K = 128
L1 = {512: (64, 128), 256: (64, 128), 128: (32, 128)}  # block_m -> per-core (l1_m, l1_n)

# (N, K) of the linears this was tuned for; q/k/v and gate/up fused into one GEMM each.
SHAPES = {
    "gemma4-e2b qkv": (4096, 1536),
    "gemma4-e2b gate_up": (24576, 1536),
    "gemma4-e2b down": (1536, 12288),
    "qwen3-0.6b qkv": (4096, 1024),
    "qwen3-0.6b o": (1024, 2048),
    "qwen3-0.6b gate_up": (6144, 1024),
    "qwen3-0.6b down": (1024, 3072),
}


@triton.jit
def matmul_i8_k(
    A,
    B,
    C,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """C[M, N] = A[M, K] @ B[K, N]; M is the grid's, everything row-major."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = B + offs_k[:, None] * N + offs_n[None, :]
    tl.static_assert(K % BLOCK_K == 0)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for _ in range(0, K // BLOCK_K):
        acc = tl.dot(tl.load(a_ptrs), tl.load(b_ptrs), acc)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * N
    tl.store(C + offs_m[:, None] * N + offs_n[None, :], acc)


def schedule(block_m):
    """The transform script for this row block, generated next to this file."""
    from triton.backends.amd_triton_npu.matmul_transform import (
        generate_matmul_transform,
    )

    l1_m, l1_n = L1[block_m]
    path = os.path.join(HERE, f"transform_i8_k{BLOCK_K}_{l1_m}x{l1_n}.mlir")
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(
                generate_matmul_transform(
                    l1_m=l1_m,
                    l1_n=l1_n,
                    l2_k=BLOCK_K,
                    pack_sizes=(8, 8, 8),
                    accum_type="i32",
                    k_loop=True,
                )
            )
    return path


def test_shape(name, M, N, K, block_m, resident, reps=20):
    a = torch.randint(-128, 128, (M, K), dtype=torch.int8)
    w = torch.randint(-128, 128, (K, N), dtype=torch.int8)
    ar, wr = resident.to_npu(a), resident.to_npu(w, static=True)
    cr = resident.empty((M, N), dtype=torch.int32)
    grid = (M // block_m, N // BLOCK_N)

    def launch():
        matmul_i8_k[grid](
            ar, wr, cr, N, K, BLOCK_M=block_m, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
        )

    launch()
    ok = torch.equal(cr.cpu(), a.int() @ w.int())
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        launch()
        ts.append(time.perf_counter() - t)
    ms = sorted(ts)[reps // 2] * 1e3
    tops = 2 * M * N * K / ms / 1e9
    print(
        f"{'PASS' if ok else 'FAIL'}: {name:20s} M={M} N={N} K={K}: {ms:.3f} ms, {tops:.2f} TOPS"
    )
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block-m", type=int, default=512, choices=sorted(L1))
    ap.add_argument("--rows", type=int, default=None, help="M (default: one row block)")
    args = ap.parse_args()
    M = args.rows or args.block_m
    assert M % args.block_m == 0, "M must be a multiple of --block-m"

    os.environ["AIR_TRANSFORM_TILING_SCRIPT"] = schedule(args.block_m)
    benchmark.select_npu_backend()
    from triton.backends.amd_triton_npu import resident

    torch.manual_seed(0)
    ok = all(
        test_shape(name, M, N, K, args.block_m, resident)
        for name, (N, K) in SHAPES.items()
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
