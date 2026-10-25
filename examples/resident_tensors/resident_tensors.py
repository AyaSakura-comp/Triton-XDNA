# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""NPU-resident tensors through the ordinary ``kernel[grid](...)`` launch.

Checks, against torch on the CPU:

  * a launch on resident inputs and output (passed as their own BOs, no copy);
  * views that start inside an allocation (the launcher's sub-buffer path),
    and that the rest of the allocation is left alone;
  * a static weight, rewritten with ``resident.update``;
  * resident and ordinary tensors mixed in one launch;
  * a kernel whose output is not its last argument, on ordinary tensors (the
    launcher copies back what the kernel stores to, not the last pointer);
  * that repeated launches reuse one launcher instead of building one per call;

and prints what a launch costs with ordinary vs resident tensors.

Exit status: 0 pass, 1 failure, 77 skip (no XRT ELF runtime here).
"""

import sys
import time

import torch
import triton
import triton.language as tl

sys.path.append("..")
import benchmark

BLOCK = 1024


@triton.jit
def vecadd(A, B, C, BLOCK_SIZE_N: tl.constexpr):
    offs = tl.program_id(0) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    tl.store(C + offs, tl.load(A + offs) + tl.load(B + offs))


@triton.jit
def vecadd_out_first(C, A, B, BLOCK_SIZE_N: tl.constexpr):
    offs = tl.program_id(0) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    tl.store(C + offs, tl.load(A + offs) + tl.load(B + offs))


def add(a, b, c):
    vecadd[(a.numel() // BLOCK,)](a, b, c, BLOCK_SIZE_N=BLOCK)


def check(name, got, want):
    ok = torch.equal(got, want)
    print(f"{'PASS' if ok else 'FAIL'}: {name}")
    return ok


def main():
    benchmark.select_npu_backend()
    from triton.backends.amd_triton_npu import driver, resident

    if driver._get_output_format() != "elf" or driver.xrt_resident_lib_dir() is None:
        print("SKIP: resident tensors need the XRT ELF runtime (NPU2, Linux)")
        return 77

    torch.manual_seed(0)
    N = 64 * BLOCK
    dt = torch.bfloat16

    # Small integers, so every sum is exact in bf16 and results can be
    # compared bit for bit (the NPU rounds a bf16 add differently from torch).
    def data():
        return torch.randint(-64, 64, (N,)).to(dt)

    a, b = data(), data()
    ok = True

    ra, rb, rc = resident.to_npu(a), resident.to_npu(b), resident.empty(N, dtype=dt)
    add(ra, rb, rc)
    ok &= check("resident inputs and output", rc, a + b)

    # Views starting at an offset: sub-buffers of the allocations.
    lo, hi = 8 * BLOCK, 24 * BLOCK
    rc.zero_()
    add(ra[lo:hi], rb[lo:hi], rc[lo:hi])
    want = torch.zeros(N, dtype=dt)
    want[lo:hi] = a[lo:hi] + b[lo:hi]
    ok &= check("offset views, rest untouched", rc, want)

    # Static weight: written once, then rewritten through update().
    w = resident.to_npu(b, static=True)
    ok &= check("is_static", torch.tensor(resident.is_static(w)), torch.tensor(True))
    add(ra, w, rc)
    ok &= check("static weight", rc, a + b)
    b2 = data()
    resident.update(w, b2)
    add(ra, w, rc)
    ok &= check("static weight after update()", rc, a + b2)

    c = torch.empty(N, dtype=dt)
    add(ra, b, c)
    ok &= check("resident + ordinary inputs, ordinary output", c, a + b)

    c = torch.empty(N, dtype=dt)
    vecadd_out_first[(N // BLOCK,)](c, a, b, BLOCK_SIZE_N=BLOCK)
    ok &= check("output as the first argument", c, a + b)

    # One launcher per compiled kernel, however many launches.
    built = [0]
    init = driver.NPULauncher.__init__

    def counting_init(self, *args, **kwargs):
        built[0] += 1
        init(self, *args, **kwargs)

    driver.NPULauncher.__init__ = counting_init
    try:
        for _ in range(5):
            add(ra, rb, rc)
    finally:
        driver.NPULauncher.__init__ = init
    ok &= check("no launcher rebuilt per launch", torch.tensor(built[0]), torch.tensor(0))

    def per_call(x, y, z, n=50):
        add(x, y, z)
        t = time.perf_counter()
        for _ in range(n):
            add(x, y, z)
        return (time.perf_counter() - t) / n * 1e3

    plain = per_call(a, b, torch.empty(N, dtype=dt))
    res = per_call(ra, w, rc)
    print(f"per launch, {N} bf16: ordinary tensors {plain:.3f} ms, resident {res:.3f} ms")
    print(f"resident memory in use: {resident.usage()}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
