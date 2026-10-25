# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# air statically links LLVM; a ROCm PyTorch install pulls in its own
# libLLVM.so.23. When a script imports torch before triton (as every example
# does), ROCm's LLVM wins the global namespace and air's constructor binds the
# wrong globals and segfaults (issue #102). Load air's bindings with
# RTLD_DEEPBIND so air resolves its own LLVM regardless of torch.
#
# DEEPBIND also binds air's malloc/free to glibc ahead of an allocator
# preloaded with LD_PRELOAD (tcmalloc, as vLLM's CPU backend uses), so memory
# one allocates the other frees and the import segfaults. With such an
# allocator preloaded, load air normally. TRITON_XDNA_DEEPBIND=0/1 overrides.
try:
    import os
    import sys

    _deepbind = os.environ.get("TRITON_XDNA_DEEPBIND")
    if _deepbind is None:
        _preload = os.environ.get("LD_PRELOAD", "")
        _deepbind = "0" if any(a in _preload for a in ("tcmalloc", "jemalloc", "mimalloc")) else "1"
    if _deepbind == "1":
        _prev_flags = sys.getdlopenflags()
        sys.setdlopenflags(os.RTLD_NOW | os.RTLD_GLOBAL | os.RTLD_DEEPBIND)
        try:
            import air._mlir_libs._mlir  # noqa: F401
        finally:
            sys.setdlopenflags(_prev_flags)
except Exception:
    pass
