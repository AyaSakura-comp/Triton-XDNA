# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""NPU-resident tensors for the ordinary ``kernel[grid](...)`` launch.

Under XRT a launch stages every pointer argument: the tensor is copied into a
buffer object, the BO synced, and after the run the output is synced and
copied back. For a model that is most of a launch -- the weights are copied in
on every call -- and none of it is the NPU's work.

A tensor from this module is an ordinary CPU tensor whose pages are an XRT BO.
Passed to a kernel, the launcher recognises it and hands the BO to the NPU as
it is: nothing is copied, and only the bytes the argument covers are synced, so
a view such as ``x[:rows]`` of a large buffer syncs ``rows`` rows. A *static*
tensor is not synced at all after it is created -- it is for weights::

    from triton.backends.amd_triton_npu import resident

    w = resident.to_npu(weight_int8, static=True)  # written once
    x = resident.empty((4096, K), dtype=torch.int8)
    y = resident.empty((4096, N), dtype=torch.int32)
    x[:rows].copy_(activations)
    matmul[grid](x, w, y, ...)                       # no copies
    result = y[:rows]                                # the BO's pages, synced

Everything else about them is torch: they can be read, written, viewed and
passed to any CPU op. Two rules follow from the device reading the pages
directly:

* Write a static tensor from the host only through ``update`` (or call
  ``flush`` afterwards), since launches do not sync it.
* A tensor's pages go back to the NPU driver when the last view of it dies.

Only the XRT ELF launcher (NPU2) knows about them. Elsewhere -- xclbin, HSA,
Windows -- they are still valid CPU tensors and are staged like any other.
"""

from __future__ import annotations

import ctypes
import os
import threading
import weakref

import numpy as np
import torch

__all__ = [
    "empty",
    "zeros",
    "empty_like",
    "to_npu",
    "is_resident",
    "is_static",
    "set_static",
    "update",
    "flush",
    "usage",
]

_lib = None
_lib_lock = threading.Lock()
# Base address -> static flag, mirrored from the registry so is_static() needs
# no call into the library.
_static = {}


def _library() -> ctypes.CDLL:
    global _lib
    with _lib_lock:
        if _lib is None:
            from .driver import xrt_resident_lib_dir, _XRT_RESIDENT_LIB

            lib_dir = xrt_resident_lib_dir()
            if lib_dir is None:
                raise RuntimeError("NPU-resident tensors need the XRT ELF runtime (Linux)")
            lib = ctypes.CDLL(os.path.join(lib_dir, _XRT_RESIDENT_LIB))
            lib.tnpu_alloc.restype = ctypes.c_void_p
            lib.tnpu_alloc.argtypes = [ctypes.c_size_t, ctypes.c_int]
            lib.tnpu_free.argtypes = [ctypes.c_void_p]
            lib.tnpu_set_static.argtypes = [ctypes.c_void_p, ctypes.c_int]
            lib.tnpu_flush.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            lib.tnpu_lookup.restype = ctypes.c_size_t
            lib.tnpu_lookup.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
            lib.tnpu_usage.argtypes = [ctypes.POINTER(ctypes.c_size_t)] * 2
            _lib = lib
    return _lib


def _free(ptr: int) -> None:
    _static.pop(ptr, None)
    _lib.tnpu_free(ptr)


def _shape(size) -> tuple[int, ...]:
    if len(size) == 1 and isinstance(size[0], (tuple, list, torch.Size)):
        return tuple(size[0])
    return tuple(size)


def empty(*size, dtype: torch.dtype = torch.float32, static: bool = False) -> torch.Tensor:
    """A zero-filled CPU tensor whose pages are an NPU buffer object.

    (Zero-filled because the driver hands out cleared pages; ``empty`` is the
    name torch code reaches for.)
    """
    shape = _shape(size)
    nbytes = int(np.prod(shape, dtype=np.int64)) * torch.empty((), dtype=dtype).element_size()
    lib = _library()
    ptr = lib.tnpu_alloc(nbytes, int(static))
    if not ptr:
        raise RuntimeError(f"could not allocate {nbytes} bytes of NPU-resident memory")
    _static[ptr] = static
    # numpy rather than ctypes as the owner, because the release has to hang
    # off an object that lives exactly as long as the tensor's storage does:
    # from_numpy keeps the array alive, and arrays take weak references.
    raw = np.ctypeslib.as_array(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint8)), (max(nbytes, 1),))
    weakref.finalize(raw, _free, ptr)
    return torch.from_numpy(raw)[:nbytes].view(dtype).view(shape)


def zeros(*size, dtype: torch.dtype = torch.float32, static: bool = False) -> torch.Tensor:
    return empty(*size, dtype=dtype, static=static)


def empty_like(t: torch.Tensor, static: bool = False) -> torch.Tensor:
    return empty(tuple(t.shape), dtype=t.dtype, static=static)


def to_npu(t: torch.Tensor, static: bool = False) -> torch.Tensor:
    """A resident copy of ``t`` (contiguous). ``static`` for weights."""
    out = empty(tuple(t.shape), dtype=t.dtype)
    out.copy_(t)
    if static:
        set_static(out, True)
    return out


def _base(t: torch.Tensor) -> int:
    base = ctypes.c_void_p()
    if not _library().tnpu_lookup(t.data_ptr(), ctypes.byref(base)):
        return 0
    return base.value


def is_resident(t: torch.Tensor) -> bool:
    """Whether ``t``'s data lies in NPU-resident memory."""
    return t.device.type == "cpu" and _lib is not None and _base(t) != 0


def is_static(t: torch.Tensor) -> bool:
    return bool(_static.get(_base(t))) if is_resident(t) else False


def set_static(t: torch.Tensor, static: bool = True) -> None:
    """Mark the allocation holding ``t`` static (syncing it now) or not."""
    if _library().tnpu_set_static(t.data_ptr(), int(static)) != 0:
        raise ValueError("not an NPU-resident tensor")
    _static[_base(t)] = static


def flush(t: torch.Tensor) -> None:
    """Push host writes to ``t`` (contiguous) to the device."""
    nbytes = t.numel() * t.element_size()
    if _library().tnpu_flush(t.data_ptr(), nbytes) != 0:
        raise ValueError("not an NPU-resident tensor")


def update(t: torch.Tensor, value: torch.Tensor) -> None:
    """``t.copy_(value)``, then ``flush(t)`` -- the way to write a static tensor."""
    t.copy_(value)
    flush(t)


def usage() -> dict:
    """Bytes and allocations currently resident."""
    nbytes, count = ctypes.c_size_t(), ctypes.c_size_t()
    _library().tnpu_usage(ctypes.byref(nbytes), ctypes.byref(count))
    return {"bytes": nbytes.value, "allocations": count.value}
