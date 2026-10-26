# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""``tl.extra.npu.conv3x3`` -- 3x3 stride-1 int8 convolution on the whole NPU.

A convolution's reduction runs over (ky, kx, input channel). Here it is split across
the array:

* every core keeps the weights of its slice in local memory and reduces a block of 16
  output pixels x 16 output channels over 3 ky x Ci/2 channels x 3 kx without its
  accumulators leaving the registers (``kernels/conv3x3_cascade.cc``, Peano
  intrinsics);
* the two halves of the input channels sit on two vertically adjacent cores, and the
  head's partial sums reach the tail over the AIE cascade, where they seed the tail's
  accumulators;
* the tail shift-round-saturates to int8 (or to uint8: a fused ReLU) and writes the
  output line; the shift travels with the weights, so one build of a geometry serves
  every layer of it.

Each column runs two such chains, one per 32-pixel strip of a 64-pixel band, for the
column's 32 output channels; eight columns give 256 output channels. Input lines slide
through a four-line window in each core; memtile ``2h+s`` broadcasts half ``h`` of
strip ``s`` to the eight cores that need it, memtile ``c`` holds column ``c``'s weights
and gathers its output rows.

This is not a ``@triton.jit`` kernel: the dataflow -- weights resident across the
image, partial sums passed core to core -- has no tile semantics in the language. It
is an op in the sense of ``fused_decode``: real on this device, built from an AIE
design plus a kernel object, cached.

Measured on Strix Halo (AIE2P at 1.86 GHz, 61 TOPS int8 peak), Ci = Co = 256, 64 wide:
1536 rows in 2.21 ms of NPU time = 52.6 TOPS (86%); each core's steady state is 90%
VMAC.

    op = Conv3x3(weight_int8, height=H, width=W, shift=12)  # weight [Co, Ci, 3, 3]
    y = op(x_int8)                       # x [Ci, H, W] -> [Co, H, W] int8
    op.set_shift(10)                     # no rebuild

Limits: Ci a multiple of 16 up to 256, Co a multiple of 32 (more than 256 runs in slabs
of 256), W a multiple of 64 (bands stacked vertically in one launch), padding 1, no
bias (add it after).
"""

import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
KERNEL_SOURCE = HERE / "kernels" / "conv3x3_cascade.cc"
DESIGN_SOURCE = HERE / "kernels" / "conv3x3_design.py"
HOST_SOURCE = HERE / "kernels" / "conv3x3_host.c"

NC = 8  # columns
OWS = 32  # strip output width
WPS = 40  # strip input line width: 32 + halo, a multiple of 8
BAND = 2 * OWS  # pixels per band (two strips)
OGP = 2  # 32 output channels per chain = 2 pairs of 16
MAX_CG = 16  # channel groups per core that fit local memory (weights + 4 lines)
WTRAILER = 64  # bytes after each core's weights; the first int32 is the shift

_BUILD_LOCK = threading.Lock()


class Conv3x3ConfigError(ValueError):
    pass


# ------------------------------------------------------------------ build (cached)
def _cache_dir():
    root = os.environ.get("AMD_TRITON_NPU_CONV_CACHE") or os.path.join(
        os.path.expanduser("~"), ".cache", "amd_triton_npu", "conv3x3"
    )
    os.makedirs(root, exist_ok=True)
    return root


def _peano():
    from triton.backends.amd_triton_npu.driver import find_peano_root

    root = find_peano_root()
    if not root:
        raise RuntimeError("no Peano (llvm-aie) install found")
    return root


def _toolchain():
    vers = []
    for dist in ("llvm-aie", "mlir_aie_no_rtti", "mlir_aie"):
        try:
            vers.append(f"{dist}={importlib.metadata.version(dist)}")
        except importlib.metadata.PackageNotFoundError:
            pass
    return ";".join(vers)


def _key(*parts):
    h = hashlib.sha256()
    for p in parts:
        h.update(p if isinstance(p, bytes) else str(p).encode())
    return h.hexdigest()[:16]


def kernel_object(cg: int) -> str:
    """Compile the cascade kernel for ``cg`` channel groups per core (cached)."""
    defs = [
        f"-DCONV_CG={cg}",
        f"-DCONV_OGP={OGP}",
        f"-DCONV_OW={OWS}",
        f"-DCONV_WP={WPS}",
    ]
    key = _key(KERNEL_SOURCE.read_bytes(), " ".join(defs), _toolchain())
    obj = os.path.join(_cache_dir(), f"conv3x3_cascade_cg{cg}_{key}.o")
    with _BUILD_LOCK:
        if not os.path.exists(obj):
            tmp = f"{obj}.{os.getpid()}.tmp"
            cmd = [
                os.path.join(_peano(), "bin", "clang++"),
                "-O2",
                "-std=c++20",
                "--target=aie2p-none-unknown-elf",
                "-DNDEBUG",
                "-Wno-deprecated-declarations",
                *defs,
                "-c",
                str(KERNEL_SOURCE),
                "-o",
                tmp,
            ]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode:
                raise RuntimeError(f"conv3x3 kernel build failed:\n{r.stderr[-4000:]}")
            os.replace(tmp, obj)
    return obj


def design_artifacts(rows, cg, ncol, relu):
    """xclbin + instruction stream for one geometry, built in a subprocess, cached."""
    obj = kernel_object(cg)
    params = dict(ROWS=rows, CG=cg, NCOL=ncol, relu=bool(relu), obj=obj)
    key = _key(
        DESIGN_SOURCE.read_bytes(), json.dumps(params, sort_keys=True), _toolchain()
    )
    out = os.path.join(_cache_dir(), f"design_{key}")
    xclbin = os.path.join(out, "final.xclbin")
    insts = os.path.join(out, "insts.bin")
    with _BUILD_LOCK:
        if not (os.path.exists(xclbin) and os.path.exists(insts)):
            tmp = f"{out}.{os.getpid()}.tmp"
            shutil.rmtree(tmp, ignore_errors=True)
            r = subprocess.run(
                [sys.executable, str(DESIGN_SOURCE), json.dumps(params), tmp],
                capture_output=True,
                text=True,
                env=dict(os.environ, PEANO_INSTALL_DIR=_peano()),
                cwd=os.path.dirname(obj),
            )
            if r.returncode:
                log = (r.stdout + r.stderr)[-6000:]
                raise RuntimeError(f"conv3x3 design build failed:\n{log}")
            shutil.rmtree(out, ignore_errors=True)
            os.replace(tmp, out)
    return xclbin, insts


_HOST = {}


def host_lib():
    """The threaded layout routines (``kernels/conv3x3_host.c``), built with the
    system C compiler; None without one or with AMD_TRITON_NPU_CONV_HOST=torch, and
    the layouts are torch / numpy permutes instead."""
    with _BUILD_LOCK:
        if "lib" in _HOST:
            return _HOST["lib"]
        lib = None
        try:
            import ctypes

            if os.environ.get("AMD_TRITON_NPU_CONV_HOST") == "torch":
                raise RuntimeError("AMD_TRITON_NPU_CONV_HOST=torch")
            cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc")
            if not cc:
                raise RuntimeError("no C compiler")
            flags = ["-O3", "-march=native", "-fopenmp", "-shared", "-fPIC"]
            key = _key(HOST_SOURCE.read_bytes(), cc, " ".join(flags))
            so = os.path.join(_cache_dir(), f"conv3x3_host_{key}.so")
            if not os.path.exists(so):
                tmp = f"{so}.{os.getpid()}.tmp"
                cmd = [cc, *flags, str(HOST_SOURCE), "-o", tmp]
                r = subprocess.run(cmd, capture_output=True, text=True)
                if r.returncode:
                    raise RuntimeError(f"host build failed:\n{r.stderr[-2000:]}")
                os.replace(tmp, so)
            lib = ctypes.CDLL(so)
            # (src, channels, h, w, dst, threads)
            args = [ctypes.c_void_p] + [ctypes.c_int] * 3 + [ctypes.c_void_p]
            for f in (lib.conv3x3_pack, lib.conv3x3_unpack):
                f.restype = None
                f.argtypes = args + [ctypes.c_int]
        except Exception as e:
            _HOST["error"] = f"{type(e).__name__}: {e}"
        _HOST["lib"] = lib
        return lib


def _threads():
    """torch's thread count (the physical cores): the layout loops are memory bound."""
    torch = _torch()
    if torch is not None:
        return torch.get_num_threads()
    return max(1, (os.cpu_count() or 2) // 2)


def _ptr(a):
    return a.data_ptr() if hasattr(a, "data_ptr") else a.ctypes.data


def _is_contiguous(a):
    return a.is_contiguous() if hasattr(a, "is_contiguous") else a.flags.c_contiguous


# ------------------------------------------------------------------ run (pyxrt)
_DEVICE = {}
_PROGRAMS = {}  # xclbin path -> _Program: ops of one geometry share a hw context


def _device():
    import pyxrt

    if "dev" not in _DEVICE:
        _DEVICE["dev"] = pyxrt.device(0)
    return _DEVICE["dev"]


class _Program:
    """A loaded xclbin with its instruction BO and hardware context, kept for reuse."""

    def __init__(self, xclbin_path, insts_path):
        import pyxrt

        dev = _device()
        self.xclbin = pyxrt.xclbin(xclbin_path)
        dev.register_xclbin(self.xclbin)
        self.ctx = pyxrt.hw_context(dev, self.xclbin.get_uuid())
        name = self.xclbin.get_kernels()[0].get_name()
        self.kernel = pyxrt.kernel(self.ctx, name)
        insts = np.fromfile(insts_path, dtype=np.uint32)
        self.insts_bo = pyxrt.bo(
            dev, insts.nbytes, pyxrt.bo.cacheable, self.kernel.group_id(1)
        )
        self.insts_bo.write(insts.view(np.uint8), 0)
        self.insts_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        self.insts_bytes = insts.nbytes

    def bo(self, nbytes, arg):
        """A host buffer for runtime argument ``arg`` (0 input, 1 weights, 2 output)."""
        import pyxrt

        group = self.kernel.group_id(3 + arg)
        b = pyxrt.bo(_device(), nbytes, pyxrt.bo.host_only, group)
        return b, np.frombuffer(b.map(), dtype=np.int8, count=nbytes)

    def run(self, *bos):
        import pyxrt

        h = self.kernel(3, self.insts_bo, self.insts_bytes, *bos)
        state = h.wait()
        if state != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            raise RuntimeError(f"conv3x3 run ended in {state}")


def _program(xclbin, insts):
    with _BUILD_LOCK:
        if xclbin not in _PROGRAMS:
            _PROGRAMS[xclbin] = _Program(xclbin, insts)
        return _PROGRAMS[xclbin]


def _sync_dir(d):
    import pyxrt

    if d == "to":
        return pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
    return pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE


# ------------------------------------------------------------------ host layouts
def _torch():
    try:
        import torch

        return torch
    except ImportError:  # pragma: no cover - numpy fallback below
        return None


def pack_weights(weight, shift):
    """[Co_slab, Ci, 3, 3] int8 -> [NCOL][2 halves] of
    ([2 pairs][ky][CG][kx][2][8 ci][8 co], 64-byte trailer holding the shift)."""
    co, ci = weight.shape[:2]
    ncol, cg = co // 32, ci // 16
    w = np.asarray(weight, dtype=np.int8).reshape(ncol, OGP, 2, 8, 2, cg, 8, 3, 3)
    w = w.transpose(0, 4, 1, 7, 5, 8, 2, 6, 3)  # [c, h, p, ky, g, kx, j, ci8, co8]
    w = w.reshape(ncol, 2, -1)
    out = np.zeros((ncol, 2, w.shape[2] + WTRAILER), np.int8)
    out[:, :, : w.shape[2]] = w
    _set_shift(out, shift)
    return out.reshape(-1)


def _set_shift(packed, shift):
    """Write ``shift`` into every core's trailer of a packed [NCOL, 2, slice] array."""
    packed[:, :, -WTRAILER : -WTRAILER + 4] = np.array([shift], "<i4").view(np.int8)


def pack_input(x, out, pad=None):
    """x [Ci, H, W] int8 -> out [2 halves][2 strips][B*(H+2)][CG][40][8].

    Band b covers output columns [64b, 64b+64); strip s of it reads padded columns
    [32s, 32s+40) of the band (original column 64b + 32s - 1 + i), zero outside the
    image: the strips of all bands are the size-40, step-32 windows of the padded rows.
    Bands are stacked vertically, each with its own top and bottom zero row, so a window
    never mixes two bands; the two output rows that straddle a band boundary are
    dropped by `unpack_output`. ``pad`` ([Ci, H+2, W+8], zero border) may be kept by the
    caller of the torch path so only its interior is written per call.
    """
    ci, h, w = x.shape
    bands, cg = w // BAND, ci // 16
    lib = host_lib()
    if lib is not None:
        x = x.contiguous() if hasattr(x, "contiguous") else np.ascontiguousarray(x)
        lib.conv3x3_pack(_ptr(x), ci, h, w, _ptr(out), _threads())
        return out
    torch = _torch()
    if torch is not None:
        if pad is None:
            pad = torch.zeros((ci, h + 2, w + WPS - OWS), dtype=torch.int8)
        pad[:, 1 : h + 1, 1 : w + 1] = torch.as_tensor(x)
        win = pad.unfold(2, WPS, OWS)  # [ci, h+2, 2B, 40], a view
        win = win.reshape(2, cg, 8, h + 2, bands, 2, WPS)
        win = win.permute(0, 5, 4, 3, 1, 6, 2)  # [half, s, B, h+2, cg, 40, 8]
        torch.as_tensor(out).view(win.shape).copy_(win)
        return out
    pad = np.zeros((ci, h + 2, w + WPS - OWS), np.int8)
    pad[:, 1 : h + 1, 1 : w + 1] = x
    win = np.lib.stride_tricks.sliding_window_view(pad, WPS, axis=2)[:, :, ::OWS]
    win = win.reshape(2, cg, 8, h + 2, bands, 2, WPS).transpose(0, 5, 4, 3, 1, 6, 2)
    out.reshape(2, 2, bands, h + 2, cg, WPS, 8)[...] = win
    return out


def unpack_output(y, co, h, w, out):
    """[B*(H+2) rows][NCOL][2 strips][4 groups][32][8] -> out [Co_slab, H, W].

    ``y`` holds two rows more than the design writes, so it reshapes into whole bands;
    the last two rows of every band (the window straddling a band boundary, or past
    the end) are dropped. Bytes are copied as they are (int8, or uint8 with relu).
    """
    bands, ncol = w // BAND, co // 32
    lib = host_lib()
    if lib is not None and _is_contiguous(out):
        lib.conv3x3_unpack(_ptr(y), co, h, w, _ptr(out), _threads())
        return out
    torch = _torch()
    if torch is not None:
        yt = torch.as_tensor(y).view(bands, h + 2, ncol, 2, 4, OWS, 8)[:, :h]
        dst = torch.as_tensor(out).view(ncol, 4, 8, h, bands, 2, OWS)
        dst.copy_(yt.permute(2, 4, 6, 1, 0, 3, 5))
        return out
    yv = y.reshape(bands, h + 2, ncol, 2, 4, OWS, 8)[:, :h]
    out.reshape(ncol, 4, 8, h, bands, 2, OWS)[...] = yv.transpose(2, 4, 6, 1, 0, 3, 5)
    return out


# ------------------------------------------------------------------ the op
class Conv3x3:
    """A 3x3, stride-1, padding-1 int8 convolution with fixed weights, on the NPU.

    Args:
        weight: int8 [Co, Ci, 3, 3] (numpy or torch). Packed once into each core's
            order and kept in a device buffer.
        height, width: the input (= output) size the op is built for.
        shift: output = saturate(round_half_up(sum >> shift)) as int8 (uint8 with
            ``relu``); `set_shift` changes it without a rebuild.
        relu: fuse ReLU: the tail saturates to [0, 255] instead of [-128, 127].
    """

    def __init__(self, weight, *, height, width, shift, relu=False):
        w = np.asarray(weight.cpu() if hasattr(weight, "cpu") else weight)
        if w.dtype != np.int8 or w.ndim != 4 or w.shape[2:] != (3, 3):
            raise Conv3x3ConfigError(
                f"weight must be int8 [Co, Ci, 3, 3], got {w.dtype} {w.shape}"
            )
        co, ci = w.shape[:2]
        if ci % 16 or ci // 16 > MAX_CG:
            raise Conv3x3ConfigError(
                f"Ci must be a multiple of 16 up to {16 * MAX_CG}, got {ci}"
            )
        if co % 32:
            raise Conv3x3ConfigError(f"Co must be a multiple of 32, got {co}")
        if width % BAND or height < 1:
            raise Conv3x3ConfigError(
                f"width must be a multiple of {BAND}, got {width}x{height}"
            )
        if not 1 <= shift <= 31:
            raise Conv3x3ConfigError(f"shift must be in [1, 31], got {shift}")
        self.ci, self.co, self.h, self.w = ci, co, height, width
        self.shift, self.relu = shift, relu
        self.weight = w
        self.cg = ci // 16
        self.bands = width // BAND
        self.rows = self.bands * (height + 2) - 2
        x_bytes = 2 * 2 * (self.rows + 2) * self.cg * WPS * 8
        torch = _torch()
        self._pad = None
        if torch is not None:
            pad = (ci, height + 2, width + WPS - OWS)
            self._pad = torch.zeros(pad, dtype=torch.int8)
        self._ystage = {}
        self.slabs = []  # (first channel, channels, program, x, w, y buffers)
        for c0 in range(0, co, NC * 32):
            ws = w[c0 : c0 + NC * 32]
            n = ws.shape[0]
            prog = _program(*design_artifacts(self.rows, self.cg, n // 32, relu))
            packed = pack_weights(ws, shift)
            wbo, wview = prog.bo(packed.nbytes, 1)
            wview[:] = packed
            wbo.sync(_sync_dir("to"))
            xbo, xview = prog.bo(x_bytes, 0)
            # [rows][n channels][64 px] int8, two spare rows: it reshapes into bands
            ybo, yview = prog.bo((self.rows + 2) * n * BAND, 2)
            self.slabs.append((c0, n, prog, (xbo, xview), (wbo, wview), (ybo, yview)))

    def set_shift(self, shift):
        """Requantize by ``shift`` from the next call on (rewrites the trailers)."""
        if not 1 <= shift <= 31:
            raise Conv3x3ConfigError(f"shift must be in [1, 31], got {shift}")
        for _, n, _, _, (wbo, wview), _ in self.slabs:
            _set_shift(wview.reshape(n // 32, 2, -1), shift)
            wbo.sync(_sync_dir("to"))
        self.shift = shift

    def __call__(self, x, out=None):
        """x int8 [Ci, H, W] (or [1, Ci, H, W]) -> [Co, H, W] int8, uint8 with relu.

        Returns a torch tensor when torch is installed, else numpy; ``out`` may be a
        contiguous [Co, H, W] buffer of that type.
        """
        torch = _torch()
        x = x[0] if x.ndim == 4 else x
        dtype = str(x.dtype).split(".")[-1]
        if tuple(x.shape) != (self.ci, self.h, self.w) or dtype != "int8":
            raise Conv3x3ConfigError(
                f"built for int8 {(self.ci, self.h, self.w)}, "
                f"got {dtype} {tuple(x.shape)}"
            )
        x = x.cpu() if hasattr(x, "cpu") else x
        odt = np.uint8 if self.relu else np.int8
        if out is None and torch is not None:
            tdt = torch.uint8 if self.relu else torch.int8
            out = torch.empty((self.co, self.h, self.w), dtype=tdt)
        elif out is None:
            out = np.empty((self.co, self.h, self.w), odt)
        packed_for = None
        for c0, n, prog, (xbo, xview), (wbo, _), (ybo, yview) in self.slabs:
            if packed_for is None:
                pack_input(x, xview, self._pad)
                packed_for = xview
            elif xview is not packed_for:
                xview[:] = packed_for
            xbo.sync(_sync_dir("to"))
            prog.run(xbo, wbo, ybo)
            ybo.sync(_sync_dir("from"))
            dst = out[c0 : c0 + n]
            if host_lib() is not None:  # reads the buffer's mapping directly
                unpack_output(yview.view(odt), n, self.h, self.w, dst)
                continue
            # torch / numpy: one bulk copy out of the buffer's mapping, then the
            # strided reshuffle in ordinary memory (gathering from the mapping is
            # slower)
            host = self._ystage.setdefault(n, np.empty_like(yview))
            if torch is not None:
                torch.as_tensor(host).copy_(torch.as_tensor(yview))
            else:
                np.copyto(host, yview)
            unpack_output(host.view(odt), n, self.h, self.w, dst)
        return out

    def reference(self, x):
        """The same integers on the CPU: int32 sums, round half up, saturate."""
        return reference(x, self.weight, self.shift, self.relu)


def reference(x, weight, shift, relu=False):
    torch = _torch()
    if torch is None:
        raise RuntimeError("reference needs torch")
    x = x[0] if x.ndim == 4 else x
    x = np.asarray(x.cpu() if hasattr(x, "cpu") else x)
    xt = torch.as_tensor(x, dtype=torch.float64)[None]
    wt = torch.as_tensor(np.asarray(weight), dtype=torch.float64)
    # float64 sums of int8 products are exact (far below 2^53)
    acc = torch.nn.functional.conv2d(xt, wt, padding=1)[0].to(torch.int64)
    q = torch.div(acc + (1 << (shift - 1)), 1 << shift, rounding_mode="floor")
    if relu:
        return q.clamp(0, 255).to(torch.uint8)
    return q.clamp(-128, 127).to(torch.int8)
