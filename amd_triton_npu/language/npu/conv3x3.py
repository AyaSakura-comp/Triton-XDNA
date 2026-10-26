# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""``tl.extra.npu.conv3x3`` -- 3x3 stride-1 int8 convolution on the whole NPU.

A convolution's reduction runs over (ky, kx, input channel). Here it is split across
the array:

* every core keeps the weights of its slice in local memory and reduces a block of 16
  output pixels x 16 output channels over 3 ky x its input channels x 3 kx without
  its accumulators leaving the registers (``kernels/conv3x3_cascade.cc``, Peano
  intrinsics);
* the input channels are split over a chain of L = 2 (Ci <= 256) or 4 (Ci <= 512)
  vertically adjacent cores; each core's partial sums seed the next one's
  accumulators over the AIE cascade;
* the tail shift-round-saturates to int8 (or to uint8: a fused ReLU) and writes the
  output line; the shift travels with the weights, so one build of a geometry serves
  every layer of it.

A chain computes 16*OGP output channels of a 32-pixel strip; the array runs R
partitions x 2 strips x G channel groups of chains (``kernels/conv3x3_design.py``),
chosen per layer so that small Co still fills the array. The host places the images
on those strips (`make_plan`): 32-column tiles, rows spread evenly over the
partitions, several narrow tiles side by side in one strip, any H and W, a batch in
one launch. Ci is padded to the chain's channels, Co to whole slabs of G*16*OGP
channels, which one launch runs one after the other.

This is not a ``@triton.jit`` kernel: the dataflow -- weights resident across the
image, partial sums passed core to core -- has no tile semantics in the language. It
is an op in the sense of ``fused_decode``: real on this device, built from an AIE
design plus a kernel object, cached.

Measured on Strix Halo (AIE2P at 1.86 GHz, 61 TOPS int8 peak), Ci = Co = 256, 64 wide:
1536 rows in 2.21 ms of NPU time = 52.6 TOPS (86%); each core's steady state is 90%
VMAC.

    op = Conv3x3(weight_int8, height=H, width=W, shift=12)  # weight [Co, Ci, 3, 3]
    y = op(x_int8)                       # x [N, Ci, H, W] -> [N, Co, H, W] int8
    op.set_shift(10)                     # no rebuild

Limits: Ci up to 512, stride 1, padding 1, no bias (add it after).
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

OWS = 32  # strip output width
WPS = 40  # strip input line width: 32 + halo, a multiple of 8
WTRAILER = 64  # bytes after each core's weights; the first int32 is the shift
L1_BUDGET = 65536 - 0x800  # a core's data memory minus its stack
# (G channel groups, R partitions) the design can route, per chain length
SPLITS = {2: ((8, 1), (4, 2), (2, 4)), 4: ((4, 1),)}

_BUILD_LOCK = threading.Lock()


class Conv3x3ConfigError(ValueError):
    pass


def geometry(cg, ogp):
    """Bytes of one input line, one weight slice and one chain's output line."""
    return cg * WPS * 8, ogp * 3 * cg * 3 * 2 * 64 + WTRAILER, 2 * ogp * OWS * 8


def configure(ci, co):
    """(L, CG, OGP, G, R) for a layer: the chain length from Ci, then the channel split
    with the least work per output (launches x rows per partition), fewest launches,
    least padding."""
    if ci > 512:
        raise Conv3x3ConfigError(f"Ci must be at most 512, got {ci}")
    chain = 2 if ci <= 256 else 4
    cg = -(-ci // (8 * chain))
    best = None
    for ogp in (1, 2, 4, 8):
        line, wsl, ochain = geometry(cg, ogp)
        if wsl + 4 * line + 4 * ochain > L1_BUDGET:
            continue
        for g, r in SPLITS[chain]:
            slab = g * 16 * ogp
            nslab = -(-co // slab)
            key = (nslab / r, nslab, slab)
            if best is None or key < best[0]:
                best = (key, (chain, cg, ogp, g, r))
    return best[1]


def make_plan(n, h, w, r_parts):
    """Place n images of h x w on 2 * r_parts strips of 32-pixel lines.

    Returns (pieces, rowmap, rows): pieces int32 [P, 9] as in ``conv3x3_host.c``,
    rowmap int32 [n*h, tiles, 4] (output line, partition, strip, pixel offset of every
    image row and 32-column tile), and the output lines per strip.
    """
    tiles = [(c0, min(OWS, w - c0)) for c0 in range(0, w, OWS)]
    wmax = max(tw for _, tw in tiles)
    k = max(1, (OWS + 2) // (wmax + 2))  # narrow tiles side by side in a line
    slots = [
        (r, s, j * (wmax + 2))
        for r in range(r_parts)
        for s in range(2)
        for j in range(k)
    ]
    per = -(-n * len(tiles) * h // len(slots))
    cursor, fill, si, pieces = [0] * len(slots), [0] * len(slots), 0, []
    for img in range(n):
        for c0, tw in tiles:
            row = 0
            while row < h:
                take = min(h - row, per - fill[si])
                r, s, o = slots[si]
                pieces.append((img, r, s, o, cursor[si], take, row, c0, tw))
                cursor[si] += take + 2  # the next piece's halo lines
                fill[si] += take
                row += take
                if fill[si] == per:
                    si += 1
    rowmap = np.zeros((n * h, len(tiles), 4), np.int32)
    for img, r, s, o, l0, nr, r0, c0, _ in pieces:
        rows = slice(img * h + r0, img * h + r0 + nr)
        rowmap[rows, c0 // OWS, 0] = l0 + np.arange(nr)
        rowmap[rows, c0 // OWS, 1:] = (r, s, o)
    return np.array(pieces, np.int32), rowmap, max(c for c in cursor) - 2


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


def kernel_object(cg: int, ogp: int) -> str:
    """Compile the cascade kernel for ``cg`` channel groups, ``ogp`` pairs (cached)."""
    defs = [
        f"-DCONV_CG={cg}",
        f"-DCONV_OGP={ogp}",
        f"-DCONV_OW={OWS}",
        f"-DCONV_WP={WPS}",
    ]
    key = _key(KERNEL_SOURCE.read_bytes(), " ".join(defs), _toolchain())
    obj = os.path.join(_cache_dir(), f"conv3x3_cascade_cg{cg}_ogp{ogp}_{key}.o")
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


def program_format():
    """ "elf" (default): a full ELF per geometry, which reloads its PDI on every run
    (~0.3 ms) but switches between programs without the ~2 ms hardware-context switch
    of an xclbin; "xclbin" (AMD_TRITON_NPU_CONV_FORMAT=xclbin): xclbin + instruction
    stream, cheaper back to back."""
    return os.environ.get("AMD_TRITON_NPU_CONV_FORMAT", "elf")


def design_artifacts(rows, cg, chain, ogp, g, r, nslab, relu):
    """The program for one geometry, built in a subprocess, cached: (format, path,
    kernel name or instruction stream path)."""
    obj = kernel_object(cg, ogp)
    fmt = program_format()
    params = dict(
        ROWS=rows,
        CG=cg,
        L=chain,
        OGP=ogp,
        G=g,
        R=r,
        NSLAB=nslab,
        relu=bool(relu),
        obj=obj,
    )
    if fmt == "elf":
        params["format"] = "elf"
    key = _key(
        DESIGN_SOURCE.read_bytes(), json.dumps(params, sort_keys=True), _toolchain()
    )
    out = os.path.join(_cache_dir(), f"design_{key}")
    if fmt == "elf":
        a, b = os.path.join(out, "conv.elf"), os.path.join(out, "design.mlir")
    else:
        a, b = os.path.join(out, "final.xclbin"), os.path.join(out, "insts.bin")
    with _BUILD_LOCK:
        if not (os.path.exists(a) and os.path.exists(b)):
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
    if fmt == "elf":
        with open(os.path.join(out, "kernel.txt")) as f:
            return fmt, a, f.read().strip()
    return fmt, a, b


_HOST = {}


def host_lib():
    """The threaded layout routines (``kernels/conv3x3_host.c``), built with the
    system C compiler; None without one or with AMD_TRITON_NPU_CONV_HOST=torch, and
    the layouts are done with torch instead."""
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
            p, i = ctypes.c_void_p, ctypes.c_int
            lib.conv3x3_pack.restype = lib.conv3x3_unpack.restype = None
            lib.conv3x3_pack.argtypes = [p, i, i, i, p, p, i, i, i, i, i, i]
            lib.conv3x3_unpack.argtypes = [p, i, i, i, i, i, p, p, i, i, i, i]
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


# ------------------------------------------------------------------ run (pyxrt)
_DEVICE = {}
_PROGRAMS = {}  # program path -> program: ops of one geometry share a hw context


def _device():
    import pyxrt

    if "dev" not in _DEVICE:
        _DEVICE["dev"] = pyxrt.device(0)
    return _DEVICE["dev"]


class _ElfProgram:
    """A loaded full ELF with its hardware context, kept for reuse."""

    def __init__(self, elf_path, kernel_name):
        import pyxrt

        self.elf = pyxrt.elf(elf_path)
        self.ctx = pyxrt.hw_context(_device(), self.elf)
        self.kernel = pyxrt.ext.kernel(self.ctx, kernel_name)

    def bo(self, nbytes, arg):
        """A host buffer for runtime argument ``arg`` (0 input, 1 weights, 2 output)."""
        import pyxrt

        b = pyxrt.bo(_device(), nbytes, pyxrt.bo.host_only, 0)
        return b, np.frombuffer(b.map(), dtype=np.int8, count=nbytes)

    def run(self, *bos):
        import pyxrt

        run = pyxrt.run(self.kernel)
        for i, b in enumerate(bos):
            run.set_arg(i, b)
        run.start()
        state = run.wait()
        if state != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            raise RuntimeError(f"conv3x3 run ended in {state}")


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


def _program(fmt, path, extra):
    with _BUILD_LOCK:
        if path not in _PROGRAMS:
            cls = _ElfProgram if fmt == "elf" else _Program
            _PROGRAMS[path] = cls(path, extra)
        return _PROGRAMS[path]


class _SharedProgram:
    """One kernel of a merged full ELF (`share_context`)."""

    def __init__(self, ctx, name):
        import pyxrt

        self.kernel = pyxrt.ext.kernel(ctx, name)

    bo = _ElfProgram.bo
    run = _ElfProgram.run


_SHARED = {}  # merged ELF path -> (elf, hw_context)


def share_context(ops):
    """Run ``ops`` as the kernels of one full ELF: one hardware context for all of them.

    Separately each design holds a context, and a launch after another design's pays a
    context switch (~2.3 ms measured for two of these designs). Merged, every launch
    rewrites its own configuration instead (~0.3-0.5 ms), whatever ran before: the way
    to run the layers of a model one after the other. Ops of one design share a kernel;
    with a single design there is nothing to merge (its context never switches). The
    ops keep their buffers. Returns the merged ELF's path, or None.
    """
    ops = list(ops)
    if any(op.design_mlir is None for op in ops):
        raise Conv3x3ConfigError("share_context needs ops built as full ELFs")
    mlirs = sorted({op.design_mlir for op in ops})
    if len(mlirs) < 2:
        return None
    sources = [open(m, "rb").read() for m in mlirs]
    key = _key(DESIGN_SOURCE.read_bytes(), _toolchain(), *sources)
    out = os.path.join(_cache_dir(), f"model_{key}")
    elf_path = os.path.join(out, "model.elf")
    with _BUILD_LOCK:
        if not os.path.exists(elf_path):
            tmp = f"{out}.{os.getpid()}.tmp"
            shutil.rmtree(tmp, ignore_errors=True)
            r = subprocess.run(
                [sys.executable, str(DESIGN_SOURCE), "--merge", tmp, *mlirs],
                capture_output=True,
                text=True,
                env=dict(os.environ, PEANO_INSTALL_DIR=_peano()),
            )
            if r.returncode:
                log = (r.stdout + r.stderr)[-6000:]
                raise RuntimeError(f"conv3x3 merge failed:\n{log}")
            shutil.rmtree(out, ignore_errors=True)
            os.replace(tmp, out)
        if elf_path not in _SHARED:
            import pyxrt

            elf = pyxrt.elf(elf_path)
            _SHARED[elf_path] = (elf, pyxrt.hw_context(_device(), elf))
        ctx = _SHARED[elf_path][1]
        kernels = {
            m: _SharedProgram(ctx, f"conv{k}:sequence") for k, m in enumerate(mlirs)
        }
        for op in ops:
            op.prog = kernels[op.design_mlir]
            # its own context goes once no op uses it
            _PROGRAMS.pop(op.program_path, None)
    return elf_path


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
    except ImportError:  # pragma: no cover - the op returns numpy then
        return None


def pack_weights(weight, shift, groups, chain):
    """[slab, Ci_pad, 3, 3] int8 -> [G][L] of
    ([OGP pairs][ky][CG][kx][2][8 ci][8 co], 64-byte trailer holding the shift)."""
    co, ci = weight.shape[:2]
    ogp, cg = co // (16 * groups), ci // (8 * chain)
    w = np.asarray(weight, dtype=np.int8)
    w = w.reshape(groups, ogp, 2, 8, chain, cg, 8, 3, 3)
    w = w.transpose(0, 4, 1, 7, 5, 8, 2, 6, 3)  # [g, p, pair, ky, cg, kx, j, ci8, co8]
    w = w.reshape(groups, chain, -1)
    out = np.zeros((groups, chain, w.shape[2] + WTRAILER), np.int8)
    out[:, :, : w.shape[2]] = w
    _set_shift(out, shift)
    return out.reshape(-1)


def _set_shift(packed, shift):
    """Write ``shift`` into every core's trailer of a packed [G, L, slice] array."""
    packed[:, :, -WTRAILER : -WTRAILER + 4] = np.array([shift], "<i4").view(np.int8)


def _pack_torch(x, out, pieces, chain, cg, parts, rows):
    """`conv3x3_host.c`'s pack, with torch: one piece at a time."""
    torch = _torch()
    n, ci, h, w = x.shape
    xp = torch.nn.functional.pad(torch.as_tensor(x), (1, 1, 1, 1))
    xp = torch.nn.functional.pad(xp, (0, 0, 0, 0, 0, chain * cg * 8 - ci))
    dst = torch.as_tensor(out).view(chain, 2, rows + 2, parts, cg, WPS, 8)
    for img, r, s, o, l0, nr, r0, c0, tw in pieces.tolist():
        src = xp[img, :, r0 : r0 + nr + 2, c0 : c0 + tw + 2]  # padded coordinates
        src = src.reshape(chain, cg, 8, nr + 2, tw + 2).permute(0, 3, 1, 4, 2)
        dst[:, s, l0 : l0 + nr + 2, r, :, o : o + tw + 2] = src
    return out


def _unpack_torch(y, out, pieces, co0, chain, groups, ogp, rows):
    """`conv3x3_host.c`'s unpack, with torch: one piece at a time."""
    torch = _torch()
    co = out.shape[1]
    cpc = 4 // chain
    yt = torch.as_tensor(y)[: rows * 8 * cpc * 2 * ogp * OWS * 8]
    yt = yt.view(rows, 8, cpc, 2 * ogp, OWS, 8)
    for img, r, s, o, l0, nr, r0, c0, tw in pieces.tolist():
        for g in range(groups):
            ch = co0 + g * 16 * ogp
            if ch >= co:
                break
            col, i = (
                (r * groups + g, s) if chain == 2 else ((r * 2 + s) * groups + g, 0)
            )
            blk = yt[l0 : l0 + nr, col, i, :, o : o + tw]  # [nr, 2*OGP, tw, 8]
            blk = blk.permute(1, 3, 0, 2).reshape(16 * ogp, nr, tw)
            m = min(16 * ogp, co - ch)
            out[img, ch : ch + m, r0 : r0 + nr, c0 : c0 + tw] = blk[:m]
    return out


# ------------------------------------------------------------------ the op
class Conv3x3:
    """A 3x3, stride-1, padding-1 int8 convolution with fixed weights, on the NPU.

    Args:
        weight: int8 [Co, Ci, 3, 3] (numpy or torch), Ci <= 512. Packed once into each
            core's order and kept in device buffers.
        height, width: the input (= output) size the op is built for.
        shift: output = saturate(round_half_up(sum >> shift)) as int8 (uint8 with
            ``relu``); `set_shift` changes it without a rebuild.
        relu: fuse ReLU: the tail saturates to [0, 255] instead of [-128, 127].
        batch: images per call, all in one launch.
    """

    def __init__(self, weight, *, height, width, shift, relu=False, batch=1):
        w = np.asarray(weight.cpu() if hasattr(weight, "cpu") else weight)
        if w.dtype != np.int8 or w.ndim != 4 or w.shape[2:] != (3, 3):
            raise Conv3x3ConfigError(
                f"weight must be int8 [Co, Ci, 3, 3], got {w.dtype} {w.shape}"
            )
        if height < 1 or width < 1 or batch < 1:
            raise Conv3x3ConfigError(f"bad size {batch}x{height}x{width}")
        if not 1 <= shift <= 31:
            raise Conv3x3ConfigError(f"shift must be in [1, 31], got {shift}")
        co, ci = w.shape[:2]
        self.chain, self.cg, self.ogp, self.groups, self.parts = configure(ci, co)
        self.ci, self.co, self.h, self.w, self.n = ci, co, height, width, batch
        self.shift, self.relu, self.weight = shift, relu, w
        self.pieces, self.rowmap, self.rows = make_plan(
            batch, height, width, self.parts
        )
        line, wsl, ochain = geometry(self.cg, self.ogp)
        self.slab = self.groups * 16 * self.ogp
        self.nslab = -(-co // self.slab)
        fmt, self.program_path, extra = design_artifacts(
            self.rows,
            self.cg,
            self.chain,
            self.ogp,
            self.groups,
            self.parts,
            self.nslab,
            relu,
        )
        self.prog = _program(fmt, self.program_path, extra)
        # the design's module, for `share_context`
        mlir = os.path.join(os.path.dirname(self.program_path), "design.mlir")
        self.design_mlir = mlir if fmt == "elf" else None
        x_bytes = 2 * self.chain * (self.rows + 2) * self.parts * line
        self.x_bo, self.x_view = self.prog.bo(x_bytes, 0)
        self.x_view[:] = 0
        ci_pad = self.chain * self.cg * 8
        ws = np.zeros((self.nslab * self.slab, ci_pad, 3, 3), np.int8)
        ws[:co, :ci] = w
        packed = np.concatenate(
            [
                pack_weights(
                    ws[k * self.slab : (k + 1) * self.slab],
                    shift,
                    self.groups,
                    self.chain,
                )
                for k in range(self.nslab)
            ]
        )
        self.w_bo, self.w_view = self.prog.bo(packed.nbytes, 1)
        self.w_view[:] = packed
        self.w_bo.sync(_sync_dir("to"))
        # one slab's output lines; slack: the unpack reads whole 8-pixel groups past
        # a narrow tile
        self.y_slab = self.rows * 8 * (4 // self.chain) * ochain
        self.y_bo, self.y_view = self.prog.bo(self.nslab * self.y_slab + 64, 2)

    def set_shift(self, shift):
        """Requantize by ``shift`` from the next call on (rewrites the trailers)."""
        if not 1 <= shift <= 31:
            raise Conv3x3ConfigError(f"shift must be in [1, 31], got {shift}")
        _set_shift(self.w_view.reshape(self.nslab * self.groups, self.chain, -1), shift)
        self.w_bo.sync(_sync_dir("to"))
        self.shift = shift

    def launch(self):
        """Run the launch on the packed input (the NPU alone, for timing)."""
        self.prog.run(self.x_bo, self.w_bo, self.y_bo)

    def __call__(self, x, out=None):
        """x int8 [N, Ci, H, W] (or [Ci, H, W] when N = 1) -> [N, Co, H, W] (or
        [Co, H, W]) int8, uint8 with relu.

        Returns a torch tensor when torch is installed, else numpy; ``out`` may be a
        contiguous buffer of that shape and type.
        """
        torch = _torch()
        single = x.ndim == 3
        x = x[None] if single else x
        shape = (self.n, self.ci, self.h, self.w)
        dtype = str(x.dtype).split(".")[-1]
        if tuple(x.shape) != shape or dtype != "int8":
            raise Conv3x3ConfigError(
                f"built for int8 {shape}, got {dtype} {tuple(x.shape)}"
            )
        x = x.cpu() if hasattr(x, "cpu") else x
        x = x.contiguous() if hasattr(x, "contiguous") else np.ascontiguousarray(x)
        oshape = (self.n, self.co, self.h, self.w)
        if out is None and torch is not None:
            out = torch.empty(oshape, dtype=torch.uint8 if self.relu else torch.int8)
        elif out is None:
            out = np.empty(oshape, np.uint8 if self.relu else np.int8)
        out4 = out.reshape(oshape) if single else out
        lib = host_lib()
        if lib is not None:
            pc = self.pieces
            lib.conv3x3_pack(
                _ptr(x), self.ci, self.h, self.w, _ptr(self.x_view), _ptr(pc), len(pc),
                self.chain, self.cg, self.parts, self.rows, _threads(),
            )  # fmt: skip
        else:
            _pack_torch(x, self.x_view, self.pieces, self.chain, self.cg, self.parts,
                        self.rows)  # fmt: skip
        self.x_bo.sync(_sync_dir("to"))
        self.prog.run(self.x_bo, self.w_bo, self.y_bo)
        self.y_bo.sync(_sync_dir("from"))
        ydt = np.uint8 if self.relu else np.int8
        for k in range(self.nslab):
            co0, y = k * self.slab, self.y_view[k * self.y_slab :]
            if lib is not None:
                lib.conv3x3_unpack(
                    _ptr(y), self.n, self.co, co0, self.h, self.w, _ptr(out4),
                    _ptr(self.rowmap), self.chain, self.groups, self.ogp, _threads(),
                )  # fmt: skip
            else:
                _unpack_torch(y.view(ydt), out4, self.pieces, co0, self.chain,
                              self.groups, self.ogp, self.rows)  # fmt: skip
        return out

    def reference(self, x):
        """The same integers on the CPU: int32 sums, round half up, saturate."""
        return reference(x, self.weight, self.shift, self.relu)


def reference(x, weight, shift, relu=False):
    """[N, Ci, H, W] (or [Ci, H, W]) int8 -> the op's output, computed on the CPU."""
    torch = _torch()
    if torch is None:
        raise RuntimeError("reference needs torch")
    single = x.ndim == 3
    x = np.asarray(x.cpu() if hasattr(x, "cpu") else x)
    xt = torch.as_tensor(x, dtype=torch.float64)
    xt = xt[None] if single else xt
    wt = torch.as_tensor(np.asarray(weight), dtype=torch.float64)
    # float64 sums of int8 products are exact (far below 2^53)
    acc = torch.nn.functional.conv2d(xt, wt, padding=1).to(torch.int64)
    q = torch.div(acc + (1 << (shift - 1)), 1 << shift, rounding_mode="floor")
    q = q.clamp(0, 255).to(torch.uint8) if relu else q.clamp(-128, 127).to(torch.int8)
    return q[0] if single else q
