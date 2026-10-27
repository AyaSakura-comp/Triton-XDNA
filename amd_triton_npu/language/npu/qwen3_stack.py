# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""``tl.extra.npu.qwen3_stack`` -- a Qwen3-0.6B-sized transformer stack on the whole NPU,
every layer, the final norm and a linear head in ONE launch.

What a call computes, for two sequences of hidden states x0 [n0, 1024], x1 [n1, 1024]:

    for each layer:  h = x + o_proj(attn(rope(q_norm(q)), rope(k_norm(k)), v))
                     x = h + down(silu(gate(rms(h))) * up(rms(h)))      (q, k, v of rms(x))
    logits = head(rms(x) * norm)

Qwen3-0.6B's shapes: hidden 1024, 16 query and 8 kv heads of 128, MLP 3072, RMSNorm eps
1e-6, RoPE from the given inverse frequencies; any number of layers; a head of any width,
no biases. Attention is bidirectional within each sequence: every position sees the
sequence's first n positions (n0 or n1) -- the layout of a masked-diffusion model such as
OmniVoice's backbone, with its conditional and unconditional sequences side by side.

How it runs (``kernels/qwen3_stack_design.py``, ``kernels/qwen3_stack_core.cc``): operands
bfp16 (8 values per shared exponent), fp32 accumulation. A row of 8 cores is a cascade chain
over K; column c's weights stream from DDR through memtile c to its 4 cores; the tails
(column 7) finish each output tile in the registers of its last MAC (RMSNorm, residual,
logits). Every core then does q/k-norm, RoPE and the V transpose of its rows; column c runs
kv head c's attention with an online softmax. Rows come in groups of 64 positions per
sequence; a call uses the build with the fewest groups that hold the longer sequence (built
on first use, cached).

Measured on Strix Halo (OmniVoice: 28 layers, an 8 x 1025 head; the two sequences of a
denoising step): 12.5 ms per launch at 58 / 43 positions, 21.7 ms at 113 / 92, 55 ms at
264 / 222; logits within 2-3% (relative norm) of the fp32 model.

    stack = Qwen3Stack(layers, norm=w_norm, head=w_head, inv_freq=rotary.inv_freq)
    y0, y1 = stack(x0, x1)          # [n0, 1024], [n1, 1024] -> [n0, N], [n1, N] float32

``layers`` is a list of dicts of weights (torch or numpy): q, k, v, o, gate, up, down
([out, in], as nn.Linear stores them), input_norm, post_norm ([1024]), q_norm, k_norm
([128]). The weights are packed once (~10 s; cached on disk by their contents) into one
device buffer that every build shares.

Limits: sequences of 1 to 384 positions (MAX_GROUPS row groups of 64); torch is required.
"""

import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from .conv3x3 import _device, _key, _peano, _toolchain

HERE = Path(__file__).resolve().parent
KERNEL_SOURCE = HERE / "kernels" / "qwen3_stack_core.cc"
DESIGN_SOURCE = HERE / "kernels" / "qwen3_stack_design.py"

D, NH, NKV, DH, INTER = 1024, 16, 8, 128, 3072
EPS = 1e-6
TILE, BLK = 4608, 72
GROUP_ROWS = 64  # positions of each sequence per row group
MAX_GROUPS = 6
WEIGHTS_FORMAT = 1  # the packed layout; part of the weight cache's key

_BUILD_LOCK = threading.Lock()
_DESIGN = {}


class Qwen3StackConfigError(ValueError):
    pass


def _design():
    """The design module, for its layout (it imports aie only inside its functions)."""
    if "mod" not in _DESIGN:
        spec = importlib.util.spec_from_file_location(
            "_qwen3_stack_design", DESIGN_SOURCE
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _DESIGN["mod"] = mod
    return _DESIGN["mod"]


def _cache_dir():
    root = os.environ.get("AMD_TRITON_NPU_STACK_CACHE") or os.path.join(
        os.path.expanduser("~"), ".cache", "amd_triton_npu", "qwen3_stack"
    )
    os.makedirs(root, exist_ok=True)
    return root


# ------------------------------------------------------------------ bfp16 tiles
def _torch():
    import torch

    return torch


def bfp_encode(x):
    """float32 torch tensor [..., 8] -> uint8 [..., 9] bfp16ebs8 groups (exponent, 8
    mantissas), as the cores convert: bf16 values, exponent of the largest, mantissa =
    v * 2^(6 - e)."""
    torch = _torch()
    xb = x.to(torch.bfloat16).to(torch.float32)
    emax = ((xb.view(torch.int32) >> 23) & 0xFF).amax(-1)
    sc = torch.ldexp(torch.ones(emax.shape), (133 - emax).float())
    m = torch.round(xb * sc[..., None])
    over = m.abs().amax(-1) > 127
    emax = torch.where(over, emax + 1, emax)
    sc = torch.where(over, sc * 0.5, sc)
    m = torch.round(xb * sc[..., None])
    small = emax < 8
    m[small] = 0
    emax[small] = 0
    out = torch.empty(x.shape[:-1] + (9,), dtype=torch.uint8)
    out[..., 0] = emax.to(torch.uint8)
    out[..., 1:] = m.to(torch.int8).view(torch.uint8)
    return out


def bfp_decode(g):
    """uint8 torch tensor [..., 9] -> float32 [..., 8]."""
    torch = _torch()
    e = g[..., 0].to(torch.int32)
    m = g[..., 1:].view(torch.int8).to(torch.float32)
    return torch.ldexp(m, (e - 127 - 6)[..., None].to(torch.float32))


def tiles(m):
    """float32 [R, C] (R % 32 == 0, C % 128 == 0) -> uint8 [R/32, C/128, 4608] bfp16 tiles
    ([row pair][column block][row block of the pair][72], 8 values of a row per exponent).
    """
    R, C = m.shape
    g = bfp_encode(m.reshape(R // 8, 8, C // 8, 8).transpose(1, 2))
    g = g.reshape(R // 32, 2, 2, C // 128, 16, BLK)
    return g.permute(0, 3, 1, 4, 2, 5).reshape(R // 32, C // 128, TILE)


def untile(t):
    """uint8 [..., 4608] -> float32 [..., 32, 128]."""
    lead = t.shape[:-1]
    v = bfp_decode(t.reshape(-1, 2, 16, 2, 8, 9))  # [n, p, kb, j, row, 8]
    return v.permute(0, 1, 3, 4, 2, 5).reshape(*lead, 32, 128)


def untile_vt(t):
    """A V^T tile [16 d blocks][4 key blocks][72] -> V [32 keys, 128 d] float32."""
    v = bfp_decode(t.reshape(16, 4, 8, 9))  # [db, kb, d, key]
    return v.permute(1, 3, 0, 2).reshape(32, 128)


def _bf16_bits(x):
    """float32 torch tensor -> uint8 bytes of its bf16 values (round to nearest even)."""
    torch = _torch()
    return x.to(torch.bfloat16).view(torch.uint8)


# ------------------------------------------------------------------ weights
LAYER_KEYS = (
    "q",
    "k",
    "v",
    "o",
    "gate",
    "up",
    "down",
    "input_norm",
    "post_norm",
    "q_norm",
    "k_norm",
)
LAYER_SHAPES = dict(
    q=(NH * DH, D),
    k=(NKV * DH, D),
    v=(NKV * DH, D),
    o=(D, NH * DH),
    gate=(INTER, D),
    up=(INTER, D),
    down=(D, INTER),
    input_norm=(D,),
    post_norm=(D,),
    q_norm=(DH,),
    k_norm=(DH,),
)


def _f32(t):
    torch = _torch()
    t = torch.as_tensor(np.asarray(t)) if not isinstance(t, torch.Tensor) else t
    return t.detach().to("cpu", torch.float32)


def column_stream(lay):
    """One layer's weights (norms folded in) as each column's objects in the cores'
    order: uint8 [8][480][4608] (QKV 128, O 64, gate/up 192, down 96)."""
    torch = _torch()
    g_in, g_post = _f32(lay["input_norm"]), _f32(lay["post_norm"])
    tq = tiles(
        torch.cat([_f32(lay["q"]), _f32(lay["k"]), _f32(lay["v"])]) * g_in
    )  # [128, 8]
    to = tiles(_f32(lay["o"]))  # [32, 16]
    tg, tu = tiles(_f32(lay["gate"]) * g_post), tiles(
        _f32(lay["up"]) * g_post
    )  # [96, 8]
    td = tiles(_f32(lay["down"]))  # [32, 24]
    cols = []
    for c in range(8):
        # gate/up units alternate gate block i, up block i: (u, g) -> n-tile 4(u >> 1) + g
        gu = torch.stack(
            [tg[:, c].reshape(24, 4, TILE), tu[:, c].reshape(24, 4, TILE)], 1
        )
        cols.append(
            torch.cat(
                [
                    tq[:, c],
                    to[:, 2 * c : 2 * c + 2].reshape(-1, TILE),
                    gu.reshape(-1, TILE),
                    td[:, 3 * c : 3 * c + 3].reshape(-1, TILE),
                ]
            )
        )
    return torch.stack(cols)


def head_stream(head, norm, units):
    """The head [N, 1024] with the final norm folded in, padded to 64 * units rows:
    uint8 [8][2 units][4608]."""
    torch = _torch()
    w = torch.zeros((64 * units, D))
    w[: head.shape[0]] = _f32(head) * _f32(norm)
    return tiles(w).transpose(0, 1).contiguous()


def _digest(arrays):
    """A key for the packed weights: the bytes of every source tensor."""
    torch = _torch()
    h = hashlib.sha256(f"qwen3_stack weights v{WEIGHTS_FORMAT}".encode())
    for a in arrays:
        t = (
            a.detach().cpu().contiguous()
            if isinstance(a, torch.Tensor)
            else torch.as_tensor(np.ascontiguousarray(a))
        )
        h.update(f"{tuple(t.shape)} {t.dtype};".encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy())
    return h.hexdigest()[:24]


def pack_weights(layers, norm, head, units):
    """The whole weight stream (cached on disk by content): uint8 [8][NL * 480 + 2 units][4608],
    per column every layer's objects, then the head's."""
    torch = _torch()
    srcs = [lay[k] for lay in layers for k in LAYER_KEYS] + [norm, head]
    path = os.path.join(_cache_dir(), f"weights_{_digest(srcs)}_u{units}.npy")
    if os.path.exists(path):
        return np.load(path, mmap_mode="r")
    wl = _design().w_objects_per_layer()
    buf = torch.empty((8, len(layers) * wl + 2 * units, TILE), dtype=torch.uint8)
    for i, lay in enumerate(layers):
        buf[:, i * wl : (i + 1) * wl] = column_stream(lay)
    buf[:, len(layers) * wl :] = head_stream(head, norm, units)
    tmp = f"{path}.{os.getpid()}.tmp.npy"
    np.save(tmp, buf.numpy())
    os.replace(tmp, path)
    return buf.numpy()


def rope_tiles(inv_freq, g):
    """Row group g's RoPE tiles: row block rb holds positions 64 g + 32 (rb & 1) + 0..31 as
    [cos | sin] of 64 frequencies -> uint8 [4][4608]."""
    torch = _torch()
    f = _f32(inv_freq).double()
    out = []
    for rb in range(4):
        pos = torch.arange(32, dtype=torch.float64) + 32 * (rb & 1) + GROUP_ROWS * g
        ang = pos[:, None] * f[None, :]
        out.append(tiles(torch.cat([ang.cos(), ang.sin()], 1).float())[0, 0])
    return torch.stack(out)


# ------------------------------------------------------------------ build (cached)
def _include_dir():
    return os.path.join(os.path.dirname(_peano()), "mlir_aie", "include")


def kernel_objects(units, defines=()):
    """The core kernels for a head of ``units`` (cached): {"body": path, "tail": path}."""
    out = {}
    for v in ("body", "tail"):
        defs = [
            f"-DHEAD_U={units}",
            *(["-DSTACK_TAIL"] if v == "tail" else []),
            *defines,
        ]
        key = _key(KERNEL_SOURCE.read_bytes(), " ".join(defs), _toolchain())
        obj = os.path.join(_cache_dir(), f"qwen3_stack_{v}_{key}.o")
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
                    "-Wno-unused-function",
                    f"-I{_include_dir()}",
                    # frame sizes, so aiecc can size each core's stack from the call graph
                    "-fstack-size-section",
                    "-D__AIE_API_AIE_ADF_HPP__",
                    *defs,
                    "-c",
                    str(KERNEL_SOURCE),
                    "-o",
                    tmp,
                ]
                r = subprocess.run(cmd, capture_output=True, text=True)
                if r.returncode:
                    raise RuntimeError(
                        f"qwen3_stack kernel build failed:\n{r.stderr[-4000:]}"
                    )
                os.replace(tmp, obj)
        out[v] = obj
    return out


def design_artifacts(layers, groups, units, defines=(), trace=0):
    """The program of one (layers, groups, head units) build, made in a subprocess and
    cached: the directory holding final.xclbin and insts.bin."""
    objs = kernel_objects(units, defines)
    params = dict(
        NL=layers,
        NG=groups,
        HU=units,
        obj_body=objs["body"],
        obj_tail=objs["tail"],
        trace=trace,
    )
    key = _key(
        DESIGN_SOURCE.read_bytes(), json.dumps(params, sort_keys=True), _toolchain()
    )
    out = os.path.join(_cache_dir(), f"design_nl{layers}_g{groups}_u{units}_{key}")
    with _BUILD_LOCK:
        if not os.path.exists(os.path.join(out, "insts.bin")):
            tmp = f"{out}.{os.getpid()}.tmp"
            shutil.rmtree(tmp, ignore_errors=True)
            r = subprocess.run(
                [sys.executable, str(DESIGN_SOURCE), json.dumps(params), tmp],
                capture_output=True,
                text=True,
                env=dict(os.environ, PEANO_INSTALL_DIR=_peano()),
                cwd=_cache_dir(),
            )
            if r.returncode:
                raise RuntimeError(
                    f"qwen3_stack design build failed:\n{(r.stdout + r.stderr)[-6000:]}"
                )
            shutil.rmtree(out, ignore_errors=True)
            os.replace(tmp, out)
    return out


# ------------------------------------------------------------------ run (pyxrt)
class _Program:
    """One build loaded on the NPU: its hardware context, instruction stream, activation and
    logits buffers; the weights come from the stack's shared buffer."""

    def __init__(self, stack, groups, trace=0):
        import pyxrt

        self.stack, self.NG = stack, groups
        self.L = _design().Layout(stack.layers, groups)
        self.dir = design_artifacts(
            stack.layers, groups, stack.units, stack.defines, trace
        )
        dev = _device()
        self.xclbin = pyxrt.xclbin(os.path.join(self.dir, "final.xclbin"))
        dev.register_xclbin(self.xclbin)
        self.ctx = pyxrt.hw_context(dev, self.xclbin.get_uuid())
        self.kernel = pyxrt.kernel(self.ctx, self.xclbin.get_kernels()[0].get_name())
        words = np.fromfile(os.path.join(self.dir, "insts.bin"), dtype=np.uint32)
        self.ibo = pyxrt.bo(
            dev, words.nbytes, pyxrt.bo.cacheable, self.kernel.group_id(1)
        )
        self.ibo.write(words.view(np.uint8), 0)
        self.ibo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        self.act, self.actv = self.bo(self.L.ntiles * TILE, 0)
        self.log, self.logv = self.bo(groups * stack.units * 4 * TILE, 2)
        torch = _torch()
        a = torch.from_numpy(self.actv).view(-1, TILE)
        a.zero_()
        for g in range(groups):
            # every column's copy of the group's RoPE tiles (its QKV post-pass input)
            rt = rope_tiles(stack.inv_freq, g)
            for c in range(8):
                q0 = self.L["QR", g] + _design().QR_COL * c
                a[q0 : q0 + 4] = rt
        # q/k-norm weights times sqrt(128), as bf16 (qwen3_stack_core.cc qp_qk)
        a[self.L["GAMMA"] : self.L["GAMMA"] + stack.layers, :512] = stack.gamma
        self.act.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        args = [3, self.ibo, words.nbytes, self.act, stack.weights_bo(self), self.log]
        if trace:
            self.trace, self.tracev = self.bo(trace, 3)
            self.tracev[:] = 0
            self.trace.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            args.append(self.trace)
        self.run_ = pyxrt.run(self.kernel)
        for i, v in enumerate(args):
            self.run_.set_arg(i, v)

    def bo(self, nbytes, arg):
        """A host buffer for runtime argument ``arg`` (0 activations, 1 weights, 2 logits, 3 trace)."""
        import pyxrt

        b = pyxrt.bo(
            _device(), nbytes, pyxrt.bo.host_only, self.kernel.group_id(3 + arg)
        )
        return b, np.frombuffer(b.map(), dtype=np.uint8, count=nbytes)

    def region(self, name, n=1, g=0):
        """``n`` tiles of activation region ``name`` of row group ``g`` (as the last run left them)."""
        a = self.actv.reshape(-1, TILE)
        return a[self.L[name, g] : self.L[name, g] + n]

    def __call__(self, x0, x1):
        import pyxrt

        torch = _torch()
        n0, n1, NG = len(x0), len(x1), self.NG
        a = torch.from_numpy(self.actv).view(-1, TILE)
        a[self.L["PARAMS"], :8] = torch.tensor([n0, n1], dtype=torch.int32).view(
            torch.uint8
        )
        # position p of sequence s -> row group p // 64, row 64 s + p % 64
        x = torch.zeros((2, NG, GROUP_ROWS, D))
        x[0].view(-1, D)[:n0] = torch.as_tensor(x0, dtype=torch.float32)
        x[1].view(-1, D)[:n1] = torch.as_tensor(x1, dtype=torch.float32)
        t = tiles(x.transpose(0, 1).reshape(NG * 128, D))  # [4 NG row blocks, 8 units]
        for g in range(NG):
            a[self.L["X", g] : self.L["X", g] + 32] = (
                t[4 * g : 4 * g + 4].transpose(0, 1).reshape(32, TILE)
            )
        self.act.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        t = time.perf_counter()
        self.run_.start()
        state = self.run_.wait()
        self.stack.last_npu_s = time.perf_counter() - t
        if state != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            raise RuntimeError(f"qwen3_stack run ended in {state}")
        self.log.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        self.act.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        HU = self.stack.units
        lg = (
            torch.from_numpy(self.logv)
            .view(torch.int16)
            .view(NG, HU, 4, TILE // 2)[..., :2048]
        )
        # [group][unit][row block of 32][group of 32 columns][rb][nb][8][8]
        #   -> [sequence][64 NG positions][units * 64 columns]
        lg = (
            (lg.to(torch.int32) << 16)
            .view(torch.float32)
            .view(NG, HU, 2, 2, 2, 4, 4, 8, 8)
        )
        out = lg.permute(2, 0, 3, 5, 7, 1, 4, 6, 8).reshape(
            2, NG * GROUP_ROWS, HU * 64
        )  # a fresh tensor
        N = self.stack.out_features
        return out[0, :n0, :N], out[1, :n1, :N]

    def trace_events(self):
        """The last run's hardware trace in Trace Event Format (needs ``trace`` at build)."""
        import pyxrt
        from aie.utils.trace.parse import parse_trace

        self.trace.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        mlir = open(
            os.path.join(self.dir, "final.prj", "input_with_addresses.mlir")
        ).read()
        return parse_trace(self.tracev.view(np.uint32), mlir)


# ------------------------------------------------------------------ the op
class Qwen3Stack:
    """A Qwen3-0.6B-sized stack, its final RMSNorm and a linear head on the NPU, one launch
    per call of two sequences.

    Args:
        layers: one dict per decoder layer of q, k, v, o, gate, up, down ([out, in]),
            input_norm, post_norm ([1024]), q_norm, k_norm ([128]) weights (torch or numpy).
        norm: the final RMSNorm's weight [1024].
        head: the head's weight [N, 1024] (no bias).
        inv_freq: RoPE inverse frequencies [64].
        trace: bytes of hardware trace per build (profiling; 0 = off).
        defines: extra -D flags for the core kernels (experiments).
    """

    def __init__(self, layers, *, norm, head, inv_freq, trace=0, defines=()):
        torch = _torch()
        if not layers:
            raise Qwen3StackConfigError("no layers")
        for i, lay in enumerate(layers):
            for k in LAYER_KEYS:
                if k not in lay:
                    raise Qwen3StackConfigError(f"layer {i}: no {k!r} weight")
                if tuple(lay[k].shape) != LAYER_SHAPES[k]:
                    raise Qwen3StackConfigError(
                        f"layer {i} {k}: expected {LAYER_SHAPES[k]}, got {tuple(lay[k].shape)}"
                    )
        if (
            tuple(norm.shape) != (D,)
            or head.ndim != 2
            or head.shape[1] != D
            or tuple(inv_freq.shape) != (DH // 2,)
        ):
            raise Qwen3StackConfigError(
                f"norm [1024], head [N, 1024], inv_freq [64] expected, got {tuple(norm.shape)}, "
                f"{tuple(head.shape)}, {tuple(inv_freq.shape)}"
            )
        self.layers = len(layers)
        self.out_features = int(head.shape[0])
        self.units = -(-self.out_features // 64)
        self.inv_freq = _f32(inv_freq)
        self.trace, self.defines = trace, tuple(defines)
        self.max_tokens = MAX_GROUPS * GROUP_ROWS
        gamma = torch.stack(
            [torch.cat([_f32(lay["q_norm"]), _f32(lay["k_norm"])]) for lay in layers]
        )
        self.gamma = _bf16_bits(
            gamma * torch.tensor(math.sqrt(DH), dtype=torch.float32)
        )  # [NL, 512] bytes
        self.packed = pack_weights(layers, norm, head, self.units)
        self._wbo = None
        self.programs = {}  # row groups -> _Program
        self.last_npu_s = (
            None  # the last launch's time on the NPU (start to completion)
        )
        self._lock = threading.Lock()

    def weights_bo(self, prog):
        """The device copy of the packed weights, made once and shared by every build."""
        if self._wbo is None:
            b, view = prog.bo(self.packed.size, 1)
            view[:] = np.asarray(self.packed).reshape(-1)
            import pyxrt

            b.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            self._wbo = b
        return self._wbo

    def program(self, groups):
        """The loaded build for ``groups`` row groups (built and loaded on first use)."""
        if not 1 <= groups <= MAX_GROUPS:
            raise Qwen3StackConfigError(
                f"row groups must be 1..{MAX_GROUPS}, got {groups}"
            )
        with self._lock:
            if groups not in self.programs:
                self.programs[groups] = _Program(self, groups, self.trace)
            return self.programs[groups]

    def groups_for(self, n0, n1):
        return -(-max(n0, n1) // GROUP_ROWS)

    def __call__(self, x0, x1):
        """Hidden states of two sequences, [n0, 1024] and [n1, 1024] (1 <= n <= 384) -> their
        logits [n0, N] and [n1, N], float32 torch tensors (views of one new tensor)."""
        n0, n1 = len(x0), len(x1)
        if not (1 <= n0 <= self.max_tokens and 1 <= n1 <= self.max_tokens):
            raise Qwen3StackConfigError(
                f"sequences must have 1..{self.max_tokens} positions, got {n0}, {n1}"
            )
        if x0.shape[-1] != D or x1.shape[-1] != D:
            raise Qwen3StackConfigError(f"hidden size must be {D}")
        prog = self.program(self.groups_for(n0, n1))
        with self._lock:
            return prog(x0, x1)


def reference(layers, norm, head, inv_freq, x):
    """One sequence x [n, 1024] through the stack in fp32 -> logits [n, N]: what the op
    computes, without its bfp16 rounding."""
    torch = _torch()
    F = torch.nn.functional
    x = _f32(x)
    n = len(x)
    ang = (
        torch.arange(n, dtype=torch.float64)[:, None] * _f32(inv_freq).double()[None, :]
    )
    cos, sin = (
        torch.cat([ang.cos()] * 2, 1).float(),
        torch.cat([ang.sin()] * 2, 1).float(),
    )

    def rms(t, w):
        return t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + EPS) * _f32(w)

    def rope(t):  # [heads, n, 128]
        rot = torch.cat([-t[..., DH // 2 :], t[..., : DH // 2]], -1)
        return t * cos + rot * sin

    for lay in layers:
        h = rms(x, lay["input_norm"])
        q = rms((h @ _f32(lay["q"]).T).view(n, NH, DH), lay["q_norm"]).transpose(0, 1)
        k = rms((h @ _f32(lay["k"]).T).view(n, NKV, DH), lay["k_norm"]).transpose(0, 1)
        v = (h @ _f32(lay["v"]).T).view(n, NKV, DH).transpose(0, 1)
        k, v = k.repeat_interleave(NH // NKV, 0), v.repeat_interleave(NH // NKV, 0)
        o = (
            F.scaled_dot_product_attention(rope(q), rope(k), v)
            .transpose(0, 1)
            .reshape(n, NH * DH)
        )
        x = x + o @ _f32(lay["o"]).T
        h = rms(x, lay["post_norm"])
        x = (
            x
            + (F.silu(h @ _f32(lay["gate"]).T) * (h @ _f32(lay["up"]).T))
            @ _f32(lay["down"]).T
        )
    return rms(x, norm) @ _f32(head).T
