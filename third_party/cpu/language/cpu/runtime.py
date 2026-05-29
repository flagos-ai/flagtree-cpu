"""Python callables wrapping libTritonCPURuntime.so C functions.

Exposes the pre-compiled C runtime helpers (whole-layer fused kernels,
in-place KV cache write, etc.) as regular Python functions operating on
torch tensors.  These are complementary to Triton @triton.jit kernels —
use them when the operation is too coarse-grained to fit inside a
single @triton.jit (e.g. fused gate+up+silu+mul spans 3 GEMVs and 2
elementwise ops that don't compose cleanly as one dot product).

Analogous to `triton.language.extra.cuda.libdevice` — a thin Python
veneer over a C runtime. No JIT compilation, just ctypes dispatch.

Usage:
    from triton.language.extra.cpu.runtime import fused_mlp_bf16
    fused_mlp_bf16(x, gate_packed, up_packed, gate_scale, up_scale, out, K, N)
"""

import ctypes
import os
from pathlib import Path

# libTritonCPURuntime.so lives in triton._C alongside libtriton.so.
# Use triton package location rather than __file__ (this module is symlinked
# from third_party/cpu/language/cpu/ into python/triton/language/extra/cpu/,
# so __file__.resolve() walks into third_party instead of python/triton/).
import triton as _triton
_LIB_PATH = Path(_triton.__file__).parent / "_C" / "libTritonCPURuntime.so"

if not _LIB_PATH.exists():
    raise ImportError(
        f"libTritonCPURuntime.so not found at {_LIB_PATH}. "
        f"Build triton-cpu or set LD_LIBRARY_PATH to include python/triton/_C."
    )

_rt = ctypes.CDLL(str(_LIB_PATH))

# ── Signatures (must match runtime_transformer_layer.cpp / runtime_gemv.cpp) ──
_rt.fused_mlp_bf16.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int64] * 2
_rt.fused_mlp_bf16.restype = None

_rt.flash_attn_decode_bf16.argtypes = (
    [ctypes.c_void_p] * 4
    + [ctypes.c_int64, ctypes.c_int64, ctypes.c_float]
    + [ctypes.c_int64] * 4
)
_rt.flash_attn_decode_bf16.restype = None

_rt.standalone_kv_cache_write_bf16.argtypes = (
    [ctypes.c_void_p] * 4 + [ctypes.c_int64] * 4
)
_rt.standalone_kv_cache_write_bf16.restype = None

_rt.standalone_rope_bf16.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int64] * 3
_rt.standalone_rope_bf16.restype = None

_rt.standalone_residual_add_bf16.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_int64]
_rt.standalone_residual_add_bf16.restype = None


# ── Python wrappers ──────────────────────────────────────────────────────────

def fused_mlp_bf16(x, gate_packed, up_packed, gate_scale, up_scale, out, K, N):
    """Fused INT8 SDOT gate+up+SWIGLU for MLP decode (M=1, BF16 I/O).

    Computes:  out = silu(gate_packed @ x_int8) * (up_packed @ x_int8)
    where x is dynamically quantized to INT8 inside the kernel.

    Args:
        x:            [K] bf16 activation (flattened)
        gate_packed:  [K//4, N//4, 4, 4] int8 SDOT-packed weight
        up_packed:    [K//4, N//4, 4, 4] int8 SDOT-packed weight
        gate_scale:   [N] fp32 per-column scale
        up_scale:     [N] fp32 per-column scale
        out:          [N] bf16 output buffer (written in-place)
        K, N:         ints
    """
    _rt.fused_mlp_bf16(
        x.data_ptr(),
        gate_packed.data_ptr(),
        up_packed.data_ptr(),
        gate_scale.data_ptr(),
        up_scale.data_ptr(),
        out.data_ptr(),
        K, N,
    )


def flash_attn_decode_bf16(q, k, v, out, seq_len, head_dim, sm_scale,
                           n_heads_q, n_heads_kv, stride_k, stride_v):
    """M=1 Flash Attention decode. BF16 inputs, online softmax, GQA.

    Args:
        q:            [Hq, D] bf16
        k, v:         [Hkv, seq_len, D] bf16
        out:          [Hq, D] bf16 (written in-place)
        seq_len:      int, length of k/v
        head_dim:     int, D
        sm_scale:     float, 1/sqrt(D) typically
        n_heads_q:    int, Hq
        n_heads_kv:   int, Hkv (GQA: Hq >= Hkv, group size = Hq / Hkv)
        stride_k:     int, k's seq-dim stride in elements
        stride_v:     int, v's seq-dim stride in elements
    """
    _rt.flash_attn_decode_bf16(
        q.data_ptr(), k.data_ptr(), v.data_ptr(), out.data_ptr(),
        seq_len, head_dim, sm_scale,
        n_heads_q, n_heads_kv, stride_k, stride_v,
    )


def kv_cache_write_bf16(k_cache, v_cache, k_new, v_new,
                        n_kv_heads, max_seq_len, head_dim, pos):
    """In-place write of k_new/v_new into pre-allocated KV cache at position pos.

    Args:
        k_cache:     [n_kv_heads, max_seq_len, head_dim] bf16
        v_cache:     [n_kv_heads, max_seq_len, head_dim] bf16
        k_new:       [n_kv_heads, head_dim] bf16 (single timestep)
        v_new:       [n_kv_heads, head_dim] bf16
        n_kv_heads:  int
        max_seq_len: int
        head_dim:    int
        pos:         int, timestep index (0 <= pos < max_seq_len)
    """
    _rt.standalone_kv_cache_write_bf16(
        k_cache.data_ptr(), v_cache.data_ptr(),
        k_new.data_ptr(), v_new.data_ptr(),
        n_kv_heads, max_seq_len, head_dim, pos,
    )


def rope_bf16(q, k, cos, sin, n_heads_q, n_heads_kv, head_dim):
    """Apply RoPE (rotary position embedding) to q and k in-place."""
    _rt.standalone_rope_bf16(
        q.data_ptr(), k.data_ptr(), cos.data_ptr(), sin.data_ptr(),
        n_heads_q, n_heads_kv, head_dim,
    )


def residual_add_bf16(out, residual, numel):
    """In-place: out += residual  (BF16)."""
    _rt.standalone_residual_add_bf16(
        out.data_ptr(), residual.data_ptr(), numel,
    )


__all__ = [
    "fused_mlp_bf16",
    "flash_attn_decode_bf16",
    "kv_cache_write_bf16",
    "rope_bf16",
    "residual_add_bf16",
]
