# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Thin ctypes veneer over libTritonCPURuntime for the TLE-CPU subset.

Exposes ``flash_attn_decode_bf16`` as a host callable operating on torch
tensors (no JIT compilation). Analogous to ``triton.language.extra.cuda.
libdevice``: a regular Python function dispatching straight into the
pre-compiled C runtime. The named TLE micro-kernels used inside @triton.jit
kernels live in ``tle_ops`` instead; this module is for the coarse-grained
decode attention that FlagGems calls directly from the host.
"""
import ctypes
from pathlib import Path

import triton as _triton

_rt = None


def _get_rt():
    """Lazily dlopen libTritonCPURuntime and bind ``flash_attn_decode_bf16``.

    Deferred to first call rather than import time: on macOS the runtime's
    OpenMP symbols are left undefined (dynamic_lookup) and resolved from the
    libomp that PyTorch loads, so the library can only be opened once torch is
    imported. Opening it eagerly here would break ``import triton`` in
    torch-free processes.
    """
    global _rt
    if _rt is not None:
        return _rt
    # libTritonCPURuntime lives in triton._C alongside libtriton. Use the
    # triton package location rather than __file__ (this module is symlinked
    # from third_party/cpu/language/cpu/ into triton/language/extra/cpu/).
    c_dir = Path(_triton.__file__).parent / "_C"
    # macOS may emit a .dylib while Linux emits .so; accept either.
    lib_path = None
    for name in ("libTritonCPURuntime.so", "libTritonCPURuntime.dylib"):
        candidate = c_dir / name
        if candidate.exists():
            lib_path = candidate
            break
    if lib_path is None:
        raise ImportError(
            f"libTritonCPURuntime not found in {c_dir}. Build triton-cpu first."
        )
    rt = ctypes.CDLL(str(lib_path))
    # Must match runtime_flash_attn_decode.cpp:
    #   void flash_attn_decode_bf16(const uint16_t *Q, *K, *V; uint16_t *Out;
    #       int64_t seq_len, head_dim; float sm_scale;
    #       int64_t num_heads, num_kv_heads, stride_kn, stride_vn)
    rt.flash_attn_decode_bf16.argtypes = (
        [ctypes.c_void_p] * 4
        + [ctypes.c_int64, ctypes.c_int64, ctypes.c_float]
        + [ctypes.c_int64] * 4
    )
    rt.flash_attn_decode_bf16.restype = None
    _rt = rt
    return _rt


def flash_attn_decode_bf16(q, k, v, out, seq_len, head_dim, sm_scale,
                           n_heads_q, n_heads_kv, stride_k, stride_v):
    """M=1 Flash Attention decode. BF16 inputs, online softmax, GQA.

    Args:
        q:          [Hq, D] bf16
        k, v:       [Hkv, seq_len, D] bf16
        out:        [Hq, D] bf16 (written in-place)
        seq_len:    int, length of k/v
        head_dim:   int, D
        sm_scale:   float, 1/sqrt(D) typically
        n_heads_q:  int, Hq
        n_heads_kv: int, Hkv (GQA: Hq >= Hkv, group size = Hq / Hkv)
        stride_k:   int, k's seq-dim stride in elements
        stride_v:   int, v's seq-dim stride in elements
    """
    _get_rt().flash_attn_decode_bf16(
        q.data_ptr(), k.data_ptr(), v.data_ptr(), out.data_ptr(),
        seq_len, head_dim, sm_scale,
        n_heads_q, n_heads_kv, stride_k, stride_v,
    )