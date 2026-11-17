# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Numerical tests for the TLE named micro-kernels (route C).

Each tle_ops builtin is a thin cpu.create_raw_call wrapper over a C
micro-kernel compiled into libTritonCPURuntime. These tests drive every one
of them through a @triton.jit kernel (plus the ctypes runtime veneer for
flash_attn_decode) and compare against plain PyTorch float32 references.

The C kernels round bf16 results by truncation (vshrn, no RNE) and compute
silu/rms in fp32, so bf16 comparisons use a 1-2 ULP tolerance instead of
exact equality. The INT8 SDOT path is exact integer arithmetic and is
compared with torch.equal.
"""
import math
import platform

import pytest
import torch
import torch.nn.functional as F

import triton
import triton.language as tl
from triton.language.extra.cpu import runtime, tle_ops

IS_ARM64 = platform.machine() in ("aarch64", "arm64")

pytestmark = pytest.mark.skipif(not IS_ARM64, reason="TLE micro-kernels are ARM64-only")


# ---------------------------------------------------------------------------
# @triton.jit harnesses: one trivial kernel per builtin. Size arguments are
# tl.constexpr because the builtins marshal them as compile-time i64 scalars.
# ---------------------------------------------------------------------------


@triton.jit
def _k_pack_weights(b_ptr, packed_ptr, K: tl.constexpr, N: tl.constexpr):
    tle_ops.sdot_pack_weights(b_ptr, packed_ptr, K, N)


@triton.jit
def _k_sdot_gemv(a_ptr, packed_ptr, c_ptr, K: tl.constexpr, N: tl.constexpr):
    tle_ops.sdot_gemv(a_ptr, packed_ptr, c_ptr, K, N)


@triton.jit
def _k_sdot_gemv_fused(x_ptr, packed_ptr, w_scale_ptr, out_ptr,
                       K: tl.constexpr, N: tl.constexpr):
    tle_ops.sdot_gemv_fused_bf16(x_ptr, packed_ptr, w_scale_ptr, out_ptr, K, N)


@triton.jit
def _k_fused_mlp(x_ptr, gate_packed_ptr, up_packed_ptr, gate_scale_ptr,
                 up_scale_ptr, out_ptr, K: tl.constexpr, N: tl.constexpr):
    tle_ops.fused_mlp(x_ptr, gate_packed_ptr, up_packed_ptr, gate_scale_ptr,
                      up_scale_ptr, out_ptr, K, N)


@triton.jit
def _k_swiglu(gate_ptr, up_ptr, out_ptr, N: tl.constexpr):
    tle_ops.swiglu(gate_ptr, up_ptr, out_ptr, N)


@triton.jit
def _k_rms_norm(x_ptr, w_ptr, out_ptr, D: tl.constexpr):
    tle_ops.rms_norm(x_ptr, w_ptr, out_ptr, D, 1e-5)


@triton.jit
def _k_rms_norm_gated(x_ptr, gate_ptr, w_ptr, out_ptr,
                      M: tl.constexpr, D: tl.constexpr):
    tle_ops.rms_norm_gated(x_ptr, gate_ptr, w_ptr, out_ptr, M, D, 1e-5)


@triton.jit
def _k_flash_attn_decode(q_ptr, k_ptr, v_ptr, out_ptr,
                         T: tl.constexpr, D: tl.constexpr,
                         Hq: tl.constexpr, Hkv: tl.constexpr,
                         stride_kn: tl.constexpr, stride_vn: tl.constexpr):
    tle_ops.flash_attn_decode(q_ptr, k_ptr, v_ptr, out_ptr, T, D,
                              1.0 / math.sqrt(D), Hq, Hkv,
                              stride_kn, stride_vn)


@triton.jit
def _k_gated_delta(q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, state_ptr, out_ptr,
                   B: tl.constexpr, H: tl.constexpr,
                   k_dim: tl.constexpr, v_dim: tl.constexpr,
                   use_l2norm: tl.constexpr):
    tle_ops.gated_delta_decode(q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr,
                               state_ptr, out_ptr, B, H, k_dim, v_dim,
                               use_l2norm)


@triton.jit
def _k_causal_conv1d(hidden_ptr, state_ptr, w_ptr, bias_ptr, out_ptr,
                     B: tl.constexpr, C: tl.constexpr):
    tle_ops.causal_conv1d_update(hidden_ptr, state_ptr, w_ptr, bias_ptr,
                                 out_ptr, B, C, 4, 1, 1)


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------


def _pack_ref(b):
    """[K, N] int8 -> flat blocks of (kb, nb, ni, ki), 16 bytes each."""
    K, N = b.shape
    K4, N4 = K // 4, N // 4
    out = torch.zeros(K4, N4, 4, 4, dtype=torch.int8)
    for kb in range(K4):
        for nb in range(N4):
            for ni in range(4):
                for ki in range(4):
                    out[kb, nb, ni, ki] = b[kb * 4 + ki, nb * 4 + ni]
    return out.reshape(-1)


def _quantize_int8(x_f32, amax_floor):
    """absmax symmetric quantization matching the C kernels (RNE, clamp)."""
    amax = x_f32.abs().max().item()
    amax = max(amax, amax_floor)
    inv = 127.0 / amax
    q = torch.round(x_f32 * inv).clamp(-128, 127).to(torch.int8)
    return q, amax / 127.0


def _sdpa_ref(q, k, v, scale, Hq, Hkv):
    """Single-token decode attention with GQA, fp32 reference."""
    ref = torch.zeros(Hq, q.shape[1])
    for h in range(Hq):
        kvh = h // (Hq // Hkv)
        s = (k[kvh].float() @ q[h].float()) * scale
        p = torch.softmax(s, dim=-1)
        ref[h] = p @ v[kvh].float()
    return ref


def _gated_delta_ref(q, k, v, g, beta, state, use_l2norm):
    B, H, k_dim = q.shape
    v_dim = v.shape[2]
    scale = 1.0 / math.sqrt(k_dim)
    out = torch.zeros(B, H, v_dim)
    st = state.clone()
    for b in range(B):
        for h in range(H):
            qh, kh = q[b, h].clone(), k[b, h].clone()
            if use_l2norm:
                qh = qh / torch.sqrt((qh * qh).sum() + 1e-6)
                kh = kh / torch.sqrt((kh * kh).sum() + 1e-6)
            qh = qh * scale
            S = st[b, h]
            S = S * math.exp(g[b, h].item())
            kv = kh @ S
            delta = (v[b, h] - kv) * beta[b, h]
            S = S + torch.outer(kh, delta)
            out[b, h] = qh @ S
            st[b, h] = S
    return out, st


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

BF16_TOL = dict(rtol=1 / 32, atol=1e-2)


def test_tle_ops_exports():
    names = [n for n in dir(tle_ops) if not n.startswith("_")]
    expected = [
        "sdot_pack_weights", "sdot_gemv", "sdot_gemv_fused_bf16",
        "fused_mlp", "swiglu", "rms_norm", "rms_norm_gated",
        "flash_attn_decode", "gated_delta_decode", "causal_conv1d_update",
    ]
    for e in expected:
        assert e in names, f"tle_ops is missing {e}"


def test_sdot_pack_weights():
    K, N = 64, 48
    b = torch.randint(-127, 128, (K, N), dtype=torch.int8)
    packed = torch.zeros(K // 4 * N // 4 * 16, dtype=torch.int8)
    _k_pack_weights[(1,)](b, packed, K=K, N=N)
    assert torch.equal(packed, _pack_ref(b))


def test_sdot_gemv_exact():
    K, N = 256, 128
    a = torch.randint(-127, 128, (K,), dtype=torch.int8)
    b = torch.randint(-127, 128, (K, N), dtype=torch.int8)
    packed = _pack_ref(b)
    c = torch.zeros(N, dtype=torch.int32)
    _k_sdot_gemv[(1,)](a, packed, c, K=K, N=N)
    ref = (a.long().unsqueeze(0) @ b.long()).squeeze(0).to(torch.int32)
    assert torch.equal(c, ref)


def test_sdot_gemv_fused_bf16():
    K, N = 128, 64
    x = torch.randn(K, dtype=torch.bfloat16)
    b = torch.randint(-127, 128, (K, N), dtype=torch.int8)
    w_scale = torch.rand(N, dtype=torch.float32) * 0.05 + 1e-3
    packed = _pack_ref(b)
    out = torch.zeros(N, dtype=torch.bfloat16)
    _k_sdot_gemv_fused[(1,)](x, packed, w_scale, out, K=K, N=N)

    xq, x_scale = _quantize_int8(x.float(), 1e-8)
    acc = xq.long().unsqueeze(0) @ b.long()
    ref = (acc.float() * x_scale * w_scale).squeeze(0)
    assert not torch.all(out == 0), "output is all zero (empty stub?)"
    assert torch.allclose(out.float(), ref, **BF16_TOL)


def test_fused_mlp():
    K, N = 128, 64
    x = torch.randn(K, dtype=torch.bfloat16)
    gate_w = torch.randint(-127, 128, (K, N), dtype=torch.int8)
    up_w = torch.randint(-127, 128, (K, N), dtype=torch.int8)
    gate_scale = torch.rand(N, dtype=torch.float32) * 0.05 + 1e-3
    up_scale = torch.rand(N, dtype=torch.float32) * 0.05 + 1e-3
    gate_packed, up_packed = _pack_ref(gate_w), _pack_ref(up_w)
    out = torch.zeros(N, dtype=torch.bfloat16)
    _k_fused_mlp[(1,)](x, gate_packed, up_packed, gate_scale, up_scale,
                       out, K=K, N=N)

    xq, x_scale = _quantize_int8(x.float(), 1e-10)
    gate = (xq.long().unsqueeze(0) @ gate_w.long()).float() * x_scale * gate_scale
    up = (xq.long().unsqueeze(0) @ up_w.long()).float() * x_scale * up_scale
    ref = (F.silu(gate) * up).squeeze(0)
    assert not torch.all(out == 0), "output is all zero (empty stub?)"
    assert torch.allclose(out.float(), ref, **BF16_TOL)


def test_swiglu():
    N = 1024
    gate = torch.randn(N, dtype=torch.bfloat16)
    up = torch.randn(N, dtype=torch.bfloat16)
    out = torch.zeros(N, dtype=torch.bfloat16)
    _k_swiglu[(1,)](gate, up, out, N=N)
    ref = F.silu(gate.float()) * up.float()
    assert not torch.all(out == 0), "output is all zero (empty stub?)"
    assert torch.allclose(out.float(), ref, **BF16_TOL)


def test_rms_norm():
    D = 128
    x = torch.randn(D, dtype=torch.bfloat16)
    w = torch.randn(D, dtype=torch.bfloat16)
    out = torch.zeros(D, dtype=torch.bfloat16)
    _k_rms_norm[(1,)](x, w, out, D=D)
    xf, wf = x.float(), w.float()
    ref = xf * torch.rsqrt(xf.pow(2).mean() + 1e-5) * wf
    assert not torch.all(out == 0), "output is all zero (empty stub?)"
    assert torch.allclose(out.float(), ref, **BF16_TOL)


def test_rms_norm_gated():
    M, D = 8, 128
    x = torch.randn(M, D, dtype=torch.bfloat16)
    gate = torch.randn(M, D, dtype=torch.bfloat16)
    w = torch.randn(D, dtype=torch.bfloat16)
    out = torch.zeros(M, D, dtype=torch.bfloat16)
    _k_rms_norm_gated[(1,)](x, gate, w, out, M=M, D=D)
    xf, wf, gf = x.float(), w.float(), gate.float()
    xn = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-5)
    ref = xn * wf * F.silu(gf)
    assert not torch.all(out == 0), "output is all zero (empty stub?)"
    assert torch.allclose(out.float(), ref, **BF16_TOL)


def _run_flash_decode(q, k, v, out, T, D, Hq, Hkv):
    _k_flash_attn_decode[(1,)](q, k, v, out, T=T, D=D, Hq=Hq, Hkv=Hkv,
                               stride_kn=k.stride(1), stride_vn=v.stride(1))


def test_flash_attn_decode_jit():
    Hq, Hkv, D, T = 8, 2, 64, 32
    q = torch.randn(Hq, D, dtype=torch.bfloat16)
    k = torch.randn(Hkv, T, D, dtype=torch.bfloat16)
    v = torch.randn(Hkv, T, D, dtype=torch.bfloat16)
    out = torch.zeros(Hq, D, dtype=torch.bfloat16)
    _run_flash_decode(q, k, v, out, T, D, Hq, Hkv)
    ref = _sdpa_ref(q, k, v, 1.0 / math.sqrt(D), Hq, Hkv)
    assert not torch.all(out == 0), "output is all zero (empty stub?)"
    assert torch.allclose(out.float(), ref, rtol=1 / 16, atol=2e-2)


def test_flash_attn_decode_runtime():
    Hq, Hkv, D, T = 8, 2, 64, 32
    q = torch.randn(Hq, D, dtype=torch.bfloat16)
    k = torch.randn(Hkv, T, D, dtype=torch.bfloat16)
    v = torch.randn(Hkv, T, D, dtype=torch.bfloat16)
    out = torch.zeros(Hq, D, dtype=torch.bfloat16)
    runtime.flash_attn_decode_bf16(q, k, v, out, T, D, 1.0 / math.sqrt(D),
                                   Hq, Hkv, k.stride(1), v.stride(1))
    ref = _sdpa_ref(q, k, v, 1.0 / math.sqrt(D), Hq, Hkv)
    assert not torch.all(out == 0), "output is all zero (empty stub?)"
    assert torch.allclose(out.float(), ref, rtol=1 / 16, atol=2e-2)


@pytest.mark.parametrize("use_l2norm", [0, 1])
def test_gated_delta_decode(use_l2norm):
    B, H, k_dim, v_dim = 2, 4, 16, 16
    q = torch.randn(B, H, k_dim)
    k = torch.randn(B, H, k_dim)
    v = torch.randn(B, H, v_dim)
    g = torch.randn(B, H) * 0.1
    beta = torch.rand(B, H)
    state = torch.randn(B, H, k_dim, v_dim)
    state_in = state.clone()
    out = torch.zeros(B, H, v_dim)
    _k_gated_delta[(1,)](q, k, v, g, beta, state, out, B=B, H=H,
                         k_dim=k_dim, v_dim=v_dim, use_l2norm=use_l2norm)
    out_ref, state_ref = _gated_delta_ref(q, k, v, g, beta, state_in,
                                          use_l2norm)
    assert not torch.all(out == 0), "output is all zero (empty stub?)"
    assert torch.allclose(out, out_ref, rtol=1e-3, atol=1e-4)
    assert torch.allclose(state, state_ref, rtol=1e-3, atol=1e-4)


def test_causal_conv1d_update():
    B, C = 4, 64
    hidden = torch.randn(B, C, dtype=torch.bfloat16)
    state = torch.randn(B, C, 4, dtype=torch.bfloat16)
    weight = torch.randn(C, 4, dtype=torch.bfloat16)
    bias = torch.randn(C, dtype=torch.bfloat16)
    state_in = state.clone()
    out = torch.zeros(B, C, dtype=torch.bfloat16)
    _k_causal_conv1d[(1,)](hidden, state, weight, bias, out, B=B, C=C)

    # Conv reads state[1..3] + hidden (not state[0..2] + hidden):
    # y = s1*w0 + s2*w1 + s3*w2 + h*w3 (+ bias), then silu, bf16 out.
    buf = torch.cat([state_in[:, :, 1:], hidden.unsqueeze(2)], dim=2)
    y = (buf.float() * weight.float().unsqueeze(0)).sum(-1)
    y = y + bias.float().unsqueeze(0)
    ref = F.silu(y)
    assert not torch.all(out == 0), "output is all zero (empty stub?)"
    assert torch.allclose(out.float(), ref, rtol=1 / 16, atol=1e-2)
    # State rolls forward by one: [s1, s2, s3, h].
    expected_state = torch.cat([state_in[:, :, 1:], hidden.unsqueeze(2)], dim=2)
    assert torch.equal(state, expected_state)
