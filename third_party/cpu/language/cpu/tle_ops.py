"""
TLE-CPU: ARM NEON/SVE2 intrinsics as Triton language builtins.

Usage in @triton.jit kernels:
    from triton.language.extra.cpu import tle_ops as tle_cpu

    @triton.jit
    def my_kernel(...):
        acc = tl.zeros([4], dtype=tl.int32)
        a_bc = ...  # [16] int8, A broadcast
        b_pk = ...  # [16] int8, B pre-packed
        acc = tle_cpu.sdot(acc, a_bc, b_pk)
"""

import triton.language as tl
from triton.language.core import builtin, tensor, _unwrap_if_constexpr


@builtin
def sdot_gemv(a_ptr, b_packed_ptr, c_ptr, K, N, _builder=None):
    """TLE-CPU: M=1 INT8 GEMV micro-kernel using NEON SDOT with pre-packed weights.

    A complete micro-kernel that performs:
      C[N] = A[K] (int8) @ B_packed[K//4, N//4, 4, 4] (int8) → C[N] (int32)

    Uses K-outer loop with NEON SDOT, OMP parallelized across N.
    Calls sdot_gemv_m1_prepacked() from libTritonCPURuntime.so.

    Args:
        a_ptr: pointer to [K] int8 activation
        b_packed_ptr: pointer to pre-packed int8 weights (SDOT format)
        c_ptr: pointer to [N] int32 output
        K: activation/weight inner dimension
        N: output dimension
    """
    K_raw = _unwrap_if_constexpr(K)
    N_raw = _unwrap_if_constexpr(N)
    K_val = K_raw.handle if hasattr(K_raw, 'handle') else _builder.get_int64(K_raw)
    N_val = N_raw.handle if hasattr(N_raw, 'handle') else _builder.get_int64(N_raw)
    _builder.create_cpu_sdot_gemv(
        a_ptr.handle, b_packed_ptr.handle, c_ptr.handle, K_val, N_val)
    # Return None (void op)
    return None


@builtin
def fused_decode_step(
    token_id, pos,
    embed_table_ptr, layer_ptrs_ptr,
    k_cache_ptr, v_cache_ptr,
    rope_cos_ptr, rope_sin_ptr,
    final_norm_ptr,
    lm_head_packed_ptr, lm_head_scale_ptr,
    hidden_dim, head_dim, n_heads, n_kv_heads,
    intermediate, vocab_size, n_layers, max_seq,
    rms_eps,
    _builder=None):
    """TLE-CPU: Full decode step. Returns next token ID (i64).

    embedding → n_layers × transformer layer → final norm → lm_head → argmax.
    One Triton kernel launch per token.
    """
    def _i64(v):
        raw = _unwrap_if_constexpr(v)
        if hasattr(raw, 'handle'):
            handle = raw.handle
            i64_ty = _builder.get_int64_ty()
            try:
                handle = _builder.create_int_cast(handle, i64_ty, True)
            except Exception:
                pass
            return handle
        return _builder.get_int64(raw)

    rms_f = float(_unwrap_if_constexpr(rms_eps))

    result = _builder.create_cpu_fused_decode_step(
        _i64(token_id), _i64(pos),
        embed_table_ptr.handle, layer_ptrs_ptr.handle,
        k_cache_ptr.handle, v_cache_ptr.handle,
        rope_cos_ptr.handle, rope_sin_ptr.handle,
        final_norm_ptr.handle,
        lm_head_packed_ptr.handle, lm_head_scale_ptr.handle,
        _i64(hidden_dim), _i64(head_dim), _i64(n_heads), _i64(n_kv_heads),
        _i64(intermediate), _i64(vocab_size), _i64(n_layers), _i64(max_seq),
        rms_f)
    return tensor(result, tl.int64)


@builtin
def fused_transformer_layer(
    hidden_ptr,
    wq_ptr, wk_ptr, wv_ptr, wo_ptr,
    wq_s_ptr, wk_s_ptr, wv_s_ptr, wo_s_ptr,
    q_norm_ptr, k_norm_ptr,
    cos_ptr, sin_ptr,
    k_cache_ptr, v_cache_ptr,
    cache_pos, max_seq_len,
    gate_ptr, up_ptr, down_ptr,
    gate_s_ptr, up_s_ptr, down_s_ptr,
    input_norm_ptr, post_norm_ptr,
    hidden_dim, head_dim, n_heads, n_kv_heads, intermediate,
    rms_eps,
    _builder=None):
    """TLE-CPU: Full transformer decode layer in one C call.

    RMSNorm → QKV GEMV → QK_Norm → RoPE → KV_cache → Attention →
    O_GEMV → Residual → RMSNorm → Gate+Up+SWIGLU → Down_GEMV → Residual.

    Zero tensor allocation. Zero Python dispatch within layer.
    """
    def _i64(v):
        raw = _unwrap_if_constexpr(v)
        if hasattr(raw, 'handle'):
            # Always cast to i64 to handle i32 kernel args
            handle = raw.handle
            i64_ty = _builder.get_int64_ty()
            try:
                handle = _builder.create_int_cast(handle, i64_ty, True)
            except Exception:
                pass
            return handle
        return _builder.get_int64(raw)

    rms_f = float(_unwrap_if_constexpr(rms_eps))

    _builder.create_cpu_fused_transformer_layer(
        hidden_ptr.handle,
        wq_ptr.handle, wk_ptr.handle, wv_ptr.handle, wo_ptr.handle,
        wq_s_ptr.handle, wk_s_ptr.handle, wv_s_ptr.handle, wo_s_ptr.handle,
        q_norm_ptr.handle, k_norm_ptr.handle,
        cos_ptr.handle, sin_ptr.handle,
        k_cache_ptr.handle, v_cache_ptr.handle,
        _i64(cache_pos), _i64(max_seq_len),
        gate_ptr.handle, up_ptr.handle, down_ptr.handle,
        gate_s_ptr.handle, up_s_ptr.handle, down_s_ptr.handle,
        input_norm_ptr.handle, post_norm_ptr.handle,
        _i64(hidden_dim), _i64(head_dim), _i64(n_heads), _i64(n_kv_heads),
        _i64(intermediate), rms_f)
    return None


@builtin
def fused_mlp(x_ptr, gate_packed_ptr, up_packed_ptr,
               gate_scale_ptr, up_scale_ptr, out_ptr, K, N, _builder=None):
    """TLE-CPU: Fused MLP = gate SDOT GEMV + up SDOT GEMV + SWIGLU.

    Single OMP region replaces 3 separate ops (gate_proj, up_proj, silu_and_mul).

    Args:
        x_ptr: [K] bf16 activation
        gate_packed_ptr, up_packed_ptr: [K/4, N/4, 16] int8 SDOT-packed weights
        gate_scale_ptr, up_scale_ptr: [N] fp32 per-channel weight scales
        out_ptr: [N] bf16 output
        K, N: dimensions
    """
    K_raw = _unwrap_if_constexpr(K)
    N_raw = _unwrap_if_constexpr(N)
    K_val = K_raw.handle if hasattr(K_raw, 'handle') else _builder.get_int64(K_raw)
    N_val = N_raw.handle if hasattr(N_raw, 'handle') else _builder.get_int64(N_raw)
    _builder.create_cpu_fused_mlp(
        x_ptr.handle, gate_packed_ptr.handle, up_packed_ptr.handle,
        gate_scale_ptr.handle, up_scale_ptr.handle, out_ptr.handle,
        K_val, N_val)
    return None


@builtin
def flash_attn_decode(q_ptr, k_ptr, v_ptr, out_ptr,
                       seq_len, head_dim, sm_scale,
                       num_heads, num_kv_heads,
                       stride_kn, stride_vn, _builder=None):
    """TLE-CPU: M=1 Flash Attention with NEON online softmax.

    Replaces ATen SDPA fallback for decode (M=1). Per-row online softmax,
    NEON dot product for Q·K^T, OMP parallelized across heads.

    Args:
        q_ptr: [num_heads, head_dim] bf16
        k_ptr: [num_kv_heads, seq_len, head_dim] bf16
        v_ptr: [num_kv_heads, seq_len, head_dim] bf16
        out_ptr: [num_heads, head_dim] bf16
        seq_len, head_dim: dimensions
        sm_scale: softmax scale (typically head_dim^-0.5)
        num_heads, num_kv_heads: head counts (GQA support)
        stride_kn, stride_vn: strides for K,V along seq_len dim
    """
    def _i64(v):
        # Runtime int args (e.g. decode seq_len, which varies per token and so
        # cannot be a constexpr) come in as i32 handles; cast to i64 to match the
        # op's I64 operands. constexpr ints go straight through get_int64.
        raw = _unwrap_if_constexpr(v)
        if hasattr(raw, 'handle'):
            handle = raw.handle
            i64_ty = _builder.get_int64_ty()
            try:
                handle = _builder.create_int_cast(handle, i64_ty, True)
            except Exception:
                pass
            return handle
        return _builder.get_int64(raw)

    vals = {}
    for name, v in [('seq_len', seq_len), ('head_dim', head_dim),
                     ('num_heads', num_heads), ('num_kv_heads', num_kv_heads),
                     ('stride_kn', stride_kn), ('stride_vn', stride_vn)]:
        vals[name] = _i64(v)
    sm_scale_val = _unwrap_if_constexpr(sm_scale)
    if hasattr(sm_scale_val, 'handle'):
        sm_f = 0.0  # will be set from handle
    else:
        sm_f = float(sm_scale_val)
    _builder.create_cpu_flash_attn_decode(
        q_ptr.handle, k_ptr.handle, v_ptr.handle, out_ptr.handle,
        vals['seq_len'], vals['head_dim'], sm_f,
        vals['num_heads'], vals['num_kv_heads'],
        vals['stride_kn'], vals['stride_vn'])
    return None


@builtin
def rms_norm(x_ptr, weight_ptr, out_ptr, D, eps, _builder=None):
    """TLE-CPU: RMSNorm — out = (x / rms(x)) * weight.

    Single NEON kernel replaces 5 ATen decomposed ops.
    BF16 input/output, single-threaded (optimal for decode D <= 4096).

    Args:
        x_ptr: pointer to [D] bfloat16 input
        weight_ptr: pointer to [D] bfloat16 weight
        out_ptr: pointer to [D] bfloat16 output
        D: hidden dimension
        eps: epsilon for numerical stability
    """
    D_raw = _unwrap_if_constexpr(D)
    D_val = D_raw.handle if hasattr(D_raw, 'handle') else _builder.get_int64(D_raw)
    eps_f = float(_unwrap_if_constexpr(eps))
    _builder.create_cpu_rms_norm(x_ptr.handle, weight_ptr.handle, out_ptr.handle, D_val, eps_f)
    return None


@builtin
def swiglu(gate_ptr, up_ptr, out_ptr, N, _builder=None):
    """TLE-CPU: Fused SWIGLU activation: out = silu(gate) * up.

    Single NEON kernel replaces F.silu(gate) * up (2 ATen calls).
    BF16 input/output. Single-threaded (optimal for decode N <= 6144).

    Args:
        gate_ptr: pointer to [N] bfloat16 gate values
        up_ptr: pointer to [N] bfloat16 up values
        out_ptr: pointer to [N] bfloat16 output
        N: number of elements
    """
    N_raw = _unwrap_if_constexpr(N)
    N_val = N_raw.handle if hasattr(N_raw, 'handle') else _builder.get_int64(N_raw)
    _builder.create_cpu_swiglu(gate_ptr.handle, up_ptr.handle, out_ptr.handle, N_val)
    return None


@builtin
def sdot_gemv_fused_bf16(x_ptr, b_packed_ptr, w_scale_ptr, out_ptr, K, N, _builder=None):
    """TLE-CPU: Fused BF16→INT8 quant + SDOT GEMV + dequant→BF16.

    Single call replaces: abs().max() → div → clamp → to(int8) → gemv → mul(scale) → to(bf16)

    Args:
        x_ptr: pointer to [K] bfloat16 activation
        b_packed_ptr: pointer to pre-packed int8 weights (SDOT format)
        w_scale_ptr: pointer to [N] float32 per-channel weight scale
        out_ptr: pointer to [N] bfloat16 output
        K, N: dimensions
    """
    K_raw = _unwrap_if_constexpr(K)
    N_raw = _unwrap_if_constexpr(N)
    K_val = K_raw.handle if hasattr(K_raw, 'handle') else _builder.get_int64(K_raw)
    N_val = N_raw.handle if hasattr(N_raw, 'handle') else _builder.get_int64(N_raw)
    _builder.create_cpu_sdot_gemv_fused_bf16(
        x_ptr.handle, b_packed_ptr.handle, w_scale_ptr.handle,
        out_ptr.handle, K_val, N_val)
    return None


@builtin
def sdot_pack_weights(b_ptr, b_packed_ptr, K, N, _builder=None):
    """TLE-CPU: Pack INT8 weights from row-major [K,N] to SDOT format [K//4, N//4, 4, 4].

    Args:
        b_ptr: pointer to [K, N] int8 weights (row-major)
        b_packed_ptr: pointer to output buffer (pre-allocated)
        K, N: dimensions
    """
    K_raw = _unwrap_if_constexpr(K)
    N_raw = _unwrap_if_constexpr(N)
    K_val = K_raw.handle if hasattr(K_raw, 'handle') else _builder.get_int64(K_raw)
    N_val = N_raw.handle if hasattr(N_raw, 'handle') else _builder.get_int64(N_raw)
    _builder.create_cpu_sdot_pack_weights(
        b_ptr.handle, b_packed_ptr.handle, K_val, N_val)
    return None


@builtin
def sme_gemm(ap_ptr, bp_ptr, c_ptr, Mp, Np, K4, _builder=None):
    """TLE-CPU: INT8 GEMM via ARM SME (SMOPA outer-products).

    C[Mp][Np] int32 = Ap @ Bp^T, computed on the M4 SME unit in 16x64 SMOPA tiles.
      ap_ptr: [Mp//16, K//4, 16, 4] packed activation panels (int8)
      bp_ptr: [Np//64, K//4, 4, 16, 4] packed weight panels (int8)
      c_ptr:  [Mp, Np] int32 output
    Calls sme_gemm_int32() in libTritonCPURuntime.
    """
    def _i64(v):
        r = _unwrap_if_constexpr(v)
        return r.handle if hasattr(r, 'handle') else _builder.get_int64(r)
    _builder.create_cpu_sme_gemm(
        ap_ptr.handle, bp_ptr.handle, c_ptr.handle, _i64(Mp), _i64(Np), _i64(K4))
    return None


@builtin
def smmla_uk(ap_ptr, wp_ptr, c_ptr, xs_ptr, ws_ptr, K8, MP, N, mp0, np0, _builder=None):
    """TLE-Struct micro-kernel: one fixed 8x8 SMMLA output tile (the Raw leaf).

    The M/N tiling is orchestrated by the surrounding @triton.jit kernel (the
    Struct layer); this op computes ONE 8x8 register-blocked tile at (mp0, np0).
    Calls smmla_uk() in libTritonCPURuntime.
    """
    def _i64(v):
        r = _unwrap_if_constexpr(v)
        return r.handle if hasattr(r, 'handle') else _builder.get_int64(r)
    _builder.create_cpu_smmla_uk(
        ap_ptr.handle, wp_ptr.handle, c_ptr.handle, xs_ptr.handle, ws_ptr.handle,
        _i64(K8), _i64(MP), _i64(N), _i64(mp0), _i64(np0))
    return None


def _i64(v, _builder):
    r = _unwrap_if_constexpr(v)
    return r.handle if hasattr(r, 'handle') else _builder.get_int64(r)


@builtin
def sme_uk(ap, bp, c, K4, Np, mt, nt, _builder=None):
    """TLE-Struct micro-kernel: one 16x64 SME GEMM tile."""
    _builder.create_cpu_sme_uk(ap.handle, bp.handle, c.handle,
                               _i64(K4,_builder), _i64(Np,_builder), _i64(mt,_builder), _i64(nt,_builder))
    return None


@builtin
def sdot_gemv_uk(a, b, c, K4, N4, BN4, blk, _builder=None):
    """TLE-Struct micro-kernel: one block of the decode int8 GEMV."""
    _builder.create_cpu_sdot_gemv_uk(a.handle, b.handle, c.handle,
                                     _i64(K4,_builder), _i64(N4,_builder), _i64(BN4,_builder), _i64(blk,_builder))
    return None


@builtin
def swiglu_uk(gate, up, out, off, n, _builder=None):
    """TLE-Struct micro-kernel: SwiGLU on one block."""
    _builder.create_cpu_swiglu_uk(gate.handle, up.handle, out.handle, _i64(off,_builder), _i64(n,_builder))
    return None


@builtin
def rmsnorm_uk(x, weight, out, D, row, _builder=None):
    """TLE-Struct micro-kernel: RMSNorm on one row."""
    _builder.create_cpu_rmsnorm_uk(x.handle, weight.handle, out.handle, _i64(D,_builder), _i64(row,_builder))
    return None


@builtin
def residual_uk(residual, x, off, n, _builder=None):
    """TLE-Struct micro-kernel: residual add on one block."""
    _builder.create_cpu_residual_uk(residual.handle, x.handle, _i64(off,_builder), _i64(n,_builder))
    return None


@builtin
def sdot(acc, a, b, _builder=None):
    """NEON SDOT: 4-lane signed int8 dot product accumulate.

    acc: tensor([4], int32)  — accumulator
    a:   tensor([16], int8)  — first operand (typically A broadcast)
    b:   tensor([16], int8)  — second operand (typically B pre-packed)

    Returns: tensor([4], int32)

    Each lane computes:
        result[i] = acc[i] + sum_{k=0}^{3}(a[i*4+k] * b[i*4+k])

    Lowered to: llvm.aarch64.neon.sdot.v4i32.v16i8 via TTC_NeonSdotOp.
    """
    return tensor(
        _builder.create_cpu_neon_sdot(acc.handle, a.handle, b.handle),
        acc.type,
    )
