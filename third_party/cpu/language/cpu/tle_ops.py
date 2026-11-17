# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""CPU TLE native leaves, reusing the existing ARM runtime-call architecture.

This scalar C ABI entry extends this CPU backend's native-call support.
It does not implement GPU TLE source-region APIs. The caller loads the native
operator library before the compiled kernel is loaded.
"""
from triton._C.libtriton import cpu
from triton.language import core

if not hasattr(cpu, "create_raw_call"):
    raise ImportError("Rebuild the CPU compiler with the TLE native-call binding first")


@core.builtin
def raw_call(symbol, *args, result_dtype=core.int32, _semantic=None):
    """Call a scalar C ABI leaf with conservative buffer effects.

    Pass pointer addresses as uint64 on 64-bit hosts. Values retain their
    explicit integer/float widths. Set result_dtype=None for a void function.
    This call is retained when its result is unused.

    Kernels used as native single-program leaves may request the CPU compiler
    option ``enable_single_program_entry=True``. The additional symbol in
    ``compiled.metadata.single_program_name`` accepts only kernel arguments;
    it fixes program IDs to zero and all grid dimensions to one. The ordinary
    compiled entry and Triton launch interface keep their full grid ABI.

    With ``single_program_return_type="i32"`` (also i64/f32/f64), the additional
    C entry omits the kernel's final output pointer and returns that scalar.
    After inlining, the kernel may only store matching scalar values directly
    through that pointer. Pointer arithmetic, loads, tensor stores and escape
    are rejected before lowering. Its temporary
    storage starts at zero. The ordinary launcher still takes the original
    output pointer. LLVM can promote the temporary to an SSA return value,
    avoiding a status-buffer store/load around a scalar native call.

    CPU ``extern_libs={name: bitcode_path}`` can link leaves from LLVM bitcode
    before optimization. Use a compatible LLVM toolchain, host architecture,
    target features and C ABI. Compile shared-object leaves with ``-fPIC``;
    CPU linking overrides PIE module flags because the JIT loads a DSO.
    Bitcode contents participate in the persistent compilation cache key.
    Keep files immutable while a prepared kernel is in use: the process-local
    JIT cache retains already compiled entries, like loaded shared libraries.
    """
    symbol = core._unwrap_if_constexpr(symbol)
    result_dtype = core._unwrap_if_constexpr(result_dtype)
    values = [_semantic.to_tensor(x) for x in args]
    if not isinstance(symbol, str) or not symbol:
        raise TypeError("CPU Raw requires a nonempty compile-time C ABI symbol")
    if any(x.type.is_block() or not (x.dtype.is_int() or x.dtype.is_floating()) for x in values):
        raise TypeError("CPU Raw native-call arguments must be scalar integers or floats")
    if result_dtype is not None and not (result_dtype.is_int() or result_dtype.is_floating()):
        raise TypeError("CPU Raw native-call result must be a scalar integer, float or None")
    handle = cpu.create_raw_call(_semantic.builder, symbol,
                                 [x.handle for x in values],
                                 None if result_dtype is None else result_dtype.to_ir(_semantic.builder))
    if result_dtype is None:
        return None
    return core.tensor(handle, result_dtype)


# --- TLE-CPU named micro-kernels (route C: raw_call over the ARM runtime) ---
# FlagGems calls these with K/N/M/D as tl.constexpr and runtime tensor pointers.
# C ABI: pointers pass as uint64 (aarch64 integer register), integer scalars as
# i64, eps/sm_scale as f32, all kernels return void. Symbols resolve through the
# compiler's existing -lTritonCPURuntime link.
import triton.language as tl


def _u64_ptr(ptr, _semantic):
    # tt.ptr -> i64 address (pointer passed in an integer register on aarch64).
    return _semantic.cast(ptr, tl.uint64).handle


def _i64(x, _semantic):
    return _semantic.builder.get_int64(int(core._unwrap_if_constexpr(x)))


def _f32(x, _semantic):
    t = _semantic.to_tensor(float(core._unwrap_if_constexpr(x)))
    return _semantic.cast(t, tl.float32).handle


def _raw_void(symbol, arg_handles, _semantic):
    # result=None -> void; same mechanism as raw_call(result_dtype=None).
    cpu.create_raw_call(_semantic.builder, symbol, arg_handles, None)


@core.builtin
def sdot_pack_weights(b_ptr, b_packed_ptr, K, N, _semantic=None):
    """TLE-CPU: pack INT8 weights [K,N] -> SDOT format [K//4,N//4,4,4]."""
    _raw_void("sdot_pack_weights",
              [_u64_ptr(b_ptr, _semantic), _u64_ptr(b_packed_ptr, _semantic),
               _i64(K, _semantic), _i64(N, _semantic)], _semantic)


@core.builtin
def sdot_gemv(a_ptr, b_packed_ptr, c_ptr, K, N, _semantic=None):
    """TLE-CPU: M=1 INT8 GEMV, NEON SDOT, pre-packed weights."""
    Nv = int(core._unwrap_if_constexpr(N))
    _raw_void("sdot_gemv_m1_prepacked",
              [_u64_ptr(a_ptr, _semantic), _u64_ptr(b_packed_ptr, _semantic),
               _u64_ptr(c_ptr, _semantic), _i64(K, _semantic), _i64(N, _semantic),
               _i64(Nv // 4, _semantic)], _semantic)


@core.builtin
def sdot_gemv_fused_bf16(x_ptr, b_packed_ptr, w_scale_ptr, out_ptr, K, N, _semantic=None):
    """TLE-CPU: fused BF16->INT8 quant + SDOT GEMV + dequant->BF16."""
    Nv = int(core._unwrap_if_constexpr(N))
    _raw_void("sdot_gemv_m1_fused_bf16",
              [_u64_ptr(x_ptr, _semantic), _u64_ptr(b_packed_ptr, _semantic),
               _u64_ptr(w_scale_ptr, _semantic), _u64_ptr(out_ptr, _semantic),
               _i64(K, _semantic), _i64(N, _semantic), _i64(Nv // 4, _semantic)], _semantic)


@core.builtin
def fused_mlp(x_ptr, gate_packed_ptr, up_packed_ptr, gate_scale_ptr, up_scale_ptr,
              out_ptr, K, N, _semantic=None):
    """TLE-CPU: fused SwiGLU MLP over pre-packed INT8 gate/up weights."""
    _raw_void("fused_mlp_bf16",
              [_u64_ptr(x_ptr, _semantic), _u64_ptr(gate_packed_ptr, _semantic),
               _u64_ptr(up_packed_ptr, _semantic), _u64_ptr(gate_scale_ptr, _semantic),
               _u64_ptr(up_scale_ptr, _semantic), _u64_ptr(out_ptr, _semantic),
               _i64(K, _semantic), _i64(N, _semantic)], _semantic)


@core.builtin
def swiglu(gate_ptr, up_ptr, out_ptr, N, _semantic=None):
    """TLE-CPU: SwiGLU activation silu(gate) * up over BF16 buffers."""
    _raw_void("swiglu_bf16",
              [_u64_ptr(gate_ptr, _semantic), _u64_ptr(up_ptr, _semantic),
               _u64_ptr(out_ptr, _semantic), _i64(N, _semantic)], _semantic)


@core.builtin
def rms_norm(x_ptr, weight_ptr, out_ptr, D, eps, _semantic=None):
    """TLE-CPU: standalone RMSNorm over a BF16 [D] vector."""
    _raw_void("standalone_rms_norm_bf16",
              [_u64_ptr(x_ptr, _semantic), _u64_ptr(weight_ptr, _semantic),
               _u64_ptr(out_ptr, _semantic), _i64(D, _semantic), _f32(eps, _semantic)],
              _semantic)


@core.builtin
def rms_norm_gated(x_ptr, gate_ptr, weight_ptr, out_ptr, M, D, eps, _semantic=None):
    """TLE-CPU: gated RMSNorm over BF16 [M,D] with a [D] weight."""
    _raw_void("standalone_rms_norm_gated_bf16",
              [_u64_ptr(x_ptr, _semantic), _u64_ptr(gate_ptr, _semantic),
               _u64_ptr(weight_ptr, _semantic), _u64_ptr(out_ptr, _semantic),
               _i64(M, _semantic), _i64(D, _semantic), _f32(eps, _semantic)], _semantic)


@core.builtin
def flash_attn_decode(q_ptr, k_ptr, v_ptr, out_ptr, seq_len, head_dim, sm_scale,
                      num_heads, num_kv_heads, stride_kn, stride_vn, _semantic=None):
    """TLE-CPU: single-token flash attention decode over BF16 Q/K/V."""
    _raw_void("flash_attn_decode_bf16",
              [_u64_ptr(q_ptr, _semantic), _u64_ptr(k_ptr, _semantic),
               _u64_ptr(v_ptr, _semantic), _u64_ptr(out_ptr, _semantic),
               _i64(seq_len, _semantic), _i64(head_dim, _semantic), _f32(sm_scale, _semantic),
               _i64(num_heads, _semantic), _i64(num_kv_heads, _semantic),
               _i64(stride_kn, _semantic), _i64(stride_vn, _semantic)], _semantic)


@core.builtin
def gated_delta_decode(q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, state_ptr, out_ptr,
                       B, H, k_dim, v_dim, use_l2norm, _semantic=None):
    """TLE-CPU: gated delta-rule decode step over FP32 buffers."""
    _raw_void("standalone_gated_delta_decode_fp32",
              [_u64_ptr(q_ptr, _semantic), _u64_ptr(k_ptr, _semantic),
               _u64_ptr(v_ptr, _semantic), _u64_ptr(g_ptr, _semantic),
               _u64_ptr(beta_ptr, _semantic), _u64_ptr(state_ptr, _semantic),
               _u64_ptr(out_ptr, _semantic), _i64(B, _semantic), _i64(H, _semantic),
               _i64(k_dim, _semantic), _i64(v_dim, _semantic), _i64(use_l2norm, _semantic)],
              _semantic)


@core.builtin
def causal_conv1d_update(hidden_ptr, state_ptr, weight_ptr, bias_ptr, out_ptr,
                         B, C, kernel_size, silu, has_bias, _semantic=None):
    """TLE-CPU: causal conv1d decode update over BF16 (kernel_size must be 4)."""
    _raw_void("standalone_causal_conv1d_update_bf16",
              [_u64_ptr(hidden_ptr, _semantic), _u64_ptr(state_ptr, _semantic),
               _u64_ptr(weight_ptr, _semantic), _u64_ptr(bias_ptr, _semantic),
               _u64_ptr(out_ptr, _semantic), _i64(B, _semantic), _i64(C, _semantic),
               _i64(kernel_size, _semantic), _i64(silu, _semantic), _i64(has_bias, _semantic)],
              _semantic)
