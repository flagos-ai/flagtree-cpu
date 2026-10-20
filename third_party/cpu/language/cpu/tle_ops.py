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
