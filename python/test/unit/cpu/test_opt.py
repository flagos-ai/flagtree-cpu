import os
import pytest
import torch

import triton
import triton.language as tl
"""
This test is only for the CPU backend.
"""


def is_interpreter():
    return os.environ.get('TRITON_INTERPRET', '0') == '1'


def is_cpu():
    return not is_interpreter() and \
        triton.runtime.driver.active.get_current_target().backend == "cpu"


def is_x86():
    return is_cpu() and \
        triton.runtime.driver.active.get_current_target().arch == "x86_64"


def test_scalar_pointer_arith(device):

    @triton.jit
    def kernel(src, dst, BLOCK_SIZE: tl.constexpr):
        offs = tl.arange(0, BLOCK_SIZE)
        x = tl.load(src + offs)
        tl.store(dst + offs, x)

    src = torch.rand((128, ), dtype=torch.float32, device=device)
    res = torch.empty_like(src)
    meta = kernel[(1, )](src, res, BLOCK_SIZE=128)
    assert (src == res).all()

    # Check TTCIR doesn't have pointer extraction from a tensor.
    ttcir = meta.asm["ttcir"]
    assert ttcir.count("extract") == 0


@pytest.mark.parametrize("size", [32, 128, 135])
@pytest.mark.parametrize("tile_size", [16])
def test_optimize_tile_mask(size, tile_size, device):
    pytest.xfail("masked vector ops no longer eliminated by upstream canonicalizers")

    @triton.jit
    def kernel(src, dst, size, TILE_SIZE: tl.constexpr):
        for i in range(0, tl.cdiv(size, TILE_SIZE)):
            offs = tl.arange(0, TILE_SIZE) + i * TILE_SIZE
            mask = offs < size
            x = tl.load(src + offs, mask=mask, other=0)
            tl.store(dst + offs, x, mask=mask)

    src = torch.rand((size, ), dtype=torch.float32, device='cpu')
    res = torch.empty_like(src)
    meta = kernel[(1, )](src, res, size, TILE_SIZE=tile_size)
    assert (src == res).all()

    # Check number of masked loads and stores.
    tttcir = meta.asm["tttcir"]
    masked_loads = tttcir.count("maskedload")
    masked_stores = tttcir.count("maskedstore")
    if size % tile_size == 0:
        assert masked_loads == 0
        assert masked_stores == 0
    else:
        assert masked_loads == 1
        assert masked_stores == 1


# Regression test for compilation failure in masks optimization
def test_vec_cdiv(device):

    @triton.jit
    def kernel(in_ptr, out_ptr):
        offs = tl.arange(0, 16)
        x = tl.load(in_ptr + offs)
        res = (x + 15) // 16
        tl.store(out_ptr + offs, res)

    arg0 = torch.zeros((16, ), dtype=torch.int32)
    arg1 = torch.empty_like(arg0)
    kernel[(1, )](arg0, arg1)


@pytest.mark.parametrize("enabled", [False, True])
def test_single_program_c_entry_preserves_grid_launcher(enabled):
    import ctypes

    @triton.jit
    def kernel(out, value):
        x, y, z = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        nx, ny, nz = tl.num_programs(0), tl.num_programs(1), tl.num_programs(2)
        tl.store(out + x + nx * (y + ny * z), value + 100 * nx + 10 * ny + nz)

    output = torch.full((8,), -1, dtype=torch.int32)
    # Warmup exports metadata but never executes the kernel.
    compiled = kernel.warmup(output, 7, grid=(2, 2, 2), enable_single_program_entry=enabled)
    assert torch.equal(output, torch.full_like(output, -1))
    entry_name = getattr(compiled.metadata, "single_program_name", None)
    if enabled:
        assert entry_name == compiled.name + "__single_program"
        library = ctypes.CDLL(compiled.metadata_group[compiled.name + ".so"])
        entry = getattr(library, entry_name)
        entry.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        entry.restype = None
        entry(output.data_ptr(), 7)
        assert output.tolist() == [118, -1, -1, -1, -1, -1, -1, -1]
    else:
        assert entry_name is None
    # Enabling the extra entry must not specialize the ordinary grid to one.
    kernel[(2, 2, 2)](output, 7, enable_single_program_entry=enabled)
    assert output.tolist() == [229] * 8


@pytest.mark.parametrize("kind,dtype,ctype", [
    ("i32", torch.int32, "c_int32"), ("i64", torch.int64, "c_int64"),
    ("f32", torch.float32, "c_float"), ("f64", torch.float64, "c_double"),
])
def test_single_program_c_scalar_return(kind, dtype, ctype):
    import ctypes

    @triton.jit
    def kernel(out, value, result):
        tl.store(out + tl.program_id(0), value * 2)
        tl.store(result, value + 3)

    output = torch.full((2,), -1, dtype=torch.int32)
    result = torch.full((1,), -1, dtype=dtype)
    compiled = kernel.warmup(output, 7, result, grid=(1,),
                             enable_single_program_entry=True,
                             single_program_return_type=kind)
    library = ctypes.CDLL(compiled.metadata_group[compiled.name + ".so"])
    entry = getattr(library, compiled.metadata.single_program_name)
    entry.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    entry.restype = getattr(ctypes, ctype)
    assert entry(output.data_ptr(), 7) == 10
    assert output.tolist() == [14, -1]
    assert result.item() == -1  # No result pointer crosses the additional C ABI.
    kernel[(2,)](output, 7, result, enable_single_program_entry=True,
                 single_program_return_type=kind)
    assert output.tolist() == [14, 14]
    assert result.item() == 10


def test_single_program_scalar_return_rejects_invalid_abi():
    from triton.backends.cpu.compiler import CPUOptions

    with pytest.raises(ValueError, match="return type"):
        CPUOptions(enable_single_program_entry=True, single_program_return_type="i8")
    with pytest.raises(ValueError, match="requires"):
        CPUOptions(single_program_return_type="i32")

    @triton.jit
    def kernel(out, value):
        tl.store(out, value)

    with pytest.raises(RuntimeError, match="final output pointer"):
        kernel.warmup(torch.empty(1, dtype=torch.int32), 7, grid=(1,),
                      enable_single_program_entry=True, single_program_return_type="i32")


@pytest.mark.parametrize("case", ["type", "offset", "vector", "escape"])
def test_single_program_scalar_return_rejects_unsafe_output(case):
    @triton.jit
    def kernel(out, result, CASE: tl.constexpr):
        if CASE == "offset":
            tl.store(result + 1, 7)
        elif CASE == "vector":
            tl.store(result + tl.arange(0, 4), 7)
        elif CASE == "escape":
            tl.store(out, result.to(tl.uint64))
        else:
            tl.store(result, 7)

    dtype = torch.float32 if case == "type" else torch.int32
    with pytest.raises(RuntimeError, match="Scalar C return"):
        kernel.warmup(torch.empty(1, dtype=torch.int64), torch.empty(4, dtype=dtype), case,
                      grid=(1,), enable_single_program_entry=True,
                      single_program_return_type="i32")


@pytest.mark.parametrize("kind,dtype", [("f32", torch.float32), ("f64", torch.float64)])
def test_cpu_raw_call_float_and_void_abi(tmp_path, kind, dtype):
    from triton.language.extra.cpu.tle_ops import raw_call

    llvm_type = "float" if kind == "f32" else "double"
    leaf = tmp_path / "float_leaf.ll"
    leaf.write_text(
        f"define {llvm_type} @cpu_raw_add({llvm_type} %x) {{\n"
        f" %y = fadd {llvm_type} %x, 1.000000e+00\n ret {llvm_type} %y\n}}\n"
        f"define void @cpu_raw_store(i64 %address, {llvm_type} %x) {{\n"
        f" %out = inttoptr i64 %address to ptr\n store {llvm_type} %x, ptr %out\n ret void\n}}\n"
    )

    @triton.jit
    def kernel(input, out, TYPE: tl.constexpr):
        value = raw_call("cpu_raw_add", tl.load(input).to(TYPE), result_dtype=TYPE)
        raw_call("cpu_raw_store", out.to(tl.uint64), value, result_dtype=None)

    x = torch.tensor([1.25], dtype=dtype)
    out = torch.zeros_like(x)
    kernel[(1,)](x, out, tl.float32 if kind == "f32" else tl.float64,
                  extern_libs={"leaf": str(leaf)})
    assert out.item() == 2.25


def test_cpu_raw_call_rejects_block_arguments():
    from triton.language.extra.cpu.tle_ops import raw_call
    from triton.compiler.errors import CompilationError

    @triton.jit
    def kernel(out):
        value = raw_call("not_reached", tl.arange(0, 4))
        tl.store(out, value)

    with pytest.raises(CompilationError, match="scalar integers or floats"):
        kernel.warmup(torch.empty(1, dtype=torch.int32), grid=(1,))


def test_cpu_external_ir_link_and_disk_cache_invalidation(tmp_path):
    from triton.language.extra.cpu.tle_ops import raw_call

    @triton.jit
    def kernel(out, value):
        result = raw_call("cpu_test_add", value.to(tl.int32))
        tl.store(out, result)

    library = tmp_path / "leaf.ll"
    output = torch.empty(1, dtype=torch.int32)
    compiled_hashes = []
    for increment in (17, 29):
        library.write_text(
            "define i32 @cpu_test_add(i32 %x) {\n"
            f"  %y = add i32 %x, {increment}\n"
            "  ret i32 %y\n}\n"
        )
        # As with loaded DSOs, library files remain immutable while a prepared
        # kernel is used. A fresh JIT object exercises persistent cache lookup.
        fresh = triton.jit(kernel.fn)
        compiled = fresh.warmup(output, 7, grid=(1,), extern_libs={"leaf": str(library)})
        fresh[(1,)](output, 7, extern_libs={"leaf": str(library)})
        assert output.item() == 7 + increment
        compiled_hashes.append(compiled.hash)
    assert compiled_hashes[0] != compiled_hashes[1]


def test_cpu_external_ir_tls_uses_shared_library_relocations(tmp_path):
    import ctypes

    from triton.language.extra.cpu.tle_ops import raw_call

    library = tmp_path / "tls.ll"
    library.write_text(
        "@counter = internal thread_local global i32 0\n"
        "define i32 @cpu_test_tls(i32 %x) {\n"
        "  %old = load i32, ptr @counter\n"
        "  %sum = add i32 %old, %x\n"
        "  store i32 %sum, ptr @counter\n"
        "  ret i32 %sum\n}\n"
        "!llvm.module.flags = !{!0, !1}\n"
        '!0 = !{i32 8, !"PIC Level", i32 2}\n'
        '!1 = !{i32 7, !"PIE Level", i32 2}\n'
    )

    @triton.jit
    def kernel(out, value):
        tl.store(out, raw_call("cpu_test_tls", value.to(tl.int32)))

    output = torch.zeros(1, dtype=torch.int32)
    compiled = kernel.warmup(output, 7, grid=(1,), enable_single_program_entry=True,
                             extern_libs={"tls": str(library)})
    assert '\"PIE Level\", i32 0' in compiled.asm["llir"]
    dso = ctypes.CDLL(compiled.metadata_group[compiled.name + ".so"])
    entry = getattr(dso, compiled.metadata.single_program_name)
    entry.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    entry.restype = None
    for expected in (7, 14):
        entry(output.data_ptr(), 7)
        assert output.item() == expected
