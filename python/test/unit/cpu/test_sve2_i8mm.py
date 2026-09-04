import os
import platform

import pytest
import torch

import triton
import triton.language as tl
from triton._C.libtriton import llvm
from triton.backends.cpu.target_info import (
    get_sve_vector_bits,
    supplement_aarch64_features,
)


def has_sve2_i8mm_128() -> bool:
    if platform.machine() not in ("aarch64", "arm64"):
        return False
    features = supplement_aarch64_features(llvm.get_cpu_features())
    return "sve2" in features and "i8mm" in features and get_sve_vector_bits() == 128


def has_arm_i8mm() -> bool:
    if platform.machine() not in ("aarch64", "arm64"):
        return False
    features = supplement_aarch64_features(llvm.get_cpu_features())
    return "dotprod" in features and "i8mm" in features


def uses_fixed_i8mm() -> bool:
    return os.environ.get("TRITON_CPU_FIXED_I8MM", "0") == "1" or not has_sve2_i8mm_128()


requires_sve2_i8mm = pytest.mark.skipif(
    not has_sve2_i8mm_128() or os.environ.get("TRITON_CPU_FIXED_I8MM", "0") == "1"
    or os.environ.get("TRITON_CPU_DISABLE_SVE2_I8MM", "0") == "1",
    reason="requires the 128-bit Arm SVE2/i8mm lowering",
)

requires_arm_i8mm = pytest.mark.skipif(
    not has_arm_i8mm() or os.environ.get("TRITON_CPU_DISABLE_SVE2_I8MM", "0") == "1",
    reason="requires the Arm DotProd/i8mm lowering",
)

requires_cortex_a720 = pytest.mark.skipif(
    platform.machine() not in ("aarch64", "arm64") or llvm.get_cpu_name().lower() != "cortex-a720",
    reason="requires Cortex-A720 store-pair code generation",
)


@requires_cortex_a720
def test_a720_independent_bf16_loop_store_avoids_pairing(monkeypatch):
    monkeypatch.setenv("TRITON_CPU_A720_BF16_STORE_WORKAROUND", "1")

    @triton.jit
    def kernel(input_ptr, weight_ptr, output_ptr, N: tl.constexpr):
        for offset in range(0, N, 16):
            lanes = offset + tl.arange(0, 16)
            value = tl.load(input_ptr + lanes).to(tl.float32)
            weight = tl.load(weight_ptr + lanes).to(tl.float32)
            tl.store(output_ptr + lanes, (value * weight).to(tl.bfloat16))

    torch.manual_seed(720)
    input = torch.randn(1024, dtype=torch.bfloat16)
    weight = torch.randn(1024, dtype=torch.bfloat16)
    output = torch.empty_like(input)
    compiled = kernel[(1, )](input, weight, output, N=1024)

    expected = (input.float() * weight.float()).to(torch.bfloat16)
    assert torch.equal(output, expected)
    llir = compiled.asm["llir"].lower()
    assert "store volatile <16 x bfloat>" in llir
    assembly = compiled.asm["asm"].lower()
    assert "stp\tq" not in assembly
    assert assembly.count("str\tq") == 2


@requires_sve2_i8mm
def test_i8_dot_uses_smmla():

    @triton.jit
    def kernel(a_ptr, b_ptr, c_ptr):
        m = tl.arange(0, 32)
        n = tl.arange(0, 32)
        k = tl.arange(0, 128)
        a = tl.load(a_ptr + m[:, None] * 128 + k[None, :])
        b = tl.load(b_ptr + k[:, None] * 32 + n[None, :])
        result = tl.dot(a, b, out_dtype=tl.int32)
        tl.store(c_ptr + m[:, None] * 32 + n[None, :], result)

    torch.manual_seed(1701)
    a = torch.randint(-128, 128, (32, 128), dtype=torch.int8, device="cpu")
    b = torch.randint(-128, 128, (128, 32), dtype=torch.int8, device="cpu")
    actual = torch.empty((32, 32), dtype=torch.int32, device="cpu")
    compiled = kernel[(1, )](a, b, actual)

    expected = a.to(torch.int32) @ b.to(torch.int32)
    assert torch.equal(actual, expected)
    assembly = compiled.asm["asm"].lower()
    assert assembly.count("smmla") == 16
    assert "smull" not in assembly
    assert "smlal" not in assembly
    # The SMMLA lowering must write its 2x2 accumulator fragments straight to
    # the final strided output descriptor.  Materializing the whole 32x32 i32
    # result in a stack buffer regresses medium and large tiles significantly.
    llir = compiled.asm["llir"].lower()
    assert "alloca [1024 x i32]" not in llir
    tttcir = compiled.asm["tttcir"].lower()
    assert "memref.alloca() {alignment = 64 : i64} : memref<32x32xi32>" not in tttcir
    assert "memref<32x32xi32, strided<[?, 1]>>" in tttcir


@requires_arm_i8mm
def test_w4a8_m1_dot_fuses_nibble_unpack_to_sdot():

    @triton.jit
    def kernel(x_ptr, packed_ptr, out_ptr):
        packed_flat = tl.load(packed_ptr + tl.arange(0, 16))
        packed = tl.trans(packed_flat.reshape((4, 4)))
        weight_low = ((packed & 0x0F).to(tl.int8) - 8).to(tl.int8)
        weight_high = ((packed >> 4).to(tl.int8) - 8).to(tl.int8)
        k = tl.arange(0, 4)
        x_low = tl.load(x_ptr + k).reshape((1, 4))
        x_high = tl.load(x_ptr + 16 + k).reshape((1, 4))
        result = tl.dot(x_low, weight_low, out_dtype=tl.int32)
        result += tl.dot(x_high, weight_high, out_dtype=tl.int32)
        tl.store(out_ptr + tl.arange(0, 4), result.reshape((4, )))

    x = torch.tensor(
        [17, -5, 11, 3] + [0] * 12 + [-7, 13, 2, -9] + [0] * 12,
        dtype=torch.int8,
        device="cpu",
    )
    weight_low = torch.tensor(
        [[-8, 7, -3, 2], [1, -4, 6, -7], [5, 0, -2, 4], [-1, 3, -5, 7]],
        dtype=torch.int8,
        device="cpu",
    )
    weight_high = torch.tensor(
        [[6, -2, 1, -8], [-5, 4, -1, 3], [7, -7, 2, 0], [3, 5, -4, -6]],
        dtype=torch.int8,
        device="cpu",
    )
    packed = ((weight_low.to(torch.int16) + 8) | ((weight_high.to(torch.int16) + 8) << 4)).to(torch.uint8).reshape(-1)
    actual = torch.empty(4, dtype=torch.int32, device="cpu")
    compiled = kernel[(1, )](x, packed, actual)

    expected = (x[:4].to(torch.int32) @ weight_low.to(torch.int32).T +
                x[16:20].to(torch.int32) @ weight_high.to(torch.int32).T)
    assert torch.equal(actual, expected)
    assembly = compiled.asm["asm"].lower()
    assert assembly.count("sdot") == 2
    assert "smmla" not in assembly
    llir = compiled.asm["llir"].lower()
    assert "llvm.aarch64.neon.sdot" in llir


@requires_arm_i8mm
def test_packed_w8a8_honors_loop_unroll_and_uses_sdot():

    @triton.jit
    def kernel(x_ptr, packed_ptr, out_ptr):
        dot = tl.zeros((1, 4), dtype=tl.int32)
        for group in tl.range(0, 64, loop_unroll_factor=2):
            packed_flat = tl.load(packed_ptr + group * 16 + tl.arange(0, 16))
            weight = tl.trans(packed_flat.reshape((4, 4)))
            x = tl.load(x_ptr + group * 4 + tl.arange(0, 4)).reshape((1, 4))
            dot += tl.dot(x, weight, out_dtype=tl.int32)
        tl.store(out_ptr + tl.arange(0, 4), dot.reshape((4, )))

    torch.manual_seed(801)
    x = torch.randint(-127, 128, (256, ), dtype=torch.int8, device="cpu")
    weight = torch.randint(-127, 128, (256, 4), dtype=torch.int8, device="cpu")
    packed = (weight.reshape(64, 4, 4).permute(0, 2, 1).contiguous())
    actual = torch.empty(4, dtype=torch.int32, device="cpu")
    compiled = kernel[(1, )](x, packed, actual)

    expected = x.to(torch.int32) @ weight.to(torch.int32)
    assert torch.equal(actual, expected)
    assembly = compiled.asm["asm"].lower()
    assert assembly.count("sdot") == 2
    assert "smull" not in assembly
    assert "smlal" not in assembly
    llir = compiled.asm["llir"].lower()
    assert llir.count("call <4 x i32> @llvm.aarch64.neon.sdot") == 2


@requires_arm_i8mm
def test_kai_w8_physical_layout_mul_sum_fuses_sdot_and_addp():

    @triton.jit
    def kernel(x_ptr, packed_ptr, out_ptr):
        partial01 = tl.zeros((4, ), dtype=tl.int32)
        partial23 = tl.zeros((4, ), dtype=tl.int32)
        q_offsets = tl.arange(0, 16)
        x_offsets = tl.arange(0, 8)
        for group in tl.range(0, 64, loop_unroll_factor=1):
            base = group * 32
            weight01 = tl.load(packed_ptr + base + q_offsets).reshape((4, 4))
            weight23 = tl.load(packed_ptr + base + 16 + q_offsets).reshape((4, 4))
            x = tl.load(x_ptr + group * 8 + x_offsets).reshape((2, 4))
            x_repeated = tl.cat(x, x, dim=0)
            partial01 += tl.sum(weight01.to(tl.int32) * x_repeated.to(tl.int32), axis=1)
            partial23 += tl.sum(weight23.to(tl.int32) * x_repeated.to(tl.int32), axis=1)
        partial = tl.cat(
            partial01.reshape((2, 2)),
            partial23.reshape((2, 2)),
            dim=0,
        )
        result = tl.sum(partial, axis=1)
        tl.store(out_ptr + tl.arange(0, 4), result)

    torch.manual_seed(8128)
    x = torch.randint(-127, 128, (512, ), dtype=torch.int8, device="cpu")
    weight = torch.randint(-127, 128, (4, 512), dtype=torch.int8, device="cpu")
    packed = (weight.reshape(4, 64, 8).permute(1, 0, 2).contiguous().reshape(-1))
    actual = torch.empty(4, dtype=torch.int32, device="cpu")
    compiled = kernel[(1, )](x, packed, actual)

    expected = x.to(torch.int32) @ weight.to(torch.int32).T
    assert torch.equal(actual, expected)
    assembly = compiled.asm["asm"].lower()
    assert assembly.count("sdot") == 2
    assert assembly.count("addp") == 1
    assert "smull" not in assembly
    assert "smlal" not in assembly
    llir = compiled.asm["llir"].lower()
    assert llir.count("call <4 x i32> @llvm.aarch64.neon.sdot") == 2
    assert llir.count("call <4 x i32> @llvm.aarch64.neon.addp") == 1


@requires_arm_i8mm
def test_kai_w4_physical_layout_fuses_sdot_addp_and_fixed_scvtf():

    @triton.jit
    def kernel(x_ptr, packed_ptr, out_ptr):
        q_lanes = tl.arange(0, 16)
        x_lanes = tl.arange(0, 8)
        q0 = tl.load(packed_ptr + q_lanes)
        q1 = tl.load(packed_ptr + 16 + q_lanes)
        q2 = tl.load(packed_ptr + 32 + q_lanes)
        q3 = tl.load(packed_ptr + 48 + q_lanes)
        q0_low = (q0 << 4).to(tl.int8).reshape((4, 4))
        q1_low = (q1 << 4).to(tl.int8).reshape((4, 4))
        q2_low = (q2 << 4).to(tl.int8).reshape((4, 4))
        q3_low = (q3 << 4).to(tl.int8).reshape((4, 4))
        q0_high = (q0 & 0xF0).to(tl.int8).reshape((4, 4))
        q1_high = (q1 & 0xF0).to(tl.int8).reshape((4, 4))
        q2_high = (q2 & 0xF0).to(tl.int8).reshape((4, 4))
        q3_high = (q3 & 0xF0).to(tl.int8).reshape((4, 4))

        x0 = tl.load(x_ptr + x_lanes).to(tl.int8).reshape((2, 4))
        x1 = tl.load(x_ptr + 8 + x_lanes).to(tl.int8).reshape((2, 4))
        x2 = tl.load(x_ptr + 16 + x_lanes).to(tl.int8).reshape((2, 4))
        x3 = tl.load(x_ptr + 24 + x_lanes).to(tl.int8).reshape((2, 4))
        x0 = tl.cat(x0, x0, dim=0)
        x1 = tl.cat(x1, x1, dim=0)
        x2 = tl.cat(x2, x2, dim=0)
        x3 = tl.cat(x3, x3, dim=0)

        partial01 = tl.sum(q0_low.to(tl.int32) * x0.to(tl.int32), axis=1)
        partial23 = tl.sum(q1_low.to(tl.int32) * x0.to(tl.int32), axis=1)
        partial01 += tl.sum(q2_low.to(tl.int32) * x1.to(tl.int32), axis=1)
        partial23 += tl.sum(q3_low.to(tl.int32) * x1.to(tl.int32), axis=1)
        partial01 += tl.sum(q0_high.to(tl.int32) * x2.to(tl.int32), axis=1)
        partial23 += tl.sum(q1_high.to(tl.int32) * x2.to(tl.int32), axis=1)
        partial01 += tl.sum(q2_high.to(tl.int32) * x3.to(tl.int32), axis=1)
        partial23 += tl.sum(q3_high.to(tl.int32) * x3.to(tl.int32), axis=1)
        partial = tl.cat(partial01.reshape((2, 2)), partial23.reshape((2, 2)), dim=0)
        result = tl.sum(partial, axis=1).to(tl.float32) * (1.0 / 16.0)
        tl.store(out_ptr + tl.arange(0, 4), result)

    torch.manual_seed(4328)
    x = torch.randint(-127, 128, (32, ), dtype=torch.int8, device="cpu")
    weight = torch.randint(-8, 8, (4, 32), dtype=torch.int8, device="cpu")
    vectors = []
    for k_begin, outputs in ((0, (0, 1)), (0, (2, 3)), (8, (0, 1)), (8, (2, 3))):
        low = weight[list(outputs), k_begin:k_begin + 8].to(torch.int16) & 15
        high = weight[list(outputs), k_begin + 16:k_begin + 24].to(torch.int16) & 15
        vectors.append((low | (high << 4)).to(torch.uint8).reshape(-1))
    packed = torch.cat(vectors)
    actual = torch.empty(4, dtype=torch.float32, device="cpu")
    compiled = kernel[(1, )](x.view(torch.uint8), packed, actual)

    expected = x.to(torch.int32) @ weight.to(torch.int32).T
    assert torch.equal(actual, expected.to(torch.float32))
    assembly = compiled.asm["asm"].lower()
    assert assembly.count("sdot") == 8
    assert assembly.count("addp") == 1
    assert "scvtf" in assembly and "#4" in assembly
    assert "smull" not in assembly
    assert "smlal" not in assembly
    llir = compiled.asm["llir"].lower()
    assert llir.count("call <4 x i32> @llvm.aarch64.neon.sdot") == 8
    assert "llvm.aarch64.neon.vcvtfxs2fp" in llir


@requires_arm_i8mm
@pytest.mark.parametrize("groups", (2, 8, 32))
def test_kai_q4_prefill_m16_fuses_ordinary_tl_dot_to_smmla(groups):

    @triton.jit
    def kernel(
        lhs_data_ptr,
        lhs_scale_ptr,
        rhs_data_ptr,
        rhs_scale_ptr,
        out_ptr,
        N: tl.constexpr,
        K: tl.constexpr,
        HIGH_MASK: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        groups: tl.constexpr = K // 32
        cols = pid_n * 4 + tl.arange(0, 4)
        rows = tl.arange(0, 4)
        result0 = tl.zeros((4, 4), tl.float32)
        result1 = tl.zeros((4, 4), tl.float32)
        result2 = tl.zeros((4, 4), tl.float32)
        result3 = tl.zeros((4, 4), tl.float32)

        for group in range(0, groups):
            rhs_base = (pid_n * groups + group) * 72
            packed = tl.load(rhs_data_ptr + rhs_base + 8 + tl.arange(0, 64)).reshape(
                (2, 4, 8)).permute(0, 2, 1).reshape((16, 4))
            weight_low = (packed << 4).to(tl.int8)
            weight_high = (packed & HIGH_MASK).to(tl.int8)
            weight = tl.join(weight_low, weight_high).permute(0, 2, 1).reshape((32, 4))
            rhs_scale = tl.load(rhs_scale_ptr + rhs_base // 2 + tl.arange(0, 4)).to(tl.float32)

            lhs_base = group * 136
            panel_stride: tl.constexpr = groups * 136
            lhs0_seq = tl.load(lhs_data_ptr + lhs_base + 8 + tl.arange(0, 128)).reshape(
                (4, 4, 8)).permute(1, 0, 2).reshape((4, 32))
            lhs0 = lhs0_seq.reshape((4, 2, 16)).permute(0, 2, 1).reshape((4, 32))
            scale0 = tl.load(lhs_scale_ptr + lhs_base // 2 + rows).to(tl.float32)
            result0 += (tl.dot(lhs0, weight, out_dtype=tl.int32).to(tl.float32) * (1.0 / 16.0) * scale0[:, None] *
                        rhs_scale[None, :])

            lhs1_base = lhs_base + panel_stride
            lhs1_seq = tl.load(lhs_data_ptr + lhs1_base + 8 + tl.arange(0, 128)).reshape(
                (4, 4, 8)).permute(1, 0, 2).reshape((4, 32))
            lhs1 = lhs1_seq.reshape((4, 2, 16)).permute(0, 2, 1).reshape((4, 32))
            scale1 = tl.load(lhs_scale_ptr + lhs1_base // 2 + rows).to(tl.float32)
            result1 += (tl.dot(lhs1, weight, out_dtype=tl.int32).to(tl.float32) * (1.0 / 16.0) * scale1[:, None] *
                        rhs_scale[None, :])

            lhs2_base = lhs_base + 2 * panel_stride
            lhs2_seq = tl.load(lhs_data_ptr + lhs2_base + 8 + tl.arange(0, 128)).reshape(
                (4, 4, 8)).permute(1, 0, 2).reshape((4, 32))
            lhs2 = lhs2_seq.reshape((4, 2, 16)).permute(0, 2, 1).reshape((4, 32))
            scale2 = tl.load(lhs_scale_ptr + lhs2_base // 2 + rows).to(tl.float32)
            result2 += (tl.dot(lhs2, weight, out_dtype=tl.int32).to(tl.float32) * (1.0 / 16.0) * scale2[:, None] *
                        rhs_scale[None, :])

            lhs3_base = lhs_base + 3 * panel_stride
            lhs3_seq = tl.load(lhs_data_ptr + lhs3_base + 8 + tl.arange(0, 128)).reshape(
                (4, 4, 8)).permute(1, 0, 2).reshape((4, 32))
            lhs3 = lhs3_seq.reshape((4, 2, 16)).permute(0, 2, 1).reshape((4, 32))
            scale3 = tl.load(lhs_scale_ptr + lhs3_base // 2 + rows).to(tl.float32)
            result3 += (tl.dot(lhs3, weight, out_dtype=tl.int32).to(tl.float32) * (1.0 / 16.0) * scale3[:, None] *
                        rhs_scale[None, :])

        tl.store(out_ptr + rows[:, None] * N + cols[None, :], result0.to(tl.bfloat16))
        tl.store(out_ptr + (rows + 4)[:, None] * N + cols[None, :], result1.to(tl.bfloat16))
        tl.store(out_ptr + (rows + 8)[:, None] * N + cols[None, :], result2.to(tl.bfloat16))
        tl.store(out_ptr + (rows + 12)[:, None] * N + cols[None, :], result3.to(tl.bfloat16))

    torch.manual_seed(41632)
    lhs_blob = torch.zeros(4 * groups * 136, dtype=torch.uint8)
    rhs_blob = torch.zeros(groups * 72, dtype=torch.uint8)
    for panel in range(4):
        for group in range(groups):
            base = (panel * groups + group) * 136
            lhs_blob.view(torch.float16)[base // 2:base // 2 + 4] = (torch.rand(4, dtype=torch.float16) + 0.25)
            lhs_blob[base + 8:base + 136] = torch.randint(0, 256, (128, ), dtype=torch.uint8)
    for group in range(groups):
        base = group * 72
        rhs_blob.view(torch.float16)[base // 2:base // 2 + 4] = (torch.rand(4, dtype=torch.float16) + 0.25)
        rhs_blob[base + 8:base + 72] = torch.randint(0, 256, (64, ), dtype=torch.uint8)

    def reference(high_mask):
        expected = torch.zeros((16, 4), dtype=torch.float32)
        for group in range(groups):
            rhs_base = group * 72
            packed = rhs_blob[rhs_base + 8:rhs_base + 72].reshape(2, 4, 8).permute(0, 2, 1).reshape(16, 4)
            low = (packed << 4).view(torch.int8)
            high = (packed & high_mask).view(torch.int8)
            weight = torch.stack((low, high), dim=2).permute(0, 2, 1).reshape(32, 4)
            rhs_scale = rhs_blob.view(torch.float16)[rhs_base // 2:rhs_base // 2 + 4].float()
            for panel in range(4):
                lhs_base = (panel * groups + group) * 136
                lhs_seq = lhs_blob[lhs_base + 8:lhs_base + 136].view(torch.int8).reshape(4, 4,
                                                                                         8).permute(1, 0,
                                                                                                    2).reshape(4, 32)
                lhs = lhs_seq.reshape(4, 2, 16).permute(0, 2, 1).reshape(4, 32)
                lhs_scale = lhs_blob.view(torch.float16)[lhs_base // 2:lhs_base // 2 + 4].float()
                expected[panel * 4:(panel + 1) * 4] += ((lhs.to(torch.int32) @ weight.to(torch.int32)).float() *
                                                        (1.0 / 16.0) * lhs_scale[:, None] * rhs_scale[None, :])
        return expected.to(torch.bfloat16)

    actual = torch.empty((16, 4), dtype=torch.bfloat16)
    compiled = kernel[(1, )](
        lhs_blob.view(torch.int8),
        lhs_blob.view(torch.float16),
        rhs_blob,
        rhs_blob.view(torch.float16),
        actual,
        N=4,
        K=groups * 32,
        HIGH_MASK=0xF0,
    )
    assert torch.equal(actual, reference(0xF0))
    assembly = compiled.asm["asm"].lower()
    assert assembly.count("smmla") == 64
    assert "smull" not in assembly and "smlal" not in assembly
    # LLVM may save four callee-saved SIMD pairs in the prologue/epilogue.  A
    # separate GPR pair can also be saved as loop state, so count vector spills
    # rather than every line carrying LLVM's generic "Folded Spill" comment.
    vector_spills = [
        line for line in assembly.splitlines()
        if "folded spill" in line and any(opcode in line for opcode in ("stp\td", "str\td", "stp\tq", "str\tq"))
    ]
    vector_reloads = [
        line for line in assembly.splitlines()
        if "folded reload" in line and any(opcode in line for opcode in ("ldp\td", "ldr\td", "ldp\tq", "ldr\tq"))
    ]
    assert len(vector_spills) <= 4
    assert len(vector_reloads) <= 4
    llir = compiled.asm["llir"].lower()
    assert llir.count("call <4 x i32> @llvm.aarch64.neon.smmla") == 64
    assert "triton_cpu.dot" not in llir
    assert "sdot_gemv" not in llir and "fused_mlp" not in llir

    # A graph that differs by even one packed-layout constant must remain on
    # the generic path.  This guards the matcher against a silent miscompile.
    negative = torch.empty_like(actual)
    generic = kernel[(1, )](
        lhs_blob.view(torch.int8),
        lhs_blob.view(torch.float16),
        rhs_blob,
        rhs_blob.view(torch.float16),
        negative,
        N=4,
        K=groups * 32,
        HIGH_MASK=0xE0,
    )
    assert torch.equal(negative, reference(0xE0))
    assert "llvm.aarch64.neon.smmla" not in generic.asm["llir"].lower()


@requires_arm_i8mm
@pytest.mark.parametrize("k_size", (256, 1024, 3072))
def test_kai_w8_prefill_m16_fuses_ordinary_tl_dot_to_smmla(k_size):

    @triton.jit
    def kernel(
        lhs_packed_ptr,
        rhs_packed_ptr,
        out_ptr,
        K: tl.constexpr,
    ):
        panel_offsets = tl.arange(0, 128)
        panel_stride: tl.constexpr = 4 * K
        accumulator = tl.zeros((16, 4), tl.int32)

        # This is the ordinary Triton description of KleidiAI's M4/K8
        # interleaving.  Target-aware lowering recognizes the exact physical
        # graph while the frontend kernel keeps a regular logical tl.dot.
        for chunk in range(0, K // 32):
            lhs_base = chunk * 128
            lhs0 = tl.load(lhs_packed_ptr + lhs_base + panel_offsets).reshape((4, 4, 8)).permute(1, 0, 2).reshape(
                (4, 32))
            lhs1 = tl.load(lhs_packed_ptr + lhs_base + panel_stride + panel_offsets).reshape(
                (4, 4, 8)).permute(1, 0, 2).reshape((4, 32))
            lhs2 = tl.load(lhs_packed_ptr + lhs_base + 2 * panel_stride + panel_offsets).reshape(
                (4, 4, 8)).permute(1, 0, 2).reshape((4, 32))
            lhs3 = tl.load(lhs_packed_ptr + lhs_base + 3 * panel_stride + panel_offsets).reshape(
                (4, 4, 8)).permute(1, 0, 2).reshape((4, 32))
            lhs01 = tl.join(lhs0, lhs1).permute(2, 0, 1).reshape((8, 32))
            lhs23 = tl.join(lhs2, lhs3).permute(2, 0, 1).reshape((8, 32))
            lhs = tl.join(lhs01, lhs23).permute(2, 0, 1).reshape((16, 32))

            rhs = tl.load(rhs_packed_ptr + chunk * 128 + panel_offsets).reshape((4, 4, 8)).permute(0, 2, 1).reshape(
                (32, 4))
            accumulator += tl.dot(lhs, rhs, out_dtype=tl.int32)

        rows = tl.arange(0, 16)
        cols = tl.arange(0, 4)
        tl.store(out_ptr + rows[:, None] * 4 + cols[None, :], accumulator)

    torch.manual_seed(81632)
    lhs = torch.randint(-127, 128, (16, k_size), dtype=torch.int8)
    rhs = torch.randint(-127, 128, (4, k_size), dtype=torch.int8)
    lhs_packed = torch.cat([
        panel.reshape(4, k_size // 8, 8).permute(1, 0, 2).contiguous().reshape(-1)
        for panel in lhs.reshape(4, 4, k_size)
    ])
    rhs_packed = (rhs.reshape(4, k_size // 8, 8).permute(1, 0, 2).contiguous().reshape(-1))
    actual = torch.empty((16, 4), dtype=torch.int32)
    compiled = kernel[(1, )](lhs_packed, rhs_packed, actual, K=k_size)

    expected = lhs.to(torch.int32) @ rhs.to(torch.int32).T
    assert torch.equal(actual, expected)
    assembly = compiled.asm["asm"].lower()
    assert assembly.count("smmla") == 64
    assert "smull" not in assembly and "smlal" not in assembly
    # Four M4 panels remain in native 2x2 accumulators throughout the K loop;
    # no accumulator or operand should be folded through the stack.
    assert "folded spill" not in assembly
    assert "folded reload" not in assembly
    llir = compiled.asm["llir"].lower()
    if uses_fixed_i8mm():
        assert llir.count("call <4 x i32> @llvm.aarch64.neon.smmla.v4i32.v16i8") == 64
        assert "llvm.aarch64.sve.smmla" not in llir
    else:
        assert llir.count("call <vscale x 4 x i32> @llvm.aarch64.sve.smmla.nxv4i32") == 64
    assert "triton_cpu.dot" not in llir
    assert "sdot_gemv" not in llir and "fused_mlp" not in llir

    @triton.jit
    def wrong_rhs_layout_kernel(lhs_ptr, rhs_ptr, out_ptr):
        lanes = tl.arange(0, 128)
        lhs_tile = tl.load(lhs_ptr + lanes).reshape((4, 4, 8)).permute(1, 0, 2).reshape((4, 32))
        # The same 128 physical bytes but a different permutation describe a
        # different logical matrix.  It must never match the KAI ABI rewrite.
        rhs_tile = tl.load(rhs_ptr + lanes).reshape((4, 4, 8)).permute(1, 0, 2).reshape((32, 4))
        result = tl.dot(lhs_tile, rhs_tile, out_dtype=tl.int32)
        rows = tl.arange(0, 4)
        cols = tl.arange(0, 4)
        tl.store(out_ptr + rows[:, None] * 4 + cols[None, :], result)

    wrong_actual = torch.empty((4, 4), dtype=torch.int32)
    generic = wrong_rhs_layout_kernel[(1, )](lhs_packed[:128], rhs_packed[:128], wrong_actual)
    wrong_lhs = lhs_packed[:128].reshape(4, 4, 8).permute(1, 0, 2).reshape(4, 32)
    wrong_rhs = rhs_packed[:128].reshape(4, 4, 8).permute(1, 0, 2).reshape(32, 4)
    assert torch.equal(
        wrong_actual,
        wrong_lhs.to(torch.int32) @ wrong_rhs.to(torch.int32),
    )
    assert "llvm.aarch64.sve.smmla" not in generic.asm["llir"].lower()
