from pathlib import Path

import pytest

from triton.backends.cpu import target_info
from triton.backends.cpu import compiler as cpu_compiler
from triton.backends.cpu.compiler import CPUBackend, _normalize_darwin_aarch64_assembly


def test_parse_linux_cpuinfo_uses_features_common_to_all_cpus():
    cpuinfo = """
processor : 0
Features  : fp asimd asimddp sve sve2 i8mm bf16

processor : 1
Features  : fp asimd asimddp sve sve2 bf16
"""
    assert target_info.parse_linux_cpuinfo_features(cpuinfo) == {
        "fp",
        "asimd",
        "asimddp",
        "sve",
        "sve2",
        "bf16",
    }


def test_supplement_aarch64_features(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(target_info.platform, "system", lambda: "Linux")
    monkeypatch.setattr(target_info.platform, "machine", lambda: "aarch64")
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text("Features : fp asimd asimdhp asimddp i8mm bf16 sve sve2 svebf16\n")

    assert target_info.supplement_aarch64_features({"neon", "sve2"}, cpuinfo_path=cpuinfo) == {
        "fp-armv8",
        "neon",
        "sve",
        "sve2",
        "dotprod",
        "fullfp16",
        "i8mm",
        "bf16",
    }


def test_supplement_aarch64_features_removes_non_common_llvm_features(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(target_info.platform, "system", lambda: "Linux")
    monkeypatch.setattr(target_info.platform, "machine", lambda: "aarch64")
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text("processor : 0\n"
                       "Features : fp asimd asimddp i8mm sve sve2\n"
                       "processor : 1\n"
                       "Features : fp asimd asimddp\n")

    assert target_info.supplement_aarch64_features(
        {"fp-armv8", "neon", "dotprod", "i8mm", "sve", "sve2"},
        cpuinfo_path=cpuinfo,
    ) == {"fp-armv8", "neon", "dotprod"}


def test_supplement_features_is_noop_off_aarch64(monkeypatch):
    monkeypatch.setattr(target_info.platform, "system", lambda: "Linux")
    monkeypatch.setattr(target_info.platform, "machine", lambda: "x86_64")
    assert target_info.supplement_aarch64_features({"avx2"}) == {"avx2"}


def test_supplement_aarch64_features_uses_darwin_sysctl(monkeypatch):
    monkeypatch.setattr(target_info.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(target_info.platform, "machine", lambda: "arm64")

    class Result:
        returncode = 0

        def __init__(self, enabled):
            self.stdout = "1\n" if enabled else "0\n"

    reported = {
        "hw.optional.arm.FEAT_DotProd": True,
        "hw.optional.arm.FEAT_FP16": True,
        "hw.optional.arm.FEAT_BF16": True,
        "hw.optional.arm.FEAT_I8MM": True,
    }

    def run(command, **_kwargs):
        return Result(reported[command[-1]])

    monkeypatch.setattr(target_info.subprocess, "run", run)
    assert target_info.supplement_aarch64_features({"neon"}) == {
        "neon",
        "dotprod",
        "fullfp16",
        "bf16",
        "i8mm",
    }


def test_fixed_i8mm_mode_is_used_without_sve2(monkeypatch):
    backend = object.__new__(CPUBackend)
    backend.cpu_arch = "aarch64"
    backend.cpu_features = {"neon", "dotprod", "i8mm"}
    backend.sve_vector_bits = 0
    monkeypatch.delenv("TRITON_CPU_DISABLE_SVE2_I8MM", raising=False)
    monkeypatch.delenv("TRITON_CPU_FIXED_I8MM", raising=False)

    assert backend.supports_fixed_i8mm()
    assert backend.use_fixed_i8mm()
    assert not backend.use_sve2_i8mm()

    backend.cpu_features.add("sve2")
    backend.sve_vector_bits = 128
    assert backend.supports_sve2_i8mm()
    assert backend.use_sve2_i8mm()
    assert not backend.use_fixed_i8mm()


def test_a720_bf16_store_workaround_requires_deployment_profile(monkeypatch):
    backend = object.__new__(CPUBackend)
    backend.cpu_arch = "aarch64"
    backend.cpu_name = "cortex-a720"

    monkeypatch.delenv("TRITON_CPU_A720_BF16_STORE_WORKAROUND", raising=False)
    generic = backend.parse_options({})
    assert not generic.a720_bf16_store_workaround

    monkeypatch.setenv("TRITON_CPU_A720_BF16_STORE_WORKAROUND", "1")
    profiled = backend.parse_options({})
    assert profiled.a720_bf16_store_workaround
    assert generic.hash() != profiled.hash()

    backend.cpu_name = "apple-m4"
    assert not backend.parse_options({}).a720_bf16_store_workaround


def test_arm_assembler_flags_select_host_compatible_isa(monkeypatch):
    backend = object.__new__(CPUBackend)
    backend.cpu_arch = "arm64"
    backend.cpu_features = {"neon", "dotprod", "i8mm"}
    backend.sve_vector_bits = 0
    monkeypatch.delenv("TRITON_CPU_FIXED_I8MM", raising=False)
    monkeypatch.delenv("TRITON_CPU_DISABLE_SVE2_I8MM", raising=False)

    monkeypatch.setattr(cpu_compiler.platform, "system", lambda: "Darwin")
    assert backend.arm_assembler_flags() == ["-march=armv8.6-a+dotprod+i8mm"]

    monkeypatch.setattr(cpu_compiler.platform, "system", lambda: "Linux")
    assert backend.arm_assembler_flags() == ["-march=armv8.6-a+dotprod+i8mm"]

    backend.cpu_features.update(("bf16", "fullfp16"))
    assert backend.arm_assembler_flags() == ["-march=armv8.6-a+bf16+dotprod+fp16+i8mm"]

    backend.cpu_arch = "aarch64"
    backend.cpu_features.update({
        "aes",
        "crc",
        "lse",
        "sha2",
        "sha3",
        "sm4",
        "sve",
        "sve2",
        "sve-aes",
        "sve-sha3",
        "sve-sm4",
    })
    backend.sve_vector_bits = 128
    assert backend.arm_assembler_flags() == [
        "-march=armv8.6-a+aes+bf16+crc+dotprod+fp16+i8mm+lse+sha2+sha3+sm4"
        "+sve+sve2+sve2-aes+sve2-sha3+sve2-sm4"
    ]

    monkeypatch.setenv("TRITON_CPU_FIXED_I8MM", "1")
    assert not backend.use_sve2_i8mm()
    assert backend.use_fixed_i8mm()
    assert backend.arm_assembler_flags() == ["-march=armv8.6-a+aes+bf16+crc+dotprod+fp16+i8mm+lse+sha2+sha3+sm4"]
    target_features = set(backend.llvm_target_features().split(","))
    assert "+dotprod" in target_features and "+i8mm" in target_features
    assert "-sve2" in target_features

    monkeypatch.delenv("TRITON_CPU_FIXED_I8MM")
    monkeypatch.setenv("TRITON_CPU_DISABLE_SVE2_I8MM", "1")
    assert not backend.use_sve2_i8mm()
    assert not backend.use_fixed_i8mm()


@pytest.mark.parametrize("sve_vector_bits", (0, 128))
def test_streaming_features_are_not_implicitly_enabled(monkeypatch, sve_vector_bits):
    backend = object.__new__(CPUBackend)
    backend.cpu_arch = "aarch64"
    backend.cpu_features = {
        "neon",
        "dotprod",
        "i8mm",
        "sve",
        "sve2",
        "sme",
        "sme2",
    }
    backend.sve_vector_bits = sve_vector_bits
    monkeypatch.delenv("TRITON_CPU_FIXED_I8MM", raising=False)
    monkeypatch.delenv("TRITON_CPU_DISABLE_SVE2_I8MM", raising=False)

    target_features = set(backend.llvm_target_features().split(","))
    assert "+sme" not in target_features
    assert "+sme2" not in target_features
    expected_prefix = "+" if sve_vector_bits == 128 else "-"
    assert f"{expected_prefix}sve" in target_features
    assert f"{expected_prefix}sve2" in target_features


def test_normalize_darwin_aarch64_smmla_syntax():
    assembly = "\tsmmla.4s v23, v24, v20\n\tret\n"

    assert _normalize_darwin_aarch64_assembly(assembly) == ("\tsmmla v23.4s, v24.16b, v20.16b\n\tret\n")


@pytest.mark.parametrize("prctl_value, expected", [(16, 128), (32 | 0x40000, 256), (-1, 0)])
def test_get_sve_vector_bits(monkeypatch, prctl_value, expected):
    monkeypatch.setattr(target_info.platform, "system", lambda: "Linux")
    monkeypatch.setattr(target_info.platform, "machine", lambda: "aarch64")
    assert target_info.get_sve_vector_bits(lambda *_: prctl_value) == expected
