import ctypes
import platform
import subprocess
from pathlib import Path
from typing import Callable, Iterable, Optional, Set

# Linux /proc/cpuinfo names mapped to LLVM AArch64 target-feature names. Keep
# this list to architectural extensions that can affect generated instructions.
_LINUX_AARCH64_FEATURE_MAP = {
    "fp": "fp-armv8",
    "asimd": "neon",
    "asimddp": "dotprod",
    # Linux reports scalar and Advanced-SIMD FP16 separately.  LLVM's
    # ``fullfp16`` feature is the one that permits vector .8h arithmetic.
    "asimdhp": "fullfp16",
    "bf16": "bf16",
    "i8mm": "i8mm",
    "sve": "sve",
    "sve2": "sve2",
}

_DARWIN_AARCH64_FEATURE_MAP = {
    "hw.optional.arm.FEAT_DotProd": "dotprod",
    "hw.optional.arm.FEAT_FP16": "fullfp16",
    "hw.optional.arm.FEAT_BF16": "bf16",
    "hw.optional.arm.FEAT_I8MM": "i8mm",
}

_PR_SVE_GET_VL = 51
_PR_SVE_VL_LEN_MASK = 0xFFFF


def parse_linux_cpuinfo_features(text: str) -> Set[str]:
    """Return features common to every CPU listed in /proc/cpuinfo.

    Using the intersection matters on heterogeneous systems: compiled code
    may run on any CPU allowed by the process affinity mask, so dispatch must
    not be based on a feature present on only one core.
    """
    per_cpu = []
    for line in text.splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip().lower() == "features":
            per_cpu.append(set(value.split()))
    if not per_cpu:
        return set()
    return set.intersection(*per_cpu)


def supplement_aarch64_features(
        llvm_features: Iterable[str],
        *,
        cpuinfo_path: Path = Path("/proc/cpuinfo"),
) -> Set[str]:
    """Supplement LLVM host detection with OS-reported AArch64 features."""
    features = set(llvm_features)
    if platform.machine() not in ("aarch64", "arm64"):
        return features
    if platform.system() == "Darwin":
        for oid, llvm_name in _DARWIN_AARCH64_FEATURE_MAP.items():
            try:
                result = subprocess.run(["sysctl", "-n", oid], capture_output=True, text=True)
            except OSError:
                continue
            if result.returncode == 0 and result.stdout.strip() == "1":
                features.add(llvm_name)
        return features
    if platform.system() != "Linux":
        return features
    try:
        cpuinfo = cpuinfo_path.read_text()
    except OSError:
        return features
    linux_features = parse_linux_cpuinfo_features(cpuinfo)
    if not linux_features:
        return features

    # LLVM host detection and /proc/cpuinfo can observe different CPUs on a
    # heterogeneous machine.  Treat the OS-reported intersection as
    # authoritative for every AArch64 feature that controls code generation:
    # retaining a feature seen by LLVM on only the current core can otherwise
    # produce an illegal instruction after the process migrates to another
    # allowed core.
    mapped_features = set(_LINUX_AARCH64_FEATURE_MAP.values())
    common_features = {
        llvm_name
        for linux_name, llvm_name in _LINUX_AARCH64_FEATURE_MAP.items()
        if linux_name in linux_features
    }
    features.difference_update(mapped_features - common_features)
    features.update(common_features)
    return features


def get_sve_vector_bits(prctl: Optional[Callable[[int, int, int, int, int], int]] = None, ) -> int:
    """Return the current Linux thread's SVE vector length, or zero.

    SVE vector length is process/thread state, not merely a CPU property.  A
    lowering specialized for one vector length must therefore query PR_SVE_GET_VL
    instead of inferring it from the presence of the SVE feature.
    """
    if platform.system() != "Linux" or platform.machine() not in ("aarch64", "arm64"):
        return 0
    try:
        if prctl is None:
            prctl = ctypes.CDLL(None, use_errno=True).prctl
        value = prctl(_PR_SVE_GET_VL, 0, 0, 0, 0)
    except (AttributeError, OSError):
        return 0
    if value < 0:
        return 0
    return (value & _PR_SVE_VL_LEN_MASK) * 8
