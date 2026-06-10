"""Host CPU feature detection for TLE-CPU kernel-variant selection.

CPU Triton JIT-compiles on the target machine, so feature detection at
import/JIT time is sufficient to pick the right TLE-Struct micro-kernel
family (the arm64 capability ladder):

    sme2 > sme > i8mm (SMMLA) > dotprod (SDOT) > baseline (tl.dot)

Use ``features()`` for the raw set, or ``best_int8_tier()`` for the ladder
rung. Environment variables can only DISABLE a feature, never enable one:
``TLE_CPU_DISABLE_SME`` / ``TLE_CPU_DISABLE_I8MM`` / ``TLE_CPU_DISABLE_DOTPROD``
(same names the C runtime honors, so Python selection and runtime dispatch
stay in agreement).
"""
import os
import platform
import subprocess
from functools import lru_cache

_SYSCTL_MAP = {
    "hw.optional.arm.FEAT_DotProd": "dotprod",
    "hw.optional.arm.FEAT_I8MM": "i8mm",
    "hw.optional.arm.FEAT_SME": "sme",
    "hw.optional.arm.FEAT_SME2": "sme2",
}

# /proc/cpuinfo "Features" tokens -> canonical names
_CPUINFO_MAP = {
    "asimddp": "dotprod",
    "i8mm": "i8mm",
    "sve2": "sve2",
    "sme": "sme",
    "sme2": "sme2",
}


def _env_disabled(name):
    v = os.environ.get(name)
    return v is not None and v != "0"


@lru_cache(maxsize=None)
def features():
    """Detected arm64 features as a set of canonical names (cached)."""
    feats = set()
    machine = platform.machine()
    if machine not in ("arm64", "aarch64"):
        return frozenset()
    system = platform.system()
    if system == "Darwin":
        for oid, name in _SYSCTL_MAP.items():
            try:
                out = subprocess.run(["sysctl", "-n", oid], capture_output=True,
                                     text=True)
                if out.returncode == 0 and out.stdout.strip() == "1":
                    feats.add(name)
            except OSError:
                pass
    elif system == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("Features"):
                        tokens = set(line.split(":", 1)[1].split())
                        for tok, name in _CPUINFO_MAP.items():
                            if tok in tokens:
                                feats.add(name)
                        break
        except OSError:
            pass

    if _env_disabled("TLE_CPU_DISABLE_DOTPROD"):
        feats -= {"dotprod", "i8mm", "sme", "sme2"}
    if _env_disabled("TLE_CPU_DISABLE_I8MM"):
        feats.discard("i8mm")
    if _env_disabled("TLE_CPU_DISABLE_SME"):
        feats -= {"sme", "sme2"}
    return frozenset(feats)


def has(name):
    return name in features()


def best_int8_tier():
    """Highest usable rung of the arm64 int8 ladder.

    Returns one of "sme2", "sme", "i8mm", "dotprod", or "baseline".
    "baseline" means: use the portable tl.dot (TLE-Lite) path.
    """
    f = features()
    for tier in ("sme2", "sme", "i8mm", "dotprod"):
        if tier in f:
            return tier
    return "baseline"
