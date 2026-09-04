import functools
import hashlib
import os
import platform
import re
import tempfile
from pathlib import Path

from dataclasses import dataclass
from types import ModuleType
from typing import Any, Dict, Optional, Tuple

from triton._C.libtriton import cpu, ir, llvm, passes, getenv_bool
from triton.backends.compiler import BaseBackend, GPUTarget, Language
from triton.runtime.build import _build
import triton.backends.cpu.driver as cpu_driver
from triton.backends.cpu.target_info import get_sve_vector_bits, supplement_aarch64_features


def min_dot_size(target: GPUTarget):
    # Decode GEMV naturally expresses a [1, K] x [K, N] dot.  The Arm
    # lowering has an M=1 SDOT path; keep the upstream lower bound elsewhere.
    minimum = (1, 4, 4) if target.arch in ("aarch64", "arm64", "armv8") else (4, 4, 4)
    return lambda lhsType, rhsType: minimum


VecLib = cpu.passes.ttcpuir.VecLib
Ukernels = cpu.passes.ttcpuir.Ukernels

# LLVM and GNU/Clang use different names for a few AArch64 architecture
# extensions. Any feature LLVM may use while emitting assembly must also be
# enabled for the system assembler; a later ``-march`` otherwise overrides
# ``-mcpu=native`` and can reject valid host instructions such as EOR3.
_AARCH64_MARCH_EXTENSIONS = {
    "aes": "aes",
    "bf16": "bf16",
    "crc": "crc",
    "dotprod": "dotprod",
    "fullfp16": "fp16",
    "i8mm": "i8mm",
    "lse": "lse",
    "sha2": "sha2",
    "sha3": "sha3",
    "sm4": "sm4",
    "sve": "sve",
    "sve2": "sve2",
    "sve-aes": "sve2-aes",
    "sve-sha3": "sve2-sha3",
    "sve-sm4": "sve2-sm4",
    "sve2-aes": "sve2-aes",
    "sve2-sha3": "sve2-sha3",
    "sve2-sm4": "sve2-sm4",
}


def _normalize_darwin_aarch64_assembly(assembly: str) -> str:
    """Translate LLVM's GNU-style I8MM spelling for AppleClang."""
    return re.sub(
        r"\bsmmla\.4s\s+v(\d+),\s*v(\d+),\s*v(\d+)",
        r"smmla v\1.4s, v\2.16b, v\3.16b",
        assembly,
    )


@dataclass(frozen=True)
class CPUOptions:
    # GPU-specific options are used in several places.
    # For now, we just provide dummy values.
    backend_name: str = "cpu"
    # These options provide compatibility with GPU kernel calls.
    # All of them are ignored.
    num_warps: int = 0
    num_stages: int = 0
    num_ctas: int = 0
    # Max number of threads to be used for a kernel call.
    # Zero value is used to utilize all available CPU cores.
    num_cpu_threads: int = 0
    cluster_dims: tuple = (1, 1, 1)
    extern_libs: dict = None
    debug: bool = False
    supported_fp8_dtypes: Tuple[str] = ("fp8e5", "fp8e5b16", "fp8e4nv")
    deprecated_fp8_dtypes: Tuple[str] = ()
    allowed_dot_input_precisions: Tuple[str] = ("ieee", "tf32", "tf32x3")
    deprecated_fp8_dot_operand_dtypes: Tuple[str, ...] = ()
    allow_fp8e4nv: bool = True
    allow_fp8e4b15: bool = True
    enable_fp_fusion: bool = True
    launch_cooperative_grid: bool = False
    max_num_imprecise_acc_default: int = 0
    enable_fast_math: bool = True
    vec_lib: Optional[str] = 'libsleef'
    # TODO: Try to enable it.
    sanitize_overflow: bool = False
    instrumentation_mode: str = ""

    # TODO: We may introduce CPU-specific options like # of cores.
    ukernels: str = None
    assume_in_bounds: bool = False
    # Cortex-A720 deployment profiles may enable the store-pair workaround.
    # Keep microarchitecture policy out of automatic ISA selection: Linux can
    # report the same LLVM CPU name for heterogeneous A720/A520 affinities.
    a720_bf16_store_workaround: bool = False

    def __post_init__(self):
        pass

    def hash(self):
        hash_dict = dict(self.__dict__)
        key = "_".join([f"{name}-{val}" for name, val in sorted(hash_dict.items())])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def get_vec_lib(self) -> VecLib:
        if self.vec_lib is None:
            return None
        # Parse enum from str here (instead of in parse_options()) because the options have to be JSON-serializable,
        # and pybind enums are not serializable.
        vec_lib = VecLib.__members__.get(self.vec_lib, None)
        if vec_lib is None:
            raise ValueError(
                f"Unexpected value for vec_lib: {self.vec_lib}, should be one of {{{', '.join(VecLib.__members__.keys())}}}"
            )
        return vec_lib

    def get_ukernels(self) -> Ukernels:
        raw_ukernels = os.getenv("TRITON_CPU_UKERNELS_LIB", self.ukernels)
        if raw_ukernels is None or raw_ukernels == "None":
            return None
        ukernels = Ukernels.__members__.get(raw_ukernels, None)
        if ukernels is None:
            raise ValueError(
                f"Unexpected value for ukernels: {raw_ukernels}, should be one of {{{', '.join(Ukernels.__members__.keys())}}}"
            )

        if ukernels == Ukernels.OneDNN and not cpu.onednn_available():
            import warnings
            # Warns on each compileation
            warnings.simplefilter('once', category=UserWarning)
            warnings.warn(
                "Warning! Triton build was made without OneDNN support. Check if \"CMAKE_PREFIX_PATH\" contains path to OneDNN during build. \n\t -------OneDNN will NOT be used-------",
                stacklevel=1)
            return None

        if ukernels == Ukernels.XSMM and not cpu.xsmm_available():
            import warnings
            # Warns on each compileation
            warnings.simplefilter('once', category=UserWarning)
            warnings.warn(
                "Warning! Triton build was made without XSMM support. Check if \"CMAKE_PREFIX_PATH\" contains path to XSMM during build. \n\t -------XSMM will NOT be used-------",
                stacklevel=1)
            return None

        return ukernels


class CPUBackend(BaseBackend):

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == "cpu"

    def __init__(self, target: tuple) -> None:
        super().__init__(target)
        self.binary_ext = "so"
        self.cpu_arch = llvm.get_cpu_tripple().split("-")[0]
        self.cpu_name = llvm.get_cpu_name()
        self.cpu_features = supplement_aarch64_features(llvm.get_cpu_features())
        self.sve_vector_bits = get_sve_vector_bits() if "sve" in self.cpu_features else 0
        if 'amx-tile' in self.cpu_features:
            if not cpu.enable_amx():
                import warnings
                warnings.warn("Warning! Couldn't enable AMX for the process. AMX optimizations are disabled.")
                self.cpu_features.discard('amx-tile')
                self.cpu_features.discard('amx-int8')
                self.cpu_features.discard('amx-fp16')
                self.cpu_features.discard('amx-bf16')

    def supports_sve2_i8mm(self) -> bool:
        return (self.cpu_arch in ("aarch64", "arm64", "armv8") and "sve2" in self.cpu_features
                and "i8mm" in self.cpu_features and self.sve_vector_bits == 128)

    def use_sve2_i8mm(self) -> bool:
        return (self.supports_sve2_i8mm() and not getenv_bool("TRITON_CPU_FIXED_I8MM", False)
                and not getenv_bool("TRITON_CPU_DISABLE_SVE2_I8MM", False))

    def supports_fixed_i8mm(self) -> bool:
        return (self.cpu_arch in ("aarch64", "arm64", "armv8") and "dotprod" in self.cpu_features
                and "i8mm" in self.cpu_features)

    def use_fixed_i8mm(self) -> bool:
        return (self.supports_fixed_i8mm()
                and (getenv_bool("TRITON_CPU_FIXED_I8MM", False) or not self.supports_sve2_i8mm())
                and not getenv_bool("TRITON_CPU_DISABLE_SVE2_I8MM", False))

    def arm_assembler_flags(self) -> list[str]:
        features = set(self.cpu_features)
        if self.use_fixed_i8mm():
            features = {feature for feature in features if not feature.startswith("sve")}
        elif not self.supports_sve2_i8mm() and not self.supports_fixed_i8mm():
            return []

        extensions = {
            assembler_name
            for feature, assembler_name in _AARCH64_MARCH_EXTENSIONS.items()
            if feature in features
        }
        return ["-march=armv8.6-a+" + "+".join(sorted(extensions))]

    def llvm_target_features(self) -> str:
        if self.cpu_arch not in ("aarch64", "arm64", "armv8"):
            return ""
        fixed_width = self.use_fixed_i8mm()
        features = []
        for feature in sorted(self.cpu_features):
            is_streaming = (feature == "sme" or feature.startswith("sme-") or feature == "sme2"
                            or feature.startswith("sme2-"))
            # SME requires an explicitly streaming-aware lowering and ABI.
            # Ordinary Triton CPU kernels must not inherit SME merely because
            # the JIT host advertises it.  A future SME backend should create
            # and dispatch a separate kernel variant with the required attrs.
            if is_streaming:
                continue
            # Forcing the fixed path on an SVE machine must model a genuinely
            # non-SVE target. Otherwise LLVM can introduce scalable vector
            # instructions while optimizing ordinary integer expressions even
            # though the custom dot lowering itself emitted only Neon.
            disable = fixed_width and (feature == "sve" or feature.startswith("sve-") or feature == "sve2"
                                       or feature.startswith("sve2-"))
            features.append(("-" if disable else "+") + feature)
        return ",".join(features)

    def parse_options(self, opts) -> Any:
        args = {k: opts[k] for k in CPUOptions.__dataclass_fields__.keys() if k in opts}
        if "enable_fast_math" not in args:
            args["enable_fast_math"] = os.getenv("TRITON_CPU_FAST_MATH", "1") != "0"
        if "supported_fp8_dtypes" not in args:
            supported_fp8_dtypes = set(CPUOptions.supported_fp8_dtypes)
            args["supported_fp8_dtypes"] = tuple(sorted(supported_fp8_dtypes))
        if "assume_in_bounds" not in args:
            args["assume_in_bounds"] = getenv_bool("TRITON_CPU_ASSUME_IN_BOUNDS", False)
        if "a720_bf16_store_workaround" not in args:
            requested = getenv_bool("TRITON_CPU_A720_BF16_STORE_WORKAROUND", False)
            args["a720_bf16_store_workaround"] = (requested and self.cpu_arch in ("aarch64", "arm64", "armv8")
                                                  and self.cpu_name.lower() == "cortex-a720")
        return CPUOptions(**args)

    def pack_metadata(self, metadata):
        return metadata

    def get_codegen_implementation(self, options):
        codegen_fns = {"min_dot_size": min_dot_size(self.target)}
        return codegen_fns

    def get_module_map(self) -> Dict[str, ModuleType]:
        from triton.language.extra.cpu import libdevice
        return {"triton.language.extra.libdevice": libdevice}

    def load_dialects(self, ctx):
        cpu.load_dialects(ctx)

    @staticmethod
    def make_ttir(mod, metadata, opt):
        # This is the same as the Nvidia backend.
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_combine(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_licm(pm)
        passes.common.add_symbol_dce(pm)
        # Honor tl.range(..., loop_unroll_factor=...).  The GPU backends run
        # this standard TTIR pass, but upstream Triton-CPU 3.7.2 omitted it,
        # leaving the annotation in IR without changing generated code.
        passes.ttir.add_loop_unroll(pm)
        pm.run(mod, "make_ttir")
        return mod

    @staticmethod
    def make_ttcir(mod, metadata, opt):
        # TTIR -> TTCIR
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        cpu.passes.ttcpuir.add_scalarize(pm, True)
        cpu.passes.ttcpuir.add_convert_memory_ops(pm, True, opt.assume_in_bounds)
        cpu.passes.ttcpuir.add_convert_ptr_ops(pm)
        cpu.passes.ttcpuir.add_convert_elementwise_ops(pm)
        cpu.passes.ttcpuir.add_convert_elem_manip_ops(pm)
        cpu.passes.ttcpuir.add_convert_dot_op(pm)
        cpu.passes.ttcpuir.add_convert_histogram_op(pm)
        # Keep multidimensional reductions intact until the target pass.
        # Besides the existing BF16 mul/sum matcher, this lets AArch64 fuse
        # packed INT8 4x2x4 reductions into two SDOT accumulators instead of
        # losing the microtile structure in scalar shuffle trees.
        cpu.passes.ttcpuir.add_convert_reduction_op(pm, True, True)
        cpu.passes.ttcpuir.add_convert_scan_op(pm)
        cpu.passes.ttcpuir.add_convert_cf_ops(pm)
        cpu.passes.ttcpuir.add_convert_atomic_ops(pm)
        cpu.passes.ttcpuir.add_convert_debug_ops(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        passes.common.add_canonicalizer(pm)
        pm.run(mod, "make_ttcir")
        metadata["cluster_dims"] = (opt.cluster_dims[0], opt.cluster_dims[1], opt.cluster_dims[2])
        return mod

    def make_tttcir(self, mod, metadata, opt):
        # TTCIR -> Target TTCIR
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        cpu.passes.ttcpuir.add_triton_cpu_canonicalizer(pm)
        cpu.passes.ttcpuir.add_optimize_masks(pm)
        passes.common.add_canonicalizer(pm)
        if (ukernels := opt.get_ukernels()):
            # For further analysis simplification
            cpu.passes.ttcpuir.add_loop_invariant_code_motion(pm)
            cpu.passes.ttcpuir.add_convert_dot_to_ukernels(pm, ukernels)
            passes.common.add_canonicalizer(pm)
            passes.common.add_cse(pm)
        # LLVM reports Darwin AArch64 triples as ``arm64-apple-*``. Treat that
        # spelling exactly like ``aarch64`` here; otherwise the packed INT8
        # recognizer is silently disabled and lowers to generic vector math.
        is_aarch64_neon = (self.cpu_arch in ("aarch64", "arm64", "armv8") and 'neon' in self.cpu_features)
        convert_bf16_dot_product = (is_aarch64_neon and 'fp-armv8' in self.cpu_features and 'bf16' in self.cpu_features)
        convert_i8_dot_product = is_aarch64_neon and 'dotprod' in self.cpu_features
        if convert_bf16_dot_product or convert_i8_dot_product:
            use_horizontal_sum = os.getenv("TRITON_CPU_DOT_PROD_HORIZ_SUM", "1") == "1"
            cpu.passes.ttcpuir.add_convert_dot_product(pm, use_horizontal_sum, convert_bf16_dot_product,
                                                       convert_i8_dot_product)
        if 'amx-tile' in self.cpu_features:
            amx_int8 = 'amx-int8' in self.cpu_features
            # amx_fp16 = 'amx-fp16' in self.cpu_features
            # FP16 support is not in AMX dialect yet
            amx_fp16 = False
            amx_bf16 = 'amx-bf16' in self.cpu_features
            cpu.passes.ttcpuir.add_convert_dot_to_amx(pm, amx_int8, amx_fp16, amx_bf16)
        if 'avx512f' in self.cpu_features:
            cpu.passes.ttcpuir.add_convert_dot_to_fma(pm)
        if self.use_sve2_i8mm():
            cpu.passes.ttcpuir.add_convert_dot_to_sve2_i8mm(pm, False)
        elif self.use_fixed_i8mm():
            # Fixed-width Arm targets use the M1 and packed Q4/Q8 NEON
            # SDOT/SMMLA recognizers.  Generic M>=8 candidates remain off
            # because their lowering requires SVE vectors.
            cpu.passes.ttcpuir.add_convert_dot_to_sve2_i8mm(pm, True)
        cpu.passes.ttcpuir.add_convert_dot_generic(pm)
        promote_bf16_to_fp32 = self.cpu_arch == "x86_64" and "avx512bf16" not in self.cpu_features
        # We don't have any lowering for mixed precision matmuls, so always use casts for now
        convert_mixed_precision_matmul = True
        # We don't have math lib functions for FP8, FP16, BF16. Promote such operations to FP32.
        promote_lib_math_to_fp32 = True
        cpu.passes.ttcpuir.add_convert_unsupported_ops(pm, promote_bf16_to_fp32, convert_mixed_precision_matmul,
                                                       promote_lib_math_to_fp32)
        decompose_bf16_conv = self.cpu_arch == "x86_64" and "avx512bf16" not in self.cpu_features
        decompose_fp8_conv = True
        cpu.passes.ttcpuir.add_decompose_fp_conversions(pm, decompose_bf16_conv, decompose_fp8_conv)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        passes.common.add_canonicalizer(pm)
        pm.run(mod, "make_tttcir")
        return mod

    def make_llir(self, src, metadata, options):
        # warp-specialization mutates num_warps
        num_warp_groups = src.get_int_attr("triton_gpu.num-warp-groups-per-cta")
        if num_warp_groups is not None:
            metadata["num_warps"] *= num_warp_groups
        metadata["threads_per_warp"] = 1
        mod = src
        # TritonCPU -> LLVM-IR (MLIR)
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        if options.get_ukernels() == Ukernels.OneDNN:
            cpu.passes.ttcpuir.add_ukernels_to_onednn_llvmir(pm)
        if options.get_ukernels() == Ukernels.XSMM:
            cpu.passes.ttcpuir.add_ukernels_to_xsmm_llvmir(pm)
        cpu.passes.ttcpuir.add_lower_vector_multi_dim(pm)
        cpu.passes.ttcpuir.add_expand_strided_metadata(pm)
        cpu.passes.ttcpuir.add_vector_to_scf(pm, True, 1, False)
        cpu.passes.ttcpuir.add_lower_affine(pm)
        passes.convert.add_scf_to_cf(pm)
        passes.convert.add_index_to_llvmir(pm)
        cpu.passes.ttcpuir.add_func_op_to_llvmir(pm)
        cpu.passes.ttcpuir.add_program_id_to_llvmir(pm)
        cpu.passes.ttcpuir.add_memory_op_to_llvmir(pm)
        cpu.passes.ttcpuir.add_atomic_ops_to_llvmir(pm)
        cpu.passes.ttcpuir.add_debug_ops_to_llvmir(pm)

        vec_lib_requirements = {
            VecLib.libsleef: {"neon", "sse", "avx"},
            VecLib.libmvec: {"avx512f"},
        }
        if (vec_lib := options.get_vec_lib()) and vec_lib_requirements[vec_lib] & self.cpu_features:
            cpu.passes.ttcpuir.add_math_to_vec_lib(pm, vec_lib, self.cpu_features)

        passes.convert.add_math_to_llvmir(pm)
        cpu.passes.ttcpuir.add_math_to_libm(pm)
        cpu.passes.ttcpuir.add_vector_to_llvmir(pm, options.enable_fast_math)
        cpu.passes.ttcpuir.add_memref_to_llvmir(pm)
        passes.convert.add_reconcile_unrealized(pm)
        passes.convert.add_arith_to_llvmir(pm)
        # passes.convert.add_cf_to_llvmir(pm)
        cpu.passes.ttcpuir.add_func_to_llvmir(pm)
        cpu.passes.ttcpuir.add_ub_to_llvmir(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        # Cortex-A720 has a reproducible low-12-bit false dependency when a
        # 256-bit BF16 vector store becomes STP Q,Q and its two store addresses
        # overlap a following 32-byte load's page offsets.  Run this after CSE
        # so in-place kernels can be excluded by exact load/store addresses.
        # Keeping that one independent-output store as two STRs removes the
        # alignment cliff while preserving the 16-element compute tile.
        if options.a720_bf16_store_workaround:
            cpu.passes.ttcpuir.add_mark_wide_bf16_stores_volatile(pm)
        if os.environ.get("TRITON_DISABLE_LINE_INFO", "0") == "0":
            passes.llvmir.add_di_scope(pm)
        pm.run(mod, "make_llir")

        # Find kernel fn
        kernel_names = cpu.find_kernel_names(mod)
        assert len(kernel_names) == 1, f"expected exactly 1 kernel in a module, got {kernel_names}"

        # LLVM-IR (MLIR) -> LLVM-IR (LLVM)
        llvm.init_targets()
        context = llvm.context()
        llvm_mod = llvm.to_module(mod, context)
        if llvm_mod is None:
            raise RuntimeError("Failed to convert to LLVM IR")
        llvm.set_host_target(llvm_mod)
        #if options.extern_libs:
        #    paths = [path for (name, path) in options.extern_libs]
        #   llvm.link_extern_libs(llvm_mod, paths)
        llvm.optimize_module(llvm_mod, llvm.OPTIMIZE_O3)
        # Get some metadata
        metadata["shared"] = 0
        metadata["name"] = kernel_names[0]
        ret = str(llvm_mod)
        del llvm_mod
        del context
        return ret

    def make_asm(self, src, metadata, options):
        return llvm.translate_to_host_asm(src, options.enable_fp_fusion, options.enable_fast_math,
                                          self.llvm_target_features())

    def make_so(self, src, metadata, options):
        with tempfile.TemporaryDirectory() as tmpdir:
            asm_path = os.path.join(tmpdir, "kernel.s")
            if platform.system() == "Darwin" and self.cpu_arch in (
                    "aarch64",
                    "arm64",
                    "armv8",
            ):
                src = _normalize_darwin_aarch64_assembly(src)
            Path(asm_path).write_text(src)
            lib_dirs = cpu_driver.library_dirs
            libs = ["m", "TritonCPURuntime", "sleef"]
            # GCC 12 does not always infer SVE i8mm from -mcpu=native even
            # when Linux reports the feature.  LLVM has already emitted the
            # assembly, so this flag only teaches the system assembler which
            # instructions are valid for the detected target.
            ccflags = self.arm_assembler_flags()
            so = _build("kernel", asm_path, tmpdir, lib_dirs, cpu_driver.include_dirs, libs, ccflags)
            with open(so, "rb") as f:
                return f.read()

    def add_stages(self, stages, options, language):
        assert language == Language.TRITON
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["ttcir"] = lambda src, metadata: self.make_ttcir(src, metadata, options)
        stages["tttcir"] = lambda src, metadata: self.make_tttcir(src, metadata, options)
        stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options)
        stages["asm"] = lambda src, metadata: self.make_asm(src, metadata, options)
        stages["so"] = lambda src, metadata: self.make_so(src, metadata, options)

    @functools.lru_cache()
    def hash(self):
        features = ",".join(sorted(self.cpu_features))
        return (f"{self.cpu_arch}-{self.cpu_name}-{features}-sve{self.sve_vector_bits}"
                f"-sve2-i8mm-{int(self.use_sve2_i8mm())}"
                f"-fixed-i8mm-{int(self.use_fixed_i8mm())}")
