"""
TLE-CPU: NEON intrinsics support for Triton CPU backend.

Provides a mechanism to register C NEON functions that get compiled and linked
into Triton kernel .so files. This enables using NEON SDOT/I8MM instructions
without going through MLIR vector lowering (which adds shuffle overhead).

Usage:
    from triton.language.extra.cpu import neon

    # Register a C function (done once, at import time)
    neon.register_c_function("sdot_m1_kernel", '''
    #include <arm_neon.h>
    void sdot_m1_kernel(const int8_t* A, const int8_t* B_packed,
                         int32_t* C, int64_t K4, int64_t N4,
                         int64_t nb_start, int64_t nb_count) {
        // ... NEON SDOT implementation ...
    }
    ''')

    # In a Triton kernel, the registered function is available as an extern
    # symbol that gets linked at make_so time.

The C code is compiled with gcc -O3 -march=armv8.2-a+dotprod -fPIC during
Triton's make_so stage and linked into the kernel .so.
"""

import os
import hashlib
import tempfile
import subprocess
import platform
from pathlib import Path

# Registry of C functions to compile and link
_C_FUNCTION_REGISTRY = {}
# Cache of compiled .o files (keyed by content hash)
_COMPILED_OBJECT_CACHE = {}


def register_c_function(name, c_source, extra_cflags=None):
    """Register a C function to be compiled and linked into Triton kernels.

    Args:
        name: Function name (must match the C function name)
        c_source: Complete C source code string
        extra_cflags: Additional compiler flags (e.g., ['-march=armv8.2-a+dotprod'])
    """
    _C_FUNCTION_REGISTRY[name] = {
        'source': c_source,
        'cflags': extra_cflags or [],
    }


def get_registered_functions():
    """Return all registered C functions."""
    return _C_FUNCTION_REGISTRY


def compile_c_to_object(c_source, extra_cflags=None, cache=True):
    """Compile C source to object file (.o), with caching.

    Returns path to .o file.
    """
    content_hash = hashlib.md5(c_source.encode()).hexdigest()
    if cache and content_hash in _COMPILED_OBJECT_CACHE:
        obj_path = _COMPILED_OBJECT_CACHE[content_hash]
        if os.path.exists(obj_path):
            return obj_path

    import shutil
    is_macos = platform.system() == "Darwin"
    cc = os.environ.get("CC")
    if cc is None:
        if is_macos:
            # macOS: /usr/bin/gcc is an Apple clang shim that does NOT support
            # -fopenmp. Use Homebrew LLVM clang (libomp-capable) instead.
            cc = ("/opt/homebrew/opt/llvm@18/bin/clang"
                  if os.path.exists("/opt/homebrew/opt/llvm@18/bin/clang")
                  else (shutil.which("clang") or shutil.which("gcc")))
        else:
            cc = shutil.which("gcc") or shutil.which("clang")
    if cc is None:
        raise RuntimeError("No C compiler found")

    # Use a persistent cache directory
    cache_dir = os.path.join(tempfile.gettempdir(), "triton_tle_cpu_cache")
    os.makedirs(cache_dir, exist_ok=True)
    obj_path = os.path.join(cache_dir, f"tle_{content_hash}.o")

    if os.path.exists(obj_path):
        _COMPILED_OBJECT_CACHE[content_hash] = obj_path
        return obj_path

    src_path = os.path.join(cache_dir, f"tle_{content_hash}.c")
    with open(src_path, 'w') as f:
        f.write(c_source)

    machine = platform.machine()
    cc_cmd = [cc, src_path, "-c", "-O3", "-fPIC", "-o", obj_path]

    if machine in ("aarch64", "arm64"):
        if is_macos:
            # Apple M-series: no SVE; apple-m3 implies i8mm/dotprod/bf16/fp16
            # (clang-18 rejects -mcpu=apple-m4). libomp is keg-only -> add its
            # include dir so <omp.h> resolves.
            cc_cmd += ["-mcpu=apple-m3", "-fopenmp"]
            omp_inc = "/opt/homebrew/opt/libomp/include"
            if os.path.isdir(omp_inc):
                cc_cmd += [f"-I{omp_inc}"]
        else:
            cc_cmd += ["-march=armv8.2-a+dotprod+i8mm+bf16", "-fopenmp"]

    if extra_cflags:
        cc_cmd += extra_cflags

    subprocess.check_call(cc_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    _COMPILED_OBJECT_CACHE[content_hash] = obj_path
    return obj_path


def get_all_object_files():
    """Compile all registered C functions and return deduplicated list of .o paths."""
    obj_files = set()
    for name, info in _C_FUNCTION_REGISTRY.items():
        obj = compile_c_to_object(info['source'], info['cflags'])
        obj_files.add(obj)
    return list(obj_files)


# ============================================================================
# Pre-registered NEON SDOT functions for INT8 GEMV
# ============================================================================

_SDOT_GEMV_SOURCE = r"""
#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

/* Pack weights: B[K,N] row-major → B_packed[K//4, N//4, 4, 4] SDOT format */
void tle_sdot_pack_weights(const int8_t *B, int8_t *B_packed,
                            int64_t K, int64_t N) {
    int64_t K4 = K / 4, N4 = N / 4;
    #pragma omp parallel for collapse(2)
    for (int64_t kb = 0; kb < K4; kb++) {
        for (int64_t nb = 0; nb < N4; nb++) {
            int8_t *dst = B_packed + (kb * N4 + nb) * 16;
            for (int ni = 0; ni < 4; ni++)
                for (int ki = 0; ki < 4; ki++)
                    dst[ni * 4 + ki] = B[(kb * 4 + ki) * N + nb * 4 + ni];
        }
    }
}

/* Core SDOT GEMV range: K-outer loop with accumulator array */
static void sdot_range(const int8_t *A, const int8_t *B_packed,
                        int32_t *C, int64_t K4, int64_t N4,
                        int64_t nb_start, int64_t nb_count) {
    /* Heap alloc for large N (lm_head: ~4748 groups per thread) */
    int32x4_t *acc = (int32x4_t*)malloc(nb_count * sizeof(int32x4_t));
    for (int64_t i = 0; i < nb_count; i++)
        acc[i] = vdupq_n_s32(0);

    for (int64_t kb = 0; kb < K4; kb++) {
        int32_t a4;
        memcpy(&a4, A + kb * 4, 4);
        int8x16_t av = vreinterpretq_s8_s32(vdupq_n_s32(a4));
        const int8_t *bp = B_packed + (kb * N4 + nb_start) * 16;
        int64_t i = 0;
        for (; i + 4 <= nb_count; i += 4) {
            acc[i]   = vdotq_s32(acc[i],   av, vld1q_s8(bp)); bp += 16;
            acc[i+1] = vdotq_s32(acc[i+1], av, vld1q_s8(bp)); bp += 16;
            acc[i+2] = vdotq_s32(acc[i+2], av, vld1q_s8(bp)); bp += 16;
            acc[i+3] = vdotq_s32(acc[i+3], av, vld1q_s8(bp)); bp += 16;
        }
        for (; i < nb_count; i++) {
            acc[i] = vdotq_s32(acc[i], av, vld1q_s8(bp)); bp += 16;
        }
    }
    for (int64_t i = 0; i < nb_count; i++)
        vst1q_s32(C + (nb_start + i) * 4, acc[i]);
    free(acc);
}

/* INT8 GEMV: A[K] int8 × B_packed → C[N] int32, OMP parallel */
void tle_sdot_gemv_m1(const int8_t *A, const int8_t *B_packed,
                       int32_t *C, int64_t K, int64_t N) {
    int64_t K4 = K / 4, N4 = N / 4;
    #pragma omp parallel
    {
        int nt = omp_get_num_threads(), tid = omp_get_thread_num();
        int64_t chunk = (N4 + nt - 1) / nt;
        int64_t start = tid * chunk, count = chunk;
        if (start + count > N4) count = N4 - start;
        if (start < N4 && count > 0)
            sdot_range(A, B_packed, C, K4, N4, start, count);
    }
}

/* BF16 → FP32 inline */
static inline float32x4_t bf16_to_fp32(uint16x4_t bf16) {
    return vreinterpretq_f32_u32(vshll_n_u16(bf16, 16));
}

/* Fused: BF16 input → dynamic quant → SDOT GEMV → dequant → BF16 output */
void tle_sdot_gemv_m1_fused_bf16(const uint16_t *x_bf16,
                                   const int8_t *B_packed,
                                   const float *w_scale,
                                   uint16_t *out_bf16,
                                   int64_t K, int64_t N) {
    int64_t K4 = K / 4, N4 = N / 4;

    /* Step 1: Quantize BF16 → INT8 */
    int8_t x_int8[16384];
    float32x4_t vmax = vdupq_n_f32(0.0f);
    int64_t k = 0;
    for (; k + 4 <= K; k += 4) {
        float32x4_t f = bf16_to_fp32(vld1_u16(x_bf16 + k));
        vmax = vmaxq_f32(vmax, vabsq_f32(f));
    }
    float amax = vmaxvq_f32(vmax);
    for (; k < K; k++) {
        uint32_t bits = (uint32_t)x_bf16[k] << 16;
        float v; memcpy(&v, &bits, 4);
        float av = fabsf(v);
        if (av > amax) amax = av;
    }
    if (amax < 1e-8f) amax = 1e-8f;
    float x_scale = amax / 127.0f;
    float inv_scale = 127.0f / amax;
    k = 0;
    for (; k + 4 <= K; k += 4) {
        float32x4_t f = bf16_to_fp32(vld1_u16(x_bf16 + k));
        float32x4_t sc = vmulq_n_f32(f, inv_scale);
        int32x4_t r = vcvtnq_s32_f32(sc);
        int16x4_t n16 = vqmovn_s32(r);
        int8x8_t n8 = vqmovn_s16(vcombine_s16(n16, n16));
        vst1_lane_s32((int32_t*)(x_int8 + k), vreinterpret_s32_s8(n8), 0);
    }
    for (; k < K; k++) {
        uint32_t bits = (uint32_t)x_bf16[k] << 16;
        float v; memcpy(&v, &bits, 4);
        int32_t r = (int32_t)roundf(v * inv_scale);
        if (r > 127) r = 127; if (r < -128) r = -128;
        x_int8[k] = (int8_t)r;
    }

    /* Step 2+3: SDOT GEMV + Dequant (OMP parallel, per-thread buffer) */
    #pragma omp parallel
    {
        int nt = omp_get_num_threads(), tid = omp_get_thread_num();
        int64_t chunk = (N4 + nt - 1) / nt;
        int64_t start = tid * chunk, count = chunk;
        if (start + count > N4) count = N4 - start;
        if (start >= N4) count = 0;
        if (count > 0) {
            /* GEMV into thread-local heap buffer (lm_head: N=151936, 8 threads → ~19K cols) */
            int32_t *local_buf = (int32_t*)malloc(count * 4 * sizeof(int32_t));
            int32_t *local_c = local_buf - start * 4;
            sdot_range(x_int8, B_packed, local_c, K4, N4, start, count);

            /* Dequant: int32 → bf16 */
            float32x4_t xs = vdupq_n_f32(x_scale);
            int64_t n = start * 4, end = (start + count) * 4;
            for (; n + 4 <= end; n += 4) {
                float32x4_t oi = vcvtq_f32_s32(vld1q_s32(local_c + n));
                float32x4_t ws = vld1q_f32(w_scale + n);
                float32x4_t res = vmulq_f32(vmulq_f32(oi, xs), ws);
                vst1_u16(out_bf16 + n, vshrn_n_u32(vreinterpretq_u32_f32(res), 16));
            }
            for (; n < end; n++) {
                float r = (float)local_c[n] * x_scale * w_scale[n];
                uint32_t bits; memcpy(&bits, &r, 4);
                out_bf16[n] = (uint16_t)(bits >> 16);
            }
            free(local_buf);
        }
    }
}

#endif /* __aarch64__ && __ARM_NEON */
"""

# Auto-register on import
if platform.machine() in ("aarch64", "arm64"):
    register_c_function("tle_sdot_pack_weights", _SDOT_GEMV_SOURCE,
                        extra_cflags=["-funroll-loops"])
    register_c_function("tle_sdot_gemv_m1", _SDOT_GEMV_SOURCE,
                        extra_cflags=["-funroll-loops"])
    register_c_function("tle_sdot_gemv_m1_fused_bf16", _SDOT_GEMV_SOURCE,
                        extra_cflags=["-funroll-loops"])
