/*
 * NEON SDOT runtime for M=1 INT8 GEMV.
 *
 * Compiled into TritonCPURuntime.so. Called from FlagGems dispatch layer
 * for high-performance M=1 INT8 GEMV with pre-packed weights.
 *
 * API:
 *   sdot_pack_weights(B_ptr, B_packed_ptr, K, N)
 *     B: [K, N] int8 (row-major)
 *     B_packed: [K//4, N//4, 4, 4] int8 (output, pre-allocated)
 *
 *   sdot_gemv_m1_prepacked(A_ptr, B_packed_ptr, C_ptr, K, N, N4)
 *     A: [K] int8, B_packed: SDOT format, C: [N] int32
 *
 *   sdot_gemv_m1_fused_bf16(x_bf16, B_packed, w_scale, out_bf16, K, N, N4)
 *     Fused: BF16 activation → dynamic quant → SDOT GEMV → dequant → BF16 output
 *     Eliminates Python-side abs/div/clamp/to overhead (~29ms → ~1ms)
 */

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#include <cmath>
#include <cstdint>
#include <cstring>
#include <omp.h>
#include <vector>

#if defined(_MSC_VER)
#define EXPORT __declspec(dllexport)
#elif defined(__GNUC__)
#define EXPORT __attribute__((visibility("default")))
#else
#define EXPORT
#endif

extern "C" {

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)

EXPORT void sdot_pack_weights(const int8_t *B, int8_t *B_packed,
                               int64_t K, int64_t N) {
  int64_t K4 = K / 4;
  int64_t N4 = N / 4;
  for (int64_t kb = 0; kb < K4; kb++) {
    for (int64_t nb = 0; nb < N4; nb++) {
      int8_t *dst = B_packed + (kb * N4 + nb) * 16;
      for (int ni = 0; ni < 4; ni++) {
        for (int ki = 0; ki < 4; ki++) {
          dst[ni * 4 + ki] = B[(kb * 4 + ki) * N + nb * 4 + ni];
        }
      }
    }
  }
}

static void sdot_gemv_range(const int8_t *A, const int8_t *B_packed,
                             int32_t *C, int64_t K4, int64_t N4,
                             int64_t nb_start, int64_t nb_count) {
  // Accumulator array. For lm_head (N=151936, 8 threads) need up to 4748 groups.
  // Use heap allocation for large counts, stack for small.
  std::vector<int32x4_t> acc_heap;
  int32x4_t acc_stack[2048];
  int32x4_t *acc;
  if (nb_count <= 2048) {
    acc = acc_stack;
  } else {
    acc_heap.resize(nb_count);
    acc = acc_heap.data();
  }
  for (int64_t i = 0; i < nb_count; i++)
    acc[i] = vdupq_n_s32(0);

  for (int64_t kb = 0; kb < K4; kb++) {
    int32_t a4;
    std::memcpy(&a4, A + kb * 4, 4);
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
}

EXPORT void sdot_gemv_m1_prepacked(const int8_t *A,
                                    const int8_t *B_packed,
                                    int32_t *C,
                                    int64_t K, int64_t N, int64_t N4) {
  int64_t K4 = K / 4;

  #pragma omp parallel
  {
    int nt = omp_get_num_threads();
    int tid = omp_get_thread_num();
    int64_t chunk = (N4 + nt - 1) / nt;
    int64_t start = tid * chunk;
    int64_t count = chunk;
    if (start + count > N4) count = N4 - start;
    if (start >= N4) count = 0;
    if (count > 0)
      sdot_gemv_range(A, B_packed, C, K4, N4, start, count);
  }
}

/*
 * Fused BF16→INT8 dynamic quantization + SDOT GEMV + dequantization→BF16.
 *
 * Replaces the Python-side pipeline:
 *   abs().max() → div → clamp → to(int8) → _int_mm → to(fp32) → mul(scale) → to(bf16)
 *
 * Single C call does: quantize activation (NEON), SDOT GEMV (OMP), dequant to BF16.
 */

// BF16 → FP32: shift left 16 bits
static inline float32x4_t bf16_to_fp32(uint16x4_t bf16) {
  return vreinterpretq_f32_u32(vshll_n_u16(bf16, 16));
}

static float quantize_activation_bf16(const uint16_t *x_bf16, int8_t *x_int8,
                                       int64_t K) {
  float32x4_t vmax = vdupq_n_f32(0.0f);
  int64_t k = 0;
  for (; k + 4 <= K; k += 4) {
    float32x4_t f = bf16_to_fp32(vld1_u16(x_bf16 + k));
    vmax = vmaxq_f32(vmax, vabsq_f32(f));
  }
  float amax = vmaxvq_f32(vmax);
  for (; k < K; k++) {
    uint32_t bits = (uint32_t)x_bf16[k] << 16;
    float v;
    std::memcpy(&v, &bits, 4);
    float av = std::abs(v);
    if (av > amax) amax = av;
  }
  if (amax < 1e-8f) amax = 1e-8f;
  float inv_scale = 127.0f / amax;

  k = 0;
  for (; k + 4 <= K; k += 4) {
    float32x4_t f = bf16_to_fp32(vld1_u16(x_bf16 + k));
    float32x4_t scaled = vmulq_n_f32(f, inv_scale);
    int32x4_t rounded = vcvtnq_s32_f32(scaled);
    int16x4_t n16 = vqmovn_s32(rounded);
    int8x8_t n8 = vqmovn_s16(vcombine_s16(n16, n16));
    vst1_lane_s32(reinterpret_cast<int32_t *>(x_int8 + k),
                  vreinterpret_s32_s8(n8), 0);
  }
  for (; k < K; k++) {
    uint32_t bits = (uint32_t)x_bf16[k] << 16;
    float v;
    std::memcpy(&v, &bits, 4);
    int32_t r = static_cast<int32_t>(std::round(v * inv_scale));
    if (r > 127) r = 127;
    if (r < -128) r = -128;
    x_int8[k] = static_cast<int8_t>(r);
  }
  return amax / 127.0f;
}

static void dequant_range_bf16(const int32_t *out_int32, float x_scale,
                                const float *w_scale, uint16_t *out_bf16,
                                int64_t start, int64_t count) {
  float32x4_t xs = vdupq_n_f32(x_scale);
  int64_t n = start;
  int64_t end = start + count;
  for (; n + 4 <= end; n += 4) {
    float32x4_t oi = vcvtq_f32_s32(vld1q_s32(out_int32 + n));
    float32x4_t ws = vld1q_f32(w_scale + n);
    float32x4_t result = vmulq_f32(vmulq_f32(oi, xs), ws);
    uint32x4_t ru = vreinterpretq_u32_f32(result);
    uint16x4_t bf = vshrn_n_u32(ru, 16);
    vst1_u16(out_bf16 + n, bf);
  }
  for (; n < end; n++) {
    float r = static_cast<float>(out_int32[n]) * x_scale * w_scale[n];
    uint32_t bits;
    std::memcpy(&bits, &r, 4);
    out_bf16[n] = static_cast<uint16_t>(bits >> 16);
  }
}

EXPORT void sdot_gemv_m1_fused_bf16(const uint16_t *x_bf16,
                                     const int8_t *B_packed,
                                     const float *w_scale,
                                     uint16_t *out_bf16,
                                     int64_t K, int64_t N, int64_t N4) {
  // Step 1: Quantize BF16 activation → INT8 (single-threaded, K is small ~2-6K)
  int8_t x_int8[16384];
  float x_scale = quantize_activation_bf16(x_bf16, x_int8, K);

  // Step 2+3: SDOT GEMV + Dequantize (fused per-thread to keep data in L1)
  int64_t K4 = K / 4;

  #pragma omp parallel
  {
    int nt = omp_get_num_threads();
    int tid = omp_get_thread_num();
    int64_t chunk_n4 = (N4 + nt - 1) / nt;
    int64_t start_n4 = tid * chunk_n4;
    int64_t count_n4 = chunk_n4;
    if (start_n4 + count_n4 > N4) count_n4 = N4 - start_n4;
    if (start_n4 >= N4) count_n4 = 0;

    if (count_n4 > 0) {
      // Allocate thread-local int32 buffer on heap for large N (lm_head: N=151936)
      std::vector<int32_t> local_buf(count_n4 * 4);
      int32_t *local_int32 = local_buf.data() - start_n4 * 4;
      sdot_gemv_range(x_int8, B_packed, local_int32,
                       K4, N4, start_n4, count_n4);
      dequant_range_bf16(local_int32, x_scale, w_scale,
                          out_bf16, start_n4 * 4, count_n4 * 4);
    }
  }
}

#else
// Stub for non-ARM platforms
EXPORT void sdot_pack_weights(const int8_t *, int8_t *, int64_t, int64_t) {}
EXPORT void sdot_gemv_m1_prepacked(const int8_t *, const int8_t *,
                                    int32_t *, int64_t, int64_t, int64_t) {}
EXPORT void sdot_gemv_m1_fused_bf16(const uint16_t *, const int8_t *,
                                     const float *, uint16_t *,
                                     int64_t, int64_t, int64_t) {}
#endif

} // extern "C"
