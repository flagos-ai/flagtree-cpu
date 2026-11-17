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

/* ═══════════════════════════════════════════════════════════
 * W4A8 per-channel SDOT GEMV (decode T=1, fused BF16 in/out).
 *
 * Weight packing format:  [K/4, N/4, 4, 2] int8
 *   For each (kb, nb) block (4 K-elements × 4 N-channels = 16 i4 weights):
 *     8 packed bytes; byte (ni*2 + p) holds:
 *       low nibble  = w[ni, ki=2p]
 *       high nibble = w[ni, ki=2p+1]
 *   Quantization: per-output-channel symmetric, w in [-7, 7] (4-bit signed).
 *
 * NEON unpack uses vshl/vshr-by-4 sign-extend trick + vzip1q to interleave
 * 8 low + 8 high nibbles into the 16 int8 layout SDOT wants
 * (ni0_ki0..3, ni1_ki0..3, ni2_ki0..3, ni3_ki0..3).
 *
 * Replaces W8 SDOT-fused GEMV at 2× memory throughput on weights (1 byte
 * per 2 weights vs 1 byte per 1 weight).
 * ═══════════════════════════════════════════════════════════ */

static void sdot_gemv_w4_range(const int8_t *A_int8, const int8_t *B_w4,
                                int32_t *C_int32, int64_t K4, int64_t N4,
                                int64_t nb_start, int64_t nb_count) {
  std::vector<int32x4_t> acc_heap;
  int32x4_t acc_stack[2048];
  int32x4_t *acc;
  if (nb_count <= 2048) {
    acc = acc_stack;
  } else {
    acc_heap.resize(nb_count);
    acc = acc_heap.data();
  }
  for (int64_t i = 0; i < nb_count; i++) acc[i] = vdupq_n_s32(0);

  for (int64_t kb = 0; kb < K4; kb++) {
    int32_t a4;
    std::memcpy(&a4, A_int8 + kb * 4, 4);
    int8x16_t av = vreinterpretq_s8_s32(vdupq_n_s32(a4));
    // Each (kb, nb) block is 8 bytes (16 i4 weights).
    const int8_t *bp = B_w4 + (kb * N4 + nb_start) * 8;

    int64_t i = 0;
    for (; i < nb_count; i++) {
      // Load 8 packed bytes.
      int8x8_t packed = vld1_s8(bp);
      // Sign-extend low nibble: (packed << 4) >> 4 (arithmetic).
      int8x8_t lo = vshr_n_s8(vshl_n_s8(packed, 4), 4);
      // Sign-extend high nibble: packed >> 4 (arithmetic).
      int8x8_t hi = vshr_n_s8(packed, 4);
      // Interleave to (lo[0], hi[0], lo[1], hi[1], ..., lo[7], hi[7]) ==
      //   (w[ni0_ki0], w[ni0_ki1], w[ni0_ki2], w[ni0_ki3],
      //    w[ni1_ki0], ..., w[ni3_ki3]).
      int8x16_t bv = vzip1q_s8(
          vcombine_s8(lo, vdup_n_s8(0)),
          vcombine_s8(hi, vdup_n_s8(0)));
      acc[i] = vdotq_s32(acc[i], av, bv);
      bp += 8;
    }
  }
  for (int64_t i = 0; i < nb_count; i++)
    vst1q_s32(C_int32 + (nb_start + i) * 4, acc[i]);
}

EXPORT void sdot_gemv_m1_w4_fused_bf16(const uint16_t *x_bf16,
                                        const int8_t *B_w4,
                                        const float *w_scale,
                                        uint16_t *out_bf16,
                                        int64_t K, int64_t N) {
  int8_t x_int8[16384];
  float x_scale = quantize_activation_bf16(x_bf16, x_int8, K);

  int64_t K4 = K / 4;
  int64_t N4 = N / 4;

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
      std::vector<int32_t> local_buf(count_n4 * 4);
      int32_t *local_int32 = local_buf.data() - start_n4 * 4;
      sdot_gemv_w4_range(x_int8, B_w4, local_int32,
                          K4, N4, start_n4, count_n4);
      dequant_range_bf16(local_int32, x_scale, w_scale,
                          out_bf16, start_n4 * 4, count_n4 * 4);
    }
  }
}

/* ═══════════════════════════════════════════════════════════
 * Q4_0-style W4A8 SDOT GEMV: per-block-32 W4 with fp16 scale per block per
 * output channel. Mirrors llama.cpp Q4_0 quantization (1 fp16 scale per
 * 32 K-elements per output channel).
 *
 * Layouts:
 *   weights:      [K/4, N/4, 4, 2] int8           (16 i4 weights per (kb, nb) block, same as W4 per-channel)
 *   block_scales: [K/32, N] fp16                   (1 scale per K-block-32 per output channel)
 *   activation:   [K] bf16                         (dynamically int8-quantized in kernel)
 *   out:          [N] bf16
 *
 * Per super-block of 32 K, accumulate 8 SDOTs into int32, then multiply by
 * (block_scale × activation_scale) to fp32, accumulate fp32 across blocks,
 * then convert to bf16. Avoids per-channel scale's outlier sensitivity.
 *
 * Constraint: K must be a multiple of 32 (and N a multiple of 4).
 * ═══════════════════════════════════════════════════════════ */

static void sdot_gemv_q4_0_range(const int8_t *A_int8, const int8_t *B_w4,
                                  const uint16_t *block_scales_fp16,
                                  float x_scale,
                                  uint16_t *out_bf16,
                                  int64_t K, int64_t N4,
                                  int64_t nb_start, int64_t nb_count) {
  // K must be multiple of 32 (caller guarantees).
  const int64_t K_super = K / 32;

  // Per-N-stripe accumulators. Two arrays:
  //   int_acc[i]: int32 SDOT accumulator within the current K-block-32
  //   out_fp[i]:  fp32 accumulator across K-blocks (after per-block dequant)
  // Stack-allocate when small; fall back to heap for very large N (e.g.
  // lm_head N=248320 → count_n4 up to 7760 per thread).
  std::vector<int32x4_t> int_heap;
  std::vector<float32x4_t> out_heap;
  int32x4_t int_stack[1024];
  float32x4_t out_stack[1024];
  int32x4_t *int_acc;
  float32x4_t *out_fp;
  if (nb_count <= 1024) {
    int_acc = int_stack;
    out_fp = out_stack;
  } else {
    int_heap.resize(nb_count);
    out_heap.resize(nb_count);
    int_acc = int_heap.data();
    out_fp = out_heap.data();
  }
  for (int64_t i = 0; i < nb_count; i++) out_fp[i] = vdupq_n_f32(0.0f);

  for (int64_t kb_super = 0; kb_super < K_super; kb_super++) {
    // Reset int32 accumulator at the start of each K-block-32.
    for (int64_t i = 0; i < nb_count; i++) int_acc[i] = vdupq_n_s32(0);

    // 8 K-stripes (kb = kb_super*8 .. kb_super*8+7) within this block.
    const int8_t *xp = A_int8 + kb_super * 32;
    for (int j = 0; j < 8; j++) {
      int32_t a4;
      std::memcpy(&a4, xp + j * 4, 4);
      int8x16_t av = vreinterpretq_s8_s32(vdupq_n_s32(a4));

      // Inner loop: iterate N-stripes sequentially (cache-friendly).
      // For 2 adjacent N-stripes at the same K-stripe, packed weights are
      // contiguous (16 bytes total). One vld1q_s8 + one shift sequence
      // produces unpacked operands for 2 SDOTs, halving the unpack cost
      // per SDOT vs the 8-byte-load-per-SDOT pattern.
      // Weight ptr: stripe (kb=kb_super*8+j, nb) is at offset
      //   (kb*N4 + nb) * 8 bytes.
      const int8_t *bp = B_w4 + (((kb_super * 8 + j) * N4 + nb_start) * 8);

      int64_t i = 0;
      // 8-way unroll: 4 × (2-stripe load) = 64 bytes of weight per iter,
      // 8 SDOTs per iter, ILP across 8 independent accumulators.
      for (; i + 8 <= nb_count; i += 8) {
        int8x16_t pp0 = vld1q_s8(bp);       bp += 16;
        int8x16_t pp1 = vld1q_s8(bp);       bp += 16;
        int8x16_t pp2 = vld1q_s8(bp);       bp += 16;
        int8x16_t pp3 = vld1q_s8(bp);       bp += 16;
        // Sign-extend low and high nibbles of each 16-byte register.
        int8x16_t lo0 = vshrq_n_s8(vshlq_n_s8(pp0, 4), 4);
        int8x16_t hi0 = vshrq_n_s8(pp0, 4);
        int8x16_t lo1 = vshrq_n_s8(vshlq_n_s8(pp1, 4), 4);
        int8x16_t hi1 = vshrq_n_s8(pp1, 4);
        int8x16_t lo2 = vshrq_n_s8(vshlq_n_s8(pp2, 4), 4);
        int8x16_t hi2 = vshrq_n_s8(pp2, 4);
        int8x16_t lo3 = vshrq_n_s8(vshlq_n_s8(pp3, 4), 4);
        int8x16_t hi3 = vshrq_n_s8(pp3, 4);
        // Each 16-byte load packs 2 (4K, 4N) blocks at adjacent nb.
        // vzip1q gives lo[0..7]+hi[0..7] interleaved = first SDOT operand
        // (covers nb+0); vzip2q gives lo[8..15]+hi[8..15] = second SDOT
        // operand (covers nb+1).
        int8x16_t b0 = vzip1q_s8(lo0, hi0);
        int8x16_t b1 = vzip2q_s8(lo0, hi0);
        int8x16_t b2 = vzip1q_s8(lo1, hi1);
        int8x16_t b3 = vzip2q_s8(lo1, hi1);
        int8x16_t b4 = vzip1q_s8(lo2, hi2);
        int8x16_t b5 = vzip2q_s8(lo2, hi2);
        int8x16_t b6 = vzip1q_s8(lo3, hi3);
        int8x16_t b7 = vzip2q_s8(lo3, hi3);
        int_acc[i]   = vdotq_s32(int_acc[i],   av, b0);
        int_acc[i+1] = vdotq_s32(int_acc[i+1], av, b1);
        int_acc[i+2] = vdotq_s32(int_acc[i+2], av, b2);
        int_acc[i+3] = vdotq_s32(int_acc[i+3], av, b3);
        int_acc[i+4] = vdotq_s32(int_acc[i+4], av, b4);
        int_acc[i+5] = vdotq_s32(int_acc[i+5], av, b5);
        int_acc[i+6] = vdotq_s32(int_acc[i+6], av, b6);
        int_acc[i+7] = vdotq_s32(int_acc[i+7], av, b7);
      }
      // 2-way unroll for remaining pairs.
      for (; i + 2 <= nb_count; i += 2) {
        int8x16_t pp = vld1q_s8(bp);     bp += 16;
        int8x16_t lo = vshrq_n_s8(vshlq_n_s8(pp, 4), 4);
        int8x16_t hi = vshrq_n_s8(pp, 4);
        int8x16_t b0 = vzip1q_s8(lo, hi);
        int8x16_t b1 = vzip2q_s8(lo, hi);
        int_acc[i]   = vdotq_s32(int_acc[i],   av, b0);
        int_acc[i+1] = vdotq_s32(int_acc[i+1], av, b1);
      }
      // Single-stripe tail.
      for (; i < nb_count; i++) {
        int8x8_t p = vld1_s8(bp);
        int8x8_t lo = vshr_n_s8(vshl_n_s8(p, 4), 4);
        int8x8_t hi = vshr_n_s8(p, 4);
        int8x16_t bv = vzip1q_s8(vcombine_s8(lo, vdup_n_s8(0)),
                                  vcombine_s8(hi, vdup_n_s8(0)));
        int_acc[i] = vdotq_s32(int_acc[i], av, bv);
        bp += 8;
      }
    }

    // After this K-block-32: convert int32 → fp32, multiply by per-channel
    // per-block scale, accumulate into out_fp[i].
    const uint16_t *sp = block_scales_fp16 + kb_super * (N4 * 4) + nb_start * 4;
    for (int64_t i = 0; i < nb_count; i++) {
      float16x4_t scales_h = vld1_f16(reinterpret_cast<const float16_t *>(sp));
      sp += 4;
      float32x4_t scales = vcvt_f32_f16(scales_h);
      out_fp[i] = vfmaq_f32(out_fp[i], vcvtq_f32_s32(int_acc[i]), scales);
    }
  }

  // Final: multiply by per-token activation scale, convert fp32 → bf16.
  float32x4_t xs = vdupq_n_f32(x_scale);
  for (int64_t i = 0; i < nb_count; i++) {
    float32x4_t out = vmulq_f32(out_fp[i], xs);
    uint32x4_t ru = vreinterpretq_u32_f32(out);
    uint16x4_t bf = vshrn_n_u32(ru, 16);
    vst1_u16(out_bf16 + (nb_start + i) * 4, bf);
  }
}

EXPORT void sdot_gemv_m1_q4_0_fused_bf16(const uint16_t *x_bf16,
                                          const int8_t *B_w4,
                                          const uint16_t *block_scales_fp16,
                                          uint16_t *out_bf16,
                                          int64_t K, int64_t N) {
  // Quantize activation
  int8_t x_int8[16384];
  float x_scale = quantize_activation_bf16(x_bf16, x_int8, K);

  int64_t N4 = N / 4;

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
      sdot_gemv_q4_0_range(x_int8, B_w4, block_scales_fp16,
                            x_scale, out_bf16, K, N4,
                            start_n4, count_n4);
    }
  }
}

/* ═══════════════════════════════════════════════════════════
 * Q4_0 v2 SDOT GEMV — llama.cpp-style layout (K-major per N output row).
 *
 * Layout (per output row n, then K-block-32 j):
 *   bytes 0..15: 32 i4 weights for K=[j*32 .. j*32+31].
 *                Q4_0 nibble convention: byte b stores
 *                  w[k=b]      → low nibble of byte b   (unsigned [0,15], decoded as v - 8)
 *                  w[k=b+16]   → high nibble of byte b
 *                So byte 0..15 lows give w[k=0..15], byte 0..15 highs give w[k=16..31].
 *   bytes 16..17: fp16 per-block per-output-channel scale.
 *
 * Total per (N, K_block): 18 bytes. Total weight tensor: N × K/32 × 18 bytes.
 *
 * This matches llama.cpp's block_q4_0 packing in memory, allowing one
 * uint8x16 load per K-block and only two SDOT instructions per K-block per
 * output channel (one for low nibbles × x_low, one for high nibbles × x_high).
 *
 * Activation: per-token int8 (single fp32 scale across full K, computed
 * once at the start of the kernel call). Less accurate than llama.cpp's Q8_0
 * per-block-32 activation, but simpler and matches our existing W8 path.
 * ═══════════════════════════════════════════════════════════ */

EXPORT void sdot_gemv_m1_q4_0_v2_fused_bf16(
    const uint16_t *x_bf16,             // [K] bf16 activation
    const int8_t   *W_packed,           // [N × K/32 × 18] int8 — pack format is **kxor** (data bytes XOR'd with 0x88)
    uint16_t       *out_bf16,           // [N] bf16 output
    int64_t K, int64_t N) {
  // Quantize activation once (per-token int8).
  int8_t x_int8[16384];
  float x_scale = quantize_activation_bf16(x_bf16, x_int8, K);

  const int64_t K_blocks = K / 32;
  const int64_t row_bytes = K_blocks * 18;
  const uint8x16_t mhi = vdupq_n_u8(0xF0);

  #pragma omp parallel for schedule(static)
  for (int64_t n = 0; n < N; n++) {
    const int8_t *wp = W_packed + n * row_bytes;
    // Prefetch the weight row a few cache lines ahead.
    __builtin_prefetch(wp + 64);
    __builtin_prefetch(wp + 128);
    float32x4_t fp_acc = vdupq_n_f32(0.0f);

    for (int64_t kb = 0; kb < K_blocks; kb++) {
      // Prefetch a couple K-blocks ahead (each K-block is 18 bytes, prefetch
      // up to 4 K-blocks ahead = 72 bytes ≈ next cache line).
      if (kb + 4 < K_blocks) __builtin_prefetch(wp + 4 * 18);
      // Bytes are stored as (signed_4bit_2's_complement_low) | (signed_4bit_high << 4),
      // i.e. caller pre-XOR'd with 0x88 to convert from unsigned [0..15] = signed - 8.
      // After `shl #4`, low nibble moves to upper 4 bits and is sign-extended → original signed × 16.
      // After `and 0xF0`, high nibble stays in upper 4 bits → original signed × 16.
      // Final dequant uses vcvtq_n_f32_s32(_, 4) which divides by 16 implicitly.
      int8x16_t v0 = vld1q_s8(wp);
      uint16_t scale_h;
      std::memcpy(&scale_h, wp + 16, 2);
      wp += 18;
      int8x16_t lo16x = vshlq_n_s8(v0, 4);
      int8x16_t hi16x = vreinterpretq_s8_u8(vandq_u8(vreinterpretq_u8_s8(v0), mhi));
      const int8_t *xp = x_int8 + kb * 32;
      int8x16_t y_lo = vld1q_s8(xp);
      int8x16_t y_hi = vld1q_s8(xp + 16);
      int32x4_t int_acc = vdotq_s32(vdupq_n_s32(0), lo16x, y_lo);
      int_acc = vdotq_s32(int_acc, hi16x, y_hi);
      __fp16 h;
      std::memcpy(&h, &scale_h, 2);
      fp_acc = vmlaq_n_f32(fp_acc, vcvtq_n_f32_s32(int_acc, 4), (float)h);
    }
    float dot = vaddvq_f32(fp_acc) * x_scale;
    uint32_t bits;
    std::memcpy(&bits, &dot, 4);
    out_bf16[n] = (uint16_t)(bits >> 16);
  }
}

/* ═══════════════════════════════════════════════════════════
 * Q4_0 v2 SMMLA prefill GEMM (M >= 2). Uses NEON i8mm SMMLA
 * (vmmlaq_s32) which does a 2x8 × 8x2 int8 matmul → 2x2 int32 in one
 * instruction. Pairs M tokens into groups of 2 and N output channels
 * into groups of 2, so one SMMLA computes 4 partial outputs.
 *
 * Inner per-(M-pair, N-pair, K-block-32):
 *   load 16 packed bytes for n0's K-block + fp16 scale
 *   load 16 packed bytes for n1's K-block + fp16 scale
 *   unpack to 4 int8x16: lo0/hi0/lo1/hi1 (each = 16 K-elements for one N row)
 *   load activation: 4 int8x16 for (m0/m1) × (lo/hi K-halves)
 *   4 SMMLA instructions, accumulating int32 within this K-block
 *   dequant: int_acc[lanes] × (w_scale * x_scale) → fp32 accumulators
 * After all K-blocks: write 4 bf16 outputs.
 *
 * For odd M, the last row falls back to M=1 sdot_gemv_m1_q4_0_v2_fused_bf16.
 * ═══════════════════════════════════════════════════════════ */

EXPORT void sdot_gemm_q4_0_v2_smmla_bf16(
    const uint16_t *x_bf16,   // [M, K] bf16 activation, M outer
    const int8_t   *W_packed, // [N × K/32 × 18 bytes] int8 (Q4_0 v2 kxor layout)
    uint16_t       *out_bf16, // [M, N] bf16 output
    int64_t M, int64_t K, int64_t N) {
  if (M < 2) return;  // caller falls back for M=1
  const int64_t K_blocks = K / 32;
  const int64_t row_bytes = K_blocks * 18;

  // Quantize all M activation rows to int8 with per-row fp32 scale.
  std::vector<int8_t> x_int8_buf(M * K);
  std::vector<float>  x_scales(M);
  for (int64_t m = 0; m < M; m++) {
    x_scales[m] = quantize_activation_bf16(x_bf16 + m * K,
                                            x_int8_buf.data() + m * K, K);
  }

  const uint8x16_t mhi = vdupq_n_u8(0xF0);

  const int64_t M_pairs = M / 2;
  const int64_t M_left  = M % 2;

  #pragma omp parallel for schedule(static)
  for (int64_t np = 0; np < N / 2; np++) {
    int64_t n0 = np * 2;
    int64_t n1 = n0 + 1;

    for (int64_t mp = 0; mp < M_pairs; mp++) {
      int64_t m0 = mp * 2;
      int64_t m1 = m0 + 1;
      float xs0 = x_scales[m0];
      float xs1 = x_scales[m1];

      float32x4_t fp_acc = vdupq_n_f32(0.0f);

      for (int64_t kb = 0; kb < K_blocks; kb++) {
        const int8_t *wp0 = W_packed + n0 * row_bytes + kb * 18;
        const int8_t *wp1 = W_packed + n1 * row_bytes + kb * 18;
        int8x16_t w0 = vld1q_s8(wp0);
        int8x16_t w1 = vld1q_s8(wp1);
        uint16_t s0_h, s1_h;
        std::memcpy(&s0_h, wp0 + 16, 2);
        std::memcpy(&s1_h, wp1 + 16, 2);
        __fp16 h0, h1;
        std::memcpy(&h0, &s0_h, 2);
        std::memcpy(&h1, &s1_h, 2);
        // kxor: divide-by-16 happens via vcvtq_n_f32_s32(_, 4); fold that into
        // the scale so we can keep using vmlaq_n_f32 with one constant per N.
        float ws0 = (float)h0 / 16.0f;
        float ws1 = (float)h1 / 16.0f;

        // shl/and trick: low nibble in upper 4 bits via vshlq_n; high nibble
        // already in upper 4 bits via vandq_u8(_, 0xF0). Original signed
        // values × 16, undone later by per-N scale fold.
        int8x16_t lo0 = vshlq_n_s8(w0, 4);
        int8x16_t hi0 = vreinterpretq_s8_u8(vandq_u8(vreinterpretq_u8_s8(w0), mhi));
        int8x16_t lo1 = vshlq_n_s8(w1, 4);
        int8x16_t hi1 = vreinterpretq_s8_u8(vandq_u8(vreinterpretq_u8_s8(w1), mhi));

        const int8_t *xp0 = x_int8_buf.data() + m0 * K + kb * 32;
        const int8_t *xp1 = x_int8_buf.data() + m1 * K + kb * 32;
        int8x16_t a0_lo = vld1q_s8(xp0);       // m0 K=0..15
        int8x16_t a0_hi = vld1q_s8(xp0 + 16);  // m0 K=16..31
        int8x16_t a1_lo = vld1q_s8(xp1);
        int8x16_t a1_hi = vld1q_s8(xp1 + 16);

        // SMMLA: 4 calls, each consuming 8 K-elements (lo halves of a/lo,
        // hi halves of a/lo, lo halves of a/hi, hi halves of a/hi).
        int32x4_t int_acc = vdupq_n_s32(0);
        // K=0..7 (lo half of K=0..15 block)
        int_acc = vmmlaq_s32(int_acc,
            vcombine_s8(vget_low_s8(a0_lo), vget_low_s8(a1_lo)),
            vcombine_s8(vget_low_s8(lo0),   vget_low_s8(lo1)));
        // K=8..15 (high half of K=0..15 block)
        int_acc = vmmlaq_s32(int_acc,
            vcombine_s8(vget_high_s8(a0_lo), vget_high_s8(a1_lo)),
            vcombine_s8(vget_high_s8(lo0),   vget_high_s8(lo1)));
        // K=16..23 (lo half of K=16..31 block)
        int_acc = vmmlaq_s32(int_acc,
            vcombine_s8(vget_low_s8(a0_hi), vget_low_s8(a1_hi)),
            vcombine_s8(vget_low_s8(hi0),   vget_low_s8(hi1)));
        // K=24..31 (high half of K=16..31 block)
        int_acc = vmmlaq_s32(int_acc,
            vcombine_s8(vget_high_s8(a0_hi), vget_high_s8(a1_hi)),
            vcombine_s8(vget_high_s8(hi0),   vget_high_s8(hi1)));

        // Dequant: int_acc lanes are (m0n0, m0n1, m1n0, m1n1).
        // Build per-block float scale per lane: lane i scale = w_scale[n] * x_scale[m].
        float lane_scales_arr[4] = {
            ws0 * xs0,  // m0n0
            ws1 * xs0,  // m0n1
            ws0 * xs1,  // m1n0
            ws1 * xs1,  // m1n1
        };
        float32x4_t lane_scales = vld1q_f32(lane_scales_arr);
        fp_acc = vfmaq_f32(fp_acc, vcvtq_f32_s32(int_acc), lane_scales);
      }

      // Write 4 bf16 outputs at out[m0, n0..n1] and out[m1, n0..n1].
      uint32_t bits[4];
      float vals[4];
      vst1q_f32(vals, fp_acc);
      for (int i = 0; i < 4; i++) std::memcpy(&bits[i], &vals[i], 4);
      out_bf16[m0 * N + n0] = (uint16_t)(bits[0] >> 16);
      out_bf16[m0 * N + n1] = (uint16_t)(bits[1] >> 16);
      out_bf16[m1 * N + n0] = (uint16_t)(bits[2] >> 16);
      out_bf16[m1 * N + n1] = (uint16_t)(bits[3] >> 16);
    }
  }

  // Handle leftover M row (if M is odd). Use the M=1 GEMV per remaining row.
  if (M_left) {
    int64_t m_last = M - 1;
    sdot_gemv_m1_q4_0_v2_fused_bf16(
        x_bf16 + m_last * K, W_packed,
        out_bf16 + m_last * N, K, N);
  }
}

/* Pack a row-major [N, K] int8 weight (already quantized to Q4_0 nibbles in
 * [0..15], with subtract-8 decoding) plus per-(N, K_block_32) fp16 scales,
 * into the v2 layout [N × K/32 × 18 bytes]. Used by the Python pack helper
 * via ctypes; kernel itself only consumes the packed buffer. */
/* Pack [K, N] unsigned-nibble weights ([0..15] = signed - 8) plus
 * per-(K_block, N) fp16 scales into the v2 kxor layout
 * [N × K/32 × 18 bytes]. The data bytes are XOR'd with 0x88 so each
 * nibble becomes the 4-bit two's-complement of the original signed
 * weight, letting the SDOT kernel use shl/and tricks to avoid an
 * explicit subtract-8 step. */
EXPORT void sdot_pack_weights_q4_0_v2(
    const int8_t   *w_nibbles_kn,       // [K, N] int8 with values [0..15]
    const uint16_t *block_scales_kn,    // [K/32, N] fp16
    int8_t         *out,                // [N × K/32 × 18] int8 packed (kxor)
    int64_t K, int64_t N) {
  const int64_t K_blocks = K / 32;
  for (int64_t n = 0; n < N; n++) {
    int8_t *dst = out + n * K_blocks * 18;
    for (int64_t kb = 0; kb < K_blocks; kb++) {
      for (int b = 0; b < 16; b++) {
        int low_k  = kb * 32 + b;
        int high_k = kb * 32 + b + 16;
        uint8_t lo = (uint8_t)(w_nibbles_kn[low_k  * N + n] & 0x0F);
        uint8_t hi = (uint8_t)(w_nibbles_kn[high_k * N + n] & 0x0F);
        // XOR 0x88: convert nibble [0..15] (encoding signed - 8) to its
        // 4-bit two's-complement representation of the signed value.
        dst[b] = (int8_t)(((hi << 4) | lo) ^ 0x88);
      }
      uint16_t s = block_scales_kn[kb * N + n];
      std::memcpy(dst + 16, &s, 2);
      dst += 18;
    }
  }
}

/* W4 weight packer: [K, N] int8 (values in -7..7) → [K/4, N/4, 4, 2] int8. */
EXPORT void sdot_pack_weights_w4(const int8_t *B, int8_t *B_packed,
                                  int64_t K, int64_t N) {
  int64_t K4 = K / 4;
  int64_t N4 = N / 4;
  for (int64_t kb = 0; kb < K4; kb++) {
    for (int64_t nb = 0; nb < N4; nb++) {
      int8_t *dst = B_packed + (kb * N4 + nb) * 8;
      for (int ni = 0; ni < 4; ni++) {
        for (int p = 0; p < 2; p++) {
          int ki_lo = 2 * p;
          int ki_hi = 2 * p + 1;
          int8_t lo = B[(kb * 4 + ki_lo) * N + nb * 4 + ni];
          int8_t hi = B[(kb * 4 + ki_hi) * N + nb * 4 + ni];
          // Pack low nibble of `lo` into low nibble of byte;
          // low nibble of `hi` into high nibble of byte.
          uint8_t b = (uint8_t)((lo & 0x0F) | ((hi & 0x0F) << 4));
          dst[ni * 2 + p] = (int8_t)b;
        }
      }
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
EXPORT void sdot_gemv_m1_w4_fused_bf16(const uint16_t *, const int8_t *,
                                        const float *, uint16_t *,
                                        int64_t, int64_t) {}
EXPORT void sdot_pack_weights_w4(const int8_t *, int8_t *, int64_t, int64_t) {}
EXPORT void sdot_gemv_m1_q4_0_fused_bf16(const uint16_t *, const int8_t *,
                                          const uint16_t *, uint16_t *,
                                          int64_t, int64_t) {}
EXPORT void sdot_gemv_m1_q4_0_v2_fused_bf16(const uint16_t *, const int8_t *,
                                             uint16_t *, int64_t, int64_t) {}
EXPORT void sdot_pack_weights_q4_0_v2(const int8_t *, const uint16_t *,
                                       int8_t *, int64_t, int64_t) {}
EXPORT void sdot_gemm_q4_0_v2_smmla_bf16(const uint16_t *, const int8_t *,
                                          uint16_t *, int64_t, int64_t, int64_t) {}
#endif

} // extern "C"
