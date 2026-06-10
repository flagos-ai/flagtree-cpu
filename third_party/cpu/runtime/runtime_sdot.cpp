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
#include "cpu_features.h"

#include <arm_neon.h>
#include <cstdio>
#include <cstdlib>
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

// nb-major GEMV: weight laid out [N4][K4][16] so each output group's K-stream is
// contiguous. Register-blocked over NR=8 groups (acc stays in registers, no L1
// sweep) → 8 sequential weight streams. ~2x the kb-major kernel (DRAM-BW bound,
// matches ggml's q8_0 GEMV structure). Threads split [nb_start, nb_start+nb_count).
static void sdot_gemv_nmajor_range(const int8_t *A, const int8_t *B_n,
                                   int32_t *C, int64_t K4, int64_t N4,
                                   int64_t nb_start, int64_t nb_count) {
  int64_t nb = nb_start, end = nb_start + nb_count;
  for (; nb + 8 <= end; nb += 8) {
    int32x4_t a0 = vdupq_n_s32(0), a1 = a0, a2 = a0, a3 = a0,
              a4 = a0, a5 = a0, a6 = a0, a7 = a0;
    const int8_t *p0 = B_n + (nb + 0) * K4 * 16, *p1 = B_n + (nb + 1) * K4 * 16,
                 *p2 = B_n + (nb + 2) * K4 * 16, *p3 = B_n + (nb + 3) * K4 * 16,
                 *p4 = B_n + (nb + 4) * K4 * 16, *p5 = B_n + (nb + 5) * K4 * 16,
                 *p6 = B_n + (nb + 6) * K4 * 16, *p7 = B_n + (nb + 7) * K4 * 16;
    for (int64_t kb = 0; kb < K4; kb++) {
      int32_t a; std::memcpy(&a, A + kb * 4, 4);
      int8x16_t av = vreinterpretq_s8_s32(vdupq_n_s32(a));
      a0 = vdotq_s32(a0, av, vld1q_s8(p0)); p0 += 16;
      a1 = vdotq_s32(a1, av, vld1q_s8(p1)); p1 += 16;
      a2 = vdotq_s32(a2, av, vld1q_s8(p2)); p2 += 16;
      a3 = vdotq_s32(a3, av, vld1q_s8(p3)); p3 += 16;
      a4 = vdotq_s32(a4, av, vld1q_s8(p4)); p4 += 16;
      a5 = vdotq_s32(a5, av, vld1q_s8(p5)); p5 += 16;
      a6 = vdotq_s32(a6, av, vld1q_s8(p6)); p6 += 16;
      a7 = vdotq_s32(a7, av, vld1q_s8(p7)); p7 += 16;
    }
    vst1q_s32(C + (nb + 0) * 4, a0); vst1q_s32(C + (nb + 1) * 4, a1);
    vst1q_s32(C + (nb + 2) * 4, a2); vst1q_s32(C + (nb + 3) * 4, a3);
    vst1q_s32(C + (nb + 4) * 4, a4); vst1q_s32(C + (nb + 5) * 4, a5);
    vst1q_s32(C + (nb + 6) * 4, a6); vst1q_s32(C + (nb + 7) * 4, a7);
  }
  for (; nb < end; nb++) {
    int32x4_t a = vdupq_n_s32(0);
    const int8_t *p = B_n + nb * K4 * 16;
    for (int64_t kb = 0; kb < K4; kb++) {
      int32_t x; std::memcpy(&x, A + kb * 4, 4);
      a = vdotq_s32(a, vreinterpretq_s8_s32(vdupq_n_s32(x)), vld1q_s8(p + kb * 16));
    }
    vst1q_s32(C + nb * 4, a);
  }
}

// Transpose a kb-major pack [K4][N4][16] into an nb-major pack [N4][K4][16].
EXPORT void sdot_pack_nmajor(const int8_t *B_kb, int8_t *B_n,
                             int64_t K4, int64_t N4) {
  for (int64_t nb = 0; nb < N4; nb++)
    for (int64_t kb = 0; kb < K4; kb++)
      std::memcpy(B_n + (nb * K4 + kb) * 16, B_kb + (kb * N4 + nb) * 16, 16);
}

// Block-kb-major pack: chop N4 into blocks of BN4 groups; each block stored
// kb-major internally -> [nblk][K4][BN4][16] (last block zero-padded to BN4).
// A thread owning whole blocks reads each block as ONE sequential DRAM stream
// (hits multi-core BW ceiling) while acc[BN4] stays L1-resident. nblk*K4*BN4*16
// bytes (slightly padded). Build from a kb-major pack [K4][N4][16].
EXPORT void sdot_pack_blk(const int8_t *B_kb, int8_t *B_blk,
                          int64_t K4, int64_t N4, int64_t BN4) {
  int64_t nblk = (N4 + BN4 - 1) / BN4;
  std::memset(B_blk, 0, (size_t) nblk * K4 * BN4 * 16);
  for (int64_t blk = 0; blk < nblk; blk++) {
    int64_t g0 = blk * BN4;
    int64_t gc = (N4 - g0 < BN4) ? (N4 - g0) : BN4;
    int8_t *base = B_blk + blk * K4 * BN4 * 16;
    for (int64_t kb = 0; kb < K4; kb++)
      for (int64_t j = 0; j < gc; j++)
        std::memcpy(base + (kb * BN4 + j) * 16,
                    B_kb + ((kb * N4) + g0 + j) * 16, 16);
  }
}

// Block-kb-major GEMV. Threads split the block range [blk_start, blk_start+blk_count).
static void sdot_gemv_blk_range(const int8_t *A, const int8_t *B_blk, int32_t *C,
                                int64_t K4, int64_t N4, int64_t BN4,
                                int64_t blk_start, int64_t blk_count) {
  int32x4_t acc[256];  // BN4 <= 256
  for (int64_t blk = blk_start; blk < blk_start + blk_count; blk++) {
    int64_t g0 = blk * BN4;
    int64_t gc = (N4 - g0 < BN4) ? (N4 - g0) : BN4;
    if (gc <= 0) break;
    const int8_t *base = B_blk + blk * K4 * BN4 * 16;
    for (int64_t j = 0; j < gc; j++) acc[j] = vdupq_n_s32(0);
    for (int64_t kb = 0; kb < K4; kb++) {
      int32_t a; std::memcpy(&a, A + kb * 4, 4);
      int8x16_t av = vreinterpretq_s8_s32(vdupq_n_s32(a));
      const int8_t *bp = base + kb * BN4 * 16;   // contiguous within the block
      for (int64_t j = 0; j < gc; j++)
        acc[j] = vdotq_s32(acc[j], av, vld1q_s8(bp + j * 16));
    }
    for (int64_t j = 0; j < gc; j++) vst1q_s32(C + (g0 + j) * 4, acc[j]);
  }
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

// ============================================================================
// i8mm SMMLA GEMM (M>1 prefill): weights read once, reused across all M rows.
// vmmlaq_s32 does a 2x2 output tile (2 m-rows x 2 n-cols) 8-K deep -> ~2x SDOT.
// Guarded: i8mm is armv8.6 optional-from-8.2; toolchains/targets without it
// (e.g. plain armv8.2 server builds) still get a symbol-complete library via
// the stubs below, and runtime dispatch keeps these paths unreachable.
// ============================================================================
#if defined(__ARM_FEATURE_MATMUL_INT8)

// Pack int8 [rows, K] -> [rows/2, K/8, 16] (2 rows x 8 K per SMMLA tile).
EXPORT void smmla_pack_weights(const int8_t *W, int8_t *P, int64_t rows, int64_t K) {
  int64_t K8 = K / 8, RP = rows / 2;
  for (int64_t rp = 0; rp < RP; rp++)
    for (int64_t ko = 0; ko < K8; ko++)
      for (int r = 0; r < 2; r++)
        for (int kk = 0; kk < 8; kk++)
          P[(rp * K8 + ko) * 16 + r * 8 + kk] = W[(rp * 2 + r) * K + ko * 8 + kk];
}

// Per-row dynamic int8 quant of f32 activation A[M,K] + pack to ko-major SMMLA
// layout Ap[K/8, M/2, 16] (so the GEMM's inner mp loop is contiguous), xs[M].
// Processes rows [m_start, m_start+m_count) -> caller can split across threads.
EXPORT void smmla_quant_pack_act_f32(const float *A, int8_t *Ap, float *xs,
                                     int64_t M, int64_t K,
                                     int64_t m_start, int64_t m_count) {
  int64_t K8 = K / 8, MP = M / 2;
  for (int64_t m = m_start; m < m_start + m_count; m++) {
    const float *row = A + m * K;
    float32x4_t vmax = vdupq_n_f32(0.0f);
    int64_t k = 0;
    for (; k + 4 <= K; k += 4) vmax = vmaxq_f32(vmax, vabsq_f32(vld1q_f32(row + k)));
    float amax = vmaxvq_f32(vmax);
    for (; k < K; k++) { float a = std::fabs(row[k]); if (a > amax) amax = a; }
    if (amax < 1e-10f) amax = 1e-10f;
    xs[m] = amax / 127.0f;
    float inv = 127.0f / amax;
    int64_t mp = m / 2, mr = m % 2;
    for (int64_t ko = 0; ko < K8; ko++) {
      int8_t *dst = Ap + (ko * MP + mp) * 16 + mr * 8;
      for (int kk = 0; kk < 8; kk++) {
        int32_t q = (int32_t)std::lround(row[ko * 8 + kk] * inv);
        if (q > 127) q = 127; if (q < -128) q = -128;
        dst[kk] = (int8_t)q;
      }
    }
  }
}

// C[M,N] = A @ W^T, dequant by xs[m]*ws[n]. Ap/Wp SMMLA-packed. Thread takes
// N-pair range [np_start, np_start+np_count). Reads each W n-pair once.
static inline void smmla_store(float *C, int64_t N, const float *xs, const float *ws,
                               int64_t mp, int64_t np, int32x4_t acc) {
  int32_t a[4]; vst1q_s32(a, acc);
  int64_t m0 = mp * 2, m1 = mp * 2 + 1, n0 = np * 2, n1 = np * 2 + 1;
  C[m0 * N + n0] = a[0] * xs[m0] * ws[n0];
  C[m0 * N + n1] = a[1] * xs[m0] * ws[n1];
  C[m1 * N + n0] = a[2] * xs[m1] * ws[n0];
  C[m1 * N + n1] = a[3] * xs[m1] * ws[n1];
}

EXPORT void smmla_gemm_range(const int8_t *Ap, const float *xs, const int8_t *Wp,
                             const float *ws, float *C, int64_t M, int64_t N, int64_t K,
                             int64_t np_start, int64_t np_count) {
  int64_t K8 = K / 8, MP = M / 2, np_end = np_start + np_count;
  int64_t np = np_start;
  // Main: 4 N-pairs x 4 M-pairs register tile (16 acc, in registers; no L1 sweep).
  for (; np + 4 <= np_end; np += 4) {
    const int8_t *w0b = Wp + np * K8 * 16, *w1b = w0b + K8 * 16,
                 *w2b = w1b + K8 * 16, *w3b = w2b + K8 * 16;
    int64_t mp = 0;
    for (; mp + 4 <= MP; mp += 4) {
      int32x4_t acc[4][4];
      for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) acc[i][j] = vdupq_n_s32(0);
      for (int64_t ko = 0; ko < K8; ko++) {
        int8x16_t w0 = vld1q_s8(w0b + ko * 16), w1 = vld1q_s8(w1b + ko * 16),
                  w2 = vld1q_s8(w2b + ko * 16), w3 = vld1q_s8(w3b + ko * 16);
        const int8_t *apk = Ap + ko * MP * 16 + mp * 16;
        int8x16_t a0 = vld1q_s8(apk), a1 = vld1q_s8(apk + 16),
                  a2 = vld1q_s8(apk + 32), a3 = vld1q_s8(apk + 48);
        acc[0][0]=vmmlaq_s32(acc[0][0],a0,w0); acc[0][1]=vmmlaq_s32(acc[0][1],a0,w1);
        acc[0][2]=vmmlaq_s32(acc[0][2],a0,w2); acc[0][3]=vmmlaq_s32(acc[0][3],a0,w3);
        acc[1][0]=vmmlaq_s32(acc[1][0],a1,w0); acc[1][1]=vmmlaq_s32(acc[1][1],a1,w1);
        acc[1][2]=vmmlaq_s32(acc[1][2],a1,w2); acc[1][3]=vmmlaq_s32(acc[1][3],a1,w3);
        acc[2][0]=vmmlaq_s32(acc[2][0],a2,w0); acc[2][1]=vmmlaq_s32(acc[2][1],a2,w1);
        acc[2][2]=vmmlaq_s32(acc[2][2],a2,w2); acc[2][3]=vmmlaq_s32(acc[2][3],a2,w3);
        acc[3][0]=vmmlaq_s32(acc[3][0],a3,w0); acc[3][1]=vmmlaq_s32(acc[3][1],a3,w1);
        acc[3][2]=vmmlaq_s32(acc[3][2],a3,w2); acc[3][3]=vmmlaq_s32(acc[3][3],a3,w3);
      }
      for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++)
        smmla_store(C, N, xs, ws, mp + i, np + j, acc[i][j]);
    }
    for (; mp < MP; mp++) {  // leftover M-pairs for this 4-N block
      int32x4_t a0=vdupq_n_s32(0),a1=a0,a2=a0,a3=a0;
      for (int64_t ko = 0; ko < K8; ko++) {
        int8x16_t a = vld1q_s8(Ap + (ko*MP+mp)*16);
        a0=vmmlaq_s32(a0,a,vld1q_s8(w0b+ko*16)); a1=vmmlaq_s32(a1,a,vld1q_s8(w1b+ko*16));
        a2=vmmlaq_s32(a2,a,vld1q_s8(w2b+ko*16)); a3=vmmlaq_s32(a3,a,vld1q_s8(w3b+ko*16));
      }
      smmla_store(C,N,xs,ws,mp,np,a0);   smmla_store(C,N,xs,ws,mp,np+1,a1);
      smmla_store(C,N,xs,ws,mp,np+2,a2); smmla_store(C,N,xs,ws,mp,np+3,a3);
    }
  }
  for (; np < np_end; np++) {  // leftover N-pairs (1-3)
    const int8_t *wp = Wp + np * K8 * 16;
    for (int64_t mp = 0; mp < MP; mp++) {
      int32x4_t acc = vdupq_n_s32(0);
      for (int64_t ko = 0; ko < K8; ko++)
        acc = vmmlaq_s32(acc, vld1q_s8(Ap + (ko*MP+mp)*16), vld1q_s8(wp + ko*16));
      smmla_store(C, N, xs, ws, mp, np, acc);
    }
  }
}

#endif // __ARM_FEATURE_MATMUL_INT8

// Non-OMP range variant: the CALLER's thread pool (e.g. llama.cpp/ggml) splits
// [n4_start, n4_start+n4_count) (in N4 = N/4 groups) across its own threads.
// Each call quantizes the full activation locally (cheap: 2*K ops vs the slice
// GEMV) so no shared state / barrier is needed inside the runtime.
EXPORT void sdot_gemv_m1_fused_bf16_range(const uint16_t *x_bf16,
                                          const int8_t *B_packed,
                                          const float *w_scale,
                                          uint16_t *out_bf16,
                                          int64_t K, int64_t N, int64_t N4,
                                          int64_t n4_start, int64_t n4_count) {
  if (n4_count <= 0) return;
  int8_t x_int8[16384];
  float x_scale = quantize_activation_bf16(x_bf16, x_int8, K);
  int64_t K4 = K / 4;
  std::vector<int32_t> local_buf(n4_count * 4);
  int32_t *local_int32 = local_buf.data() - n4_start * 4;
  sdot_gemv_range(x_int8, B_packed, local_int32, K4, N4, n4_start, n4_count);
  dequant_range_bf16(local_int32, x_scale, w_scale, out_bf16,
                     n4_start * 4, n4_count * 4);
}

// nb-major fused bf16 GEMV range (same contract as the kb-major variant above).
EXPORT void sdot_gemv_m1_fused_bf16_nmajor_range(const uint16_t *x_bf16,
                                                 const int8_t *B_n,
                                                 const float *w_scale,
                                                 uint16_t *out_bf16,
                                                 int64_t K, int64_t N, int64_t N4,
                                                 int64_t n4_start, int64_t n4_count) {
  if (n4_count <= 0) return;
  int8_t x_int8[16384];
  float x_scale = quantize_activation_bf16(x_bf16, x_int8, K);
  int64_t K4 = K / 4;
  std::vector<int32_t> local_buf(n4_count * 4);
  int32_t *local_int32 = local_buf.data() - n4_start * 4;
  sdot_gemv_nmajor_range(x_int8, B_n, local_int32, K4, N4, n4_start, n4_count);
  dequant_range_bf16(local_int32, x_scale, w_scale, out_bf16,
                     n4_start * 4, n4_count * 4);
}

// Block-kb-major fused bf16 GEMV. Threads split BLOCKS (each BN4 groups); this
// call handles blocks [blk_start, blk_start+blk_count). Per-thread sequential
// stream -> reaches the multi-core DRAM-BW ceiling (best of kb- and nb-major).
EXPORT void sdot_gemv_m1_fused_bf16_blk_range(const uint16_t *x_bf16,
                                              const int8_t *B_blk,
                                              const float *w_scale,
                                              uint16_t *out_bf16,
                                              int64_t K, int64_t N, int64_t N4,
                                              int64_t BN4,
                                              int64_t blk_start, int64_t blk_count) {
  if (blk_count <= 0) return;
  int8_t x_int8[16384];
  float x_scale = quantize_activation_bf16(x_bf16, x_int8, K);
  int64_t K4 = K / 4;
  int64_t g0 = blk_start * BN4;
  int64_t gend = (blk_start + blk_count) * BN4;
  if (gend > N4) gend = N4;
  if (g0 >= N4) return;
  std::vector<int32_t> local_buf((gend - g0) * 4);
  int32_t *local_int32 = local_buf.data() - g0 * 4;
  sdot_gemv_blk_range(x_int8, B_blk, local_int32, K4, N4, BN4, blk_start, blk_count);
  dequant_range_bf16(local_int32, x_scale, w_scale, out_bf16, g0 * 4, (gend - g0) * 4);
}

// ---- Decode fast path: SHARED activation quant (f32->int8 once, on ith==0) +
//      f32-DIRECT dequant (skip the int32->bf16->f32 roundtrip). ----
// Quantize an f32 activation row to int8 in one pass; returns scale = amax/127.
EXPORT float sdot_quant_act_f32(const float *x, int8_t *xq, int64_t K) {
  float32x4_t vmax = vdupq_n_f32(0.0f);
  int64_t k = 0;
  for (; k + 4 <= K; k += 4) vmax = vmaxq_f32(vmax, vabsq_f32(vld1q_f32(x + k)));
  float amax = vmaxvq_f32(vmax);
  for (; k < K; k++) { float av = std::abs(x[k]); if (av > amax) amax = av; }
  if (amax < 1e-8f) amax = 1e-8f;
  float inv = 127.0f / amax;
  k = 0;
  for (; k + 4 <= K; k += 4) {
    int32x4_t r = vcvtnq_s32_f32(vmulq_n_f32(vld1q_f32(x + k), inv));
    int8x8_t n8 = vqmovn_s16(vcombine_s16(vqmovn_s32(r), vqmovn_s32(r)));
    vst1_lane_s32(reinterpret_cast<int32_t *>(xq + k), vreinterpret_s32_s8(n8), 0);
  }
  for (; k < K; k++) { int32_t r = (int32_t) std::lround(x[k] * inv);
    if (r > 127) r = 127; if (r < -128) r = -128; xq[k] = (int8_t) r; }
  return amax / 127.0f;
}
static void dequant_range_f32(const int32_t *oi, float xs, const float *ws,
                              float *out, int64_t start, int64_t count) {
  float32x4_t vxs = vdupq_n_f32(xs);
  int64_t n = start, end = start + count;
  for (; n + 4 <= end; n += 4)
    vst1q_f32(out + n, vmulq_f32(vmulq_f32(vcvtq_f32_s32(vld1q_s32(oi + n)), vxs),
                                 vld1q_f32(ws + n)));
  for (; n < end; n++) out[n] = (float) oi[n] * xs * ws[n];
}
// Block-kb GEMV from a PRE-quantized int8 activation, dequant straight to f32.
EXPORT void sdot_gemv_blk_prequant_f32_range(const int8_t *xq, float x_scale,
                                             const int8_t *B_blk, const float *w_scale,
                                             float *out_f32, int64_t K, int64_t N, int64_t N4,
                                             int64_t BN4, int64_t blk_start, int64_t blk_count) {
  if (blk_count <= 0) return;
  int64_t K4 = K / 4;
  int64_t g0 = blk_start * BN4, gend = (blk_start + blk_count) * BN4;
  if (gend > N4) gend = N4;
  if (g0 >= N4) return;
  std::vector<int32_t> local_buf((gend - g0) * 4);
  int32_t *local = local_buf.data() - g0 * 4;
  sdot_gemv_blk_range(xq, B_blk, local, K4, N4, BN4, blk_start, blk_count);
  dequant_range_f32(local, x_scale, w_scale, out_f32, g0 * 4, (gend - g0) * 4);
}

// ---- SME int8 GEMM (prefill). One shared SME unit -> GEMM runs on ONE thread;
//      activation pack + dequant parallelize across the other ggml threads. ----
#ifndef TLE_NO_SME_KERNEL
extern "C" void sme_tile_16x64(const int8_t *, const int8_t *, int32_t *,
                               int64_t, int64_t);  // runtime_sme_kernel.S
#else
// Assembler lacks SME2; the .S kernel is not in this build. Fail loudly
// instead of computing garbage -- runtime feature dispatch must keep the
// SME paths unreachable on such builds.
static void sme_tile_16x64(const int8_t *, const int8_t *, int32_t *,
                           int64_t, int64_t) {
  fprintf(stderr,
          "TritonCPURuntime: sme_tile_16x64 called but the runtime was built "
          "without the SME kernel (toolchain lacks SME2)\n");
  abort();
}
#endif

// Pack weight [N][K] int8 -> SME Bp [Np/64][K4][4][16][4] (Np = ceil(N/64)*64).
EXPORT void sme_pack_weights(const int8_t *W, int8_t *Bp, int64_t N, int64_t K) {
  int64_t K4 = K / 4, Np = (N + 63) / 64 * 64, NB = Np / 64;
  for (int64_t nt = 0; nt < NB; nt++)
    for (int64_t kb = 0; kb < K4; kb++)
      for (int b = 0; b < 4; b++)
        for (int j = 0; j < 16; j++) {
          int64_t c = nt * 64 + b * 16 + j;
          int8_t *d = Bp + nt * K4 * 256 + kb * 256 + b * 64 + j * 4;
          for (int k4 = 0; k4 < 4; k4++) d[k4] = (c < N) ? W[c * K + 4 * kb + k4] : 0;
        }
}

// Per-row symmetric int8 quant of activation A[M][K] (row stride lda bytes) +
// pack to SME Ap [Mp/16][K4][16][4]; xs[M] = per-row scale. Threaded over rows.
EXPORT void sme_quant_pack_act_f32(const float *A, int8_t *Ap, float *xs,
                                   int64_t M, int64_t K, int64_t lda,
                                   int64_t m_start, int64_t m_count) {
  int64_t K4 = K / 4;
  for (int64_t r = m_start; r < m_start + m_count && r < M; r++) {
    const float *arow = (const float *) ((const char *) A + r * lda);
    float amax = 0.0f;
    for (int64_t k = 0; k < K; k++) { float v = arow[k]; v = v < 0 ? -v : v; if (v > amax) amax = v; }
    xs[r] = amax / 127.0f;
    float inv = amax > 1e-12f ? 127.0f / amax : 0.0f;
    int64_t mt = r / 16, i = r % 16;
    int8_t *base = Ap + mt * K4 * 64 + i * 4;
    for (int64_t kb = 0; kb < K4; kb++)
      for (int k4 = 0; k4 < 4; k4++) {
        int val = (int) lrintf(arow[4 * kb + k4] * inv);
        if (val > 127) val = 127; if (val < -128) val = -128;
        base[kb * 64 + k4] = (int8_t) val;
      }
  }
}

// SME GEMM: Ap x Bp -> C int32 [Mp][Np] (padded). Runs on ONE thread (single SME unit).
// Defense-in-depth: callers must consult tle_cpu_has_sme() before selecting
// this path; if they didn't, fail loudly here instead of SIGILLing in the
// tile kernel. (The per-tile sme_uk hot path is gated by the caller only.)
EXPORT void sme_gemm_int32(const int8_t *Ap, const int8_t *Bp, int32_t *C,
                           int64_t Mp, int64_t Np, int64_t K4) {
  if (!tle_cpu_has_sme()) {
    fprintf(stderr, "TritonCPURuntime: sme_gemm_int32 called on a CPU/build "
                    "without usable SME (check tle_cpu_has_sme() first)\n");
    abort();
  }
  int64_t MT = Mp / 16, NT = Np / 64;
  for (int64_t mt = 0; mt < MT; mt++)
    for (int64_t nt = 0; nt < NT; nt++)
      sme_tile_16x64(Ap + mt * K4 * 64, Bp + nt * K4 * 256,
                     C + (mt * 16) * Np + nt * 64, K4, Np);
}

// ===== TLE-Struct micro-kernel: ONE fixed 8x8 SMMLA output tile (4 mp-pairs x
//       4 np-pairs, 16 accumulators in registers). This is the "Raw leaf"; the
//       M/N tiling is orchestrated by the upper Triton kernel (the Struct layer).
//       Ap ko-major [K8][MP][16], Wp n-major [N/2][K8][16], C row-major + dequant.
#if defined(__ARM_FEATURE_MATMUL_INT8)
EXPORT void smmla_uk(const int8_t *Ap, const int8_t *Wp, float *C,
                     const float *xs, const float *ws,
                     int64_t K8, int64_t MP, int64_t N, int64_t mp0, int64_t np0) {
  const int8_t *w0b = Wp + np0 * K8 * 16, *w1b = w0b + K8 * 16,
               *w2b = w1b + K8 * 16, *w3b = w2b + K8 * 16;
  int32x4_t acc[4][4];
  for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) acc[i][j] = vdupq_n_s32(0);
  for (int64_t ko = 0; ko < K8; ko++) {
    int8x16_t w0 = vld1q_s8(w0b + ko*16), w1 = vld1q_s8(w1b + ko*16),
              w2 = vld1q_s8(w2b + ko*16), w3 = vld1q_s8(w3b + ko*16);
    const int8_t *apk = Ap + ko * MP * 16 + mp0 * 16;
    int8x16_t a0 = vld1q_s8(apk), a1 = vld1q_s8(apk+16),
              a2 = vld1q_s8(apk+32), a3 = vld1q_s8(apk+48);
    acc[0][0]=vmmlaq_s32(acc[0][0],a0,w0);acc[0][1]=vmmlaq_s32(acc[0][1],a0,w1);acc[0][2]=vmmlaq_s32(acc[0][2],a0,w2);acc[0][3]=vmmlaq_s32(acc[0][3],a0,w3);
    acc[1][0]=vmmlaq_s32(acc[1][0],a1,w0);acc[1][1]=vmmlaq_s32(acc[1][1],a1,w1);acc[1][2]=vmmlaq_s32(acc[1][2],a1,w2);acc[1][3]=vmmlaq_s32(acc[1][3],a1,w3);
    acc[2][0]=vmmlaq_s32(acc[2][0],a2,w0);acc[2][1]=vmmlaq_s32(acc[2][1],a2,w1);acc[2][2]=vmmlaq_s32(acc[2][2],a2,w2);acc[2][3]=vmmlaq_s32(acc[2][3],a2,w3);
    acc[3][0]=vmmlaq_s32(acc[3][0],a3,w0);acc[3][1]=vmmlaq_s32(acc[3][1],a3,w1);acc[3][2]=vmmlaq_s32(acc[3][2],a3,w2);acc[3][3]=vmmlaq_s32(acc[3][3],a3,w3);
  }
  for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++)
    smmla_store(C, N, xs, ws, mp0 + i, np0 + j, acc[i][j]);
}
#endif // __ARM_FEATURE_MATMUL_INT8

#if !defined(__ARM_FEATURE_MATMUL_INT8)
// i8mm not available in this build: keep the C ABI symbol-complete. The GEMM
// entry points abort (silent wrong results are worse); runtime dispatch
// (tle_cpu_has_i8mm) must keep them unreachable.
static void tle_no_i8mm_abort(const char *fn) {
  fprintf(stderr, "TritonCPURuntime: %s called but the runtime was built "
                  "without i8mm (SMMLA) support\n", fn);
  abort();
}
EXPORT void smmla_pack_weights(const int8_t *, int8_t *, int64_t, int64_t) {}
EXPORT void smmla_quant_pack_act_f32(const float *, int8_t *, float *, int64_t,
                                     int64_t, int64_t, int64_t) {}
EXPORT void smmla_gemm_range(const int8_t *, const float *, const int8_t *,
                             const float *, float *, int64_t, int64_t, int64_t,
                             int64_t, int64_t) {
  tle_no_i8mm_abort("smmla_gemm_range");
}
EXPORT void smmla_uk(const int8_t *, const int8_t *, float *, const float *,
                     const float *, int64_t, int64_t, int64_t, int64_t,
                     int64_t) {
  tle_no_i8mm_abort("smmla_uk");
}
#endif // !__ARM_FEATURE_MATMUL_INT8

// ===== More TLE-Struct micro-kernels: each is a fixed Raw leaf; the M/N/row/block
//       tiling is orchestrated by the upper @triton.jit kernel (the Struct layer).
//       These wrap the existing monolithic ops at single-tile granularity. =====
extern "C" void swiglu_bf16(const uint16_t *, const uint16_t *, uint16_t *, int64_t);
extern "C" void standalone_rms_norm_bf16(const uint16_t *, const uint16_t *, uint16_t *, int64_t, float);
extern "C" void standalone_residual_add_bf16(uint16_t *, const uint16_t *, int64_t);

// SME int8 GEMM: ONE 16x64 tile (mt,nt). Raw leaf = sme_tile_16x64.
EXPORT void sme_uk(const int8_t *Ap, const int8_t *Bp, int32_t *C,
                   int64_t K4, int64_t Np, int64_t mt, int64_t nt) {
  sme_tile_16x64(Ap + mt * K4 * 64, Bp + nt * K4 * 256,
                 C + (mt * 16) * Np + nt * 64, K4, Np);
}
// Decode int8 GEMV: ONE block (BN4 output groups) over full K. Raw leaf = block-kb GEMV.
EXPORT void sdot_gemv_uk(const int8_t *A, const int8_t *B_blk, int32_t *C,
                         int64_t K4, int64_t N4, int64_t BN4, int64_t blk) {
  sdot_gemv_blk_range(A, B_blk, C, K4, N4, BN4, blk, 1);
}
// SwiGLU: ONE block [off, off+n) of a bf16 vector. Raw leaf = swiglu_bf16.
EXPORT void swiglu_uk(const uint16_t *gate, const uint16_t *up, uint16_t *out,
                      int64_t off, int64_t n) {
  swiglu_bf16(gate + off, up + off, out + off, n);
}
// RMSNorm: ONE row (row*D .. +D) of a [rows][D] bf16 tensor. Raw leaf = rms_norm_bf16.
EXPORT void rmsnorm_uk(const uint16_t *x, const uint16_t *weight, uint16_t *out,
                       int64_t D, int64_t row) {
  standalone_rms_norm_bf16(x + row * D, weight, out + row * D, D, 1e-6f);
}
// Residual add: ONE block [off, off+n). Raw leaf = standalone_residual_add_bf16.
EXPORT void residual_uk(uint16_t *residual, const uint16_t *x, int64_t off, int64_t n) {
  standalone_residual_add_bf16(residual + off, x + off, n);
}

#else
// Stub for non-ARM platforms
EXPORT void sdot_pack_weights(const int8_t *, int8_t *, int64_t, int64_t) {}
EXPORT void sdot_gemv_m1_prepacked(const int8_t *, const int8_t *,
                                    int32_t *, int64_t, int64_t, int64_t) {}
EXPORT void sdot_gemv_m1_fused_bf16(const uint16_t *, const int8_t *,
                                     const float *, uint16_t *,
                                     int64_t, int64_t, int64_t) {}
EXPORT void sdot_gemv_m1_fused_bf16_range(const uint16_t *, const int8_t *,
                                          const float *, uint16_t *,
                                          int64_t, int64_t, int64_t,
                                          int64_t, int64_t) {}
EXPORT void sdot_pack_nmajor(const int8_t *, int8_t *, int64_t, int64_t) {}
EXPORT void sdot_gemv_m1_fused_bf16_nmajor_range(const uint16_t *, const int8_t *,
                                                 const float *, uint16_t *,
                                                 int64_t, int64_t, int64_t,
                                                 int64_t, int64_t) {}
EXPORT void sdot_pack_blk(const int8_t *, int8_t *, int64_t, int64_t, int64_t) {}
EXPORT float sdot_quant_act_f32(const float *, int8_t *, int64_t) { return 0; }
EXPORT void sdot_gemv_blk_prequant_f32_range(const int8_t *, float, const int8_t *,
                                             const float *, float *, int64_t, int64_t,
                                             int64_t, int64_t, int64_t, int64_t) {}
EXPORT void sdot_gemv_m1_fused_bf16_blk_range(const uint16_t *, const int8_t *,
                                              const float *, uint16_t *,
                                              int64_t, int64_t, int64_t,
                                              int64_t, int64_t, int64_t) {}
EXPORT void sme_pack_weights(const int8_t *, int8_t *, int64_t, int64_t) {}
EXPORT void sme_quant_pack_act_f32(const float *, int8_t *, float *, int64_t, int64_t,
                                   int64_t, int64_t, int64_t) {}
EXPORT void sme_gemm_int32(const int8_t *, const int8_t *, int32_t *,
                           int64_t, int64_t, int64_t) {}
EXPORT void smmla_pack_weights(const int8_t *, int8_t *, int64_t, int64_t) {}
EXPORT void smmla_quant_pack_act_f32(const float *, int8_t *, float *, int64_t, int64_t,
                                     int64_t, int64_t) {}
EXPORT void smmla_gemm_range(const int8_t *, const float *, const int8_t *, const float *,
                             float *, int64_t, int64_t, int64_t, int64_t, int64_t) {}
EXPORT void smmla_uk(const int8_t *, const int8_t *, float *, const float *, const float *,
                     int64_t, int64_t, int64_t, int64_t, int64_t) {}
EXPORT void sme_uk(const int8_t *, const int8_t *, int32_t *, int64_t, int64_t, int64_t, int64_t) {}
EXPORT void sdot_gemv_uk(const int8_t *, const int8_t *, int32_t *, int64_t, int64_t, int64_t, int64_t) {}
EXPORT void swiglu_uk(const uint16_t *, const uint16_t *, uint16_t *, int64_t, int64_t) {}
EXPORT void rmsnorm_uk(const uint16_t *, const uint16_t *, uint16_t *, int64_t, int64_t) {}
EXPORT void residual_uk(uint16_t *, const uint16_t *, int64_t, int64_t) {}
#endif

} // extern "C"
