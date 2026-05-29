/*
 * Flash Attention decode kernel for M=1 (single token generation).
 * Replaces ATen SDPA fallback with NEON-optimized per-row online softmax.
 *
 * Algorithm: for each Q head, iterate over KV cache entries:
 *   1. Compute q . k[i] via NEON dot product (BF16→FP32)
 *   2. Online softmax: track running max, sum of exp
 *   3. Accumulate weighted V
 *   4. Final normalization
 *
 * OMP parallelized across heads.
 * Supports GQA (num_kv_heads < num_heads).
 */

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#include <cmath>
#include <cstdint>
#include <cstring>
#include <omp.h>

#if defined(_MSC_VER)
#define EXPORT __declspec(dllexport)
#elif defined(__GNUC__)
#define EXPORT __attribute__((visibility("default")))
#else
#define EXPORT
#endif

extern "C" {

#if defined(__aarch64__) && defined(__ARM_NEON)

static inline float32x4_t bf16x4_to_fp32(const uint16_t *p) {
  return vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(p), 16));
}

/*
 * flash_attn_decode_bf16:
 *   Q:   [num_heads, 1, head_dim] bf16 (contiguous, already squeezed from [B,H,1,D])
 *   K:   [num_kv_heads, seq_len, head_dim] bf16
 *   V:   [num_kv_heads, seq_len, head_dim] bf16
 *   Out: [num_heads, 1, head_dim] bf16
 *
 * Strides are passed explicitly for K and V (may not be contiguous in cache).
 */
EXPORT void flash_attn_decode_bf16(
    const uint16_t *Q,    /* [num_heads, head_dim] */
    const uint16_t *K,    /* [num_kv_heads, seq_len, head_dim] */
    const uint16_t *V,    /* [num_kv_heads, seq_len, head_dim] */
    uint16_t *Out,        /* [num_heads, head_dim] */
    int64_t seq_len,
    int64_t head_dim,
    float sm_scale,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t stride_kn,    /* K stride along seq_len dimension (in bf16 elements) */
    int64_t stride_vn     /* V stride along seq_len dimension */
) {
  int64_t gqa_ratio = num_heads / num_kv_heads;

  #pragma omp parallel for schedule(static)
  for (int64_t h = 0; h < num_heads; h++) {
    int64_t kv_h = h / gqa_ratio;

    const uint16_t *q_row = Q + h * head_dim;
    const uint16_t *k_base = K + kv_h * seq_len * stride_kn;
    const uint16_t *v_base = V + kv_h * seq_len * stride_vn;
    uint16_t *out_row = Out + h * head_dim;

    /* Convert Q to FP32 once */
    float q_fp32[256]; /* max head_dim = 256 */
    for (int64_t d = 0; d < head_dim; d += 4) {
      float32x4_t qv = bf16x4_to_fp32(q_row + d);
      qv = vmulq_n_f32(qv, sm_scale);
      vst1q_f32(q_fp32 + d, qv);
    }

    /* Online softmax + weighted V accumulation */
    float running_max = -1e30f;
    float running_sum = 0.0f;
    float acc[256]; /* accumulated weighted V */
    std::memset(acc, 0, head_dim * sizeof(float));

    for (int64_t s = 0; s < seq_len; s++) {
      const uint16_t *k_row = k_base + s * stride_kn;
      const uint16_t *v_row = v_base + s * stride_vn;

      /* Compute q . k[s] */
      float32x4_t dot_acc = vdupq_n_f32(0.0f);
      for (int64_t d = 0; d < head_dim; d += 4) {
        float32x4_t q4 = vld1q_f32(q_fp32 + d);
        float32x4_t k4 = bf16x4_to_fp32(k_row + d);
        dot_acc = vfmaq_f32(dot_acc, q4, k4);
      }
      float qk = vaddvq_f32(dot_acc); /* horizontal sum */

      /* Online softmax update */
      float old_max = running_max;
      if (qk > running_max) running_max = qk;

      float exp_old = expf(old_max - running_max);
      float exp_new = expf(qk - running_max);

      /* Rescale existing accumulator */
      running_sum = running_sum * exp_old + exp_new;

      /* Accumulate: acc = acc * exp_old + v[s] * exp_new */
      float32x4_t scale_old = vdupq_n_f32(exp_old);
      float32x4_t scale_new = vdupq_n_f32(exp_new);
      for (int64_t d = 0; d < head_dim; d += 4) {
        float32x4_t a = vld1q_f32(acc + d);
        float32x4_t v4 = bf16x4_to_fp32(v_row + d);
        a = vfmaq_f32(vmulq_f32(a, scale_old), v4, scale_new);
        vst1q_f32(acc + d, a);
      }
    }

    /* Normalize and convert to BF16 */
    float inv_sum = 1.0f / running_sum;
    for (int64_t d = 0; d < head_dim; d += 4) {
      float32x4_t a = vmulq_n_f32(vld1q_f32(acc + d), inv_sum);
      uint32x4_t au = vreinterpretq_u32_f32(a);
      uint16x4_t bf = vshrn_n_u32(au, 16);
      vst1_u16(out_row + d, bf);
    }
  }
}

#else
EXPORT void flash_attn_decode_bf16(
    const uint16_t *, const uint16_t *, const uint16_t *, uint16_t *,
    int64_t, int64_t, float, int64_t, int64_t, int64_t, int64_t) {}
#endif

} // extern "C"
