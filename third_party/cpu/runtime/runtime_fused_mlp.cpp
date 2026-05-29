/*
 * Fused MLP: quantize + gate SDOT + up SDOT + SWIGLU in one OMP region.
 * Saves: 1 OMP fork/join, 2 intermediate tensors, 1 Python dispatch.
 */

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include <omp.h>

#if defined(__GNUC__)
#define EXPORT __attribute__((visibility("default")))
#else
#define EXPORT
#endif

extern "C" {

#if defined(__aarch64__) && defined(__ARM_NEON)

static inline float32x4_t fast_exp_mlp(float32x4_t x) {
  x = vmaxq_f32(x, vdupq_n_f32(-88.0f));
  x = vminq_f32(x, vdupq_n_f32(88.0f));
  float32x4_t t = vmulq_n_f32(x, 1.4426950408889634f); /* x * log2(e) */
  float32x4_t shift = vdupq_n_f32(12582912.0f);
  float32x4_t ti = vsubq_f32(vaddq_f32(t, shift), shift);
  float32x4_t tf = vsubq_f32(t, ti);
  float32x4_t p = vaddq_f32(vmulq_f32(tf, vaddq_f32(
      vmulq_f32(tf, vdupq_n_f32(0.2402265f)), vdupq_n_f32(0.6931472f))),
      vdupq_n_f32(1.0f));
  int32x4_t ii = vshlq_n_s32(vcvtq_s32_f32(ti), 23);
  return vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(p), ii));
}

/*
 * fused_mlp_bf16:
 *   x_bf16: [K] bf16 activation
 *   gate_packed, up_packed: [K/4, N/4, 16] int8 SDOT-format weights
 *   gate_scale, up_scale: [N] float32 per-channel weight scales
 *   out_bf16: [N] bf16 output = silu(gate_proj(x)) * up_proj(x)
 */
EXPORT void fused_mlp_bf16(
    const uint16_t *x_bf16,
    const int8_t *gate_packed,
    const int8_t *up_packed,
    const float *gate_scale,
    const float *up_scale,
    uint16_t *out_bf16,
    int64_t K, int64_t N
) {
  int64_t K4 = K / 4;
  int64_t N4 = N / 4;

  /* Step 1: Quantize x_bf16 → x_int8 + x_scale (sequential, ~2us for K=2048) */
  int8_t x_int8[32768];
  float32x4_t vmax = vdupq_n_f32(0.0f);
  for (int64_t k = 0; k + 4 <= K; k += 4) {
    float32x4_t v = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(x_bf16 + k), 16));
    vmax = vmaxq_f32(vmax, vabsq_f32(v));
  }
  float absmax = vmaxvq_f32(vmax);
  if (absmax < 1e-10f) absmax = 1e-10f;
  float x_scale = absmax / 127.0f;
  float inv_x = 127.0f / absmax;
  float32x4_t vinv = vdupq_n_f32(inv_x);
  for (int64_t k = 0; k + 4 <= K; k += 4) {
    float32x4_t v = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(x_bf16 + k), 16));
    int32x4_t r = vcvtnq_s32_f32(vmulq_f32(v, vinv));
    r = vmaxq_s32(r, vdupq_n_s32(-128));
    r = vminq_s32(r, vdupq_n_s32(127));
    int16x4_t n16 = vmovn_s32(r);
    int8x8_t n8 = vmovn_s16(vcombine_s16(n16, n16));
    vst1_lane_s32((int32_t *)(x_int8 + k), vreinterpret_s32_s8(n8), 0);
  }

  /* Step 2+3: gate SDOT + up SDOT + SWIGLU (K-outer, N-range per thread) */
  #pragma omp parallel
  {
    int nt = omp_get_num_threads();
    int tid = omp_get_thread_num();
    int64_t chunk = (N4 + nt - 1) / nt;
    int64_t start = tid * chunk;
    int64_t count = chunk;
    if (start + count > N4) count = N4 - start;
    if (start >= N4) count = 0;

    if (count > 0) {
      /* Accumulators for gate and up */
      std::vector<int32x4_t> gate_acc(count), up_acc(count);
      for (int64_t i = 0; i < count; i++) {
        gate_acc[i] = vdupq_n_s32(0);
        up_acc[i] = vdupq_n_s32(0);
      }

      /* K-outer loop: load A once, apply to all N-groups in range */
      for (int64_t k4 = 0; k4 < K4; k4++) {
        int32_t a4;
        std::memcpy(&a4, x_int8 + k4 * 4, 4);
        int8x16_t av = vreinterpretq_s8_s32(vdupq_n_s32(a4));

        const int8_t *gp = gate_packed + (k4 * N4 + start) * 16;
        const int8_t *up = up_packed + (k4 * N4 + start) * 16;

        for (int64_t i = 0; i < count; i++) {
          gate_acc[i] = vdotq_s32(gate_acc[i], av, vld1q_s8(gp + i * 16));
          up_acc[i]   = vdotq_s32(up_acc[i],   av, vld1q_s8(up + i * 16));
        }
      }

      /* Dequant + SWIGLU: silu(gate * x_scale * gate_scale) * (up * x_scale * up_scale) */
      for (int64_t i = 0; i < count; i++) {
        int64_t n = start + i;
        float32x4_t gf = vcvtq_f32_s32(gate_acc[i]);
        float32x4_t uf = vcvtq_f32_s32(up_acc[i]);
        float32x4_t gs = vld1q_f32(gate_scale + n * 4);
        float32x4_t us = vld1q_f32(up_scale + n * 4);
        gf = vmulq_f32(vmulq_n_f32(gf, x_scale), gs);
        uf = vmulq_f32(vmulq_n_f32(uf, x_scale), us);

        /* silu(gate) * up */
        float32x4_t exp_neg = fast_exp_mlp(vnegq_f32(gf));
        float32x4_t denom = vaddq_f32(vdupq_n_f32(1.0f), exp_neg);
        float32x4_t recip = vrecpeq_f32(denom);
        recip = vmulq_f32(recip, vrecpsq_f32(denom, recip));
        float32x4_t result = vmulq_f32(vmulq_f32(gf, recip), uf);

        uint16x4_t bf = vshrn_n_u32(vreinterpretq_u32_f32(result), 16);
        vst1_u16(out_bf16 + n * 4, bf);
      }
    }
  }
}

#else
EXPORT void fused_mlp_bf16(
    const uint16_t *, const int8_t *, const int8_t *,
    const float *, const float *, uint16_t *,
    int64_t, int64_t) {}
#endif

} // extern "C"
