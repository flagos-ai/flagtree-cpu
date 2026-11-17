/*
 * Fused SWIGLU: silu(gate) * up in a single pass.
 * Replaces F.silu(x1) * x2 (two ATen calls) with one NEON kernel.
 *
 * BF16 input/output. Single-threaded for decode shapes (N <= 6144).
 */

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(_MSC_VER)
#define EXPORT __declspec(dllexport)
#elif defined(__GNUC__)
#define EXPORT __attribute__((visibility("default")))
#else
#define EXPORT
#endif

extern "C" {

#if defined(__aarch64__) && defined(__ARM_NEON)

static inline float32x4_t bf16_to_fp32(uint16x4_t bf16) {
  return vreinterpretq_f32_u32(vshll_n_u16(bf16, 16));
}

static inline uint16x4_t fp32_to_bf16(float32x4_t fp32) {
  return vshrn_n_u32(vreinterpretq_u32_f32(fp32), 16);
}

/*
 * Fast NEON exp approximation (Schraudolph's method, ~1% relative error).
 * exp(x) ≈ 2^(x * log2(e)) via integer bit manipulation.
 * Good enough for BF16 precision (7-bit mantissa).
 */
static inline float32x4_t neon_exp_fast(float32x4_t x) {
  /* Clamp to avoid overflow/underflow */
  x = vmaxq_f32(x, vdupq_n_f32(-88.0f));
  x = vminq_f32(x, vdupq_n_f32(88.0f));
  /* exp(x) = 2^(x/ln2) */
  const float32x4_t log2e = vdupq_n_f32(1.4426950408889634f);
  const float32x4_t shift = vdupq_n_f32(12582912.0f); /* 1.5 * 2^23 */
  const float32x4_t C1 = vdupq_n_f32(121.2740575f);   /* 2^23 / ln2 */
  const float32x4_t C2 = vdupq_n_f32(27.7280233f);    /* correction */
  const float32x4_t C3 = vdupq_n_f32(4.84252568f);    /* correction */
  /* Compute 2^(x*log2e) using Cephes polynomial */
  float32x4_t t = vmulq_f32(x, log2e);
  float32x4_t ti = vsubq_f32(vaddq_f32(t, shift), shift); /* floor(t) */
  float32x4_t tf = vsubq_f32(t, ti);  /* fractional part */
  /* Polynomial for 2^tf on [0,1] */
  float32x4_t p = vaddq_f32(tf, vdupq_n_f32(1.0f));
  p = vaddq_f32(vmulq_f32(tf, vaddq_f32(vmulq_f32(tf, vdupq_n_f32(0.2402265f)),
                                          vdupq_n_f32(0.6931472f))),
                vdupq_n_f32(1.0f));
  /* Scale by 2^floor(t) */
  int32x4_t ii = vcvtq_s32_f32(ti);
  ii = vshlq_n_s32(ii, 23);  /* shift to exponent field */
  int32x4_t pi = vreinterpretq_s32_f32(p);
  return vreinterpretq_f32_s32(vaddq_s32(pi, ii));
}

static inline float32x4_t neon_sigmoid(float32x4_t x) {
  float32x4_t exp_neg = neon_exp_fast(vnegq_f32(x));
  float32x4_t one = vdupq_n_f32(1.0f);
  /* Use Newton-Raphson for 1/(1+exp_neg) via NEON reciprocal estimate */
  float32x4_t denom = vaddq_f32(one, exp_neg);
  float32x4_t recip = vrecpeq_f32(denom);
  recip = vmulq_f32(recip, vrecpsq_f32(denom, recip)); /* 1 Newton step */
  return recip;
}

/*
 * swiglu_bf16: out[i] = silu(gate[i]) * up[i]
 *            = gate[i] * sigmoid(gate[i]) * up[i]
 *
 * gate: [N] bf16, up: [N] bf16, out: [N] bf16
 * Single-threaded (decode N <= 6144, OMP fork overhead > compute).
 */
EXPORT void swiglu_bf16(const uint16_t *gate, const uint16_t *up,
                         uint16_t *out, int64_t N) {
  int64_t n = 0;
  for (; n + 4 <= N; n += 4) {
    float32x4_t g = bf16_to_fp32(vld1_u16(gate + n));
    float32x4_t u = bf16_to_fp32(vld1_u16(up + n));
    /* silu(g) * u = g * sigmoid(g) * u */
    float32x4_t sig = neon_sigmoid(g);
    float32x4_t result = vmulq_f32(vmulq_f32(g, sig), u);
    vst1_u16(out + n, fp32_to_bf16(result));
  }
  for (; n < N; n++) {
    uint32_t gb = (uint32_t)gate[n] << 16;
    uint32_t ub = (uint32_t)up[n] << 16;
    float gf, uf;
    std::memcpy(&gf, &gb, 4);
    std::memcpy(&uf, &ub, 4);
    float sig = 1.0f / (1.0f + expf(-gf));
    float r = gf * sig * uf;
    uint32_t rb;
    std::memcpy(&rb, &r, 4);
    out[n] = (uint16_t)(rb >> 16);
  }
}

#else
EXPORT void swiglu_bf16(const uint16_t *, const uint16_t *,
                         uint16_t *, int64_t) {}
#endif

} // extern "C"
