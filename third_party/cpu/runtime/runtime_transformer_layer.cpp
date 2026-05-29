/*
 * Fused Transformer Decode Layer (M=1) for Qwen3-style models.
 * Single C function replaces ~125 Python dispatches per layer.
 *
 * Flow: RMSNorm → QKV GEMV → QK_Norm → RoPE → KV_cache_write →
 *       Attention → O_GEMV → Residual → RMSNorm → Gate+Up+SWIGLU → Down_GEMV → Residual
 *
 * All intermediate results on stack or pre-allocated buffer. Zero tensor allocation.
 */

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#include <algorithm>
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

#if defined(__aarch64__) && defined(__ARM_NEON)

/* ── Helpers ───────────────────────────────────────────── */

static inline float32x4_t bf16x4_to_f32(const uint16_t *p) {
  return vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(p), 16));
}
static inline void f32_to_bf16x4(uint16_t *p, float32x4_t v) {
  vst1_u16(p, vshrn_n_u32(vreinterpretq_u32_f32(v), 16));
}

/* ── RMSNorm (inline, no allocation) ───────────────────── */

static void rms_norm_bf16(const uint16_t *x, const uint16_t *weight,
                           uint16_t *out, int64_t D, float eps) {
  /* Compute sum of squares */
  float32x4_t ss = vdupq_n_f32(0.0f);
  for (int64_t d = 0; d + 4 <= D; d += 4) {
    float32x4_t v = bf16x4_to_f32(x + d);
    ss = vfmaq_f32(ss, v, v);
  }
  float sum_sq = vaddvq_f32(ss);
  float rms = 1.0f / sqrtf(sum_sq / (float)D + eps);

  /* Normalize and scale */
  float32x4_t vrms = vdupq_n_f32(rms);
  for (int64_t d = 0; d + 4 <= D; d += 4) {
    float32x4_t v = vmulq_f32(bf16x4_to_f32(x + d), vrms);
    float32x4_t w = bf16x4_to_f32(weight + d);
    f32_to_bf16x4(out + d, vmulq_f32(v, w));
  }
}

/* ── Quantize BF16 → INT8 ─────────────────────────────── */

static float quantize_bf16_to_int8(const uint16_t *x, int8_t *out, int64_t K) {
  float32x4_t vmax = vdupq_n_f32(0.0f);
  for (int64_t k = 0; k + 4 <= K; k += 4) {
    float32x4_t v = bf16x4_to_f32(x + k);
    vmax = vmaxq_f32(vmax, vabsq_f32(v));
  }
  float absmax = vmaxvq_f32(vmax);
  if (absmax < 1e-10f) absmax = 1e-10f;
  float inv = 127.0f / absmax;
  float32x4_t vinv = vdupq_n_f32(inv);
  for (int64_t k = 0; k + 4 <= K; k += 4) {
    float32x4_t v = bf16x4_to_f32(x + k);
    int32x4_t r = vcvtnq_s32_f32(vmulq_f32(v, vinv));
    r = vmaxq_s32(r, vdupq_n_s32(-128));
    r = vminq_s32(r, vdupq_n_s32(127));
    int16x4_t n16 = vmovn_s32(r);
    int8x8_t n8 = vmovn_s16(vcombine_s16(n16, n16));
    vst1_lane_s32((int32_t *)(out + k), vreinterpret_s32_s8(n8), 0);
  }
  return absmax / 127.0f;
}

/* ── SDOT GEMV M=1 (single-thread, for use inside OMP region) ── */

static void sdot_gemv_st(const int8_t *A, const int8_t *B_packed,
                          int32_t *C, int64_t K4, int64_t N4,
                          int64_t start, int64_t count) {
  /* Same as sdot_gemv_range but no OMP (called from within OMP region) */
  int32x4_t acc_stack[4096];
  int32x4_t *acc = acc_stack;
  std::vector<int32x4_t> acc_heap;
  if (count > 4096) { acc_heap.resize(count); acc = acc_heap.data(); }
  for (int64_t i = 0; i < count; i++) acc[i] = vdupq_n_s32(0);

  for (int64_t kb = 0; kb < K4; kb++) {
    int32_t a4;
    std::memcpy(&a4, A + kb * 4, 4);
    int8x16_t av = vreinterpretq_s8_s32(vdupq_n_s32(a4));
    const int8_t *bp = B_packed + (kb * N4 + start) * 16;
    for (int64_t i = 0; i < count; i++) {
      acc[i] = vdotq_s32(acc[i], av, vld1q_s8(bp + i * 16));
    }
  }
  for (int64_t i = 0; i < count; i++)
    vst1q_s32(C + (start + i) * 4, acc[i]);
}

/* ── Dequant INT32 → BF16 with per-channel scale ──────── */

static void dequant_to_bf16(const int32_t *acc, const float *w_scale,
                              float x_scale, uint16_t *out, int64_t N) {
  for (int64_t n = 0; n + 4 <= N; n += 4) {
    float32x4_t v = vcvtq_f32_s32(vld1q_s32(acc + n));
    float32x4_t ws = vld1q_f32(w_scale + n);
    v = vmulq_f32(vmulq_n_f32(v, x_scale), ws);
    f32_to_bf16x4(out + n, v);
  }
}

/* ── SDOT GEMV full (OMP parallel, bf16 in → bf16 out) ── */

static void gemv_int8_bf16(const uint16_t *x_bf16, int64_t K,
                            const int8_t *w_packed, const float *w_scale,
                            uint16_t *out_bf16, int64_t N,
                            int8_t *x_int8_buf, float *x_scale_out) {
  int64_t K4 = K / 4, N4 = N / 4;
  float xs = quantize_bf16_to_int8(x_bf16, x_int8_buf, K);
  *x_scale_out = xs;

  std::vector<int32_t> acc_vec(N);
  int32_t *acc_buf = acc_vec.data();

  #pragma omp parallel
  {
    int nt = omp_get_num_threads();
    int tid = omp_get_thread_num();
    int64_t chunk = (N4 + nt - 1) / nt;
    int64_t s = tid * chunk, c = chunk;
    if (s + c > N4) c = N4 - s;
    if (s < N4 && c > 0)
      sdot_gemv_st(x_int8_buf, w_packed, acc_buf, K4, N4, s, c);
  }

  dequant_to_bf16(acc_buf, w_scale, xs, out_bf16, N);
}

/* ── RoPE (inline, cos/sin embedding) ──────────────────── */

static void apply_rope_bf16(uint16_t *q, uint16_t *k,
                              const uint16_t *cos_emb, const uint16_t *sin_emb,
                              int64_t n_heads, int64_t n_kv_heads, int64_t head_dim) {
  int64_t half = head_dim / 2;
  /* Apply to Q heads */
  for (int64_t h = 0; h < n_heads; h++) {
    uint16_t *qh = q + h * head_dim;
    for (int64_t d = 0; d + 4 <= half; d += 4) {
      float32x4_t q0 = bf16x4_to_f32(qh + d);
      float32x4_t q1 = bf16x4_to_f32(qh + half + d);
      float32x4_t c = bf16x4_to_f32(cos_emb + d);
      float32x4_t s = bf16x4_to_f32(sin_emb + d);
      float32x4_t r0 = vsubq_f32(vmulq_f32(q0, c), vmulq_f32(q1, s));
      float32x4_t r1 = vaddq_f32(vmulq_f32(q0, s), vmulq_f32(q1, c));
      f32_to_bf16x4(qh + d, r0);
      f32_to_bf16x4(qh + half + d, r1);
    }
  }
  /* Apply to K heads */
  for (int64_t h = 0; h < n_kv_heads; h++) {
    uint16_t *kh = k + h * head_dim;
    for (int64_t d = 0; d + 4 <= half; d += 4) {
      float32x4_t k0 = bf16x4_to_f32(kh + d);
      float32x4_t k1 = bf16x4_to_f32(kh + half + d);
      float32x4_t c = bf16x4_to_f32(cos_emb + d);
      float32x4_t s = bf16x4_to_f32(sin_emb + d);
      float32x4_t r0 = vsubq_f32(vmulq_f32(k0, c), vmulq_f32(k1, s));
      float32x4_t r1 = vaddq_f32(vmulq_f32(k0, s), vmulq_f32(k1, c));
      f32_to_bf16x4(kh + d, r0);
      f32_to_bf16x4(kh + half + d, r1);
    }
  }
}

/* ── Flash Attention Decode (M=1, online softmax) ──────── */

static void flash_attn_decode(
    const uint16_t *Q, const uint16_t *K_cache, const uint16_t *V_cache,
    uint16_t *Out, int64_t seq_len, int64_t head_dim, float sm_scale,
    int64_t n_heads, int64_t n_kv_heads,
    int64_t max_seq_len) {

  int64_t gqa = n_heads / n_kv_heads;

  #pragma omp parallel for schedule(static)
  for (int64_t h = 0; h < n_heads; h++) {
    int64_t kvh = h / gqa;
    const uint16_t *q = Q + h * head_dim;
    const uint16_t *kb = K_cache + kvh * max_seq_len * head_dim;
    const uint16_t *vb = V_cache + kvh * max_seq_len * head_dim;
    uint16_t *out = Out + h * head_dim;

    float q_fp32[256];
    for (int64_t d = 0; d + 4 <= head_dim; d += 4)
      vst1q_f32(q_fp32 + d, vmulq_n_f32(bf16x4_to_f32(q + d), sm_scale));

    float running_max = -1e30f, running_sum = 0.0f;
    float acc[256];
    std::memset(acc, 0, head_dim * sizeof(float));

    for (int64_t s = 0; s < seq_len; s++) {
      const uint16_t *kr = kb + s * head_dim;
      const uint16_t *vr = vb + s * head_dim;

      float32x4_t dot = vdupq_n_f32(0.0f);
      for (int64_t d = 0; d + 4 <= head_dim; d += 4)
        dot = vfmaq_f32(dot, vld1q_f32(q_fp32 + d), bf16x4_to_f32(kr + d));
      float qk = vaddvq_f32(dot);

      float old_max = running_max;
      if (qk > running_max) running_max = qk;
      float exp_old = expf(old_max - running_max);
      float exp_new = expf(qk - running_max);
      running_sum = running_sum * exp_old + exp_new;

      float32x4_t so = vdupq_n_f32(exp_old), sn = vdupq_n_f32(exp_new);
      for (int64_t d = 0; d + 4 <= head_dim; d += 4) {
        float32x4_t a = vld1q_f32(acc + d);
        a = vfmaq_f32(vmulq_f32(a, so), bf16x4_to_f32(vr + d), sn);
        vst1q_f32(acc + d, a);
      }
    }

    float inv = 1.0f / running_sum;
    for (int64_t d = 0; d + 4 <= head_dim; d += 4)
      f32_to_bf16x4(out + d, vmulq_n_f32(vld1q_f32(acc + d), inv));
  }
}

/* ── Fast exp + SWIGLU ─────────────────────────────────── */

static inline float32x4_t fast_exp(float32x4_t x) {
  x = vmaxq_f32(x, vdupq_n_f32(-88.0f));
  x = vminq_f32(x, vdupq_n_f32(88.0f));
  float32x4_t t = vmulq_n_f32(x, 1.4426950408889634f);
  float32x4_t shift = vdupq_n_f32(12582912.0f);
  float32x4_t ti = vsubq_f32(vaddq_f32(t, shift), shift);
  float32x4_t tf = vsubq_f32(t, ti);
  float32x4_t p = vaddq_f32(vmulq_f32(tf, vaddq_f32(
      vmulq_f32(tf, vdupq_n_f32(0.2402265f)), vdupq_n_f32(0.6931472f))),
      vdupq_n_f32(1.0f));
  int32x4_t ii = vshlq_n_s32(vcvtq_s32_f32(ti), 23);
  return vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(p), ii));
}

/* ── Fused MLP: gate+up GEMV + SWIGLU + down GEMV ─────── */

static void fused_mlp_full(
    const uint16_t *x_bf16, int64_t hidden,
    const int8_t *gate_packed, const int8_t *up_packed, const int8_t *down_packed,
    const float *gate_scale, const float *up_scale, const float *down_scale,
    uint16_t *out_bf16, int64_t intermediate,
    int8_t *scratch_int8, uint16_t *scratch_bf16) {

  int64_t K4 = hidden / 4, N4 = intermediate / 4;

  /* Quantize x */
  float xs = quantize_bf16_to_int8(x_bf16, scratch_int8, hidden);

  /* Gate + Up GEMV + SWIGLU (OMP parallel) */
  #pragma omp parallel
  {
    int nt = omp_get_num_threads();
    int tid = omp_get_thread_num();
    int64_t chunk = (N4 + nt - 1) / nt;
    int64_t start = tid * chunk, count = chunk;
    if (start + count > N4) count = N4 - start;
    if (start >= N4) count = 0;

    if (count > 0) {
      std::vector<int32x4_t> gate_acc(count), up_acc(count);
      for (int64_t i = 0; i < count; i++) {
        gate_acc[i] = vdupq_n_s32(0);
        up_acc[i] = vdupq_n_s32(0);
      }

      for (int64_t kb = 0; kb < K4; kb++) {
        int32_t a4;
        std::memcpy(&a4, scratch_int8 + kb * 4, 4);
        int8x16_t av = vreinterpretq_s8_s32(vdupq_n_s32(a4));
        const int8_t *gp = gate_packed + (kb * N4 + start) * 16;
        const int8_t *up = up_packed + (kb * N4 + start) * 16;
        for (int64_t i = 0; i < count; i++) {
          gate_acc[i] = vdotq_s32(gate_acc[i], av, vld1q_s8(gp + i * 16));
          up_acc[i]   = vdotq_s32(up_acc[i],   av, vld1q_s8(up + i * 16));
        }
      }

      /* Dequant + SWIGLU → scratch_bf16 */
      for (int64_t i = 0; i < count; i++) {
        int64_t n = start + i;
        float32x4_t gf = vmulq_f32(vmulq_n_f32(vcvtq_f32_s32(gate_acc[i]), xs),
                                     vld1q_f32(gate_scale + n * 4));
        float32x4_t uf = vmulq_f32(vmulq_n_f32(vcvtq_f32_s32(up_acc[i]), xs),
                                     vld1q_f32(up_scale + n * 4));
        float32x4_t exp_neg = fast_exp(vnegq_f32(gf));
        float32x4_t denom = vaddq_f32(vdupq_n_f32(1.0f), exp_neg);
        float32x4_t recip = vmulq_f32(vrecpeq_f32(denom), vrecpsq_f32(denom, vrecpeq_f32(denom)));
        float32x4_t result = vmulq_f32(vmulq_f32(gf, recip), uf);
        f32_to_bf16x4(scratch_bf16 + n * 4, result);
      }
    }
  }

  /* Down GEMV: scratch_bf16[intermediate] → out_bf16[hidden] */
  int64_t DK4 = intermediate / 4, DN4 = hidden / 4;
  float ds = quantize_bf16_to_int8(scratch_bf16, scratch_int8, intermediate);

  std::vector<int32_t> down_acc_vec(hidden);
  int32_t *down_acc = down_acc_vec.data();
  #pragma omp parallel
  {
    int nt = omp_get_num_threads();
    int tid = omp_get_thread_num();
    int64_t chunk = (DN4 + nt - 1) / nt;
    int64_t s = tid * chunk, c = chunk;
    if (s + c > DN4) c = DN4 - s;
    if (s < DN4 && c > 0)
      sdot_gemv_st(scratch_int8, down_packed, down_acc, DK4, DN4, s, c);
  }
  dequant_to_bf16(down_acc, down_scale, ds, out_bf16, hidden);
}

/* ── Residual add (inline) ─────────────────────────────── */

static void residual_add_bf16(uint16_t *residual, const uint16_t *x, int64_t D) {
  for (int64_t d = 0; d + 4 <= D; d += 4) {
    float32x4_t r = bf16x4_to_f32(residual + d);
    float32x4_t v = bf16x4_to_f32(x + d);
    f32_to_bf16x4(residual + d, vaddq_f32(r, v));
  }
}

/* ═══════════════════════════════════════════════════════════
 * Main entry: fused_transformer_decode_layer
 * ═══════════════════════════════════════════════════════════ */

extern "C" {

EXPORT void fused_transformer_decode_layer(
    /* I/O */
    uint16_t *hidden_states,    /* [hidden] bf16, in-place updated */
    /* Attention weights (INT8 SDOT packed) */
    const int8_t *wq, const int8_t *wk, const int8_t *wv, const int8_t *wo,
    const float *wq_s, const float *wk_s, const float *wv_s, const float *wo_s,
    /* QK norm weights */
    const uint16_t *q_norm_w, const uint16_t *k_norm_w,
    /* RoPE embeddings for current position */
    const uint16_t *cos_emb, const uint16_t *sin_emb,
    /* KV cache (pre-allocated, contiguous per head) */
    uint16_t *k_cache, uint16_t *v_cache, /* [n_kv_heads, max_seq, head_dim] */
    int64_t cache_pos, int64_t max_seq_len,
    /* MLP weights (INT8 SDOT packed) */
    const int8_t *gate_w, const int8_t *up_w, const int8_t *down_w,
    const float *gate_s, const float *up_s, const float *down_s,
    /* LayerNorm weights */
    const uint16_t *input_norm_w, const uint16_t *post_norm_w,
    /* Dimensions */
    int64_t hidden, int64_t head_dim, int64_t n_heads, int64_t n_kv_heads,
    int64_t intermediate, float rms_eps
) {
  int64_t seq_len = cache_pos + 1;

  /* Scratch buffers — heap allocated for large models (4B+: intermediate=9728) */
  int64_t q_dim = n_heads * head_dim;
  int64_t kv_dim = n_kv_heads * head_dim;
  int64_t max_dim = std::max({hidden, q_dim, intermediate});

  std::vector<uint16_t> norm_out_v(max_dim);
  std::vector<uint16_t> q_buf_v(q_dim);
  std::vector<uint16_t> k_buf_v(kv_dim);
  std::vector<uint16_t> v_buf_v(kv_dim);
  std::vector<uint16_t> attn_out_v(q_dim);
  std::vector<uint16_t> o_out_v(hidden);
  std::vector<uint16_t> mlp_out_v(hidden);
  std::vector<uint16_t> mlp_scratch_v(intermediate);
  std::vector<int8_t> int8_scratch_v(max_dim);
  std::vector<uint16_t> residual_v(hidden);

  uint16_t *norm_out = norm_out_v.data();
  uint16_t *q_buf = q_buf_v.data();
  uint16_t *k_buf = k_buf_v.data();
  uint16_t *v_buf = v_buf_v.data();
  uint16_t *attn_out = attn_out_v.data();
  uint16_t *o_out = o_out_v.data();
  uint16_t *mlp_out = mlp_out_v.data();
  uint16_t *mlp_scratch = mlp_scratch_v.data();
  int8_t *int8_scratch = int8_scratch_v.data();
  uint16_t *residual = residual_v.data();

  /* Save residual */
  std::memcpy(residual, hidden_states, hidden * 2);

  /* 1. Input RMSNorm */
  rms_norm_bf16(hidden_states, input_norm_w, norm_out, hidden, rms_eps);

  /* 2. Q/K/V GEMV — fused: quantize once, single OMP region for all 3 */
  {
    float xs = quantize_bf16_to_int8(norm_out, int8_scratch, hidden);
    int64_t K4 = hidden / 4;
    int64_t q_N4 = q_dim / 4, kv_N4 = kv_dim / 4;
    std::vector<int32_t> q_acc(q_dim), k_acc(kv_dim), v_acc(kv_dim);

    #pragma omp parallel
    {
      int nt = omp_get_num_threads();
      int tid = omp_get_thread_num();
      /* Q proj */
      { int64_t ch = (q_N4+nt-1)/nt, s = tid*ch, c = std::min(ch, q_N4-s);
        if (s < q_N4 && c > 0)
          sdot_gemv_st(int8_scratch, wq, q_acc.data(), K4, q_N4, s, c); }
      /* K proj */
      { int64_t ch = (kv_N4+nt-1)/nt, s = tid*ch, c = std::min(ch, kv_N4-s);
        if (s < kv_N4 && c > 0)
          sdot_gemv_st(int8_scratch, wk, k_acc.data(), K4, kv_N4, s, c); }
      /* V proj */
      { int64_t ch = (kv_N4+nt-1)/nt, s = tid*ch, c = std::min(ch, kv_N4-s);
        if (s < kv_N4 && c > 0)
          sdot_gemv_st(int8_scratch, wv, v_acc.data(), K4, kv_N4, s, c); }
    }
    dequant_to_bf16(q_acc.data(), wq_s, xs, q_buf, q_dim);
    dequant_to_bf16(k_acc.data(), wk_s, xs, k_buf, kv_dim);
    dequant_to_bf16(v_acc.data(), wv_s, xs, v_buf, kv_dim);
  }

  /* 3. QK Norm */
  for (int64_t h = 0; h < n_heads; h++)
    rms_norm_bf16(q_buf + h * head_dim, q_norm_w, q_buf + h * head_dim, head_dim, rms_eps);
  for (int64_t h = 0; h < n_kv_heads; h++)
    rms_norm_bf16(k_buf + h * head_dim, k_norm_w, k_buf + h * head_dim, head_dim, rms_eps);

  /* 4. RoPE */
  apply_rope_bf16(q_buf, k_buf, cos_emb, sin_emb, n_heads, n_kv_heads, head_dim);

  /* 5. KV cache write (in-place, no cat) */
  for (int64_t h = 0; h < n_kv_heads; h++) {
    std::memcpy(k_cache + h * max_seq_len * head_dim + cache_pos * head_dim,
                k_buf + h * head_dim, head_dim * 2);
    std::memcpy(v_cache + h * max_seq_len * head_dim + cache_pos * head_dim,
                v_buf + h * head_dim, head_dim * 2);
  }

  /* 6. Attention (online softmax) */
  float sm_scale = 1.0f / sqrtf((float)head_dim);
  flash_attn_decode(q_buf, k_cache, v_cache, attn_out,
                     seq_len, head_dim, sm_scale, n_heads, n_kv_heads,
                     max_seq_len);

  /* 7. O projection */
  float xs_o;
  gemv_int8_bf16(attn_out, q_dim, wo, wo_s, o_out, hidden, int8_scratch, &xs_o);

  /* 8. Residual add */
  residual_add_bf16(residual, o_out, hidden);

  /* 9. Post-attention RMSNorm */
  rms_norm_bf16(residual, post_norm_w, norm_out, hidden, rms_eps);

  /* 10. Fused MLP: gate+up GEMV + SWIGLU + down GEMV */
  fused_mlp_full(norm_out, hidden,
                  gate_w, up_w, down_w,
                  gate_s, up_s, down_s,
                  mlp_out, intermediate,
                  int8_scratch, mlp_scratch);

  /* 11. Final residual add → output */
  residual_add_bf16(residual, mlp_out, hidden);
  std::memcpy(hidden_states, residual, hidden * 2);
}

/* ═══════════════════════════════════════════════════════════
 * Standalone ops: RMSNorm, RoPE, residual add, KV cache write
 * Exposed for Phase 1+ op-level optimization (no layer fusion).
 * ═══════════════════════════════════════════════════════════ */

EXPORT void standalone_rms_norm_bf16(
    const uint16_t *x, const uint16_t *weight,
    uint16_t *out, int64_t D, float eps) {
  rms_norm_bf16(x, weight, out, D, eps);
}

EXPORT void standalone_rope_bf16(
    uint16_t *q, uint16_t *k,
    const uint16_t *cos_emb, const uint16_t *sin_emb,
    int64_t n_heads, int64_t n_kv_heads, int64_t head_dim) {
  apply_rope_bf16(q, k, cos_emb, sin_emb, n_heads, n_kv_heads, head_dim);
}

EXPORT void standalone_residual_add_bf16(
    uint16_t *residual, const uint16_t *x, int64_t D) {
  residual_add_bf16(residual, x, D);
}

EXPORT void standalone_kv_cache_write_bf16(
    uint16_t *k_cache, uint16_t *v_cache,
    const uint16_t *k_buf, const uint16_t *v_buf,
    int64_t n_kv_heads, int64_t max_seq_len, int64_t head_dim,
    int64_t cache_pos) {
  for (int64_t h = 0; h < n_kv_heads; h++) {
    std::memcpy(k_cache + h * max_seq_len * head_dim + cache_pos * head_dim,
                k_buf + h * head_dim, head_dim * 2);
    std::memcpy(v_cache + h * max_seq_len * head_dim + cache_pos * head_dim,
                v_buf + h * head_dim, head_dim * 2);
  }
}

} // extern "C"

#else
extern "C" {
EXPORT void fused_transformer_decode_layer(
    uint16_t *, const int8_t *, const int8_t *, const int8_t *, const int8_t *,
    const float *, const float *, const float *, const float *,
    const uint16_t *, const uint16_t *,
    const uint16_t *, const uint16_t *,
    uint16_t *, uint16_t *, int64_t, int64_t,
    const int8_t *, const int8_t *, const int8_t *,
    const float *, const float *, const float *,
    const uint16_t *, const uint16_t *,
    int64_t, int64_t, int64_t, int64_t, int64_t, float) {}
EXPORT void standalone_rms_norm_bf16(
    const uint16_t *, const uint16_t *, uint16_t *, int64_t, float) {}
EXPORT void standalone_rope_bf16(
    uint16_t *, uint16_t *, const uint16_t *, const uint16_t *,
    int64_t, int64_t, int64_t) {}
EXPORT void standalone_residual_add_bf16(uint16_t *, const uint16_t *, int64_t) {}
EXPORT void standalone_kv_cache_write_bf16(
    uint16_t *, uint16_t *, const uint16_t *, const uint16_t *,
    int64_t, int64_t, int64_t, int64_t) {}
}
#endif
