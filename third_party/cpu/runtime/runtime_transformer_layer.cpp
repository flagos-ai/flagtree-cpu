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

/* Multi-row RMSNormGated (Qwen3_5RMSNormGated) in one kernel call:
 *   per row m: out[m] = rms_norm(x[m]) * weight * silu(gate[m])
 * where rms_norm(x) = x / sqrt(mean(x²) + eps).
 *
 * Replaces a 6-op ATen sequence per row × M rows × per-token call site.
 * For Qwen3.5 GDN this is M=16 (head count) × D=128 (head_v_dim).
 */
static inline float silu_bf16_to_f32(float x) {
  return x / (1.0f + expf(-x));
}

EXPORT void standalone_rms_norm_gated_bf16(
    const uint16_t *x_in,    // [M, D] bf16
    const uint16_t *gate_in, // [M, D] bf16
    const uint16_t *weight,  // [D]    bf16
    uint16_t       *out,     // [M, D] bf16
    int64_t M, int64_t D, float eps) {
  // Threshold: M=16 D=128 (Qwen3.5-2B) is faster single-threaded because
  // OMP fork/join (~3-5μs) exceeds per-thread compute (~1μs for 2 rows).
  // M=32 D=128 (Qwen3.5-4B linear_num_value_heads=32) benefits from OMP
  // (each thread does 4 rows × ~6μs = 24μs, vs 192μs single-threaded).
  // Threshold: enable OMP when total work (M × D) ≥ 2048 bytes equivalent.
  if (M >= 24) {
    #pragma omp parallel for schedule(static)
    for (int64_t m = 0; m < M; m++) {
      const uint16_t *xr  = x_in    + m * D;
      const uint16_t *gr  = gate_in + m * D;
      uint16_t       *or_ = out     + m * D;
      float32x4_t ss = vdupq_n_f32(0.0f);
      int64_t d;
      for (d = 0; d + 4 <= D; d += 4) {
        float32x4_t v = bf16x4_to_f32(xr + d);
        ss = vfmaq_f32(ss, v, v);
      }
      float sum_sq = vaddvq_f32(ss);
      for (; d < D; d++) {
        uint32_t bits = (uint32_t)xr[d] << 16;
        float v;
        std::memcpy(&v, &bits, 4);
        sum_sq += v * v;
      }
      float rms = 1.0f / sqrtf(sum_sq / (float)D + eps);
      float32x4_t vrms = vdupq_n_f32(rms);
      for (d = 0; d + 4 <= D; d += 4) {
        float32x4_t xn = vmulq_f32(bf16x4_to_f32(xr + d), vrms);
        float32x4_t w  = bf16x4_to_f32(weight + d);
        float32x4_t g  = bf16x4_to_f32(gr + d);
        float xn_arr[4], w_arr[4], g_arr[4];
        vst1q_f32(xn_arr, xn);
        vst1q_f32(w_arr, w);
        vst1q_f32(g_arr, g);
        uint16_t bf[4];
        for (int i = 0; i < 4; i++) {
          float silu = silu_bf16_to_f32(g_arr[i]);
          float val = xn_arr[i] * w_arr[i] * silu;
          uint32_t bits;
          std::memcpy(&bits, &val, 4);
          bf[i] = (uint16_t)(bits >> 16);
        }
        vst1_u16(or_ + d, vld1_u16(bf));
      }
      for (; d < D; d++) {
        uint32_t bx = (uint32_t)xr[d] << 16;
        uint32_t bw = (uint32_t)weight[d] << 16;
        uint32_t bg = (uint32_t)gr[d] << 16;
        float vx, vw, vg;
        std::memcpy(&vx, &bx, 4);
        std::memcpy(&vw, &bw, 4);
        std::memcpy(&vg, &bg, 4);
        float val = vx * rms * vw * silu_bf16_to_f32(vg);
        uint32_t bits;
        std::memcpy(&bits, &val, 4);
        or_[d] = (uint16_t)(bits >> 16);
      }
    }
    return;
  }
  // Small-M path (single-threaded).
  for (int64_t m = 0; m < M; m++) {
    const uint16_t *xr  = x_in    + m * D;
    const uint16_t *gr  = gate_in + m * D;
    uint16_t       *or_ = out     + m * D;

    // Pass 1: sum of squares for normalization.
    float32x4_t ss = vdupq_n_f32(0.0f);
    int64_t d;
    for (d = 0; d + 4 <= D; d += 4) {
      float32x4_t v = bf16x4_to_f32(xr + d);
      ss = vfmaq_f32(ss, v, v);
    }
    float sum_sq = vaddvq_f32(ss);
    for (; d < D; d++) {
      uint32_t bits = (uint32_t)xr[d] << 16;
      float v;
      std::memcpy(&v, &bits, 4);
      sum_sq += v * v;
    }
    float rms = 1.0f / sqrtf(sum_sq / (float)D + eps);

    // Pass 2: out = (x * rms) * weight * silu(gate).
    float32x4_t vrms = vdupq_n_f32(rms);
    for (d = 0; d + 4 <= D; d += 4) {
      float32x4_t xn = vmulq_f32(bf16x4_to_f32(xr + d), vrms);
      float32x4_t w  = bf16x4_to_f32(weight + d);
      float32x4_t g  = bf16x4_to_f32(gr + d);
      // SiLU: g / (1 + exp(-g))
      // No native NEON exp; do it scalar lane-wise.
      float xn_arr[4], w_arr[4], g_arr[4];
      vst1q_f32(xn_arr, xn);
      vst1q_f32(w_arr, w);
      vst1q_f32(g_arr, g);
      uint16_t bf[4];
      for (int i = 0; i < 4; i++) {
        float silu = silu_bf16_to_f32(g_arr[i]);
        float val = xn_arr[i] * w_arr[i] * silu;
        uint32_t bits;
        std::memcpy(&bits, &val, 4);
        bf[i] = (uint16_t)(bits >> 16);
      }
      vst1_u16(or_ + d, vld1_u16(bf));
    }
    for (; d < D; d++) {
      uint32_t bx = (uint32_t)xr[d] << 16;
      uint32_t bw = (uint32_t)weight[d] << 16;
      uint32_t bg = (uint32_t)gr[d] << 16;
      float vx, vw, vg;
      std::memcpy(&vx, &bx, 4);
      std::memcpy(&vw, &bw, 4);
      std::memcpy(&vg, &bg, 4);
      float val = vx * rms * vw * silu_bf16_to_f32(vg);
      uint32_t bits;
      std::memcpy(&bits, &val, 4);
      or_[d] = (uint16_t)(bits >> 16);
    }
  }
}

/* ═══════════════════════════════════════════════════════════
 * Gated Delta Net recurrent decode (T=1) — fp32, NEON.
 *
 * Per (b, h) head:
 *   1. state *= exp(g)
 *   2. kv_mem[v]  = sum_k state[k, v] * k_in[k]
 *   3. delta[v]   = (v_in[v] - kv_mem[v]) * beta
 *   4+5. fused:  state[k, v] += k_in[k] * delta[v]
 *                out[v]      += state_post[k, v] * (q_in[k] * scale)
 *
 * State update + output dot fused into a single pass over state, matching
 * llama.cpp Metal/SYCL backends (their CPU kernel does these as 2 passes).
 * Saves ~one full state read+write per token.
 *
 * Optional in-kernel L2 norm of q & k (use_qk_l2norm_in_kernel=True path)
 * to match torch_recurrent_gated_delta_rule.
 *
 * Constraints:
 *   k_dim, v_dim multiples of 4, ≤ 256 (stack alloc cap).
 * ═══════════════════════════════════════════════════════════ */

static void gated_delta_decode_fp32(
    const float *q,        // [B, H, k_dim]
    const float *k,        // [B, H, k_dim]
    const float *v,        // [B, H, v_dim]
    const float *g,        // [B, H]   (raw, exp'd internally)
    const float *beta,     // [B, H]
    float *state,          // [B, H, k_dim, v_dim]   IN-OUT
    float *out,            // [B, H, v_dim]          OUT
    int64_t B, int64_t H,
    int64_t k_dim, int64_t v_dim,
    int use_l2norm) {
  const float scale = 1.0f / sqrtf((float)k_dim);
  const float l2_eps = 1e-6f;

  #pragma omp parallel for collapse(2) schedule(static)
  for (int64_t b = 0; b < B; b++) {
    for (int64_t h = 0; h < H; h++) {
      float *S = state + ((b * H + h) * k_dim) * v_dim;
      const float *qh_in = q + (b * H + h) * k_dim;
      const float *kh_in = k + (b * H + h) * k_dim;
      const float *vh    = v + (b * H + h) * v_dim;
      float g_exp  = expf(g[b * H + h]);
      float beta_h = beta[b * H + h];

      // Local q/k buffers (don't mutate caller tensors).
      float qh[256], kh[256];
      float kv_mem[256] = {0};
      float delta[256];
      float out_acc[256] = {0};

      std::memcpy(qh, qh_in, k_dim * sizeof(float));
      std::memcpy(kh, kh_in, k_dim * sizeof(float));

      // L2 norm along head_dim (matches torch l2norm with eps=1e-6).
      if (use_l2norm) {
        float ssq_q = 0.0f, ssq_k = 0.0f;
        for (int64_t kk = 0; kk < k_dim; kk++) {
          ssq_q += qh[kk] * qh[kk];
          ssq_k += kh[kk] * kh[kk];
        }
        float inv_q = 1.0f / sqrtf(ssq_q + l2_eps);
        float inv_k = 1.0f / sqrtf(ssq_k + l2_eps);
        for (int64_t kk = 0; kk < k_dim; kk++) {
          qh[kk] *= inv_q;
          kh[kk] *= inv_k;
        }
      }
      // Pre-scale q by 1/sqrt(k_dim) once per head, baked into the fused step.
      for (int64_t kk = 0; kk < k_dim; kk++) qh[kk] *= scale;

      // Step 1: state *= exp(g)
      const int64_t total = k_dim * v_dim;
      {
        float32x4_t vg = vdupq_n_f32(g_exp);
        int64_t i = 0;
        for (; i + 4 <= total; i += 4) {
          vst1q_f32(S + i, vmulq_f32(vld1q_f32(S + i), vg));
        }
        for (; i < total; i++) S[i] *= g_exp;
      }

      // Step 2: kv_mem[v] = sum_k state[k, v] * k_in[k]
      for (int64_t kk = 0; kk < k_dim; kk++) {
        const float k_val = kh[kk];
        const float *row = S + kk * v_dim;
        int64_t vv = 0;
        for (; vv + 4 <= v_dim; vv += 4) {
          float32x4_t s = vld1q_f32(row + vv);
          float32x4_t m = vld1q_f32(kv_mem + vv);
          vst1q_f32(kv_mem + vv, vfmaq_n_f32(m, s, k_val));
        }
        for (; vv < v_dim; vv++) kv_mem[vv] += row[vv] * k_val;
      }

      // Step 3: delta[v] = (v_in[v] - kv_mem[v]) * beta_h
      {
        float32x4_t vb = vdupq_n_f32(beta_h);
        int64_t vv = 0;
        for (; vv + 4 <= v_dim; vv += 4) {
          float32x4_t d = vsubq_f32(vld1q_f32(vh + vv), vld1q_f32(kv_mem + vv));
          vst1q_f32(delta + vv, vmulq_f32(d, vb));
        }
        for (; vv < v_dim; vv++) delta[vv] = (vh[vv] - kv_mem[vv]) * beta_h;
      }

      // Step 4+5 fused: state[k,v] += k_in[k]*delta[v];  out[v] += state_post[k,v] * q_scaled[k]
      for (int64_t kk = 0; kk < k_dim; kk++) {
        const float k_val = kh[kk];
        const float q_val = qh[kk];   // already × scale
        float *row = S + kk * v_dim;
        int64_t vv = 0;
        for (; vv + 4 <= v_dim; vv += 4) {
          float32x4_t s = vld1q_f32(row + vv);
          float32x4_t d = vld1q_f32(delta + vv);
          s = vfmaq_n_f32(s, d, k_val);              // state update
          vst1q_f32(row + vv, s);
          float32x4_t o = vld1q_f32(out_acc + vv);
          vst1q_f32(out_acc + vv, vfmaq_n_f32(o, s, q_val));  // out accumulate (post-update)
        }
        for (; vv < v_dim; vv++) {
          float s = row[vv] + k_val * delta[vv];
          row[vv] = s;
          out_acc[vv] += s * q_val;
        }
      }

      std::memcpy(out + (b * H + h) * v_dim, out_acc, v_dim * sizeof(float));
    }
  }
}

EXPORT void standalone_gated_delta_decode_fp32(
    const float *q, const float *k, const float *v,
    const float *g, const float *beta,
    float *state, float *out,
    int64_t B, int64_t H, int64_t k_dim, int64_t v_dim,
    int64_t use_l2norm) {
  gated_delta_decode_fp32(q, k, v, g, beta, state, out, B, H, k_dim, v_dim,
                          use_l2norm ? 1 : 0);
}

/* ═══════════════════════════════════════════════════════════
 * Causal depthwise conv1d update (T=1 decode), bf16, kernel_size=4.
 *
 * Qwen3.5/Qwen3-Next store kernel_size=4 elements in conv_state (NOT
 * kernel_size-1). The torch reference does:
 *   cat = [state, hidden_in]                      // [B, C, kernel_size+1]
 *   state_new = cat[..., -kernel_size:]           // = [s1, s2, s3, h]
 *   out = conv1d(cat, weight, kernel_size=4, padding=0)[:, :, -1:]
 *       = sum_k cat[..., 1+k] * weight[c, k]      // = s1*w0+s2*w1+s3*w2+h*w3
 *
 * For each (b, c):
 *   y = state[1]*w0 + state[2]*w1 + state[3]*w2 + hidden_in*w3
 *   if bias: y += bias[c]
 *   round y to bf16 with FTZ (matches mkldnn's intermediate output)
 *   if silu: y = silu_f32(y)
 *   out[b, c] = bf16(y)
 *   state[0..3] = [state[1], state[2], state[3], hidden_in]
 *
 * Replaces aten::conv1d (depthwise, mkldnn) which has high dispatch
 * overhead (~700us/call) on Qwen3.5 conv_dim=6144.
 * ═══════════════════════════════════════════════════════════ */

static inline float silu_f32(float x) {
  return x / (1.0f + expf(-x));
}

static inline float bf16_to_f32_scalar(uint16_t b) {
  uint32_t v = (uint32_t)b << 16;
  float f;
  std::memcpy(&f, &v, sizeof(f));
  return f;
}

static inline uint16_t f32_to_bf16_scalar(float f) {
  uint32_t v;
  std::memcpy(&v, &f, sizeof(v));
  // round-to-nearest-even
  uint32_t lsb = (v >> 16) & 1u;
  uint32_t rounding = 0x7FFFu + lsb;
  return (uint16_t)((v + rounding) >> 16);
}

// FTZ variant: flush bf16 subnormals to zero (preserve sign). Matches
// mkldnn's behavior on intermediate bf16 conv outputs.
static inline uint16_t f32_to_bf16_scalar_ftz(float f) {
  uint16_t r = f32_to_bf16_scalar(f);
  if ((r & 0x7F80u) == 0) r &= 0x8000u;
  return r;
}

static void causal_conv1d_update_bf16_kn4(
    const uint16_t *hidden_in,  // [B, C]
    uint16_t *conv_state,       // [B, C, 4]  IN-OUT  (state_len = kernel_size)
    const uint16_t *weight,     // [C, 4]
    const uint16_t *bias,       // [C] or NULL
    uint16_t *out,              // [B, C]
    int silu,
    int64_t B, int64_t C) {
  #pragma omp parallel for collapse(2) schedule(static)
  for (int64_t b = 0; b < B; b++) {
    for (int64_t c = 0; c < C; c++) {
      uint16_t *st = conv_state + (b * C + c) * 4;
      const uint16_t *w = weight + c * 4;
      const uint16_t h = hidden_in[b * C + c];
      // Conv reads state[1..3] + hidden_in (NOT state[0..2]+hidden_in):
      // mirrors the last position output of mkldnn's bf16 conv1d on
      // cat([state, h]) with kernel=4.
      uint16_t buf[4] = {st[1], st[2], st[3], h};
      float32x4_t v_v = bf16x4_to_f32(buf);
      float32x4_t v_w = bf16x4_to_f32(w);
      float y = vaddvq_f32(vmulq_f32(v_v, v_w));
      if (bias) y += bf16_to_f32_scalar(bias[c]);
      // F.conv1d outputs bf16 (round + FTZ), F.silu upcasts to fp32, applies
      // silu, rounds back to bf16 (no FTZ). Round-trip the post-conv value
      // through bf16-with-FTZ to align numerics with mkldnn's bf16 conv1d.
      y = bf16_to_f32_scalar(f32_to_bf16_scalar_ftz(y));
      if (silu) y = silu_f32(y);
      out[b * C + c] = f32_to_bf16_scalar(y);
      // Roll state forward by one position.
      st[0] = st[1];
      st[1] = st[2];
      st[2] = st[3];
      st[3] = h;
    }
  }
}

EXPORT void standalone_causal_conv1d_update_bf16(
    const uint16_t *hidden_in,
    uint16_t *conv_state,
    const uint16_t *weight,
    const uint16_t *bias,            // valid pointer; ignored when has_bias==0
    uint16_t *out,
    int64_t B, int64_t C, int64_t kernel_size,
    int64_t silu, int64_t has_bias) {
  if (kernel_size != 4) return;
  causal_conv1d_update_bf16_kn4(hidden_in, conv_state, weight,
                                 has_bias ? bias : nullptr, out,
                                 silu ? 1 : 0, B, C);
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
EXPORT void standalone_gated_delta_decode_fp32(
    const float *, const float *, const float *,
    const float *, const float *,
    float *, float *,
    int64_t, int64_t, int64_t, int64_t, int64_t) {}
EXPORT void standalone_causal_conv1d_update_bf16(
    const uint16_t *, uint16_t *, const uint16_t *,
    const uint16_t *, uint16_t *,
    int64_t, int64_t, int64_t, int64_t, int64_t) {}
EXPORT void standalone_rms_norm_gated_bf16(
    const uint16_t *, const uint16_t *, const uint16_t *,
    uint16_t *, int64_t, int64_t, float) {}
}
#endif
