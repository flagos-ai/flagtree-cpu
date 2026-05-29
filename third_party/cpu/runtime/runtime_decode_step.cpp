/*
 * Fused Decode Step: embedding → 28 layers → final norm → lm_head → argmax
 * One C call per token. Zero Python dispatch in the decode loop.
 *
 * All layer weights are passed via a flat pointer array (layer_ptrs).
 * Layout per layer (12 pointers + 4 pointers = 16 pointers):
 *   [wq, wk, wv, wo, wq_s, wk_s, wv_s, wo_s, q_norm, k_norm, in_norm, post_norm,
 *    gate, up, down, gate_s, up_s, down_s]
 * → 18 pointers per layer, n_layers layers.
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

/* Forward-declare the fused layer function from runtime_transformer_layer.cpp */
extern "C" void fused_transformer_decode_layer(
    uint16_t *hidden_states,
    const int8_t *wq, const int8_t *wk, const int8_t *wv, const int8_t *wo,
    const float *wq_s, const float *wk_s, const float *wv_s, const float *wo_s,
    const uint16_t *q_norm_w, const uint16_t *k_norm_w,
    const uint16_t *cos_emb, const uint16_t *sin_emb,
    uint16_t *k_cache, uint16_t *v_cache,
    int64_t cache_pos, int64_t max_seq_len,
    const int8_t *gate_w, const int8_t *up_w, const int8_t *down_w,
    const float *gate_s, const float *up_s, const float *down_s,
    const uint16_t *input_norm_w, const uint16_t *post_norm_w,
    int64_t hidden, int64_t head_dim, int64_t n_heads, int64_t n_kv_heads,
    int64_t intermediate, float rms_eps
);

#if defined(__aarch64__) && defined(__ARM_NEON)

static inline float32x4_t bf16x4_to_f32(const uint16_t *p) {
  return vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(p), 16));
}
static inline void f32_to_bf16x4(uint16_t *p, float32x4_t v) {
  vst1_u16(p, vshrn_n_u32(vreinterpretq_u32_f32(v), 16));
}

static void rms_norm_bf16_ds(const uint16_t *x, const uint16_t *weight,
                              uint16_t *out, int64_t D, float eps) {
  float32x4_t ss = vdupq_n_f32(0.0f);
  for (int64_t d = 0; d + 4 <= D; d += 4) {
    float32x4_t v = bf16x4_to_f32(x + d);
    ss = vfmaq_f32(ss, v, v);
  }
  float rms = 1.0f / sqrtf(vaddvq_f32(ss) / (float)D + eps);
  float32x4_t vrms = vdupq_n_f32(rms);
  for (int64_t d = 0; d + 4 <= D; d += 4) {
    float32x4_t v = vmulq_f32(bf16x4_to_f32(x + d), vrms);
    float32x4_t w = bf16x4_to_f32(weight + d);
    f32_to_bf16x4(out + d, vmulq_f32(v, w));
  }
}

/* Quantize BF16 → INT8 and return scale */
static float quantize_bf16_ds(const uint16_t *x, int8_t *out, int64_t K) {
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

/* SDOT GEMV range (single-thread, for use inside OMP) */
static void sdot_range_ds(const int8_t *A, const int8_t *B_packed,
                           int32_t *C, int64_t K4, int64_t N4,
                           int64_t start, int64_t count) {
  std::vector<int32x4_t> acc(count);
  for (int64_t i = 0; i < count; i++) acc[i] = vdupq_n_s32(0);
  for (int64_t kb = 0; kb < K4; kb++) {
    int32_t a4;
    std::memcpy(&a4, A + kb * 4, 4);
    int8x16_t av = vreinterpretq_s8_s32(vdupq_n_s32(a4));
    const int8_t *bp = B_packed + (kb * N4 + start) * 16;
    for (int64_t i = 0; i < count; i++)
      acc[i] = vdotq_s32(acc[i], av, vld1q_s8(bp + i * 16));
  }
  for (int64_t i = 0; i < count; i++)
    vst1q_s32(C + (start + i) * 4, acc[i]);
}

extern "C" {

/*
 * fused_decode_step: one complete decode token in C.
 *
 * Args:
 *   token_id:     input token ID
 *   pos:          current position in sequence
 *   embed_table:  [vocab_size, hidden] bf16 embedding weights
 *   layer_ptrs:   flat array of pointers, 18 per layer:
 *                 [wq, wk, wv, wo, wq_s, wk_s, wv_s, wo_s,
 *                  q_norm, k_norm, in_norm, post_norm,
 *                  gate, up, down, gate_s, up_s, down_s]
 *   k_cache:      [n_layers, n_kv, max_seq, head_dim] bf16
 *   v_cache:      [n_layers, n_kv, max_seq, head_dim] bf16
 *   rope_cos:     [max_seq, head_dim] bf16 (pre-computed)
 *   rope_sin:     [max_seq, head_dim] bf16 (pre-computed)
 *   final_norm_w: [hidden] bf16
 *   lm_head_packed: SDOT-packed lm_head weights
 *   lm_head_scale:  [vocab_size] float32
 *   hidden, head_dim, n_heads, n_kv_heads, intermediate:  dims
 *   vocab_size, n_layers, max_seq: dims
 *   rms_eps: float
 *
 * Returns: next token ID
 */
EXPORT int64_t fused_decode_step(
    int64_t token_id,
    int64_t pos,
    const uint16_t *embed_table,
    const void **layer_ptrs,
    uint16_t *k_cache,
    uint16_t *v_cache,
    const uint16_t *rope_cos,
    const uint16_t *rope_sin,
    const uint16_t *final_norm_w,
    const int8_t *lm_head_packed,
    const float *lm_head_scale,
    int64_t hidden,
    int64_t head_dim,
    int64_t n_heads,
    int64_t n_kv_heads,
    int64_t intermediate,
    int64_t vocab_size,
    int64_t n_layers,
    int64_t max_seq,
    float rms_eps
) {
  int64_t kv_dim = n_kv_heads * head_dim;
  int64_t kv_layer_stride = n_kv_heads * max_seq * head_dim;

  /* 1. Embedding lookup */
  std::vector<uint16_t> h_vec(hidden);
  uint16_t *h = h_vec.data();
  std::memcpy(h, embed_table + token_id * hidden, hidden * sizeof(uint16_t));

  /* 2. RoPE cos/sin for this position */
  const uint16_t *cos = rope_cos + pos * head_dim;
  const uint16_t *sin = rope_sin + pos * head_dim;

  /* 3. Run all layers */
  for (int64_t i = 0; i < n_layers; i++) {
    const void **lp = layer_ptrs + i * 18;
    fused_transformer_decode_layer(
        h,
        (const int8_t *)lp[0],   /* wq */
        (const int8_t *)lp[1],   /* wk */
        (const int8_t *)lp[2],   /* wv */
        (const int8_t *)lp[3],   /* wo */
        (const float *)lp[4],    /* wq_s */
        (const float *)lp[5],    /* wk_s */
        (const float *)lp[6],    /* wv_s */
        (const float *)lp[7],    /* wo_s */
        (const uint16_t *)lp[8], /* q_norm */
        (const uint16_t *)lp[9], /* k_norm */
        cos, sin,
        k_cache + i * kv_layer_stride,
        v_cache + i * kv_layer_stride,
        pos, max_seq,
        (const int8_t *)lp[12],  /* gate */
        (const int8_t *)lp[13],  /* up */
        (const int8_t *)lp[14],  /* down */
        (const float *)lp[15],   /* gate_s */
        (const float *)lp[16],   /* up_s */
        (const float *)lp[17],   /* down_s */
        (const uint16_t *)lp[10], /* in_norm */
        (const uint16_t *)lp[11], /* post_norm */
        hidden, head_dim, n_heads, n_kv_heads, intermediate, rms_eps
    );
  }

  /* 4. Final RMSNorm */
  rms_norm_bf16_ds(h, final_norm_w, h, hidden, rms_eps);

  /* 5. Quantize h → INT8 for lm_head GEMV */
  std::vector<int8_t> h_int8(hidden);
  float x_scale = quantize_bf16_ds(h, h_int8.data(), hidden);

  /* 6. lm_head SDOT GEMV (OMP parallel) */
  int64_t K4 = hidden / 4;
  int64_t N4 = vocab_size / 4;
  std::vector<int32_t> logits_i32(vocab_size);

  #pragma omp parallel
  {
    int nt = omp_get_num_threads();
    int tid = omp_get_thread_num();
    int64_t chunk = (N4 + nt - 1) / nt;
    int64_t s = tid * chunk, c = chunk;
    if (s + c > N4) c = N4 - s;
    if (s < N4 && c > 0)
      sdot_range_ds(h_int8.data(), lm_head_packed, logits_i32.data(), K4, N4, s, c);
  }

  /* 7. Dequant + argmax (fused: find max during dequant scan) */
  float best_val = -1e30f;
  int64_t best_idx = 0;
  for (int64_t n = 0; n + 4 <= vocab_size; n += 4) {
    float32x4_t vi = vcvtq_f32_s32(vld1q_s32(logits_i32.data() + n));
    float32x4_t ws = vld1q_f32(lm_head_scale + n);
    float32x4_t val = vmulq_f32(vmulq_n_f32(vi, x_scale), ws);
    /* Horizontal max of 4 elements */
    float vals[4];
    vst1q_f32(vals, val);
    for (int j = 0; j < 4; j++) {
      if (vals[j] > best_val) {
        best_val = vals[j];
        best_idx = n + j;
      }
    }
  }

  return best_idx;
}

} // extern "C"

#else
extern "C" {
EXPORT int64_t fused_decode_step(
    int64_t, int64_t, const uint16_t *, const void **,
    uint16_t *, uint16_t *, const uint16_t *, const uint16_t *,
    const uint16_t *, const int8_t *, const float *,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, float) {
  return 0;
}
}
#endif
