#include "transformer.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
namespace {

half* gpu_alloc_half(size_t n) {
    half* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, n * sizeof(half));
    if (err != cudaSuccess) {
        fprintf(stderr, "[Transformer] cudaMalloc(%zu) failed: %s\n",
                n * sizeof(half), cudaGetErrorString(err));
        std::exit(1);
    }
    cudaMemset(ptr, 0, n * sizeof(half));
    return ptr;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
Transformer::Transformer(const Model& model) : model_(model) {
    const ModelConfig& c = model_.config;
    max_tokens_ = c.max_seq_len;

    size_t kv_cache_elems = static_cast<size_t>(c.n_layers) *
                            c.max_seq_len * c.n_kv_heads * c.head_dim;
    k_cache_ = gpu_alloc_half(kv_cache_elems);
    v_cache_ = gpu_alloc_half(kv_cache_elems);

    size_t mt = max_tokens_;
    hidden_buf_   = gpu_alloc_half(mt * c.hidden_dim);
    hidden_buf2_  = gpu_alloc_half(mt * c.hidden_dim);
    residual_buf_ = gpu_alloc_half(mt * c.hidden_dim);
    q_buf_        = gpu_alloc_half(mt * c.q_dim());
    k_buf_        = gpu_alloc_half(mt * c.kv_dim());
    v_buf_        = gpu_alloc_half(mt * c.kv_dim());
    attn_out_buf_ = gpu_alloc_half(mt * c.q_dim());
    gate_buf_     = gpu_alloc_half(mt * c.intermediate_dim);
    up_buf_       = gpu_alloc_half(mt * c.intermediate_dim);
    ffn_out_buf_  = gpu_alloc_half(mt * c.hidden_dim);
    logits_buf_   = gpu_alloc_half(c.vocab_size);

    // INT8 scratch — large enough for the biggest activation vector in decode
    size_t max_dim = std::max({
        static_cast<size_t>(c.hidden_dim),
        static_cast<size_t>(c.intermediate_dim),
        static_cast<size_t>(c.q_dim()),
        static_cast<size_t>(c.kv_dim())
    });
    // Pad to multiple of 4 for DP4A alignment
    max_dim = ((max_dim + 3) / 4) * 4;
    cudaMalloc(&int8_buf_, max_dim * sizeof(int8_t));
    cudaMemset(int8_buf_, 0, max_dim * sizeof(int8_t));
    cudaMalloc(&int8_scale_, sizeof(float));

    fprintf(stderr, "[Transformer] KV cache: %.1f MB  Scratch: %.1f MB  INT8 buf: %zu B\n",
            2 * kv_cache_elems * sizeof(half) / 1e6,
            (2 * mt * c.hidden_dim + mt * c.hidden_dim +
             mt * c.q_dim() + 2 * mt * c.kv_dim() +
             mt * c.q_dim() + 2 * mt * c.intermediate_dim +
             mt * c.hidden_dim + c.vocab_size) * sizeof(half) / 1e6,
            max_dim);
}

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------
Transformer::~Transformer() {
    cudaFree(k_cache_);    cudaFree(v_cache_);
    cudaFree(hidden_buf_); cudaFree(hidden_buf2_);
    cudaFree(residual_buf_);
    cudaFree(q_buf_);      cudaFree(k_buf_);      cudaFree(v_buf_);
    cudaFree(attn_out_buf_);
    cudaFree(gate_buf_);   cudaFree(up_buf_);
    cudaFree(ffn_out_buf_);
    cudaFree(logits_buf_);
    cudaFree(int8_buf_);   cudaFree(int8_scale_);
}

// ---------------------------------------------------------------------------
// reset
// ---------------------------------------------------------------------------
void Transformer::reset() {
    const ModelConfig& c = model_.config;
    size_t bytes = static_cast<size_t>(c.n_layers) *
                   c.max_seq_len * c.n_kv_heads * c.head_dim * sizeof(half);
    cudaMemset(k_cache_, 0, bytes);
    cudaMemset(v_cache_, 0, bytes);
}

// ---------------------------------------------------------------------------
// ternary_linear — legacy FP16 path (used during prefill, n_tokens > 1)
// ---------------------------------------------------------------------------
void Transformer::ternary_linear(const TernaryWeight& w,
                                 const half* input, half* output,
                                 int n_tokens) {
    int gs = static_cast<int>(model_.config.group_size);
    if (n_tokens == 1) {
        ternary::ternary_gemv(
            w.nonzero_masks, w.sign_masks, w.scales,
            input, output,
            w.out_features, w.in_features, gs);
    } else {
        ternary::ternary_gemm(
            input,
            w.nonzero_masks, w.sign_masks, w.scales,
            output,
            n_tokens, w.out_features, w.in_features, gs);
    }
}

// ---------------------------------------------------------------------------
// ternary_linear_int8 — fast DP4A path (decode only, n_tokens == 1)
//
// 1. Quantise FP16 input → INT8 (absmax)
// 2. Call ternary_gemv_int8 with DP4A accumulation
// ---------------------------------------------------------------------------
void Transformer::ternary_linear_int8(const TernaryWeight& w,
                                       const half* input, half* output,
                                       int dim) {
    int gs = static_cast<int>(model_.config.group_size);

    // Step 1: quantise input to INT8
    ternary::quantize_absmax_int8(input, int8_buf_, int8_scale_, dim);

    // Step 2: copy scale to host (single float)
    float h_scale = 0.0f;
    cudaMemcpy(&h_scale, int8_scale_, sizeof(float), cudaMemcpyDeviceToHost);

    // Step 3: DP4A GEMV
    ternary::ternary_gemv_int8(
        w.nonzero_masks, w.sign_masks, w.scales,
        int8_buf_, h_scale, output,
        w.out_features, w.in_features, gs);
}

// ---------------------------------------------------------------------------
// fp16_linear — for lm_head when not ternarised
// ---------------------------------------------------------------------------
void Transformer::fp16_linear(const half* weight, const half* input,
                              half* output, int n_tokens,
                              int out_dim, int in_dim) {
    if (n_tokens == 1) {
        ternary::fp16_gemv(weight, input, output, out_dim, in_dim);
    } else {
        ternary::fp16_gemm(input, weight, output, n_tokens, out_dim, in_dim);
    }
}

// ---------------------------------------------------------------------------
// forward — full decoder forward pass
//
// Uses INT8 fast path for decode (n_tokens == 1) and FP16 path for prefill.
// ---------------------------------------------------------------------------
half* Transformer::forward(const int* token_ids_gpu, int n_tokens,
                           int start_pos) {
    const ModelConfig& c = model_.config;
    int hidden = static_cast<int>(c.hidden_dim);
    int q_dim  = static_cast<int>(c.q_dim());
    int kv_dim = static_cast<int>(c.kv_dim());
    int inter  = static_cast<int>(c.intermediate_dim);
    int n_heads    = static_cast<int>(c.n_heads);
    int n_kv_heads = static_cast<int>(c.n_kv_heads);
    int head_dim   = static_cast<int>(c.head_dim);
    int max_seq    = static_cast<int>(c.max_seq_len);

    // Decode vs prefill — pick fast INT8 path for single-token decode
    const bool use_int8 = (n_tokens == 1);

    // ---- Embedding lookup ----
    ternary::embedding_lookup(
        model_.embedding_table, token_ids_gpu,
        residual_buf_, n_tokens, hidden);

    // ---- Decoder layers ----
    for (uint32_t layer_idx = 0; layer_idx < c.n_layers; ++layer_idx) {
        const TransformerLayer& layer = model_.layers[layer_idx];

        size_t kv_layer_off = static_cast<size_t>(layer_idx) *
                              max_seq * n_kv_heads * head_dim;
        half* k_cache_layer = k_cache_ + kv_layer_off;
        half* v_cache_layer = v_cache_ + kv_layer_off;

        // 1. Pre-attention RMSNorm
        ternary::rmsnorm_forward(
            residual_buf_, layer.attn_norm_weight, hidden_buf_,
            nullptr, nullptr,
            n_tokens, hidden, c.rms_norm_eps);

        // 2. QKV projections
        if (use_int8) {
            // Quantise hidden_buf_ once, reuse for Q/K/V
            ternary::quantize_absmax_int8(hidden_buf_, int8_buf_, int8_scale_, hidden);
            float h_scale;
            cudaMemcpy(&h_scale, int8_scale_, sizeof(float), cudaMemcpyDeviceToHost);
            int gs = static_cast<int>(c.group_size);

            ternary::ternary_gemv_int8(
                layer.q_proj.nonzero_masks, layer.q_proj.sign_masks, layer.q_proj.scales,
                int8_buf_, h_scale, q_buf_,
                layer.q_proj.out_features, layer.q_proj.in_features, gs);
            ternary::ternary_gemv_int8(
                layer.k_proj.nonzero_masks, layer.k_proj.sign_masks, layer.k_proj.scales,
                int8_buf_, h_scale, k_buf_,
                layer.k_proj.out_features, layer.k_proj.in_features, gs);
            ternary::ternary_gemv_int8(
                layer.v_proj.nonzero_masks, layer.v_proj.sign_masks, layer.v_proj.scales,
                int8_buf_, h_scale, v_buf_,
                layer.v_proj.out_features, layer.v_proj.in_features, gs);
        } else {
            ternary_linear(layer.q_proj, hidden_buf_, q_buf_, n_tokens);
            ternary_linear(layer.k_proj, hidden_buf_, k_buf_, n_tokens);
            ternary_linear(layer.v_proj, hidden_buf_, v_buf_, n_tokens);
        }

        // 3. RoPE
        ternary::rope_forward(
            q_buf_, k_buf_, 1, n_tokens,
            n_heads, n_kv_heads, head_dim,
            start_pos, c.rope_theta);

        // 4. Copy K, V into cache
        ternary::copy_to_kv_cache(
            k_buf_, v_buf_,
            k_cache_layer, v_cache_layer,
            1, n_kv_heads, head_dim,
            n_tokens, start_pos, max_seq);

        // 5. Attention
        int kv_len = start_pos + n_tokens;
        ternary::attention_forward(
            q_buf_, k_cache_layer, v_cache_layer,
            attn_out_buf_,
            1, n_heads, n_kv_heads, head_dim,
            n_tokens, kv_len);

        // 6. Output projection
        if (use_int8) {
            ternary_linear_int8(layer.o_proj, attn_out_buf_, hidden_buf2_, q_dim);
        } else {
            ternary_linear(layer.o_proj, attn_out_buf_, hidden_buf2_, n_tokens);
        }

        // 7. Residual
        ternary::add_forward(
            residual_buf_, hidden_buf2_, residual_buf_,
            n_tokens * hidden);

        // 8. Pre-FFN RMSNorm
        ternary::rmsnorm_forward(
            residual_buf_, layer.ffn_norm_weight, hidden_buf_,
            nullptr, nullptr,
            n_tokens, hidden, c.rms_norm_eps);

        // 9. FFN gate + up projections
        if (use_int8) {
            // Quantise hidden_buf_ once, reuse for gate + up
            ternary::quantize_absmax_int8(hidden_buf_, int8_buf_, int8_scale_, hidden);
            float h_scale;
            cudaMemcpy(&h_scale, int8_scale_, sizeof(float), cudaMemcpyDeviceToHost);
            int gs = static_cast<int>(c.group_size);

            ternary::ternary_gemv_int8(
                layer.gate_proj.nonzero_masks, layer.gate_proj.sign_masks, layer.gate_proj.scales,
                int8_buf_, h_scale, gate_buf_,
                layer.gate_proj.out_features, layer.gate_proj.in_features, gs);
            ternary::ternary_gemv_int8(
                layer.up_proj.nonzero_masks, layer.up_proj.sign_masks, layer.up_proj.scales,
                int8_buf_, h_scale, up_buf_,
                layer.up_proj.out_features, layer.up_proj.in_features, gs);
        } else {
            ternary_linear(layer.gate_proj, hidden_buf_, gate_buf_, n_tokens);
            ternary_linear(layer.up_proj,   hidden_buf_, up_buf_,   n_tokens);
        }

        // 10. SwiGLU
        ternary::silu_mul_forward(gate_buf_, up_buf_, gate_buf_,
                                  n_tokens * inter);

        // 11. Down projection
        if (use_int8) {
            ternary_linear_int8(layer.down_proj, gate_buf_, ffn_out_buf_, inter);
        } else {
            ternary_linear(layer.down_proj, gate_buf_, ffn_out_buf_, n_tokens);
        }

        // 12. Residual
        ternary::add_forward(
            residual_buf_, ffn_out_buf_, residual_buf_,
            n_tokens * hidden);
    }

    // ---- Final RMSNorm (last token only) ----
    half* last_hidden = residual_buf_ + static_cast<size_t>(n_tokens - 1) * hidden;
    ternary::rmsnorm_forward(
        last_hidden, model_.final_norm_weight, hidden_buf_,
        nullptr, nullptr, 1, hidden, c.rms_norm_eps);

    // ---- lm_head ----
    if (model_.lm_head_is_ternary) {
        // Ternary lm_head — use INT8 fast path
        ternary_linear_int8(model_.lm_head_ternary, hidden_buf_, logits_buf_, hidden);
    } else {
        // FP16 lm_head (legacy)
        fp16_linear(model_.lm_head_weight, hidden_buf_, logits_buf_,
                    1, static_cast<int>(c.vocab_size), hidden);
    }

    return logits_buf_;
}
