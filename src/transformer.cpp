#include "transformer.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

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

    // Scratch buffers — sized for full prefill (max_seq_len tokens)
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
    logits_buf_   = gpu_alloc_half(c.vocab_size);   // only last token

    fprintf(stderr, "[Transformer] KV cache: %.1f MB  Scratch: %.1f MB\n",
            2 * kv_cache_elems * sizeof(half) / 1e6,
            (2 * mt * c.hidden_dim +      // hidden_buf_ + hidden_buf2_
             mt * c.hidden_dim +           // residual_buf_
             mt * c.q_dim() +              // q_buf_
             2 * mt * c.kv_dim() +         // k_buf_ + v_buf_
             mt * c.q_dim() +              // attn_out_buf_
             2 * mt * c.intermediate_dim + // gate_buf_ + up_buf_
             mt * c.hidden_dim +           // ffn_out_buf_
             c.vocab_size                  // logits_buf_
            ) * sizeof(half) / 1e6);
}

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------
Transformer::~Transformer() {
    cudaFree(k_cache_);
    cudaFree(v_cache_);
    cudaFree(hidden_buf_);
    cudaFree(hidden_buf2_);
    cudaFree(residual_buf_);
    cudaFree(q_buf_);
    cudaFree(k_buf_);
    cudaFree(v_buf_);
    cudaFree(attn_out_buf_);
    cudaFree(gate_buf_);
    cudaFree(up_buf_);
    cudaFree(ffn_out_buf_);
    cudaFree(logits_buf_);
}

// ---------------------------------------------------------------------------
// reset — clear KV cache between sequences
// ---------------------------------------------------------------------------
void Transformer::reset() {
    const ModelConfig& c = model_.config;
    size_t bytes = static_cast<size_t>(c.n_layers) *
                   c.max_seq_len * c.n_kv_heads * c.head_dim * sizeof(half);
    cudaMemset(k_cache_, 0, bytes);
    cudaMemset(v_cache_, 0, bytes);
}

// ---------------------------------------------------------------------------
// ternary_linear — dispatch GEMV (n_tokens==1) or GEMM (n_tokens>1)
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
// fp16_linear — for lm_head (not ternarised)
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

    // ---- Embedding lookup ----
    // residual_buf_ = embedding(token_ids)   [n_tokens, hidden_dim]
    ternary::embedding_lookup(
        model_.embedding_table, token_ids_gpu,
        residual_buf_, n_tokens, hidden);

    // ---- Decoder layers ----
    for (uint32_t layer_idx = 0; layer_idx < c.n_layers; ++layer_idx) {
        const TransformerLayer& layer = model_.layers[layer_idx];

        // Offset into per-layer KV cache slice:
        //   [layer_idx][0..max_seq_len-1][n_kv_heads][head_dim]
        size_t kv_layer_off = static_cast<size_t>(layer_idx) *
                              max_seq * n_kv_heads * head_dim;
        half* k_cache_layer = k_cache_ + kv_layer_off;
        half* v_cache_layer = v_cache_ + kv_layer_off;

        // 1. Pre-attention RMSNorm:  hidden_buf_ = RMSNorm(residual_buf_)
        ternary::rmsnorm_forward(
            residual_buf_, layer.attn_norm_weight, hidden_buf_,
            /*residual=*/nullptr, /*residual_out=*/nullptr,
            n_tokens, hidden, c.rms_norm_eps);

        // 2. QKV projections
        ternary_linear(layer.q_proj, hidden_buf_, q_buf_, n_tokens);
        ternary_linear(layer.k_proj, hidden_buf_, k_buf_, n_tokens);
        ternary_linear(layer.v_proj, hidden_buf_, v_buf_, n_tokens);

        // 3. RoPE (in-place on q_buf_, k_buf_)
        ternary::rope_forward(
            q_buf_, k_buf_,
            /*batch=*/1, n_tokens,
            n_heads, n_kv_heads, head_dim,
            start_pos, c.rope_theta);

        // 4. Copy K, V into cache at start_pos
        ternary::copy_to_kv_cache(
            k_buf_, v_buf_,
            k_cache_layer, v_cache_layer,
            /*batch=*/1, n_kv_heads, head_dim,
            n_tokens, start_pos, max_seq);

        // 5. Attention:  attn_out_buf_ = MHA(q, k_cache, v_cache)
        //    kv_len = start_pos + n_tokens (total cached length)
        int kv_len = start_pos + n_tokens;
        ternary::attention_forward(
            q_buf_, k_cache_layer, v_cache_layer,
            attn_out_buf_,
            /*batch=*/1, n_heads, n_kv_heads, head_dim,
            n_tokens, kv_len);

        // 6. Output projection
        ternary_linear(layer.o_proj, attn_out_buf_, hidden_buf2_, n_tokens);

        // 7. Residual:  residual_buf_ += hidden_buf2_
        ternary::add_forward(
            residual_buf_, hidden_buf2_, residual_buf_,
            n_tokens * hidden);

        // 8. Pre-FFN RMSNorm:  hidden_buf_ = RMSNorm(residual_buf_)
        ternary::rmsnorm_forward(
            residual_buf_, layer.ffn_norm_weight, hidden_buf_,
            /*residual=*/nullptr, /*residual_out=*/nullptr,
            n_tokens, hidden, c.rms_norm_eps);

        // 9. FFN gate + up projections
        ternary_linear(layer.gate_proj, hidden_buf_, gate_buf_, n_tokens);
        ternary_linear(layer.up_proj,   hidden_buf_, up_buf_,   n_tokens);

        // 10. SwiGLU:  gate_buf_ = SiLU(gate_buf_) * up_buf_
        ternary::silu_mul_forward(gate_buf_, up_buf_, gate_buf_,
                                  n_tokens * inter);

        // 11. Down projection
        ternary_linear(layer.down_proj, gate_buf_, ffn_out_buf_, n_tokens);

        // 12. Residual:  residual_buf_ += ffn_out_buf_
        ternary::add_forward(
            residual_buf_, ffn_out_buf_, residual_buf_,
            n_tokens * hidden);
    }

    // ---- Final RMSNorm ----
    // Only need the last token for logits computation.
    // Point to the last token's hidden state in residual_buf_.
    half* last_hidden = residual_buf_ + static_cast<size_t>(n_tokens - 1) * hidden;

    // Normalise just the last token into hidden_buf_ (reuse first hidden_dim slots)
    ternary::rmsnorm_forward(
        last_hidden, model_.final_norm_weight, hidden_buf_,
        /*residual=*/nullptr, /*residual_out=*/nullptr,
        /*batch=*/1, hidden, c.rms_norm_eps);

    // ---- lm_head (FP16, NOT ternarised) ----
    fp16_linear(model_.lm_head_weight, hidden_buf_, logits_buf_,
                /*n_tokens=*/1,
                static_cast<int>(c.vocab_size), hidden);

    return logits_buf_;
}
