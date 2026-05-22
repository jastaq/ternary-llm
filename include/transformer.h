#pragma once
#include "model.h"
#include "kernels.cuh"
#include <cuda_fp16.h>
#include <cstdint>

// ---------------------------------------------------------------------------
// Transformer — runs the full decoder forward pass on the GPU.
//
// Owns the KV cache and all scratch buffers.  Thread-safety: NOT thread-safe.
// Call reset() between independent sequences.
// ---------------------------------------------------------------------------
class Transformer {
public:
    explicit Transformer(const Model& model);
    ~Transformer();

    // Run one forward step:
    //   token_ids_gpu  — device pointer to int[n_tokens]
    //   n_tokens       — number of tokens (>1 for prefill, ==1 for decode)
    //   start_pos      — KV-cache position for the first token
    // Returns device pointer to half[vocab_size] logits (owned by this object).
    half* forward(const int* token_ids_gpu, int n_tokens, int start_pos);

    // Clear the KV cache.
    void reset();

private:
    const Model& model_;
    int          max_tokens_;

    // KV cache  [n_layers, max_seq_len, n_kv_heads, head_dim]
    half* k_cache_   = nullptr;
    half* v_cache_   = nullptr;

    // Scratch buffers (FP16)
    half* hidden_buf_   = nullptr;
    half* hidden_buf2_  = nullptr;
    half* residual_buf_ = nullptr;
    half* q_buf_        = nullptr;
    half* k_buf_        = nullptr;
    half* v_buf_        = nullptr;
    half* attn_out_buf_ = nullptr;
    half* gate_buf_     = nullptr;
    half* up_buf_       = nullptr;
    half* ffn_out_buf_  = nullptr;
    half* logits_buf_   = nullptr;

    // INT8 quantisation scratch (for DP4A fast path during decode)
    int8_t* int8_buf_   = nullptr;   // [max(hidden, intermediate, q_dim)]
    float*  int8_scale_ = nullptr;   // [1] scalar on GPU

    // Helpers
    void ternary_linear(const TernaryWeight& w, const half* input,
                        half* output, int n_tokens);
    void ternary_linear_int8(const TernaryWeight& w, const half* input,
                             half* output, int dim);
    void fp16_linear(const half* weight, const half* input,
                     half* output, int n_tokens, int out_dim, int in_dim);
};
