#pragma once
#include "model.h"
#include "kernels.cuh"

// ---------------------------------------------------------------------------
// Transformer — runs the full decoder-only forward pass on the GPU.
//
// Usage:
//   Transformer xfm(model);
//   // prefill
//   half* logits = xfm.forward(token_ids_gpu, n_prompt_tokens, 0);
//   // decode
//   logits = xfm.forward(&next_token_gpu, 1, n_prompt_tokens);
//   xfm.reset();  // clear KV cache for next sequence
// ---------------------------------------------------------------------------
class Transformer {
public:
    explicit Transformer(const Model& model);
    ~Transformer();

    // Run forward pass.
    //   token_ids_gpu : device pointer to int32 token IDs  [n_tokens]
    //   n_tokens      : number of tokens (>1 for prefill, 1 for decode)
    //   start_pos     : position of the first token in the sequence
    // Returns device pointer to logits for the *last* token  [vocab_size].
    // The pointer is owned by the Transformer and valid until the next call.
    half* forward(const int* token_ids_gpu, int n_tokens, int start_pos);

    // Reset KV cache (call between sequences).
    void reset();

private:
    const Model& model_;

    // ---- KV cache ----
    // Layout: [n_layers][max_seq_len][n_kv_heads][head_dim]
    half* k_cache_ = nullptr;
    half* v_cache_ = nullptr;

    // ---- Scratch buffers (pre-allocated once) ----
    half* hidden_buf_     = nullptr;   // [max_tokens, hidden_dim]
    half* hidden_buf2_    = nullptr;   // [max_tokens, hidden_dim]
    half* residual_buf_   = nullptr;   // [max_tokens, hidden_dim]
    half* q_buf_          = nullptr;   // [max_tokens, q_dim]
    half* k_buf_          = nullptr;   // [max_tokens, kv_dim]
    half* v_buf_          = nullptr;   // [max_tokens, kv_dim]
    half* attn_out_buf_   = nullptr;   // [max_tokens, q_dim]
    half* gate_buf_       = nullptr;   // [max_tokens, intermediate_dim]
    half* up_buf_         = nullptr;   // [max_tokens, intermediate_dim]
    half* ffn_out_buf_    = nullptr;   // [max_tokens, hidden_dim]
    half* logits_buf_     = nullptr;   // [vocab_size]  (only last token)

    int max_tokens_;  // capacity of scratch buffers (= max_seq_len)

    // Dispatch to GEMV (n_tokens==1) or GEMM (n_tokens>1).
    void ternary_linear(const TernaryWeight& w,
                        const half* input, half* output, int n_tokens);

    void fp16_linear(const half* weight, const half* input, half* output,
                     int n_tokens, int out_dim, int in_dim);
};
