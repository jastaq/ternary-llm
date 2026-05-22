#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace ternary {

// =========================================================================
// Ternary GEMV — original FP16 input version (fallback / legacy)
// =========================================================================
void ternary_gemv(
    const uint32_t* nonzero_masks, const uint32_t* sign_masks,
    const half* scales, const half* input, half* output,
    int rows, int cols, int group_size, cudaStream_t stream = 0);

// =========================================================================
// Ternary GEMV — optimised INT8 activations + DP4A (primary fast path)
// =========================================================================
void ternary_gemv_int8(
    const uint32_t* nonzero_masks, const uint32_t* sign_masks,
    const half* weight_scales,
    const int8_t* input_int8, float input_scale,
    half* output,
    int rows, int cols, int group_size, cudaStream_t stream = 0);

// =========================================================================
// INT8 absmax quantisation: half[n] → int8[n] + float scale
// scale = max(|x|) / 127,  x_int8 = clamp(round(x / scale), -127, 127)
// =========================================================================
void quantize_absmax_int8(
    const half* input, int8_t* output, float* scale,
    int n, cudaStream_t stream = 0);

// =========================================================================
// Ternary GEMM — prefill (FP16 activations, kept for now)
// =========================================================================
void ternary_gemm(
    const half* A, const uint32_t* nz_B, const uint32_t* sign_B,
    const half* scales_B, half* C, int M, int N, int K, int group_size,
    cudaStream_t stream = 0);

// =========================================================================
// Standard ops
// =========================================================================
void rmsnorm_forward(
    const half* input, const half* weight, half* output,
    const half* residual, half* residual_out,
    int batch, int dim, float eps, cudaStream_t stream = 0);

void rope_forward(
    half* q, half* k, int batch, int seq_len,
    int n_heads, int n_kv_heads, int head_dim,
    int start_pos, float theta, cudaStream_t stream = 0);

void attention_forward(
    const half* q, const half* k_cache, const half* v_cache,
    half* output, int batch, int n_heads, int n_kv_heads, int head_dim,
    int seq_len, int kv_len, cudaStream_t stream = 0);

void silu_mul_forward(
    const half* gate, const half* up, half* output, int size,
    cudaStream_t stream = 0);

void add_forward(
    const half* a, const half* b, half* output, int size,
    cudaStream_t stream = 0);

void embedding_lookup(
    const half* table, const int* indices, half* output,
    int n_tokens, int dim, cudaStream_t stream = 0);

void copy_to_kv_cache(
    const half* k, const half* v, half* k_cache, half* v_cache,
    int batch, int n_kv_heads, int head_dim,
    int seq_len, int start_pos, int max_seq_len,
    cudaStream_t stream = 0);

void fp16_gemv(
    const half* weight, const half* input, half* output,
    int rows, int cols, cudaStream_t stream = 0);

void fp16_gemm(
    const half* A, const half* B, half* C,
    int M, int N, int K, cudaStream_t stream = 0);

} // namespace ternary
