/*
 * Rotary Position Embeddings (RoPE)
 *
 * Applied in-place to Q and K tensors.
 * Q layout: [batch, seq_len, n_heads, head_dim]
 * K layout: [batch, seq_len, n_kv_heads, head_dim]
 *
 * For each (batch, seq_pos, head, pair_i):
 *   pos = start_pos + seq_idx
 *   freq = 1.0 / pow(theta, 2*pair_i / head_dim)
 *   angle = pos * freq
 *   (x0, x1) = (q[2i], q[2i+1])
 *   q[2i]   = x0 * cos(angle) - x1 * sin(angle)
 *   q[2i+1] = x0 * sin(angle) + x1 * cos(angle)
 */

#include "../include/kernels.cuh"
#include <cuda_fp16.h>
#include <math.h>

namespace ternary {

__global__ void rope_kernel(
    half* __restrict__ q,
    half* __restrict__ k,
    int batch,
    int seq_len,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int start_pos,
    float theta)
{
    const int half_dim = head_dim / 2;
    const int max_heads = max(n_heads, n_kv_heads);
    const int total = batch * seq_len * max_heads * half_dim;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    // Decode indices
    int pair_i = idx % half_dim;
    int tmp = idx / half_dim;
    int head_idx = tmp % max_heads;
    tmp = tmp / max_heads;
    int seq_idx = tmp % seq_len;
    int batch_idx = tmp / seq_len;

    // Compute rotation angle
    int pos = start_pos + seq_idx;
    float freq = 1.0f / powf(theta, (2.0f * pair_i) / (float)head_dim);
    float angle = (float)pos * freq;
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    // Apply to Q (if head_idx < n_heads)
    if (head_idx < n_heads) {
        int q_offset = ((batch_idx * seq_len + seq_idx) * n_heads + head_idx) * head_dim;
        int i0 = q_offset + 2 * pair_i;
        int i1 = q_offset + 2 * pair_i + 1;

        float q0 = __half2float(q[i0]);
        float q1 = __half2float(q[i1]);

        q[i0] = __float2half(q0 * cos_val - q1 * sin_val);
        q[i1] = __float2half(q0 * sin_val + q1 * cos_val);
    }

    // Apply to K (if head_idx < n_kv_heads)
    if (head_idx < n_kv_heads) {
        int k_offset = ((batch_idx * seq_len + seq_idx) * n_kv_heads + head_idx) * head_dim;
        int i0 = k_offset + 2 * pair_i;
        int i1 = k_offset + 2 * pair_i + 1;

        float k0 = __half2float(k[i0]);
        float k1 = __half2float(k[i1]);

        k[i0] = __float2half(k0 * cos_val - k1 * sin_val);
        k[i1] = __float2half(k0 * sin_val + k1 * cos_val);
    }
}

void rope_forward(
    half* q,
    half* k,
    int batch,
    int seq_len,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int start_pos,
    float theta,
    cudaStream_t stream)
{
    const int half_dim = head_dim / 2;
    const int max_heads = (n_heads > n_kv_heads) ? n_heads : n_kv_heads;
    const int total = batch * seq_len * max_heads * half_dim;
    const int block_size = 256;
    const int grid_size = (total + block_size - 1) / block_size;

    rope_kernel<<<grid_size, block_size, 0, stream>>>(
        q, k, batch, seq_len, n_heads, n_kv_heads, head_dim,
        start_pos, theta);
}

} // namespace ternary
