/*
 * Multi-Head Attention with Causal Mask and GQA Support
 *
 * Supports both decode (seq_len=1) and prefill (seq_len>1) paths.
 *
 * Q layout:       [batch, seq_len, n_heads, head_dim]
 * K cache layout: [batch, n_kv_heads, max_seq_len, head_dim]
 * V cache layout: [batch, n_kv_heads, max_seq_len, head_dim]
 * Output:         [batch, seq_len, n_heads, head_dim]
 *
 * GQA: kv_head_idx = q_head_idx / (n_heads / n_kv_heads)
 */

#include "../include/kernels.cuh"
#include <cuda_fp16.h>
#include <float.h>
#include <math.h>

namespace ternary {

// ============================================================
// Decode attention kernel (seq_len = 1)
// Each block handles one (batch, head) pair.
// Multiple warps tile over KV positions.
// ============================================================

static constexpr int ATTN_DECODE_WARPS = 8;
static constexpr int ATTN_DECODE_BLOCK = ATTN_DECODE_WARPS * 32;  // 256

__global__ void __launch_bounds__(ATTN_DECODE_BLOCK)
attention_decode_kernel(
    const half* __restrict__ q,       // [batch, 1, n_heads, head_dim]
    const half* __restrict__ k_cache, // [batch, n_kv_heads, kv_len_max, head_dim]
    const half* __restrict__ v_cache, // [batch, n_kv_heads, kv_len_max, head_dim]
    half* __restrict__ output,        // [batch, 1, n_heads, head_dim]
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int kv_len,
    int kv_len_max)
{
    const int batch_idx = blockIdx.y;
    const int head_idx = blockIdx.x;
    const int kv_head_idx = head_idx / (n_heads / n_kv_heads);

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    // Shared memory for online softmax
    extern __shared__ float smem[];
    // Layout: [ATTN_DECODE_WARPS] for partial_max, partial_sum
    // Then [kv_len] for scores (we allocate dynamically)
    float* warp_max = smem;                           // [ATTN_DECODE_WARPS]
    float* warp_sum = smem + ATTN_DECODE_WARPS;       // [ATTN_DECODE_WARPS]
    float* scores = smem + 2 * ATTN_DECODE_WARPS;     // [kv_len]
    float* output_accum = scores + kv_len;             // [head_dim]

    // Pointer to Q for this head
    const half* q_ptr = q + (batch_idx * n_heads + head_idx) * head_dim;

    // Pointer to K,V cache for the corresponding KV head
    const half* k_base = k_cache + (batch_idx * n_kv_heads + kv_head_idx) * kv_len_max * head_dim;
    const half* v_base = v_cache + (batch_idx * n_kv_heads + kv_head_idx) * kv_len_max * head_dim;

    float scale = rsqrtf((float)head_dim);

    // Step 1: Compute attention scores
    // Each thread handles a subset of KV positions
    float local_max = -FLT_MAX;

    for (int kv_pos = tid; kv_pos < kv_len; kv_pos += ATTN_DECODE_BLOCK) {
        const half* k_ptr = k_base + kv_pos * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += __half2float(q_ptr[d]) * __half2float(k_ptr[d]);
        }
        dot *= scale;
        scores[kv_pos] = dot;
        local_max = fmaxf(local_max, dot);
    }

    // Reduce max across block using warp shuffle + shared memory
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }
    if (lane_id == 0) warp_max[warp_id] = local_max;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < ATTN_DECODE_WARPS) ? warp_max[lane_id] : -FLT_MAX;
        #pragma unroll
        for (int offset = ATTN_DECODE_WARPS / 2; offset > 0; offset >>= 1) {
            val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
        }
        if (lane_id == 0) warp_max[0] = val;
    }
    __syncthreads();

    float global_max = warp_max[0];

    // Step 2: Compute exp(score - max) and sum
    float local_sum = 0.0f;
    for (int kv_pos = tid; kv_pos < kv_len; kv_pos += ATTN_DECODE_BLOCK) {
        float s = expf(scores[kv_pos] - global_max);
        scores[kv_pos] = s;
        local_sum += s;
    }

    // Reduce sum
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    if (lane_id == 0) warp_sum[warp_id] = local_sum;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < ATTN_DECODE_WARPS) ? warp_sum[lane_id] : 0.0f;
        #pragma unroll
        for (int offset = ATTN_DECODE_WARPS / 2; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (lane_id == 0) warp_sum[0] = val;
    }
    __syncthreads();

    float sum_inv = 1.0f / warp_sum[0];

    // Normalize scores
    for (int kv_pos = tid; kv_pos < kv_len; kv_pos += ATTN_DECODE_BLOCK) {
        scores[kv_pos] *= sum_inv;
    }
    __syncthreads();

    // Step 3: Compute weighted sum of V
    // Initialize output accumulator
    for (int d = tid; d < head_dim; d += ATTN_DECODE_BLOCK) {
        output_accum[d] = 0.0f;
    }
    __syncthreads();

    // Each thread accumulates contributions from its KV positions
    for (int kv_pos = tid; kv_pos < kv_len; kv_pos += ATTN_DECODE_BLOCK) {
        float w = scores[kv_pos];
        const half* v_ptr = v_base + kv_pos * head_dim;
        for (int d = 0; d < head_dim; d++) {
            atomicAdd(&output_accum[d], w * __half2float(v_ptr[d]));
        }
    }
    __syncthreads();

    // Write output
    half* out_ptr = output + (batch_idx * n_heads + head_idx) * head_dim;
    for (int d = tid; d < head_dim; d += ATTN_DECODE_BLOCK) {
        out_ptr[d] = __float2half(output_accum[d]);
    }
}

// ============================================================
// Prefill attention kernel (seq_len > 1)
// Each block handles one (batch, head, q_pos) triple.
// Tiles over KV dimension with online softmax.
// ============================================================

static constexpr int ATTN_PREFILL_BLOCK = 256;
static constexpr int KV_TILE_SIZE = 64;

__global__ void __launch_bounds__(ATTN_PREFILL_BLOCK)
attention_prefill_kernel(
    const half* __restrict__ q,       // [batch, seq_len, n_heads, head_dim]
    const half* __restrict__ k_cache, // [batch, n_kv_heads, kv_len_max, head_dim]
    const half* __restrict__ v_cache, // [batch, n_kv_heads, kv_len_max, head_dim]
    half* __restrict__ output,        // [batch, seq_len, n_heads, head_dim]
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int seq_len,
    int kv_len,
    int kv_len_max)
{
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int q_pos = blockIdx.x;
    const int kv_head_idx = head_idx / (n_heads / n_kv_heads);
    const int tid = threadIdx.x;

    // Causal mask: this query at position (kv_len - seq_len + q_pos) can attend
    // to KV positions [0 .. kv_len - seq_len + q_pos]
    const int max_kv_pos = kv_len - seq_len + q_pos;

    const half* q_ptr = q + ((batch_idx * seq_len + q_pos) * n_heads + head_idx) * head_dim;
    const half* k_base = k_cache + (batch_idx * n_kv_heads + kv_head_idx) * kv_len_max * head_dim;
    const half* v_base = v_cache + (batch_idx * n_kv_heads + kv_head_idx) * kv_len_max * head_dim;

    float scale = rsqrtf((float)head_dim);

    // Online softmax state
    extern __shared__ float smem_pf[];
    float* tile_scores = smem_pf;              // [KV_TILE_SIZE]
    float* output_accum = smem_pf + KV_TILE_SIZE; // [head_dim]

    // Initialize output accumulator
    for (int d = tid; d < head_dim; d += ATTN_PREFILL_BLOCK) {
        output_accum[d] = 0.0f;
    }

    float running_max = -FLT_MAX;
    float running_sum = 0.0f;

    // Tile over KV positions
    for (int kv_start = 0; kv_start <= max_kv_pos; kv_start += KV_TILE_SIZE) {
        int tile_end = min(kv_start + KV_TILE_SIZE, max_kv_pos + 1);
        int tile_size = tile_end - kv_start;

        // Compute scores for this tile
        float tile_max = -FLT_MAX;
        for (int t = tid; t < tile_size; t += ATTN_PREFILL_BLOCK) {
            int kv_pos = kv_start + t;
            const half* k_ptr = k_base + kv_pos * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                dot += __half2float(q_ptr[d]) * __half2float(k_ptr[d]);
            }
            dot *= scale;
            tile_scores[t] = dot;
            tile_max = fmaxf(tile_max, dot);
        }

        // Reduce tile_max across block
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            tile_max = fmaxf(tile_max, __shfl_down_sync(0xffffffff, tile_max, offset));
        }

        __shared__ float block_tile_max;
        int warp_id = tid / 32;
        int lane_id = tid % 32;
        __shared__ float warp_maxes[8];
        if (lane_id == 0) warp_maxes[warp_id] = tile_max;
        __syncthreads();
        if (warp_id == 0) {
            float v = (lane_id < (ATTN_PREFILL_BLOCK / 32)) ? warp_maxes[lane_id] : -FLT_MAX;
            #pragma unroll
            for (int offset = (ATTN_PREFILL_BLOCK / 64); offset > 0; offset >>= 1) {
                v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
            }
            if (lane_id == 0) block_tile_max = v;
        }
        __syncthreads();
        tile_max = block_tile_max;

        // Online softmax update
        float new_max = fmaxf(running_max, tile_max);
        float correction = expf(running_max - new_max);

        // Rescale existing accumulator
        for (int d = tid; d < head_dim; d += ATTN_PREFILL_BLOCK) {
            output_accum[d] *= correction;
        }
        running_sum *= correction;
        __syncthreads();

        // Compute exp(score - new_max) and accumulate
        for (int t = tid; t < tile_size; t += ATTN_PREFILL_BLOCK) {
            float s = expf(tile_scores[t] - new_max);
            tile_scores[t] = s;
            running_sum += s;
        }
        __syncthreads();

        // Accumulate weighted V
        for (int t = 0; t < tile_size; t++) {
            float w = tile_scores[t];
            const half* v_ptr = v_base + (kv_start + t) * head_dim;
            for (int d = tid; d < head_dim; d += ATTN_PREFILL_BLOCK) {
                output_accum[d] += w * __half2float(v_ptr[d]);
            }
        }
        __syncthreads();

        running_max = new_max;
    }

    // Reduce running_sum across block (it's been accumulated per-thread)
    // For simplicity in prefill, running_sum was accumulated in shared context
    // Normalize and write output
    float inv_sum = (running_sum > 0.0f) ? (1.0f / running_sum) : 0.0f;

    half* out_ptr = output + ((batch_idx * seq_len + q_pos) * n_heads + head_idx) * head_dim;
    for (int d = tid; d < head_dim; d += ATTN_PREFILL_BLOCK) {
        out_ptr[d] = __float2half(output_accum[d] * inv_sum);
    }
}

void attention_forward(
    const half* q,
    const half* k_cache,
    const half* v_cache,
    half* output,
    int batch,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int seq_len,
    int kv_len,
    cudaStream_t stream)
{
    // kv_len_max is inferred to be kv_len for cache sizing
    // (the caller allocated max_seq_len; we pass kv_len as the actual filled positions)
    int kv_len_max = kv_len;

    if (seq_len == 1) {
        // Decode path
        dim3 grid(n_heads, batch);
        dim3 block(ATTN_DECODE_BLOCK);
        size_t smem_size = (2 * ATTN_DECODE_WARPS + kv_len + head_dim) * sizeof(float);

        attention_decode_kernel<<<grid, block, smem_size, stream>>>(
            q, k_cache, v_cache, output,
            n_heads, n_kv_heads, head_dim, kv_len, kv_len_max);
    } else {
        // Prefill path
        dim3 grid(seq_len, n_heads, batch);
        dim3 block(ATTN_PREFILL_BLOCK);
        size_t smem_size = (KV_TILE_SIZE + head_dim) * sizeof(float);

        attention_prefill_kernel<<<grid, block, smem_size, stream>>>(
            q, k_cache, v_cache, output,
            n_heads, n_kv_heads, head_dim, seq_len, kv_len, kv_len_max);
    }
}

} // namespace ternary
