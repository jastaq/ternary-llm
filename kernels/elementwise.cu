/*
 * Elementwise and utility kernels:
 * 1. add_forward       - vector addition with half2 vectorization
 * 2. embedding_lookup  - token embedding table lookup
 * 3. copy_to_kv_cache  - copy K,V into KV cache at position
 * 4. fp16_gemv         - standard FP16 matrix-vector multiply (for lm_head)
 * 5. fp16_gemm         - standard FP16 tiled GEMM (for prefill lm_head)
 */

#include "../include/kernels.cuh"
#include <cuda_fp16.h>

namespace ternary {

// ============================================================
// 1. Vector Addition: out[i] = a[i] + b[i]
// ============================================================

__global__ void add_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ output,
    int size)
{
    const int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;

    if (idx + 1 < size) {
        const half2* a2 = reinterpret_cast<const half2*>(a);
        const half2* b2 = reinterpret_cast<const half2*>(b);
        half2* out2 = reinterpret_cast<half2*>(output);
        int idx2 = idx / 2;
        out2[idx2] = __hadd2(a2[idx2], b2[idx2]);
    } else if (idx < size) {
        output[idx] = __hadd(a[idx], b[idx]);
    }
}

void add_forward(
    const half* a,
    const half* b,
    half* output,
    int size,
    cudaStream_t stream)
{
    int num_threads = (size + 1) / 2;
    const int block = 256;
    int grid = (num_threads + block - 1) / block;
    add_kernel<<<grid, block, 0, stream>>>(a, b, output, size);
}

// ============================================================
// 2. Embedding Lookup
// ============================================================

__global__ void embedding_lookup_kernel(
    const half* __restrict__ table,
    const int* __restrict__ indices,
    half* __restrict__ output,
    int n_tokens,
    int dim)
{
    const int token_idx = blockIdx.x;
    const int tid = threadIdx.x;

    if (token_idx >= n_tokens) return;

    int token_id = indices[token_idx];
    const half* src = table + token_id * dim;
    half* dst = output + token_idx * dim;

    // Each thread copies multiple elements
    for (int i = tid; i < dim; i += blockDim.x) {
        dst[i] = src[i];
    }
}

void embedding_lookup(
    const half* table,
    const int* indices,
    half* output,
    int n_tokens,
    int dim,
    cudaStream_t stream)
{
    const int block = 256;
    embedding_lookup_kernel<<<n_tokens, block, 0, stream>>>(
        table, indices, output, n_tokens, dim);
}

// ============================================================
// 3. Copy to KV Cache
// K input:  [batch, seq_len, n_kv_heads, head_dim]
// K cache:  [batch, n_kv_heads, max_seq_len, head_dim]
// Copy position seq_idx -> cache position (start_pos + seq_idx)
// ============================================================

__global__ void copy_to_kv_cache_kernel(
    const half* __restrict__ k,
    const half* __restrict__ v,
    half* __restrict__ k_cache,
    half* __restrict__ v_cache,
    int n_kv_heads,
    int head_dim,
    int seq_len,
    int start_pos,
    int max_seq_len)
{
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int seq_idx = blockIdx.x;
    const int tid = threadIdx.x;

    if (seq_idx >= seq_len) return;

    int cache_pos = start_pos + seq_idx;
    if (cache_pos >= max_seq_len) return;

    // Source: [batch, seq_len, n_kv_heads, head_dim]
    int src_offset = ((batch_idx * seq_len + seq_idx) * n_kv_heads + head_idx) * head_dim;

    // Destination: [batch, n_kv_heads, max_seq_len, head_dim]
    int dst_offset = ((batch_idx * n_kv_heads + head_idx) * max_seq_len + cache_pos) * head_dim;

    for (int d = tid; d < head_dim; d += blockDim.x) {
        k_cache[dst_offset + d] = k[src_offset + d];
        v_cache[dst_offset + d] = v[src_offset + d];
    }
}

void copy_to_kv_cache(
    const half* k,
    const half* v,
    half* k_cache,
    half* v_cache,
    int batch,
    int n_kv_heads,
    int head_dim,
    int seq_len,
    int start_pos,
    int max_seq_len,
    cudaStream_t stream)
{
    const int block = 128;
    dim3 grid(seq_len, n_kv_heads, batch);
    copy_to_kv_cache_kernel<<<grid, block, 0, stream>>>(
        k, v, k_cache, v_cache, n_kv_heads, head_dim,
        seq_len, start_pos, max_seq_len);
}

// ============================================================
// 4. FP16 GEMV: y[row] = dot(weight[row], input)
// Each warp handles one output row.
// ============================================================

static constexpr int FP16_GEMV_WARPS = 4;
static constexpr int FP16_GEMV_BLOCK = FP16_GEMV_WARPS * 32;

__global__ void __launch_bounds__(FP16_GEMV_BLOCK)
fp16_gemv_kernel(
    const half* __restrict__ weight,
    const half* __restrict__ input,
    half* __restrict__ output,
    int rows,
    int cols)
{
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int row = blockIdx.x * FP16_GEMV_WARPS + warp_id;

    if (row >= rows) return;

    const half* w_row = weight + row * cols;

    float sum = 0.0f;
    for (int c = lane_id; c < cols; c += 32) {
        sum += __half2float(__ldg(&w_row[c])) * __half2float(__ldg(&input[c]));
    }

    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane_id == 0) {
        output[row] = __float2half(sum);
    }
}

void fp16_gemv(
    const half* weight,
    const half* input,
    half* output,
    int rows,
    int cols,
    cudaStream_t stream)
{
    dim3 block(FP16_GEMV_BLOCK);
    dim3 grid((rows + FP16_GEMV_WARPS - 1) / FP16_GEMV_WARPS);
    fp16_gemv_kernel<<<grid, block, 0, stream>>>(weight, input, output, rows, cols);
}

// ============================================================
// 5. FP16 GEMM: C[M,N] = A[M,K] @ B[N,K]^T
// Tiled approach with shared memory.
// ============================================================

static constexpr int FP16_GEMM_TILE_M = 64;
static constexpr int FP16_GEMM_TILE_N = 64;
static constexpr int FP16_GEMM_TILE_K = 16;
static constexpr int FP16_GEMM_THREADS_M = 16;
static constexpr int FP16_GEMM_THREADS_N = 16;
static constexpr int FP16_GEMM_SUB_M = FP16_GEMM_TILE_M / FP16_GEMM_THREADS_M; // 4
static constexpr int FP16_GEMM_SUB_N = FP16_GEMM_TILE_N / FP16_GEMM_THREADS_N; // 4

__global__ void __launch_bounds__(256)
fp16_gemm_kernel(
    const half* __restrict__ A,  // [M, K]
    const half* __restrict__ B,  // [N, K] (stored row-major, C = A @ B^T)
    half* __restrict__ C,        // [M, N]
    int M, int N, int K)
{
    const int tile_m = blockIdx.y * FP16_GEMM_TILE_M;
    const int tile_n = blockIdx.x * FP16_GEMM_TILE_N;
    const int tx = threadIdx.x; // 0..15
    const int ty = threadIdx.y; // 0..15
    const int tid = ty * FP16_GEMM_THREADS_N + tx;

    __shared__ half smem_A[FP16_GEMM_TILE_M][FP16_GEMM_TILE_K];
    __shared__ half smem_B[FP16_GEMM_TILE_N][FP16_GEMM_TILE_K];

    float acc[FP16_GEMM_SUB_M][FP16_GEMM_SUB_N];
    #pragma unroll
    for (int i = 0; i < FP16_GEMM_SUB_M; i++)
        #pragma unroll
        for (int j = 0; j < FP16_GEMM_SUB_N; j++)
            acc[i][j] = 0.0f;

    for (int k_tile = 0; k_tile < K; k_tile += FP16_GEMM_TILE_K) {
        // Load A tile: 64x16 = 1024 elements, 256 threads -> 4 each
        for (int i = tid; i < FP16_GEMM_TILE_M * FP16_GEMM_TILE_K; i += 256) {
            int row = i / FP16_GEMM_TILE_K;
            int col = i % FP16_GEMM_TILE_K;
            int gr = tile_m + row;
            int gc = k_tile + col;
            smem_A[row][col] = (gr < M && gc < K) ? __ldg(&A[gr * K + gc]) : __float2half(0.0f);
        }

        // Load B tile: 64x16 = 1024 elements
        for (int i = tid; i < FP16_GEMM_TILE_N * FP16_GEMM_TILE_K; i += 256) {
            int row = i / FP16_GEMM_TILE_K;
            int col = i % FP16_GEMM_TILE_K;
            int gr = tile_n + row;
            int gc = k_tile + col;
            smem_B[row][col] = (gr < N && gc < K) ? __ldg(&B[gr * K + gc]) : __float2half(0.0f);
        }

        __syncthreads();

        // Compute
        #pragma unroll
        for (int kk = 0; kk < FP16_GEMM_TILE_K; kk++) {
            #pragma unroll
            for (int sm = 0; sm < FP16_GEMM_SUB_M; sm++) {
                float a_val = __half2float(smem_A[ty * FP16_GEMM_SUB_M + sm][kk]);
                #pragma unroll
                for (int sn = 0; sn < FP16_GEMM_SUB_N; sn++) {
                    float b_val = __half2float(smem_B[tx * FP16_GEMM_SUB_N + sn][kk]);
                    acc[sm][sn] += a_val * b_val;
                }
            }
        }

        __syncthreads();
    }

    // Write results
    #pragma unroll
    for (int sm = 0; sm < FP16_GEMM_SUB_M; sm++) {
        int gm = tile_m + ty * FP16_GEMM_SUB_M + sm;
        if (gm >= M) continue;
        #pragma unroll
        for (int sn = 0; sn < FP16_GEMM_SUB_N; sn++) {
            int gn = tile_n + tx * FP16_GEMM_SUB_N + sn;
            if (gn >= N) continue;
            C[gm * N + gn] = __float2half(acc[sm][sn]);
        }
    }
}

void fp16_gemm(
    const half* A,
    const half* B,
    half* C,
    int M, int N, int K,
    cudaStream_t stream)
{
    dim3 block(FP16_GEMM_THREADS_N, FP16_GEMM_THREADS_M); // 16x16
    dim3 grid(
        (N + FP16_GEMM_TILE_N - 1) / FP16_GEMM_TILE_N,
        (M + FP16_GEMM_TILE_M - 1) / FP16_GEMM_TILE_M
    );
    fp16_gemm_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}

} // namespace ternary
