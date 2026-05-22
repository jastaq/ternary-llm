/*
 * Ternary GEMM - Prefill phase matrix-matrix multiply
 *
 * C[M,N] = A[M,K] @ TernaryB[N,K]^T
 *
 * A is [M, K] row-major FP16.
 * B is stored as:
 *   nz_B[N, K/32]   - nonzero masks
 *   sign_B[N, K/32] - sign masks  
 *   scales_B[N, K/group_size] - per-group scales
 * C is [M, N] row-major FP16.
 *
 * Tiled approach with TILE_M=64, TILE_N=64, TILE_K=128 (group-aligned).
 */

#include "../include/kernels.cuh"
#include <cuda_fp16.h>
#include <cstdint>

namespace ternary {

static constexpr int TILE_M_GEMM = 64;
static constexpr int TILE_N_GEMM = 64;
static constexpr int TILE_K_GEMM = 128;  // Aligned with typical group_size

// Thread block: 16x16 = 256 threads
// Each thread computes a 4x4 sub-tile of C
static constexpr int THREADS_M = 16;
static constexpr int THREADS_N = 16;
static constexpr int SUB_M = TILE_M_GEMM / THREADS_M;  // 4
static constexpr int SUB_N = TILE_N_GEMM / THREADS_N;  // 4

__global__ void __launch_bounds__(256)
ternary_gemm_kernel(
    const half* __restrict__ A,
    const uint32_t* __restrict__ nz_B,
    const uint32_t* __restrict__ sign_B,
    const half* __restrict__ scales_B,
    half* __restrict__ C,
    int M, int N, int K, int group_size)
{
    const int tile_m = blockIdx.y * TILE_M_GEMM;
    const int tile_n = blockIdx.x * TILE_N_GEMM;
    const int tx = threadIdx.x;  // 0..15
    const int ty = threadIdx.y;  // 0..15

    const int masks_per_row_B = K / 32;
    const int groups_per_row_B = K / group_size;
    const int masks_per_tile_k = TILE_K_GEMM / 32;  // 4

    // Shared memory
    __shared__ half smem_A[TILE_M_GEMM][TILE_K_GEMM];            // 64 * 128 * 2 = 16KB
    __shared__ uint32_t smem_nz_B[TILE_N_GEMM][masks_per_tile_k]; // 64 * 4 * 4 = 1KB
    __shared__ uint32_t smem_sg_B[TILE_N_GEMM][masks_per_tile_k]; // 64 * 4 * 4 = 1KB

    // Accumulator for this thread's 4x4 sub-tile
    float acc[SUB_M][SUB_N];
    #pragma unroll
    for (int i = 0; i < SUB_M; i++)
        #pragma unroll
        for (int j = 0; j < SUB_N; j++)
            acc[i][j] = 0.0f;

    const int tid = ty * THREADS_N + tx;  // 0..255

    // Iterate over K dimension in tiles
    for (int k_tile = 0; k_tile < K; k_tile += TILE_K_GEMM) {
        // --- Load A tile into shared memory ---
        // A tile: TILE_M x TILE_K = 64 x 128 = 8192 elements
        // 256 threads, each loads 32 elements
        for (int i = tid; i < TILE_M_GEMM * TILE_K_GEMM; i += 256) {
            int row = i / TILE_K_GEMM;
            int col = i % TILE_K_GEMM;
            int global_row = tile_m + row;
            int global_col = k_tile + col;
            if (global_row < M && global_col < K) {
                smem_A[row][col] = __ldg(&A[global_row * K + global_col]);
            } else {
                smem_A[row][col] = __float2half(0.0f);
            }
        }

        // --- Load B mask tiles into shared memory ---
        // B masks tile: TILE_N x masks_per_tile_k = 64 x 4 = 256 elements per mask type
        for (int i = tid; i < TILE_N_GEMM * masks_per_tile_k; i += 256) {
            int n_local = i / masks_per_tile_k;
            int w_local = i % masks_per_tile_k;
            int global_n = tile_n + n_local;
            int global_w = k_tile / 32 + w_local;
            if (global_n < N && global_w < masks_per_row_B) {
                smem_nz_B[n_local][w_local] = __ldg(&nz_B[global_n * masks_per_row_B + global_w]);
                smem_sg_B[n_local][w_local] = __ldg(&sign_B[global_n * masks_per_row_B + global_w]);
            } else {
                smem_nz_B[n_local][w_local] = 0;
                smem_sg_B[n_local][w_local] = 0;
            }
        }

        __syncthreads();

        // --- Compute partial products ---
        // This thread handles rows [ty*SUB_M .. ty*SUB_M+3] of the M tile
        // and columns [tx*SUB_N .. tx*SUB_N+3] of the N tile

        // Determine the group index for scaling
        int group_idx = k_tile / group_size;

        // For each sub-tile N column this thread handles
        #pragma unroll
        for (int sn = 0; sn < SUB_N; sn++) {
            int n_local = tx * SUB_N + sn;
            int global_n = tile_n + n_local;

            // Get scale for this N row at this group
            float scale_val = 0.0f;
            if (global_n < N) {
                // Check if this k_tile spans multiple groups
                // For simplicity, assume TILE_K == group_size (both 128)
                scale_val = __half2float(__ldg(&scales_B[global_n * groups_per_row_B + group_idx]));
            }

            // Process each mask word in the tile
            for (int w = 0; w < masks_per_tile_k; w++) {
                uint32_t nz = smem_nz_B[n_local][w];
                uint32_t sg = smem_sg_B[n_local][w];
                uint32_t pos_mask = nz & sg;
                uint32_t neg_mask = nz & (~sg);

                // For positive bits
                uint32_t bits = pos_mask;
                while (bits) {
                    int bit = __ffs(bits) - 1;
                    int k_local = w * 32 + bit;

                    #pragma unroll
                    for (int sm = 0; sm < SUB_M; sm++) {
                        int m_local = ty * SUB_M + sm;
                        acc[sm][sn] += scale_val * __half2float(smem_A[m_local][k_local]);
                    }
                    bits &= bits - 1;
                }

                // For negative bits
                bits = neg_mask;
                while (bits) {
                    int bit = __ffs(bits) - 1;
                    int k_local = w * 32 + bit;

                    #pragma unroll
                    for (int sm = 0; sm < SUB_M; sm++) {
                        int m_local = ty * SUB_M + sm;
                        acc[sm][sn] -= scale_val * __half2float(smem_A[m_local][k_local]);
                    }
                    bits &= bits - 1;
                }
            }
        }

        __syncthreads();
    }

    // --- Write results ---
    #pragma unroll
    for (int sm = 0; sm < SUB_M; sm++) {
        int global_m = tile_m + ty * SUB_M + sm;
        if (global_m >= M) continue;

        #pragma unroll
        for (int sn = 0; sn < SUB_N; sn++) {
            int global_n = tile_n + tx * SUB_N + sn;
            if (global_n >= N) continue;

            C[global_m * N + global_n] = __float2half(acc[sm][sn]);
        }
    }
}

void ternary_gemm(
    const half* A,
    const uint32_t* nz_B,
    const uint32_t* sign_B,
    const half* scales_B,
    half* C,
    int M, int N, int K, int group_size,
    cudaStream_t stream)
{
    dim3 block(THREADS_N, THREADS_M);  // 16x16 = 256
    dim3 grid(
        (N + TILE_N_GEMM - 1) / TILE_N_GEMM,
        (M + TILE_M_GEMM - 1) / TILE_M_GEMM
    );

    ternary_gemm_kernel<<<grid, block, 0, stream>>>(
        A, nz_B, sign_B, scales_B, C, M, N, K, group_size);
}

} // namespace ternary
