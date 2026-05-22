/*
 * Ternary GEMV - The most performance-critical kernel
 * 
 * Computes y[row] = sum_groups { scale[g] * (sum(x[i] for +1 bits) - sum(x[i] for -1 bits)) }
 *
 * Weight packing:
 *   nonzero_masks[row][word]: bit i=1 => weight at col (word*32+i) is nonzero
 *   sign_masks[row][word]:   bit i=1 => +1, bit i=0 => -1
 *   positive_mask = nonzero & sign        (nonzero and positive)
 *   negative_mask = nonzero & (~sign)     (nonzero and negative)
 *
 * Key insight: ternary weights mean NO multiplications for the weight part,
 * only additions and subtractions of input values.
 */

#include "../include/kernels.cuh"
#include <cuda_fp16.h>
#include <cstdint>

namespace ternary {

static constexpr int WARPS_PER_BLOCK_GEMV = 4;
static constexpr int WARP_SIZE = 32;
static constexpr int BLOCK_SIZE_GEMV = WARPS_PER_BLOCK_GEMV * WARP_SIZE; // 128

/*
 * Each warp computes one output row.
 * Multiple warps per block share the input vector loaded into shared memory.
 * Input is loaded cooperatively by all threads in the block.
 */
__global__ void __launch_bounds__(BLOCK_SIZE_GEMV)
ternary_gemv_kernel(
    const uint32_t* __restrict__ nonzero_masks,
    const uint32_t* __restrict__ sign_masks,
    const half* __restrict__ scales,
    const half* __restrict__ input,
    half* __restrict__ output,
    int rows,
    int cols,
    int group_size)
{
    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;
    const int row = blockIdx.x * WARPS_PER_BLOCK_GEMV + warp_id;

    if (row >= rows) return;

    const int masks_per_row = cols / 32;
    const int groups_per_row = cols / group_size;
    const int masks_per_group = group_size / 32;

    // Shared memory for input vector tile
    // We process the input in tiles that fit in shared memory
    // Each tile = group_size elements (aligned with scale groups)
    extern __shared__ half smem_input[];

    float row_sum = 0.0f;

    // Process each group
    for (int g = 0; g < groups_per_row; g++) {
        const int col_start = g * group_size;
        const int mask_start = g * masks_per_group;

        // Cooperatively load input tile into shared memory
        // All threads in the block participate
        for (int i = threadIdx.y * WARP_SIZE + threadIdx.x;
             i < group_size;
             i += BLOCK_SIZE_GEMV) {
            smem_input[i] = __ldg(&input[col_start + i]);
        }
        __syncthreads();

        // Each thread in the warp processes a stripe of mask words
        float group_sum = 0.0f;

        for (int w = lane_id; w < masks_per_group; w += WARP_SIZE) {
            const int mask_idx = row * masks_per_row + mask_start + w;
            uint32_t nz = __ldg(&nonzero_masks[mask_idx]);
            uint32_t sg = __ldg(&sign_masks[mask_idx]);

            uint32_t pos_mask = nz & sg;
            uint32_t neg_mask = nz & (~sg);

            // Sum positive contributions using __ffs() to iterate set bits
            float pos_sum = 0.0f;
            uint32_t bits = pos_mask;
            while (bits) {
                int bit = __ffs(bits) - 1;  // find lowest set bit (0-indexed)
                int local_col = w * 32 + bit;
                pos_sum += __half2float(smem_input[local_col]);
                bits &= bits - 1;  // clear lowest set bit
            }

            // Sum negative contributions
            float neg_sum = 0.0f;
            bits = neg_mask;
            while (bits) {
                int bit = __ffs(bits) - 1;
                int local_col = w * 32 + bit;
                neg_sum += __half2float(smem_input[local_col]);
                bits &= bits - 1;
            }

            group_sum += pos_sum - neg_sum;
        }

        // Warp-level reduction
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            group_sum += __shfl_down_sync(0xffffffff, group_sum, offset);
        }

        // Lane 0 applies the group scale and accumulates
        if (lane_id == 0) {
            float scale_val = __half2float(__ldg(&scales[row * groups_per_row + g]));
            row_sum += scale_val * group_sum;
        }

        __syncthreads();  // Ensure smem is safe to overwrite in next iteration
    }

    // Lane 0 writes the result
    if (lane_id == 0) {
        output[row] = __float2half(row_sum);
    }
}

void ternary_gemv(
    const uint32_t* nonzero_masks,
    const uint32_t* sign_masks,
    const half* scales,
    const half* input,
    half* output,
    int rows,
    int cols,
    int group_size,
    cudaStream_t stream)
{
    dim3 block(WARP_SIZE, WARPS_PER_BLOCK_GEMV);
    dim3 grid((rows + WARPS_PER_BLOCK_GEMV - 1) / WARPS_PER_BLOCK_GEMV);

    // Shared memory for the input tile (one group at a time)
    size_t smem_size = group_size * sizeof(half);

    ternary_gemv_kernel<<<grid, block, smem_size, stream>>>(
        nonzero_masks, sign_masks, scales, input, output,
        rows, cols, group_size);
}

} // namespace ternary
