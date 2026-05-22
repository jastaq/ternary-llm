/*
 * Ternary GEMV kernels — two implementations:
 *
 *   1. ternary_gemv()      — FP16 input, __ffs() bit-scanning (legacy)
 *   2. ternary_gemv_int8() — INT8 input, DP4A accumulation   (fast path)
 *
 * The INT8 + DP4A path is ~2-3× faster because:
 *   - No branch-divergent __ffs() loop
 *   - __dp4a() processes 4 int8 MACs in 1 instruction
 *   - INT8 input = 2× less bandwidth vs FP16
 *   - All threads execute the same instruction count (zero divergence)
 */

#include "../include/kernels.cuh"
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>

namespace ternary {

// =========================================================================
//  KERNEL 1 (legacy): FP16 input + __ffs() bit scanning
// =========================================================================
static constexpr int WARPS_LEGACY = 4;
static constexpr int WARP_SIZE    = 32;

__global__ void __launch_bounds__(WARPS_LEGACY * WARP_SIZE)
ternary_gemv_legacy_kernel(
    const uint32_t* __restrict__ nonzero_masks,
    const uint32_t* __restrict__ sign_masks,
    const half*     __restrict__ scales,
    const half*     __restrict__ input,
    half*           __restrict__ output,
    int rows, int cols, int group_size)
{
    const int warp_id = threadIdx.y;
    const int lane    = threadIdx.x;
    const int row     = blockIdx.x * WARPS_LEGACY + warp_id;
    if (row >= rows) return;

    const int masks_per_row  = cols / 32;
    const int groups_per_row = cols / group_size;
    const int masks_per_grp  = group_size / 32;

    extern __shared__ half smem_input[];

    float row_sum = 0.0f;

    for (int g = 0; g < groups_per_row; g++) {
        int col_start  = g * group_size;
        int mask_start = g * masks_per_grp;

        // Load input tile cooperatively
        for (int i = threadIdx.y * WARP_SIZE + threadIdx.x;
             i < group_size; i += WARPS_LEGACY * WARP_SIZE)
            smem_input[i] = __ldg(&input[col_start + i]);
        __syncthreads();

        float group_sum = 0.0f;
        for (int w = lane; w < masks_per_grp; w += WARP_SIZE) {
            int idx = row * masks_per_row + mask_start + w;
            uint32_t nz = __ldg(&nonzero_masks[idx]);
            uint32_t sg = __ldg(&sign_masks[idx]);

            uint32_t pos = nz & sg;
            uint32_t neg = nz & (~sg);

            float ps = 0.0f;
            while (pos) { int b = __ffs(pos)-1; ps += __half2float(smem_input[w*32+b]); pos &= pos-1; }
            float ns = 0.0f;
            while (neg) { int b = __ffs(neg)-1; ns += __half2float(smem_input[w*32+b]); neg &= neg-1; }
            group_sum += ps - ns;
        }

        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            group_sum += __shfl_down_sync(0xffffffff, group_sum, off);

        if (lane == 0) {
            float s = __half2float(__ldg(&scales[row * groups_per_row + g]));
            row_sum += s * group_sum;
        }
        __syncthreads();
    }

    if (lane == 0) output[row] = __float2half(row_sum);
}

void ternary_gemv(
    const uint32_t* nonzero_masks, const uint32_t* sign_masks,
    const half* scales, const half* input, half* output,
    int rows, int cols, int group_size, cudaStream_t stream)
{
    dim3 block(WARP_SIZE, WARPS_LEGACY);
    dim3 grid((rows + WARPS_LEGACY - 1) / WARPS_LEGACY);
    size_t smem = group_size * sizeof(half);
    ternary_gemv_legacy_kernel<<<grid, block, smem, stream>>>(
        nonzero_masks, sign_masks, scales, input, output,
        rows, cols, group_size);
}

// =========================================================================
//  KERNEL 2 (fast): INT8 input + DP4A
//
//  Each warp computes one output row.
//  Within each scale-group (128 elements, default):
//    - 32 threads × 4 elements = 128 elements covered
//    - Each thread: unpack 4 ternary bits → int8x4, load int8x4 input, DP4A
//    - 1 warp reduction at the very end (NOT per-group)
//    - Group scale applied per-thread before accumulation
//
//  Shared memory holds the INT8 input vector for reuse across warps.
// =========================================================================
static constexpr int WARPS_DP4A = 4;

__global__ void __launch_bounds__(WARPS_DP4A * WARP_SIZE)
ternary_gemv_int8_kernel(
    const uint32_t* __restrict__ nonzero_masks,   // [rows, cols/32]
    const uint32_t* __restrict__ sign_masks,       // [rows, cols/32]
    const half*     __restrict__ weight_scales,    // [rows, cols/group_size]
    const int8_t*   __restrict__ input_int8,       // [cols]
    float                        input_scale,
    half*           __restrict__ output,           // [rows]
    int rows, int cols, int group_size)
{
    const int warp_id = threadIdx.y;
    const int lane    = threadIdx.x;
    const int row     = blockIdx.x * WARPS_DP4A + warp_id;
    if (row >= rows) return;

    const int masks_per_row  = cols / 32;
    const int groups_per_row = cols / group_size;

    // --- Load INT8 input into shared memory (once per block) ---
    extern __shared__ int8_t smem_i8[];
    for (int i = threadIdx.y * WARP_SIZE + threadIdx.x;
         i < cols; i += WARPS_DP4A * WARP_SIZE)
        smem_i8[i] = input_int8[i];
    __syncthreads();

    float acc = 0.0f;

    for (int g = 0; g < groups_per_row; g++) {
        const int col_base = g * group_size;
        int group_acc = 0;   // integer accumulator for this group

        // Each thread handles group_size/32 chunks of 4 consecutive elements.
        // For group_size=128: 1 chunk per thread.  For 256: 2 chunks, etc.
        for (int t = 0; t < group_size; t += WARP_SIZE * 4) {
            const int elem = col_base + t + lane * 4;

            // Identify mask word and bit offset for these 4 elements
            const int mask_word = elem / 32;
            const int bit_off   = elem % 32;   // = ((t/4 + lane) * 4) % 32

            uint32_t nz = __ldg(&nonzero_masks[row * masks_per_row + mask_word]);
            uint32_t sg = __ldg(&sign_masks   [row * masks_per_row + mask_word]);

            // Unpack 4 ternary weights → int8 quartet
            //   nz_bit=0            → 0
            //   nz_bit=1, sg_bit=1  → +1
            //   nz_bit=1, sg_bit=0  → -1
            int8_t w[4];
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                const int b  = bit_off + j;
                const int nz_b = (nz >> b) & 1;
                const int sg_b = (sg >> b) & 1;
                w[j] = static_cast<int8_t>(nz_b * (2 * sg_b - 1));
            }

            // Pack into int32 for DP4A
            int32_t w_packed;
            __builtin_memcpy(&w_packed, w, 4);

            // Load 4 INT8 input values (coalesced: adjacent threads read adjacent int32)
            int32_t x_packed;
            __builtin_memcpy(&x_packed, &smem_i8[elem], 4);

            // DP4A: group_acc += w[0]*x[0] + w[1]*x[1] + w[2]*x[2] + w[3]*x[3]
            group_acc = __dp4a(w_packed, x_packed, group_acc);
        }

        // Apply per-group weight scale (each thread applies to its own partial)
        float wscale = __half2float(__ldg(&weight_scales[row * groups_per_row + g]));
        acc += wscale * static_cast<float>(group_acc);
    }

    // Apply input quantisation scale
    acc *= input_scale;

    // Warp-level reduction (only 1 reduction at the end — not per-group!)
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, off);

    if (lane == 0)
        output[row] = __float2half(acc);
}

void ternary_gemv_int8(
    const uint32_t* nonzero_masks, const uint32_t* sign_masks,
    const half* weight_scales,
    const int8_t* input_int8, float input_scale,
    half* output,
    int rows, int cols, int group_size, cudaStream_t stream)
{
    dim3 block(WARP_SIZE, WARPS_DP4A);
    dim3 grid((rows + WARPS_DP4A - 1) / WARPS_DP4A);
    size_t smem = cols * sizeof(int8_t);   // INT8 = half the shared mem vs FP16

    ternary_gemv_int8_kernel<<<grid, block, smem, stream>>>(
        nonzero_masks, sign_masks, weight_scales,
        input_int8, input_scale, output,
        rows, cols, group_size);
}

} // namespace ternary
