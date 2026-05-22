/*
 * RMSNorm with optional fused residual add
 *
 * Algorithm:
 * 1. If residual != nullptr: tmp[i] = input[i] + residual[i], store to residual_out
 * 2. var = sum(tmp[i]^2) / dim
 * 3. rms = 1.0 / sqrt(var + eps)
 * 4. output[i] = tmp[i] * rms * weight[i]
 *
 * One block per batch element. BlockDim = 256.
 */

#include "../include/kernels.cuh"
#include <cuda_fp16.h>

namespace ternary {

static constexpr int RMSNORM_BLOCK_SIZE = 256;

__global__ void __launch_bounds__(RMSNORM_BLOCK_SIZE)
rmsnorm_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    half* __restrict__ output,
    const half* __restrict__ residual,
    half* __restrict__ residual_out,
    int dim,
    float eps)
{
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = RMSNORM_BLOCK_SIZE;

    const half* x = input + batch_idx * dim;
    half* out = output + batch_idx * dim;

    // Shared memory for block-level reduction
    __shared__ float smem_reduce[RMSNORM_BLOCK_SIZE / 32];  // one per warp

    // Optional residual pointer setup
    const half* res = residual ? residual + batch_idx * dim : nullptr;
    half* res_out = residual_out ? residual_out + batch_idx * dim : nullptr;

    // Pass 1: Optionally add residual, compute sum of squares
    float ss = 0.0f;
    for (int i = tid; i < dim; i += stride) {
        float val = __half2float(x[i]);

        if (res) {
            val += __half2float(res[i]);
            if (res_out) {
                res_out[i] = __float2half(val);
            }
        }

        ss += val * val;
    }

    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        ss += __shfl_down_sync(0xffffffff, ss, offset);
    }

    // Write warp results to shared memory
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    if (lane_id == 0) {
        smem_reduce[warp_id] = ss;
    }
    __syncthreads();

    // First warp reduces across warps
    if (warp_id == 0) {
        float val = (lane_id < (RMSNORM_BLOCK_SIZE / 32)) ? smem_reduce[lane_id] : 0.0f;
        #pragma unroll
        for (int offset = (RMSNORM_BLOCK_SIZE / 64); offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (lane_id == 0) {
            smem_reduce[0] = val;
        }
    }
    __syncthreads();

    float variance = smem_reduce[0] / (float)dim;
    float rms_inv = rsqrtf(variance + eps);

    // Pass 2: Normalize and apply weight
    for (int i = tid; i < dim; i += stride) {
        float val;
        if (res) {
            // Re-read from residual_out if we stored it, or recompute
            if (res_out) {
                val = __half2float(res_out[i]);
            } else {
                val = __half2float(x[i]) + __half2float(res[i]);
            }
        } else {
            val = __half2float(x[i]);
        }

        float w = __half2float(weight[i]);
        out[i] = __float2half(val * rms_inv * w);
    }
}

void rmsnorm_forward(
    const half* input,
    const half* weight,
    half* output,
    const half* residual,
    half* residual_out,
    int batch,
    int dim,
    float eps,
    cudaStream_t stream)
{
    dim3 block(RMSNORM_BLOCK_SIZE);
    dim3 grid(batch);

    rmsnorm_kernel<<<grid, block, 0, stream>>>(
        input, weight, output, residual, residual_out, dim, eps);
}

} // namespace ternary
