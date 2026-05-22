/*
 * Fused SiLU * Gate for SwiGLU activation
 *
 * output[i] = gate[i] * sigmoid(gate[i]) * up[i]
 *           = (gate[i] / (1 + exp(-gate[i]))) * up[i]
 *
 * Vectorized with half2 for 2x throughput.
 * FP32 intermediate computation for precision.
 */

#include "../include/kernels.cuh"
#include <cuda_fp16.h>

namespace ternary {

static constexpr int ACTIVATION_BLOCK_SIZE = 256;

__global__ void silu_mul_kernel(
    const half* __restrict__ gate,
    const half* __restrict__ up,
    half* __restrict__ output,
    int size)
{
    // Process 2 elements per thread using half2
    const int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;

    if (idx + 1 < size) {
        // Vectorized load
        const half2* gate2 = reinterpret_cast<const half2*>(gate);
        const half2* up2 = reinterpret_cast<const half2*>(up);
        half2* out2 = reinterpret_cast<half2*>(output);

        int idx2 = idx / 2;
        half2 g = gate2[idx2];
        half2 u = up2[idx2];

        // Convert to float for precision
        float g0 = __half2float(g.x);
        float g1 = __half2float(g.y);
        float u0 = __half2float(u.x);
        float u1 = __half2float(u.y);

        // SiLU: x * sigmoid(x) = x / (1 + exp(-x))
        float s0 = g0 / (1.0f + expf(-g0));
        float s1 = g1 / (1.0f + expf(-g1));

        // Multiply with up
        float o0 = s0 * u0;
        float o1 = s1 * u1;

        out2[idx2] = make_half2(__float2half(o0), __float2half(o1));
    } else if (idx < size) {
        // Handle last odd element
        float g = __half2float(gate[idx]);
        float u = __half2float(up[idx]);
        float s = g / (1.0f + expf(-g));
        output[idx] = __float2half(s * u);
    }
}

void silu_mul_forward(
    const half* gate,
    const half* up,
    half* output,
    int size,
    cudaStream_t stream)
{
    // Each thread handles 2 elements
    int num_threads = (size + 1) / 2;
    dim3 block(ACTIVATION_BLOCK_SIZE);
    dim3 grid((num_threads + ACTIVATION_BLOCK_SIZE - 1) / ACTIVATION_BLOCK_SIZE);

    silu_mul_kernel<<<grid, block, 0, stream>>>(gate, up, output, size);
}

// =========================================================================
//  INT8 Absmax Quantisation:  half[n] → int8[n] + float scale
//
//  Phase 1: find global absmax via hierarchical reduction
//  Phase 2: quantise each element:  out = clamp(round(x * 127 / absmax), -127, 127)
//
//  Both phases are fused into a single kernel using cooperative groups or
//  a two-pass approach.  For the typical vector sizes in LLM inference
//  (≤ 14336) a single-block kernel is sufficient and avoids launch overhead.
// =========================================================================
static constexpr int QUANT_BLOCK = 256;

__global__ void quantize_absmax_int8_kernel(
    const half* __restrict__ input,
    int8_t*     __restrict__ output,
    float*      __restrict__ out_scale,   // [1] — scalar
    int n)
{
    __shared__ float smem_max[QUANT_BLOCK / 32];  // per-warp max

    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int wid  = tid >> 5;

    // ---- Phase 1: find absmax ----
    float local_max = 0.0f;
    for (int i = tid; i < n; i += QUANT_BLOCK) {
        float v = fabsf(__half2float(__ldg(&input[i])));
        local_max = fmaxf(local_max, v);
    }

    // Warp reduce
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, off));

    if (lane == 0) smem_max[wid] = local_max;
    __syncthreads();

    // Block reduce (first warp reads all per-warp maxes)
    float block_max = 0.0f;
    if (wid == 0) {
        block_max = (tid < (QUANT_BLOCK / 32)) ? smem_max[tid] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            block_max = fmaxf(block_max, __shfl_down_sync(0xffffffff, block_max, off));
    }

    // Broadcast absmax to all threads
    __shared__ float shared_absmax;
    if (tid == 0) {
        shared_absmax = block_max;
        *out_scale = (block_max > 0.0f) ? (block_max / 127.0f) : 1.0f;
    }
    __syncthreads();

    float absmax = shared_absmax;
    float inv_scale = (absmax > 0.0f) ? (127.0f / absmax) : 0.0f;

    // ---- Phase 2: quantise ----
    for (int i = tid; i < n; i += QUANT_BLOCK) {
        float v = __half2float(__ldg(&input[i]));
        int   q = __float2int_rn(v * inv_scale);
        q = max(-127, min(127, q));
        output[i] = static_cast<int8_t>(q);
    }
}

void quantize_absmax_int8(
    const half* input, int8_t* output, float* scale,
    int n, cudaStream_t stream)
{
    // Single-block launch — sufficient for n ≤ ~100k (covers all LLM dims)
    quantize_absmax_int8_kernel<<<1, QUANT_BLOCK, 0, stream>>>(
        input, output, scale, n);
}

} // namespace ternary
