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

} // namespace ternary
