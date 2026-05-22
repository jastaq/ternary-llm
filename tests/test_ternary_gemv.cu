/*
 * Test program for ternary_gemv kernel
 *
 * 1. Generate random packed ternary weights (nonzero_masks + sign_masks)
 * 2. Generate random FP16 input
 * 3. Run ternary_gemv kernel on GPU
 * 4. Compute reference on CPU
 * 5. Compare results, print max error and pass/fail
 * 6. Test multiple sizes
 */

#include "../include/kernels.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,\
                    cudaGetErrorString(err));                                 \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

static float rand_float() {
    return (float)rand() / RAND_MAX * 2.0f - 1.0f;
}

static uint32_t rand_uint32() {
    // Generate a full 32-bit random value from two rand() calls
    return ((uint32_t)(rand() & 0xFFFF) << 16) | (uint32_t)(rand() & 0xFFFF);
}

void cpu_reference(
    const uint32_t* nonzero_masks,
    const uint32_t* sign_masks,
    const float* scales,    // float copy of scales
    const float* input,     // float copy of input
    float* output,
    int rows, int cols, int group_size)
{
    int masks_per_row = cols / 32;
    int groups_per_row = cols / group_size;

    for (int r = 0; r < rows; r++) {
        float sum = 0.0f;
        for (int g = 0; g < groups_per_row; g++) {
            float group_sum = 0.0f;
            int col_start = g * group_size;

            for (int j = 0; j < group_size; j++) {
                int col = col_start + j;
                int word = col / 32;
                int bit = col % 32;

                uint32_t nz_word = nonzero_masks[r * masks_per_row + word];
                uint32_t sg_word = sign_masks[r * masks_per_row + word];

                bool is_nonzero = (nz_word >> bit) & 1;
                bool is_positive = (sg_word >> bit) & 1;

                float w = is_nonzero ? (is_positive ? 1.0f : -1.0f) : 0.0f;
                group_sum += w * input[col];
            }
            sum += scales[r * groups_per_row + g] * group_sum;
        }
        output[r] = sum;
    }
}

bool run_test(int rows, int cols, int group_size) {
    printf("Testing ternary_gemv: rows=%d, cols=%d, group_size=%d\n", rows, cols, group_size);

    int masks_per_row = cols / 32;
    int groups_per_row = cols / group_size;

    size_t nz_size = (size_t)rows * masks_per_row;
    size_t sg_size = (size_t)rows * masks_per_row;
    size_t scales_size = (size_t)rows * groups_per_row;

    // Host allocations
    uint32_t* h_nz = (uint32_t*)malloc(nz_size * sizeof(uint32_t));
    uint32_t* h_sg = (uint32_t*)malloc(sg_size * sizeof(uint32_t));
    half* h_scales = (half*)malloc(scales_size * sizeof(half));
    float* h_scales_f = (float*)malloc(scales_size * sizeof(float));
    half* h_input = (half*)malloc(cols * sizeof(half));
    float* h_input_f = (float*)malloc(cols * sizeof(float));
    half* h_output = (half*)malloc(rows * sizeof(half));
    float* h_ref = (float*)malloc(rows * sizeof(float));

    // Generate random data
    for (size_t i = 0; i < nz_size; i++) {
        h_nz[i] = rand_uint32();
    }
    for (size_t i = 0; i < sg_size; i++) {
        h_sg[i] = rand_uint32();
    }
    for (size_t i = 0; i < scales_size; i++) {
        float s = rand_float() * 0.1f;  // Small scales typical for quantization
        h_scales_f[i] = s;
        h_scales[i] = __float2half(s);
    }
    for (int i = 0; i < cols; i++) {
        float v = rand_float();
        h_input_f[i] = v;
        h_input[i] = __float2half(v);
    }

    // CPU reference
    cpu_reference(h_nz, h_sg, h_scales_f, h_input_f, h_ref, rows, cols, group_size);

    // GPU allocations
    uint32_t *d_nz, *d_sg;
    half *d_scales, *d_input, *d_output;

    CUDA_CHECK(cudaMalloc(&d_nz, nz_size * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_sg, sg_size * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_scales, scales_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_input, cols * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output, rows * sizeof(half)));

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_nz, h_nz, nz_size * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sg, h_sg, sg_size * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scales, h_scales, scales_size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, cols * sizeof(half), cudaMemcpyHostToDevice));

    // Run kernel
    ternary::ternary_gemv(d_nz, d_sg, d_scales, d_input, d_output,
                          rows, cols, group_size, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy back
    CUDA_CHECK(cudaMemcpy(h_output, d_output, rows * sizeof(half), cudaMemcpyDeviceToHost));

    // Compare
    float max_err = 0.0f;
    float max_rel_err = 0.0f;
    int worst_row = -1;
    for (int r = 0; r < rows; r++) {
        float gpu_val = __half2float(h_output[r]);
        float ref_val = h_ref[r];
        float err = fabsf(gpu_val - ref_val);
        float rel = (fabsf(ref_val) > 1e-6f) ? err / fabsf(ref_val) : err;

        if (err > max_err) {
            max_err = err;
            worst_row = r;
        }
        max_rel_err = fmaxf(max_rel_err, rel);
    }

    // Tolerance: FP16 accumulation can have significant error for large vectors
    float tolerance = 1e-2f;
    bool pass = max_rel_err < tolerance || max_err < tolerance;

    printf("  Max absolute error: %e (row %d)\n", max_err, worst_row);
    printf("  Max relative error: %e\n", max_rel_err);
    printf("  GPU[%d] = %f, Ref[%d] = %f\n",
           worst_row, __half2float(h_output[worst_row]),
           worst_row, h_ref[worst_row]);
    printf("  Result: %s\n\n", pass ? "PASS" : "FAIL");

    // Cleanup
    free(h_nz); free(h_sg); free(h_scales); free(h_scales_f);
    free(h_input); free(h_input_f); free(h_output); free(h_ref);
    CUDA_CHECK(cudaFree(d_nz));
    CUDA_CHECK(cudaFree(d_sg));
    CUDA_CHECK(cudaFree(d_scales));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return pass;
}

int main() {
    srand((unsigned int)time(nullptr));

    printf("=== Ternary GEMV Correctness Tests ===\n\n");

    int group_size = 128;

    struct TestCase {
        int rows, cols;
    };

    TestCase tests[] = {
        {1024,  1024},
        {4096,  4096},
        {4096, 11008},  // Note: 11008 is not divisible by 128 cleanly
                        // 11008 / 128 = 86, so it works
    };

    // Verify 11008 is divisible by group_size and 32
    if (11008 % group_size != 0 || 11008 % 32 != 0) {
        // Adjust to nearest valid size
        printf("Warning: 11008 not aligned, adjusting to 11008\n");
        // 11008 = 86 * 128 = 344 * 32, so it's fine
    }

    int total = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;

    for (int i = 0; i < total; i++) {
        if (run_test(tests[i].rows, tests[i].cols, group_size)) {
            passed++;
        }
    }

    printf("=== Summary: %d / %d tests passed ===\n", passed, total);

    return (passed == total) ? 0 : 1;
}
