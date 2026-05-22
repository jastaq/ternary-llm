/*
 * Benchmark program for ternary_gemv vs fp16_gemv
 *
 * 1. Benchmark ternary_gemv at multiple sizes
 * 2. Run 100 iterations, measure average time
 * 3. Compute effective bandwidth
 * 4. Compare with fp16_gemv
 * 5. Print results as a table
 */

#include "../include/kernels.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

struct BenchmarkSize {
    int rows;
    int cols;
};

void benchmark_ternary_gemv(
    int rows, int cols, int group_size,
    int warmup_iters, int bench_iters,
    float& avg_ms, float& bandwidth_gbs)
{
    int masks_per_row = cols / 32;
    int groups_per_row = cols / group_size;

    size_t nz_bytes = (size_t)rows * masks_per_row * sizeof(uint32_t);
    size_t sg_bytes = (size_t)rows * masks_per_row * sizeof(uint32_t);
    size_t scales_bytes = (size_t)rows * groups_per_row * sizeof(half);
    size_t input_bytes = (size_t)cols * sizeof(half);
    size_t output_bytes = (size_t)rows * sizeof(half);

    uint32_t *d_nz, *d_sg;
    half *d_scales, *d_input, *d_output;

    CUDA_CHECK(cudaMalloc(&d_nz, nz_bytes));
    CUDA_CHECK(cudaMalloc(&d_sg, sg_bytes));
    CUDA_CHECK(cudaMalloc(&d_scales, scales_bytes));
    CUDA_CHECK(cudaMalloc(&d_input, input_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, output_bytes));

    // Initialize with zeros (content doesn't matter for benchmarking)
    CUDA_CHECK(cudaMemset(d_nz, 0, nz_bytes));
    CUDA_CHECK(cudaMemset(d_sg, 0, sg_bytes));
    CUDA_CHECK(cudaMemset(d_scales, 0, scales_bytes));
    CUDA_CHECK(cudaMemset(d_input, 0, input_bytes));

    // Warmup
    for (int i = 0; i < warmup_iters; i++) {
        ternary::ternary_gemv(d_nz, d_sg, d_scales, d_input, d_output,
                              rows, cols, group_size, 0);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < bench_iters; i++) {
        ternary::ternary_gemv(d_nz, d_sg, d_scales, d_input, d_output,
                              rows, cols, group_size, 0);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    avg_ms = total_ms / bench_iters;

    // Effective bytes read:
    // masks: rows * (cols/32) * 4 * 2 (nz + sign)
    // scales: rows * (cols/group_size) * 2
    // input: cols * 2
    // output (write): rows * 2
    size_t bytes_read = nz_bytes + sg_bytes + scales_bytes + input_bytes;
    size_t bytes_written = output_bytes;
    size_t total_bytes = bytes_read + bytes_written;
    bandwidth_gbs = (float)total_bytes / (avg_ms * 1e-3f) / 1e9f;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_nz));
    CUDA_CHECK(cudaFree(d_sg));
    CUDA_CHECK(cudaFree(d_scales));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

void benchmark_fp16_gemv(
    int rows, int cols,
    int warmup_iters, int bench_iters,
    float& avg_ms, float& bandwidth_gbs)
{
    size_t weight_bytes = (size_t)rows * cols * sizeof(half);
    size_t input_bytes = (size_t)cols * sizeof(half);
    size_t output_bytes = (size_t)rows * sizeof(half);

    half *d_weight, *d_input, *d_output;

    CUDA_CHECK(cudaMalloc(&d_weight, weight_bytes));
    CUDA_CHECK(cudaMalloc(&d_input, input_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, output_bytes));

    CUDA_CHECK(cudaMemset(d_weight, 0, weight_bytes));
    CUDA_CHECK(cudaMemset(d_input, 0, input_bytes));

    // Warmup
    for (int i = 0; i < warmup_iters; i++) {
        ternary::fp16_gemv(d_weight, d_input, d_output, rows, cols, 0);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < bench_iters; i++) {
        ternary::fp16_gemv(d_weight, d_input, d_output, rows, cols, 0);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    avg_ms = total_ms / bench_iters;

    size_t total_bytes = weight_bytes + input_bytes + output_bytes;
    bandwidth_gbs = (float)total_bytes / (avg_ms * 1e-3f) / 1e9f;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_weight));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

int main() {
    // Print GPU info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s\n", prop.name);
    printf("Memory Bandwidth: %.1f GB/s\n", prop.memoryBusWidth * prop.memoryClockRate * 2.0 / 1e6);
    printf("SM Count: %d\n\n", prop.multiProcessorCount);

    BenchmarkSize sizes[] = {
        {4096,  4096},
        {4096, 11008},
        {11008, 4096},
        {8192,  8192},
    };
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    int group_size = 128;
    int warmup_iters = 10;
    int bench_iters = 100;

    // Print header
    printf("%-20s | %-14s | %-14s | %-14s | %-14s | %-10s\n",
           "Size (rows x cols)", "Ternary (ms)", "Ternary BW", "FP16 (ms)", "FP16 BW", "Speedup");
    printf("%-20s-+-%-14s-+-%-14s-+-%-14s-+-%-14s-+-%-10s\n",
           "--------------------", "--------------", "--------------",
           "--------------", "--------------", "----------");

    for (int i = 0; i < num_sizes; i++) {
        int rows = sizes[i].rows;
        int cols = sizes[i].cols;

        // Verify alignment
        if (cols % 32 != 0 || cols % group_size != 0) {
            printf("Skipping %d x %d: not aligned to group_size=%d and 32\n",
                   rows, cols, group_size);
            continue;
        }

        float ternary_ms, ternary_bw;
        benchmark_ternary_gemv(rows, cols, group_size,
                               warmup_iters, bench_iters,
                               ternary_ms, ternary_bw);

        float fp16_ms, fp16_bw;
        benchmark_fp16_gemv(rows, cols,
                            warmup_iters, bench_iters,
                            fp16_ms, fp16_bw);

        float speedup = fp16_ms / ternary_ms;

        char size_str[32];
        snprintf(size_str, sizeof(size_str), "%d x %d", rows, cols);

        printf("%-20s | %10.3f ms  | %10.1f GB/s | %10.3f ms  | %10.1f GB/s | %8.2fx\n",
               size_str, ternary_ms, ternary_bw, fp16_ms, fp16_bw, speedup);
    }

    printf("\n");

    // Memory savings summary
    printf("=== Memory Savings ===\n");
    for (int i = 0; i < num_sizes; i++) {
        int rows = sizes[i].rows;
        int cols = sizes[i].cols;

        size_t fp16_bytes = (size_t)rows * cols * 2;  // FP16 weights
        size_t ternary_bytes = (size_t)rows * (cols / 32) * 4 * 2  // nz + sign masks
                             + (size_t)rows * (cols / group_size) * 2; // scales

        float compression = (float)fp16_bytes / ternary_bytes;

        printf("  %5d x %5d: FP16 = %8.2f MB, Ternary = %8.2f MB, Compression = %.1fx\n",
               rows, cols,
               fp16_bytes / (1024.0f * 1024.0f),
               ternary_bytes / (1024.0f * 1024.0f),
               compression);
    }

    return 0;
}
