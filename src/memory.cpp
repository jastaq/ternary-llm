#include "memory.h"
#include <cuda_runtime.h>
#include <cstdio>

// ---------------------------------------------------------------------------
// MemoryPool implementation
// ---------------------------------------------------------------------------

MemoryPool::~MemoryPool() {
    reset();
}

void* MemoryPool::allocate(size_t bytes) {
    if (bytes == 0) return nullptr;

    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "[MemoryPool] cudaMalloc failed for %zu bytes: %s\n",
                bytes, cudaGetErrorString(err));
        return nullptr;
    }

    blocks_.push_back({ptr, bytes});
    total_ += bytes;
    return ptr;
}

void MemoryPool::reset() {
    for (auto& b : blocks_) {
        if (b.ptr) cudaFree(b.ptr);
    }
    blocks_.clear();
    total_ = 0;
}
