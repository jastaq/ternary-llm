#pragma once
#include <cstddef>
#include <vector>

// ---------------------------------------------------------------------------
// MemoryPool — arena-style GPU allocator.
//
// Every allocation is recorded; reset() frees them all at once.
// This avoids per-kernel cudaMalloc overhead during inference.
// ---------------------------------------------------------------------------
class MemoryPool {
public:
    MemoryPool() = default;
    ~MemoryPool();

    MemoryPool(const MemoryPool&)            = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;

    // Allocate `bytes` of device memory.  Returns nullptr on failure.
    void* allocate(size_t bytes);

    // Free every allocation made through this pool.
    void reset();

    size_t total_allocated() const { return total_; }

private:
    struct Block { void* ptr; size_t size; };
    std::vector<Block> blocks_;
    size_t total_ = 0;
};
