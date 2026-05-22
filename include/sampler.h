#pragma once
#include <cuda_fp16.h>
#include <cstdint>
#include <random>
#include <vector>

// ---------------------------------------------------------------------------
// SamplerConfig — tuneable generation knobs.
// ---------------------------------------------------------------------------
struct SamplerConfig {
    float    temperature       = 0.7f;
    float    top_p             = 0.9f;
    int      top_k             = 40;
    float    repetition_penalty = 1.1f;
    uint64_t seed              = 42;
};

// ---------------------------------------------------------------------------
// Sampler — picks the next token from a logits vector on the GPU.
//
// Internally copies logits D→H, applies penalties / temperature / top-k /
// top-p filtering, then samples from the resulting distribution.
// ---------------------------------------------------------------------------
class Sampler {
public:
    Sampler(const SamplerConfig& config, int vocab_size);

    // Sample one token from device-side logits [vocab_size].
    int sample(const half* logits_gpu);

    // Track generated tokens for repetition penalty.
    void add_to_history(int token_id);
    void reset_history();

private:
    SamplerConfig       config_;
    int                 vocab_size_;
    std::vector<float>  logits_;     // host staging buffer
    std::vector<int>    history_;    // recently generated tokens
    std::mt19937        rng_;
};
