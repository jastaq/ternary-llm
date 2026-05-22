#include "sampler.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <cstdio>

// ---------------------------------------------------------------------------
// Portable host-side half → float conversion (no CUDA device intrinsics)
// ---------------------------------------------------------------------------
static float half_to_float(uint16_t bits) {
    unsigned sign = (bits >> 15) & 1;
    unsigned exp  = (bits >> 10) & 0x1F;
    unsigned mant = bits & 0x3FF;

    if (exp == 0) {
        // Subnormal or zero
        if (mant == 0) return sign ? -0.0f : 0.0f;
        return (sign ? -1.0f : 1.0f) * std::ldexp(static_cast<float>(mant), -24);
    }
    if (exp == 31) {
        return mant ? NAN : (sign ? -INFINITY : INFINITY);
    }
    return (sign ? -1.0f : 1.0f) *
           std::ldexp(static_cast<float>(mant + 1024), exp - 25);
}

// ---------------------------------------------------------------------------
// Sampler implementation
// ---------------------------------------------------------------------------

Sampler::Sampler(const SamplerConfig& config, int vocab_size)
    : config_(config),
      vocab_size_(vocab_size),
      logits_(vocab_size),
      rng_(config.seed) {}

int Sampler::sample(const half* logits_gpu) {
    // 1. Copy raw FP16 logits from GPU → host buffer
    //    logits_ is vector<float> but we need raw half bytes first.
    std::vector<uint16_t> raw(vocab_size_);
    cudaMemcpy(raw.data(), logits_gpu,
               vocab_size_ * sizeof(uint16_t), cudaMemcpyDeviceToHost);

    // 2. Convert FP16 → FP32 (host-side, no CUDA intrinsics)
    for (int i = 0; i < vocab_size_; ++i) {
        logits_[i] = half_to_float(raw[i]);
    }

    // 3. Apply repetition penalty
    if (config_.repetition_penalty != 1.0f) {
        for (int tok : history_) {
            if (tok >= 0 && tok < vocab_size_) {
                if (logits_[tok] > 0.0f)
                    logits_[tok] /= config_.repetition_penalty;
                else
                    logits_[tok] *= config_.repetition_penalty;
            }
        }
    }

    // 4. Apply temperature
    if (config_.temperature > 0.0f && config_.temperature != 1.0f) {
        float inv_t = 1.0f / config_.temperature;
        for (int i = 0; i < vocab_size_; ++i)
            logits_[i] *= inv_t;
    }

    // 5. Build index array sorted by descending logit
    std::vector<int> indices(vocab_size_);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(),
                      indices.begin() + std::min(config_.top_k, vocab_size_),
                      indices.end(),
                      [&](int a, int b) {
                          return logits_[a] > logits_[b];
                      });

    // 6. Top-K truncation
    int k = std::min(config_.top_k, vocab_size_);
    indices.resize(k);

    // 7. Softmax over top-k candidates
    float max_logit = logits_[indices[0]];
    std::vector<float> probs(k);
    float sum = 0.0f;
    for (int i = 0; i < k; ++i) {
        probs[i] = std::exp(logits_[indices[i]] - max_logit);
        sum += probs[i];
    }
    for (int i = 0; i < k; ++i) probs[i] /= sum;

    // 8. Top-P (nucleus) truncation
    float cumulative = 0.0f;
    int cutoff = k;
    for (int i = 0; i < k; ++i) {
        cumulative += probs[i];
        if (cumulative >= config_.top_p) {
            cutoff = i + 1;
            break;
        }
    }
    probs.resize(cutoff);
    indices.resize(cutoff);

    // Re-normalise after top-p cut
    sum = 0.0f;
    for (float p : probs) sum += p;
    for (float& p : probs) p /= sum;

    // 9. Sample from the distribution
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    int sampled_idx = dist(rng_);
    int token_id = indices[sampled_idx];

    return token_id;
}

void Sampler::add_to_history(int token_id) {
    history_.push_back(token_id);
    // Keep history bounded to avoid unbounded growth
    if (history_.size() > 256) {
        history_.erase(history_.begin(),
                       history_.begin() + static_cast<long>(history_.size() - 256));
    }
}

void Sampler::reset_history() {
    history_.clear();
}
