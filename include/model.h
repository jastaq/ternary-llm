#pragma once
#include "config.h"
#include <cuda_fp16.h>
#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// TernaryWeight — one linear-layer's packed ternary weights on the GPU.
//
// Storage layout (per row of the weight matrix):
//   nonzero_masks : uint32[in_features/32]   — bit is 1 where w != 0
//   sign_masks    : uint32[in_features/32]   — bit is 1 where w  > 0
//   scales        : half[in_features/group]  — per-group scale factor
// ---------------------------------------------------------------------------
struct TernaryWeight {
    uint32_t* nonzero_masks = nullptr;  // GPU [out_features, in_features/32]
    uint32_t* sign_masks    = nullptr;  // GPU [out_features, in_features/32]
    half*     scales        = nullptr;  // GPU [out_features, in_features/group_size]
    int       out_features  = 0;
    int       in_features   = 0;        // already padded to multiple of 32

    size_t mask_bytes() const {
        return static_cast<size_t>(out_features) * (in_features / 32) * sizeof(uint32_t);
    }
    size_t scale_elements(int group_size) const {
        return static_cast<size_t>(out_features) * (in_features / group_size);
    }
    size_t scale_bytes(int group_size) const {
        return scale_elements(group_size) * sizeof(half);
    }
    size_t total_bytes(int group_size) const {
        return 2 * mask_bytes() + scale_bytes(group_size);
    }
};

// ---------------------------------------------------------------------------
// TransformerLayer — all weights for one decoder layer.
// ---------------------------------------------------------------------------
struct TransformerLayer {
    // Attention projections (ternary)
    TernaryWeight q_proj;
    TernaryWeight k_proj;
    TernaryWeight v_proj;
    TernaryWeight o_proj;

    // FFN / SwiGLU projections (ternary)
    TernaryWeight gate_proj;
    TernaryWeight up_proj;
    TernaryWeight down_proj;

    // Norm weights (FP16, NOT ternarized)
    half* attn_norm_weight = nullptr;   // GPU [hidden_dim]
    half* ffn_norm_weight  = nullptr;   // GPU [hidden_dim]
};

// ---------------------------------------------------------------------------
// Model — complete model state living on the GPU.
// ---------------------------------------------------------------------------
struct Model {
    ModelConfig                config;
    half*                      embedding_table  = nullptr;  // GPU [vocab_size, hidden_dim]
    std::vector<TransformerLayer> layers;
    half*                      final_norm_weight = nullptr; // GPU [hidden_dim]

    // lm_head can be FP16 or ternary (controlled by header flag)
    bool                       lm_head_is_ternary = false;
    half*                      lm_head_weight    = nullptr; // GPU [vocab, hidden] (FP16 path)
    TernaryWeight              lm_head_ternary;             // (ternary path)

    // Raw tokenizer data extracted from .tllm file
    std::vector<uint8_t> tokenizer_data;

    // Load a model from a .tllm file.  Allocates all GPU memory and copies
    // weights to the device.
    static Model load(const std::string& path);

    // Free all GPU memory held by this model.
    void free_gpu();

    // Total GPU memory consumed by weights (approximate).
    size_t gpu_memory_bytes() const;
};
