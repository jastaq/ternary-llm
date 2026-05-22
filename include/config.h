#pragma once
#include <cstdint>
#include <string>

// ---------------------------------------------------------------------------
// ModelConfig — holds all architectural hyper-parameters for the loaded model.
// Populated directly from the .tllm binary header written by the Python
// ternarizer.  All dimension fields refer to the *padded* (multiple-of-32)
// values for ternary weights.
// ---------------------------------------------------------------------------
struct ModelConfig {
    uint32_t vocab_size        = 0;
    uint32_t hidden_dim        = 0;
    uint32_t intermediate_dim  = 0;
    uint32_t n_layers          = 0;
    uint32_t n_heads           = 0;
    uint32_t n_kv_heads        = 0;
    uint32_t max_seq_len       = 0;
    uint32_t head_dim          = 0;
    float    rope_theta        = 10000.0f;
    uint32_t group_size        = 128;
    float    rms_norm_eps      = 1e-5f;

    // Derived helpers
    uint32_t kv_dim()  const { return n_kv_heads * head_dim; }
    uint32_t q_dim()   const { return n_heads * head_dim;    }
    uint32_t head_groups() const {
        return (n_kv_heads > 0) ? (n_heads / n_kv_heads) : 1;
    }
};
