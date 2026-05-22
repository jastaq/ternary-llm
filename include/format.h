#pragma once
#include <cstdint>
#include <cstring>

// ---------------------------------------------------------------------------
// .tllm binary format constants
// ---------------------------------------------------------------------------
constexpr uint32_t TLLM_MAGIC       = 0x544C4C4D;  // "TLLM"
constexpr uint32_t TLLM_VERSION     = 1;
constexpr size_t   TLLM_HEADER_SIZE = 256;          // fixed header size (bytes)

// Architecture IDs
constexpr uint32_t TLLM_ARCH_LLAMA   = 0;
constexpr uint32_t TLLM_ARCH_MISTRAL = 1;

// ---------------------------------------------------------------------------
// On-disk header — exactly 256 bytes, zero-padded.
// Fields after rms_norm_eps_bits are reserved for future use.
// ---------------------------------------------------------------------------
struct TLLMHeader {
    uint32_t magic;                  // Must be TLLM_MAGIC
    uint32_t version;                // Must be TLLM_VERSION
    uint32_t arch;                   // TLLM_ARCH_*
    uint32_t vocab_size;
    uint32_t hidden_dim;
    uint32_t intermediate_dim;
    uint32_t n_layers;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t max_seq_len;
    uint32_t head_dim;
    uint32_t rope_theta_bits;        // float reinterpreted as uint32
    uint32_t group_size;
    uint32_t rms_norm_eps_bits;      // float reinterpreted as uint32
    uint8_t  reserved[TLLM_HEADER_SIZE - 14 * sizeof(uint32_t)];
};

static_assert(sizeof(TLLMHeader) == TLLM_HEADER_SIZE,
              "TLLMHeader must be exactly 256 bytes");

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
inline float bits_to_float(uint32_t bits) {
    float f;
    std::memcpy(&f, &bits, sizeof(float));
    return f;
}

inline uint32_t float_to_bits(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(uint32_t));
    return bits;
}
