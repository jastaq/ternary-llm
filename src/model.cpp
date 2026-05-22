#include "model.h"
#include "format.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
namespace {

// Read `bytes` from file into a freshly-allocated GPU buffer.
// Uses a host-side staging buffer to avoid pinned-memory complexity.
void* read_to_gpu(FILE* f, size_t bytes) {
    if (bytes == 0) return nullptr;

    std::vector<uint8_t> staging(bytes);
    size_t n = fread(staging.data(), 1, bytes, f);
    if (n != bytes) {
        fprintf(stderr, "[Model] Short read: expected %zu, got %zu\n", bytes, n);
        return nullptr;
    }

    void* gpu_ptr = nullptr;
    cudaError_t err = cudaMalloc(&gpu_ptr, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "[Model] cudaMalloc(%zu) failed: %s\n",
                bytes, cudaGetErrorString(err));
        return nullptr;
    }

    err = cudaMemcpy(gpu_ptr, staging.data(), bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "[Model] cudaMemcpy failed: %s\n",
                cudaGetErrorString(err));
        cudaFree(gpu_ptr);
        return nullptr;
    }

    return gpu_ptr;
}

// Read a ternary weight block from file.
TernaryWeight read_ternary_weight(FILE* f, int out_features, int in_features,
                                  int group_size) {
    TernaryWeight w;
    w.out_features = out_features;
    w.in_features  = in_features;   // already padded to multiple of 32

    size_t mask_bytes  = w.mask_bytes();
    size_t scale_bytes = w.scale_bytes(group_size);

    w.nonzero_masks = reinterpret_cast<uint32_t*>(read_to_gpu(f, mask_bytes));
    w.sign_masks    = reinterpret_cast<uint32_t*>(read_to_gpu(f, mask_bytes));
    w.scales        = reinterpret_cast<half*>(read_to_gpu(f, scale_bytes));

    return w;
}

// Pad a dimension up to the nearest multiple of 32.
int pad32(int dim) {
    return ((dim + 31) / 32) * 32;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Model::load
// ---------------------------------------------------------------------------
Model Model::load(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "[Model] Cannot open %s\n", path.c_str());
        std::exit(1);
    }

    // ---- Header ----
    TLLMHeader header;
    if (fread(&header, sizeof(TLLMHeader), 1, f) != 1) {
        fprintf(stderr, "[Model] Failed to read header\n");
        fclose(f);
        std::exit(1);
    }
    if (header.magic != TLLM_MAGIC) {
        fprintf(stderr, "[Model] Bad magic: 0x%08X (expected 0x%08X)\n",
                header.magic, TLLM_MAGIC);
        fclose(f);
        std::exit(1);
    }
    if (header.version != TLLM_VERSION) {
        fprintf(stderr, "[Model] Unsupported version: %u\n", header.version);
        fclose(f);
        std::exit(1);
    }

    Model model;
    ModelConfig& c = model.config;
    c.vocab_size       = header.vocab_size;
    c.hidden_dim       = header.hidden_dim;
    c.intermediate_dim = header.intermediate_dim;
    c.n_layers         = header.n_layers;
    c.n_heads          = header.n_heads;
    c.n_kv_heads       = header.n_kv_heads;
    c.max_seq_len      = header.max_seq_len;
    c.head_dim         = header.head_dim;
    c.rope_theta       = bits_to_float(header.rope_theta_bits);
    c.group_size       = header.group_size;
    c.rms_norm_eps     = bits_to_float(header.rms_norm_eps_bits);

    model.lm_head_is_ternary = (header.flags & TLLM_FLAG_LM_HEAD_TERNARY) != 0;

    fprintf(stderr, "[Model] %s: vocab=%u  hidden=%u  layers=%u  heads=%u/%u  "
            "max_seq=%u  group=%u  lm_head=%s\n",
            path.c_str(), c.vocab_size, c.hidden_dim, c.n_layers,
            c.n_heads, c.n_kv_heads, c.max_seq_len, c.group_size,
            model.lm_head_is_ternary ? "ternary" : "fp16");

    // Precompute padded dimensions
    int hidden_pad       = pad32(c.hidden_dim);
    int q_dim            = c.n_heads * c.head_dim;
    int kv_dim           = c.n_kv_heads * c.head_dim;
    int q_dim_pad        = pad32(q_dim);
    int kv_dim_pad       = pad32(kv_dim);
    int inter_pad        = pad32(c.intermediate_dim);
    int gs               = static_cast<int>(c.group_size);

    // ---- Embedding table (FP16) ----
    size_t embed_bytes = static_cast<size_t>(c.vocab_size) * c.hidden_dim * sizeof(half);
    model.embedding_table = reinterpret_cast<half*>(read_to_gpu(f, embed_bytes));
    fprintf(stderr, "[Model]   embedding: %.1f MB\n", embed_bytes / 1e6);

    // ---- Layers ----
    model.layers.resize(c.n_layers);
    for (uint32_t i = 0; i < c.n_layers; ++i) {
        TransformerLayer& layer = model.layers[i];

        // Attention projections
        //   q_proj : [q_dim, hidden_dim]
        //   k_proj : [kv_dim, hidden_dim]
        //   v_proj : [kv_dim, hidden_dim]
        //   o_proj : [hidden_dim, q_dim]
        layer.q_proj    = read_ternary_weight(f, q_dim,       hidden_pad,  gs);
        layer.k_proj    = read_ternary_weight(f, kv_dim,      hidden_pad,  gs);
        layer.v_proj    = read_ternary_weight(f, kv_dim,      hidden_pad,  gs);
        layer.o_proj    = read_ternary_weight(f, c.hidden_dim, q_dim_pad,  gs);

        // FFN / SwiGLU projections
        //   gate_proj : [intermediate_dim, hidden_dim]
        //   up_proj   : [intermediate_dim, hidden_dim]
        //   down_proj : [hidden_dim, intermediate_dim]
        layer.gate_proj = read_ternary_weight(f, c.intermediate_dim, hidden_pad, gs);
        layer.up_proj   = read_ternary_weight(f, c.intermediate_dim, hidden_pad, gs);
        layer.down_proj = read_ternary_weight(f, c.hidden_dim, inter_pad, gs);

        // Norm weights (FP16, not ternarised)
        size_t norm_bytes = c.hidden_dim * sizeof(half);
        layer.attn_norm_weight = reinterpret_cast<half*>(read_to_gpu(f, norm_bytes));
        layer.ffn_norm_weight  = reinterpret_cast<half*>(read_to_gpu(f, norm_bytes));

        if ((i + 1) % 8 == 0 || i == c.n_layers - 1)
            fprintf(stderr, "[Model]   layers 0-%u loaded\n", i);
    }

    // ---- Final norm ----
    size_t norm_bytes = c.hidden_dim * sizeof(half);
    model.final_norm_weight = reinterpret_cast<half*>(read_to_gpu(f, norm_bytes));

    // ---- lm_head ----
    if (model.lm_head_is_ternary) {
        model.lm_head_ternary = read_ternary_weight(f, c.vocab_size, hidden_pad, gs);
        fprintf(stderr, "[Model]   lm_head (ternary): %.1f MB\n",
                model.lm_head_ternary.total_bytes(gs) / 1e6);
    } else {
        size_t head_bytes = static_cast<size_t>(c.vocab_size) * c.hidden_dim * sizeof(half);
        model.lm_head_weight = reinterpret_cast<half*>(read_to_gpu(f, head_bytes));
        fprintf(stderr, "[Model]   lm_head (fp16): %.1f MB\n", head_bytes / 1e6);
    }

    // ---- Tokenizer blob ----
    uint64_t tok_size = 0;
    if (fread(&tok_size, sizeof(uint64_t), 1, f) == 1 && tok_size > 0) {
        model.tokenizer_data.resize(tok_size);
        size_t n = fread(model.tokenizer_data.data(), 1, tok_size, f);
        if (n != tok_size) {
            fprintf(stderr, "[Model] Warning: tokenizer short read %zu/%lu\n",
                    n, static_cast<unsigned long>(tok_size));
        }
        fprintf(stderr, "[Model]   tokenizer: %lu bytes\n",
                static_cast<unsigned long>(tok_size));
    }

    fclose(f);

    fprintf(stderr, "[Model] Total GPU memory: %.1f MB\n",
            model.gpu_memory_bytes() / 1e6);

    return model;
}

// ---------------------------------------------------------------------------
// Model::free_gpu
// ---------------------------------------------------------------------------
static void safe_free(void* ptr) {
    if (ptr) cudaFree(ptr);
}

static void free_ternary(TernaryWeight& w) {
    safe_free(w.nonzero_masks);
    safe_free(w.sign_masks);
    safe_free(w.scales);
    w.nonzero_masks = nullptr;
    w.sign_masks    = nullptr;
    w.scales        = nullptr;
}

void Model::free_gpu() {
    safe_free(embedding_table);  embedding_table = nullptr;
    for (auto& l : layers) {
        free_ternary(l.q_proj);
        free_ternary(l.k_proj);
        free_ternary(l.v_proj);
        free_ternary(l.o_proj);
        free_ternary(l.gate_proj);
        free_ternary(l.up_proj);
        free_ternary(l.down_proj);
        safe_free(l.attn_norm_weight);  l.attn_norm_weight = nullptr;
        safe_free(l.ffn_norm_weight);   l.ffn_norm_weight  = nullptr;
    }
    layers.clear();
    safe_free(final_norm_weight); final_norm_weight = nullptr;
    if (lm_head_is_ternary) {
        free_ternary(lm_head_ternary);
    } else {
        safe_free(lm_head_weight); lm_head_weight = nullptr;
    }
}

// ---------------------------------------------------------------------------
// Model::gpu_memory_bytes
// ---------------------------------------------------------------------------
size_t Model::gpu_memory_bytes() const {
    size_t total = 0;
    int gs = static_cast<int>(config.group_size);

    // Embedding
    total += static_cast<size_t>(config.vocab_size) * config.hidden_dim * sizeof(half);

    // Layers
    for (const auto& l : layers) {
        auto add_tw = [&](const TernaryWeight& w) {
            total += 2 * w.mask_bytes() + w.scale_bytes(gs);
        };
        add_tw(l.q_proj);  add_tw(l.k_proj);  add_tw(l.v_proj);  add_tw(l.o_proj);
        add_tw(l.gate_proj); add_tw(l.up_proj); add_tw(l.down_proj);
        total += 2 * config.hidden_dim * sizeof(half);  // norms
    }

    // Final norm + lm_head
    total += config.hidden_dim * sizeof(half);
    if (lm_head_is_ternary) {
        total += lm_head_ternary.total_bytes(gs);
    } else {
        total += static_cast<size_t>(config.vocab_size) * config.hidden_dim * sizeof(half);
    }

    return total;
}
