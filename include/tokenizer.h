#pragma once
#include <memory>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Tokenizer — thin wrapper around SentencePiece for BPE encode / decode.
// ---------------------------------------------------------------------------
class Tokenizer {
public:
    Tokenizer();
    ~Tokenizer();

    // Load from a .model file on disk.
    bool load(const std::string& model_path);

    // Load from an in-memory serialised protobuf (embedded in .tllm).
    bool load_from_memory(const void* data, size_t size);

    // Encode text to token IDs.
    std::vector<int> encode(const std::string& text, bool add_bos = true) const;

    // Decode a single token ID to its string piece.
    std::string decode(int token_id) const;

    // Decode a sequence of token IDs.
    std::string decode(const std::vector<int>& ids) const;

    int bos_id()    const;
    int eos_id()    const;
    int vocab_size() const;

private:
    struct Impl;                   // PImpl — keeps sentencepiece out of header
    std::unique_ptr<Impl> impl_;
};
