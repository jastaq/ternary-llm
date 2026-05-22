#include "tokenizer.h"
#include <sentencepiece_processor.h>
#include <cstdio>

// ---------------------------------------------------------------------------
// PImpl for SentencePiece
// ---------------------------------------------------------------------------
struct Tokenizer::Impl {
    sentencepiece::SentencePieceProcessor sp;
};

Tokenizer::Tokenizer() : impl_(std::make_unique<Impl>()) {}
Tokenizer::~Tokenizer() = default;

bool Tokenizer::load(const std::string& model_path) {
    auto status = impl_->sp.Load(model_path);
    if (!status.ok()) {
        fprintf(stderr, "[Tokenizer] Failed to load %s: %s\n",
                model_path.c_str(), status.ToString().c_str());
        return false;
    }
    return true;
}

bool Tokenizer::load_from_memory(const void* data, size_t size) {
    // SentencePiece accepts a serialised ModelProto via LoadFromSerializedProto
    auto sv = absl::string_view(reinterpret_cast<const char*>(data), size);
    auto status = impl_->sp.LoadFromSerializedProto(sv);
    if (!status.ok()) {
        fprintf(stderr, "[Tokenizer] Failed to load from memory: %s\n",
                status.ToString().c_str());
        return false;
    }
    return true;
}

std::vector<int> Tokenizer::encode(const std::string& text, bool add_bos) const {
    std::vector<int> ids;
    auto status = impl_->sp.Encode(text, &ids);
    if (!status.ok()) {
        fprintf(stderr, "[Tokenizer] Encode failed: %s\n",
                status.ToString().c_str());
        return {};
    }
    if (add_bos) {
        ids.insert(ids.begin(), bos_id());
    }
    return ids;
}

std::string Tokenizer::decode(int token_id) const {
    return impl_->sp.IdToPiece(token_id);
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    std::string text;
    auto status = impl_->sp.Decode(ids, &text);
    if (!status.ok()) {
        fprintf(stderr, "[Tokenizer] Decode failed: %s\n",
                status.ToString().c_str());
        return "";
    }
    return text;
}

int Tokenizer::bos_id() const    { return impl_->sp.bos_id(); }
int Tokenizer::eos_id() const    { return impl_->sp.eos_id(); }
int Tokenizer::vocab_size() const { return impl_->sp.GetPieceSize(); }
