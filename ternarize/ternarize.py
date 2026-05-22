#!/usr/bin/env python3
"""
ternarize.py – Convert a HuggingFace LLaMA / Mistral model to the .tllm
ternary-packed binary format.

Usage
-----
    python ternarize.py \
        --model meta-llama/Llama-2-7b-hf \
        --output llama2-7b.tllm \
        --group-size 128 \
        --threshold-factor 0.7 \
        --device cuda
"""

import argparse
import os
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from pack import pad_to_multiple, pack_ternary_to_bitmasks, ternarize_tensor

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

TLLM_MAGIC = 0x544C4C4D
TLLM_VERSION = 1
HEADER_SIZE = 256

ARCH_LLAMA = 0

LINEAR_WEIGHT_NAMES = [
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
]


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _float_to_uint32_bits(f: float) -> int:
    """Reinterpret the bits of a 32-bit float as a uint32."""
    return struct.unpack("I", struct.pack("f", f))[0]


def _write_raw(fout, array: np.ndarray) -> int:
    """Write a numpy array to *fout* and return bytes written."""
    data = array.tobytes()
    fout.write(data)
    return len(data)


# ──────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────

def write_header(
    fout,
    vocab_size: int,
    hidden_dim: int,
    intermediate_dim: int,
    n_layers: int,
    n_heads: int,
    n_kv_heads: int,
    max_seq_len: int,
    head_dim: int,
    rope_theta: float,
    group_size: int,
    rms_norm_eps: float,
):
    """Write the 256-byte .tllm header."""
    header_fields = struct.pack(
        "<" + "I" * 13,
        TLLM_MAGIC,
        TLLM_VERSION,
        ARCH_LLAMA,
        vocab_size,
        hidden_dim,
        intermediate_dim,
        n_layers,
        n_heads,
        n_kv_heads,
        max_seq_len,
        head_dim,
        _float_to_uint32_bits(rope_theta),
        group_size,
    )
    # rms_norm_eps as float-bits uint32
    header_fields += struct.pack("<I", _float_to_uint32_bits(rms_norm_eps))

    assert len(header_fields) <= HEADER_SIZE
    # Zero-pad to HEADER_SIZE
    header_fields += b"\x00" * (HEADER_SIZE - len(header_fields))
    fout.write(header_fields)


# ──────────────────────────────────────────────────────────────────────
# Tokenizer
# ──────────────────────────────────────────────────────────────────────

def find_tokenizer_model(model_name_or_path: str) -> bytes:
    """Return the raw bytes of the SentencePiece tokenizer.model file."""
    # 1. Try the local model directory first
    model_dir = Path(model_name_or_path)
    if model_dir.is_dir():
        sp_path = model_dir / "tokenizer.model"
        if sp_path.exists():
            return sp_path.read_bytes()

    # 2. Try the HuggingFace cache
    from transformers.utils import TRANSFORMERS_CACHE
    try:
        from huggingface_hub import hf_hub_download
        sp_path = hf_hub_download(
            repo_id=model_name_or_path,
            filename="tokenizer.model",
        )
        return Path(sp_path).read_bytes()
    except Exception:
        pass

    # 3. Try loading via the tokenizer object itself (SentencePiece backend)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if hasattr(tokenizer, "sp_model"):
            return tokenizer.sp_model.serialized_model_proto()
    except Exception:
        pass

    print("[WARN] Could not find tokenizer.model – writing 0-length tokenizer blob")
    return b""


# ──────────────────────────────────────────────────────────────────────
# Main conversion
# ──────────────────────────────────────────────────────────────────────

def convert(
    model_name: str,
    output_path: str,
    group_size: int = 128,
    threshold_factor: float = 0.7,
    device: str = "cuda",
):
    print(f"[*] Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    model.eval()
    state = model.state_dict()

    config = model.config
    vocab_size = config.vocab_size
    hidden_dim = config.hidden_size
    intermediate_dim = config.intermediate_size
    n_layers = config.num_hidden_layers
    n_heads = config.num_attention_heads
    n_kv_heads = getattr(config, "num_key_value_heads", n_heads)
    max_seq_len = getattr(config, "max_position_embeddings", 4096)
    rope_theta = float(getattr(config, "rope_theta", 10000.0))
    rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-5))
    head_dim = hidden_dim // n_heads

    print(f"    vocab_size       = {vocab_size}")
    print(f"    hidden_dim       = {hidden_dim}")
    print(f"    intermediate_dim = {intermediate_dim}")
    print(f"    n_layers         = {n_layers}")
    print(f"    n_heads          = {n_heads}")
    print(f"    n_kv_heads       = {n_kv_heads}")
    print(f"    max_seq_len      = {max_seq_len}")
    print(f"    head_dim         = {head_dim}")
    print(f"    rope_theta       = {rope_theta}")
    print(f"    rms_norm_eps     = {rms_norm_eps}")
    print(f"    group_size       = {group_size}")

    # ── Collect statistics ──
    total_bytes = 0
    original_params = 0
    layer_sparsities = []

    with open(output_path, "wb") as fout:
        # ── 1. Header ──
        write_header(
            fout,
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            max_seq_len=max_seq_len,
            head_dim=head_dim,
            rope_theta=rope_theta,
            group_size=group_size,
            rms_norm_eps=rms_norm_eps,
        )
        total_bytes += HEADER_SIZE

        # ── 2. Embedding ──
        embed_key = "model.embed_tokens.weight"
        embed_w = state[embed_key].to(torch.float16).numpy()
        print(f"[*] Writing embedding: {embed_w.shape}")
        total_bytes += _write_raw(fout, embed_w)
        original_params += embed_w.size

        # ── 3. Layers ──
        for layer_idx in tqdm(range(n_layers), desc="Layers"):
            prefix = f"model.layers.{layer_idx}."
            layer_nonzero = 0
            layer_total = 0

            for wname in LINEAR_WEIGHT_NAMES:
                key = prefix + wname
                w = state[key].float()                        # (out, in)
                original_params += w.numel()

                # Pad in-features to multiple of 32
                w_padded = pad_to_multiple(w, multiple=32)

                # Ternarize
                ternary, scales = ternarize_tensor(
                    w_padded, group_size=group_size, threshold_factor=threshold_factor,
                )

                # Statistics
                nz = (ternary != 0).sum().item()
                layer_nonzero += nz
                layer_total += ternary.numel()

                # Pack
                ternary_2d = ternary.reshape(w_padded.shape[0], w_padded.shape[1])
                nonzero_masks, sign_masks = pack_ternary_to_bitmasks(ternary_2d)

                # Write bitmasks
                total_bytes += _write_raw(fout, nonzero_masks)
                total_bytes += _write_raw(fout, sign_masks)

                # Write scales (float16)
                scales_np = scales.numpy()
                total_bytes += _write_raw(fout, scales_np)

            # Norm weights (fp16)
            attn_norm_key = prefix + "input_layernorm.weight"
            ffn_norm_key = prefix + "post_attention_layernorm.weight"

            attn_norm = state[attn_norm_key].to(torch.float16).numpy()
            ffn_norm = state[ffn_norm_key].to(torch.float16).numpy()
            total_bytes += _write_raw(fout, attn_norm)
            total_bytes += _write_raw(fout, ffn_norm)
            original_params += attn_norm.size + ffn_norm.size

            sparsity = 1.0 - (layer_nonzero / layer_total) if layer_total > 0 else 0.0
            layer_sparsities.append(sparsity)

        # ── 4. Final norm ──
        final_norm = state["model.norm.weight"].to(torch.float16).numpy()
        total_bytes += _write_raw(fout, final_norm)
        original_params += final_norm.size

        # ── 5. LM head ──
        # Handle tied embeddings: if lm_head.weight is missing, reuse embedding
        if "lm_head.weight" in state:
            lm_head = state["lm_head.weight"].to(torch.float16).numpy()
        else:
            print("[*] lm_head.weight not found – using tied embedding weights")
            lm_head = state[embed_key].to(torch.float16).numpy()
        print(f"[*] Writing lm_head: {lm_head.shape}")
        total_bytes += _write_raw(fout, lm_head)
        original_params += lm_head.size

        # ── 6. Tokenizer ──
        print("[*] Writing tokenizer data")
        tok_data = find_tokenizer_model(model_name)
        fout.write(struct.pack("<Q", len(tok_data)))
        total_bytes += 8
        fout.write(tok_data)
        total_bytes += len(tok_data)

    # ── Statistics ──
    original_size_mb = (original_params * 4) / (1024 * 1024)  # fp32
    output_size_mb = total_bytes / (1024 * 1024)
    compression = original_size_mb / output_size_mb if output_size_mb > 0 else 0.0

    print()
    print("=" * 60)
    print(f"  Output file         : {output_path}")
    print(f"  Output size         : {output_size_mb:.2f} MB")
    print(f"  Original (fp32)     : {original_size_mb:.2f} MB")
    print(f"  Compression ratio   : {compression:.2f}x")
    print(f"  Tokenizer blob      : {len(tok_data)} bytes")
    print()
    print("  Per-layer sparsity:")
    for i, sp in enumerate(layer_sparsities):
        print(f"    Layer {i:3d}: {sp * 100:5.1f}%")
    print("=" * 60)
    print("[✓] Done!")


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert a HuggingFace LLaMA/Mistral model to .tllm ternary format",
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="HuggingFace model name or local path",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output .tllm file path",
    )
    parser.add_argument(
        "--group-size", type=int, default=128,
        help="Quantization group size (default: 128)",
    )
    parser.add_argument(
        "--threshold-factor", type=float, default=0.7,
        help="Dead-zone threshold as fraction of mean|w| (default: 0.7)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Torch device for model loading (default: cuda)",
    )
    args = parser.parse_args()

    convert(
        model_name=args.model,
        output_path=args.output,
        group_size=args.group_size,
        threshold_factor=args.threshold_factor,
        device=args.device,
    )


if __name__ == "__main__":
    main()
