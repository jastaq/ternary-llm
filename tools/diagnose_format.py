#!/usr/bin/env python3
"""
Diagnostic: compare what Python WROTE vs what C++ EXPECTS to read.
Run this after ternarization to find any format mismatch.
"""
import struct, sys, os
from pathlib import Path

def pad32(x):
    return ((x + 31) // 32) * 32

def ternary_weight_bytes(out_features, in_features, group_size):
    masks_per_row = in_features // 32
    groups_per_row = in_features // group_size
    nz_bytes  = out_features * masks_per_row * 4
    sg_bytes  = out_features * masks_per_row * 4
    sc_bytes  = out_features * groups_per_row * 2
    return nz_bytes + sg_bytes + sc_bytes

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "qwen3-4b.tllm"
    file_size = os.path.getsize(path)
    print(f"File: {path}  ({file_size:,} bytes = {file_size/1e6:.1f} MB)\n")

    with open(path, "rb") as f:
        # Header
        hdr = f.read(256)
        fields = struct.unpack("<" + "I" * 15, hdr[:60])
        magic, version, arch = fields[0], fields[1], fields[2]
        vocab    = fields[3]
        hidden   = fields[4]
        inter    = fields[5]
        n_layers = fields[6]
        n_heads  = fields[7]
        n_kv     = fields[8]
        max_seq  = fields[9]
        head_dim = fields[10]
        gs       = fields[12]
        flags    = fields[14]

        print(f"Header: magic=0x{magic:08X} ver={version} arch={arch}")
        print(f"  vocab={vocab} hidden={hidden} inter={inter}")
        print(f"  layers={n_layers} heads={n_heads} kv_heads={n_kv}")
        print(f"  head_dim={head_dim} group_size={gs} flags={flags}")

        q_dim    = n_heads * head_dim
        kv_dim   = n_kv * head_dim
        q_pad    = pad32(q_dim)
        kv_pad   = pad32(kv_dim)
        hidden_p = pad32(hidden)
        inter_p  = pad32(inter)

        print(f"  q_dim={q_dim} kv_dim={kv_dim}")
        print(f"  q_pad={q_pad} kv_pad={kv_pad} hidden_pad={hidden_p} inter_pad={inter_p}")

        pos = 256
        print(f"\n{'Offset':>12}  {'Section':30}  {'Size':>12}  {'C++ expects':>12}")
        print("-" * 75)

        def show(name, size, cpp_size=None):
            nonlocal pos
            if cpp_size is None:
                cpp_size = size
            match = "✓" if size == cpp_size else f"✗ MISMATCH!"
            print(f"{pos:>12}  {name:30}  {size:>12}  {cpp_size:>12}  {match}")
            pos += size

        # Embedding
        embed_bytes = vocab * hidden * 2
        show("Embedding (fp16)", embed_bytes)

        # C++ weight read order and sizes
        cpp_weights = [
            ("q_proj",    q_dim,  hidden_p),
            ("k_proj",    kv_dim, hidden_p),
            ("v_proj",    kv_dim, hidden_p),
            ("o_proj",    hidden, q_pad),
            ("gate_proj", inter,  hidden_p),
            ("up_proj",   inter,  hidden_p),
            ("down_proj", hidden, inter_p),
        ]

        # We need actual weight shapes from the model to compare
        # For now, let's read what's actually in the file and compare
        # Python writes based on ACTUAL weight shapes (out, pad32(in))
        # C++ reads based on COMPUTED shapes from config

        # Let's just compute what Python would write for standard LLaMA shapes
        py_weights = [
            ("q_proj",    q_dim,  pad32(hidden)),   # (q_dim, hidden) → pad in_features
            ("k_proj",    kv_dim, pad32(hidden)),
            ("v_proj",    kv_dim, pad32(hidden)),
            ("o_proj",    hidden, pad32(q_dim)),     # (hidden, q_dim) → pad in_features
            ("gate_proj", inter,  pad32(hidden)),
            ("up_proj",   inter,  pad32(hidden)),
            ("down_proj", hidden, pad32(inter)),
        ]

        for layer_idx in range(n_layers):
            for (wname, py_out, py_in), (_, cpp_out, cpp_in) in zip(py_weights, cpp_weights):
                py_size  = ternary_weight_bytes(py_out, py_in, gs)
                cpp_size = ternary_weight_bytes(cpp_out, cpp_in, gs)
                show(f"L{layer_idx}.{wname}", py_size, cpp_size)

            # Norms
            norm_bytes = hidden * 2
            show(f"L{layer_idx}.attn_norm", norm_bytes)
            show(f"L{layer_idx}.ffn_norm", norm_bytes)

            if layer_idx == 0:
                # Only show first layer in detail, then skip
                print(f"  ... (layers 1-{n_layers-1} same pattern)")

            if layer_idx > 0:
                continue  # Don't print all 36 layers

        # Skip to actual position after all layers
        per_layer = sum(ternary_weight_bytes(o, i, gs) for _, o, i in py_weights) + hidden * 2 * 2
        pos = 256 + embed_bytes + per_layer * n_layers

        # Final norm
        show("Final norm (fp16)", hidden * 2)

        # lm_head
        if flags & 1:
            lm_bytes = ternary_weight_bytes(vocab, hidden_p, gs)
            show("lm_head (ternary)", lm_bytes)
        else:
            lm_bytes = vocab * hidden * 2
            show("lm_head (fp16)", lm_bytes)

        # Tokenizer
        print(f"\n{pos:>12}  Expected position before tokenizer")
        print(f"{file_size:>12}  Actual file size")
        remaining = file_size - pos
        print(f"{remaining:>12}  Remaining (should be 8 + tokenizer_data)")

        # Read actual tokenizer size from file
        f.seek(pos)
        tok_size_raw = f.read(8)
        if len(tok_size_raw) == 8:
            tok_size = struct.unpack("<Q", tok_size_raw)[0]
            print(f"\n  Tokenizer size at expected position: {tok_size:,} bytes")
            if tok_size < 100_000_000:
                print(f"  ✓ Looks valid")
            else:
                print(f"  ✗ Looks bogus! Format mismatch detected.")

                # Try to find actual tokenizer by searching for the right size
                # We know it should be ~11.4 MB
                actual_pos = file_size - 11_422_654 - 8
                f.seek(actual_pos)
                maybe_size = struct.unpack("<Q", f.read(8))[0]
                print(f"\n  Trying position {actual_pos}: size = {maybe_size:,}")
                if abs(maybe_size - 11_422_654) < 1000:
                    delta = actual_pos - pos
                    print(f"  ✓ Found tokenizer! Offset delta = {delta:,} bytes")
                    print(f"    This means C++ reads {-delta:,} bytes MORE than Python writes")

if __name__ == "__main__":
    main()
