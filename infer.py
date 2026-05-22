#!/usr/bin/env python3
"""
infer.py — Python wrapper for ternary-llm inference.

Handles tokenization via HuggingFace (works with any tokenizer: tiktoken, BPE,
SentencePiece) and calls the C++ engine for the actual forward pass.

Usage:
    python infer.py \
        --model model.tllm \
        --hf-model Qwen/Qwen3-4B-Thinking-2507 \
        --prompt "Explain quantum computing" \
        --max-tokens 128
"""

import argparse
import subprocess
import sys

from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Ternary LLM inference with HuggingFace tokenizer")
    parser.add_argument("--model", required=True, help="Path to .tllm model file")
    parser.add_argument("--hf-model", required=True, help="HuggingFace model name (for tokenizer)")
    parser.add_argument("--prompt", default="Hello", help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max tokens to generate")
    parser.add_argument("--max-seq-len", type=int, default=4096, help="Max sequence length")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--engine", default="./build/ternary-llm", help="Path to C++ engine binary")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    # 1. Load HuggingFace tokenizer
    print(f"[Python] Loading tokenizer: {args.hf_model}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model)

    # 2. Tokenize prompt
    input_ids = tokenizer.encode(args.prompt, add_special_tokens=True)
    token_ids_str = ",".join(str(t) for t in input_ids)
    print(f"[Python] Prompt: {repr(args.prompt)}", file=sys.stderr)
    print(f"[Python] Token IDs ({len(input_ids)} tokens): {token_ids_str[:100]}...", file=sys.stderr)

    # 3. Run C++ engine
    cmd = [
        args.engine,
        "--model", args.model,
        "--token-ids", token_ids_str,
        "--max-tokens", str(args.max_tokens),
        "--max-seq-len", str(args.max_seq_len),
        "--temperature", str(args.temperature),
        "--top-p", str(args.top_p),
        "--top-k", str(args.top_k),
    ]
    if args.benchmark:
        cmd.append("--benchmark")

    print(f"[Python] Running engine...", file=sys.stderr)

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Print stderr (model loading info, benchmark stats)
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")

    if result.returncode != 0:
        print(f"[Python] Engine exited with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)

    # 4. Decode output token IDs
    raw_output = result.stdout.strip()
    if not raw_output:
        print("[Python] No output from engine", file=sys.stderr)
        sys.exit(1)

    output_ids = [int(x) for x in raw_output.split(",") if x.strip()]

    # 5. Decode and print
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"[Python] Generated {len(output_ids)} tokens", file=sys.stderr)
    print(f"{'='*60}\n", file=sys.stderr)

    # Print prompt + generated text
    full_ids = input_ids + output_ids
    text = tokenizer.decode(full_ids, skip_special_tokens=True)
    print(text)


if __name__ == "__main__":
    main()
