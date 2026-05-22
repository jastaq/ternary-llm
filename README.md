# Ternary LLM — 2-bit Inference Engine

High-performance inference engine for ternary-quantized Large Language Models.
Weights are compressed to **2 bits per parameter** using dual-bitmask encoding,
and inference runs entirely in **C++ / CUDA** — no Python or PyTorch at runtime.

## Features

| Feature | Details |
|---------|---------|
| **2-bit ternary weights** | Each weight ∈ {-1, 0, +1} stored as two bitmasks (nonzero + sign) |
| **Custom CUDA kernels** | Hand-written GEMV/GEMM kernels that exploit ternary structure — additions/subtractions only, no multiplications |
| **~7.5× model compression** | 7B model: 14 GB → ~1.9 GB |
| **Per-group scaling** | FP16 scale factor per 128-weight group preserves accuracy |
| **C++ inference** | Zero Python dependency at runtime |
| **Streaming generation** | Tokens printed as they are generated |
| **GQA / MQA support** | Grouped-query attention for Mistral / LLaMA 3 |
| **KV cache** | Pre-allocated, zero-allocation decode loop |

## Supported Models

- **LLaMA** (1B, 3B, 7B, 8B, 13B, 70B)
- **Mistral** (7B)
- Any HuggingFace model with LLaMA-compatible architecture (RoPE, RMSNorm, SwiGLU)

## Requirements

- **NVIDIA GPU** with Compute Capability ≥ 7.0 (Volta or newer)
- **CUDA Toolkit** ≥ 11.0
- **CMake** ≥ 3.18
- **Python 3.8+** (for ternarization only)

## Quick Start

### 1. Build the C++ Engine

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### 2. Ternarize a Model (Python)

```bash
cd ternarize
pip install -r requirements.txt

python ternarize.py \
    --model meta-llama/Llama-3.2-1B \
    --output ../models/llama-1b.tllm \
    --group-size 128 \
    --threshold-factor 0.7
```

### 3. Run Inference

```bash
# Single prompt
./ternary-llm --model models/llama-1b.tllm \
    --prompt "The meaning of life is" \
    --max-tokens 128 \
    --temperature 0.7 \
    --benchmark

# Interactive chat
./ternary-llm --model models/llama-1b.tllm --interactive
```

### 4. Run Tests & Benchmarks

```bash
# GEMV kernel correctness test
./test_gemv

# Performance benchmark (ternary vs FP16)
./benchmark
```

## CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model <path>` | required | Path to `.tllm` model file |
| `--prompt <text>` | "Hello" | Input prompt |
| `--max-tokens <n>` | 256 | Maximum tokens to generate |
| `--temperature <f>` | 0.7 | Sampling temperature |
| `--top-p <f>` | 0.9 | Nucleus sampling threshold |
| `--top-k <n>` | 40 | Top-K filtering |
| `--rep-penalty <f>` | 1.1 | Repetition penalty |
| `--seed <n>` | 42 | Random seed |
| `--interactive` | off | Interactive chat mode |
| `--benchmark` | off | Print timing statistics |

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                   Python Ternarizer                   │
│  HuggingFace Model → Ternarize → Pack → .tllm file  │
└─────────────────────────┬────────────────────────────┘
                          │
                    ┌─────▼─────┐
                    │ .tllm file │  (~2 bits/weight)
                    └─────┬─────┘
                          │
┌─────────────────────────▼────────────────────────────┐
│              C++ / CUDA Inference Engine              │
│                                                       │
│  ┌─────────┐  ┌──────────────┐  ┌──────────────────┐ │
│  │Tokenizer│  │  Transformer  │  │    CUDA Kernels   │ │
│  │  (SPM)  │  │ Forward Pass  │  │ ternary_gemv/gemm │ │
│  └────┬────┘  └──────┬───────┘  │ rmsnorm, rope     │ │
│       │              │          │ attention, silu    │ │
│       │        ┌─────▼──────┐  └──────────────────┘ │
│       │        │  Sampler    │                       │
│       │        │ temp/top-p  │                       │
│       │        └─────┬──────┘                        │
│       │              │                               │
│       └──────────────▼───────────────────────────────│
│                   Output Text                        │
└──────────────────────────────────────────────────────┘
```

## Weight Encoding

Each ternary weight (∈ {-1, 0, +1}) is stored using **dual bitmask encoding**:

| Value | `nonzero_bit` | `sign_bit` |
|-------|---------------|------------|
|   0   |       0       |     0      |
|  +1   |       1       |     1      |
|  -1   |       1       |     0      |

32 weights are packed into 2 × `uint32_t` = 8 bytes → **2 bits per weight**.

With per-group FP16 scale factors (1 scale per 128 weights), the effective
storage is **2.125 bits per weight**.

### CUDA Kernel Trick

The dual bitmask encoding enables an elegant kernel optimisation:

```cuda
uint32_t pos = nonzero & sign;     // weights that are +1
uint32_t neg = nonzero & ~sign;    // weights that are -1

// Iterate only over non-zero positions using __ffs()
while (pos) {
    int bit = __ffs(pos) - 1;
    sum += input[base + bit];      // just ADD — no multiply!
    pos &= pos - 1;
}
while (neg) {
    int bit = __ffs(neg) - 1;
    sum -= input[base + bit];      // just SUBTRACT
    neg &= neg - 1;
}
```

Zero weights are **skipped for free** — the bit iteration naturally ignores them.

## Model Size Estimates

| Model | FP16 | Ternary (2.125 bit) | Compression |
|-------|------|---------------------|-------------|
| LLaMA 1B  |  2.0 GB |  ~0.3 GB | 7.1× |
| Mistral 7B | 14.0 GB |  ~1.9 GB | 7.5× |
| LLaMA 8B  | 16.0 GB |  ~2.2 GB | 7.3× |
| LLaMA 70B | 140 GB  | ~19.0 GB | 7.4× |

## Project Structure

```
ternary-llm/
├── CMakeLists.txt          # Build system
├── README.md               # This file
├── include/                # C++ headers
│   ├── config.h            # Model configuration
│   ├── format.h            # .tllm binary format
│   ├── kernels.cuh         # CUDA kernel declarations
│   ├── memory.h            # GPU memory pool
│   ├── model.h             # Model data structures
│   ├── sampler.h           # Token sampling
│   ├── tokenizer.h         # SentencePiece tokenizer
│   └── transformer.h       # Forward pass
├── src/                    # C++ implementation
│   ├── main.cpp            # CLI entry point
│   ├── memory.cpp          # Memory pool
│   ├── model.cpp           # Model loading
│   ├── sampler.cpp         # Sampling strategies
│   ├── tokenizer.cpp       # Tokenizer wrapper
│   └── transformer.cpp     # Forward pass orchestration
├── kernels/                # CUDA kernels
│   ├── activations.cu      # SiLU × gate (SwiGLU)
│   ├── attention.cu        # Multi-head attention
│   ├── elementwise.cu      # Add, embedding, KV cache, FP16 GEMV/GEMM
│   ├── rmsnorm.cu          # RMSNorm + fused residual
│   ├── rope.cu             # Rotary position embeddings
│   ├── ternary_gemm.cu     # Ternary GEMM (prefill)
│   └── ternary_gemv.cu     # Ternary GEMV (decode)
├── ternarize/              # Python ternarization tool
│   ├── pack.py             # Weight packing utilities
│   ├── requirements.txt    # Python dependencies
│   └── ternarize.py        # Main ternarization script
└── tests/                  # Tests & benchmarks
    ├── benchmark.cu        # Performance benchmarks
    ├── test_packing.py     # Python packing tests
    └── test_ternary_gemv.cu# GEMV correctness test
```

## License

MIT
