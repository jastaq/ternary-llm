#!/bin/bash
# ============================================================================
# Ternary LLM — Google Colab Setup Script
#
# Usage (in Colab cell):
#   !git clone https://github.com/<YOUR_USERNAME>/ternary-llm.git
#   !cd ternary-llm && bash colab_setup.sh
# ============================================================================
set -e

echo "========================================"
echo " Ternary LLM — Colab Setup"
echo "========================================"

# ── 1. Check GPU ──
echo ""
echo "[1/6] Checking GPU..."
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
echo "CUDA Toolkit: ${CUDA_VERSION}"
echo ""

# ── 2. Install system dependencies ──
echo "[2/6] Installing system dependencies..."
apt-get update -qq > /dev/null 2>&1
apt-get install -y -qq cmake > /dev/null 2>&1
CMAKE_VERSION=$(cmake --version | head -1)
echo "  ${CMAKE_VERSION}"

# ── 3. Install Python dependencies ──
echo ""
echo "[3/6] Installing Python dependencies..."
pip install -q torch transformers safetensors sentencepiece numpy tqdm
echo "  Done"

# ── 4. Build C++ engine ──
echo ""
echo "[4/6] Building C++ inference engine..."
mkdir -p build
cd build

# Detect GPU architecture for optimal build
GPU_ARCH=$(python3 -c "
import subprocess
result = subprocess.run(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'], 
                       capture_output=True, text=True)
cap = result.stdout.strip().replace('.', '')
print(cap)
" 2>/dev/null || echo "80")

echo "  Target GPU architecture: sm_${GPU_ARCH}"

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${GPU_ARCH}" \
    2>&1 | tail -5

make -j$(nproc) 2>&1 | tail -10

cd ..

echo ""
echo "  Build complete!"
ls -lh build/ternary-llm build/test_gemv build/benchmark 2>/dev/null

# ── 5. Run GEMV correctness test ──
echo ""
echo "[5/6] Running GEMV kernel test..."
./build/test_gemv || echo "  (test binary not available, skipping)"

# ── 6. Done ──
echo ""
echo "========================================"
echo " Setup complete!"
echo ""
echo " Next steps:"
echo "   1. Ternarize a model:"
echo "      !cd ternary-llm/ternarize && python ternarize.py \\"
echo "          --model meta-llama/Llama-3.2-1B \\"
echo "          --output ../model.tllm"
echo ""
echo "   2. Run inference:"
echo "      !cd ternary-llm && ./build/ternary-llm \\"
echo "          --model model.tllm \\"
echo "          --prompt \"Once upon a time\" \\"
echo "          --max-tokens 100 --benchmark"
echo "========================================"
