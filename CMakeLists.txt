cmake_minimum_required(VERSION 3.18)
project(ternary-llm LANGUAGES CXX CUDA)

# ---------------------------------------------------------------------------
# Standards & defaults
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Target GPU architectures  (Volta → Hopper)
set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90")

# Default to Release build
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE  "-O3 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG --use_fast_math")

# ---------------------------------------------------------------------------
# SentencePiece (static)
# ---------------------------------------------------------------------------
include(FetchContent)
FetchContent_Declare(
    sentencepiece
    GIT_REPOSITORY https://github.com/google/sentencepiece.git
    GIT_TAG        v0.2.0
)
set(SPM_ENABLE_SHARED  OFF CACHE BOOL "" FORCE)
set(SPM_ENABLE_TCMALLOC OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(sentencepiece)

# ---------------------------------------------------------------------------
# Include paths
# ---------------------------------------------------------------------------
include_directories(${CMAKE_SOURCE_DIR}/include)

# ---------------------------------------------------------------------------
# Main executable
# ---------------------------------------------------------------------------
set(KERNEL_SOURCES
    kernels/ternary_gemv.cu
    kernels/ternary_gemm.cu
    kernels/rmsnorm.cu
    kernels/rope.cu
    kernels/attention.cu
    kernels/activations.cu
    kernels/elementwise.cu
)

set(CPP_SOURCES
    src/main.cpp
    src/model.cpp
    src/transformer.cpp
    src/tokenizer.cpp
    src/sampler.cpp
    src/memory.cpp
)

add_executable(ternary-llm ${CPP_SOURCES} ${KERNEL_SOURCES})
target_link_libraries(ternary-llm PRIVATE sentencepiece-static)

# Ensure .cpp files that include CUDA headers can be compiled
set_source_files_properties(
    src/transformer.cpp src/model.cpp src/sampler.cpp src/memory.cpp
    PROPERTIES LANGUAGE CUDA
)

# ---------------------------------------------------------------------------
# Test: ternary GEMV correctness
# ---------------------------------------------------------------------------
add_executable(test_gemv
    tests/test_ternary_gemv.cu
    kernels/ternary_gemv.cu
)
target_include_directories(test_gemv PRIVATE ${CMAKE_SOURCE_DIR}/include)

# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
add_executable(benchmark
    tests/benchmark.cu
    kernels/ternary_gemv.cu
    kernels/ternary_gemm.cu
    kernels/elementwise.cu
)
target_include_directories(benchmark PRIVATE ${CMAKE_SOURCE_DIR}/include)
