#include "model.h"
#include "transformer.h"
#include "tokenizer.h"
#include "sampler.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>
#include <iostream>

// ---------------------------------------------------------------------------
// CLI argument parsing
// ---------------------------------------------------------------------------
struct Args {
    std::string model_path;
    std::string prompt       = "Hello";
    int         max_tokens   = 256;
    float       temperature  = 0.7f;
    float       top_p        = 0.9f;
    int         top_k        = 40;
    float       rep_penalty  = 1.1f;
    uint64_t    seed         = 42;
    bool        interactive  = false;
    bool        benchmark    = false;
};

static void print_usage(const char* prog) {
    fprintf(stderr,
        "Ternary LLM Inference Engine\n"
        "\n"
        "Usage: %s [options]\n"
        "\n"
        "Options:\n"
        "  --model <path>        Path to .tllm model file (required)\n"
        "  --prompt <text>       Input prompt (default: \"Hello\")\n"
        "  --max-tokens <n>      Maximum tokens to generate (default: 256)\n"
        "  --temperature <f>     Sampling temperature (default: 0.7)\n"
        "  --top-p <f>           Nucleus sampling threshold (default: 0.9)\n"
        "  --top-k <n>           Top-K sampling (default: 40)\n"
        "  --rep-penalty <f>     Repetition penalty (default: 1.1)\n"
        "  --seed <n>            Random seed (default: 42)\n"
        "  --interactive         Interactive chat mode\n"
        "  --benchmark           Print timing statistics\n"
        "  --help                Show this help\n"
        "\n", prog);
}

static Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--model"       && i+1 < argc) args.model_path  = argv[++i];
        else if (a == "--prompt"      && i+1 < argc) args.prompt      = argv[++i];
        else if (a == "--max-tokens"  && i+1 < argc) args.max_tokens  = std::atoi(argv[++i]);
        else if (a == "--temperature" && i+1 < argc) args.temperature = std::atof(argv[++i]);
        else if (a == "--top-p"       && i+1 < argc) args.top_p       = std::atof(argv[++i]);
        else if (a == "--top-k"       && i+1 < argc) args.top_k       = std::atoi(argv[++i]);
        else if (a == "--rep-penalty" && i+1 < argc) args.rep_penalty = std::atof(argv[++i]);
        else if (a == "--seed"        && i+1 < argc) args.seed        = std::strtoull(argv[++i], nullptr, 10);
        else if (a == "--interactive") args.interactive = true;
        else if (a == "--benchmark")  args.benchmark   = true;
        else if (a == "--help") { print_usage(argv[0]); std::exit(0); }
        else { fprintf(stderr, "Unknown option: %s\n", a.c_str()); print_usage(argv[0]); std::exit(1); }
    }
    if (args.model_path.empty()) {
        fprintf(stderr, "Error: --model is required\n");
        print_usage(argv[0]);
        std::exit(1);
    }
    return args;
}

// ---------------------------------------------------------------------------
// Generation loop
// ---------------------------------------------------------------------------
static void generate(Transformer& xfm, Tokenizer& tok, Sampler& sampler,
                     const std::string& prompt, int max_tokens,
                     bool benchmark) {
    using Clock = std::chrono::high_resolution_clock;

    // Encode prompt
    std::vector<int> prompt_ids = tok.encode(prompt, /*add_bos=*/true);
    int n_prompt = static_cast<int>(prompt_ids.size());

    if (n_prompt == 0) {
        fprintf(stderr, "[Error] Empty prompt after tokenisation\n");
        return;
    }

    fprintf(stderr, "[Generate] prompt tokens: %d\n", n_prompt);

    // Upload prompt tokens to GPU
    int* tokens_gpu = nullptr;
    cudaMalloc(&tokens_gpu, n_prompt * sizeof(int));
    cudaMemcpy(tokens_gpu, prompt_ids.data(),
               n_prompt * sizeof(int), cudaMemcpyHostToDevice);

    // ---- Prefill ----
    auto t_prefill_start = Clock::now();
    half* logits = xfm.forward(tokens_gpu, n_prompt, /*start_pos=*/0);
    cudaDeviceSynchronize();
    auto t_prefill_end = Clock::now();

    double prefill_ms = std::chrono::duration<double, std::milli>(
        t_prefill_end - t_prefill_start).count();

    // Print prompt back
    printf("%s", prompt.c_str());
    fflush(stdout);

    // Sample first token
    int next_token = sampler.sample(logits);
    sampler.add_to_history(next_token);

    // Print first generated token
    std::string piece = tok.decode(next_token);
    printf("%s", piece.c_str());
    fflush(stdout);

    // ---- Decode loop ----
    int pos = n_prompt;
    int generated = 1;

    auto t_decode_start = Clock::now();

    // Prepare single-token GPU buffer
    int* one_token_gpu = nullptr;
    cudaMalloc(&one_token_gpu, sizeof(int));

    for (int i = 1; i < max_tokens; ++i) {
        // Check for EOS
        if (next_token == tok.eos_id()) break;

        // Upload single token
        cudaMemcpy(one_token_gpu, &next_token, sizeof(int),
                   cudaMemcpyHostToDevice);

        // Forward (single token, incremental)
        logits = xfm.forward(one_token_gpu, 1, pos);
        cudaDeviceSynchronize();

        // Sample
        next_token = sampler.sample(logits);
        sampler.add_to_history(next_token);

        // Decode and stream
        piece = tok.decode(next_token);
        printf("%s", piece.c_str());
        fflush(stdout);

        pos++;
        generated++;
    }

    auto t_decode_end = Clock::now();
    double decode_ms = std::chrono::duration<double, std::milli>(
        t_decode_end - t_decode_start).count();

    printf("\n");

    // ---- Statistics ----
    if (benchmark) {
        double prefill_tok_s = (n_prompt > 0) ? (n_prompt / (prefill_ms / 1000.0)) : 0;
        double decode_tok_s  = (generated > 1) ? ((generated - 1) / (decode_ms / 1000.0)) : 0;

        fprintf(stderr, "\n--- Benchmark ---\n");
        fprintf(stderr, "Prefill:  %d tokens in %.1f ms  (%.1f tokens/s)\n",
                n_prompt, prefill_ms, prefill_tok_s);
        fprintf(stderr, "Decode:   %d tokens in %.1f ms  (%.1f tokens/s)\n",
                generated - 1, decode_ms, decode_tok_s);
        fprintf(stderr, "Total:    %d tokens generated\n", generated);

        // GPU memory info
        size_t free_mem = 0, total_mem = 0;
        cudaMemGetInfo(&free_mem, &total_mem);
        fprintf(stderr, "GPU VRAM: %.0f MB used / %.0f MB total\n",
                (total_mem - free_mem) / 1e6, total_mem / 1e6);
    }

    cudaFree(tokens_gpu);
    cudaFree(one_token_gpu);
}

// ---------------------------------------------------------------------------
// Interactive mode
// ---------------------------------------------------------------------------
static void interactive_loop(Transformer& xfm, Tokenizer& tok,
                             Sampler& sampler, int max_tokens,
                             bool benchmark) {
    printf("Ternary LLM Interactive Mode\n");
    printf("Type your prompt and press Enter. Type 'quit' to exit.\n\n");

    std::string line;
    while (true) {
        printf("> ");
        fflush(stdout);
        if (!std::getline(std::cin, line)) break;
        if (line == "quit" || line == "exit") break;
        if (line.empty()) continue;

        xfm.reset();
        sampler.reset_history();
        generate(xfm, tok, sampler, line, max_tokens, benchmark);
        printf("\n");
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    // Check CUDA
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        fprintf(stderr, "Error: No CUDA-capable GPU found\n");
        return 1;
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    fprintf(stderr, "[GPU] %s  (%.0f MB, SM %d.%d)\n",
            prop.name, prop.totalGlobalMem / 1e6,
            prop.major, prop.minor);

    // Load model
    fprintf(stderr, "\n[Loading model] %s\n", args.model_path.c_str());
    Model model = Model::load(args.model_path);

    // Initialise tokenizer
    Tokenizer tokenizer;
    if (!model.tokenizer_data.empty()) {
        if (!tokenizer.load_from_memory(model.tokenizer_data.data(),
                                         model.tokenizer_data.size())) {
            fprintf(stderr, "Error: Failed to load embedded tokenizer\n");
            return 1;
        }
        fprintf(stderr, "[Tokenizer] Loaded from embedded data  (vocab: %d)\n",
                tokenizer.vocab_size());
    } else {
        fprintf(stderr, "Error: No tokenizer data in model file\n");
        return 1;
    }

    // Build transformer
    Transformer transformer(model);

    // Build sampler
    SamplerConfig sc;
    sc.temperature       = args.temperature;
    sc.top_p             = args.top_p;
    sc.top_k             = args.top_k;
    sc.repetition_penalty = args.rep_penalty;
    sc.seed              = args.seed;
    Sampler sampler(sc, static_cast<int>(model.config.vocab_size));

    fprintf(stderr, "\n");

    if (args.interactive) {
        interactive_loop(transformer, tokenizer, sampler,
                         args.max_tokens, args.benchmark);
    } else {
        generate(transformer, tokenizer, sampler,
                 args.prompt, args.max_tokens, args.benchmark);
    }

    // Cleanup
    model.free_gpu();

    return 0;
}
