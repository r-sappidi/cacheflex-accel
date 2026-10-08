#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
GEMM="$ROOT/benchmarks/cacheflex_spm_gemm/run_vl16_best_8core.sh"

run_pair() {
    local name="$1" m="$2" k="$3" n="$4" iterations="$5"
    for variant in baseline spm; do
        CACHEFLEX_M="$m" CACHEFLEX_K="$k" CACHEFLEX_N="$n" \
        CACHEFLEX_ITERS="$iterations" CACHEFLEX_KC=256 \
        CACHEFLEX_INPUT_MODE=pattern \
        CACHEFLEX_GEMM_OUT="$ROOT/results/ai_workloads/${name}_${variant}_8c_vl16" \
            "$GEMM" "$variant"
    done
}

# One representative shape for each dense multicore reuse regime.  Some
# higher-level workloads intentionally share a shape because their dense
# projection has the same architectural dataflow.
run_pair batched_transformer 256 512 1024 2
run_pair cnn_pointwise       196 512  512 1
# Low-row decode shapes require N-only partitioning; the 4M-by-2N production
# mapping correctly rejects them rather than creating empty M partitions.
run_pair lora_base            128 512  512 1
run_pair diffusion_projection 256 512  512 1
