#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RUNNER="$ROOT/benchmarks/cacheflex_spm_gemm/run_vl16_best_8core.sh"

run_pair() {
    local name="$1" m="$2" k="$3" n="$4" iterations="$5"
    for variant in baseline spm; do
        local out="$ROOT/results/ai_workloads/promising/${name}_${variant}"
        if [[ -f "$out/run.log" ]] && rg -q ' PASS ' "$out/run.log"; then
            echo "SKIP completed $name $variant"
            continue
        fi
        CACHEFLEX_M="$m" CACHEFLEX_K="$k" CACHEFLEX_N="$n" \
        CACHEFLEX_ITERS="$iterations" CACHEFLEX_KC=256 \
        CACHEFLEX_INPUT_MODE=pattern CACHEFLEX_GEMM_OUT="$out" \
            "$RUNNER" "$variant"
    done
}

# Batched transformer projection: vary temporal weight reuse.
run_pair batch_m256_i1 256 512 1024 1
run_pair batch_m256_i4 256 512 1024 4

# Pointwise convolution: vary spatial reuse of a 512x512 weight tensor.
run_pair conv_s96  96 512 512 1
run_pair conv_s784 784 512 512 1

# Diffusion/transformer dense projections: vary channel width and activation rows.
run_pair diffusion_m128_d768 128 768 768 1
run_pair diffusion_m512_d768 512 768 768 1

# LoRA base projection upper bound: vary the number of tokens sharing the base.
run_pair lora_base_m256 256 512 512 1
run_pair lora_base_m512 512 512 512 1
