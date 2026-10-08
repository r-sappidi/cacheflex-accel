#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RUNNER="$ROOT/benchmarks/cacheflex_spm_gemm/run_vl16_best_8core.sh"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data2/rsappidi/conda-envs/cacheflex-gem5
export CONDA_PKGS_DIRS=/data2/rsappidi/conda-pkgs
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

run_pair() {
    local name="$1" op="$2" m="$3" k="$4" n="$5"
    for variant in baseline spm; do
        local out="$ROOT/results/ai_workloads/real_operators/${name}_${variant}"
        if [[ -f "$out/run.log" ]] && grep -q ' PASS ' "$out/run.log"; then
            echo "SKIP completed $name $variant"
            continue
        fi
        CACHEFLEX_M="$m" CACHEFLEX_K="$k" CACHEFLEX_N="$n" \
        CACHEFLEX_ITERS=1 CACHEFLEX_KC=256 CACHEFLEX_OPERATOR="$op" \
        CACHEFLEX_INPUT_MODE=pattern CACHEFLEX_GEMM_OUT="$out" \
            "$RUNNER" "$variant"
    done
}

# MobileNet/ConvNeXt-style NHWC 1x1 convolution with a fused bias and SiLU.
run_pair pointwise_1x1_silu_s96 conv_silu 96 512 512
run_pair pointwise_1x1_silu_s784 conv_silu 784 512 512

# Diffusion transformer feed-forward projection with fused bias, GELU, and
# residual addition.
run_pair diffusion_ffn_m128 diffusion_gelu_residual 128 768 768
run_pair diffusion_ffn_m512 diffusion_gelu_residual 512 768 768

# Complete LoRA linear: XW + (X A_down) A_up, rank 16.
run_pair lora_linear_m128 lora_residual 128 512 512
run_pair lora_linear_m512 lora_residual 512 512 512
