#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
variant="${1:-}"

case "$variant" in
baseline|spm)
    export CACHEFLEX_INPUT_MODE=pattern
    export CACHEFLEX_GEMM_OUT="${CACHEFLEX_GEMM_OUT:-$ROOT/results/cacheflex_spm_gemm/vl2048_${variant}_best_correctness_ruby_8c_w7}"
    exec "$SCRIPT_DIR/run_vl16_best_8core.sh" "$variant"
    ;;
*)
    echo "usage: $0 baseline|spm" >&2
    exit 2
    ;;
esac
