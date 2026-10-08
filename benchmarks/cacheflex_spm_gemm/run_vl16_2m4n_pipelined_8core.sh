#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export CACHEFLEX_GEMM_BIN="$SCRIPT_DIR/bin_gem5/cacheflex_spm_vl16_2m4n_pipelined"
export CACHEFLEX_GEMM_OUT="$ROOT/results/cacheflex_spm_gemm/vl2048_spm_2m4n_pipelined_ruby_8c_w7"
exec "$SCRIPT_DIR/run_vl16_2m4n_8core.sh"
