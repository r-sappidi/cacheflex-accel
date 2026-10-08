#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
variant="${1:?usage: $0 baseline|spm}"

case "$variant" in
    baseline)
        export CACHEFLEX_GEMM_BIN="$SCRIPT_DIR/bin_gem5/cache_vl16_acl_scheduler"
        export CACHEFLEX_GEMM_OUT="$ROOT/results/cacheflex_spm_gemm/vl2048_cache_acl_scheduler_ruby_8c_w7"
        exec "$SCRIPT_DIR/run_vl16_2m4n_baseline_8core.sh"
        ;;
    spm)
        export CACHEFLEX_GEMM_BIN="$SCRIPT_DIR/bin_gem5/cacheflex_spm_vl16_acl_scheduler"
        export CACHEFLEX_GEMM_OUT="$ROOT/results/cacheflex_spm_gemm/vl2048_spm_acl_scheduler_ruby_8c_w7"
        exec "$SCRIPT_DIR/run_vl16_2m4n_8core.sh"
        ;;
    *)
        echo "variant must be baseline or spm" >&2
        exit 2
        ;;
esac
