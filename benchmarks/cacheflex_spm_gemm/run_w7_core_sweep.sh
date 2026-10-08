#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

for cores in 1 2 4 8; do
    echo "=== W7 baseline: ${cores} core(s) ==="
    bash "$SCRIPT_DIR/run_matched_acl_garnet.sh" 784 1024 256 1 "$cores"

    echo "=== W7 CacheFlex SPMCP: ${cores} core(s) ==="
    bash "$SCRIPT_DIR/run_mesi3_spm_garnet.sh" 784 1024 256 1 "$cores"
done

python3 "$ROOT/scripts/analyze_gemm_epochs.py" \
    --results "$ROOT/results/cacheflex_spm_gemm" \
    --output "$ROOT/results/cacheflex_spm_gemm/w7_core_sweep_per_epoch.csv"

echo "$ROOT/results/cacheflex_spm_gemm/w7_core_sweep_per_epoch.csv"
