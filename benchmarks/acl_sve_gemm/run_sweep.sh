#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Representative model-layer GEMMs. Columns are ID, M, K, N. The runner
# accepts M, N, K, so keep the table order explicit here to avoid transposing
# the model dimensions. Override ITERATIONS in the environment when needed.
: "${ITERATIONS:=1}"
echo "Running ACL SVE GEMM sweep with NeoverseV2, MESI Three Level, and Garnet"
cases=(
    "W1 128 4096 11008"
    "W2 256 4096 4096"
    "W3 256 2048 2048"
    "W4 512 768 768"
    "W5 2048 2048 2048"
    "W6 2048 64 2048"
    "W7 784 256 1024"
)

for case_spec in "${cases[@]}"; do
    read -r workload m k n <<< "$case_spec"
    echo "=== $workload: M=$m K=$k N=$n iterations=$ITERATIONS ==="
    bash "$SCRIPT_DIR/run_mesi3_garnet.sh" \
        "$m" "$n" "$k" "$ITERATIONS"
done
