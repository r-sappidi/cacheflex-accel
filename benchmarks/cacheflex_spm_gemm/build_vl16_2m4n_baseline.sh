#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MICRO_ROOT="$(cd "$ROOT/../cacheflex_micro" && pwd)"
source "$ROOT/setup_spm_env.sh" >/dev/null

OUTDIR="$SCRIPT_DIR/bin_gem5"
OUT="${CACHEFLEX_BUILD_OUT:-$OUTDIR/cache_vl16_2m4n}"
mkdir -p "$OUTDIR"

link_out="$(mktemp "${TMPDIR:-/tmp}/cache-vl16-2m4n.XXXXXX")"
trap 'rm -f "$link_out"' EXIT
"$CROSS_CXX" \
    -std=c++17 -O3 -march=armv8.2-a+sve+fp16 -mbranch-protection=none \
    -static -pthread -DGEM5 -DGEM5_SE -DIDENTICAL_CACHE_BASELINE \
    ${CACHEFLEX_PROFILE_DEFINE:-} \
    -I"$M5_INCLUDE" \
    -I"$MICRO_ROOT/experiments/vl_length/tile_shape" \
    -I"$MICRO_ROOT/kernels/llama_bench" \
    -o "$link_out" "$SCRIPT_DIR/vl16_2m4n_driver.cpp" "$M5OP_OBJ" -lm
chmod +x "$link_out"
mv -f "$link_out" "$OUT"
trap - EXIT
echo "$OUT"
