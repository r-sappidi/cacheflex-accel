#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MICRO_ROOT="$(cd "$ROOT/../cacheflex_micro" && pwd)"
source "$ROOT/setup_spm_env.sh" >/dev/null

OUTDIR="$SCRIPT_DIR/bin_gem5"
OUT="${CACHEFLEX_BUILD_OUT:-$OUTDIR/cacheflex_spm_vl16_2m4n}"
ASM="$OUTDIR/vl16_2m4n_driver.s"
ENC="$OUTDIR/vl16_2m4n_driver_enc.s"
OBJ="$OUTDIR/vl16_2m4n_driver.o"
mkdir -p "$OUTDIR"

"$CROSS_CXX" \
    -std=c++17 -O3 -march=armv8.2-a+sve+fp16 -mbranch-protection=none \
    -DGEM5 -DGEM5_SE ${CACHEFLEX_PROFILE_DEFINE:-} \
    -I"$M5_INCLUDE" \
    -I"$MICRO_ROOT/experiments/vl_length/tile_shape" \
    -I"$MICRO_ROOT/kernels/llama_bench_spm" \
    -S -o "$ASM" "$SCRIPT_DIR/vl16_2m4n_driver.cpp"
python3 "$SPM_COMPILER" "$ASM" "$ENC"
"$CROSS_CXX" -c -o "$OBJ" "$ENC"

link_out="$(mktemp "${TMPDIR:-/tmp}/cacheflex-vl16-2m4n.XXXXXX")"
trap 'rm -f "$link_out"' EXIT
"$CROSS_CXX" -static -pthread -o "$link_out" "$OBJ" "$M5OP_OBJ" -lm
chmod +x "$link_out"
mv -f "$link_out" "$OUT"
trap - EXIT
echo "$OUT"
