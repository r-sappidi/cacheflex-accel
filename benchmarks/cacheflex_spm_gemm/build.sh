#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$ROOT/setup_spm_env.sh" >/dev/null

OUTDIR="$SCRIPT_DIR/bin_gem5"
OUT="$OUTDIR/cacheflex_spm_gemm"
mkdir -p "$OUTDIR"

objects=()
for unit in gemm_driver fp16_kernel; do
    asm="$OUTDIR/${unit}.s"
    enc="$OUTDIR/${unit}_enc.s"
    obj="$OUTDIR/${unit}.o"
    "$CROSS_CXX" \
        -std=c++17 -O3 -march=armv8.2-a+sve+fp16 -mbranch-protection=none \
        -DARM_COMPUTE_ENABLE_SVE -DCACHEFLEX_SPM_GEMM -DGEM5 -DGEM5_SE \
        -I"$M5_INCLUDE" -S -o "$asm" \
        "$ROOT/benchmarks/acl_sve_gemm/${unit}.cpp"
    python3 "$SPM_COMPILER" "$asm" "$enc"
    "$CROSS_CXX" -c -o "$obj" "$enc"
    objects+=("$obj")
done
link_out="$(mktemp "${TMPDIR:-/tmp}/cacheflex-spm-gemm.XXXXXX")"
trap 'rm -f "$link_out"' EXIT
"$CROSS_CXX" -static -pthread -o "$link_out" \
    "${objects[@]}" "$M5OP_OBJ" -lm
chmod +x "$link_out"
mv -f "$link_out" "$OUT"
trap - EXIT
echo "$OUT"
