#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
out="${1:?fresh absolute output directory}"
mkdir "$out"
source "$root/setup_spm_env.sh" > /dev/null
extra=()
if [[ "${2:-}" == "--no-wb-barrier" ]]; then
    extra+=(-DROUNDTRIP_NO_WB_BARRIER)
fi
"$CROSS_CXX" -O3 -std=c++17 -static -pthread -march=armv8.2-a+sve+fp16 \
    -mbranch-protection=none -DGEM5 -DGEM5_SE -I"$M5_INCLUDE" "${extra[@]}" \
    -S "$root/benchmarks/spm_smoke/spm_roundtrip.cpp" -o "$out/guest.s"
python3 "$SPM_COMPILER" "$out/guest.s" "$out/guest.enc.s"
"$CROSS_CXX" -static -pthread "$out/guest.enc.s" "$M5OP_OBJ" -lm -o "$out/guest"
