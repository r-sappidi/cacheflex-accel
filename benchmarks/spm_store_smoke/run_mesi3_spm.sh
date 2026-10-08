#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

: "${GEM5_SPM:=$ROOT/gem5/build/ARM_MESI_Three_Level_SPM/gem5.opt}"
: "${GEM5_SE:=$ROOT/gem5/configs/deprecated/example/se.py}"

BIN="$SCRIPT_DIR/bin_gem5/spm_store_smoke"
OUTDIR="${1:-$ROOT/results/spm_store_smoke}"

if [ ! -x "$BIN" ]; then
    echo "missing $BIN; run benchmarks/spm_store_smoke/build.sh first" >&2
    exit 1
fi

mkdir -p "$OUTDIR"
"$GEM5_SPM" \
    --outdir="$OUTDIR" \
    "$GEM5_SE" \
    --ruby \
    --num-cpus=1 \
    --cpu-type=DerivO3CPU \
    --sys-clock=1.5GHz \
    --cpu-clock=1.5GHz \
    --mem-size=2GB \
    --cacheline_size=64 \
    --l1i_size=64kB \
    --l1d_size=256kB \
    --l2_size=512kB \
    --l1i_assoc=4 \
    --l1d_assoc=16 \
    --l1d_num_spm_ways=1 \
    --l2_assoc=8 \
    -P system.cpu[0].isa[0].sve_vl_se=4 \
    --cmd "$BIN"

echo "$OUTDIR"
