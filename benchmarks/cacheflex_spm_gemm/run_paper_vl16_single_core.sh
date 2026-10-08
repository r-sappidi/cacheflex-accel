#!/usr/bin/env bash
set -euo pipefail

# Run the exact paper VL=16 tile-shape driver on this repository's Ruby/Garnet
# system. Usage: run_paper_vl16_single_core.sh baseline|spm

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
: "${GEM5_SPM:=$ROOT/gem5/build/ARM_MESI_Three_Level_SPM/gem5.opt}"
: "${GEM5_SE:=$ROOT/gem5/configs/deprecated/example/se.py}"

variant="${1:?usage: $0 baseline|spm}"
case "$variant" in
    baseline)
        bin="$SCRIPT_DIR/bin_gem5/paper_vl16_baseline"
        spm_ways=0
        ;;
    spm)
        bin="$SCRIPT_DIR/bin_gem5/paper_vl16_spm"
        # The paper's VL=16 24x1VL layout stripes each 256-byte vector over
        # four physical 64-byte ways. VL=4 uses only three ways.
        spm_ways=4
        ;;
    *) echo "variant must be baseline or spm" >&2; exit 2 ;;
esac

if [[ "$variant" == spm ]]; then
    result_variant=spm4way
else
    result_variant=baseline
fi
out="$ROOT/results/cacheflex_spm_gemm/paperdriver_vl2048_${result_variant}_ruby_1c_w7"
mkdir -p "$out"

"$GEM5_SPM" --outdir="$out" "$GEM5_SE" \
    --ruby --network=garnet --topology=Mesh_XY --mesh-rows=1 \
    --routing-algorithm=1 --router-latency=1 --link-latency=1 \
    --link-width-bits=128 --vcs-per-vnet=4 \
    --num-cpus=1 --num-l2caches=1 --num-dirs=1 \
    --cpu-type=DerivO3CPU --prog-interval=0Hz \
    --sys-clock=1.5GHz --cpu-clock=2.5GHz --mem-size=2GB \
    --cacheline_size=64 --l0i_size=64kB --l0d_size=64kB \
    --l1d_size=512kB --l2_size=4MB \
    --l0i_assoc=4 --l0d_assoc=4 --l1d_assoc=8 \
    --l1d_num_spm_ways="$spm_ways" --l2_assoc=16 --enable-prefetch \
    -P 'system.cpu[0].isa[0].sve_vl_se=16' \
    -P 'system.cpu[0].cacheLoadPorts=2' \
    -P 'system.cpu[0].cacheStorePorts=1' \
    -P 'system.cpu[0].spmLoadPorts=1' \
    -P 'system.cpu[0].spmStorePorts=1' \
    -P 'system.cpu[0].fetchWidth=5' \
    -P 'system.cpu[0].decodeWidth=5' \
    -P 'system.cpu[0].renameWidth=5' \
    -P 'system.cpu[0].commitWidth=5' \
    -P 'system.cpu[0].dispatchWidth=8' \
    -P 'system.cpu[0].issueWidth=8' \
    -P 'system.cpu[0].wbWidth=8' \
    -P 'system.cpu[0].numROBEntries=128' \
    -P 'system.cpu[0].numIQEntries=80' \
    -P 'system.cpu[0].LQEntries=32' \
    -P 'system.cpu[0].SQEntries=48' \
    -P 'system.cpu[0].numPhysIntRegs=128' \
    -P 'system.cpu[0].numPhysFloatRegs=192' \
    -P 'system.cpu[0].numPhysVecRegs=192' \
    -P 'system.cpu[0].numPhysVecPredRegs=64' \
    -P 'system.cpu[0].fuPool.FUList[0].count=2' \
    -P 'system.cpu[0].fuPool.FUList[2].count=2' \
    -P 'system.cpu[0].fuPool.FUList[5].count=2' \
    -P 'system.cpu[0].fuPool.FUList[7].count=2' \
    -P 'system.cpu[0].fuPool.FUList[9].count=2' \
    --cmd "$bin" --options '784 256 1024 1 256 784' \
    2>&1 | tee "$out/run.log"

echo "$out"
