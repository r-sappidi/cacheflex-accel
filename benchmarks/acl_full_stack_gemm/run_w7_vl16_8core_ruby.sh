#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONDA_ENV=/data2/rsappidi/conda-envs/cacheflex-gem5
export LD_LIBRARY_PATH="$CONDA_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
GEM5="$ROOT/gem5/build/ARM_MESI_Three_Level_SPM/gem5.opt"
SE="$ROOT/gem5/configs/deprecated/example/se.py"
BIN="$ROOT/benchmarks/acl_full_stack_gemm/w7_fp16_acl"
OUT="$ROOT/results/cacheflex_spm_gemm/acl_fullstack_fp16_vl2048_ruby_8c_w7"
mkdir -p "$OUT"

cpu_args=()
for ((core=0; core<8; core++)); do
    cpu_args+=(
        -P "system.cpu[$core].isa[0].sve_vl_se=16"
        -P "system.cpu[$core].cacheLoadPorts=2"
        -P "system.cpu[$core].cacheStorePorts=1"
        -P "system.cpu[$core].spmLoadPorts=1"
        -P "system.cpu[$core].spmStorePorts=1"
        -P "system.cpu[$core].fetchWidth=5"
        -P "system.cpu[$core].decodeWidth=5"
        -P "system.cpu[$core].renameWidth=5"
        -P "system.cpu[$core].commitWidth=5"
        -P "system.cpu[$core].dispatchWidth=8"
        -P "system.cpu[$core].issueWidth=8"
        -P "system.cpu[$core].wbWidth=8"
        -P "system.cpu[$core].numROBEntries=128"
        -P "system.cpu[$core].numIQEntries=80"
        -P "system.cpu[$core].LQEntries=32"
        -P "system.cpu[$core].SQEntries=48"
        -P "system.cpu[$core].numPhysIntRegs=128"
        -P "system.cpu[$core].numPhysFloatRegs=192"
        -P "system.cpu[$core].numPhysVecRegs=192"
        -P "system.cpu[$core].numPhysVecPredRegs=64"
    )
    for fu in 0 2 5 7 9; do
        cpu_args+=(-P "system.cpu[$core].fuPool.FUList[$fu].count=2")
    done
done

"$GEM5" --outdir="$OUT" "$SE" \
    --ruby --network=garnet --topology=Mesh_XY --mesh-rows=2 \
    --routing-algorithm=1 --router-latency=1 --link-latency=1 \
    --link-width-bits=128 --vcs-per-vnet=4 \
    --num-cpus=8 --num-l2caches=8 --num-dirs=8 \
    --cpu-type=DerivO3CPU --prog-interval=50kHz \
    --sys-clock=1.5GHz --cpu-clock=2.5GHz --mem-size=2GB \
    --cacheline_size=64 --l0i_size=64kB --l0d_size=64kB \
    --l1d_size=512kB --l2_size=4MB \
    --l0i_assoc=4 --l0d_assoc=4 --l1d_assoc=8 \
    --l1d_num_spm_ways=0 --l2_assoc=16 --enable-prefetch \
    "${cpu_args[@]}" --cmd "$BIN" 2>&1 | tee "$OUT/run.log"
