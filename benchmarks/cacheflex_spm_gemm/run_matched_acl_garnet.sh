#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
: "${GEM5_SPM:=$ROOT/gem5/build/ARM_MESI_Three_Level_SPM/gem5.opt}"
: "${GEM5_SE:=$ROOT/gem5/configs/deprecated/example/se.py}"
: "${PROGRESS_INTERVAL:=50kHz}"
: "${ENABLE_PREFETCH:=1}"
prefetch_args=()
[[ "$ENABLE_PREFETCH" == 1 ]] && prefetch_args+=(--enable-prefetch)

M_SIZE="${1:-64}"
N_SIZE="${2:-192}"
K_SIZE="${3:-128}"
ITERATIONS="${4:-1}"
CORES="${5:-8}"
case "$CORES" in
    1|2) MESH_ROWS=1 ;;
    4|8) MESH_ROWS=2 ;;
    *) echo "cores must be one of: 1 2 4 8" >&2; exit 2 ;;
esac
BIN="$ROOT/benchmarks/acl_sve_gemm/bin_gem5/acl_sve_gemm"
TAG="m${M_SIZE}_n${N_SIZE}_k${K_SIZE}_i${ITERATIONS}"
OUTDIR="$ROOT/results/cacheflex_spm_gemm/mesi3_spmproto_derivo3paperdriverpf_fp16_vl512_garnet_${CORES}c_$TAG"
RUN_LOG="$OUTDIR/run.log"

M_PADDED=$(( (M_SIZE + 7) / 8 * 8 ))
N_PADDED=$(( (N_SIZE + 95) / 96 * 96 ))
INPUT_BYTES=$(( (M_PADDED * K_SIZE + N_PADDED * K_SIZE) * 2 ))
INPUT_FILE="$OUTDIR/zero_inputs.bin"

[[ -x "$GEM5_SPM" ]] || { echo "missing $GEM5_SPM" >&2; exit 1; }
[[ -x "$BIN" ]] || { echo "missing $BIN" >&2; exit 1; }
mkdir -p "$OUTDIR"
truncate -s "$INPUT_BYTES" "$INPUT_FILE"

cpu_args=()
for ((core = 0; core < CORES; core++)); do
    cpu_args+=("-P" "system.cpu[$core].isa[0].sve_vl_se=4")
    cpu_args+=("-P" "system.cpu[$core].cacheLoadPorts=2")
    cpu_args+=("-P" "system.cpu[$core].cacheStorePorts=1")
    cpu_args+=("-P" "system.cpu[$core].spmLoadPorts=2")
    cpu_args+=("-P" "system.cpu[$core].spmStorePorts=1")
    cpu_args+=("-P" "system.cpu[$core].fetchWidth=5")
    cpu_args+=("-P" "system.cpu[$core].decodeWidth=5")
    cpu_args+=("-P" "system.cpu[$core].renameWidth=5")
    cpu_args+=("-P" "system.cpu[$core].commitWidth=5")
    cpu_args+=("-P" "system.cpu[$core].dispatchWidth=8")
    cpu_args+=("-P" "system.cpu[$core].issueWidth=8")
    cpu_args+=("-P" "system.cpu[$core].wbWidth=8")
    cpu_args+=("-P" "system.cpu[$core].numROBEntries=128")
    cpu_args+=("-P" "system.cpu[$core].numIQEntries=80")
    cpu_args+=("-P" "system.cpu[$core].LQEntries=32")
    cpu_args+=("-P" "system.cpu[$core].SQEntries=48")
    cpu_args+=("-P" "system.cpu[$core].numPhysIntRegs=128")
    cpu_args+=("-P" "system.cpu[$core].numPhysFloatRegs=192")
    cpu_args+=("-P" "system.cpu[$core].numPhysVecRegs=192")
    cpu_args+=("-P" "system.cpu[$core].numPhysVecPredRegs=64")
    for fu in 0 2 5 7 9; do
        cpu_args+=("-P" "system.cpu[$core].fuPool.FUList[$fu].count=2")
    done
done

"$GEM5_SPM" --outdir="$OUTDIR" "$GEM5_SE" \
    --ruby --network=garnet --topology=Mesh_XY --mesh-rows="$MESH_ROWS" \
    --routing-algorithm=1 --router-latency=1 --link-latency=1 \
    --link-width-bits=128 --vcs-per-vnet=4 \
    --num-cpus="$CORES" --num-l2caches="$CORES" --num-dirs="$CORES" \
    --cpu-type=DerivO3CPU --prog-interval="$PROGRESS_INTERVAL" \
    --sys-clock=1.5GHz --cpu-clock=2.5GHz --mem-size=2GB \
    --cacheline_size=64 --l0i_size=64kB --l0d_size=64kB \
    --l1d_size=512kB --l2_size=4MB \
    --l0i_assoc=4 --l0d_assoc=4 --l1d_assoc=8 \
    --l1d_num_spm_ways=0 --l2_assoc=16 "${prefetch_args[@]}" \
    -P "system.readfile='$INPUT_FILE'" "${cpu_args[@]}" \
    --cmd "$BIN" --options "$M_SIZE $N_SIZE $K_SIZE $ITERATIONS $CORES" \
    2>&1 | tee "$RUN_LOG"

echo "$OUTDIR"
