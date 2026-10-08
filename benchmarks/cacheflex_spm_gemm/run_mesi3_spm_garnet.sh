#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
: "${GEM5_SPM:=$ROOT/gem5/build/ARM_MESI_Three_Level_SPM/gem5.opt}"
: "${GEM5_SE:=$ROOT/gem5/configs/deprecated/example/se.py}"
: "${PROGRESS_INTERVAL:=50kHz}"
: "${ENABLE_PREFETCH:=1}"
: "${PRIVATE_L2_ASSOC:=8}"
: "${PRIVATE_L2_SIZE:=512kB}"
: "${L1_SIZE:=64kB}"
: "${L1_ASSOC:=4}"
: "${LLC_SIZE:=4MB}"
: "${LLC_ASSOC:=16}"
prefetch_args=()
[[ "$ENABLE_PREFETCH" == 1 ]] && prefetch_args+=(--enable-prefetch)
debug_args=()
[[ -n "${GEM5_DEBUG_FLAGS:-}" ]] && debug_args+=(--debug-flags="$GEM5_DEBUG_FLAGS")
[[ -n "${GEM5_DEBUG_FILE:-}" ]] && debug_args+=(--debug-file="$GEM5_DEBUG_FILE")

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
SVE_VL_BITS=512
SVE_VL_QUADWORDS=4
TILE_N=96
BIN="$SCRIPT_DIR/bin_gem5/cacheflex_spm_gemm"
TAG="m${M_SIZE}_n${N_SIZE}_k${K_SIZE}_i${ITERATIONS}"
OUTDIR="$ROOT/results/cacheflex_spm_gemm/mesi3_spm_derivo3paperdriverpf_fp16_vl512_garnet_${CORES}c_$TAG"
RUN_LOG="$OUTDIR/run.log"

M_PADDED=$(( (M_SIZE + 7) / 8 * 8 ))
N_PADDED=$(( (N_SIZE + TILE_N - 1) / TILE_N * TILE_N ))
INPUT_BYTES=$(( (M_PADDED * K_SIZE + N_PADDED * K_SIZE) * 2 ))
INPUT_FILE="$OUTDIR/zero_inputs.bin"

[[ -x "$GEM5_SPM" ]] || { echo "missing $GEM5_SPM" >&2; exit 1; }
[[ -x "$BIN" ]] || { echo "missing $BIN; run $SCRIPT_DIR/build.sh" >&2; exit 1; }
mkdir -p "$OUTDIR"
truncate -s "$INPUT_BYTES" "$INPUT_FILE"

sve_args=()
for ((core = 0; core < CORES; core++)); do
    sve_args+=("-P" "system.cpu[$core].isa[0].sve_vl_se=$SVE_VL_QUADWORDS")
    sve_args+=("-P" "system.cpu[$core].cacheLoadPorts=2")
    sve_args+=("-P" "system.cpu[$core].cacheStorePorts=1")
    sve_args+=("-P" "system.cpu[$core].spmLoadPorts=2")
    sve_args+=("-P" "system.cpu[$core].spmStorePorts=1")
    sve_args+=("-P" "system.cpu[$core].fetchWidth=5")
    sve_args+=("-P" "system.cpu[$core].decodeWidth=5")
    sve_args+=("-P" "system.cpu[$core].renameWidth=5")
    sve_args+=("-P" "system.cpu[$core].commitWidth=5")
    sve_args+=("-P" "system.cpu[$core].dispatchWidth=8")
    sve_args+=("-P" "system.cpu[$core].issueWidth=8")
    sve_args+=("-P" "system.cpu[$core].wbWidth=8")
    sve_args+=("-P" "system.cpu[$core].numROBEntries=128")
    sve_args+=("-P" "system.cpu[$core].numIQEntries=80")
    sve_args+=("-P" "system.cpu[$core].LQEntries=32")
    sve_args+=("-P" "system.cpu[$core].SQEntries=48")
    sve_args+=("-P" "system.cpu[$core].numPhysIntRegs=128")
    sve_args+=("-P" "system.cpu[$core].numPhysFloatRegs=192")
    sve_args+=("-P" "system.cpu[$core].numPhysVecRegs=192")
    sve_args+=("-P" "system.cpu[$core].numPhysVecPredRegs=64")
    for fu in 0 2 5 7 9; do
        sve_args+=("-P" "system.cpu[$core].fuPool.FUList[$fu].count=2")
    done
done

"$GEM5_SPM" "${debug_args[@]}" --outdir="$OUTDIR" "$GEM5_SE" \
    --ruby --network=garnet --topology=Mesh_XY --mesh-rows="$MESH_ROWS" \
    --routing-algorithm=1 --router-latency=1 --link-latency=1 \
    --link-width-bits=128 --vcs-per-vnet=4 \
    --num-cpus="$CORES" --num-l2caches="$CORES" --num-dirs="$CORES" \
    --cpu-type=DerivO3CPU --prog-interval="$PROGRESS_INTERVAL" \
    --sys-clock=1.5GHz --cpu-clock=2.5GHz --mem-size=2GB \
    --cacheline_size=64 --l0i_size="$L1_SIZE" --l0d_size="$L1_SIZE" \
    --l1d_size="$PRIVATE_L2_SIZE" --l2_size="$LLC_SIZE" \
    --l0i_assoc="$L1_ASSOC" --l0d_assoc="$L1_ASSOC" \
    --l1d_assoc="$PRIVATE_L2_ASSOC" \
    --l1d_num_spm_ways=3 --l2_assoc="$LLC_ASSOC" "${prefetch_args[@]}" \
    -P "system.readfile='$INPUT_FILE'" "${sve_args[@]}" \
    --cmd "$BIN" --options "$M_SIZE $N_SIZE $K_SIZE $ITERATIONS $CORES" \
    2>&1 | tee "$RUN_LOG"

echo "$OUTDIR"
