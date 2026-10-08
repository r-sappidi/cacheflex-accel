#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

: "${GEM5_MESI3:=$ROOT/gem5/build/ARM_MESI_Three_Level/gem5.opt}"
: "${GEM5_SE:=$ROOT/gem5/configs/deprecated/example/se.py}"

M_SIZE="${1:-64}"
N_SIZE="${2:-192}"
K_SIZE="${3:-128}"
ITERATIONS="${4:-1}"
: "${PROGRESS_INTERVAL:=50kHz}"
: "${ENABLE_FDP:=1}"
: "${SVE_VL_BITS:=128}"
if [[ "$ENABLE_FDP" != 0 && "$ENABLE_FDP" != 1 ]]; then
    echo "ENABLE_FDP must be 0 or 1" >&2
    exit 2
fi
if (( SVE_VL_BITS < 128 || SVE_VL_BITS > 2048 || SVE_VL_BITS % 128 != 0 )); then
    echo "SVE_VL_BITS must be a multiple of 128 from 128 through 2048" >&2
    exit 2
fi
SVE_VL_QUADWORDS=$(( SVE_VL_BITS / 128 ))
SVE_FP32_LANES=$(( SVE_VL_BITS / 32 ))
TILE_N=$(( 3 * SVE_FP32_LANES ))
BIN="$SCRIPT_DIR/bin_gem5/acl_sve_gemm"
TAG="m${M_SIZE}_n${N_SIZE}_k${K_SIZE}_i${ITERATIONS}"
OUTDIR="$ROOT/results/acl_sve_gemm/mesi3_neoversev2_fdp${ENABLE_FDP}_vl${SVE_VL_BITS}_garnet_8c_$TAG"
RUN_LOG="$OUTDIR/run.log"

# A and B are loaded through m5_read_file before the ROI. A sparse zero file
# avoids instruction-accurately simulating synthetic test-data generation.
M_PADDED=$(( (M_SIZE + 7) / 8 * 8 ))
N_PADDED=$(( (N_SIZE + TILE_N - 1) / TILE_N * TILE_N ))
INPUT_BYTES=$(( (M_PADDED * K_SIZE + N_PADDED * K_SIZE) * 4 ))
INPUT_FILE="$OUTDIR/zero_inputs.bin"

if [ ! -x "$GEM5_MESI3" ]; then
    echo "missing $GEM5_MESI3; build ARM_MESI_Three_Level first" >&2
    exit 1
fi
if [ ! -x "$BIN" ]; then
    echo "missing $BIN; run benchmarks/acl_sve_gemm/build.sh first" >&2
    exit 1
fi

mkdir -p "$OUTDIR"
truncate -s "$INPUT_BYTES" "$INPUT_FILE"
echo "Progress log: $RUN_LOG"
echo "gem5 CPU progress interval: $PROGRESS_INTERVAL"
echo "Neoverse V2 decoupled front end enabled: $ENABLE_FDP"
echo "SVE vector length: $SVE_VL_BITS bits ($SVE_FP32_LANES FP32 lanes)"
sve_args=()
for ((core = 0; core < 8; core++)); do
    sve_args+=("-P" "system.cpu[$core].isa[0].sve_vl_se=$SVE_VL_QUADWORDS")
    if [[ "$ENABLE_FDP" == 0 ]]; then
        sve_args+=("-P" "system.cpu[$core].decoupledFrontEnd=False")
    fi
done

"$GEM5_MESI3" \
    --outdir="$OUTDIR" \
    "$GEM5_SE" \
    --ruby \
    --network=garnet \
    --topology=Mesh_XY \
    --mesh-rows=2 \
    --routing-algorithm=1 \
    --router-latency=1 \
    --link-latency=1 \
    --link-width-bits=128 \
    --vcs-per-vnet=4 \
    --num-cpus=8 \
    --num-l2caches=8 \
    --num-dirs=8 \
    --cpu-type=NeoverseV2 \
    --prog-interval="$PROGRESS_INTERVAL" \
    --sys-clock=1.5GHz \
    --cpu-clock=1.5GHz \
    --mem-size=2GB \
    --cacheline_size=64 \
    --l1i_size=64kB \
    --l1d_size=256kB \
    --l2_size=256kB \
    --l1i_assoc=4 \
    --l1d_assoc=16 \
    --l2_assoc=16 \
    -P "system.readfile='$INPUT_FILE'" \
    "${sve_args[@]}" \
    --cmd "$BIN" \
    --options "$M_SIZE $N_SIZE $K_SIZE $ITERATIONS" \
    2>&1 | tee "$RUN_LOG"

echo "$OUTDIR"
