#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
variant="${1:?usage: $0 cache|spm|multicast [T Br Bc]}"
T="${2:-256}"; Br="${3:-64}"; Bc="${4:-96}"
case "$variant" in
  cache) bin="$SCRIPT_DIR/bin/cache_flash_multicore"; ways=0;;
  cache_pretouch) bin="$SCRIPT_DIR/bin/cache_pretouch_flash_multicore"; ways=0;;
  spm) bin="$SCRIPT_DIR/bin/spm_flash_multicore"; ways=3;;
  spm_pretouch) bin="$SCRIPT_DIR/bin/spm_pretouch_flash_multicore"; ways=3;;
  spm_group1) bin="$SCRIPT_DIR/bin/spm_group1_flash_multicore"; ways=3;;
  spm_group2) bin="$SCRIPT_DIR/bin/spm_group2_flash_multicore"; ways=3;;
  spm_group4) bin="$SCRIPT_DIR/bin/spm_group4_flash_multicore"; ways=3;;
  spm_konly) bin="$SCRIPT_DIR/bin/spm_konly_flash_multicore"; ways=3;;
  spm_konly_overlap) bin="$SCRIPT_DIR/bin/spm_konly_overlap_flash_multicore"; ways=3;;
  spm_konly_overlap_multicast) bin="$SCRIPT_DIR/bin/spm_konly_overlap_multicast_flash_multicore"; ways=3;;
  spm_konly_multicast_pinall) bin="$SCRIPT_DIR/bin/spm_konly_multicast_pinall_flash_multicore"; ways=3;;
  spm_kv_multicast_pinall) bin="$SCRIPT_DIR/bin/spm_kv_multicast_pinall_flash_multicore"; ways=6;;
  spm_kv_dma_pinall) bin="$SCRIPT_DIR/bin/spm_kv_dma_pinall_flash_multicore"; ways=6;;
  spm_konly_multicast_ready) bin="$SCRIPT_DIR/bin/spm_konly_multicast_ready_flash_multicore"; ways=3;;
  spm_konly_multicast_ready_interleave) bin="$SCRIPT_DIR/bin/spm_konly_multicast_ready_interleave_flash_multicore"; ways=3;;
  spm_konly_multicast_ready_batch3) bin="$SCRIPT_DIR/bin/spm_konly_multicast_ready_batch3_flash_multicore"; ways=3;;
  spm_konly_multicast_ready_dma) bin="$SCRIPT_DIR/bin/spm_konly_multicast_ready_dma_flash_multicore"; ways=3;;
  spm_konly_multicast_ready_dma_interleave) bin="$SCRIPT_DIR/bin/spm_konly_multicast_ready_dma_interleave_flash_multicore"; ways=3;;
  spm_konly_dma_pinall_interleave) bin="$SCRIPT_DIR/bin/spm_konly_dma_pinall_interleave_flash_multicore"; ways=3;;
  spm_vonly) bin="$SCRIPT_DIR/bin/spm_vonly_flash_multicore"; ways=2;;
  multicast) bin="$SCRIPT_DIR/bin/spm_multicast_flash_multicore"; ways=3;;
  multicast_preclaim) bin="$SCRIPT_DIR/bin/spm_multicast_preclaim_flash_multicore"; ways=3;;
  *) exit 2;;
esac
ways="${CACHEFLEX_FLASH_SPM_WAYS:-$ways}"
bin="${CACHEFLEX_FLASH_BIN:-$bin}"
rob="${CACHEFLEX_FLASH_ROB:-128}"
iq="${CACHEFLEX_FLASH_IQ:-80}"
lq="${CACHEFLEX_FLASH_LQ:-32}"
sq="${CACHEFLEX_FLASH_SQ:-48}"
cache_load_ports="${CACHEFLEX_FLASH_CACHE_LOAD_PORTS:-2}"
cache_store_ports="${CACHEFLEX_FLASH_CACHE_STORE_PORTS:-1}"
spm_load_ports="${CACHEFLEX_FLASH_SPM_LOAD_PORTS:-2}"
spm_store_ports="${CACHEFLEX_FLASH_SPM_STORE_PORTS:-1}"
private_l2_size="${CACHEFLEX_FLASH_PRIVATE_L2_SIZE:-512kB}"
private_l2_assoc="${CACHEFLEX_FLASH_PRIVATE_L2_ASSOC:-8}"
l0d_size="${CACHEFLEX_FLASH_L0D_SIZE:-64kB}"
l0d_assoc="${CACHEFLEX_FLASH_L0D_ASSOC:-4}"
sequencer_outstanding="${CACHEFLEX_FLASH_SEQUENCER_OUTSTANDING:-}"
ideal_simd="${CACHEFLEX_FLASH_IDEAL_SIMD:-0}"
sve_vl="${CACHEFLEX_FLASH_VL:-4}"
out="${CACHEFLEX_FLASH_OUT:-$ROOT/results/cacheflex_flash_attention/${variant}_vl$((sve_vl*128))_8c_t${T}_br${Br}_bc${Bc}}"
mkdir -p "$out"
params=()
if [[ "${CACHEFLEX_FLASH_SPM_SEPARATE_BACKING:-0}" == 1 ]]; then
 for ctrl in $(seq 0 7); do
  params+=(-P "system.ruby.l1_cntrl$ctrl.cache.spm_separate_backing=True")
 done
fi
if [[ -n "${CACHEFLEX_FLASH_READFILE:-}" ]]; then
 params+=(-P "system.readfile='$CACHEFLEX_FLASH_READFILE'")
fi
if [[ "${CACHEFLEX_FLASH_PROFILE:-0}" == 1 ]]; then
 params+=( -P "system.ruby.network.cacheflex_message_profile=True" )
 for cpu in $(seq 0 7); do
  params+=( -P "system.cpu[$cpu].cacheFlexPcProfile=True" )
 done
fi
if [[ "${CACHEFLEX_FLASH_FAST_MEMCTRL:-0}" == 1 ]]; then
 for ctrl in $(seq 0 7); do
  params+=( -P "system.mem_ctrls[$ctrl].static_frontend_latency='1ns'"
            -P "system.mem_ctrls[$ctrl].static_backend_latency='1ns'" )
 done
fi
if [[ "${CACHEFLEX_FLASH_FAST_DRAM:-0}" == 1 ]]; then
 for ctrl in $(seq 0 7); do
  for timing in tRCD tRCD_WR tCL tRP; do
   params+=( -P "system.mem_ctrls[$ctrl].dram.$timing='1ns'" )
  done
  params+=( -P "system.mem_ctrls[$ctrl].static_frontend_latency='1ns'"
            -P "system.mem_ctrls[$ctrl].static_backend_latency='1ns'" )
 done
fi
for cpu in $(seq 0 7); do
 params+=(
  -P "system.cpu[$cpu].isa[0].sve_vl_se=$sve_vl"
  -P "system.cpu[$cpu].cacheLoadPorts=$cache_load_ports" -P "system.cpu[$cpu].cacheStorePorts=$cache_store_ports"
  -P "system.cpu[$cpu].spmLoadPorts=$spm_load_ports" -P "system.cpu[$cpu].spmStorePorts=$spm_store_ports"
  -P "system.cpu[$cpu].fetchWidth=5" -P "system.cpu[$cpu].decodeWidth=5"
  -P "system.cpu[$cpu].renameWidth=5" -P "system.cpu[$cpu].commitWidth=5"
  -P "system.cpu[$cpu].dispatchWidth=8" -P "system.cpu[$cpu].issueWidth=8"
  -P "system.cpu[$cpu].wbWidth=8" -P "system.cpu[$cpu].numROBEntries=$rob"
  -P "system.cpu[$cpu].numIQEntries=$iq" -P "system.cpu[$cpu].LQEntries=$lq"
  -P "system.cpu[$cpu].SQEntries=$sq" -P "system.cpu[$cpu].numPhysIntRegs=128"
  -P "system.cpu[$cpu].numPhysFloatRegs=192" -P "system.cpu[$cpu].numPhysVecRegs=192"
  -P "system.cpu[$cpu].numPhysVecPredRegs=64"
 )
done
if [[ "$ideal_simd" == 1 ]]; then
 for cpu in $(seq 0 7); do
  # One-cycle upper bound for the vector arithmetic/reduction unit.  This is
  # an ablation control, not a proposed paper configuration.
  for op in $(seq 13 30); do
   params+=( -P "system.cpu[$cpu].fuPool.FUList[5].opList[$op].opLat=1" )
  done
 done
fi
if [[ -n "$sequencer_outstanding" ]]; then
 for ctrl in $(seq 0 7); do
  params+=(
   -P "system.ruby.l0_cntrl$ctrl.sequencer.max_outstanding_requests=$sequencer_outstanding"
  )
 done
fi
export LD_LIBRARY_PATH="/data2/rsappidi/conda-envs/cacheflex-gem5/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
prefetch_args=()
if [[ "${CACHEFLEX_FLASH_MATCHED_LRU:-0}" == 1 ]]; then prefetch_args+=(--l1d-force-lru); fi
if [[ "${CACHEFLEX_FLASH_PREFETCH:-1}" == 1 ]]; then prefetch_args+=(--enable-prefetch); fi
debug_args=()
if [[ -n "${CACHEFLEX_FLASH_DEBUG_FLAGS:-}" ]]; then
 debug_args+=(--debug-flags="$CACHEFLEX_FLASH_DEBUG_FLAGS" --debug-file=debug.trace)
fi
"${CACHEFLEX_GEM5_BIN:-$ROOT/gem5/build/ARM_MESI_Three_Level_SPM/gem5.opt}" "${debug_args[@]}" --outdir="$out" \
 "$ROOT/gem5/configs/deprecated/example/se.py" --ruby --network=garnet \
 --topology=Mesh_XY --mesh-rows=2 --routing-algorithm=1 --router-latency=1 \
 --link-latency=1 --link-width-bits="${CACHEFLEX_FLASH_LINK_BITS:-128}" --vcs-per-vnet=4 --num-cpus=8 \
 --num-l2caches=8 --num-dirs=8 --cpu-type=DerivO3CPU --prog-interval=0Hz \
 --sys-clock=1.5GHz --cpu-clock=2.5GHz --mem-size=2GB --cacheline_size=64 \
 --l0i_size=64kB --l0d_size="$l0d_size" --l1d_size="$private_l2_size" --l2_size=4MB \
 --l0i_assoc=4 --l0d_assoc="$l0d_assoc" --l1d_assoc="$private_l2_assoc" --l1d_num_spm_ways="$ways" \
 --l2_assoc=16 "${prefetch_args[@]}" "${params[@]}" --cmd "$bin" \
 --options "${CACHEFLEX_FLASH_OPTIONS:-$T 64 $Br $Bc}" 2>&1 | tee "$out/run.log"
echo "$out"
