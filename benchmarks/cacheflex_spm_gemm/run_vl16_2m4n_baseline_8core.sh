#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
: "${GEM5_SPM:=$ROOT/gem5/build/ARM_MESI_Three_Level_SPM/gem5.opt}"
: "${GEM5_SE:=$ROOT/gem5/configs/deprecated/example/se.py}"

bin="${CACHEFLEX_GEMM_BIN:-$SCRIPT_DIR/bin_gem5/cache_vl16_2m4n}"
out="${CACHEFLEX_GEMM_OUT:-$ROOT/results/cacheflex_spm_gemm/vl2048_cache_2m4n_ruby_8c_w7}"
gemm_m="${CACHEFLEX_M:-784}"
gemm_k="${CACHEFLEX_K:-256}"
gemm_n="${CACHEFLEX_N:-1024}"
gemm_iters="${CACHEFLEX_ITERS:-1}"
gemm_kc="${CACHEFLEX_KC:-256}"
gemm_operator="${CACHEFLEX_OPERATOR:-gemm}"
input="$out/gemm_fp16_zero_inputs.bin"
mkdir -p "$out"
input_mode="${CACHEFLEX_INPUT_MODE:-zero}"
check_reference=0
case "$input_mode" in
zero)
    truncate -s $(( (gemm_m * gemm_k + gemm_k * gemm_n) * 2 )) "$input"
    ;;
pattern)
    input="$out/w7_fp16_pattern_reference.bin"
    python3 "$SCRIPT_DIR/generate_fp16_correctness_input.py" \
        "$gemm_m" "$gemm_k" "$gemm_n" "$input"
    check_reference=1
    ;;
*)
    echo "unsupported CACHEFLEX_INPUT_MODE=$input_mode" >&2
    exit 2
    ;;
esac

params=()
kernel_start=0
kernel_end=0
b_load_start=0
b_load_end=0
if [[ "${CACHEFLEX_KERNEL_PC_PROFILE:-0}" == 1 ]]; then
    nm_tool="${CROSS_NM:-aarch64-linux-gnu-nm}"
    symbol_bin="${CACHEFLEX_KERNEL_SYMBOL_BIN:-$bin}"
    kernel_start="0x$($nm_tool -n "$symbol_bin" | awk '$3 == "cacheflex_cache_vl16_kernel_begin" {print $1}')"
    kernel_end="0x$($nm_tool -n "$symbol_bin" | awk '$3 == "cacheflex_cache_vl16_kernel_end" {print $1}')"
    b_load_start="0x$($nm_tool -n "$symbol_bin" | awk '$3 == "cacheflex_cache_vl16_b_load_begin" {print $1}')"
    b_load_end="0x$($nm_tool -n "$symbol_bin" | awk '$3 == "cacheflex_cache_vl16_b_load_end" {print $1}')"
    [[ "$kernel_start" != 0x && "$kernel_end" != 0x &&
       "$b_load_start" != 0x && "$b_load_end" != 0x ]] || {
        echo "missing cache-kernel PC symbols in $symbol_bin" >&2
        exit 2
    }
    echo "KERNEL_PC_PROFILE start=$kernel_start end=$kernel_end b_load=$b_load_start:$b_load_end"
fi
for cpu in $(seq 0 7); do
    params+=(
        -P "system.cpu[$cpu].isa[0].sve_vl_se=16"
        -P "system.cpu[$cpu].cacheLoadPorts=2"
        -P "system.cpu[$cpu].cacheStorePorts=1"
        -P "system.cpu[$cpu].spmLoadPorts=1"
        -P "system.cpu[$cpu].spmStorePorts=1"
        -P "system.cpu[$cpu].fetchWidth=5"
        -P "system.cpu[$cpu].decodeWidth=5"
        -P "system.cpu[$cpu].renameWidth=5"
        -P "system.cpu[$cpu].commitWidth=5"
        -P "system.cpu[$cpu].dispatchWidth=8"
        -P "system.cpu[$cpu].issueWidth=8"
        -P "system.cpu[$cpu].wbWidth=8"
        -P "system.cpu[$cpu].numROBEntries=128"
        -P "system.cpu[$cpu].numIQEntries=80"
        -P "system.cpu[$cpu].LQEntries=32"
        -P "system.cpu[$cpu].SQEntries=48"
        -P "system.cpu[$cpu].numPhysIntRegs=128"
        -P "system.cpu[$cpu].numPhysFloatRegs=192"
        -P "system.cpu[$cpu].numPhysVecRegs=192"
        -P "system.cpu[$cpu].numPhysVecPredRegs=64"
        -P "system.cpu[$cpu].fuPool.FUList[0].count=2"
        -P "system.cpu[$cpu].fuPool.FUList[2].count=2"
        -P "system.cpu[$cpu].fuPool.FUList[5].count=2"
        -P "system.cpu[$cpu].fuPool.FUList[7].count=2"
        -P "system.cpu[$cpu].fuPool.FUList[9].count=2"
        -P "system.cpu[$cpu].cacheFlexKernelStartPc=$kernel_start"
        -P "system.cpu[$cpu].cacheFlexKernelEndPc=$kernel_end"
        -P "system.cpu[$cpu].cacheFlexKernelBLoadStartPc=$b_load_start"
        -P "system.cpu[$cpu].cacheFlexKernelBLoadEndPc=$b_load_end"
    )
done

"$GEM5_SPM" --outdir="$out" "$GEM5_SE" \
    --ruby --network=garnet --topology=Mesh_XY --mesh-rows=2 \
    --routing-algorithm=1 --router-latency=1 --link-latency=1 \
    --link-width-bits=128 --vcs-per-vnet=4 \
    --num-cpus=8 --num-l2caches=8 --num-dirs=8 \
    --cpu-type=DerivO3CPU --prog-interval=0Hz \
    --sys-clock=1.5GHz --cpu-clock=2.5GHz --mem-size=2GB \
    --cacheline_size=64 --l0i_size=64kB --l0d_size=64kB \
    --l1d_size=512kB --l2_size=4MB \
    --l0i_assoc=4 --l0d_assoc=4 --l1d_assoc=8 \
    --l1d_num_spm_ways=0 --l2_assoc=16 --enable-prefetch \
    -P "system.readfile='$input'" "${params[@]}" \
    --cmd "$bin" --options \
    "$gemm_m $gemm_k $gemm_n $gemm_iters $gemm_kc $check_reference $gemm_operator" \
    2>&1 | tee "$out/run.log"

echo "$out"
