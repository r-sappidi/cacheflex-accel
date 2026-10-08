#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$ROOT/setup_spm_env.sh" >/dev/null
mkdir -p "$SCRIPT_DIR/bin"

"$CROSS_CXX" -std=c++17 -O3 -march=armv8.2-a+sve+fp16 \
  -mbranch-protection=none -DGEM5 -DGEM5_SE -I"$M5_INCLUDE" \
  -static -pthread "$SCRIPT_DIR/cache_multicore.cpp" "$M5OP_OBJ" -lm \
  -o "$SCRIPT_DIR/bin/cache_flash_multicore"

echo "$SCRIPT_DIR/bin/cache_flash_multicore"

"$CROSS_CXX" -std=c++17 -O3 -march=armv8.2-a+sve+fp16 \
  -mbranch-protection=none -DGEM5 -DGEM5_SE -DCACHEFLEX_FLASH_PRETOUCH=1 -I"$M5_INCLUDE" \
  -static -pthread "$SCRIPT_DIR/cache_multicore.cpp" "$M5OP_OBJ" -lm \
  -o "$SCRIPT_DIR/bin/cache_pretouch_flash_multicore"

build_spm() {
  local name="$1" define="$2"
  local asm="$SCRIPT_DIR/bin/${name}.s" enc="$SCRIPT_DIR/bin/${name}_enc.s"
  "$CROSS_CXX" -std=c++17 -O3 -march=armv8.2-a+sve+fp16 \
    -mbranch-protection=none -DGEM5 -DGEM5_SE -DVL_4 $define \
    -I"$M5_INCLUDE" -I"$ROOT/../cacheflex_micro/kernels/llama_bench_spm" \
    -I"$ROOT/../cacheflex_micro/kernels/attention_kernels_spm" \
    -S "$SCRIPT_DIR/spm_multicore.cpp" -o "$asm"
  python3 "$SPM_COMPILER" "$asm" "$enc"
  "$CROSS_CXX" -std=c++17 -O3 -march=armv8.2-a+sve+fp16 -mbranch-protection=none \
    -DGEM5 -DGEM5_SE -I"$M5_INCLUDE" -I"$ROOT/../cacheflex_micro/experiments/flash_attention/src" \
    -c "$SCRIPT_DIR/cache_qk_wrapper.cpp" -o "$SCRIPT_DIR/bin/cache_qk_wrapper.o"
  "$CROSS_CXX" -static -pthread "$enc" "$SCRIPT_DIR/bin/cache_qk_wrapper.o" "$M5OP_OBJ" -lm -o "$SCRIPT_DIR/bin/$name"
  echo "$SCRIPT_DIR/bin/$name"
}

build_spm spm_flash_multicore ""
build_spm spm_multicast_flash_multicore "-DCACHEFLEX_FLASH_MULTICAST=1"
build_spm spm_multicast_preclaim_flash_multicore "-DCACHEFLEX_FLASH_MULTICAST=1 -DCACHEFLEX_FLASH_PRECLAIM=1"
build_spm spm_pretouch_flash_multicore "-DCACHEFLEX_FLASH_PRETOUCH=1"
build_spm spm_group1_flash_multicore "-DCACHEFLEX_FLASH_PRETOUCH=1 -DCACHEFLEX_FLASH_IB_GROUP=1"
build_spm spm_group2_flash_multicore "-DCACHEFLEX_FLASH_PRETOUCH=1 -DCACHEFLEX_FLASH_IB_GROUP=2"
build_spm spm_group4_flash_multicore "-DCACHEFLEX_FLASH_PRETOUCH=1 -DCACHEFLEX_FLASH_IB_GROUP=4"
build_spm spm_konly_flash_multicore "-DCACHEFLEX_FLASH_PRETOUCH=1 -DCACHEFLEX_FLASH_STAGE_V=0"
build_spm spm_konly_overlap_flash_multicore "-DCACHEFLEX_FLASH_PRETOUCH=1 -DCACHEFLEX_FLASH_STAGE_V=0 -DCACHEFLEX_FLASH_K_OVERLAP=1"
build_spm spm_konly_overlap_multicast_flash_multicore "-DCACHEFLEX_FLASH_PRETOUCH=1 -DCACHEFLEX_FLASH_STAGE_V=0 -DCACHEFLEX_FLASH_K_OVERLAP=1 -DCACHEFLEX_FLASH_MULTICAST=1"
build_spm spm_konly_multicast_pinall_flash_multicore "-DCACHEFLEX_FLASH_PRETOUCH=1 -DCACHEFLEX_FLASH_STAGE_V=0 -DCACHEFLEX_FLASH_K_OVERLAP=1 -DCACHEFLEX_FLASH_MULTICAST=1 -DCACHEFLEX_FLASH_MULTICAST_PIN_ALL=1"
build_spm spm_kv_multicast_pinall_flash_multicore "-DCACHEFLEX_FLASH_PRETOUCH=1 -DCACHEFLEX_FLASH_STAGE_V=1 -DCACHEFLEX_FLASH_K_OVERLAP=1 -DCACHEFLEX_FLASH_MULTICAST=1 -DCACHEFLEX_FLASH_MULTICAST_PIN_ALL=1 -DCACHEFLEX_FLASH_PIN_KV=1 -DCACHEFLEX_FLASH_INTERLEAVE_IB=1"
build_spm spm_kv_dma_pinall_flash_multicore "-DCACHEFLEX_FLASH_PRETOUCH=1 -DCACHEFLEX_FLASH_STAGE_V=1 -DCACHEFLEX_FLASH_K_OVERLAP=1 -DCACHEFLEX_FLASH_MULTICAST=1 -DCACHEFLEX_FLASH_MULTICAST_PIN_ALL=1 -DCACHEFLEX_FLASH_PIN_KV=1 -DCACHEFLEX_FLASH_INTERLEAVE_IB=1 -DCACHEFLEX_FLASH_DMA=1"
build_spm spm_konly_multicast_ready_flash_multicore "-DCACHEFLEX_FLASH_PRETOUCH=1 -DCACHEFLEX_FLASH_STAGE_V=0 -DCACHEFLEX_FLASH_K_OVERLAP=1 -DCACHEFLEX_FLASH_MULTICAST=1 -DCACHEFLEX_FLASH_MULTICAST_READY=1"
build_spm spm_konly_multicast_ready_interleave_flash_multicore "-DCACHEFLEX_FLASH_PRETOUCH=1 -DCACHEFLEX_FLASH_STAGE_V=0 -DCACHEFLEX_FLASH_K_OVERLAP=1 -DCACHEFLEX_FLASH_MULTICAST=1 -DCACHEFLEX_FLASH_MULTICAST_READY=1 -DCACHEFLEX_FLASH_INTERLEAVE_IB=1"
build_spm spm_konly_multicast_ready_batch3_flash_multicore "-DCACHEFLEX_FLASH_PRETOUCH=1 -DCACHEFLEX_FLASH_STAGE_V=0 -DCACHEFLEX_FLASH_K_OVERLAP=1 -DCACHEFLEX_FLASH_MULTICAST=1 -DCACHEFLEX_FLASH_MULTICAST_READY=1 -DCACHEFLEX_FLASH_BATCH3=1"
build_spm spm_konly_multicast_ready_dma_flash_multicore "-DCACHEFLEX_FLASH_PRETOUCH=1 -DCACHEFLEX_FLASH_STAGE_V=0 -DCACHEFLEX_FLASH_K_OVERLAP=1 -DCACHEFLEX_FLASH_MULTICAST=1 -DCACHEFLEX_FLASH_MULTICAST_READY=1 -DCACHEFLEX_FLASH_DMA=1"
build_spm spm_konly_multicast_ready_dma_interleave_flash_multicore "-DCACHEFLEX_FLASH_PRETOUCH=1 -DCACHEFLEX_FLASH_STAGE_V=0 -DCACHEFLEX_FLASH_K_OVERLAP=1 -DCACHEFLEX_FLASH_MULTICAST=1 -DCACHEFLEX_FLASH_MULTICAST_READY=1 -DCACHEFLEX_FLASH_DMA=1 -DCACHEFLEX_FLASH_INTERLEAVE_IB=1"
build_spm spm_konly_dma_pinall_interleave_flash_multicore "-DCACHEFLEX_FLASH_PRETOUCH=1 -DCACHEFLEX_FLASH_STAGE_V=0 -DCACHEFLEX_FLASH_K_OVERLAP=1 -DCACHEFLEX_FLASH_MULTICAST=1 -DCACHEFLEX_FLASH_MULTICAST_PIN_ALL=1 -DCACHEFLEX_FLASH_DMA=1 -DCACHEFLEX_FLASH_INTERLEAVE_IB=1"
build_spm spm_vonly_flash_multicore "-DCACHEFLEX_FLASH_PRETOUCH=1 -DCACHEFLEX_FLASH_STAGE_K=0"
