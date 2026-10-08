#!/usr/bin/env bash
# Separate pristine upstream source/export and logging-free production archive.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
out="${1:?new absolute directory required}"
[[ "$out" == /* && ! -e "$out" ]]
: "${ACL_REFERENCE:=/data2/rsappidi/cacheflex_mc_eval_cleanup_20260817/third_party/ComputeLibrary}"
: "${ACL_REVISION:=c245aa08b328fe7ef1d356f40b89ebc587524dae}"
mkdir -p "$out/source"
git -C "$ACL_REFERENCE" archive "$ACL_REVISION" | tar -x -C "$out/source"
printf '%s\n' "$ACL_REVISION" > "$out/source/UPSTREAM_REVISION"
prefix="$ROOT/tools/arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-"
cd "$out/source"
scons -j8 Werror=0 debug=0 asserts=0 logging=0 neon=1 opencl=0 \
 arch=arm64-v8.2-a-sve os=linux build=cross_compile \
 toolchain_prefix="$prefix" cppthreads=1 openmp=0 examples=0 validation_tests=0 \
 gemm_tuner=0 standalone=1 data_type_support=fp16,fp32 \
 build/libarm_compute-static.a 2>&1 | tee "$out/build.log"
ACL_ROOT="$out/source" ACL_LIB="$out/source/build/libarm_compute-static.a" \
 bash "$ROOT/benchmarks/ai_workloads/build_mlp_acl.sh" "$out/guest"
