#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
out="${1:?new absolute build directory required}"
[[ "$out" == /* && ! -e "$out" ]]
mkdir -p "$out"
: "${ACL_ROOT:=/data2/rsappidi/cacheflex_mc_eval_cleanup_20260817/third_party/ComputeLibrary}"
: "${ACL_LIB:=$ACL_ROOT/build/build_sve/libarm_compute-static.a}"
source "$ROOT/setup_spm_env.sh" >/dev/null
: "${MLP_ACL_SOURCE:=$ROOT/benchmarks/ai_workloads/mlp_acl.cpp}"
"$CROSS_CXX" -std=c++17 -O3 -march=armv8.2-a+sve+fp16+dotprod \
 -mbranch-protection=none -DARM_COMPUTE_CPP_SCHEDULER=1 \
 -DARM_COMPUTE_ENABLE_NEON -DARM_COMPUTE_ENABLE_SVE -DARM_COMPUTE_ENABLE_FP16 \
 -I"$ACL_ROOT" -I"$ACL_ROOT/include" -I"$M5_INCLUDE" \
 "$MLP_ACL_SOURCE" "$M5OP_OBJ" "$ACL_LIB" \
 -static -pthread -ldl -lrt -lm -o "$out/acl"
if [[ -f "$ACL_ROOT/UPSTREAM_REVISION" ]]; then
 cp "$ACL_ROOT/UPSTREAM_REVISION" "$out/acl_commit.txt"
 printf '%s\n' 'Pristine git archive; no source modifications.' > "$out/acl_source.diff"
else
 git -C "$ACL_ROOT" rev-parse HEAD > "$out/acl_commit.txt"
 git -C "$ACL_ROOT" diff > "$out/acl_source.diff"
fi
sha256sum "$ACL_LIB" "$out/acl" "$MLP_ACL_SOURCE" > "$out/hashes.txt"
echo "$out/acl"
