#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export CACHEFLEX_PROFILE_DEFINE="-DCACHEFLEX_M_PARTITIONS=4 -DCACHEFLEX_CACHE_VL16_PIPELINED ${CACHEFLEX_EXTRA_DEFINE:-}"
export CACHEFLEX_BUILD_OUT="${CACHEFLEX_BUILD_OUT:-$SCRIPT_DIR/bin_gem5/cache_vl16_best}"
exec "$SCRIPT_DIR/build_vl16_2m4n_baseline.sh"
