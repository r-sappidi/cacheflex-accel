#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
m_partitions="${CACHEFLEX_M_PARTITIONS_VALUE:-4}"
export CACHEFLEX_PROFILE_DEFINE="-DCACHEFLEX_M_PARTITIONS=$m_partitions -DCACHEFLEX_VL16_PIPELINED -DCACHEFLEX_MULTICAST -DCACHEFLEX_MULTICAST_WAY_CLAIM ${CACHEFLEX_EXTRA_DEFINE:-}"
export CACHEFLEX_BUILD_OUT="${CACHEFLEX_BUILD_OUT:-$SCRIPT_DIR/bin_gem5/cacheflex_spm_vl16_best_multicast_claim}"
exec "$SCRIPT_DIR/build_vl16_2m4n.sh"
