#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export CACHEFLEX_PROFILE_DEFINE="-DCACHEFLEX_VL16_PIPELINED -DCACHEFLEX_VL16_INTERLEAVED"
export CACHEFLEX_BUILD_OUT="$SCRIPT_DIR/bin_gem5/cacheflex_spm_vl16_2m4n_interleaved"
exec "$SCRIPT_DIR/build_vl16_2m4n.sh"
