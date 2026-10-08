#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export CACHEFLEX_PROFILE_DEFINE=-DCACHEFLEX_PHASE_PROFILE
export CACHEFLEX_BUILD_OUT="$SCRIPT_DIR/bin_gem5/cacheflex_spm_vl16_2m4n_profile"
exec "$SCRIPT_DIR/build_vl16_2m4n.sh"
