#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
out="${1:?usage: run_scalar_matched_sweep.sh NEW_OUTPUT_DIRECTORY}"
mkdir "$out"
sha256sum "$root/gem5/build/ARM_MESI_Three_Level_SPM/gem5.opt" \
  "$here/bin/cache_pretouch_flash_multicore" \
  "$here/bin/spm_konly_overlap_flash_multicore" > "$out/binaries.sha256"
# Explicit clean environment prevents inherited experimental parameter overrides.
# Fixed Br/Bc for a controlled data-path comparison, not independent tile tuning.
for tokens in 128 256 512 1024; do
  pids=()
  for variant in cache_pretouch spm_konly_overlap; do
    env -i PATH="$PATH" HOME="$HOME" \
      CACHEFLEX_FLASH_OUT="$out/${variant}_t${tokens}" \
      bash "$here/run_8core.sh" "$variant" "$tokens" 64 96 \
      > "$out/${variant}_t${tokens}.log" 2>&1 &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  [[ "$failed" == 0 ]] || exit 1
  echo "Completed T=$tokens"
done
