#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
out="${1:?new absolute output directory required}"
mkdir "$out"
for experiment in ${CACHEFLEX_ABLATIONS:-fast_memctrl wide_noc}; do
  extra=()
  if [[ "$experiment" == fast_memctrl ]]; then
    extra+=(CACHEFLEX_FLASH_FAST_MEMCTRL=1)
  elif [[ "$experiment" == fast_dram ]]; then
    extra+=(CACHEFLEX_FLASH_FAST_DRAM=1)
  else
    extra+=(CACHEFLEX_FLASH_LINK_BITS=512)
  fi
  pids=()
  for variant in cache_pretouch spm_konly_overlap; do
    env -i PATH="$PATH" HOME="$HOME" "${extra[@]}" \
      CACHEFLEX_FLASH_OUT="$out/${experiment}_${variant}" \
      bash "$here/run_8core.sh" "$variant" 1024 64 96 \
      > "$out/${experiment}_${variant}.log" 2>&1 &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  [[ "$failed" == 0 ]] || exit 1
  echo "Completed $experiment"
done
