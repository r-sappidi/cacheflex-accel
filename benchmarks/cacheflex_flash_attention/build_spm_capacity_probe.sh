#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"; root="$(cd "$here/../.." && pwd)"
out="${1:?new absolute build directory}"
mkdir "$out"
cp "$root/gem5/build/ARM_MESI_Three_Level_SPM/gem5.opt" "$out/gem5.original.opt"
export PATH="/data2/rsappidi/conda-envs/cacheflex-gem5/bin:$PATH"
export PYTHON_CONFIG=/data2/rsappidi/conda-envs/cacheflex-gem5/bin/python3-config
export PKG_CONFIG_PATH=/data2/rsappidi/conda-envs/cacheflex-gem5/lib/pkgconfig
export LD_LIBRARY_PATH=/data2/rsappidi/conda-envs/cacheflex-gem5/lib
export CACHEFLEX_LINK_MANIFEST="$out/link_manifest.json"
cd "$root/gem5"
scons CC=gcc CXX=g++ build/ARM_MESI_Three_Level_SPM/gem5.opt -j8 > "$out/build.log" 2>&1
python3 "$here/link_spm_capacity_probe.py" "$out"
