"""Isolated near-L3 build using preserved storage-only source/config baseline."""
import os
from pathlib import Path
import build_l3_spm as build
ROOT=Path(__file__).resolve().parents[1]
build.BASE=ROOT/'results/cacheflex_l3_spm/reduction_20260928'
build.OLD=ROOT/'results/cacheflex_l3_spm/initial_20260928/build_v4'
build.BUILD=build.BASE/os.environ.get('CACHEFLEX_L3_REDUCTION_BUILD','build_v1')
if __name__=='__main__':build.main()
