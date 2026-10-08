"""Independent snapshot/build; previous pipeline experiments remain untouched."""
import os
import build_l3_spm as build
build.BASE=build.ROOT/'results/cacheflex_l3_spm/lifecycle_20260928'
build.BUILD=build.BASE/os.environ.get('CACHEFLEX_L3_LIFECYCLE_BUILD','build_v1')
if __name__=='__main__':build.main()
