"""Build isolated Qwen projections using unchanged accepted GEMM primitives."""
import json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path('/data2/rsappidi/cacheflex_mc');sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'benchmarks/ai_workloads')]
from build_access_timing import save,sha
from repair_screen_export import patch
BASE=ROOT/'results/ai_workloads/qwen_projection_20260928'
OLD=ROOT/'results/ai_workloads/qwen_swiglu_20260928/guests/custom_g0'

def main():
 out=BASE/'guest_v1';out.mkdir(parents=True)
 src=(OLD/'mlp_study.cpp').read_text();prefix=src[:src.index('size_t check(')]
 # Reuse allocation, staging, packing and GEMM bodies byte-for-byte.
 (out/'projection_core.hpp').write_text(prefix+'\n}\n')
 for name in ('mlp_helpers.hpp','mlp_fp32_kernel.hpp'):shutil.copy2(OLD/name,out/name)
 shutil.copy2(ROOT/'benchmarks/ai_workloads/qwen_projection.cpp',out/'driver.cpp')
 build=r'''set -euo pipefail
source /data2/rsappidi/cacheflex_mc/setup_spm_env.sh >/dev/null
out="$1"
flags=(-std=c++17 -O3 -march=armv8.2-a+sve+fp16 -mbranch-protection=none -DGEM5 -DGEM5_SE -DMLP_ROWS=8 -DMLP_ACC32=1 -DMLP_COMBINED=0 -I"$out" -I"$M5_INCLUDE" -I"$CACHEFLEX_MC_ROOT/../cacheflex_micro/experiments/vl_length/tile_shape")
"$CROSS_CXX" "${flags[@]}" -I"$CACHEFLEX_MC_ROOT/../cacheflex_micro/kernels/llama_bench" -DIDENTICAL_CACHE_BASELINE -DCACHEFLEX_CACHE_VL16_PIPELINED -static -pthread "$out/driver.cpp" "$M5OP_OBJ" -lm -o "$out/cache"
"$CROSS_CXX" "${flags[@]}" -I"$CACHEFLEX_MC_ROOT/../cacheflex_micro/kernels/llama_bench_spm" -DCACHEFLEX_VL16_PIPELINED -S "$out/driver.cpp" -o "$out/spm.s"
python3 "$SPM_COMPILER" "$out/spm.s" "$out/spm_enc.s"
"$CROSS_CXX" -static -pthread "$out/spm_enc.s" "$M5OP_OBJ" -lm -o "$out/spm"
'''
 (out/'build.sh').write_text(build)
 with (out/'build.log').open('w') as log:subprocess.run(['bash',str(out/'build.sh'),str(out)],stdout=log,stderr=subprocess.STDOUT,check=True)
 for arm in ('cache','spm'):patch(out/arm)
 files=[p for p in out.iterdir() if p.suffix in ('.cpp','.hpp','.sh') or p.name in ('cache','spm')]
 files += [OLD/n for n in ('mlp_study.cpp','mlp_helpers.hpp','mlp_fp32_kernel.hpp')]
 save(out/'frozen.json',{str(p):sha(p) for p in files})
 save(out/'audit.json',dict(gemm_prefix_exact=(out/'projection_core.hpp').read_text()==prefix+'\n}\n',new_code='Projection driver, explicit FP32 bias from preserved accumulator and head layout conversions; original GEMM half store is retained and overwritten for QKV bias.'))
 save(out/'status.json',dict(stage='built_requires_validation',cache_sha256=sha(out/'cache'),spm_sha256=sha(out/'spm')))
if __name__=='__main__':main()
