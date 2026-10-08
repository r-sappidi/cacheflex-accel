"""Freeze existing validated microkernels and build the new mapping driver."""
from pathlib import Path
import json
import shutil
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from build_access_timing import sha,save

def build(out):
    out.mkdir(parents=True)
    base=ROOT/'results/cacheflex_flash_attention/output_resident_20260924/build_spm'
    sources=[HERE/'driver.cpp',base/'kernels.hpp',base/'pv_fullwidth.hpp',
             HERE.parent/'correct_fa_conversions.hpp',HERE.parent/'driver_vector_pack.hpp']
    for p in sources:shutil.copy2(p,out/p.name)
    # Reuse the same exp polynomial, without importing an entire old main().
    src=ROOT.parent/'cacheflex_micro/kernels/attention_kernels_spm/11_flash_attn_kv_spm_vl4.cpp'
    text=src.read_text();start=text.index('static inline svfloat32_t sve_exp_f32(')
    end=text.index('\n}',start)+3
    (out/'math_helpers.hpp').write_text('#pragma once\n'+text[start:end]+'\n')
    sources.append(src)
    save(out/'source_provenance.json',{str(p):sha(p) for p in sources})
    script=f'''set -eu
source "{ROOT}/setup_spm_env.sh" >/dev/null
"$CROSS_CXX" --version > "{out}/compiler.txt"
"$CROSS_CXX" -std=c++17 -O3 -g -ffp-contract=off -march=armv8.2-a+sve+fp16 -mbranch-protection=none -I"$M5_INCLUDE" -I"{out}" -S "{out}/driver.cpp" -o "{out}/guest.s"
python3 "$SPM_COMPILER" "{out}/guest.s" "{out}/guest.enc.s"
"$CROSS_CXX" -static -pthread "{out}/guest.enc.s" "$M5OP_OBJ" -lm -o "{out}/guest"
'''
    (out/'build.sh').write_text(script)
    with (out/'build.log').open('w') as log:
        subprocess.run(['bash',str(out/'build.sh')],check=True,stdout=log,stderr=subprocess.STDOUT)
    save(out/'frozen.json',{str(p):sha(p) for p in out.iterdir() if p.is_file() and p.name!='frozen.json'})

if __name__=='__main__':build(Path(sys.argv[1]).resolve())
