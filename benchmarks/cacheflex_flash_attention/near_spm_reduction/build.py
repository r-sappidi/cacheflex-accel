"""Frozen same-ELF FA modes and directed reduction test; existing microkernels."""
import sys, shutil, subprocess, json
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(HERE.parent)]
from build_access_timing import sha,save
from export_safe_elf import patch
BASE=ROOT/'results/cacheflex_flash_attention/near_spm_reduction_20260927'
OLD=ROOT/'results/cacheflex_flash_attention/model_tuning_20260926/build'
def main():
    out=BASE/'guest_v2';out.mkdir(parents=True)
    for name in ('kernels.hpp','pv_fullwidth.hpp','math_helpers.hpp',
                 'correct_fa_conversions.hpp','driver_vector_pack.hpp','fast_qpack.hpp'):
        shutil.copy2(OLD/name,out/name)
    for name in ('driver.cpp','reduction.hpp','smoke.cpp'):shutil.copy2(HERE/name,out/name)
    commands=[]
    for name,src in [('guest','driver.cpp'),('smoke','smoke.cpp')]:
        # The existing exact compiler/assembler pipeline, with new source/output paths.
        script=(OLD/'build.sh').read_text().replace(str(OLD),str(out))
        script=script.replace('/driver.cpp','/'+src).replace(str(out)+'/guest',str(out)+'/'+name)
        commands.append(script)
        with (out/(name+'.build.log')).open('w') as f:
            subprocess.run(['bash','-c',script],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    patch(out/'guest',out/'guest.export-safe')
    save(out/'build_commands.json',commands)
    save(out/'frozen.json',{str(p):sha(p) for p in out.iterdir() if p.is_file()})
if __name__=='__main__':main()
