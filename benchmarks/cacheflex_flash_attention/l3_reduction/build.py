"""Frozen guests; unchanged QK/PV and arithmetic order, alternate output backend."""
import sys,os,shutil,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(HERE.parent)]
from build_access_timing import sha,save
from export_safe_elf import patch
BASE=ROOT/'results/cacheflex_l3_spm/reduction_20260928'
OLD=ROOT/'results/cacheflex_flash_attention/model_tuning_20260926/build'
def main():
    out=BASE/os.environ.get('CACHEFLEX_L3_REDUCTION_GUEST','guest_v1');out.mkdir(parents=True)
    for n in ('kernels.hpp','pv_fullwidth.hpp','math_helpers.hpp','correct_fa_conversions.hpp','driver_vector_pack.hpp','fast_qpack.hpp'):
        shutil.copy2(OLD/n,out/n)
    shutil.copy2(HERE.parent/'l3_spm/shared_spm.hpp',out/'shared_spm.hpp')
    for n in ('reduction.hpp','smoke.cpp'):shutil.copy2(HERE/n,out/n)
    src=HERE.parent/'near_spm_reduction/driver.cpp'
    driver=src.read_text().replace('const int rows=s.br*s.group,local=s.t/s.cores;',
        'shared_output_bank=c.id;\n    const int rows=s.br*s.group,local=s.t/s.cores;')
    driver=driver.replace('std::fill(o,o+nr*64,0.f);','if(accumulator_mode<2)std::fill(o,o+nr*64,0.f);').replace('br*g>1024','br*g>512')
    (out/'driver.cpp').write_text(driver)
    commands=[]
    for name,source in [('guest','driver.cpp'),('smoke','smoke.cpp')]:
        cmd=(OLD/'build.sh').read_text().replace(str(OLD),str(out))
        cmd=cmd.replace('/driver.cpp','/'+source).replace(str(out)+'/guest',str(out)+'/'+name)
        commands.append(cmd)
        with (out/(name+'.build.log')).open('w') as f:
            subprocess.run(['bash','-c',cmd],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    patch(out/'guest',out/'guest.export-safe')
    save(out/'commands.json',commands);save(out/'source_reference.json',dict(path=str(src),sha256=sha(src)))
    save(out/'frozen.json',{str(p):sha(p) for p in out.iterdir() if p.is_file()})
if __name__=='__main__':main()
