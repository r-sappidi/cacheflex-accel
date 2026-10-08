"""Use the same existing Q64 packing improvement in all four FA families."""
from pathlib import Path
import json,shutil,subprocess,sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(HERE.parent)]
from build_access_timing import sha,save
from export_safe_elf import patch
BASE=ROOT/'results/cacheflex_flash_attention/model_tuning_20260926'

def main():
    old=ROOT/'results/cacheflex_flash_attention/cache_tuning_20260925/build'
    out=BASE/'build';out.mkdir(parents=True)
    for name in ('driver.cpp','kernels.hpp','pv_fullwidth.hpp','math_helpers.hpp',
                 'correct_fa_conversions.hpp','driver_vector_pack.hpp','fast_qpack.hpp'):
        shutil.copy2(old/name,out/name)
    src=(out/'driver.cpp').read_text();assert src.count('if(kv||spm)return 2;')==1
    (out/'driver.cpp').write_text(src.replace('if(kv||spm)return 2;',''))
    # No new arithmetic, microkernel, loop ordering or SPM instructions.
    script=(old/'build.sh').read_text().replace(str(old),str(out))
    (out/'build.sh').write_text(script)
    with (out/'build.log').open('w') as f:
        subprocess.run(['bash',str(out/'build.sh')],stdout=f,stderr=subprocess.STDOUT,check=True)
    patch(out/'guest',out/'guest.export-safe')
    save(out/'provenance.json',dict(source={str(p):sha(p) for p in old.iterdir() if p.is_file()},
         change='Enable existing Q64 optimized packer for SPM and KV mappings too; safe post-ROI export'))
    save(out/'frozen.json',{str(p):sha(p) for p in out.iterdir() if p.is_file() and p.name!='frozen.json'})

if __name__=='__main__':main()
