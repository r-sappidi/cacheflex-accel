"""Preserve original FA; derive one isolated Q-packing optimization."""
from pathlib import Path
import shutil,subprocess,sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from build_access_timing import save,sha
BASE=ROOT/'results/cacheflex_flash_attention/cache_tuning_20260925'
OLD=ROOT/'results/cacheflex_flash_attention/kv_partition_20260925/build'

def main():
    out=BASE/'build';out.mkdir(parents=True)
    for name in ('driver.cpp','kernels.hpp','pv_fullwidth.hpp','math_helpers.hpp','correct_fa_conversions.hpp','driver_vector_pack.hpp'):
        shutil.copy2(OLD/name,out/name)
    helper=(out/'driver_vector_pack.hpp').read_text()
    # Specialize the existing bit-permutation network for 64 reduction elements.
    # Generic packer uses a scalar tail at VL16 because 64 < svcnth()==128.
    start=helper.index('            svfloat16_t r0=')
    end=helper.index('        }\n        for (; ki',start)
    network=helper[start:end]
    network='\n'.join(line for line in network.splitlines() if not any('ki*8+'+str(i)+'*VLh' in line for i in range(4,8)))
    fast='''static void cf_pack_q64(const __fp16* src,__fp16* dst,size_t rows) {
    const size_t VLh=svcnth(),ki=0;
    const auto pg=svptrue_b16();
    const auto active=svwhilelt_b16_u64(0,64);
    for(size_t row=0;row<rows;row+=8) {
        const __fp16* rp[8];for(int r=0;r<8;++r)rp[r]=src+(row+r)*64;
'''+network.replace('svld1_f16(pg,','svld1_f16(active,')+'''
        dst+=8*64;
    }
}
'''
    (out/'fast_qpack.hpp').write_text('#pragma once\n'+fast)
    src=(out/'driver.cpp').read_text()
    src=src.replace('#include "driver_vector_pack.hpp"','#include "driver_vector_pack.hpp"\n#include "fast_qpack.hpp"')
    old='cf_vector_pack_A(s.q.p,64,qa.p,s.t,qi,64,0,nr,64);'
    assert src.count(old)==1
    src=src.replace(old,'cf_pack_q64(s.q.p+qi*64,qa.p,nr);')
    # Reject SPM/KV mode: this experiment is explicitly cache-only.
    src=src.replace('setvbuf(stdout,nullptr,_IONBF,0);','if(kv||spm)return 2;\n    setvbuf(stdout,nullptr,_IONBF,0);')
    (out/'driver.cpp').write_text(src)
    script=(OLD/'build.sh').read_text().replace(str(OLD),str(out))
    (out/'build.sh').write_text(script)
    with (out/'build.log').open('w') as log:subprocess.run(['bash',str(out/'build.sh')],stdout=log,stderr=subprocess.STDOUT,check=True)
    save(out/'provenance.json',dict(original={str(p):sha(p) for p in OLD.iterdir() if p.is_file()},
        change='Q64 packing: same zip network, predicated loads, four valid output vectors; no arithmetic/model changes'))
    save(out/'frozen.json',{str(p):sha(p) for p in out.iterdir() if p.is_file() and p.name!='frozen.json'})
if __name__=='__main__':main()
