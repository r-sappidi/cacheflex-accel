"""Single FA ELF: cache/private/FP32 command/FP16 command; frozen old math."""
import json,os,shutil,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(HERE.parent)]
from build_access_timing import sha,save
from export_safe_elf import patch
OLD=ROOT/'results/cacheflex_l3_spm/reduction_20260928/guest_v1'
BASE=ROOT/'results/cacheflex_l3_spm/pipeline_20260928'

def main():
    out=BASE/os.environ.get('CACHEFLEX_L3_PIPELINE_GUEST','guest_v2');out.mkdir(parents=True)
    for p in OLD.glob('*.hpp'):shutil.copy2(p,out/p.name)
    shutil.copy2(HERE/'half_contribution.hpp',out/'half_contribution.hpp')
    driver=(OLD/'driver.cpp').read_text()
    driver=driver.replace('#include "reduction.hpp"','#include "reduction.hpp"\n#include "half_contribution.hpp"')
    before='auto contribution=cf_dense_f16_to_f32(pf,svld1_f16(ph,pv.p+a*64));'
    assert driver.count(before)==1
    driver=driver.replace(before,'''if(accumulator_mode==4) {
                            output_reduce_half(bi+r+a,alpha.p[r+a],svld1_f16(ph,pv.p+a*64));
                            continue;
                        }
                        '''+before)
    assert 'accumulator_mode>3' in driver
    driver=driver.replace('accumulator_mode>3','accumulator_mode>4')
    (out/'driver.cpp').write_text(driver)
    smoke=(OLD/'smoke.cpp').read_text()
    # Separate directed half-ABI test; existing FP32 smoke remains byte-identical.
    (out/'smoke.cpp').write_text(smoke)
    helper='''
#include "half_contribution.hpp"
static void test_reduce(uint64_t slot,float scale,svfloat32_t v) {
    alignas(64) float values[64];alignas(64) float16_t half[64];
    svst1_f32(svptrue_b32(),values,v);
    for(int i=0;i<64;++i)half[i]=float16_t(values[i]);
    uint32_t bits;std::memcpy(&bits,&scale,4);
    auto addr=slot|(uint64_t(bits)<<24)|(uint64_t(1)<<60)|(uint64_t(1)<<58);
    auto packed=svld1_f16(svwhilelt_b16(0,64),half);auto p=svptrue_b16();
    asm volatile("spm.st1h %0.h, %1, [%2]\\n"::"w"(packed),"Upl"(p),"r"(addr):"memory");
}
'''
    half=smoke.replace('shared_reduce(','test_reduce(')
    half=half.replace('#include "reduction.hpp"','#include "reduction.hpp"\n'+helper)
    (out/'smoke_half.cpp').write_text(half)
    mixed=half.replace('static void test_reduce(uint64_t slot,float scale,svfloat32_t v) {',
        '''static void test_reduce(uint64_t slot,float scale,svfloat32_t v) {
    static thread_local unsigned count=0;
    if((count++ & 1)==0) { shared_reduce(slot,scale,v);return; }''')
    (out/'smoke_mixed.cpp').write_text(mixed)
    template=json.loads((OLD/'commands.json').read_text())[0]
    commands=[]
    for name,source in [('guest','driver.cpp'),('smoke','smoke.cpp'),('smoke_half','smoke_half.cpp'),('smoke_mixed','smoke_mixed.cpp')]:
        cmd=template.replace(str(OLD),str(out)).replace('/driver.cpp','/'+source)
        cmd=cmd.replace(str(out)+'/guest',str(out)+'/'+name)
        commands.append(cmd)
        with (out/(name+'.build.log')).open('w') as f:
            subprocess.run(['bash','-c',cmd],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    patch(out/'guest',out/'guest.export-safe')
    save(out/'commands.json',commands)
    save(out/'frozen.json',{str(p):sha(p) for p in out.iterdir() if p.is_file()})

if __name__=='__main__':main()
