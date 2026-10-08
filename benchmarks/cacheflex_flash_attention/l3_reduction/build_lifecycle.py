"""Same-ELF lifecycle ablations; preserve existing corrected QK/PV/softmax."""
import json,os,shutil,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(HERE.parent)]
from build_access_timing import sha,save
from export_safe_elf import patch
OLD=ROOT/'results/cacheflex_l3_spm/pipeline_20260928/guest_v2'
BASE=ROOT/'results/cacheflex_l3_spm/lifecycle_20260928'
def main():
    out=BASE/os.environ.get('CACHEFLEX_L3_LIFECYCLE_GUEST','guest_v1');out.mkdir(parents=True)
    for p in OLD.glob('*.hpp'):shutil.copy2(p,out/p.name)
    for n in ('lifecycle.hpp','lifecycle_smoke.cpp'):shutil.copy2(HERE/n,out/n)
    driver=(OLD/'driver.cpp').read_text()
    driver=driver.replace('#include "half_contribution.hpp"','#include "lifecycle.hpp"')
    driver=driver.replace('if(accumulator_mode>=2)output_claim(o,nr);',
        'if(accumulator_mode>=2 && accumulator_mode!=5 && accumulator_mode!=7)output_claim(o,nr);')
    old='''if(accumulator_mode==4) {
                            output_reduce_half(bi+r+a,alpha.p[r+a],svld1_f16(ph,pv.p+a*64));'''
    assert driver.count(old)==1
    driver=driver.replace(old,'''if(accumulator_mode>=4) {
                            auto contribution=svld1_f16(ph,pv.p+a*64);
                            if((accumulator_mode==5||accumulator_mode==7)&&j==kbegin)
                                output_reduce_initialize(bi+r+a,alpha.p[r+a],contribution);
                            else output_reduce_half(bi+r+a,alpha.p[r+a],contribution);''')
    old='''for(int r=0;r<nr;++r)
                svst1_f16'''
    assert driver.count(old)==1
    driver=driver.replace(old,'''if(accumulator_mode>=6) {
                for(int r=0;r<nr;++r)output_finalize(r,1.f/l[r],s.out.p+(qi+r)*64);
                shared_drain();
            } else for(int r=0;r<nr;++r)
                svst1_f16''')
    driver=driver.replace('if(accumulator_mode>=2)output_release(nr);','if(accumulator_mode>=2 && accumulator_mode<6)output_release(nr);')
    driver=driver.replace('accumulator_mode>4','accumulator_mode>7')
    (out/'driver.cpp').write_text(driver)
    template=json.loads((OLD/'commands.json').read_text())[0];commands=[]
    for name,source in [('guest','driver.cpp'),('lifecycle_smoke','lifecycle_smoke.cpp')]:
        cmd=template.replace(str(OLD),str(out)).replace('/driver.cpp','/'+source)
        cmd=cmd.replace(str(out)+'/guest',str(out)+'/'+name);commands.append(cmd)
        with (out/(name+'.build.log')).open('w') as f:
            subprocess.run(['bash','-c',cmd],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    patch(out/'guest',out/'guest.export-safe');save(out/'commands.json',commands)
    save(out/'frozen.json',{str(p):sha(p) for p in out.iterdir() if p.is_file()})
if __name__=='__main__':main()
