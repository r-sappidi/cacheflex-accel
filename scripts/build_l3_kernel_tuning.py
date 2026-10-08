"""Software-only direct PV epilogues; reuse the frozen lifecycle simulator."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path('/data2/rsappidi/cacheflex_mc')
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'benchmarks/cacheflex_flash_attention')]
from build_access_timing import sha, save
from export_safe_elf import patch

OLD = ROOT/'results/cacheflex_l3_spm/lifecycle_20260928/guest_v1'
BASE = ROOT/'results/cacheflex_l3_spm/kernel_tuning_20260928'


def make_kernel(spm, reduction, paired):
    """Keep original literal assembly through the final FP16 FMA unchanged."""
    import ast
    source = (OLD/'pv_fullwidth.hpp').read_text()
    name = 'cf_pv16_spm' if spm else 'cf_pv16_cache'
    body = source.split('static void '+name+'(', 1)[1]
    literal = body.split('__asm__ __volatile__(', 1)[1].split('    : :', 1)[0]
    asm = ast.literal_eval('('+literal.strip()+')')
    compute = asm[:asm.index('mov x5, x0\n')]
    # C/ld setup was only needed for the old temporary; the loop stays exact.
    compute = compute.replace('mov x0, %x[C]\n', '').replace('lsl x13, %x[ld], #1\n', '')
    order = [i for k in range(8) for i in (k, k+8)] if paired else list(range(16))
    epilogue = 'mov x7, %x[alpha]\nmov x8, %x[slot]\nmov x10, %x[output]\nptrue p3.s\n'
    for row in order:
        reg = 8 + row % 8
        if row >= 8:
            epilogue += f'ext z{reg}.b, z{reg}.b, z{reg}.b, #128\n'
        if reduction:
            epilogue += (f'ldr w4, [x7, #{row*4}]\n'
                         'orr x4, x8, x4, lsl #24\n'
                         f'add x4, x4, #{row*64}\n'
                         f'spm.st1h z{reg}.h, p0, [x4]\n')
        else:
            epilogue += (f'zip1 z3.h, z{reg}.h, z{reg}.h\n'
                         'fcvt z3.s, p3/m, z3.h\n'
                         f'ld1rw {{z4.s}}, p3/z, [x7, #{row*4}]\n'
                         f'add x9, x10, #{row*256}\n'
                         'ld1w {z5.s}, p3/z, [x9]\n'
                         'fmul z5.s, p3/m, z5.s, z4.s\n'
                         'fadd z5.s, p3/m, z5.s, z3.s\n'
                         'st1w {z5.s}, p3, [x9]\n')
    suffix = ('reduce' if reduction else 'cpu') + ('_paired' if paired else '_sequential')
    lines = '\n'.join(json.dumps(line+'\n') for line in (compute+epilogue).splitlines())
    clobbers = ['cc','memory','p0','p1','p2','p3'] + [f'x{i}' for i in (0,3,4,5,6,7,8,9,10,13,14,16,20,21,22,23)] + [f'z{i}' for i in (0,1,2,3,4,5,8,9,10,11,12,13,14,15)]
    return f'''static __attribute__((noinline,aligned(64))) void {name}_{suffix}(
        const __fp16* p,size_t v,size_t k,const float* alpha,uint64_t slot,float* output) {{
    asm volatile(
{lines}
    : : [A] "r"(p),[B] "r"(v),[K] "r"(k),[alpha] "r"(alpha),
        [slot] "r"(slot),[output] "r"(output)
    : {','.join(json.dumps(c) for c in clobbers)});
}}
'''


def main():
    out = BASE/os.environ.get('CACHEFLEX_L3_KERNEL_GUEST','guest_v1')
    out.mkdir(parents=True)
    for p in OLD.glob('*.hpp'):
        shutil.copy2(p, out/p.name)
    kernels = '#pragma once\n// Generated from the frozen, checked FP16 PV FMA loop.\n'
    for spm, reduction in ((True,True),(True,False),(False,False)):
        for paired in (True,False):
            kernels += make_kernel(spm,reduction,paired)
    (out/'pv_direct.hpp').write_text(kernels)
    driver = (OLD/'driver.cpp').read_text()
    driver = driver.replace('#include "lifecycle.hpp"', '#include "lifecycle.hpp"\n#include "pv_direct.hpp"\nstatic int kernel_variant=0;')
    anchor = '                    if(s.spm) cf_pv16_spm(pa.p,vbase(s.kv?j-kbegin:0,s.sets),pv.p,bc,64);'
    assert driver.count(anchor) == 1
    direct = '''                    if(kernel_variant) {
                        auto slot=shared_slot(shared_output_bank,bi+r)|(uint64_t(1)<<60)|
                                  (uint64_t(1)<<58);
                        if(j==kbegin)slot|=uint64_t(1)<<56;
                        const auto vptr=s.spm?vbase(0,s.sets):reinterpret_cast<size_t>(s.v.p+j*64);
                        if(accumulator_mode==5) {
                            if(kernel_variant==1)cf_pv16_spm_reduce_paired(pa.p,vptr,bc,alpha.p+r,slot,nullptr);
                            else cf_pv16_spm_reduce_sequential(pa.p,vptr,bc,alpha.p+r,slot,nullptr);
                        } else if(s.spm) {
                            if(kernel_variant==1)cf_pv16_spm_cpu_paired(pa.p,vptr,bc,alpha.p+r,0,o+(bi+r)*64);
                            else cf_pv16_spm_cpu_sequential(pa.p,vptr,bc,alpha.p+r,0,o+(bi+r)*64);
                        } else {
                            if(kernel_variant==1)cf_pv16_cache_cpu_paired(pa.p,vptr,bc,alpha.p+r,0,o+(bi+r)*64);
                            else cf_pv16_cache_cpu_sequential(pa.p,vptr,bc,alpha.p+r,0,o+(bi+r)*64);
                        }
                        continue;
                    }
'''
    driver = driver.replace(anchor,direct+anchor)
    driver = driver.replace('if(argc!=10)', 'if(argc!=11)')
    driver = driver.replace('spm accumulator_mode\\n', 'spm accumulator_mode kernel_variant\\n')
    driver = driver.replace('accumulator_mode=atoi(argv[9]);', '''accumulator_mode=atoi(argv[9]);kernel_variant=atoi(argv[10]);
    if(kernel_variant<0||kernel_variant>2 ||
       (kernel_variant && accumulator_mode!=1 && accumulator_mode!=5))return 2;''')
    (out/'driver.cpp').write_text(driver)
    template = json.loads((OLD/'commands.json').read_text())[0]
    command = template.replace(str(OLD),str(out))
    (out/'build.sh').write_text(command)
    with (out/'build.log').open('w') as log:
        subprocess.run(['bash',str(out/'build.sh')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    patch(out/'guest',out/'guest.export-safe')
    save(out/'source_reference.json',dict(source=str(OLD),ledger=['FA-29','L3-03','L3-04']))
    save(out/'frozen.json',{str(p):sha(p) for p in out.iterdir() if p.is_file()})
    print(out,sha(out/'guest.export-safe'),flush=True)


if __name__=='__main__':main()
