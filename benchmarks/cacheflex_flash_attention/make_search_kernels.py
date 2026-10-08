"""Generate local native-kernel variants; never modify the micro reference.

Unrolls the existing assembly K loop by exact body replication. Valid only
for K divisible by the factor (search uses d64 and Bc multiples of 128).
"""
from pathlib import Path
import re
import sys
import json

root=Path(__file__).resolve().parents[2]
src=root.parent/'cacheflex_micro/experiments/end2end/src'
out=Path(sys.argv[1]); factor=int(sys.argv[2]); order=int(sys.argv[3])
assert factor in (1,2,4) and order in (0,1)
wide=int(sys.argv[4]) if len(sys.argv)>4 else 0
pipeline=int(sys.argv[5]) if len(sys.argv)>5 else 0
assert wide in (0,8,16,24) and pipeline in (0,1)

def wide_kernel(spm):
    n=wide//8
    a=['ptrue p0.h','ptrue p1.d','mov x21, %x[Ap]','mov x22, %x[Bp]',
       'mov x25, %x[base]','mov x20, %x[K]','lsl x14, %x[K], #4',
       'add x23, x21, x14','add x24, x23, x14']
    a += [f'mov z{8+r}.b, #0' for r in range(wide)]
    def load(bank):
        ar=([0,1,4],[5,6,7])[bank]; br=2+bank
        lines=[]
        if spm: lines += ['lsl x16, x25, #6',f'spm.ld1qd z{br}.d, p1/z, [x16]','add x25, x25, #1']
        else: lines += [f'ld1h {{z{br}.h}}, p0/z, [x22]','addvl x22, x22, #1']
        for ptr,z in zip([21,23,24][:n],ar[:n]):
            lines += [f'ld1rqh {{z{z}.h}}, p0/z, [x{ptr}]',f'add x{ptr}, x{ptr}, #16']
        return lines
    def compute(bank):
        ar=([0,1,4],[5,6,7])[bank]
        return [f'fmla z{8+r}.h, z{2+bank}.h, z{ar[r//8]}.h[{r%8}]' for r in range(wide)]
    if pipeline:
        a+=load(0)+['1:']+load(1)+compute(0)+['subs x20, x20, #2','beq 2f']+load(0)+compute(1)+['b 1b','2:']+compute(1)
    else:
        a+=['1:']+load(0)+compute(0)+['subs x20, x20, #1','bne 1b']
    a+=['mov x5, %x[Cp]','lsl x13, %x[ldc], #1']
    for r in range(wide): a += [f'st1h {{z{8+r}.h}}, p0, [x5]','add x5, x5, x13']
    name='cf_wide_qk_'+('spm' if spm else 'cache')
    code='static void '+name+'(const __fp16* a,const __fp16* b,__fp16* c,int k,int ld,size_t base) {\n__asm__ __volatile__(\n'
    code+='\n'.join(json.dumps(x+'\n') for x in a)
    code+='\n: : [Ap] "r"(a),[Bp] "r"(b),[Cp] "r"(c),[K] "r"((size_t)k),[ldc] "r"((size_t)ld),[base] "r"(base)\n: '
    code+=','.join(json.dumps(x) for x in ['cc','memory','p0','p1']+[f'x{i}' for i in (5,13,14,16,20,21,22,23,24,25)]+[f'z{i}' for i in range(32)])
    return code+');\n}\n'

def extract(file,name):
    s=(src/file).read_text()
    start=s.index('static void '+name+'(')
    return s[start:s.index('\n}',start)+3]

def transform(s):
    if order and 'spm_kernel_1vl_vl16' in s:
        lines=s.splitlines(True)
        a=next(i for i,x in enumerate(lines) if '"spm.ld1qd z2.d' in x)
        b=next(i for i,x in enumerate(lines) if '"ld1rqh { z0.h' in x)
        lines[a],lines[b]=lines[b],lines[a]
        s=''.join(lines)
    tokens=list(re.finditer(r'"(?:[^"\\]|\\.)*"',s))
    label=next(m for m in tokens if m.group()=='"1:\\n"')
    sub=next(m for m in tokens if re.match(r'"subs x20,\s*x20,\s*#1\\n"',m.group()))
    body=s[label.end():sub.start()]
    return s[:label.end()]+body*factor+sub.group().replace('#1',f'#{factor}')+s[sub.end():]

specs=[('cache_flash_attn.cpp','cache_qk_8x1VL_ldc'),
       ('spm_qk_kernel_vl16.hpp','spm_kernel_1vl_vl16'),
       ('pv_kernel_8x64_vl16.hpp','cache_pv_8x64_vl16'),
       ('pv_kernel_8x64_vl16.hpp','spm_pv_8x64_vl16')]
out.write_text('#pragma once\n#include <arm_sve.h>\n#include <cstddef>\n'+
               '\n'.join(transform(extract(f,n)) for f,n in specs)+
               (wide_kernel(False)+wide_kernel(True) if wide else ''))
