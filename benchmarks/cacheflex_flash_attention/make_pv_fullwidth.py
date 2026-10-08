"""VL16: two eight-row groups share the two vector halves (16x64 output).

P remains packed in two ordinary8-row chunks; replicate64-column V across
halves. Indexed FMLA selects different P values in each128-bit segment.
Sequential FP16 accumulation order for each output is unchanged.
"""
import json
from pathlib import Path

def kernel(spm):
    a=['ptrue p0.h','mov x3, #16','whilelt p1.d, xzr, x3',
       'mov x3, #64','whilelt p2.h, xzr, x3',
       'mov x21, %x[A]','lsl x14, %x[K], #4','add x23, x21, x14',
       'mov x22, %x[B]','mov x0, %x[C]','mov x20, %x[K]',
       'lsl x13, %x[ld], #1']
    a += [f'mov z{8+r}.b, #0' for r in range(8)]
    a+=['b 1f','.p2align 6','1:']
    if spm:a+=['lsl x16, x22, #6','spm.ld1qd z2.d, p1/z, [x16]']
    else:a+=['ld1h {z2.h}, p2/z, [x22]']
    a+=['splice z2.d, p1, z2.d, z2.d',
        'ld1rqh {z0.h}, p0/z, [x21]','ld1rqh {z1.h}, p0/z, [x23]',
        'splice z0.d, p1, z0.d, z1.d']
    a += [f'fmla z{8+r}.h, z2.h, z0.h[{r}]' for r in range(8)]
    a+=['add x21, x21, #16','add x23, x23, #16',
        'add x22, x22, #'+('1' if spm else '128'),
        'subs x20, x20, #1','b.ne 1b','mov x5, x0','add x6, x0, x13, lsl #3']
    for r in range(8):
        a += [f'st1h {{z{8+r}.h}}, p2, [x5]',
              f'ext z{8+r}.b, z{8+r}.b, z{8+r}.b, #128',
              f'st1h {{z{8+r}.h}}, p2, [x6]',
              'add x5, x5, x13','add x6, x6, x13']
    name='cf_pv16_'+('spm' if spm else 'cache')
    clob=['cc','memory','p0','p1','p2']+[f'x{i}' for i in [0,3,5,6,13,14,16,20,21,22,23]]+['z0','z1','z2']+[f'z{i}' for i in range(8,16)]
    return f'''static void {name}(const __fp16* p, size_t v, __fp16* out, size_t k, size_t ld) {{
    __asm__ __volatile__(
'''+ '\n'.join(json.dumps(x+'\n') for x in a)+'''
    : : [A] "r"(p),[B] "r"(v),[C] "r"(out),[K] "r"(k),[ld] "r"(ld)
    : '''+','.join(json.dumps(x) for x in clob)+');\n}\n'

def generate(path):Path(path).write_text('#pragma once\n#include <cstddef>\n'+kernel(False)+kernel(True))

if __name__=='__main__':
    import sys
    generate(sys.argv[1])
