"""Generate local 8/16-row variants of the existing pipelined 24x1VL kernels.
Keep the micro tree unchanged. Drop unused accumulators/A groups, adjust A
stride, and emit a row-count-correct store/RMW epilogue. Original 24-row
implementation remains an independent contender.
"""
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[2]
out=Path(sys.argv[1]);rows=int(sys.argv[2]);assert rows in (8,16)
micro=ROOT.parent/'cacheflex_micro/experiments/vl_length/tile_shape'
for variant,filename in (('cache','v1_24x1vl.cpp'),('spm','kernels_tile_spm.hpp')):
    source=(micro/filename).read_text()
    if variant=='cache':
        begin=source.index('static void cache_gemm_fused_24x1VL(')
        first='x4';epilogue=source.index('        // Epilogue: 24 rows',begin)
        stop=source.index('        CACHEFLEX_CACHE_KERNEL_END',epilogue)
    else:
        begin=source.index('static void spm_gemm_fused_tile(',source.index('#elif defined(VL_16)'))
        first='x17';epilogue=source.index('        // Store 24 rows',begin)
        stop=source.index('        CACHEFLEX_VL16_KERNEL_END',epilogue)
    def inst(x):return '        "'+x+'\\n"\n'
    ep=inst('mov x5,x0')+inst(f'cbz {first},7f')
    for r in range(rows):ep+=inst(f'st1h {{z{r+8}.h}},p0,[x5]')+inst('add x5,x5,x13')
    ep+=inst('b 8f')+inst('7:')+inst('mov x5,x0')
    for r in range(rows):
        ep+=inst('ld1h {z0.h},p0/z,[x5]')+inst(f'fadd z0.h,z{r+8}.h,z0.h')+inst('st1h {z0.h},p0,[x5]')+inst('add x5,x5,x13')
    ep+=inst('8:')+inst('add x26,x26,x25')+inst('mov x27,x5')+inst('subs x23,x23,#1')+inst('bgt 2b')
    source=source[:epilogue]+ep+source[stop:]
    # Apply only instruction strings and the selected function's A stride.
    def remove(m):
        instruction=m[1]
        acc=re.match(r'(?:fmla|mov) z(\d+)\.',instruction)
        if acc and int(acc[1])>=8+rows:return '""'
        load=re.match(r'ld1rqh \{z(\d+)\.h\}',instruction)
        unused=(1,5,6,7) if rows==8 else (5,7)
        if load and int(load[1]) in unused:return '""'
        return m[0]
    # Don't rewrite the unrelated VL8 implementation in the same header.
    macro_start=source.index('#elif defined(VL_16)') if variant=='spm' else source.index('#ifdef CACHEFLEX_KERNEL_PC_LABELS')
    prefix,body=source[:macro_start],source[macro_start:]
    body=re.sub(r'"([^"\n]*\\n)"',remove,body)
    body=body.replace('K * 48',f'K * {rows*2}').replace('x21, x21, #48',f'x21, x21, #{rows*2}')
    (out/filename).write_text(prefix+body)
