"""Keep per-core packed Q resident across all K/V panels; coherent K and V.

Initial layout supports <=64 query rows/group in one 8KiB physical SPM way.
SPM loads fetch aligned64B then TBL selects/replicates eight FP16 A operands, preserving
the existing 16-row pipelined QK FMLA order. No hardware changes.
"""
from pathlib import Path
import json
import subprocess
import sys
from build_single_head import build as single_build


def q_kernel(source):
    start=source.index('static void cf_wide_qk_cache(')
    end=source.index('\n}',start)+2
    body=source[start:end].replace('cf_wide_qk_cache','cf_wide_qk_qspm')
    anchor='"ptrue p1.d\\n"'
    assert body.count(anchor)==1
    body=body.replace(anchor,anchor+'\n"mov x16, #8\\n"\n"whilelt p2.d, xzr, x16\\n"\n"index z31.b, #0, #1\\n"\n"and z31.b, z31.b, #15\\n"')
    for z,ptr in [(0,21),(1,23),(5,21),(6,23)]:
        old=f'"ld1rqh {{z{z}.h}}, p0/z, [x{ptr}]\\n"'
        assert body.count(old)==(2 if z in (0,1) else 1),old
        # Ruby returns line-sized SPM responses: align the load, then select
        # the correct 16B quarter and replicate it with the byte table.
        body=body.replace(old,
            f'"and x16, x{ptr}, #0x1fc0\\n"\n"and x17, x{ptr}, #48\\n"\n'
            f'"dup z30.b, w17\\n"\n"add z30.b, z30.b, z31.b\\n"\n'
            f'"spm.ld1qd z{z}.d, p2/z, [x16]\\n"\n"tbl z{z}.b, {{z{z}.b}}, z30.b\\n"')
    body=body.replace('"p0","p1",','"p0","p1","p2",')
    body=body.replace('"x16",','"x16","x17",')
    return source+'\n'+body+'\n'


def build(out,c):
    assert c['rows']==16 and c['pipeline']==1 and c['placement']=='cache'
    assert not c['overlap'] and not c['offset']
    single_build(out,c)
    source=(out/'driver.cpp').read_text()
    if c.get('q_resident',False):
        kernels=out/'kernels.hpp';kernels.write_text(q_kernel(kernels.read_text()))
        anchor='     std::fill_n(Oacc.data(),active_rows*d,0.f);'
        assert source.count(anchor)==1
        source=source.replace(anchor,'''     // Packed Q is unique to this worker and remains resident for this group.
     for(size_t off=0;off<group_count*abmax*MT_VAL*d*2;off+=64)
       copy_line(off,reinterpret_cast<const char*>(qap.data())+off);
'''+anchor)
        old='cf_wide_qk_cache(q+mb*8*d,s.Kcache.data()'
        new='cf_wide_qk_qspm(reinterpret_cast<const __fp16*>((local_ib*abmax+mb)*8*d*2),s.Kcache.data()'
        assert source.count(old)==1
        source=source.replace(old,new)
        # Every tile is exactly 16 rows or larger; no coherent fallback may
        # consume source Q after its destructive move into SPM.
        anchor='if(T<=0 || Br<=0 || T%(Cores*Br)!=0) return 2;'
        assert source.count(anchor)==1
        source=source.replace(anchor,anchor+'\n if(Br%16 || Br*CACHEFLEX_FLASH_IB_GROUP>64) return 2;')
    (out/'driver.cpp').write_text(source)
    subprocess.run(['bash',str(out/'build_command.sh')],check=True)
    (out/'q_resident.json').write_text(json.dumps(c,indent=2))


if __name__=='__main__':build(Path(sys.argv[1]).resolve(),json.loads(Path(sys.argv[2]).read_text()))
