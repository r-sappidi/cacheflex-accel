"""Independent two-way O-resident guest and matched coherent control.

Leaves all shared drivers/builders unchanged. K/V and Q stay coherent;
FP16 GEMM and FP32 online-state rounding order are preserved.
"""
import json
from pathlib import Path
import subprocess
import sys
from build_single_head import build as single_build

HERE=Path(__file__).resolve().parent
CONFIG=dict(placement='cache',group=4,overlap=0,unroll=1,order=0,
            rows=16,pipeline=1,fusion=0,offset=0,pinned=0,interleave=0,
            stride_fix=0,pv_fullwidth=1,rich=1)

def transform(source):
    def replace(a,b):
        nonlocal source
        assert source.count(a)==1,(a,source.count(a))
        source=source.replace(a,b)
    source='#include "output_resident.hpp"\n'+source
    replace('    for(size_t group_begin=0;group_begin<n_ib;group_begin+=ib_group){',
            '    cf_o_claim(Oacc.data(),group_rows);\n'
            '    for(size_t group_begin=0;group_begin<n_ib;group_begin+=ib_group){')
    replace('std::fill_n(Oacc.data(),active_rows*d,0.f);',
            'for(size_t rr=0;rr<active_rows;++rr) cf_o_store(Oacc.data(),rr,svdup_n_f32(0.f));')
    replace('else scale_f32_sve(Oacc.data()+(local_i+r)*d,a,d);',
            'else cf_o_scale(Oacc.data(),local_i+r,a);')
    replace('} else acc_f16_to_f32_sve(PV.data()+r*d,Oacc.data()+(local_i+row+r)*d,d);',
            '} else cf_o_add(Oacc.data(),local_i+row+r,PV.data()+r*d);')
    replace('svld1_f32(pg,Oacc.data()+r*d+x)','cf_o_load(Oacc.data(),r)')
    replace('    CF_PHASE(final_sync);','    cf_o_release(group_rows);\n    CF_PHASE(final_sync);')
    replace('if(T<=0 || Br<=0 || T%(Cores*Br)!=0) return 2;',
            'if(T<=0 || Br<=0 || T%(Cores*Br)!=0 || Br*CACHEFLEX_FLASH_IB_GROUP>64) return 2;')
    return source

def build(out,spm):
    single_build(out,CONFIG)
    (out/'driver.cpp').write_text(transform((out/'driver.cpp').read_text()))
    (out/'output_resident.hpp').write_bytes((HERE/'output_resident.hpp').read_bytes())
    script=(out/'build_command.sh').read_text()
    script=script.replace('-std=c++17',f'-DCACHEFLEX_OUTPUT_SPM={int(spm)} -std=c++17')
    (out/'build_command.sh').write_text(script)
    subprocess.run(['bash',str(out/'build_command.sh')],check=True)
    # Two existing exporter repairs only touch post-ROI code.
    from fa_export_patch import patch
    sys.path.insert(0,str(HERE.parents[1]/'scripts'))
    from patch_timing_export import patch as refresh
    refresh(patch(out/'guest'),out/'guest.final')
    (out/'output_resident.json').write_text(json.dumps(dict(CONFIG,output_spm=spm),indent=2))

if __name__=='__main__':build(Path(sys.argv[1]).resolve(),sys.argv[2]=='spm')
