"""Isolated variants: constant d64 staging stride, full-lane PV, or both."""
from pathlib import Path
import json,subprocess,sys
from build_correct_search import build
from make_pv_fullwidth import generate

def build_variant(out,c):
    build(out,c)
    source=(out/'driver.cpp').read_text()
    if c.get('stride_fix'):
        # Shared stage_kv is called only after main's d==64 guard.
        before='static void stage_kv(const __fp16 *KT,const __fp16 *V,int T,int d,'
        assert source.count(before)==1
        source=source.replace(before,before.replace('int d,','int runtime_d,'))
        start=source.index('static void stage_kv(');body=source.index('{',start)
        source=source[:body+1]+'\n    constexpr int d=64; // validated by main before workers start\n'+source[body+1:]
    if c.get('pv_fullwidth'):
        generate(out/'pv_fullwidth.hpp')
        source='#include "pv_fullwidth.hpp"\n'+source
        replacements=[('PV(8*d),pap(8*Bc);','PV(16*d),pap(16*Bc);'),
          ('std::memset(PV.data(),0,8*d*2);','std::memset(PV.data(),0,16*d*2);'),
          ('std::memset(pap.data(),0,8*Bc*2);','std::memset(pap.data(),0,16*Bc*2);'),
          ('for(size_t mb=0;mb<abs;++mb){size_t row=mb*MT_VAL,rows=std::min((size_t)MT_VAL,(size_t)br-row);',
           'for(size_t mb=0;mb<abs;){size_t step=abs-mb>=2?2:1;size_t row=mb*MT_VAL,rows=std::min(step*(size_t)MT_VAL,(size_t)br-row); mb+=step;'),
          ('if(CACHEFLEX_FLASH_STAGE_V) sve_spm_pv_8x64_vl4(pap.data(),PV.data(),bc,d,vb);\n          else sve_cache_pv_8x64(pap.data(),s.V.data()+(size_t)j*d,PV.data(),bc,d);',
           f'''if(step==2) {{
            if(CACHEFLEX_FLASH_STAGE_V) cf_pv16_spm(pap.data(),vb+{c['offset']},PV.data(),bc,d);
            else cf_pv16_cache(pap.data(),(size_t)(s.V.data()+(size_t)j*d),PV.data(),bc,d);
          }} else {{
            if(CACHEFLEX_FLASH_STAGE_V) sve_spm_pv_8x64_vl4(pap.data(),PV.data(),bc,d,vb);
            else sve_cache_pv_8x64(pap.data(),s.V.data()+(size_t)j*d,PV.data(),bc,d);
          }}''')]
        for old,new in replacements:
            assert source.count(old)==1,old
            source=source.replace(old,new)
    (out/'driver.cpp').write_text(source)
    subprocess.run(['bash',str(out/'build_command.sh')],check=True)

if __name__=='__main__':build_variant(Path(sys.argv[1]).resolve(),json.loads(Path(sys.argv[2]).read_text()))
