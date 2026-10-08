"""Opt-in Bc32/Br16/G1 single-head VL16 proxy; original sources untouched."""
from pathlib import Path
import json
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'benchmarks/cacheflex_flash_attention'))
from fa_export_patch import patch

def build(out,arch):
    old=ROOT/'results/cacheflex_flash_attention/single_head_20260915/builds/spm_g4'
    out.mkdir(parents=True)
    for name in ('driver.cpp','kernels.hpp','pv_fullwidth.hpp','build_command.sh'):
        shutil.copy2(old/name,out/name)
    s=(out/'driver.cpp').read_text()
    s=s.replace('#define NT_VL4 size_t(128)','#define NT_VL4 size_t(32)')
    # Kcache remains panel packed, but each reduction step now has 32 columns.
    for a,b in [('search_bc/128','search_bc/32'),('j/128','j/32'),('nb*128','nb*32'),('d*128','d*32'),('(Bc/128)','(Bc/32)')]:s=s.replace(a,b)
    s=s.replace('if(T<=0 || Br<=0 || T%(Cores*Br)!=0)',
                'if(Bc!=32 || Br!=16 || T<=0 || Br<=0 || T%(Cores*Br)!=0)')
    (out/'driver.cpp').write_text(s)
    k=(out/'kernels.hpp').read_text();start=k.index('static void cf_wide_qk_cache(');end=k.index('static void cf_wide_qk_spm(',start)
    block=k[start:end]
    assert block.count('"ptrue p0.h\\n"')==1 and block.count('"addvl x22, x22, #1\\n"')==3
    block=block.replace('"ptrue p0.h\\n"','"ptrue p0.h, vl32\\n"').replace('"addvl x22, x22, #1\\n"','"add x22, x22, #64\\n"')
    (out/'kernels.hpp').write_text(k[:start]+block+k[end:])
    script=(out/'build_command.sh').read_text().replace(str(old),str(out))
    assert '-DCACHEFLEX_FLASH_IB_GROUP=4' in script
    script=script.replace('-DCACHEFLEX_FLASH_IB_GROUP=4','-DCACHEFLEX_FLASH_IB_GROUP=1')
    script=script.replace('-DCACHEFLEX_FLASH_STAGE_V=1',f'-DCACHEFLEX_FLASH_STAGE_V={int(arch=="spm")}')
    (out/'build_command.sh').write_text(script)
    subprocess.run(['bash',str(out/'build_command.sh')],check=True)
    guest=patch(out/'guest')
    (out/'scaled.json').write_text(json.dumps(dict(arch=arch,Br=16,Bc=32,G=1,d=64,
      V_panel_bytes=4096,guest=str(guest),note='32 active FP16 lanes in QK; full-width PV unchanged. Not an exact timing replica of Bc128.'),indent=2))
    return guest

if __name__=='__main__':build(Path(sys.argv[1]).resolve(),sys.argv[2])
