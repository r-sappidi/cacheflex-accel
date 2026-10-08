"""Preserve/build an isolated finite SPM address-only resolution model."""
import os
import shutil
import sys
os.environ['CACHEFLEX_TIMING_ROOT']='results/cacheflex_flash_attention/spm_early_address_20260924'
os.environ['CACHEFLEX_TIMING_BUILD']='build_v4'
import build_access_timing as b

FILES=['cpu/o3/BaseO3CPU.py','cpu/o3/lsq_unit.hh','cpu/o3/lsq_unit.cc',
       'cpu/static_inst.hh','arch/arm/isa/templates/sve_mem.isa',
       'arch/arm/isa/insts/sve_mem.isa']
if __name__=='__main__':
    if '--preserve' in sys.argv:
        for rel in FILES:
            p=b.OUT/'preserved/gem5/src'/rel;p.parent.mkdir(parents=True,exist_ok=True)
            assert not p.exists(),p
            shutil.copy2(b.ROOT/'gem5/src'/rel,p)
        b.save(b.OUT/'preserved/hashes.json',{rel:b.sha(b.ROOT/'gem5/src'/rel) for rel in FILES})
    else:
        b.main()
        old=b.ROOT/'results/cacheflex_flash_attention/spm_forwarding_20260924/build'
        shutil.copytree(old/'configs',b.BUILD/'configs',dirs_exist_ok=True)
        rels=set(FILES)|{str(p.relative_to(old/'sources')) for p in (old/'sources').rglob('*') if p.is_file()}
        for rel in rels:
            p=b.BUILD/'sources'/rel;p.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(b.ROOT/'gem5/src'/rel,p)
        files=[b.BUILD/'gem5.opt']+[p for p in (b.BUILD/'sources').rglob('*') if p.is_file()]
        files+=list((b.BUILD/'configs').rglob('*.py'))
        b.save(b.BUILD/'frozen.json',{str(p):b.sha(p) for p in files})
