"""Freeze debug-only coherent load lifetime instrumentation."""
import os
import shutil
os.environ['CACHEFLEX_TIMING_ROOT']='results/cacheflex_flash_attention/coherent_load_path_20260923'
os.environ['CACHEFLEX_TIMING_BUILD']=os.environ.get('CACHEFLEX_LOAD_BUILD','build_v2')
import build_access_timing as b

if __name__=='__main__':
    b.main()
    prior=b.ROOT/'results/cacheflex_flash_attention/ifetch_edge_20260923/build'
    shutil.copytree(prior/'configs',b.BUILD/'configs',dirs_exist_ok=True)
    paths=[p.relative_to(prior/'sources') for p in (prior/'sources').rglob('*') if p.is_file()]
    paths.append(__import__('pathlib').Path('cpu/o3/cacheflex_load_trace.hh'))
    for rel in paths:
        dest=b.BUILD/'sources'/rel
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(b.ROOT/'gem5/src'/rel,dest)
    files=[b.BUILD/'gem5.opt']+[p for p in (b.BUILD/'sources').rglob('*') if p.is_file()]
    files+=list((b.BUILD/'configs').rglob('*.py'))
    b.save(b.BUILD/'frozen.json',{str(p):b.sha(p) for p in files})
