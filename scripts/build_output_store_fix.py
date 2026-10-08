"""Freeze the wide-SPM-store acknowledgment correctness repair."""
import os
import shutil
os.environ['CACHEFLEX_TIMING_ROOT']='results/cacheflex_flash_attention/output_resident_20260924'
os.environ['CACHEFLEX_TIMING_BUILD']=os.environ.get('CACHEFLEX_OUTPUT_BUILD','build_model_v4')
import build_access_timing as b

if __name__=='__main__':
    b.main()
    prior=b.ROOT/'results/cacheflex_flash_attention/ifetch_edge_20260923/build'
    shutil.copytree(prior/'configs',b.BUILD/'configs',dirs_exist_ok=True)
    for p in (prior/'sources').rglob('*'):
        if p.is_file():
            src=b.ROOT/'gem5/src'/p.relative_to(prior/'sources')
            dst=b.BUILD/'sources'/p.relative_to(prior/'sources')
            dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    extra='mem/ruby/common/DataBlock.cc'
    (b.BUILD/'sources'/extra).parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(b.ROOT/'gem5/src'/extra,b.BUILD/'sources'/extra)
    files=[b.BUILD/'gem5.opt']+[p for p in (b.BUILD/'sources').rglob('*') if p.is_file()]
    files+=list((b.BUILD/'configs').rglob('*.py'))
    b.save(b.BUILD/'frozen.json',{str(p):b.sha(p) for p in files})
