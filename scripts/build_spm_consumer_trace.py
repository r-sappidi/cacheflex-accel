"""Isolated build of observational CPU-dependency hooks."""
import os
from pathlib import Path
import shutil
os.environ['CACHEFLEX_TIMING_ROOT']='results/cacheflex_flash_attention/spm_consumer_trace_20260923'
os.environ['CACHEFLEX_TIMING_BUILD']='build'
import build_access_timing as b

if __name__=='__main__':
    b.main()
    shutil.copytree(b.ROOT/'results/cacheflex_flash_attention/timing_alignment_20260922/fa_v3/configs',
                    b.BUILD/'configs',dirs_exist_ok=True)
    for f in ('Sequencer.cc','Sequencer.hh','Sequencer.py'):
        src=b.ROOT/'gem5/src/mem/ruby/system'/f
        dst=b.BUILD/'sources/mem/ruby/system'/f
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    paths=[b.BUILD/'gem5.opt']+[p for p in (b.BUILD/'sources').rglob('*') if p.is_file()]
    paths+=list((b.BUILD/'configs').rglob('*.py'))
    b.save(b.BUILD/'frozen.json',{str(p):b.sha(p) for p in paths})
