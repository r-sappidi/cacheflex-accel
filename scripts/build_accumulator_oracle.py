"""Isolated diagnostic build; never replaces any active frozen executable."""
import os
from pathlib import Path
import shutil
import sys
os.environ['CACHEFLEX_TIMING_ROOT']='results/ai_workloads/swiglu_accumulator_oracle_20260925'
os.environ['CACHEFLEX_TIMING_BUILD']='build'
import build_access_timing as b

FILES=['mem/ruby/system/Sequencer.cc','mem/ruby/system/Sequencer.hh',
       'mem/ruby/system/Sequencer.py']

if __name__=='__main__':
    b.OUT.mkdir(exist_ok=True)
    if '--preserve' in sys.argv:
        for rel in FILES:
            dest=b.OUT/'preserved/gem5/src'/rel
            dest.parent.mkdir(parents=True,exist_ok=True)
            assert not dest.exists()
            shutil.copy2(b.ROOT/'gem5/src'/rel,dest)
    else:
        b.main()
        # Original pilot configs; no MICRO resource/timing wrapper added.
        shutil.copytree(b.ROOT/'gem5/configs',b.BUILD/'pilot_configs',
                        ignore=shutil.ignore_patterns('__pycache__'))
        b.save(b.BUILD/'pilot_frozen.json',{
            str(p):b.sha(p) for p in [b.BUILD/'gem5.opt']+
            list((b.BUILD/'pilot_configs').rglob('*.py'))})
