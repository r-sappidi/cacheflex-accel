"""Rebuild pristine MICRO for timing calibration only; no old FA queue resumed."""
import json
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import time
from build_access_timing import ROOT, sha, save

OUT=ROOT/'results/cacheflex_flash_attention/timing_alignment_20260922'
UP=ROOT/'results/cacheflex_flash_attention/micro_upstream_20260918'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--finish',action='store_true');args=parser.parse_args()
    out=OUT/'classic_build_v2'
    if args.finish:
        assert (UP/'gem5/build/ARM/gem5.opt').is_file()
        shutil.copy2(out/'status.json',out/'first_attempt_status.json')
    else:out.mkdir(parents=True,exist_ok=False)
    def status(stage,**kw):save(out/'status.json',dict(stage=stage,pid=os.getpid(),updated=time.time(),**kw))
    try:
        assert subprocess.check_output(['git','-C',str(UP),'rev-parse','HEAD'],text=True).strip()=='82ac0b9de1ed408ce5cd53067267c8a855b4eef3'
        assert not subprocess.check_output(['git','-C',str(UP),'diff','--name-only','--','gem5/src','gem5/configs'],text=True).strip()
        env=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib',
            PYTHON_CONFIG='/data2/rsappidi/conda-envs/cacheflex-gem5/bin/python3-config',
            CCFLAGS_EXTRA='-gdwarf-2',GEM5PY_CCFLAGS_EXTRA='-gdwarf-2',LINKFLAGS_EXTRA='-L/usr/lib64')
        cmd=['/data2/rsappidi/conda-envs/cacheflex-gem5/bin/scons','build/ARM/gem5.opt','-j16','--linker=gold','--config=force',
             'CXX=g++ -gdwarf-2','CC=gcc -gdwarf-2','CCFLAGS_EXTRA=-gdwarf-2','GEM5PY_CCFLAGS_EXTRA=-gdwarf-2',
             'PYTHON_CONFIG=/data2/rsappidi/conda-envs/cacheflex-gem5/bin/python3-config']
        if not args.finish:
            save(out/'command.json',cmd);status('building')
            with (out/'build.log').open('w') as log:
                subprocess.run(cmd,cwd=UP/'gem5',env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        else:
            retry=[c for c in cmd if c!='--config=force']+['--verbose','--debug=explain']
            save(out/'retry_command.json',retry)
        shutil.copy2(UP/'gem5/build/ARM/gem5.opt',out/'classic.opt')
        shutil.copytree(UP/'gem5/configs',out/'configs')
        save(out/'manifest.json',dict(revision='82ac0b9de1ed408ce5cd53067267c8a855b4eef3',
            source_clean=True,sha256=sha(out/'classic.opt'),configs={str(p.relative_to(out)):sha(p) for p in (out/'configs').rglob('*.py')},
            note='Second identical-flag SCons invocation rebuilt generated dependencies and emitted the executable; source/configs unmodified.'))
        status('built',sha256=sha(out/'classic.opt'))
    except BaseException as exc:status('failed',error=repr(exc));raise

if __name__=='__main__':main()
