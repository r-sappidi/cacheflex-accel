"""Build opt-in retry-isolation model and freeze a standalone binary."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/cacheflex_flash_attention/async_staging_20260921'
BUILD=OUT/'build_v2'

def save(name,value):
    (BUILD/name).write_text(json.dumps(value,indent=2)+'\n')

def main():
    BUILD.mkdir()
    env=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib',
        PYTHON_CONFIG='/data2/rsappidi/conda-envs/cacheflex-gem5/bin/python3-config',
        PROTOC='/usr/bin/protoc',CCFLAGS_EXTRA='-gdwarf-2',GEM5PY_CCFLAGS_EXTRA='-gdwarf-2',
        LINKFLAGS_EXTRA='-L/usr/lib64',CACHEFLEX_LINK_MANIFEST=str(BUILD/'link_manifest.json'))
    save('status.json',dict(stage='building',pid=os.getpid(),updated=time.time()))
    cmd=['/data2/rsappidi/conda-envs/cacheflex-gem5/bin/scons','build/ARM_MESI_Three_Level_SPM/gem5.opt','-j8','--linker=gold']
    save('command.json',cmd)
    try:
        with (BUILD/'build.log').open('w') as log:
            subprocess.run(cmd,cwd=ROOT/'gem5',env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        manifest=json.loads((BUILD/'link_manifest.json').read_text())
        link=manifest['argv'];link[link.index('-o')+1]=str(BUILD/'gem5.opt')
        le=dict(manifest['environment']);le['LD_LIBRARY_PATH']=env['LD_LIBRARY_PATH']
        with (BUILD/'link.log').open('w') as log:
            subprocess.run(link,cwd=ROOT/'gem5/build/ARM_MESI_Three_Level_SPM',env=le,stdout=log,stderr=subprocess.STDOUT,check=True)
        shutil.copytree(ROOT/'gem5/configs',BUILD/'configs',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        h=hashlib.sha256((BUILD/'gem5.opt').read_bytes()).hexdigest()
        save('status.json',dict(stage='built_requires_validation',pid=os.getpid(),sha256=h,updated=time.time()))
        (BUILD/'simulator.sha256').write_text(h+'\n')
    except BaseException as e:
        save('status.json',dict(stage='failed',error=repr(e),updated=time.time()));raise

if __name__=='__main__':main()
