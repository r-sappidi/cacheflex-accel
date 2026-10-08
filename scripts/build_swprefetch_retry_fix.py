"""Build a frozen simulator with prefetch-completion retry notification."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/ruby_swprefetch_retry_fix_20260921'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,data):(OUT/name).write_text(json.dumps(data,indent=2))

env=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib',
         PYTHON_CONFIG='/data2/rsappidi/conda-envs/cacheflex-gem5/bin/python3-config',
         PROTOC='/usr/bin/protoc',CCFLAGS_EXTRA='-gdwarf-2',GEM5PY_CCFLAGS_EXTRA='-gdwarf-2',
         LINKFLAGS_EXTRA='-L/usr/lib64',CACHEFLEX_LINK_MANIFEST=str(OUT/'link_manifest.json'))
save('status.json',dict(stage='building',pid=os.getpid(),updated=time.time()))
shutil.copy2(ROOT/'gem5/src/mem/ruby/system/Sequencer.cc',OUT/'Sequencer.fixed.cc')
cmd=['/data2/rsappidi/conda-envs/cacheflex-gem5/bin/scons','build/ARM_MESI_Three_Level_SPM/gem5.opt','-j8','--linker=gold']
save('command.json',cmd)
try:
    with (OUT/'build.log').open('w') as log:
        subprocess.run(cmd,cwd=ROOT/'gem5',env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    manifest=json.loads((OUT/'link_manifest.json').read_text())
    cmd=manifest['argv'];cmd[cmd.index('-o')+1]=str(OUT/'gem5.opt')
    linkenv=dict(manifest['environment']);linkenv['LD_LIBRARY_PATH']=env['LD_LIBRARY_PATH']
    # Root linking has previously been skipped by this workspace's SCons setup.
    # Use its freshly captured exact link argv, never the stale target binary.
    with (OUT/'link.log').open('w') as log:
        subprocess.run(cmd,cwd=ROOT/'gem5/build/ARM_MESI_Three_Level_SPM',env=linkenv,stdout=log,stderr=subprocess.STDOUT,check=True)
    save('status.json',dict(stage='built_requires_validation',pid=os.getpid(),updated=time.time(),simulator_sha256=sha(OUT/'gem5.opt'),source_sha256=sha(OUT/'Sequencer.fixed.cc')))
except BaseException as e:
    save('status.json',dict(stage='failed',error=repr(e),updated=time.time()));raise
