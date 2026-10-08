"""Build/freeze debug-only message/flit instrumentation without editing live inputs."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/os.environ.get('CACHEFLEX_TIMING_ROOT',
    'results/cacheflex_flash_attention/access_timing_20260922')
BUILD = OUT/os.environ.get('CACHEFLEX_TIMING_BUILD','build')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def save(path, value):
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(value, indent=2)); tmp.replace(path)


def main():
    BUILD.mkdir(exist_ok=True)
    def state(stage, **kw):
        save(BUILD/'status.json',dict(stage=stage,pid=os.getpid(),updated=time.time(),**kw))
    env=dict(os.environ, LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib',
        PYTHON_CONFIG='/data2/rsappidi/conda-envs/cacheflex-gem5/bin/python3-config',
        PROTOC='/usr/bin/protoc',CCFLAGS_EXTRA='-gdwarf-2',GEM5PY_CCFLAGS_EXTRA='-gdwarf-2',
        LINKFLAGS_EXTRA='-L/usr/lib64',CACHEFLEX_LINK_MANIFEST=str(BUILD/'link_manifest.json'))
    try:
        state('building')
        with (BUILD/'build.log').open('a') as log:
            subprocess.run(['/data2/rsappidi/conda-envs/cacheflex-gem5/bin/scons',
                'build/ARM_MESI_Three_Level_SPM/gem5.opt','-j8','--linker=gold'],
                cwd=ROOT/'gem5',env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        manifest=json.loads((BUILD/'link_manifest.json').read_text())
        cmd=manifest['argv'];cmd[cmd.index('-o')+1]=str(BUILD/'gem5.opt')
        state('linking')
        with (BUILD/'link.log').open('w') as log:
            subprocess.run(cmd,cwd=ROOT/'gem5/build/ARM_MESI_Three_Level_SPM',
                env=dict(manifest['environment'],LD_LIBRARY_PATH=env['LD_LIBRARY_PATH']),
                stdout=log,stderr=subprocess.STDOUT,check=True)
        shutil.copytree(ROOT/'results/cacheflex_flash_attention/micro_resources_20260921/build_v5/configs',
                        BUILD/'configs',dirs_exist_ok=True)
        files=list((OUT/'preserved/gem5/src').rglob('*'))
        sources=[ROOT/'gem5/src'/p.relative_to(OUT/'preserved/gem5/src') for p in files if p.is_file()]
        sources.append(ROOT/'gem5/src/mem/ruby/network/CacheFlexTiming.hh')
        for extra in ('CacheFlexAggregate.hh','CacheFlexAggregate.cc','IngressBypass.hh'):
            p=ROOT/'gem5/src/mem/ruby/network'/extra
            if p.exists():sources.append(p)
        # Include unchanged observation hooks in aggregate builds as well.
        for p in (ROOT/'results/cacheflex_flash_attention/access_timing_20260922/preserved/gem5/src').rglob('*'):
            if p.is_file():
                s=ROOT/'gem5/src'/p.relative_to(ROOT/'results/cacheflex_flash_attention/access_timing_20260922/preserved/gem5/src')
                if s not in sources:sources.append(s)
        for p in (ROOT/'results/cacheflex_flash_attention/access_aggregate_20260922/preserved/gem5/src').rglob('*'):
            if p.is_file():
                s=ROOT/'gem5/src'/p.relative_to(ROOT/'results/cacheflex_flash_attention/access_aggregate_20260922/preserved/gem5/src')
                if s not in sources:sources.append(s)
        for p in sources:
            dest=BUILD/'sources'/p.relative_to(ROOT/'gem5/src');dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(p,dest)
        save(BUILD/'manifest.json',{str(p):sha(p) for p in sources+[BUILD/'gem5.opt']})
        state('built_requires_validation',sha256=sha(BUILD/'gem5.opt'))
    except BaseException as exc:
        state('failed',error=repr(exc));raise


if __name__=='__main__':main()
