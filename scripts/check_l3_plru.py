"""Run native TreePLRU regressions after the isolated simulator build finishes."""
import json, os, subprocess, time
from pathlib import Path
from build_access_timing import save

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/cacheflex_l3_spm/reduction_20260928'
BUILD = BASE / os.environ.get('CACHEFLEX_L3_REDUCTION_BUILD', 'build_v2')
OUT = BUILD / os.environ.get('CACHEFLEX_PLRU_TEST_ATTEMPT', 'plru_tests_v3')

def main():
    OUT.mkdir()
    save(OUT/'status.json', dict(stage='waiting', pid=os.getpid()))
    try:
        while True:
            b=json.loads((BUILD/'status.json').read_text())
            assert b['stage'] != 'failed', b
            if b['stage'] == 'built_requires_validation': break
            time.sleep(5)
        env=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib',
            PYTHON_CONFIG='/data2/rsappidi/conda-envs/cacheflex-gem5/bin/python3-config',
            PROTOC='/usr/bin/protoc',CCFLAGS_EXTRA='-gdwarf-2',GEM5PY_CCFLAGS_EXTRA='-gdwarf-2',
            LINKFLAGS_EXTRA='-L/usr/lib64',CACHEFLEX_UNIT_LINK_MANIFEST=str(OUT/'link_manifest.json'))
        target='build/ARM_MESI_Three_Level_SPM/mem/cache/replacement_policies/tree_plru_rp.test.opt'
        cmd=['/data2/rsappidi/conda-envs/cacheflex-gem5/bin/scons',target,'-j4','--linker=gold']
        save(OUT/'command.json',cmd)
        save(OUT/'status.json',dict(stage='building_tests',pid=os.getpid()))
        with (OUT/'build.log').open('w') as f:
            subprocess.run(cmd,cwd=ROOT/'gem5',env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        link=json.loads((OUT/'link_manifest.json').read_text())
        link['argv'][link['argv'].index('-o')+1]=str(OUT/'tree_plru_rp.test.opt')
        with (OUT/'link.log').open('w') as f:
            subprocess.run(link['argv'],cwd=ROOT/'gem5/build/ARM_MESI_Three_Level_SPM',
                env=link['environment'],stdout=f,stderr=subprocess.STDOUT,check=True)
        with (OUT/'run.log').open('w') as f:
            subprocess.run([str(OUT/'tree_plru_rp.test.opt')],cwd=ROOT,env=env,
                stdout=f,stderr=subprocess.STDOUT,check=True)
        save(OUT/'status.json',dict(stage='passed',pid=os.getpid()))
    except BaseException as e:
        save(OUT/'status.json',dict(stage='failed',error=repr(e)));raise

if __name__=='__main__': main()
