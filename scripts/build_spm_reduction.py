"""Isolated reduction simulator build; never replace any frozen active input."""
import json, os, shutil, subprocess, time
from pathlib import Path
from build_access_timing import sha, save
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/cacheflex_flash_attention/near_spm_reduction_20260927'
OLD=ROOT/'results/cacheflex_flash_attention/spm_set_spread_20260927/build_retry_v3'
BUILD=BASE/os.environ.get('CACHEFLEX_REDUCTION_BUILD','build')

def main():
    BUILD.mkdir(parents=True)
    def status(stage,**kw): save(BUILD/'status.json',dict(stage=stage,pid=os.getpid(),updated=time.time(),**kw))
    try:
        status('preserving')
        rels={str(p.relative_to(OLD/'sources')) for p in (OLD/'sources').rglob('*') if p.is_file()}
        rels.update(['mem/cacheflex_reduction.hh','cpu/o3/lsq.cc',
            'mem/ruby/system/SequencerReduction.cc','mem/ruby/system/Sequencer.cc',
            'mem/ruby/system/Sequencer.hh','mem/ruby/system/Sequencer.py','mem/ruby/system/SConscript'])
        hashes={}
        for rel in sorted(rels):
            p=ROOT/'gem5/src'/rel;dst=BUILD/'sources'/rel
            dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst);hashes[str(p)]=sha(p)
        save(BUILD/'source_inputs.json',hashes)
        save(BUILD/'prior_reference.json',dict(path=str(OLD),frozen=json.loads((OLD/'frozen.json').read_text())))
        env=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib',
            PYTHON_CONFIG='/data2/rsappidi/conda-envs/cacheflex-gem5/bin/python3-config',
            PROTOC='/usr/bin/protoc',CCFLAGS_EXTRA='-gdwarf-2',GEM5PY_CCFLAGS_EXTRA='-gdwarf-2',
            LINKFLAGS_EXTRA='-L/usr/lib64',CACHEFLEX_LINK_MANIFEST=str(BUILD/'link_manifest.json'))
        status('building')
        with (BUILD/'build.log').open('w') as f:
            subprocess.run(['/data2/rsappidi/conda-envs/cacheflex-gem5/bin/scons',
                'build/ARM_MESI_Three_Level_SPM/gem5.opt','-j8','--linker=gold'],cwd=ROOT/'gem5',env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        link=json.loads((BUILD/'link_manifest.json').read_text());cmd=link['argv']
        cmd[cmd.index('-o')+1]=str(BUILD/'gem5.opt');status('linking')
        with (BUILD/'link.log').open('w') as f:
            subprocess.run(cmd,cwd=ROOT/'gem5/build/ARM_MESI_Three_Level_SPM',
                env=dict(link['environment'],LD_LIBRARY_PATH=env['LD_LIBRARY_PATH']),stdout=f,stderr=subprocess.STDOUT,check=True)
        for p,h in hashes.items(): assert sha(p)==h,('source changed',p)
        shutil.copytree(OLD/'configs',BUILD/'configs',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        shutil.copy2(ROOT/'scripts/reduction_se.py',BUILD/'configs/deprecated/example/reduction_se.py')
        files=[BUILD/'gem5.opt',*list((BUILD/'configs').rglob('*.py')),
               *[p for p in (BUILD/'sources').rglob('*') if p.is_file()]]
        save(BUILD/'frozen.json',{str(p):sha(p) for p in files})
        status('built_requires_validation',sha256=sha(BUILD/'gem5.opt'))
    except BaseException as e:status('failed',error=repr(e));raise
if __name__=='__main__':main()
