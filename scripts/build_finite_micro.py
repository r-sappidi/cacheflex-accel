"""Freeze and build finite tracking + MICRO credits without changing live inputs."""
import json,os,shutil,subprocess,time
from pathlib import Path
from build_access_timing import sha,save
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/cacheflex_flash_attention/finite_model_20260926'
BUILD=BASE/os.environ.get('CACHEFLEX_FINITE_BUILD','build')
OLD=ROOT/'results/cacheflex_flash_attention/spm_early_address_20260924/build_v4'

def main():
    BUILD.mkdir()
    def status(stage,**kw):save(BUILD/'status.json',dict(stage=stage,pid=os.getpid(),updated=time.time(),**kw))
    try:
        status('preserving')
        # Capture all source changes relative to the previous frozen simulator.
        rels={str(p.relative_to(OLD/'sources')) for p in (OLD/'sources').rglob('*') if p.is_file()}
        rels|={'mem/ruby/protocol/MESI_Three_Level_SPM-L1cache.sm',
               'mem/ruby/protocol/MESI_Three_Level_SPM-L0cache.sm',
               'mem/ruby/protocol/MESI_Three_Level_SPM-dir.sm','cpu/o3/lsq_unit.cc',
               'cpu/o3/lsq_unit.hh','mem/ruby/system/Sequencer.cc','mem/ruby/system/Sequencer.hh'}
        rels|={'mem/ruby/network/garnet/'+f for f in (
            'GarnetNetwork.py','GarnetNetwork.hh','GarnetNetwork.cc',
            'OutputUnit.hh','OutputUnit.cc','NetworkInterface.hh','NetworkInterface.cc','SwitchAllocator.cc')}
        source_hashes={}
        for rel in rels:
            src=ROOT/'gem5/src'/rel;dst=BUILD/'sources'/rel
            dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
            source_hashes[str(src)]=sha(src)
        save(BUILD/'source_inputs.json',source_hashes)
        env=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib',
            PYTHON_CONFIG='/data2/rsappidi/conda-envs/cacheflex-gem5/bin/python3-config',
            PROTOC='/usr/bin/protoc',CCFLAGS_EXTRA='-gdwarf-2',GEM5PY_CCFLAGS_EXTRA='-gdwarf-2',
            LINKFLAGS_EXTRA='-L/usr/lib64',CACHEFLEX_LINK_MANIFEST=str(BUILD/'link_manifest.json'))
        status('building')
        with (BUILD/'build.log').open('w') as log:
            subprocess.run(['/data2/rsappidi/conda-envs/cacheflex-gem5/bin/scons',
                'build/ARM_MESI_Three_Level_SPM/gem5.opt','-j8','--linker=gold'],cwd=ROOT/'gem5',env=env,
                stdout=log,stderr=subprocess.STDOUT,check=True)
        manifest=json.loads((BUILD/'link_manifest.json').read_text())
        cmd=manifest['argv'];cmd[cmd.index('-o')+1]=str(BUILD/'gem5.opt')
        status('linking')
        with (BUILD/'link.log').open('w') as log:
            subprocess.run(cmd,cwd=ROOT/'gem5/build/ARM_MESI_Three_Level_SPM',
                env=dict(manifest['environment'],LD_LIBRARY_PATH=env['LD_LIBRARY_PATH']),
                stdout=log,stderr=subprocess.STDOUT,check=True)
        for p,h in source_hashes.items():assert sha(p)==h,('source changed during build',p)
        shutil.copytree(OLD/'configs',BUILD/'configs',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        shutil.copy2(ROOT/'scripts/finite_micro_se.py',BUILD/'configs/deprecated/example/bounded_se.py')
        files=[BUILD/'gem5.opt',*list((BUILD/'configs').rglob('*.py')),
               *[p for p in (BUILD/'sources').rglob('*') if p.is_file()]]
        save(BUILD/'frozen.json',{str(p):sha(p) for p in files})
        status('built_requires_validation',sha256=sha(BUILD/'gem5.opt'))
    except BaseException as e:status('failed',error=repr(e));raise

if __name__=='__main__':main()
