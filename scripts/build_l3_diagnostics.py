"""Isolated diagnostic build; four compile workers, resource guard, frozen inputs."""
import json, os, shutil, subprocess, time
from pathlib import Path
from build_access_timing import sha,save
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/cacheflex_l3_spm/diagnostics_20260929'
OLD=ROOT/'results/cacheflex_l3_spm/direct_link_20260928/build_v4'
BUILD=BASE/os.environ.get('CACHEFLEX_DIAGNOSTIC_BUILD','build_v1')
def main():
    BUILD.mkdir(parents=True)
    def status(stage,**kw):save(BUILD/'status.json',dict(stage=stage,pid=os.getpid(),updated=time.time(),**kw))
    try:
        status('resource_wait')
        from l3_diagnostic_profiles import resources, can_launch
        while not can_launch(resources(), simulator=False):time.sleep(10)
        status('preserving')
        rels={str(p.relative_to(OLD/'sources')) for p in (OLD/'sources').rglob('*') if p.is_file()}
        rels.update(['mem/ruby/network/garnet/'+n for n in ('GarnetNetwork.py','GarnetNetwork.hh','GarnetNetwork.cc','NetworkInterface.hh','NetworkInterface.cc')])
        rels.update(['mem/ruby/system/SConscript','mem/request.hh','cpu/o3/lsq_unit.cc','cpu/o3/lsq.hh','cpu/o3/cpu.cc',
            'mem/ruby/network/Network.cc',
            'mem/cache/replacement_policies/tree_plru_rp.cc',
            'mem/cache/replacement_policies/tree_plru_rp.test.cc',
            'mem/ruby/protocol/RubySlicc_Exports.sm','mem/ruby/protocol/RubySlicc_Types.sm',
            'mem/ruby/protocol/MESI_Three_Level_SPM-msg.sm',
            'mem/ruby/protocol/MESI_Three_Level_SPM-L1cache.sm',
            'mem/ruby/protocol/MESI_Three_Level_SPM-L2cache.sm',
            'mem/ruby/structures/CacheMemory.hh','mem/ruby/structures/CacheMemory.cc'])
        inputs={}
        for rel in sorted(rels):
            p=ROOT/'gem5/src'/rel;dst=BUILD/'sources'/rel
            dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst);inputs[str(p)]=sha(p)
        save(BUILD/'source_inputs.json',inputs)
        env=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib',
            PYTHON_CONFIG='/data2/rsappidi/conda-envs/cacheflex-gem5/bin/python3-config',
            PROTOC='/usr/bin/protoc',CCFLAGS_EXTRA='-gdwarf-2',GEM5PY_CCFLAGS_EXTRA='-gdwarf-2',
            LINKFLAGS_EXTRA='-L/usr/lib64',CACHEFLEX_LINK_MANIFEST=str(BUILD/'link_manifest.json'))
        status('building')
        with (BUILD/'build.log').open('w') as f:
            subprocess.run(['/data2/rsappidi/conda-envs/cacheflex-gem5/bin/scons',
                'build/ARM_MESI_Three_Level_SPM/gem5.opt','-j4','--linker=gold'],
                cwd=ROOT/'gem5',env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        link=json.loads((BUILD/'link_manifest.json').read_text());cmd=link['argv']
        cmd[cmd.index('-o')+1]=str(BUILD/'gem5.opt');status('linking')
        with (BUILD/'link.log').open('w') as f:
            subprocess.run(cmd,cwd=ROOT/'gem5/build/ARM_MESI_Three_Level_SPM',
                env=dict(link['environment'],LD_LIBRARY_PATH=env['LD_LIBRARY_PATH']),stdout=f,stderr=subprocess.STDOUT,check=True)
        for p,h in inputs.items():assert sha(p)==h,('source changed',p)
        shutil.copytree(OLD/'configs',BUILD/'configs',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        shutil.copy2(ROOT/'scripts/l3_diagnostics_se.py',BUILD/'configs/deprecated/example/l3_diagnostics_se.py')
        files=[BUILD/'gem5.opt',*list((BUILD/'configs').rglob('*.py')),
               *[p for p in (BUILD/'sources').rglob('*') if p.is_file()]]
        save(BUILD/'frozen.json',{str(p):sha(p) for p in files})
        status('built_requires_validation',sha256=sha(BUILD/'gem5.opt'))
    except BaseException as e:status('failed',error=repr(e));raise
if __name__=='__main__':main()
