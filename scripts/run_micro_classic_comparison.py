"""Build latest MICRO, gate identical guests, then compare Classic to Ruby.

Core parameters follow the already-running Ruby controls; Classic hierarchy
parameters follow upstream's script. These are not claimed identical models.
"""
import concurrent.futures as cf
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time
from run_micro_fa_ruby_pilot import ROOT, save, sha, numerical
from run_micro_fa_ruby_target import validate

BASE=ROOT/'results/cacheflex_flash_attention'
UP=BASE/'micro_upstream_20260918'
OUT=BASE/'micro_classic_comparison_v7_20260918'
GATE=BASE/'micro_upstream_multitile_gate_v2_20260918'
RUBY=BASE/'micro_upstream_t4096_20260918'
ENV=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',CCFLAGS_EXTRA='-gdwarf-2',GEM5PY_CCFLAGS_EXTRA='-gdwarf-2',PYTHON_CONFIG='/data2/rsappidi/conda-envs/cacheflex-gem5/bin/python3-config',LINKFLAGS_EXTRA='-L/usr/lib64')

def main():
    OUT.mkdir();(OUT/'runs').mkdir()
    state=dict(pid=os.getpid(),completed=[])
    def status(stage):
        state.update(stage=stage,updated=time.time());save(OUT/'status.json',state)
        print(stage,flush=True)
    try:
        assert json.loads((GATE/'status.json').read_text())['stage']=='gates_passed_target_configuration_pending'
        commit=subprocess.check_output(['git','-C',str(UP),'rev-parse','HEAD'],text=True).strip()
        assert commit=='82ac0b9de1ed408ce5cd53067267c8a855b4eef3'
        status('building_latest_micro_classic')
        build=['/data2/rsappidi/conda-envs/cacheflex-gem5/bin/scons','build/ARM/gem5.opt','-j8','--linker=gold','CXX=g++ -gdwarf-2','CC=gcc -gdwarf-2','CCFLAGS_EXTRA=-gdwarf-2','GEM5PY_CCFLAGS_EXTRA=-gdwarf-2','PYTHON_CONFIG=/data2/rsappidi/conda-envs/cacheflex-gem5/bin/python3-config']
        save(OUT/'build_command.json',build)
        with (OUT/'build.log').open('w') as log:
            p=subprocess.Popen(build,cwd=UP/'gem5',env=ENV,stdout=log,stderr=subprocess.STDOUT)
            save(OUT/'build_process.json',dict(pid=p.pid));rc=p.wait()
        assert rc==0,('build',rc)
        sim=UP/'gem5/build/ARM/gem5.opt'
        script=(UP/'experiments/end2end/scripts/sweep_vl16.sh').read_text()
        options=shlex.split(re.search(r'GEM5_CONFIG="(.*?)"',script,re.S).group(1).replace('\\\n',' '))
        ruby_cpu=json.loads((RUBY/'runs/cache_vl16/config.json').read_text())['system']['cpu'][0]
        cpu_params={}
        command=json.loads((RUBY/'runs/cache_vl16/command.json').read_text())
        for i,x in enumerate(command[:-1]):
            if x=='-P':
                key,value=command[i+1].split('=',1)
                if key.startswith('system.cpu[0].') and '.workload' not in key:
                    cpu_params[key]=value
        for i,fu in enumerate(ruby_cpu['fuPool']['FUList']):
            cpu_params[f'system.cpu[0].fuPool.FUList[{i}].count']=str(fu['count'])
        fixed=[];i=0
        while i<len(options):
            x=options[i]
            if x=='-P':
                key,value=options[i+1].split('=',1)
                fixed+=['-P',key+'='+cpu_params.get(key,value)];i+=2
            else:fixed.append(x);i+=1
        present={fixed[i+1].split('=',1)[0] for i,x in enumerate(fixed[:-1]) if x=='-P'}
        for key,value in cpu_params.items():
            if key not in present:fixed+=['-P',key+'='+value]
        save(OUT/'plan.json',dict(commit=commit,simulator=str(sim),simulator_sha=sha(sim),max_simulators=2,
            target=dict(T=4096,Hq=32,Hkv=8,d=64,Br=64,Bc=384,VL=[16,4]),
            precision='FP16 operands/GEMM, FP32 softmax/running O, FP16 stored O',
            source='Identical corrected guest ELF files as active Ruby runs',
            core='Common settings and FU counts match active Ruby, NOT original paper FU-count overrides',
            memory='Upstream Classic cache latencies/prefetch/MSHR configuration; Ruby coherence, network and admission differ',
            roi='First cold stats epoch for both architectures; cache warm epoch excluded from primary comparison'))
        def run(arch,vl,target):
            t=4096 if target else 320
            p=OUT/'runs'/f'{arch}_vl{vl}_t{t}';p.mkdir()
            guest=GATE/'builds'/f'{arch}_vl{vl}/guest'
            argv=f'{t} 32 8 64 1 64 384' if target else '320 1 1 64 1 32 128'
            opts=[f'system.cpu[0].isa[0].sve_vl_se={vl}' if x.startswith('system.cpu[0].isa[0].sve_vl_se=') else x for x in fixed]
            cmd=[str(sim),f'--outdir={p}',str(UP/'gem5/configs/deprecated/example/se.py'),*opts,'--num-cpus=1','--prog-interval=100kHz','-c',str(guest),'-o',argv]
            save(p/'command.json',cmd);save(p/'hashes.json',{str(guest):sha(guest),str(sim):sha(sim)})
            start=time.monotonic()
            with (p/'run.log').open('w') as log:
                child=subprocess.Popen(cmd,env=ENV,stdout=log,stderr=subprocess.STDOUT)
                try:
                    while child.poll() is None:
                        save(p/'progress.json',dict(pid=child.pid,stage='simulating',host_elapsed_s=time.monotonic()-start))
                        try:child.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            if time.monotonic()-start>48*3600:raise RuntimeError('48h timeout')
                finally:
                    if child.poll() is None:child.terminate();child.wait()
            assert child.returncode==0,(p,child.returncode)
            actual=json.loads((p/'config.json').read_text())['system']['cpu'][0]
            fields=['fetchWidth','decodeWidth','renameWidth','commitWidth','dispatchWidth','issueWidth','wbWidth','numROBEntries','numIQEntries','LQEntries','SQEntries','numPhysIntRegs','numPhysFloatRegs','numPhysVecRegs','numPhysVecPredRegs','cacheLoadPorts','cacheStorePorts','spmLoadPorts','spmStorePorts']
            for key in fields:assert actual[key]==ruby_cpu[key],(key,actual[key],ruby_cpu[key])
            # FU operation latencies/issue behavior must also agree, not just counts.
            for a,b in zip(actual['fuPool']['FUList'],ruby_cpu['fuPool']['FUList']):
                assert a['count']==b['count']
                assert [(o['opClass'],o['opLat'],o['pipelined']) for o in a['opList']]==[(o['opClass'],o['opLat'],o['pipelined']) for o in b['opList']]
            save(p/'core_audit.json',dict(passed=True,fields=fields,FU_operations_matched=True))
            save(p/'progress.json',dict(stage='numerical_validation'))
            check=validate(p) if target else numerical(p,320,128)
            assert check['errors']==check['nonfinite']==0,check
            if not target:
                other=GATE/'runs'/f'{arch}_vl{vl}_t320'
                equality={n:sha(p/f'{n}.fp16')==sha(other/f'{n}.fp16') for n in ('Q','K','V','O')}
                save(p/'ruby_output_equivalence.json',equality)
                assert all(equality.values()),equality
            stats={}
            for line in (p/'stats.txt').read_text().split('End Simulation Statistics')[0].splitlines():
                w=line.split()
                if len(w)>1:stats[w[0]]=w[1]
            row=dict(arch=arch,VL=vl,T=t,roi_us=float(stats['simTicks'])/1e6,numerical=check)
            save(p/'accepted.json',row);return row
        for target in (False,True):
            for vl in (16,4):
                status(f'{"target" if target else "gate"}_vl{vl}')
                with cf.ThreadPoolExecutor(max_workers=2) as pool:
                    for row in pool.map(lambda a:run(a,vl,target),('cache','spm')):
                        state['completed'].append(row);status(state['stage'])
        save(OUT/'summary.json',[dict(VL=vl,speedup=next(r['roi_us'] for r in state['completed'] if r['VL']==vl and r['T']==4096 and r['arch']=='cache')/next(r['roi_us'] for r in state['completed'] if r['VL']==vl and r['T']==4096 and r['arch']=='spm')) for vl in (16,4)])
        status('complete')
    except BaseException as e:
        state['error']=repr(e);status('failed');raise

if __name__=='__main__':main()
