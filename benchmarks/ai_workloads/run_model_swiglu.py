"""Frozen Llama-3.2-1B-shaped SwiGLU pilots, not a model accuracy claim.

Small precision/copy/config gates precede T128/512/2048 custom-cache/SPM/ACL.
Five calls: first separately, second transitional, last three repeated-call
samples. Variability is descriptive, never a stopping condition.
"""
import concurrent.futures as cf
import configparser
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
import numpy as np
from run_mapping_screen import ROOT,HERE,sha,save
from sweep_mlp_study import epochs
from audit_mlp_math import audit as math_audit
from model_swiglu_reference import generate,validate_acl

OUT=ROOT/'results/ai_workloads/swiglu_model_pilot_v4_20260921'
PRIOR=ROOT/'results/ai_workloads/swiglu_model_pilot_v3_20260921'
FIX=ROOT/'results/ruby_swprefetch_retry_fix_20260921'
SEED=ROOT/'results/ai_workloads/swiglu_confirmation_resumed_20260921'
ENV=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4')

def config_audit(old,new):
    a=configparser.ConfigParser(interpolation=None,strict=False);a.read(old/'config.ini')
    b=configparser.ConfigParser(interpolation=None,strict=False);b.read(new/'config.ini')
    assert a.sections()==b.sections();changes=[]
    for section in a.sections():
        extra=set(b[section])-set(a[section]);missing=set(a[section])-set(b[section])
        assert not missing,(section,missing)
        for key in extra:
            assert section.endswith('.sequencer') and key in ('cacheflex_bridge_credits','cacheflex_fetch_credits') and b[section][key]=='0',(section,key,b[section][key])
            changes.append([section,key,'absent',b[section][key]])
        for key,x in a[section].items():
            y=b[section][key]
            if x.replace(str(old),'<RUN>')==y.replace(str(new),'<RUN>'):continue
            assert (section=='system' and key=='readfile') or (section=='system.cpu0.workload' and key in ('cmd','executable')),(section,key,x,y)
            changes.append([section,key,x,y])
    return changes

def expected_copies(t,d,f,mp):
    return min(mp,(t+7)//8)*4*(2*d*(f//128)+f*(d//128))

def command(template,sim,path,guest,inp,arch,c,repeats):
    cmd=json.loads((template/'command.json').read_text())
    t,d,f,mp,kc,g=c
    args=[t,d,f,'first',repeats] if arch=='acl' else [t,d,f,mp,kc,0,repeats,1,8]
    cmd[0]=str(sim);cmd[cmd.index('--cmd')+1]=str(guest)
    cmd[cmd.index('--options')+1]=' '.join(map(str,args))
    return [f'--outdir={path}' if x.startswith('--outdir=') else f"system.readfile='{inp}'" if x.startswith('system.readfile=') else
            'system.cpu[0].workload[0].cmd='+repr(['model-swiglu']+list(map(str,args))) if x.startswith('system.cpu[0].workload[0].cmd=') else x for x in cmd]

def main():
    OUT.mkdir()
    save(OUT/'status.json',dict(pid=os.getpid(),stage='waiting_for_retry_fix_build',completed=[],updated=time.time()))
    while True:
        build=json.loads((FIX/'status.json').read_text())
        if build['stage']=='failed':raise RuntimeError(('simulator build failed',build))
        if build['stage']=='built_requires_validation':break
        time.sleep(10)
    shutil.copytree(PRIOR/'builds',OUT/'builds')
    for name in ('runs','inputs','preserved'):(OUT/name).mkdir()
    state=dict(pid=os.getpid(),stage='preparing',completed=[],updated=time.time())
    lock=threading.Lock();stop=threading.Event()
    def status(stage):
        with lock:
            state.update(stage=stage,updated=time.time());save(OUT/'status.json',state)
        print(stage,flush=True)
    old_rows=json.loads((SEED/'status.json').read_text())['completed']
    templates={a:Path(next(r['path'] for r in old_rows if r['arch']==a)) for a in ('cache','spm','acl')}
    sim=OUT/'preserved/gem5.opt'
    shutil.copy2(FIX/'gem5.opt',sim)
    assert sha(sim)==build['simulator_sha256']
    lib=OUT/'preserved/ordered_gemm.so'
    subprocess.run(['g++','-O3','-march=native','-ffp-contract=off','-fopenmp','-shared','-fPIC',str(HERE/'model_swiglu_reference.cpp'),'-o',str(lib)],check=True)
    files=list((ROOT/'gem5/configs').rglob('*.py'))+list((OUT/'builds').rglob('*'))
    files += [sim,lib,Path(__file__),HERE/'model_swiglu_reference.py',HERE/'model_swiglu_reference.cpp',HERE/'build_model_swiglu.py',HERE/'sweep_mlp_study.py',HERE/'run_swiglu_confirmation.py',HERE/'run_mapping_screen.py',HERE/'audit_mlp_math.py']
    for p in templates.values():files.extend([p/'command.json',p/'config.ini'])
    manifest={str(p.resolve()):sha(p) for p in files if p.is_file()}
    save(OUT/'manifest.json',manifest)
    save(OUT/'plan.json',dict(model='Llama-3.2-1B',source='https://github.com/meta-llama/llama-models/blob/main/models/sku_list.py',d=2048,f=8192,t=[128,512,2048],
        scope='Model-shaped SwiGLU operator, synthetic Gaussian inputs/weights, not checkpoint/model quality or full transformer latency.',
        precision='FP16 tensors, FP32 GEMM accumulation; native checkpoint is not being executed. Custom fused SiLU-product rounding differs from ACL materialized FP16 SiLU.',
        hardware='8core VL16;64KiB L1I/D,512KiB private L2,4MiB total LLC; four eligible SPM ways; no extra storage/DMA/multicast/batch/retention',
        acl='Unmodified library via public experimental CpuGemmAssemblyDispatch API, use_fp32_acc=true, fast_math=false; normal ACL packing and scheduler. Pinned NEGEMM silently omits this flag.',
        roi='Complete gate/up, SiLU-product, down projection, packing/staging/synchronization. Tensor allocation/input loading/thread creation outside. First invocation includes weight packing; later reuse allowed.',
        measurement='Five calls; first separate, second transitional, last three warm samples. Output check reads between calls retained for every backend. Spread and drift reported, never gated.',
        initial_custom=dict(mp=2,kc=128,combined=0,rows=8,reason='Prior T128 separate/2M warm winner transferred as starting point, not assumed optimal at FP32/model dimensions.'),
        future='Review pilot latency/traffic first; equal-budget independent KC/mapping/fusion refinement, then T4096 and second-model held-out validation. Not an exhaustive sweep.',
        weight_bytes=6*2048*8192,max_simulators=2))
    def frozen():
        for p,h in manifest.items():assert sha(p)==h,('dependency changed',p)
    inputs={}
    def get_input(t,d,f):
        key=t,d,f
        if key not in inputs:
            p=OUT/'inputs'/f't{t}_d{d}_f{f}.bin'
            generate(p,t,d,f,lib)
            inputs[key]=(p,sha(p))
        return inputs[key][0]
    def run(arch,c,stage,repeats=5):
        if stop.is_set():raise RuntimeError('peer failed')
        t,d,f,mp,kc,g=c;inp=inputs[t,d,f][0]
        guest=OUT/'builds'/('acl/acl' if arch=='acl' else f'custom_g{g}/{arch}')
        p=OUT/'runs'/f'{stage}_{arch}_t{t}_d{d}_f{f}_mp{mp}_kc{kc}_g{g}';p.mkdir()
        cmd=command(templates[arch],sim,p,guest,inp,arch,c,repeats)
        if stage=='gate':cmd+=['--abs-max-tick=10000000000']
        save(p/'command.json',cmd);save(p/'case.json',dict(arch=arch,shape=list(c),repeats=repeats))
        start=time.monotonic();proc=None
        try:
            frozen();assert sha(inp)==inputs[t,d,f][1]
            with (p/'run.log').open('w') as log:
                proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=ENV)
                while proc.poll() is None:
                    save(p/'progress.json',dict(stage='simulating',pid=proc.pid,host_elapsed_s=time.monotonic()-start,updated=time.time()))
                    if stop.is_set():raise RuntimeError('peer failed')
                    if time.monotonic()-start>96*3600:raise RuntimeError('96h case timeout')
                    if shutil.disk_usage(OUT).free<50*1024**3:raise RuntimeError('disk guard')
                    if (p/'run.log').stat().st_size>128*1024**2:raise RuntimeError('log guard')
                    try:proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:pass
            assert proc.returncode==0,('simulator exit',proc.returncode)
            text=(p/'run.log').read_text(errors='replace')
            assert not re.search(r'panic:|fatal:|Simulated exit code not 0|\bFAIL\b',text)
            changes=config_audit(templates[arch],p)
            checks=[dict(re.findall(r'(\w+)=([^\s]+)',l)) for l in text.splitlines() if l.startswith('MLP_RESULT')]
            assert len(checks)==repeats and all(x['nonfinite']=='0' for x in checks)
            assert len({x['hash'] for x in checks})==1
            data=np.fromfile(p/'output.bin',dtype='<f2');assert data.size==t*d and np.isfinite(data).all()
            h=1469598103934665603
            for x in data.view(np.uint16):h=((h^int(x))*1099511628211)&((1<<64)-1)
            assert f'{h:016x}'==checks[-1]['hash'],('export hash',p)
            if arch=='acl':
                numeric=validate_acl(p,inp,t,d,f,lib)
                assert not re.search(r'\[.*(?:INFO|CORE).*\]',text),'ACL logging enabled'
            else:
                assert all(x['errors']=='0' for x in checks)
                ref=np.load(inp.with_suffix('.stages.npz'))['y'].reshape(-1)
                error=np.abs(data.astype(float)-ref.astype(float))
                assert np.all(error<=2e-6+.01*np.abs(ref.astype(float)))
                numeric=dict(reference_passed=True,max_abs=float(error.max()))
            math=math_audit(p,inp,t,d,f);assert math['passed']
            timings=epochs(p,repeats);copies=expected_copies(t,d,f,mp) if arch=='spm' else 0
            assert all(e['copies_issued']==e['copies_completed']==copies for e in timings),(copies,timings)
            values=[e['us'] for e in timings[2:]] if repeats>2 else [e['us'] for e in timings]
            median=statistics.median(values)
            row=dict(path=str(p),arch=arch,case=list(c),stage=stage,epochs=timings,first_us=timings[0]['us'],warm_us=median,
                warm_spread=(max(values)-min(values))/median,warm_samples=len(values),checks=checks,numerical=numeric,math=math,config_changes=changes,host_elapsed_s=time.monotonic()-start)
            prior=PRIOR/'runs'/p.name/'accepted.json'
            if stage=='gate' and prior.exists():
                old=json.loads(prior.read_text())
                row['prior_counter_equivalence']=timings==old['epochs']
                assert [x['hash'] for x in checks]==[x['hash'] for x in old['checks']]
                save(p/'prior_equivalence.json',dict(exact_extracted_counters=row['prior_counter_equivalence'],old_epochs=old['epochs'],new_epochs=timings))
            frozen();assert sha(inp)==inputs[t,d,f][1]
            save(p/'accepted.json',row);save(p/'progress.json',dict(stage='accepted',host_elapsed_s=time.monotonic()-start))
            with lock:
                state['completed'].append(row);state['updated']=time.time();save(OUT/'status.json',state)
            return row
        except BaseException as e:
            stop.set();save(p/'failed.json',dict(error=repr(e)))
            if proc and proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=30)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
            raise
    def batch(jobs):
        with cf.ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(run,**job) for job in jobs]
            return [f.result() for f in futures]
    try:
        status('precision_and_correctness_gates')
        get_input(24,256,512)
        cases=[(24,256,512,2,128,0),(24,256,512,1,256,1)]
        batch([dict(arch=a,c=c,stage='gate',repeats=3) for c in cases for a in ('cache','spm')]+[dict(arch='acl',c=cases[0],stage='gate',repeats=3)])
        save(OUT/'gates_passed.json',dict(passed=True))
        for t in (128,512,2048):
            status(f'preparing_reference_t{t}')
            get_input(t,2048,8192)
            status(f'pilot_t{t}')
            c=(t,2048,8192,2,128,0)
            batch([dict(arch=a,c=c,stage='pilot') for a in ('cache','spm','acl')])
        save(OUT/'summary.json',dict(results=state['completed'],scope='Pre-tuning model-shaped FP16/FP32 pilots; synthetic, not full-model accuracy.'))
        status('pilots_complete_requires_targeted_tuning_review')
    except BaseException as e:
        state['error']=repr(e);status('failed');raise

if __name__=='__main__':main()
