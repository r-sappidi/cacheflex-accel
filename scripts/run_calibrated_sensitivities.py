"""Paired old/calibrated-timing FA sensitivities, with immutable inputs.

Every intervention uses the same current simulator, guests, resource policy and
8-core T4096 geometry. No extra SPM backing. Reports end-to-end ROI changes,
not summed copy/stall latency as a runtime fraction.
"""
import concurrent.futures as cf
import copy
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
import numpy as np
from build_access_timing import ROOT,sha,save
from run_access_timing import stats
from run_ingress_bypass import aggregate_metrics
from test_micro_timing import check_global_guard
sys.path.insert(0,str(ROOT/'benchmarks/cacheflex_flash_attention'))
from single_head_reference import check

BASE=ROOT/'results/cacheflex_flash_attention'
OUT=BASE/os.environ.get('CACHEFLEX_SENSITIVITY_OUT','calibrated_sensitivities_20260922')
CAL=BASE/'timing_alignment_20260922'
OLD=BASE/'demand_isolation_20260922/experiment'
SIM=BASE/'demand_isolation_20260922/build/gem5.opt'
REF=BASE/'micro_resources_20260921/experiment_v6/preserved'
VARIANTS={
    'control':'Unmodified timing-profile control',
    'demand_isolation':'Copy-stalled FIFO may select an independent ordinary demand; diagnostic arbitration',
    'ingress4':'Search4 younger FIFO entries within existing transition budget',
    'ingress16':'Search16 younger FIFO entries within existing transition budget',
    'copy_service':'Prior fast-copy bundle: local request/ack1 cycle and SPM store ports8',
    'load_issue':'Prior fast-load bundle: local forward1 cycle and SPM load ports8',
    'copy_load':'Both prior copy/load bundles; no new data storage',
    'load_service':'Additional isolated pure-SPM response floor1 cycle (no port expansion)',
    'no_retention':'Disable SPM silent-fetch LLC retention',
    'restrained_prefetch':'2 streams, train after8 misses',
    'no_prefetch':'Disable existing hardware prefetcher',
    'network256':'256-bit instead of128-bit links; explicit bandwidth/cost change for BOTH backends',
    'guard4096':'Nonbinding global bookkeeping guard4096 instead of1024; exact-stats check',
}
ENV=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib',
         OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
for k in list(ENV):
    if k.startswith('CACHEFLEX_AGGREGATE') or k=='CACHEFLEX_TRACE_LINE':ENV.pop(k)

def prior(profile,arch,t):
    if profile=='micro':return CAL/f'fa_v3/{arch}_t{t}'
    if t==512:return CAL/f'fa_v3/{arch}_t512_export_control'
    return OLD/f'{arch}_t{t}_off'

def parameters(variant):
    assert variant in VARIANTS
    p={}
    for i in range(8):
        l0=f'system.ruby.l0_cntrl{i}';l1=f'system.ruby.l1_cntrl{i}'
        if variant=='demand_isolation':p[l0+'.mandatoryQueue.cacheflex_copy_demand_bypass']=True
        if variant.startswith('ingress'):p[l0+'.mandatoryQueue.cacheflex_ingress_lookahead']=int(variant[7:])
        if variant in ('copy_service','copy_load'):
            p[l0+'.spm_copy_request_latency']=1;p[l1+'.spm_response_latency']=1
            p[f'system.cpu[{i}].spmStorePorts']=8
            p[l0+'.bufferToL1.clamp_fifo_arrival']=True
        if variant in ('load_issue','copy_load'):
            p[l0+'.spm_load_request_latency']=1;p[f'system.cpu[{i}].spmLoadPorts']=8
            p[l0+'.bufferToL1.clamp_fifo_arrival']=True
        if variant=='load_service':p[l1+'.spm_access_response_latency']=1
        if variant=='no_retention':p[f'system.ruby.l2_cntrl{i}.spm_retain_llc']=False
        if variant=='no_prefetch':p[l0+'.enable_prefetch']=False
        if variant=='restrained_prefetch':
            p[l0+'.prefetcher.num_streams']=2;p[l0+'.prefetcher.train_misses']=8
        if variant=='guard4096':p[l0+'.sequencer.max_outstanding_requests']=4096
    return p

def audit(old,new,variant):
    expected=parameters(variant);seen=set();changes=[]
    def walk(a,b,path=''):
        if path in expected:
            assert b==expected[path],(path,b,expected[path]);seen.add(path)
            if a!=b:changes.append([path,a,b])
            return
        if isinstance(a,dict):
            assert isinstance(b,dict) and set(a)==set(b),(path,set(a)^set(b))
            for k in a:walk(a[k],b[k],path+'.'+k if path else k)
        elif isinstance(a,list):
            assert len(a)==len(b),path
            for i,(x,y) in enumerate(zip(a,b)):walk(x,y,path+f'[{i}]')
        elif a!=b:
            if isinstance(a,str) and a.replace(str(old),'<OUT>')==b.replace(str(new),'<OUT>'):return
            if path=='system.cpu[0].workload[0].executable':
                assert Path(b)==CAL/'fa_export_guests'/Path(a).name,(path,a,b)
            elif variant=='network256' and path.startswith('system.ruby.network.') and path.rsplit('.',1)[-1] in ('ni_flit_size','width'):
                assert b==(32 if a==16 else 256),(path,a,b)
            else:raise AssertionError((path,a,b))
            changes.append([path,a,b])
    walk(json.loads((old/'config.json').read_text()),json.loads((new/'config.json').read_text()))
    assert seen==set(expected),(seen^set(expected))
    return changes

def make_command(profile,arch,t,variant,p):
    old=prior(profile,arch,t);cmd=json.loads((old/'command.json').read_text())
    params=parameters(variant);clean=[];i=0
    while i<len(cmd):
        if cmd[i]=='-P' and cmd[i+1].split('=',1)[0] in params:i+=2;continue
        v=cmd[i]
        if v.startswith('--outdir='):v=f'--outdir={p}'
        if variant=='network256' and v.startswith('--link-width-bits='):v='--link-width-bits=256'
        clean.append(v);i+=1
    clean[0]=str(SIM);clean[2]=str(OUT/'configs/deprecated/example/micro_timing_se.py')
    clean[clean.index('--cmd')+1]=str(CAL/'fa_export_guests'/arch)
    for k,v in params.items():clean+=['-P',f'{k}={v!r}']
    assert ('--cacheflex-micro-timing' in clean)==(profile=='micro')
    return clean

def write_report(state):
    rows=state['completed']
    lines=['# FA sensitivity after MICRO timing calibration','',
        'Fixed8-core single-head VL16, d64, T4096,64KiB L1I/D and privateL2,4MiB LLC. '
        'Same current simulator/resource policy and guest bytes for both timing profiles. '
        'Each intervention is compared to its own profile/backend control. '
        'Old and new historical experiments with different resource models are NOT mixed.',
        '', 'T512 results are numerical/configuration gates, not target-performance conclusions. '
        'Only accepted cases appear below. Service/issue floors are diagnostic bounds; network widening has a cost. '
        'There is no extra SPM data backing. Effects need not add because overlap/bottlenecks change.',
        '', '| T | Timing | Variant | Cache us | SPM us | Cache/SPM | SPM runtime reduction vs control |',
        '|---|---|---|---:|---:|---:|---:|']
    for t in (512,4096):
        for v in VARIANTS:
            for prof in ('old','micro'):
                pair={r['arch']:r for r in rows if r['T']==t and r['variant']==v and r['profile']==prof}
                if not pair:continue
                ctrl=next((r for r in rows if r['T']==t and r['variant']=='control' and r['profile']==prof and r['arch']=='spm'),None)
                c=pair.get('cache',{}).get('roi_us');s=pair.get('spm',{}).get('roi_us')
                fmt=lambda x:f'{x:.4f}' if x is not None else 'pending'
                ratio=f'{c/s:.6f}x' if c is not None and s is not None else 'pending'
                delta=f'{100*(1-s/ctrl["roi_us"]):+.3f}%' if ctrl and s is not None else 'pending'
                lines.append(f'| {t} | {prof} | {v} | {fmt(c)} | {fmt(s)} | {ratio} | {delta} |')
    lines+=['','Controls marked reused in status.json reference prior fully accepted data. '
            'All newly executed cases require the unchanged full numerical oracle, exact full output hash, '
            'guest/export hash, copy count, and full configuration audit. '
            'Per-copy aggregate latency is auxiliary data, not a fraction of FA runtime.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')

def main():
    OUT.mkdir();(OUT/'runs').mkdir();shutil.copytree(CAL/'fa_v3/configs',OUT/'configs')
    shutil.copy2(__file__,OUT/'runner_snapshot.py')
    state=dict(pid=os.getpid(),stage='preparing',active={},completed=[],updated=time.time())
    lock=threading.RLock();stop=threading.Event()
    def status(stage=None):
        with lock:
            if stage:state['stage']=stage
            state['updated']=time.time();save(OUT/'status.json',state);write_report(state)
    paths=[Path(__file__).resolve(),SIM,CAL/'calibration.json',CAL/'fa_v3/verification.json',
           ROOT/'scripts/build_access_timing.py',ROOT/'scripts/run_access_timing.py',
           ROOT/'scripts/run_ingress_bypass.py',ROOT/'scripts/run_access_aggregate.py',
           ROOT/'scripts/test_micro_timing.py',ROOT/'scripts/run_timing_fa.py',
           ROOT/'benchmarks/cacheflex_flash_attention/single_head_reference.py']
    paths+=list((OUT/'configs').rglob('*.py'))
    for arch in ('cache','spm'):paths += [CAL/'fa_export_guests'/arch,CAL/'fa_export_guests'/f'{arch}.audit.json']
    for t in (512,4096):
        paths.append(REF/f't{t}_bc128.npy')
        for prof in ('old','micro'):
            for arch in ('cache','spm'):
                p=prior(prof,arch,t);paths += [p/n for n in ('validated.json','command.json','config.json','stats.txt','output.fp16')]
    manifest={str(p):sha(p) for p in paths};save(OUT/'inputs.json',manifest)
    save(OUT/'plan.json',dict(variants=VARIANTS,profiles=['old','micro'],parameters={v:parameters(v) for v in VARIANTS},
        stages='T512 controls then all intervention gates; T4096 both profiles/backends for each intervention',
        max_simulators=2,new_runs=100,reused_t4096_controls=4,causal_metric='End-to-end ROI latency, not summed stall/copy cycles',
        exclusions='No separate backing, no new DMA/multicast, no tile retuning; no artificial16-entry global gate',
        network_cost='Network256 affects both architectures and doubles link width; not free CacheFlex benefit'))
    def frozen():
        for p,h in manifest.items():assert sha(p)==h,('Changed frozen input',p)
    def accept(row,p=None):
        frozen()
        if p:save(p/'validated.json',row)
        with lock:state['completed'].append(row);status()
    def execute(p,cmd,env):
        proc=None;started=time.monotonic()
        try:
            with (p/'run.log').open('w') as log:
                proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                with lock:state['active'][p.name]=dict(pid=proc.pid,path=str(p));status()
                while proc.poll() is None:
                    assert not stop.is_set(),'Peer failed'
                    assert time.monotonic()-started<10800,'Three-hour case cap'
                    assert shutil.disk_usage(OUT).free>50*2**30,'Disk guard'
                    try:proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:pass
            assert proc.returncode==0,(p.name,proc.returncode)
            with lock:state['active'].pop(p.name);status()
            return time.monotonic()-started
        except BaseException:
            stop.set()
            if proc and proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
            raise
    def run(prof,arch,t,v):
        frozen();p=OUT/'runs'/f'{prof}_{arch}_t{t}_{v}';p.mkdir()
        old=prior(prof,arch,t);cmd=make_command(prof,arch,t,v,p);save(p/'command.json',cmd)
        env=dict(ENV)
        if arch=='spm':env['CACHEFLEX_AGGREGATE_PATH']=str(p/'aggregate.json')
        seconds=execute(p,cmd,env);changes=audit(old,p,v)
        assert sha(p/'output.fp16')==sha(old/'output.fp16'),'Full output differs'
        arithmetic=check(p/'output.fp16',np.load(REF/f't{t}_bc128.npy'))
        assert arithmetic['errors']==arithmetic['nonfinite']==0,arithmetic
        h=1469598103934665603
        for w in np.fromfile(p/'output.fp16',dtype=np.uint16):h=((h^int(w))*1099511628211)&((1<<64)-1)
        printed=re.search(r'FLASH_RESULT.*?hash=([0-9a-f]+)',(p/'run.log').read_text())[1]
        assert f'{h:016x}'==printed,'Guest/export mismatch'
        s=stats(p)[0];check_global_guard(s);expected=(t//64)*t*2 if arch=='spm' else 0
        for suffix in ('.m_spmcpRequestsIssued','.m_spmcpRequestsCompleted'):
            assert sum(x for k,x in s.items() if k.endswith(suffix))==expected
        exact=(v in ('control','guard4096') or (arch=='cache' and v in
            ('demand_isolation','copy_service','load_issue','copy_load','load_service','no_retention')))
        if exact:assert s==stats(old)[0],('Required control-equivalence failed',p.name)
        row=dict(profile=prof,arch=arch,T=t,variant=v,path=str(p),passed=True,reused=False,
            roi_us=s['simTicks']/1e6,control_roi_us=stats(old)[0]['simTicks']/1e6,host_seconds=seconds,
            arithmetic=arithmetic,output_hash=printed,config_changes=changes,expected_copies=expected,
            exact_control_stats=exact,global_guard_rejects=0)
        if arch=='spm':
            a=aggregate_metrics(p,expected);row['aggregate']={k:x for k,x in a.items() if k!='numeric_aggregate'}
        accept(row,p)
    def pair(prof,t,v):
        status(f't{t}_{prof}_{v}')
        with cf.ThreadPoolExecutor(max_workers=2) as pool:
            fs=[pool.submit(run,prof,a,t,v) for a in ('cache','spm')]
            try:
                for f in cf.as_completed(fs):f.result()
            except BaseException:stop.set();raise
    try:
        calibration=json.loads((CAL/'calibration.json').read_text())
        assert calibration['passed'] and calibration['profile_sha256']==sha(OUT/'configs/deprecated/example/micro_timing_se.py')
        assert json.loads((CAL/'fa_v3/verification.json').read_text())['complete']
        for prof in ('old','micro'):
            for arch in ('cache','spm'):
                p=prior(prof,arch,4096);s=stats(p)[0];check_global_guard(s)
                num=check(p/'output.fp16',np.load(REF/'t4096_bc128.npy'))
                assert num['errors']==num['nonfinite']==0
                accept(dict(profile=prof,arch=arch,T=4096,variant='control',path=str(p),passed=True,
                    reused=True,roi_us=s['simTicks']/1e6,arithmetic=num,source_validation=str(p/'validated.json')))
        for v in VARIANTS:
            for prof in ('micro','old'):pair(prof,512,v)
        save(OUT/'gates_passed.json',dict(passed=True,cases=52))
        for v in VARIANTS:
            if v=='control':continue
            for prof in ('micro','old'):pair(prof,4096,v)
        status('completed_validated')
    except BaseException as exc:
        stop.set();state['error']=repr(exc);status('failed');raise

if __name__=='__main__':main()
