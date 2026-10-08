"""Causal L1D miss-credit sensitivity, same frozen IFETCH-on simulator/guests.

8-credit accepted controls are reused. Only dataMissCredits.capacity changes
on every actual L1D. No LLC/L2/CPU/cache/transport/oracle changes. Two sims max.
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
from test_micro_timing import check_global_guard
sys.path.insert(0,str(ROOT/'benchmarks/cacheflex_flash_attention'))
from single_head_reference import check

BASE=ROOT/'results/cacheflex_flash_attention'
OUT=BASE/'l1d_miss_credit_20260923'
BUILD=BASE/'ifetch_edge_20260923/build'
REF=BASE/'micro_resources_20260921/experiment_v6/preserved'
BACKENDS=('cache','spm','spm_fast')
CAPACITIES=(16,32,64)
ENV=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib',
         OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
for k in list(ENV):
    if k.startswith('CACHEFLEX_AGGREGATE') or k in ('CACHEFLEX_TRACE_LINE','CACHEFLEX_LOAD_PCS'):ENV.pop(k)

def prior(backend,t):return BASE/f'ifetch_edge_20260923/experiment/runs/{backend}_t{t}_edge1'

def command(backend,t,capacity,path):
    old=json.loads((prior(backend,t)/'command.json').read_text())
    i=next(i for i,x in enumerate(old) if x.endswith('/micro_timing_se.py'))
    cmd=[str(BUILD/'gem5.opt'),f'--outdir={path}',str(OUT/'configs/deprecated/example/micro_timing_se.py')]+old[i+1:]
    for core in range(8):cmd+=['-P',f'system.ruby.l0_cntrl{core}.dataMissCredits.capacity={capacity}']
    return cmd

def audit_objects(old,new,capacity):
    a=copy.deepcopy(old);b=copy.deepcopy(new)
    for i in range(8):
        aa=a['system']['ruby'][f'l0_cntrl{i}'];bb=b['system']['ruby'][f'l0_cntrl{i}']
        assert aa['dataMissCredits']['capacity']==8
        assert bb['dataMissCredits']['capacity']==capacity
        bb['dataMissCredits']['capacity']=8
        assert bb['sequencer']['cacheflex_ifetch_same_tick_response'] is True
        assert bb['sequencer']['cacheflex_ideal_staging_cache'] is None
        assert bb['sequencer']['cacheflex_ideal_staging_controller'] is None
    assert a==b,'Only eight L1D dataMissCredits capacities may change'

def audit(old,path,capacity):
    a=json.loads((old/'config.json').read_text().replace(str(old),'<OUT>'))
    b=json.loads((path/'config.json').read_text().replace(str(path),'<OUT>'))
    audit_objects(a,b,capacity)

def counter(s,name):
    return sum(s[f'system.ruby.l0_cntrl{i}.sequencer.{name}'] for i in range(8))

def validate(path,backend,t,capacity):
    old=prior(backend,t);audit(old,path,capacity)
    assert sha(path/'output.fp16')==sha(old/'output.fp16'),'Output hash changed'
    numerical=check(path/'output.fp16',np.load(REF/f't{t}_bc128.npy'))
    assert numerical['errors']==numerical['nonfinite']==0,numerical
    h=1469598103934665603
    for x in np.fromfile(path/'output.fp16',dtype=np.uint16):h=((h^int(x))*1099511628211)&((1<<64)-1)
    match=re.search(r'FLASH_RESULT.*?hash=([0-9a-f]+)',(path/'run.log').read_text())
    assert match and f'{h:016x}'==match[1],'Guest/export mismatch'
    blocks=stats(path);s=blocks[0]
    for block in blocks:check_global_guard(block)
    copies=0 if backend=='cache' else (t//64)*t*2
    for name in ('m_spmcpRequestsIssued','m_spmcpRequestsCompleted'):
        assert counter(s,name)==copies,(name,counter(s,name),copies)
    assert counter(s,'m_idealStagingCopies')==0
    assert counter(s,'ifetchEdgeDirect')>0
    pools=[]
    for i in range(8):
        key=f'system.ruby.l0_cntrl{i}.dataMissCredits.'
        row={k:s[key+k] for k in ('fullChecks','maxLive','allocations','releases','occupancy')}
        assert row['maxLive']<=capacity
        pools.append(row)
    diagnostic={k:v for k,v in s.items() if
                any(x in k for x in ('.missCredits.','.llcMissCredits.','.writeCredits.',
                                     'pcNotExecutedCycles::','m_ingressFullRejects'))}
    return dict(path=str(path),backend=backend,T=t,credits=capacity,roi_us=s['simTicks']/1e6,
                arithmetic=numerical,output_hash=match[1],copy_count=copies,config_audit=True,
                global_guard_nonbinding=True,pools=pools,
                l1d_full_checks=sum(p['fullChecks'] for p in pools),
                max_l1d_occupied=max(p['maxLive'] for p in pools),other_counters=diagnostic)

def report(state):
    rows=[r for r in state['completed'] if r['T']==4096]
    lines=['# FA L1D miss-credit sensitivity','',
           'Same frozen IFETCH-on model; only actual L1D data miss credits changed. '
           'First-invocation ROI includes the original packing/staging/synchronization boundaries. '
           'Credits are diagnostic transaction-capacity assumptions, not extra cache/SPM SRAM or calibrated area-neutral hardware.', '',
           '| Credits | Backend | ROI us | Runtime reduction vs own 8-credit control | L1D failed checks | Max occupied/core |',
           '|---:|---|---:|---:|---:|---:|']
    for r in sorted(rows,key=lambda r:(r['credits'],BACKENDS.index(r['backend']))):
        control=next(x for x in rows if x['credits']==8 and x['backend']==r['backend'])
        gain=100*(1-r['roi_us']/control['roi_us'])
        lines.append(f"| {r['credits']} | {r['backend']} | {r['roi_us']:.4f} | {gain:.3f}% | {r['l1d_full_checks']:.0f} | {r['max_l1d_occupied']:.0f} |")
    lines+=['','## Matched comparisons','']
    for cap in (8,16,32,64):
        group={r['backend']:r for r in rows if r['credits']==cap}
        if 'cache' in group and 'spm' in group:
            lines.append(f"- {cap} credits: matched cache/SPM speedup = {group['cache']['roi_us']/group['spm']['roi_us']:.6f}x.")
        if 'spm' in group and 'spm_fast' in group:
            gain=100*(1-group['spm_fast']['roi_us']/group['spm']['roi_us'])
            lines.append(f'- {cap} credits: faster direct-SPM loads reduce SPM ROI by {gain:.4f}% (negative = slower).')
    lines += ['', 'Failed-check counts are retries, NOT lost cycles. Lower checks with unchanged ROI means that relieving '
              'this resource at the tested setting did not expose an end-to-end gain. Persistent checks mean the intervention '
              'did not remove the bottleneck completely. Plateau may reflect unchanged private-L2/LLC/CPU limits; '
              'do not infer zero memory-system headroom or attribute the entire admission span to this pool.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')

def main():
    OUT.mkdir();(OUT/'runs').mkdir()
    state=dict(pid=os.getpid(),stage='validating_reused_controls',active={},completed=[],
               target_cases=[dict(backend=b,credits=c,T=4096) for c in (64,16,32) for b in BACKENDS])
    manifest={};lock=threading.RLock();halt=threading.Event()
    def status(stage=None):
        with lock:
            if stage:state['stage']=stage
            state['updated']=time.time();save(OUT/'status.json',state);report(state)
    def frozen():
        for p,h in manifest.items():assert sha(p)==h,('Frozen input changed',p)
    def run(backend,t,capacity):
        frozen();tag=f'{backend}_t{t}_credits{capacity}';p=OUT/'runs'/tag;p.mkdir()
        cmd=command(backend,t,capacity,p);save(p/'command.json',cmd)
        proc=None;begin=time.monotonic()
        try:
            with (p/'run.log').open('w') as log:
                proc=subprocess.Popen(cmd,cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT)
                with lock:state['active'][tag]=dict(pid=proc.pid,path=str(p));status()
                while proc.poll() is None:
                    assert not halt.is_set(),'Peer failed'
                    assert time.monotonic()-begin<10800,'Three-hour case limit'
                    assert shutil.disk_usage(OUT).free>50*2**30,'Disk guard'
                    try:proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:pass
            assert proc.returncode==0,(tag,proc.returncode)
            row=validate(p,backend,t,capacity);row['host_seconds']=time.monotonic()-begin
            row['reused']=False;frozen();save(p/'validated.json',row)
            with lock:state['active'].pop(tag);state['completed'].append(row);status()
        except BaseException as exc:
            halt.set();save(p/'failed.json',dict(error=repr(exc)))
            if proc and proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
            with lock:state['active'].pop(tag,None);status()
            raise
    def parallel(cases):
        with cf.ThreadPoolExecutor(max_workers=2) as pool:
            fs=[pool.submit(run,*c) for c in cases]
            for f in cf.as_completed(fs):f.result()
    status()
    try:
        shutil.copytree(BUILD/'configs',OUT/'configs');shutil.copy2(__file__,OUT/'runner_snapshot.py')
        manifest.update(json.loads((BUILD/'frozen.json').read_text()))
        paths=list((OUT/'configs').rglob('*.py'))+[Path(__file__).resolve(),
              ROOT/'scripts/build_access_timing.py',ROOT/'scripts/run_access_timing.py',
              ROOT/'scripts/test_micro_timing.py',ROOT/'benchmarks/cacheflex_flash_attention/single_head_reference.py']
        for t in (512,4096):
            paths.append(REF/f't{t}_bc128.npy')
            for backend in BACKENDS:
                old=prior(backend,t)
                paths += [old/n for n in ('validated.json','command.json','config.json','stats.txt','output.fp16','run.log')]
                cmd=json.loads((old/'command.json').read_text());paths.append(Path(cmd[cmd.index('--cmd')+1]))
        manifest.update({str(p):sha(p) for p in paths});save(OUT/'inputs.json',manifest);frozen()
        for t in (512,4096):
            for backend in BACKENDS:
                row=validate(prior(backend,t),backend,t,8);row['reused']=True
                state['completed'].append(row);status()
        save(OUT/'reused_controls_validated.json',dict(same_frozen_simulator_guests=True,full_math=True))
        status('t512_changed_capacity_gates')
        parallel([(b,512,c) for c in (64,16,32) for b in BACKENDS])
        save(OUT/'gates_passed.json',dict(full_math_hash_config_copy_guard=True))
        # Strong finite intervention first, then intermediate points test dose
        # response and saturation. Lower-level limits remain unchanged.
        for c in (64,16,32):
            status(f't4096_credits{c}')
            parallel([(b,4096,c) for b in BACKENDS])
        frozen();status('completed_validated')
    except BaseException as exc:
        halt.set();state['error']=repr(exc);status('failed');raise

if __name__=='__main__':main()
