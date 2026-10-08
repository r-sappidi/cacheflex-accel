"""Fixed 2x2 execution/on-chip-memory sensitivity, cache and SPM, max two sims."""
import concurrent.futures as cf
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
from fa_headroom_profiles import parameters,expected_changes,audit_objects,VARIANTS,EXECUTION_FUS
sys.path.insert(0,str(ROOT/'benchmarks/cacheflex_flash_attention'))
from single_head_reference import check

BASE=ROOT/'results/cacheflex_flash_attention'
OUT=BASE/os.environ.get('CACHEFLEX_HEADROOM_EXPERIMENT','headroom_2x2_v2_20260923')
BUILD=BASE/'ifetch_edge_20260923/build'
REF=BASE/'micro_resources_20260921/experiment_v6/preserved'
BACKENDS=('cache','spm')
ENV=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib',
         OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
for k in list(ENV):
    if k.startswith('CACHEFLEX_AGGREGATE') or k in ('CACHEFLEX_TRACE_LINE','CACHEFLEX_LOAD_PCS'):ENV.pop(k)

def prior(backend,t):return BASE/f'ifetch_edge_20260923/experiment/runs/{backend}_t{t}_edge1'

def command(backend,t,variant,path):
    old=prior(backend,t);config=json.loads((old/'config.json').read_text())
    overrides=parameters(config,variant);original=json.loads((old/'command.json').read_text())
    index=next(i for i,x in enumerate(original) if x.endswith('/micro_timing_se.py'))
    cmd=[str(BUILD/'gem5.opt'),f'--outdir={path}',str(OUT/'configs/deprecated/example/micro_timing_se.py')]
    i=index+1
    while i<len(original):
        if original[i]=='-P' and original[i+1].split('=',1)[0] in overrides:i+=2;continue
        arg=original[i]
        if VARIANTS[variant][1] and arg.startswith('--link-width-bits='):arg='--link-width-bits=1024'
        cmd.append(arg);i+=1
    for k,v in overrides.items():cmd+=['-P',f'{k}={v!r}']
    return cmd

def counter(s,name):return sum(s[f'system.ruby.l0_cntrl{i}.sequencer.{name}'] for i in range(8))

def diagnostics(s):
    out={k:v for k,v in s.items() if any(x in k for x in
        ('Credits.','FullEvents','statFuBusy::','issueRate','icacheStallCycles',
         'pcNotExecutedCycles::','m_ingressFullRejects','m_lineTargetFullRejects',
         'avgMemAccLat','busUtil','bytesReadSys','bytesWrittenSys','readReqs','writeReqs',
         'average_packet','flits_received::total','packets_received::total'))}
    out['sum_l1d_failed_checks']=sum(s[f'system.ruby.l0_cntrl{i}.dataMissCredits.fullChecks'] for i in range(8))
    out['sum_fu_busy']=sum(s[f'system.cpu{i}.fuBusy'] for i in range(8))
    copies=counter(s,'m_spmcpRequestsCompleted')
    out['copy_mean_lifetime_cycles']=counter(s,'m_spmcpOutstandingCycleArea')/copies if copies else None
    out['fma_issues']=sum(s[f'system.cpu{i}.statIssuedInstType_0::SimdFloatMultAcc'] for i in range(8))
    return out

def validate(path,backend,t,variant):
    old=prior(backend,t)
    a=json.loads((old/'config.json').read_text().replace(str(old),'<OUT>'))
    b=json.loads((path/'config.json').read_text().replace(str(path),'<OUT>'))
    changes=audit_objects(a,b,variant)
    assert sha(path/'output.fp16')==sha(old/'output.fp16'),'Output changed'
    numerical=check(path/'output.fp16',np.load(REF/f't{t}_bc128.npy'))
    assert numerical['errors']==numerical['nonfinite']==0,numerical
    h=1469598103934665603
    for x in np.fromfile(path/'output.fp16',dtype=np.uint16):h=((h^int(x))*1099511628211)&((1<<64)-1)
    match=re.search(r'FLASH_RESULT.*?hash=([0-9a-f]+)',(path/'run.log').read_text())
    assert match and f'{h:016x}'==match[1],'Printed/export hash mismatch'
    blocks=stats(path);s=blocks[0]
    for block in blocks:check_global_guard(block)
    copies=0 if backend=='cache' else (t//64)*t*2
    for field in ('m_spmcpRequestsIssued','m_spmcpRequestsCompleted'):assert counter(s,field)==copies
    assert counter(s,'m_idealStagingCopies')==0
    assert counter(s,'ifetchEdgeDirect')>0
    d=diagnostics(s)
    assert d['fma_issues']>0,'Arithmetic path not exercised'
    assert s['system.ruby.network.packets_received::total']>0,'Network path not exercised'
    for i in range(8):
        seq=b['system']['ruby'][f'l0_cntrl{i}']['sequencer']
        assert seq['cacheflex_ideal_staging_cache'] is None and seq['cacheflex_ideal_staging_controller'] is None
        for level,names in (('l0',('dataMissCredits','instMissCredits','writeCredits')),('l1',('missCredits','writeCredits'))):
            for name in names:
                obj=b['system']['ruby'][f'{level}_cntrl{i}'][name]
                assert s[f'system.ruby.{level}_cntrl{i}.{name}.maxLive']<=obj['capacity']
    return dict(path=str(path),backend=backend,T=t,variant=variant,roi_us=s['simTicks']/1e6,
                arithmetic=numerical,output_hash=match[1],copy_count=copies,config_changes=changes,
                config_audit=True,global_guard_nonbinding=True,diagnostics=d)

def constraints_audit():
    rows=[];configs={b:json.loads((prior(b,4096)/'config.json').read_text()) for b in BACKENDS}
    for backend,c in configs.items():
        assert len(c['system']['cpu'])==8
        for cpu in c['system']['cpu']:
            assert cpu['fetchBufferSize']==64
            assert cpu['fuPool']['FUList'][9]['count']==4
        r=c['system']['ruby']
        for i in range(8):
            for l,k in (('l0','Icache'),('l0','Dcache'),('l1','cache'),('l2','L2cache')):
                cache=r[f'{l}_cntrl{i}'][k]
                assert cache['resourceStalls'] is False
            assert r[f'l0_cntrl{i}']['number_of_TBEs']==256
        assert not(set(parameters(c,'execution'))&set(parameters(c,'memory')))
        assert parameters(c,'combined')==parameters(c,'execution')|parameters(c,'memory')
        s=stats(prior(backend,4096))[0]
        rows.append(dict(backend=backend,normal_roi_us=s['simTicks']/1e6,
                         dram_bus_utilization={k:v for k,v in s.items() if k.endswith('.dram.busUtil')},
                         diagnostics=diagnostics(s)))
    # Backends must receive exactly the same architectural changes.
    for v in VARIANTS:assert expected_changes(configs['cache'],v)==expected_changes(configs['spm'],v)
    save(OUT/'constraints.json',dict(checked=True,profiles={v:expected_changes(configs['spm'],v) for v in VARIANTS},
        evidence=rows,remaining=['DRAM timings/bandwidth/queues','branch prediction and fixed pipeline delays',
           '64-byte single fetch buffer and control flow','256 controller TBEs and finite coalescing/queue limits',
           'cache capacity, replacement, translation, coherence dependencies and topology',
           'positive one-cycle per-hop/controller floors; no SRAM bank conflicts modeled'],
        scope='Finite aggressive on-chip/execution sensitivity, not a perfect-memory or universal upper bound'))

def report(state):
    rows=[r for r in state['completed'] if r['T']==4096]
    lines=['# FA execution × on-chip-memory headroom','',
           'Fixed IFETCH-on8core/T4096/singlehead/VL16. Same kernels, data capacities and real coherence. '
           'Finite bundles, not perfect memory or realizable area-neutral hardware. See constraints.json.', '',
           '| Variant | Cache us | SPM us | Matched cache/SPM speedup |','|---|---:|---:|---:|']
    for variant in VARIANTS:
        pair={r['backend']:r for r in rows if r['variant']==variant}
        c=pair.get('cache',{}).get('roi_us');s=pair.get('spm',{}).get('roi_us')
        fmt=lambda x:f'{x:.4f}' if x is not None else 'pending'
        lines.append(f'| {variant} | {fmt(c)} | {fmt(s)} | '+(f'{c/s:.6f}x' if c and s else 'pending')+' |')
    lines+=['','## Conditional effects (end-to-end ROI)','']
    for backend in BACKENDS:
        g={r['variant']:r['roi_us'] for r in rows if r['backend']==backend}
        for start,end,label in (('control','execution','execution with normal memory'),
                               ('control','memory','memory with normal execution'),
                               ('execution','combined','memory with relaxed execution'),
                               ('memory','combined','execution with relaxed memory'),
                               ('control','combined','combined')):
            if start in g and end in g:
                lines.append(f'- {backend}: {label} reduces ROI by {100*(1-g[end]/g[start]):.3f}%.')
        if len(g)==4:
            interaction=g['execution']+g['memory']-g['combined']-g['control']
            lines.append(f'- {backend}: interaction savings beyond additive individual deltas = {interaction:.4f}us. Positive denotes extra joint savings; not an additive bottleneck breakdown.')
    lines+=['','A remaining stall/credit limit must be inspected before interpreting a small effect as absent headroom. '
            'Cache and SPM get identical bundles. General-platform improvements are not CacheFlex-specific gains. '
            'T512 is a correctness/configuration gate only. No automatic expansion beyond this six-target screen.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')

def main():
    OUT.mkdir();(OUT/'runs').mkdir()
    state=dict(pid=os.getpid(),stage='constraints_audit',active={},completed=[])
    manifest={};lock=threading.RLock();halt=threading.Event()
    def status(stage=None):
        with lock:
            if stage:state['stage']=stage
            state['updated']=time.time();save(OUT/'status.json',state);report(state)
    def frozen():
        for p,h in manifest.items():assert sha(p)==h,('Changed frozen input',p)
    def run(backend,t,variant):
        frozen();tag=f'{backend}_t{t}_{variant}';p=OUT/'runs'/tag;p.mkdir()
        cmd=command(backend,t,variant,p);save(p/'command.json',cmd)
        proc=None;begin=time.monotonic()
        try:
            with (p/'run.log').open('w') as log:
                proc=subprocess.Popen(cmd,cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT)
                with lock:state['active'][tag]=dict(pid=proc.pid,path=str(p));status()
                while proc.poll() is None:
                    assert not halt.is_set(),'Peer failed'
                    assert time.monotonic()-begin<10800,'Three-hour limit'
                    assert shutil.disk_usage(OUT).free>50*2**30,'Disk guard'
                    try:proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:pass
            assert proc.returncode==0,(tag,proc.returncode)
            row=validate(p,backend,t,variant);row.update(host_seconds=time.monotonic()-begin,reused=False)
            frozen();save(p/'validated.json',row)
            with lock:state['active'].pop(tag);state['completed'].append(row);status()
        except BaseException as exc:
            halt.set();save(p/'failed.json',dict(error=repr(exc)))
            if proc and proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
            with lock:state['active'].pop(tag,None);status()
            raise
    def pair(t,variant):
        status(f't{t}_{variant}')
        with cf.ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(run,b,t,variant) for b in BACKENDS]
            for f in cf.as_completed(futures):f.result()
    status()
    try:
        shutil.copytree(BUILD/'configs',OUT/'configs');shutil.copy2(__file__,OUT/'runner_snapshot.py')
        shutil.copy2(ROOT/'scripts/fa_headroom_profiles.py',OUT/'profiles_snapshot.py')
        constraints_audit()
        manifest.update(json.loads((BUILD/'frozen.json').read_text()))
        paths=list((OUT/'configs').rglob('*.py'))+[Path(__file__).resolve(),ROOT/'scripts/fa_headroom_profiles.py',
              ROOT/'scripts/build_access_timing.py',ROOT/'scripts/run_access_timing.py',ROOT/'scripts/test_micro_timing.py',
              ROOT/'benchmarks/cacheflex_flash_attention/single_head_reference.py',OUT/'constraints.json']
        for t in (512,4096):
            paths.append(REF/f't{t}_bc128.npy')
            for backend in BACKENDS:
                old=prior(backend,t)
                paths += [old/n for n in ('validated.json','command.json','config.json','stats.txt','output.fp16','run.log')]
                cmd=json.loads((old/'command.json').read_text());paths.append(Path(cmd[cmd.index('--cmd')+1]))
        manifest.update({str(p):sha(p) for p in paths});save(OUT/'inputs.json',manifest);frozen()
        for t in (512,4096):
            for backend in BACKENDS:
                row=validate(prior(backend,t),backend,t,'control');row['reused']=True
                state['completed'].append(row);status()
        save(OUT/'reused_controls_validated.json',dict(same_frozen_simulator_guests=True,full_math=True))
        for v in ('execution','memory','combined'):pair(512,v)
        save(OUT/'gates_passed.json',dict(full_math_hash_config_copy_guard=True))
        # One finite screen; no automatic parameter chase after these targets.
        for v in ('execution','memory','combined'):pair(4096,v)
        frozen();status('completed_validated')
    except BaseException as exc:
        halt.set();state['error']=repr(exc);status('failed');raise

if __name__=='__main__':main()
