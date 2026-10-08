"""Matched IFETCH response-edge x SPMLD-latency experiment, fixed kernels.

All changes default off; no data-path retiming or extra storage. Full numerical
and config gates, mechanism traces, then six full-ROI T4096 cases. Max two sims.
"""
import concurrent.futures as cf
from collections import defaultdict
import gzip
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
from build_access_timing import ROOT, sha, save
from run_access_timing import stats
from test_micro_timing import check_global_guard
from analyze_spm_consumers import parse_lines, distribution
sys.path.insert(0,str(ROOT/'benchmarks/cacheflex_flash_attention'))
from single_head_reference import check

BASE=ROOT/'results/cacheflex_flash_attention'
OUT=BASE/'ifetch_edge_20260923';BUILD=OUT/'build';EX=OUT/'experiment'
REF=BASE/'micro_resources_20260921/experiment_v6/preserved'
ENV=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib',
         OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
for k in list(ENV):
    if k.startswith('CACHEFLEX_AGGREGATE') or k=='CACHEFLEX_TRACE_LINE':ENV.pop(k)


def prior(backend,t):
    if backend=='spm_fast':
        return BASE/f'calibrated_sensitivities_20260922/runs/micro_spm_t{t}_load_service'
    return BASE/f'timing_alignment_20260922/fa_v3/{backend}_t{t}'


def command(backend,t,edge,p,flags=None,lo=0,hi=0):
    cmd=json.loads((prior(backend,t)/'command.json').read_text())
    cmd=[f'--outdir={p}' if x.startswith('--outdir=') else x for x in cmd]
    cmd[0]=str(BUILD/'gem5.opt');cmd[2]=str(EX/'configs/deprecated/example/micro_timing_se.py')
    for i in range(8):
        cmd+=['-P',f'system.ruby.l0_cntrl{i}.sequencer.cacheflex_ifetch_same_tick_response={edge}']
    if flags:
        cmd[1:1]=[f'--debug-flags={flags}',f'--debug-start={lo}',f'--debug-end={hi}',
                  '--debug-file=trace.gz','--debug-ignore='+':'.join(f'system.cpu{i}' for i in range(1,7))]
    return cmd


def audit_config(old,new,edge):
    def clean(obj,allow_edge=False):
        if isinstance(obj,list):return [clean(v,allow_edge) for v in obj]
        if not isinstance(obj,dict):return obj
        out={}
        for k,v in obj.items():
            if k=='cacheflex_ifetch_same_tick_response':
                expected=bool(allow_edge and re.fullmatch(r'system.ruby.l0_cntrl[0-7].sequencer',obj.get('path','')))
                assert v is expected,(obj.get('path'),k,v,expected)
            elif k in ('cacheflex_ideal_staging_cache','cacheflex_ideal_staging_controller'):
                assert v is None, 'Oracle must stay off'
            else:out[k]=clean(v,allow_edge)
        return out
    assert clean(old)==clean(new,edge), 'Undeclared configuration change'


def control_stats(old,new):
    assert len(old)==len(new)
    for a,b in zip(old,new):
        extra={k:v for k,v in b.items() if k not in a}
        assert all(('.ifetchEdge' in k or '.m_idealStaging' in k) and v==0 for k,v in extra.items()),extra
        assert {k:v for k,v in b.items() if k in a}==a,'Default changed non-host stats'


def counter(s,name):
    vals=[v for k,v in s.items() if re.fullmatch(r'system.ruby.l0_cntrl[0-7].sequencer\.'+name,k)]
    assert len(vals)==8,(name,len(vals))
    return sum(vals)


def first_loads(p):
    found={};pat=re.compile(r'^(\d+): system.cpu([07])\..*\[SPM\] Load Issue to Cache:')
    with gzip.open(p/'trace.gz','rt') as f:
        for line in f:
            m=pat.match(line)
            if m:found.setdefault(int(m[2]),int(m[1]))
    assert set(found)=={0,7},found
    return found


def mechanism(p):
    with gzip.open(p/'trace.gz','rt') as f:nodes,_,_=parse_lines(f)
    pat=re.compile(r'^(\d+): system.cpu([07]).fetch: .*Instruction PC .*created \[sn:(\d+)\]')
    fetched={}
    with gzip.open(p/'trace.gz','rt') as f:
        for line in f:
            m=pat.match(line)
            if m:fetched[(int(m[2]),int(m[3]))]=int(m[1])
    rows=[]
    for core in (0,7):
        ls=sorted((n for n in nodes.values() if n['core']==core and n.get('spm') and n.get('load')
                   and 'commit' in n and 'dispatch' in n and len(n['issues'])==1
                   and (core,n['sn']) in fetched),key=lambda n:n['sn'])
        for a,b in zip(ls,ls[1:]):
            if b['va']!=a['va']+64:continue
            rows.append(dict(core=core,sn=a['sn'],next_sn=b['sn'],
                             fetch=(fetched[(core,b['sn'])]-fetched[(core,a['sn'])])/400,
                             dispatch=(b['dispatch']-a['dispatch'])/400,
                             issue=(b['issues'][0]-a['issues'][0])/400))
    assert len(rows)>=64,('Too few mechanism samples',len(rows))
    result=dict(pairs=len(rows),cores=sorted({r['core'] for r in rows}),
                metrics={k:distribution([r[k] for r in rows]) for k in ('fetch','dispatch','issue')})
    assert result['cores']==[0,7]
    assert result['metrics']['fetch']['median']<7,('Did not reduce observed fetch cadence',result)
    save(p/'mechanism.json',result);save(p/'mechanism_pairs.json',rows)
    return result


def report(state):
    rows=[r for r in state['completed'] if r.get('kind')=='fa']
    lines=['# Matched instruction-fetch edge sensitivity','',
           'Fixed8core single-head VL16 FA; calibrated64KiB privateL2 and4MiB totalLLC. '
           'IFETCH-only completed-response bypass when its response queue is empty; ordinary '
           'retry/ordering fallback retained. No changed cache service latencies, data paths, '
           'kernels, issue widths, storage, or SPMCP. Diagnostic, not hardware calibration.', '',
           '| T | Backend | IFETCH edge | ROI us | Direct IFETCH responses | Cache/SPM ratio on same platform |',
           '|---|---|---|---:|---:|---:|']
    for r in rows:
        c=next((x for x in rows if x['T']==r['T'] and x['backend']=='cache' and x['edge']==r['edge']),None)
        ratio=f'{c["roi_us"]/r["roi_us"]:.6f}x' if c else 'pending'
        lines.append(f'| {r["T"]} | {r["backend"]} | {r["edge"]} | {r["roi_us"]:.4f} | '
                     f'{r["ifetch_direct"]:.0f} | {ratio} |')
    lines+=['','## Matched causal comparisons','']
    for t in (512,4096):
        for b in ('cache','spm','spm_fast'):
            group={r['edge']:r for r in rows if r['T']==t and r['backend']==b}
            if len(group)==2:
                gain=100*(1-group[True]['roi_us']/group[False]['roi_us'])
                lines.append(f'- T{t} {b}: IFETCH intervention reduces its own ROI by {gain:.4f}%.')
        for edge in (False,True):
            group={r['backend']:r for r in rows if r['T']==t and r['edge']==edge}
            if 'spm' in group and 'spm_fast' in group:
                gain=100*(1-group['spm_fast']['roi_us']/group['spm']['roi_us'])
                lines.append(f'- T{t}, IFETCH edge={edge}: faster SPMLD reduces SPM ROI by {gain:.4f}%.')
    lines+=['','Only fully numerically/configuration-validated cases appear. Negative reduction '
            'means slowdown. Do not compare faster-front-end SPM against slower-front-end cache. '
            'The change in SPMLD benefit between front ends is the interaction of interest; '
            'these intervention deltas are not an additive attribution of FA runtime.']
    (EX/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    EX.mkdir();(EX/'runs').mkdir()
    state=dict(pid=os.getpid(),stage='waiting_for_build',active={},completed=[])
    lock=threading.RLock();halt=threading.Event();manifest={}
    def status(stage=None):
        with lock:
            if stage:state['stage']=stage
            state['updated']=time.time();save(EX/'status.json',state);report(state)
    def frozen():
        for p,h in manifest.items():assert sha(p)==h,('Changed frozen input',p)
    status()
    def run(backend,t,edge,trace=False):
        frozen();tag=f'{backend}_t{t}_edge{int(edge)}'+('_trace' if trace else '')
        p=EX/'runs'/tag;p.mkdir();old=prior(backend,t)
        flags=None;lo=0;hi=0
        if t==512 and edge and backend!='cache':
            flags='SPM_CPU';hi=int(stats(old)[-1]['finalTick']*2)
        if trace:
            times=first_loads(EX/'runs'/f'{backend}_t{t}_edge1')
            lo=min(times.values())-1_000_000;hi=max(times.values())+3_000_000
            flags='CacheFlexDeps,Fetch,Decode,Rename'
        cmd=command(backend,t,edge,p,flags,lo,hi);save(p/'command.json',cmd)
        proc=None;begin=time.monotonic()
        try:
            with (p/'run.log').open('w') as f:
                proc=subprocess.Popen(cmd,cwd=ROOT,env=ENV,stdout=f,stderr=subprocess.STDOUT)
                with lock:state['active'][tag]=dict(pid=proc.pid,path=str(p));status()
                while proc.poll() is None:
                    assert not halt.is_set(),'Peer failed'
                    assert time.monotonic()-begin<10800,'3-hour case cap'
                    assert shutil.disk_usage(EX).free>50*2**30,'Disk guard'
                    tr=p/'trace.gz';assert not tr.exists() or tr.stat().st_size<2**30,'Trace cap'
                    try:proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:pass
            assert proc.returncode==0,(tag,proc.returncode)
            a=json.loads((old/'config.json').read_text().replace(str(old),'<OUT>'))
            b=json.loads((p/'config.json').read_text().replace(str(p),'<OUT>'))
            audit_config(a,b,edge)
            assert sha(p/'output.fp16')==sha(old/'output.fp16'),'Export differs'
            num=check(p/'output.fp16',np.load(REF/f't{t}_bc128.npy'))
            assert num['errors']==num['nonfinite']==0,num
            h=1469598103934665603
            for w in np.fromfile(p/'output.fp16',dtype=np.uint16):h=((h^int(w))*1099511628211)&((1<<64)-1)
            printed=re.search(r'FLASH_RESULT.*?hash=([0-9a-f]+)',(p/'run.log').read_text())[1]
            assert f'{h:016x}'==printed,'Guest hash/export mismatch'
            allstats=stats(p);s=allstats[0];check_global_guard(s)
            copies=0 if backend=='cache' else (t//64)*t*2
            for name in ('m_spmcpRequestsIssued','m_spmcpRequestsCompleted'):
                assert counter(s,name)==copies,(name,counter(s,name),copies)
            assert counter(s,'m_idealStagingCopies')==0
            direct=counter(s,'ifetchEdgeDirect');queued=counter(s,'ifetchEdgeQueued')
            eligible=counter(s,'ifetchEdgeEligible')
            assert direct+queued==eligible
            if edge:assert direct>0,'Intervention never exercised'
            else:
                assert eligible==0
                control_stats(stats(old),allstats)
            if trace:
                assert stats(EX/'runs'/f'{backend}_t{t}_edge1')==allstats,'Trace changed timing'
                diag=mechanism(p)
            else:diag=None
            row=dict(kind='trace' if trace else 'fa',backend=backend,T=t,edge=edge,
                     path=str(p),roi_us=s['simTicks']/1e6,arithmetic=num,output_hash=printed,
                     copy_count=copies,ifetch_direct=direct,ifetch_queued=queued,
                     config_audit=True,exact_old_stats=not edge,mechanism=diag,
                     host_seconds=time.monotonic()-begin)
            frozen();save(p/'validated.json',row)
            with lock:state['active'].pop(tag);state['completed'].append(row);status()
        except BaseException:
            halt.set()
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
    try:
        while not (BUILD/'frozen.json').exists():
            p=BUILD/'status.json'
            if p.exists():assert json.loads(p.read_text())['stage']!='failed','Build failed'
            time.sleep(5)
        shutil.copytree(BUILD/'configs',EX/'configs');shutil.copy2(__file__,EX/'runner_snapshot.py')
        manifest.update(json.loads((BUILD/'frozen.json').read_text()))
        paths=[Path(__file__).resolve(),Path(sys.modules['single_head_reference'].__file__),
               ROOT/'scripts/build_access_timing.py',ROOT/'scripts/run_access_timing.py',
               ROOT/'scripts/test_micro_timing.py',ROOT/'scripts/analyze_spm_consumers.py']
        paths+=list((EX/'configs').rglob('*.py'))
        for t in (512,4096):
            paths.append(REF/f't{t}_bc128.npy')
            for backend in ('cache','spm','spm_fast'):
                old=prior(backend,t);cmd=json.loads((old/'command.json').read_text())
                paths+=[old/n for n in ('command.json','config.json','stats.txt','validated.json','output.fp16')]
                paths.append(Path(cmd[cmd.index('--cmd')+1]))
        manifest.update({str(p):sha(p) for p in paths});save(EX/'inputs.json',manifest)
        status('t512_default_equivalence')
        parallel([(b,512,False) for b in ('cache','spm','spm_fast')])
        status('t512_intervention_numerical')
        parallel([(b,512,True) for b in ('cache','spm','spm_fast')])
        status('t512_mechanism_and_observer_equivalence')
        parallel([(b,512,True,True) for b in ('spm','spm_fast')])
        save(EX/'gates_passed.json',dict(full_math_export_config=True,default_exact=True,
                                       mechanism_exercised=True,trace_timing_exact=True))
        status('t4096_matched_controls')
        parallel([(b,4096,False) for b in ('cache','spm','spm_fast')])
        status('t4096_matched_interventions')
        parallel([(b,4096,True) for b in ('cache','spm','spm_fast')])
        status('completed_validated')
    except BaseException as e:
        halt.set();state['error']=repr(e);status('failed');raise


if __name__=='__main__':main()
