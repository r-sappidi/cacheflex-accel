"""Bounded matched traces: normal versus fast SPMLD, unchanged real SPMCP.

Full numerical default-equivalence gates precede diagnostic prefixes. Trace
ON/OFF prefixes must have identical finite non-host stats for each variant.
"""
import concurrent.futures as cf
import copy
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
sys.path.insert(0,str(ROOT/'benchmarks/cacheflex_flash_attention'))
from single_head_reference import check

BASE=ROOT/'results/cacheflex_flash_attention'
OUT=BASE/'spm_consumer_trace_20260923'
BUILD=OUT/'build'
EX=OUT/os.environ.get('CACHEFLEX_CONSUMER_EXPERIMENT','experiment_v2')
REF=BASE/'micro_resources_20260921/experiment_v6/preserved'
ENV=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib',
         OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
for k in list(ENV):
    if k.startswith('CACHEFLEX_AGGREGATE') or k=='CACHEFLEX_TRACE_LINE':ENV.pop(k)

def prior(variant,t):
    if variant=='normal':return BASE/f'timing_alignment_20260922/fa_v3/spm_t{t}'
    return BASE/f'calibrated_sensitivities_20260922/runs/micro_spm_t{t}_load_service'

def command(variant,t,p,start=None,end=None,flags=None,stop=None):
    cmd=json.loads((prior(variant,t)/'command.json').read_text())
    cmd=[f'--outdir={p}' if x.startswith('--outdir=') else x for x in cmd]
    cmd[0]=str(BUILD/'gem5.opt');cmd[2]=str(EX/'configs/deprecated/example/micro_timing_se.py')
    if flags:
        cmd[1:1]=[f'--debug-flags={flags}',f'--debug-start={start}',f'--debug-end={end}',
                  '--debug-file=trace.gz','--debug-ignore='+':'.join(f'system.cpu{i}' for i in range(1,7))]
    if stop is not None:cmd += [f'--abs-max-tick={stop}']
    return cmd

def config_check(old,new):
    a=json.loads((old/'config.json').read_text().replace(str(old),'<OUT>'))
    b=json.loads((new/'config.json').read_text().replace(str(new),'<OUT>'))
    for i in range(8):
        seq=b['system']['ruby'][f'l0_cntrl{i}']['sequencer']
        for k in ('cacheflex_ideal_staging_cache','cacheflex_ideal_staging_controller'):
            assert seq.pop(k) is None, 'Oracle must be OFF'
    assert a==b,'Unexpected configuration change'

def default_stats(old,new):
    for a,b in zip(old,new):
        extra={k:v for k,v in b.items() if k not in a}
        assert all('.m_idealStaging' in k and v==0 for k,v in extra.items()),extra
        assert {k:v for k,v in b.items() if k in a}==a,'Default observer build changes counters'
    assert len(old)==len(new)

def first_loads(p):
    found={}
    pat=re.compile(r'^(\d+): system.cpu([07])\..*\[SPM\] Load Issue to Cache:')
    with gzip.open(p/'trace.gz','rt') as stream:
        for line in stream:
            m=pat.search(line)
            if m:found.setdefault(int(m[2]),int(m[1]))
    assert set(found)=={0,7},('Missing sampled PV phase',found)
    return found

def main():
    EX.mkdir();state=dict(pid=os.getpid(),stage='waiting_for_build',active={},completed=[],windows={})
    lock=threading.RLock();halt=threading.Event();manifest={}
    def status(stage=None):
        with lock:
            if stage:state['stage']=stage
            state['updated']=time.time();save(EX/'status.json',state)
    def frozen():
        for p,h in manifest.items():assert sha(p)==h,('Frozen input changed',p)
    status()
    def run(variant,t,name,start=None,end=None,flags=None,stop=None):
        frozen();p=EX/name/variant;p.mkdir(parents=True)
        cmd=command(variant,t,p,start,end,flags,stop);save(p/'command.json',cmd)
        proc=None;beg=time.monotonic()
        try:
            with (p/'run.log').open('w') as log:
                proc=subprocess.Popen(cmd,cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT)
                with lock:state['active'][f'{name}/{variant}']=dict(pid=proc.pid,path=str(p));status()
                while proc.poll() is None:
                    assert not halt.is_set(),'Peer failed'
                    assert time.monotonic()-beg<3600,'One-hour case cap'
                    assert shutil.disk_usage(EX).free>50*2**30,'Disk guard'
                    trace=p/'trace.gz'
                    assert not trace.exists() or trace.stat().st_size<2**30,'1GiB compressed trace cap'
                    try:proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:pass
            assert proc.returncode==0,(name,variant,proc.returncode)
            config_check(prior(variant,t),p);s=stats(p)
            if stop is None:
                default_stats(stats(prior(variant,t)),s)
                assert sha(p/'output.fp16')==sha(prior(variant,t)/'output.fp16')
                num=check(p/'output.fp16',np.load(REF/f't{t}_bc128.npy'))
                assert num['errors']==num['nonfinite']==0,num
            else:
                assert s[-1]['finalTick']==stop
                assert 'simulate() limit reached' in (p/'run.log').read_text()
                num=None
            frozen()
            row=dict(variant=variant,stage=name,path=str(p),T=t,host_seconds=time.monotonic()-beg,
                     config_equal=True,arithmetic=num,diagnostic_prefix=stop is not None,
                     full_correctness_reference=str(prior(variant,t)/'validated.json'),stop=stop)
            save(p/'validated_prefix.json' if stop else p/'validated.json',row)
            with lock:state['active'].pop(f'{name}/{variant}');state['completed'].append(row);status()
            return p
        except BaseException:
            halt.set()
            if proc and proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
            with lock:state['active'].pop(f'{name}/{variant}',None);status()
            raise
    def pair(name,t=4096,windows=None,flags=None):
        status(name)
        with cf.ThreadPoolExecutor(max_workers=2) as pool:
            fs=[]
            for v in ('normal','fast'):
                args=windows[v] if windows else {}
                fs.append(pool.submit(run,v,t,name,flags=flags,**args))
            return [f.result() for f in fs]
    try:
        while not (BUILD/'frozen.json').exists():
            b=BUILD/'status.json'
            if b.exists():assert json.loads(b.read_text())['stage']!='failed','Build failed'
            time.sleep(5)
        shutil.copytree(BUILD/'configs',EX/'configs');shutil.copy2(__file__,EX/'runner_snapshot.py')
        manifest.update(json.loads((BUILD/'frozen.json').read_text()))
        paths=[Path(__file__).resolve(),ROOT/'scripts/run_access_timing.py',ROOT/'scripts/build_access_timing.py',
               ROOT/'benchmarks/cacheflex_flash_attention/single_head_reference.py',REF/'t512_bc128.npy']
        paths+=list((EX/'configs').rglob('*.py'))
        for t in (512,4096):
            for v in ('normal','fast'):
                p=prior(v,t);cmd=json.loads((p/'command.json').read_text())
                paths += [p/n for n in ('validated.json','command.json','config.json','stats.txt','output.fp16')]
                paths.append(Path(cmd[cmd.index('--cmd')+1]))
        manifest.update({str(p):sha(p) for p in paths});save(EX/'inputs.json',manifest)
        pair('t512_default_equivalence',t=512)
        save(EX/'gates_passed.json',dict(full_arithmetic=True,all_nonhost_stats_exact=True))
        # Predeclared early and mid-ROI samples, not selected by longest stalls.
        # Existing progress logs show worker computation starts near2.9e9;
        # the preceding timed single-core preparation contains no SPMLDs.
        for label,lo,hi in [('early',2800000000,3100000000),('middle',3800000000,4040000000)]:
            scan={v:dict(start=lo,end=hi,stop=hi+4000) for v in ('normal','fast')}
            paths=pair(label+'_scan',windows=scan,flags='SPM_CPU')
            windows={}
            for v,p in zip(('normal','fast'),paths):
                first=first_loads(p)
                start=min(first.values())-1000000
                end=max(first.values())+12000000
                windows[v]=dict(start=start,end=end,stop=end+400000)
            state['windows'][label]=windows;status();save(EX/f'{label}_windows.json',windows)
            pair(label+'_trace',windows=windows,flags='CacheFlexDeps')
            pair(label+'_observer_off',windows=windows)
            for v in ('normal','fast'):
                assert stats(EX/(label+'_trace')/v)==stats(EX/(label+'_observer_off')/v),'Tracing changed timing/counters'
            save(EX/f'{label}_observer_equivalence.json',dict(passed=True,all_finite_nonhost_stats=True))
            status(label+'_analyzing')
            subprocess.run([sys.executable,str(ROOT/'scripts/analyze_spm_consumers.py'),str(EX),label],check=True,cwd=ROOT)
        status('completed_validated_diagnostic')
    except BaseException as exc:
        halt.set();state['error']=repr(exc);status('failed');raise

if __name__=='__main__':main()
