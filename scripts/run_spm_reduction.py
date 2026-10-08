"""Gated near-SPM FA experiment. Two new simulators, frozen inputs, fail closed."""
import concurrent.futures as futures
import json, math, os, re, shutil, subprocess, sys, threading, time
from pathlib import Path
import numpy as np
import run_finite_micro as finite
from build_access_timing import sha,save
from run_access_timing import stats
from fa_headroom_profiles import leaves
from test_micro_timing import check_global_guard
from export_safe_elf import export_hash
from reference import validate
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/cacheflex_flash_attention/near_spm_reduction_20260927'
BUILD=BASE/os.environ.get('CACHEFLEX_REDUCTION_BUILD','build')
GUEST=BASE/'guest_v2'
OUT=BASE/os.environ.get('CACHEFLEX_REDUCTION_EXPERIMENT','experiment')
OLD=ROOT/'results/cacheflex_flash_attention/spm_set_spread_20260927/experiment_retry_v2'
SEARCH=ROOT/'results/cacheflex_flash_attention/finite_model_20260926/analytical_experiment_v5/experiment'

def report(state):
    rows=['# Near-SPM reduction: accepted results only','',f"Stage: {state['stage']}",'',
          '| Case | FA ROI (us) | Completed updates | Arithmetic |','|---|---:|---:|---|']
    for r in state['completed']:
        rows.append(f"| {r['name']} | {r.get('roi_us','—')} | {r.get('commands','—')} | pass |")
    rows+=['','Do not compare smoke or gate runtimes as full-size speedup. See docs/experiments/NEAR_SPM_REDUCTION.md.',
           'SPM SRAM accesses use the existing timed controller path; no calibrated bank-level timing claim.']
    (OUT/'REPORT.md').write_text('\n'.join(rows)+'\n')

def audit(source,path,job,args,guest):
    def fields(p):return dict(leaves(json.loads((p/'config.json').read_text().replace(str(p),'<OUT>'))))
    old,new=fields(source),fields(path)
    delta={k:[old.get(k),new.get(k)] for k in old.keys()|new.keys() if old.get(k)!=new.get(k)}
    allow=r'system.cpu\[\d+\].workload\[0\].(cmd\[\d+\]|executable)|system.ruby.l0_cntrl\d+.sequencer.cacheflex_reduction.*'
    bad={k:v for k,v in delta.items() if not re.fullmatch(allow,k)}
    assert not bad,bad
    for i in range(8):
        seq=f'system.ruby.l0_cntrl{i}.sequencer.'
        assert new[seq+'cacheflex_reduction']==(job['mode']==3 or job.get('smoke',False))
        assert new[seq+'cacheflex_reduction_async']==not_sync(job)
        assert new[seq+'cacheflex_reduction_source_entries']==job.get('depth',4)
        assert new[seq+'cacheflex_reduction_sink_entries']==job.get('depth',4)
        assert not new[f'system.cpu[{i}].cacheFlexSpmStoreForwarding']
    assert new['system.cpu[0].workload[0].executable']==str(guest)
    assert [new[f'system.cpu[0].workload[0].cmd[{i+1}]'] for i in range(len(args))]==args
    return delta

def not_sync(job):return not job.get('sync',False)

def main():
    OUT.mkdir(parents=True);(OUT/'runs').mkdir()
    state=dict(pid=os.getpid(),stage='waiting_for_build',active={},completed=[])
    lock=threading.RLock();stop=threading.Event()
    def status(stage=None):
        with lock:
            if stage:state['stage']=stage
            state['updated']=time.time();save(OUT/'status.json',state);report(state)
    status()
    try:
        while True:
            if (BUILD/'status.json').exists():
                b=json.loads((BUILD/'status.json').read_text());assert b['stage']!='failed',b
                if b['stage']=='built_requires_validation':break
            time.sleep(5)
        assert sha(BUILD/'gem5.opt')==b['sha256']
        assert (GUEST/'frozen.json').exists(),'guest not built'
        template=json.loads((OLD/'runs/repaired_spm/command.json').read_text())
        shutil.copytree(Path(template[2]).parents[2],OUT/'configs',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        shutil.copy2(ROOT/'scripts/reduction_se.py',OUT/'configs/deprecated/example/reduction_se.py')
        frozen=json.loads((BUILD/'frozen.json').read_text())|json.loads((GUEST/'frozen.json').read_text())
        for p in [Path(__file__),ROOT/'scripts/reduction_se.py',*list((OUT/'configs').rglob('*.py'))]:frozen[str(p)]=sha(p)
        refs={}
        for t in (1024,4096,16384):
            source=OLD/'runs'/('repaired_spm' if t==1024 else f't{t}_spm_control')
            args=json.loads((source/'command.json').read_text())
            bc=int(args[args.index('--options')+1].split()[3])
            name=f't{t}_bc{bc}_parts1.npy'
            candidates=list((ROOT/'results/cacheflex_flash_attention/finite_model_20260926').glob('**/references/'+name))
            assert candidates, name
            p=candidates[0]
            assert sha(p)==json.loads(p.with_suffix('.json').read_text())['sha256']
            refs[t]=np.load(p);frozen[str(p)]=sha(p)
        save(OUT/'inputs.json',frozen)
        def unchanged():
            for p,h in frozen.items():assert sha(p)==h,('changed frozen input',p)
        def run(job):
            path=OUT/'runs'/job['name'];path.mkdir();proc=None;start=time.monotonic()
            try:
                assert not stop.is_set()
                t=job.get('t',1024);spm=job.get('spm',1)
                source=OLD/'runs'/((f'repaired_{"spm" if spm else "cache"}') if t==1024 else f't{t}_{"spm_control" if spm else "cache"}')
                cmd=json.loads((source/'command.json').read_text())
                cmd[0]=str(BUILD/'gem5.opt');cmd[1]='--outdir='+str(path)
                cmd[2]=str(OUT/'configs/deprecated/example/reduction_se.py')
                args=cmd[cmd.index('--options')+1].split()
                guest=Path(cmd[cmd.index('--cmd')+1])
                if job.get('smoke'):
                    guest=GUEST/'smoke';args=[]
                elif not job.get('original'):
                    guest=GUEST/'guest.export-safe';args+=[str(job['mode'])]
                cmd[cmd.index('--cmd')+1]=str(guest)
                cmd[cmd.index('--options')+1]=' '.join(args)
                for i,x in enumerate(cmd):
                    if x.startswith('system.cpu[0].workload[0].cmd='):
                        cmd[i]=f"system.cpu[0].workload[0].cmd={['kv-fa']+args!r}"
                if job['mode']==3 or job.get('smoke'):cmd+=['--cacheflex-reduction']
                if job.get('sync'):cmd+=['--cacheflex-reduction-sync']
                for i in range(8):
                    for suffix in ('source_entries','sink_entries'):
                        cmd+=['-P',f'system.ruby.l0_cntrl{i}.sequencer.cacheflex_reduction_{suffix}={job.get("depth",4)}']
                if job.get('slow'):
                    for i in range(8):
                        cmd+=['-P',f'system.ruby.l0_cntrl{i}.sequencer.cacheflex_reduction_link_bytes=4',
                              '-P',f'system.ruby.l0_cntrl{i}.sequencer.cacheflex_reduction_lanes=4']
                save(path/'command.json',cmd);save(path/'job.json',job);unchanged()
                with (path/'run.log').open('w') as log:
                    proc=subprocess.Popen(cmd,cwd=ROOT,env=finite.ENV,stdout=log,stderr=subprocess.STDOUT)
                    with lock:state['active'][job['name']]=dict(pid=proc.pid,start=time.time());status()
                    while proc.poll() is None:
                        assert not stop.is_set(),'peer failure'
                        assert time.monotonic()-start<172800,'timeout'
                        assert shutil.disk_usage(OUT).free>50*2**30,'disk guard'
                        try:proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:pass
                log=(path/'run.log').read_text()
                marker='REDUCTION_SMOKE PASS' if job.get('smoke') else 'KV_ROI_END'
                assert proc.returncode==0 and marker in log and 'Simulated exit code not 0!' not in log, log[-4000:]
                assert 'because exiting with last active thread context' in log,log[-2000:]
                row=dict(name=job['name'],job=job,source=str(source),config_delta=audit(source,path,job,args,guest))
                epochs=stats(path);ss=epochs[0]
                for epoch in epochs:
                    check_global_guard(epoch)
                    for i in range(8):
                        p=f'system.ruby.l0_cntrl{i}.sequencer.'
                        for label in ('Source','Sink'):assert epoch[p+'reductionMax'+label]<=job.get('depth',4)
                def total(k):return sum(ss[f'system.ruby.l0_cntrl{i}.sequencer.'+k] for i in range(8))
                expected=518 if job.get('smoke') else (t*math.ceil(t/int(args[3])) if job['mode']==3 else 0)
                assert total('reductionAccepted')==total('reductionCompleted')==expected,(total('reductionAccepted'),total('reductionCompleted'),expected)
                row['commands']=expected
                row['queue_full_retries']=total('reductionSourceFull')
                if job.get('smoke'):
                    if not job.get('sync'):assert row['queue_full_retries']>0,'queue saturation not exercised'
                else:
                    export_hash(path/'output.fp16',log)
                    row['arithmetic']=validate(path/'output.fp16',refs[t],t,dense=t==1024)
                    assert sha(path/'output.fp16')==sha(source/'output.fp16'),'output bits changed'
                    rows=int(args[2])*int(args[4]);bc=int(args[3])
                    copies=16*t*math.ceil((t//8)/rows) if spm else 0
                    if job['mode']>=2:copies+=4*t
                    assert total('m_spmcpRequestsIssued')==total('m_spmcpRequestsCompleted')==copies,(total('m_spmcpRequestsCompleted'),copies)
                    row.update(roi_us=ss['simTicks']/1e6,copies=copies,output_sha256=sha(path/'output.fp16'))
                    if job.get('original'):
                        old=stats(source);assert len(old)==len(epochs)
                        for a,b in zip(old,epochs):
                            assert all(b.get(k)==v for k,v in a.items()),[(k,v,b.get(k)) for k,v in a.items() if b.get(k)!=v][:15]
                            assert all(v==0 and '.reduction' in k for k,v in b.items() if k not in a)
                unchanged();row['host_seconds']=time.monotonic()-start;save(path/'accepted.json',row)
                with lock:state['completed'].append(row)
                return row
            except BaseException as e:
                stop.set();save(path/'failed.json',dict(error=repr(e)));raise
            finally:
                if proc and proc.poll() is None:
                    proc.terminate()
                    try:proc.wait(timeout=20)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()
                with lock:state['active'].pop(job['name'],None);status()
        def phase(name,jobs):
            status(name)
            with futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(run,jobs))
        phase('default_equivalence',[dict(name='old_'+a,spm=s,mode=0,original=True) for a,s in [('cache',0),('spm',1)]])
        phase('directed_ordering',[dict(name='smoke_'+n,mode=3,smoke=True,sync=sync,depth=d,slow=slow)
            for n,sync,d,slow in [('async',False,4,False),('sync',True,4,False),('stress',False,1,True)]])
        variants=[('cache_original',0,0,False),('cache_fused',0,1,False),('spm_original',1,0,False),
                  ('spm_fused',1,1,False),('spm_output',1,2,False),('reduce_sync',1,3,True),('reduce_async',1,3,False)]
        for t in (1024,4096,16384):
            phase('fa_'+str(t),[dict(name=f't{t}_{n}',t=t,spm=s,mode=m,sync=sync) for n,s,m,sync in variants])
            save(OUT/f't{t}_passed.json',dict(cases=[r for r in state['completed'] if r['job'].get('t')==t]))
        status('completed_requires_analysis')
    except BaseException as e:
        state['error']=repr(e);status('failed');raise
if __name__=='__main__':main()
