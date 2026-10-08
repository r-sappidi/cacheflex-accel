"""Fail-closed shared-L3 tests and FA comparisons, two additional sims maximum."""
import concurrent.futures as futures
import json,math,os,re,shutil,subprocess,threading,time
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
BASE=ROOT/'results/cacheflex_l3_spm/initial_20260928'
BUILD=BASE/os.environ.get('CACHEFLEX_L3_BUILD','build_v4')
GUEST=BASE/os.environ.get('CACHEFLEX_L3_GUEST','guest_v4')
OUT=BASE/os.environ.get('CACHEFLEX_L3_EXPERIMENT','experiment_v7')
OLD=ROOT/'results/cacheflex_flash_attention/spm_set_spread_20260927/experiment_retry_v2'

def audit(source,path,job,args):
    def fields(p):return dict(leaves(json.loads((p/'config.json').read_text().replace(str(p),'<OUT>'))))
    old,new=fields(source),fields(path)
    delta={k:[old.get(k),new.get(k)] for k in old.keys()|new.keys() if old.get(k)!=new.get(k)}
    allowed=(r'system.cpu\[\d+\].workload\[0\].(cmd\[\d+\]|executable)|'
       r'system.ruby.l[12]_cntrl\d+.l3_.*|system.ruby.l2_cntrl\d+.L2cache.num_spm_ways|'
       r'system.ruby.l0_cntrl\d+.sequencer.cacheflex_reduction.*')
    if job.get('stress'):allowed+=r'|system.ruby.l2_cntrl\d+.(L1RequestToL2Cache|responseFromL2Cache).buffer_size'
    if 'latency' in job:allowed+=r'|system.ruby.l2_cntrl\d+.l3_spm_latency'
    bad={k:v for k,v in delta.items() if not re.fullmatch(allowed,k)}
    assert not bad,bad
    for i in range(8):
        p=f'system.ruby.l2_cntrl{i}.'
        assert new[p+'l3_spm_enabled']==job.get('shared',False)
        assert new[p+'l3_shared_service_port']==job.get('port',False)
        assert new[p+'L2cache.num_spm_ways']==(8 if job.get('shared') else 0)
        assert new[p+'L2cache.size']==524288
        assert not new[p+'L2cache.spm_separate_backing']
        assert new[f'system.ruby.l0_cntrl{i}.sequencer.cacheflex_reduction']==False
    assert new['system.cpu[0].workload[0].cmd[0]']=='kv-fa'
    return delta

def main():
    OUT.mkdir(parents=True);(OUT/'runs').mkdir()
    state=dict(pid=os.getpid(),stage='waiting_for_build',active={},completed=[])
    stop=threading.Event();lock=threading.RLock()
    def status(stage=None):
        with lock:
            if stage:state['stage']=stage
            state['updated']=time.time();save(OUT/'status.json',state)
    status()
    try:
        while True:
            if (BUILD/'status.json').exists():
                b=json.loads((BUILD/'status.json').read_text());assert b['stage']!='failed',b
                if b['stage']=='built_requires_validation':break
            time.sleep(5)
        assert sha(BUILD/'gem5.opt')==b['sha256']
        assert (GUEST/'frozen.json').exists()
        template=json.loads((OLD/'runs/repaired_spm/command.json').read_text())
        shutil.copytree(Path(template[2]).parents[2],OUT/'configs',
            ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        shutil.copy2(ROOT/'scripts/l3_spm_se.py',OUT/'configs/deprecated/example/l3_spm_se.py')
        frozen=json.loads((BUILD/'frozen.json').read_text())|json.loads((GUEST/'frozen.json').read_text())
        frozen[str(Path(__file__))]=sha(Path(__file__))
        for p in (OUT/'configs').rglob('*.py'):frozen[str(p)]=sha(p)
        refs={}
        for t in (1024,4096):
            src=OLD/'runs'/('repaired_spm' if t==1024 else 't4096_spm_control')
            oldcmd=json.loads((src/'command.json').read_text())
            bc=int(oldcmd[oldcmd.index('--options')+1].split()[3])
            name=f't{t}_bc{bc}_parts1.npy'
            candidates=list((ROOT/'results/cacheflex_flash_attention/finite_model_20260926').glob('**/references/'+name))
            assert candidates,name
            p=candidates[0];assert sha(p)==json.loads(p.with_suffix('.json').read_text())['sha256']
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
                cmd=json.loads((source/'command.json').read_text());args=cmd[cmd.index('--options')+1].split()
                cmd[0]=str(BUILD/'gem5.opt');cmd[1]='--outdir='+str(path)
                cmd[2]=str(OUT/'configs/deprecated/example/l3_spm_se.py')
                guest=Path(cmd[cmd.index('--cmd')+1])
                if job.get('smoke'):guest=GUEST/'smoke';args=['4']
                elif job.get('micro'):guest=GUEST/'latency';args=[str(job['mode']),str(job['workers'])]
                elif not job.get('original'):guest=GUEST/'guest.export-safe';args+=[str(job['mode'])]
                cmd[cmd.index('--cmd')+1]=str(guest);cmd[cmd.index('--options')+1]=' '.join(args)
                for i,x in enumerate(cmd):
                    if x.startswith('system.cpu[0].workload[0].cmd='):
                        cmd[i]=f"system.cpu[0].workload[0].cmd={['kv-fa']+args!r}"
                if job.get('shared'):cmd+=['--cacheflex-l3-spm']
                if job.get('port'):cmd+=['--cacheflex-l3-service-port']
                for i in range(8):
                    if job.get('stress'):
                        for field in ('L1RequestToL2Cache','responseFromL2Cache'):
                            cmd+=['-P',f'system.ruby.l2_cntrl{i}.{field}.buffer_size=2']
                    if 'latency' in job:cmd+=['-P',f'system.ruby.l2_cntrl{i}.l3_spm_latency={job["latency"]}']
                save(path/'command.json',cmd);save(path/'job.json',job);unchanged()
                with (path/'run.log').open('w') as log:
                    proc=subprocess.Popen(cmd,cwd=ROOT,env=finite.ENV,stdout=log,stderr=subprocess.STDOUT)
                    with lock:state['active'][job['name']]=dict(pid=proc.pid,start=time.time());status()
                    while proc.poll() is None:
                        assert not stop.is_set(),'peer failure'
                        assert time.monotonic()-start<14400,'timeout'
                        assert shutil.disk_usage(OUT).free>50*2**30,'disk guard'
                        try:proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:status()
                log=(path/'run.log').read_text()
                marker='L3_SPM_SMOKE PASS' if job.get('smoke') else ('L3_LATENCY PASS' if job.get('micro') else 'KV_ROI_END')
                assert proc.returncode==0 and marker in log and 'Simulated exit code not 0!' not in log,log[-4000:]
                assert 'because exiting with last active thread context' in log,log[-2000:]
                row=dict(name=job['name'],job=job,config_delta=audit(source,path,job,args))
                epochs=stats(path);ss=epochs[0]
                for epoch in epochs:check_global_guard(epoch)
                total=lambda k:sum(ss[f'system.ruby.l0_cntrl{i}.sequencer.'+k] for i in range(8))
                assert total('m_spmcpRequestsIssued')==total('m_spmcpRequestsCompleted')
                row.update(roi_us=ss['simTicks']/1e6,copies=total('m_spmcpRequestsCompleted'))
                if not job.get('smoke') and not job.get('micro'):
                    export_hash(path/'output.fp16',log)
                    row['arithmetic']=validate(path/'output.fp16',refs[t],t,dense=t==1024)
                    assert sha(path/'output.fp16')==sha(source/'output.fp16'),'output bits changed'
                    rows=int(args[2])*int(args[4]);bc=int(args[3])
                    expected=16*t*math.ceil((t//8)/rows) if spm else 0
                    assert total('m_spmcpRequestsCompleted')==expected,(row['copies'],expected)
                    if job.get('original'):
                        prior=stats(source);assert len(prior)==len(epochs)
                        for a,b in zip(prior,epochs):
                            assert all(b.get(k)==v for k,v in a.items()),[(k,v,b.get(k)) for k,v in a.items() if b.get(k)!=v][:15]
                        row['exact_prior_numeric_stats']=True
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
        phase('default_equivalence',[dict(name='default_'+a,spm=s,original=True) for a,s in [('cache',0),('spm',1)]])
        phase('shared_correctness',[dict(name='smoke_'+n,smoke=True,shared=True,port=True,stress=s)
            for n,s in [('normal',False),('small_queues',True)]])
        phase('latency_sanity',[dict(name=f'latency_m{m}_c{c}',micro=True,mode=m,workers=c,shared=m==2,port=True)
            for c in (1,8) for m in (0,1,2)])
        phase('latency_sensitivity',[dict(name='latency_l3_slow',micro=True,mode=2,workers=1,shared=True,port=True,latency=70)])
        for t in (1024,4096):
            phase(f'fa_{t}',[dict(name=f't{t}_{n}',t=t,spm=s,mode=m,shared=l3,port=True)
                for n,s,m,l3 in [('cache',0,1,False),('private_spm',1,1,False),('l3_output',1,2,True)]])
        status('completed_requires_analysis')
    except BaseException as e:state['error']=repr(e);status('failed');raise
if __name__=='__main__':main()
