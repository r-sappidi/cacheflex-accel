"""Fail-closed near-L3 comparison; two new simulations, frozen inputs."""
import concurrent.futures as futures
import json,math,os,re,shutil,subprocess,threading,time,traceback
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
BASE=ROOT/'results/cacheflex_l3_spm/reduction_20260928'
BUILD=BASE/os.environ.get('CACHEFLEX_L3_REDUCTION_BUILD','build_v1')
GUEST=BASE/os.environ.get('CACHEFLEX_L3_REDUCTION_GUEST','guest_v1')
OUT=BASE/os.environ.get('CACHEFLEX_L3_REDUCTION_EXPERIMENT','experiment_v2')
OLD=ROOT/'results/cacheflex_l3_spm/initial_20260928/experiment_v7'

def audit(source,path,job,args):
    def fields(p):return dict(leaves(json.loads((p/'config.json').read_text().replace(str(p),'<OUT>'))))
    a,b=fields(source),fields(path)
    delta={k:[a.get(k),b.get(k)] for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}
    allowed=(r'system.cpu\[\d+\].workload\[0\].(cmd\[\d+\]|executable)|'
        r'system.ruby.l[12]_cntrl\d+.l3_.*|system.ruby.l2_cntrl\d+.l3ReductionQueue(\..*)?|'
        r'system.ruby.l2_cntrl\d+.L2cache.num_spm_ways|'
        r'system.ruby.l0_cntrl\d+.sequencer.cacheflex_(l3_reduction|reduction.*)')
    assert not {k:v for k,v in delta.items() if not re.fullmatch(allowed,k)},delta
    reduction=job.get('mode')==3 or job.get('smoke',False)
    for i in range(8):
        q=f'system.ruby.l0_cntrl{i}.sequencer.';p=f'system.ruby.l2_cntrl{i}.'
        assert b[q+'cacheflex_l3_reduction']==reduction
        assert b[q+'cacheflex_reduction']==reduction
        assert b[p+'l3_reduction_enabled']==reduction
        assert b[p+'l3_shared_service_port']
        assert b[p+'l3_spm_enabled']==job.get('shared',False)
        assert b[p+'L2cache.num_spm_ways']==(8 if job.get('shared') else 0)
        assert b[p+'L2cache.size']==524288 and not b[p+'L2cache.spm_separate_backing']
        assert b[p+'l3ReductionQueue.buffer_size']==job.get('depth',4)
        assert b[p+'l3_reduction_entries']==job.get('depth',4)
        assert b[p+'l3_reduction_arithmetic']==job.get('alu',9)
        assert b[p+'l3_spm_latency']==35
        if reduction:
            assert not b[f'system.cpu[{i}].cacheFlexSpmStoreForwarding']
            assert b[q+'cacheflex_reduction_async']==(not job.get('sync',False))
    assert b['system.cpu[0].workload[0].cmd[0]']=='kv-fa'
    return delta

def report(state):
    lines=['# Near-L3 reduction — accepted results only','',f"Queue: {state['stage']}",'',
        '| Case | ROI us | Completed reductions |','|---|---:|---:|']
    for r in state['completed']:lines.append(f"| {r['name']} | {r['roi_us']:.4f} | {r['commands']} |")
    lines+=['','No extra accumulator SRAM. Finite additional queues/ALUs and full network payload costs.',
        'Directed tests are not FA speedups. Compare all FA arms and retain stronger prior controls.',
        'Explicit timing assumptions, not calibrated bank RTL or a general coherence/ISA proof.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')

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
        shutil.copytree(OLD/'configs',OUT/'configs',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        shutil.copy2(ROOT/'scripts/l3_reduction_se.py',OUT/'configs/deprecated/example/l3_reduction_se.py')
        frozen=json.loads((BUILD/'frozen.json').read_text())|json.loads((GUEST/'frozen.json').read_text())
        for p in [Path(__file__),*list((OUT/'configs').rglob('*.py'))]:frozen[str(p)]=sha(p)
        refs={}
        for t,bc in ((1024,128),(4096,512)):
            candidates=list((ROOT/'results/cacheflex_flash_attention/finite_model_20260926').glob(f'**/references/t{t}_bc{bc}_parts1.npy'))
            p=candidates[0];assert sha(p)==json.loads(p.with_suffix('.json').read_text())['sha256']
            refs[t]=np.load(p);frozen[str(p)]=sha(p)
        save(OUT/'inputs.json',frozen)
        def unchanged():
            for p,h in frozen.items():assert sha(p)==h,('modified input',p)
        def run(job):
            path=OUT/'runs'/job['name'];path.mkdir();proc=None;start=time.monotonic()
            try:
                assert not stop.is_set()
                t=job.get('t',1024);source=OLD/'runs'/f't{t}_{job.get("source","private_spm")}'
                assert (source/'accepted.json').exists(),('unaccepted reference',source)
                cmd=json.loads((source/'command.json').read_text());args=cmd[cmd.index('--options')+1].split()
                cmd[0]=str(BUILD/'gem5.opt');cmd[1]='--outdir='+str(path)
                cmd[2]=str(OUT/'configs/deprecated/example/l3_reduction_se.py')
                guest=Path(cmd[cmd.index('--cmd')+1])
                if job.get('smoke'):guest=GUEST/'smoke';args=[]
                elif not job.get('original'):guest=GUEST/'guest.export-safe';args[-1]=str(job['mode'])
                cmd[cmd.index('--cmd')+1]=str(guest);cmd[cmd.index('--options')+1]=' '.join(args)
                for i,x in enumerate(cmd):
                    if x.startswith('system.cpu[0].workload[0].cmd='):cmd[i]=f"system.cpu[0].workload[0].cmd={['kv-fa']+args!r}"
                if job.get('shared') and '--cacheflex-l3-spm' not in cmd:cmd+=['--cacheflex-l3-spm']
                reduction=job.get('mode')==3 or job.get('smoke',False)
                if reduction:cmd+=['--cacheflex-l3-reduction']
                if job.get('sync'):cmd+=['--cacheflex-l3-reduction-sync']
                cmd+=['--cacheflex-l3-reduction-depth='+str(job.get('depth',4)),
                      '--cacheflex-l3-reduction-arithmetic='+str(job.get('alu',9))]
                for i in range(8):
                    cmd+=['-P',f'system.ruby.l2_cntrl{i}.l3ReductionQueue.buffer_size={job.get("depth",4)}']
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
                log=(path/'run.log').read_text();marker='L3_REDUCTION_SMOKE PASS' if job.get('smoke') else 'KV_ROI_END'
                assert proc.returncode==0 and marker in log and 'Simulated exit code not 0!' not in log,log[-4000:]
                assert 'because exiting with last active thread context' in log
                row=dict(name=job['name'],job=job,config_delta=audit(source,path,job,args))
                epochs=stats(path);s=epochs[0]
                total=lambda key:sum(s[f'system.ruby.l0_cntrl{i}.sequencer.'+key] for i in range(8))
                for epoch in epochs:
                    check_global_guard(epoch)
                    for i in range(8):
                        for q in ('Source','Sink'):
                            assert epoch[f'system.ruby.l0_cntrl{i}.sequencer.reductionMax{q}']<=job.get('depth',4)
                expected=1040 if job.get('smoke') else (t*math.ceil(t/int(args[3])) if reduction else 0)
                assert total('reductionAccepted')==total('reductionCompleted')==expected,(total('reductionAccepted'),total('reductionCompleted'),expected)
                row.update(roi_us=s['simTicks']/1e6,commands=expected)
                assert total('m_spmcpRequestsIssued')==total('m_spmcpRequestsCompleted')
                if not job.get('smoke'):
                    export_hash(path/'output.fp16',log)
                    row['arithmetic']=validate(path/'output.fp16',refs[t],t,dense=t==1024)
                    assert sha(path/'output.fp16')==sha(source/'output.fp16'),'output bits changed'
                    rows=int(args[2])*int(args[4])
                    assert total('m_spmcpRequestsCompleted')==(16*t*math.ceil((t//8)/rows) if int(args[7]) else 0)
                    if job.get('original'):
                        prior=stats(source);assert len(prior)==len(epochs)
                        for a,b in zip(prior,epochs):
                            assert all(b.get(k)==v for k,v in a.items()),[(k,v,b.get(k)) for k,v in a.items() if b.get(k)!=v][:12]
                        row['exact_prior_stats']=True
                unchanged();row['host_seconds']=time.monotonic()-start;save(path/'accepted.json',row)
                with lock:state['completed'].append(row);status()
            except BaseException as e:stop.set();save(path/'failed.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
            finally:
                if proc and proc.poll() is None:
                    proc.terminate()
                    try:proc.wait(timeout=20)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()
                with lock:state['active'].pop(job['name'],None);status()
        def phase(name,jobs):
            status(name)
            workers=int(os.environ.get('CACHEFLEX_L3_REDUCTION_WORKERS','2'));assert 1<=workers<=2
            with futures.ThreadPoolExecutor(max_workers=workers) as pool:list(pool.map(run,jobs))
        phase('disabled_equivalence',[dict(name='disabled_'+n,source=n,original=True,shared=n=='l3_output')
            for n in ('cache','private_spm','l3_output')])
        phase('directed_correctness',[dict(name='smoke_'+n,smoke=True,shared=True,sync=sync,depth=d,alu=a)
            for n,sync,d,a in [('async',False,4,9),('sync',True,4,9),('one_entry_slow',False,1,40)]])
        for t in (1024,4096):
            phase(f'fa_{t}',[dict(name=f't{t}_{n}',t=t,source=src,mode=m,shared=m>=2,sync=sync)
                for n,src,m,sync in [('cache','cache',1,False),('private_spm','private_spm',1,False),
                    ('l3_output','private_spm',2,False),('l3_async','private_spm',3,False),('l3_sync','private_spm',3,True)]])
        status('completed_requires_analysis')
    except BaseException as e:state['error']=repr(e);status('failed');raise
if __name__=='__main__':main()
