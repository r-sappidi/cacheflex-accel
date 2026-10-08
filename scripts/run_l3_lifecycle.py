"""Frozen, fail-closed first-update and near-L3 finalization experiment."""
import concurrent.futures as futures
import importlib,json,math,os,re,shutil,subprocess,threading,time,traceback
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
BASE=ROOT/'results/cacheflex_l3_spm/lifecycle_20260928'
BUILD=BASE/os.environ.get('CACHEFLEX_L3_LIFECYCLE_BUILD','build_v2')
GUEST=BASE/os.environ.get('CACHEFLEX_L3_LIFECYCLE_GUEST','guest_v1')
OUT=BASE/os.environ.get('CACHEFLEX_L3_LIFECYCLE_EXPERIMENT','experiment_v1')
OLD=ROOT/'results/cacheflex_l3_spm/pipeline_20260928/experiment_v2'

def source_for(job):
    if job.get('smoke'):return OLD/'runs'/job.get('template','half_pipeline')
    t=job.get('t',1024);arm=job.get('arm','base')
    return OLD/'runs'/f"t{t}_{arm if arm in ('cache','private_spm') else ('both' if t==1024 else 'half')}"

def audit(source,path,job):
    def fields(p):return dict(leaves(json.loads((p/'config.json').read_text().replace(str(p),'<OUT>'))))
    a,b=fields(source),fields(path)
    delta={k:[a.get(k),b.get(k)] for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}
    allowed=(r'system.ruby.l2_cntrl\d+.(l3_reduction_(initialize|finalize)|l3_finalize_arithmetic)|'
        r'system.cpu\[\d+\].workload\[0\].(cmd\[\d+\]|executable)')
    assert not {k:v for k,v in delta.items() if not re.fullmatch(allowed,k)},delta
    for i in range(8):
        p=f'system.ruby.l2_cntrl{i}.'
        assert b[p+'l3_reduction_initialize']==job.get('init',False)
        assert b[p+'l3_reduction_finalize']==job.get('final',False)
        assert b[p+'l3_finalize_arithmetic']==8
        assert b[p+'L2cache.size']==524288 and not b[p+'L2cache.spm_separate_backing']
        assert b[p+'l3_spm_latency']==35 and b[p+'l3_shared_service_port']
    return delta,b

def report(state):
    rows={r['name']:r for r in state['completed']}
    lines=['# L3 accumulator lifecycle','',f"Stage: {state['stage']}",'',
        '| Case | ROI us | Initializations | Final64B writes | DRAM bytes written |',
        '|---|---:|---:|---:|---:|']
    for r in state['completed']:
        lines.append(f"| {r['name']} | {r['roi_us']:.4f} | {r['initializations']} | {r['finalizations']} | {r['dram_write_bytes']:.0f} |")
    lines+=['','## Same-ELF end-to-end comparison','',
        '| T | Variant | ROI us | speedup vs reduction | vs cache | vs private SPM |',
        '|---|---|---:|---:|---:|---:|']
    for t in (1024,4096):
        controls=[rows.get(f't{t}_{x}') for x in ('base','cache','private_spm')]
        if not all(controls):continue
        for arm in ('base','init','final','both'):
            r=rows.get(f't{t}_{arm}')
            if r:
                v=r['roi_us'];ratios=[c['roi_us']/v for c in controls]
                lines.append(f'| {t} | {arm} | {v:.4f} | {ratios[0]:.6f}x | {ratios[1]:.6f}x | {ratios[2]:.6f}x |')
    lines+=['','Accepted full arithmetic/exact output/configuration/finite-resource checks only.',
        'T1024 uses pipelined FP16; T4096 uses faster serialized FP16, fixed within each comparison.',
        'No extra accumulator SRAM or larger queues. Same kernel mapping and tiles.',
        'Finalization reuses translated coherent SPMWB: two64B writes/row, with directory/DRAM ack.',
        'This write-through output policy and extra controller hop remain a limitation, not free L3 arithmetic.',
        'Raw command/stall cycles are not additive runtime fractions. Original controls remain preserved.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')

def main():
    OUT.mkdir(parents=True);(OUT/'runs').mkdir()
    state=dict(pid=os.getpid(),stage='preparing',active={},completed=[])
    lock=threading.RLock();stop=threading.Event()
    def status(stage=None):
        with lock:
            if stage:state['stage']=stage
            state['updated']=time.time();save(OUT/'status.json',state);report(state)
    status()
    try:
        build=json.loads((BUILD/'status.json').read_text())
        assert build['stage']=='built_requires_validation' and sha(BUILD/'gem5.opt')==build['sha256']
        shutil.copytree(OLD/'configs',OUT/'configs',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        shutil.copy2(ROOT/'scripts/l3_lifecycle_se.py',OUT/'configs/deprecated/example/l3_lifecycle_se.py')
        frozen=json.loads((BUILD/'frozen.json').read_text())|json.loads((GUEST/'frozen.json').read_text())
        frozen.update(json.loads((OLD.parent/'guest_v2/frozen.json').read_text()))
        for p in [Path(__file__),*list((OUT/'configs').rglob('*.py'))]:frozen[str(p)]=sha(p)
        for name in ('run_finite_micro','reference','run_access_timing','test_micro_timing','fa_headroom_profiles','export_safe_elf','build_access_timing'):
            p=Path(importlib.import_module(name).__file__);frozen[str(p)]=sha(p)
        refs={}
        for t,bc in ((1024,128),(4096,512)):
            p=next((ROOT/'results/cacheflex_flash_attention/finite_model_20260926').glob(f'**/references/t{t}_bc{bc}_parts1.npy'))
            assert sha(p)==json.loads(p.with_suffix('.json').read_text())['sha256']
            refs[t]=np.load(p);frozen[str(p)]=sha(p)
        save(OUT/'inputs.json',frozen)
        def unchanged():
            for p,h in frozen.items():assert sha(p)==h,('modified input',p)
        def run(job):
            path=OUT/'runs'/job['name'];path.mkdir();proc=None;start=time.monotonic()
            try:
                assert not stop.is_set()
                source=source_for(job);assert (source/'accepted.json').exists(),source
                prior=json.loads((source/'accepted.json').read_text())
                save(path/'prior_files.json',{str(source/n):sha(source/n) for n in ('accepted.json','command.json','stats.txt','config.json')})
                cmd=json.loads((source/'command.json').read_text())
                cmd[0]=str(BUILD/'gem5.opt');cmd[1]='--outdir='+str(path)
                cmd[2]=str(OUT/'configs/deprecated/example/l3_lifecycle_se.py')
                args=cmd[cmd.index('--options')+1].split();smoke=job.get('smoke',False)
                if not job.get('exact'):
                    cmd[cmd.index('--cmd')+1]=str(GUEST/('lifecycle_smoke' if smoke else 'guest.export-safe'))
                    if smoke:args=[str(int(job.get('init',False))+2*int(job.get('final',False)))]
                    elif job.get('arm','base') not in ('cache','private_spm'):
                        args[-1]=str(4+int(job.get('init',False))+2*int(job.get('final',False)))
                    cmd[cmd.index('--options')+1]=' '.join(args)
                    for i,x in enumerate(cmd):
                        if x.startswith('system.cpu[0].workload[0].cmd='):
                            cmd[i]=f"system.cpu[0].workload[0].cmd={['kv-fa']+args!r}"
                if job.get('init'):cmd+=['--cacheflex-l3-initialize']
                if job.get('final'):cmd+=['--cacheflex-l3-finalize']
                save(path/'command.json',cmd);save(path/'job.json',job);unchanged()
                with (path/'run.log').open('w') as log:
                    proc=subprocess.Popen(cmd,cwd=ROOT,env=finite.ENV,stdout=log,stderr=subprocess.STDOUT)
                    with lock:state['active'][job['name']]=dict(pid=proc.pid,start=time.time());status()
                    while proc.poll() is None:
                        assert not stop.is_set(),'peer failed'
                        assert time.monotonic()-start<14400,'timeout'
                        assert shutil.disk_usage(OUT).free>50*2**30,'disk guard'
                        try:proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:status()
                log=(path/'run.log').read_text()
                assert proc.returncode==0 and ('L3_LIFECYCLE_SMOKE PASS errors=0' if smoke else 'KV_ROI_END') in log,log[-5000:]
                assert 'because exiting with last active thread context' in log and 'Simulated exit code not 0!' not in log
                epochs=stats(path);old=stats(source);s=epochs[0]
                total=lambda suffix:sum(v for k,v in s.items() if k.endswith(suffix))
                t=job.get('t',1024);reduction=smoke or job.get('arm','base') not in ('cache','private_spm')
                expected=768 if smoke else (t*math.ceil(t/int(args[3])) if reduction else 0)
                initializations=(192 if smoke else t) if job.get('init') else 0
                finalizations=(384 if smoke else 2*t) if job.get('final') else 0
                assert total('.reductionAccepted')==total('.reductionCompleted')==expected
                assert total('.l3ReductionStarts')==total('.l3ReductionFinishes')==expected+finalizations
                assert total('.l3ReductionHalfCommands')==expected
                assert total('.l3ReductionInitializations')==initializations
                assert total('.l3Finalizations')==finalizations
                assert total('.reductionTransportBytes')==144*expected
                delta,config=audit(source,path,job)
                for ep in epochs:
                    check_global_guard(ep)
                    for i in range(8):
                        p=f'system.ruby.l2_cntrl{i}.';depth=config[p+'l3_reduction_entries']
                        for q in ('Source','Sink'):assert ep[f'system.ruby.l0_cntrl{i}.sequencer.reductionMax{q}']<=depth
                        assert ep[p+'L2cache.l3ReductionMaxActive']<=(depth if config[p+'l3_reduction_pipeline'] else 1)
                row=dict(name=job['name'],job=job,source=str(source),roi_us=s['simTicks']/1e6,
                    prior_us=prior['roi_us'],commands=expected,initializations=initializations,finalizations=finalizations,
                    dram_write_bytes=sum(v for k,v in s.items() if k.endswith('.dram.bytesWritten::total')),
                    config_delta=delta)
                if not smoke:
                    export_hash(path/'output.fp16',log)
                    row['arithmetic']=validate(path/'output.fp16',refs[t],t,dense=t==1024)
                    assert sha(path/'output.fp16')==sha(source/'output.fp16'),'output bits changed'
                    assert total('.m_spmcpRequestsIssued')==total('.m_spmcpRequestsCompleted')
                    assert total('.m_spmcpRequestsCompleted')==sum(v for k,v in old[0].items() if k.endswith('.m_spmcpRequestsCompleted'))
                if job.get('exact'):
                    assert len(epochs)==len(old)
                    for a,b in zip(old,epochs):
                        assert all(b.get(k)==v for k,v in a.items()),[(k,v,b.get(k)) for k,v in a.items() if b.get(k)!=v][:15]
                    row['exact_prior_stats']=True
                unchanged();row['host_seconds']=time.monotonic()-start;save(path/'accepted.json',row)
                with lock:state['completed'].append(row);status()
            except BaseException as e:
                stop.set();save(path/'failed.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
            finally:
                if proc and proc.poll() is None:
                    proc.terminate()
                    try:proc.wait(timeout=20)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()
                with lock:state['active'].pop(job['name'],None);status()
        def phase(name,jobs):
            status(name)
            with futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(run,jobs))
        phase('exact_controls',[dict(name='exact_'+a,arm=a,exact=True) for a in ('cache','private_spm','base')])
        phase('directed',[
            dict(name='init',smoke=True,init=True),dict(name='final',smoke=True,final=True),
            dict(name='both',smoke=True,init=True,final=True),
            dict(name='both_one_entry',smoke=True,init=True,final=True,template='half_one_entry'),
            dict(name='both_sync',smoke=True,init=True,final=True,template='half_sync'),
            dict(name='both_serial',smoke=True,init=True,final=True,template='half_serial')])
        save(OUT/'gates_passed.json',dict(time=time.time(),completed=[r['name'] for r in state['completed']]))
        for t in (1024,4096):
            phase(f'factorial_t{t}',[
                dict(name=f't{t}_cache',t=t,arm='cache'),dict(name=f't{t}_private_spm',t=t,arm='private_spm'),
                dict(name=f't{t}_base',t=t),dict(name=f't{t}_init',t=t,init=True),
                dict(name=f't{t}_final',t=t,final=True),dict(name=f't{t}_both',t=t,init=True,final=True)])
        status('completed_requires_analysis')
    except BaseException as e:state['error']=repr(e);status('failed');raise
if __name__=='__main__':main()
