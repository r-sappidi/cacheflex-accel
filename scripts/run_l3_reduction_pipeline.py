"""Fail-closed incremental near-L3 experiment: pipeline then compact payload."""
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
BASE=ROOT/'results/cacheflex_l3_spm/pipeline_20260928'
BUILD=BASE/os.environ.get('CACHEFLEX_L3_PIPELINE_BUILD','build_v2')
GUEST=BASE/os.environ.get('CACHEFLEX_L3_PIPELINE_GUEST','guest_v2')
OUT=BASE/os.environ.get('CACHEFLEX_L3_PIPELINE_EXPERIMENT','experiment_v2')
PREVIOUS=ROOT/'results/cacheflex_l3_spm/reduction_20260928'
RETRY=PREVIOUS/'retry_v2'
OLD=PREVIOUS/'experiment_v3'

def source_for(job):
    if job.get('smoke'):return RETRY/'runs'/job.get('template','smoke_async')
    t=job.get('t',1024);arm=job.get('arm','l3_async')
    p=RETRY/'runs'/f'isolated_t{t}_{arm}'
    return p if (p/'accepted.json').exists() else OLD/'runs'/f't{t}_{arm}'

def audit(source,path,job):
    def fields(p):return dict(leaves(json.loads((p/'config.json').read_text().replace(str(p),'<OUT>'))))
    a,b=fields(source),fields(path)
    delta={k:[a.get(k),b.get(k)] for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}
    allowed=(r'system.ruby.l2_cntrl\d+.l3_reduction_(pipeline|interval|fp16|convert)|'
        r'system.cpu\[\d+\].workload\[0\].(cmd\[\d+\]|executable)|'
        r'system.cpu\[\d+\].(cacheFlexIndependentReduction|reduction_port\.(role|peer|is_source))|'
        r'system.ruby.l0_cntrl\d+.sequencer.in_ports.peer\[\d+\]')
    assert not {k:v for k,v in delta.items() if not re.fullmatch(allowed,k)},delta
    for i in range(8):
        p=f'system.ruby.l2_cntrl{i}.';q=f'system.ruby.l0_cntrl{i}.sequencer.'
        assert b[p+'l3_reduction_pipeline']==job.get('pipeline',False)
        assert b[p+'l3_reduction_fp16']==job.get('half',False)
        assert b[p+'l3_reduction_interval']==4 and b[p+'l3_reduction_convert']==6
        assert b[p+'l3_reduction_entries']==(1 if job.get('template')=='smoke_one_entry_slow' else 4)
        assert b[f'system.cpu[{i}].cacheFlexIndependentReduction']
        assert b[p+'L2cache.size']==524288 and not b[p+'L2cache.spm_separate_backing']
        assert b[p+'l3_spm_latency']==35 and b[p+'l3_shared_service_port']
        # Append a retry endpoint only when the previous build lacked it.
        peers=[k for k in a if re.fullmatch(re.escape(q)+r'in_ports.peer\[\d+\]',k)]
        assert all(a[k]==b[k] for k in peers)
    return delta

def report(state):
    rows={r['name']:r for r in state['completed']}
    lines=['# Bounded bank pipeline / typed FP16 contributions','',f"Stage: {state['stage']}",'',
        '| Case | ROI us | Prior/source us | Source / new | Bank max active | Wire bytes |',
        '|---|---:|---:|---:|---:|---:|']
    for r in state['completed']:
        lines.append(f"| {r['name']} | {r['roi_us']:.4f} | {r['prior_us']:.4f} | {r['prior_us']/r['roi_us']:.6f}x | {r['max_active']:.0f} | {r['wire_bytes']:.0f} |")
    lines+=['','## Same-ELF factorial comparisons','',
        '| T | Variant | ROI us | Speedup vs serialized FP32 | vs cache | vs private SPM |',
        '|---|---|---:|---:|---:|---:|']
    for t in (1024,4096):
        base=rows.get(f't{t}_base');cache=rows.get(f't{t}_cache');private=rows.get(f't{t}_private_spm')
        for variant in ('base','pipeline','half','both'):
            r=rows.get(f't{t}_{variant}')
            if r and base and cache and private:
                v=r['roi_us'];lines.append(f"| {t} | {variant} | {v:.4f} | {base['roi_us']/v:.6f}x | {cache['roi_us']/v:.6f}x | {private['roi_us']/v:.6f}x |")
    lines+=['','Only accepted full-arithmetic, exact-output, resource and configuration checks.',
        'No extra accumulator SRAM. Four finite records at source/flight/bank; unchanged SRAM/network bandwidth.',
        'Pipelined 16-lane arithmetic and added 16-lane conversion are explicit hardware assumptions, not free acceleration.',
        'Core capture remains256B; typed FP16 bank-bound payload128B plus16B header.',
        'Old-ELF pipeline cases isolate hardware alone; same-new-ELF factorial isolates guest-layout changes.',
        'Retain strongest original private controls117.3456us(T1024),1167.5820us(T4096) when assessing net benefit.',
        'A speedup against reduction alone is not a speedup against standard CacheFlex.']
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
            p=BUILD/'status.json'
            if p.exists():
                build=json.loads(p.read_text());assert build['stage']!='failed',build
                if build['stage']=='built_requires_validation':break
            time.sleep(5)
        assert sha(BUILD/'gem5.opt')==build['sha256']
        shutil.copytree(RETRY/'configs',OUT/'configs',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        shutil.copy2(ROOT/'scripts/l3_reduction_pipeline_se.py',OUT/'configs/deprecated/example/l3_reduction_pipeline_se.py')
        frozen=json.loads((BUILD/'frozen.json').read_text())|json.loads((GUEST/'frozen.json').read_text())
        frozen.update(json.loads((PREVIOUS/'guest_v1/frozen.json').read_text()))
        for p in [Path(__file__),*list((OUT/'configs').rglob('*.py'))]:frozen[str(p)]=sha(p)
        # Freeze all imported analysis helpers used for acceptance as well.
        for module in (finite,):frozen[module.__file__]=sha(Path(module.__file__))
        for name in ('reference','run_access_timing','test_micro_timing','fa_headroom_profiles','export_safe_elf','build_access_timing'):
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
                # Source provenance is retained independently, not overwritten.
                save(path/'prior_files.json',{str(source/n):sha(source/n) for n in ('accepted.json','command.json','stats.txt','config.json')})
                cmd=json.loads((source/'command.json').read_text())
                cmd[0]=str(BUILD/'gem5.opt');cmd[1]='--outdir='+str(path)
                cmd[2]=str(OUT/'configs/deprecated/example/l3_reduction_pipeline_se.py')
                args=cmd[cmd.index('--options')+1].split()
                if not job.get('original'):
                    guest=GUEST/(job.get('smoke_guest','smoke') if job.get('smoke') else 'guest.export-safe')
                    cmd[cmd.index('--cmd')+1]=str(guest)
                    if not job.get('smoke') and job.get('half'):args[-1]='4'
                    cmd[cmd.index('--options')+1]=' '.join(args)
                    for i,x in enumerate(cmd):
                        if x.startswith('system.cpu[0].workload[0].cmd='):
                            cmd[i]=f"system.cpu[0].workload[0].cmd={['kv-fa']+args!r}"
                if '--cacheflex-reduction-retry-isolation' not in cmd:cmd+=['--cacheflex-reduction-retry-isolation']
                if job.get('pipeline'):cmd+=['--cacheflex-l3-reduction-pipeline']
                if job.get('half'):cmd+=['--cacheflex-l3-reduction-fp16']
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
                log=(path/'run.log').read_text();smoke=job.get('smoke',False)
                assert proc.returncode==0 and ('L3_REDUCTION_SMOKE PASS' if smoke else 'KV_ROI_END') in log,log[-5000:]
                assert 'because exiting with last active thread context' in log and 'Simulated exit code not 0!' not in log
                epochs=stats(path);old=stats(source);s=epochs[0]
                total=lambda suffix:sum(v for k,v in s.items() if k.endswith(suffix))
                reduction=smoke or job.get('arm','l3_async')=='l3_async'
                t=job.get('t',1024)
                expected=1040 if smoke else (t*math.ceil(t/int(args[3])) if reduction else 0)
                half_count=520 if job.get('smoke_guest')=='smoke_mixed' else (expected if job.get('half') else 0)
                depth=1 if job.get('template')=='smoke_one_entry_slow' else 4
                assert total('.reductionAccepted')==total('.reductionCompleted')==expected
                assert total('.l3ReductionStarts')==total('.l3ReductionFinishes')==expected
                assert total('.l3ReductionHalfCommands')==half_count
                assert total('.reductionTransportBytes')==272*(expected-half_count)+144*half_count
                max_active=max(v for k,v in s.items() if k.endswith('.l3ReductionMaxActive'))
                assert max_active<=(depth if job.get('pipeline') else 1)
                for ep in epochs:
                    check_global_guard(ep)
                    for i in range(8):
                        for q in ('Source','Sink'):assert ep[f'system.ruby.l0_cntrl{i}.sequencer.reductionMax{q}']<=depth
                row=dict(name=job['name'],job=job,source=str(source),roi_us=s['simTicks']/1e6,
                    prior_us=prior['roi_us'],commands=expected,half_commands=half_count,max_active=max_active,
                    wire_bytes=total('.reductionTransportBytes'),config_delta=audit(source,path,job))
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
                unchanged();row['host_seconds']=time.monotonic()-start
                save(path/'accepted.json',row)
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
        phase('old_elf_exact_controls',[dict(name='exact_'+a,arm=a,original=True,exact=True)
            for a in ('cache','private_spm','l3_async')])
        phase('pipeline_directed',[dict(name='pipeline_'+s,smoke=True,template=s,original=True,pipeline=True)
            for s in ('smoke_async','smoke_sync','smoke_one_entry_slow')])
        phase('pipeline_old_elf_fa',[dict(name=f'old_t{t}_pipeline',t=t,original=True,pipeline=True) for t in (1024,4096)])
        phase('typed_directed',[
            dict(name='half_serial',smoke=True,half=True,smoke_guest='smoke_half'),
            dict(name='half_pipeline',smoke=True,half=True,pipeline=True,smoke_guest='smoke_half'),
            dict(name='mixed_pipeline',smoke=True,half=True,pipeline=True,smoke_guest='smoke_mixed'),
            dict(name='half_one_entry',smoke=True,half=True,pipeline=True,smoke_guest='smoke_half',template='smoke_one_entry_slow'),
            dict(name='half_sync',smoke=True,half=True,pipeline=True,smoke_guest='smoke_half',template='smoke_sync')])
        for t in (1024,4096):
            phase(f'factorial_t{t}',[
                dict(name=f't{t}_cache',t=t,arm='cache'),dict(name=f't{t}_private_spm',t=t,arm='private_spm'),
                dict(name=f't{t}_base',t=t),dict(name=f't{t}_pipeline',t=t,pipeline=True),
                dict(name=f't{t}_half',t=t,half=True),dict(name=f't{t}_both',t=t,half=True,pipeline=True)])
        status('completed_requires_analysis')
    except BaseException as e:state['error']=repr(e);status('failed');raise
if __name__=='__main__':main()
