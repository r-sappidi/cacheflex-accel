"""Matched reduction-only retry ablation. Preserve old queue, guests and capacities."""
import concurrent.futures as futures
import json,os,re,shutil,subprocess,threading,time,traceback
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
BUILD=BASE/os.environ.get('CACHEFLEX_L3_RETRY_BUILD','build_v3')
OLD=BASE/'experiment_v3'
OUT=BASE/os.environ.get('CACHEFLEX_L3_RETRY_EXPERIMENT','retry_v2')
REUSE=BASE/'retry_v1'

def audit(old,new,isolated):
    def fields(p):return dict(leaves(json.loads((p/'config.json').read_text().replace(str(p),'<OUT>'))))
    a,b=fields(old),fields(new)
    delta={k:[a.get(k),b.get(k)] for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}
    allowed=r'system.cpu\[\d+\].(cacheFlexIndependentReduction|reduction_port\.(role|peer|is_source))|system.ruby.l0_cntrl\d+.sequencer.in_ports.peer\[\d+\]'
    assert not {k:v for k,v in delta.items() if not re.fullmatch(allowed,k)},delta
    for i in range(8):
        assert b[f'system.cpu[{i}].cacheFlexIndependentReduction']==isolated
        q=f'system.ruby.l0_cntrl{i}.sequencer.'
        old_ports=[(k,v) for k,v in a.items() if re.fullmatch(re.escape(q)+r'in_ports.peer\[\d+\]',k)]
        new_ports=[(k,v) for k,v in b.items() if re.fullmatch(re.escape(q)+r'in_ports.peer\[\d+\]',k)]
        assert all(b[k]==v for k,v in old_ports),'original port remapped'
        assert len(new_ports)==len(old_ports)+int(isolated),(old_ports,new_ports)
        if isolated:
            assert any(v==f'system.cpu{i}.reduction_port' for k,v in new_ports),new_ports
        # Capacities, guest, flags, clocks, port budgets and all timing settings
        # are covered by the exhaustive delta check, not just a selected list.
    return delta

def main():
    OUT.mkdir(parents=True);(OUT/'runs').mkdir()
    state=dict(pid=os.getpid(),stage='waiting_for_build',active={},completed=[])
    lock=threading.RLock();stop=threading.Event()
    def status(stage=None):
        with lock:
            if stage:state['stage']=stage
            state['updated']=time.time();save(OUT/'status.json',state)
            lines=['# Reduction-only retry isolation','',f"Stage: {state['stage']}",'',
                '| Case | ROI us | Prior us | Speedup | Isolated rejects | Coherent accepted while blocked |',
                '|---|---:|---:|---:|---:|---:|']
            for r in state['completed']:
                lines.append(f"| {r['name']} | {r['roi_us']:.4f} | {r['prior_us']:.4f} | {r['speedup']:.6f}x | {r['isolated_rejects']:.0f} | {r['coherent_progress']:.0f} |")
            lines+=['','Only accepted arithmetic/config/resource-checked cases.',
                'Old/new same guest; only reduction admission/retry interface changes.',
                'Queue sizes, SRAM, service latency and issue/network bandwidth unchanged.',
                'Retry counts are not additive removable runtime cycles.']
            (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    status()
    try:
        while True:
            p=BUILD/'status.json'
            if p.exists():
                b=json.loads(p.read_text());assert b['stage']!='failed',b
                if b['stage']=='built_requires_validation':break
            time.sleep(5)
        shutil.copytree(OLD/'configs',OUT/'configs',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        shutil.copy2(ROOT/'scripts/l3_reduction_retry_se.py',OUT/'configs/deprecated/example/l3_reduction_retry_se.py')
        frozen=json.loads((BUILD/'frozen.json').read_text())
        old_inputs=json.loads((OLD/'inputs.json').read_text())
        # Preserve all previous guest/reference/config/helper inputs too.
        frozen.update(old_inputs)
        for p in [Path(__file__),*list((OUT/'configs').rglob('*.py'))]:frozen[str(p)]=sha(p)
        refs={}
        for t,bc in ((1024,128),(4096,512)):
            p=next((ROOT/'results/cacheflex_flash_attention/finite_model_20260926').glob(f'**/references/t{t}_bc{bc}_parts1.npy'))
            assert sha(p)==json.loads(p.with_suffix('.json').read_text())['sha256']
            refs[t]=np.load(p);frozen[str(p)]=sha(p)
        save(OUT/'inputs.json',frozen)
        def unchanged():
            for p,h in frozen.items():assert sha(p)==h,('modified input',p)
        def run(job):
            name,source_name,isolated,exact=job
            path=OUT/'runs'/name;path.mkdir();source=OLD/'runs'/source_name
            proc=None;start=time.monotonic()
            try:
                assert not stop.is_set()
                previous=REUSE/'runs'/name
                if exact and (previous/'accepted.json').exists():
                    # Runner-only accounting repair: reuse simulations only
                    # after verifying every prior input (including preserved
                    # runner source) and independently rechecking outputs/stats.
                    inputs=json.loads((REUSE/'inputs.json').read_text())
                    for p,h in inputs.items():
                        actual=REUSE/'runner_at_failure.py' if p==str(Path(__file__)) else Path(p)
                        assert sha(actual)==h,('reuse input changed',p)
                    row=json.loads((previous/'accepted.json').read_text())
                    assert row['exact_prior_stats'] and row['source']==str(source)
                    audit(source,previous,isolated)
                    a,b=stats(source),stats(previous);assert len(a)==len(b)
                    for old,new in zip(a,b):
                        assert all(new.get(k)==v for k,v in old.items())
                        check_global_guard(new)
                    t=int(source_name.split('_')[0][1:])
                    export_hash(previous/'output.fp16',(previous/'run.log').read_text())
                    validate(previous/'output.fp16',refs[t],t,dense=t==1024)
                    assert sha(previous/'output.fp16')==sha(source/'output.fp16')
                    row['reused_from']=str(previous)
                    save(path/'reused_files.json',{str(p):sha(p) for p in previous.iterdir() if p.is_file()})
                    save(path/'accepted.json',row)
                    with lock:state['completed'].append(row);status()
                    return
                while not (source/'accepted.json').exists():
                    assert not (source/'failed.json').exists(),('prior failed',source)
                    assert not stop.is_set();assert time.monotonic()-start<14400
                    time.sleep(5)
                prior=json.loads((source/'accepted.json').read_text())
                cmd=json.loads((source/'command.json').read_text())
                cmd[0]=str(BUILD/'gem5.opt');cmd[1]='--outdir='+str(path)
                cmd[2]=str(OUT/'configs/deprecated/example/l3_reduction_retry_se.py')
                if isolated:cmd+=['--cacheflex-reduction-retry-isolation']
                save(path/'command.json',cmd);unchanged()
                with (path/'run.log').open('w') as f:
                    proc=subprocess.Popen(cmd,cwd=ROOT,env=finite.ENV,stdout=f,stderr=subprocess.STDOUT)
                    with lock:state['active'][name]=dict(pid=proc.pid,start=time.time());status()
                    while proc.poll() is None:
                        assert not stop.is_set(),'peer failed'
                        assert time.monotonic()-start<14400
                        assert shutil.disk_usage(OUT).free>50*2**30
                        try:proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:status()
                log=(path/'run.log').read_text();smoke=source_name.startswith('smoke_')
                assert proc.returncode==0 and ('L3_REDUCTION_SMOKE PASS' if smoke else 'KV_ROI_END') in log,log[-5000:]
                assert 'because exiting with last active thread context' in log and 'Simulated exit code not 0!' not in log
                row=dict(name=name,source=str(source),config_delta=audit(source,path,isolated))
                epochs=stats(path);s=epochs[0];old_epochs=stats(source)
                for ep in epochs:check_global_guard(ep)
                total=lambda suffix:sum(v for k,v in s.items() if k.endswith(suffix))
                row.update(roi_us=s['simTicks']/1e6,prior_us=prior['roi_us'],
                    isolated_rejects=total('.reductionIsolatedRejects'),
                    coherent_progress=total('.coherentSentWhileReductionBlocked'))
                row['speedup']=row['prior_us']/row['roi_us']
                assert total('.reductionAccepted')==total('.reductionCompleted')==prior['commands']
                depth=prior['job'].get('depth',4)
                for ep in epochs:
                    for i in range(8):
                        for q in ('Source','Sink'):assert ep[f'system.ruby.l0_cntrl{i}.sequencer.reductionMax{q}']<=depth
                if isolated:
                    # OrderingRetries includes BOTH command-side older-access
                    # rejection and ordinary accesses held behind posted updates.
                    # SourceFull is command-only; all command rejects are isolated.
                    assert total('.reductionSourceFull')<=row['isolated_rejects']<=total('.reductionSourceFull')+total('.reductionOrderingRetries')
                    if prior['commands']:assert row['isolated_rejects']>0,'backpressure not exercised'
                else:assert row['isolated_rejects']==row['coherent_progress']==0
                if not smoke:
                    t=int(source_name.split('_')[0][1:]);export_hash(path/'output.fp16',log)
                    row['arithmetic']=validate(path/'output.fp16',refs[t],t,dense=t==1024)
                    assert sha(path/'output.fp16')==sha(source/'output.fp16')
                    assert total('.m_spmcpRequestsCompleted')==sum(v for k,v in old_epochs[0].items() if k.endswith('.m_spmcpRequestsCompleted'))
                    assert total('.m_spmcpRequestsIssued')==total('.m_spmcpRequestsCompleted')
                if exact:
                    assert len(epochs)==len(old_epochs)
                    for a,b in zip(old_epochs,epochs):
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
                with lock:state['active'].pop(name,None);status()
        def phase(name,jobs):
            status(name)
            with futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(run,jobs))
        phase('default_exact',[('off_'+n,n,False,True) for n in ('t1024_cache','t1024_private_spm','t1024_l3_async')])
        phase('isolated_directed',[(n,n,True,False) for n in ('smoke_async','smoke_sync','smoke_one_entry_slow')])
        phase('no_reduction_exact',[('isolated_'+n,n,True,True) for n in ('t1024_cache','t1024_private_spm')])
        for t in (1024,4096):
            phase(f'fa_{t}',[(f'isolated_t{t}_l3_{a}',f't{t}_l3_{a}',True,False) for a in ('async','sync')])
        status('completed_requires_analysis')
    except BaseException as e:state['error']=repr(e);status('failed');raise
if __name__=='__main__':main()
