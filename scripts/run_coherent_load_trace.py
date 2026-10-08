"""Fixed T4096 coherent-load diagnosis; observation-only, max two simulators."""
import concurrent.futures as cf
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import numpy as np
from build_access_timing import ROOT, sha, save
from run_access_timing import stats
from test_micro_timing import check_global_guard
sys.path.insert(0,str(ROOT/'benchmarks/cacheflex_flash_attention'))
from single_head_reference import check

BASE=ROOT/'results/cacheflex_flash_attention'
OUT=BASE/'coherent_load_path_20260923'; BUILD=OUT/os.environ.get('CACHEFLEX_LOAD_BUILD','build_v2'); EX=OUT/os.environ.get('CACHEFLEX_LOAD_EXPERIMENT','experiment_v4')
REF=BASE/'micro_resources_20260921/experiment_v6/preserved/t512_bc128.npy'
PCS={'spm':'0x4020c0,0x402120,0x402a58','cache':'0x402040,0x4020a0,0x4029d8'}
ENV=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib',
         OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
for k in list(ENV):
    if k.startswith('CACHEFLEX_AGGREGATE') or k in ('CACHEFLEX_TRACE_LINE','CACHEFLEX_LOAD_PCS'):ENV.pop(k)

def prior(v,t):return BASE/f'ifetch_edge_20260923/experiment/runs/{v}_t{t}_edge1'

def command(v,t,p,window=None,traced=False):
    cmd=json.loads((prior(v,t)/'command.json').read_text())
    config_index=next(i for i,x in enumerate(cmd) if x.endswith('/micro_timing_se.py'))
    # Some accepted reference runs also contain debug flags; discard their
    # host-side tracing options rather than assuming a fixed config index.
    cmd=[str(BUILD/'gem5.opt'),f'--outdir={p}',
         str(EX/'configs/deprecated/example/micro_timing_se.py')]+cmd[config_index+1:]
    if window:
        lo,hi=window
        cmd += [f'--abs-max-tick={hi+400000}']
        if traced:
            cmd[1:1]=['--debug-flags=CacheFlexDeps,CacheFlexLoads,CacheFlexTiming,ProtocolTrace',
                      f'--debug-start={lo}',f'--debug-end={hi}','--debug-file=trace.gz',
                      '--debug-ignore='+':'.join(f'system.cpu{i}' for i in range(1,7))]
    return cmd

def config_check(old,new):
    a=json.loads((old/'config.json').read_text().replace(str(old),'<OUT>'))
    b=json.loads((new/'config.json').read_text().replace(str(new),'<OUT>'))
    assert a==b,'Undeclared configuration change'

def main():
    EX.mkdir(); state=dict(pid=os.getpid(),stage='waiting_for_build',active={},completed=[])
    lock=threading.RLock();halt=threading.Event();manifest={}
    def status(stage=None):
        with lock:
            if stage:state['stage']=stage
            state['updated']=time.time();save(EX/'status.json',state)
    def frozen():
        for p,h in manifest.items():assert sha(p)==h,('Changed frozen input',p)
    status()
    def run(v,t,label,window=None,traced=False):
        frozen();p=EX/label/v;p.mkdir(parents=True)
        cmd=command(v,t,p,window,traced);save(p/'command.json',cmd)
        env=dict(ENV,CACHEFLEX_LOAD_PCS=PCS[v]);save(p/'trace_environment.json',{'CACHEFLEX_LOAD_PCS':PCS[v]})
        proc=None;begin=time.monotonic()
        try:
            with (p/'run.log').open('w') as log:
                proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                with lock:state['active'][f'{label}/{v}']=dict(pid=proc.pid,path=str(p));status()
                while proc.poll() is None:
                    assert not halt.is_set(),'Peer failed'
                    assert time.monotonic()-begin<3600,'One-hour limit'
                    assert shutil.disk_usage(EX).free>50*2**30,'Disk guard'
                    trace=p/'trace.gz'
                    assert not trace.exists() or trace.stat().st_size<2**30,'Trace limit'
                    try:proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:pass
            assert proc.returncode==0,(v,label,proc.returncode)
            config_check(prior(v,t),p);s=stats(p)
            if window:
                assert s[-1]['finalTick']==window[1]+400000
                assert 'simulate() limit reached' in (p/'run.log').read_text()
                numerical=None
            else:
                assert s==stats(prior(v,t)),'Observer-off changes non-host stats'
                assert sha(p/'output.fp16')==sha(prior(v,t)/'output.fp16')
                numerical=check(p/'output.fp16',np.load(REF))
                assert numerical['errors']==numerical['nonfinite']==0,numerical
                for block in s:check_global_guard(block)
            frozen()
            row=dict(path=str(p),backend=v,T=t,config_equal=True,arithmetic=numerical,
                     diagnostic_prefix=bool(window),host_seconds=time.monotonic()-begin,
                     full_run_reference=str(prior(v,t)/'validated.json'))
            save(p/('validated_prefix.json' if window else 'validated.json'),row)
            with lock:state['active'].pop(f'{label}/{v}');state['completed'].append(row);status()
        except BaseException as error:
            save(p/'failed.json',dict(error=repr(error)))
            halt.set()
            if proc and proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
            with lock:state['active'].pop(f'{label}/{v}',None);status()
            raise
    def pair(label,t=4096,window=None,traced=False):
        status(label)
        with cf.ThreadPoolExecutor(max_workers=2) as pool:
            fs=[pool.submit(run,v,t,label,window,traced) for v in ('cache','spm')]
            for f in fs:f.result()
    try:
        while not (BUILD/'frozen.json').exists():
            b=BUILD/'status.json'
            if b.exists():assert json.loads(b.read_text())['stage']!='failed','Build failed'
            time.sleep(5)
        shutil.copytree(BUILD/'configs',EX/'configs');shutil.copy2(__file__,EX/'runner_snapshot.py')
        manifest.update(json.loads((BUILD/'frozen.json').read_text()))
        paths=[Path(__file__).resolve(),ROOT/'scripts/run_access_timing.py',ROOT/'scripts/build_access_timing.py',
               ROOT/'scripts/test_micro_timing.py',ROOT/'scripts/analyze_coherent_load_trace.py',
               ROOT/'scripts/analyze_spm_consumers.py',ROOT/'benchmarks/cacheflex_flash_attention/single_head_reference.py',REF]
        paths+=list((EX/'configs').rglob('*.py'))
        for t in (512,4096):
            for v in ('cache','spm'):
                p=prior(v,t);cmd=json.loads((p/'command.json').read_text())
                paths += [p/n for n in ('validated.json','command.json','config.json','stats.txt','output.fp16')]
                paths.append(Path(cmd[cmd.index('--cmd')+1]))
        reuse=OUT/'experiment_v3'
        # Reuse completed, exact-equivalence prefixes after an analyzer-only
        # repair. Verify original frozen inputs (except explicitly repaired
        # analysis/coordinator source, whose old bytes were preserved).
        old_inputs=json.loads((reuse/'inputs.json').read_text())
        repaired={str(Path(__file__).resolve()),str(ROOT/'scripts/analyze_coherent_load_trace.py')}
        for p,h in old_inputs.items():
            if p not in repaired:assert sha(p)==h,('Changed prior input',p)
        assert json.loads((reuse/'early_equivalence.json').read_text())['all_finite_nonhost_stats_equal']
        for label in ('t512_equivalence','early_trace','early_observer_off'):
            for v in ('cache','spm'):
                p=reuse/label/v
                paths += [p/n for n in ('command.json','config.json','stats.txt')]
                if label=='t512_equivalence':
                    config_check(prior(v,512),p)
                    assert stats(p)==stats(prior(v,512))
                    assert sha(p/'output.fp16')==sha(prior(v,512)/'output.fp16')
                    paths += [p/'validated.json',p/'output.fp16']
                else:paths += [p/'validated_prefix.json']
                if label=='early_trace':paths.append(p/'trace.gz')
            (EX/label).symlink_to(reuse/label,target_is_directory=True)
        for v in ('cache','spm'):
            assert stats(reuse/'early_trace'/v)==stats(reuse/'early_observer_off'/v)
        save(EX/'reused_results.json',dict(source=str(reuse),reason='Analyzer-only readiness/retry handling repair',
                                         original_inputs_verified=True,all_nonhost_stats_equal=True))
        manifest.update({str(p):sha(p) for p in paths});save(EX/'inputs.json',manifest)
        save(EX/'gates_passed.json',dict(full_arithmetic=True,all_nonhost_stats_exact=True,reused_from=str(reuse)))
        # Predeclared windows; no selection by stall length. Prefixes, not full-ROI timings.
        for label,window in [('early',(2890000000,2914000000)),('middle',(3800000000,3824000000))]:
            if label!='early':
                pair(label+'_trace',window=window,traced=True)
                pair(label+'_observer_off',window=window)
            for v in ('cache','spm'):
                assert stats(EX/(label+'_trace')/v)==stats(EX/(label+'_observer_off')/v),'Trace perturbed timing'
            save(EX/f'{label}_equivalence.json',dict(all_finite_nonhost_stats_equal=True))
            status(label+'_analyzing')
            subprocess.run([sys.executable,str(ROOT/'scripts/analyze_coherent_load_trace.py'),
                            str(EX),label],cwd=ROOT,check=True)
        status('completed_validated_diagnostic')
    except BaseException as e:
        halt.set();state['error']=repr(e);status('failed');raise

if __name__=='__main__':main()
