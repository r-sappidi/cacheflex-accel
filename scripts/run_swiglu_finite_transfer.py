"""Revalidate the accepted SwiGLU pair on the current finite FA platform.

AI-06/07/08: older accepted speedups are not transferable across models.
Keeps guests, tensors, arithmetic, tiling and five-call first/warm policy.
Waits for the placement queue to release its two simulation slots.
"""
import concurrent.futures as cf
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'benchmarks/ai_workloads'))
from build_access_timing import save,sha
from run_access_timing import stats
from test_micro_timing import check_global_guard
from fa_headroom_profiles import leaves
from run_model_swiglu import expected_copies
from sweep_mlp_study import epochs
from audit_mlp_math import audit as math_audit

OLD=ROOT/'results/ai_workloads/swiglu_model_pilot_v4_20260921'
FINITE=ROOT/'results/cacheflex_flash_attention/finite_model_20260926'
OUT=ROOT/'results/ai_workloads/swiglu_finite_transfer_20260927'
WAIT=ROOT/'results/cacheflex_flash_attention/spm_set_spread_20260927/experiment/status.json'
ENV=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib',
         OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4')


def compare_config(control,target,args,guest,inp,ways):
    def flat(p):return dict(leaves(json.loads((p/'config.json').read_text().replace(str(p),'<OUT>'))))
    a,b=flat(control),flat(target)
    delta={k:[a.get(k),b.get(k)] for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}
    allowed=r'system.readfile|system.cpu\[\d+\].workload\[0\].(cmd\[\d+\]|executable)|system.ruby.l1_cntrl\d+.cache.num_spm_ways'
    assert all(re.fullmatch(allowed,k) for k in delta),delta
    config=json.loads((target/'config.json').read_text())
    assert config['system']['cpu'][0]['workload'][0]['cmd']==args
    assert config['system']['cpu'][0]['workload'][0]['executable']==guest
    assert config['system']['readfile']==inp
    for i in range(8):assert config['system']['ruby'][f'l1_cntrl{i}']['cache']['num_spm_ways']==ways
    return delta


def main():
    OUT.mkdir();(OUT/'runs').mkdir()
    state=dict(pid=os.getpid(),stage='preparing',active={},completed=[])
    lock=threading.RLock();stop=threading.Event()
    def status(stage=None):
        with lock:
            if stage:state['stage']=stage
            state['updated']=time.time();save(OUT/'status.json',state)
    rows=json.loads((OLD/'status.json').read_text())['completed']
    jobs=[r for r in rows if r['arch'] in ('cache','spm') and
          (r['stage']=='gate' or r['case'][0] in (512,2048))]
    assert len(jobs)==8
    control=FINITE/'experiment_v5/runs/finite_kv0_spm0'
    template=json.loads((control/'command.json').read_text())
    assert Path(template[0])==FINITE/'build_v3/gem5.opt'
    frozen={str(p):sha(p) for p in [Path(__file__),control/'command.json',control/'config.json',Path(template[0])]}
    for p in Path(template[2]).parents[2].rglob('*.py'):frozen[str(p)]=sha(p)
    for r in jobs:
        p=Path(r['path']);cmd=json.loads((p/'command.json').read_text())
        inp=Path(next(x.split('=',1)[1].strip("'") for x in cmd if x.startswith('system.readfile=')))
        for dep in [p/'command.json',p/'accepted.json',p/'config.json',p/'output.bin',
                    Path(cmd[cmd.index('--cmd')+1]),inp,inp.with_suffix('.stages.npz')]:frozen[str(dep)]=sha(dep)
    for name in ('run_model_swiglu.py','sweep_mlp_study.py','audit_mlp_math.py'):
        p=ROOT/'benchmarks/ai_workloads'/name;frozen[str(p)]=sha(p)
    save(OUT/'inputs.json',frozen)
    save(OUT/'plan.json',dict(ledger=['AI-06','AI-07','AI-08','MOD-03'],
        scope='Fixed-kernel model transfer, NOT tuning or accumulator optimization',
        cases=[dict(arch=r['arch'],case=r['case'],source=r['path']) for r in jobs],
        primary='full operator first-call and median of final three calls, unchanged five-call policy',
        resources='exact current finite FA platform, with original SPM way eligibility four',
        model_change='includes bounded pools/queues/shadows and newer timing, one SPM load port; not a single-knob causal experiment',
        max_simulators=2,wait_for=str(WAIT)))
    def unchanged():
        for p,h in frozen.items():assert sha(p)==h,('changed input',p)
    def run(old):
        source=Path(old['path']);name=source.name;path=OUT/'runs'/name;path.mkdir()
        proc=None;start=time.monotonic()
        try:
            assert not stop.is_set();unchanged()
            oldcmd=json.loads((source/'command.json').read_text())
            guest=oldcmd[oldcmd.index('--cmd')+1];options=oldcmd[oldcmd.index('--options')+1]
            args=['model-swiglu']+options.split()
            inp=next(x.split('=',1)[1].strip("'") for x in oldcmd if x.startswith('system.readfile='))
            ways=4 if old['arch']=='spm' else 0
            cmd=list(template);cmd[1]='--outdir='+str(path)
            cmd[cmd.index('--cmd')+1]=guest;cmd[cmd.index('--options')+1]=options
            for i,x in enumerate(cmd):
                if x.startswith('system.cpu[0].workload[0].cmd='):cmd[i]='system.cpu[0].workload[0].cmd='+repr(args)
                if x.startswith('--l1d_num_spm_ways='):cmd[i]='--l1d_num_spm_ways='+str(ways)
            cmd+=['-P',f'system.readfile={inp!r}']
            save(path/'command.json',cmd);save(path/'case.json',dict(source=str(source),case=old['case']))
            with (path/'run.log').open('w') as log:
                proc=subprocess.Popen(cmd,cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT)
                with lock:state['active'][name]=dict(pid=proc.pid,case=old['case']);status()
                while proc.poll() is None:
                    assert not stop.is_set(),'peer failure'
                    assert time.monotonic()-start<96*3600,'96h timeout'
                    assert shutil.disk_usage(OUT).free>50*2**30,'disk guard'
                    save(path/'progress.json',dict(pid=proc.pid,host_seconds=time.monotonic()-start))
                    try:proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:pass
            text=(path/'run.log').read_text()
            assert proc.returncode==0 and not re.search(r'panic:|fatal:|Simulated exit code not 0|\bFAIL\b',text),text[-3000:]
            assert 'because exiting with last active thread context' in text
            changes=compare_config(control,path,args,guest,inp,ways)
            checks=[dict(re.findall(r'(\w+)=([^\s]+)',l)) for l in text.splitlines() if l.startswith('MLP_RESULT')]
            assert len(checks)==len(old['checks'])
            assert all(c['nonfinite']=='0' and c['errors']=='0' for c in checks)
            assert [c['hash'] for c in checks]==[c['hash'] for c in old['checks']]
            t,d,f,mp,kc,g=old['case']
            data=np.fromfile(path/'output.bin',dtype='<f2');assert data.size==t*d and np.isfinite(data).all()
            h=1469598103934665603
            for x in data.view(np.uint16):h=((h^int(x))*1099511628211)&((1<<64)-1)
            assert f'{h:016x}'==checks[-1]['hash']
            ref=np.load(Path(inp).with_suffix('.stages.npz'))['y'].reshape(-1)
            error=np.abs(data.astype(float)-ref.astype(float))
            assert np.all(error<=2e-6+.01*np.abs(ref.astype(float)))
            accuracy=math_audit(path,Path(inp),t,d,f);assert accuracy['passed']
            assert sha(path/'output.bin')==sha(source/'output.bin')
            for s in stats(path):
                check_global_guard(s)
                for i in range(8):
                    for domain in ('regular','spm'):
                        for kind,cap in [('Load',32),('Store',48)]:
                            assert s[f'system.cpu{i}.{domain}Lsq0.boundedMax{kind}Occupancy']<=cap
            timing=epochs(path,len(checks));copies=expected_copies(t,d,f,mp) if ways else 0
            assert all(e['copies_issued']==e['copies_completed']==copies for e in timing)
            samples=[e['us'] for e in timing[2:]]
            row=dict(arch=old['arch'],case=old['case'],path=str(path),epochs=timing,
                first_us=timing[0]['us'],warm_us=statistics.median(samples),
                warm_spread=(max(samples)-min(samples))/statistics.median(samples),
                checks=checks,math=accuracy,full_reference_max_abs=float(error.max()),
                config_delta=changes,output_sha256=sha(path/'output.bin'),host_seconds=time.monotonic()-start)
            unchanged();save(path/'accepted.json',row)
            with lock:state['completed'].append(row)
        except BaseException as e:
            stop.set();save(path/'failed.json',dict(error=repr(e)))
            with lock:state['error']=name+': '+repr(e)
            raise
        finally:
            if proc and proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
            with lock:state['active'].pop(name,None);status()
    try:
        status('queued_behind_fa_placement')
        while True:
            if WAIT.exists():
                s=json.loads(WAIT.read_text())
                if s['stage'] in ('completed_requires_analysis','failed') and not s['active']:break
            time.sleep(10)
        unchanged()
        for stage,t in [('gates',24),('t512_transfer',512),('t2048_transfer',2048)]:
            status(stage)
            with cf.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(run,[r for r in jobs if r['case'][0]==t]))
        status('completed_requires_analysis')
    except BaseException as e:state.setdefault('error',repr(e));status('failed');raise

if __name__=='__main__':main()
