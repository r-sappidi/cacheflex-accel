"""Warm confirmation, same-platform ACL audit, independent SwiGLU tuning.

Preserve frozen screen kernels. FP16 accumulation is explicit, not a production
FP32-accumulation claim. Two extra simulator slots alongside the FA queue.
"""
import concurrent.futures as cf
import configparser
import itertools
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import threading
import time
import numpy as np
from run_mapping_screen import ROOT,HERE,sha,save,expected_copies
from sweep_mlp_study import epochs
from audit_mlp_math import audit as math_audit
from validate_mlp_acl import validate as acl_validate

OLD=ROOT/'results/ai_workloads/mapping_screen_export_safe_v2_20260917'
PREVIOUS=ROOT/'results/ai_workloads/swiglu_confirmation_20260920'
OUT=ROOT/'results/ai_workloads/swiglu_confirmation_resumed_20260921'
ENV=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')

def config_audit(old,new):
    a=configparser.ConfigParser(interpolation=None,strict=False);a.read(old/'config.ini')
    b=configparser.ConfigParser(interpolation=None,strict=False);b.read(new/'config.ini')
    assert a.sections()==b.sections()
    changes=[]
    for s in a.sections():
        assert set(a[s])==set(b[s])
        for k,x in a[s].items():
            y=b[s][k]
            if x.replace(str(old),'<RUN>')==y.replace(str(new),'<RUN>'):continue
            assert (s=='system' and k=='readfile') or (s=='system.cpu0.workload' and k in ('cmd','executable')),(s,k,x,y)
            changes.append([s,k,x,y])
    return changes

def main():
    # Preserve the failed study; reuse only its independently accepted results.
    for name,digest in json.loads((PREVIOUS/'manifest.json').read_text()).items():
        path=Path(name)
        if path.resolve()==Path(__file__).resolve():
            path=PREVIOUS/'runner.before_gate_removal.py'
        assert sha(path)==digest,('prior dependency changed',name)
    prior_rows=json.loads((PREVIOUS/'status.json').read_text())['completed']
    for row in prior_rows:
        assert json.loads((Path(row['path'])/'accepted.json').read_text())==row
    assert {(r['arch'],r['case']['m']) for r in prior_rows if r['stage']=='confirmation'}=={('cache',64),('spm',64),('cache',128),('spm',128)}
    OUT.mkdir();(OUT/'runs').mkdir();(OUT/'inputs').mkdir();(OUT/'preserved').mkdir()
    save(OUT/'reused_confirmation.json',prior_rows)
    state=dict(pid=os.getpid(),completed=[]);lock=threading.Lock();stop=threading.Event()
    def status(stage):
        state.update(stage=stage,updated=time.time());save(OUT/'status.json',state);print(stage,flush=True)
    summary=json.loads((OLD/'summary.json').read_text())
    winners=[w for w in summary['winners'] if w['shape'][0]=='swiglu' and w['objective']=='warm_us']
    templates={};manifest={}
    for row in summary['rows']:
        if row['case']['family']=='swiglu':templates.setdefault((row['variant'],row['case']['combined']),Path(row['path']))
    for p in templates.values():
        cmd=json.loads((p/'command.json').read_text())
        for src in (Path(cmd[0]),Path(cmd[cmd.index('--cmd')+1])):
            manifest[str(src)]=sha(src);dst=OUT/'preserved'/src.relative_to(ROOT)
            if not dst.exists():dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    acl=ROOT/'results/ai_workloads/mlp_acl_20260909/runs_audited/acl'
    manifest[str(acl)]=sha(acl)
    for p in list((ROOT/'gem5/configs').rglob('*.py'))+[Path(__file__),HERE/'mlp_reference.py',HERE/'audit_mlp_math.py',HERE/'validate_mlp_acl.py']:
        manifest[str(p.resolve())]=sha(p)
    save(OUT/'manifest.json',manifest)
    save(OUT/'plan.json',dict(hardware='8core VL16,64KiB L1I/D,512KiB private L2,4MiB total LLC; same frozen repaired simulator',
      precision='FP16 operands and GEMM accumulation; custom fused SiLU-times-up rounds once, ACL materializes FP16 SiLU first. Common math error gate; no bit equality required vs ACL.',
      measurement='First invocation separate; discard first two invocations for warm median; at least five measured warm epochs. Output validation between calls is retained symmetrically.',
      confirmation='Reuse accepted T64/T128 confirmations. Spread/drift reported descriptively; no variability admission gate (user authorized removal).',
      audit='Same-platform pristine-ACL contender plus independent KC-aware and common float64 math checks. ACL legacy warm binary has one measured invocation; not a converged steady-state claim.',
      sweep=dict(KC=[64,128,256,512],M_partitions=[1,2,4,8],combined_gate_up=[0,1],T=[64,128],tile_rows=24),
      selection='Rank cache and SPM separately by warm latency, never maximize ratios. Confirm top2 each on two extra seeds.',
      unresolved=['FP32-accumulate production precision','8/16-row microkernels and alternate placement families','real-model dimensions and held-out shapes','1/2/4/8core scaling','ACL converged multi-epoch driver'],max_extra_simulators=2))
    def frozen():
        for p,h in manifest.items():assert sha(p)==h,('dependency changed',p)
    inputs={}
    def inp(t,kc,seed=20260909):
        key=t,kc,seed
        if key not in inputs:
            p=OUT/'inputs'/f't{t}_kc{kc}_seed{seed}.bin'
            subprocess.run(['python3',str(HERE/'mlp_reference.py'),str(t),'512','1536',str(p),str(kc),'0',str(seed)],env=ENV,check=True,stdout=subprocess.DEVNULL)
            inputs[key]=p
        return inputs[key]
    def run(arch,c,stage,total=7,seed=20260909,original=None):
        if stop.is_set():raise RuntimeError('peer failed')
        t=c['m'];input_path=inputs[t,c['kc'],seed]
        label=f'{stage}_{arch}_t{t}_kc{c["kc"]}_mp{c["mp"]}_g{c["combined"]}_n{total}_s{seed}'
        p=OUT/'runs'/label;p.mkdir();old=original or templates[('cache' if arch=='acl' else arch,c['combined'])]
        cmd=json.loads((old/'command.json').read_text())
        args=[t,512,1536,c['mp'],c['kc'],0,total,1,8]
        if arch=='acl':args=[t,512,1536,'warm'];cmd[cmd.index('--cmd')+1]=str(acl)
        cmd=[f'--outdir={p}' if x.startswith('--outdir=') else f"system.readfile='{input_path}'" if x.startswith('system.readfile=') else
             'system.cpu[0].workload[0].cmd='+repr(['operator-screen']+list(map(str,args))) if x.startswith('system.cpu[0].workload[0].cmd=') else x for x in cmd]
        cmd[cmd.index('--options')+1]=' '.join(map(str,args))
        save(p/'command.json',cmd);save(p/'case.json',c);frozen();start=time.monotonic();proc=None
        try:
            with (p/'run.log').open('w') as log:
                proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=ENV)
                while proc.poll() is None:
                    save(p/'progress.json',dict(pid=proc.pid,stage='simulating',host_elapsed_s=time.monotonic()-start,updated=time.time()))
                    if stop.is_set() or time.monotonic()-start>8*3600:raise RuntimeError('peer failed/timeout')
                    if shutil.disk_usage(OUT).free<50*1024**3:raise RuntimeError('disk guard')
                    if (p/'run.log').stat().st_size>128*1024**2:raise RuntimeError('log guard')
                    try:proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:pass
            assert proc.returncode==0,(label,proc.returncode)
            text=(p/'run.log').read_text(errors='replace');assert not re.search(r'panic:|fatal:|Simulated exit code not 0',text)
            changes=config_audit(old,p)
            if arch=='acl':
                numeric=acl_validate(p,input_path,t,512,1536)
                assert not re.search(r'\[.*(?:INFO|CORE).*\]',text),'ACL ROI logging'
                times=epochs(p,1);tail=times;digest=None
            else:
                checks=[dict(re.findall(r'(\w+)=([^\s]+)',l)) for l in text.splitlines() if l.startswith('MLP_RESULT')]
                assert len(checks)==total and all(x['errors']==x['nonfinite']=='0' for x in checks)
                assert len({x['hash'] for x in checks})==1
                data=np.fromfile(input_path,dtype='<f2');actual=np.fromfile(p/'output.bin',dtype='<f2');expected=data[-t*512:]
                assert actual.shape==expected.shape and np.isfinite(actual).all()
                assert (np.abs(actual.astype(float)-expected.astype(float))<=2e-6+.01*np.abs(expected.astype(float))).all()
                h=1469598103934665603
                for x in actual.view(np.uint16):h=((h^int(x))*1099511628211)&((1<<64)-1)
                digest=f'{h:016x}';assert digest==checks[-1]['hash']
                times=epochs(p,total);tail=times[2:]
                numeric=dict(KC_reference_passed=True,export_hash=digest)
                if original:
                    prior=json.loads((old/'accepted.json').read_text());assert digest==prior['checks'][-1]['hash']
                    # Final old epoch may speculatively overlap exit; compare first.
                    assert times[0]==prior['epochs'][0],('first epoch drift',label)
            copies=expected_copies(c) if arch=='spm' else 0
            assert all(x['copies_issued']==x['copies_completed']==copies for x in times)
            math=math_audit(p,input_path,t,512,1536);assert math['passed']
            values=[x['us'] for x in tail];median=statistics.median(values)
            spread=(max(values)-min(values))/median
            drift=abs(statistics.median(values[-3:])-statistics.median(values[:3]))/median
            row=dict(path=str(p),arch=arch,case=c,stage=stage,seed=seed,epochs=times,warm_us=median,warm_spread=spread,warm_drift=drift,warm_samples=len(values),numerical=numeric,math=math,config_changes=changes)
            frozen();save(p/'accepted.json',row)
            with lock:state['completed'].append(row);save(OUT/'status.json',state)
            save(p/'progress.json',dict(stage='accepted',host_elapsed_s=time.monotonic()-start))
            return row
        except BaseException as e:
            stop.set();save(p/'failed.json',dict(error=repr(e)))
            if proc and proc.poll() is None:proc.terminate();proc.wait(timeout=30)
            raise
    def batch(jobs):
        with cf.ThreadPoolExecutor(max_workers=2) as pool:
            fut=[pool.submit(run,**j) for j in jobs]
            return [x.result() for x in fut]
    try:
        save(OUT/'warm_confirmation_passed.json',dict(passed=True,reused_from=str(PREVIOUS),variability_gate_removed=True,note='Numerical/config/copy/hash gates unchanged; variability remains reported, not proof of convergence.'))
        status('same_platform_acl_and_precision_audit')
        for t in (64,128):inp(t,256)
        acl_rows=batch([dict(arch='acl',c=dict(family='swiglu',m=t,k=512,n=1536,groups=1,shared=0,team=8,mp=1,kc=256,combined=0),stage='acl_audit',total=1) for t in (64,128)])
        save(OUT/'audit_passed.json',dict(passed=True,precision='FP16 accumulation only; FP32 production audit remains',acl=acl_rows))
        status('independent_tuning')
        for t in (64,128):
            for kc in (64,128,256,512):inp(t,kc)
        cases=[dict(family='swiglu',m=t,k=512,n=1536,groups=1,shared=0,team=8,mp=mp,kc=kc,combined=g) for t,kc,mp,g in itertools.product((64,128),(64,128,256,512),(1,2,4,8),(0,1))]
        rows=batch([dict(arch=a,c=c,stage='tuning') for c in cases for a in ('cache','spm')])
        selected=[]
        for t,a in itertools.product((64,128),('cache','spm')):
            candidates=sorted([r for r in rows if r['arch']==a and r['case']['m']==t],key=lambda r:r['warm_us'])
            for r in candidates[:2]:
                selected.append(r)
        save(OUT/'finalists.json',selected);status('additional_seed_confirmation')
        for r in selected:
            for seed in (137,2026):inp(r['case']['m'],r['case']['kc'],seed)
        batch([dict(arch=r['arch'],c=r['case'],stage='heldout_seed',seed=seed) for r in selected for seed in (137,2026)])
        save(OUT/'summary.json',dict(finalists=selected,scope='Bounded best-to-best 24-row FP16-accumulate study; not a production-precision or full-model claim. See plan unresolved dimensions.'))
        status('bounded_tuning_complete_requires_review')
    except BaseException as e:state['error']=repr(e);status('failed');raise
if __name__=='__main__':main()
