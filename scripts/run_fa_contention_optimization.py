"""Capacity-preserving, interaction-aware first round; not a universal bound."""
import concurrent.futures as cf
import configparser
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
import run_single_diagnosis as sd
from single_head_reference import reference, check
from run_clean_controls import sha, save, OLD
from validate_attribution import read
from fa_export_patch import verify_dump
import numpy as np

ROOT=sd.ROOT
OUT=ROOT/'results/cacheflex_flash_attention/contention_optimization_v2_20260920'
# Network expansion is an explicit hardware-cost tradeoff, never called free.
VARIANTS={
 'control':dict(width=128,pf='stock',seq=16,retain=True),
 'no_retention':dict(width=128,pf='stock',seq=16,retain=False),
 'restrained_prefetch':dict(width=128,pf='restrained',seq=16,retain=True),
 'no_prefetch':dict(width=128,pf='off',seq=16,retain=True),
 'network256':dict(width=256,pf='stock',seq=16,retain=True),
 'network256_restrained':dict(width=256,pf='restrained',seq=16,retain=True),
 'admission8':dict(width=128,pf='stock',seq=8,retain=True),
 'network256_restrained_admission8':dict(width=256,pf='restrained',seq=8,retain=True),
 'network256_no_retention':dict(width=256,pf='stock',seq=16,retain=False),
}
ENV=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')

def audit(old,new,params,t,width,arch):
    a=configparser.ConfigParser(interpolation=None,strict=False);a.read(old/'config.ini')
    b=configparser.ConfigParser(interpolation=None,strict=False);b.read(new/'config.ini')
    assert a.sections()==b.sections()
    expected={sd.common.config_key(k):str(v).lower() for k,v in params.items()}
    changes=[];seen=set()
    for s in a.sections():
        assert set(a[s])==set(b[s]),s
        for k,x in a[s].items():
            y=b[s][k]
            if (s,k) in expected:
                assert y.lower()==expected[s,k],(s,k,y,expected[s,k]);seen.add((s,k))
            elif k=='cmd' and s.endswith('.workload'):
                assert y==x.replace("512",str(t)),(s,k,x,y)
            elif s.startswith('system.ruby.network') and k in ('ni_flit_size','width'):
                assert int(y)==(width//8 if int(x)==16 else width),(s,k,x,y)
            else:assert x.replace(str(old),'<OUT>')==y.replace(str(new),'<OUT>'),(s,k,x,y)
            if x!=y:changes.append([s,k,x,y])
    assert seen==set(expected),(seen,set(expected))
    return changes

def main():
    OUT.mkdir();(OUT/'runs').mkdir();(OUT/'preserved').mkdir();(OUT/'references').mkdir()
    state=dict(pid=os.getpid(),completed=[]);stop=threading.Event();lock=threading.Lock()
    def status(stage):
        state.update(stage=stage,updated=time.time());save(OUT/'status.json',state);print(stage,flush=True)
    status('preserving_and_preparing')
    inputs={}
    for arch in ('cache','spm'):
        for t in (512,16384):
            old=sd.prior(arch,t)
            cmd=json.loads((old/'command.json').read_text())
            for p in [Path(cmd[0]),Path(cmd[cmd.index('--cmd')+1])]+[old/n for n in ('accepted.json','command.json','config.ini','stats.txt')]:
                inputs[str(p)]=sha(p)
                dest=OUT/'preserved'/p.relative_to(ROOT)
                if not dest.exists():dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    for p in list((ROOT/'gem5/configs').rglob('*.py'))+[Path(__file__)]:inputs[str(p.resolve())]=sha(p)
    save(OUT/'inputs.json',inputs)
    save(OUT/'plan.json',dict(variants=VARIANTS,cores=8,VL=16,Hq=1,Hkv=1,d=64,L1_KiB=64,L2_private_KiB=64,LLC_total_MiB=4,
      cache_geometry='Br32/Bc128/G2',spm_geometry='Br16/Bc128/G4,V-only,two ways',
      forbidden='No separate SPM backing, extra cache capacity, DMA, multicast, or altered arithmetic',
      stages='T512 gates all candidates; T4096 unchanged-hardware-size screening; promote controls and top three SPM candidates at T16K with matched cache candidates',
      scope='T4096 screening is not assumed timing-equivalent to T16K. No fundamental-impossibility claim from a finite search.',
      success_marker='20% speedup as an operational substantial-gain marker, not a paper acceptance criterion',
      next_iteration='Use measured winners and phase/network counters to decide whether targeted traffic-class arbitration or staging throttle should be implemented.',max_simulators=4))
    refs={512:np.load(OLD/'references/t512_bc128.npy'),16384:np.load(OLD/'references/t16384_bc128.npy')}
    status('generating_t4096_reference')
    cached=ROOT/'results/cacheflex_flash_attention/contention_optimization_20260920/references/t4096_bc128.npy'
    refs[4096]=np.load(cached) if cached.exists() else reference(4096,128)
    np.save(OUT/'references/t4096_bc128.npy',refs[4096])
    def frozen():
        for p,h in inputs.items():assert sha(p)==h,('changed dependency',p)
    def run(arch,t,name):
        if stop.is_set():raise RuntimeError('peer failed before launch')
        v=VARIANTS[name];old=sd.prior(arch,512);p=OUT/'runs'/f'{arch}_t{t}_{name}';p.mkdir()
        original=configparser.ConfigParser(interpolation=None);original.read(old/'config.ini')
        cmd=json.loads((old/'command.json').read_text());params={}
        for i in range(8):
            params[f'system.ruby.l0_cntrl{i}.sequencer.max_outstanding_requests']=v['seq']
            params[f'system.ruby.l0_cntrl{i}.enable_prefetch']=v['pf']!='off'
            section=f'system.ruby.l0_cntrl{i}.prefetcher'
            params[section+'.num_streams']=2 if v['pf']=='restrained' else original[section]['num_streams']
            params[section+'.train_misses']=8 if v['pf']=='restrained' else original[section]['train_misses']
            params[f'system.ruby.l2_cntrl{i}.spm_retain_llc']=arch=='spm' and v['retain']
        clean=[];i=0
        while i<len(cmd):
            if cmd[i]=='-P' and cmd[i+1].split('=',1)[0] in params:i+=2;continue
            x=cmd[i]
            if x.startswith('--outdir='):x=f'--outdir={p}'
            elif x.startswith('--link-width-bits='):x=f"--link-width-bits={v['width']}"
            elif x.startswith('system.cpu[0].workload[0].cmd='):x=x.replace("'512'",f"'{t}'")
            clean.append(x);i+=1
        cmd=clean
        for k,val in params.items():cmd+=['-P',f'{k}={val}']
        save(p/'command.json',cmd);frozen();start=time.monotonic();proc=None
        try:
            with (p/'run.log').open('w') as log:
                proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=ENV)
                while proc.poll() is None:
                    save(p/'progress.json',dict(pid=proc.pid,stage='simulating',host_elapsed_s=time.monotonic()-start,updated=time.time()))
                    if stop.is_set():raise RuntimeError('peer failed')
                    if time.monotonic()-start>24*3600:raise RuntimeError('timeout')
                    if shutil.disk_usage(OUT).free<50*1024**3:raise RuntimeError('disk guard')
                    if (p/'run.log').stat().st_size>256*1024**2:raise RuntimeError('log guard')
                    try:proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:pass
            assert proc.returncode==0,(p,proc.returncode)
            changes=audit(old,p,params,t,v['width'],arch)
            output,counters,_=read(p,16)
            assert output['T']==str(t) and output['Hq']=='1'
            copies=(t//64)*t*2 if arch=='spm' else 0
            assert counters['m_spmcpRequestsIssued']==counters['m_spmcpRequestsCompleted']==copies
            row=dict(path=str(p),arch=arch,T=t,name=name,variant=v,output=output,counters=counters,config_changes=changes)
            verify_dump(row);row['validation']=check(p/'output.fp16',refs[t])
            if name=='control' and t in (512,16384):
                prior=json.loads((sd.prior(arch,t)/'accepted.json').read_text())
                assert output==prior['output']
                assert {k:x for k,x in counters.items() if k!='path'}=={k:x for k,x in prior['counters'].items() if k!='path'},'control drift'
            frozen();save(p/'accepted.json',row)
            with lock:
                state['completed'].append(dict(arch=arch,T=t,name=name,roi_us=counters['roi_us'],path=str(p)))
                save(OUT/'status.json',state)
            save(p/'progress.json',dict(stage='accepted',host_elapsed_s=time.monotonic()-start))
            return row
        except BaseException as e:
            stop.set();save(p/'failed.json',dict(error=repr(e)))
            if proc and proc.poll() is None:proc.terminate();proc.wait(timeout=30)
            raise
    def batch(cases):
        with cf.ThreadPoolExecutor(max_workers=4) as pool:
            tasks=[pool.submit(run,*x) for x in cases]
            for task in cf.as_completed(tasks):task.result()
    def report(t):
        rows=[r for r in state['completed'] if r['T']==t]
        baseline=min(r['roi_us'] for r in rows if r['arch']=='cache' and VARIANTS[r['name']]['width']==128)
        records=[]
        for r in rows:
            if r['arch']!='spm':continue
            width=VARIANTS[r['name']]['width']
            same=min(x['roi_us'] for x in rows if x['arch']=='cache' and VARIANTS[x['name']]['width']==width)
            records.append(dict(**r,speedup_vs_best_fixed_platform_cache=baseline/r['roi_us'],speedup_vs_best_same_width_cache=same/r['roi_us']))
        save(OUT/f'effects_t{t}.json',records)
        return sorted(records,key=lambda r:r['roi_us'])
    try:
        cases=[(a,n) for n in VARIANTS for a in ('cache','spm') if not(a=='cache' and 'no_retention' in n)]
        for t in (512,4096):
            status(f'gates_t{t}' if t==512 else 'screening_t4096')
            batch([(a,t,n) for a,n in cases]);report(t)
        ranked=report(4096);selected=['control']+[r['name'] for r in ranked[:3] if r['name']!='control']
        # Always include best fixed-width and upgraded-width cache choices.
        controls={min([r for r in state['completed'] if r['T']==4096 and r['arch']=='cache' and VARIANTS[r['name']]['width']==w],key=lambda r:r['roi_us'])['name'] for w in (128,256)}
        target=sorted(set([('spm',16384,n) for n in selected]+[('cache',16384,n) for n in controls|{'control'}]+[('cache',16384,n.replace('network256_no_retention','network256').replace('no_retention','control')) for n in selected]))
        save(OUT/'promoted.json',target);status('confirming_t16384');batch(target);report(16384)
        status('round_complete_requires_analysis_and_next_iteration')
    except BaseException as e:state['error']=repr(e);status('failed');raise
if __name__=='__main__':main()
