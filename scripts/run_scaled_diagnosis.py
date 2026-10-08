"""Gated capacity-scaled FA pilot and bounded admission interventions.

Two additional simulators maximum. Does not modify existing queues/guests.
"""
import concurrent.futures as cf
import configparser
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import shutil

from build_scaled_fa import ROOT, build
from run_single_diagnosis import prior
from run_clean_controls import sha,save
from validate_attribution import read
from single_head_reference import reference,check
from fa_export_patch import verify_dump
from run_causal import parameters,config_key
import numpy as np

def audit(old,p,params,guest,t,newbuild):
    a=configparser.ConfigParser(interpolation=None,strict=False);b=configparser.ConfigParser(interpolation=None,strict=False)
    a.read(old/'config.ini');b.read(p/'config.ini');assert a.sections()==b.sections()
    expected={config_key(k):str(v).lower() for k,v in params.items()}
    changed=[];seen=set()
    for s in a.sections():
        for k,v in b[s].items():
            if k not in a[s]:
                assert newbuild and k in ('cacheflex_fetch_credits','cacheflex_bridge_credits') and s.endswith('.sequencer'),(s,k)
                assert v==expected.get((s,k),'0'),(s,k,v)
                if (s,k) in expected:seen.add((s,k))
            elif (s,k) in expected:
                assert v.lower()==expected[s,k],(s,k,v);seen.add((s,k))
            elif s=='system.cpu0.workload' and k in ('cmd','executable'):
                assert v==(f'cacheflex-fa {t} 64 16 32' if k=='cmd' else str(guest)),(s,k,v)
            elif k=='size' and s.endswith(('.Dcache','.cache','.L2cache')):
                size=131072 if s.endswith('.L2cache') else 16384
                assert v==str(size),(s,k,v)
            else:
                assert a[s][k].replace(str(old),'<OUT>')==v.replace(str(p),'<OUT>'),(s,k,a[s][k],v)
            if a[s].get(k)!=v:changed.append([s,k,a[s].get(k),v])
        assert not (set(a[s])-set(b[s])),s
    assert seen==set(expected),(seen,set(expected))
    return changed

def main():
    root=Path(sys.argv[1]).resolve();out=root/'experiment';out.mkdir();(out/'runs').mkdir();(out/'references').mkdir()
    state=dict(pid=os.getpid(),stage='building_guests',completed=[]);stop=threading.Event();lock=threading.Lock()
    def status(stage):
        state.update(stage=stage,updated=time.time());save(out/'status.json',state);print(stage,flush=True)
    status('building_guests')
    try:
        guests={a:build(out/'builds'/a,a) for a in ('cache','spm')}
        watched=[Path(__file__).resolve(),Path(__file__).with_name('build_scaled_fa.py')]
        watched+=list((ROOT/'gem5/configs').rglob('*.py'))+list(guests.values())
        manifest={str(p):sha(p) for p in watched};save(out/'inputs.json',manifest)
        save(out/'plan.json',dict(T=4096,gate_T=256,Hq=1,d=64,cores=8,VL=16,Br=16,Bc=32,G=1,
          L1I_KiB=64,L1D_KiB=16,private_L2_KiB=16,LLC_MiB=1,SPM_ways=2,SPM_bytes=4096,
          note='L1I unchanged: instruction footprint does not scale with T. Not a timing-equivalent proxy; validate against T16K before extrapolation.',
          max_simulators=2,implemented=['scaled baseline','fetch credit isolation','bridge credit partition','equal-total-credit controls','existing service diagnostics'],
          pending_implementation=['independent core-side bridge retry','claim-wait-only intervention','bank/controller arbitration isolation','bounded asynchronous staging'],
          controls='Old/new default exact extracted-counter equivalence at T256 and T4096, independent Bc32 mathematical output reference.',
          boundary='Bridge credit partition leaves shared LSQ retry coupling intact; cannot call it full path isolation. No optimized design or complete causal closure claimed.'))
        env=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        refs={}
        for t in (256,4096):
            status(f'reference_t{t}');refs[t]=reference(t,32);np.save(out/'references'/f't{t}_bc32.npy',refs[t])
        oldsim=Path(json.loads((prior('spm',512)/'command.json').read_text())[0])
        newsim=root/'build/gem5.opt'
        results={}
        def run(arch,t,name,total=16,fetch=0,bridge=0,mask=0,new=False):
            for fn,h in manifest.items():assert sha(fn)==h,('input changed',fn)
            p=out/'runs'/f'{arch}_t{t}_{name}';p.mkdir();old=prior(arch,512)
            cmd=json.loads((old/'command.json').read_text());cmd[0]=str(newsim if new else oldsim)
            cmd=[f'--outdir={p}' if x.startswith('--outdir=') else x for x in cmd]
            # This Ruby configuration interprets l2_size PER LLC BANK.
            sizes={'--l0d_size=':'16kB','--l1d_size=':'16kB','--l2_size=':'128kB'}
            for prefix,val in sizes.items():
                assert sum(x.startswith(prefix) for x in cmd)==1
                cmd=[prefix+val if x.startswith(prefix) else x for x in cmd]
            cmd[cmd.index('--cmd')+1]=str(guests[arch])
            cmd=[f"system.cpu[0].workload[0].cmd=['cacheflex-fa','{t}','64','16','32']" if x.startswith('system.cpu[0].workload[0].cmd=') else x for x in cmd]
            params=parameters(mask)
            for i in range(8):
                prefix=f'system.ruby.l0_cntrl{i}.sequencer.'
                params[prefix+'max_outstanding_requests']=str(total)
                if new:
                    params[prefix+'cacheflex_fetch_credits']=str(fetch)
                    params[prefix+'cacheflex_bridge_credits']=str(bridge)
            for k,v in params.items():cmd+=['-P',f'{k}={v}']
            save(p/'command.json',cmd);start=time.monotonic();proc=None
            try:
                with (p/'run.log').open('w') as f:
                    proc=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
                    while True:
                        save(p/'progress.json',dict(pid=proc.pid,host_elapsed_s=time.monotonic()-start,updated=time.time()))
                        if stop.is_set():raise RuntimeError('Peer failed')
                        if time.monotonic()-start>12*3600:raise RuntimeError('12h timeout')
                        if shutil.disk_usage(root).free<50*1024**3:raise RuntimeError('Disk guard')
                        try:rc=proc.wait(timeout=10);break
                        except subprocess.TimeoutExpired:pass
                save(p/'exit.json',dict(returncode=rc,host_elapsed_s=time.monotonic()-start));assert rc==0,(p,rc)
                changes=audit(old,p,params,guests[arch],t,new)
                output,counters,cfg=read(p,16)
                for k,v in dict(T=t,Hq=1,Hkv=1,d=64,cores=8,Br=16,Bc=32).items():assert output[k]==str(v),(k,output)
                copies=t*t//8 if arch=='spm' else 0
                assert counters['m_spmcpRequestsIssued']==counters['m_spmcpRequestsCompleted']==copies
                row=dict(path=str(p),arch=arch,T=t,name=name,output=output,counters=counters,config_changes=changes,sim_sha=sha(cmd[0]),guest_sha=sha(guests[arch]))
                verify_dump(row);row['validation']=check(p/'output.fp16',refs[t])
                if name=='new_default':
                    ctrl=results[arch,t,'old_default']
                    assert output==ctrl['output']
                    assert {k:v for k,v in counters.items() if k!='path'}=={k:v for k,v in ctrl['counters'].items() if k!='path'},'Default build counter mismatch'
                save(p/'accepted.json',row)
                with lock:
                    results[arch,t,name]=row
                    state['completed'].append(dict(arch=arch,T=t,name=name,roi_us=counters['roi_us']))
                    save(out/'status.json',state)
                    lines=['# Scaled FA: end-to-end latency','', '| Architecture | T | Intervention | Runtime us | Change vs same-build default |','|---|---:|---|---:|---:|']
                    for r in state['completed']:
                        base=results.get((r['arch'],r['T'],'new_default')) or results.get((r['arch'],r['T'],'old_default'))
                        delta=100*(r['roi_us']/base['counters']['roi_us']-1) if base else 0
                        lines.append(f"| {r['arch']} | {r['T']} | {r['name']} | {r['roi_us']:.4f} | {delta:+.3f}% |")
                    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
                print('accepted',arch,t,name,counters['roi_us'],flush=True)
            except BaseException:
                stop.set()
                if proc is not None and proc.poll() is None:
                    proc.terminate()
                    try:proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()
                raise
        def pair(t,name,**kw):
            status(f'{name}_t{t}')
            with cf.ThreadPoolExecutor(max_workers=2) as pool:
                jobs=[pool.submit(run,a,t,name,**kw) for a in ('cache','spm')]
                for j in jobs:j.result()
        for t in (256,4096):pair(t,'old_default')
        status('waiting_for_instrumented_build')
        while not (root/'build/simulator.sha256').exists():
            if time.time()-state['updated']>6*3600:raise RuntimeError('Build wait exceeded 6h')
            time.sleep(10)
        assert sha(newsim)==(root/'build/simulator.sha256').read_text().strip()
        manifest[str(newsim)]=sha(newsim);save(out/'inputs.json',manifest)
        for t in (256,4096):pair(t,'new_default',new=True)
        modes=[('shared17',17,0,0),('fetch17',17,1,0),('fetch16',16,1,0),
               ('shared32',32,0,0),('bridge32',32,0,16),('shared33',33,0,0),('both33',33,1,16)]
        for t in (256,4096):
            for name,total,fetch,bridge in modes:pair(t,name,total=total,fetch=fetch,bridge=bridge,new=True)
        for mask in (1,2,3):
            for t in (256,4096):pair(t,f'service{mask}',mask=mask,new=True)
        status('implemented_stages_complete')
    except BaseException as exc:
        stop.set();state['error']=repr(exc);status('failed');raise

if __name__=='__main__':main()
