"""Rebuild repaired admission accounting; gated resumption with stall guards."""
import concurrent.futures as cf
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
from run_scaled_diagnosis import ROOT, save, sha, audit, prior, parameters, read, verify_dump, check
import numpy as np

B=ROOT/'results/cacheflex_flash_attention'
OLD=B/'scaled_diagnosis_20260918/experiment'
OUT=B/'scaled_admission_repaired_v2_20260918'
ENV=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib',PYTHON_CONFIG='/data2/rsappidi/conda-envs/cacheflex-gem5/bin/python3-config',PROTOC='/usr/bin/protoc',CCFLAGS_EXTRA='-gdwarf-2',GEM5PY_CCFLAGS_EXTRA='-gdwarf-2',LINKFLAGS_EXTRA='-L/usr/lib64',OPENBLAS_NUM_THREADS='1')

def main():
    OUT.mkdir();(OUT/'runs').mkdir();(OUT/'build').mkdir()
    state=dict(pid=os.getpid(),completed=[]);stop=threading.Event();lock=threading.Lock()
    def status(stage):
        state.update(stage=stage,updated=time.time());save(OUT/'status.json',state);print(stage,flush=True)
    try:
        status('building')
        sources=[ROOT/'gem5/src/mem/ruby/system'/n for n in ('Sequencer.cc','Sequencer.hh','HTMSequencer.cc')]
        for p in sources:shutil.copy2(p,OUT/'build'/p.name)
        save(OUT/'source_hashes.json',{str(p):sha(p) for p in sources})
        cmd=['/data2/rsappidi/conda-envs/cacheflex-gem5/bin/scons','build/ARM_MESI_Three_Level_SPM/gem5.opt','-j8','--linker=gold']
        save(OUT/'build/command.json',cmd)
        with (OUT/'build/build.log').open('w') as log:subprocess.run(cmd,cwd=ROOT/'gem5',env=ENV,stdout=log,stderr=subprocess.STDOUT,check=True)
        sim=OUT/'build/gem5.opt';shutil.copy2(ROOT/'gem5/build/ARM_MESI_Three_Level_SPM/gem5.opt',sim)
        save(OUT/'build/simulator.json',dict(sha256=sha(sim)))
        refs={t:np.load(OLD/'references'/f't{t}_bc32.npy') for t in (256,4096)}
        save(OUT/'plan.json',dict(predecessor=str(OLD),geometry='unchanged scaled T4096 proxy, T256 gates',max_simulators=2,
            fix='Completed request releases partition credit before synchronous retry callback',
            guards='256MiB per log,120 host seconds without retirement progress,12h timeout,50GiB free disk',
            baseline='Old accepted new_default controls must exactly match extracted counters and output',
            methodology='Same bounded partitions and masks as original plan; no core retry isolation or timing change'))
        def run(arch,t,name,total=16,fetch=0,bridge=0,mask=0):
            p=OUT/'runs'/f'{arch}_t{t}_{name}';p.mkdir()
            ctrl=OLD/'runs'/f'{arch}_t{t}_new_default'
            cmd=json.loads((ctrl/'command.json').read_text());cmd[0]=str(sim)
            cmd=[f'--outdir={p}' if x.startswith('--outdir=') else x for x in cmd]
            guest=Path(cmd[cmd.index('--cmd')+1]);params=parameters(mask)
            for i in range(8):
                prefix=f'system.ruby.l0_cntrl{i}.sequencer.'
                params.update({prefix+'max_outstanding_requests':str(total),prefix+'cacheflex_fetch_credits':str(fetch),prefix+'cacheflex_bridge_credits':str(bridge)})
            # Remove previous overrides before inserting exactly one of each.
            clean=[];i=0
            while i<len(cmd):
                if cmd[i]=='-P' and cmd[i+1].split('=',1)[0] in params:i+=2
                else:clean.append(cmd[i]);i+=1
            cmd=clean
            for k,v in params.items():cmd+=['-P',f'{k}={v}']
            save(p/'command.json',cmd);save(p/'inputs.json',{str(guest):sha(guest),str(sim):sha(sim)})
            start=time.monotonic();last_progress=start;last_total=-1;counts={};child=None
            try:
                with (p/'run.log').open('w') as log:
                    child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=ENV)
                    while child.poll() is None:
                        now=time.monotonic()
                        save(p/'progress.json',dict(pid=child.pid,host_elapsed_s=now-start,total_committed=sum(counts.values()),updated=time.time()))
                        if stop.is_set():raise RuntimeError('peer failed')
                        if (p/'run.log').stat().st_size>256*1024**2:raise RuntimeError('log-size guard')
                        with (p/'run.log').open('rb') as f:
                            f.seek(max(0,(p/'run.log').stat().st_size-65536));tail=f.read().decode(errors='replace')
                        for cpu,count in re.findall(r'system.cpu(\d+) progress event, total committed:(\d+)',tail):counts[cpu]=int(count)
                        total_retired=sum(counts.values())
                        if total_retired>last_total:last_total=total_retired;last_progress=now
                        if now-last_progress>120:raise RuntimeError('120s without instruction retirement progress')
                        if now-start>12*3600:raise RuntimeError('12h timeout')
                        if shutil.disk_usage(OUT).free<50*1024**3:raise RuntimeError('disk guard')
                        try:child.wait(timeout=5)
                        except subprocess.TimeoutExpired:pass
                assert child.returncode==0,(p,child.returncode)
                changes=audit(prior(arch,512),p,params,guest,t,True)
                output,counters,cfg=read(p,16)
                copies=t*t//8 if arch=='spm' else 0
                assert counters['m_spmcpRequestsIssued']==counters['m_spmcpRequestsCompleted']==copies
                row=dict(path=str(p),arch=arch,T=t,name=name,output=output,counters=counters,config_changes=changes,guest_sha=sha(guest),sim_sha=sha(sim))
                verify_dump(row);row['validation']=check(p/'output.fp16',refs[t])
                if name=='default':
                    old=json.loads((ctrl/'accepted.json').read_text())
                    assert output==old['output']
                    assert {k:v for k,v in counters.items() if k!='path'}=={k:v for k,v in old['counters'].items() if k!='path'},'default-equivalence mismatch'
                save(p/'accepted.json',row)
                with lock:
                    state['completed'].append(dict(arch=arch,T=t,name=name,roi_us=counters['roi_us']));save(OUT/'status.json',state)
                print('accepted',arch,t,name,counters['roi_us'],flush=True)
            except BaseException as e:
                stop.set();save(p/'failed.json',dict(error=repr(e)))
                if child and child.poll() is None:
                    child.terminate()
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:child.kill();child.wait()
                raise
        def pair(t,name,**kw):
            status(f'{name}_t{t}')
            with cf.ThreadPoolExecutor(max_workers=2) as pool:
                jobs=[pool.submit(run,a,t,name,**kw) for a in ('cache','spm')]
                for job in cf.as_completed(jobs):job.result()
        pair(256,'default')
        pair(256,'fetch17',total=17,fetch=1) # regression before expensive controls
        pair(4096,'default')
        modes=[('shared17',17,0,0),('fetch17',17,1,0),('fetch16',16,1,0),('shared32',32,0,0),('bridge32',32,0,16),('shared33',33,0,0),('both33',33,1,16)]
        for t in (256,4096):
            for name,total,fetch,bridge in modes:
                if t==256 and name=='fetch17':continue
                pair(t,name,total=total,fetch=fetch,bridge=bridge)
        for mask in (1,2,3):
            for t in (256,4096):pair(t,f'service{mask}',mask=mask)
        status('complete')
    except BaseException as e:state['error']=repr(e);status('failed');raise

if __name__=='__main__':main()
