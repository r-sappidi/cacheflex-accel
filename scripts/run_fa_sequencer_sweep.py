"""Matched sequencer sensitivity on the exact traced multi-head FA setup."""
import concurrent.futures as cf
import configparser
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT/'benchmarks/cacheflex_flash_attention'
sys.path.insert(0, str(HERE))
import numpy as np
from validate_attribution import read
from fa_independent_reference import check
from fa_export_patch import verify_dump
from run_clean_controls import sha, save

BASE = ROOT/'results/cacheflex_flash_attention/llc_retain_20260915'

def prior(arch, t):
    return BASE/'runs'/f'{arch}_t{t}_retain{int(arch=="spm")}_small0_old0'

def reference(t):
    name = 'long_context_pv_20260914' if t == 16384 else 'pv_improvements_20260913'
    return ROOT/f'results/cacheflex_flash_attention/{name}/references/t{t}_bc512.npy'

def audit(oldpath, newpath, entries):
    old = configparser.ConfigParser(interpolation=None, strict=False)
    new = configparser.ConfigParser(interpolation=None, strict=False)
    old.read(oldpath/'config.ini'); new.read(newpath/'config.ini')
    assert old.sections() == new.sections()
    changed=[]
    for section in old.sections():
        assert set(old[section]) == set(new[section]), section
        for key, value in old[section].items():
            actual = new[section][key]
            if section in [f'system.ruby.l0_cntrl{i}.sequencer' for i in range(8)] and key=='max_outstanding_requests':
                assert value=='16' and actual==str(entries)
                changed.append([section,key,value,actual])
            else:
                assert value.replace(str(oldpath),'<OUT>') == actual.replace(str(newpath),'<OUT>'), (section,key,value,actual)
    assert len(changed)==8
    return changed

def main():
    out=Path(sys.argv[1]).resolve(); out.mkdir(); (out/'runs').mkdir()
    stop=threading.Event()
    state=dict(pid=os.getpid(),stage='preparing',completed=[])
    def status(stage):
        state.update(stage=stage,updated=time.time());save(out/'status.json',state);print(stage,flush=True)
    try:
        paths=[Path(__file__).resolve(),BASE/'build/gem5.opt']+list((ROOT/'gem5/configs').rglob('*.py'))
        paths += [HERE/n for n in ('validate_attribution.py','analyze_traffic.py','fa_independent_reference.py','fa_export_patch.py','run_clean_controls.py')]
        for t in (1024,16384):
            paths.append(reference(t))
            for arch in ('cache','spm'):
                p=prior(arch,t)
                assert (p/'accepted.json').exists()
                cmd=json.loads((p/'command.json').read_text())
                paths += [p/n for n in ('accepted.json','command.json','config.ini')]
                paths.append(Path(cmd[cmd.index('--cmd')+1]))
        manifest={str(p):sha(p) for p in paths};save(out/'manifest.json',manifest)
        save(out/'plan.json',dict(entries=[16,32,64,128],gates_T=1024,target_T=16384,
             Hq=8,Hkv=1,d=64,cores=8,VL=16,Br=16,Bc=512,group=16,
             private_L2_KiB=512,LLC_MiB=4,spm_retention=True,cache_retention=False,
             max_simulators=2,simulator=str(BASE/'build/gem5.opt'),
             note='Exact traced multi-head setup, not scaled single-head. Only sequencer limits change within each architecture. All 8 small gates precede all 8 target runs. One timing per point; no global optimum or hardware-equivalence claim.'))
        def frozen():
            for name,digest in manifest.items():assert sha(name)==digest,('input changed',name)
        env=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        def run(arch,t,entries):
            frozen();p=out/'runs'/f'{arch}_t{t}_seq{entries}';p.mkdir()
            old=prior(arch,t);cmd=json.loads((old/'command.json').read_text())
            cmd=[f'--outdir={p}' if x.startswith('--outdir=') else x for x in cmd]
            assert not any('max_outstanding_requests' in x or x.startswith('--debug-') for x in cmd)
            for i in range(8):cmd+=['-P',f'system.ruby.l0_cntrl{i}.sequencer.max_outstanding_requests={entries}']
            save(p/'command.json',cmd);start=time.monotonic();proc=None
            try:
                with (p/'run.log').open('w') as log:
                    proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
                    while True:
                        save(p/'progress.json',dict(stage='running',pid=proc.pid,updated=time.time(),host_elapsed_s=time.monotonic()-start,kernel_percentage=None))
                        if stop.is_set():raise RuntimeError('Peer failed; stopping queue')
                        try:rc=proc.wait(timeout=10);break
                        except subprocess.TimeoutExpired:pass
                save(p/'exit.json',dict(returncode=rc,host_elapsed_s=time.monotonic()-start));assert rc==0,(p,rc)
                frozen();changes=audit(old,p,entries)
                output,counters,_=read(p,16)
                for key,val in dict(T=t,Hq=8,Hkv=1,d=64,cores=8,Br=16,Bc=512).items():assert output[key]==str(val),(key,output)
                copies=8*math.ceil(t/256)*t*2 if arch=='spm' else 0
                assert counters['m_spmcpRequestsIssued']==counters['m_spmcpRequestsCompleted']==copies
                row=dict(path=str(p),arch=arch,T=t,entries=entries,output=output,counters=counters,config_changes=changes)
                verify_dump(row)
                row['validation']=check(p/'output.fp16',t,expected=np.load(reference(t)))
                if entries==16:
                    control=json.loads((old/'accepted.json').read_text())
                    assert output==control['output']
                    assert {k:v for k,v in counters.items() if k!='path'}=={k:v for k,v in control['counters'].items() if k!='path'},'16-entry control mismatch'
                # Preserve full ROI sequencer/CPU stall/TBE counters, not just extracted aggregates.
                first=(p/'stats.txt').read_text().split('End Simulation Statistics')[0]
                selected={}
                for line in first.splitlines():
                    fields=line.split()
                    if len(fields)>1 and any(x in fields[0] for x in ('sequencer.','blockedByCache','TBE','headNot','noRetire')):
                        selected[fields[0]]=fields[1]
                save(p/'mechanism_counters.json',selected)
                save(p/'accepted.json',row);save(p/'progress.json',dict(stage='accepted',roi_us=counters['roi_us']))
                return row
            except BaseException:
                stop.set()
                if proc is not None and proc.poll() is None:
                    proc.terminate()
                    try:proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()
                raise
        for t in (1024,16384):
            for entries in (16,32,64,128):
                status(f't{t}_seq{entries}')
                with cf.ThreadPoolExecutor(max_workers=2) as pool:
                    jobs=[pool.submit(run,a,t,entries) for a in ('cache','spm')]
                    for job in cf.as_completed(jobs):
                        row=job.result();state['completed'].append(dict(path=row['path'],arch=row['arch'],T=t,entries=entries,roi_us=row['counters']['roi_us']));status(state['stage'])
            if t==1024:save(out/'gates_passed.json',state['completed'])
        summary=[]
        for n in (16,32,64,128):
            pair={r['arch']:r['roi_us'] for r in state['completed'] if r['T']==16384 and r['entries']==n}
            summary.append(dict(entries=n,**pair,speedup=pair['cache']/pair['spm']))
        save(out/'summary.json',summary);status('complete')
    except BaseException as exc:
        stop.set();state['error']=repr(exc);status('failed');raise

if __name__=='__main__':main()
