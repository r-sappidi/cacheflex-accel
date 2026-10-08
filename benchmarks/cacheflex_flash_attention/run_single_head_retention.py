"""Isolated LLC-retention test on frozen single-head T16K V-SPM pilot.

Stock and forced-miss T512 numerical gates precede one T16K retention run.
Reuse accepted recording-off controls from the identical frozen simulator.
No kernel, default hardware, or active workload-screen dependencies change.
"""
import concurrent.futures as cf
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time

import numpy as np
from run_clean_controls import ROOT, OLD, BUILD, prior_path, command, config_audit, counter_delta, sha, save
from validate_attribution import read
from fa_export_patch import verify_dump
from single_head_reference import check

CONTROLS=ROOT/'results/cacheflex_flash_attention/clean_controls_20260917'


def main():
    out=Path(sys.argv[1]).resolve();out.mkdir();(out/'runs').mkdir()
    state=dict(pid=os.getpid(),stage='freezing',completed=[],updated=time.time())
    stop=threading.Event()
    def status(stage):
        state.update(stage=stage,updated=time.time());save(out/'status.json',state);print(stage,flush=True)
    try:
        status('freezing')
        assert json.loads((CONTROLS/'status.json').read_text())['stage']=='complete'
        assert sha(CONTROLS/'gem5.opt')==sha(BUILD/'gem5.opt')==(BUILD/'simulator.sha256').read_text().strip()
        shutil.copy2(CONTROLS/'gem5.opt',out/'gem5.opt')
        watched=[out/'gem5.opt',Path(__file__).resolve()]
        watched+=list((ROOT/'gem5/configs').rglob('*.py'))
        watched += [Path(__file__).with_name(n) for n in
                    ('run_clean_controls.py','validate_attribution.py','analyze_traffic.py','fa_export_patch.py','single_head_reference.py')]
        for t in (512,16384):
            p=prior_path('spm',t)
            cmd=json.loads((p/'command.json').read_text())
            watched += [p/'command.json',p/'config.ini',Path(cmd[cmd.index('--cmd')+1]),
                        OLD/'references'/f't{t}_bc128.npy',CONTROLS/'runs'/f'spm_t{t}_rec0'/'accepted.json']
        watched += [CONTROLS/'runs/cache_t16384_rec0/accepted.json']
        manifest={str(p):sha(p) for p in watched};save(out/'manifest.json',manifest)
        save(out/'plan.json',dict(T=16384,Hq=1,Hkv=1,d=64,cores=8,vl=16,
             Br=16,Bc=128,group=4,placement='V-only single buffer, offset0, two eligible ways',
             l1i=65536,l1d=65536,private_l2=65536,llc_total=4194304,
             forced_gate_llc_total=65536,recorder_entries=0,max_simulators=2,
             controls=str(CONTROLS),stages=['stock T512 off/on','forced-miss T512 off/on','T16384 retention on'],
             note='Same frozen simulator/guest; only runtime LLC-retention flag differs at target. No retuning.'))
        def frozen():
            for p,digest in manifest.items():assert sha(p)==digest,('dependency changed',p)
        env=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        def run(t,retain,forced=False):
            frozen();prior=prior_path('spm',t)
            p=out/'runs'/f'spm_t{t}_retain{int(retain)}_forced{int(forced)}';p.mkdir()
            original=json.loads((prior/'command.json').read_text())
            cmd=command(original,out/'gem5.opt',p,0)
            if forced:
                assert sum(x.startswith('--l2_size=') for x in cmd)==1
                cmd=['--l2_size=8kB' if x.startswith('--l2_size=') else x for x in cmd]
            for i in range(8):cmd+=['-P',f'system.ruby.l2_cntrl{i}.spm_retain_llc={retain}']
            save(p/'command.json',cmd);start=time.monotonic();proc=None
            try:
                with (p/'run.log').open('w') as log:
                    proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
                    while True:
                        text=(p/'run.log').read_text(errors='replace')
                        events=re.findall(r'[^\n]*progress event[^\n]*',text)
                        save(p/'progress.json',dict(stage='running',pid=proc.pid,updated=time.time(),host_elapsed_s=time.monotonic()-start,recent_cpu_events=events[-8:]))
                        if stop.is_set():raise RuntimeError('Peer failed')
                        try:rc=proc.wait(timeout=10);break
                        except subprocess.TimeoutExpired:pass
                elapsed=time.monotonic()-start;save(p/'exit.json',dict(returncode=rc,host_elapsed_s=elapsed));assert rc==0,(p,rc)
                frozen();output,counters,cfg=read(p,16)
                for i in range(8):
                    s=f'system.ruby.l2_cntrl{i}'
                    assert cfg[s]['spm_retain_llc']==str(retain).lower()
                    cfg[s]['spm_retain_llc']='false'
                    assert cfg[s+'.L2cache']['size']==str(8192 if forced else 524288)
                    if forced:cfg[s+'.L2cache']['size']='524288'
                audit=config_audit(prior,cfg,0)
                assert output['Hq']==output['Hkv']=='1' and output['T']==str(t)
                copies=8*math.ceil((t//8//16)/4)*t*2
                assert counters['m_spmcpRequestsIssued']==counters['m_spmcpRequestsCompleted']==copies
                assert not list(p.glob('*.flight.bin'))
                row=dict(path=str(p),output=output,counters=counters,retain=retain,forced=forced,T=t,
                         host_elapsed_s=elapsed,config_audit=audit,guest_sha256=sha(Path(cmd[cmd.index('--cmd')+1])))
                verify_dump(row)
                row['validation']=check(p/'output.fp16',np.load(OLD/'references'/f't{t}_bc128.npy'))
                stats=(p/'stats.txt').read_text().split('End Simulation Statistics')[0]
                matches=re.findall(r'^system\.ruby\.L2Cache_Controller\.SPM_IS\.Mem_Data_SPM_Retain::total\s+(\d+)',stats,re.M)
                retained=sum(map(int,matches));row['retained_fills']=retained
                if not retain:assert retained==0
                if forced and retain:assert retained>0,'Forced gate did not exercise retention'
                control=json.loads((CONTROLS/'runs'/f'spm_t{t}_rec0'/'accepted.json').read_text())
                assert control['guest_sha256']==row['guest_sha256']
                row['control_counter_changes']=counter_delta(control['counters'],counters)
                if not retain and not forced:
                    assert output==control['output'] and not row['control_counter_changes']
                save(p/'accepted.json',row);save(p/'progress.json',dict(stage='accepted',roi_us=counters['roi_us'],host_elapsed_s=elapsed))
                return row
            except BaseException as exc:
                stop.set()
                if proc is not None and proc.poll() is None:
                    proc.terminate()
                    try:proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()
                save(p/'progress.json',dict(stage='failed',error=repr(exc)));raise
        gates=[]
        for forced in (False,True):
            status('forced_miss_t512' if forced else 'stock_t512')
            with cf.ThreadPoolExecutor(max_workers=2) as pool:
                jobs=[pool.submit(run,512,r,forced) for r in (False,True)]
                for job in cf.as_completed(jobs):
                    row=job.result();gates.append(row)
                    state['completed'].append(dict(path=row['path'],roi_us=row['counters']['roi_us'],retained_fills=row['retained_fills']))
                    status(state['stage'])
        save(out/'gates_passed.json',dict(rows=gates))
        status('t16384_retention_on');target=run(16384,True)
        control=json.loads((CONTROLS/'runs/spm_t16384_rec0/accepted.json').read_text())
        cache=json.loads((CONTROLS/'runs/cache_t16384_rec0/accepted.json').read_text())
        save(out/'summary.json',dict(retention=target,spm_control=control,cache_control=cache,
             speedup_over_spm=control['counters']['roi_us']/target['counters']['roi_us'],
             speedup_over_cache=cache['counters']['roi_us']/target['counters']['roi_us']))
        state['completed'].append(dict(path=target['path'],roi_us=target['counters']['roi_us'],retained_fills=target['retained_fills']))
        status('complete')
    except BaseException as exc:
        stop.set();state['error']=repr(exc);status('failed');raise


if __name__=='__main__':main()
