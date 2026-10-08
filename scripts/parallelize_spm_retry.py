"""Adopt the live FA controls and start their queued third arm exactly once.

The old coordinator is stopped (NOT its simulation children), preserving each
child's exit status as a zombie until it is captured here. After all children
finish, retire the old coordinator. No executable/config/frozen helper is edited.
The existing experiment status/results remain authoritative; a separate journal
records this scheduling-only handoff. Ledger FA-31/FA-29; max three FA sims.
"""
import concurrent.futures as cf
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import threading
import time
import numpy as np
import run_spm_retry as prior
from build_access_timing import sha,save
from run_access_timing import stats
from test_micro_timing import check_global_guard
from export_safe_elf import export_hash
from reference import validate
from analyze_spm_set_spread import analyze

ROOT=Path(__file__).resolve().parents[1]
OUT=prior.BASE/'experiment_retry_v2'
JOURNAL=prior.BASE/'parallel_handoff_20260927'
NAMES={'t16384_cache','t16384_spm_control'}

def process(pid):
    p=Path(f'/proc/{pid}/stat')
    if not p.exists():return None
    x=p.read_text().rsplit(') ',1)[1].split()
    return dict(state=x[0],ppid=int(x[1]),start_ticks=int(x[19]),wait_status=int(x[49]))

def main():
    JOURNAL.mkdir()
    original=json.loads((OUT/'status.json').read_text())
    assert original['stage']=='t16384_matched_residency_comparison',original['stage']
    assert set(original['active'])==NAMES,original['active']
    assert not (OUT/'runs/t16384_spm_spread').exists(),'third arm already launched'
    parent=original['pid']
    assert b'scripts/run_spm_retry.py' in Path(f'/proc/{parent}/cmdline').read_bytes()
    parent_start=process(parent)['start_ticks']
    frozen=json.loads((OUT/'inputs.json').read_text())
    for p,h in frozen.items():assert sha(p)==h,('changed input',p)
    frozen[str(Path(__file__))]=sha(Path(__file__))
    save(JOURNAL/'inputs.json',frozen)
    assert all(json.loads((OUT/'gates_passed.json').read_text()).values())
    os.kill(parent,signal.SIGSTOP)
    newproc=None; adopted={}; committed=False
    try:
        for _ in range(100):
            if process(parent)['state']=='T':break
            time.sleep(.01)
        assert process(parent)['state']=='T'
        original=json.loads((OUT/'status.json').read_text())
        assert set(original['active'])==NAMES and not (OUT/'runs/t16384_spm_spread').exists()
        save(JOURNAL/'original_status.json',original)
        for name,row in original['active'].items():
            ident=process(row['pid']);assert ident and ident['ppid']==parent
            assert str(OUT/'runs'/name).encode() in Path(f'/proc/{row["pid"]}/cmdline').read_bytes()
            adopted[name]=dict(row,identity=ident)
        save(JOURNAL/'handoff.json',dict(old_coordinator=parent,old_start_ticks=parent_start,
            new_coordinator=os.getpid(),adopted=adopted,scope='scheduling only',
            max_fa_simulations=3,global_intended_max=6,
            duplicate_prevention='old coordinator SIGSTOP until children validated, then retired'))
        state=dict(original,pid=os.getpid(),active=adopted,previous_coordinator=parent,
                   handoff=str(JOURNAL),max_simulators=3)
        lock=threading.RLock();stop=threading.Event()
        def status(stage=None):
            with lock:
                if stage:state['stage']=stage
                state['updated']=time.time()
                save(OUT/'status.json',state);save(JOURNAL/'status.json',state)
        def unchanged():
            for p,h in frozen.items():assert sha(p)==h,('changed input',p)
        refpaths=[Path(p) for p in frozen if Path(p).name=='t16384_bc512_parts1.npy']
        assert len(refpaths)==1,refpaths
        ref=np.load(refpaths[0])
        source=Path(adopted['t16384_spm_control']['source'])
        target=OUT/'runs/t16384_spm_spread';target.mkdir()
        cmd=json.loads((OUT/'runs/t16384_spm_control/command.json').read_text())
        cmd[1]='--outdir='+str(target)
        count=0
        for i,x in enumerate(cmd):
            if '.cache.spm_set_stride=' in x:
                assert x.endswith('=0');cmd[i]=x.rsplit('=',1)[0]+'=512';count+=1
        assert count==8
        save(target/'command.json',cmd)
        shutil.copy2(OUT/'runs/t16384_spm_control/trace_window.json',target/'trace_window.json')
        unchanged()
        with (target/'run.log').open('w') as log:
            newproc=subprocess.Popen(cmd,cwd=ROOT,env=prior.finite.ENV,stdout=log,stderr=subprocess.STDOUT)
        newrow=dict(pid=newproc.pid,source=str(source),stride=512,identity=process(newproc.pid))
        jobs={**adopted,'t16384_spm_spread':newrow}
        state['active']=dict(jobs);status();committed=True

        def run(name,row):
            path=OUT/'runs'/name;start=time.monotonic();ident=row['identity']
            try:
                while True:
                    assert not stop.is_set(),'peer failure'
                    assert time.monotonic()-start<172800,'48h timeout'
                    assert shutil.disk_usage(OUT).free>50*2**30,'disk guard'
                    if name=='t16384_spm_spread':
                        rc=newproc.poll()
                        if rc is not None:break
                    else:
                        now=process(row['pid'])
                        assert now and now['start_ticks']==ident['start_ticks'],('lost adopted child',name)
                        if now['state']=='Z':
                            rc=os.waitstatus_to_exitcode(now['wait_status'])
                            save(path/'adopted_exit.json',dict(pid=row['pid'],returncode=rc,**now))
                            break
                    save(path/'parallel_progress.json',dict(pid=row['pid'],handoff_seconds=time.monotonic()-start))
                    time.sleep(5)
                assert rc==0,(name,rc)
                log=(path/'run.log').read_text()
                assert 'KV_ROI_END' in log and 'because exiting with last active thread context' in log
                assert 'Simulated exit code not 0!' not in log
                source=Path(row['source']);stride=row['stride']
                delta=prior.audit(source,path,stride,True)
                cfg=json.loads((path/'config.json').read_text());ss=stats(path)
                for s in ss:
                    check_global_guard(s)
                    for i in range(8):
                        cpu=cfg['system']['cpu'][i]
                        for domain in ('regular','spm'):
                            for kind,cap in [('Load',cpu['LQEntries']),('Store',cpu['SQEntries'])]:
                                assert s[f'system.cpu{i}.{domain}Lsq0.boundedMax{kind}Occupancy']<=cap
                cmd=json.loads((path/'command.json').read_text())
                t,cores,br,bc,group,sets,kv,spm=map(int,cmd[cmd.index('--options')+1].split())
                assert (t,cores,br,bc,group,sets,kv)==(16384,8,16,512,48,1024,0)
                export_hash(path/'output.fp16',log)
                arithmetic=validate(path/'output.fp16',ref,t,dense=False)
                assert sha(path/'output.fp16')==sha(source/'output.fp16')
                expected=16*t*math.ceil((t//8)/(br*group)) if spm else 0
                for field in ('m_spmcpRequestsIssued','m_spmcpRequestsCompleted'):
                    assert sum(ss[0][f'system.ruby.l0_cntrl{i}.sequencer.{field}'] for i in range(8))==expected
                assert ss[0].get('system.ruby.L1Cache_Controller.X.SPM_release::total',0)==(16*bc if spm else 0)
                received=ss[0].get('system.ruby.L1Cache_Controller.SPM_SilentRetry::total',0)
                reissued=ss[0].get('system.ruby.L1Cache_Controller.SPM_SilentReissue::total',0)
                assert received==reissued
                save(path/'residency_summary.json',analyze(path/'residency.gz'))
                unchanged()
                result=dict(name=name,source=str(source),stride=stride,config_delta=delta,
                    arithmetic=arithmetic,roi_us=ss[0]['simTicks']/1e6,copies=expected,
                    output_sha256=sha(path/'output.fp16'),silent_retry_received=received,
                    silent_retry_reissued=reissued,handoff_host_seconds=time.monotonic()-start,
                    adopted=name in adopted,returncode=rc)
                save(path/'validated.json',result)
                with lock:state['completed'].append(result)
            except BaseException as e:
                stop.set();save(path/'failed.json',dict(error=repr(e)))
                with lock:state.setdefault('error',name+': '+repr(e))
                raise
            finally:
                with lock:state['active'].pop(name,None);status()
        try:
            with cf.ThreadPoolExecutor(max_workers=3) as pool:
                list(pool.map(lambda item:run(*item),jobs.items()))
            status('completed_requires_analysis')
        except BaseException as e:
            state.setdefault('error',repr(e));status('failed');raise
        finally:
            for row in jobs.values():
                now=process(row['pid'])
                if now and now['start_ticks']==row['identity']['start_ticks'] and now['state']!='Z':
                    os.kill(row['pid'],signal.SIGTERM)
            if newproc.poll() is None:
                try:newproc.wait(timeout=10)
                except subprocess.TimeoutExpired:newproc.kill();newproc.wait()
            # Old parent must never resume into its already-executed third task.
            if process(parent) and process(parent)['start_ticks']==parent_start:
                os.kill(parent,signal.SIGKILL)
            save(JOURNAL/'old_coordinator_retired.json',dict(pid=parent,simulator_inputs_unchanged=True))
            from analyze_spm_retry import main as report
            report()
    except BaseException:
        if not committed:
            if newproc and newproc.poll() is None:newproc.terminate();newproc.wait(timeout=10)
            # No external arm has been committed: the original queue can resume.
            # Preserve any prepared but unlaunched directory under the journal.
            target=OUT/'runs/t16384_spm_spread'
            if target.exists():shutil.move(str(target),str(JOURNAL/'unlaunched_spread'))
            if process(parent) and process(parent)['start_ticks']==parent_start:
                os.kill(parent,signal.SIGCONT)
        raise

if __name__=='__main__':main()
