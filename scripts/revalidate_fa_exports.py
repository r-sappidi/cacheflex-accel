"""Two-simulator immutable-ROI export repair gates, followed by failed cases.

No hardware changes or tolerance changes. Preserve original failed evidence.
After all gates pass, restart the two existing searches with repaired exports.
"""
import concurrent.futures as cf
import importlib.util
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

ROOT = Path(__file__).resolve().parents[1]
FA = ROOT/'benchmarks/cacheflex_flash_attention'
sys.path[:0] = [str(FA), str(FA/'kv_partition')]
from build_access_timing import sha, save
from run_access_timing import stats
from test_micro_timing import check_global_guard
from export_safe_elf import export_hash
from reference import validate

spec = importlib.util.spec_from_file_location('llama_run', FA/'llama_cpu/run.py')
llama = importlib.util.module_from_spec(spec)
spec.loader.exec_module(llama)
B = ROOT/'results/cacheflex_flash_attention'
OUT = B/'export_repair_20260926'
CACHE = B/'cache_tuning_20260925/experiment'
LLAMA = B/'llama_cpu_20260925/experiment_v2'


def roi(path):
    return {k:v for k,v in stats(path)[0].items()
            if not k.startswith('host') and math.isfinite(v)}


def main():
    (OUT/'runs').mkdir()
    state = dict(pid=os.getpid(), stage='starting', active={}, completed=[])
    lock, stop = threading.RLock(), threading.Event()
    frozen = {str(p):sha(p) for p in [Path(__file__), FA/'export_safe_elf.py',
        FA/'fa_export_refresh.S', FA/'kv_partition/reference.py', FA/'llama_cpu/run.py',
        FA/'cache_tuning/run.py', *list((OUT/'builds').glob('*/guest'))]}
    # Protect every original simulator/config/library/guest dependency, except
    # intentionally repaired runner source (old bytes archived in preserved/).
    edited = {str(FA/'llama_cpu/run.py'), str(FA/'cache_tuning/run.py')}
    for suite in (CACHE, LLAMA):
        for p,h in json.loads((suite/'inputs.json').read_text()).items():
            if p not in edited:
                assert sha(p)==h, ('original dependency changed',p)
                frozen[p]=h
    save(OUT/'inputs.json',frozen)
    def unchanged():
        for p,h in frozen.items():assert sha(p)==h,('changed dependency',p)
    def status(stage=None):
        with lock:
            if stage:state['stage']=stage
            state['updated']=time.time();save(OUT/'status.json',state)
    def run(case):
        mode, name, t, bc = case
        old=(LLAMA if 'sve' in mode else CACHE)/'runs'/name
        p=OUT/'runs'/name;p.mkdir();guest=OUT/'builds'/mode/'guest'
        cmd=json.loads((old/'command.json').read_text())
        cmd=[s.replace(str(old),str(p)) for s in cmd]
        cmd[cmd.index('--cmd')+1]=str(guest)
        args=cmd[cmd.index('--options')+1].split()
        save(p/'command.json',cmd);proc=None;start=time.monotonic()
        try:
            unchanged()
            assert not stop.is_set()
            with (p/'run.log').open('w') as log:
                proc=subprocess.Popen(cmd,cwd=ROOT,env=llama.ENV,stdout=log,stderr=subprocess.STDOUT)
                with lock:state['active'][name]=dict(pid=proc.pid);status()
                while proc.poll() is None:
                    assert not stop.is_set(),'peer failed'
                    assert shutil.disk_usage(OUT).free>40*2**30,'disk guard'
                    assert time.monotonic()-start<172800,'timeout'
                    save(p/'progress.json',dict(pid=proc.pid,host_seconds=time.monotonic()-start))
                    try:proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:pass
            log=(p/'run.log').read_text()
            assert proc.returncode==0 and 'Simulated exit code not 0!' not in log,log[-2000:]
            assert ('LLAMA_ROI_END status=0' if 'sve' in mode else 'KV_ROI_END') in log
            config=llama.audit(p,guest,args)
            for s in stats(p):check_global_guard(s)
            h=export_hash(p/'output.fp16',log)
            if 'sve' in mode:
                numerical=llama.numerical(p,np.load(LLAMA/'references'/f't{t}.npy'))
            else:
                numerical=validate(p/'output.fp16',np.load(CACHE/'references'/f't{t}_bc{bc}.npy'),t,dense=t==1024)
            before,after=roi(old),roi(p)
            diff={k:[before.get(k),after.get(k)] for k in before.keys()|after.keys()
                  if before.get(k)!=after.get(k)}
            save(p/'roi_comparison.json',dict(differences=diff,counters=len(before),old=str(old)))
            assert not diff, ('ROI changed',name,list(diff.items())[:10])
            oldh=re.search(r'(?:KV|LLAMA)_RESULT hash=([0-9a-f]+)',(old/'run.log').read_text())[1]
            assert oldh==h,('guest arithmetic changed',oldh,h)
            unchanged()
            row=dict(name=name,mode=mode,t=t,path=str(p),roi_us=after['simTicks']/1e6,
                     arithmetic=numerical,config=config,output_hash=h,roi_exact=True,
                     host_seconds=time.monotonic()-start)
            save(p/'validated.json',row)
            with lock:state['completed'].append(row)
            return row
        except BaseException as e:
            stop.set();save(p/'failed.json',dict(error=repr(e)))
            with lock:state.setdefault('error',f'{name}: {e!r}')
            raise
        finally:
            if proc and proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
            with lock:state['active'].pop(name,None);status()
    try:
        status('small_equivalence_gates')
        with cf.ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(run,[('qpack','qpack_t1024_br16_bc128_g4',1024,128),
                               ('ported_sve_ready','ported_sve_ready_t512',512,0)]))
        status('failed_t4096_revalidation')
        with cf.ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(run,[('qpack','qpack_t4096_br16_bc128_g16',4096,128),
                               ('ported_sve_ready','ported_sve_ready_t4096',4096,0)]))
        save(OUT/'gates_passed.json',dict(cases=4,full_numerical=True,exact_roi_counters=True,
             cause='Functional m5_write_file export lost resident data; guest timing-read hashes unchanged'))
        status('completed_validated')
        launched=[]
        for script in (FA/'cache_tuning/run.py',FA/'llama_cpu/run.py'):
            logfile=OUT/(script.parent.name+'_resumed.host.log')
            with logfile.open('w') as log:
                proc=subprocess.Popen([sys.executable,str(script)],cwd=ROOT,env=llama.ENV,
                                      stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            launched.append(dict(script=str(script),pid=proc.pid,log=str(logfile)))
        save(OUT/'resumed_queues.json',launched)
    except BaseException:
        status('failed');raise


if __name__=='__main__':main()
