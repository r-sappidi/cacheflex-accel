"""Follow up the observed 7-cycle PV-loop dispatch cadence, without retiming.

Wait for the two-slot consumer queue, then reuse its exact early prefixes with
Fetch/Decode/Rename logging. All non-host counters must match the untraced
prefix. No simulated hardware/kernel changes.
"""
import concurrent.futures as cf
import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from run_spm_consumer_trace import EX, ENV, ROOT, config_check, prior, stats, sha, save

OUT = EX/'frontend_followup'


def main():
    OUT.mkdir()
    state = dict(pid=os.getpid(), stage='waiting_for_consumer_queue', active={})
    lock = threading.Lock(); halt = threading.Event()
    def update():
        state['updated'] = time.time(); save(OUT/'status.json', state)
    update()
    try:
        while True:
            parent = json.loads((EX/'status.json').read_text())
            assert parent['stage'] != 'failed', parent
            if parent['stage'] == 'completed_validated_diagnostic':
                break
            time.sleep(5)
        manifest = json.loads((EX/'inputs.json').read_text())
        manifest[str(Path(__file__).resolve())] = sha(Path(__file__))
        save(OUT/'inputs.json', manifest)
        def frozen():
            for p,h in manifest.items():
                assert sha(p)==h, ('Changed input', p)
        state['stage']='frontend_trace';update()
        def run(v):
            frozen();p=OUT/v;p.mkdir()
            old=EX/'early_trace'/v
            cmd=json.loads((old/'command.json').read_text())
            cmd=[f'--outdir={p}' if x.startswith('--outdir=') else
                 '--debug-flags=CacheFlexDeps,Fetch,Decode,Rename' if x.startswith('--debug-flags=') else x
                 for x in cmd]
            # Retain all stop ticks, narrow only debug output around the first
            # complete PV loop. Trace observations have no timing semantics.
            start=int(next(x.split('=')[1] for x in cmd if x.startswith('--debug-start=')))
            cmd=[f'--debug-end={start+3_000_000}' if x.startswith('--debug-end=') else x for x in cmd]
            save(p/'command.json',cmd);proc=None;begin=time.monotonic()
            try:
                with (p/'run.log').open('w') as log:
                    proc=subprocess.Popen(cmd,cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT)
                    with lock:state['active'][v]=proc.pid;update()
                    while proc.poll() is None:
                        assert not halt.is_set(), 'Peer failed'
                        assert time.monotonic()-begin<3600, 'Timeout'
                        assert shutil.disk_usage(OUT).free>50*2**30, 'Disk guard'
                        tr=p/'trace.gz'
                        assert not tr.exists() or tr.stat().st_size<2**30, 'Trace size cap'
                        try:proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:pass
                assert proc.returncode==0, proc.returncode
                config_check(prior(v,4096),p)
                assert stats(p)==stats(EX/'early_observer_off'/v), 'Observer changed counters'
                frozen()
                save(p/'validated_prefix.json',dict(config_equal=True, all_finite_nonhost_stats_equal=True,
                     reference=str(EX/'early_observer_off'/v), diagnostic_prefix=True))
                with lock:state['active'].pop(v);update()
            except BaseException:
                halt.set()
                if proc and proc.poll() is None:
                    proc.terminate()
                    try:proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()
                raise
        with cf.ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(run,v) for v in ('normal','fast')]
            for f in futures:f.result()
        state['stage']='completed_validated';update()
    except BaseException as e:
        state['stage']='failed';state['error']=repr(e);update();raise


if __name__=='__main__':main()
