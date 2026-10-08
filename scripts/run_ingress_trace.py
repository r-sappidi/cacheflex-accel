"""Bounded, read-only diagnostic replays of accepted T4096 bypass cases.

Never interpret early-exit stats as completed FA performance/correctness results.
No hardware or guest modifications. At most two replay simulators at once.
"""
import argparse
import concurrent.futures as cf
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time

from build_access_timing import ROOT, sha, save
from run_access_timing import stats

BASE = ROOT/'results/cacheflex_flash_attention/ingress_bypass_20260922/experiment'
OUT = ROOT/'results/cacheflex_flash_attention/ingress_trace_20260922'


def config_equal(a, b):
    def norm(x, root):
        if isinstance(x, dict): return {k:norm(v,root) for k,v in x.items()}
        if isinstance(x, list): return [norm(v,root) for v in x]
        return x.replace(str(root), '<OUT>') if isinstance(x,str) else x
    assert norm(json.loads((a/'config.json').read_text()), a) == norm(
        json.loads((b/'config.json').read_text()), b), 'Configuration changed'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('name')
    ap.add_argument('--start', type=int, default=1316000000)
    ap.add_argument('--stop', type=int, default=1650000000)
    ap.add_argument('--flags', default='ProtocolTrace,SPM_CPU')
    ap.add_argument('--match', help='Prior bounded replay pair to compare every non-host statistic')
    args = ap.parse_args()
    assert 0 <= args.start < args.stop
    OUT.mkdir(exist_ok=True)
    ex = OUT/args.name; ex.mkdir()
    lock = threading.Lock(); halt = threading.Event()
    state = dict(pid=os.getpid(), stage='freezing', active={}, completed=[],
                 diagnostic_only=True, start=args.start, stop=args.stop)
    def status():
        with lock:
            state['updated'] = time.time(); save(ex/'status.json', state)
    paths = [Path(__file__).resolve()]
    for w in (0,4):
        old = BASE/f'spm_t4096_w{w}'
        assert (old/'validated.json').exists()
        cmd = json.loads((old/'command.json').read_text())
        paths += [old/'command.json', old/'validated.json', old/'config.json',
                  Path(cmd[0]), Path(cmd[cmd.index('--cmd')+1])]
        paths += list(Path(cmd[2]).parents[2].rglob('*.py'))
    inputs = {str(p):sha(p) for p in paths}
    save(ex/'inputs.json', inputs)
    def frozen():
        for p,h in inputs.items(): assert sha(Path(p)) == h, p
    status()
    def run(w):
        p = ex/f'w{w}'; p.mkdir(); old=BASE/f'spm_t4096_w{w}'
        cmd=json.loads((old/'command.json').read_text())
        cmd=[f'--outdir={p}' if x.startswith('--outdir=') else x for x in cmd]
        if args.flags:
            cmd[1:1]=[f'--debug-flags={args.flags}', f'--debug-start={args.start}',
                      f'--debug-end={args.stop}', '--debug-file=trace.gz',
                      '--debug-ignore='+':'.join(f'system.cpu{i}' for i in range(1,8))]
        cmd += [f'--abs-max-tick={args.stop}']
        save(p/'command.json',cmd)
        env=dict(os.environ, LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib',
                 OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
        for k in list(env):
            if k.startswith('CACHEFLEX_AGGREGATE') or k=='CACHEFLEX_TRACE_LINE': env.pop(k)
        proc=None; begin=time.monotonic()
        try:
            frozen()
            with (p/'run.log').open('w') as log:
                proc=subprocess.Popen(cmd,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                with lock: state['active'][f'w{w}']=proc.pid
                status()
                while proc.poll() is None:
                    assert not halt.is_set(), 'Peer failed'
                    assert time.monotonic()-begin < 3600, 'One-hour diagnostic cap'
                    assert shutil.disk_usage(ex).free > 50*2**30, 'Disk guard'
                    trace=p/'trace.gz'
                    assert not trace.exists() or trace.stat().st_size < 4*2**30, 'Trace size cap'
                    try: proc.wait(timeout=5)
                    except subprocess.TimeoutExpired: pass
            assert proc.returncode==0, (p,proc.returncode)
            assert 'simulate() limit reached' in (p/'run.log').read_text(), 'Expected bounded exit'
            config_equal(old,p)
            s=stats(p); assert int(s[-1]['finalTick']) == args.stop
            if args.match:
                other=OUT/args.match/f'w{w}'
                assert stats(other)==s, 'Diagnostic instrumentation changed finite non-host statistics'
            frozen()
            row=dict(lookahead=w, bounded_exit=True, exact_config=True,
                     final_tick=args.stop, host_seconds=time.monotonic()-begin,
                     exact_nonhost_stats_vs=args.match, diagnostic_only=True,
                     full_FA_correctness_reference=str(old/'validated.json'))
            save(p/'validated_prefix.json',row)
            with lock:
                state['active'].pop(f'w{w}'); state['completed'].append(row)
            status()
        except BaseException:
            halt.set()
            if proc and proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
            raise
    try:
        state['stage']='running';status()
        with cf.ThreadPoolExecutor(max_workers=2) as pool:
            tasks=[pool.submit(run,w) for w in (0,4)]
            for f in cf.as_completed(tasks):f.result()
        state['stage']='completed_bounded_diagnostic';status()
    except BaseException as exc:
        state['stage']='failed';state['error']=repr(exc);status();raise


if __name__=='__main__':main()
