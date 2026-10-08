"""AI-13b: KC-neighbor convergence at T2048/T4096 for the AI-13 16-credit comparison.

AI-13 transfers only each backend's T1024 winner(s) to T2048/T4096. Accepted data show
the best tiling can move with T (QKV cache: AI-11 8-credit KC512 beats AI-13 16-credit
KC1024 at T2048). After AI-13 finishes a (workload, backend, T), this queue runs the
KC neighbors of the current best (first-call and repeat-call) until the best has both
neighbors tested within {128,256,512,1024}. Same frozen platform, commands and
acceptance as AI-13 (imported, not copied); AI-13 sources are never edited.
"""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
import fcntl, json, re, resource, shutil, signal, subprocess, sys, threading, time, traceback
from pathlib import Path
H = Path(__file__).resolve().parent; R = Path('/data2/rsappidi/cacheflex_mc')
sys.path.insert(0, str(R / 'benchmarks/qwen/components/mshr16'))
import plan as P
import run as A13                      # AI-13 module: command(), validate(), pressure(), cpus(), expected_hashes()
import broker
from build_access_timing import sha
from parallel_budget_scheduler import owned
OUT = R / 'results/ai_workloads/qwen_mshr16_neighbors_20261005'
A13_STATUS = P.OUT / 'status.json'
SLOTS = (12, 14)
GIB = 2 ** 30


def save(p, v): A13.save(p, v)
def read(p): return json.loads(Path(p).read_text())


def neighbors(kc):
    i = P.KCS.index(kc); return [P.KCS[j] for j in (i - 1, i + 1) if 0 <= j < len(P.KCS)]


class Followup:
    def __init__(self):
        OUT.mkdir(parents=True, exist_ok=False); (OUT / 'runs').mkdir()
        self.lockfile = (OUT / 'coordinator.lock').open('w'); fcntl.flock(self.lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.sched_setaffinity(0, A13.cpus(P.COORDINATOR_SLOT)); os.nice(10)
        resource.setrlimit(resource.RLIMIT_AS, (4 * GIB, resource.RLIM_INFINITY))
        self.lock = threading.RLock(); self.free = list(SLOTS); self.threads = {}; self.stop = threading.Event()
        self.state = dict(pid=os.getpid(), stage='watching', active={}, pending=[], completed=[], failed=[], decisions=[],
                          scope='AI-13b KC-neighbor convergence at T2048/T4096, 16 credits; bounded to KC{128,256,512,1024}')
        frozen = read(P.OUT / 'inputs.json')
        frozen[str(Path(__file__))] = sha(Path(__file__)); self.frozen = frozen
        for p, h in frozen.items(): assert sha(p) == h, ('frozen drift at start', p)
        save(OUT / 'inputs.json', frozen)

    def status(self, stage=None):
        with self.lock:
            if stage: self.state['stage'] = stage
            self.state['updated'] = time.time(); save(OUT / 'status.json', self.state)

    def unchanged(self):
        for p, h in self.frozen.items(): assert sha(p) == h, ('frozen input drift', p)

    def needed(self, a13):
        """Neighbor cases required now, from accepted AI-13 + AI-13b rows."""
        rows = [r for r in a13['completed'] + self.state['completed'] if r['case']['credits'] == 16 and r['case']['t'] in (2048, 4096)]
        a13_failed = {f['name'] for f in a13['failed']}
        mine = {n for n in self.state['pending']} | set(self.state['active']) | {f['name'] for f in self.state['failed']} | {r['name'] for r in self.state['completed']}
        out = []
        for wl, sel in a13.get('selections', {}).items():
            for t in (2048, 4096):
                for arm in P.ARMS:
                    planned = [P.case(f'target{t}', wl, arm, t, kc)['name'] for kc in sel[arm]]
                    done = {r['name'] for r in a13['completed']}
                    if any(n not in done and n not in a13_failed for n in planned): continue   # AI-13 still owns this cell
                    have = {r['case']['kc']: r for r in rows if (r['case']['wl'], r['case']['arm'], r['case']['t']) == (wl, arm, t)}
                    if not have: continue
                    bests = {min(have.values(), key=lambda r: r['first_us'])['case']['kc'], min(have.values(), key=lambda r: r['repeat_us'])['case']['kc']}
                    for b in sorted(bests):
                        for kc in neighbors(b):
                            if kc in have: continue
                            c = P.case(f'nbr{t}', wl, arm, t, kc)
                            if c['name'] in mine or any(c['name'] == x['name'] for x in out): continue
                            out.append(c)
        return out

    def run_case(self, c, slot):
        path = OUT / 'runs' / c['name']; path.mkdir(); proc = None; ticket = 'ai13b_' + c['name']; leased = False
        try:
            self.unchanged()
            cmd, args, inp = A13.command(c, path); save(path / 'command.json', cmd); save(path / 'case.json', c)
            while True:
                if self.stop.is_set(): raise RuntimeError('stopped')
                with broker.locked() as s:
                    if time.time() - s['heartbeat'] < 20 and not s['hold_new'] and not A13.pressure() and broker.total(s) + 10 * GIB <= s['cap']:
                        s['leases'][ticket] = dict(bytes=10 * GIB, owner=os.getpid(), family='AI-13b'); leased = True; break
                time.sleep(5)
            limit = [8 * GIB]
            def child():
                os.sched_setaffinity(0, A13.cpus(slot)); os.nice(10); resource.setrlimit(resource.RLIMIT_AS, (limit[0], resource.RLIM_INFINITY))
            with (path / 'run.log').open('w') as log:
                proc = subprocess.Popen(cmd, cwd=R, env=A13.ENV, stdout=log, stderr=subprocess.STDOUT, preexec_fn=child)
            start = time.monotonic(); paused = False; last = start; sig = None; paused_s = 0.0; pstart = None
            with self.lock: self.state['active'][c['name']] = dict(pid=proc.pid, cpus=A13.cpus(slot), started=time.time()); self.status()
            while proc.poll() is None:
                if self.stop.is_set(): raise RuntimeError('stopped')
                now = time.monotonic(); assert now - start - paused_s < A13.WALL_S, 'case wall timeout'
                assert shutil.disk_usage(OUT).free > 75 * GIB, 'disk guard'
                with (path / 'run.log').open('rb') as f:
                    f.seek(max(0, (path / 'run.log').stat().st_size - 16384)); tail = f.read().decode(errors='replace')
                now_sig = tuple(sorted(re.findall(r'system.cpu(\d+) progress event, total committed:(\d+)', tail)))
                if paused or now_sig != sig: last = now; sig = now_sig
                assert now - last < A13.NO_PROGRESS_S, 'no committed-instruction progress'
                hold = A13.pressure(); info = owned().get(proc.pid)
                if info and info['vms'] >= limit[0] - GIB:
                    with broker.locked() as s:
                        if not hold and broker.total(s) + 4 * GIB <= s['cap']:
                            s['leases'][ticket]['bytes'] += 4 * GIB; limit[0] += 4 * GIB; resource.prlimit(proc.pid, resource.RLIMIT_AS, (limit[0], resource.RLIM_INFINITY))
                        else: hold = True
                if hold != paused:
                    proc.send_signal(signal.SIGSTOP if hold else signal.SIGCONT); paused = hold
                    if hold: pstart = now
                    else: paused_s += now - pstart
                with self.lock: self.state['active'][c['name']].update(paused=paused); self.status()
                try: proc.wait(timeout=10)
                except subprocess.TimeoutExpired: pass
            assert proc.returncode == 0, f'simulator exit {proc.returncode}'
            row = A13.validate(c, path, inp, args); self.unchanged()
            row.update(name=c['name'], case=c, path=str(path), host_seconds=time.monotonic() - start)
            save(path / 'accepted.json', row)
            with self.lock: self.state['completed'].append(row); self.status()
        except BaseException as exc:
            save(path / 'failed.json', dict(error=repr(exc), traceback=traceback.format_exc()))
            with self.lock: self.state['failed'].append(dict(name=c['name'], case=c, error=repr(exc))); self.status()
        finally:
            if proc and proc.poll() is None:
                proc.send_signal(signal.SIGCONT); proc.terminate()
                try: proc.wait(timeout=20)
                except subprocess.TimeoutExpired: proc.kill(); proc.wait()
            if leased: broker.release(ticket)
            with self.lock: self.state['active'].pop(c['name'], None); self.free.append(slot); self.status()

    def main(self):
        signal.signal(signal.SIGTERM, lambda *a: self.stop.set())
        A13.EXPECTED = A13.expected_hashes()
        try:
            while not self.stop.is_set():
                a13 = read(A13_STATUS)
                for c in self.needed(a13):
                    with self.lock:
                        self.state['pending'].append(c['name']); self.state.setdefault('queue', []).append(c)
                        self.state['decisions'].append(dict(time=time.time(), add=c['name']))
                with self.lock:
                    while self.state.get('queue') and self.free:
                        c = max(self.state['queue'], key=lambda c: (c['wl'] == 'ffn', c['t']))    # long jobs first
                        self.state['queue'].remove(c); self.state['pending'].remove(c['name']); slot = self.free.pop(0)
                        self.state['active'][c['name']] = dict(stage='starting')
                        threading.Thread(target=self.run_case, args=(c, slot), daemon=True).start()
                    idle = not self.state.get('queue') and not self.state['active']
                self.status()
                if idle and a13['stage'] not in ('workloads', 'replay_gate', 'preparing') and not self.needed(a13): break
                time.sleep(60)
            self.status('complete' if not self.state['failed'] else 'complete_with_failures')
            with (R / 'docs/experiments/LEDGER.md').open('a') as f:
                f.write(f"\n**AI-13b neighbor queue finished ({self.state['stage']}):** {len(self.state['completed'])} accepted, "
                        f"{len(self.state['failed'])} failed. Merged analysis: benchmarks/qwen/components/mshr16_neighbors/analyze.py.\n")
        except BaseException as exc:
            self.state['error'] = repr(exc); self.state['traceback'] = traceback.format_exc(); self.stop.set(); self.status('failed'); raise


if __name__ == '__main__':
    Followup().main()
