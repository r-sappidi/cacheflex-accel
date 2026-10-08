"""FA-51: per-flag T4096 attribution and Bc/R retuning with the FA-50 software (flag 15).

Guest v2 = FA-50 guest with the FA-39 geometry lock removed (diff: 3 lines). Retained
simulator/configuration; only the guest, its argv geometry/flags, SPM ways and (16-credit
arms) l0 dataMissCredits change. Acceptance per run: FA-39/FA-50 checks (oracle for the
run's T/Bc, generalized per-core work, copy counts, LSQ/global guards, zero reductions,
exact config) plus exact output bits per (T, Bc): FA-35 reference where available, else
every accepted peer at that (T, Bc). Bits do not depend on R, Br, flags, backend or credits.
"""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
import fcntl, hashlib, json, math, re, resource, shutil, signal, subprocess, sys, threading, time, traceback
from pathlib import Path
R = Path('/data2/rsappidi/cacheflex_mc'); H = Path(__file__).resolve().parent
sys.path[:0] = [str(R / 'scripts'), str(R / 'benchmarks/cacheflex_flash_attention'), str(R / 'results/parallel_overlap_20260930')]
import numpy as np
import broker
from run_access_timing import stats
from fa_headroom_profiles import leaves
from test_micro_timing import check_global_guard
from export_safe_elf import export_hash
from qwen_l3_reference import reference, validate
from parallel_budget_scheduler import owned
ROOT = R / 'results/cacheflex_flash_attention/softmax_tune_20261005'; G = ROOT / 'guest_v1'; GIB = 2 ** 30
PARENT = R / 'results/retained_protocol_restart_20261002/fa_matrix_spm_only/runs/t256_q14_legacy_v_br16_bc256_r256'
MATRIX = R / 'results/retained_protocol_restart_20261002/fa_matrix_spm_only/status.json'
ENV = dict(os.environ, LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib')
BUDGET = json.loads((R / 'results/parallel_budget_20260929/budget.json').read_text())
SLOTS = (16, 18, 20); COORD = 22
BCS = (128, 256, 512); RS = (128, 256, 512); START = (256, 256)
ATTRIB_FLAGS = (0, 1, 2, 4, 8, 15); TUNED = 15; TIE = 0.005; MAX_MOVES = 4


def read(p): return json.loads(Path(p).read_text())
def save(p, v): p = Path(p); tmp = p.with_name(p.name + '.tmp'); tmp.write_text(json.dumps(v, indent=2, default=str) + '\n'); tmp.replace(p)
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()
def cpus(off): return BUDGET['cpu_set'][off:off + 2]


def case(t, spm, bc, r, flags, credits, stage):
    c = dict(t=t, spm=spm, bc=bc, r=r, flags=flags, credits=credits, stage=stage)
    c['name'] = f"t{t}_{'spm' if spm else 'cache'}_bc{bc}_r{r}_f{flags}_cr{credits}"; return c


def argv(c):
    s = int(c['spm']); return [c['t'], 8, 16, c['bc'], c['r'] // 16, 1024, 0, s, 1, 1, 14, 2, 1, s, 0, 0, c['flags']]


def parameters(c):
    p = {f'system.ruby.l1_cntrl{i}.cache.num_spm_ways': 6 if c['spm'] else 0 for i in range(8)}
    if c['credits'] != 8: p.update({f'system.ruby.l0_cntrl{i}.dataMissCredits.capacity': c['credits'] for i in range(8)})
    return p


def work(t, bc, rows, hq=14, cores=8):
    """Mirror of legacy_worker task loop: per-core tasks, row updates and SPM copies."""
    groups = (t + rows - 1) // rows; tasks = [0] * cores; updates = [0] * cores; copies = [0] * cores
    for task in range(hq * groups):
        core = task % cores; head = task // groups; qi = ((task % groups + head) % groups) * rows
        nr = min(rows, t - qi); kend = min(t, ((qi + nr + bc - 1) // bc) * bc)
        tasks[core] += 1; updates[core] += nr * ((kend + bc - 1) // bc); copies[core] += 2 * kend
    return tasks, updates, copies


def command(c, path):
    cmd = read(PARENT / 'command.json'); cmd[1] = '--outdir=' + str(path)
    cmd[cmd.index('--cmd') + 1] = str(G / 'guest.export-safe'); cmd[cmd.index('--options') + 1] = ' '.join(map(str, argv(c)))
    cmd[cmd.index('--l1d_num_spm_ways=6')] = '--l1d_num_spm_ways=' + str(6 if c['spm'] else 0)
    for i, x in enumerate(cmd):
        if x.startswith('system.cpu[0].workload[0].cmd='): cmd[i] = 'system.cpu[0].workload[0].cmd=' + repr(['kv-fa', *map(str, argv(c))])
    for k, v in parameters(c).items():
        found = [i for i, x in enumerate(cmd) if x.startswith(k + '=')]
        if found:
            for i in found: cmd[i] = k + '=' + repr(v)
        else: cmd += ['-P', k + '=' + repr(v)]
    return cmd


def config_check(path, c, cmd):
    old = dict(leaves(read(PARENT / 'config.json'))); new = dict(leaves(read(path / 'config.json'))); expected = dict(old)
    for k, v in list(expected.items()):
        if isinstance(v, str): expected[k] = v.replace(str(PARENT), str(path))
    expected.update(parameters(c)); expected['system.cpu[0].workload[0].executable'] = str(G / 'guest.export-safe')
    for i, x in enumerate(['kv-fa', *map(str, argv(c))]): expected[f'system.cpu[0].workload[0].cmd[{i}]'] = x
    assert new == expected, {k: [expected.get(k), new.get(k)] for k in expected.keys() | new.keys() if expected.get(k) != new.get(k)}
    assert cmd[cmd.index('--cmd') + 1] == new['system.cpu[0].workload[0].executable']
    return {k: [old.get(k), new.get(k)] for k in old.keys() | new.keys() if old.get(k) != new.get(k)}


def pressure():
    m = read(R / 'results/retained_protocol_restart_20261002/spm_only_resources.json')
    mem = {l.split(':')[0]: int(l.split()[1]) * 1024 for l in Path('/proc/meminfo').read_text().splitlines()}
    rss = m['user_rss_bytes'] + sum(v['rss'] for v in owned().values()); cap = m['ram_limit_bytes']
    return time.time() - m['updated'] > 30 or rss > cap - 2 * GIB or mem['MemAvailable'] < 64 * GIB + max(0, cap - rss)


def matrix_bits():
    out = {}
    for x in read(MATRIX)['completed']:
        c = x['case']
        if c['mode'] == 1 and c['hq'] == 14:
            k = (c['t'], c['bc']); assert out.setdefault(k, x['output_sha256']) == x['output_sha256'], ('FA-35 peers disagree', k)
    return out


class Tuner:
    def __init__(self):
        (ROOT / 'runs').mkdir(); (ROOT / 'references').mkdir()
        self.lockf = (ROOT / 'coordinator.lock').open('w'); fcntl.flock(self.lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.sched_setaffinity(0, cpus(COORD)[:1]); os.nice(10); resource.setrlimit(resource.RLIMIT_AS, (6 * GIB, resource.RLIM_INFINITY))
        self.lock = threading.RLock(); self.ref_lock = threading.Lock(); self.free = list(SLOTS); self.stop = threading.Event()
        self.done = {}; self.inflight = {}
        self.state = dict(pid=os.getpid(), stage='preparing', active={}, pending=[], completed=[], failed=[], descent={}, confirm={},
                          scope='FA-51 bounded Bc/R descent (Br16), flag 15, credits 8/16; no global optimum claim')
        self.bits = matrix_bits()
        frozen = read(G / 'frozen.json'); frozen.update(read(R / 'results/retained_protocol_restart_20261002/build/frozen.json'))
        cmd = read(PARENT / 'command.json')
        for f in [*H.glob('*.py'), *PARENT.glob('*.json'), Path(cmd[0]), Path(cmd[2]), MATRIX]: frozen[str(f)] = sha(f)
        import run_access_timing, fa_headroom_profiles, qwen_l3_reference, fa_independent_reference, export_safe_elf, test_micro_timing, parallel_budget_scheduler
        for m in (broker, run_access_timing, fa_headroom_profiles, qwen_l3_reference, fa_independent_reference, export_safe_elf, test_micro_timing, parallel_budget_scheduler):
            frozen[str(Path(m.__file__))] = sha(Path(m.__file__))
        for p, h in frozen.items(): assert sha(p) == h, ('frozen drift at start', p)
        self.frozen = frozen; save(ROOT / 'inputs.json', frozen)

    def status(self, stage=None):
        with self.lock:
            if stage: self.state['stage'] = stage
            self.state['updated'] = time.time(); save(ROOT / 'status.json', self.state)

    def unchanged(self):
        for p, h in list(self.frozen.items()): assert sha(p) == h, ('frozen input drift', p)

    def ref(self, t, bc):
        with self.ref_lock:
            p = ROOT / 'references' / f't{t}_bc{bc}.npy'
            if not p.exists():
                np.save(p, reference(t, bc, 14, 2, True))
                with self.lock: self.frozen[str(p)] = sha(p); save(ROOT / 'inputs.json', self.frozen)
            return np.load(p)

    def run(self, c, off):
        path = ROOT / 'runs' / c['name']; path.mkdir(); proc = None; ticket = 'fa51_' + c['name']; leased = False
        try:
            expected_ref = self.ref(c['t'], c['bc'])
            self.unchanged(); cmd = command(c, path); save(path / 'command.json', cmd); save(path / 'case.json', c)
            while True:
                if self.stop.is_set(): raise RuntimeError('stopped')
                with broker.locked() as s:
                    if time.time() - s['heartbeat'] < 20 and not s['hold_new'] and not pressure() and broker.total(s) + 10 * GIB <= s['cap']:
                        s['leases'][ticket] = dict(bytes=10 * GIB, owner=os.getpid(), family='FA-51'); leased = True; break
                time.sleep(5)
            limit = [8 * GIB]
            def child():
                os.sched_setaffinity(0, cpus(off)); os.nice(10); resource.setrlimit(resource.RLIMIT_AS, (limit[0], resource.RLIM_INFINITY))
            with (path / 'run.log').open('w') as log:
                proc = subprocess.Popen(cmd, cwd=R, env=ENV, stdout=log, stderr=subprocess.STDOUT, preexec_fn=child)
            start = time.monotonic(); paused = False; last = start; sig = None
            with self.lock: self.state['active'][c['name']] = dict(pid=proc.pid, cpus=cpus(off), started=time.time()); self.status()
            while proc.poll() is None:
                if self.stop.is_set(): raise RuntimeError('stopped')
                assert time.monotonic() - start < 24 * 3600, 'case timeout'
                with (path / 'run.log').open('rb') as f:
                    f.seek(max(0, (path / 'run.log').stat().st_size - 16384)); tail = f.read().decode(errors='replace')
                now = tuple(sorted(re.findall(r'system.cpu(\d+) progress event, total committed:(\d+)', tail)))
                if paused or now != sig: last = time.monotonic(); sig = now
                assert time.monotonic() - last < 900, 'no committed-instruction progress for 900s'
                hold = pressure(); info = owned().get(proc.pid)
                if info and info['vms'] >= limit[0] - GIB:
                    with broker.locked() as s:
                        if not hold and broker.total(s) + 4 * GIB <= s['cap']:
                            s['leases'][ticket]['bytes'] += 4 * GIB; limit[0] += 4 * GIB; resource.prlimit(proc.pid, resource.RLIMIT_AS, (limit[0], resource.RLIM_INFINITY))
                        else: hold = True
                if hold != paused: proc.send_signal(signal.SIGSTOP if hold else signal.SIGCONT); paused = hold
                try: proc.wait(timeout=5)
                except subprocess.TimeoutExpired: pass
            log = (path / 'run.log').read_text()
            assert proc.returncode == 0 and 'because exiting with last active thread context' in log, log[-3000:]
            assert not re.search(r'panic:|fatal:|\bFAIL\b', log), log[-3000:]
            assert f"SOFTMAX_OPT flags={c['flags']}" in log and 'BOTTLENECK stage_simple=0 packed_producer=0' in log
            delta = config_check(path, c, cmd); blocks = stats(path); s = blocks[0]
            for block in blocks:
                check_global_guard(block)
                for i in range(8):
                    for domain in ('regular', 'spm'):
                        for kind, cap in [('Load', 32), ('Store', 48)]: assert block[f'system.cpu{i}.{domain}Lsq0.boundedMax{kind}Occupancy'] <= cap
                    assert block[f'system.ruby.l0_cntrl{i}.dataMissCredits.maxLive'] <= c['credits']
            export_hash(path / 'output.fp16', log)
            arithmetic = validate(path / 'output.fp16', expected_ref, c['t'], 14, 2, True, dense=c['t'] <= 256)
            tasks, updates, copies = work(c['t'], c['bc'], c['r'])
            actual = {int(i): (int(t), int(u)) for i, t, u in re.findall(r'GQA_WORK core=(\d+) tasks=(\d+) row_updates=(\d+)', log)}
            assert actual == {i: (tasks[i], updates[i]) for i in range(8)}, actual
            issued = sum(v for k, v in s.items() if k.endswith('.m_spmcpRequestsIssued'))
            assert issued == (sum(copies) if c['spm'] else 0) and issued == sum(v for k, v in s.items() if k.endswith('.m_spmcpRequestsCompleted'))
            assert sum(v for k, v in s.items() if k.endswith(('.reductionAccepted', '.l3ReductionStarts'))) == 0
            digest = sha(path / 'output.fp16')
            with self.lock:
                key = (c['t'], c['bc']); ref = self.bits.setdefault(key, digest)
                assert digest == ref, ('output bits differ for same T/Bc', key)
            row = dict(name=c['name'], case=c, path=str(path), outcome='full_fa_pass', roi_us=s['simTicks'] / 1e6, arithmetic=arithmetic,
                       copies=issued, output_sha256=digest, config_delta=delta, host_seconds=time.monotonic() - start)
            self.unchanged(); save(path / 'accepted.json', row)
            with self.lock: self.state['completed'].append(row); self.done[c['name']] = row; self.status()
        except BaseException as e:
            save(path / 'failed.json', dict(error=repr(e), traceback=traceback.format_exc()))
            with self.lock: self.state['failed'].append(dict(name=c['name'], case=c, error=repr(e))); self.done[c['name']] = None; self.status()
        finally:
            if proc is not None and proc.poll() is None:
                proc.send_signal(signal.SIGCONT); proc.terminate()
                try: proc.wait(timeout=10)
                except subprocess.TimeoutExpired: proc.kill(); proc.wait()
            if leased: broker.release(ticket)
            with self.lock: self.state['active'].pop(c['name'], None); self.free.append(off); self.status()

    def submit(self, cases):
        """Run cases (deduplicated across stages) on shared slots; returns accepted rows or None."""
        mine = []
        for c in cases:
            with self.lock:
                if c['name'] in self.done or c['name'] in self.inflight: mine.append(c['name']); continue
                self.inflight[c['name']] = c; self.state['pending'].append(c['name']); mine.append(c['name'])
        while True:
            with self.lock:
                if all(n in self.done for n in mine): return {n: self.done[n] for n in mine}
                queue = [self.inflight[n] for n in self.state['pending'] if n in self.inflight]
                while queue and self.free and not self.stop.is_set():
                    c = max(queue, key=lambda c: c['t']); queue.remove(c)
                    self.state['pending'].remove(c['name']); off = self.free.pop(0); self.state['active'][c['name']] = dict(stage='starting')
                    threading.Thread(target=self.run, args=(c, off), daemon=True).start()
            if self.stop.is_set(): raise RuntimeError('stopped')
            time.sleep(10)

    def attribution(self):
        self.submit([case(4096, spm, 256, 256, f, 8, 'attrib') for f in ATTRIB_FLAGS for spm in (False, True)])

    def descend(self, spm, credits):
        key = f"{'spm' if spm else 'cache'}_cr{credits}"; center = START; path = [center]
        for _ in range(MAX_MOVES + 1):
            bi, ri = BCS.index(center[0]), RS.index(center[1])
            geos = {center} | {(BCS[j], center[1]) for j in (bi - 1, bi + 1) if 0 <= j < 3} | {(center[0], RS[j]) for j in (ri - 1, ri + 1) if 0 <= j < 3}
            res = self.submit([case(1024, spm, bc, r, TUNED, credits, 'descent') for bc, r in sorted(geos)])
            ok = {(row['case']['bc'], row['case']['r']): row['roi_us'] for row in res.values() if row}
            with self.lock: self.state['descent'][key] = dict(path=path, evaluated={f'{b}_{r}': v for (b, r), v in sorted(ok.items())}); self.status()
            if center not in ok: raise RuntimeError(f'descent center failed {key} {center}')
            best = min(ok, key=ok.get)
            if best != center and ok[best] < ok[center] * (1 - TIE): center = best; path.append(center); continue
            break
        ranked = sorted(ok, key=ok.get); runner = ranked[1] if ranked[0] == center and len(ranked) > 1 else ranked[0]
        res = self.submit([case(4096, spm, center[0], center[1], TUNED, credits, 'confirm'), case(4096, spm, runner[0], runner[1], TUNED, credits, 'confirm'),
                           case(4096, spm, center[0], center[1], 0, credits, 'confirm')])
        with self.lock:
            self.state['confirm'][key] = dict(winner_t1024=center, runner_t1024=runner, t4096={n: (r['roi_us'] if r else None) for n, r in res.items()}); self.status()

    def main(self):
        signal.signal(signal.SIGTERM, lambda *a: self.stop.set())
        try:
            self.status('running')
            # T1024 Bc128/Bc512 have no FA-35 bits: establish them with flag-0 controls first.
            self.submit([case(1024, False, bc, 256, 0, 8, 'bits') for bc in (128, 512)])
            def guarded(fn, *args):
                try: fn(*args)
                except BaseException as e:
                    with self.lock: self.state.setdefault('stage_errors', []).append(dict(stage=fn.__name__, args=repr(args), error=repr(e), traceback=traceback.format_exc())); self.status()
            jobs = [threading.Thread(target=guarded, args=(self.attribution,), daemon=True)] + \
                   [threading.Thread(target=guarded, args=(self.descend, spm, cr), daemon=True) for cr in (8, 16) for spm in (False, True)]
            for j in jobs: j.start()
            for j in jobs: j.join()
            self.status('complete_requires_review' if not self.state['failed'] else 'complete_with_failures')
            subprocess.run([sys.executable, str(H / 'report.py')], cwd=R, check=False)
            with (R / 'docs/experiments/LEDGER.md').open('a') as f:
                f.write(f"\n**FA-51 finished ({self.state['stage']}):** {len(self.state['completed'])} accepted, {len(self.state['failed'])} failed. "
                        "Read results/cacheflex_flash_attention/softmax_tune_20261005/REPORT.md; scientific review required.\n")
        except BaseException as e:
            self.state['error'] = repr(e); self.state['traceback'] = traceback.format_exc(); self.stop.set(); self.status('failed'); raise


if __name__ == '__main__':
    Tuner().main()
