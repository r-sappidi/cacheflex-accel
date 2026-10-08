"""AI-13 coordinator: Qwen FFN/QKV/output cache-vs-SPM at 16 L1D miss credits.

Reuses the frozen AI-10/AI-11 commands, guests, inputs, simulator and complete
acceptance checks. The only allowed configuration delta beyond AI-10/AI-11 is
l0_cntrl{0..7}.dataMissCredits.capacity. Shared 20% CPU/RAM budget, shared
reservation broker, per-simulator address-space limit with admitted growth,
pause under pressure, progress watchdog. Failures are retained, never accepted.
"""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
import fcntl, json, re, resource, shutil, signal, subprocess, sys, threading, time, traceback
from pathlib import Path
H = Path(__file__).resolve().parent
R = Path('/data2/rsappidi/cacheflex_mc')
sys.path[:0] = [str(H), str(R / 'scripts'), str(R / 'benchmarks/ai_workloads'),
                str(R / 'benchmarks/cacheflex_flash_attention'), str(R / 'results/parallel_overlap_20260930')]
import numpy as np
import broker
import plan as P
import run_qwen_swiglu as FFN
import run_qwen_projection_v2 as PROJ
from build_access_timing import sha
from run_access_timing import stats
from test_micro_timing import check_global_guard
from fa_headroom_profiles import leaves
from sweep_mlp_study import epochs
from audit_mlp_math import audit as math_audit
from qwen_projection_reference import validate as proj_validate
from qwen_swiglu_model import describe as ffn_describe
from qwen_projection_model import describe as proj_describe
from parallel_budget_scheduler import owned

GIB = 2 ** 30
OUT = P.OUT
CONTROL = FFN.CONTROL
assert CONTROL == PROJ.CONTROL
ENV = FFN.ENV
BUDGET = json.loads((R / 'results/parallel_budget_20260929/budget.json').read_text())
MONITOR = R / 'results/retained_protocol_restart_20261002/spm_only_resources.json'
NO_PROGRESS_S = 900
WALL_S = 96 * 3600


def save(p, v):
    p = Path(p); tmp = p.with_name(p.name + '.tmp'); tmp.write_text(json.dumps(v, indent=2, default=str) + '\n'); tmp.replace(p)


def read(p):
    return json.loads(Path(p).read_text())


def cpus(slot):
    return BUDGET['cpu_set'][slot:slot + 2]


def pressure():
    m = read(MONITOR)
    mem = {l.split(':')[0]: int(l.split()[1]) * 1024 for l in Path('/proc/meminfo').read_text().splitlines()}
    rss = m['user_rss_bytes'] + sum(v['rss'] for v in owned().values()); cap = m['ram_limit_bytes']
    return time.time() - m['updated'] > 30 or rss > cap - 2 * GIB or mem['MemAvailable'] < 64 * GIB + max(0, cap - rss)


def guest(c):
    return (P.AI10 / 'guests/custom_g0' / c['arm']) if c['wl'] == 'ffn' else (P.AI11 / 'guest_v2' / c['arm'])


def command(c, path):
    inp = P.input_path(c['wl'], c['t'])
    if c['wl'] == 'ffn':
        cmd, args = FFN.command(path, guest(c), inp, c, 2)
    else:
        cmd, args = PROJ.command(path, guest(c), inp, c, 2)
    if c['credits'] != P.DEFAULT_CREDITS:
        for k, v in P.credit_parameters(c['credits']).items():
            cmd += ['-P', f'{k}={v}']
    return cmd, args, inp


def config_audit(path, c, inp, args):
    def flat(p): return dict(leaves(json.loads((p / 'config.json').read_text().replace(str(p), '<OUT>'))))
    a, b = flat(CONTROL), flat(path)
    delta = {k: [a.get(k), b.get(k)] for k in a.keys() | b.keys() if a.get(k) != b.get(k)}
    allowed = (r'system.readfile|system.cpu\[\d+\].workload\[0\].(cmd\[\d+\]|executable)'
               r'|system.ruby.l1_cntrl\d+.cache.num_spm_ways|system.ruby.l0_cntrl\d+.dataMissCredits.capacity')
    assert all(re.fullmatch(allowed, k) for k in delta), delta
    s = read(path / 'config.json')['system']
    assert s['cpu'][0]['workload'][0]['cmd'] == args
    assert s['cpu'][0]['workload'][0]['executable'] == str(guest(c)) and s['readfile'] == str(inp)
    for i in range(8):
        assert s['ruby'][f'l1_cntrl{i}']['cache']['num_spm_ways'] == (6 if c['arm'] == 'spm' else 0)
        assert s['ruby'][f'l2_cntrl{i}']['L2cache']['size'] == 524288
        assert not s['ruby'][f'l2_cntrl{i}']['L2cache']['spm_separate_backing']
        assert s['ruby'][f'l0_cntrl{i}']['dataMissCredits']['capacity'] == c['credits']
        assert s['ruby'][f'l0_cntrl{i}']['number_of_TBEs'] == 16
        assert s['ruby'][f'l0_cntrl{i}']['sequencer']['cacheflex_micro_targets'] == 16
        assert s['ruby'][f'l1_cntrl{i}']['missCredits']['capacity'] == 20
    return delta


def validate(c, path, inp, args):
    """AI-10 / AI-11 acceptance, unchanged, plus cross-experiment exact output bits."""
    text = (path / 'run.log').read_text()
    assert 'because exiting with last active thread context' in text, text[-3000:]
    assert not re.search(r'panic:|fatal:|Simulated exit code not 0|\bFAIL\b', text), text[-3000:]
    delta = config_audit(path, c, inp, args)
    tag = 'MLP_RESULT' if c['wl'] == 'ffn' else 'PROJ_RESULT'
    checks = [dict(re.findall(r'(\w+)=([^\s]+)', l)) for l in text.splitlines() if l.startswith(tag)]
    assert len(checks) == 2 and all(x['errors'] == x['nonfinite'] == '0' for x in checks)
    assert len({x['hash'] for x in checks}) == 1
    t = c['t']; data = np.fromfile(path / 'output.bin', dtype='<f2')
    assert data.size == t * P.COLUMNS[c['wl']] and np.isfinite(data).all()
    h = 1469598103934665603
    for x in data.view(np.uint16): h = ((h ^ int(x)) * 1099511628211) & ((1 << 64) - 1)
    assert f'{h:016x}' == checks[-1]['hash'], 'export bits differ from guest output'
    if c['wl'] == 'ffn':
        ref = np.load(inp.with_suffix('.stages.npz'))['y'].reshape(-1)
        error = np.abs(data.astype(float) - ref.astype(float)); assert np.all(error <= 2e-6 + .01 * np.abs(ref.astype(float)))
        numeric = math_audit(path, inp, t, 896, 4864); assert numeric['passed']
        structure = ffn_describe(t, 896, 4864, dict(mp=c['mp'], kc=c['kc'], g=0))
    else:
        numeric = proj_validate(path / 'output.bin', inp, t, c['wl'])
        structure = proj_describe(t, c['wl'], dict(mp=c['mp'], kc=c['kc']))
    timing = epochs(path, 2); copies = structure['totals']['copies'] if c['arm'] == 'spm' else 0
    assert all(e['copies_issued'] == e['copies_completed'] == copies for e in timing), (copies, timing)
    for ep in stats(path):
        check_global_guard(ep)
        for i in range(8):
            for domain in ('regular', 'spm'):
                for kind, cap in [('Load', 32), ('Store', 48)]:
                    assert ep[f'system.cpu{i}.{domain}Lsq0.boundedMax{kind}Occupancy'] <= cap
            for suffix in ('reductionAccepted', 'reductionCompleted'):
                assert ep[f'system.ruby.l0_cntrl{i}.sequencer.{suffix}'] == 0
            assert ep[f'system.ruby.l0_cntrl{i}.dataMissCredits.maxLive'] <= c['credits']
    output_hash = sha(path / 'output.bin')
    reference_hash = EXPECTED.get((c['wl'], t))
    if reference_hash: assert output_hash == reference_hash, 'output differs from accepted AI-10/AI-11 bits'
    return dict(epochs=timing, first_us=timing[0]['us'], repeat_us=timing[1]['us'], roi_us=timing[0]['us'],
                checks=checks, math=numeric, output_sha256=output_hash, copies=copies, structure=structure,
                config_delta=delta)


def expected_hashes():
    out = {}
    for root in (P.AI10 / 'experiment_v1/runs', P.AI11 / 'experiment_v2/runs'):
        for acc in root.glob('*/accepted.json'):
            a = read(acc); c = a['case']; wl = 'ffn' if root.parent.parent == P.AI10 else c['op']
            if wl == 'ffn' and (c.get('d'), c.get('f')) != (896, 4864): continue
            key = (wl, c['t'])
            assert out.setdefault(key, a['output_sha256']) == a['output_sha256'], ('old peers disagree', key)
    return out


EXPECTED = {}


class Coordinator:
    def __init__(self):
        OUT.mkdir(parents=True, exist_ok=False); (OUT / 'runs').mkdir()
        self.lockfile = (OUT / 'coordinator.lock').open('w'); fcntl.flock(self.lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.sched_setaffinity(0, cpus(P.COORDINATOR_SLOT)); os.nice(10)
        resource.setrlimit(resource.RLIMIT_AS, (4 * GIB, resource.RLIM_INFINITY))
        self.lock = threading.RLock(); self.stop = threading.Event(); self.failed_wl = set()
        self.state = dict(pid=os.getpid(), stage='preparing', active={}, pending=[], completed=[], failed=[],
                          selections={}, scope='AI-13 bounded MP8 KC axis; 16 L1D credits; no global-optimum claim')
        self.frozen = {}; self.jobs = {}; self.free = list(P.CPU_SLOTS)

    def status(self, stage=None):
        with self.lock:
            if stage: self.state['stage'] = stage
            self.state['updated'] = time.time(); save(OUT / 'status.json', self.state)

    def freeze(self):
        manifest = {}
        for p in (P.AI10 / 'inputs.json', P.AI10 / 'experiment_v1/inputs.json', P.AI11 / 'frozen_v2.json', P.AI11 / 'experiment_v2/inputs.json'):
            manifest.update(read(p))
        cmd = read(CONTROL / 'command.json')
        for p in [Path(cmd[0]), Path(cmd[2]), CONTROL / 'command.json', CONTROL / 'config.json', *H.glob('*.py'),
                  Path(broker.__file__), *(Path(sys.modules[m].__file__) for m in ('run_qwen_swiglu', 'run_qwen_projection_v2',
                  'run_access_timing', 'test_micro_timing', 'fa_headroom_profiles', 'sweep_mlp_study', 'audit_mlp_math',
                  'qwen_projection_reference', 'qwen_swiglu_model', 'qwen_projection_model', 'parallel_budget_scheduler'))]:
            manifest[str(p)] = sha(p)
        for wl in P.WORKLOADS:
            for t in (P.GATE_T, *P.SIZES):
                inp = P.input_path(wl, t)
                for q in ((inp, inp.with_suffix('.stages.npz')) if wl == 'ffn' else (inp, inp.with_suffix('.npz'))):
                    manifest[str(q)] = sha(q)
            for arm in P.ARMS:
                g = guest(dict(wl=wl, arm=arm)); manifest[str(g)] = sha(g)
        for p, h in manifest.items(): assert sha(p) == h, ('frozen input drift at start', p)
        self.frozen = manifest; save(OUT / 'inputs.json', manifest)

    def unchanged(self):
        for p, h in self.frozen.items(): assert sha(p) == h, ('frozen input drift', p)

    def lease(self, ticket):
        while True:
            if self.stop.is_set(): raise RuntimeError('stopped')
            with broker.locked() as s:
                if time.time() - s['heartbeat'] < 20 and not s['hold_new'] and not pressure() and broker.total(s) + 10 * GIB <= s['cap']:
                    assert ticket not in s['leases']
                    s['leases'][ticket] = dict(bytes=10 * GIB, owner=os.getpid(), family='AI-13'); return
            time.sleep(5)

    def run_case(self, c, slot):
        path = OUT / 'runs' / c['name']; path.mkdir(); proc = None; ticket = 'ai13_' + c['name']; leased = False
        try:
            self.unchanged()
            cmd, args, inp = command(c, path); save(path / 'command.json', cmd); save(path / 'case.json', c)
            if c['stage'] == 'replay':
                old = P.old_run(c['wl'], c['arm'], c['t'], c['kc'])
                oldcmd = read(old / 'command.json'); assert [x for x in oldcmd if not x.startswith('--outdir=')] == \
                    [x for x in cmd if not x.startswith('--outdir=')], 'replay command differs from accepted AI-11 command'
            with self.lock: self.state['active'][c['name']] = dict(stage='waiting_for_reservation', cpus=cpus(slot)); self.status()
            self.lease(ticket); leased = True
            limit = [8 * GIB]
            def child():
                os.sched_setaffinity(0, cpus(slot)); os.nice(10)
                resource.setrlimit(resource.RLIMIT_AS, (limit[0], resource.RLIM_INFINITY))
            with (path / 'run.log').open('w') as log:
                proc = subprocess.Popen(cmd, cwd=R, env=ENV, stdout=log, stderr=subprocess.STDOUT, preexec_fn=child)
            start = time.monotonic(); paused = False; last_change = start; last_sig = None; paused_s = 0.0; pause_start = None
            with self.lock: self.state['active'][c['name']] = dict(pid=proc.pid, cpus=cpus(slot), paused=False, simulator_as=limit[0], started=time.time()); self.status()
            while proc.poll() is None:
                if self.stop.is_set(): raise RuntimeError('coordinator stop')
                now = time.monotonic()
                assert now - start - paused_s < WALL_S, 'case wall timeout'
                assert shutil.disk_usage(OUT).free > 75 * GIB, 'disk guard'
                assert (path / 'run.log').stat().st_size < 256 * 2 ** 20, 'log guard'
                with (path / 'run.log').open('rb') as f:
                    f.seek(max(0, (path / 'run.log').stat().st_size - 16384)); tail = f.read().decode(errors='replace')
                sig = tuple(sorted({int(i): int(n) for i, n in re.findall(r'system.cpu(\d+) progress event, total committed:(\d+)', tail)}.items()))
                if paused or sig != last_sig: last_change = now; last_sig = sig
                assert now - last_change < NO_PROGRESS_S, f'no committed-instruction progress for {NO_PROGRESS_S}s'
                hold = pressure(); info = owned().get(proc.pid)
                if info and info['vms'] >= limit[0] - GIB:
                    with broker.locked() as s:
                        if not hold and broker.total(s) + 4 * GIB <= s['cap']:
                            s['leases'][ticket]['bytes'] += 4 * GIB; limit[0] += 4 * GIB
                            resource.prlimit(proc.pid, resource.RLIMIT_AS, (limit[0], resource.RLIM_INFINITY))
                        else: hold = True
                if hold != paused:
                    proc.send_signal(signal.SIGSTOP if hold else signal.SIGCONT); paused = hold
                    if hold: pause_start = now
                    else: paused_s += now - pause_start
                with self.lock: self.state['active'][c['name']].update(paused=paused, simulator_as=limit[0], committed=dict(last_sig or ())); self.status()
                try: proc.wait(timeout=10)
                except subprocess.TimeoutExpired: pass
            assert proc.returncode == 0, f'simulator exit {proc.returncode}'
            row = validate(c, path, inp, args)
            if c['stage'] == 'replay':
                old = read(P.old_run(c['wl'], c['arm'], c['t'], c['kc']) / 'accepted.json')
                assert (row['first_us'], row['repeat_us'], row['output_sha256']) == (old['first_us'], old['repeat_us'], old['output_sha256']), \
                    ('replay mismatch', row['first_us'], row['repeat_us'], old['first_us'], old['repeat_us'])
            self.unchanged()
            row.update(name=c['name'], case=c, path=str(path), host_seconds=time.monotonic() - start, paused_seconds=paused_s)
            with self.lock:
                peers = [r for r in self.state['completed'] if (r['case']['wl'], r['case']['t']) == (c['wl'], c['t'])]
                assert all(r['output_sha256'] == row['output_sha256'] for r in peers), 'peer output bits differ'
                save(path / 'accepted.json', row)
                self.state['completed'].append(row); self.status()
            return True
        except BaseException as exc:
            save(path / 'failed.json', dict(error=repr(exc), traceback=traceback.format_exc()))
            with self.lock: self.state['failed'].append(dict(name=c['name'], error=repr(exc))); self.status()
            return False
        finally:
            if proc and proc.poll() is None:
                proc.send_signal(signal.SIGCONT); proc.terminate()
                try: proc.wait(timeout=20)
                except subprocess.TimeoutExpired: proc.kill(); proc.wait()
            if leased: broker.release(ticket)
            with self.lock: self.state['active'].pop(c['name'], None); self.free.append(slot); self.status()

    def cost(self, c):
        return (3.0 if c['wl'] == 'ffn' else 0.3) * c['t'] / 1024

    def execute(self, cases):
        """Run cases on free CPU slots, longest-first; returns accepted flags by name."""
        queue = sorted(cases, key=lambda c: -self.cost(c)); threads = {}; result = {}
        with self.lock: self.state['pending'] += [c['name'] for c in queue]; self.status()
        while queue or threads:
            while queue and not self.stop.is_set():
                c = queue[0]
                if c['wl'] in self.failed_wl and c['stage'] != 'replay':
                    queue.pop(0)
                    with self.lock: self.state['pending'].remove(c['name']); self.state['failed'].append(dict(name=c['name'], error='skipped: workload failed earlier')); self.status()
                    result[c['name']] = False; continue
                with self.lock:
                    if not self.free: break
                    slot = self.free.pop(0); queue.pop(0); self.state['pending'].remove(c['name'])
                th = threading.Thread(target=lambda c=c, slot=slot: result.__setitem__(c['name'], self.run_case(c, slot)), daemon=True)
                th.start(); threads[c['name']] = (th, c)
            for n, (th, c) in list(threads.items()):
                if not th.is_alive():
                    del threads[n]
                    if not result.get(n): self.failed_wl.add(c['wl'])
            if self.stop.is_set() and not threads: break
            time.sleep(2)
        return result

    def workload(self, wl):
        if not all(self.execute(P.gate_cases(wl)).values()): return
        if not all(self.execute(P.screen_cases(wl)).values()): return
        rows = [r for r in self.state['completed'] if r['case']['wl'] == wl]
        selected = {arm: P.select(rows, wl, arm) for arm in P.ARMS}
        with self.lock: self.state['selections'][wl] = selected; self.status()
        for t in (2048, 4096):
            if not all(self.execute(P.target_cases(wl, t, selected)).values()): return

    def main(self):
        global EXPECTED
        def term(signum, frame): self.stop.set()
        signal.signal(signal.SIGTERM, term)
        try:
            self.status(); self.freeze(); EXPECTED = expected_hashes()
            save(OUT / 'plan.json', dict(workloads=P.WORKLOADS, kcs=P.KCS, mp=P.MP, credits=P.CREDITS, sizes=P.SIZES,
                 gates=P.GATE_T, cpu_slots=P.CPU_SLOTS, max_simulators=len(P.CPU_SLOTS), selection='per backend: T1024 winner by first-call and by repeat-call ROI (union)',
                 expected_output_sha256={f'{k[0]}_t{k[1]}': v for k, v in EXPECTED.items()}))
            self.status('replay_gate')
            res = self.execute(P.replay_cases())
            assert res and all(res.values()), 'replay gate failed: platform does not reproduce AI-11 exactly'
            self.status('workloads')
            ths = [threading.Thread(target=self.workload, args=(wl,), daemon=True) for wl in P.WORKLOADS]
            # Workload pipelines share the slot pool; execute() is serialized per workload but slots are shared.
            for th in ths: th.start()
            for th in ths: th.join()
            done = 'stopped' if self.stop.is_set() else ('completed_with_failures' if self.state['failed'] else 'completed_validated')
            self.status(done)
            subprocess.run([sys.executable, str(H / 'analyze.py')], cwd=R, check=False,
                           stdout=(OUT / 'analysis.log').open('w'), stderr=subprocess.STDOUT)
            with (R / 'docs/experiments/LEDGER.md').open('a') as f:
                f.write(f"\n**AI-13 coordinator finished ({done}):** {len(self.state['completed'])} accepted, "
                        f"{len(self.state['failed'])} failed/skipped. Read results/ai_workloads/qwen_mshr16_20261005/REPORT.md; scientific review required.\n")
        except BaseException as exc:
            self.state['error'] = repr(exc); self.state['traceback'] = traceback.format_exc(); self.stop.set(); self.status('failed'); raise


if __name__ == '__main__':
    Coordinator().main()
