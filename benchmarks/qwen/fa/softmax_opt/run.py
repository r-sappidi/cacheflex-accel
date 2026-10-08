"""FA-50 lane runner (one simulator at a time per lane); FA-39 acceptance unchanged.

Env: FA50_LANE (0-3). Shared references are prepared and frozen by run_all.py.
"""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
import json, time, hashlib, subprocess, resource, signal, fcntl, traceback, re, sys
from pathlib import Path
R = Path(__file__).resolve().parents[4]; H = Path(__file__).resolve().parent
sys.path[:0] = [str(H), str(R / 'scripts'), str(R / 'benchmarks/cacheflex_flash_attention'), str(R / 'results/parallel_overlap_20260930')]
import broker, numpy as np
import plan as P
from run_access_timing import stats
from fa_headroom_profiles import leaves
from test_micro_timing import check_global_guard
from export_safe_elf import export_hash
from qwen_l3_reference import validate
from parallel_budget_scheduler import owned
ROOT = R / 'results/cacheflex_flash_attention/softmax_opt_20261005'; G = ROOT / 'guest_v1'; GIB = 2 ** 30
LANE = int(os.environ['FA50_LANE']); OUT = ROOT / f'lane{LANE}'
PARENT = R / 'results/retained_protocol_restart_20261002/fa_matrix_spm_only/runs/t256_q14_legacy_v_br16_bc256_r256'
ENV = dict(os.environ, LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib')
BUDGET = json.loads((R / 'results/parallel_budget_20260929/budget.json').read_text())


def read(p): return json.loads(Path(p).read_text())


def save(p, v):
    p = Path(p); tmp = p.with_suffix('.tmp'); tmp.write_text(json.dumps(v, indent=2) + '\n'); tmp.replace(p)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def pressure():
    monitor = read(R / 'results/retained_protocol_restart_20261002/spm_only_resources.json')
    mem = {l.split(':')[0]: int(l.split()[1]) * 1024 for l in Path('/proc/meminfo').read_text().splitlines()}
    rss = monitor['user_rss_bytes'] + sum(v['rss'] for v in owned().values()); cap = monitor['ram_limit_bytes']
    return time.time() - monitor['updated'] > 30 or rss > cap - 2 * GIB or mem['MemAvailable'] < 64 * GIB + max(0, cap - rss)


def command(c, path):
    cmd = read(PARENT / 'command.json'); cmd[1] = '--outdir=' + str(path)
    cmd[cmd.index('--cmd') + 1] = str(G / 'guest.export-safe'); cmd[cmd.index('--options') + 1] = ' '.join(map(str, P.argv(c)))
    cmd[cmd.index('--l1d_num_spm_ways=6')] = '--l1d_num_spm_ways=' + str(6 if c['spm'] else 0)
    for i, x in enumerate(cmd):
        if x.startswith('system.cpu[0].workload[0].cmd='): cmd[i] = 'system.cpu[0].workload[0].cmd=' + repr(['kv-fa', *map(str, P.argv(c))])
    for k, v in P.parameters(c['spm']).items():
        found = [i for i, x in enumerate(cmd) if x.startswith(k + '=')]
        if found:
            for i in found: cmd[i] = k + '=' + repr(v)
        else: cmd += ['-P', k + '=' + repr(v)]
    return cmd


def config_check(path, c, cmd):
    old = dict(leaves(read(PARENT / 'config.json'))); new = dict(leaves(read(path / 'config.json'))); expected = dict(old)
    for k, v in list(expected.items()):
        if isinstance(v, str): expected[k] = v.replace(str(PARENT), str(path))
    expected.update(P.parameters(c['spm'])); expected['system.cpu[0].workload[0].executable'] = str(G / 'guest.export-safe')
    for i, x in enumerate(['kv-fa', *map(str, P.argv(c))]): expected[f'system.cpu[0].workload[0].cmd[{i}]'] = x
    assert new == expected, {k: [expected.get(k), new.get(k)] for k in expected.keys() | new.keys() if expected.get(k) != new.get(k)}
    assert cmd[cmd.index('--cmd') + 1] == new['system.cpu[0].workload[0].executable']
    return {k: [old.get(k), new.get(k)] for k in old.keys() | new.keys() if old.get(k) != new.get(k)}


def gate_state(flags):
    """'pass' when both T256 gates for these flags are accepted, 'fail' if either failed."""
    ok = 0
    for lane in range(4):
        p = ROOT / f'lane{lane}/status.json'
        if not p.exists(): continue
        s = read(p)
        ok += sum(1 for r in s['completed'] if r['case']['stage'] == 'gate' and r['case']['flags'] == flags)
        if any(f.get('case', {}).get('stage') == 'gate' and f.get('case', {}).get('flags') == flags for f in s['failed']): return 'fail'
        if s['stage'] == 'failed' and any(n.startswith('t256_') and n.endswith(f'_f{flags}') for n in s.get('pending', [])): return 'fail'
    return 'pass' if ok == 2 else 'wait'


def main():
    def stop(signum, frame): raise KeyboardInterrupt('termination requested')
    signal.signal(signal.SIGTERM, stop)
    OUT.mkdir(parents=True, exist_ok=False); (OUT / 'runs').mkdir()
    lock = (OUT / 'coordinator.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    off = P.LANE_OFFSETS[LANE]; os.sched_setaffinity(0, BUDGET['cpu_set'][off:off + 2]); os.nice(10)
    resource.setrlimit(resource.RLIMIT_AS, (3 * GIB, resource.RLIM_INFINITY))
    plan = P.lanes()[LANE]
    state = dict(pid=os.getpid(), lane=LANE, stage='preparing', active={}, pending=[c['name'] for c in plan], completed=[], failed=[])
    def status(): state['updated'] = time.time(); save(OUT / 'status.json', state)
    status(); frozen = read(ROOT / 'inputs.json')
    def unchanged():
        for p, h in frozen.items(): assert sha(p) == h, ('frozen input drift', p)
    def acquire(ticket):
        while True:
            state['stage'] = 'waiting_for_reservation'; status()
            with broker.locked() as s:
                if time.time() - s['heartbeat'] < 20 and not s['hold_new'] and not pressure() and broker.total(s) + 10 * GIB <= s['cap']:
                    assert ticket not in s['leases']; s['leases'][ticket] = dict(bytes=10 * GIB, owner=os.getpid(), family='FA-50'); return
            time.sleep(5)
    try:
        for c in plan:
            if c['stage'] == 'timing':
                while (g := gate_state(c['flags'])) == 'wait': state['stage'] = 'waiting_for_gates_' + c['name']; status(); time.sleep(20)
                if g == 'fail':
                    state['failed'].append(dict(name=c['name'], case=c, error='skipped: gate failed')); state['pending'].remove(c['name']); status(); continue
            ticket = 'fa50_' + c['name']; acquire(ticket); path = OUT / 'runs' / c['name']; path.mkdir(); proc = None
            try:
                unchanged(); cmd = command(c, path); save(path / 'command.json', cmd); limit = 8 * GIB
                def child(): resource.setrlimit(resource.RLIMIT_AS, (limit, resource.RLIM_INFINITY))
                with (path / 'run.log').open('w') as log:
                    proc = subprocess.Popen(cmd, cwd=R, env=ENV, stdout=log, stderr=subprocess.STDOUT, preexec_fn=child)
                state['stage'] = 'running_' + c['name']; state['active'] = {c['name']: dict(pid=proc.pid, path=str(path), paused=False, simulator_as=limit)}; status()
                start = time.monotonic(); paused = False; last = start; sig = None
                while proc.poll() is None:
                    assert time.monotonic() - start < 24 * 3600, 'case timeout'
                    with (path / 'run.log').open('rb') as f:
                        f.seek(max(0, (path / 'run.log').stat().st_size - 16384)); tail = f.read().decode(errors='replace')
                    now_sig = tuple(sorted(re.findall(r'system.cpu(\d+) progress event, total committed:(\d+)', tail)))
                    if paused or now_sig != sig: last = time.monotonic(); sig = now_sig
                    assert time.monotonic() - last < 900, 'no committed-instruction progress for 900s'
                    hold = pressure(); info = owned().get(proc.pid)
                    if info and info['vms'] >= limit - GIB:
                        with broker.locked() as s:
                            if not hold and broker.total(s) + 4 * GIB <= s['cap']:
                                s['leases'][ticket]['bytes'] += 4 * GIB; limit += 4 * GIB; resource.prlimit(proc.pid, resource.RLIMIT_AS, (limit, resource.RLIM_INFINITY))
                            else: hold = True
                    if hold != paused: proc.send_signal(signal.SIGSTOP if hold else signal.SIGCONT); paused = hold
                    state['active'][c['name']].update(paused=paused, simulator_as=limit); status()
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
                ref = ROOT / 'references' / f't{c["t"]}.npy'
                export_hash(path / 'output.fp16', log)
                arithmetic = validate(path / 'output.fp16', np.load(ref), c['t'], 14, 2, True, dense=c['t'] <= 256)
                tasks, updates, expected = P.work(c['t'])
                actual = {int(i): (int(t), int(u)) for i, t, u in re.findall(r'GQA_WORK core=(\d+) tasks=(\d+) row_updates=(\d+)', log)}
                assert actual == {i: (tasks[i], updates[i]) for i in range(8)}, actual
                copies = sum(v for k, v in s.items() if k.endswith('.m_spmcpRequestsIssued'))
                assert copies == (expected if c['spm'] else 0) and copies == sum(v for k, v in s.items() if k.endswith('.m_spmcpRequestsCompleted'))
                assert sum(v for k, v in s.items() if k.endswith(('.reductionAccepted', '.l3ReductionStarts'))) == 0
                digest = sha(path / 'output.fp16'); exact = digest == P.REFERENCE_SHA256[c['t']]
                if c['flags'] in P.EXACT: assert exact, 'exact-preserving flags changed output bits'
                row = dict(name=c['name'], case=c, path=str(path), outcome='full_fa_pass', roi_us=s['simTicks'] / 1e6, arithmetic=arithmetic,
                           copies=copies, output_sha256=digest, bit_exact_vs_fa39=exact, config_delta=delta, host_seconds=time.monotonic() - start)
                unchanged(); save(path / 'accepted.json', row); state['completed'].append(row); state['pending'].remove(c['name']); state['active'] = {}; status()
            except BaseException as e:
                save(path / 'failed.json', dict(error=repr(e), traceback=traceback.format_exc()))
                state['failed'].append(dict(name=c['name'], case=c, error=repr(e))); state['pending'].remove(c['name']); state['active'] = {}; status()
                if isinstance(e, KeyboardInterrupt): raise
            finally:
                if proc is not None and proc.poll() is None:
                    proc.send_signal(signal.SIGCONT); proc.terminate()
                    try: proc.wait(timeout=10)
                    except subprocess.TimeoutExpired: proc.kill(); proc.wait()
                broker.release(ticket)
        state['stage'] = 'complete'; status()
    except BaseException as e:
        state['stage'] = 'failed'; state['active'] = {}; state['error'] = repr(e); status(); raise


if __name__ == '__main__':
    main()
