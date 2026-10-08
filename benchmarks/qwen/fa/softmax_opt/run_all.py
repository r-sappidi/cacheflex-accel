"""FA-50 launcher: freeze inputs, compute oracle references once, run four lanes, report."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
import fcntl, hashlib, json, subprocess, sys, time, resource
from pathlib import Path
R = Path(__file__).resolve().parents[4]; H = Path(__file__).resolve().parent
sys.path[:0] = [str(H), str(R / 'scripts'), str(R / 'benchmarks/cacheflex_flash_attention'), str(R / 'results/parallel_overlap_20260930')]
import numpy as np
import plan as P
ROOT = R / 'results/cacheflex_flash_attention/softmax_opt_20261005'; G = ROOT / 'guest_v1'
PARENT = R / 'results/retained_protocol_restart_20261002/fa_matrix_spm_only/runs/t256_q14_legacy_v_br16_bc256_r256'
BUDGET = json.loads((R / 'results/parallel_budget_20260929/budget.json').read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def save(p, v): tmp = Path(p).with_suffix('.tmp'); tmp.write_text(json.dumps(v, indent=2) + '\n'); tmp.replace(p)


def report():
    rows = []
    for lane in range(4):
        p = ROOT / f'lane{lane}/status.json'
        if p.exists(): rows += json.loads(p.read_text())['completed']
    by = {(r['case']['t'], r['case']['spm'], r['case']['flags']): r for r in rows}
    meaning = {0: 'control', 1: 'four-row softmax', 2: 'skip unit alpha', 4: 'QK-epilogue max', 8: 'mask folded',
               15: 'combined exact (1+2+4+8)', 16: 'Q pre-scale', 31: 'all (15+16)'}
    lines = ['# FA-50: softmax/PV software optimizations (one ELF, runtime flags)', '',
             'Full independent oracle, FA-39 work/copy/config/LSQ checks on every row. Exact = output bits identical',
             'to accepted FA-39 controls. Reduction is versus flag 0 on the same backend and size.', '',
             '| T | Backend | Flags | Meaning | ROI us | Reduction vs f0 | Exact bits |', '|---:|---|---:|---|---:|---:|---|']
    for (t, spm, f), r in sorted(by.items()):
        base = by.get((t, spm, 0)); red = (1 - r['roi_us'] / base['roi_us']) if base else float('nan')
        lines.append(f"| {t} | {'SPM' if spm else 'cache'} | {f} | {meaning[f]} | {r['roi_us']:.4f} | {100 * red:.2f}% | {r['bit_exact_vs_fa39']} |")
    for t in (256, 1024):
        for f in P.TIMING:
            c, s = by.get((t, False, f)), by.get((t, True, f))
            if c and s: lines.append(f"\nT{t} flags {f}: cache/SPM {c['roi_us'] / s['roi_us']:.4f}x")
    failed = []
    for lane in range(4):
        p = ROOT / f'lane{lane}/status.json'
        if p.exists(): failed += json.loads(p.read_text())['failed']
    lines += ['', 'Failed/skipped: ' + (json.dumps([dict(name=x.get('name'), error=x.get('error')) for x in failed]) if failed else 'none'),
              '', 'Gates (T256) are correctness checks; T1024 is the first timing size. No T4096 claim yet.']
    (ROOT / 'REPORT.md').write_text('\n'.join(lines) + '\n'); return by, failed


def main():
    lock = (ROOT / 'launcher.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.sched_setaffinity(0, BUDGET['cpu_set'][P.LANE_OFFSETS[0]:P.LANE_OFFSETS[0] + 1]); os.nice(10)
    assert not (ROOT / 'inputs.json').exists(), 'already launched'
    from qwen_l3_reference import reference
    (ROOT / 'references').mkdir()
    for t in (256, 1024): np.save(ROOT / 'references' / f't{t}.npy', reference(t, 256, 14, 2, True))
    frozen = json.loads((G / 'frozen.json').read_text())
    frozen.update(json.loads((R / 'results/retained_protocol_restart_20261002/build/frozen.json').read_text()))
    cmd = json.loads((PARENT / 'command.json').read_text())
    for f in [*H.glob('*.py'), *PARENT.glob('*.json'), Path(cmd[0]), Path(cmd[2]), *(ROOT / 'references').glob('*.npy')]:
        if f.is_file(): frozen[str(f)] = sha(f)
    import broker, run_access_timing, fa_headroom_profiles, qwen_l3_reference, fa_independent_reference, export_safe_elf, test_micro_timing, parallel_budget_scheduler
    for m in (broker, run_access_timing, fa_headroom_profiles, qwen_l3_reference, fa_independent_reference, export_safe_elf, test_micro_timing, parallel_budget_scheduler):
        frozen[str(Path(m.__file__))] = sha(Path(m.__file__))
    for p, h in frozen.items(): assert sha(p) == h, ('frozen drift at launch', p)
    save(ROOT / 'inputs.json', frozen)
    save(ROOT / 'plan.json', dict(cases=P.cases(), lanes=[[c['name'] for c in l] for l in P.lanes()], reference_sha256=P.REFERENCE_SHA256))
    state = dict(pid=os.getpid(), stage='running', lanes={})
    procs = {}
    for lane in range(4):
        log = (ROOT / f'lane{lane}.log').open('w')
        procs[lane] = subprocess.Popen([sys.executable, str(H / 'run.py')], cwd=R, env=dict(os.environ, FA50_LANE=str(lane)), stdout=log, stderr=subprocess.STDOUT)
        state['lanes'][lane] = procs[lane].pid
    save(ROOT / 'status.json', state)
    while any(p.poll() is None for p in procs.values()):
        report(); time.sleep(30)
    by, failed = report()
    state['stage'] = 'complete_requires_review' if not failed else 'complete_with_failures'
    state['exit_codes'] = {k: p.returncode for k, p in procs.items()}; save(ROOT / 'status.json', state)
    with (R / 'docs/experiments/LEDGER.md').open('a') as f:
        f.write(f"\n**FA-50 queue finished ({state['stage']}):** {len(by)} accepted, {len(failed)} failed/skipped. "
                "Read results/cacheflex_flash_attention/softmax_opt_20261005/REPORT.md; scientific review required.\n")


if __name__ == '__main__':
    main()
