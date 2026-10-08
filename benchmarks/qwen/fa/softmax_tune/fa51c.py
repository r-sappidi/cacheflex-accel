"""FA-51c: closing runs after OPS-25 reboot (user-authorized 2026-10-06).

mode=qwen : T4096 Qwen SPM Bc256/R256 flag 15 at 16 credits (the missing tuned SPM cell).
mode=llama: T4096 Llama-3.2-1B shape (32Q/8KV) Br16/Bc256/R256 (FA-41 winner), cache and
            SPM, flags 0 and 15, 16 credits.
Reuses FA-51 Tuner.run acceptance unchanged (tune.py frozen, not edited). Llama mode patches
only shape plumbing in this process: argv heads, work(hq=32), validate(hq=32,hkv=8), the
FA-41 frozen oracle file, and FA-41 output bits as the exact-bit reference (flags are exact).
"""
import functools, json, sys, threading, time, traceback
from pathlib import Path
H = Path(__file__).resolve().parent; sys.path.insert(0, str(H))
import tune as T
import numpy as np
MODE = sys.argv[1]; assert MODE in ('qwen', 'llama')
OUT = T.ROOT / f'fa51c_{MODE}'
QWEN_REF = T.ROOT / 'references/t4096_bc256.npy'
LLAMA = T.R / 'results/llama_fa_iterative_20261004/experiment_v1'
LLAMA_REF = T.R / 'results/ordered_fa_followup_20261005/llama/references/t4096_bc256_q32_kv8.npy'
LLAMA_BITS = None
if MODE == 'llama':
    _argv, _work, _validate = T.argv, T.work, T.validate
    def argv(c):
        a = _argv(c); a[10], a[11] = 32, 8; return a
    T.argv = argv
    T.work = functools.partial(_work, hq=32)
    T.validate = lambda p, e, t, hq, hkv, causal, dense=False: _validate(p, e, t, 32, 8, causal, dense=dense)
    LLAMA_BITS = json.loads((LLAMA / 'runs/t4096_cache_br16_bc256_r256/accepted.json').read_text())['output_sha256']
    assert LLAMA_BITS == json.loads((LLAMA / 'runs/t4096_spm_br16_bc256_r256/accepted.json').read_text())['output_sha256']


def cases():
    if MODE == 'qwen': return [T.case(4096, True, 256, 256, 15, 16, 'fa51c')]
    out = []
    for f in (15, 0):
        for spm in (False, True):
            c = T.case(4096, spm, 256, 256, f, 16, 'fa51c_llama'); c['name'] = 'llama_' + c['name']; c['hq'], c['hkv'] = 32, 8; out.append(c)
    return out


class Runner(T.Tuner):
    def __init__(self, slots):
        OUT.mkdir(exist_ok=False); (OUT / 'runs').mkdir()
        self.lockf = (OUT / 'coordinator.lock').open('w'); T.fcntl.flock(self.lockf, T.fcntl.LOCK_EX | T.fcntl.LOCK_NB)
        T.os.sched_setaffinity(0, T.cpus(T.COORD)[:1]); T.os.nice(10)
        self.lock = threading.RLock(); self.ref_lock = threading.Lock(); self.free = list(slots); self.stop = threading.Event()
        self.done = {}; self.inflight = {}
        if MODE == 'qwen':
            self.bits = T.matrix_bits()
            for r in json.loads((T.ROOT / 'status.json').read_text())['completed']:
                assert self.bits.setdefault((r['case']['t'], r['case']['bc']), r['output_sha256']) == r['output_sha256']
        else:
            self.bits = {(4096, 256): LLAMA_BITS}
        self.frozen = json.loads((T.ROOT / 'inputs.json').read_text())
        extra = [Path(__file__), QWEN_REF] + ([LLAMA_REF, LLAMA / 'runs/t4096_cache_br16_bc256_r256/accepted.json'] if MODE == 'llama' else [])
        for p in extra: self.frozen[str(p)] = T.sha(p)
        for p, h in self.frozen.items(): assert T.sha(p) == h, ('frozen drift at start', p)
        T.save(OUT / 'inputs.json', self.frozen)
        T.ROOT = OUT   # run directories under this mode's root (set once, before any thread)
        self.state = dict(pid=T.os.getpid(), stage='running', mode=MODE, active={}, pending=[], completed=[], failed=[])

    def status(self, stage=None):
        with self.lock:
            if stage: self.state['stage'] = stage
            self.state['updated'] = time.time(); T.save(OUT / 'status.json', self.state)

    def ref(self, t, bc):
        assert (t, bc) == (4096, 256); return np.load(LLAMA_REF if MODE == 'llama' else QWEN_REF)

    def main(self):
        try:
            self.status(); self.submit(cases())
            self.status('complete_requires_review' if not self.state['failed'] else 'complete_with_failures')
            with (T.R / 'docs/experiments/LEDGER.md').open('a') as f:
                f.write(f"\n**FA-51c {MODE} finished ({self.state['stage']}):** {len(self.state['completed'])} accepted, "
                        f"{len(self.state['failed'])} failed; {OUT.relative_to(T.R)}/status.json.\n")
        except BaseException as e:
            self.state['error'] = repr(e); self.state['traceback'] = traceback.format_exc(); self.stop.set(); self.status('failed'); raise


if __name__ == '__main__':
    Runner(json.loads(sys.argv[2])).main()
