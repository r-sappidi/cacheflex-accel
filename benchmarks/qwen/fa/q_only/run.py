"""FA-52: Q-only SPM placement versus matched cache and V-only controls (head layout).

T4096 Qwen, Br16/Bc256/R512 (FA-35 best head-layout geometry), flag 0, credits 8 and 16.
Modes in one ELF: 2 = cache control (no SPM ways), 12 = Q-only SPM, 3 = V-only SPM.
FA-51 Tuner.run acceptance reused unchanged; expected per-core work from FA-35 head records
(tasks 14, row updates 60928 per core; V-only copies from FA-35 head_v; Q-only 2*512*14).
Exact output bits 27f14c53.. (FA-35 head and legacy layouts agree).
"""
import json, sys, threading, time, traceback
from pathlib import Path
R = Path('/data2/rsappidi/cacheflex_mc'); sys.path.insert(0, str(R / 'benchmarks/qwen/fa/softmax_tune'))
import tune as T
import numpy as np
OUT = R / 'results/cacheflex_flash_attention/q_only_20261006'
T.G = OUT / 'guest_v1'
QREF = R / 'results/cacheflex_flash_attention/softmax_tune_20261005/references/t4096_bc256.npy'
FA35 = json.loads(T.MATRIX.read_text())
HV = [x for x in FA35['completed'] if x['name'] == 't4096_q14_head_v_br16_bc256_r512'][0]
TL = threading.local()
MODES = {2: ('cache', False), 12: ('qonly', True), 3: ('vonly', True)}
_argv = T.argv
def argv(c):
    a = _argv(c); a[13] = c['mode']; return a
T.argv = argv
def work(t, bc, rows, hq=14):
    c = TL.case
    copies = [0] * 8 if c['mode'] == 2 else [2 * 512 * 14] * 8 if c['mode'] == 12 else HV['work']['copies']
    return HV['work']['tasks'], HV['work']['updates'], copies
T.work = work


def cases():
    out = []
    for cr in (8, 16):
        for mode, (label, spm) in MODES.items():
            c = T.case(4096, spm, 256, 512, 0, cr, 'fa52'); c['mode'] = mode; c['name'] = f'head_{label}_t4096_bc256_r512_cr{cr}'; out.append(c)
    return out


class Runner(T.Tuner):
    def __init__(self, slots):
        (OUT / 'runs').mkdir()
        self.lockf = (OUT / 'coordinator.lock').open('w'); T.fcntl.flock(self.lockf, T.fcntl.LOCK_EX | T.fcntl.LOCK_NB)
        T.os.sched_setaffinity(0, T.cpus(T.COORD)[:1]); T.os.nice(10)
        self.lock = threading.RLock(); self.ref_lock = threading.Lock(); self.free = list(slots); self.stop = threading.Event()
        self.done = {}; self.inflight = {}
        self.bits = {(4096, 256): HV['output_sha256']}
        frozen = json.loads((T.G / 'frozen.json').read_text())
        frozen.update(json.loads((R / 'results/retained_protocol_restart_20261002/build/frozen.json').read_text()))
        cmd = json.loads((T.PARENT / 'command.json').read_text())
        for f in [Path(__file__), Path(T.__file__), QREF, T.MATRIX, Path(cmd[0]), Path(cmd[2]), *T.PARENT.glob('*.json')]: frozen[str(f)] = T.sha(f)
        for p, h in frozen.items(): assert T.sha(p) == h, ('frozen drift at start', p)
        self.frozen = frozen; T.save(OUT / 'inputs.json', frozen)
        T.ROOT = OUT
        self.state = dict(pid=T.os.getpid(), stage='running', active={}, pending=[], completed=[], failed=[],
                          scope='FA-52 Q-only vs cache vs V-only, head layout, one ELF; flag 0; credits 8/16')

    def ref(self, t, bc):
        assert (t, bc) == (4096, 256); return np.load(QREF)

    def run(self, c, off):
        TL.case = c; return super().run(c, off)

    def main(self):
        try:
            self.status(); self.submit(cases())
            self.status('complete_requires_review' if not self.state['failed'] else 'complete_with_failures')
            with (R / 'docs/experiments/LEDGER.md').open('a') as f:
                f.write(f"\n**FA-52 finished ({self.state['stage']}):** {len(self.state['completed'])} accepted, {len(self.state['failed'])} failed; "
                        "results/cacheflex_flash_attention/q_only_20261006/status.json.\n")
        except BaseException as e:
            self.state['error'] = repr(e); self.state['traceback'] = traceback.format_exc(); self.stop.set(); self.status('failed'); raise


if __name__ == '__main__':
    Runner(json.loads(sys.argv[1])).main()
