"""FA-51b: close the T1024->T4096 transfer gap found in FA-51.

FA-51's T1024 descent moved every backend/credit cell to R128, but at T4096 R256 is
faster where both were measured (8 credits, flag 15: cache 10164.6 vs 10687.4 us; SPM
8952.1 vs 9499.6 us). 16-credit Bc256/R256 at T4096 was never run. Flag 2 alone is
slower at T4096, so flag 13 (1+4+8) is also measured. Same guest, acceptance and
resource rules as FA-51 (Tuner.run reused); new status file in followup/.
"""
import json, sys, threading, time, traceback
from pathlib import Path
H = Path(__file__).resolve().parent; sys.path.insert(0, str(H))
import tune as T

OUT = T.ROOT / 'followup'


class Followup(T.Tuner):
    def __init__(self):
        OUT.mkdir(exist_ok=False)
        self.lockf = (OUT / 'coordinator.lock').open('w'); T.fcntl.flock(self.lockf, T.fcntl.LOCK_EX | T.fcntl.LOCK_NB)
        T.os.sched_setaffinity(0, T.cpus(T.COORD)[:1]); T.os.nice(10)
        self.lock = threading.RLock(); self.ref_lock = threading.Lock(); self.free = list(T.SLOTS); self.stop = threading.Event()
        self.done = {}; self.inflight = {}
        prior = json.loads((T.ROOT / 'status.json').read_text()); assert prior['stage'].startswith('complete')
        self.bits = T.matrix_bits()
        for r in prior['completed']: assert self.bits.setdefault((r['case']['t'], r['case']['bc']), r['output_sha256']) == r['output_sha256']
        self.frozen = json.loads((T.ROOT / 'inputs.json').read_text()); self.frozen[str(Path(__file__))] = T.sha(Path(__file__))
        for p, h in self.frozen.items(): assert T.sha(p) == h, ('frozen drift at start', p)
        T.save(OUT / 'inputs.json', self.frozen)
        self.state = dict(pid=T.os.getpid(), stage='running', active={}, pending=[], completed=[], failed=[],
                          scope='FA-51b: T4096 Bc256/R256 at 16 credits (f0, f13, f15) and 8 credits (f13); both backends')

    def status(self, stage=None):
        with self.lock:
            if stage: self.state['stage'] = stage
            self.state['updated'] = time.time(); T.save(OUT / 'status.json', self.state)

    def main(self):
        try:
            self.status()
            cases = [T.case(4096, spm, 256, 256, f, 16, 'followup') for f in (0, 15, 13) for spm in (False, True)]
            cases += [T.case(4096, spm, 256, 256, 13, 8, 'followup') for spm in (False, True)]
            self.submit(cases)
            self.status('complete_requires_review' if not self.state['failed'] else 'complete_with_failures')
            with (T.R / 'docs/experiments/LEDGER.md').open('a') as f:
                f.write(f"\n**FA-51b finished ({self.state['stage']}):** {len(self.state['completed'])} accepted, {len(self.state['failed'])} failed; "
                        "results/cacheflex_flash_attention/softmax_tune_20261005/followup/status.json.\n")
        except BaseException as e:
            self.state['error'] = repr(e); self.state['traceback'] = traceback.format_exc(); self.stop.set(); self.status('failed'); raise


if __name__ == '__main__':
    Followup().main()
