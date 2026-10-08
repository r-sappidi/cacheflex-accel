"""Fresh paired study on a frozen LSQ-repaired simulator after regression.

Usage: resume_mlp_lsq_fix.py NEW_OUT SIMULATOR REGRESSION_DIR [REGRESSION_PID]
Prior results/builds are preserved; no old timing is pooled into this study.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
out, sim, regression = map(lambda s: Path(s).resolve(), sys.argv[1:4])
out.mkdir()
state = dict(stage='starting', completed=[], failed=None)
env = dict(os.environ, OPENBLAS_NUM_THREADS='1', MLP_GEM5_BIN=str(out / 'gem5.opt'))

def status(stage):
    state['stage'] = stage
    (out / 'status.json').write_text(json.dumps(state, indent=2))
    print(stage, flush=True)

def run(name, args, extra=None):
    status(name)
    with (out / f'{name}.host.log').open('w') as log:
        subprocess.run(args, env=dict(env, **(extra or {})), stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    state['completed'].append(name)
    status(name + '_complete')

try:
    status('waiting_for_regression')
    deadline = time.monotonic() + 3 * 3600
    while not (regression / 'summary.json').exists():
        assert len(sys.argv) > 4, 'Regression must finish first'
        os.kill(int(sys.argv[4]), 0)
        if time.monotonic() > deadline:
            raise TimeoutError('regression did not complete within three hours')
        time.sleep(10)
    expected = json.loads((regression / 'hashes.json').read_text())['gem5.opt']
    assert hashlib.sha256(sim.read_bytes()).hexdigest() == expected
    shutil.copy2(sim, out / 'gem5.opt')
    shutil.copy2(regression / 'summary.json', out / 'regression_summary.json')
    shutil.copy2(ROOT / 'gem5/src/cpu/o3/lsq_unit.cc', out / 'lsq_unit.cc')
    (out / 'simulator.sha256').write_text(expected + '\n')
    for rows in (24, 8, 16):
        run(f'build_r{rows}', ['bash', str(HERE / 'build_mlp_study.sh'),
                             str(out / f'build_r{rows}')],
            dict(MLP_ROWS=str(rows), MLP_COMBINED='0', MLP_ACC32='0'))
    for stage, name in ((1, 'stage1'), (11, 'stage1_convergence'), (2, 'stage2_r24')):
        run(name, [sys.executable, str(HERE / 'sweep_mlp_study.py'), str(stage),
                   str(out / name), str(out / 'build_r24')])
    run('ordered', [sys.executable, str(HERE / 'run_mlp_ordered.py'), str(out)])
    status('complete')
except BaseException as error:
    state['failed'] = repr(error)
    status('failed')
    raise
