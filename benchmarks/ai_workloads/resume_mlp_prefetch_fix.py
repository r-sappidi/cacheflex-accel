"""Gate the repaired-simulator study on build, regressions and paired controls.

Usage: python3 resume_mlp_prefetch_fix.py NEW_OUTPUT BUILD_LOG
Old results and binaries remain untouched. Failure stops downstream work.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / 'results/ai_workloads/mlp_study_20260909'
out = Path(sys.argv[1]).resolve()
build_log = Path(sys.argv[2]).resolve()
out.mkdir()
state = dict(stage='waiting_for_build', completed=[], failed=None)

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

protocol = ROOT / 'gem5/src/mem/ruby/protocol/MESI_Three_Level_SPM-L0cache.sm'
shutil.copy2(protocol, out / protocol.name)
sources = list(HERE.glob('*.py')) + list(HERE.glob('*.cpp')) + list(HERE.glob('*.sh')) + [protocol]
hashes = {str(p): digest(p) for p in sources}
(out / 'source_hashes.json').write_text(json.dumps(hashes, indent=2))

def status(name):
    state['stage'] = name
    (out / 'status.json').write_text(json.dumps(state, indent=2))
    print(name, flush=True)

def unchanged():
    for path, expected in hashes.items():
        assert digest(Path(path)) == expected, ('source changed', path)

env = dict(os.environ, OPENBLAS_NUM_THREADS='1', MLP_GEM5_BIN=str(out / 'gem5.opt'))

def run(name, args, extra=None):
    unchanged()
    status(name)
    with (out / (name + '.host.log')).open('w') as log:
        subprocess.run(args, env=dict(env, **(extra or {})), stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    state['completed'].append(name)
    status(name + '_complete')

try:
    status('waiting_for_build')
    deadline = time.monotonic() + 4 * 3600
    while True:
        log = build_log.read_text()
        if 'scons: done building targets.' in log:
            break
        if any(s in log for s in ('scons: building terminated', 'Error:', 'fatal error:', ' error:')):
            raise RuntimeError('simulator build failed; inspect ' + str(build_log))
        if time.monotonic() > deadline:
            raise TimeoutError('simulator build completion not observed')
        time.sleep(10)
    unchanged()
    shutil.copy2(ROOT / 'gem5/build/ARM_MESI_Three_Level_SPM/gem5.opt', out / 'gem5.opt')
    shutil.copy2(build_log, out / 'build.log')
    old_sim = OLD / 'stage1/gem5.opt'
    assert digest(old_sim) != digest(out / 'gem5.opt'), 'build did not replace the original executable'
    (out / 'simulator_hashes.json').write_text(json.dumps(
        dict(original=dict(path=str(old_sim), sha256=digest(old_sim)),
             repaired=dict(path=str(out / 'gem5.opt'), sha256=digest(out / 'gem5.opt'))), indent=2))
    cases = out / 'regression_cases.json'
    cases.write_text(json.dumps([[64,512,1536,mp,512,1,1,1] for mp in (1,2)]))
    # Exact previously failing guest binaries, with paired cache controls.
    run('regression', [sys.executable, str(HERE / 'sweep_mlp_study.py'), '2',
                       str(out / 'regression'), str(OLD / 'build_r24'), str(cases)])
    # Same original first/warmed-call guests and inputs, now on repaired hardware.
    run('stage1', [sys.executable, str(HERE / 'sweep_mlp_study.py'), '1',
                   str(out / 'stage1'), str(OLD / 'build')])
    run('stage1_convergence', [sys.executable, str(HERE / 'sweep_mlp_study.py'), '11',
                               str(out / 'stage1_convergence'), str(OLD / 'build')])
    # Rebuild all tuning candidates from the current frozen source.
    for rows in (24,8,16):
        run(f'build_r{rows}', ['bash', str(HERE / 'build_mlp_study.sh'), str(out / f'build_r{rows}')],
            dict(MLP_ROWS=str(rows), MLP_COMBINED='0', MLP_ACC32='0'))
    run('stage2_r24', [sys.executable, str(HERE / 'sweep_mlp_study.py'), '2',
                       str(out / 'stage2_r24'), str(out / 'build_r24')])
    run('ordered', [sys.executable, str(HERE / 'run_mlp_ordered.py'), str(out)])
    status('complete')
except BaseException as error:
    state['failed'] = repr(error)
    status('failed')
    raise
