"""Replay frozen resident-QK guests with a bounded, repaired simulator.

Usage: check_spm_addr_retry.py NEW_OUTPUT SIMULATOR
No guest rebuilding; analytic output checks and no-copy ROI are mandatory.
"""
import concurrent.futures as cf
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from analyze_traffic import analyze

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / 'results/cacheflex_flash_attention/residual_20260911'
out = Path(sys.argv[1]).resolve()
out.mkdir()
shutil.copy2(sys.argv[2], out / 'gem5.opt')

def run(variant):
    path = out / variant
    path.mkdir()
    args = json.loads((OLD / f'resident_{variant}_t128_p0/command.json').read_text())
    args[0] = str(out / 'gem5.opt')
    args = [f'--outdir={path}' if a.startswith('--outdir=') else a for a in args]
    if variant == 'spm':
        args[1:1] = ['--debug-flags=SPM_CPU', '--debug-end=40000000',
                     '--debug-file=retry.trace']
    args += ['--abs-max-tick=2000000000']
    (path / 'command.json').write_text(json.dumps(args, indent=2))
    with (path / 'run.log').open('w') as log:
        subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=True,
                       timeout=600, env=dict(os.environ,
                       LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib'))
    log = (path / 'run.log').read_text()
    assert 'RESIDENT_RESULT errors=0' in log, log[-2000:]
    assert 'exiting with last active thread context' in log
    counters = analyze(path)
    assert counters['m_spmcpRequestsIssued'] == counters['m_spmcpRequestsCompleted'] == 0
    return dict(variant=variant, counters=counters)

with cf.ThreadPoolExecutor(max_workers=2) as pool:
    rows = list(pool.map(run, ('cache', 'spm')))
(out / 'summary.json').write_text(json.dumps(dict(
    simulator_sha256=hashlib.sha256((out / 'gem5.opt').read_bytes()).hexdigest(),
    results=rows), indent=2))
print(json.dumps(rows, indent=2))
