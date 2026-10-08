"""Bounded trace of the preserved v3 T4096 writeback/recall race."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/cacheflex_flash_attention/hardware_bounds_20260921'
v4 = '--v4' in sys.argv
p = OUT / ('diagnosis/writeback_v4' if v4 else 'diagnosis/writeback_v3')
p.mkdir(parents=True)
origin = 'experiment_v4/runs/spm_bounded_t512' if v4 else 'experiment_v3/runs/cache_bounded_t4096'
cmd = json.loads((OUT / origin / 'command.json').read_text())
cmd = [f'--outdir={p}' if x.startswith('--outdir=') else x for x in cmd]
start, end = (220000000, 226000000) if v4 else (1295000000, 1298010000)
cmd[1:1] = ['--debug-flags=ProtocolTrace,RubySlicc',
            f'--debug-start={start}', f'--debug-end={end}',
            '--debug-file=protocol.log']
cmd += [f'--abs-max-tick={end}']
(p / 'command.json').write_text(json.dumps(cmd, indent=2))
with (p / 'run.log').open('w') as log:
    proc = subprocess.run(cmd, env=dict(os.environ,
        LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib'),
        stdout=log, stderr=subprocess.STDOUT)
(p / 'result.json').write_text(json.dumps(dict(returncode=proc.returncode,
    diagnostic_only=True), indent=2))
