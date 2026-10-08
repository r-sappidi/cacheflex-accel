"""Isolated short replay/ablation of finite-resource validation failure."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/cacheflex_flash_attention/hardware_bounds_20260921'
case = sys.argv[1]
p = OUT / 'diagnosis' / case; p.mkdir(parents=True)
origin = 'experiment_v2/runs/spm_stress_t512' if case.startswith('stress_') else 'experiment/runs/spm_bounded_t512'
cmd = json.loads((OUT / origin / 'command.json').read_text())
cmd = [f'--outdir={p}' if x.startswith('--outdir=') else x for x in cmd]
if case == 'stress_queues16':
    cmd = [x.replace('.buffer_size=4', '.buffer_size=16') for x in cmd]
elif case == 'stress_trace':
    cmd[1:1] = ['--debug-flags=ProtocolTrace,RubyQueue',
                '--debug-start=365000000', '--debug-end=365040000', '--debug-file=protocol.log']
    cmd += ['--abs-max-tick=365040400']
elif case == 'queues_unbounded':
    cmd = [x.replace('.buffer_size=16', '.buffer_size=0') for x in cmd]
elif case == 'tracking_off':
    cmd = [x.replace('.cacheFlexBoundedTracking=True', '.cacheFlexBoundedTracking=False') for x in cmd]
elif case == 'trace':
    cmd[1:1] = ['--debug-flags=ProtocolTrace,RubySequencer,RubyQueue',
                '--debug-start=256000000', '--debug-end=260000000', '--debug-file=protocol.log']
    cmd += ['--abs-max-tick=261000000']
else:
    raise ValueError(case)
(p / 'command.json').write_text(json.dumps(cmd, indent=2))
with (p / 'run.log').open('w') as log:
    proc = subprocess.run(cmd, env=dict(os.environ, LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib'),
                          stdout=log, stderr=subprocess.STDOUT)
(p / 'result.json').write_text(json.dumps(dict(returncode=proc.returncode,
    diagnostic_only=True, no_performance_acceptance=True), indent=2))
