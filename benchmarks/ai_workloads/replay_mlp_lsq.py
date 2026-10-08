"""Replay an archived gem5 command without changing guest/input paths.

Usage: replay_mlp_lsq.py OLD_RUN_LOG NEW_OUT [SIMULATOR] [TRACE_START]
"""
import os
from pathlib import Path
import shlex
import subprocess
import sys

log, out = Path(sys.argv[1]), Path(sys.argv[2]).resolve()
out.mkdir()
line = next(x for x in log.read_text().splitlines()
            if x.startswith('command line: '))
args = shlex.split(line.removeprefix('command line: '))
args = [f'--outdir={out}' if x.startswith('--outdir=') else x for x in args]
if len(sys.argv) > 3:
    args[0] = str(Path(sys.argv[3]).resolve())
if len(sys.argv) > 4:
    args[1:1] = ['--debug-flags=SPM_CPU,LSQUnit',
                 '--debug-file=debug.trace', f'--debug-start={sys.argv[4]}']
env = dict(os.environ, LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib')
with (out / 'run.log').open('w') as stream:
    result = subprocess.run(args, env=env, stdout=stream, stderr=subprocess.STDOUT)
sys.exit(result.returncode)
