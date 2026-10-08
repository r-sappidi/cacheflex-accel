"""Frozen T16K failure replay, debug events only near the recorded crash."""
import json,os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
base=ROOT/'results/cacheflex_flash_attention/q_resident_20260915_v3'
out=ROOT/'results/cacheflex_flash_attention/q_crash_trace_20260915'
out.mkdir()
cmd=json.loads((base/'runs/spm_t16384_br16_bc128_g4_p1/command.json').read_text())
cmd=[f'--outdir={out}' if x.startswith('--outdir=') else x for x in cmd]
cmd[1:1]=['--debug-flags=SPM_CPU,Exec','--debug-start=11628000000','--debug-end=11629000000','--debug-file=trace.log']
(out/'command.json').write_text(json.dumps(cmd,indent=2))
with (out/'run.log').open('w') as log:
    result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,
        env=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib'))
(out/'returncode.json').write_text(json.dumps(result.returncode))
