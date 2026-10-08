"""Fast separate diagnostic; original FA guest/model remain unchanged."""
import json
import os
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[1]
B=ROOT/'results/cacheflex_flash_attention'
OUT=B/'gcc15_clock_output_stress_20260920'
CC=B/'micro_toolchain_audit_20260918/arm-gnu-toolchain-15.2.rel1-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++'
OUT.mkdir()
subprocess.run([str(CC),'-O3','-static',str(ROOT/'benchmarks/cacheflex_flash_attention/clock_output_stress.cpp'),'-o',str(OUT/'guest')],check=True)
cmd=json.loads((B/'micro_gcc15_corrected_gate_20260918/runs/cache_vl16_t320/command.json').read_text())
cmd[cmd.index('--cmd')+1]=str(OUT/'guest')
cmd=[f'--outdir={OUT}' if x.startswith('--outdir=') else "system.cpu[0].workload[0].cmd=['clock-stress']" if x.startswith('system.cpu[0].workload[0].cmd=') else x for x in cmd]
cmd[1:1]=['--debug-flags=SyscallVerbose,RubySystem','--debug-file=clock.trace']
(OUT/'command.json').write_text(json.dumps(cmd,indent=2))
with (OUT/'run.log').open('w') as log:
    rc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib'),timeout=600).returncode
(OUT/'result.json').write_text(json.dumps(dict(returncode=rc)))
