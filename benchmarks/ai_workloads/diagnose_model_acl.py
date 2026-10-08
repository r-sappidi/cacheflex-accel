"""Read-only bounded-prefix replay of the unexpectedly slow ACL precision gate."""
import json
from pathlib import Path
import subprocess
import sys
from run_model_swiglu import OUT,ENV
from run_mapping_screen import save

tick=int(sys.argv[1]) if len(sys.argv)>1 else 1000000000
mode=sys.argv[2] if len(sys.argv)>2 else ''
trace=bool(mode)
p=OUT/(f'acl_bounded_diagnostic_{tick}'+('_'+mode if trace else ''));p.mkdir()
src=OUT/'runs/gate_acl_t24_d256_f512_mp2_kc128_g0'
cmd=json.loads((src/'command.json').read_text())
cmd=[f'--outdir={p}' if a.startswith('--outdir=') else a for a in cmd]+[f'--abs-max-tick={tick}']
if trace:
    begin,end=(649900000,650100000) if mode.startswith('narrow') else (580000000,850000000)
    cmd[1:1]=['--debug-flags=Exec,LSQ,LSQUnit,IQ,IEW,Commit,O3CPU,RubySequencer,RubyPort',f'--debug-start={begin}',f'--debug-end={end}',f'--debug-file={p}/debug.log']
save(p/'command.json',cmd)
with (p/'run.log').open('w') as log:
    result=subprocess.run(cmd,env=ENV,stdout=log,stderr=subprocess.STDOUT,timeout=600)
save(p/'exit.json',dict(returncode=result.returncode,intentionally_partial=True))
