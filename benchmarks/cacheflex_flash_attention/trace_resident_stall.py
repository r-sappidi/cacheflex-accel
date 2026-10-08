"""Bounded diagnostic replay, with original guest and simulator unchanged."""
import json
import os
from pathlib import Path
import subprocess
import sys
old,out=map(Path,sys.argv[1:3]);out=out.resolve();out.mkdir()
args=json.loads((old/'command.json').read_text())
args=[f'--outdir={out}' if x.startswith('--outdir=') else x for x in args]
start=int(sys.argv[3]) if len(sys.argv)>3 else 200000000
end=int(sys.argv[4]) if len(sys.argv)>4 else start+200000
flags=sys.argv[5] if len(sys.argv)>5 else 'SPM_CPU,LSQUnit,Commit'
args[1:1]=[f'--debug-flags={flags}',f'--debug-start={start}',
           f'--debug-end={end}','--debug-file=stall.trace']
args += [f'--abs-max-tick={end}']
(out/'command.json').write_text(json.dumps(args,indent=2))
with (out/'run.log').open('w') as log:
    subprocess.run(args,env=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib'),
                   stdout=log,stderr=subprocess.STDOUT,check=True)
