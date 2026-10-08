"""Bounded diagnostic replay of a preserved failed lifecycle case."""
import json,os,subprocess
from pathlib import Path
import run_finite_micro as finite
from build_access_timing import save
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/cacheflex_l3_spm/lifecycle_20260928'
def main():
    case=BASE/os.environ.get('CF_LIFECYCLE_TRACE_CASE','experiment_v3/runs/final')
    out=BASE/os.environ.get('CF_LIFECYCLE_TRACE_OUT','diagnosis/ordering_v4');out.mkdir(parents=True)
    cmd=json.loads((case/'command.json').read_text());cmd[1]='--outdir='+str(out)
    cmd[2:2]=['--debug-flags=ProtocolTrace,SPM_CPU',
        '--debug-start='+os.environ.get('CF_LIFECYCLE_TRACE_START','34000000'),
        '--debug-end='+os.environ.get('CF_LIFECYCLE_TRACE_END','37500000'),'--debug-file=trace.gz']
    if os.environ.get('CF_LIFECYCLE_TRACE_STOP'):
        cmd+=['--abs-max-tick='+os.environ['CF_LIFECYCLE_TRACE_STOP']]
    save(out/'command.json',cmd)
    with (out/'run.log').open('w') as log:
        result=subprocess.run(cmd,cwd=ROOT,env=finite.ENV,stdout=log,stderr=subprocess.STDOUT)
    save(out/'exit.json',dict(returncode=result.returncode))
if __name__=='__main__':main()
