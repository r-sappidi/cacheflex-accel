"""Bounded replay of placement full-set deadlock. No timing acceptance."""
import json,os,subprocess,time
from pathlib import Path
from build_access_timing import save,sha
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/cacheflex_flash_attention/spm_set_spread_20260927'
OUT=BASE/'diagnosis';OUT.mkdir()
source=BASE/'experiment/runs/spread_full_set'
cmd=json.loads((source/'command.json').read_text())
cmd[1]='--outdir='+str(OUT)
cmd[2:2]=['--debug-flags=ProtocolTrace,RubyQueue,CacheFlexTiming',
          '--debug-start=1060000000','--debug-end=1090000000','--debug-file=protocol.gz']
cmd+=['--abs-max-tick=1090000400']
save(OUT/'command.json',cmd)
save(OUT/'inputs.json',{str(p):sha(p) for p in [Path(cmd[0]),Path(__file__),source/'command.json']})
with (OUT/'run.log').open('w') as log:
    proc=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib'),stdout=log,stderr=subprocess.STDOUT)
    save(OUT/'status.json',dict(stage='running',pid=proc.pid))
    try:rc=proc.wait(timeout=1800)
    except BaseException:proc.terminate();proc.wait(timeout=10);raise
save(OUT/'status.json',dict(stage='finished_diagnostic_only',returncode=rc))
