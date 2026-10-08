"""Frozen late-window protocol replay of the Q-resident T16K deadlock.

Read diagnostic evidence only; do not relax deadlock thresholds or accept a
timing from this failed case. No guest, hardware or protocol changes.
"""
from pathlib import Path
import json,os,subprocess,time
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results/cacheflex_flash_attention'
prior=BASE/'q_resident_20260915_v4/runs/spm_t16384_br16_bc128_g4_p1'
out=BASE/'q_deadlock_trace_20260916'
out.mkdir()
cmd=json.loads((prior/'command.json').read_text())
cmd=[f'--outdir={out}' if x.startswith('--outdir=') else x for x in cmd]
cmd[1:1]=['--debug-flags=ProtocolTrace,RubySequencer,SPM_CPU',
    '--debug-start=73400000000','--debug-end=73800000000','--debug-file=protocol.log']
(out/'command.json').write_text(json.dumps(cmd,indent=2))
start=time.time()
def status(**data):
    p=out/'status.tmp';p.write_text(json.dumps(dict(updated=time.time(),elapsed_s=time.time()-start,**data),indent=2));p.replace(out/'status.json')
with (out/'run.log').open('w') as log:
    proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,
        env=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib'))
    while proc.poll() is None:
        status(stage='replaying',pid=proc.pid);time.sleep(30)
status(stage='replay_finished',returncode=proc.returncode)
# Keep a compact target-line view in addition to the full trace. Include all
# controllers, not just the reporting sequencer: dependency cycles span nodes.
with (out/'protocol.log').open(errors='replace') as inp, (out/'target_line.log').open('w') as dest:
    for line in inp:
        if 'fc9040' in line.lower():dest.write(line)
