"""Bounded diagnostic replay; does not produce accepted performance results."""
import json,os,subprocess,sys,time
from pathlib import Path
from build_access_timing import sha,save
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/cacheflex_flash_attention/finite_model_20260926'
name=sys.argv[1];p=BASE/'diagnosis'/name;p.mkdir(parents=True)
origin=BASE/'experiment_v3/runs/stress_cache'
cmd=json.loads((origin/'command.json').read_text())
cmd[1]='--outdir='+str(p)
if name in ('trace','trace_late'):
    lo,hi=(2129000000,2131500000) if name=='trace' else (2131500000,2140000000)
    cmd[1:1]=['--debug-flags=ProtocolTrace,RubyQueue,CacheFlexTiming',f'--debug-start={lo}',f'--debug-end={hi}','--debug-file=protocol.log']
    cmd+=[f'--abs-max-tick={hi+400}']
elif name=='normal_pools':
    # Keep all small queues, TBEs, targets, LQ/SQ; restore only miss/WB pools.
    for i in range(8):
        for attr,n in [('l0_cntrl%d.dataMissCredits'%i,8),('l0_cntrl%d.instMissCredits'%i,8),('l0_cntrl%d.writeCredits'%i,8),('l1_cntrl%d.missCredits'%i,20),('l1_cntrl%d.writeCredits'%i,8)]:
            cmd+=['-P',f'system.ruby.{attr}.capacity={n}']
    cmd+=['-P','system.ruby.microLlcMissCredits.capacity=32','-P','system.ruby.microLlcWriteCredits.capacity=16']
else:raise ValueError(name)
save(p/'command.json',cmd);save(p/'inputs.json',{str(x):sha(x) for x in [Path(cmd[0]),Path(__file__),origin/'command.json']})
with (p/'run.log').open('w') as log:
    proc=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib'),stdout=log,stderr=subprocess.STDOUT)
    save(p/'status.json',dict(pid=proc.pid,stage='running'))
    try:rc=proc.wait(timeout=1800)
    except BaseException:
        proc.terminate();proc.wait(timeout=10);raise
save(p/'status.json',dict(returncode=rc,stage='finished_diagnostic_only'))
