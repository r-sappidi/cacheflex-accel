"""Read existing control evidence; replay only the first differing time window."""
import concurrent.futures as cf
import json
from pathlib import Path
import re
import subprocess
import time
from build_access_timing import ROOT, save, sha
from run_access_timing import stats
from run_fa_headroom import ENV, validate, prior

BASE=ROOT/'results/cacheflex_flash_attention'
OUT=BASE/'output_resident_20260924/control_audit'
NEW=BASE/'output_resident_v8_20260924/runs/recheck_spm_t4096'
OLD=prior('spm',4096)

def main():
    OUT.mkdir()
    a,b=stats(OLD),stats(NEW)
    accepted=validate(NEW,'spm',4096,'control')
    diffs=[{k:[x.get(k),y.get(k)] for k in sorted(x.keys()|y.keys())
            if x.get(k)!=y.get(k)} for x,y in zip(a,b)]
    save(OUT/'counter_differences.json',diffs)
    committed={k:[a[0][k],b[0][k]] for k in a[0]
               if '.commit.committedInstType_' in k or k in ('simInsts','simOps')}
    assert all(x==y for x,y in committed.values())
    save(OUT/'committed_invariants.json',committed)
    save(OUT/'numerical_config_validation.json',accepted)
    progress=[]
    for p in (OLD,NEW):
        d={}
        for line in (p/'run.log').read_text().splitlines():
            m=re.match(r'(\d+):.*system.cpu(\d+) progress event, total committed:(\d+)',line)
            if m:d[(int(m[1]),int(m[2]))]=int(m[3])
        progress.append(d)
    x,y=progress
    first=min(k[0] for k in x.keys()&y.keys() if x[k]!=y[k])
    save(OUT/'summary.json',dict(roi_us=[a[0]['simTicks']/1e6,b[0]['simTicks']/1e6],
        runtime_change_pct=100*(b[0]['simTicks']/a[0]['simTicks']-1),
        all_committed_types_equal=True,first_differing_progress_tick=first,
        changed_counters_per_block=list(map(len,diffs))))
    def replay(label,source):
        p=OUT/label;p.mkdir();cmd=json.loads((source/'command.json').read_text())
        cmd=[f'--outdir={p}' if s.startswith('--outdir=') else s for s in cmd]
        cmd[1:1]=['--debug-flags=ProtocolTrace,SPM_CPU,Commit','--debug-file=trace.gz',
                  f'--debug-start={first-10_000_000}']
        cmd += [f'--abs-max-tick={first}']
        save(p/'command.json',cmd);save(p/'binary.json',{'path':cmd[0],'sha256':sha(cmd[0])})
        with (p/'run.log').open('w') as f:
            proc=subprocess.Popen(cmd,cwd=ROOT,env=ENV,stdout=f,stderr=subprocess.STDOUT)
            save(p/'process.json',dict(pid=proc.pid,start=time.time()))
            try:code=proc.wait(timeout=600)
            except BaseException:
                proc.terminate();proc.wait(timeout=10);raise
        assert code==0,(label,code)
        assert 'simulate() limit reached' in (p/'run.log').read_text()
        save(p/'completed.json',dict(prefix_only=True,end_tick=first))
    with cf.ThreadPoolExecutor(max_workers=2) as pool:
        fs=[pool.submit(replay,k,p) for k,p in [('old',OLD),('new',NEW)]]
        for f in cf.as_completed(fs):f.result()
    save(OUT/'complete.json',dict(prefix_replays_complete=True))

if __name__=='__main__':main()
