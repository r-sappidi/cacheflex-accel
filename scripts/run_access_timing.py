"""Timing-only instrumentation validation and full-output-equivalent FA replay."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from build_access_timing import ROOT, OUT, BUILD, sha, save

PRIOR=ROOT/'results/cacheflex_flash_attention/micro_resources_20260921/experiment_v6/runs/spm_bounded_t512'
EX=OUT/os.environ.get('CACHEFLEX_TIMING_EXPERIMENT','experiment')


def stats(p):
    blocks=[]; cur={}
    for line in (p/'stats.txt').read_text().splitlines():
        if 'Begin Simulation Statistics' in line:cur={}
        elif 'End Simulation Statistics' in line:blocks.append(cur)
        else:
            w=line.split()
            if len(w)<2 or w[0].startswith(('host','#')):continue
            try:v=float(w[1])
            except ValueError:continue
            if v==v and abs(v)!=float('inf'):cur[w[0]]=v
    return blocks


def protocol(path):
    pat=re.compile(rb'^\s*(\d+)\s+(\d+)\s+(Seq|L0Cache|L1Cache|L2Cache)\s+(\S+)\s+(\S+)\s+\[0x[0-9a-f]+, line (0x[0-9a-f]+)\]\s*(.*)')
    events=[]
    with path.open('rb') as stream:
        for i,line in enumerate(stream,1):
            m=pat.match(line)
            if m:
                t,c,ctrl,event,state,addr,detail=m.groups()
                events.append(dict(tick=int(t),core=int(c),ctrl=ctrl.decode(),event=event.decode(),
                                   state=state.decode(),addr=addr.decode(),detail=detail.decode(),line=i))
    return events


def select(path):
    events=protocol(path);paths=[]
    byaddr={}
    for e in events:byaddr.setdefault(e['addr'],[]).append(e)
    for addr,es in byaddr.items():
        starts=[e for e in es if e['core']==0 and e['ctrl']=='Seq' and e['event']=='Begin' and e['detail'].strip()=='SPMCP_fetch']
        for s in starts:
            done=next((e for e in es if e['tick']>s['tick'] and e['core']==0 and e['ctrl']=='Seq' and e['event']=='Done'),None)
            if not done:continue
            between=[e for e in es if s['tick']<=e['tick']<=done['tick']]
            p=[e for e in between if e['core']==0 and e['ctrl']=='L1Cache' and e['state']=='I>IX_D']
            l=[e for e in between if e['ctrl']=='L2Cache' and e['event']=='SPM_GETS_SILENT' and e['state']=='M>M']
            f=[e for e in between if e['core']==0 and e['ctrl']=='L1Cache' and e['state']=='IX_D>X']
            if len(p)==len(l)==len(f)==1:
                paths.append(dict(addr=addr,begin=s,private=p[0],llc=l[0],install=f[0],done=done))
    assert paths,'No unambiguous LLC-resident copy in scan'
    # Diagnostic of a long request, not an estimate of the workload average.
    chosen=max(paths,key=lambda p:p['install']['tick']-p['begin']['tick'])
    chosen.update(selection='Longest unambiguous core0 I->IX_D / LLC M-hit copy in scan',candidates=len(paths))
    return chosen


def main():
    EX.mkdir();state=dict(pid=os.getpid())
    def status(stage,**kw):
        state.update(stage=stage,updated=time.time(),**kw);save(EX/'status.json',state)
    proc=None
    try:
        status('waiting_for_build')
        while True:
            b=json.loads((BUILD/'status.json').read_text())
            assert b['stage']!='failed',b
            if b['stage']=='built_requires_validation':break
            time.sleep(10)
        original=json.loads((PRIOR/'command.json').read_text())
        paths=[BUILD/'gem5.opt',Path(original[original.index('--cmd')+1]),PRIOR/'command.json',PRIOR/'accepted.json',PRIOR/'output.fp16']
        paths+=list((BUILD/'configs').rglob('*.py'))
        manifest={str(p):sha(p) for p in paths};save(EX/'inputs.json',manifest)
        def frozen():
            for p,d in manifest.items():assert sha(p)==d,('Changed dependency',p)
        def run(name,flags,lo,hi,line=None):
            nonlocal proc
            frozen();p=EX/name;p.mkdir()
            cmd=[f'--outdir={p}' if x.startswith('--outdir=') else x for x in original]
            cmd[0]=str(BUILD/'gem5.opt');cmd[2]=str(BUILD/'configs/deprecated/example/bounded_se.py')
            cmd[1:1]=[f'--debug-flags={flags}',f'--debug-start={lo}',f'--debug-end={hi}','--debug-file=timing.trace']
            save(p/'command.json',cmd)
            env=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
            if line:env['CACHEFLEX_TRACE_LINE']=line
            start=time.monotonic()
            with (p/'run.log').open('w') as log:
                proc=subprocess.Popen(cmd,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                while True:
                    trace=p/'timing.trace';size=trace.stat().st_size if trace.exists() else 0
                    status(name,simulator=proc.pid,host_seconds=time.monotonic()-start,trace_bytes=size)
                    assert size<2*1024**3 and time.monotonic()-start<1800
                    assert shutil.disk_usage(EX).free>50*1024**3
                    try:rc=proc.wait(timeout=5);break
                    except subprocess.TimeoutExpired:pass
            assert rc==0,(name,rc)
            frozen()
            assert sha(p/'output.fp16')==sha(PRIOR/'output.fp16'),'Numerical output differs from independently validated control'
            a,z=stats(PRIOR),stats(p)
            assert a==z,('Non-host counter mismatch',name,[(i,k,a[i].get(k),v) for i,r in enumerate(z) for k,v in r.items() if i>=len(a) or a[i].get(k)!=v][:30])
            # Compare all JSON parameters, allowing only output-local redirects.
            a=json.loads((PRIOR/'config.json').read_text());z=json.loads((p/'config.json').read_text())
            def norm(x,root):
                if isinstance(x,dict):return {k:norm(v,root) for k,v in x.items()}
                if isinstance(x,list):return [norm(v,root) for v in x]
                return x.replace(str(root),'<OUT>') if isinstance(x,str) else x
            assert norm(a,PRIOR)==norm(z,p),'Configuration mismatch'
            save(p/'validated.json',dict(exact_finite_nonhost_stats=True,exact_output=True,
                inherited_independent_arithmetic_pass=True,exact_config_except_outdir=True,host_seconds=time.monotonic()-start))
            return p
        old=stats(PRIOR)[0];start=int(old['finalTick']-old['simTicks']);end=int(old['finalTick'])
        p=run('scan','ProtocolTrace',start,end)
        target=select(p/'timing.trace');save(EX/'target.json',target)
        p=run('trace','CacheFlexTiming,ProtocolTrace,IQ,IEW,Commit,LSQ,LSQUnit,SPM_CPU',
              target['begin']['tick']-1_000_000,target['done']['tick']+100_000,target['addr'])
        status('trace_validated_requires_analysis',simulator=None,target=target)
    except BaseException as exc:
        if proc and proc.poll() is None:
            proc.terminate()
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
        status('failed',error=repr(exc));raise


if __name__=='__main__':main()
