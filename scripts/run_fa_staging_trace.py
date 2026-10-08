"""Frozen T16K staging diagnosis, one extra simulator, bounded debug storage.

Locate core0 staging retirement gaps, then replay the identical prefix with
pipeline/LSQ/Ruby logs around one in-stage delayed stride load. Partial runs
are diagnostics, never accepted whole-FA performance/correctness results.
"""
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/'results/cacheflex_flash_attention/llc_retain_20260915/runs/spm_t16384_retain1_small0_old0'
LOAD=0x401e14
STAGE=0x401db0
QK=0x401e94


def save(p,data):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(p)


def sha(p):
    with Path(p).open('rb') as f:
        h=hashlib.sha256()
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
        return h.hexdigest()


def stats(p):
    result=[]
    for block in (p/'stats.txt').read_text().split('Begin Simulation Statistics')[1:]:
        vals={}
        for line in block.split('End Simulation Statistics')[0].splitlines():
            w=line.split()
            if len(w)<2 or w[0].startswith('host'):continue
            try:float(w[1])
            except ValueError:continue
            vals[w[0]]=w[1]
        result.append(vals)
    return result


def choose_window(path,start,end):
    gaps=[];last=None;inside=False
    for line in trace_lines(path):
        m=re.search(r'(\d+): system.cpu0.commit: CFBOUND pc=(0x[0-9a-f]+) sn=(\d+)',line)
        if not m:continue
        tick,pc,sn=int(m[1]),int(m[2],16),int(m[3])
        if pc==STAGE:inside=True;last=None
        elif pc==QK:inside=False;last=None
        elif pc==LOAD and inside:
            if last is not None:gaps.append(dict(begin=last[0],end=tick,previous_sn=last[1],sn=sn,gap=tick-last[0]))
            last=(tick,sn)
    assert gaps,'No completed within-staging load pairs in prefix; inspect scan, do not guess ticks'
    # Largest fully observed gap, with explicit lead-in and completion margin.
    selected=max(gaps,key=lambda g:g['gap'])
    lo=max(start,selected['begin']-200000);hi=min(end,selected['end']+200000)
    assert hi-lo<=8000000,('Selected window exceeds 8us cap',selected)
    return dict(selected=selected,begin=lo,end=hi,count=len(gaps),
                median_gap=sorted(g['gap'] for g in gaps)[len(gaps)//2],
                warning='Largest core0 in-stage retirement gap in this prefix, not representative whole-run average or load service latency.')


def trace_lines(path):
    # Debug messages may contain raw data bytes, not UTF-8 text. Escape only
    # for parsing; the focused output below retains the original bytes.
    with path.open('rb') as stream:
        for line in stream:
            yield line.decode('utf-8',errors='backslashreplace')


def extract_context(source,destination,sn):
    pattern=re.compile(rb'(?<!\d)(?:'+b'|'.join(str(i).encode() for i in range(sn-24,sn+25))+rb')(?!\d)')
    temporary=destination.with_suffix('.tmp')
    selected=0;non_utf8=0
    with source.open('rb') as src,temporary.open('wb') as dst:
        for line in src:
            try:line.decode('utf-8')
            except UnicodeDecodeError:non_utf8+=1
            if (b'system.cpu0' in line and pattern.search(line)) or b'401e14' in line:
                dst.write(line);selected+=1
    assert selected,'No target context found'
    temporary.replace(destination)
    return dict(selected_lines=selected,non_utf8_lines=non_utf8,byte_preserving=True)


def postprocess(out):
    out=out.resolve()
    state=json.loads((out/'status.json').read_text())
    assert state['stage'] in ('failed','trace_ready')
    old_manifest=json.loads((out/'manifest.json').read_text())
    for p,digest in old_manifest.items():
        if Path(p).resolve()!=Path(__file__).resolve():
            assert sha(p)==digest,('Simulation dependency changed',p)
    plan=json.loads((out/'plan.json').read_text())
    for name in ('scan','trace'):
        assert json.loads((out/name/'exit.json').read_text())['returncode']==0
        assert int(stats(out/name)[-1]['finalTick'])==plan['stop']
    a,b=stats(out/'scan'),stats(out/'trace')
    assert a==b,'Observer counter mismatch'
    if not (out/'status.before_parser_repair.json').exists():
        save(out/'status.before_parser_repair.json',state)
    window=json.loads((out/'window.json').read_text())
    source=out/'trace/diagnostic.trace';digest=sha(source)
    report=extract_context(source,out/'target_context.log',window['selected']['sn'])
    assert sha(source)==digest
    report.update(trace_sha256=digest,parser_sha256=sha(__file__),
                  original_parser_sha256=old_manifest[str(Path(__file__).resolve())],
                  counter_entries=sum(map(len,a)),simulations_rerun=False)
    save(out/'parser_repair.json',report)
    state.pop('error',None)
    state.update(stage='trace_ready',updated=time.time(),target_sn=window['selected']['sn'],observer_equivalence=True,parser_repair=report)
    save(out/'status.json',state)
    print(json.dumps(report,indent=2))


def main():
    out=Path(sys.argv[1]).resolve();out.mkdir()
    state=dict(pid=os.getpid(),stage='preparing',updated=time.time())
    def status(stage,**kw):
        state.update(stage=stage,updated=time.time(),**kw);save(out/'status.json',state);print(stage,flush=True)
    try:
        original=json.loads((PRIOR/'command.json').read_text())
        assert (PRIOR/'accepted.json').exists()
        scalar=stats(PRIOR)[0]
        start=int(scalar['finalTick'])-int(scalar['simTicks'])
        stop=start+4_000_000_000
        watched=[Path(original[0]),Path(original[original.index('--cmd')+1]),PRIOR/'command.json',PRIOR/'config.ini',Path(__file__)]
        watched+=list((ROOT/'gem5/configs').rglob('*.py'))
        manifest={str(p.resolve()):sha(p) for p in watched};save(out/'manifest.json',manifest)
        save(out/'plan.json',dict(prior=str(PRIOR),tokens=16384,Hq=8,Hkv=1,retention=True,
             roi_start=start,stop=stop,core=0,load_pc=hex(LOAD),max_trace_bytes=2*1024**3,
             max_window_ticks=8000000,max_simulators=1,
             note='Two prefix replays of identical frozen simulator and guest. No active queue modifications. No whole-output acceptance for partial diagnostics.'))
        def frozen():
            for p,d in manifest.items():assert sha(p)==d,('dependency changed',p)
        env=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib')
        def run(name,flags,lo,hi):
            frozen();p=out/name;p.mkdir()
            cmd=[f'--outdir={p}' if x.startswith('--outdir=') else x for x in original]
            assert not any(x.startswith('--debug-') or x.startswith('--abs-max-tick') for x in cmd)
            cmd[1:1]=[f'--debug-flags={flags}',f'--debug-start={lo}',f'--debug-end={hi}','--debug-file=diagnostic.trace']
            cmd += [f'--abs-max-tick={stop}','-P',f'system.cpu[0].cacheFlexBoundaryPcs={[STAGE,LOAD,QK]}']
            save(p/'command.json',cmd);begin=time.monotonic();proc=None
            try:
                with (p/'run.log').open('w') as log:
                    proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
                    while True:
                        trace=p/'diagnostic.trace';size=trace.stat().st_size if trace.exists() else 0
                        assert size<=2*1024**3,'Trace exceeded 2GiB storage budget'
                        elapsed=time.monotonic()-begin
                        assert elapsed<12*3600,'Diagnostic timeout'
                        save(p/'progress.json',dict(pid=proc.pid,stage='running',host_elapsed_s=elapsed,trace_bytes=size,updated=time.time()))
                        try:rc=proc.wait(timeout=5);break
                        except subprocess.TimeoutExpired:pass
                assert rc==0,(p,rc)
                text=(p/'run.log').read_text()
                assert not re.search(r'panic:|fatal:',text)
                assert int(stats(p)[-1]['finalTick'])==stop
                frozen();save(p/'exit.json',dict(returncode=rc,host_elapsed_s=time.monotonic()-begin,diagnostic_only=True))
                save(p/'progress.json',dict(stage='complete',host_elapsed_s=time.monotonic()-begin))
                return p
            except BaseException:
                if proc is not None and proc.poll() is None:
                    proc.terminate()
                    try:proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()
                raise
        status('locating_staging_window')
        scan=run('scan','CacheFlexBoundary',start,stop)
        window=choose_window(scan/'diagnostic.trace',start,stop);save(out/'window.json',window)
        status('detailed_replay',window=window)
        trace=run('trace','CacheFlexBoundary,Exec,IQ,IEW,Commit,MemDepUnit,LSQ,LSQUnit,SPM_CPU,RubySequencer,ProtocolTrace',window['begin'],window['end'])
        a,b=stats(scan),stats(trace)
        assert a==b,'Debug observer changed non-host prefix counters; reject diagnosis'
        save(out/'observer_equivalence.json',dict(passed=True,epochs=len(a),counter_entries=sum(map(len,a)),exclude='host* only'))
        # Preserve complete trace and focused dynamic-SN context for subsequent analysis.
        sn=window['selected']['sn']
        extract_context(trace/'diagnostic.trace',out/'target_context.log',sn)
        status('trace_ready',target_sn=sn,observer_equivalence=True)
    except BaseException as exc:
        status('failed',error=repr(exc));raise


if __name__=='__main__':
    if len(sys.argv)==3 and sys.argv[1]=='--postprocess':postprocess(Path(sys.argv[2]))
    else:main()
