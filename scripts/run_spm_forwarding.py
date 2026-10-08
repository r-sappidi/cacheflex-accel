"""Finite existing-SQ forwarding: regression, exact OFF control, matched FA."""
import concurrent.futures as cf
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
import run_fa_output_resident as m

BASE=m.BASE/'spm_forwarding_20260924'
BUILD=BASE/'build'
OUT=BASE/os.environ.get('CACHEFLEX_FORWARD_EXPERIMENT','experiment_v3')
REUSE=BASE/'experiment_v2'
TESTS=BASE/os.environ.get('CACHEFLEX_FORWARD_TESTS','tests_v2')
OLD512=m.BASE/'output_resident_v8_20260924/runs/output_spm_t512'
OLD4096=m.BASE/'output_resident_v10_20260924/runs/output_spm_t4096'
m.OUT=OUT
m.BUILD=BUILD

def audit(path,kind,t):
    old=OLD512 if t==512 else OLD4096
    a=dict(m.leaves(json.loads((old/'config.json').read_text().replace(str(old),'<OUT>'))))
    b=dict(m.leaves(json.loads((path/'config.json').read_text().replace(str(path),'<OUT>'))))
    on='_on_' in path.name
    for i in range(8):
        assert b.pop(f'system.cpu[{i}].cacheFlexSpmStoreForwarding')==on
        assert b.pop(f'system.cpu[{i}].cacheFlexSpmForwardLatency')==2
    delta={k:[a.get(k),b.get(k)] for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}
    assert not delta,delta
    return {'forwarding':on,'latency_cycles':2,'all_other_fields_identical':True}
m.config_audit=audit

def sum_stat(s,suffix):
    return sum(v for k,v in s.items() if '.spmLsq0.' in k and k.endswith(suffix))

def main():
    assert (BUILD/'frozen.json').exists()
    OUT.mkdir();(OUT/'runs').mkdir()
    shutil.copytree(BUILD/'configs',OUT/'configs')
    state=dict(pid=os.getpid(),stage='regressions',active={},completed=[])
    lock=threading.RLock();halt=threading.Event()
    manifest=json.loads((BUILD/'frozen.json').read_text())
    paths=list((OUT/'configs').rglob('*.py'))+[Path(__file__).resolve(),TESTS/'guest',
        m.GUESTS/'roundtrip/guest',m.GUESTS/'build_spm/guest.final',
        Path(m.__file__).resolve()]
    for old in (OLD512,OLD4096):
        paths += [old/n for n in ('config.json','stats.txt','output.fp16','command.json','run.log')]
    for t in (512,4096):paths.append(m.REF/f't{t}_bc128.npy')
    paths += [p for p in (REUSE/'runs').rglob('*') if p.is_file()]
    manifest.update({str(p):m.sha(p) for p in paths});m.save(OUT/'inputs.json',manifest)
    def frozen():
        for p,h in manifest.items():assert m.sha(p)==h,('Input changed',p)
    def status(stage=None):
        with lock:
            if stage:state['stage']=stage
            state['updated']=time.time();m.save(OUT/'status.json',state)
            lines=['# SPM store-to-load forwarding','',
                'Same output-resident FA guest, existing two SPM ways/SQ payload. '
                'Optional 2-cycle forwarding consumes existing load ports; no additional SPM capacity.','',
                '| T | Forwarding | ROI (us) | Forwarded loads |','|---|---|---:|---:|']
            for r in state['completed']:
                if 'T' in r:lines.append(f"| {r['T']} | {r['forwarding']} | {r['roi_us']:.4f} | {r['forwarded_loads']:.0f} |")
            for t in (512,4096):
                rows={r['forwarding']:r for r in state['completed'] if r.get('T')==t}
                if rows.keys()>={False,True}:
                    ratio=rows[False]['roi_us']/rows[True]['roi_us']
                    lines.append(f'\nT{t} OFF/ON speedup: {ratio:.6f}x; runtime reduction {(1-1/ratio)*100:.3f}%.')
            rows=[r for r in state['completed'] if r.get('T')==4096 and r.get('forwarding')]
            if rows:
                for name,lat in [('cache winner',5660.9784),('V-SPM',5715.0284),('matched coherent',6661.4028)]:
                    lines.append(f"\nT4096 {name}/forwarding: {lat/rows[0]['roi_us']:.6f}x.")
            (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    def run(on,t=None,regression='forward'):
        frozen();label=('on' if on else 'off')
        name=f'fa_{label}_t{t}' if t else f'{regression}_{label}'
        if t!=4096:
            old=REUSE/'runs'/name
            row=json.loads((old/'validated.json').read_text())
            if t:
                m.validate(old,'output_spm',t)
                assert row['arithmetic']['errors']==row['arithmetic']['nonfinite']==0
                if not on:assert row['exact_previous_stats']
            else:
                assert row['passed']
                if on and regression=='forward':assert row['forwarded_loads']>0
            row.update(reused=True)
            with lock:state['completed'].append(row);status()
            return
        path=OUT/'runs'/name;path.mkdir();proc=None;start=time.monotonic()
        try:
            if t:cmd=m.command('output_spm',t,path)
            else:
                cmd=json.loads((m.GUESTS/'roundtrip/command.json').read_text());cmd[0]=str(BUILD/'gem5.opt')
                for i,a in enumerate(cmd):
                    if a.startswith('--outdir='):cmd[i]=f'--outdir={path}'
                    if a.endswith('/micro_timing_se.py'):cmd[i]=str(OUT/'configs/deprecated/example/micro_timing_se.py')
                if regression=='forward':cmd[cmd.index('--cmd')+1]=str(TESTS/'guest')
            for i in range(8):cmd+=['-P',f'system.cpu[{i}].cacheFlexSpmStoreForwarding={on!r}']
            m.save(path/'command.json',cmd)
            with (path/'run.log').open('w') as log:
                proc=subprocess.Popen(cmd,cwd=m.ROOT,env=m.ENV,stdout=log,stderr=subprocess.STDOUT)
                with lock:state['active'][name]=dict(pid=proc.pid,path=str(path));status()
                while proc.poll() is None:
                    assert not halt.is_set(),'Peer failed'
                    assert time.monotonic()-start<(10800 if t else 300),'Timeout'
                    assert shutil.disk_usage(OUT).free>50*2**30,'Disk guard'
                    try:proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:pass
            text=(path/'run.log').read_text()
            assert proc.returncode==0 and 'Simulated exit code not 0!' not in text,(name,proc.returncode)
            blocks=m.stats(path);s=blocks[0]
            forwards=sum_stat(s,'.forwLoads')
            if t:
                row=m.validate(path,'output_spm',t)
                assert row['arithmetic']['errors']==row['arithmetic']['nonfinite']==0
                if not on:
                    old=m.stats(OLD512 if t==512 else OLD4096)
                    clean=[]
                    for block in blocks:
                        extra={k:v for k,v in block.items() if k.endswith('.spmForwardedBytes')}
                        assert len(extra)==16 and all(v==0 for v in extra.values())
                        clean.append({k:v for k,v in block.items() if k not in extra})
                    delta=[{k:[a.get(k),b.get(k)] for k in a.keys()|b.keys() if a.get(k)!=b.get(k)} for a,b in zip(old,clean)]
                    m.save(path/'off_counter_delta.json',delta)
                    assert len(old)==len(clean) and all(not d for d in delta),'OFF changed counters'
                    row['exact_previous_stats']=True
                row.update(forwarding=on,forwarded_loads=forwards,
                    forwarded_bytes=sum_stat(s,'.spmForwardedBytes'),
                    rescheduled_loads=sum_stat(s,'.rescheduledLoads'))
            else:
                marker='SPM_FORWARD_TEST errors=0 elements=16384' if regression=='forward' else 'OUTPUT_RESIDENT_ROUNDTRIP errors=0 elements=4096'
                assert marker in text,'Regression mismatch'
                if regression=='forward' and on:assert forwards>0,'Regression missed forwarding'
                if not on:assert forwards==0
                row=dict(regression=regression,forwarding=on,forwarded_loads=forwards,passed=True)
            frozen();row.update(path=str(path),host_seconds=time.monotonic()-start)
            m.save(path/'validated.json',row)
            with lock:state['active'].pop(name);state['completed'].append(row);status()
        except BaseException as exc:
            halt.set();m.save(path/'failed.json',dict(error=repr(exc)))
            if proc and proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
            with lock:state['active'].pop(name,None);status()
            raise
    def pair(t=None,regression='forward'):
        with cf.ThreadPoolExecutor(max_workers=2) as pool:
            fs=[pool.submit(run,on,t,regression) for on in (False,True)]
            for f in cf.as_completed(fs):f.result()
    try:
        status();pair();pair(regression='bridge_roundtrip')
        for t in (512,4096):
            status(f'fa_t{t}');pair(t)
            if t==512:m.save(OUT/'gates_passed.json',dict(passed=True))
        status('completed_validated')
    except BaseException as exc:
        state['error']=repr(exc);status('failed');raise
if __name__=='__main__':main()
