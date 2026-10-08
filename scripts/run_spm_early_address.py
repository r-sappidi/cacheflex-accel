"""Bounded address/data separation: correctness gates then matched T4096 FA."""
import concurrent.futures as cf
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
import run_fa_output_resident as m

BASE=m.BASE/'spm_early_address_20260924'
BUILD=BASE/'build_v4'
OUT=BASE/os.environ.get('CACHEFLEX_EARLY_EXPERIMENT','experiment_v4')
FORWARD=m.BASE/'spm_forwarding_20260924'
OLD={512:FORWARD/'experiment_v2/runs/fa_off_t512',4096:FORWARD/'experiment_v3/runs/fa_off_t4096'}
m.OUT=OUT;m.BUILD=BUILD
NEW_STATS=('spmEarlyAddrIssued','spmEarlyAddrDisjoint','spmEarlyAddrOverlap','spmEarlyAddrBusy')

def audit(path,kind,t):
    old=OLD[t]
    a=dict(m.leaves(json.loads((old/'config.json').read_text().replace(str(old),'<OUT>'))))
    b=dict(m.leaves(json.loads((path/'config.json').read_text().replace(str(path),'<OUT>'))))
    for i in range(8):
        assert b.pop(f'system.cpu[{i}].cacheFlexSpmEarlyAddress')==('_on_' in path.name)
    delta={k:[a.get(k),b.get(k)] for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}
    assert not delta,delta
    return dict(early_address='_on_' in path.name,all_other_fields_identical=True)
m.config_audit=audit

def total(s,suffix):
    return sum(v for k,v in s.items() if '.spmLsq0.' in k and k.endswith('.'+suffix))

def main():
    OUT.mkdir(parents=True);(OUT/'runs').mkdir()
    state=dict(pid=os.getpid(),stage='waiting_for_build',active={},completed=[])
    lock=threading.RLock();halt=threading.Event();manifest={}
    def status(stage=None):
        with lock:
            if stage:state['stage']=stage
            state['updated']=time.time();m.save(OUT/'status.json',state)
            lines=['# Early SPM store address resolution','',
                'One address-only unit/core, one computation/cycle, one-cycle latency. '
                'Metadata resides in existing finite SQ; store-data and commit path unchanged. '
                'Forwarding disabled in both cases.','',
                '| T | Early address | ROI (us) | Early computations |','|---|---|---:|---:|']
            for r in state['completed']:
                if 'T' in r:lines.append(f"| {r['T']} | {r['early_address']} | {r['roi_us']:.4f} | {r['early']['spmEarlyAddrIssued']:.0f} |")
            for t in (512,4096):
                rows={r['early_address']:r for r in state['completed'] if r.get('T')==t}
                if rows.keys()>={False,True}:
                    ratio=rows[False]['roi_us']/rows[True]['roi_us']
                    lines.append(f'\nT{t} OFF/ON: {ratio:.6f}x; end-to-end latency reduction {(1-1/ratio)*100:.3f}%.')
            rows=[r for r in state['completed'] if r.get('T')==4096 and r.get('early_address')]
            if rows:
                for name,lat in [('existing cache winner',5660.9784),('V-SPM',5715.0284),('matched coherent',6661.4028)]:
                    lines.append(f"\nT4096 {name}/early-address SPM: {lat/rows[0]['roi_us']:.6f}x.")
            (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    def frozen():
        for p,h in manifest.items():assert m.sha(p)==h,('Frozen input changed',p)
    def run(on,t=None,reg='early'):
        frozen();name=f'fa_{"on" if on else "off"}_t{t}' if t else f'{reg}_{"on" if on else "off"}'
        path=OUT/'runs'/name;path.mkdir();start=time.monotonic();proc=None
        try:
            if t:cmd=m.command('output_spm',t,path)
            else:
                cmd=json.loads((m.GUESTS/'roundtrip/command.json').read_text());cmd[0]=str(BUILD/'gem5.opt')
                for i,a in enumerate(cmd):
                    if a.startswith('--outdir='):cmd[i]=f'--outdir={path}'
                    if a.endswith('/micro_timing_se.py'):cmd[i]=str(OUT/'configs/deprecated/example/micro_timing_se.py')
                if reg=='early':cmd[cmd.index('--cmd')+1]=str(BASE/'tests/guest')
                if reg=='overlap':cmd[cmd.index('--cmd')+1]=str(FORWARD/'tests_v2/guest')
            for i in range(8):cmd+=['-P',f'system.cpu[{i}].cacheFlexSpmEarlyAddress={on!r}']
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
            blocks=m.stats(path);s=blocks[0];counts={k:total(s,k) for k in NEW_STATS}
            assert total(s,'forwLoads')==0,'Forwarding unexpectedly enabled'
            if t:
                row=m.validate(path,'output_spm',t)
                assert row['arithmetic']['errors']==row['arithmetic']['nonfinite']==0
                if not on:
                    clean=[]
                    for block in blocks:
                        extra={k:v for k,v in block.items() if k.rsplit('.',1)[-1] in NEW_STATS}
                        assert len(extra)==64 and all(v==0 for v in extra.values())
                        clean.append({k:v for k,v in block.items() if k not in extra})
                    old=m.stats(OLD[t]);assert len(old)==len(clean)
                    delta=[{k:[a.get(k),b.get(k)] for k in a.keys()|b.keys() if a.get(k)!=b.get(k)} for a,b in zip(old,clean)]
                    m.save(path/'off_counter_delta.json',delta)
                    assert all(not d for d in delta),'OFF changed prior model counters'
                    row['exact_previous_stats']=True
                row.update(rescheduled_loads=total(s,'rescheduledLoads'))
            else:
                marker={'early':'SPM_EARLY_ADDRESS errors=0 elements=22528',
                        'overlap':'SPM_FORWARD_TEST errors=0 elements=16384',
                        'roundtrip':'OUTPUT_RESIDENT_ROUNDTRIP errors=0 elements=4096'}[reg]
                assert marker in text,'Regression failed'
                if reg=='early' and on:
                    assert counts['spmEarlyAddrIssued']>0 and counts['spmEarlyAddrDisjoint']>0,'Early-address path unexercised'
                row=dict(regression=reg,passed=True)
            frozen();row.update(path=str(path),early_address=on,early=counts,host_seconds=time.monotonic()-start)
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
    def pair(t=None,reg='early'):
        with cf.ThreadPoolExecutor(max_workers=2) as pool:
            fs=[pool.submit(run,on,t,reg) for on in (False,True)]
            for f in cf.as_completed(fs):f.result()
    try:
        status();deadline=time.monotonic()+10800
        while not (BUILD/'frozen.json').exists():
            if (BUILD/'status.json').exists():
                assert json.loads((BUILD/'status.json').read_text())['stage']!='failed','Build failed'
            assert time.monotonic()<deadline,'Build timeout';time.sleep(5)
        assert json.loads((FORWARD/'experiment_v3/status.json').read_text())['stage']=='completed_validated','Prior pair not complete'
        shutil.copytree(BUILD/'configs',OUT/'configs')
        shutil.copy2(__file__,OUT/'runner_snapshot.py')
        manifest.update(json.loads((BUILD/'frozen.json').read_text()))
        paths=list((OUT/'configs').rglob('*.py'))+[Path(__file__).resolve(),BASE/'tests/guest',
            m.GUESTS/'roundtrip/guest',FORWARD/'tests_v2/guest',m.GUESTS/'build_spm/guest.final',Path(m.__file__).resolve()]
        for old in OLD.values():paths += [old/n for n in ('config.json','stats.txt','output.fp16','command.json','validated.json')]
        for t in (512,4096):paths.append(m.REF/f't{t}_bc128.npy')
        manifest.update({str(p):m.sha(p) for p in paths});m.save(OUT/'inputs.json',manifest)
        status('regressions');pair();pair(reg='overlap');pair(reg='roundtrip')
        for t in (512,4096):
            status(f'fa_t{t}');pair(t)
            if t==512:m.save(OUT/'gates_passed.json',dict(passed=True))
        status('completed_validated')
    except BaseException as exc:
        state['error']=repr(exc);status('failed');raise
if __name__=='__main__':main()
