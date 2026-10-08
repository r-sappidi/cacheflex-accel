"""Resume the audited output experiment without discarding completed gates.

Historical SPM counters are not an equivalence oracle after ordering repairs.
Require numerical/config/resource correctness and identical committed work;
require exact counter reproducibility against the SAME repaired simulator.
"""
import concurrent.futures as cf
import json
from pathlib import Path
import shutil
import subprocess
import threading
import time
import run_fa_output_resident as m

ROOT=m.ROOT
PREVIOUS=m.BASE/'output_resident_v8_20260924'
OUT=m.BASE/'output_resident_v10_20260924'
REPEAT=m.BASE/'output_resident_v9_20260924/runs/recheck_spm_t4096'
AUDIT=m.GUESTS/'control_audit'
m.OUT=OUT

def committed_work(block):
    return {k:v for k,v in block.items()
            if k in ('simInsts','simOps') or '.commit.committedInstType_' in k}

def validate_control(path,kind):
    backend=kind.removeprefix('recheck_')
    row=m.validate_old(path,backend,4096,'control')
    old=m.stats(m.prior(backend,4096));new=m.stats(path)
    assert len(old)==len(new)
    # The first dump is the timed FA ROI. The final dump also includes
    # thread shutdown/export, whose instruction counts can change with
    # scheduling after a model repair. Same-model replay below still checks
    # EVERY dump and EVERY finite numeric counter without exclusions.
    assert committed_work(old[0])==committed_work(new[0]), \
        'Committed architectural work changed inside the FA ROI'
    delta=[{k:[a.get(k),b.get(k)] for k in sorted(a.keys()|b.keys())
            if a.get(k)!=b.get(k)} for a,b in zip(old,new)]
    m.save(path/'historical_counter_changes.json',delta)
    if backend=='cache':
        assert old==new,'Cache-only model unexpectedly changed'
    else:
        same=m.stats(PREVIOUS/'runs/recheck_spm_t4096')
        assert same==new,'Same-model SPM replay is not exactly reproducible'
    row.update(kind=kind,reused=False,exact_historical_stats=old==new,
               committed_work_equal=True,same_model_stats_equal=True,
               changed_historical_counters=list(map(len,delta)))
    return row

def main():
    assert (AUDIT/'approved.json').exists(),'Finish causal audit before resuming'
    OUT.mkdir();(OUT/'runs').mkdir()
    state=dict(pid=__import__('os').getpid(),stage='revalidating_completed_gates',active={},completed=[])
    lock=threading.RLock();halt=threading.Event()
    def status(stage=None):
        with lock:
            if stage:state['stage']=stage
            state['updated']=time.time();m.save(OUT/'status.json',state);m.report(state)
            rows={r['kind']:r for r in state['completed'] if r['T']==4096}
            if 'output_spm' in rows:
                with (OUT/'REPORT.md').open('a') as f:
                    f.write('\n\n## Same repaired model comparisons\n\n')
                    for name in ('recheck_cache','recheck_spm','matched_cache'):
                        if name in rows:
                            f.write(f"- {name}/output-SPM: {rows[name]['roi_us']/rows['output_spm']['roi_us']:.6f}x.\n")
    status()
    manifest=json.loads((PREVIOUS/'inputs.json').read_text())
    # No old inputs have changed, including the old runner itself.
    for name,h in manifest.items():assert m.sha(name)==h,('Old frozen input changed',name)
    assert manifest[str(m.BUILD/'gem5.opt')]==m.sha(m.BUILD/'gem5.opt')
    shutil.copytree(m.BUILD/'configs',OUT/'configs')
    shutil.copy2(__file__,OUT/'runner_snapshot.py')
    for p in [Path(__file__).resolve(),AUDIT/'approved.json',AUDIT/'summary.json',
              PREVIOUS/'gates_passed.json',PREVIOUS/'runs/roundtrip/validated.json']:
        manifest[str(p)]=m.sha(p)
    for p in (OUT/'configs').rglob('*.py'):manifest[str(p)]=m.sha(p)
    assert json.loads((PREVIOUS/'runs/roundtrip/validated.json').read_text())['passed']
    for kind in ('recheck_cache','recheck_spm','matched_cache','output_spm'):
        p=PREVIOUS/'runs'/f'{kind}_t512';row=m.validate(p,kind,512)
        for name in ('command.json','config.json','run.log','stats.txt','output.fp16','validated.json'):
            f=p/name;manifest[str(f)]=m.sha(f)
        row.update(path=str(p),reused=True,reused_same_model=True);state['completed'].append(row)
    # This is a numerical/resource-validated result, not yet a reproducibility pass.
    p=PREVIOUS/'runs/recheck_spm_t4096'
    row=m.validate_old(p,'spm',4096,'control')
    for name in ('command.json','config.json','run.log','stats.txt','output.fp16'):
        f=p/name;manifest[str(f)]=m.sha(f)
    m.save(OUT/'prior_same_model_spm.json',row)
    # V9 completed this repeat and matches V8 on both complete stats blocks.
    # Its failure was the historical post-ROI instruction gate, not execution.
    repeated=validate_control(REPEAT,'recheck_spm')
    repeated.update(path=str(REPEAT),reused=True,reused_same_model=True)
    state['completed'].append(repeated)
    m.save(OUT/'reused_spm_repeat.json',repeated)
    for name in ('command.json','config.json','run.log','stats.txt','output.fp16'):
        f=REPEAT/name;manifest[str(f)]=m.sha(f)
    m.save(OUT/'inputs.json',manifest)
    def frozen():
        for p,h in manifest.items():assert m.sha(p)==h,('Frozen input changed',p)
    def run(kind):
        frozen();p=OUT/'runs'/f'{kind}_t4096';p.mkdir()
        cmd=m.command(kind,4096,p);m.save(p/'command.json',cmd)
        proc=None;begin=time.monotonic()
        try:
            with (p/'run.log').open('w') as log:
                proc=subprocess.Popen(cmd,cwd=ROOT,env=m.ENV,stdout=log,stderr=subprocess.STDOUT)
                with lock:state['active'][p.name]=dict(pid=proc.pid,path=str(p));status()
                while proc.poll() is None:
                    assert not halt.is_set(),'Peer failed'
                    assert time.monotonic()-begin<10800,'Three-hour limit'
                    assert shutil.disk_usage(OUT).free>50*2**30,'Disk guard'
                    try:proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:pass
            assert proc.returncode==0,(kind,proc.returncode)
            text=(p/'run.log').read_text()
            assert 'Simulated exit code not 0!' not in text,'Guest failed'
            row=validate_control(p,kind) if kind.startswith('recheck_') else m.validate(p,kind,4096)
            frozen();row.update(path=str(p),host_seconds=time.monotonic()-begin)
            m.save(p/'validated.json',row)
            with lock:state['active'].pop(p.name);state['completed'].append(row);status()
        except BaseException as exc:
            halt.set();m.save(p/'failed.json',dict(error=repr(exc)))
            if proc and proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
            with lock:state['active'].pop(p.name,None);status()
            raise
    try:
        for stage,kinds in [('cache_control_and_output_spm',('recheck_cache','output_spm')),
                            ('matched_cache',('matched_cache',))]:
            status(stage)
            with cf.ThreadPoolExecutor(max_workers=2) as pool:
                fs=[pool.submit(run,k) for k in kinds]
                for f in cf.as_completed(fs):f.result()
        status('completed_validated')
    except BaseException as exc:
        state['error']=repr(exc);status('failed');raise

if __name__=='__main__':main()
