"""Gated, frozen T512 observer checks then scaled eight-core T4096 FA replay."""
import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

from build_access_timing import ROOT, sha, save
from run_access_timing import stats

OUT=ROOT/'results/cacheflex_flash_attention/access_aggregate_20260922'
BUILD=OUT/os.environ.get('CACHEFLEX_AGGREGATE_BUILD','build')
EX=OUT/os.environ.get('CACHEFLEX_AGGREGATE_EXPERIMENT','experiment')
PRIOR=ROOT/'results/cacheflex_flash_attention/micro_resources_20260921/experiment_v6'
sys.path.insert(0,str(ROOT/'benchmarks/cacheflex_flash_attention'))
from single_head_reference import check
import numpy as np


def validate_aggregate(data,expected):
    assert data['begun']==data['completed']==expected, data
    for k in ('inflight','live_messages','live_packets','missing_install','overlap_ticks','llc_match_failures'):
        assert data[k]==0,(k,data[k])
    groups=data['groups'];assert groups['all']['count']==expected
    for key,group in groups.items():
        assert sum(group['category_ticks'].values())==group['total_ticks'],key
        assert sum(group['log2_tick_histogram'])==group['count'],key
    for prefix in ('core_','path_'):
        selected=[g for k,g in groups.items() if k.startswith(prefix)]
        assert sum(g['count'] for g in selected)==expected,prefix
        assert sum(g['total_ticks'] for g in selected)==groups['all']['total_ticks'],prefix
    assert not groups.get('path_unclassified')
    return dict(all_copies_accounted=True,additive_partition=True,
                no_same_priority_overlap=True,bounded_inflight_metadata=True)


def report(path,data,roi):
    a=data['groups']['all'];total=a['total_ticks'];count=a['count']
    rows=['# Whole-ROI SPMCP latency aggregate','',
          f'{count:,} copies; FA ROI {roi:.4f} us. Mean issue-to-install: {total/count/400:.3f} core cycles.',
          f'Live metadata peak: {data["max_inflight"]} copies, {data["max_messages"]} messages, {data["max_packets"]} packets.','',
          'These percentages partition summed copy lifetimes, NOT end-to-end FA runtime. Copies overlap.',
          'Router waiting is modeled Garnet contention. SRAM bank arbitration is not modeled.',
          'Source-service/destination residual includes controller scheduling, coherence/memory service, and destination readiness; it is not assumed to be DRAM time.',
          'Histograms store log2 tick ranges, not exact percentiles.','',
          '| Component | Mean core cycles/copy | Share of copy latency |','|---|---:|---:|']
    for k,v in sorted(a['category_ticks'].items(),key=lambda x:-x[1]):
        rows.append(f'| {k} | {v/count/400:.3f} | {100*v/total:.3f}% |')
    rows+=['','| Source path | Copies | Mean cycles |','|---|---:|---:|']
    for k,g in data['groups'].items():
        if k.startswith('path_'):rows.append(f'| {k[5:]} | {g["count"]:,} | {g["total_ticks"]/g["count"]/400:.3f} |')
    rows+=['',f'Completion acknowledgement follows installation by {a["ack_after_install_ticks"]/count/400:.3f} mean cycles (excluded from above).','']
    (path/'BREAKDOWN.md').write_text('\n'.join(rows))


def main():
    EX.mkdir();state=dict(pid=os.getpid(),completed=[],active={});lock=threading.RLock();stop=threading.Event()
    def status(stage=None):
        with lock:
            if stage:state['stage']=stage
            state['updated']=time.time();save(EX/'status.json',state)
    status('waiting_for_build')
    try:
        while True:
            p=BUILD/'status.json'
            b=json.loads(p.read_text()) if p.exists() else {}
            assert b.get('stage')!='failed',b
            if b.get('stage')=='built_requires_validation':break
            time.sleep(5)
        paths=[BUILD/'gem5.opt',Path(__file__),ROOT/'scripts/run_access_timing.py',
               ROOT/'scripts/build_access_timing.py',ROOT/'benchmarks/cacheflex_flash_attention/single_head_reference.py']
        for t in (512,4096):
            for arch in ('cache','spm'):
                old=PRIOR/'runs'/f'{arch}_bounded_t{t}'
                cmd=json.loads((old/'command.json').read_text())
                paths+=[old/'command.json',old/'accepted.json',old/'config.json',old/'stats.txt',
                        old/'output.fp16',Path(cmd[cmd.index('--cmd')+1])]
        paths+=list((BUILD/'configs').rglob('*.py'))
        refs={t:PRIOR/'preserved'/f't{t}_bc128.npy' for t in (512,4096)}
        paths+=list(refs.values())
        manifest={str(p):sha(p) for p in paths};save(EX/'inputs.json',manifest)
        def frozen():
            for p,h in manifest.items():assert sha(p)==h,('Changed dependency',p)
        def run(arch,t,enabled,label):
            frozen();p=EX/label;p.mkdir();old=PRIOR/'runs'/f'{arch}_bounded_t{t}'
            original=json.loads((old/'command.json').read_text())
            cmd=[f'--outdir={p}' if x.startswith('--outdir=') else x for x in original]
            cmd[0]=str(BUILD/'gem5.opt');cmd[2]=str(BUILD/'configs/deprecated/example/bounded_se.py')
            save(p/'command.json',cmd)
            before=stats(old);start=int(before[0]['finalTick']-before[0]['simTicks']);end=int(before[0]['finalTick'])
            env=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
            for k in ('CACHEFLEX_AGGREGATE_PATH','CACHEFLEX_AGGREGATE_START','CACHEFLEX_AGGREGATE_END','CACHEFLEX_TRACE_LINE'):
                env.pop(k,None)
            if enabled:env.update(CACHEFLEX_AGGREGATE_PATH=str(p/'aggregate.json'),CACHEFLEX_AGGREGATE_START=str(start),CACHEFLEX_AGGREGATE_END=str(end))
            save(p/'observer.json',dict(enabled=enabled,start=start,end=end))
            proc=None;began=time.monotonic()
            try:
                with (p/'run.log').open('w') as log:
                    proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                    with lock:state['active'][label]=dict(pid=proc.pid,path=str(p));status()
                    while proc.poll() is None:
                        assert not stop.is_set(),'Peer failed'
                        assert time.monotonic()-began<7200,'Two-hour diagnostic cap'
                        assert shutil.disk_usage(EX).free>50*1024**3,'Disk guard'
                        try:proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:pass
                assert proc.returncode==0,(label,proc.returncode)
                frozen()
                assert sha(p/'output.fp16')==sha(old/'output.fp16'),'Output bytes differ'
                after=stats(p)
                assert before==after,('Non-host stats differ',[(i,k,v,after[i].get(k)) for i,e in enumerate(before) for k,v in e.items() if i>=len(after) or after[i].get(k)!=v][:30])
                def norm(x,root):
                    if isinstance(x,dict):return {k:norm(v,root) for k,v in x.items()}
                    if isinstance(x,list):return [norm(v,root) for v in x]
                    return x.replace(str(root),'<OUT>') if isinstance(x,str) else x
                assert norm(json.loads((old/'config.json').read_text()),old)==norm(json.loads((p/'config.json').read_text()),p),'Config drift'
                arithmetic=check(p/'output.fp16',np.load(refs[t]))
                prior=json.loads((old/'accepted.json').read_text());roi=prior['counters']['roi_us']
                validation=dict(exact_output=True,exact_all_finite_nonhost_stats=True,
                                exact_config_except_outdir=True,arithmetic=arithmetic,
                                roi_us=roi,host_seconds=time.monotonic()-began)
                if enabled:
                    data=json.loads((p/'aggregate.json').read_text())
                    validation['aggregate']=validate_aggregate(data,int(prior['counters']['m_spmcpRequestsIssued']))
                    report(p,data,roi)
                save(p/'validated.json',validation)
                with lock:state['active'].pop(label);state['completed'].append(dict(name=label,**validation));status()
            except BaseException as exc:
                stop.set()
                if proc and proc.poll() is None:
                    proc.terminate()
                    try:proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()
                save(p/'failed.json',dict(error=repr(exc)));raise
        def pair(cases):
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                futures=[pool.submit(run,*args) for args in cases]
                for f in concurrent.futures.as_completed(futures):f.result()
        status('t512_observer_equivalence')
        pair([('spm',512,False,'spm_t512_off'),('spm',512,True,'spm_t512_on')])
        status('t4096_target')
        pair([('cache',4096,False,'cache_t4096'),('spm',4096,True,'spm_t4096')])
        status('completed_validated')
    except BaseException as exc:
        state['error']=repr(exc);status('failed');raise

if __name__=='__main__':main()
