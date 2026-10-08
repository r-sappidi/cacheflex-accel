"""Gated 0/4/16-entry ingress-bypass comparison, cache and SPM, fixed hardware."""
import concurrent.futures as cf
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
from run_access_aggregate import validate_aggregate
sys.path.insert(0,str(ROOT/'benchmarks/cacheflex_flash_attention'))
from single_head_reference import check
import numpy as np

OUT=ROOT/'results/cacheflex_flash_attention/ingress_bypass_20260922'
BUILD=OUT/os.environ.get('CACHEFLEX_BYPASS_BUILD','build')
EX=OUT/os.environ.get('CACHEFLEX_BYPASS_EXPERIMENT','experiment')
OLD=ROOT/'results/cacheflex_flash_attention/micro_resources_20260921/experiment_v6'
ENV=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib',
         OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
for _key in ('CACHEFLEX_AGGREGATE_PATH','CACHEFLEX_AGGREGATE_START','CACHEFLEX_AGGREGATE_END'):
    ENV.pop(_key,None)

def audit_config(old,new,lookahead):
    def walk(x,root):
        if isinstance(x,list):return [walk(v,root) for v in x]
        if isinstance(x,dict):
            result={k:walk(v,root) for k,v in x.items()}
            if x.get('type')=='MessageBuffer':
                actual=result.pop('cacheflex_ingress_lookahead',0)
                is_target=x.get('name')=='mandatoryQueue'
                # Parameter checking uses SimObject path if name is not emitted.
                result['__lookahead_actual__']=actual
            return result
        return x.replace(str(root),'<OUT>') if isinstance(x,str) else x
    a=json.loads((old/'config.json').read_text());b=json.loads((new/'config.json').read_text())
    def compare(a,b,path=''):
        if isinstance(a,dict):
            extra=set(b)-set(a)
            if a.get('type')=='MessageBuffer':
                expected=lookahead if path.endswith('.mandatoryQueue') else 0
                assert b.get('cacheflex_ingress_lookahead')==expected,(path,b.get('cacheflex_ingress_lookahead'),expected)
                extra.discard('cacheflex_ingress_lookahead')
            assert not extra and not set(a)-set(b),(path,extra,set(a)-set(b))
            for k in a:compare(a[k],b[k],path+'.'+k)
        elif isinstance(a,list):
            assert len(a)==len(b),path
            for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+f'[{i}]')
        elif isinstance(a,str):assert a.replace(str(old),'<OUT>')==b.replace(str(new),'<OUT>'),(path,a,b)
        else:assert a==b,(path,a,b)
    compare(a,b)

def baseline_stats(old,new):
    a=stats(old);b=stats(new);assert len(a)==len(b)
    for i,(before,after) in enumerate(zip(a,b)):
        added={k:v for k,v in after.items() if k not in before}
        assert all('.m_ingressBypass' in k and v==0 for k,v in added.items()),added
        assert before=={k:v for k,v in after.items() if k not in added},('Default drift',i,
            [(k,v,after.get(k)) for k,v in before.items() if after.get(k)!=v][:20])

def aggregate_metrics(p,expected):
    d=json.loads((p/'aggregate.json').read_text());validate_aggregate(d,expected)
    s=stats(p)[0];g=d['groups']['all'];c=g['category_ticks']
    for core in range(8):
        a=d['groups'][f'core_{core}'];pre=sum(v for k,v in a['category_ticks'].items() if k.startswith('CPU_'))
        actual=a['total_ticks']+a['ack_after_install_ticks']-pre
        assert actual==400*s[f'system.ruby.l0_cntrl{core}.sequencer.m_spmcpOutstandingCycleArea'],core
    return dict(copies=expected,mean_copy_cycles=g['total_ticks']/expected/400,
        ingress_to_first_attempt_cycles=c.get('L1_ingress_FIFO_wait',0)/expected/400,
        ingress_to_successful_forward_cycles=sum(v for k,v in c.items() if
            k=='L1_ingress_FIFO_wait' or k.startswith('L1_forward'))/expected/400,
        numeric_aggregate=d)

def main():
    EX.mkdir();state=dict(pid=os.getpid(),active={},completed=[]);lock=threading.RLock();stop=threading.Event()
    def status(stage=None):
        with lock:
            if stage:state['stage']=stage
            state['updated']=time.time();save(EX/'status.json',state)
    def write_report():
        fa=[r for r in state['completed'] if 'arch' in r]
        text=['# Bounded ingress lookahead: fixed-kernel FA comparison','',
              'Lookahead counts younger entries beyond the head. Stores/special operations are ordering boundaries.',
              'Same storage, transaction credits and controller transition bandwidth. Failed alternative attempts consume the existing transition budget.',
              'First-attempt FIFO time can fall merely because probing happens earlier: use ingress-to-successful-forward time and end-to-end ROI.',
              'Incomplete tables are NOT final results. No new kernel tuning.','',
              '| T | Backend | Lookahead | ROI us | Speedup vs own W0 | SPM ingress-to-forward cycles | Successful bypasses |',
              '|---|---|---:|---:|---:|---:|---:|']
        for r in fa:
            base=next((x for x in fa if x['T']==r['T'] and x['arch']==r['arch'] and x['lookahead']==0),None)
            ratio=base['roi_us']/r['roi_us'] if base else None
            ingress=r.get('aggregate',{}).get('ingress_to_successful_forward_cycles')
            text.append(f'| {r["T"]} | {r["arch"]} | {r["lookahead"]} | {r["roi_us"]:.4f} | '+
                (f'{ratio:.6f}x' if ratio is not None else 'pending')+' | '+
                (f'{ingress:.3f}' if ingress is not None else '—')+f' | {r["bypass_dequeues"]:.0f} |')
        text+=['','Matched-platform cache/SPM ratios:','']
        for t in (512,4096):
            for w in (0,4,16):
                pair={r['arch']:r for r in fa if r['T']==t and r['lookahead']==w}
                if len(pair)==2:text.append(f'- T{t}, lookahead{w}: {pair["cache"]["roi_us"]/pair["spm"]["roi_us"]:.6f}x.')
        (EX/'REPORT.md').write_text('\n'.join(text)+'\n')
    def execute(p,cmd,env):
        proc=None;started=time.monotonic()
        try:
            with (p/'run.log').open('w') as log:
                proc=subprocess.Popen(cmd,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                with lock:state['active'][p.name]=dict(pid=proc.pid,path=str(p));status()
                while proc.poll() is None:
                    assert not stop.is_set(),'Peer failed'
                    assert time.monotonic()-started<7200,'Two-hour case cap'
                    assert shutil.disk_usage(EX).free>50*1024**3,'Disk guard'
                    try:proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:pass
            assert proc.returncode==0,(p.name,proc.returncode)
            return time.monotonic()-started
        except BaseException:
            stop.set()
            if proc and proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
            raise
    def command(old,p,w):
        cmd=json.loads((old/'command.json').read_text())
        cmd=[f'--outdir={p}' if a.startswith('--outdir=') else a for a in cmd]
        cmd[0]=str(BUILD/'gem5.opt');cmd[2]=str(BUILD/'configs/deprecated/example/bounded_se.py')
        for c in range(8):cmd+=['-P',f'system.ruby.l0_cntrl{c}.mandatoryQueue.cacheflex_ingress_lookahead={w}']
        return cmd
    try:
        status('waiting_for_build')
        while True:
            p=BUILD/'status.json';b=json.loads(p.read_text()) if p.exists() else {}
            assert b.get('stage')!='failed',b
            if b.get('stage')=='built_requires_validation':break
            time.sleep(5)
        guests=[ROOT/'results/cacheflex_flash_attention/micro_resources_20260921/roundtrip_guest/guest',
                ROOT/'results/cacheflex_flash_attention/micro_resources_20260921/roundtrip_raw_guest/guest',
                ROOT/'results/ruby_getx_race_guest_20260916/guest']
        paths=[BUILD/'gem5.opt',Path(__file__),ROOT/'scripts/run_access_aggregate.py',
               ROOT/'benchmarks/cacheflex_flash_attention/single_head_reference.py',*guests]
        paths+=list((BUILD/'configs').rglob('*.py'))
        refs={t:OLD/'preserved'/f't{t}_bc128.npy' for t in (512,4096)};paths+=list(refs.values())
        for t in (512,4096):
            for a in ('cache','spm'):
                old=OLD/'runs'/f'{a}_bounded_t{t}';cmd=json.loads((old/'command.json').read_text())
                paths += [old/'command.json',old/'stats.txt',old/'config.json',old/'output.fp16',Path(cmd[cmd.index('--cmd')+1])]
        manifest={str(p):sha(p) for p in paths};save(EX/'inputs.json',manifest)
        def frozen():
            for p,h in manifest.items():assert sha(p)==h,('Changed input',p)
        status('ordering_smokes')
        for label,guest,marker in zip(('roundtrip','roundtrip_raw','release_getx'),guests,
            ('SPM_ROUNDTRIP PASS errors=0','SPM_ROUNDTRIP PASS errors=0','SPM_RELEASE_GETX PASS errors=0')):
            frozen();p=EX/label;p.mkdir();old=OLD/'runs/spm_bounded_t512';cmd=command(old,p,16)
            cmd[cmd.index('--cmd')+1]=str(guest)
            cmd=["system.cpu[0].workload[0].cmd=['spm-smoke']" if a.startswith('system.cpu[0].workload[0].cmd=') else
                 a.replace('sve_vl_se=16','sve_vl_se=4') for a in cmd]
            save(p/'command.json',cmd);seconds=execute(p,cmd,ENV)
            assert marker in (p/'run.log').read_text(),(label,marker)
            save(p/'validated.json',dict(passed=True,marker=marker,host_seconds=seconds));frozen()
            with lock:state['active'].pop(label);state['completed'].append(dict(name=label,passed=True));status()
        def run(arch,t,w):
            frozen();name=f'{arch}_t{t}_w{w}';p=EX/name;p.mkdir();old=OLD/'runs'/f'{arch}_bounded_t{t}'
            cmd=command(old,p,w);save(p/'command.json',cmd);env=dict(ENV)
            if arch=='spm':env['CACHEFLEX_AGGREGATE_PATH']=str(p/'aggregate.json')
            try:
                seconds=execute(p,cmd,env);audit_config(old,p,w)
                assert sha(p/'output.fp16')==sha(old/'output.fp16'),'Output differs from correct fixed-kernel control'
                arithmetic=check(p/'output.fp16',np.load(refs[t]));numeric=stats(p);s=numeric[0]
                copies=sum(v for k,v in s.items() if k.endswith('.m_spmcpRequestsIssued'))
                completed=sum(v for k,v in s.items() if k.endswith('.m_spmcpRequestsCompleted'))
                expected=(t//64)*t*2 if arch=='spm' else 0
                assert copies==completed==expected,(copies,completed,expected)
                if w==0:baseline_stats(old,p)
                row=dict(name=name,arch=arch,T=t,lookahead=w,roi_us=s['simTicks']/1e6,
                         host_seconds=seconds,arithmetic=arithmetic,exact_output=True,config_audit=True,
                         bypass_dequeues=sum(v for k,v in s.items() if k.endswith('.m_ingressBypassDequeues')),
                         bypass_counters={k:v for k,v in s.items() if '.m_ingressBypass' in k and '.mandatoryQueue.' in k})
                if arch=='spm':row['aggregate']=aggregate_metrics(p,expected)
                frozen();save(p/'validated.json',row)
                with lock:
                    state['active'].pop(name);state['completed'].append({k:v for k,v in row.items() if k!='bypass_counters' and k!='aggregate'} | ({'aggregate':{k:v for k,v in row['aggregate'].items() if k!='numeric_aggregate'}} if arch=='spm' else {}))
                    status();write_report()
            except BaseException as exc:
                stop.set();save(p/'failed.json',dict(error=repr(exc)));raise
        for t in (512,4096):
            for w in (0,4,16):
                status(f't{t}_lookahead{w}')
                with cf.ThreadPoolExecutor(max_workers=2) as pool:
                    fs=[pool.submit(run,a,t,w) for a in ('cache','spm')]
                    for f in cf.as_completed(fs):f.result()
        status('completed_validated');write_report()
    except BaseException as exc:
        stop.set();state['error']=repr(exc);status('failed');raise

if __name__=='__main__':main()
