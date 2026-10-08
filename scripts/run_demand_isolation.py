"""Causal sensitivity of FA to copy-induced demand head-of-line blocking.

Frozen resources and kernels, no extra request/data storage. Verify the targeted
hit-delay intervention in a bounded trace, then evaluate full FA ROI latency.
"""
import concurrent.futures as cf
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

from build_access_timing import ROOT,sha,save
from run_access_timing import stats
from run_ingress_bypass import aggregate_metrics
sys.path.insert(0,str(ROOT/'benchmarks/cacheflex_flash_attention'))
from single_head_reference import check
import numpy as np

OUT=ROOT/'results/cacheflex_flash_attention/demand_isolation_20260922'
BUILD=OUT/'build'
EX=OUT/'experiment'
OLD=ROOT/'results/cacheflex_flash_attention/ingress_bypass_20260922/experiment'
REF=ROOT/'results/cacheflex_flash_attention/micro_resources_20260921/experiment_v6/preserved'
ENV=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib',
         OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
for k in list(ENV):
    if k.startswith('CACHEFLEX_AGGREGATE') or k=='CACHEFLEX_TRACE_LINE':ENV.pop(k)


def audit(old,new,enabled):
    def compare(a,b,path=''):
        if isinstance(a,dict):
            extra=set(b)-set(a)
            if a.get('type')=='MessageBuffer':
                expected=bool(enabled and path.endswith('.mandatoryQueue'))
                assert b.get('cacheflex_copy_demand_bypass')==expected,(path,expected)
                extra.discard('cacheflex_copy_demand_bypass')
            assert not extra and not set(a)-set(b),(path,extra,set(a)-set(b))
            for k in a:compare(a[k],b[k],path+'.'+k)
        elif isinstance(a,list):
            assert len(a)==len(b),path
            for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+f'[{i}]')
        elif isinstance(a,str):assert a.replace(str(old),'<OUT>')==b.replace(str(new),'<OUT>'),(path,a,b)
        else:assert a==b,(path,a,b)
    compare(json.loads((old/'config.json').read_text()),json.loads((new/'config.json').read_text()))


def main():
    EX.mkdir();state=dict(pid=os.getpid(),stage='waiting_for_build',active={},completed=[])
    lock=threading.RLock();stop=threading.Event()
    def status(stage=None):
        with lock:
            if stage:state['stage']=stage
            state['updated']=time.time();save(EX/'status.json',state)
    def report():
        rows=[r for r in state['completed'] if r.get('kind')=='full_fa']
        text=['# Demand/copy ingress isolation: end-to-end sensitivity','',
              'Only ingress selection changes. No added data/queue capacity, miss credits, network bandwidth or controller transitions.',
              'Full-queue search cost is unmodeled; this is a diagnostic sensitivity, not an RTL performance claim.',
              'Incomplete cases are not performance results. Full-FA ROIs include the original preparation and synchronization.',
              '', '| T | Backend | Isolation | ROI us | Improvement vs same-backend OFF |',
              '|---|---|---|---:|---:|']
        for r in rows:
            off=next((v for v in rows if v['T']==r['T'] and v['arch']==r['arch'] and not v['enabled']),None)
            delta=f'{100*(1-r["roi_us"]/off["roi_us"]):.4f}%' if off else 'pending'
            text.append(f'| {r["T"]} | {r["arch"]} | {r["enabled"]} | {r["roi_us"]:.4f} | {delta} |')
        for t in (512,4096):
            for en in (False,True):
                pair={r['arch']:r for r in rows if r['T']==t and r['enabled']==en}
                if len(pair)==2:text.append(f'\nT{t}, isolation={en}: cache/SPM = {pair["cache"]["roi_us"]/pair["spm"]["roi_us"]:.6f}x.')
        if any(r.get('kind')=='trace_analysis' for r in state['completed']):
            text+=['','See mechanism.json for the matched panel/driver-load intervention check. Request waits are not whole-FA causal fractions.']
        (EX/'REPORT.md').write_text('\n'.join(text)+'\n')
    def execute(p,cmd,env):
        proc=None;start=time.monotonic()
        try:
            with (p/'run.log').open('w') as log:
                proc=subprocess.Popen(cmd,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                with lock:state['active'][p.name]=dict(pid=proc.pid,path=str(p));status()
                while proc.poll() is None:
                    assert not stop.is_set(),'Peer failed'
                    assert time.monotonic()-start<7200,'Two-hour case limit'
                    assert shutil.disk_usage(EX).free>50*2**30,'Disk guard'
                    trace=p/'trace.gz'
                    assert not trace.exists() or trace.stat().st_size<4*2**30,'Trace size guard'
                    try:proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:pass
            assert proc.returncode==0,(p.name,proc.returncode)
            with lock:state['active'].pop(p.name);status()
            return time.monotonic()-start
        except BaseException:
            stop.set()
            if proc and proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
            raise
    def command(old,p,en):
        cmd=json.loads((old/'command.json').read_text())
        cmd=[f'--outdir={p}' if x.startswith('--outdir=') else x for x in cmd]
        cmd[0]=str(BUILD/'gem5.opt');cmd[2]=str(BUILD/'configs/deprecated/example/bounded_se.py')
        for i in range(8):cmd+=['-P',f'system.ruby.l0_cntrl{i}.mandatoryQueue.cacheflex_copy_demand_bypass={en!r}']
        return cmd
    try:
        status()
        while True:
            p=BUILD/'status.json';s=json.loads(p.read_text()) if p.exists() else {}
            assert s.get('stage')!='failed',s
            if s.get('stage')=='built_requires_validation':break
            time.sleep(5)
        guests=[ROOT/'results/cacheflex_flash_attention/micro_resources_20260921/roundtrip_guest/guest',
                ROOT/'results/cacheflex_flash_attention/micro_resources_20260921/roundtrip_raw_guest/guest',
                ROOT/'results/ruby_getx_race_guest_20260916/guest']
        paths=[Path(__file__).resolve(),BUILD/'gem5.opt',*guests,
               ROOT/'scripts/analyze_ingress_trace.py',ROOT/'scripts/analyze_ingress_credits.py',
               ROOT/'scripts/run_ingress_bypass.py',ROOT/'scripts/run_access_timing.py',
               ROOT/'scripts/run_access_aggregate.py',ROOT/'benchmarks/cacheflex_flash_attention/single_head_reference.py']
        paths+=list((BUILD/'configs').rglob('*.py'))
        for t in (512,4096):
            paths.append(REF/f't{t}_bc128.npy')
            for arch in ('cache','spm'):
                old=OLD/f'{arch}_t{t}_w0';c=json.loads((old/'command.json').read_text())
                paths += [old/'validated.json',old/'command.json',old/'config.json',old/'stats.txt',old/'output.fp16',Path(c[c.index('--cmd')+1])]
        manifest={str(p):sha(p) for p in paths};save(EX/'inputs.json',manifest)
        def frozen():
            for p,h in manifest.items():assert sha(Path(p))==h,('Changed frozen input',p)
        def accepted(row,p=None):
            frozen()
            if p:save(p/'validated.json',row)
            with lock:
                state['completed'].append({k:v for k,v in row.items() if k!='numeric_aggregate'})
                status();report()
        status('ordering_regressions')
        for name,g,marker in zip(('roundtrip','roundtrip_raw','release_getx'),guests,
            ('SPM_ROUNDTRIP PASS errors=0','SPM_ROUNDTRIP PASS errors=0','SPM_RELEASE_GETX PASS errors=0')):
            frozen();p=EX/name;p.mkdir();cmd=command(OLD/'spm_t512_w0',p,True)
            cmd[cmd.index('--cmd')+1]=str(g)
            cmd=["system.cpu[0].workload[0].cmd=['spm-smoke']" if x.startswith('system.cpu[0].workload[0].cmd=')
                 else x.replace('sve_vl_se=16','sve_vl_se=4') for x in cmd]
            save(p/'command.json',cmd);seconds=execute(p,cmd,ENV)
            assert marker in (p/'run.log').read_text(),name
            accepted(dict(name=name,kind='ordering_regression',passed=True,host_seconds=seconds),p)
        def fa(arch,t,en):
            frozen();name=f'{arch}_t{t}_{"on" if en else "off"}';p=EX/name;p.mkdir()
            old=OLD/f'{arch}_t{t}_w0';cmd=command(old,p,en);save(p/'command.json',cmd)
            env=dict(ENV)
            if arch=='spm':env['CACHEFLEX_AGGREGATE_PATH']=str(p/'aggregate.json')
            seconds=execute(p,cmd,env);audit(old,p,en)
            assert sha(p/'output.fp16')==sha(old/'output.fp16'),'Output differs from validated control'
            arithmetic=check(p/'output.fp16',np.load(REF/f't{t}_bc128.npy'))
            s=stats(p)[0]
            if not en or arch=='cache':assert stats(p)==stats(old),('OFF/cache control drift',name)
            expected=(t//64)*t*2 if arch=='spm' else 0
            assert sum(v for k,v in s.items() if k.endswith('.m_spmcpRequestsIssued'))==expected
            assert sum(v for k,v in s.items() if k.endswith('.m_spmcpRequestsCompleted'))==expected
            row=dict(name=name,kind='full_fa',arch=arch,T=t,enabled=en,roi_us=s['simTicks']/1e6,
                     host_seconds=seconds,arithmetic=arithmetic,exact_output=True,config_audit=True,
                     exact_control_stats=(not en or arch=='cache'),
                     bypass_dequeues=sum(v for k,v in s.items() if k.endswith('.m_ingressBypassDequeues')))
            if arch=='spm':
                a=aggregate_metrics(p,expected)
                row['aggregate']={k:v for k,v in a.items() if k!='numeric_aggregate'}
            accepted(row,p)
        def pair(fn,args):
            with cf.ThreadPoolExecutor(max_workers=2) as pool:
                fs=[pool.submit(fn,*a) for a in args]
                try:
                    for f in cf.as_completed(fs):f.result()
                except BaseException:stop.set();raise
        for en in (False,True):
            status(f't512_{"on" if en else "off"}_gates')
            pair(fa,[(a,512,en) for a in ('cache','spm')])
        status('targeted_t4096_prefix_trace')
        def prefix(en):
            frozen();p=EX/f'prefix_{"on" if en else "off"}';p.mkdir()
            old=OLD/'spm_t4096_w0';cmd=command(old,p,en)
            cmd[1:1]=['--debug-flags=CacheFlexTiming,ProtocolTrace,SPM_CPU,Commit,LSQUnit',
                      '--debug-start=2518000000','--debug-end=2554000000','--debug-file=trace.gz',
                      '--debug-ignore='+':'.join(f'system.cpu{i}' for i in range(1,8))]
            cmd+=['--abs-max-tick=2554000000'];save(p/'command.json',cmd)
            seconds=execute(p,cmd,ENV);audit(old,p,en)
            assert 'simulate() limit reached' in (p/'run.log').read_text()
            assert int(stats(p)[-1]['finalTick'])==2554000000
            if not en:
                prior=ROOT/'results/cacheflex_flash_attention/ingress_trace_20260922/fine_control/w0'
                assert stats(p)==stats(prior),'Prefix control drift'
            row=dict(kind='bounded_prefix',enabled=en,diagnostic_only=True,final_tick=2554000000,
                     exact_config_except_intervention=True,host_seconds=seconds)
            save(p/'validated_prefix.json',row);accepted(row)
        pair(prefix,[(False,),(True,)])
        status('analyzing_targeted_trace')
        from analyze_ingress_trace import analyze
        from analyze_ingress_credits import coherent_queue_waits
        mechanism={}
        for en in (False,True):
            p=EX/f'prefix_{"on" if en else "off"}';r=analyze(p)
            panel=next(x for x in r['panels'] if x['copies'][0]['src']==0x138040 and x['complete'])
            waits=coherent_queue_waits(json.loads((p/'resource_events.json').read_text()))
            hits=[x for x in waits if x['addr']==0xa7980 and panel['first_accept']<=x['accepted']<panel['first_consumer_issue']]
            assert hits,'Missing identified driver hit; inspect trace, do not infer effect'
            target=min(hits,key=lambda x:x['accepted'])
            mechanism['on' if en else 'off']=dict(driver_hit=target,
                panel_first_accept=panel['first_accept'],panel_last_ack=panel['last_ack'],
                first_V_load=panel['first_consumer_issue'],consumer_slack_cycles=panel['all_copies_ready_before_first_consumer_cycles'])
        mechanism['target_wait_reduction_cycles']=mechanism['off']['driver_hit']['wait_cycles']-mechanism['on']['driver_hit']['wait_cycles']
        mechanism['target_intervention_succeeded']=mechanism['target_wait_reduction_cycles']>0
        mechanism['not_whole_FA_cycles']=True
        save(EX/'mechanism.json',mechanism)
        accepted(dict(kind='trace_analysis',**mechanism))
        for en in (False,True):
            status(f't4096_{"on" if en else "off"}_full_ROI')
            pair(fa,[(a,4096,en) for a in ('cache','spm')])
        status('completed_validated');report()
    except BaseException as exc:
        stop.set();state['error']=repr(exc);status('failed');raise


if __name__=='__main__':main()
