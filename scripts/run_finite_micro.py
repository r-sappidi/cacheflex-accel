"""Isolated finite-resource correctness/guard validation; at most two sims."""
import concurrent.futures as cf
import json, math, os, re, shutil, subprocess, sys, threading, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
FA=ROOT/'benchmarks/cacheflex_flash_attention'
sys.path[:0]=[str(FA/'model_tuning'),str(FA),str(FA/'kv_partition')]
from build_access_timing import sha,save
from run_access_timing import stats
from fa_headroom_profiles import leaves
from test_micro_timing import check_global_guard
from export_safe_elf import export_hash
from reference import reference,validate
BASE=ROOT/'results/cacheflex_flash_attention'
BUILD=BASE/'finite_model_20260926'/os.environ.get('CACHEFLEX_FINITE_BUILD','build')
OUT=BASE/'finite_model_20260926'/os.environ.get('CACHEFLEX_FINITE_EXPERIMENT','experiment')
OLD=BASE/'model_tuning_20260926/experiment'
CONTROL=OLD/'runs/t1024_q_cache_br32_bc128_r192'
GUEST=BASE/'model_tuning_20260926/build/guest.export-safe'
ENV=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')

def command(c,path):
    args=list(map(str,(c['t'],8,c['br'],c['bc'],c['r']//c['br'],1024,c['kv'],c['spm'])))
    cmd=json.loads((CONTROL/'command.json').read_text())
    cmd[0]=str(BUILD/'gem5.opt');cmd[1]='--outdir='+str(path)
    cmd[2]=str(OUT/'configs/deprecated/example/micro_timing_se.py')
    cmd[cmd.index('--options')+1]=' '.join(args)
    for i,x in enumerate(cmd):
        if x.startswith('system.cpu[0].workload[0].cmd='):cmd[i]=f"system.cpu[0].workload[0].cmd={['kv-fa']+args!r}"
        if x.startswith('--l1d_num_spm_ways='):cmd[i]='--l1d_num_spm_ways='+str(6 if c['spm'] else 0)
        if '.cacheFlexBoundedTracking=' in x:cmd[i]=x.split('=')[0]+'='+str(c['finite'])
        if '.max_outstanding_requests=' in x:cmd[i]=x.split('=')[0]+'='+str(c['guard'])
        if c['small'] and re.search(r'\.(LQEntries|SQEntries)=',x):cmd[i]=x.split('=')[0]+'=8'
    if c['finite']:cmd.append('--cacheflex-finite-resources')
    if c['small']:cmd.append('--cacheflex-micro-small')
    return cmd,args

def audit(path,c,args):
    a=json.loads((CONTROL/'config.json').read_text().replace(str(CONTROL),'<OUT>'))
    b=json.loads((path/'config.json').read_text().replace(str(path),'<OUT>'))
    old,new=dict(leaves(a)),dict(leaves(b))
    delta={k:[old.get(k),new.get(k)] for k in old.keys()|new.keys() if old.get(k)!=new.get(k)}
    always=r'system.cpu\[\d+\].workload\[0\].cmd\[\d+\]|system.ruby.l1_cntrl\d+.cache.num_spm_ways|system.ruby.l1_cntrl\d+.returnBufferFromL0(\..*)?|system.ruby.network.cacheflex_request_vc_partition'
    finite=r'system.cpu\[\d+\].(cacheFlexBoundedTracking|LQEntries|SQEntries)|system.ruby\..*\.(buffer_size|number_of_TBEs)|system.ruby.l0_cntrl\d+.replyToL1|system.ruby.l0_cntrl\d+.sequencer.(cacheflex_finite_ingress|cacheflex_max_line_targets|max_outstanding_requests)|system.ruby\..*Credits.capacity'
    bad={k:v for k,v in delta.items() if not re.fullmatch(always,k) and not (c['finite'] and re.fullmatch(finite,k))}
    assert not bad,bad
    ruby=b['system']['ruby'];small=c['small'];finite_on=c['finite']
    assert ruby['network'].get('cacheflex_request_vc_partition',False)==finite_on
    for i in range(8):
        cpu=b['system']['cpu'][i];l0=ruby[f'l0_cntrl{i}'];l1=ruby[f'l1_cntrl{i}'];seq=l0['sequencer']
        assert cpu['cacheFlexBoundedTracking']==finite_on
        assert cpu['LQEntries']==(8 if small else 32) and cpu['SQEntries']==(8 if small else 48)
        assert seq['max_outstanding_requests']==c['guard']
        assert seq['cacheflex_finite_ingress']==finite_on
        assert l1['cache']['num_spm_ways']==(6 if c['spm'] else 0)
        if finite_on:
            assert l0['bufferToL1']['buffer_size']==(2 if small else 12)
            assert l1['returnBufferFromL0']['buffer_size']==(2 if small else 4)
            assert l0['replyToL1']==l1['returnBufferFromL0']['path']
            assert l1['spmBufferFromL0']['buffer_size']==(4 if small else 16)
    if finite_on:
        for k,v in new.items():
            if k.startswith('system.ruby.') and k.endswith('.buffer_size') and new.get(k.rsplit('.',1)[0]+'.type')=='MessageBuffer':assert v>0,(k,v)
    assert b['system']['cpu'][0]['workload'][0]['cmd'][1:]==args
    return delta

def case(name,**kw):
    return dict(name=name,**(dict(t=1024,br=32,bc=128,r=192,kv=0,spm=0,finite=True,small=False,guard=1024)|kw))

def equivalent(a,b,extra_return=False):
    sa,sb=stats(a),stats(b);assert len(sa)==len(sb),(len(sa),len(sb))
    for epoch,(x,y) in enumerate(zip(sa,sb)):
        if extra_return:
            x={k:v for k,v in x.items() if '.returnBufferFromL0.' not in k}
            y={k:v for k,v in y.items() if '.returnBufferFromL0.' not in k}
            # New default-disabled instrumentation, absent from the old build.
            for k in list(y.keys()-x.keys()):
                if re.fullmatch(r'system.ruby.l0_cntrl\d+.sequencer.m_earlyRead(Eligible|Issued|Verified|FallbackAlias|FallbackFunctional)',k):
                    assert y[k]==0,(k,y[k])
                    del y[k]
        diff={k:[x.get(k),y.get(k)] for k in x.keys()|y.keys() if x.get(k)!=y.get(k)}
        assert not diff,(epoch,dict(list(diff.items())[:30]))
    assert sha(a/'output.fp16')==sha(b/'output.fp16')

def main():
    OUT.mkdir();(OUT/'runs').mkdir();(OUT/'references').mkdir()
    shutil.copytree(BUILD/'configs',OUT/'configs')
    shutil.copy2(OLD/'configs/topologies/KVFixedMesh.py',OUT/'configs/topologies/KVFixedMesh.py')
    inputs=[Path(__file__),GUEST,BUILD/'gem5.opt',CONTROL/'command.json',CONTROL/'config.json',
            FA/'kv_partition/reference.py',FA/'fa_independent_reference.py',FA/'export_safe_elf.py',
            *list((OUT/'configs').rglob('*.py'))]
    frozen={str(p):sha(p) for p in inputs};save(OUT/'inputs.json',frozen)
    reuse_runs=os.environ.get('CACHEFLEX_FINITE_REUSE_RUNS')
    previous_runs=BASE/'finite_model_20260926'/reuse_runs if reuse_runs else None
    if previous_runs:
        prior_manifest=json.loads((previous_runs/'inputs.json').read_text())
        for p,h in prior_manifest.items():
            if p!=str(Path(__file__)):assert sha(p)==h,('reused simulation dependency changed',p)
    state=dict(pid=os.getpid(),stage='initializing',active={},completed=[])
    lock=threading.RLock();halt=threading.Event();refs={}
    def status(stage=None):
        with lock:
            if stage:state['stage']=stage
            state['updated']=time.time();save(OUT/'status.json',state)
    def unchanged():
        for p,h in frozen.items():assert sha(p)==h,('changed dependency',p)
    def getref(c):
        key=(c['t'],c['bc'],8 if c['kv'] else 1)
        if key not in refs:
            p=OUT/'references'/('t%d_bc%d_parts%d.npy'%key)
            import numpy as np
            old=OLD/'references'/p.name
            if old.exists():
                meta=json.loads(old.with_suffix('.json').read_text());assert sha(old)==meta['sha256']
                shutil.copy2(old,p);a=np.load(p)
            else:a=reference(*key);np.save(p,a)
            save(p.with_suffix('.json'),dict(key=key,sha256=sha(p)))
            refs[key]=a
    def execute(c,guest=None,marker=None):
        assert not halt.is_set()
        p=OUT/'runs'/c['name'];p.mkdir();record=p;proc=None;start=time.monotonic()
        try:
            unchanged();cmd,args=command(c,p)
            if guest:
                cmd[cmd.index('--cmd')+1]=str(guest)
                cmd[cmd.index('--options')+1]=''
                cmd=["system.cpu[0].workload[0].cmd=['spm-smoke']" if x.startswith('system.cpu[0].workload[0].cmd=') else x.replace('sve_vl_se=16','sve_vl_se=4') for x in cmd]
                cmd=[x.replace('--l1d_num_spm_ways=6','--l1d_num_spm_ways=4') for x in cmd]
                save(p/'guest.json',dict(path=str(guest),sha256=sha(guest)))
            save(p/'command.json',cmd)
            prior=previous_runs/'runs'/c['name'] if previous_runs else None
            oldlog=(prior/'run.log').read_text() if prior and (prior/'run.log').exists() else ''
            reusable=(marker or 'KV_ROI_END') in oldlog and 'because exiting with last active thread context' in oldlog and 'Simulated exit code not 0!' not in oldlog
            if reusable:
                oldcmd=json.loads((prior/'command.json').read_text())
                assert oldcmd[:1]+oldcmd[3:]==cmd[:1]+cmd[3:],'reused command changed'
                p=prior
                save(record/'reused.json',dict(source=str(p),files={str(f):sha(f) for f in p.iterdir() if f.is_file()}))
            else:
                with (p/'run.log').open('w') as log:
                    proc=subprocess.Popen(cmd,cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT)
                    with lock:state['active'][c['name']]=dict(pid=proc.pid,case=c);status()
                    while proc.poll() is None:
                        assert not halt.is_set(),'peer failed'
                        assert time.monotonic()-start<(1800 if guest else 172800),'timeout'
                        assert shutil.disk_usage(OUT).free>50*2**30,'disk guard'
                        try:proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:pass
            log=(p/'run.log').read_text()
            assert (reusable or proc.returncode==0) and (marker or 'KV_ROI_END') in log and 'Simulated exit code not 0!' not in log,log[-3000:]
            ss=stats(p);assert ss
            for s in ss:
                check_global_guard(s)
                if c['finite']:
                    for i in range(8):
                        for domain in ('regular','spm'):
                            for kind,cap in [('Load',8 if c['small'] else 32),('Store',8 if c['small'] else 48)]:
                                k=f'system.cpu{i}.{domain}Lsq0.boundedMax{kind}Occupancy'
                                assert s[k]<=cap,(k,s[k],cap)
            row=dict(case=c,host_seconds=time.monotonic()-start)
            if not guest:
                row['config_delta']=audit(p,c,args);export_hash(p/'output.fp16',log)
                row['arithmetic']=validate(p/'output.fp16',refs[c['t'],c['bc'],8 if c['kv'] else 1],c['t'],dense=c['t']==1024)
                s=ss[0];expected=(2*c['t'] if c['kv'] else 16*c['t']*math.ceil((c['t']//8)/c['r'])) if c['spm'] else 0
                for k in ('m_spmcpRequestsIssued','m_spmcpRequestsCompleted'):
                    assert sum(s[f'system.ruby.l0_cntrl{i}.sequencer.{k}'] for i in range(8))==expected
                if c['spm'] and c['finite']:
                    assert sum(s[f'system.cpu{i}.spmLsq0.boundedShadowAllocations'] for i in range(8))>0
                    assert max(s[f'system.ruby.l0_cntrl{i}.sequencer.m_spmcpMaxOutstanding'] for i in range(8))<= (8 if c['small'] else 48)
                row.update(roi_us=s['simTicks']/1e6,copies=expected,output_sha256=sha(p/'output.fp16'))
            row['data_path']=str(p)
            unchanged();save(record/'validated.json',row)
            with lock:state['completed'].append(row)
            return p
        except BaseException as e:
            halt.set();save(record/'failed.json',dict(error=repr(e)))
            with lock:state.setdefault('error',c['name']+': '+repr(e))
            raise
        finally:
            if proc and proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
            with lock:state['active'].pop(c['name'],None);status()
    def batch(cs,stage):
        status(stage)
        for c in cs:getref(c)
        with cf.ThreadPoolExecutor(max_workers=2) as pool:return list(pool.map(execute,cs))
    try:
        status()
        reuse=os.environ.get('CACHEFLEX_FINITE_REUSE_DEFAULTS')
        if reuse:
            previous=BASE/'finite_model_20260926'/reuse
            manifest=json.loads((previous/'inputs.json').read_text())
            for p,h in manifest.items():
                if p!=str(Path(__file__)):assert sha(p)==h,('reused dependency changed',p)
            defaults=[previous/'runs'/('default_'+a) for a in ('cache','spm')]
            for p in defaults:
                row=json.loads((p/'validated.json').read_text())
                assert sha(p/'output.fp16')==row['output_sha256']
            save(OUT/'reused_defaults.json',dict(source=str(previous),files={str(p/f):sha(p/f) for p in defaults for f in ('stats.txt','config.json','output.fp16','validated.json','command.json')}))
        else:
            defaults=batch([case('default_'+a,spm=s,finite=False) for s,a in enumerate(('cache','spm'))],'default_equivalence')
        for a,p in zip(('cache','spm'),defaults):equivalent(OLD/f'runs/t1024_q_{a}_br32_bc128_r192',p,True)
        save(OUT/'default_equivalence.json',dict(all_finite_nonhost_counters_equal=True))
        status('bridge_regressions')
        for name,guest,marker in [
            ('roundtrip',BASE/'micro_resources_20260921/roundtrip_guest/guest','SPM_ROUNDTRIP PASS errors=0'),
            ('roundtrip_raw',BASE/'micro_resources_20260921/roundtrip_raw_guest/guest','SPM_ROUNDTRIP PASS errors=0'),
            ('release_getx',ROOT/'results/ruby_getx_race_guest_20260916/guest','SPM_RELEASE_GETX PASS errors=0')]:
            execute(case(name,spm=1),guest,marker)
        # Exercise the previously failing dependency cycle before expensive
        # nominal/guard repeats. This is not a performance-selection gate.
        batch([case('stress_'+a,spm=s,small=True,br=64,bc=1024,r=256) for s,a in enumerate(('cache','spm'))],'small_resource_stress')
        gates=[]
        for kv in (0,1):
            for spm in (0,1):
                gates.append(case(f'finite_kv{kv}_spm{spm}',kv=kv,spm=spm))
                gates.append(case(f'large_kv{kv}_spm{spm}',kv=kv,spm=spm,br=64,bc=128 if kv else 1024,r=256))
        nominal=batch(gates,'eight_numerical_gates')
        repeats=batch([dict(c,name=c['name']+'_guard4096',guard=4096) for c in gates],'guard_equivalence')
        for a,b in zip(nominal,repeats):equivalent(a,b)
        save(OUT/'guard_equivalence.json',dict(cases=8,all_finite_nonhost_counters_equal=True,guards=[1024,4096]))
        save(OUT/'gates_passed.json',dict(full_arithmetic=True,default_equivalence=True,guard_equivalence=True,stress=True))
        targets=[case(f't4096_kv{k}_spm{s}',t=4096,kv=k,spm=s,br=16,bc=128 if k else 512,r=128 if k else 256) for k in (0,1) for s in (0,1)]
        batch(targets,'matched_t4096_fixed_candidates')
        # These are new-model pilots, not minima from the old unlimited profile.
        batch([dict(c,name=c['name'].replace('4096','16384'),t=16384) for c in targets],'matched_t16384_fixed_candidates')
        status('completed_requires_tuning_review')
    except BaseException as e:
        state.setdefault('error',repr(e));status('failed');raise

if __name__=='__main__':main()
