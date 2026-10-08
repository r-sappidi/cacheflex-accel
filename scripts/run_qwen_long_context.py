"""L3-10: resource-gated 16K/32K Qwen FA transfer with full acceptance checks."""
import importlib,json,math,os,re,shutil,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path('/data2/rsappidi/cacheflex_mc')
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'benchmarks/cacheflex_flash_attention')]
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["OMP_NUM_THREADS"]="1"
import numpy as np
from build_access_timing import sha,save
from run_access_timing import stats
from fa_headroom_profiles import leaves
from test_micro_timing import check_global_guard
from export_safe_elf import export_hash
from qwen_l3_reference import validate
from run_qwen_l3_attention import work
from run_l3_kernel_tuning import resources,ENV
BASE=ROOT/'results/cacheflex_l3_spm/qwen_long_context_20260928'
PARENT=ROOT/'results/cacheflex_l3_spm/direct_link_20260928/experiment_v2'
BUILD=PARENT.parent/'build_v4'
GUEST=BASE/'guest_v2'
OUT=BASE/'experiment_v1'
ARMS=('cache','private','noc','direct')
PLAN=[dict(name=f'exact_{a}',t=1024,source=str(PARENT/'runs'/f't1024_{a}'),direct=a=='direct',exact=True) for a in ARMS]+[dict(name=f't{t}_{a}',t=t,source=str(PARENT/'runs'/f't4096_{a}'),direct=a=='direct',exact=False) for t in (16384,32768) for a in ARMS]
QWEN=ROOT/'results/cacheflex_l3_spm/qwen_gqa_20260928/experiment_v2'
TUNED=ROOT/'results/cacheflex_l3_spm/qwen_analytical_20260928/experiment_v1'
CACHE=ROOT/'results/cacheflex_l3_spm/qwen_cache_analytical_20260928/experiment_v1'

def validate_work_counts(c, ep):
    # Both dumps are cumulative since ROI reset; export does not reset them.
    total=lambda suffix:sum(v for k,v in ep.items() if k.endswith(suffix))
    w=work(c)
    commands=sum(w['updates']) if c['arm']=='l3' else 0
    copies=sum(w['copies']) if c['arm']!='cache' else 0
    for suffix in ('.reductionAccepted','.reductionCompleted','.l3ReductionStarts',
                   '.l3ReductionFinishes','.l3ReductionHalfCommands'):
        assert total(suffix)==commands,(suffix,total(suffix),commands)
    assert total('.l3ReductionInitializations')==(c['hq']*c['t'] if commands else 0)
    assert total('.l3Finalizations')==0
    assert total('.reductionTransportBytes')==144*commands
    assert total('.m_spmcpRequestsIssued')==total('.m_spmcpRequestsCompleted')==copies

def make_command(original,path,t,direct):
    cmd=[x for x in original if x!='--cacheflex-direct-reduction' and not x.startswith(('--cacheflex-direct-bytes=','--cacheflex-direct-latency='))]
    cmd[0]=str(BUILD/'gem5.opt');cmd[1]='--outdir='+str(path)
    cmd[2]=str(BUILD/'configs/deprecated/example/l3_direct_reduction_se.py')
    cmd[cmd.index('--cmd')+1]=str(GUEST/'guest.export-safe')
    i=cmd.index('--options')+1;args=cmd[i].split();args[0]=str(t);cmd[i]=' '.join(args)
    prefix='system.cpu[0].workload[0].cmd='
    indices=[i for i,x in enumerate(cmd) if x.startswith(prefix)];assert len(indices)==1
    cmd[indices[0]]=prefix+repr(['kv-fa',*args])
    if direct:cmd.append('--cacheflex-direct-reduction')
    cmd+=['--cacheflex-direct-bytes=16','--cacheflex-direct-latency=2']
    return cmd

def main():
    OUT.mkdir(parents=True);(OUT/'runs').mkdir()
    state=dict(pid=os.getpid(),stage='waiting_parent',active={},completed=[],pending=PLAN.copy(),parent=str(PARENT))
    save(OUT/'plan.json',PLAN)
    frozen={};proc=None
    def status(stage=None):
        if stage:state['stage']=stage
        state['updated']=time.time();save(OUT/'status.json',state)
        lines=['# Qwen 16K/32K FA transfer: accepted results','',f"Stage: {state['stage']}",'',
            '| Case | ROI us | Commands | Direct bytes (both ways) |','|---|---:|---:|---:|']
        for r in state['completed']:
            lines.append(f"| {r['name']} | {r['roi_us']:.4f} | {r['commands']} | {r['direct_bytes']} |")
        effects=[]
        for t in (16384,32768):
            rows={r['name']:r for r in state['completed']}
            for mode in ('direct','noc','private'):
                d=rows.get(f't{t}_{mode}')
                if not d:continue
                for baseline in ('noc','private','cache'):
                    if mode==baseline or (mode=='private' and baseline!='cache'):continue
                    b=rows.get(f't{t}_{baseline}')
                    if b:
                        e=dict(t=t,mode=mode,baseline=baseline,baseline_us=b['roi_us'],direct_us=d['roi_us'],
                            speedup=b['roi_us']/d['roi_us'],latency_reduction_pct=100*(1-d['roi_us']/b['roi_us']))
                        effects.append(e);lines.append(f"\nT{t} {mode} versus {baseline}: {e['latency_reduction_pct']:.3f}% lower latency, {e['speedup']:.6f}x.")
        lines+=['','Accepted only after math, exact output bits, configuration, command/copy/resource checks.',
            'Direct wire/endpoint timing is a finite modeling assumption, not calibrated physical implementation.',
            'NoC/direct use identical kernels; each backend inherits its best accepted T4096 tiles, not long-context tuning.',
            'This is FP16 Qwen-shaped causal GQA, not end-to-end model inference.']
        (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n');save(OUT/'effects.json',effects)
    def freeze(p):
        p=Path(p);h=sha(p)
        if str(p) in frozen:assert frozen[str(p)]==h,('changed input',p)
        frozen[str(p)]=h
    def unchanged():
        for p,h in frozen.items():assert sha(p)==h,('changed input',p)
    status()
    try:
        # Freeze inputs before waiting, so queued code cannot silently change.
        frozen.update(json.loads((GUEST/'frozen.json').read_text()))
        freeze(ROOT/'scripts/qwen_long_reference.py')
        assert json.loads((GUEST/'status.json').read_text())['stage']=='built_requires_validation'
        frozen.update(json.loads((BUILD/'frozen.json').read_text()))
        for mod in ('run_access_timing','fa_headroom_profiles','test_micro_timing','export_safe_elf',
                    'qwen_l3_reference','fa_independent_reference','run_qwen_l3_attention',
                    'run_l3_kernel_tuning','build_access_timing'):
            freeze(importlib.import_module(mod).__file__)
        freeze(__file__);freeze(ROOT/'docs/experiments/QWEN_LONG_CONTEXT.md')
        save(OUT/'inputs.json',frozen);unchanged()
        while not (PARENT/'complete.json').exists():
            parent=json.loads((PARENT/'status.json').read_text())
            assert parent['stage']!='failed',parent
            os.kill(parent['pid'],0)
            state['parent_stage']=parent['stage'];status('waiting_parent');time.sleep(10)
        assert json.loads((PARENT/'complete.json').read_text())['stage']=='completed_validated'
        freeze(PARENT/'complete.json')
        unchanged()

        def headroom(path):
            while True:
                r=resources();save(path/'resource_launch.json',r)
                if r['cpu_idle_pct']>=50 and r['available_gib']>=128 and r['free_disk_gib']>=100 and len(r['simulators'])<6:return
                state['resource_wait']=r;status();time.sleep(10)

        def watch(start):
            assert time.monotonic()-start<96*3600,'96h timeout'
            assert shutil.disk_usage(OUT).free>75*2**30,'disk guard'
            memory=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))
            assert memory>64*2**20,'memory guard'
            try:proc.wait(timeout=5)
            except subprocess.TimeoutExpired:status()

        def get_reference(c):
            nonlocal proc
            name=f"t{c['t']}_bc{c['bc']}_q{c['hq']}_kv{c['hkv']}_causal{int(c['causal'])}.npy"
            ref=next((p/'references'/name for p in (OUT,QWEN,TUNED,CACHE) if (p/'references'/name).exists()),None)
            if ref is None:
                ref=OUT/'references'/name;ref.parent.mkdir(exist_ok=True)
                status('reference_'+ref.stem);headroom(ref.parent);state.pop('resource_wait',None)
                cmd=[sys.executable,str(ROOT/'scripts/qwen_long_reference.py'),'--t',str(c['t']),'--bc',str(c['bc']),'--out',str(ref)]
                save(ref.with_suffix('.command.json'),cmd)
                with ref.with_suffix('.log').open('w') as log:
                    proc=subprocess.Popen(['nice','-n','10',*cmd],cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT)
                    state['active']={'reference':dict(pid=proc.pid,path=str(ref),start=time.time())};status()
                    start=time.monotonic()
                    while proc.poll() is None:watch(start)
                assert proc.returncode==0,ref.with_suffix('.log').read_text()[-4000:]
                proc=None;state['active']={}
            assert sha(ref)==json.loads(ref.with_suffix('.json').read_text())['sha256']
            freeze(ref);freeze(ref.with_suffix('.json'))
            return np.load(ref)

        def run(name,source,t,direct=False,exact=False):
            width,latency=16,2
            nonlocal proc
            source=Path(source);prior=json.loads((source/'accepted.json').read_text())
            path=OUT/'runs'/name;path.mkdir();start=time.monotonic()
            for n in ('accepted.json','command.json','config.json','stats.txt','run.log'):freeze(source/n)
            cmd=json.loads((source/'command.json').read_text());freeze(cmd[cmd.index('--cmd')+1])
            c=dict(prior['case']);c['t']=t
            assert c['hq']==14 and c['hkv']==2 and c['causal'] and c['depth']==4
            if not exact:
                assert c['br']==16 and c['r']==256
                assert c['bc']==(512 if c['arm']=='cache' else 256)
            expected=get_reference(c)
            freeze(source/'output.fp16')
            cmd=make_command(cmd,path,t,direct)
            job=dict(name=name,source=str(source),direct=direct,width=width,latency=latency,exact=exact,t=t)
            save(path/'job.json',job);save(path/'command.json',cmd);save(OUT/'inputs.json',frozen);unchanged()
            status('running_'+name)
            with (path/'run.log').open('w') as f:
                headroom(path);state.pop('resource_wait',None)
                start=time.monotonic()
                proc=subprocess.Popen(['nice','-n','10',*cmd],cwd=ROOT,env=ENV,stdout=f,stderr=subprocess.STDOUT)
                state['active']={name:dict(pid=proc.pid,start=time.time())};status()
                while proc.poll() is None:
                    watch(start)
            log=(path/'run.log').read_text()
            assert proc.returncode==0 and 'because exiting with last active thread context' in log,log[-5000:]
            assert not re.search(r'panic:|fatal:|Simulated exit code not 0|\bFAIL\b',log),log[-5000:]
            if c:assert 'KV_ROI_END' in log
            else:assert re.search(r'L3_(?:LIFECYCLE|REDUCTION)_SMOKE PASS errors=0 commands=\d+',log)
            def fields(p):return dict(leaves(json.loads((p/'config.json').read_text().replace(str(p),'<OUT>'))))
            a,b=fields(source),fields(path)
            delta={k:[a.get(k),b.get(k)] for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}
            permitted={f'system.ruby.network.cacheflex_direct_{k}':v for k,v in [('reduction',direct),('bytes',width),('latency',latency)]}
            permitted['system.cpu[0].workload[0].executable']=str(GUEST/'guest.export-safe')
            permitted['system.cpu[0].workload[0].cmd[1]']=str(t)
            assert not {k:v for k,v in delta.items() if k not in permitted or v[1]!=permitted[k]},delta
            for k,v in permitted.items():assert b[k]==v,(k,b[k],v)
            epochs=stats(path);old=stats(source);s=epochs[0]
            total=lambda ep,suffix:sum(v for k,v in ep.items() if k.endswith(suffix))
            same_counts=('.reductionAccepted','.reductionCompleted','.l3ReductionStarts','.l3ReductionFinishes',
                '.l3ReductionHalfCommands','.l3ReductionInitializations','.l3Finalizations',
                '.reductionTransportBytes','.m_spmcpRequestsIssued','.m_spmcpRequestsCompleted')
            assert len(epochs)==len(old)
            expected_work=work(c)
            for epoch,(ep,control) in enumerate(zip(epochs,old)):
                check_global_guard(ep)
                if exact:
                    for suffix in same_counts:assert total(ep,suffix)==total(control,suffix),(suffix,total(ep,suffix),total(control,suffix))
                validate_work_counts(c,ep)
                commands=total(ep,'.reductionAccepted');half=total(ep,'.l3ReductionHalfCommands')
                assert commands==total(ep,'.reductionCompleted')
                assert total(ep,'.m_spmcpRequestsIssued')==total(ep,'.m_spmcpRequestsCompleted')
                prefix='system.ruby.network.'
                def ds(k):return ep[prefix+k]
                n=commands if direct else 0
                assert ds('directCommands')==ds('directAcks')==n
                assert ds('directDelivered')==2*n
                wire=half*144+(commands-half)*272
                assert ds('directCommandBytes')==(wire if direct else 0)
                assert ds('directAckBytes')==16*n
                beats=(half*math.ceil(144/width)+(commands-half)*math.ceil(272/width)+commands*math.ceil(16/width)) if direct else 0
                assert ds('directBeats')==beats,(ds('directBeats'),beats)
                assert 0<=ds('directMaxPending')<=1
                if n:assert ds('directMaxPending')==1
                for i in range(8):
                    p=f'system.ruby.l2_cntrl{i}.';depth=b[p+'l3_reduction_entries']
                    assert b[p+'L2cache.size']==524288 and not b[p+'L2cache.spm_separate_backing']
                    assert b[p+'l3_shared_service_port'] and b[p+'l3_spm_latency']==35
                    for q in ('Source','Sink'):assert ep[f'system.ruby.l0_cntrl{i}.sequencer.reductionMax{q}']<=depth
                    assert ep[p+'L2cache.l3ReductionMaxActive']<=(depth if b[p+'l3_reduction_pipeline'] else 1)
                    for domain in ('regular','spm'):
                        for kind,cap in [('Load',32),('Store',48)]:assert ep[f'system.cpu{i}.{domain}Lsq0.boundedMax{kind}Occupancy']<=cap
                if exact:
                    mismatches=[(k,v,ep.get(k)) for k,v in control.items() if ep.get(k)!=v]
                    assert not mismatches and set(ep)==set(control),(mismatches[:15],set(ep)^set(control))
            row=dict(name=name,job=job,case=c,path=str(path),roi_us=s['simTicks']/1e6,prior_us=prior['roi_us'],
                commands=total(s,'.reductionAccepted'),direct_bytes=total(s,'.directCommandBytes')+total(s,'.directAckBytes'),
                config_delta=delta,exact_prior_stats=exact,host_seconds=time.monotonic()-start,
                direct_stats={k.rsplit('.',1)[-1]:v for k,v in s.items() if k.startswith('system.ruby.network.direct')},
                network_flits={k:v for k,v in s.items() if 'flits_injected' in k})
            if c:
                export_hash(path/'output.fp16',log)
                row['arithmetic']=validate(path/'output.fp16',expected,c['t'],c['hq'],c['hkv'],c['causal'],dense=c['t']<=1024)
                if exact:assert sha(path/'output.fp16')==sha(source/'output.fp16'),'old/new guest output bits changed'
                for peer in state['completed']:
                    pc=peer['case']
                    if pc['t']==t and pc['bc']==c['bc']:
                        assert sha(path/'output.fp16')==peer['output_sha256'],('same-Bc output bits changed',peer['name'])
                expected_work=work(c)
                actual={int(i):(int(n),int(u)) for i,n,u in re.findall(r'GQA_WORK core=(\d+) tasks=(\d+) row_updates=(\d+)',log)}
                assert actual=={i:(expected_work['tasks'][i],expected_work['updates'][i]) for i in range(8)},actual
                assert row['commands']==(sum(expected_work['updates']) if c['arm']=='l3' else 0)
                assert total(s,'.m_spmcpRequestsIssued')==(sum(expected_work['copies']) if c['arm']!='cache' else 0)
                row['output_sha256']=sha(path/'output.fp16')
            unchanged();save(path/'accepted.json',row);state['completed'].append(row);state['pending']=[j for j in state['pending'] if j['name']!=name];state['active']={};proc=None;status()
            return row
        for job in PLAN:
            run(**job)
            if job['name']=='exact_direct':
                save(OUT/'gates_passed.json',dict(cases=4,full_math=True,exact_stats=True,exact_bits=True))
        status('completed_validated');save(OUT/'complete.json',state)
        effects=json.loads((OUT/'effects.json').read_text())
        lines=['\n\n**L3-10 completion update:** four exact old/new guest controls and eight full long-context FA cases accepted.']
        for e in effects:
            if e['mode']!=e['baseline']:
                lines.append(f"T{e['t']} {e['mode']} {e['direct_us']:.4f}us vs {e['baseline']} {e['baseline_us']:.4f}us: {e['latency_reduction_pct']:.3f}% lower latency ({e['speedup']:.6f}x).")
        lines.append('Transferred T4096 tiles, not independent long-context tuning. Full oracle/exact sameBc outputs/counts/config/bounds. All accepted effects including null/regressive cases: qwen_long_context_20260928/experiment_v1/REPORT.md. Finite timing assumptions; no full-model claim.')
        with (ROOT/'docs/experiments/LEDGER.md').open('a') as f:f.write('\n'.join(lines)+'\n')
    except BaseException as e:
        state['error']=repr(e);state['traceback']=traceback.format_exc()
        if proc and proc.poll() is None:
            proc.terminate()
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
        state['active']={};status('failed');save(OUT/'failed.json',state);raise
if __name__=='__main__':main()
