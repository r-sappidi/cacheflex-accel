"""Finite direct-link validation and best-measured Qwen FA comparison, fail closed."""
import importlib,json,math,os,re,shutil,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path('/data2/rsappidi/cacheflex_mc')
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'benchmarks/cacheflex_flash_attention')]
import numpy as np
from build_access_timing import sha,save
from run_access_timing import stats
from fa_headroom_profiles import leaves
from test_micro_timing import check_global_guard
from export_safe_elf import export_hash
from qwen_l3_reference import validate
from run_qwen_l3_attention import work
from run_l3_kernel_tuning import resources,ENV
BASE=ROOT/'results/cacheflex_l3_spm/direct_link_20260928'
BUILD=BASE/os.environ.get('CACHEFLEX_DIRECT_BUILD','build_v2')
OUT=BASE/os.environ.get('CACHEFLEX_DIRECT_EXPERIMENT','experiment_v1')
QWEN=ROOT/'results/cacheflex_l3_spm/qwen_gqa_20260928/experiment_v2'
TUNED=ROOT/'results/cacheflex_l3_spm/qwen_analytical_20260928/experiment_v1'
CACHE=ROOT/'results/cacheflex_l3_spm/qwen_cache_analytical_20260928/experiment_v1'
LIFE=ROOT/'results/cacheflex_l3_spm/lifecycle_20260928/experiment_v4'
PIPE=ROOT/'results/cacheflex_l3_spm/pipeline_20260928/experiment_v2'

def main():
    OUT.mkdir(parents=True);(OUT/'runs').mkdir()
    state=dict(pid=os.getpid(),stage='waiting_build',active={},completed=[])
    frozen={};proc=None
    def status(stage=None):
        if stage:state['stage']=stage
        state['updated']=time.time();save(OUT/'status.json',state)
        lines=['# Direct reduction transport: accepted results','',f"Stage: {state['stage']}",'',
            '| Case | ROI us | Commands | Direct bytes (both ways) |','|---|---:|---:|---:|']
        for r in state['completed']:
            lines.append(f"| {r['name']} | {r['roi_us']:.4f} | {r['commands']} | {r['direct_bytes']} |")
        effects=[]
        for t in (1024,2048,4096):
            rows={r['name']:r for r in state['completed']}
            for mode in ('direct','slow'):
                d=rows.get(f't{t}_{mode}')
                if not d:continue
                for baseline in ('noc','private','cache'):
                    b=rows.get(f't{t}_{baseline}')
                    if b:
                        e=dict(t=t,mode=mode,baseline=baseline,baseline_us=b['roi_us'],direct_us=d['roi_us'],
                            speedup=b['roi_us']/d['roi_us'],latency_reduction_pct=100*(1-d['roi_us']/b['roi_us']))
                        effects.append(e);lines.append(f"\nT{t} {mode} versus {baseline}: {e['latency_reduction_pct']:.3f}% lower latency, {e['speedup']:.6f}x.")
        lines+=['','Accepted only after math, exact output bits, configuration, command/copy/resource checks.',
            'Direct wire/endpoint timing is a finite modeling assumption, not calibrated physical implementation.',
            'NoC control uses identical kernel; cache/private use their best accepted bounded-search configuration.',
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
        while True:
            if (BUILD/'status.json').exists():
                b=json.loads((BUILD/'status.json').read_text());assert b['stage']!='failed',b
                if b['stage']=='built_requires_validation':break
            time.sleep(10);status()
        frozen.update(json.loads((BUILD/'frozen.json').read_text()))
        for mod in ('run_access_timing','fa_headroom_profiles','test_micro_timing','export_safe_elf',
                    'qwen_l3_reference','fa_independent_reference','run_qwen_l3_attention',
                    'run_l3_kernel_tuning','build_access_timing'):
            freeze(importlib.import_module(mod).__file__)
        freeze(__file__);freeze(ROOT/'docs/experiments/L3_DIRECT_LINK.md')
        save(OUT/'inputs.json',frozen);unchanged()
        def run(name,source,direct=False,width=16,latency=2,exact=False):
            nonlocal proc
            source=Path(source);prior=json.loads((source/'accepted.json').read_text())
            path=OUT/'runs'/name;path.mkdir();start=time.monotonic()
            for n in ('accepted.json','command.json','config.json','stats.txt','run.log'):freeze(source/n)
            cmd=json.loads((source/'command.json').read_text());freeze(cmd[cmd.index('--cmd')+1])
            c=prior.get('case');expected=None
            if c:
                refname=f"t{c['t']}_bc{c['bc']}_q{c['hq']}_kv{c['hkv']}_causal{int(c['causal'])}.npy"
                ref=next((p/'references'/refname for p in (QWEN,TUNED,CACHE) if (p/'references'/refname).exists()),None)
                assert ref is not None,refname
                assert sha(ref)==json.loads(ref.with_suffix('.json').read_text())['sha256']
                freeze(ref);freeze(ref.with_suffix('.json'));expected=np.load(ref);freeze(source/'output.fp16')
            cmd[0]=str(BUILD/'gem5.opt');cmd[1]='--outdir='+str(path)
            cmd[2]=str(BUILD/'configs/deprecated/example/l3_direct_reduction_se.py')
            if direct:cmd+=['--cacheflex-direct-reduction']
            cmd += [f'--cacheflex-direct-bytes={width}',f'--cacheflex-direct-latency={latency}']
            job=dict(name=name,source=str(source),direct=direct,width=width,latency=latency,exact=exact)
            save(path/'job.json',job);save(path/'command.json',cmd);save(OUT/'inputs.json',frozen);unchanged()
            status('running_'+name)
            with (path/'run.log').open('w') as f:
                while True:
                    r=resources();save(path/'resource_launch.json',r)
                    if r['cpu_idle_pct']>=50 and r['available_gib']>=128 and r['free_disk_gib']>=100 and len(r['simulators'])<6:break
                    state['resource_wait']=r;status();time.sleep(10)
                state.pop('resource_wait',None)
                proc=subprocess.Popen(['nice','-n','10',*cmd],cwd=ROOT,env=ENV,stdout=f,stderr=subprocess.STDOUT)
                state['active']={name:dict(pid=proc.pid,start=time.time())};status()
                while proc.poll() is None:
                    assert time.monotonic()-start<12*3600,'12h timeout'
                    assert shutil.disk_usage(OUT).free>75*2**30,'disk guard'
                    memory=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))
                    assert memory>64*2**20,'memory guard'
                    try:proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:status()
            log=(path/'run.log').read_text()
            assert proc.returncode==0 and 'because exiting with last active thread context' in log,log[-5000:]
            assert not re.search(r'panic:|fatal:|Simulated exit code not 0|\bFAIL\b',log),log[-5000:]
            if c:assert 'KV_ROI_END' in log
            else:assert re.search(r'L3_(?:LIFECYCLE|REDUCTION)_SMOKE PASS errors=0 commands=\d+',log)
            def fields(p):return dict(leaves(json.loads((p/'config.json').read_text().replace(str(p),'<OUT>'))))
            a,b=fields(source),fields(path)
            delta={k:[a.get(k),b.get(k)] for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}
            permitted={f'system.ruby.network.cacheflex_direct_{k}':v for k,v in [('reduction',direct),('bytes',width),('latency',latency)]}
            # Older pipeline directed tests predate lifecycle parameters. Defaults
            # are explicitly checked; no existing field may change.
            for i in range(8):
                for k,v in [('l3_reduction_initialize',False),('l3_reduction_finalize',False),('l3_finalize_arithmetic',8)]:
                    p=f'system.ruby.l2_cntrl{i}.{k}'
                    if p not in a:permitted[p]=v
            assert not {k:v for k,v in delta.items() if k not in permitted or v[1]!=permitted[k]},delta
            for k,v in permitted.items():assert b[k]==v,(k,b[k],v)
            epochs=stats(path);old=stats(source);s=epochs[0]
            total=lambda ep,suffix:sum(v for k,v in ep.items() if k.endswith(suffix))
            same_counts=('.reductionAccepted','.reductionCompleted','.l3ReductionStarts','.l3ReductionFinishes',
                '.l3ReductionHalfCommands','.l3ReductionInitializations','.l3Finalizations',
                '.reductionTransportBytes','.m_spmcpRequestsIssued','.m_spmcpRequestsCompleted')
            assert len(epochs)==len(old)
            for ep,control in zip(epochs,old):
                check_global_guard(ep)
                for suffix in same_counts:assert total(ep,suffix)==total(control,suffix),(suffix,total(ep,suffix),total(control,suffix))
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
                    assert not mismatches,mismatches[:15]
            row=dict(name=name,job=job,case=c,path=str(path),roi_us=s['simTicks']/1e6,prior_us=prior['roi_us'],
                commands=total(s,'.reductionAccepted'),direct_bytes=total(s,'.directCommandBytes')+total(s,'.directAckBytes'),
                config_delta=delta,exact_prior_stats=exact,host_seconds=time.monotonic()-start,
                direct_stats={k.rsplit('.',1)[-1]:v for k,v in s.items() if k.startswith('system.ruby.network.direct')},
                network_flits={k:v for k,v in s.items() if 'flits_injected' in k})
            if c:
                export_hash(path/'output.fp16',log)
                row['arithmetic']=validate(path/'output.fp16',expected,c['t'],c['hq'],c['hkv'],c['causal'],dense=c['t']<=1024)
                assert sha(path/'output.fp16')==sha(source/'output.fp16'),'output bits changed'
                expected_work=work(c)
                actual={int(i):(int(n),int(u)) for i,n,u in re.findall(r'GQA_WORK core=(\d+) tasks=(\d+) row_updates=(\d+)',log)}
                assert actual=={i:(expected_work['tasks'][i],expected_work['updates'][i]) for i in range(8)},actual
                assert row['commands']==(sum(expected_work['updates']) if c['arm']=='l3' else 0)
                assert total(s,'.m_spmcpRequestsIssued')==(sum(expected_work['copies']) if c['arm']!='cache' else 0)
                row['output_sha256']=sha(path/'output.fp16')
            unchanged();save(path/'accepted.json',row);state['completed'].append(row);state['active']={};proc=None;status()
            return row
        small=lambda arm,depth=4:QWEN/'runs'/f't256_q14_kv2_causal1_{arm}_bc128_d{depth}'
        for arm in ('cache','private','l3'):run('exact_'+arm,small(arm),exact=True)
        for arm in ('cache','private'):run('noop_'+arm,small(arm),direct=True,exact=True)
        for name,source in [('fp32',PIPE/'runs/pipeline_smoke_async'),('mixed',PIPE/'runs/mixed_pipeline'),
            ('init',LIFE/'runs/init'),('both',LIFE/'runs/both'),('sync',LIFE/'runs/both_sync')]:
            run('directed_'+name,source,direct=True)
        run('directed_one_entry_slow',LIFE/'runs/both_one_entry',direct=True,width=8,latency=4)
        run('qwen_small_direct',small('l3'),direct=True)
        run('qwen_small_one_entry_slow',small('l3',1),direct=True,width=8,latency=4)
        save(OUT/'gates_passed.json',dict(cases=len(state['completed']),full_math=True,exact_controls=True,finite_transport=True))
        # T1024 candidate search is already complete; only later-size winners
        # can change while the independent cache queue finishes.
        def best(t,arm):
            rows=json.loads((CACHE/'status.json').read_text())['completed']
            candidates=[r for r in rows if r['case']['t']==t and r['case']['arm']==arm and r['case']['depth']==4 and r['case']['hq']==14]
            r=min(candidates,key=lambda r:r['roi_us']);return Path(r['path'])
        def target(t):
            sources={a:best(t,a) for a in ('cache','private','l3')}
            save(OUT/f'selection_t{t}.json',{a:dict(path=str(p),accepted_sha256=sha(p/'accepted.json')) for a,p in sources.items()})
            # Run the matched L3 pair first so the primary result is available early.
            run(f't{t}_noc',sources['l3'],exact=True)
            run(f't{t}_direct',sources['l3'],direct=True)
            for arm in ('private','cache'):run(f't{t}_{arm}',sources[arm],exact=True)
            if t in (1024,4096):run(f't{t}_slow',sources['l3'],direct=True,width=8,latency=4)
        cache_stage=json.loads((CACHE/'status.json').read_text())['stage']
        assert cache_stage in ('cache_t2048_screen','cache_t4096_transfer','completed_validated'),cache_stage
        target(1024)
        while not (CACHE/'complete.json').exists():
            cs=json.loads((CACHE/'status.json').read_text());assert cs['stage']!='failed',cs
            status('waiting_cache_selection');time.sleep(10)
        freeze(CACHE/'complete.json');freeze(CACHE/'status.json');save(OUT/'inputs.json',frozen)
        for t in (2048,4096):target(t)
        status('completed_validated');save(OUT/'complete.json',state)
        p=ROOT/'docs/experiments/LEDGER.md'
        effects=json.loads((OUT/'effects.json').read_text())
        lines=['\n\n**L3-09 completion update:** all default/directed/full FA gates accepted.']
        for e in effects:
            if e['mode']=='direct':lines.append(f"T{e['t']} direct {e['direct_us']:.4f}us vs {e['baseline']} {e['baseline_us']:.4f}us: {e['latency_reduction_pct']:.3f}% lower latency ({e['speedup']:.6f}x).")
        lines.append('Finite transport assumptions and unchanged bank/ALU resources; full math/exact bits/config/count/bounds. Sensitivity and all regressions retained in direct_link_20260928/experiment_v1/REPORT.md. No RTL-calibrated area/timing or full-model claim.')
        with p.open('a') as f:f.write('\n'.join(lines)+'\n')
    except BaseException as e:
        state['error']=repr(e);state['traceback']=traceback.format_exc()
        if proc and proc.poll() is None:
            proc.terminate()
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
        state['active']={};status('failed');save(OUT/'failed.json',state);raise
if __name__=='__main__':main()
