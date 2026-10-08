"""MOD-10: bounded timing diagnostics, default equivalence and full-output gates."""
import importlib,json,math,os,re,shutil,subprocess,sys,time,traceback,signal
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
from run_l3_kernel_tuning import ENV
from l3_diagnostic_profiles import resources,can_launch,command,allowed_config,trace_window,PROFILES,DIRECT
BASE=ROOT/'results/cacheflex_l3_spm/diagnostics_20260929'
BUILD=BASE/os.environ.get('CACHEFLEX_DIAGNOSTIC_BUILD','build_v1')
OUT=BASE/os.environ.get('CACHEFLEX_DIAGNOSTIC_EXPERIMENT','experiment_v1')
QWEN=ROOT/'results/cacheflex_l3_spm/qwen_gqa_20260928/experiment_v2'
TUNED=ROOT/'results/cacheflex_l3_spm/qwen_analytical_20260928/experiment_v1'
CACHE=ROOT/'results/cacheflex_l3_spm/qwen_cache_analytical_20260928/experiment_v1'
LIFE=ROOT/'results/cacheflex_l3_spm/lifecycle_20260928/experiment_v4'
PIPE=ROOT/'results/cacheflex_l3_spm/pipeline_20260928/experiment_v2'

def make_plan():
    def job(name,source,profile='default',exact=False,trace=False):
        return dict(name=name,source=str(DIRECT/'runs'/source),profile=profile,exact=exact,trace=trace)
    gates=[job('exact_'+arm,'exact_'+arm,exact=True) for arm in ('cache','private','l3')]
    gates += [job('exact_direct','qwen_small_direct',exact=True),
              job('trace_equivalence','qwen_small_direct',exact=True,trace=True)]
    gates += [job('fast_noop_'+arm,'exact_'+arm,'both_fast',exact=True) for arm in ('cache','private')]
    gates += [job('fast_'+kind,'directed_'+kind,'both_fast') for kind in
              ('fp32','mixed','init','both','sync','one_entry_slow')]
    gates += [job('lanes4_'+kind,'directed_'+kind,'lanes4') for kind in ('mixed','sync','one_entry_slow')]
    gates += [job('qwen_small_'+profile,'qwen_small_direct',profile) for profile in ('both_fast','lanes4','pipeline')]
    targets=[job('t4096_direct_trace','t4096_direct',exact=True,trace=True)]
    targets += [job('t4096_direct_'+profile,'t4096_direct',profile) for profile in
                ('reply_fast','read_fast','both_fast','lanes8','lanes4','pipeline')]
    targets += [job('t4096_noc_both_fast','t4096_noc','both_fast')]
    return dict(ledger=['L3-02','L3-03','L3-04','L3-09','MOD-09'],gates=gates,targets=targets,
                max_new_simulators=1,max_total_simulators=6)


def summarize_trace(text):
    """Event-reported ages and conditional samples; window censoring is explicit."""
    phases={k:0 for k in ('capture','issue','complete','sample')}
    source_full=flight_full=samples=0;oldest=[];issue_age=[];complete_age=[]
    for line in text.splitlines():
        fields=dict(re.findall(r'(\w+)=(\w+)',line));phase=fields.get('phase')
        if phase not in phases:continue
        phases[phase]+=1
        if phase=='sample':
            samples+=1;source_full+=int(fields['source'])==4;flight_full+=int(fields['flight'])==4
            oldest.append(int(fields['oldest_age']))
        if phase=='issue':issue_age.append(int(fields['age']))
        if phase=='complete':complete_age.append(int(fields['age']))
    return dict(events=phases,sampled_source_full=source_full,sampled_flight_full=flight_full,
                samples=samples,max_oldest_age_core_cycles=max(oldest,default=0),
                mean_issue_age_core_cycles=sum(issue_age)/len(issue_age) if issue_age else None,
                mean_completion_age_core_cycles=sum(complete_age)/len(complete_age) if complete_age else None,
                caveat='Bounded time window, event ages and conditional active-service samples; not stall-cycle fractions or full-run distributions.')


def check_config(a,b,profile):
    delta={k:[a.get(k),b.get(k)] for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}
    permitted=allowed_config(profile)
    assert not {k:v for k,v in delta.items() if k not in permitted or v[1]!=permitted[k]},delta
    for k,v in permitted.items():assert b[k]==v,(k,b.get(k),v)
    return delta


def main():
    OUT.mkdir(parents=True);(OUT/'runs').mkdir()
    state=dict(pid=os.getpid(),stage='waiting_build',active={},completed=[])
    frozen={};proc=None
    def status(stage=None):
        if stage:state['stage']=stage
        state['updated']=time.time();save(OUT/'status.json',state)
        lines=['# MOD-10 L3 diagnostic results','',f"Stage: {state['stage']}",'',
            '| Case | Profile | ROI us | Change vs same-kernel source |',
            '|---|---|---:|---:|']
        for r in state['completed']:
            lines.append(f"| {r['name']} | {r['job']['profile']} | {r['roi_us']:.4f} | {100*(1-r['roi_us']/r['prior_us']):.4f}% lower |")
        effects=[]
        for r in state['completed']:
            if not r['name'].startswith('t4096_'):continue
            private=json.loads((DIRECT/'runs/t4096_private/accepted.json').read_text())
            cache=json.loads((DIRECT/'runs/t4096_cache/accepted.json').read_text())
            e=dict(name=r['name'],roi_us=r['roi_us'],source_us=r['prior_us'],
                   latency_reduction_pct=100*(1-r['roi_us']/r['prior_us']),
                   speedup_vs_private=private['roi_us']/r['roi_us'],
                   speedup_vs_cache=cache['roi_us']/r['roi_us'])
            effects.append(e)
            lines.append(f"\n{r['name']}: {e['speedup_vs_private']:.6f}x private SPM; {e['speedup_vs_cache']:.6f}x cache. Diagnostic comparison only.")
        save(OUT/'effects.json',effects)
        lines+=['','Accepted only after full math/exact bits/config/count/finite-resource checks.',
            'Fast read/reply are optimistic diagnostic delays, not hardware-feasible performance claims.',
            'Narrow engine profiles change modeled timing, not area or physical lane implementation.',
            'Trace samples and retry counters are not removable runtime fractions. Existing controls remain frozen.']
        (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    def freeze(p):
        p=Path(p);h=sha(p)
        if str(p) in frozen:assert frozen[str(p)]==h,('changed input',p)
        frozen[str(p)]=h
    def unchanged():
        for p,h in frozen.items():assert sha(p)==h,('changed input',p)
    status()
    try:
        save(OUT/'plan.json',make_plan())
        while True:
            if (BUILD/'status.json').exists():
                b=json.loads((BUILD/'status.json').read_text());assert b['stage']!='failed',b
                if b['stage']=='built_requires_validation':break
            time.sleep(10);status()
        frozen.update(json.loads((BUILD/'frozen.json').read_text()))
        for mod in ('run_access_timing','fa_headroom_profiles','test_micro_timing','export_safe_elf',
                    'qwen_l3_reference','fa_independent_reference','run_qwen_l3_attention',
                    'run_l3_kernel_tuning','build_access_timing','l3_diagnostic_profiles'):
            freeze(importlib.import_module(mod).__file__)
        freeze(__file__);freeze(ROOT/'docs/experiments/L3_DIAGNOSTICS_20260929.md')
        for arm in ('private','cache'):
            freeze(DIRECT/f'runs/t4096_{arm}/accepted.json')
        save(OUT/'inputs.json',frozen);unchanged()
        def run(name,source,profile="default",exact=False,trace=False):
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
            cmd=command(source,BUILD,path,profile,trace_window(source) if trace else None)
            oldcfg=json.loads((source/'config.json').read_text())['system']['ruby']['network']
            direct=oldcfg['cacheflex_direct_reduction'];width=oldcfg['cacheflex_direct_bytes']
            job=dict(name=name,source=str(source),profile=profile,exact=exact,trace=trace,
                     trace_window=trace_window(source) if trace else None)
            save(path/'job.json',job);save(path/'command.json',cmd);save(OUT/'inputs.json',frozen);unchanged()
            status('running_'+name)
            with (path/'run.log').open('w') as f:
                while True:
                    r=resources();save(path/'resource_launch.json',r)
                    if can_launch(r):break
                    state['resource_wait']=r;status();time.sleep(10)
                state.pop('resource_wait',None)
                proc=subprocess.Popen(['nice','-n','10',*cmd],cwd=ROOT,env=ENV,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                start=time.monotonic();last_resource=0;paused=False;low_idle=0
                state['active']={name:dict(pid=proc.pid,start=time.time())};status()
                while proc.poll() is None:
                    assert time.monotonic()-start<96*3600,'96h timeout'
                    if time.monotonic()-last_resource>15:
                        r=resources();save(path/'resource_runtime.json',r);last_resource=time.monotonic()
                        low_idle=low_idle+1 if r['cpu_idle_pct']<25 else 0
                        too_many=len(r['simulators'])>6
                        if not paused and (too_many or low_idle>=3):
                            os.killpg(proc.pid,signal.SIGSTOP);paused=True
                            state['paused_own_simulator']=r;status()
                        elif paused and can_launch({**r,'simulators':[p for p in r['simulators'] if p!=proc.pid]}):
                            os.killpg(proc.pid,signal.SIGCONT);paused=False;low_idle=0
                            state.pop('paused_own_simulator',None);status()
                    assert shutil.disk_usage(OUT).free>75*2**30,'disk guard'
                    tracepath=path/'reduction_trace.log'
                    assert not tracepath.exists() or tracepath.stat().st_size<128*2**20,'trace size guard'
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
            delta=check_config(a,b,profile)
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
                    mismatches=[(k,control.get(k),ep.get(k)) for k in control.keys()|ep.keys() if control.get(k)!=ep.get(k)]
                    assert not mismatches,mismatches[:15]
            row=dict(name=name,job=job,case=c,path=str(path),roi_us=s['simTicks']/1e6,prior_us=prior['roi_us'],
                commands=total(s,'.reductionAccepted'),direct_bytes=total(s,'.directCommandBytes')+total(s,'.directAckBytes'),
                config_delta=delta,exact_prior_stats=exact,host_seconds=time.monotonic()-start,
                direct_stats={k.rsplit('.',1)[-1]:v for k,v in s.items() if k.startswith('system.ruby.network.direct')},
                network_flits={k:v for k,v in s.items() if 'flits_injected' in k})
            if trace:
                tracepath=path/'reduction_trace.log'
                assert tracepath.is_file() and 0<tracepath.stat().st_size<128*2**20
                text=tracepath.read_text()
                assert all('phase='+phase in text for phase in ('capture','issue','complete','sample'))
                row['trace_summary']=summarize_trace(text)
                save(path/'trace_summary.json',row['trace_summary'])
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
        plan=make_plan()
        save(OUT/'plan.json',plan)
        for job in plan['gates']:
            run(**job)
        save(OUT/'gates_passed.json',dict(cases=len(state['completed']),full_math=True,
             exact_default=True,trace_equivalence=True,finite_resources=True))
        with (ROOT/'docs/experiments/LEDGER.md').open('a') as f:
            f.write('\n**MOD-10 runtime gates accepted:** '+str(len(state['completed']))+
                    ' default/no-op/directed/full small-Qwen cases; full oracle/exact bits/config/count/bounds. Targets follow. See '+str(OUT/'REPORT.md')+'.\n')
        for job in plan['targets']:
            run(**job)
        status('completed_validated');save(OUT/'complete.json',state)
        with (ROOT/'docs/experiments/LEDGER.md').open('a') as f:
            f.write('\n**MOD-10 complete:** '+str(len(state['completed']))+' accepted cases.\n')
            for r in state['completed']:
                if r['name'].startswith('t4096_'):
                    f.write(f"{r['name']} {r['roi_us']:.4f}us vs source {r['prior_us']:.4f}us; {100*(1-r['roi_us']/r['prior_us']):.4f}% lower.\n")
            f.write('Optimistic latency sensitivities and modeled lane schedules only; no physical or area claim. '+str(OUT/'REPORT.md')+'\n')
    except BaseException as e:
        state['error']=repr(e);state['traceback']=traceback.format_exc()
        if proc and proc.poll() is None:
            os.killpg(proc.pid,signal.SIGCONT)
            proc.terminate()
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
        state['active']={};status('failed');save(OUT/'failed.json',state);raise
if __name__=='__main__':main()
