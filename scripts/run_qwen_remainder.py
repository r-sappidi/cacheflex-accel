"""AI-12: gated, equal-budget analytical remainder tuning; OPS-20 admission.

No old scheduler edits. Reserve its maximum possible future concurrency, not
merely currently live simulators. New work starts when its monotone remaining
plan releases enough slots. All children share the existing 24-CPU affinity.
"""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import concurrent.futures as cf
import fcntl,importlib,json,re,resource,shutil,subprocess,sys,threading,time,traceback
from pathlib import Path
ROOT=Path('/data2/rsappidi/cacheflex_mc')
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'benchmarks/ai_workloads'),str(ROOT/'benchmarks/cacheflex_flash_attention')]
import numpy as np
from build_access_timing import save,sha
from run_qwen_projection_v2 import CONTROL,PLATFORM,config_audit,ENV
from sweep_mlp_study import epochs
from run_access_timing import stats
from test_micro_timing import check_global_guard
from parallel_budget_scheduler import owned
from qwen_remainder_reference import OPS,V,generate,validate
BASE=ROOT/'results/ai_workloads/qwen_remainder_20260929'
OUT=BASE/'experiment_v1'
JOURNAL=ROOT/'results/parallel_budget_20260929'
GIB=2**30

def read(p):return json.loads(Path(p).read_text())
def choices(op):return [1,2,4] if op=='logits' else [1,8,32]
def name(c):return f"{c['stage']}_{c['op']}_{c['arm']}_t{c['t']}_v{c['v']}_b{c['b']}"

def describe(op,t,v,b):
    rows=v if op=='logits' else t
    per_core=[rows*(i+1)//8-rows*i//8 for i in range(8)]
    return dict(rows_per_core=per_core,block=b,
      stream_bytes={'rmsnorm':t*896*6+1792,'final_norm':t*896*6+1792,
                    'residual':t*896*6,'rope':t*(1024*4+128),
                    'rope_tables':t*128+128,'embedding':t*(896*4+4),
                    'logits':v*896*2+1792+v*2}[op],
      weight_bytes=(v*896*2 if op in ('embedding','logits') else 1792 if op in ('rmsnorm','final_norm') else 0),
      arithmetic_intensity_note='Logits has one query vector and about one FLOP/weight byte; streaming weights are consumed once. Other operators have no cross-row operand reuse except tiny norm weights/RoPE tables.',
      spm_commands=0,spm_strategy='Coherent SVE fallback on private-SPM platform; do not force staging single-use operands.',
      register_accumulators=b if op=='logits' else 1,
      objective='first full invocation including synchronization, stores and timed table generation in its own component')

def command(path,inp,c):
    guest=BASE/'guest_v1/guest';args=list(map(str,[OPS.index(c['op']),c['t'],c['v'],c['b']]))
    cmd=read(CONTROL/'command.json');cmd[1]='--outdir='+str(path)
    cmd[cmd.index('--cmd')+1]=str(guest);cmd[cmd.index('--options')+1]=' '.join(args)
    for i,x in enumerate(cmd):
        if x.startswith('system.cpu[0].workload[0].cmd='):cmd[i]='system.cpu[0].workload[0].cmd='+repr(['qwen-remainder']+args)
        if x.startswith('--l1d_num_spm_ways='):cmd[i]='--l1d_num_spm_ways='+str(6 if c['arm']=='spm' else 0)
    cmd+=['-P',f'system.readfile={str(inp)!r}']
    return cmd,['qwen-remainder']+args

def capacity():
    """Conservative future reservation; the old frozen plan only shrinks."""
    budget=read(JOURNAL/'budget.json');state=read(JOURNAL/'status.json');r=read(JOURNAL/'resources.json')
    assert not state.get('failed'),'existing budget queue has a failure; inspect before adding work'
    complete=JOURNAL/'complete.json'
    finished=complete.exists() and not read(complete).get('failed')
    assert time.time()-r['updated']<180 or finished,'stale scheduler status'
    remaining=sum(state['remaining'].values())
    # Include still-active reference workers in addition to every remaining sim.
    refs=sum(x['name'].startswith('reference_') for x in state['active'])
    old_slots=0 if finished else min(budget['max_worker_slots'],remaining+refs)
    # Always retain the complete old llama worker+sim reservation, even if done.
    reserved=budget['outside_reserve']+(old_slots+1)*6*GIB
    slots=min(22,max(0,(budget['ram_limit']-reserved)//(6*GIB)),max(0,22-old_slots))
    if not finished and (r.get('pressure') or (r['cpu_pct_of_host'] or 0)>=20 or r['user_ram_pct']>=18):slots=0
    if sum(x['rss'] for x in owned().values())>=budget['ram_limit']-4*GIB:slots=0
    return int(slots),dict(old_future_slots=old_slots,old_reserved_bytes=reserved,available_slots=int(slots),old_sample=r)

def prepare():
    OUT.mkdir(parents=True);(OUT/'runs').mkdir();(BASE/'inputs').mkdir(exist_ok=True)
    model_path=ROOT/'results/cacheflex_l3_spm/qwen_gqa_20260928/model_config.json'
    model=read(model_path)
    assert (model['hidden_size'],model['num_attention_heads'],model['num_key_value_heads'],model['vocab_size'],model['num_hidden_layers'])==(896,14,2,V,24)
    assert model['rms_norm_eps']==1e-6 and model['rope_theta']==1e6 and model['tie_word_embeddings']
    manifest=read(PLATFORM/'inputs.json');manifest.update(read(BASE/'guest_v1/frozen.json'))
    for mod in ('run_qwen_remainder','qwen_remainder_reference','run_qwen_projection_v2',
                'sweep_mlp_study','run_access_timing','test_micro_timing','fa_headroom_profiles',
                'parallel_budget_scheduler','parallel_budget_worker','build_access_timing'):
        p=Path(importlib.import_module(mod).__file__);manifest[str(p)]=sha(p)
    for p in (ROOT/'docs/experiments/QWEN_REMAINDER_TUNING.md',model_path,CONTROL/'command.json',CONTROL/'config.json',JOURNAL/'budget.json'):
        manifest[str(p)]=sha(p)
    for p,h in manifest.items():assert sha(p)==h,('changed input',p)
    save(OUT/'inputs.json',manifest)
    save(OUT/'plan.json',dict(ledger='AI-12',related=['AI-03','AI-10','AI-11','OPS-20'],
        operators=list(OPS),gates='T25/V263: all 3 choices and both arms; T1/V17 tail gates both arms all ops',
        selection={op:[describe(op,1 if op=='logits' else 1024,V,b) for b in choices(op)] for op in OPS},
        screen='T1024 (logits T1/full vocabulary), equal 3 candidates/backend/operator',
        transfers=[2048,4096],transfer_policy='baseline + independently selected first-ROI winner; logits measured once, independent of context length',
        max_parallel=22,resource_policy='OPS-20 shared CPU set; future old concurrency reserved; 1.5GiB coordinator +4.5GiB/simulator; 6GiB/slot; no old process changes',
        multiplicities=dict(rmsnorm=48,residual=48,rope=24,rope_tables=1,embedding=1,final_norm=1,logits=1),
        model_total=None,integration_required=True))

def main():
    lockfile=(OUT/'coordinator.lock').open('w');fcntl.flock(lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'status.json').exists(),'Do not duplicate/restart without inspecting evidence'
    budget=read(JOURNAL/'budget.json');os.sched_setaffinity(0,budget['cpu_set'])
    resource.setrlimit(resource.RLIMIT_AS,(budget['worker_as'],resource.getrlimit(resource.RLIMIT_AS)[1]))
    frozen=read(OUT/'inputs.json');lock=threading.RLock();math_lock=threading.Lock();stop=threading.Event();active={};rows=[]
    state=dict(pid=os.getpid(),stage='queued_resource_wait',active=active,completed=rows)
    last_cpu=[None,None]
    def resource_sample():
        processes=owned();now=time.monotonic();cpu=None
        if last_cpu[0] is not None:
            delta=sum(max(0,p['cpu_ticks']-last_cpu[1].get((pid,p['start_ticks']),p['cpu_ticks'])) for pid,p in processes.items())
            cpu=100*delta/os.sysconf('SC_CLK_TCK')/(now-last_cpu[0])/os.cpu_count()
        last_cpu[:]=[now,{(pid,p['start_ticks']):p['cpu_ticks'] for pid,p in processes.items()}]
        rss=sum(x['rss'] for x in processes.values())
        available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
        state['resources']=dict(updated=time.time(),user_rss=rss,user_cpu_pct=cpu,available_bytes=available,
                                cpu_set=budget['cpu_set'],ram_limit=budget['ram_limit'])
        state['pressure']=rss>budget['ram_limit']-4*GIB or available<64*GIB or (cpu is not None and cpu>20)
    def status():
        with lock:
            state['updated']=time.time();save(OUT/'status.json',state)
            lines=['# AI-12: accepted Qwen remainder results','',f"Stage: {state['stage']}",'',
                   '| Case | First ROI us | Second call us |','|---|---:|---:|']
            lines += [f"| {x['name']} | {x['first_us']:.4f} | {x['repeat_us']:.4f} |" for x in rows]
            lines += ['','Only accepted results. Both arms use coherent SVE loads for these streaming operators. No SPM offload or L3 speedup claimed. First-use component measurements; second call is diagnostic. Integrated-layer validation still required before model totals.']
            (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    def unchanged():
        with lock:items=list(frozen.items())
        for p,h in items:assert sha(p)==h,('changed frozen input',p)
    def input_for(c):
        p=BASE/'inputs'/f"{c['op']}_t{c['t']}_v{c['v']}.bin"
        if not p.exists():
            with math_lock:generate(p,c['op'],c['t'],c['v'])
        meta=read(p.with_suffix('.json'))
        assert meta['sha256']==sha(p) and meta['reference_sha256']==sha(p.with_suffix('.npz'))
        with lock:
            for f in (p,p.with_suffix('.npz'),p.with_suffix('.json')):
                digest=sha(f)
                assert str(f) not in frozen or frozen[str(f)]==digest
                frozen[str(f)]=digest
            save(OUT/'inputs.json',frozen)
        return p
    def run(c,inp):
        path=OUT/'runs'/name(c);path.mkdir();proc=None
        try:
            unchanged();cmd,args=command(path,inp,c);save(path/'command.json',cmd);save(path/'case.json',c)
            # prlimit avoids unsafe Python preexec_fn in this threaded runner.
            with (path/'run.log').open('w') as log:
                proc=subprocess.Popen(['prlimit',f'--as={budget["simulator_as"]}:','--','nice','-n','10',*cmd],cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT)
                with lock:active[name(c)]=dict(pid=proc.pid,path=str(path));status()
                start=time.monotonic();paused=False
                while proc.poll() is None:
                    assert not stop.is_set(),'peer failed'
                    assert time.monotonic()-start<96*3600,'96h timeout'
                    assert shutil.disk_usage(OUT).free>75*GIB,'disk guard'
                    assert (path/'run.log').stat().st_size<128*2**20,'log guard'
                    user_rss=sum(x['rss'] for x in owned().values())
                    pressure=user_rss>budget['ram_limit']-GIB or state.get('pressure',False)
                    if pressure!=paused:
                        import signal
                        proc.send_signal(signal.SIGSTOP if pressure else signal.SIGCONT);paused=pressure
                    try:proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:pass
            text=(path/'run.log').read_text()
            assert proc.returncode==0 and 'because exiting with last active thread context' in text,text[-3000:]
            assert not re.search(r'panic:|fatal:|Simulated exit code not 0|\bFAIL\b',text),text[-3000:]
            delta=config_audit(path,BASE/'guest_v1/guest',inp,args,c['arm'])
            checks=[dict(re.findall(r'(\w+)=([^\s]+)',l)) for l in text.splitlines() if l.startswith('REMAINDER_RESULT')]
            assert len(checks)==2 and all(x['nonfinite']=='0' for x in checks) and checks[0]['hash']==checks[1]['hash']
            with math_lock:numeric=validate(path/'output.bin',inp,c['op'])
            data=np.fromfile(path/'output.bin',dtype='<u2');h=1469598103934665603
            for x in data:h=((h^int(x))*1099511628211)&((1<<64)-1)
            assert f'{h:016x}'==checks[-1]['hash'],'export differs from guest'
            timing=epochs(path,2)
            assert all(x['copies_issued']==x['copies_completed']==0 for x in timing)
            for ep in stats(path):
                check_global_guard(ep)
                for i in range(8):
                    for domain in ('regular','spm'):
                        for kind,cap in [('Load',32),('Store',48)]:assert ep[f'system.cpu{i}.{domain}Lsq0.boundedMax{kind}Occupancy']<=cap
                    for suffix in ('reductionAccepted','reductionCompleted'):assert ep[f'system.ruby.l0_cntrl{i}.sequencer.{suffix}']==0
            unchanged();digest=sha(path/'output.bin')
            row=dict(name=name(c),case=c,path=str(path),first_us=timing[0]['us'],repeat_us=timing[1]['us'],epochs=timing,math=numeric,checks=checks,output_sha256=digest,config_delta=delta,structure=describe(c['op'],c['t'],c['v'],c['b']))
            with lock:
                peers=[r for r in rows if all(r['case'][k]==c[k] for k in ('op','t','v'))]
                assert all(r['output_sha256']==digest for r in peers),'backend/block changed output bits'
                save(path/'accepted.json',row);rows.append(row);status()
            return row
        except BaseException as exc:
            stop.set();save(path/'failed.json',dict(error=repr(exc),traceback=traceback.format_exc()));raise
        finally:
            if proc and proc.poll() is None:
                import signal
                proc.send_signal(signal.SIGCONT);proc.terminate()
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
            with lock:active.pop(name(c),None);status()
    def phase(label,cases):
        state['stage']=label;status();pending=list(cases);futures={}
        with cf.ThreadPoolExecutor(max_workers=22) as pool:
            while pending or futures:
                for f in list(futures):
                    if f.done():f.result();del futures[f]
                resource_sample();slots,admission=capacity();state['admission']=admission
                if state['pressure']:slots=0
                # Reference generation shares the coordinator reservation and is serial.
                if pending and len(futures)<slots:
                    c=dict(pending.pop(0),stage=label);inp=input_for(c);futures[pool.submit(run,c,inp)]=c
                else:time.sleep(10)
                status()
    def cases(t,v,selection=None):
        return [dict(op=op,arm=arm,t=1 if op=='logits' and v==V else t,v=v,b=b)
            for op in OPS for arm in ('cache','spm') for b in (selection[op][arm] if selection else choices(op))]
    try:
        unchanged();status()
        phase('tail_gates',cases(1,17,{op:{a:[1] for a in ('cache','spm')} for op in OPS}))
        phase('shape_gates',cases(25,263))
        save(OUT/'gates_passed.json',dict(cases=len(rows),full_oracle=True,peer_bits=True,config=True,finite_resources=True))
        phase('screen',cases(1024,V))
        selected={}
        for op in OPS:
            selected[op]={}
            for arm in ('cache','spm'):
                best=min((r for r in rows if r['case']['stage']=='screen' and r['case']['op']==op and r['case']['arm']==arm),key=lambda r:r['first_us'])
                selected[op][arm]=list(dict.fromkeys([1,best['case']['b']]))
        save(OUT/'transfer_selection.json',dict(selected=selected,objective='first full ROI us'))
        for t in (2048,4096):phase(f'target_t{t}',[c for c in cases(t,V,selected) if c['op']!='logits'])
        effects=[]
        for op in OPS:
            for t in ([1] if op=='logits' else [1024,2048,4096]):
                best={a:min((r for r in rows if r['case']['op']==op and r['case']['t']==t and r['case']['v']==V and r['case']['arm']==a),key=lambda r:r['first_us']) for a in ('cache','spm')}
                effects.append(dict(op=op,t=t,cache=best['cache']['name'],spm=best['spm']['name'],cache_us=best['cache']['first_us'],spm_us=best['spm']['first_us'],speedup=best['cache']['first_us']/best['spm']['first_us']))
        save(OUT/'effects.json',effects);save(OUT/'complete.json',dict(cases=len(rows),effects=effects,model_total=None))
        state['stage']='completed_validated';status()
        with (ROOT/'docs/experiments/LEDGER.md').open('a') as f:f.write('\n**AI-12 complete:** '+json.dumps(effects)+'\nFull-output/config/resource gates; shared coherent SVE implementation, no L3 offload. Integration remains pending.\n')
    except BaseException as exc:
        stop.set();state['stage']='failed';state['error']=repr(exc);status();raise
if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    else:main()
