"""Bounded, fail-closed model-shaped causal GQA transfer on frozen L3 hardware."""
import concurrent.futures as cf
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback

ROOT=Path('/data2/rsappidi/cacheflex_mc')
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'benchmarks/cacheflex_flash_attention')]
import numpy as np
from build_access_timing import sha,save
from run_access_timing import stats
from fa_headroom_profiles import leaves
from test_micro_timing import check_global_guard
from export_safe_elf import export_hash
from qwen_l3_reference import reference,validate
from run_l3_kernel_tuning import resources

BASE=ROOT/'results/cacheflex_l3_spm/qwen_gqa_20260928'
OUT=BASE/os.environ.get('CACHEFLEX_QWEN_EXPERIMENT','experiment_v2')
GUEST=BASE/os.environ.get('CACHEFLEX_QWEN_GUEST','guest_v4')
OLD=ROOT/'results/cacheflex_l3_spm/lifecycle_20260928/experiment_v4'
KERNEL=ROOT/'results/cacheflex_l3_spm/kernel_tuning_20260928/experiment_v1'
ENV=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib',
         OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')


def case(t,arm,bc=None,hq=14,hkv=2,causal=True,depth=4):
    return dict(t=t,arm=arm,bc=bc or (512 if arm=='cache' else 256),br=16,r=256,
                hq=hq,hkv=hkv,causal=causal,depth=depth)


def label(c):
    return f"t{c['t']}_q{c['hq']}_kv{c['hkv']}_causal{int(c['causal'])}_{c['arm']}_bc{c['bc']}_d{c['depth']}"


def work(c):
    """Expected ownership/counts from complete logical tasks, independent of stats."""
    t,r,bc=c['t'],c['r'],c['bc'];groups=t//r
    tasks=[0]*8;updates=[0]*8;copies=[0]*8
    for task in range(c['hq']*groups):
        core=task%8;head=task//groups;qi=((task%groups+head)%groups)*r
        end=min(t,((qi+r+bc-1)//bc)*bc) if c['causal'] else t
        tasks[core]+=1;updates[core]+=r*(end//bc);copies[core]+=2*end
    return dict(tasks=tasks,updates=updates,copies=copies)


def key(c):return (c['t'],c['bc'],c['hq'],c['hkv'],c['causal'])


def main():
    OUT.mkdir(parents=True);(OUT/'runs').mkdir();(OUT/'references').mkdir()
    state=dict(pid=os.getpid(),stage='preparing',active={},completed=[])
    lock=threading.RLock();launch_lock=threading.Lock();stop=threading.Event()
    def status(stage=None):
        with lock:
            if stage:state['stage']=stage
            state['updated']=time.time();save(OUT/'status.json',state)
            lines=['# Qwen-shaped causal GQA: accepted results','',f"Stage: {state['stage']}",'',
                   'Hq14/Hkv2/d64, batch1, 8 cores. Existing FP16 kernel, not full Qwen inference.',
                   'Br16/R256; inherited cache Bc512 and private/L3 Bc256. No new tile sweep.',
                   '', '| Case | Complete ROI us | Updates | Copies |', '|---|---:|---:|---:|']
            for x in state['completed']:
                lines.append(f"| {x['name']} | {x['roi_us']:.4f} | {x['commands']} | {x['copies']} |")
            effects=[]
            for t in (1024,2048,4096):
                arms={x['case']['arm']:x for x in state['completed'] if x['case']['t']==t and
                      x['case']['hq']==14 and x['case']['depth']==4 and
                      x['case']['bc']==(512 if x['case']['arm']=='cache' else 256)}
                if 'l3' in arms:
                    for arm in ('cache','private'):
                        if arm in arms:
                            a,b=arms[arm]['roi_us'],arms['l3']['roi_us']
                            effects.append(dict(t=t,control=arm,control_us=a,l3_us=b,
                                                latency_reduction_pct=100*(1-b/a),speedup=a/b))
                            lines.append(f"\nT{t} L3 versus {arm}: {100*(1-b/a):.3f}% lower latency, {a/b:.6f}x.")
            lines+=['','All rows passed full-output independent arithmetic/configuration/resource gates.',
                    'Different Bc has its own oracle; same-Bc outputs must match exactly.',
                    'Depth1 is stress only. Positive/negative effects retained. No full-model or BF16 claim.']
            (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n');save(OUT/'effects.json',effects)
    status()
    try:
        frozen=json.loads((GUEST/'frozen.json').read_text())
        frozen.update(json.loads((OLD/'inputs.json').read_text()))
        for mod in ('qwen_l3_reference','fa_independent_reference','run_access_timing','test_micro_timing',
                    'fa_headroom_profiles','export_safe_elf','build_access_timing','run_l3_kernel_tuning'):
            p=Path(importlib.import_module(mod).__file__);frozen[str(p)]=sha(p)
        for p in (Path(__file__),ROOT/'scripts/build_qwen_l3_attention.py',BASE/'model_config.json',
                  ROOT/'docs/experiments/QWEN_GQA_L3.md'):
            frozen[str(p)]=sha(p)
        for arm in ('cache','private_spm','init'):
            source=OLD/'runs'/f't4096_{arm}'
            for n in ('accepted.json','command.json','config.json'):frozen[str(source/n)]=sha(source/n)
        for arm in ('cache','private','l3'):
            p=KERNEL/'runs'/f't1024_{arm}_k1_br32_bc128_r192_d4'/'output.fp16'
            frozen[str(p)]=sha(p)
        def unchanged():
            for p,h in frozen.items():assert sha(p)==h,('changed frozen input',p)
        unchanged();save(OUT/'inputs.json',frozen);save(OUT/'resource_preflight.json',resources())
        save(OUT/'plan.json',dict(ledger=['FA-06','FA-07','FA-08','FA-09','L3-05','L3-06'],
            model='Qwen2.5-0.5B-Instruct attention shape',heads=[14,2],d=64,causal=True,
            precision='FP16 GEMM with FP32 online state; not published BF16 inference',
            max_new_simulators=2,targets=[2048,4096],cache_retuning=False))
        references={}
        def get_reference(c):
            k=key(c)
            if k in references:return
            name=f't{c["t"]}_bc{c["bc"]}_q{c["hq"]}_kv{c["hkv"]}_causal{int(c["causal"])}'
            status('reference_'+name)
            def progress(done,total):
                state['reference_progress']=dict(name=name,heads_done=done,heads_total=total);status()
            dest=OUT/'references'/(name+'.npy')
            expected=reference(*k,progress=progress);np.save(dest,expected)
            save(dest.with_suffix('.json'),dict(independent=True,sha256=sha(dest)))
            frozen[str(dest)]=sha(dest);save(OUT/'inputs.json',frozen)
            references[k]=expected;state.pop('reference_progress',None)
        def run(c):
            name=label(c);path=OUT/'runs'/name;path.mkdir();proc=None;start=time.monotonic()
            try:
                assert not stop.is_set();unchanged()
                arm=c['arm'];spm=arm!='cache';reduction=arm=='l3'
                source=OLD/'runs'/f"t4096_{'init' if reduction else ('private_spm' if spm else 'cache')}"
                args=list(map(str,(c['t'],8,c['br'],c['bc'],c['r']//c['br'],1024,0,int(spm),
                                   5 if reduction else 1,1,c['hq'],c['hkv'],int(c['causal']))))
                cmd=json.loads((source/'command.json').read_text());cmd[1]='--outdir='+str(path)
                cmd[cmd.index('--cmd')+1]=str(GUEST/'guest.export-safe')
                cmd[cmd.index('--options')+1]=' '.join(args)
                for i,x in enumerate(cmd):
                    if x.startswith('system.cpu[0].workload[0].cmd='):cmd[i]=f"system.cpu[0].workload[0].cmd={['kv-fa']+args!r}"
                    if x.startswith('--cacheflex-l3-reduction-depth='):cmd[i]=f"--cacheflex-l3-reduction-depth={c['depth']}"
                    if '.l3ReductionQueue.buffer_size=' in x:cmd[i]=x.split('=')[0]+'='+str(c['depth'])
                save(path/'command.json',cmd);save(path/'case.json',c)
                with (path/'run.log').open('w') as log:
                    with launch_lock:
                        while True:
                            assert not stop.is_set();r=resources();save(path/'resource_launch.json',r)
                            if r['cpu_idle_pct']>=50 and r['available_gib']>=128 and r['free_disk_gib']>=100 and len(r['simulators'])<6:break
                            with lock:state['resource_wait']=dict(job=name,resources=r);status()
                            time.sleep(10)
                        proc=subprocess.Popen(['nice','-n','10',*cmd],cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT)
                        with lock:state.pop('resource_wait',None);state['active'][name]=dict(pid=proc.pid,start=time.time());status()
                    while proc.poll() is None:
                        assert not stop.is_set(),'peer failed'
                        assert time.monotonic()-start<12*3600,'12h timeout'
                        assert shutil.disk_usage(OUT).free>75*2**30,'disk guard'
                        memory=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))
                        assert memory>64*2**20,'memory guard'
                        try:proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:status()
                logtext=(path/'run.log').read_text()
                assert proc.returncode==0 and 'KV_ROI_END' in logtext and 'because exiting with last active thread context' in logtext,logtext[-3000:]
                assert not re.search(r'panic:|fatal:|Simulated exit code not 0|\bFAIL\b',logtext),logtext[-3000:]
                def fields(p):return dict(leaves(json.loads((p/'config.json').read_text().replace(str(p),'<OUT>'))))
                old,new=fields(source),fields(path)
                delta={k:[old.get(k),new.get(k)] for k in old.keys()|new.keys() if old.get(k)!=new.get(k)}
                allowed=r'system.cpu\[\d+\].workload\[0\].(cmd\[\d+\]|executable)'
                if c['depth']==1:allowed+=r'|system.ruby.l0_cntrl\d+.sequencer.cacheflex_reduction_(source|sink)_entries|system.ruby.l2_cntrl\d+.(l3_reduction_entries|l3ReductionQueue.buffer_size)'
                assert not {k:v for k,v in delta.items() if not re.fullmatch(allowed,k)},delta
                assert new['system.cpu[0].workload[0].executable']==str(GUEST/'guest.export-safe')
                assert [new[f'system.cpu[0].workload[0].cmd[{i}]'] for i in range(14)]==['kv-fa']+args
                expected_work=work(c)
                actual={int(i):(int(n),int(u)) for i,n,u in re.findall(r'GQA_WORK core=(\d+) tasks=(\d+) row_updates=(\d+)',logtext)}
                assert actual=={i:(expected_work['tasks'][i],expected_work['updates'][i]) for i in range(8)},actual
                epochs=stats(path);s=epochs[0];total=lambda suffix:sum(v for k,v in s.items() if k.endswith(suffix))
                commands=sum(expected_work['updates']) if reduction else 0
                copies=sum(expected_work['copies']) if spm else 0
                assert total('.reductionAccepted')==total('.reductionCompleted')==commands
                assert total('.l3ReductionStarts')==total('.l3ReductionFinishes')==commands
                assert total('.l3ReductionHalfCommands')==commands
                assert total('.l3ReductionInitializations')==(c['hq']*c['t'] if reduction else 0)
                assert total('.l3Finalizations')==0
                assert total('.reductionTransportBytes')==144*commands
                assert total('.m_spmcpRequestsIssued')==total('.m_spmcpRequestsCompleted')==copies
                for ep in epochs:
                    check_global_guard(ep)
                    for i in range(8):
                        assert new[f'system.ruby.l2_cntrl{i}.L2cache.size']==524288
                        assert not new[f'system.ruby.l2_cntrl{i}.L2cache.spm_separate_backing']
                        for domain in ('regular','spm'):
                            for kind,cap in [('Load',32),('Store',48)]:assert ep[f'system.cpu{i}.{domain}Lsq0.boundedMax{kind}Occupancy']<=cap
                        for q in ('Source','Sink'):assert ep[f'system.ruby.l0_cntrl{i}.sequencer.reductionMax{q}']<=c['depth']
                        maximum=c['depth'] if new[f'system.ruby.l2_cntrl{i}.l3_reduction_pipeline'] else 1
                        assert ep[f'system.ruby.l2_cntrl{i}.L2cache.l3ReductionMaxActive']<=maximum
                export_hash(path/'output.fp16',logtext)
                arithmetic=validate(path/'output.fp16',references[key(c)],c['t'],c['hq'],c['hkv'],c['causal'],dense=c['t']<=1024)
                exact_source=None
                if c['hq']==1 and not c['causal']:
                    exact_source=KERNEL/'runs'/f't1024_{arm}_k1_br32_bc128_r192_d4'/'output.fp16'
                else:
                    with lock:
                        peer=next((r for r in state['completed'] if key(r['case'])==key(c)),None)
                    if peer:exact_source=Path(peer['path'])/'output.fp16'
                if exact_source:assert sha(path/'output.fp16')==sha(exact_source),'same-Bc output bits changed'
                row=dict(name=name,case=c,path=str(path),roi_us=s['simTicks']/1e6,commands=commands,copies=copies,
                    arithmetic=arithmetic,work=expected_work,exact_output_source=str(exact_source) if exact_source else None,
                    output_sha256=sha(path/'output.fp16'),config_delta=delta,host_seconds=time.monotonic()-start,
                    source_full=total('.reductionSourceFull'),transport_bytes=total('.reductionTransportBytes'),
                    dram_read_bytes=total('.dram.bytesRead::total'),dram_write_bytes=total('.dram.bytesWritten::total'))
                unchanged();save(path/'accepted.json',row)
                with lock:state['completed'].append(row);status()
                return row
            except BaseException as exc:
                stop.set();save(path/'failed.json',dict(error=repr(exc),traceback=traceback.format_exc()));raise
            finally:
                if proc and proc.poll() is None:
                    proc.terminate()
                    try:proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()
                with lock:state['active'].pop(name,None);status()
        def phase(name,cases):
            for c in cases:get_reference(c)
            status(name)
            with cf.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(run,cases))
            for k in {key(c) for c in cases}:
                assert len({r['output_sha256'] for r in state['completed'] if key(r['case'])==k})==1,('output equality',k)
        phase('one_head_compatibility',[dict(case(1024,a,128,hq=1,hkv=1,causal=False),r=128) for a in ('cache','private','l3')])
        phase('causal_gqa_small',[case(256,a,128) for a in ('cache','private','l3')])
        phase('causal_gqa_one_entry',[case(256,'l3',128,depth=1)])
        phase('t1024_validation',[case(1024,a) for a in ('cache','private','l3')]+[case(1024,'cache',256)])
        save(OUT/'gates_passed.json',dict(cases=len(state['completed']),full_math=True,exact_output=True,
              causal=True,gqa_head_mapping=True,one_entry=True))
        for t in (2048,4096):phase(f't{t}_performance',[case(t,a) for a in ('cache','private','l3')])
        status('completed_validated');save(OUT/'complete.json',dict(cases=len(state['completed'])))
        ledger=ROOT/'docs/experiments/LEDGER.md';text=ledger.read_text()
        update=['','**L3-06 completion update:** all '+str(len(state['completed']))+' cases accepted.']
        for e in json.loads((OUT/'effects.json').read_text()):
            update.append(f"T{e['t']} L3 {e['l3_us']:.4f}us versus {e['control']} {e['control_us']:.4f}us: {e['latency_reduction_pct']:.3f}% lower latency ({e['speedup']:.6f}x).")
        update+=['Full causal GQA oracle/exact output/configuration/resource checks; model-shaped FP16 kernel, not full BF16 inference.','']
        anchor='## Deliberation template and maintenance';assert text.count(anchor)==1
        ledger.write_text(text.replace(anchor,'\n'.join(update)+'\n'+anchor))
        save(OUT/'ledger_recorded.json',dict(time=time.time()))
        with (OUT/'inventory_refresh.log').open('w') as log:
            subprocess.run([sys.executable,str(ROOT/'scripts/index_experiment_history.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    except BaseException as exc:
        state['error']=repr(exc);status('failed');raise


if __name__=='__main__':main()
