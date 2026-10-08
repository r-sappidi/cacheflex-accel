"""AI-11 projection tuning; same finite FA/FFN platform, full acceptance gates."""
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
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'benchmarks/ai_workloads'),str(ROOT/'benchmarks/cacheflex_flash_attention')]
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["OMP_NUM_THREADS"]="1"
import numpy as np
from build_access_timing import save,sha
from run_access_timing import stats
from test_micro_timing import check_global_guard
from fa_headroom_profiles import leaves
from sweep_mlp_study import epochs
from qwen_projection_reference import generate,validate
from run_l3_kernel_tuning import resources
from qwen_projection_model import BASE,BASELINE,describe

OUT=BASE/'experiment_v2'
OLD=ROOT/'results/ai_workloads/swiglu_model_pilot_v4_20260921'
PLATFORM=ROOT/'results/cacheflex_l3_spm/lifecycle_20260928/experiment_v4'
CONTROL=PLATFORM/'runs/t4096_cache'
WAIT=ROOT/'results/cacheflex_l3_spm/qwen_analytical_20260928/coordinator.json'
ENV=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib',
         OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')


def command(path,guest,inp,c,repeats):
    args=list(map(str,[0 if c['op']=='qkv' else 1,c['t'],c['mp'],c['kc'],repeats]))
    cmd=json.loads((CONTROL/'command.json').read_text());cmd[1]='--outdir='+str(path)
    cmd[cmd.index('--cmd')+1]=str(guest);cmd[cmd.index('--options')+1]=' '.join(args)
    for i,x in enumerate(cmd):
        if x.startswith('system.cpu[0].workload[0].cmd='):cmd[i]='system.cpu[0].workload[0].cmd='+repr(['qwen-projection']+args)
        if x.startswith('--l1d_num_spm_ways='):cmd[i]='--l1d_num_spm_ways='+str(6 if c['arm']=='spm' else 0)
    cmd+=['-P',f'system.readfile={str(inp)!r}']
    return cmd,['qwen-projection']+args


def config_audit(path,guest,inp,args,arm):
    def flat(p):return dict(leaves(json.loads((p/'config.json').read_text().replace(str(p),'<OUT>'))))
    a,b=flat(CONTROL),flat(path)
    delta={k:[a.get(k),b.get(k)] for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}
    allowed=r'system.readfile|system.cpu\[\d+\].workload\[0\].(cmd\[\d+\]|executable)|system.ruby.l1_cntrl\d+.cache.num_spm_ways'
    assert all(re.fullmatch(allowed,k) for k in delta),delta
    cfg=json.loads((path/'config.json').read_text());s=cfg['system']
    assert s['cpu'][0]['workload'][0]['cmd']==args
    assert s['cpu'][0]['workload'][0]['executable']==str(guest) and s['readfile']==str(inp)
    for i in range(8):
        assert s['ruby'][f'l1_cntrl{i}']['cache']['num_spm_ways']==(6 if arm=='spm' else 0)
        assert s['ruby'][f'l2_cntrl{i}']['L2cache']['size']==524288
        assert not s['ruby'][f'l2_cntrl{i}']['L2cache']['spm_separate_backing']
    return delta


def prepare():
    assert not (BASE/'prepared_v2.json').exists(),'Already prepared'
    (BASE/'inputs').mkdir(exist_ok=True);(BASE/'reference').mkdir(exist_ok=True)
    model=json.loads((ROOT/'results/cacheflex_l3_spm/qwen_gqa_20260928/model_config.json').read_text())
    assert (model['hidden_size'],model['num_attention_heads'],model['num_key_value_heads'])==(896,14,2)
    manifest=json.loads((PLATFORM/'inputs.json').read_text())
    manifest.update(json.loads((BASE/'guest_v2/frozen.json').read_text()))
    library=BASE/'reference/ordered_gemm_v2.so'
    cpp=ROOT/'benchmarks/ai_workloads/model_swiglu_reference.cpp'
    subprocess.run(['g++','-O3','-march=native','-ffp-contract=off','-shared','-fPIC',str(cpp),'-o',str(library)],check=True)
    for mod in ('run_qwen_projection_v2','qwen_projection_model','qwen_projection_reference','run_access_timing',
                'test_micro_timing','fa_headroom_profiles','sweep_mlp_study','run_l3_kernel_tuning','build_access_timing'):
        p=Path(importlib.import_module(mod).__file__);manifest[str(p)]=sha(p)
    for p in (library,cpp,BASE/'selection.json',ROOT/'docs/experiments/QWEN_PROJECTION_TUNING_V2.md',
              ROOT/'results/cacheflex_l3_spm/qwen_gqa_20260928/model_config.json',CONTROL/'command.json',CONTROL/'config.json'):
        manifest[str(p)]=sha(p)
    for p,h in manifest.items():assert sha(p)==h,('Changed frozen input',p)
    save(BASE/'frozen_v2.json',manifest);save(BASE/'prepared_v2.json',dict(library=str(library),time=time.time()))


def main():
    OUT.mkdir();(OUT/'runs').mkdir()
    state=dict(pid=os.getpid(),stage='preparing',active={},completed=[])
    lock=threading.RLock();launch_lock=threading.Lock();stop=threading.Event()
    frozen=json.loads((BASE/'frozen_v2.json').read_text())
    prepared=json.loads((BASE/'prepared_v2.json').read_text());selection=json.loads((BASE/'selection.json').read_text())
    def status(stage=None):
        with lock:
            if stage:state['stage']=stage
            state['updated']=time.time();save(OUT/'status.json',state)
            lines=['# Qwen projections: accepted results','',f"Stage: {state['stage']}",'',
                   '| Case | First full ROI us | Second call us |', '|---|---:|---:|']
            for r in state['completed']:lines.append(f"| {r['name']} | {r['first_us']:.4f} | {r['repeat_us']:.4f} |")
            lines+=['','First call includes packing; second call is a repeat diagnostic, not a stable warm median.',
                    'Same Qwen FA hardware; synthetic FP16/FP32 operator, not full BF16 inference. Only accepted rows shown.']
            (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    def unchanged():
        for p,h in frozen.items():assert sha(p)==h,('changed frozen input',p)
    inputs={}
    def get_input(t,op):
        key=t,op
        if key not in inputs:
            status(f'reference_{op}_t{t}')
            p=BASE/'inputs'/f'v2_{op}_t{t}.bin'
            generate(p,t,op,Path(prepared['library']))
            for q in (p,p.with_suffix('.npz'),p.with_suffix('.json')):frozen[str(q)]=sha(q)
            save(OUT/'inputs.json',frozen);inputs[key]=p
        return inputs[key]
    def run(c):
        name=f"{c['stage']}_{c['op']}_{c['arm']}_t{c['t']}_mp{c['mp']}_kc{c['kc']}"
        path=OUT/'runs'/name;path.mkdir();proc=None;start=time.monotonic()
        inp=Path(c['inp']);guest=BASE/'guest_v2'/c['arm'];repeats=2
        try:
            assert not stop.is_set();unchanged()
            cmd,args=command(path,guest,inp,c,repeats)
            save(path/'command.json',cmd);save(path/'case.json',c)
            with (path/'run.log').open('w') as log:
                with launch_lock:
                    while True:
                        assert not stop.is_set();r=resources();save(path/'resource_launch.json',r)
                        if r['cpu_idle_pct']>=50 and r['available_gib']>=128 and r['free_disk_gib']>=100 and len(r['simulators'])<6:break
                        with lock:state['resource_wait']=dict(job=name,resources=r);status()
                        time.sleep(10)
                    proc=subprocess.Popen(['nice','-n','10',*cmd],cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT)
                    start=time.monotonic()
                    with lock:state.pop('resource_wait',None);state['active'][name]=dict(pid=proc.pid,start=time.time());status()
                while proc.poll() is None:
                    assert not stop.is_set(),'peer failed'
                    assert time.monotonic()-start<96*3600,'96h simulation timeout'
                    assert shutil.disk_usage(OUT).free>75*2**30,'disk guard'
                    mem=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))
                    assert mem>64*2**20,'memory guard'
                    assert (path/'run.log').stat().st_size<128*2**20,'log guard'
                    try:proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:status()
            text=(path/'run.log').read_text()
            assert proc.returncode==0 and 'because exiting with last active thread context' in text,text[-3000:]
            assert not re.search(r'panic:|fatal:|Simulated exit code not 0|\bFAIL\b',text),text[-3000:]
            delta=config_audit(path,guest,inp,args,c['arm'])
            checks=[dict(re.findall(r'(\w+)=([^\s]+)',l)) for l in text.splitlines() if l.startswith('PROJ_RESULT')]
            assert len(checks)==repeats and all(x['errors']==x['nonfinite']=='0' for x in checks)
            assert len({x['hash'] for x in checks})==1
            t,op=c['t'],c['op'];data=np.fromfile(path/'output.bin',dtype='<f2')
            assert data.size==t*(1152 if op=='qkv' else 896) and np.isfinite(data).all()
            h=1469598103934665603
            for x in data.view(np.uint16):h=((h^int(x))*1099511628211)&((1<<64)-1)
            assert f'{h:016x}'==checks[-1]['hash'],'export bits differ from guest output'
            numeric=validate(path/'output.bin',inp,t,op)
            structure=describe(t,op,{k:c[k] for k in ('mp','kc')})
            timing=epochs(path,repeats);copies=structure['totals']['copies'] if c['arm']=='spm' else 0
            assert all(e['copies_issued']==e['copies_completed']==copies for e in timing),(copies,timing)
            for ep in stats(path):
                check_global_guard(ep)
                for i in range(8):
                    for domain in ('regular','spm'):
                        for kind,cap in [('Load',32),('Store',48)]:assert ep[f'system.cpu{i}.{domain}Lsq0.boundedMax{kind}Occupancy']<=cap
                    for suffix in ('reductionAccepted','reductionCompleted'):
                        assert ep[f'system.ruby.l0_cntrl{i}.sequencer.{suffix}']==0
            output_hash=sha(path/'output.bin')
            if c.get('old'):assert output_hash==sha(Path(c['old'])/'output.bin'),'compatibility output changed'
            unchanged()
            row=dict(name=name,case=c,path=str(path),epochs=timing,first_us=timing[0]['us'],repeat_us=timing[1]['us'],
                     checks=checks,math=numeric,full_reference_exact=True,output_sha256=output_hash,
                     copies=copies,structure=structure,config_delta=delta,host_seconds=time.monotonic()-start)
            with lock:
                peers=[r for r in state['completed'] if all(r['case'][k]==c[k] for k in ('t','op'))]
                assert all(r['output_sha256']==output_hash for r in peers),'partition/KC/backend changed output bits'
                save(path/'accepted.json',row);state['completed'].append(row);status()
            return row
        except BaseException as exc:
            stop.set();save(path/'failed.json',dict(error=repr(exc),traceback=traceback.format_exc()));raise
        finally:
            if proc and proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
            with lock:state['active'].pop(name,None);status()
    def phase(name,cases,workers):
        status(name)
        with cf.ThreadPoolExecutor(max_workers=workers) as pool:list(pool.map(run,[dict(c,stage=name) for c in cases]))
    def cases(t,selected):
        result=[]
        for op,arms in selected.items():
            inp=get_input(t,op)
            for arm,cs in arms.items():
                for c in cs:result.append(dict(c,arm=arm,t=t,op=op,inp=str(inp)))
        return result
    def record(message):
        with (ROOT/'docs/experiments/LEDGER.md').open('a') as f:
            f.write('\n**AI-11 '+message+'** '+str(len(state['completed']))+' accepted cases. Full ordered bits, float64 reference, peer hashes, config/count/resource checks. Report: results/ai_workloads/qwen_projection_20260928/experiment_v2/REPORT.md.\n')
    try:
        status();unchanged();save(OUT/'inputs.json',frozen);save(OUT/'resource_preflight.json',resources())
        baseline={op:{arm:[BASELINE] for arm in ('cache','spm')} for op in ('qkv','out')}
        save(OUT/'plan.json',dict(gates='4 T25 tail gates +16 T24 selected cases',screen='16 T1024 cases',targets=[2048,4096],workers=1,max_total_simulators=6))
        phase('tail_gates',cases(25,baseline),1)
        phase('shape_gates',cases(24,selection['selected']),1)
        save(OUT/'gates_passed.json',dict(cases=len(state['completed']),full_math=True,exact_output=True,config=True,resources=True))
        record('gates accepted; targets follow.')
        phase('screen_t1024',cases(1024,selection['selected']),1)
        winners={}
        for op in ('qkv','out'):
            winners[op]={}
            for arm in ('cache','spm'):
                rows=[r for r in state['completed'] if r['case']['stage']=='screen_t1024' and r['case']['arm']==arm and r['case']['op']==op]
                winner=min(rows,key=lambda r:r['first_us']);c={k:winner['case'][k] for k in ('mp','kc')}
                winners[op][arm]=[BASELINE]+([c] if c!=BASELINE else [])
        save(OUT/'transfer_selection.json',dict(objective='first full invocation us',selected=winners))
        for t in (2048,4096):phase(f'target_t{t}',cases(t,winners),1)
        effects=[]
        for t in (1024,2048,4096):
            for op in ('qkv','out'):
                best={arm:min((r for r in state['completed'] if r['case']['t']==t and r['case']['op']==op and r['case']['arm']==arm),key=lambda r:r['first_us']) for arm in ('cache','spm')}
                effects.append(dict(t=t,op=op,cache=best['cache']['name'],spm=best['spm']['name'],cache_us=best['cache']['first_us'],spm_us=best['spm']['first_us'],speedup=best['cache']['first_us']/best['spm']['first_us']))
        save(OUT/'effects.json',effects);save(OUT/'complete.json',dict(cases=len(state['completed']),effects=effects))
        status('completed_validated');record('completed.')
        with (ROOT/'docs/experiments/LEDGER.md').open('a') as f:f.write('AI-11 accepted effects: '+json.dumps(effects)+'\n')
    except BaseException as exc:
        state['error']=repr(exc);status('failed');record('stopped with accepted subset.');raise


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    else:main()
