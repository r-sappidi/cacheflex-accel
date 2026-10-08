"""Bounded cache counterpart to L3-07, without changing its frozen queue."""
import ast
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from build_access_timing import save,sha
from qwen_analytical_model import ROOT,describe,case

BASE=ROOT/'results/cacheflex_l3_spm/qwen_cache_analytical_20260928'
PRIOR=ROOT/'results/cacheflex_l3_spm/qwen_analytical_20260928'
SOURCE=ROOT/'results/cacheflex_l3_spm/qwen_gqa_20260928/experiment_v2'


def features(c):
    m=describe(c)
    # Logical K/V span per query group, not actual cache loads or DRAM bytes.
    grouped_span=describe(dict(c,arm='private'))['max_core_staged_bytes']*2
    return dict(compute=m['max_core_pairs'],updates=m['max_core_updates'],
                tasks=m['max_core_tasks'],span=grouped_span,
                pressure=max(0,m['l1_reuse_pressure']-1),structure=m)


def select():
    baseline=features(case(2048,'cache',256,512));pool=[];deferred=[]
    for r,bc in itertools.product((64,128,256,512),(128,256,512,1024)):
        c=case(2048,'cache',r,bc);x=features(c);m=x['structure']
        reason=None
        if r==64:reason='same first-round R64 staging-budget deferral as L3-07'
        elif bc==1024:reason='same first-round large-tile deferral as L3-07; L3-05 regression'
        elif m['imbalance']>1.15:reason='same >15% busiest-core imbalance screen as L3-07'
        elif m['coherent_task_reuse_bytes']>m['coherent_l2_capacity_bytes']:reason='aggregate coherent footprint exceeds L2'
        row=dict(case=c,features=x)
        if reason:row['reason']=reason;deferred.append(row)
        else:pool.append(row)
    scenarios=list(itertools.product((0,.02,.1,.25),(.01,.05,.15,.3),(0,.005,.02),(0,.05,.15)))
    scores=[[] for _ in pool];ranks=[[] for _ in pool]
    for span,update,tasks,pressure in scenarios:
        current=[]
        for i,row in enumerate(pool):
            x=row['features']
            score=((1-span-update-tasks)*x['compute']/baseline['compute']+
                   span*x['span']/baseline['span']+update*x['updates']/baseline['updates']+
                   tasks*x['tasks']/baseline['tasks']+pressure*(x['pressure']-baseline['pressure']))
            scores[i].append(score);current.append(score)
        for i,s in enumerate(current):ranks[i].append(1+sum(v<s-1e-12 for v in current))
    for i,row in enumerate(pool):row.update(mean_rank=sum(ranks[i])/len(scenarios),score_range=[min(scores[i]),max(scores[i])])
    pool.sort(key=lambda x:(x['mean_rank'],x['case']['r'],x['case']['bc']))
    control=case(2048,'cache',256,512)
    best=next(x['case'] for x in pool if x['case']!=control)
    # Explicitly include the R128/Bc256 geometry which improved both SPM paths.
    diagnostic=case(2048,'cache',128,256)
    if best==diagnostic:
        diagnostic=next(x['case'] for x in pool if x['case']['r']==256 and x['case']['bc']!=512)
    selected=[best,diagnostic]
    result=dict(ledger=['FA-06','FA-29','L3-05','L3-06','L3-07','L3-08'],selected=selected,
                rankings=pool,deferred=deferred,baseline=control,scenarios=len(scenarios),
                method='Same bounded geometry screens and 144 sensitivity scenarios as L3-07, evaluated with cache coherent K/V footprints; no fitted latency.',
                budget='Two candidate geometries + inherited control per backend; reuse exact existing cases; at most two T1024, two T2048, one winning T4096 transfer.',
                max_new_simulators=1,wait='L3-07 completion; existing FA two + FFN one + older three already occupy six slots.',
                caveat='Logical grouped operand span is not cache misses/DRAM traffic. Bounded best-measured comparison, not global optimality.')
    save(BASE/'selection.json',result);save(BASE/'model.json',result)
    return result


PHASES='''        selected=json.loads((BASE/'selection.json').read_text())['selected']
        def already(c):
            return any(all(r['case'][k]==c[k] for k in ('t','arm','r','bc','br','hq','hkv','causal','depth')) for r in baseline_rows)
        gates=[dict(c,t=1024) for c in selected if not already(dict(c,t=1024))]
        save(OUT/'reused_cases.json',[r['name'] for r in baseline_rows if any(all(r['case'][k]==dict(c,t=1024)[k] for k in ('t','arm','r','bc','br','hq','hkv','causal','depth')) for c in selected)])
        if gates:phase('cache_t1024_gates',gates)
        save(OUT/'gates_passed.json',dict(new_cases=len(gates),reused_baseline_cases=len(baseline_rows),full_math=True,exact_output=True))
        phase('cache_t2048_screen',[c for c in selected if not already(c)])
        candidates=[r for r in state['completed'] if r['case']['t']==2048 and r['case']['hq']==14 and r['case']['arm']=='cache' and r['case']['depth']==4]
        winner=min(candidates,key=lambda r:r['roi_us'])
        baseline=min((r for r in baseline_rows if r['case']['t']==2048 and r['case']['arm']=='cache'),key=lambda r:r['roi_us'])
        transfers=[dict(winner['case'],t=4096)] if winner['roi_us']<baseline['roi_us'] and not already(dict(winner['case'],t=4096)) else []
        save(OUT/'transfer_decision.json',dict(selected=transfers,reason='Same measured-improvement transfer rule as private/L3; inherited controls remain eligible winners'))
        if transfers:phase('cache_t4096_transfer',transfers)
'''


def prepare():
    src=PRIOR/'selected_runner.py';s=src.read_text();original=s
    def change(a,b):
        nonlocal s
        assert s.count(a)==1,a;s=s.replace(a,b)
    change("BASE=ROOT/'results/cacheflex_l3_spm/qwen_analytical_20260928'", "BASE=ROOT/'results/cacheflex_l3_spm/qwen_cache_analytical_20260928'\nTUNED=ROOT/'results/cacheflex_l3_spm/qwen_analytical_20260928/experiment_v1'")
    change("baseline_rows=json.loads((SOURCE/'status.json').read_text())['completed']", "baseline_rows=json.loads((TUNED/'status.json').read_text())['completed']")
    change('Same frozen guest/model; two model-selected geometries, private and L3 only.', 'Cache tuning added to completed private/L3 search; same frozen guest and hardware.')
    change('Inherited cache controls retained. Rows include reused baseline results.', 'All three backends have bounded analytical candidate searches; rows include accepted reused controls.')
    change('Original R256 controls remain eligible winners. This is a bounded search, not global optimality.', 'Inherited controls remain eligible winners. Compare best accepted cache/private/L3; not global optimality.')
    change("ROOT/'docs/experiments/QWEN_ANALYTICAL_TUNING.md']:", "ROOT/'docs/experiments/QWEN_ANALYTICAL_TUNING.md',TUNED/'status.json',TUNED/'complete.json',\n                  ROOT/'scripts/qwen_cache_tuning.py',ROOT/'docs/experiments/QWEN_CACHE_TUNING.md']:")
    start=s.index("        selected=json.loads((BASE/'selection.json').read_text())['selected']")
    end=s.index("        status('completed_validated')",start)
    s=s[:start]+PHASES+s[end:]
    change('with cf.ThreadPoolExecutor(max_workers=2) as pool:', 'with cf.ThreadPoolExecutor(max_workers=1) as pool:')
    change('**L3-07 completion update:** baseline plus selected follow-up, ', '**L3-08 completion update:** all-backend accepted comparison with cache tuning, ')
    before,after=ast.parse(original),ast.parse(s)
    def fn(tree,name):return next(x for x in ast.walk(tree) if isinstance(x,ast.FunctionDef) and x.name==name)
    checked={}
    for name in ('run','work','get_reference'):
        checked[name]=ast.dump(fn(before,name))==ast.dump(fn(after,name));assert checked[name],name
    phase=fn(after,'phase')
    for node in ast.walk(phase):
        if isinstance(node,ast.keyword) and node.arg=='max_workers':node.value=ast.Constant(value=2)
    assert ast.dump(fn(before,'phase'))==ast.dump(phase),'Other phase acceptance logic changed'
    dest=BASE/'selected_runner.py';dest.write_text(s)
    save(BASE/'acceptance_audit.json',dict(source=str(src),source_sha256=sha(src),derived=str(dest),derived_sha256=sha(dest),
         identical_functions=checked,phase_only_change='max_workers 2 -> 1; exact output checks unchanged'))
    return dest


def launch():
    def status(stage,**kw):save(BASE/'coordinator.json',dict(pid=os.getpid(),stage=stage,updated=time.time(),**kw))
    try:
        status('queued_waiting_for_spm_l3_tuning')
        while True:
            s=json.loads((PRIOR/'coordinator.json').read_text())
            assert s['stage'] not in ('failed','blocked_by_original_failure'),('predecessor failed',s)
            if s['stage']=='completed':
                assert (PRIOR/'experiment_v1/complete.json').exists();break
            time.sleep(10)
        assert not (BASE/'experiment_v1').exists(),'Do not duplicate simulation launch'
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT/'scripts'))
        with (BASE/'selected_runner.log').open('x') as log:
            p=subprocess.Popen([sys.executable,str(BASE/'selected_runner.py')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            status('cache_tuning_running',child=p.pid);rc=p.wait()
        status('completed' if rc==0 else 'failed',returncode=rc)
    except BaseException as exc:
        status('failed',error=repr(exc),traceback=traceback.format_exc());raise


if __name__=='__main__':
    if '--prepare' in sys.argv:
        BASE.mkdir();result=select();prepare();print(json.dumps(result['selected'],indent=2))
    else:launch()
