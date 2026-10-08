"""L3-12: bounded adaptive R/Bc neighbor search, identical acceptance."""
import ast
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path('/data2/rsappidi/cacheflex_mc')
sys.path.insert(0,str(ROOT/'scripts'))
from build_access_timing import save,sha
from qwen_analytical_model import describe,case
from qwen_t4096_tuning import fn,run_parts

BASE=ROOT/'results/cacheflex_l3_spm/qwen_neighbors_20260929'
PRIOR=ROOT/'results/cacheflex_l3_spm/qwen_t4096_tuning_20260929'
RECOVERY=ROOT/'results/parallel_memory_recovery_20260929'
ARMS=('cache','private','l3')
RS=(64,128,256,512)
BCS=(128,256,512,1024)


def read(p):return json.loads(Path(p).read_text())


def geometry(c):return c['r'],c['bc']


def frontier(rows):
    """Same axis-neighbor rule and analytic screens for every backend."""
    winners={a:min((x for x in rows if x['case']['t']==4096 and x['case']['arm']==a
                    and x['case']['hq']==14 and x['case']['depth']==4),key=lambda x:x['roi_us']) for a in ARMS}
    needed=set();rejected=[]
    for a,w in winners.items():
        r,bc=geometry(w['case']);ri=RS.index(r);bi=BCS.index(bc)
        candidates=[(RS[i],bc) for i in (ri-1,ri+1) if 0<=i<len(RS)]
        candidates += [(r,BCS[i]) for i in (bi-1,bi+1) if 0<=i<len(BCS)]
        for g in candidates:
            if any(x['case']['t']==4096 and x['case']['arm']==a and geometry(x['case'])==g for x in rows):continue
            models={arm:describe(case(4096,arm,*g)) for arm in ARMS}
            reasons=[]
            for arm,m in models.items():
                if not m['valid']:reasons.append(arm+':invalid checked geometry')
                elif m['imbalance']>1.15:reasons.append(arm+':>15% busiest-core imbalance')
                elif m['coherent_task_reuse_bytes']>m['coherent_l2_capacity_bytes']:reasons.append(arm+':coherent reuse footprint exceeds remaining L2')
            if reasons:rejected.append(dict(geometry=g,trigger=a,reasons=reasons,models=models))
            else:needed.add(g)
    return sorted(needed),dict(winners={a:dict(name=w['name'],roi_us=w['roi_us'],case=w['case']) for a,w in winners.items()},rejected=rejected,
        scope='axis-local optimum only within the declared analytically screened R/Bc grid; Br fixed16; no global-optimum claim')


def merge_final_prior(rows,frozen,out):
    state=read(PRIOR/'experiment_v1/status.json')
    assert state['stage']!='failed',state.get('error')
    if state['stage']!='completed_validated':return False
    known={x['name'] for x in rows}
    for x in state['completed']:
        if x['name'] not in known:
            for n in ('accepted.json','output.fp16'):
                p=Path(x['path'])/n;frozen[str(p)]=sha(p)
            rows.append(x)
    save(out/'inputs.json',frozen)
    save(out/'prior_completion_import.json',dict(source=str(PRIOR/'experiment_v1/status.json'),source_sha256=sha(PRIOR/'experiment_v1/status.json'),cases=len(state['completed'])))
    return True


def prepare():
    assert not BASE.exists(),'Do not duplicate neighbor queue'
    state=read(PRIOR/'experiment_v1/status.json')
    assert state['stage'] in ('t4096_equal_candidate_completion','completed_validated')
    for p,h in read(PRIOR/'experiment_v1/inputs.json').items():assert sha(p)==h,('changed input',p)
    initial=[(512,256),(256,128)]
    for g in initial:
        for a in ARMS:
            c=case(4096,a,*g);m=describe(c)
            assert m['valid'] and m['imbalance']<=1.15 and m['coherent_task_reuse_bytes']<=m['coherent_l2_capacity_bytes']
            assert not any(x['case']==c for x in state['completed']),('duplicate',c)
    BASE.mkdir();snap=BASE/'baseline_snapshot';snap.mkdir()
    save(snap/'status.json',dict(stage='immutable_accepted_subset_snapshot',completed=state['completed'],source=str(PRIOR/'experiment_v1/status.json')))
    save(snap/'complete.json',dict(snapshot_complete=True,source_experiment_complete=state['stage']=='completed_validated'))
    save(snap/'inputs.json',read(PRIOR/'experiment_v1/inputs.json'))
    plan=dict(ledger='L3-12',related=['FA-29','L3-07','L3-08','L3-11'],initial=initial,max_new_geometries=6,
        arms=ARMS,grid=dict(r=RS,bc=BCS,br=[16]),gates_t=1024,target_t=4096,
        selection='union of unmeasured axis neighbors of each backend winner; same model screens; measure each chosen geometry on all3arms',
        stopping='stop when no unmeasured admissible axis neighbors remain; if6-new-geometry budget exhausted publish unresolved frontier, not convergence',
        finite_limits='same guest_v4/build_v5,NoC L3 FP16+init; original full checks',max_parallel=2,
        reserved_bytes=18*2**30,initial_models=[dict(case=case(4096,a,*g),model=describe(case(4096,a,*g))) for g in initial for a in ARMS])
    save(BASE/'selection.json',plan);save(BASE/'model.json',plan)
    original=(PRIOR/'selected_runner.py').read_text();s=original
    def change(a,b):
        nonlocal s
        assert s.count(a)==1,a
        s=s.replace(a,b)
    change("BASE=ROOT/'results/cacheflex_l3_spm/qwen_t4096_tuning_20260929'", "BASE=ROOT/'results/cacheflex_l3_spm/qwen_neighbors_20260929'")
    change("TUNED=ROOT/'results/cacheflex_l3_spm/qwen_cache_analytical_20260928/experiment_v1'", "TUNED=BASE/'baseline_snapshot'")
    change('from qwen_t4096_tuning import BudgetGuard','from qwen_t4096_tuning import BudgetGuard\nfrom qwen_neighbor_tuning import frontier,merge_final_prior,describe')
    change('T4096 equal-candidate completion; same frozen guest and hardware. New five cases plus reused accepted controls.', 'Adaptive target-size R/Bc neighbors; same frozen guest/model. Accepted controls reused.')
    change('All three backends get R128/Bc256, R256/Bc256, R256/Bc512 at Br16. L3 uses the original NoC transport.', 'Same candidate geometry and stopping rule for all3backends; Br16 fixed. L3 uses original NoC transport.')
    change("ROOT/'results/parallel_budget_20260929/budget.json']:", "ROOT/'results/parallel_budget_20260929/budget.json',\n                  ROOT/'scripts/qwen_neighbor_tuning.py',ROOT/'docs/experiments/QWEN_NEIGHBOR_TUNING.md']:")
    change('with cf.ThreadPoolExecutor(max_workers=5) as pool:', 'with cf.ThreadPoolExecutor(max_workers=2) as pool:')
    start=s.index("        selected=json.loads((BASE/'selection.json').read_text())['selected']")
    end=s.index("        status('completed_validated')",start)
    s=s[:start]+'''        plan=json.loads((BASE/'selection.json').read_text())
        visited=[];batch=[tuple(g) for g in plan['initial']];iteration=0
        def measured(c):return any(all(x['case'][k]==c[k] for k in ('t','arm','r','bc','br','hq','hkv','causal','depth')) for x in state['completed'])
        while batch:
            iteration+=1
            candidates=[dict(case(t,a,bc),r=r) for t in (1024,4096) for r,bc in batch for a in ('cache','private','l3')]
            save(OUT/f'iteration_{iteration}_selection.json',dict(geometries=batch,models=[dict(case=c,model=describe(c)) for c in candidates],before_measurement=True))
            gates=[c for c in candidates if c['t']==1024 and not measured(c)]
            if gates:phase(f'neighbor_{iteration}_gates',gates)
            save(OUT/f'iteration_{iteration}_gates_passed.json',dict(geometries=batch,full_original_acceptance=True))
            targets=[c for c in candidates if c['t']==4096 and not measured(c)]
            if targets:phase(f'neighbor_{iteration}_targets',targets)
            visited.extend(batch)
            while not merge_final_prior(state['completed'],frozen,OUT):
                status('waiting_for_final_L3_11_case');time.sleep(10)
            proposed,decision=frontier(state['completed'])
            decision.update(iteration=iteration,visited=visited,proposed=proposed,budget=plan['max_new_geometries'])
            budget=plan['max_new_geometries']-len(visited)
            batch=[g for g in proposed if g not in visited][:max(0,budget)]
            decision.update(next_batch=batch,converged=not proposed,budget_exhausted=bool(proposed) and not batch)
            save(OUT/'neighbor_decision.json',decision)
        save(OUT/'gates_passed.json',dict(iterations=iteration,full_original_acceptance=True))
''' +s[end:]
    change('**L3-11 completion update:** equal three-geometry T4096 comparison, ', '**L3-12 completion update:** bounded adaptive R/Bc neighbor comparison, ')
    a,b=ast.parse(original),ast.parse(s)
    checks={n:ast.dump(fn(a,n))==ast.dump(fn(b,n)) for n in ('work','get_reference','run')}
    assert all(checks.values()),checks
    dest=BASE/'selected_runner.py';compile(s,str(dest),'exec');dest.write_text(s)
    save(BASE/'acceptance_audit.json',dict(source=str(PRIOR/'selected_runner.py'),source_sha256=sha(PRIOR/'selected_runner.py'),
        derived_sha256=sha(dest),identical_functions=checks,max_parallel=2))
    print('Prepared2initial geometries across3backends,6gates+6targets; at most6adaptive geometries total.')


if __name__=='__main__':
    assert sys.argv[1:]==['--prepare'];prepare()
