"""Generate auditable predictions/held-out evaluation and a fail-closed plan.

This is the model-to-runner interface, not a new unattended simulation queue.
Unsupported guests and failed validation cannot silently become executable jobs.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
import re
import resource
import shutil

from qwen_placement_model import Candidate, HW, account, calibrate, candidates, estimate, predict

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'results/cacheflex_flash_attention/qwen_qo_ring_20260930/experiment_v3'
DEFAULT=ROOT/'results/qwen_placement_model_20260930'
LEDGER=['FA-03','FA-10','FA-14','FA-25','FA-26','FA-27','FA-28','FA-29','FA-31','FA-33','L3-11','L3-12']


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')


def from_row(row):
    x=row['case'];m=x['mode']
    return Candidate(t=x['t'],bc=x['bc'],r=x['r'],causal=x['causal'],
                     ownership='query_legacy' if m<2 else 'query_head',
                     placement={0:'cache',1:'v',2:'cache',3:'v',4:'qo',5:'qov'}[m])


def hardware_signature(path):
    config=read(path/'config.json');system=config['system'];cpus=system['cpu'];ruby=system['ruby']
    keys=('cacheLoadPorts','cacheStorePorts','spmLoadPorts','spmStorePorts','commitWidth',
          'issueWidth','numROBEntries','LQEntries','SQEntries','cacheFlexBoundedTracking')
    cpu=[{k:x[k] for k in keys} for x in cpus]
    caches=[]
    for i in range(8):
        x=ruby[f'l1_cntrl{i}']['cache']
        assert int(x['size'])==524288 and x['assoc']==8 and not x['spm_separate_backing']
        assert x['spm_set_stride']==0
        caches.append({k:x[k] for k in ('size','assoc','dataAccessLatency','spm_separate_backing','spm_set_stride')})
    assert len(cpu)==8 and all(x['spmLoadPorts']==1 and x['LQEntries']==32 and x['SQEntries']==48 for x in cpu)
    def normalized(obj):
        if isinstance(obj,list):return [normalized(x) for x in obj]
        if not isinstance(obj,dict):return obj
        return {k:normalized(v) for k,v in obj.items()
                if k not in ('cmd','host_paths','num_spm_ways')}
    # Full config comparison too: exclude only experimental placement eligibility,
    # guest arguments, and per-outdir filesystem redirects. No hidden queue/NoC drift.
    hardware_hash=hashlib.sha256(json.dumps(normalized(config),sort_keys=True).encode()).hexdigest()
    return dict(cpu=cpu,caches=caches,normalized_full_config_sha256=hardware_hash)


def observations(source):
    state=read(source/'status.json');rows=state['completed'];result=[];audit=[];signature=None
    assert state['stage']=='completed_validated', 'require a completed accepted factorial'
    for row in rows:
        path=Path(row['path']);accepted=path/'accepted.json'
        assert read(accepted)==row, ('status/acceptance mismatch',path)
        assert row['arithmetic']['errors']==row['arithmetic']['nonfinite']==row['commands']==0
        sig=hardware_signature(path)
        if signature is None:signature=sig
        assert sig==signature, 'mixed hardware profiles'
        c=from_row(row);m=account(c)
        assert m['totals']['copies']==row['copies'],row['name']
        for k in ('tasks','updates','copies'):
            assert [x[k] for x in m['cores']]==row['work'][k],(row['name'],k)
        audit.append(dict(name=row['name'],source=str(accepted),accepted_sha256=sha(accepted),
                          config_sha256=sha(path/'config.json'),counts_exact=True))
        if c.t>=1024:result.append((c,row['roi_us'],row['name']))
    assert {c.t for c,y,n in result}=={1024,4096}, 'require both calibration and holdout sizes'
    return result,audit,signature


def evaluate(rows,fitted):
    predictions=[]
    for c,y,name in rows:
        p=predict(c,fitted);predictions.append(dict(name=name,case=asdict(c),measured_us=y,
                   **p,relative_error=(p['median_us']/y-1),envelope_covers=p['min_us']<=y<=p['max_us']))
    correct=total=0
    for i,a in enumerate(predictions):
        for b in predictions[i+1:]:
            if abs(a['measured_us']/b['measured_us']-1)<.01:continue
            total+=1
            correct+=int((a['median_us']-b['median_us'])*(a['measured_us']-b['measured_us'])>0)
    head={p['case']['placement']:p for p in predictions if p['case']['ownership']=='query_head'}
    checks={
        'qo_regresses_vs_cache':head['qo']['median_us']>head['cache']['median_us'],
        'adding_v_improves_qo':head['qov']['median_us']<head['qo']['median_us'],
        'qov_loses_to_v':head['qov']['median_us']>head['v']['median_us']}
    mape=statistics.mean(abs(p['relative_error']) for p in predictions)
    agreement=correct/total if total else 1
    # Chosen before evaluation; success is not an optimality certificate.
    passed=mape<=.15 and agreement>=.8 and all(checks.values())
    return dict(predictions=predictions,mape=mape,pairwise_ordering_agreement=agreement,
                pairwise_comparisons=total,directional_checks=checks,passed=passed,
                thresholds=dict(max_mape=.15,min_pairwise_agreement=.8,ignore_measured_pairs_within=.01))


def launchable(c):
    # Only the actual frozen modes; no invented argv for not-yet-built kernels.
    if c.stride or c.sync!='batch' or c.q_adapter!='six_instruction':return None
    if c.ownership=='query_legacy' and c.placement in ('cache','v'):
        return {'runner':'scripts/run_qwen_qo.py','mode':int(c.placement=='v'),'requires_adapter':True}
    if c.ownership=='query_head' and c.placement in ('cache','v','qo','qov'):
        return {'runner':'scripts/run_qwen_qo.py','mode':{'cache':2,'v':3,'qo':4,'qov':5}[c.placement],
                'requires_adapter':True}
    return None


def counter_diagnosis(rows, source):
    """Diagnostic only: holdout counters never enter the fitted predictor."""
    result=[]
    suffixes={'ops':r'system.cpu\d+.commitStats0.numOps',
              'l1d_misses':r'system.ruby.l0_cntrl\d+.Dcache.m_demand_misses',
              'l2_misses':r'system.ruby.l1_cntrl\d+.cache.m_demand_misses',
              'flits':r'system.ruby.network.flits_injected::total',
              'spm_service_cycles':r'system.cpu\d+.spmLsq0.spmLoadServiceCycles',
              'spm_responses':r'system.cpu\d+.spmLsq0.spmLoadResponses',
              'suspended_cycles':r'system.cpu\d+.commit.pcSuspendedCycles'}
    for c,y,name in rows:
        p=source/'runs'/name/'stats.txt';d={}
        for line in p.read_text().split('---------- End Simulation Statistics')[0].splitlines():
            fields=line.split()
            if len(fields)<2:continue
            try:d[fields[0]]=float(fields[1])
            except ValueError:pass
        counters={k:sum(v for n,v in d.items() if re.fullmatch(p,n)) for k,p in suffixes.items()}
        result.append(dict(name=name,case=asdict(c),measured_us=y,counters=counters,
                           stats_sha256=sha(p),used_for_calibration=False))
    return dict(rows=result,interpretation=[
        'Accounting-only capacity/reuse ratios do not determine actual coherent misses.',
        'Inspect placement-dependent miss and suspension changes before attributing timing error to SPM service.',
        'Cannot infer per-array evictions or removable runtime from these totals.'],
        next_measurements=[
            'Trace physical set reuse for packed K/V/Q/O and scratch, preserving exact observer equivalence.',
            'Match Q+V and O+V at T4096 to separate Q adapter from O dependency/claim costs.',
            'Calibrate KV software merge with full-output correctness before promoting a KV timing prediction.'])


def generate(out,source=SOURCE):
    rows,audit,signature=observations(source)
    train=[(c,y) for c,y,n in rows if c.t==1024]
    held=[r for r in rows if r[0].t==4096]
    fitted=calibrate(train);validation=evaluate(held,fitted)
    # Do not refit after holdout. Scores below remain the frozen T1024 model.
    measured={c.key:y for c,y,n in rows}
    ranked=[];rejected=[]
    baseline=min(y for c,y,n in held if c.placement=='v')
    for c in candidates(4096):
        try:m=account(c)
        except ValueError as e:
            rejected.append(dict(case=asdict(c),reason=str(e)));continue
        p=predict(c,fitted)
        calibrated_family=c.ownership!='kv' and c.placement in ('cache','v','qo','qov') and not c.stride
        ranked.append(dict(id=c.key,case=asdict(c),prediction=p,measured_us=measured.get(c.key),
            exact={k:m[k] for k in ('staged_bytes','gemm_flops','padded_pairs','imbalance','merge_payload_bytes','merge_fp32_flops')},
            layout={k:v for k,v in m['layout'].items() if k!='slots'},
            calibration_domain='transfer within observed families' if calibrated_family else 'extrapolation/unvalidated family',
            runner=launchable(c),performance_selection_allowed=validation['passed'] and calibrated_family,
            possible_gain_under_model=p['min_us']<baseline))
    ranked.sort(key=lambda x:x['prediction']['median_us'])
    # Information-first queue explicitly separated from runtime-ranked promotion.
    wanted=[Candidate(placement='qv'),Candidate(placement='ov'),
            Candidate(ownership='kv',placement='cache',lifetime='kv_group'),
            Candidate(ownership='kv',placement='kv',lifetime='kv_group'),
            Candidate(ownership='kv',placement='kv',lifetime='all_heads')]
    plan=[]
    for c in wanted:
        r=next(x for x in ranked if x['id']==c.key)
        kv=c.ownership=='kv'
        plan.append(dict(**r,selection_basis='information gain; not a validated predicted speedup',
            prior_ids=['FA-28','FA-29'] if kv else ['FA-10','FA-25','FA-26','FA-27','FA-33'],
            material_delta='Qwen seven-query-head reuse with resident K+V and software merge' if kv else 'matched Qwen V-resident placement decomposition',
            control_case=asdict(replace(c,placement='cache')) if kv else asdict(replace(c,placement='v')),
            acceptance=['full independent numerical oracle','causal first-row identity','configuration/finite-resource audit',
                        'exact modeled per-core work/copy counts','full ROI includes packing, staging, merge, export',
                        'same-order exact output bits; changed reduction order requires separate documented numerical validation'],
            decision='lower full ROI than matched control and strongest independently tuned private reference',
            state='requires_guest_implementation' if r['runner'] is None else 'requires_isolated_runner_adapter'))
    out.mkdir(parents=True,exist_ok=False)
    model=dict(version=1,created=time.time(),source=str(source),ledger_ids=LEDGER,
        hardware=asdict(HW),hardware_signature=signature,structural_validation=audit,
        calibration=dict(training_names=[n for c,y,n in rows if c.t==1024],heldout_names=[n for c,y,n in held],
                         policy='T256 correctness only; fit T1024, evaluate T4096 once; no holdout refit',
                         ensemble=[dict(training_log_rmse=e,scenario=asdict(s)) for e,s in fitted]),
        heldout=validation,ranked=ranked,rejected=rejected,
        limitations=['No exact coherent miss prediction without address traces.',
                     'Finite issue/window bounds included; no calibrated queue contention model.',
                     'Phase-level dependency graph aggregates tile order; KV merge timing uncalibrated.',
                     'Scenarios are sensitivity envelopes, not guaranteed bounds or confidence intervals.',
                     'Instruction and cache traffic estimates are distinct from exact work/copy counts.',
                     'No new architecture, guest, or simulation is implemented by this model.'])
    save(out/'model.json',model)
    save(out/'experiment_plan.json',dict(schema='cacheflex-placement-plan-v1',auto_launch=False,
        model_validation_passed=validation['passed'],jobs=plan,
        resource_policy=dict(cpu_host_fraction=.20,ram_host_fraction=.20,max_new_simulators=2,
                             admission='existing shared atomic OPS-23 broker; no independent budget'),
        baseline=dict(same_elf_best_private_us=baseline,strongest_prior_private_us=9473.3184,
                      note='prior best is a performance target, not a causal same-ELF control')))
    save(out/'diagnosis.json',counter_diagnosis(rows,source))
    sources=[Path(__file__),ROOT/'scripts/qwen_placement_model.py',ROOT/'scripts/check_qwen_placement_plan.py',
             ROOT/'scripts/test_qwen_placement_model.py']
    (out/'source_snapshot').mkdir()
    for p in sources:shutil.copy2(p,out/'source_snapshot'/p.name)
    save(out/'provenance.json',{str(p):sha(p) for p in sources+[
                             source/'status.json',source.parent/'guest_v2/guest.export-safe']})
    # Text artifact avoids implying a paper-ready report or modifying frozen docs.
    lines=['Qwen placement/ownership analytical model v1',
           f'Exact accounting checks: {len(audit)} accepted cases.',
           f'Held-out T4096 MAPE: {validation["mape"]:.1%}; pair ordering: {validation["pairwise_ordering_agreement"]:.1%}.',
           f'Validation passed: {validation["passed"]}. No holdout data used in calibration.',
           'Predictions (us), measured (us):']
    for p in validation['predictions']:
        lines.append(f'{p["name"]}: {p["median_us"]:.2f} [{p["min_us"]:.2f}, {p["max_us"]:.2f}], measured {p["measured_us"]:.2f}')
    lines+=['','Runtime ranks are exploratory unless validation and family coverage pass.',
            'KV/software merge is uncalibrated even if query-family validation passes.',
            'experiment_plan.json lists information-gain cases and missing implementation gates.',
            'No simulator launched; no active/frozen experiment inputs changed.']
    (out/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    return model


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=DEFAULT);p.add_argument('--source',type=Path,default=SOURCE)
    args=p.parse_args()
    # Bound analytical work to one available CPU. No BLAS/multithreaded fitting.
    allowed=os.sched_getaffinity(0)
    budget=ROOT/'results/parallel_budget_20260929/budget.json'
    if budget.exists():allowed &= set(read(budget)['cpu_set'])
    if not allowed:raise RuntimeError('no CPU available within shared resource policy')
    os.sched_setaffinity(0,{min(allowed)})
    soft,hard=resource.getrlimit(resource.RLIMIT_AS)
    cap=2*1024**3
    if soft!=resource.RLIM_INFINITY:cap=min(cap,soft)
    if hard!=resource.RLIM_INFINITY:cap=min(cap,hard)
    resource.setrlimit(resource.RLIMIT_AS,(cap,hard))
    model=generate(args.out,args.source)
    print((args.out/'REPORT.txt').read_text())


if __name__=='__main__':main()
