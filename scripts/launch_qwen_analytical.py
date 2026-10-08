"""Preserve the running baseline, then launch a bounded model-selected follow-up."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from build_access_timing import save,sha
from qwen_analytical_model import ROOT,SOURCE,OUT,analyze


STATUS_FUNCTION='''    def status(stage=None):
        with lock:
            if stage:state['stage']=stage
            state['updated']=time.time();save(OUT/'status.json',state)
            lines=['# Analytical Qwen tuning: accepted baseline and new results','',
                   f"Stage: {state['stage']}",'',
                   'Same frozen guest/model; two model-selected geometries, private and L3 only.',
                   'Inherited cache controls retained. Rows include reused baseline results.',
                   '', '| Case | R | Br | Bc | Complete ROI us |', '|---|---:|---:|---:|---:|']
            for x in state['completed']:
                c=x['case'];lines.append(f"| {x['name']} | {c['r']} | {c['br']} | {c['bc']} | {x['roi_us']:.4f} |")
            effects=[]
            for t in (1024,2048,4096):
                best={}
                for arm in ('cache','private','l3'):
                    xs=[x for x in state['completed'] if x['case']['t']==t and x['case']['hq']==14 and x['case']['depth']==4 and x['case']['arm']==arm]
                    if xs:best[arm]=min(xs,key=lambda x:x['roi_us'])
                if 'l3' in best:
                    for arm in ('cache','private'):
                        if arm in best:
                            a,b=best[arm]['roi_us'],best['l3']['roi_us']
                            effects.append(dict(t=t,control=arm,control_us=a,l3_us=b,latency_reduction_pct=100*(1-b/a),speedup=a/b))
                            lines.append(f"\\nT{t} best accepted L3 versus {arm}: {100*(1-b/a):.3f}% lower latency, {a/b:.6f}x.")
            lines+=['','Model sensitivity scores are NOT predicted latency. Only accepted timings above.',
                    'All new rows require full independent arithmetic, same-Bc exact bits, counts/config/resource checks.',
                    'Original R256 controls remain eligible winners. This is a bounded search, not global optimality.']
            (OUT/'REPORT.md').write_text('\\n'.join(lines)+'\\n');save(OUT/'effects.json',effects)
'''

PHASES='''        selected=json.loads((BASE/'selection.json').read_text())['selected']
        phase('selected_t1024_gates',[dict(c,t=1024) for c in selected])
        save(OUT/'gates_passed.json',dict(new_cases=4,reused_baseline_cases=len(baseline_rows),full_math=True,exact_output=True))
        phase('selected_t2048_screen',selected)
        transfers=[]
        for arm in ('private','l3'):
            candidates=[x for x in state['completed'] if x['case']['t']==2048 and x['case']['hq']==14 and x['case']['arm']==arm and x['case']['depth']==4]
            winner=min(candidates,key=lambda x:x['roi_us'])
            baseline=min((x for x in baseline_rows if x['case']['t']==2048 and x['case']['arm']==arm),key=lambda x:x['roi_us'])
            if winner['roi_us']<baseline['roi_us']:
                transfers.append(dict(winner['case'],t=4096))
        save(OUT/'transfer_decision.json',dict(selected=transfers,reason='Only measured T2048 improvements transfer; original T4096 controls reused'))
        if transfers:phase('selected_t4096_transfer',transfers)
'''


def prepare():
    src=ROOT/'scripts/run_qwen_l3_attention.py';s=src.read_text()
    def change(a,b):
        nonlocal s
        assert s.count(a)==1,a;s=s.replace(a,b)
    change("BASE=ROOT/'results/cacheflex_l3_spm/qwen_gqa_20260928'", "BASE=ROOT/'results/cacheflex_l3_spm/qwen_analytical_20260928'\nSOURCE=ROOT/'results/cacheflex_l3_spm/qwen_gqa_20260928/experiment_v2'")
    change("OUT=BASE/os.environ.get('CACHEFLEX_QWEN_EXPERIMENT','experiment_v2')", "OUT=BASE/'experiment_v1'")
    change("GUEST=BASE/os.environ.get('CACHEFLEX_QWEN_GUEST','guest_v4')", "GUEST=ROOT/'results/cacheflex_l3_spm/qwen_gqa_20260928/guest_v4'")
    change("return f\"t{c['t']}_q{c['hq']}_kv{c['hkv']}_causal{int(c['causal'])}_{c['arm']}_bc{c['bc']}_d{c['depth']}\"",
           "return f\"t{c['t']}_q{c['hq']}_kv{c['hkv']}_causal{int(c['causal'])}_{c['arm']}_br{c['br']}_bc{c['bc']}_r{c['r']}_d{c['depth']}\"")
    change("state=dict(pid=os.getpid(),stage='preparing',active={},completed=[])",
           "baseline_rows=json.loads((SOURCE/'status.json').read_text())['completed']\n    state=dict(pid=os.getpid(),stage='preparing',active={},completed=list(baseline_rows))")
    start=s.index('    def status(stage=None):');end=s.index('\n    status()\n    try:',start)
    s=s[:start]+STATUS_FUNCTION+s[end:]
    change("BASE/'model_config.json'", "SOURCE.parent/'model_config.json'")
    anchor="        def unchanged():"
    insert="""        for p in [SOURCE/'status.json',SOURCE/'complete.json',BASE/'selection.json',BASE/'model.json',
                  ROOT/'scripts/qwen_analytical_model.py',ROOT/'scripts/launch_qwen_analytical.py',
                  ROOT/'docs/experiments/QWEN_ANALYTICAL_TUNING.md']:
            frozen[str(p)]=sha(p)
        for row in baseline_rows:
            for name in ('accepted.json','output.fp16'):
                p=Path(row['path'])/name;frozen[str(p)]=sha(p)
"""
    change(anchor,insert+anchor)
    start=s.index("        save(OUT/'plan.json',dict(");end=s.index('        references={}',start)
    s=s[:start]+"        save(OUT/'plan.json',json.loads((BASE/'selection.json').read_text()))\n"+s[end:]
    change('            expected=reference(*k,progress=progress);np.save(dest,expected)', '''            old=SOURCE/'references'/dest.name
            if old.exists():
                assert sha(old)==json.loads(old.with_suffix('.json').read_text())['sha256']
                shutil.copy2(old,dest);expected=np.load(dest)
            else:
                expected=reference(*k,progress=progress);np.save(dest,expected)''')
    start=s.index("        phase('one_head_compatibility'");end=s.index("        status('completed_validated')",start)
    s=s[:start]+PHASES+s[end:]
    change('**L3-06 completion update:** all ', '**L3-07 completion update:** baseline plus selected follow-up, ')
    ast.parse(s)
    out=OUT/'selected_runner.py';out.write_text(s)
    (OUT/'runner_derivation.json').write_text(json.dumps(dict(source=str(src),source_sha256=sha(src),
       derived=str(out),derived_sha256=sha(out),changes=['isolated output paths','reuse baseline/reference evidence',
       'model-selected phase list','geometry-aware labels','best accepted reporting'],
       acceptance='Original numerical/config/count/finite-resource assertions retained'),indent=2))
    return out


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    def state(stage,**kw):save(OUT/'coordinator.json',dict(pid=os.getpid(),stage=stage,updated=time.time(),**kw))
    try:
        state('waiting_for_original_qwen_targets')
        while True:
            d=json.loads((SOURCE/'status.json').read_text())
            if d['stage']=='failed':state('blocked_by_original_failure',error=d.get('error'));return
            live=Path('/proc/'+str(d['pid'])+'/cmdline')
            try:active=b'run_qwen_l3_attention.py' in live.read_bytes()
            except FileNotFoundError:active=False
            if d['stage']=='completed_validated' and (SOURCE/'complete.json').exists() and not active:break
            time.sleep(10)
        assert (SOURCE/'gates_passed.json').exists()
        analyze();runner=prepare()
        assert not (OUT/'experiment_v1').exists(),'Do not duplicate launch'
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        with (OUT/'selected_runner.log').open('w') as log:
            p=subprocess.Popen([sys.executable,str(runner)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            state('selected_followup_running',child=p.pid)
            rc=p.wait()
        if rc==0:
            accepted=json.loads((OUT/'experiment_v1/status.json').read_text())['completed']
            original=json.loads((SOURCE/'status.json').read_text())['completed']
            model=json.loads((OUT/'model.json').read_text());checks=[]
            for row in accepted:
                c=row['case']
                if c['t']!=2048 or c['arm'] not in ('private','l3') or str(OUT) not in row['path']:continue
                baseline=next(x for x in original if x['case']['t']==2048 and x['case']['arm']==c['arm'])
                ranking=next(x for x in model['rankings']['2048'][c['arm']]['ranked'] if all(x['case'][k]==c[k] for k in ('br','bc','r')))
                checks.append(dict(case=c,measured_latency_ratio=row['roi_us']/baseline['roi_us'],
                    sensitivity_score_range=ranking['score_range'],mean_rank=ranking['mean_rank'],
                    note='Dimensionless ranking scenarios are not calibrated latency predictions'))
            save(OUT/'measurement_feedback.json',checks)
        state('completed' if rc==0 else 'failed',returncode=rc)
    except BaseException as exc:
        state('failed',error=repr(exc),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
