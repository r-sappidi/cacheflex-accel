"""Wait for validated finite-model pilots, then tune using ONLY their timings."""
import importlib.util,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'benchmarks/cacheflex_flash_attention/model_tuning'
sys.path[:0]=[str(HERE),str(ROOT/'scripts')]
from build_access_timing import save,sha
from model import label,select

def finite_rows(ex):
    state=json.loads((ex/'status.json').read_text())
    assert state['stage']=='completed_requires_tuning_review',state['stage']
    for gate in ('default_equivalence.json','guard_equivalence.json','gates_passed.json'):
        assert (ex/gate).exists(),gate
    rows=[]
    for x in state['completed']:
        c=x['case']
        if c['t'] not in (4096,16384):continue
        assert c['finite'] and not c['small'] and c['guard']==1024
        case=dict(t=c['t'],br=c['br'],bc=c['bc'],r=c['r'],family=('kv_' if c['kv'] else 'q_')+('spm' if c['spm'] else 'cache'))
        rows.append(dict(case=case,roi_us=x['roi_us'],source=x['data_path'],packer='qpack'))
    assert len(rows)==8 and len({label(x['case']) for x in rows})==8
    return rows

def main():
    ex=ROOT/'results/cacheflex_flash_attention/finite_model_20260926'/os.environ['CACHEFLEX_FINITE_SOURCE_EXPERIMENT']
    base=ex.parent/('analytical_'+ex.name);base.mkdir()
    def status(stage,**kw):save(base/'coordinator.json',dict(stage=stage,pid=os.getpid(),updated=time.time(),source=str(ex),**kw))
    status('waiting_for_finite_validation_and_pilots')
    while True:
        if (ex/'status.json').exists():
            s=json.loads((ex/'status.json').read_text())
            if s['stage']=='failed':status('blocked_by_validation_failure',error=s.get('error'));return
            if s['stage']=='completed_requires_tuning_review':break
        time.sleep(10)
    rows=finite_rows(ex);save(base/'finite_only_seed_observations.json',rows)
    # Import the existing equal-budget, arithmetic-gated search engine without
    # editing the running historical cache/llama queues or their dependencies.
    spec=importlib.util.spec_from_file_location('finite_tuning_engine',HERE/'run.py')
    engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)
    engine.BASE=base;engine.OUT=base/'experiment';engine.KV=ex
    engine.CONTROL=ex/'runs/t4096_kv0_spm0'
    engine.GUEST=ROOT/'results/cacheflex_flash_attention/model_tuning_20260926/build/guest.export-safe'
    engine.observations=lambda:list(rows)
    seed_seen={label(r['case']) for r in rows}
    def finite_select(t,data,seen,**kw):
        plan=select(t,data,set(seen)|seed_seen,**kw)
        plan['calibration']['profile']=str(ex)
        plan['calibration']['limitations']='Sparse finite-profile measurements with one common packer; ranking model, not causal latency decomposition'
        return plan
    engine.select=finite_select
    (base/'build').mkdir()
    deps=[Path(__file__),engine.GUEST,ex/'status.json',ex/'gates_passed.json',
          ex/'guard_equivalence.json',base/'finite_only_seed_observations.json']
    save(base/'build/frozen.json',{str(p):sha(p) for p in deps})
    status('analytical_search_running')
    try:engine.main()
    except BaseException as e:status('failed',error=repr(e));raise
    status('completed_requires_review')

if __name__=='__main__':main()
