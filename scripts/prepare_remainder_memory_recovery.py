"""Prepare AI-12 continuation after OPS-21 drains; reuse all accepted cases."""
import ast
from pathlib import Path
import time
from recover_parallel_memory import ROOT,BASE,REMAINDER,read,save,sha,GIB


def main():
    source=ROOT/'scripts/run_qwen_remainder.py'
    original=source.read_text();s=original
    previous=read(REMAINDER/'status.json')
    assert previous['stage']=='failed' and not previous['active']
    assert 'existing budget queue has a failure' in previous['error']
    assert len(previous['completed'])==59
    def change(a,b):
        nonlocal s
        assert s.count(a)==1,a
        s=s.replace(a,b)
    change("OUT=BASE/'experiment_v1'", "OUT=BASE/'experiment_memory_v2'\nPRIOR=BASE/'experiment_v1'\nRECOVERY=ROOT/'results/parallel_memory_recovery_20260929'")
    start=s.index('def capacity():');end=s.index('\ndef prepare():',start)
    s=s[:start]+'''def capacity():
    budget=read(JOURNAL/'budget.json')
    complete=RECOVERY/'complete.json'
    if not complete.exists():return 0,dict(reason='waiting for preserved queues and two retries to finish')
    assert not read(complete).get('failed'),'recovery queue failed'
    # The stopped original AI-12 record stays intact; this queue owns a new outdir.
    l3=read(ROOT/'results/cacheflex_l3_spm/qwen_t4096_tuning_20260929/experiment_v1/status.json')
    alive=Path('/proc')/str(l3['pid'])
    reserve=(20.5+(34 if alive.exists() else 0))*GIB
    slots=min(12,max(0,(budget['ram_limit']-reserve)//(8*GIB)))
    rss=sum(x['rss'] for x in owned().values())
    if rss>=budget['ram_limit']-4*GIB:slots=0
    return int(slots),dict(reserved_bytes=reserve,simulator_as=8*GIB,available_slots=int(slots))
''' +s[end:]
    change("budget=read(JOURNAL/'budget.json');os.sched_setaffinity(0,budget['cpu_set'])", "budget=dict(read(JOURNAL/'budget.json'),simulator_as=8*GIB,worker_as=2*GIB);os.sched_setaffinity(0,budget['cpu_set'])")
    change("stop=threading.Event();active={};rows=[]", "stop=threading.Event();active={};rows=list(read(PRIOR/'status.json')['completed'])")
    change("state['stage']=label;status();pending=list(cases);futures={}", "state['stage']=label;status();accepted={r['name'] for r in rows};pending=[c for c in cases if name(dict(c,stage=label)) not in accepted];futures={}")
    change("**AI-12 complete:** ","**AI-12 memory-recovery continuation complete:** ")
    # Preserve ALL scientific validation and input generation unchanged.
    a,b=ast.parse(original),ast.parse(s)
    def fn(t,n):return next(x for x in ast.walk(t) if isinstance(x,ast.FunctionDef) and x.name==n)
    checks={n:ast.dump(fn(a,n))==ast.dump(fn(b,n)) for n in ('run','input_for','command','describe','cases')}
    assert all(checks.values()),checks
    runner=BASE/'remainder_runner.py';compile(s,str(runner),'exec');runner.write_text(s)
    out=REMAINDER.parent/'experiment_memory_v2';assert not out.exists();out.mkdir();(out/'runs').mkdir()
    manifest=read(REMAINDER/'inputs.json')
    for p,h in manifest.items():assert sha(p)==h,('changed frozen input',p)
    for row in previous['completed']:
        for name in ('accepted.json','output.bin'):
            p=Path(row['path'])/name;manifest[str(p)]=sha(p)
    for p in (runner,source,Path(__file__),REMAINDER/'status.json'):
        manifest[str(p)]=sha(p)
    save(out/'inputs.json',manifest)
    save(out/'continuation_plan.json',dict(reused_accepted=59,original=str(REMAINDER),
        waiting_for=str(BASE/'complete.json'),max_parallel=12,simulator_as=8*GIB,
        host_scope='same24CPU group and20%RAM; reserve L3-11 capacity if still live',
        identical_scientific_functions=checks,original_source_sha256=sha(source),derived_sha256=sha(runner)))
    print('Prepared AI-12 continuation:59 cases reused; original validation unchanged; starts after recovery targets finish.')


if __name__=='__main__':main()
