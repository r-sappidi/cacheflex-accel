"""State-preserving OPS-21 supervisor handoff to reserve L3-12 resources."""
import ast
from pathlib import Path
from recover_parallel_memory import ROOT,BASE,OLD,read,save,sha,process

EXT=ROOT/'results/cacheflex_l3_spm/qwen_neighbors_20260929/budget_handoff'
INIT='''    cfg=read(OLD/'budget.json');constrain(os.getpid(),cfg['cpu_set'],GIB)
    extension=ROOT/'results/cacheflex_l3_spm/qwen_neighbors_20260929/budget_handoff'
    assert not (extension/'committed.json').exists(),'Do not duplicate handoff'
    previous=read(BASE/'status.json');old_pid=previous['pid'];before=process(old_pid)
    assert before and before['state']!='Z'
    os.kill(old_pid,signal.SIGSTOP)
    committed=False
    try:
        deadline=time.time()+5
        while process(old_pid)['state']!='T':
            assert time.time()<deadline;time.sleep(.02)
        previous=read(BASE/'status.json')
        assert not previous['failed'],'Inspect new failure before handoff'
        for n in ('status.json','resources.json','handoff.json'):
            shutil.copy2(BASE/n,extension/('previous_'+n))
        states={f:read(out/'status.json') for f,out in FAMILIES.items()}
        handoff=read(OLD/'handoff.json')
        pending={f:[j for j in handoff['families'][f]['all_jobs'] if j['name'] not in {r['name'] for r in states[f]['completed']}] for f in FAMILIES}
        running={};adopted_ids=[]
        for row in previous['active']:
            family,name=row['family'],row['name']
            journal=BASE/'workers'/family/name
            if not journal.exists():journal=OLD/'workers'/family/name
            spec=read(journal/'spec.json');ws=read(journal/'worker_status.json')
            worker=process(row['worker_pid']);assert worker and worker['state']!='Z'
            # No simulator restart. Freeze both long enough to record identities.
            os.kill(row['worker_pid'],signal.SIGSTOP)
            assert ws['stage']=='simulating',('worker crossed validation boundary; inspect',row,ws)
            sim_pid=ws['simulator_pid'];sim=process(sim_pid)
            assert sim and sim['start_ticks']==ws['simulator_identity']['start_ticks']
            if sim['state']!='Z':os.kill(sim_pid,signal.SIGSTOP)
            limit=resource.prlimit(sim_pid,resource.RLIMIT_AS)[0]
            assert limit in (int(4.5*GIB),SIM_AS)
            running[(family,name)]=dict(family=family,name=name,journal=journal,spec=spec,
                 proc=Adopted(dict(pid=row['worker_pid'],identity=worker)),sim_pid=sim_pid,
                 sim_ident=sim,limit=limit,adopted=True,admitted=row['admitted'],start=time.time())
            adopted_ids.append(dict(name=name,worker_pid=row['worker_pid'],simulator_pid=sim_pid,limit=limit))
        for f,state in states.items():state.update(pid=os.getpid(),budget_extension=str(extension))
        save(extension/'committed.json',dict(pid=os.getpid(),old_pid=old_pid,old_identity=before,time=time.time(),adopted=adopted_ids))
        committed=True
    finally:
        if not committed:
            # Original supervisor remains authoritative and restores its admitted workers.
            os.kill(old_pid,signal.SIGCONT)

'''

RESERVATION='''def base_reservation():
    l3=read(L3/'status.json');alive=process(l3['pid'])
    # Preflight proved the fixed five-case L3-11 runner has <=one unfinished
    # simulation and no possible replenishment. Reserve8GiB+2GiB while alive.
    l3_bytes=10*GIB if alive and alive['state']!='Z' else 0
    neighbor=ROOT/'results/cacheflex_l3_spm/qwen_neighbors_20260929'
    launch=neighbor/'launch.json';done=neighbor/'experiment_v1/complete.json'
    proc=process(read(launch)['pid']) if launch.exists() else None
    finished=done.exists() and (proc is None or proc['state']=='Z')
    return int(18.5*GIB)+l3_bytes+(0 if finished else 18*GIB)

'''


def main():
    assert not EXT.exists();EXT.mkdir()
    source=ROOT/'scripts/recover_parallel_memory.py';s=source.read_text();original=s
    prior=read(ROOT/'results/cacheflex_l3_spm/qwen_t4096_tuning_20260929/experiment_v1/status.json')
    completed=[r for r in prior['completed'] if 'qwen_t4096_tuning_20260929/experiment_v1/runs' in r['path']]
    assert len(completed)>=4 and len(prior['active'])<=1
    assert len(read(ROOT/'results/cacheflex_l3_spm/qwen_t4096_tuning_20260929/selection.json')['selected'])==5
    start=s.index('def base_reservation():');end=s.index('\ndef normalized_worker',start)
    s=s[:start]+RESERVATION+s[end:]
    start=s.index("    assert not (BASE/'handoff.json').exists(),'Already handed off; do not duplicate'")
    end=s.index("    failed={};published=",start)
    s=s[:start]+INIT+s[end:]
    s=s.replace("lock=(BASE/'supervisor.lock').open('w')", "lock=(ROOT/'results/cacheflex_l3_spm/qwen_neighbors_20260929/budget_handoff/supervisor.lock').open('w')")
    # Cleanup the first (already stopped) supervisor after every old job is accepted.
    anchor="    # Recovery of AI-12 is prepared separately with unchanged scientific gates."
    assert s.count(anchor)==1
    wait='''    oldest=read(BASE/'handoff.json')
    ident=process(oldest['old_pid'])
    if ident and ident['start_ticks']==oldest['old_identity']['start_ticks']:os.kill(oldest['old_pid'],signal.SIGKILL)
    # AI-12 has its original independent capacity policy. Start only once both
    # FA coordinators have exited, so reservations cannot overlap invisibly.
    while base_reservation()>int(18.5*GIB):
        metric['updated']=time.time();metric['reserved_bytes']=base_reservation()
        metric['workers']=0;metric['admitted_workers']=0
        save(BASE/'resources.json',metric);save(OLD/'resources.json',metric)
        for path in (BASE/'status.json',OLD/'status.json'):
            final=read(path);final['updated']=time.time();final['stage']='waiting_for_fa_queues';save(path,final)
        time.sleep(5)
'''
    s=s.replace(anchor,wait+anchor)
    # The main scheduling/admission/validation loop is byte-for-byte unchanged.
    def loop(text):return text[text.index('    while any(pending.values()) or running:'):text.index("    save(BASE/'complete.json'")]
    assert loop(original)==loop(s)
    dest=EXT/'supervisor.py';compile(s,str(dest),'exec');dest.write_text(s)
    save(EXT/'audit.json',dict(source=str(source),source_sha256=sha(source),derived_sha256=sha(dest),
        unchanged_main_loop=True,changes='live-worker handoff and explicit neighbor reservation only',
        l3_original_runner_sha256=sha(ROOT/'results/cacheflex_l3_spm/qwen_t4096_tuning_20260929/selected_runner.py'),
        unfinished_l3_cases=5-len(completed),neighbor_reservation=18*2**30))
    print('Prepared supervisor handoff; original scheduling/acceptance loop unchanged;18GiB neighbor reservation.')


if __name__=='__main__':main()
