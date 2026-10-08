"""OPS-22: larger context-dependent host limits plus proactive bounded growth."""
import ast
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path('/data2/rsappidi/cacheflex_mc/scripts')))
from recover_parallel_memory import ROOT,BASE,OLD,FAMILIES,read,save,sha,GIB
from parallel_budget_scheduler import owned,memory

EXT=ROOT/'results/long_memory_recovery_20260930'
SOURCE=ROOT/'results/cacheflex_l3_spm/qwen_neighbors_20260929/budget_handoff/supervisor.py'

INIT='''    cfg=read(OLD/'budget.json');constrain(os.getpid(),cfg['cpu_set'],GIB)
    extension=ROOT/'results/long_memory_recovery_20260930'
    assert not (extension/'committed.json').exists(),'Do not duplicate handoff'
    previous=read(BASE/'status.json');old_pid=previous['pid'];before=process(old_pid)
    assert before and before['state']!='Z'
    os.kill(old_pid,signal.SIGSTOP)
    committed=False
    try:
        deadline=time.time()+5
        while process(old_pid)['state']!='T':
            assert time.time()<deadline;time.sleep(.02)
        previous=read(BASE/'status.json');assert set(previous['failed'])=={'long'}
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
            os.kill(row['worker_pid'],signal.SIGSTOP)
            assert ws['stage']=='simulating',('worker crossed validation boundary',row,ws)
            sim_pid=ws['simulator_pid'];sim=process(sim_pid)
            assert sim and sim['start_ticks']==ws['simulator_identity']['start_ticks']
            if sim['state']!='Z':os.kill(sim_pid,signal.SIGSTOP)
            limit=resource.prlimit(sim_pid,resource.RLIMIT_AS)[0]
            running[(family,name)]=dict(family=family,name=name,journal=journal,spec=spec,
                 proc=Adopted(dict(pid=row['worker_pid'],identity=worker)),sim_pid=sim_pid,
                 sim_ident=sim,limit=limit,adopted=True,admitted=(family!='long'),start=time.time())
            adopted_ids.append(dict(name=name,worker_pid=row['worker_pid'],simulator_pid=sim_pid,limit=limit))
        failure=OLD/'workers/long/t16384_direct/failed.json'
        assert 'bad_alloc' in str(read(failure)) and 'allocation failed' in str(read(failure))
        src=FAMILIES['long']/'runs/t16384_direct'
        assert not (src/'accepted.json').exists()
        assert not any(p.startswith(str(src)) for p in read(FAMILIES['long']/'inputs.json'))
        hashes={str(p.relative_to(src)):sha(p) for p in src.rglob('*') if p.is_file()}
        dest=extension/'failed_t16384_direct';src.rename(dest)
        assert all(sha(dest/p)==h for p,h in hashes.items())
        for f,state in states.items():
            state['previous_failure_ops22']=state.pop('error',None)
            state.update(pid=os.getpid(),budget_extension=str(extension))
        save(extension/'committed.json',dict(pid=os.getpid(),old_pid=old_pid,old_identity=before,time=time.time(),adopted=adopted_ids,archived=dict(path=str(dest),hashes=hashes)))
        committed=True
    finally:
        if not committed:os.kill(old_pid,signal.SIGCONT)

'''


def main():
    assert not EXT.exists(),'Inspect existing recovery before duplicate launch'
    state=read(BASE/'status.json');assert set(state['failed'])=={'long'}
    # The FA tuning queues have finished; their reservations can be released.
    for name in ('qwen_t4096_tuning_20260929','qwen_neighbors_20260929'):
        path=ROOT/'results/cacheflex_l3_spm'/name/'experiment_v1'
        assert (path/'complete.json').exists() and read(path/'status.json')['stage']=='completed_validated'
    for family,out in FAMILIES.items():
        for p,h in read(out/'inputs.json').items():assert sha(p)==h,('changed frozen file',family,p)
    mem=memory();rss=sum(p['rss'] for p in owned().values());cap=read(OLD/'budget.json')['ram_limit']
    assert mem['MemAvailable']-max(0,cap-rss)>64*GIB,'insufficient server headroom beyond user cap'
    EXT.mkdir()
    worker_source=(BASE/'worker.py').read_text();assert worker_source.count('int(8*GIB)')==2
    worker=worker_source.replace('int(8*GIB)',"int(spec.get('simulator_as',8*GIB))")
    (EXT/'worker.py').write_text(worker)
    # Reversing exactly these two scheduling substitutions recovers original AST.
    assert ast.dump(ast.parse(worker.replace("int(spec.get('simulator_as',8*GIB))",'int(8*GIB)')))==ast.dump(ast.parse(worker_source))
    s=SOURCE.read_text();original=s
    def change(a,b):
        nonlocal s
        assert s.count(a)==1,a
        s=s.replace(a,b)
    change("WORKER=BASE/'worker.py'", "WORKER=ROOT/'results/long_memory_recovery_20260930/worker.py'")
    change("FAILED=[('long','t16384_noc'),('ffn','target_t2048_spm_t2048_mp8_kc1024_g0')]", "FAILED=[('long','t16384_direct')]")
    start=s.index('def base_reservation():');end=s.index('\ndef normalized_worker',start)
    s=s[:start]+'''def base_reservation():
    return int(18.5*GIB)  # outside8 + llama9.5 + supervisor1

def target_limit(family,name):
    if family=='long':return (48 if name.startswith('t32768') else 24)*GIB
    return SIM_AS

'''+s[end:]
    change("lock=(ROOT/'results/cacheflex_l3_spm/qwen_neighbors_20260929/budget_handoff/supervisor.lock').open('w')", "lock=(ROOT/'results/long_memory_recovery_20260930/supervisor.lock').open('w')")
    start=s.index("    cfg=read(OLD/'budget.json');constrain(os.getpid(),cfg['cpu_set'],GIB)",s.index('def main():'))
    end=s.index('    failed={};published=',start)
    s=s[:start]+INIT+s[end:]
    change("stage=('paused_resource' if not r['admitted'] else w.get('stage','starting'))", "stage=('paused_memory_growth' if r.get('memory_wait') else 'paused_resource' if not r['admitted'] else w.get('stage','starting'))")
    change("mem['MemAvailable']<64*GIB", "mem['MemAvailable']<64*GIB+max(0,cfg['ram_limit']-rss)")
    change("if reserved+WORKER_AS+SIM_AS>cfg['ram_limit']:continue", "wanted=target_limit(family,name)\n                if reserved+WORKER_AS+wanted>cfg['ram_limit']:continue")
    change("spec=dict(old_spec,journal=str(journal),adopted=None)", "spec=dict(old_spec,journal=str(journal),adopted=None,simulator_as=wanted)")
    change("limit=SIM_AS,adopted=False,admitted=True,start=time.time())", "limit=wanted,adopted=False,admitted=True,start=time.time())")
    change("reserved+=WORKER_AS+SIM_AS", "reserved+=WORKER_AS+wanted")
    change("extra=SIM_AS-r['limit']", "wanted=max(r['limit'],target_limit(r['family'],r['name']))\n                extra=wanted-r['limit']")
    change("constrain(r['sim_pid'],cfg['cpu_set'],SIM_AS)", "constrain(r['sim_pid'],cfg['cpu_set'],wanted)")
    change("r.update(limit=SIM_AS,admitted=True);reserved+=extra", "r.update(limit=wanted,admitted=True);reserved+=extra")
    anchor="        for r in running.values():\n            ws=r['journal']/'worker_status.json';w=read(ws) if ws.exists() else {}"
    growth='''        # Grow well before allocation failure. Pausing preserves simulation
        # state if another job must finish before8GiB of quota can be reserved.
        for r in running.values():
            if not r['admitted'] or r['family'] in failed:continue
            ws=r['journal']/'worker_status.json'
            w=read(ws) if ws.exists() else {}
            pid=w.get('simulator_pid',r.get('sim_pid'))
            info=now.get(pid)
            if not info or info['state']=='Z':continue
            if info['vms']>=r['limit']-2*GIB or r.get('memory_wait'):
                if not pressure and reserved+8*GIB<=cfg['ram_limit']:
                    old_limit=r['limit'];r['limit']+=8*GIB
                    constrain(pid,cfg['cpu_set'],r['limit']);reserved+=8*GIB
                    r['memory_wait']=False
                    with (ROOT/'results/long_memory_recovery_20260930/quota_growth.jsonl').open('a') as f:
                        f.write(json.dumps(dict(time=time.time(),name=r['name'],pid=pid,old=old_limit,new=r['limit'],reserved=reserved))+'\\n')
                else:r['memory_wait']=True
'''
    change(anchor,growth+anchor)
    change("pause=pressure or r['family'] in failed or not r['admitted']", "pause=pressure or r['family'] in failed or not r['admitted'] or r.get('memory_wait',False)")
    change("reserved_bytes=reserved,logical_cpu_cap=", "reserved_bytes=reserved,server_headroom_after_full_user_budget=mem['MemAvailable']-max(0,cfg['ram_limit']-rss),\n            memory_growth_wait=[r['name'] for r in running.values() if r.get('memory_wait')],logical_cpu_cap=")
    dest=EXT/'supervisor.py';compile(s,str(dest),'exec');dest.write_text(s)
    # Publication and its complete original scientific peer validation stay exact.
    def publication(text):return text[text.index('        for key,r in list(running.items()):'):text.index('        mem=memory();rss=')]
    assert publication(original)==publication(s)
    compile(worker,str(EXT/'worker.py'),'exec')
    save(EXT/'audit.json',dict(source=str(SOURCE),source_sha256=sha(SOURCE),derived_sha256=sha(dest),
        worker_source_sha256=sha(BASE/'worker.py'),worker_derived_sha256=sha(EXT/'worker.py'),
        publication_identical=True,worker_only_host_limit_parameters=True,
        new_limits_gib=dict(t16384=24,t32768=48),growth_step_gib=8,growth_margin_gib=2,
        initial_server_available_gib=mem['MemAvailable']/GIB,initial_user_rss_gib=rss/GIB,
        headroom_after_full_user_budget_gib=(mem['MemAvailable']-max(0,cap-rss))/GIB,minimum_headroom_gib=64))
    # All3 T16K can run alongside2FFN and four preservedT32K processes.
    assert (18.5+2*9.5+3*25.5+9.5+3*6)*GIB<=cap
    print('Prepared24/48GiB limits, bounded8GiB growth,64GiB extra server headroom; original acceptance unchanged.')


if __name__=='__main__':main()
