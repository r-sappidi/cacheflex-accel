"""Create a host-only supervisor handoff; original simulation acceptance unchanged."""
from pathlib import Path
import json,hashlib,ast
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/resource_scheduler_20261001'
old=ROOT/'results/parallel_overlap_20260930/supervisor_v2.py';s=old.read_text()
s=s.replace("import broker\n", "import broker\nsys.path.insert(0,str(ROOT/'benchmarks/infrastructure'))\nfrom admission_policy import initial_limit,GROW_MARGIN,GROW_STEP,BASE_RESERVE\nAUDIT=ROOT/'results/resource_scheduler_20261001'\n",1)
s=s.replace("return int(11*GIB)  # outside8 + supervisor1 + AI-12 coordinator2; retiredllama reserve released", "return BASE_RESERVE  # outside processes and supervisor; retired AI-12 allowance removed")
a=s.index('def main():');b=s.index('    failed={};published=',a)
init='''def auxiliary_pending():
    paths=[ROOT/'results/cacheflex_flash_attention/qwen_matrix_20260930/experiment_v4/status.json',
           ROOT/'results/rumur_refresh_20261001/verification_v5/status.json',
           ROOT/'results/cacheflex_flash_attention/optimistic_ring_20261001/experiment_v2/status.json']
    for p in paths:
        if not p.exists():continue
        state=read(p);ident=process(state.get('pid',0))
        if ident and ident['state']!='Z' and state.get('stage') not in ('complete','completed_validated','failed'):
            return True
    return False


def main():
    lock=(AUDIT/'supervisor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cfg=read(OLD/'budget.json');constrain(os.getpid(),cfg['cpu_set'],GIB)
    extension=EXT
    assert not (AUDIT/'committed.json').exists(),'Do not duplicate handoff'
    previous=read(BASE/'status.json');old_pid=previous['pid'];before=process(old_pid)
    assert old_pid==4168836 and before and before['state'] not in ('Z','T') and not previous['failed']
    assert (EXT/'remainder_finished.json').exists() and not read(EXT/'remainder_finished.json').get('error')
    checked={}
    # Check immutable inputs before interrupting any scheduler.
    for family,out in FAMILIES.items():
        manifest=read(out/'inputs.json')
        for p,h in manifest.items():assert sha(p)==h,('frozen drift',family,p)
        checked[family]=dict(files=len(manifest),manifest_sha256=sha(out/'inputs.json'))
    original_states={};original_limits={};committed=False
    # Stop the previous writer only while we hold its broker lock: it cannot
    # be suspended holding that lock. Simulator/worker PIDs are preserved.
    with broker.locked() as ledger:
        old_ledger=json.loads(json.dumps(ledger))
        os.kill(old_pid,signal.SIGSTOP)
        try:
            deadline=time.time()+5
            while process(old_pid)['state']!='T':
                assert time.time()<deadline;time.sleep(.02)
            previous=read(BASE/'status.json');assert not previous['failed']
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
                original_states[row['worker_pid']]=worker['state'];os.kill(row['worker_pid'],signal.SIGSTOP)
                assert ws['stage']=='simulating',('validation boundary',row,ws)
                sim_pid=ws['simulator_pid'];sim=process(sim_pid)
                assert sim and sim['state']!='Z' and sim['start_ticks']==ws['simulator_identity']['start_ticks']
                original_states[sim_pid]=sim['state'];os.kill(sim_pid,signal.SIGSTOP)
                deadline=time.time()+5
                while process(sim_pid)['state']!='T':
                    assert time.time()<deadline;time.sleep(.02)
                old_limit=resource.prlimit(sim_pid,resource.RLIMIT_AS)[0];original_limits[sim_pid]=old_limit
                info=owned()[sim_pid];limit=initial_limit(info['vms'],old_limit)
                assert limit>=info['vms']+3*GIB
                constrain(sim_pid,cfg['cpu_set'],limit)
                running[(family,name)]=dict(family=family,name=name,journal=journal,spec=spec,
                    proc=Adopted(dict(pid=row['worker_pid'],identity=worker)),sim_pid=sim_pid,
                    sim_ident=sim,limit=limit,adopted=True,admitted=True,start=time.time(),memory_wait=False)
                adopted_ids.append(dict(name=name,worker_pid=row['worker_pid'],simulator_pid=sim_pid,
                    old_limit=old_limit,limit=limit,vms=info['vms'],rss=info['rss'],identity=sim))
            managed={p for r in adopted_ids for p in (r['worker_pid'],r['simulator_pid'])}
            # Auxiliary families carry their own leases; this conservative check
            # includes their current RSS too and requires outside usage <=6GiB.
            outside=sum(p['rss'] for pid,p in owned().items() if pid not in managed and not p['name'].startswith('gem5'))
            assert outside<BASE_RESERVE,('outside usage exceeds base reserve',outside)
            new_reserved=base_reservation()+sum(WORKER_AS+r['limit'] for r in running.values())
            assert new_reserved+sum(x['bytes'] for x in ledger['leases'].values())<=ledger['cap']
            save(AUDIT/'before.json',dict(ledger=old_ledger,states=states,identities=adopted_ids,checked=checked,outside_rss=outside))
            ledger.update(old_reserved=new_reserved,heartbeat=time.time(),hold_new=False,owner=os.getpid())
            for f,state in states.items():state.update(pid=os.getpid(),budget_extension=str(AUDIT))
            save(AUDIT/'committed.json',dict(pid=os.getpid(),old_pid=old_pid,old_identity=before,time=time.time(),adopted=adopted_ids))
            committed=True
        finally:
            if not committed:
                ledger.clear();ledger.update(old_ledger)
                for pid,limit in original_limits.items():
                    if process(pid):constrain(pid,cfg['cpu_set'],limit)
                for pid,st in original_states.items():
                    if process(pid) and st!='T':os.kill(pid,signal.SIGCONT)
                os.kill(old_pid,signal.SIGCONT)
    # Publish ownership before resuming adopted processes.
    save(BASE/'status.json',dict(previous,pid=os.getpid(),updated=time.time()))
    for r in running.values():
        for pid in (r['proc'].pid,r['sim_pid']):os.kill(pid,signal.SIGCONT)
    remainder_finished=True;remainder_error=None

'''
s=s[:a]+init+s[b:]
s=s.replace('while any(pending.values()) or running or not remainder_finished:', 'while any(pending.values()) or running or not remainder_finished or auxiliary_pending():')
s=s.replace("if info['vms']>=r['limit']-4*GIB or r.get('memory_wait'):","if info['vms']>=r['limit']-GROW_MARGIN or r.get('memory_wait'):")
s=s.replace("if not pressure and reserved+8*GIB<=cfg['ram_limit']:\n                        old_limit=r['limit'];r['limit']+=8*GIB", "if not pressure and reserved+GROW_STEP<=cfg['ram_limit']:\n                        old_limit=r['limit'];r['limit']+=GROW_STEP")
s=s.replace("constrain(pid,cfg['cpu_set'],r['limit']);reserved+=8*GIB", "constrain(pid,cfg['cpu_set'],r['limit']);reserved+=GROW_STEP")
s=s.replace("with (EXT/'quota_growth.jsonl').open('a')", "with (AUDIT/'quota_growth.jsonl').open('a')")
s=s.replace('        update(metric)\n',"        update(metric)\n        save(AUDIT/'status.json',dict(stage='running',pid=os.getpid(),resources=metric,updated=time.time()))\n")
s=s.replace("save(BASE/'supervisor_failed.json'", "save(AUDIT/'supervisor_failed.json'")
# Prove that result acceptance and update/publication functions are unchanged.
def subtree(src,name):
 return ast.dump(next(n for n in ast.walk(ast.parse(src)) if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
original=old.read_text();assert subtree(original,'update')==subtree(s,'update')
a='        for key,r in list(running.items()):';b='        mem=memory();'
assert original[original.index(a):original.index(b)]==s[s.index(a):s.index(b)]
ast.parse(s)
(OUT/'supervisor.py').write_text(s)
(OUT/'audit.json').write_text(json.dumps(dict(original_sha256=hashlib.sha256(original.encode()).hexdigest(),derived_sha256=hashlib.sha256(s.encode()).hexdigest(),result_acceptance_identical=True,result_publication_identical=True,changes=['preserve/adopt all existing workers','trim unused AS allowance with >=3GiB remaining','long-context growth margin2GiB / step1GiB','base reserve11GiB->6GiB after outside-RSS check','retain heartbeat while auxiliary queues pending'],cap_unchanged=True),indent=2)+'\n')
print('Scheduler prepared; acceptance and publication unchanged')
