"""OPS-23: derive scheduling-only concurrent AI-12/recovery continuation."""
import ast,json,sys
from pathlib import Path
ROOT=Path('/data2/rsappidi/cacheflex_mc');EXT=ROOT/'results/parallel_overlap_20260930'
sys.path.insert(0,str(ROOT/'scripts'))
from parallel_budget_worker import sha,read,save
SOURCE=ROOT/'results/long_memory_recovery_20260930/supervisor.py'
REMAINDER=ROOT/'results/parallel_memory_recovery_20260929/remainder_runner.py'

INIT='''    cfg=read(OLD/'budget.json');constrain(os.getpid(),cfg['cpu_set'],GIB)
    extension=EXT
    assert not (extension/'committed.json').exists(),'Do not duplicate handoff'
    assert not (REMAINDER_OUT/'status.json').exists(),'Remainder already started'
    previous=read(BASE/'status.json');old_pid=previous['pid'];before=process(old_pid)
    assert before and before['state'] not in ('Z','T') and not previous['failed']
    original_states={};original_limits={};committed=False
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
            assert ws['stage']=='simulating',('validation boundary; retry handoff later',row,ws)
            sim_pid=ws['simulator_pid'];sim=process(sim_pid)
            assert sim and sim['state']!='Z' and sim['start_ticks']==ws['simulator_identity']['start_ticks']
            original_states[sim_pid]=sim['state'];os.kill(sim_pid,signal.SIGSTOP)
            old_limit=resource.prlimit(sim_pid,resource.RLIMIT_AS)[0];original_limits[sim_pid]=old_limit
            info=owned()[sim_pid]
            # Preserve >=8GiB virtual headroom; only trim unused16K reservations.
            limit=old_limit
            if family=='long' and name.startswith('t16384'):
                limit=min(old_limit,max(16*GIB,((info['vms']+GIB-1)//GIB+8)*GIB))
            assert limit>info['vms'];constrain(sim_pid,cfg['cpu_set'],limit)
            running[(family,name)]=dict(family=family,name=name,journal=journal,spec=spec,
                proc=Adopted(dict(pid=row['worker_pid'],identity=worker)),sim_pid=sim_pid,
                sim_ident=sim,limit=limit,adopted=True,admitted=row['admitted'],start=time.time())
            adopted_ids.append(dict(name=name,worker_pid=row['worker_pid'],simulator_pid=sim_pid,old_limit=old_limit,limit=limit,identity=sim))
        for f,state in states.items():state.update(pid=os.getpid(),budget_extension=str(extension))
        old_reserved=base_reservation()+sum(WORKER_AS+r['limit'] for r in running.values())
        assert old_reserved<=cfg['ram_limit']
        broker.save(EXT/'broker.json',dict(cap=cfg['ram_limit'],old_reserved=old_reserved,leases={},hold_new=False,heartbeat=time.time(),updated=time.time()))
        save(extension/'committed.json',dict(pid=os.getpid(),old_pid=old_pid,old_identity=before,time=time.time(),adopted=adopted_ids))
        committed=True
    finally:
        if not committed:
            for pid,limit in original_limits.items():
                if process(pid):constrain(pid,cfg['cpu_set'],limit)
            for pid,st in original_states.items():
                if process(pid) and st!='T':os.kill(pid,signal.SIGCONT)
            os.kill(old_pid,signal.SIGCONT)
    with (EXT/'remainder.log').open('x') as log:
        remainder_proc=subprocess.Popen(['/usr/bin/python3',str(EXT/'remainder_runner.py')],cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(EXT/'remainder_launch.json',dict(pid=remainder_proc.pid,time=time.time()))
    remainder_finished=False;remainder_error=None

'''
PHASE='''    def phase(label,cases):
        state['stage']=label;status();accepted={r['name'] for r in rows};pending=[c for c in cases if name(dict(c,stage=label)) not in accepted];futures={}
        try:
            with cf.ThreadPoolExecutor(max_workers=4) as pool:
                while pending or futures:
                    for f in list(futures):
                        if f.done():
                            ticket=futures.pop(f)
                            try:f.result()
                            finally:broker.release(ticket)
                    resource_sample();slots,admission=capacity();state['admission']=admission
                    if state['pressure']:slots=0
                    if pending and len(futures)<slots:
                        c=dict(pending[0],stage=label);ticket=name(c)
                        if broker.acquire(ticket):
                            pending.pop(0)
                            try:inp=input_for(c);futures[pool.submit(run,c,inp)]=ticket
                            except BaseException:
                                broker.release(ticket);raise
                        else:time.sleep(2)
                    else:time.sleep(2)
                    status()
        finally:
            for ticket in futures.values():broker.release(ticket)
'''

def main():
 assert not (EXT/'committed.json').exists()
 s=SOURCE.read_text();original=s
 def change(a,b):
  nonlocal s
  assert s.count(a)==1,a
  s=s.replace(a,b)
 change("FAILED=[('long','t16384_direct')]",'''FAILED=[]
EXT=ROOT/'results/parallel_overlap_20260930'
REMAINDER_OUT=ROOT/'results/ai_workloads/qwen_remainder_20260929/experiment_memory_v2'
sys.path.insert(0,str(EXT))
import broker''')
 change('return int(18.5*GIB)  # outside8 + llama9.5 + supervisor1','return int(20.5*GIB)  # outside8 + llama9.5 + supervisor1 + AI-12 coordinator2')
 change("if family=='long':return (48 if name.startswith('t32768') else 24)*GIB","if family=='long':return (12 if name.startswith('t32768') else 16)*GIB")
 change("lock=(ROOT/'results/long_memory_recovery_20260930/supervisor.lock').open('w')","lock=(EXT/'supervisor.lock').open('w')")
 start=s.index("    cfg=read(OLD/'budget.json');constrain(os.getpid(),cfg['cpu_set'],GIB)",s.index('def main():'));end=s.index('    failed={};published=',start);s=s[:start]+INIT+s[end:]
 change('    while any(pending.values()) or running:', '    while any(pending.values()) or running or not remainder_finished:')
 # Start32K at12GiB with monitored growth, admit at most5long jobs while AI-12 runs.
 change("            for r in waiting:\n                wanted=", "            for r in waiting:\n                if r['family']=='long' and sum(x['family']=='long' and x['admitted'] for x in running.values()) >= (7 if remainder_finished else 5):continue\n                wanted=")
 change("if info['vms']>=r['limit']-2*GIB or r.get('memory_wait'):", "if info['vms']>=r['limit']-4*GIB or r.get('memory_wait'):")
 change("ROOT/'results/long_memory_recovery_20260930/quota_growth.jsonl'", "EXT/'quota_growth.jsonl'")
 # Lock BOTH admission domains around all mutations to old reservations.
 start=s.index('        reserved=base_reservation()+sum(');end=s.index("        for r in running.values():\n            ws=r['journal']/'worker_status.json';w=read(ws) if ws.exists() else {}",start)
 block=s[start:end]
 block=block.replace("reserved=base_reservation()+sum(WORKER_AS+r['limit'] for r in running.values())", "reserved=base_reservation()+sum(WORKER_AS+r['limit'] for r in running.values())+sum(x['bytes'] for x in budget_state['leases'].values())")
 tail='''        budget_state['old_reserved']=base_reservation()+sum(WORKER_AS+r['limit'] for r in running.values())
        budget_state['heartbeat']=time.time()
        budget_state['hold_new']=pressure or any(r.get('memory_wait') for r in running.values())
        # Grow/pause remainder children under the SAME atomic reservation ledger.
        remstate=read(REMAINDER_OUT/'status.json') if (REMAINDER_OUT/'status.json').exists() else {}
        for name,lease in budget_state['leases'].items():
            pid=remstate.get('active',{}).get(name,{}).get('pid');info=now.get(pid)
            if not info or info['state']=='Z':continue
            if info['vms']>=lease['bytes']-2*GIB or lease.get('memory_wait'):
                if not pressure and broker.total(budget_state)+8*GIB<=cfg['ram_limit']:
                    lease['bytes']+=8*GIB;constrain(pid,cfg['cpu_set'],lease['bytes']);lease['memory_wait']=False
                else:lease['memory_wait']=True
            pause=pressure or lease.get('memory_wait',False)
            if pause and info['state']!='T':os.kill(pid,signal.SIGSTOP)
            elif not pause and info['state']=='T':os.kill(pid,signal.SIGCONT)
            budget_state['hold_new'] |= lease.get('memory_wait',False)
        reserved=broker.total(budget_state)
        remainder_leases=len(budget_state['leases'])
'''
 s=s[:start]+"        with broker.locked() as budget_state:\n"+''.join('    '+line if line.strip() else line for line in (block+tail).splitlines(True))+s[end:]
 change('        now=owned();stamp=time.monotonic();cpu=None', '''        if not remainder_finished and remainder_proc.poll() is not None:
            remainder_finished=True
            remainder_error=None if remainder_proc.returncode==0 and (REMAINDER_OUT/'complete.json').exists() else str(remainder_proc.returncode)
            save(EXT/'remainder_finished.json',dict(returncode=remainder_proc.returncode,error=remainder_error))
        now=owned();stamp=time.monotonic();cpu=None''')
 change('workers=len(running),admitted_workers=', 'remainder_leases=remainder_leases,remainder_finished=remainder_finished,remainder_error=remainder_error,workers=len(running),admitted_workers=')
 # Retire only this predecessor after work; no automatic duplicate AI-12 launch.
 end=s.index('    # AI-12 has its original independent capacity policy.')
 stop=s.index("\n\nif __name__=='__main__':",end)
 s=s[:end]+"    save(EXT/'complete.json',dict(failed=failed,remainder_error=remainder_error,time=time.time()))\n"+s[stop:]
 s=s.replace('        time.sleep(5)','        time.sleep(2)')
 def publication(t):return t[t.index('        for key,r in list(running.items()):'):t.index('        mem=memory();rss=')]
 assert publication(s)==publication(original)
 compile(s,str(EXT/'supervisor.py'),'exec');(EXT/'supervisor.py').write_text(s)
 r=REMAINDER.read_text();r=r.replace("GIB=2**30", "GIB=2**30\nsys.path.insert(0,str(ROOT/'results/parallel_overlap_20260930'))\nimport broker")
 start=r.index('def capacity():');end=r.index('\ndef prepare():',start);r=r[:start]+"def capacity():\n    return broker.capacity()\n\n"+r[end:]
 start=r.index('    def phase(');end=r.index('    def cases(',start);r=r[:start]+PHASE+r[end:]
 r=r.replace("available<64*GIB", "available<64*GIB+max(0,budget['ram_limit']-rss)")
 def fn(t,n):return next(x for x in ast.walk(ast.parse(t)) if isinstance(x,ast.FunctionDef) and x.name==n)
 checks={n:ast.dump(fn(r,n))==ast.dump(fn(REMAINDER.read_text(),n)) for n in ('run','input_for','command','describe','cases','unchanged')}
 assert all(checks.values()),checks
 compile(r,str(EXT/'remainder_runner.py'),'exec');(EXT/'remainder_runner.py').write_text(r)
 save(EXT/'audit.json',dict(source_supervisor=sha(SOURCE),derived_supervisor=sha(EXT/'supervisor.py'),original_publication_identical=True,remainder_scientific_functions=checks,broker_sha256=sha(EXT/'broker.py'),derived_remainder=sha(EXT/'remainder_runner.py'),initial16k_min_gib=16,initial32k_gib=12,old_growth_margin_gib=4,growth_step_gib=8,max_remainder_simulators=4,old_pids_preserved=True))
 print('Prepared atomic shared reservations, preserved scientific acceptance and process adoption.')
if __name__=='__main__':main()
