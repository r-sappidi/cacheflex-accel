"""OPS-21: adopt live workers and raise host AS allowances, preserving checks.

The old source/budget files stay frozen. Reserve each paused child at its old
limit, and upgrade/admit 8GiB simulators only when aggregate AS reservations fit.
The independent L3-11 queue always gets its full potential 34GiB reservation.
"""
import ast
import fcntl
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path('/data2/rsappidi/cacheflex_mc')
sys.path.insert(0,str(ROOT/'scripts'))
from parallel_budget_scheduler import FAMILIES,owned,memory,check_peers,publish_report
from parallel_budget_worker import read,save,sha,process,constrain,Adopted

GIB=2**30
OLD=ROOT/'results/parallel_budget_20260929'
BASE=ROOT/'results/parallel_memory_recovery_20260929'
L3=ROOT/'results/cacheflex_l3_spm/qwen_t4096_tuning_20260929/experiment_v1'
REMAINDER=ROOT/'results/ai_workloads/qwen_remainder_20260929/experiment_v1'
WORKER=BASE/'worker.py'
ENV=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
         NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT/'scripts'))
SIM_AS=8*GIB
WORKER_AS=int(1.5*GIB)
FAILED=[('long','t16384_noc'),('ffn','target_t2048_spm_t2048_mp8_kc1024_g0')]


def base_reservation():
    # Outside work8GiB, existing llama9.5GiB, this supervisor1GiB.
    # Retain full potential L3-11 allocation until its coordinator exits.
    l3=read(L3/'status.json')
    alive=process(l3['pid'])
    return int(18.5*GIB)+(34*GIB if alive and alive['state']!='Z' else 0)


def normalized_worker(source):
    tree=ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node,ast.Constant) and node.value==8:
            node.value=4.5
    return ast.dump(tree)


def prepare():
    assert not BASE.exists(),'Recovery already exists; inspect, do not duplicate'
    cfg=read(OLD/'budget.json');state=read(OLD/'status.json')
    assert set(state.get('failed',{}))=={'long','ffn'}
    identities={}
    for row in state['active']:
        p=process(row['worker_pid']);assert p and p['state']=='T',row
        ws=read(OLD/'workers'/row['family']/row['name']/'worker_status.json')
        sim=process(ws['simulator_pid'])
        assert sim and sim['state']=='T' and sim['start_ticks']==ws['simulator_identity']['start_ticks']
        identities[row['name']]=dict(worker_pid=row['worker_pid'],worker=p,
                                    simulator_pid=ws['simulator_pid'],simulator=sim)
    checked={}
    for family,out in FAMILIES.items():
        manifest=read(out/'inputs.json')
        for p,h in manifest.items():assert sha(p)==h,('changed frozen input',family,p)
        checked[family]=dict(files=len(manifest),manifest_sha256=sha(out/'inputs.json'))
    for family,name in FAILED:
        journal=OLD/'workers'/family/name
        failure=read(journal/'failed.json')
        assert 'bad_alloc' in str(failure) and 'allocation failed' in str(failure)
        assert not (FAMILIES[family]/'runs'/name/'accepted.json').exists()
        assert not any(p.startswith(str(FAMILIES[family]/'runs'/name)) for p in read(FAMILIES[family]/'inputs.json'))
    source=(ROOT/'scripts/parallel_budget_worker.py').read_text()
    assert source.count('int(4.5*GIB)')==2
    derived=source.replace('int(4.5*GIB)','int(8*GIB)')
    # The only source change is the two simulator host address-space limits.
    assert normalized_worker(derived)==normalized_worker(source)
    BASE.mkdir();(BASE/'workers').mkdir();(BASE/'preserved').mkdir()
    WORKER.write_text(derived)
    save(BASE/'worker_audit.json',dict(source=str(ROOT/'scripts/parallel_budget_worker.py'),
        source_sha256=sha(ROOT/'scripts/parallel_budget_worker.py'),derived_sha256=sha(WORKER),
        change='Only two host simulator AS constants:4.5GiB->8GiB. All command/reference/acceptance functions identical.'))
    save(BASE/'preflight.json',dict(time=time.time(),old_state=state,identities=identities,
         verified_manifests=checked,simulator_as=SIM_AS,worker_as=WORKER_AS,
         ram_limit=cfg['ram_limit'],cpu_set=cfg['cpu_set'],base_reserved=base_reservation(),
         failed_retries=FAILED))
    # Admission invariants for old paused workers, resumed workers and retries.
    for l3alive in (False,True):
        base=(18.5+34*int(l3alive))*GIB
        total=base+len(identities)*6*GIB
        assert total<cfg['ram_limit']
        upgrades=0
        while total+int(3.5*GIB)<=cfg['ram_limit']:
            total+=int(3.5*GIB);upgrades+=1
            if upgrades==len(identities):break
        assert total<=cfg['ram_limit']
    print(json.dumps(dict(prepared=str(BASE),preserved_workers=len(identities),retries=FAILED,
                         manifests=checked,initial_reservation_gib=(base_reservation()+len(identities)*6*GIB)/GIB)))


def main():
    lock=(BASE/'supervisor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (BASE/'handoff.json').exists(),'Already handed off; do not duplicate'
    cfg=read(OLD/'budget.json');constrain(os.getpid(),cfg['cpu_set'],GIB)
    pre=read(BASE/'preflight.json');old_pid=pre['old_state']['pid']
    assert process(old_pid)['state']!='Z'
    before=process(old_pid)
    os.kill(old_pid,signal.SIGSTOP)
    committed=False
    try:
        deadline=time.time()+5
        while process(old_pid)['state']!='T':
            assert time.time()<deadline;time.sleep(.02)
        old_state=read(OLD/'status.json')
        assert {(x['family'],x['name'],x['worker_pid']) for x in old_state['active']}=={(x['family'],x['name'],x['worker_pid']) for x in pre['old_state']['active']}
        for n in ('status.json','resources.json','budget.json','handoff.json'):
            shutil.copy2(OLD/n,BASE/'preserved'/('old_'+n))
        states={f:read(out/'status.json') for f,out in FAMILIES.items()}
        for f,state in states.items():
            save(BASE/'preserved'/f'{f}_status.json',state)
            state['previous_failure']=state.pop('error',None)
            state.update(pid=os.getpid(),recovery=str(BASE))
        handoff=read(OLD/'handoff.json')
        pending={f:[j for j in handoff['families'][f]['all_jobs']
                    if j['name'] not in {r['name'] for r in states[f]['completed']}] for f in FAMILIES}
        running={}
        for row in old_state['active']:
            family,name=row['family'],row['name'];journal=OLD/'workers'/family/name
            spec=read(journal/'spec.json');ws=read(journal/'worker_status.json')
            ident=process(row['worker_pid']);sim=process(ws['simulator_pid'])
            assert ident['state']==sim['state']=='T'
            running[(family,name)]=dict(family=family,name=name,journal=journal,spec=spec,
                proc=Adopted(dict(pid=row['worker_pid'],identity=ident)),sim_pid=ws['simulator_pid'],
                sim_ident=sim,limit=int(4.5*GIB),adopted=True,admitted=False,start=time.time())
        archived={}
        for family,name in FAILED:
            src=FAMILIES[family]/'runs'/name
            dest=BASE/'preserved'/family/name;dest.parent.mkdir(exist_ok=True)
            hashes={str(p.relative_to(src)):sha(p) for p in src.rglob('*') if p.is_file()}
            src.rename(dest)
            assert all(sha(dest/p)==h for p,h in hashes.items())
            archived[family]=dict(name=name,path=str(dest),hashes=hashes)
        save(BASE/'handoff.json',dict(pid=os.getpid(),old_pid=old_pid,old_identity=before,
            old_paused=True,time=time.time(),preserved_workers=pre['identities'],archived=archived))
        committed=True
    finally:
        if not committed:os.kill(old_pid,signal.SIGCONT)

    # The long-lived unrelated llama simulator gets the same higher allowance;
    # its9.5GiB worker+sim reservation is included above, with PID preserved.
    llama=[]
    for pid,p in owned().items():
        if p['name'].startswith('gem5') and 'llama_cpu_20260925' in p['cmd'] and p['state']!='Z':
            previous=resource.prlimit(pid,resource.RLIMIT_AS)
            constrain(pid,cfg['cpu_set'],SIM_AS)
            llama.append(dict(pid=pid,old_limit=previous,new_limit=SIM_AS))
    save(BASE/'llama_limit_change.json',llama)
    failed={};published={f for f,s in states.items() if s['stage']=='completed_validated'}
    last=None;retired=set()
    def update(metric):
        for family,out in FAMILIES.items():
            state=states[family];active={}
            for (f,name),r in running.items():
                if f!=family:continue
                ws=r['journal']/'worker_status.json';w=read(ws) if ws.exists() else {}
                pid=w.get('simulator_pid',r.get('sim_pid',r['proc'].pid))
                ident=process(pid)
                active[name]=dict(pid=pid,worker_pid=r['proc'].pid,start=r['start'],
                    kind='simulation',stage=('paused_resource' if not r['admitted'] else w.get('stage','starting')),
                    process_state=ident['state'] if ident else 'exited',simulator_as=r['limit'])
            state['active']=active;state['pending']=pending[family]
            state['stage']='failed_parallel' if family in failed else ('running_parallel' if pending[family] or active else 'completed_validated')
            if family in failed:state['error']=failed[family]
            state['updated']=time.time();save(out/'status.json',state)
            effects=publish_report(family,state,out)
            if state['stage']=='completed_validated' and family not in published:
                save(out/'complete.json',dict(cases=len(state['completed']),effects=effects,recovery=str(BASE)))
                with (ROOT/'docs/experiments/LEDGER.md').open('a') as f:
                    f.write('\n**OPS-21 '+family+' complete:** '+str(len(state['completed']))+' accepted cases; original acceptance unchanged. Effects '+json.dumps(effects)+'.\n')
                published.add(family)
        state=dict(pid=os.getpid(),stage='running_with_failed_queue' if failed else 'running',
             recovery=str(BASE),active=[dict(family=r['family'],name=r['name'],worker_pid=r['proc'].pid,
                 admitted=r['admitted']) for r in running.values()],
             remaining={f:len(j) for f,j in pending.items()},failed=failed,updated=time.time())
        save(BASE/'status.json',state);save(OLD/'status.json',state)
        save(BASE/'resources.json',metric);save(OLD/'resources.json',metric)

    while any(pending.values()) or running:
        now=owned();stamp=time.monotonic();cpu=None
        if last:
            delta=sum(max(0,p['cpu_ticks']-last[1].get((pid,p['start_ticks']),p['cpu_ticks'])) for pid,p in now.items())
            cpu=100*delta/os.sysconf('SC_CLK_TCK')/(stamp-last[0])/os.cpu_count()
        last=stamp,{(pid,p['start_ticks']):p['cpu_ticks'] for pid,p in now.items()}
        for key,r in list(running.items()):
            rc=r['proc'].poll()
            if rc is None:continue
            family=r['family'];out=FAMILIES[family];journal=r['journal']
            try:
                assert rc==0 and (journal/'done.json').exists(), read(journal/'failed.json') if (journal/'failed.json').exists() else ('worker exit',rc)
                row=read(journal/'validated.json');check_peers(family,row,states[family]['completed'])
                assert not (out/'runs'/r['name']/'accepted.json').exists()
                save(out/'runs'/r['name']/'accepted.json',row);states[family]['completed'].append(row)
                pending[family]=[j for j in pending[family] if j['name']!=r['name']]
                manifest=read(out/'inputs.json')
                for p,h in read(journal/'inputs.json').items():
                    assert p not in manifest or manifest[p]==h
                    manifest[p]=h
                save(out/'inputs.json',manifest)
            except BaseException as exc:
                failed[family]=repr(exc);save(journal/'recovery_publication_failed.json',dict(error=repr(exc)))
            del running[key]
        mem=memory();rss=sum(p['rss'] for p in now.values());disk=shutil.disk_usage(BASE).free
        pressure=(rss>=cfg['ram_limit']-4*GIB or mem['MemAvailable']<64*GIB or disk<100*GIB or (cpu is not None and cpu>20))
        reserved=base_reservation()+sum(WORKER_AS+r['limit'] for r in running.values())
        assert reserved<=cfg['ram_limit'],('reservation exceeds budget',reserved)
        # An existing paused worker cannot reset its simulator cap: it has
        # already reached its simulation wait loop. Raise the limit before resume.
        if not pressure:
            # Prefer retrying terminated cases as soon as a full slot fits.
            for family,name in FAILED:
                if family in failed or (family,name) in running or not any(j['name']==name for j in pending[family]):continue
                if reserved+WORKER_AS+SIM_AS>cfg['ram_limit']:continue
                old_spec=read(OLD/'workers'/family/name/'spec.json')
                journal=BASE/'workers'/family/name;journal.mkdir(parents=True)
                spec=dict(old_spec,journal=str(journal),adopted=None)
                spec['_spec_path']=str(journal/'spec.json');save(journal/'spec.json',spec)
                with (journal/'worker.log').open('x') as log:
                    proc=subprocess.Popen(['/usr/bin/python3',str(WORKER),str(journal/'spec.json')],cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                running[(family,name)]=dict(family=family,name=name,journal=journal,spec=spec,proc=proc,
                    limit=SIM_AS,adopted=False,admitted=True,start=time.time())
                reserved+=WORKER_AS+SIM_AS
            # Long-context parents closest to their old cap first, then oldest.
            waiting=sorted((r for r in running.values() if not r['admitted'] and r['family'] not in failed),
                           key=lambda r:(0 if r['name'].startswith('t16384') else 1,r['start']))
            for r in waiting:
                extra=SIM_AS-r['limit']
                if reserved+extra>cfg['ram_limit']:continue
                sim=process(r['sim_pid']);assert sim and sim['start_ticks']==r['sim_ident']['start_ticks']
                constrain(r['sim_pid'],cfg['cpu_set'],SIM_AS)
                r.update(limit=SIM_AS,admitted=True);reserved+=extra
        for r in running.values():
            ws=r['journal']/'worker_status.json';w=read(ws) if ws.exists() else {}
            sim_pid=w.get('simulator_pid',r.get('sim_pid'))
            pause=pressure or r['family'] in failed or not r['admitted']
            for pid in (sim_pid,r['proc'].pid):
                if not pid:continue
                ident=process(pid)
                if not ident or ident['state']=='Z':continue
                if pause and ident['state']!='T':os.kill(pid,signal.SIGSTOP)
                elif not pause and ident['state']=='T':os.kill(pid,signal.SIGCONT)
        # Original CPU-job coordinators must never resume and duplicate jobs.
        for parent in handoff['old_coordinators']:
            family=parent['family'];pid=parent['pid']
            if pid in retired:continue
            names=set(handoff['families'][family]['state']['active'])
            if names<={x['name'] for x in states[family]['completed']}:
                ident=process(pid)
                if ident and ident['start_ticks']==parent['identity']['start_ticks']:os.kill(pid,signal.SIGKILL)
                retired.add(pid)
        metric=dict(updated=time.time(),pid=os.getpid(),cpu_pct_of_host=cpu,user_rss_bytes=rss,
            user_ram_pct=100*rss/mem['MemTotal'],ram_limit_bytes=cfg['ram_limit'],
            reserved_bytes=reserved,logical_cpu_cap=len(cfg['cpu_set']),cpu_cap_pct=100*len(cfg['cpu_set'])/os.cpu_count(),
            available_bytes=mem['MemAvailable'],disk_free_bytes=disk,pressure=pressure,
            workers=len(running),admitted_workers=sum(r['admitted'] for r in running.values()),
            simulators=sum(p['name'].startswith('gem5') and p['state']!='Z' for p in now.values()),failed=failed)
        update(metric)
        with (BASE/'resource_history.jsonl').open('a') as f:f.write(json.dumps(metric)+'\n')
        time.sleep(5)
    save(BASE/'complete.json',dict(time=time.time(),failed=failed))
    save(OLD/'complete.json',dict(time=time.time(),failed=failed,recovery=str(BASE)))
    old_ident=process(old_pid)
    if old_ident and old_ident['start_ticks']==before['start_ticks']:os.kill(old_pid,signal.SIGKILL)
    # Recovery of AI-12 is prepared separately with unchanged scientific gates.
    remainder=BASE/'remainder_runner.py'
    if remainder.exists() and not failed:
        with (BASE/'remainder.log').open('x') as log:
            proc=subprocess.Popen(['/usr/bin/python3',str(remainder)],cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        save(BASE/'remainder_launch.json',dict(pid=proc.pid,time=time.time()))


if __name__=='__main__':
    try:
        if sys.argv[1:]==['--prepare']:prepare()
        else:main()
    except BaseException as exc:
        BASE.mkdir(exist_ok=True)
        save(BASE/'supervisor_failed.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
