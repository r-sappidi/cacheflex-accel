"""Adopt authorized queues, preserve acceptance, enforce a shared host budget.

No frozen simulator, guest, configuration, or acceptance helper is edited.
CPU affinity bounds the experiment group to 24 of 128 logical CPUs. Address
space reservations plus RLIMIT_AS bound worker/simulator allocations below
20% host RAM; all-UID RSS is also monitored, with an 8 GiB outside-work reserve.
This is not a privileged per-user cgroup and cannot constrain unrelated future
programs started outside this scheduler.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time
import traceback
from parallel_budget_worker import ROOT,GIB,read,save,sha,process,constrain

JOURNAL=ROOT/'results/parallel_budget_20260929'
WORKER=ROOT/'scripts/parallel_budget_worker.py'
FAMILIES={
 'diagnostics':ROOT/'results/cacheflex_l3_spm/diagnostics_20260929/experiment_v1',
 'long':ROOT/'results/cacheflex_l3_spm/qwen_long_context_20260928/experiment_v1',
 'ffn':ROOT/'results/ai_workloads/qwen_swiglu_20260928/experiment_v1',
 'projection':ROOT/'results/ai_workloads/qwen_projection_20260928/experiment_v2'}
ENV=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
         NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')


def memory():
    return {line.split(':')[0]:int(line.split()[1])*1024
            for line in Path('/proc/meminfo').read_text().splitlines()}


def owned():
    rows={}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit(): continue
        try:
            if p.stat().st_uid!=os.getuid(): continue
            fields={l.split(':',1)[0]:l.split(':',1)[1].strip() for l in (p/'status').read_text().splitlines()}
            cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
            ident=process(int(p.name))
            if not ident:continue
            rows[int(p.name)]={**ident,'rss':int(fields.get('VmRSS','0 kB').split()[0])*1024,
                'vms':int(fields.get('VmSize','0 kB').split()[0])*1024,
                'name':fields['Name'],'cmd':cmd}
        except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    return rows


def choose_cpus(n):
    def sample():
        return {int(l.split()[0][3:]):list(map(int,l.split()[1:9]))
                for l in Path('/proc/stat').read_text().splitlines()
                if l.startswith('cpu') and l.split()[0]!='cpu'}
    a=sample();time.sleep(1);b=sample();load=[]
    for cpu in os.sched_getaffinity(0):
        delta=[y-x for x,y in zip(a[cpu],b[cpu])]
        load.append((1-delta[3]/max(1,sum(delta)),cpu))
    result=[];cores=set()
    for _,cpu in sorted(load):
        p=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        core=((p/'physical_package_id').read_text(),(p/'core_id').read_text())
        if core in cores:continue
        cores.add(core);result.append(cpu)
        if len(result)==n:break
    assert len(result)==n
    return sorted(result)


def plans():
    result={}
    for family,out in FAMILIES.items():
        state=read(out/'status.json')
        assert (out/'gates_passed.json').exists(), ('gates incomplete',family)
        assert not (out/'failed.json').exists(),('failed queue',family)
        if family=='diagnostics':jobs=read(out/'plan.json')['targets']
        elif family=='long':jobs=[j for j in read(out/'plan.json') if not j['exact']]
        else:
            selected=read(out/'transfer_selection.json')['selected'];jobs=[]
            for t in (2048,4096):
                if family=='ffn':
                    for arm,cs in selected.items():
                        for c in cs:
                            c=dict(c,arm=arm,t=t,d=896,f=4864,stage=f'target_t{t}',
                                inp=str(out.parent/'inputs'/f't{t}_d896_f4864.bin'))
                            jobs.append(dict(name=f"target_t{t}_{arm}_t{t}_mp{c['mp']}_kc{c['kc']}_g{c['g']}",case=c))
                else:
                    for op,arms in selected.items():
                        for arm,cs in arms.items():
                            for c in cs:
                                c=dict(c,arm=arm,t=t,op=op,stage=f'target_t{t}',
                                    inp=str(out.parent/'inputs'/f'v2_{op}_t{t}.bin'))
                                jobs.append(dict(name=f"target_t{t}_{op}_{arm}_t{t}_mp{c['mp']}_kc{c['kc']}",case=c))
        completed={x['name'] for x in state['completed']}
        pending=[j for j in jobs if j['name'] not in completed]
        assert set(state['active']) <= {j['name'] for j in pending}
        result[family]=dict(out=str(out),jobs=pending,all_jobs=jobs,state=state)
    return result


def reference_for(family,job):
    out=FAMILIES[family]
    if family=='diagnostics':return None
    if family=='long':
        prior=read(Path(job['source'])/'accepted.json')['case']
        t=job['t'];bc=prior['bc']
        p=out/'references'/f't{t}_bc{bc}_q14_kv2_causal1.npy'
        return dict(path=str(p),t=t,bc=bc)
    c=job['case'];p=Path(c['inp'])
    prep=read(out.parent/('prepared.json' if family=='ffn' else 'prepared_v2.json'))
    return dict(path=str(p),t=c['t'],library=prep['library'],op=c.get('op'))


def reference_ready(ref,family):
    if ref is None:return True
    p=Path(ref['path']);meta=p.with_suffix('.json')
    if not p.exists() or not meta.exists():return False
    if family in ('ffn','projection') and not p.with_suffix('.stages.npz' if family=='ffn' else '.npz').exists():return False
    return True


def spec_for(family,job,cpus,journal,adopted=None):
    out=FAMILIES[family];name=job['name']
    return dict(family=family,name=name,job=job.get('case',job),out=str(out),
                journal=str(journal),manifest=str(out/'inputs.json'),cpus=cpus,
                build=str(out.parent/'build_v2'),adopted=adopted)


def check_peers(family,row,rows):
    if family=='diagnostics':return
    keys={'long':('t','bc'),'ffn':('t','d','f'),'projection':('t','op')}[family]
    for peer in rows:
        if all(peer['case'][k]==row['case'][k] for k in keys):
            assert peer['output_sha256']==row['output_sha256'],('peer output mismatch',row['name'],peer['name'])


def publish_report(family,state,out):
    rows=state['completed'];effects=[]
    if family in ('ffn','projection'):
        lines=['# Accepted component results — parallel resource-bounded continuation','',
            'First call includes packing; second call is a repeat diagnostic, not a stable warm median.','',
            '| Case | First full ROI us | Second call us |','|---|---:|---:|']
        for r in rows:lines.append(f"| {r['name']} | {r['first_us']:.4f} | {r['repeat_us']:.4f} |")
        for t in (1024,2048,4096):
            for op in (('qkv','out') if family=='projection' else (None,)):
                selected={a:[r for r in rows if r['case']['t']==t and r['case']['arm']==a
                             and (op is None or r['case']['op']==op)] for a in ('cache','spm')}
                # Only publish a pair when ALL previously selected cases for its
                # token count/operator have completed, preserving tuning scope.
                plan=read(JOURNAL/'handoff.json')['families'][family]['all_jobs']
                required=[j['name'] for j in plan if j['case']['t']==t and (op is None or j['case']['op']==op)]
                if not all(selected.values()) or not set(required)<={r['name'] for r in rows}:continue
                best={a:min(rs,key=lambda r:r['first_us']) for a,rs in selected.items()}
                effects.append(dict(t=t,**({'op':op} if op else {}),
                    cache=best['cache']['name'],spm=best['spm']['name'],
                    cache_us=best['cache']['first_us'],spm_us=best['spm']['first_us'],
                    speedup=best['cache']['first_us']/best['spm']['first_us']))
    else:
        lines=['# Accepted L3 results — parallel resource-bounded continuation','',
            '| Case | Full ROI us |','|---|---:|']
        for r in rows:lines.append(f"| {r['name']} | {r['roi_us']:.4f} |")
        if family=='diagnostics':
            direct=ROOT/'results/cacheflex_l3_spm/direct_link_20260928/experiment_v2/runs'
            private=read(direct/'t4096_private/accepted.json')['roi_us'];cache=read(direct/'t4096_cache/accepted.json')['roi_us']
            for r in rows:
                if r['name'].startswith('t4096_'):
                    effects.append(dict(name=r['name'],roi_us=r['roi_us'],source_us=r['prior_us'],
                       latency_reduction_pct=100*(1-r['roi_us']/r['prior_us']),
                       speedup_vs_private=private/r['roi_us'],speedup_vs_cache=cache/r['roi_us']))
            lines+=['','Fast read/reply are optimistic diagnostics, not calibrated hardware.',
                    'Narrow-engine timing is not area evidence; trace samples are not runtime fractions.']
        else:
            byname={r['name']:r for r in rows}
            for t in (16384,32768):
                for mode in ('direct','noc','private'):
                    for baseline in ('noc','private','cache'):
                        if mode==baseline or (mode=='private' and baseline!='cache'):continue
                        d=byname.get(f't{t}_{mode}');b=byname.get(f't{t}_{baseline}')
                        if d and b:effects.append(dict(t=t,mode=mode,baseline=baseline,
                          baseline_us=b['roi_us'],direct_us=d['roi_us'],speedup=b['roi_us']/d['roi_us'],
                          latency_reduction_pct=100*(1-d['roi_us']/b['roi_us'])))
            lines+=['','Transferred T4096 tiling, not independently tuned at long context.']
    lines+=['',f"Stage: {state['stage']}",'Full original arithmetic/configuration/count/resource checks retained.',
            'Model-shaped component results, not full-model inference.',
            f'Parallel scheduler journal: {JOURNAL}']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n');save(out/'effects.json',effects)
    return effects


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    JOURNAL.mkdir(exist_ok=True)
    lock=(JOURNAL/'scheduler.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.prepare:
        assert not (JOURNAL/'handoff.json').exists()
        snapshot=plans();n=min(24,math.floor(os.cpu_count()*.20)-1)
        cfg=dict(cpu_fraction=.20,ram_fraction=.20,logical_cpus=os.cpu_count(),
                 cpu_set=choose_cpus(n),mem_total=memory()['MemTotal'],
                 ram_limit=math.floor(memory()['MemTotal']*.20),
                 worker_as=int(1.5*GIB),simulator_as=int(4.5*GIB),
                 outside_reserve=8*GIB,max_worker_slots=n-1,
                 explanation='Reserve one CPU for existing llama; all user RSS monitored; no delegated cgroup.')
        save(JOURNAL/'budget.json',cfg);save(JOURNAL/'preflight.json',snapshot)
        print(json.dumps(cfg));return
    assert not (JOURNAL/'handoff.json').exists(),'Already handed off; do not duplicate scheduler'
    cfg=read(JOURNAL/'budget.json');cpus=set(cfg['cpu_set'])
    constrain(os.getpid(),cpus,GIB)
    current=plans()
    # Verify all frozen files before stopping any parent.
    for family,item in current.items():
        manifest=read(Path(item['out'])/'inputs.json')
        for p,h in manifest.items():assert sha(p)==h,('changed input',family,p)
    old=[];committed=False
    try:
        for family,item in current.items():
            pid=item['state']['pid'];ident=process(pid);assert ident and ident['state']!='Z'
            old.append(dict(family=family,pid=pid,identity=ident))
            os.kill(pid,signal.SIGSTOP)
        for parent in old:
            deadline=time.time()+5
            while process(parent['pid'])['state']!='T':
                assert time.time()<deadline;time.sleep(.02)
        # Re-read after quiescence; never rely on a racing pre-stop snapshot.
        current=plans()
        for family,item in current.items():
            out=Path(item['out']);save(JOURNAL/f'original_{family}_status.json',item['state'])
            for name,row in item['state']['active'].items():
                ident=process(row['pid']);assert ident and ident['ppid']==item['state']['pid']
                row['identity']=ident
        handoff=dict(pid=os.getpid(),started=time.time(),families=current,old_coordinators=old,
                     budget=cfg,scope='Scheduling only; no new experiment configurations',
                     related_ledger=['MOD-10','L3-10','AI-10','AI-11'])
        save(JOURNAL/'handoff.json',handoff);committed=True
    finally:
        if not committed:
            for parent in old:
                now=process(parent['pid'])
                if now and now['start_ticks']==parent['identity']['start_ticks']:os.kill(parent['pid'],signal.SIGCONT)

    states={f:dict(v['state'],pid=os.getpid(),previous_coordinator=v['state']['pid'],
                 active={},handoff=str(JOURNAL)) for f,v in current.items()}
    pending={f:list(v['jobs']) for f,v in current.items()}
    running={};done_refs=set();failed={};retired=set();published=set();last_cpu=None
    # Preserve the unrelated llama coordinator but bound it and its child.
    external={}
    for pid,row in owned().items():
        if 'restart_interrupted_20260928.py llama' in row['cmd'] or (row['name'].startswith('gem5') and
             'llama_cpu_20260925' in row['cmd']):
            limit=int(4.5*GIB) if row['name'].startswith('gem5') else int(1.5*GIB)
            external[pid]=dict(identity=row,old_affinity=sorted(os.sched_getaffinity(pid)),
                old_limit=resource.prlimit(pid,resource.RLIMIT_AS),limit=limit)
            constrain(pid,cpus,limit)
    save(JOURNAL/'external_constraints.json',external)

    def update():
        for family,out in FAMILIES.items():
            state=states[family];active={}
            for key,r in running.items():
                if r['family']!=family:continue
                ws=Path(r['spec']['journal'])/'worker_status.json'
                w=read(ws) if ws.exists() else {}
                active[r['name']]=dict(pid=w.get('simulator_pid',r['proc'].pid),worker_pid=r['proc'].pid,
                                      start=r['start'],kind='reference' if r.get('reference') else 'simulation',
                                      stage=w.get('stage','starting'))
            state['active']=active;state['pending']=[j for j in pending[family]]
            state['stage']='failed_parallel' if family in failed else ('running_parallel' if active or pending[family] else 'completed_validated')
            state['updated']=time.time()
            if family in failed:state['error']=failed[family]
            save(out/'status.json',state);effects=publish_report(family,state,out)
            if state['stage']=='completed_validated' and family not in published:
                save(out/'complete.json',dict(state,cases=len(state['completed']),effects=effects))
                with (ROOT/'docs/experiments/LEDGER.md').open('a') as f:
                    f.write('\n**Parallel-budget completion '+family+':** All '+str(len(state['completed']))+
                            ' cases accepted using unmodified original acceptance checks. Effects '+json.dumps(effects)+
                            '. Scheduling-only continuation: results/parallel_budget_20260929.\n')
                published.add(family)

    while pending or running:
        now=owned();nowtime=time.monotonic();ticks=sum(p['cpu_ticks'] for p in now.values())
        cpu_pct=None
        if last_cpu:
            elapsed=nowtime-last_cpu[0]
            # Match identities so exited processes never create negative CPU deltas.
            delta=sum(max(0,p['cpu_ticks']-last_cpu[1].get((pid,p['start_ticks']),p['cpu_ticks'])) for pid,p in now.items())
            cpu_pct=100*delta/os.sysconf('SC_CLK_TCK')/elapsed/os.cpu_count()
        last_cpu=(nowtime,{(pid,p['start_ticks']):p['cpu_ticks'] for pid,p in now.items()})
        rss=sum(p['rss'] for p in now.values());mem=memory();disk=shutil.disk_usage(JOURNAL).free
        # Reap/validate workers before admitting replacements.
        for key,r in list(running.items()):
            rc=r['proc'].poll()
            if rc is None:continue
            journal=Path(r['spec']['journal']);family=r['family'];out=FAMILIES[family]
            try:
                assert rc==0 and (journal/'done.json').exists(), read(journal/'failed.json') if (journal/'failed.json').exists() else ('worker exit',rc)
                if r.get('reference'):
                    done_refs.add(r['reference']['path'])
                else:
                    row=read(journal/'validated.json');check_peers(family,row,states[family]['completed'])
                    assert not (out/'runs'/r['name']/'accepted.json').exists(),'duplicate acceptance'
                    save(out/'runs'/r['name']/'accepted.json',row)
                    states[family]['completed'].append(row)
                    pending[family]=[j for j in pending[family] if j['name']!=r['name']]
                    manifest=read(out/'inputs.json')
                    for p,h in read(journal/'inputs.json').items():
                        assert p not in manifest or manifest[p]==h
                        manifest[p]=h
                    save(out/'inputs.json',manifest)
            except BaseException as exc:
                failed[family]=repr(exc);save(journal/'publication_failed.json',dict(error=repr(exc)))
            del running[key]
        # Fail closed per experiment: pause peer workers/simulators, preserve
        # partials, and leave other independently validated queues progressing.
        for r in running.values():
            if r['family'] not in failed:continue
            ws=Path(r['spec']['journal'])/'worker_status.json'
            if ws.exists():
                w=read(ws);pid=w.get('simulator_pid');ident=w.get('simulator_identity')
                p=process(pid) if pid else None
                if p and ident and p['start_ticks']==ident['start_ticks'] and p['state'] not in ('T','Z'):os.kill(pid,signal.SIGSTOP)
            p=process(r['proc'].pid)
            if p and p['state'] not in ('T','Z'):os.kill(r['proc'].pid,signal.SIGSTOP)
        # Keep frozen parents stopped until their original children have exited,
        # then retire them so they cannot launch duplicate jobs.
        for parent in old:
            family=parent['family']
            if family in retired:continue
            adopted=current[family]['state']['active']
            if all(any(x['name']==name for x in states[family]['completed']) for name in adopted):
                ident=process(parent['pid'])
                if ident and ident['start_ticks']==parent['identity']['start_ticks']:os.kill(parent['pid'],signal.SIGKILL)
                retired.add(family);save(JOURNAL/f'retired_{family}.json',dict(pid=parent['pid'],time=time.time()))
        sim_count=sum(p['name'].startswith('gem5') and p['state']!='Z' for p in now.values())
        external_reserved=sum(r['limit'] for pid,r in external.items() if pid in now and now[pid]['state']!='Z')
        reserved=cfg['outside_reserve']+external_reserved+len(running)*6*GIB
        pressure=rss>=cfg['ram_limit']-4*GIB or mem['MemAvailable']<64*GIB or disk<100*GIB
        metric=dict(updated=time.time(),pid=os.getpid(),cpu_pct_of_host=cpu_pct,
                    user_rss_bytes=rss,user_ram_pct=100*rss/mem['MemTotal'],
                    logical_cpu_cap=len(cpus),cpu_cap_pct=100*len(cpus)/os.cpu_count(),
                    ram_limit_bytes=cfg['ram_limit'],reserved_bytes=reserved,
                    workers=len(running),simulators=sim_count,available_bytes=mem['MemAvailable'],
                    disk_free_bytes=disk,pressure=pressure,failed=failed)
        save(JOURNAL/'resources.json',metric)
        with (JOURNAL/'resource_history.jsonl').open('a') as f:f.write(json.dumps(metric)+'\n')
        if pressure or (cpu_pct is not None and cpu_pct>20):
            # CPU affinity is already a hard experiment-group ceiling. This
            # additional guard accounts for outside processes owned by this UID.
            for r in running.values():
                if r['family'] in failed:continue
                ws=Path(r['spec']['journal'])/'worker_status.json'
                if ws.exists():
                    w=read(ws);pid=w.get('simulator_pid');p=process(pid) if pid else None
                    if p and p['state'] not in ('T','Z'):os.kill(pid,signal.SIGSTOP)
                if process(r['proc'].pid)['state']!='T':os.kill(r['proc'].pid,signal.SIGSTOP)
                r['budget_paused']=True
        else:
            for r in running.values():
                if r.pop('budget_paused',False) and r['family'] not in failed:
                    ws=Path(r['spec']['journal'])/'worker_status.json'
                    if ws.exists():
                        pid=read(ws).get('simulator_pid');p=process(pid) if pid else None
                        if p and p['state']=='T':os.kill(pid,signal.SIGCONT)
                    os.kill(r['proc'].pid,signal.SIGCONT)
        if not pressure:
            # Prioritize adoption, then short L3 diagnostics, then round-robin
            # existing target queues; a case cannot be admitted twice.
            ordered=[]
            for family,jobs in pending.items():
                if family in failed:continue
                for j in jobs:
                    if (family,j['name']) in running:continue
                    adopted=current[family]['state']['active'].get(j['name'])
                    ordered.append((0 if adopted else 1 if family=='diagnostics' else 2,
                                    jobs.index(j),family,j,adopted))
            for _,__,family,job,adopted in sorted(ordered,key=lambda x:x[:3]):
                if len(running)>=cfg['max_worker_slots']:break
                if cfg['outside_reserve']+external_reserved+(len(running)+1)*6*GIB>cfg['ram_limit']:break
                ref=reference_for(family,job)
                reference=None;name=job['name']
                if not reference_ready(ref,family):
                    name='reference_'+Path(ref['path']).stem;reference=ref
                    if (family,name) in running:continue
                key=(family,name)
                journal=JOURNAL/'workers'/family/name;journal.mkdir(parents=True,exist_ok=False)
                spec=spec_for(family,job,sorted(cpus),journal,adopted)
                if reference:spec['reference']=reference
                spec['_spec_path']=str(journal/'spec.json');save(journal/'spec.json',spec)
                with (journal/'worker.log').open('w') as log:
                    proc=subprocess.Popen(['/usr/bin/python3',str(WORKER),str(journal/'spec.json')],
                       cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                running[key]=dict(proc=proc,family=family,name=name,spec=spec,reference=reference,start=time.time())
                if reference:continue
        update()
        if all(not jobs for jobs in pending.values()) and not running:break
        save(JOURNAL/'status.json',dict(pid=os.getpid(),stage='running' if not failed else 'running_with_failed_queue',
              active=[dict(family=r['family'],name=r['name'],worker_pid=r['proc'].pid) for r in running.values()],
              remaining={f:len(j) for f,j in pending.items()},failed=failed,updated=time.time()))
        time.sleep(5)
    save(JOURNAL/'complete.json',dict(pid=os.getpid(),time=time.time(),failed=failed))


if __name__=='__main__':
    try:main()
    except BaseException as e:
        JOURNAL.mkdir(exist_ok=True);save(JOURNAL/'scheduler_failed.json',dict(error=repr(e),traceback=traceback.format_exc()))
        raise
