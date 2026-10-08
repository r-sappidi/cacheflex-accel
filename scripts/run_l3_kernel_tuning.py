"""Ordered, fail-closed software tuning on the frozen L3 lifecycle simulator."""
import concurrent.futures as cf
import importlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback

ROOT=Path('/data2/rsappidi/cacheflex_mc')
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'benchmarks/cacheflex_flash_attention'),
             str(ROOT/'benchmarks/cacheflex_flash_attention/kv_partition')]
import numpy as np
from build_access_timing import sha,save
from run_access_timing import stats
from fa_headroom_profiles import leaves
from test_micro_timing import check_global_guard
from export_safe_elf import export_hash
from reference import reference,validate

BASE=ROOT/'results/cacheflex_l3_spm/kernel_tuning_20260928'
OUT=BASE/os.environ.get('CACHEFLEX_L3_KERNEL_EXPERIMENT','experiment_v1')
GUEST=BASE/os.environ.get('CACHEFLEX_L3_KERNEL_GUEST','guest_v2')
OLD=ROOT/'results/cacheflex_l3_spm/lifecycle_20260928/experiment_v4'
ENV=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib',
         OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')


def resources():
    def cpu():return list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:]))
    a=cpu();time.sleep(.5);b=cpu();d=[y-x for x,y in zip(a,b)]
    memory={k:int(v.split()[0]) for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())}
    simulators=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:cmd=(p/'cmdline').read_bytes().split(b'\0')
        except OSError:continue
        if cmd and cmd[0].endswith(b'gem5.opt'):simulators.append(int(p.name))
    return dict(cpu_idle_pct=100*(d[3]+d[4])/sum(d),load=os.getloadavg(),
        available_gib=memory['MemAvailable']/2**20,swap_used_gib=(memory['SwapTotal']-memory['SwapFree'])/2**20,
        free_disk_gib=shutil.disk_usage(BASE).free/2**30,simulators=simulators)


def case(t,arm,k,br=None,bc=None,r=None,depth=4):
    return dict(t=t,arm=arm,k=k,br=br or (32 if t==1024 else 16),
                bc=bc or (128 if t==1024 else 512),r=r or (192 if t==1024 else 256),depth=depth)


def label(c):
    return f"t{c['t']}_{c['arm']}_k{c['k']}_br{c['br']}_bc{c['bc']}_r{c['r']}_d{c['depth']}"


def main():
    OUT.mkdir(parents=True);(OUT/'runs').mkdir();(OUT/'references').mkdir()
    state=dict(pid=os.getpid(),stage='preparing',active={},completed=[])
    lock=threading.RLock();launch_lock=threading.Lock();stop=threading.Event()
    def report():
        lines=['# L3 software kernel tuning','',f"Stage: {state['stage']}",'',
               '| Case | ROI us | Updates | Copies | Exact output reference |',
               '|---|---:|---:|---:|---|']
        for row in state['completed']:
            lines.append(f"| {row['name']} | {row['roi_us']:.4f} | {row['commands']} | {row['copies']} | {row['exact_output_source']} |")
        lines += ['','Only fully validated cases above. Different Bc uses a separate independent oracle.',
                  'Fixed simulator/storage/queues/bandwidth. Same-Bc outputs must be bit-identical.',
                  'Depth1 is a correctness stress gate, excluded from performance selection.']
        for t in (1024,4096):
            byarm={}
            for arm in ('cache','private','l3'):
                rows=[x for x in state['completed'] if x['case']['t']==t and x['case']['arm']==arm and x['case']['depth']==4]
                if rows:byarm[arm]=min(rows,key=lambda x:x['roi_us'])
            if 'private' in byarm and 'l3' in byarm:
                a,b=byarm['private'],byarm['l3']
                lines += ['',f"T{t} best accepted private {a['roi_us']:.4f}us ({a['name']}); L3 {b['roi_us']:.4f}us ({b['name']}).",
                          f"Private/L3 speedup {a['roi_us']/b['roi_us']:.6f}x; L3 latency reduction {100*(1-b['roi_us']/a['roi_us']):.3f}%."]
        (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    def status(stage=None):
        with lock:
            if stage:state['stage']=stage
            state['updated']=time.time();save(OUT/'status.json',state);report()
    status()
    try:
        frozen=json.loads((GUEST/'frozen.json').read_text())
        # The old experiment manifest pins the simulator, model, and config tree.
        frozen.update(json.loads((OLD/'inputs.json').read_text()))
        for mod in ('reference','run_access_timing','test_micro_timing','fa_headroom_profiles','export_safe_elf','build_access_timing'):
            p=Path(importlib.import_module(mod).__file__);frozen[str(p)]=sha(p)
        for p in (Path(__file__),ROOT/'scripts/build_l3_kernel_tuning.py'):
            frozen[str(p)]=sha(p)
        for t in (1024,4096):
            for arm in ('cache','private_spm','init'):
                p=OLD/'runs'/f't{t}_{arm}'
                for n in ('accepted.json','command.json','config.json','output.fp16'):
                    frozen[str(p/n)]=sha(p/n)
        save(OUT/'inputs.json',frozen)
        def unchanged():
            for p,h in frozen.items():assert sha(p)==h,('changed frozen input',p)
        unchanged()
        save(OUT/'plan.json',dict(ledger=['FA-06','FA-07','FA-29','L3-02','L3-03','L3-04'],
            new='direct PV register epilogue; then L3-specific transfer of tile/group tuning',
            max_new_simulators=2,stages=['same-tile epilogue','Bc tuning','group and row scheduling'],
            primary='complete ROI; best L3 versus independently tuned private and stronger historical controls'))
        references={}
        def get_reference(t,bc):
            key=(t,bc)
            if key in references:return references[key]
            name=f't{t}_bc{bc}_parts1.npy'
            paths=list((ROOT/'results/cacheflex_flash_attention/finite_model_20260926').glob('**/references/'+name))
            if paths:
                p=paths[0];assert sha(p)==json.loads(p.with_suffix('.json').read_text())['sha256']
                dest=OUT/'references'/name;shutil.copy2(p,dest)
                save(dest.with_suffix('.json'),dict(source=str(p),sha256=sha(dest)))
            else:
                status(f'generating_reference_t{t}_bc{bc}')
                dest=OUT/'references'/name;np.save(dest,reference(t,bc,1))
                save(dest.with_suffix('.json'),dict(independent=True,sha256=sha(dest)))
            frozen[str(dest)]=sha(dest);save(OUT/'inputs.json',frozen)
            references[key]=np.load(dest);return references[key]
        def run(c):
            name=label(c)
            with lock:
                prior=next((x for x in state['completed'] if x['name']==name),None)
            if prior:return prior
            path=OUT/'runs'/name;path.mkdir();proc=None;start=time.monotonic()
            try:
                assert not stop.is_set();unchanged()
                t=c['t'];arm=c['arm'];spm=arm!='cache';reduction=arm=='l3'
                source=OLD/'runs'/f"t{t}_{'init' if reduction else ('private_spm' if spm else 'cache')}"
                args=list(map(str,(t,8,c['br'],c['bc'],c['r']//c['br'],1024,0,int(spm),5 if reduction else 1,c['k'])))
                assert c['r']%c['br']==0 and c['r']<=512
                cmd=json.loads((source/'command.json').read_text());cmd[1]='--outdir='+str(path)
                cmd[cmd.index('--cmd')+1]=str(GUEST/'guest.export-safe')
                cmd[cmd.index('--options')+1]=' '.join(args)
                for i,x in enumerate(cmd):
                    if x.startswith('system.cpu[0].workload[0].cmd='):cmd[i]=f"system.cpu[0].workload[0].cmd={['kv-fa']+args!r}"
                    if x.startswith('--cacheflex-l3-reduction-depth='):cmd[i]=f"--cacheflex-l3-reduction-depth={c['depth']}"
                    if '.l3ReductionQueue.buffer_size=' in x:cmd[i]=x.split('=')[0]+'='+str(c['depth'])
                save(path/'command.json',cmd);save(path/'case.json',c)
                with (path/'run.log').open('w') as log:
                    with launch_lock:
                        while True:
                            assert not stop.is_set()
                            resource=resources();save(path/'resource_launch.json',resource)
                            if resource['cpu_idle_pct']>=50 and resource['available_gib']>=128 and resource['free_disk_gib']>=100 and len(resource['simulators'])<6:break
                            with lock:state['resource_wait']=dict(job=name,resources=resource);status()
                            time.sleep(10)
                        proc=subprocess.Popen(['nice','-n','10',*cmd],cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT)
                        with lock:
                            state.pop('resource_wait',None);state['active'][name]=dict(pid=proc.pid,start=time.time());status()
                    while proc.poll() is None:
                        assert not stop.is_set(),'peer failed'
                        assert time.monotonic()-start<4*3600,'4h timeout'
                        assert shutil.disk_usage(OUT).free>75*2**30,'disk guard'
                        memory=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))
                        assert memory>64*2**20,'memory guard'
                        try:proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:status()
                logtext=(path/'run.log').read_text()
                assert proc.returncode==0 and 'KV_ROI_END' in logtext and 'because exiting with last active thread context' in logtext,logtext[-4000:]
                assert not re.search(r'panic:|fatal:|Simulated exit code not 0|\bFAIL\b',logtext),logtext[-4000:]
                def fields(p):return dict(leaves(json.loads((p/'config.json').read_text().replace(str(p),'<OUT>'))))
                old,new=fields(source),fields(path)
                delta={k:[old.get(k),new.get(k)] for k in old.keys()|new.keys() if old.get(k)!=new.get(k)}
                allowed=r'system.cpu\[\d+\].workload\[0\].(cmd\[\d+\]|executable)'
                if c['depth']==1:
                    allowed+=r'|system.ruby.l0_cntrl\d+.sequencer.cacheflex_reduction_(source|sink)_entries|system.ruby.l2_cntrl\d+.(l3_reduction_entries|l3ReductionQueue.buffer_size)'
                assert not {k:v for k,v in delta.items() if not re.fullmatch(allowed,k)},delta
                assert new['system.cpu[0].workload[0].executable']==str(GUEST/'guest.export-safe')
                assert [new[f'system.cpu[0].workload[0].cmd[{i}]'] for i in range(11)]==['kv-fa']+args
                epochs=stats(path);s=epochs[0];total=lambda suffix:sum(v for k,v in s.items() if k.endswith(suffix))
                commands=t*math.ceil(t/c['bc']) if reduction else 0
                copies=8*math.ceil((t//8)/c['r'])*t*2 if spm else 0
                assert total('.reductionAccepted')==total('.reductionCompleted')==commands
                assert total('.l3ReductionStarts')==total('.l3ReductionFinishes')==commands
                assert total('.l3ReductionHalfCommands')==commands
                assert total('.l3ReductionInitializations')==(t if reduction else 0)
                assert total('.l3Finalizations')==0
                assert total('.reductionTransportBytes')==144*commands
                assert total('.m_spmcpRequestsIssued')==total('.m_spmcpRequestsCompleted')==copies
                for ep in epochs:
                    check_global_guard(ep)
                    for i in range(8):
                        assert new[f'system.ruby.l2_cntrl{i}.L2cache.size']==524288
                        assert not new[f'system.ruby.l2_cntrl{i}.L2cache.spm_separate_backing']
                        for domain in ('regular','spm'):
                            for kind,cap in [('Load',32),('Store',48)]:assert ep[f'system.cpu{i}.{domain}Lsq0.boundedMax{kind}Occupancy']<=cap
                        for q in ('Source','Sink'):assert ep[f'system.ruby.l0_cntrl{i}.sequencer.reductionMax{q}']<=c['depth']
                        maximum=c['depth'] if new[f'system.ruby.l2_cntrl{i}.l3_reduction_pipeline'] else 1
                        assert ep[f'system.ruby.l2_cntrl{i}.L2cache.l3ReductionMaxActive']<=maximum
                export_hash(path/'output.fp16',logtext)
                arithmetic=validate(path/'output.fp16',references[(t,c['bc'])],t,dense=t==1024)
                exact_source=None
                if c['bc']==(128 if t==1024 else 512):exact_source=source/'output.fp16'
                else:
                    with lock:
                        peer=next((r for r in state['completed'] if r['case']['t']==t and r['case']['bc']==c['bc']),None)
                    if peer:exact_source=Path(peer['path'])/'output.fp16'
                if exact_source:assert sha(path/'output.fp16')==sha(exact_source),'same-Bc output bits changed'
                row=dict(name=name,case=c,path=str(path),roi_us=s['simTicks']/1e6,commands=commands,copies=copies,
                    arithmetic=arithmetic,exact_output_source=str(exact_source) if exact_source else 'first independent-checked case at this Bc',
                    output_sha256=sha(path/'output.fp16'),config_delta=delta,host_seconds=time.monotonic()-start,
                    source_full=total('.reductionSourceFull'),transport_bytes=total('.reductionTransportBytes'),
                    dram_write_bytes=total('.dram.bytesWritten::total'))
                unchanged();save(path/'accepted.json',row)
                with lock:state['completed'].append(row);status()
                return row
            except BaseException as exc:
                stop.set();save(path/'failed.json',dict(error=repr(exc),traceback=traceback.format_exc()));raise
            finally:
                if proc and proc.poll() is None:
                    proc.terminate()
                    try:proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()
                with lock:state['active'].pop(name,None);status()
        def phase(name,cases):
            for c in cases:get_reference(c['t'],c['bc'])
            status(name)
            with cf.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(run,cases))
            # Concurrent first-at-Bc completions must also agree exactly.
            for t,bc in {(c['t'],c['bc']) for c in cases}:
                hashes={r['output_sha256'] for r in state['completed'] if r['case']['t']==t and r['case']['bc']==bc}
                assert len(hashes)==1,('same-Bc output equality',t,bc)
        # Full FA tests validate actual epilogues including normalization/reuse.
        phase('phase1_t1024_controls',[case(1024,a,0) for a in ('private','l3','cache')])
        phase('phase1_t1024_direct',[case(1024,a,1) for a in ('private','l3','cache')])
        phase('phase1_one_entry',[case(1024,'l3',1,depth=1)])
        save(OUT/'gates_passed.json',dict(full_math=True,exact_output=True,one_entry=True,cases=len(state['completed'])))
        phase('phase1_t4096',[case(4096,a,k) for k in (0,1) for a in ('private','l3','cache')])
        save(OUT/'phase1_complete.json',dict(cases=len(state['completed'])))
        def best(arm):
            return min((r for r in state['completed'] if r['case']['t']==4096 and r['case']['arm']==arm and r['case']['depth']==4),key=lambda r:r['roi_us'])
        selected={a:best(a)['case']['k'] for a in ('private','l3')}
        phase('phase2_bc',[case(4096,a,selected[a],bc=bc) for bc in (256,1024) for a in ('private','l3')])
        save(OUT/'phase2_complete.json',{a:best(a) for a in ('private','l3')})
        choices={a:best(a)['case'] for a in ('private','l3')}
        phase('phase3_group',[case(4096,a,choices[a]['k'],br=br,bc=choices[a]['bc'],r=r)
                             for br,r in ((32,256),(16,128),(16,512)) for a in ('private','l3')])
        schedule=[]
        for a in ('private','l3'):
            c=best(a)['case'];schedule.extend([dict(c,k=k) for k in (1,2)])
        phase('phase3_row_order',schedule)
        status('completed_validated')
        save(OUT/'complete.json',dict(best={a:best(a) for a in ('cache','private','l3')},completed=len(state['completed'])))
    except BaseException as exc:
        state['error']=repr(exc);status('failed');raise


if __name__=='__main__':main()
