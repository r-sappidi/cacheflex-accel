"""Corrected MICRO single-core reconstruction, then fixed-workload strong scaling.

Two simulator slots. Fail closed on arithmetic, configuration or export failure.
Published best-case tiles are unavailable; this is a fixed repository candidate,
not a best-to-best sweep or exact Classic reproduction.
"""
import concurrent.futures as cf
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
import run_micro_paper_reproduction as oracle
from build_micro_head_scaling import ROOT, BASE, UP, build, sha
from run_clean_controls import save
from run_micro_resources import ini, numeric_stats

OUT=BASE/'micro_head_scaling_20260921'
MODEL=BASE/'micro_resources_20260921/build_v5'
CONTROL=BASE/'micro_resources_20260921/experiment_paper_geometry_v2/runs'
ENV=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib',
         OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
oracle.OUT=OUT
stop=threading.Event()
lock=threading.RLock()


def make_command(arch, mode, cores, t, hq, hkv, p, cap=1024):
    old=json.loads((CONTROL/f'{arch}_bounded_t4096/command.json').read_text())
    cmd=[];i=0
    args=['micro-fa',str(t),str(hq),str(hkv),'64','1','64','384','0',str(cores)]
    changes={'--num-cpus':cores,'--num-l2caches':8,'--num-dirs':2,
             '--topology':'MicroHeadMesh','--mesh-rows':2,'--mem-size':'16GiB',
             '--l1d_size':'512KiB','--l1d_num_spm_ways':4 if arch=='spm' else 0,
             '--l2_size':'512KiB'}
    while i<len(old):
        arg=old[i];i+=1
        if arg=='-P':
            value=old[i];i+=1
            m=re.search(r'system.cpu\[(\d+)\]|system.ruby.l[01]_cntrl(\d+)',value)
            if m and int(m[1] if m[1] is not None else m[2])>=cores:continue
            if value.startswith('system.cpu[0].workload[0].cmd='):value=f'system.cpu[0].workload[0].cmd={args!r}'
            if '.max_outstanding_requests=' in value:value=value.split('=')[0]+f'={cap}'
            cmd+=['-P',value];continue
        key=arg.split('=')[0]
        if key in changes:arg=f'{key}={changes[key]}'
        if arg.startswith('--outdir='):arg=f'--outdir={p}'
        cmd.append(arg)
    cmd[0]=str(MODEL/'gem5.opt')
    cmd[2]=str(OUT/'configs/deprecated/example/bounded_se.py')
    cmd[cmd.index('--cmd')+1]=str(OUT/'builds'/f'{arch}_{mode}'/'guest')
    cmd+=['--mem-channels=2','--mem-type=DDR4_2400_8x8']
    for i in range(cores):
        for fu in (0,2,5,7,9):cmd+=['-P',f'system.cpu[{i}].fuPool.FUList[{fu}].count=2']
    return cmd,args


def audit(p,arch,cores,args,cap):
    c=json.loads((p/'config.json').read_text());s=c['system'];r=s['ruby']
    assert len(s['cpu'])==cores
    fields=dict(fetchWidth=5,decodeWidth=5,renameWidth=5,commitWidth=5,
        dispatchWidth=8,issueWidth=8,wbWidth=8,numROBEntries=128,numIQEntries=80,
        LQEntries=32,SQEntries=48,cacheLoadPorts=2,cacheStorePorts=1,
        spmLoadPorts=1,spmStorePorts=1,cacheFlexBoundedTracking=False)
    for cpu in s['cpu']:
        for key,value in fields.items():assert cpu[key]==value,(key,cpu[key],value)
        assert cpu['isa'][0]['sve_vl_se']==16
        for index in (0,2,5,7,9):assert cpu['fuPool']['FUList'][index]['count']==2
    assert s['cpu'][0]['workload'][0]['cmd']==args
    pools={}
    for i in range(cores):
        l0=r[f'l0_cntrl{i}'];l1=r[f'l1_cntrl{i}']
        assert l0['Dcache']['size']==l0['Icache']['size']==65536
        assert l1['cache']['size']==524288 and l1['cache']['assoc']==8
        assert l1['cache']['num_spm_ways']==(4 if arch=='spm' else 0)
        seq=l0['sequencer'];assert seq['max_outstanding_requests']==cap
        assert seq['cacheflex_spm_ingress'] is not None
        for obj,n in ((l0['dataMissCredits'],8),(l0['instMissCredits'],8),
                      (l0['writeCredits'],8),(l1['missCredits'],20),(l1['writeCredits'],8)):
            assert obj['capacity']==n;pools[obj['path']]=n
    for i in range(8):
        llc=r[f'l2_cntrl{i}'];assert llc['L2cache']['size']==524288
        assert not llc['spm_retain_llc']
    for key,n in [('microLlcMissCredits',32),('microLlcWriteCredits',16)]:
        assert r[key]['capacity']==n;pools[r[key]['path']]=n
    assert len(r['network']['routers'])==8
    assert len(s['mem_ctrls'])==2
    assert all(m['dram']['type']=='DRAMInterface' and m['dram']['tCK']==833
               and m['dram']['bank_groups_per_rank']==4 and m['dram']['device_bus_width']==8
               for m in s['mem_ctrls'])
    assert s['mem_ranges']==['0:17179869184']
    # Compare every core's scalar timing/resource field with the frozen model,
    # allowing only declared paper FU counts and core-count-derived metadata.
    template=json.loads((CONTROL/f'{arch}_bounded_t4096/config.json').read_text())['system']['cpu'][0]
    for cpu in s['cpu']:
        for k,v in template.items():
            if isinstance(v,(int,float,bool)) and k not in ('cpu_id',):
                assert cpu[k]==v,(k,cpu[k],v)
    return dict(passed=True,cores=cores,fields=fields,pools=pools,LLC_bytes=4194304,
                DRAM='2xDDR4_2400_8x8,16GiB',fabric='Fixed eight-router mesh',
                limitations='Ruby prefetch/controller/link timing remains distinct from Classic')


def main():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'status.json').exists(),'Refuse to overwrite a previous queue'
    state=dict(pid=os.getpid(),stage='preparing',active={},completed=[])
    def status(stage=None):
        with lock:
            if stage:state['stage']=stage
            state['updated']=time.time();save(OUT/'status.json',state)
    status()
    try:
        if not (OUT/'builds/manifest.json').exists():build(OUT/'builds')
        shutil.copytree(MODEL/'configs',OUT/'configs',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        shutil.copy2(ROOT/'scripts/MicroHeadMesh.py',OUT/'configs/topologies/MicroHeadMesh.py')
        (OUT/'runs').mkdir();(OUT/'references').mkdir()
        deps=[MODEL/'gem5.opt',Path(__file__),ROOT/'scripts/build_micro_head_scaling.py',
              ROOT/'scripts/run_micro_paper_reproduction.py',ROOT/'scripts/run_micro_resources.py',
              ROOT/'scripts/run_micro_fa_ruby_pilot.py',ROOT/'scripts/revalidate_gcc15_export.py',
              ROOT/'benchmarks/cacheflex_flash_attention/fa_independent_reference.py']
        deps += [p for folder in ('builds','configs') for p in (OUT/folder).rglob('*') if p.is_file() and p.suffix!='.pyc']
        manifest={str(p):sha(p) for p in deps};save(OUT/'inputs.json',manifest)
        save(OUT/'plan.json',dict(Hq=32,Hkv=8,d=64,VL=16,Br=64,Bc=384,
            core_order=[1,4,8],T_order_at_8cores=[4096,8192,16384],max_simulators=2,
            precision='FP16 GEMM operands/accumulators; FP32 softmax/running output; FP16 stored output',
            source_commit='82ac0b9de1ed408ce5cd53067267c8a855b4eef3',
            stages=['Original corrected and common-harness T128 numerical gates at1/4/8 cores',
                    'Original corrected single-core T4096 pair','Common harness single-core T4096 pair',
                    'Same workload at4 cores','Same workload at8 cores','Eight-core T8192','Eight-core T16384'],
            policy='Fixed repository tile candidate, no winner search; no DMA/multicast/new backing/retention',
            caveats=['Published winning manifest unavailable; not unchanged paper reproduction',
                     'Original SPM VL16 stages V but consumes coherent V; preserved, not optimized here',
                     'SE flush is an upstream no-op; report first invocation, not guaranteed cold cache',
                     'Fixed shared4MiB LLC/two DRAM channels/eight-router fabric across core counts',
                     'SPM auto query grouping retained; 4 eligible ways in512KiB L2']))
        def frozen():
            for f,digest in manifest.items():assert sha(f)==digest,('changed dependency',f)
        def run(arch,mode,cores,t,gate=False):
            hq,hkv=32,8
            name=f'{arch}_{mode}_c{cores}_t{t}';p=OUT/'runs'/name;p.mkdir()
            cmd,args=make_command(arch,mode,cores,t,hq,hkv,p)
            save(p/'command.json',cmd);frozen();start=time.monotonic();proc=None
            try:
                with (p/'run.log').open('w') as log:
                    proc=subprocess.Popen(cmd,env=ENV,stdout=log,stderr=subprocess.STDOUT)
                    with lock:state['active'][name]=dict(pid=proc.pid,path=str(p));status()
                    audited=None
                    while proc.poll() is None:
                        if stop.is_set():raise RuntimeError('peer failed')
                        assert time.monotonic()-start<96*3600,'96h per-case timeout'
                        assert shutil.disk_usage(OUT).free>50*1024**3,'disk guard'
                        if audited is None and (p/'config.json').exists():
                            try:audited=audit(p,arch,cores,args,1024)
                            except json.JSONDecodeError:pass
                            if audited:save(p/'config_audit.json',audited)
                        save(p/'progress.json',dict(stage='simulating',pid=proc.pid,host_seconds=time.monotonic()-start,updated=time.time()))
                        try:proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:pass
                assert proc.returncode==0,(name,proc.returncode)
                audited=audit(p,arch,cores,args,1024);save(p/'config_audit.json',audited)
                stats=numeric_stats(p/'stats.txt');assert stats and stats[0]['simTicks']>0
                for epoch in stats:
                    for key,value in epoch.items():
                        if key.endswith('m_globalAdmissionFullRejects'):assert value==0,(key,value)
                    for resource,cap in audited['pools'].items():assert epoch.get(resource+'.maxLive',0)<=cap
                if mode=='parallel':
                    log=(p/'run.log').read_text(errors='replace')
                    for tid in range(cores):assert f'CF_DONE tid={tid} heads={hq//cores}' in log
                save(p/'progress.json',dict(stage='independent_numerical_reference',updated=time.time()))
                numerical=oracle.numerical(p,t,hq,hkv,384)
                # Same complete outputs, not just checksums, across scheduler mappings.
                reference=OUT/'runs'/f'{arch}_original_c1_t{t}'/'accepted.json'
                if reference.exists():
                    other=json.loads(reference.read_text())
                    assert numerical['hashes']==other['numerical']['hashes'],'Mapping changed numerical output'
                frozen()
                row=dict(arch=arch,mode=mode,cores=cores,T=t,gate=gate,path=str(p),
                         roi_us=stats[0]['simTicks']/1e6,numerical=numerical,host_seconds=time.monotonic()-start)
                save(p/'accepted.json',row)
                with lock:
                    state['active'].pop(name);state['completed'].append(row);status()
                    lines=['# MICRO-derived head scaling','',
                           'First invocation; corrected arithmetic; fixed Br64/Bc384. Not a tuned paper reproduction.','',
                           '| Mode | Cores | T | Cache us | SPM us | Speedup |','|---|---:|---:|---:|---:|---:|']
                    cases={(r['mode'],r['cores'],r['T']) for r in state['completed']}
                    for m,c,tok in sorted(cases):
                        pair={r['arch']:r for r in state['completed'] if (r['mode'],r['cores'],r['T'])==(m,c,tok)}
                        if len(pair)==2:
                            a,b=pair['cache']['roi_us'],pair['spm']['roi_us']
                            lines.append(f'| {m} | {c} | {tok} | {a:.4f} | {b:.4f} | {a/b:.6f} |')
                    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
                return row
            except BaseException as exc:
                stop.set();save(p/'failed.json',dict(error=repr(exc)))
                if proc and proc.poll() is None:
                    proc.terminate()
                    try:proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()
                raise
        for mode,cores,t,gate in [('original',1,128,True)]+[('parallel',c,128,True) for c in (1,4,8)]+[
                ('original',1,4096,False),('parallel',1,4096,False),('parallel',4,4096,False),
                ('parallel',8,4096,False),('parallel',8,8192,False),('parallel',8,16384,False)]:
            status(f'{mode}_c{cores}_t{t}')
            with cf.ThreadPoolExecutor(max_workers=2) as pool:
                futures=[pool.submit(run,a,mode,cores,t,gate) for a in ('cache','spm')]
                for f in cf.as_completed(futures):f.result()
        status('completed_requires_analysis')
    except BaseException as exc:
        stop.set();state['error']=repr(exc);status('failed');raise


if __name__=='__main__':main()
