"""Gated, capacity-matched retry-isolation x double-buffering experiment."""
import concurrent.futures as cf
import configparser
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'benchmarks/cacheflex_flash_attention'))
from run_clean_controls import sha, save, OLD
from single_head_reference import check
from fa_export_patch import verify_dump
from validate_attribution import read
from run_single_diagnosis import prior
import numpy as np

OUT=ROOT/'results/cacheflex_flash_attention/async_staging_20260921'
EX=OUT/'experiment'
BUILD=OUT/'build_v2'
ENV=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
# Cap limits only bridges: all16 entries remain usable by normal traffic.
CASES={
 'cache':('cache',False,0,None),
 'cache_isolated':('cache',True,0,None),
 'spm':('spm',False,0,None),
 'spm_cap12':('spm',False,12,None),
 'spm_retry':('spm',True,0,None),
 'spm_isolated':('spm',True,12,None),
 'db_jit':('spm',False,0,0),
 'db_jit_isolated':('spm',True,12,0),
 'db_early':('spm',False,0,1),
 'db_early_isolated':('spm',True,12,1),
 'db_chunk':('spm',False,0,2),
 'db_chunk_isolated':('spm',True,12,2),
}


def ini(path):
    c=configparser.ConfigParser(interpolation=None,strict=False);c.read(path)
    return {s:dict(c[s]) for s in c.sections()}


def audit(old,p,guest,t,arch,isolated,cap,mode):
    expected=ini(old/'config.ini');actual=ini(p/'config.ini')
    args=['cacheflex-fa',str(t),'64','32' if arch=='cache' else '16','128']
    if mode is not None:args.append(str(mode))
    for s,values in expected.items():
        for k,v in list(values.items()):values[k]=v.replace(str(old),str(p))
        if re.fullmatch(r'system.cpu\d+',s):
            i=int(s[10:]); seq=f'system.ruby.l0_cntrl{i}.sequencer'
            values['cacheflexindependentbridge']=str(isolated).lower()
            values['bridge_port']=seq+'.in_ports[3]'
        if s.endswith('.mmu.walker'):
            values['port']=values['port'].replace('.in_ports[3]','.in_ports[4]')
        if re.fullmatch(r'system.ruby.l0_cntrl\d+.sequencer',s):
            i=int(re.search(r'l0_cntrl(\d+)',s)[1]);ports=values['in_ports'].split()
            ports.insert(3,f'system.cpu{i}.bridge_port');values['in_ports']=' '.join(ports)
            values['cacheflex_max_bridge_requests']=str(cap)
            values['cacheflex_fetch_credits']='0';values['cacheflex_bridge_credits']='0'
        if s=='system.cpu0.workload':
            values['cmd']=' '.join(args);values['executable']=str(guest)
        if re.fullmatch(r'system.ruby.l1_cntrl\d+.cache',s) and mode is not None:
            values['num_spm_ways']='4'
    differences={}
    for s in set(expected)|set(actual):
        for k in set(expected.get(s,{}))|set(actual.get(s,{})):
            a=expected.get(s,{}).get(k);b=actual.get(s,{}).get(k)
            if a!=b:differences[s+'.'+k]=[a,b]
    assert not differences,differences
    return dict(passed=True,scope='Only opt-in retry flag/bridge cap, extra port wiring, eligible ways and explicit guest argv/paths')


def main():
    EX.mkdir();(EX/'runs').mkdir();(EX/'references').mkdir();(EX/'preserved').mkdir()
    state=dict(pid=os.getpid(),completed=[],active={},stage='waiting_for_build')
    lock=threading.RLock();stop=threading.Event();results={}
    def status(stage=None):
        with lock:
            if stage:state['stage']=stage
            state['updated']=time.time();save(EX/'status.json',state)
    status()
    try:
        start=time.time()
        while not (BUILD/'simulator.sha256').exists():
            if (BUILD/'status.json').exists():assert json.loads((BUILD/'status.json').read_text())['stage']!='failed'
            assert time.time()-start<6*3600
            time.sleep(5)
        sim=BUILD/'gem5.opt';assert sha(sim)==(BUILD/'simulator.sha256').read_text().strip()
        guests={}
        for arch in ('cache','spm'):
            old=prior(arch,512);cmd=json.loads((old/'command.json').read_text())
            dst=EX/'preserved'/arch;shutil.copy2(cmd[cmd.index('--cmd')+1],dst);guests[arch]=dst
        guests['double']=OUT/'guests/double/guest.export-safe'
        refs={}
        for t in (512,4096,16384):
            src=(ROOT/'results/cacheflex_flash_attention/contention_optimization_v2_20260920/references/t4096_bc128.npy'
                 if t==4096 else OLD/'references'/f't{t}_bc128.npy')
            dst=EX/'references'/src.name;shutil.copy2(src,dst);refs[t]=np.load(dst)
        files=[Path(__file__),sim,*guests.values(),*list((BUILD/'configs').rglob('*.py')),
               *list((EX/'references').glob('*.npy'))]
        files += [Path(sys.modules[m].__file__) for m in ['run_clean_controls','single_head_reference','fa_export_patch','validate_attribution','run_single_diagnosis']]
        manifest={str(p):sha(p) for p in files};save(EX/'inputs.json',manifest)
        save(EX/'plan.json',dict(cases=CASES,screen_T=4096,gate_T=512,target_T=16384,
             cores=8,VL=16,d=64,Hq=1,Hkv=1,L1_KiB=64,L2_KiB=64,LLC_MiB=4,network_bits=128,
             SPM_ways_single=2,SPM_ways_double=4,total_sequencer_entries=16,bridge_cap=12,
             no_added_data_SRAM=True,no_extra_issue_bandwidth=True,no_dma_or_multicast=True,
             max_simulators=2,primary='Whole ROI latency',
             caveats=['SPMCP remains issued through ordered regular SQ; independent stores not fully decoupled',
                      'Independent retry is not independent downstream bandwidth',
                      'Four-way JIT is a storage-matched overlap control, not best cache baseline',
                      'Small/reference checks establish tested cases, not exhaustive protocol verification']))

        def frozen():
            for f,h in manifest.items():assert sha(Path(f))==h,('changed input',f)

        def report():
            lines=['# Async staging: correctness-gated end-to-end latency','',
                   '64KiB private L2 unchanged; double buffering claims4ways versus2 originally. No additional network bandwidth or data SRAM.','',
                   '| T | Case | ROI ms |','|---:|---|---:|']
            lines += [f"| {r['T']} | {r['name']} | {r['roi_us']/1000:.6f} |" for r in state['completed']]
            for t in (512,4096,16384):
                if (t,'cache') not in results or (t,'spm') not in results:continue
                cache=results[t,'cache']['counters']['roi_us'];spm=results[t,'spm']['counters']['roi_us']
                lines += ['',f'T{t} original SPM/cache speedup: {cache/spm:.6f}x.']
                for label in ('db_early_isolated','db_chunk_isolated'):
                    if (t,label) in results:
                        new=results[t,label]['counters']['roi_us']
                        lines += [f'{label}: {cache/new:.6f}x over full-credit cache, {100*(1-new/spm):+.3f}% lower latency than original SPM.']
            (EX/'REPORT.md').write_text('\n'.join(lines)+'\n')

        def run(t,name):
            frozen();arch,isolated,cap,mode=CASES[name];old=prior(arch,512)
            p=EX/'runs'/f'{name}_t{t}';p.mkdir();guest=guests[arch if mode is None else 'double']
            cmd=json.loads((old/'command.json').read_text());cmd[0]=str(sim)
            cmd[2]=str(BUILD/'configs/deprecated/example/se.py')
            cmd[cmd.index('--cmd')+1]=str(guest)
            args=['cacheflex-fa',str(t),'64','32' if arch=='cache' else '16','128']
            if mode is not None:args.append(str(mode))
            cmd=[f'--outdir={p}' if s.startswith('--outdir=') else
                 f'system.cpu[0].workload[0].cmd={args!r}' if s.startswith('system.cpu[0].workload[0].cmd=') else
                 '--l1d_num_spm_ways=4' if mode is not None and s.startswith('--l1d_num_spm_ways=') else s for s in cmd]
            for i in range(8):
                cmd+=['-P',f'system.cpu[{i}].cacheFlexIndependentBridge={isolated}',
                      '-P',f'system.ruby.l0_cntrl{i}.sequencer.cacheflex_max_bridge_requests={cap}']
            save(p/'command.json',cmd);start=time.monotonic();proc=None
            try:
                if stop.is_set():raise RuntimeError('peer failed')
                with (p/'run.log').open('w') as log:
                    proc=subprocess.Popen(cmd,env=ENV,stdout=log,stderr=subprocess.STDOUT)
                    with lock:state['active'][f'{name}_t{t}']=dict(pid=proc.pid,path=str(p));status()
                    audited=False
                    while proc.poll() is None:
                        save(p/'progress.json',dict(pid=proc.pid,host_seconds=time.monotonic()-start,updated=time.time()))
                        if stop.is_set():raise RuntimeError('peer failed')
                        if time.monotonic()-start>30*3600:raise RuntimeError('timeout')
                        if shutil.disk_usage(EX).free<50*1024**3:raise RuntimeError('disk guard')
                        if not audited and (p/'config.json').exists():
                            try:json.loads((p/'config.json').read_text())
                            except json.JSONDecodeError:pass
                            else:save(p/'config_audit.json',audit(old,p,guest,t,arch,isolated,cap,mode));audited=True
                        try:proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:pass
                assert proc.returncode==0,(p,proc.returncode)
                save(p/'config_audit.json',audit(old,p,guest,t,arch,isolated,cap,mode))
                output,counters,_=read(p,16)
                for k,v in dict(T=t,Hq=1,Hkv=1,d=64,cores=8,Bc=128,Br=32 if arch=='cache' else 16).items():assert output[k]==str(v),(k,output)
                count=(t//64)*t*2 if arch=='spm' else 0
                assert counters['m_spmcpRequestsIssued']==counters['m_spmcpRequestsCompleted']==count
                row=dict(path=str(p),name=name,T=t,output=output,counters=counters,host_seconds=time.monotonic()-start)
                verify_dump(row);row['validation']=check(p/'output.fp16',refs[t])
                if t==512 and name in ('cache','spm'):
                    original=json.loads((old/'accepted.json').read_text())
                    assert output==original['output']
                    assert {k:v for k,v in counters.items() if k!='path'}=={k:v for k,v in original['counters'].items() if k!='path'},'Default counter drift'
                    row['default_equivalence']='Exact extracted counters and output versus prior accepted control'
                with lock:
                    if (t,arch) in results:assert output['hash']==results[t,arch]['output']['hash'],'Output changed'
                frozen();save(p/'accepted.json',row)
                with lock:
                    results[t,name]=row;state['completed'].append(dict(name=name,T=t,roi_us=counters['roi_us'],path=str(p)))
                    state['active'].pop(f'{name}_t{t}',None);report();status()
            except BaseException as e:
                stop.set();save(p/'failed.json',dict(error=repr(e)))
                if proc is not None and proc.poll() is None:
                    proc.terminate()
                    try:proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()
                raise

        def batch(t,names):
            status(f't{t}_'+','.join(names))
            with cf.ThreadPoolExecutor(max_workers=2) as pool:
                jobs=[pool.submit(run,t,n) for n in names]
                for j in cf.as_completed(jobs):j.result()

        for t in (512,4096):
            batch(t,['cache','spm'])
            for names in [['cache_isolated','spm_cap12'],['spm_retry','spm_isolated'],
                          ['db_jit','db_jit_isolated'],['db_early','db_early_isolated'],['db_chunk','db_chunk_isolated']]:batch(t,names)
        winner=min(['db_early_isolated','db_chunk_isolated'],key=lambda n:results[4096,n]['counters']['roi_us'])
        save(EX/'promoted.json',dict(name=winner,selection='Minimum end-to-end T4096 latency among combined variants; fixed matching controls retained'))
        for names in [['cache','spm'],['db_jit','db_jit_isolated'],[winner.replace('_isolated',''),winner]]:batch(16384,names)
        status('completed_requires_analysis')
    except BaseException as e:
        stop.set();state['error']=repr(e);status('failed');raise


if __name__=='__main__':main()
