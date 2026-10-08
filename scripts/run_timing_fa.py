"""Verify MICRO timing profile with ordering tests and unchanged FA guests.

No other experiment inputs are edited. Full-size execution requires an
explicit calibration record, not just a successful simulator invocation.
"""
import concurrent.futures as cf
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from build_access_timing import ROOT, sha, save
from run_access_timing import stats
sys.path.insert(0, str(ROOT/'benchmarks/cacheflex_flash_attention'))
from single_head_reference import check
import numpy as np

BASE=ROOT/'results/cacheflex_flash_attention'
OUT=BASE/'timing_alignment_20260922'
OLD=BASE/'demand_isolation_20260922/experiment'
EX=OUT/os.environ.get('CACHEFLEX_TIMING_FA_EXPERIMENT','fa_v3')
SIM=BASE/'demand_isolation_20260922/build/gem5.opt'
REF=BASE/'micro_resources_20260921/experiment_v6/preserved'
ENV=dict(os.environ, LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib',
         OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
for k in list(ENV):
    if k.startswith('CACHEFLEX_AGGREGATE') or k=='CACHEFLEX_TRACE_LINE': ENV.pop(k)

def config_audit(old, new, profile=True):
    """Compare all fields, allowing only the enumerated timing translation."""
    changes=[]
    def walk(a,b,path=''):
        if isinstance(a,dict):
            assert isinstance(b,dict) and set(a)==set(b),(path,set(a)^set(b))
            for k in a: walk(a[k],b[k],path+'.'+k)
        elif isinstance(a,list):
            assert len(a)==len(b),path
            for i,(x,y) in enumerate(zip(a,b)):walk(x,y,path+f'[{i}]')
        elif a!=b:
            if isinstance(a,str) and a.replace(str(old),'<OUT>')==b.replace(str(new),'<OUT>'):return
            if path=='.system.cpu[0].workload[0].executable':
                assert Path(a).name==Path(b).name and Path(b).parent==OUT/'fa_export_guests'
                return
            allowed=False
            for core in range(8):
                l0=f'.system.ruby.l0_cntrl{core}';l1=f'.system.ruby.l1_cntrl{core}'
                fields={l0+'.clk_domain':'system.cpu_clk_domain',
                    l1+'.clk_domain':'system.cpu_clk_domain',
                    l0+'.mandatory_queue_latency':1,l0+'.request_latency':2,
                    l1+'.l1_response_latency':11,l1+'.l1_request_latency':8,
                    l0+'.spm_load_request_latency':1,l1+'.spm_access_response_latency':5,
                    l0+'.spm_request_latency':1,
                    l0+'.spm_copy_request_latency':4,l1+'.spm_response_latency':8,
                    l0+'.bufferToL1.clamp_fifo_arrival':True,
                    l0+'.bufferFromL1.clamp_fifo_arrival':True}
                allowed |= profile and path in fields and b==fields.get(path)
            for bank in range(8):
                fields={f'.system.ruby.l2_cntrl{bank}.clk_domain':'system.cpu_clk_domain',
                    f'.system.ruby.l2_cntrl{bank}.l2_request_latency':35,
                    f'.system.ruby.l2_cntrl{bank}.l2_response_latency':35}
                allowed |= profile and path in fields and b==fields.get(path)
            assert allowed,(path,a,b)
            changes.append(dict(path=path,old=a,new=b))
    walk(json.loads((old/'config.json').read_text()),json.loads((new/'config.json').read_text()))
    save(new/'config_changes.json',changes)
    return changes

def main():
    EX.mkdir()
    shutil.copytree(OUT/'configs', EX/'configs')
    state=dict(pid=os.getpid(),stage='initializing',active={},completed=[])
    lock=threading.RLock();stop=threading.Event()
    def status(stage=None):
        with lock:
            if stage:state['stage']=stage
            state['updated']=time.time();save(EX/'status.json',state)
    guests=[BASE/'micro_resources_20260921/roundtrip_guest/guest',
            BASE/'micro_resources_20260921/roundtrip_raw_guest/guest',
            ROOT/'results/ruby_getx_race_guest_20260916/guest']
    paths=[SIM,Path(__file__).resolve(),*guests,
           ROOT/'benchmarks/cacheflex_flash_attention/single_head_reference.py']
    paths+=list((OUT/'fa_export_guests').glob('*'))
    paths+=list((EX/'configs').rglob('*.py'))
    for t in (512,4096):
        paths.append(REF/f't{t}_bc128.npy')
        for arch in ('cache','spm'):
            old=OLD/f'{arch}_t{t}_off';cmd=json.loads((old/'command.json').read_text())
            paths.extend([old/'command.json',old/'config.json',old/'output.fp16',
                          Path(cmd[cmd.index('--cmd')+1])])
    manifest={str(p):sha(p) for p in paths};save(EX/'inputs.json',manifest)
    def frozen():
        for p,h in manifest.items():assert sha(p)==h,('Changed frozen input',p)
    def command(old,p,profile=True):
        cmd=json.loads((old/'command.json').read_text())
        cmd=[f'--outdir={p}' if v.startswith('--outdir=') else v for v in cmd]
        cmd[0]=str(SIM);cmd[2]=str(EX/'configs/deprecated/example/micro_timing_se.py')
        return cmd+(['--cacheflex-micro-timing'] if profile else [])
    def execute(p,cmd):
        frozen();save(p/'command.json',cmd);proc=None;start=time.monotonic()
        try:
            with (p/'run.log').open('w') as log:
                proc=subprocess.Popen(cmd,cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT)
                with lock:state['active'][p.name]=dict(pid=proc.pid,path=str(p));status()
                while proc.poll() is None:
                    assert not stop.is_set(),'Peer failed'
                    assert time.monotonic()-start<10800,'Three-hour case cap'
                    assert shutil.disk_usage(EX).free>50*2**30,'Disk guard'
                    try:proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:pass
            assert proc.returncode==0,(p.name,proc.returncode)
            frozen()
            with lock:state['active'].pop(p.name);status()
            return time.monotonic()-start
        except BaseException:
            stop.set()
            if proc and proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
            raise
    def accept(p,row):
        save(p/'validated.json',row)
        with lock:state['completed'].append(row);status()
    try:
        status('ordering_regressions')
        for name,g,marker in zip(('roundtrip','roundtrip_raw','release_getx'),guests,
            ('SPM_ROUNDTRIP PASS errors=0','SPM_ROUNDTRIP PASS errors=0','SPM_RELEASE_GETX PASS errors=0')):
            p=EX/name;p.mkdir();cmd=command(OLD/'spm_t512_off',p)
            cmd[cmd.index('--cmd')+1]=str(g)
            cmd=["system.cpu[0].workload[0].cmd=['spm-smoke']" if v.startswith('system.cpu[0].workload[0].cmd=')
                 else v.replace('sve_vl_se=16','sve_vl_se=4') for v in cmd]
            seconds=execute(p,cmd);assert marker in (p/'run.log').read_text(),name
            accept(p,dict(name=name,kind='ordering',passed=True,host_seconds=seconds))
        def fa(arch,t,profile=True):
            p=EX/(f'{arch}_t{t}'+('' if profile else '_export_control'));p.mkdir();old=OLD/f'{arch}_t{t}_off'
            cmd=command(old,p,profile);cmd[cmd.index('--cmd')+1]=str(OUT/'fa_export_guests'/arch)
            seconds=execute(p,cmd);changes=config_audit(old,p,profile)
            assert sha(p/'output.fp16')==sha(old/'output.fp16'),'Output changed'
            arithmetic=check(p/'output.fp16',np.load(REF/f't{t}_bc128.npy'))
            s=stats(p)[0];expected=(t//64)*t*2 if arch=='spm' else 0
            if not profile:assert s==stats(old)[0], 'Export repair changed timed ROI stats'
            for suffix in ('.m_spmcpRequestsIssued','.m_spmcpRequestsCompleted'):
                assert sum(v for k,v in s.items() if k.endswith(suffix))==expected
            assert sum(v for k,v in s.items() if k.endswith('.m_GlobalAdmissionFull'))==0
            accept(p,dict(name=p.name,kind='fa',arch=arch,T=t,passed=True,roi_us=s['simTicks']/1e6,
                timing_profile=profile,exact_roi_stats=(not profile),
                old_roi_us=stats(old)[0]['simTicks']/1e6,host_seconds=seconds,
                arithmetic=arithmetic,exact_output=True,config_audit=True,changed_fields=len(changes),copies=expected))
        status('export_equivalence_controls')
        with cf.ThreadPoolExecutor(max_workers=2) as pool:
            fs=[pool.submit(fa,a,512,False) for a in ('cache','spm')]
            for f in cf.as_completed(fs):f.result()
        for t in (512,4096):
            if t==4096:
                status('waiting_for_calibration')
                record=OUT/'calibration.json'
                while not record.exists():time.sleep(5)
                assert json.loads(record.read_text())['passed'], 'Timing calibration failed'
                save(EX/'calibration.json',json.loads(record.read_text()))
            status(f't{t}_pair')
            with cf.ThreadPoolExecutor(max_workers=2) as pool:
                fs=[pool.submit(fa,a,t) for a in ('cache','spm')]
                for f in cf.as_completed(fs):f.result()
        status('completed_validated')
    except BaseException as exc:
        stop.set();state['error']=repr(exc);status('failed');raise

if __name__=='__main__':main()
