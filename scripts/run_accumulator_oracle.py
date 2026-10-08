"""Matched accumulator load latency sensitivity, with real cache traffic retained."""
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
sys.path.insert(0,str(ROOT/'benchmarks/ai_workloads'))
import numpy as np
from run_model_swiglu import command, expected_copies
from sweep_mlp_study import epochs
from audit_mlp_math import audit as math_audit
from build_access_timing import sha,save

OLD=ROOT/'results/ai_workloads/swiglu_model_pilot_v4_20260921'
OUT=ROOT/'results/ai_workloads/swiglu_accumulator_oracle_20260925'
BUILD=OUT/'build'
EX=OUT/os.environ.get('ACC_ORACLE_EXPERIMENT','experiment_v4')
ENV=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
STOP=threading.Event();LOCK=threading.Lock()
STATE=dict(pid=os.getpid(),stage='waiting_for_build',active={},completed=[])
PCS=json.loads((OUT/'pc_profile.json').read_text())


def status():
    STATE['updated']=time.time();save(EX/'status.json',STATE)


def cfg(path,old,on,arch):
    a=configparser.ConfigParser(interpolation=None,strict=False)
    b=configparser.ConfigParser(interpolation=None,strict=False)
    a.read(old/'config.ini');b.read(path/'config.ini')
    # Existing hardware fields must be unchanged from the pilot. Newly added
    # model defaults are recorded, not mistaken for a byte-identical old model.
    extra={};changes=[]
    for sec in a.sections():
        assert sec in b,sec
        for key,x in a[sec].items():
            y=b[sec].get(key)
            assert y is not None,(sec,key)
            if x.replace(str(old),'<RUN>')==y.replace(str(path),'<RUN>'):continue
            # New disabled resource objects and the default shared bridge port
            # are plumbing required by the current simulator, not enabled knobs.
            if key=='children':
                added=set(y.split())-set(x.split())
                assert set(x.split())<=set(y.split()) and added<=set(('microLlcMissCredits','microLlcWriteCredits','dataMissCredits','instMissCredits','microSpmQueue','writeCredits','missCredits')),(sec,x,y)
                changes.append([sec,key,x,y]);continue
            if sec.endswith('.mmu.walker') and key=='port':
                assert x.endswith('.in_ports[3]') and y==x.replace('.in_ports[3]','.in_ports[4]')
                changes.append([sec,key,x,y]);continue
            if sec.endswith('.sequencer') and key=='in_ports':
                core=re.search(r'l0_cntrl(\d+)',sec)[1]
                assert y.split()==x.split()[:3]+[f'system.cpu{core}.bridge_port']+x.split()[3:]
                changes.append([sec,key,x,y]);continue
            assert (sec=='system' and key=='readfile') or (sec=='system.cpu0.workload' and key in ('cmd','executable')),(sec,key,x,y)
            changes.append([sec,key,x,y])
        for key in set(b[sec])-set(a[sec]):extra[sec+'.'+key]=b[sec][key]
    for i in range(8):
        sec=f'system.ruby.l0_cntrl{i}.sequencer'
        values=[int(x,0) for x in b[sec]['cacheflex_early_read_pcs'].split()]
        assert values==([p['pc'] for p in PCS[arch]['loads']] if on else [])
        assert b[f'system.cpu{i}']['cacheflexindependentbridge']=='false'
    for sec in b.sections():
        if b[sec].get('type')=='MicroResourcePool':assert b[sec]['capacity']=='0'
    return dict(changes=changes,new_fields=extra)


def compare_cfg(off,on):
    a=configparser.ConfigParser(interpolation=None,strict=False)
    b=configparser.ConfigParser(interpolation=None,strict=False)
    a.read(off/'config.ini');b.read(on/'config.ini');assert a.sections()==b.sections()
    for sec in a.sections():
        assert a[sec].keys()==b[sec].keys()
        for key,x in a[sec].items():
            y=b[sec][key]
            if x.replace(str(off),'<RUN>')==y.replace(str(on),'<RUN>'):continue
            assert key=='cacheflex_early_read_pcs' and sec.endswith('.sequencer'),(sec,key,x,y)


def counters(path,count):
    rows=[]
    for block in (path/'stats.txt').read_text().split('Begin Simulation Statistics')[1:count+1]:
        row={}
        for line in block.split('End Simulation Statistics')[0].splitlines():
            w=line.split()
            if len(w)>1 and '.m_earlyRead' in w[0]:
                key=w[0].split('.')[-1];row[key]=row.get(key,0)+float(w[1])
        rows.append(row)
    return rows


def run(arch,on,t,d,f,stage):
    label=f'{stage}_{arch}_{"on" if on else "off"}_t{t}'
    path=EX/'runs'/label;path.mkdir()
    template=OLD/'runs'/f'pilot_{arch}_t2048_d2048_f8192_mp2_kc128_g0'
    inp=OLD/'inputs'/f't{t}_d{d}_f{f}.bin'
    cmd=command(template,BUILD/'gem5.opt',path,OLD/'builds/custom_g0'/arch,inp,arch,(t,d,f,2,128,0),2)
    cmd=[str(EX/'configs/deprecated/example/micro_resource_se.py') if x.endswith('/configs/deprecated/example/se.py') else x for x in cmd]
    if on:
        for i in range(8):cmd+=['-P',f'system.ruby.l0_cntrl{i}.sequencer.cacheflex_early_read_pcs='+repr([p['pc'] for p in PCS[arch]['loads']])]
    save(path/'command.json',cmd);proc=None;start=time.monotonic()
    try:
        with (path/'run.log').open('w') as log:
            proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=ENV)
            with LOCK:STATE['active'][label]=dict(pid=proc.pid,path=str(path));status()
            audit=None
            while proc.poll() is None:
                if STOP.is_set():raise RuntimeError('peer failed')
                if time.monotonic()-start>120*3600:raise RuntimeError('120h timeout')
                if shutil.disk_usage(OUT).free<50*1024**3:raise RuntimeError('disk guard')
                if audit is None and 'MLP_CONFIG' in (path/'run.log').read_text():
                    audit=cfg(path,template,on,arch);save(path/'config_audit.json',audit)
                save(path/'progress.json',dict(stage='simulating',pid=proc.pid,host_seconds=time.monotonic()-start))
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:pass
        assert proc.returncode==0,('exit',proc.returncode)
        log=(path/'run.log').read_text();assert not re.search(r'panic:|fatal:|\bFAIL\b|Simulated exit code not 0',log)
        audit=audit or cfg(path,template,on,arch)
        checks=[dict(re.findall(r'(\w+)=([^\s]+)',l)) for l in log.splitlines() if l.startswith('MLP_RESULT')]
        assert len(checks)==2 and all(c['errors']==c['nonfinite']=='0' for c in checks)
        data=np.fromfile(path/'output.bin',dtype='<f2');ref=np.load(inp.with_suffix('.stages.npz'))['y'].reshape(-1)
        assert data.size==t*d and np.isfinite(data).all() and np.array_equal(data,ref)
        h=1469598103934665603
        for x in data.view(np.uint16):h=((h^int(x))*1099511628211)&((1<<64)-1)
        assert all(c['hash']==f'{h:016x}' for c in checks)
        accuracy=math_audit(path,inp,t,d,f);assert accuracy['passed']
        timing=epochs(path,2);copies=expected_copies(t,d,f,2) if arch=='spm' else 0
        assert all(e['copies_issued']==e['copies_completed']==copies for e in timing)
        cumulative=counters(path,2)
        diag=[cumulative[0],{k:v-cumulative[0][k] for k,v in cumulative[1].items()}]
        if on:assert all(r['m_earlyReadIssued']>0 for r in diag), 'oracle unexercised'
        else:assert all(all(v==0 for v in r.values()) for r in diag)
        # These diagnostic scalars are lifetime counters (Ruby's resetStats
        # explicitly resets selected old counters, not these new observers).
        all_diag=counters(path,100)
        assert len(all_diag) in (2,3),len(all_diag)
        totals=all_diag[-1]
        assert totals['m_earlyReadIssued']==totals['m_earlyReadVerified'],totals
        row=dict(path=str(path),arch=arch,on=on,T=t,epochs=timing,oracle=diag,oracle_total=totals,
                 exact_reference=True,hash=f'{h:016x}',math=accuracy,config=audit,host_seconds=time.monotonic()-start)
        save(path/'accepted.json',row)
        with LOCK:STATE['completed'].append(row);STATE['active'].pop(label,None);status()
        return row
    except BaseException as e:
        STOP.set();save(path/'failed.json',dict(error=repr(e)))
        if proc and proc.poll() is None:
            proc.terminate()
            try:proc.wait(timeout=30)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
        raise


def main():
    EX.mkdir();(EX/'runs').mkdir();status()
    shutil.copy2(__file__,EX/'runner_snapshot.py')
    try:
        while not (BUILD/'pilot_frozen.json').exists():
            if (BUILD/'status.json').exists():
                b=json.loads((BUILD/'status.json').read_text());assert b['stage']!='failed',b
            time.sleep(10)
        frozen=json.loads((BUILD/'pilot_frozen.json').read_text())
        shutil.copytree(BUILD/'pilot_configs',EX/'configs')
        shutil.copy2(ROOT/'scripts/micro_resource_se.py',EX/'configs/deprecated/example/micro_resource_se.py')
        for p in (EX/'configs').rglob('*.py'):frozen[str(p)]=sha(p)
        for arch in ('cache','spm'):
            guest=OLD/'builds/custom_g0'/arch;assert sha(guest)==PCS[arch]['sha256']
            frozen[str(guest)]=sha(guest)
        for t,d,f in ((24,256,512),(2048,2048,8192)):
            for suffix in ('.bin','.stages.npz'):
                p=OLD/'inputs'/f't{t}_d{d}_f{f}{suffix}';frozen[str(p)]=sha(p)
        save(EX/'inputs.json',frozen)
        # V3's four gates have full numerical/config acceptance. Only the
        # reporting convention changes: no new simulator/guest/config bytes.
        prior=OUT/'experiment_v3'
        gates=[r for r in json.loads((prior/'status.json').read_text())['completed'] if r['T']==24]
        assert {(r['arch'],r['on']) for r in gates}=={(a,o) for a in ('cache','spm') for o in (False,True)}
        for row in gates:
            path=Path(row['path']);assert (path/'accepted.json').exists()
            c=counters(path,100)
            row['oracle']=[c[0],{k:v-c[0][k] for k,v in c[1].items()}]
            row['oracle_total']=c[-1]
            assert c[-1]['m_earlyReadIssued']==c[-1]['m_earlyReadVerified']
            row['reused']=True
        STATE['completed']=gates;save(EX/'reused_gates.json',gates);status()
        for stage,t,d,f in (('target',2048,2048,8192),):
            STATE['stage']=stage;status()
            for on in (False,True):
                with cf.ThreadPoolExecutor(max_workers=2) as pool:
                    jobs=[pool.submit(run,a,on,t,d,f,stage) for a in ('cache','spm')]
                    for j in jobs:j.result()
            for arch in ('cache','spm'):
                off=EX/'runs'/f'{stage}_{arch}_off_t{t}';on=EX/'runs'/f'{stage}_{arch}_on_t{t}'
                compare_cfg(off,on)
            assert all(sha(p)==h for p,h in frozen.items()),'frozen dependency changed'
        STATE['stage']='completed_validated';status()
    except BaseException as e:
        STATE['stage']='failed';STATE['error']=repr(e);status();raise


if __name__=='__main__':main()
