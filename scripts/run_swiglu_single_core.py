"""Frozen one-core SwiGLU comparison; two extra simulations, no active edits."""
import concurrent.futures
import configparser
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'benchmarks/ai_workloads'))
import numpy as np
from sweep_mlp_study import epochs
from audit_mlp_math import audit as math_audit

OLD = ROOT/'results/ai_workloads/swiglu_model_pilot_v4_20260921'
OUT = ROOT/'results/ai_workloads'/os.environ.get('SWIGLU_SINGLE_OUT','swiglu_single_core_v2_20260925')
ENV = dict(os.environ, LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib',
           OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
LOCK = threading.Lock()
STOP = threading.Event()
STATE = dict(pid=os.getpid(), stage='preparing', active={}, completed=[])


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def save(p, data):
    tmp = p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(data, indent=2)+'\n')
    tmp.replace(p)


def status():
    STATE['updated'] = time.time()
    save(OUT/'status.json', STATE)


def config_audit(path, old, arch):
    a = configparser.ConfigParser(interpolation=None, strict=False)
    b = configparser.ConfigParser(interpolation=None, strict=False)
    a.read(old/'config.ini'); b.read(path/'config.ini')
    cpus = [s for s in b.sections() if b[s].get('type') == a['system.cpu0']['type']]
    assert len(cpus) == 1, cpus
    cpu = cpus[0]
    # Compare every numeric/bool CPU parameter (widths, queues, ports, etc.).
    numeric = re.compile(r'(?:-?\d+(?:\.\d+)?|true|false)', re.I)
    checked = 0
    for key, value in a['system.cpu0'].items():
        if numeric.fullmatch(value) and key not in ('cpu_id',):
            assert b[cpu].get(key) == value, (cpu,key,value,b[cpu].get(key))
            checked += 1
    assert b[cpu+'.isa']['sve_vl_se'] == '16'
    for sec in ('system.clk_domain','system.cpu_clk_domain'):
        assert dict(a[sec]) == dict(b[sec]), sec
    for section in a.sections():
        # Surviving controller/cache parameters, excluding plumbing/addresses.
        if not re.match(r'system\.ruby\.(?:l0_cntrl0|l1_cntrl0|l2_cntrl\d+|dir_cntrl\d+)(?:\.|$)',section):
            continue
        assert section in b, section
        for key,value in a[section].items():
            if numeric.fullmatch(value):
                assert b[section].get(key) == value, (section,key,value,b[section].get(key))
    assert b['system.ruby.l1_cntrl0.cache']['size'] == '524288'
    assert b['system.ruby.l1_cntrl0.cache']['num_spm_ways'] == ('4' if arch=='spm' else '0')
    assert sum(int(b[f'system.ruby.l2_cntrl{i}.L2cache']['size']) for i in range(8)) == 4*1024**2
    routers = [s for s in b.sections() if b[s].get('type') == 'GarnetRouter']
    assert len(routers) == 8
    # All eight bank/directory attachments and core0 remain at original routers.
    def links(cfg, typ):
        return [dict(cfg[s]) for s in cfg.sections() if cfg[s].get('type') == typ]
    oldext = {x['ext_node']:x['int_node'] for x in links(a,'GarnetExtLink')}
    newext = {x['ext_node']:x['int_node'] for x in links(b,'GarnetExtLink')}
    assert len(newext)==18, newext  # L0 + private L2 + eight LLC + eight dirs
    assert all(oldext[k]==v for k,v in newext.items()), (oldext,newext)
    def fabric(cfg):
        return sorted(tuple(x[k] for k in ('src_node','dst_node','src_outport','dst_inport','latency','weight'))
                      for x in links(cfg,'GarnetIntLink'))
    assert fabric(a)==fabric(b), 'fabric changed'
    return dict(single_cpu=True, cpu_numeric_parameters_checked=checked,
                surviving_controller_numeric_parameters_equal=True,
                fabric_and_placement_equal=True, LLC_bytes=4*1024**2)


def command(arch,t,d,f,repeats,path):
    template=OLD/'runs'/f'pilot_{arch}_t2048_d2048_f8192_mp2_kc128_g0'
    cmd=json.loads((template/'command.json').read_text())
    # Remove per-CPU overrides for removed CPUs, without touching CPU0.
    clean=[]; i=0
    while i<len(cmd):
        if cmd[i]=='-P' and re.match(r'system\.cpu\[[1-7]\]',cmd[i+1]):
            i+=2; continue
        clean.append(cmd[i]); i+=1
    cmd=clean
    inp=OLD/'inputs'/f't{t}_d{d}_f{f}.bin'
    args=list(map(str,[t,d,f,1,128,0,repeats,1,1]))
    for i,value in enumerate(cmd):
        if value.startswith('--outdir='):cmd[i]=f'--outdir={path}'
        elif value=='--num-cpus=8':cmd[i]='--num-cpus=1'
        elif value=='--topology=Mesh_XY':cmd[i]='--topology=SwiGLUFixedMesh'
        elif value.endswith('/configs/deprecated/example/se.py'):cmd[i]=str(OUT/'configs/deprecated/example/se.py')
        elif value.startswith('system.readfile='):cmd[i]=f'system.readfile={str(inp)!r}'
        elif value.startswith('system.cpu[0].workload[0].cmd='):cmd[i]='system.cpu[0].workload[0].cmd='+repr(['model-swiglu']+args)
    cmd[cmd.index('--options')+1]=' '.join(args)
    return cmd, inp, template


def run(arch,t,d,f,repeats,stage):
    path=OUT/'runs'/f'{stage}_{arch}_t{t}_d{d}_f{f}'
    path.mkdir()
    cmd,inp,template=command(arch,t,d,f,repeats,path)
    save(path/'command.json',cmd)
    proc=None; start=time.monotonic()
    try:
        with (path/'run.log').open('w') as log:
            proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=ENV)
            with LOCK:
                STATE['active'][arch]=dict(pid=proc.pid,path=str(path),T=t)
                status()
            audited=None
            while proc.poll() is None:
                if STOP.is_set():raise RuntimeError('peer failed')
                if time.monotonic()-start>120*3600:raise RuntimeError('120h timeout')
                if shutil.disk_usage(OUT).free<50*1024**3:raise RuntimeError('disk guard')
                if audited is None and (path/'config.ini').exists():
                    # Audit once simulation has reached the guest (config fully written).
                    if 'MLP_CONFIG' in (path/'run.log').read_text():
                        audited=config_audit(path,template,arch)
                        save(path/'config_audit.json',audited)
                save(path/'progress.json',dict(pid=proc.pid,stage='simulating',host_seconds=time.monotonic()-start))
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:pass
        assert proc.returncode==0, ('exit',proc.returncode)
        log=(path/'run.log').read_text()
        assert not re.search(r'panic:|fatal:|\bFAIL\b|Simulated exit code not 0',log)
        audited=audited or config_audit(path,template,arch)
        checks=[dict(re.findall(r'(\w+)=([^\s]+)',l)) for l in log.splitlines() if l.startswith('MLP_RESULT')]
        assert len(checks)==repeats and all(c['errors']==c['nonfinite']=='0' for c in checks)
        assert len({c['hash'] for c in checks})==1
        actual=np.fromfile(path/'output.bin',dtype='<f2')
        reference=np.load(inp.with_suffix('.stages.npz'))['y'].reshape(-1)
        assert actual.size==t*d and np.isfinite(actual).all()
        assert np.array_equal(actual,reference), 'ordered reference differs'
        digest=1469598103934665603
        for x in actual.view(np.uint16):digest=((digest^int(x))*1099511628211)&((1<<64)-1)
        assert f'{digest:016x}'==checks[0]['hash']
        # Same precision/order as eight-core run: require exact full-output hash.
        oldlog=(template/'run.log').read_text() if stage=='target' else ''
        if stage=='target':assert f'hash={digest:016x}' in oldlog
        math=math_audit(path,inp,t,d,f);assert math['passed']
        timing=epochs(path,repeats)
        expected=4*(2*d*(f//128)+f*(d//128)) if arch=='spm' else 0
        assert all(e['copies_issued']==e['copies_completed']==expected for e in timing)
        row=dict(arch=arch,T=t,path=str(path),epochs=timing,checks=checks,
                 math=math,config=audited,exact_reference=True,output_hash=f'{digest:016x}',
                 host_seconds=time.monotonic()-start)
        save(path/'accepted.json',row)
        with LOCK:
            STATE['completed'].append(row);STATE['active'].pop(arch,None);status()
        return row
    except BaseException as e:
        STOP.set();save(path/'failed.json',dict(error=repr(e)))
        if proc and proc.poll() is None:
            proc.terminate()
            try:proc.wait(timeout=30)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
        raise


def main():
    OUT.mkdir();(OUT/'runs').mkdir();status()
    original=json.loads((OLD/'manifest.json').read_text())
    assert all(sha(p)==h for p,h in original.items()), 'active dependency changed'
    shutil.copytree(ROOT/'gem5/configs',OUT/'configs',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(ROOT/'scripts/SwiGLUFixedMesh.py',OUT/'configs/topologies/SwiGLUFixedMesh.py')
    shutil.copy2(__file__,OUT/'runner_snapshot.py')
    files=[p for p in (OUT/'configs').rglob('*.py')]+[OUT/'runner_snapshot.py',
        OLD/'preserved/gem5.opt',OLD/'builds/custom_g0/cache',OLD/'builds/custom_g0/spm']
    for t,d,f in ((24,256,512),(2048,2048,8192)):
        files.extend([OLD/'inputs'/f't{t}_d{d}_f{f}{suffix}' for suffix in ('.bin','.stages.npz')])
    manifest={str(p):sha(p) for p in files};save(OUT/'manifest.json',manifest)
    save(OUT/'plan.json',dict(cores=1,shared_fabric='unchanged 8-router 2x4 mesh',
        LLC='eight 512KiB banks',memory_controllers=8,private_L2='512KiB',VL=16,
        mapping='1M x 1N; original eight-core 2M x 4N',kernel='same frozen ELF',
        target=[2048,2048,8192],invocations=2,measurement='first and second compared separately, not converged warm median',
        scope='fixed workload/core scaling, not independently retuned best-to-best',max_extra_simulators=2))
    try:
        for stage,t,d,f in (('gate',24,256,512),('target',2048,2048,8192)):
            STATE['stage']=stage;status()
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                jobs=[pool.submit(run,a,t,d,f,2,stage) for a in ('cache','spm')]
                for j in jobs:j.result()
            assert all(sha(p)==h for p,h in manifest.items()), 'frozen input changed'
            assert all(sha(p)==h for p,h in original.items()), 'old dependency changed'
        STATE['stage']='completed_validated';status()
    except BaseException as e:
        STATE['stage']='failed';STATE['error']=repr(e);status();raise


if __name__=='__main__':main()
