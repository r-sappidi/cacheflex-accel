"""Isolated, correctness-gated KV ownership experiment; max two new simulators."""
import concurrent.futures as cf
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
import numpy as np
from reference import reference,validate

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from build_access_timing import sha,save
from run_access_timing import stats
from test_micro_timing import check_global_guard
from fa_headroom_profiles import leaves

BASE=ROOT/'results/cacheflex_flash_attention/kv_partition_20260925'
OUT=BASE/os.environ.get('CACHEFLEX_KV_EXPERIMENT','experiment_v3')
GUEST=BASE/os.environ.get('CACHEFLEX_KV_BUILD','build')/'guest'
MODEL=ROOT/'results/cacheflex_flash_attention/spm_early_address_20260924/build_v4'
CONTROL=ROOT/'results/cacheflex_flash_attention/spm_early_address_20260924/experiment_v4/runs/fa_off_t4096'
ENV=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib',
         OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')

def case(t,n,kv,spm,br=16,group=4,l2=512,bc=128,stage='target'):
    return dict(t=t,n=n,kv=kv,spm=spm,br=br,group=group,l2=l2,bc=bc,stage=stage)

def label(c):
    return f"{c['stage']}_t{c['t']}_n{c['n']}_{'kv' if c['kv'] else 'q'}_{'spm' if c['spm'] else 'cache'}_br{c['br']}_g{c['group']}_l2{c['l2']}"

def command(c,path):
    cmd=json.loads((CONTROL/'command.json').read_text());cmd[0]=str(MODEL/'gem5.opt')
    cmd[1]=f'--outdir={path}';cmd[2]=str(OUT/'configs/deprecated/example/micro_timing_se.py')
    args=list(map(str,(c['t'],c['n'],c['br'],c['bc'],c['group'],c['l2']*2,int(c['kv']),int(c['spm']))))
    cmd[cmd.index('--cmd')+1]=str(GUEST);cmd[cmd.index('--options')+1]=' '.join(args)
    out=[];i=0
    while i<len(cmd):
        a=cmd[i]
        if a=='-P':
            p=cmd[i+1];i+=2
            m=re.match(r'system\.(?:cpu\[(\d+)\]|ruby\.l[01]_cntrl(\d+))',p)
            if m and int(m[1] or m[2])>=c['n']:continue
            if p.startswith('system.cpu[0].workload[0].cmd='):
                p=f"system.cpu[0].workload[0].cmd={['kv-fa']+args!r}"
            out+=['-P',p];continue
        replacements={'--num-cpus=':str(c['n']),'--topology=':'KVFixedMesh',
            '--l1d_size=':f"{c['l2']}kB",'--l1d_num_spm_ways=':'6' if c['spm'] else '0'}
        for prefix,value in replacements.items():
            if a.startswith(prefix):a=prefix+value;break
        out.append(a);i+=1
    return out

def audit(c,path):
    original=json.loads((CONTROL/'config.json').read_text())['system']
    syscfg=json.loads((path/'config.json').read_text())['system'];ruby=syscfg['ruby']
    assert len(syscfg['cpu'])==c['n']
    # All scalar numerical CPU parameters must equal the calibrated control.
    for cpu in syscfg['cpu']:
        for k,v in original['cpu'][0].items():
            if isinstance(v,(int,float,bool)) and k not in ('cpu_id',):assert cpu[k]==v,(k,v,cpu[k])
        assert cpu['isa'][0]['sve_vl_se']==16
        workload=cpu['workload'][0]
        if isinstance(workload,str):
            assert workload==syscfg['cpu'][0]['workload'][0]['path']
            workload=syscfg['cpu'][0]['workload'][0]
        assert workload['executable']==str(GUEST)
        assert workload['cmd'][1:]==list(map(str,(c['t'],c['n'],c['br'],c['bc'],c['group'],c['l2']*2,int(c['kv']),int(c['spm']))))
    for i in range(c['n']):
        l0=ruby[f'l0_cntrl{i}'];l1=ruby[f'l1_cntrl{i}']
        assert l1['cache']['size']==c['l2']*1024
        assert l1['cache']['assoc']==8 and l1['cache']['num_spm_ways']==(6 if c['spm'] else 0)
        assert l0['sequencer']['max_outstanding_requests']==1024
        for name,obj in ((f'l0_cntrl{i}',l0),(f'l1_cntrl{i}',l1)):
            old=original['ruby'][name]
            for k,v in old.items():
                if isinstance(v,(int,float,bool)):assert obj[k]==v,(name,k,v,obj[k])
    assert len(ruby['network']['routers'])==8
    for i in range(8):
        llc=ruby[f'l2_cntrl{i}'];old=original['ruby'][f'l2_cntrl{i}']
        assert llc['L2cache']['size']==512*1024
        for k,v in old.items():
            if isinstance(v,(int,float,bool)):assert llc[k]==v,(k,v,llc[k])
    # All internal links retain the old network's numerical values/topology.
    assert ruby['network']['int_links']==original['ruby']['network']['int_links']
    return dict(cpu_numeric_parameters_matched=True,private_l2_kib=c['l2'],
        llc_mib=4,routers=8,cores=c['n'],spm_eligible_ways=6 if c['spm'] else 0,
        retention=True,added_data_sram=False)

def report(state):
    lines=['# KV-partitioned persistent-V FA','',
        'Single head, noncausal d64/VL16; existing FP16 GEMMs and FP32 state. '
        'All packing, staging, merge, synchronization and release included. '
        'First invocation after initialization; NOT guaranteed cold. '
        'Software mapping experiment, not a new hardware mechanism or tuned optimum.','',
        '| Case | ROI us | SPMCP | Host seconds |','|---|---:|---:|---:|']
    for r in state['completed']:
        lines.append(f"| {r['name']} | {r['roi_us']:.4f} | {r['copies']} | {r['host_seconds']:.1f} |")
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')

def main():
    OUT.mkdir();(OUT/'runs').mkdir();(OUT/'references').mkdir()
    shutil.copytree(MODEL/'configs',OUT/'configs')
    # Freeze the already-used eight-router placement and extend to four CPUs.
    topo=(ROOT/'scripts/SwiGLUFixedMesh.py').read_text().replace('SwiGLUFixedMesh','KVFixedMesh').replace('in (1, 8)','in (1, 4, 8)')
    # Preserve internal link identifiers as well as wiring when private nodes
    # are absent; holes in the external-ID range do not add hardware.
    topo=topo.replace("        for direction in ('East'", "        link_id=32\n        for direction in ('East'")
    (OUT/'configs/topologies/KVFixedMesh.py').write_text(topo)
    paths=list((OUT/'configs').rglob('*.py'))+list(HERE.glob('*.py'))+[HERE/'driver.cpp',GUEST,MODEL/'gem5.opt',CONTROL/'command.json',CONTROL/'config.json']
    paths+=list(p for p in GUEST.parent.iterdir() if p.is_file())
    manifest={str(p):sha(p) for p in paths};save(OUT/'inputs.json',manifest)
    state=dict(pid=os.getpid(),stage='gates',active={},completed=[],updated=time.time())
    lock=threading.RLock();halt=threading.Event();refs={}
    def status(stage=None):
        with lock:
            if stage:state['stage']=stage
            state['updated']=time.time();save(OUT/'status.json',state);report(state)
    def frozen():
        for p,h in manifest.items():assert sha(p)==h,('Frozen input changed',p)
    def run(c):
        name=label(c);path=OUT/'runs'/name;path.mkdir();proc=None;start=time.monotonic()
        try:
            frozen();cmd=command(c,path);save(path/'command.json',cmd)
            with (path/'run.log').open('w') as log:
                proc=subprocess.Popen(cmd,cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT)
                with lock:state['active'][name]=dict(pid=proc.pid,path=str(path),case=c);status()
                while proc.poll() is None:
                    assert not halt.is_set(),'Peer failure'
                    assert time.monotonic()-start<(172800 if c['t']>=4096 else 7200),'Timeout'
                    assert shutil.disk_usage(OUT).free>50*2**30,'Disk guard'
                    try:proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:pass
            text=(path/'run.log').read_text()
            assert proc.returncode==0 and 'Simulated exit code not 0!' not in text,(name,proc.returncode)
            assert 'KV_ROI_END' in text,'Missing ROI completion'
            hardware=audit(c,path);blocks=stats(path);s=blocks[0]
            for b in blocks:check_global_guard(b,c['n'])
            expected=2*c['t'] if c['kv'] else 2*c['t']*c['n']*((c['t']//c['n']+c['br']*c['group']-1)//(c['br']*c['group']))
            if not c['spm']:expected=0
            for key in ('m_spmcpRequestsIssued','m_spmcpRequestsCompleted'):
                actual=sum(s[f'system.ruby.l0_cntrl{i}.sequencer.{key}'] for i in range(c['n']))
                assert actual==expected,(name,key,actual,expected)
            counter='system.ruby.L1Cache_Controller.X.SPM_release'
            releases=s.get(counter+'::total',s.get(counter,0))
            expected_releases=(2*c['t'] if c['kv'] else 2*c['bc']*c['n']) if c['spm'] else 0
            assert releases==expected_releases,(name,releases,expected_releases)
            key=(c['t'],c['bc'],c['n'] if c['kv'] else 1)
            numerical=validate(path/'output.fp16',refs[key],c['t'],dense=c['stage']=='gate')
            data=np.fromfile(path/'output.fp16',dtype=np.uint16);h=1469598103934665603
            for x in data:h=((h^int(x))*1099511628211)&((1<<64)-1)
            match=re.search(r'KV_RESULT hash=([0-9a-f]+) nonfinite=0',text)
            assert match and match[1]==f'{h:016x}','Printed/export hash mismatch'
            frozen();row=dict(case=c,name=name,path=str(path),roi_us=s['simTicks']/1e6,
                copies=expected,releases=releases,arithmetic=numerical,config=hardware,
                output_sha256=sha(path/'output.fp16'),host_seconds=time.monotonic()-start)
            save(path/'validated.json',row)
            with lock:state['active'].pop(name);state['completed'].append(row);status()
            return row
        except BaseException as exc:
            halt.set();save(path/'failed.json',dict(error=repr(exc)))
            if proc and proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
            with lock:state['active'].pop(name,None);status()
            raise
    def pair(c):
        key=(c['t'],c['bc'],c['n'] if c['kv'] else 1)
        if key not in refs:
            status('reference_'+str(key));p=OUT/'references'/('t%d_bc%d_parts%d.npy'%key)
            refs[key]=reference(*key);np.save(p,refs[key]);save(p.with_suffix('.json'),dict(sha256=sha(p),key=key))
        status(label(c).replace('_cache_','_pair_'))
        with cf.ThreadPoolExecutor(max_workers=2) as pool:
            fs=[pool.submit(run,dict(c,spm=spm)) for spm in (False,True)]
            rows=[f.result() for f in cf.as_completed(fs)]
        assert rows[0]['output_sha256']==rows[1]['output_sha256'],'Matched cache/SPM output bits differ'
        # Audit every field between matched arms, allowing only mode/capacity eligibility.
        a,b=rows
        ca=dict(leaves(json.loads((Path(a['path'])/'config.json').read_text().replace(a['path'],'<OUT>'))))
        cb=dict(leaves(json.loads((Path(b['path'])/'config.json').read_text().replace(b['path'],'<OUT>'))))
        delta={k:[ca.get(k),cb.get(k)] for k in ca.keys()|cb.keys() if ca.get(k)!=cb.get(k)}
        allowed=(r'system.cpu\[\d+\].workload\[0\].cmd\[8\]',r'system.ruby.l1_cntrl\d+.cache.num_spm_ways')
        assert all(any(re.fullmatch(p,k) for p in allowed) for k in delta),delta
        save(OUT/(label(c)+'_pair.json'),dict(bitwise_equal=True,config_delta=delta,rows=rows))
    try:
        status()
        for n,t,br,g in ((1,384,32,5),(4,1024,32,3),(8,1024,16,4)):
            for kv in (False,True):pair(case(t,n,kv,False,br,g,l2=64,stage='gate'))
        save(OUT/'gates_passed.json',dict(passed=True,cases=12,independent_math=True,multi_way_pairs=True))
        # Eight-core discovery first; both mappings get the same two group sizes.
        for group in (4,8):
            for kv in (False,True):pair(case(4096,8,kv,False,group=group))
        # Common query mapping establishes same-workload strong scaling.
        for n in (1,4):pair(case(4096,n,False,False))
        # Long-context confirmation keeps both group sizes, permitting independent
        # backend/mapping selection afterwards without assuming a T4096 winner.
        for group in (4,8):
            for kv in (False,True):pair(case(16384,8,kv,False,group=group))
        status('completed_validated')
    except BaseException as exc:
        state['error']=repr(exc);status('failed');raise

if __name__=='__main__':main()
