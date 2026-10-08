"""Single-head replacement for stopped multi-head sequencer/causal queues."""
import concurrent.futures as cf
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import threading
import time

import run_fa_overnight_diagnosis as common
from run_causal import parameters
from single_head_reference import check
from run_clean_controls import OLD, sha, save

ROOT=common.ROOT
CONTROLS=ROOT/'results/cacheflex_flash_attention/clean_controls_20260917'
RETAIN=ROOT/'results/cacheflex_flash_attention/single_head_retention_20260917'

def prior(arch,t):
    return (RETAIN/'runs'/f'spm_t{t}_retain1_forced0' if arch=='spm' else
            CONTROLS/'runs'/f'cache_t{t}_rec0')

def patch(src,dst):
    original=src.read_bytes();data=bytearray(original);va=0x401e58
    assert common.instruction_bytes(data,va)==bytes.fromhex('e2f740f9')
    ph=struct.unpack_from('<Q',data,32)[0];size,n=struct.unpack_from('<HH',data,54)
    found=[]
    for i in range(n):
        typ,_,off,addr,_,filesz,_,_=struct.unpack_from('<IIQQQQQQ',data,ph+i*size)
        if typ==1 and addr<=va and va+4<=addr+filesz:found.append(off+va-addr)
    assert len(found)==1
    off=found[0];data[off:off+4]=bytes.fromhex('021080d2') # mov x2,#128
    assert data[:off]==original[:off] and data[off+4:]==original[off+4:]
    dst.write_bytes(data);shutil.copymode(src,dst)
    return dict(pc=hex(va),offset=off,original_sha=sha(src),patched_sha=sha(dst),
                old='ldr x2,[sp,#488]',new='mov x2,#128',scope='d64 only')

def phase(frames,shift=0):
    # Symbol names remain valid; old multi-head driver line ranges DO NOT.
    names=' '.join(f for f,_ in frames)
    for symbol,label in [('cf_pv16','PV_microkernel'),('cf_wide_qk','QK_microkernel'),
      ('cf_vector_pack_A','Q_P_packing'),('pack_A_fp16','Q_P_packing'),
      ('stage_kv','V_staging'),('scale_f32','output_rescale'),('acc_f16','PV_output_accumulate'),
      ('sve_exp_f32','vector_exp'),('expf','scalar_exp'),('pthread_barrier','synchronization'),
      ('futex','synchronization')]:
        if symbol in names:return label
    return 'other_unclassified'

def main():
    reuse=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else None
    if reuse:
        old_hashes=json.loads((reuse/'inputs.json').read_text())
        for p,h in old_hashes.items():
            actual=reuse/'runner.before_skip_controls.py' if Path(p)==Path(__file__).resolve() else Path(p)
            assert sha(actual)==h,('reuse dependency changed',p)
        old_gates=json.loads((reuse/'gates_complete.json').read_text())
        assert len(old_gates)==24
        for row in old_gates:
            accepted=json.loads((Path(row['path'])/'accepted.json').read_text())
            assert accepted['case']==row['case'] and accepted['counters']['roi_us']==row['roi_us']
    out=Path(sys.argv[1]).resolve();out.mkdir();(out/'runs').mkdir();(out/'preserved').mkdir()
    state=dict(pid=os.getpid(),stage='preparing',completed=[]);stop=threading.Event();lock=threading.Lock()
    def status(stage):
        state.update(stage=stage,updated=time.time());save(out/'status.json',state);print(stage,flush=True)
    status('preparing')
    try:
        guests={};debug={};watched=[Path(__file__).resolve(),Path(common.__file__).resolve()]
        for arch in ('cache','spm'):
            cmd=json.loads((prior(arch,512)/'command.json').read_text())
            guests[arch]=Path(cmd[cmd.index('--cmd')+1]);debug[arch]=guests[arch].with_name('guest.debug')
            watched += [guests[arch],debug[arch],Path(cmd[0])]
            for t in (512,16384):
                p=prior(arch,t)
                assert (p/'accepted.json').exists()
                watched += [p/n for n in ('accepted.json','config.ini','command.json')]
                watched.append(OLD/'references'/f't{t}_bc128.npy')
        assert sha(json.loads((prior('cache',512)/'command.json').read_text())[0])==sha(json.loads((prior('spm',512)/'command.json').read_text())[0])
        patched=out/'spm.stride128.guest';patched_debug=out/'spm.stride128.debug'
        save(out/'patch.json',patch(guests['spm'],patched));patch(debug['spm'],patched_debug)
        watched += [patched,patched_debug]
        watched+=list((ROOT/'gem5/configs').rglob('*.py'))
        watched += [common.HERE/n for n in ('run_causal.py','single_head_reference.py','validate_attribution.py','analyze_attribution.py','analyze_traffic.py','fa_export_patch.py','run_clean_controls.py')]
        hashes={str(p):sha(p) for p in watched};save(out/'inputs.json',hashes)
        for p in set([Path(__file__).resolve(),guests['cache'],guests['spm'],debug['spm'],patched]):
            dest=out/'preserved'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
        common.GUEST=guests['spm'] # strict config audit normalizes ONLY this patched executable
        common.a.phase=phase
        def frozen():
            for p,h in hashes.items():assert sha(p)==h,('input changed',p)
        # Full 3-factor diagnostic factorial at both admission endpoints, plus
        # matched baseline capacities and exact-layout software intervention.
        gates=[('cache',512,n,0,False) for n in (16,32,64,128)]
        gates += [('spm',512,n,m,False) for n in (16,128) for m in (0,1,2,8,3,9,10,11)]
        gates += [('spm',512,n,0,False) for n in (32,64)]
        gates += [('spm',512,n,0,True) for n in (16,128)]
        # Existing independent T16K controls already establish these two points.
        targets=[('spm',16384,16,0,True),('spm',16384,16,8,False)]
        targets += [(a,16384,n,0,False) for n in (32,64,128) for a in ('cache','spm')]
        targets += [('spm',16384,16,m,False) for m in (1,2,11)]
        targets += [('spm',16384,128,11,False),('spm',16384,128,0,True)]
        save(out/'plan.json',dict(Hq=1,Hkv=1,T=16384,d=64,cores=8,VL=16,
          L1_KiB=64,private_L2_KiB=64,LLC_MiB=4,cache=dict(Br=32,Bc=128,G=2),
          spm=dict(Br=16,Bc=128,G=4,ways=2,retention=True),max_simulators=4,
          gates=gates,targets=targets,recorder_entries=0,
          caution='Different cache capacity/tiling from stopped multi-head study; never merge their timings. Separate backing adds storage. Service interventions change delays/ports/FIFO clamping. PC attribution is not causal cycle fractions.'))
        save(out/'reused_t16384_seq16_controls.json',{
            arch:json.loads((prior(arch,16384)/'accepted.json').read_text()) for arch in ('cache','spm')})
        if reuse:
            save(out/'reused_gates.json',dict(source=str(reuse),rows=old_gates,dependency_hashes_verified=True))
            state['completed'].extend(old_gates)
            gates=[]
        env=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        def report():
            lines=['# Single-head diagnosis','', 'Accepted single-head results only; separate backing is not area matched.','',
              '| Architecture | T | Entries | Mask | Patch | Runtime us |','|---|---:|---:|---:|:---:|---:|']
            for r in state['completed']:
                a,t,n,m,b=r['case'];lines.append(f"| {a} | {t} | {n} | {m} | {b} | {r['roi_us']:.4f} |")
            (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
        def run(case):
            arch,t,n,m,patch_on=case;frozen()
            p=out/'runs'/f'{arch}_t{t}_seq{n}_m{m}_patch{int(patch_on)}';p.mkdir()
            old=prior(arch,t);cmd=json.loads((old/'command.json').read_text())
            cmd=[f'--outdir={p}' if x.startswith('--outdir=') else x for x in cmd]
            if patch_on:cmd[cmd.index('--cmd')+1]=str(patched)
            params=parameters(m)
            params.update({f'system.ruby.l0_cntrl{i}.sequencer.max_outstanding_requests':str(n) for i in range(8)})
            for k,v in params.items():cmd+=['-P',f'{k}={v}']
            save(p/'command.json',cmd);start=time.monotonic();proc=None
            try:
                with (p/'run.log').open('w') as f:
                    proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env)
                    while True:
                        elapsed=time.monotonic()-start
                        save(p/'progress.json',dict(pid=proc.pid,updated=time.time(),host_elapsed_s=elapsed,kernel_percentage=None))
                        if stop.is_set() or elapsed>30*3600:raise RuntimeError('Peer failed or timeout')
                        if shutil.disk_usage(out).free<50*1024**3:raise RuntimeError('Disk free below 50GiB')
                        try:rc=proc.wait(timeout=10);break
                        except subprocess.TimeoutExpired:pass
                save(p/'exit.json',dict(returncode=rc,host_elapsed_s=time.monotonic()-start));assert rc==0,(p,rc)
                frozen();audit=common.audit(old,p,params,patched if patch_on else None)
                output,counters,_=common.read(p,16)
                for k,v in dict(Hq=1,Hkv=1,T=t,d=64,cores=8,Bc=128,Br=16 if arch=='spm' else 32).items():assert output[k]==str(v),(k,output)
                copies=(t//64)*t*2 if arch=='spm' else 0
                assert counters['m_spmcpRequestsIssued']==counters['m_spmcpRequestsCompleted']==copies
                row=dict(path=str(p),case=case,output=output,counters=counters,config_audit=audit)
                common.verify_dump(row);row['validation']=check(p/'output.fp16',common.np.load(OLD/'references'/f't{t}_bc128.npy'))
                ctrl=json.loads((old/'accepted.json').read_text())
                assert output==ctrl['output'],'Output changed'
                if n==16 and m==0 and not patch_on:
                    assert {k:v for k,v in counters.items() if k!='path'}=={k:v for k,v in ctrl['counters'].items() if k!='path'},'No-op control mismatch'
                save(p/'accepted.json',row)
                with lock:
                    result=common.a.analyze(p,patched_debug if patch_on else debug[arch])
                    gd=Path(cmd[cmd.index('--cmd')+1]).read_bytes();dd=(patched_debug if patch_on else debug[arch]).read_bytes()
                    for pc in result['pcs']:
                        addr=int(pc['pc'],16);assert common.instruction_bytes(gd,addr)==common.instruction_bytes(dd,addr)
                    save(p/'phase_pc_analysis.json',result)
                    state['completed'].append(dict(case=case,path=str(p),roi_us=counters['roi_us']))
                    save(out/'status.json',state);report()
                print('accepted',case,counters['roi_us'],flush=True)
            except BaseException as exc:
                stop.set();save(p/'failure.json',dict(error=repr(exc)))
                if proc is not None and proc.poll() is None:
                    proc.terminate()
                    try:proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()
                raise
        for name,cases in [('gates',gates),('targets',targets)]:
            status(name)
            with cf.ThreadPoolExecutor(max_workers=4) as pool:
                for i in range(0,len(cases),4):
                    jobs=[pool.submit(run,c) for c in cases[i:i+4]]
                    for j in jobs:j.result()
            save(out/(name+'_complete.json'),state['completed'])
        status('complete')
    except BaseException as exc:
        stop.set();state['error']=repr(exc);status('failed');raise

if __name__=='__main__':main()
