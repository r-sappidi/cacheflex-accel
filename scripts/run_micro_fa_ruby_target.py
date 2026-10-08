"""Correctness-gated one-core T4096 MICRO-driver transfer to Ruby.

Uses repo LLaMA Hq32/Hkv8 and one declared tile candidate, not a tuned claim.
"""
import concurrent.futures as cf
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import numpy as np
from run_micro_fa_ruby_pilot import ROOT,save,sha,gemm

def validate(p):
    text=(p/'run.log').read_text(errors='replace');arrays={}
    for name,heads in [('Q',32),('K',8),('V',8),('O',32)]:
        raw=np.fromfile(p/f'{name}.fp16',dtype=np.uint16);h=1469598103934665603
        for x in raw:h=((h^int(x))*1099511628211)&((1<<64)-1)
        assert f'EXPORT {name}.fp16 bytes={raw.size*2} hash={h:016x}' in text
        arrays[name]=raw.view(np.float16).reshape(heads,4096,64)
    errors=0;nonfinite=0;maxabs=0.
    for h in range(32):
        q=arrays['Q'][h];k=arrays['K'][h//4];v=arrays['V'][h//4];actual=arrays['O'][h]
        for i in range(0,4096,32):
            qq=q[i:i+32];m=np.full(32,-1e30,np.float32);l=np.zeros(32,np.float32);acc=np.zeros((32,64),np.float32)
            for j in range(0,4096,384):
                s=gemm(qq,k[j:j+384].T).astype(np.float32)*np.float32(.125)
                mn=np.maximum(m,s.max(axis=1));alpha=np.exp((m-mn).astype(np.float64)).astype(np.float32)
                z=np.exp((s-mn[:,None]).astype(np.float64)).astype(np.float32)
                l=(l*alpha+z.sum(axis=1,dtype=np.float32)).astype(np.float32)
                acc=(acc*alpha[:,None]+gemm(z.astype(np.float16),v[j:j+384]).astype(np.float32)).astype(np.float32);m=mn
            expected=(acc/l[:,None]).astype(np.float16);o=actual[i:i+32]
            err=np.abs(o.astype(np.float64)-expected.astype(np.float64))
            errors+=int((err>2e-6+.01*np.abs(expected.astype(np.float64))).sum());nonfinite+=int((~np.isfinite(o)).sum());maxabs=max(maxabs,float(err.max()))
        save(p/'reference_progress.json',dict(heads_checked=h+1,heads_total=32,errors=errors,nonfinite=nonfinite,maxabs=maxabs))
    assert errors==nonfinite==0,(errors,nonfinite,maxabs)
    return dict(elements=32*4096*64,errors=errors,nonfinite=nonfinite,maxabs=maxabs)

def main():
    gate=ROOT/'results/cacheflex_flash_attention/micro_upstream_multitile_gate_v2_20260918'
    assert json.loads((gate/'status.json').read_text())['stage']=='gates_passed_target_configuration_pending'
    provenance=json.loads((gate/'upstream.json').read_text())
    assert provenance['commit']=='82ac0b9de1ed408ce5cd53067267c8a855b4eef3'
    out=Path(sys.argv[1]).resolve();out.mkdir();(out/'runs').mkdir()
    state=dict(pid=os.getpid(),stage='starting',completed=[])
    def status(stage):state.update(stage=stage,updated=time.time());save(out/'status.json',state);print(stage,flush=True)
    save(out/'upstream.json',provenance)
    save(out/'plan.json',dict(T=4096,Hq=32,Hkv=8,d=64,cores=1,VL=[16,4],Br=64,Bc=384,
      heads_source='cacheflex_micro/experiments/end2end/scripts/sweep_vl16.sh LLaMA argv',
      target_speedup={'4':1.11,'16':1.34},original_micro_drivers=False,corrected_dense_fp16_conversions=True,
      caution='One fixed tile candidate, no best-to-best tuning claim. Head count follows repo sweep, not confirmed winning table manifest. Ruby is not Classic. Kernel-level, not layer-level timing.',
      hardware='64KiB L1I/D,512KiB private L2,4MiB LLC,DDR4-2400 two channels, Ruby sequencer16, prefetch enabled, four eligible SPM ways, retention default off',
      compare='Cold ROI stats for both; cache warm timing separately reported, never silently compared to cold SPM.',max_simulators=2))
    save(out/'precision.json',dict(operands='FP16',gemm_accumulation='FP16',softmax_state='FP32',running_output_accumulation='FP32',stored_output='FP16',
        fix='ZIP1 before widening and UZP1 after narrowing for dense arrays; corrected upstream, not unchanged paper binary',
        baseline='same precision and corrected conversion handling',gate=str(gate)))
    def run(arch,vl):
        p=out/'runs'/f'{arch}_vl{vl}';p.mkdir()
        source=gate/'runs'/f'{arch}_vl{vl}_t320'
        cmd=json.loads((source/'command.json').read_text())
        cmd=[f'--outdir={p}' if x.startswith('--outdir=') else x for x in cmd]
        cmd=["system.cpu[0].workload[0].cmd=['micro-fa','4096','32','8','64','1','64','384']" if x.startswith('system.cpu[0].workload[0].cmd=') else x for x in cmd]
        save(p/'command.json',cmd);save(p/'hashes.json',{cmd[0]:sha(cmd[0]),cmd[cmd.index('--cmd')+1]:sha(cmd[cmd.index('--cmd')+1])})
        env=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        start=time.monotonic()
        with (p/'run.log').open('w') as f:
            proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env)
            try:
                while True:
                    save(p/'progress.json',dict(stage='simulating',pid=proc.pid,host_elapsed_s=time.monotonic()-start,updated=time.time()))
                    try:rc=proc.wait(timeout=10);break
                    except subprocess.TimeoutExpired:
                        if time.monotonic()-start>48*3600:raise RuntimeError('48h timeout')
            finally:
                if proc.poll() is None:proc.terminate();proc.wait()
        assert rc==0,(arch,vl,rc)
        stats={}
        for line in (p/'stats.txt').read_text().split('End Simulation Statistics')[0].splitlines():
            w=line.split()
            if len(w)>1:stats[w[0]]=w[1]
        save(p/'progress.json',dict(stage='full_numerical_reference',updated=time.time()))
        row=dict(arch=arch,VL=vl,roi_us=float(stats['simTicks'])/1e6,numerical=validate(p),path=str(p))
        save(p/'accepted.json',row);return row
    try:
        for vl in (16,4):
            status(f't4096_vl{vl}')
            with cf.ThreadPoolExecutor(max_workers=2) as pool:
                for r in pool.map(lambda arch:run(arch,vl),('cache','spm')):
                    state['completed'].append(r);status(state['stage'])
        results=[]
        for vl in (4,16):
            pair={r['arch']:r['roi_us'] for r in state['completed'] if r['VL']==vl}
            results.append(dict(VL=vl,**pair,speedup=pair['cache']/pair['spm']))
        save(out/'summary.json',results);status('complete_candidate_not_full_reproduction')
    except BaseException as e:state['error']=repr(e);status('failed');raise

if __name__=='__main__':main()
