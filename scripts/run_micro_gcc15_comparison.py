"""Identical GCC15 guests on Classic and Ruby, single CPU, correctness gated."""
import concurrent.futures as cf
import json
import os
from pathlib import Path
import subprocess
import time
from run_micro_fa_ruby_target import validate
from run_micro_fa_ruby_pilot import ROOT,save,sha

B=ROOT/'results/cacheflex_flash_attention'
G=B/'micro_gcc15_corrected_gate_20260918'
C=B/'micro_gcc15_classic_gate_20260918'
O=B/'micro_gcc15_comparison_20260918'
ENV=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')

def main():
    assert json.loads((C/'passed.json').read_text())['all_numerical_and_cross_model_hash_checks_passed']
    assert json.loads((G/'status.json').read_text())['stage']=='gates_passed_target_configuration_pending'
    O.mkdir();(O/'runs').mkdir()
    state=dict(pid=os.getpid(),completed=[])
    def status(stage):
        state.update(stage=stage,updated=time.time());save(O/'status.json',state)
        print(stage,flush=True)
    save(O/'plan.json',dict(T=4096,Hq=32,Hkv=8,d=64,cores=1,Br=64,Bc=384,VL_order=[16,4],max_simulators=4,
        compiler='Arm GNU15.2.1',upstream_commit=json.loads((G/'upstream.json').read_text())['commit'],
        precision='FP16 operands/GEMM, FP32 softmax/running output accumulation, FP16 output',
        validation='Full independent numerical/reference and guest-export hash checks; cross-model bit equality',
        comparison='First cold ROI for each; cache also executes a warm epoch, excluded from comparison',
        limits='One fixed tile candidate. Hq32/Hkv8 follows upstream sweep, not confirmed winning table manifest. Classic memory latencies/prefetch/MSHR/buses differ from Ruby. Common core fields/FU operations audited. Corrected conversion kernels, not original paper binaries.',
        supersedes='micro_upstream_t4096_20260918 GCC11 jobs stopped; old data retained separately'))
    guests={f'{a}_vl{vl}':str(G/'builds'/f'{a}_vl{vl}/guest') for vl in (16,4) for a in ('cache','spm')}
    save(O/'guest_hashes.json',{p:sha(p) for p in guests.values()})
    def run(model,arch,vl):
        p=O/'runs'/f'{model}_{arch}_vl{vl}';p.mkdir()
        template=G/'runs'/f'{arch}_vl{vl}_t320' if model=='ruby' else C/f'{arch}_vl{vl}'
        cmd=json.loads((template/'command.json').read_text())
        cmd=[f'--outdir={p}' if x.startswith('--outdir=') else x for x in cmd]
        if model=='ruby':
            cmd=["system.cpu[0].workload[0].cmd=['micro-fa','4096','32','8','64','1','64','384']" if x.startswith('system.cpu[0].workload[0].cmd=') else x for x in cmd]
            guest=cmd[cmd.index('--cmd')+1]
        else:
            cmd[cmd.index('-o')+1]='4096 32 8 64 1 64 384'
            guest=cmd[cmd.index('-c')+1]
        assert guest==guests[f'{arch}_vl{vl}']
        save(p/'command.json',cmd);save(p/'hashes.json',{guest:sha(guest),cmd[0]:sha(cmd[0])})
        start=time.monotonic()
        with (p/'run.log').open('w') as log:
            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=ENV)
            try:
                while child.poll() is None:
                    save(p/'progress.json',dict(stage='simulating',pid=child.pid,host_elapsed_s=time.monotonic()-start,updated=time.time()))
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        if time.monotonic()-start>48*3600:raise RuntimeError('48h timeout')
            finally:
                if child.poll() is None:child.terminate();child.wait()
        assert child.returncode==0,(p,child.returncode)
        save(p/'progress.json',dict(stage='full_numerical_validation',host_elapsed_s=time.monotonic()-start,updated=time.time()))
        check=validate(p)
        stats={}
        for line in (p/'stats.txt').read_text().split('End Simulation Statistics')[0].splitlines():
            w=line.split()
            if len(w)>1:stats[w[0]]=w[1]
        row=dict(model=model,arch=arch,VL=vl,roi_us=float(stats['simTicks'])/1e6,numerical=check,path=str(p))
        save(p/'accepted.json',row);save(p/'progress.json',dict(stage='accepted',updated=time.time()))
        return row
    try:
        for vl in (16,4):
            status(f't4096_vl{vl}')
            with cf.ThreadPoolExecutor(max_workers=4) as pool:
                futures=[pool.submit(run,m,a,vl) for m in ('classic','ruby') for a in ('cache','spm')]
                for future in cf.as_completed(futures):
                    state['completed'].append(future.result());status(state['stage'])
            for arch in ('cache','spm'):
                paths=[O/'runs'/f'{m}_{arch}_vl{vl}' for m in ('classic','ruby')]
                eq={n:sha(paths[0]/f'{n}.fp16')==sha(paths[1]/f'{n}.fp16') for n in ('Q','K','V','O')}
                save(O/f'output_equivalence_{arch}_vl{vl}.json',eq)
                assert all(eq.values()),eq
                cpus=[json.loads((p/'config.json').read_text())['system']['cpu'][0] for p in paths]
                fields=['fetchWidth','decodeWidth','renameWidth','commitWidth','dispatchWidth','issueWidth','wbWidth','numROBEntries','numIQEntries','LQEntries','SQEntries','numPhysIntRegs','numPhysFloatRegs','numPhysVecRegs','numPhysVecPredRegs','cacheLoadPorts','cacheStorePorts','spmLoadPorts','spmStorePorts']
                for key in fields:assert cpus[0][key]==cpus[1][key],key
                def fus(cpu):return [(f['count'],[(o['opClass'],o['opLat'],o['pipelined']) for o in f['opList']]) for f in cpu['fuPool']['FUList']]
                assert fus(cpus[0])==fus(cpus[1]),'FU mismatch'
                save(O/f'core_equivalence_{arch}_vl{vl}.json',dict(passed=True,fields=fields,FU=True))
            selected=[r for r in state['completed'] if r['VL']==vl]
            times={m:{r['arch']:r['roi_us'] for r in selected if r['model']==m} for m in ('classic','ruby')}
            save(O/f'summary_vl{vl}.json',dict(latency_us=times,speedup={m:t['cache']/t['spm'] for m,t in times.items()},ruby_over_classic_latency={a:times['ruby'][a]/times['classic'][a] for a in ('cache','spm')}))
        status('complete')
    except BaseException as e:
        state['error']=repr(e);status('failed');raise

if __name__=='__main__':main()
