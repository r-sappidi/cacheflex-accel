"""Bounded mapping-aware operator screen, with two simulator slots.

First/warmed invocations are separate objectives. This is not a model benchmark
or proof of optimality. New output directory required; failures stop promotion.
"""
import concurrent.futures as cf
import configparser
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time
import numpy as np
from screen_reference import generate, validate
from sweep_mlp_study import epochs, audit
from audit_mlp_math import audit as math_audit

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SIM=ROOT/'results/ruby_getx_fix_20260916_build_v3/gem5.opt'
SEED=ROOT/'results/cacheflex_flash_attention/single_head_20260915/runs/cache_t512_br32_bc128_g2_p1'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p,data):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(p)


def candidates():
    result=[]
    for family, shapes in {
        'grouped_projection':[(32,512,512,8,0),(64,1024,1024,8,0)],
        'shared_weight_batch':[(32,512,512,8,1),(64,1024,1024,8,1)],
        'pointwise_1x1':[(96,512,512,1,0),(784,1024,1024,1,0)],
    }.items():
        for m,k,n,groups,shared in shapes:
            mappings=[(8,1),(8,2),(8,4),(1,1)] if groups>1 else [(8,1),(8,2),(8,4),(8,8)]
            # Two KC values across two rounds; every mapping gets both.
            for kc in (128,256):
                for team,mp in mappings:
                    result.append(dict(family=family,m=m,k=k,n=n,groups=groups,shared=shared,team=team,mp=mp,kc=kc,combined=0))
    for t in (64,128):
        for kc,mp,combined in itertools.product((128,256),(1,2,4,8),(0,1)):
            result.append(dict(family='swiglu',m=t,k=512,n=1536,groups=1,shared=0,team=8,mp=mp,kc=kc,combined=combined))
    return result


def expected_copies(c):
    if c['family']=='swiglu':
        return min(c['mp'],math.ceil(c['m']/24))*4*(2*c['k']*(c['n']//128)+c['n']*(c['k']//128))
    return c['groups']*min(c['mp'],math.ceil(c['m']/24))*4*c['k']*(c['n']//128)


def main():
    out=Path(sys.argv[1]).resolve();out.mkdir()
    for name in ('builds','runs','inputs'):(out/name).mkdir()
    state=dict(pid=os.getpid(),stage='preparing',completed=[],updated=time.time())
    def status(stage,**kw):
        state.update(stage=stage,updated=time.time(),**kw);save(out/'status.json',state);print(stage,flush=True)
    env=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    try:
        plan=candidates()
        screening=[]
        for c in plan:
            mlp=c['family']=='swiglu'
            weights=6*c['k']*c['n'] if mlp else 2*(1 if c['shared'] else c['groups'])*c['k']*c['n']
            screening.append(dict(case=c,weight_bytes=weights,cache_packed_weight_bytes=weights,
                 useful_flops=(6 if mlp else 2*c['groups'])*c['m']*c['k']*c['n'],
                 spm_panel_bytes_per_active_core=4*c['kc']*64,
                 spm_copied_bytes_per_invocation=expected_copies(c)*64,
                 packed_a_upper_bound_per_core=math.ceil(c['m']/24)*24*c['kc']*2,
                 weight_to_total_llc_ratio=weights/4194304,
                 note='Storage/copy model only: no prediction of DRAM misses or elapsed time.'))
        save(out/'analytical_screen.json',screening)
        save(out/'plan.json',dict(candidates=plan,simulations=len(plan)*2,
             hardware=dict(cores=8,vl=16,l1i=65536,l1d=65536,private_l2=524288,llc_total=4194304),
             kernels='Existing pipelined 24x128 FP16 GEMM; corrected SwiGLU conversion',
             epochs='Three invocations: first includes weight preparation; next two reuse packed weights. Validation touches outputs between calls.',
             scope='Grouped projection only (no routing), batched shared-weight GEMM, plain NHWC 1x1 convolution, complete SwiGLU.',
             pending=['3x3 convolution','end-to-end routed MoE','activation-resident SPM families','microkernel row variants','finalist confirmation on held-out shapes/seeds','production-library controls'],
             screening_only=True,recorder_entries=0,max_concurrent_simulators=2))
        status('building')
        assert sha(SIM)==(SIM.parent/'simulator.sha256').read_text().strip()
        shutil.copy2(SIM,out/'gem5.opt')
        for name in ('projection','mlp_separate','mlp_combined'):
            build=out/'builds'/name
            cmd=['bash',str(HERE/'build_screen_projection.sh'),str(build)] if name=='projection' else [sys.executable,str(HERE/'build_screen_mlp.py'),str(build)]
            with (out/'builds'/f'{name}.log').open('w') as log:
                subprocess.run(cmd,env=dict(env,MLP_COMBINED=str(int(name=='mlp_combined'))),stdout=log,stderr=subprocess.STDOUT,check=True)
        paths=list((out/'builds').rglob('*'))+list(HERE.glob('*.py'))+list(HERE.glob('*.cpp'))+list(HERE.glob('*.sh'))+list((ROOT/'gem5/configs').rglob('*.py'))
        paths+=[out/'gem5.opt',SEED/'command.json',SEED/'config.ini']
        manifest={str(p):sha(p) for p in paths if p.is_file()}
        save(out/'manifest.json',manifest)
        def frozen():
            for p,digest in manifest.items():assert sha(p)==digest,('changed dependency',p)
        seed=json.loads((SEED/'command.json').read_text())

        def inp(c):
            key={k:c[k] for k in ('family','m','k','n','groups','shared','kc')}
            path=out/'inputs'/(hashlib.sha256(json.dumps(key,sort_keys=True).encode()).hexdigest()[:16]+'.bin')
            if not path.exists():
                if c['family']=='swiglu':
                    subprocess.run([sys.executable,str(HERE/'mlp_reference.py'),str(c['m']),str(c['k']),str(c['n']),str(path),str(c['kc'])],env=env,check=True)
                else:generate(path,c['m'],c['k'],c['n'],c['groups'],c['shared'],c['kc'])
            return path

        def run(c,arch,input_path):
            name=hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()[:16]+'_'+arch
            path=out/'runs'/name;path.mkdir()
            mlp=c['family']=='swiglu'
            build='mlp_combined' if mlp and c['combined'] else 'mlp_separate' if mlp else 'projection'
            guest=out/'builds'/build/arch
            args=[c['m'],c['k'],c['n'],c['mp'],c['kc'],0,3,1,8] if mlp else [c['m'],c['k'],c['n'],c['groups'],c['shared'],c['team'],c['mp'],c['kc'],3,8]
            cmd=list(seed);cmd[0]=str(out/'gem5.opt')
            values={'--outdir':str(path),'--l1d_size':'512kB','--l1d_num_spm_ways':'4' if arch=='spm' else '0','--prog-interval':'0Hz'}
            for key,value in values.items():
                indexes=[i for i,x in enumerate(cmd) if x.startswith(key+'=')];assert len(indexes)==1
                cmd[indexes[0]]=key+'='+value
            cmd[cmd.index('--cmd')+1]=str(guest)
            cmd[cmd.index('--options')+1]=' '.join(map(str,args))
            i=next(i for i,x in enumerate(cmd) if x.startswith('system.cpu[0].workload[0].cmd='))
            cmd[i]='system.cpu[0].workload[0].cmd='+repr(['operator-screen']+list(map(str,args)))
            cmd+=['--ruby-flight-recorder-entries=0','-P',f"system.readfile='{input_path}'"]
            save(path/'command.json',cmd);save(path/'case.json',c)
            start=time.monotonic()
            with (path/'run.log').open('w') as log:
                proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
                while True:
                    save(path/'progress.json',dict(pid=proc.pid,stage='running',updated=time.time(),host_elapsed_s=time.monotonic()-start))
                    try:rc=proc.wait(timeout=10);break
                    except subprocess.TimeoutExpired:pass
            elapsed=time.monotonic()-start
            save(path/'exit.json',dict(returncode=rc,host_elapsed_s=elapsed));assert rc==0,(path,rc)
            text=(path/'run.log').read_text();assert 'exiting with last active thread context' in text
            assert not re.search(r'panic:|fatal:| FAIL|Simulated exit code not 0',text)
            checks=[dict(re.findall(r'(\w+)=([^\s]+)',line)) for line in text.splitlines() if line.startswith('MLP_RESULT' if mlp else 'SCREEN_RESULT')]
            assert len(checks)==3 and all(r['errors']==r['nonfinite']=='0' for r in checks)
            assert len({r['hash'] for r in checks})==1
            times=epochs(path,3);copies=expected_copies(c) if arch=='spm' else 0
            assert all(r['copies_issued']==r['copies_completed']==copies for r in times),(c,copies,times)
            if mlp:
                data=np.fromfile(input_path,dtype='<f2');expected=data[-c['m']*c['k']:]
                actual=np.fromfile(path/'output.bin',dtype='<f2');assert actual.shape==expected.shape
                error=np.abs(actual.astype(float)-expected.astype(float))
                assert np.isfinite(actual).all() and (error<=2e-6+.01*np.abs(expected.astype(float))).all()
                accuracy=math_audit(path,input_path,c['m'],c['k'],c['n']);assert accuracy['passed']
                h=1469598103934665603
                for word in actual.view(np.uint16):h=((h^int(word))*1099511628211)&((1<<64)-1)
                digest=f'{h:016x}'
            else:
                accuracy=validate(path,input_path,c['m'],c['k'],c['n'],c['groups'],c['shared']);digest=accuracy['hash']
            assert digest==checks[-1]['hash'],('export mismatch',path)
            cfg=configparser.ConfigParser(interpolation=None,strict=False);cfg.read(path/'config.ini')
            prior=configparser.ConfigParser(interpolation=None,strict=False);prior.read(SEED/'config.ini')
            # All prior instantiated fields must match, except explicit launch,
            # private-cache geometry and SPM eligibility changes in this plan.
            for sec in prior.sections():
                for key,value in prior[sec].items():
                    if (sec=='system' and key=='readfile') or (sec=='system.cpu0.workload' and key in ('cmd','executable')):continue
                    if re.fullmatch(r'system.cpu\d+',sec) and key=='progress_interval':continue
                    if re.fullmatch(r'system.ruby.l1_cntrl\d+.cache',sec) and key in ('size','num_spm_ways'):continue
                    assert cfg[sec][key].replace(str(path),'<RUN>')==value.replace(str(SEED),'<RUN>'),(sec,key)
            for i in range(8):
                assert cfg[f'system.cpu{i}.isa']['sve_vl_se']=='16'
                assert cfg[f'system.ruby.l1_cntrl{i}.cache']['size']=='524288'
                assert cfg[f'system.ruby.l1_cntrl{i}.cache']['num_spm_ways']==str(4 if arch=='spm' else 0)
                assert cfg[f'system.ruby.l2_cntrl{i}.L2cache']['size']=='524288'
                assert cfg[f'system.ruby.l2_cntrl{i}']['spm_retain_llc']=='false'
            controllers=[s for s in cfg.sections() if 'flight_recorder_entries' in cfg[s]]
            assert len(controllers)==32 and all(cfg[s]['flight_recorder_entries']=='0' for s in controllers)
            assert not list(path.glob('*.flight.bin'))
            row=dict(path=str(path),case=c,variant=arch,vl=16,checks=checks,epochs=times,accuracy=accuracy,
                     first_us=times[0]['us'],warm_us=statistics.median(r['us'] for r in times[1:]),
                     warm_spread=(max(r['us'] for r in times[1:])-min(r['us'] for r in times[1:]))/statistics.median(r['us'] for r in times[1:]),
                     host_elapsed_s=elapsed,input_sha256=sha(input_path),guest_sha256=sha(guest))
            save(path/'accepted.json',row);save(path/'progress.json',dict(stage='accepted',host_elapsed_s=elapsed))
            return row

        rows=[]
        def pair(c):
            frozen();input_path=inp(c)
            with cf.ThreadPoolExecutor(max_workers=2) as pool:
                jobs=[pool.submit(run,c,a,input_path) for a in ('cache','spm')]
                a,b=[j.result() for j in jobs]
            audit(a,b);assert a['checks'][-1]['hash']==b['checks'][-1]['hash']
            rows.extend((a,b));save(out/'results.json',rows)
            state['completed'].append(dict(case=c,cache_first=a['first_us'],spm_first=b['first_us'],cache_warm=a['warm_us'],spm_warm=b['warm_us']))
            status(state['stage'])

        status('numerical_and_mapping_gates')
        # Padded rows, all teams, shared/distinct weights, and both MLP families.
        for shared,team in itertools.product((0,1),(1,8)):
            pair(dict(family='projection_gate',m=25,k=128,n=256,groups=8,shared=shared,team=team,mp=1 if team==1 else 2,kc=128,combined=0))
        for combined in (0,1):
            pair(dict(family='swiglu',m=24,k=128,n=256,groups=1,shared=0,team=8,mp=1,kc=128,combined=combined))
        save(out/'gates_passed.json',dict(pairs=6))
        # Breadth first: initial point per shape/family before refinements.
        grouped={}
        for c in plan:grouped.setdefault((c['family'],c['m'],c['k'],c['n']),[]).append(c)
        for round_index in range(max(map(len,grouped.values()))):
            status('screening',round=round_index)
            for points in grouped.values():
                if round_index<len(points):pair(points[round_index])
        winners=[]
        for shape in grouped:
            choices=[r for r in rows if (r['case']['family'],r['case']['m'],r['case']['k'],r['case']['n'])==shape]
            for objective in ('first_us','warm_us'):
                best={a:min((r for r in choices if r['variant']==a),key=lambda r:r[objective]) for a in ('cache','spm')}
                winners.append(dict(shape=shape,objective=objective,cache=best['cache']['path'],spm=best['spm']['path'],
                     cache_us=best['cache'][objective],spm_us=best['spm'][objective],
                     speedup=best['cache'][objective]/best['spm'][objective],needs_finalist_confirmation=True))
        save(out/'summary.json',dict(rows=rows,winners=winners,scope='Bounded initial screen, not final best-to-best or full models'))
        status('initial_screen_complete')
    except BaseException as exc:
        state['error']=repr(exc);status('failed');raise


if __name__=='__main__':main()
