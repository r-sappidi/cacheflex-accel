"""Q-resident single-head pilot then optional bounded persistent-Q sweep.

Progress uses gem5 CPUProgressEvent, not guest instructions or ROI printing.
No percentage/ETA is inferred from committed instructions.
"""
from pathlib import Path
import argparse
import concurrent.futures as cf
import hashlib
import itertools
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import numpy as np
from fa_export_patch import patch, verify_dump
from validate_attribution import read
from single_head_reference import reference, check

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT/'results/cacheflex_flash_attention/pv_improvements_20260913'


def save(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2)); temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--tokens', type=int, default=16384)
    parser.add_argument('--sweep', action='store_true')
    parser.add_argument('--spm-only', action='store_true', help='Reuse separately retained cache controls; run only SPM')
    parser.add_argument('--gate-tokens', type=int, nargs='+', default=[512,2048])
    args = parser.parse_args(); out = args.out.resolve(); out.mkdir()
    for name in ('builds', 'runs', 'references'): (out/name).mkdir()
    shutil.copy2(BASE/'gem5.opt', out/'gem5.opt')
    env = dict(os.environ, LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    seeds = {a:json.loads((BASE/'runs'/r/'command.json').read_text()) for a,r in
             [('cache','3c716544869382b2'),('spm','550744dbae7b1832')]}
    config = dict(group=4, overlap=0, unroll=1, order=0, rows=16,
                  pipeline=1, fusion=0, offset=0, pinned=0, interleave=0,
                  stride_fix=0, pv_fullwidth=1, rich=1)
    state = dict(stage='building', pid=os.getpid(), completed=[])
    plans = [('cache',32,128,2), ('spm',16,128,4)]
    if args.sweep:
        plans += [c for a in ('cache','spm') for br,bc,g in itertools.product(
            (16,32,64), (128,256,512), (1,2,4,8))
            if br*g<=64 and (c:=(a,br,bc,g)) not in plans]
    if args.spm_only:
        plans=[c for c in plans if c[0]=='spm']
    pilot_count=1 if args.spm_only else 2
    save(out/'plan.json', dict(T=args.tokens, heads=1, d=64, cores=8,
        l1i_bytes=65536,l1d_bytes=65536,private_l2_bytes=65536,llc_total_bytes=4194304,
        spm_eligible_ways=1, plans=plans,
        limits='Noncausal query-parallel only; 36 bounded points, not exhaustive best-to-best. SPM pins Q only, <=64 rows/group in one way. Full reference per Bc; no DMA/multicast/batch.'))
    watched=[HERE/n for n in ('build_q_resident.py','build_single_head.py','build_pv_improvements.py',
        'build_correct_search.py','make_search_kernels.py','make_pv_fullwidth.py',
        'spm_multicore.cpp','cache_qk_wrapper.cpp','correct_fa_conversions.hpp',
        'driver_vector_pack.hpp','single_head_reference.py','run_q_resident.py')]
    save(out/'sources.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in watched})
    def status(stage):
        state.update(stage=stage,updated=time.time());save(out/'status.json',state)
        print(stage,flush=True)
    def guest(a,g):
        p=out/'builds'/f'{a}_g{g}'
        if not p.exists():
            c=dict(config,placement='cache',q_resident=a=='spm',group=g)
            cp=p.with_suffix('.json');save(cp,c)
            with p.with_suffix('.log').open('w') as log:
                subprocess.run([sys.executable,str(HERE/'build_q_resident.py'),str(p),str(cp)],
                               check=True,env=env,stdout=log,stderr=subprocess.STDOUT)
        return patch(p/'guest')
    def run(case,t,progress=True):
        a,br,bc,g=case
        assert t%(8*br)==0
        p=out/'runs'/f'{a}_t{t}_br{br}_bc{bc}_g{g}_p{int(progress)}';p.mkdir()
        binary=guest(a,g)
        cmd=list(seeds[a]);cmd[0]=str(out/'gem5.opt')
        values={'--outdir':str(p),'--l0i_size':'64kB','--l0d_size':'64kB',
            '--l1d_size':'64kB','--l2_size':'512kB','--l1d_num_spm_ways':str(1 if a=='spm' else 0),
            '--prog-interval':'100kHz' if progress else '0Hz'}
        for key,val in values.items():
            hits=[i for i,x in enumerate(cmd) if x.startswith(key+'=')];assert len(hits)==1,(key,hits)
            cmd[hits[0]]=key+'='+val
        cmd[cmd.index('--cmd')+1]=str(binary)
        matches=[i for i,x in enumerate(cmd) if x.startswith('system.cpu[0].workload[0].cmd=')]
        assert len(matches)==1
        cmd[matches[0]]=f"system.cpu[0].workload[0].cmd=['cacheflex-fa','{t}','64','{br}','{bc}']"
        save(p/'command.json',cmd)
        start=time.time()
        with (p/'run.log').open('w') as log:
            proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
            while proc.poll() is None:
                text=(p/'run.log').read_text(errors='replace')
                events=re.findall(r'([^\n]*progress event[^\n]*)',text)
                save(p/'progress.json',dict(pid=proc.pid,host_elapsed_s=time.time()-start,
                    updated=time.time(),stage='ROI' if 'FLASH_ROI begin' in text else 'startup',
                    recent_cpu_events=events[-8:],kernel_percentage=None))
                time.sleep(10)
            if proc.returncode!=0:
                save(p/'progress.json',dict(stage='failed',returncode=proc.returncode,
                    host_elapsed_s=time.time()-start,updated=time.time(),pid=proc.pid))
                raise RuntimeError(f'{p}: simulator exited {proc.returncode}; see run.log')
        output,counters,cfg=read(p,16)
        assert output['Hq']=='1' and output['Hkv']=='1' and output['T']==str(t)
        assert (p/'output.fp16').stat().st_size==t*64*2
        copies=0 if a=='cache' else t*2
        assert counters['m_spmcpRequestsIssued']==counters['m_spmcpRequestsCompleted']==copies
        # Verify instantiated geometry, not just requested command arguments.
        caches=[cfg[s] for s in cfg.sections() if cfg[s].get('type')=='RubyCache']
        sizes=[int(c['size']) for c in caches]
        assert sizes.count(524288)==8,(sizes,'LLC must be eight 512KiB banks')
        assert sizes.count(65536)==24,(sizes,'8 each L1I, L1D and private L2')
        row=dict(path=str(p),case=case,T=t,output=output,counters=counters,roi_us=counters['roi_us'])
        verify_dump(row);save(p/'measured.json',row)
        save(p/'progress.json',dict(stage='simulation_complete_pending_validation',host_elapsed_s=time.time()-start))
        return row
    refs={}
    def ref(t,bc):
        key=(t,bc)
        if key not in refs:
            refs[key]=reference(t,bc);np.save(out/'references'/f't{t}_bc{bc}.npy',refs[key])
        return refs[key]
    def accept(row):
        row['validation']=check(Path(row['path'])/'output.fp16',ref(row['T'],row['case'][2]))
        save(Path(row['path'])/'accepted.json',row)
        save(Path(row['path'])/'progress.json',dict(stage='validated',roi_us=row['roi_us']))
        state['completed'].append(row);status('case_validated')
    try:
        status('smoke_and_progress_controls')
        for case in plans[:pilot_count]:
            for gate_t in args.gate_tokens:
                off=run(case,gate_t,False);on=run(case,gate_t,True)
                assert off['output']==on['output']
                assert {k:v for k,v in off['counters'].items() if k!='path'}=={k:v for k,v in on['counters'].items() if k!='path'}
                accept(off);accept(on)
        status('q_resident_pilot')
        with cf.ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(run,c,args.tokens) for c in plans[:pilot_count]]
            try:
                for f in cf.as_completed(futures):accept(f.result())
            except BaseException as e:
                # Report immediately, before executor shutdown waits for peers.
                state['error']=repr(e);status('failed_waiting_for_peer');raise
        status('pilot_complete')
        for case in plans[pilot_count:]:
            status('sweep');accept(run(case,args.tokens))
        status('complete')
    except BaseException as e:
        state['error']=repr(e);status('failed');raise


if __name__=='__main__':main()
