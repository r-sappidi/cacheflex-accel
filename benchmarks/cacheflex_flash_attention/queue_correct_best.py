"""Bounded, independently tuned corrected FA queue; four simulations maximum.

Usage: queue_correct_best.py NEW_ABSOLUTE_OUT [--resume]
Complete means executable stages completed, NOT unresolved kernel-development
items listed explicitly in plan.json. Source mutation and failed gates stop queue.
"""
from pathlib import Path
import concurrent.futures as cf
import hashlib, json, math, os, shutil, subprocess, sys, threading, time
import numpy as np
from validate_attribution import read
from run_causal import audit
from fa_independent_reference import reference, check

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
OLD=ROOT/'results/cacheflex_flash_attention/residual_retry_20260911'
DEFAULT=dict(placement='cache',group=4,overlap=1,unroll=1,order=0,rows=0,
             pipeline=0,fusion=0,offset=0,pinned=0,interleave=0)

def key(c):
    return hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()[:16]

def eligible(c,t,br,bc):
    k=c['placement'] in ('k','kv'); v=c['placement'] in ('v','kv')
    sets=((t//128)*64 if c['pinned'] else (1+c['overlap'])*(bc//128)*64)*k
    sets+=(1+c['overlap'])*bc*v
    return sets+c['offset']<=1024 and not(c['interleave'] and c['fusion'])

def main():
    out=Path(sys.argv[1]).resolve(); resume='--resume' in sys.argv
    out.mkdir(exist_ok=resume)
    for n in ('builds','runs','references'): (out/n).mkdir(exist_ok=resume)
    # Prevent duplicate orchestrators for this output directory.
    import fcntl
    lease=(out/'queue.lock').open('w');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    watched=list(HERE.glob('*.py'))+list(HERE.glob('*.hpp'))+[HERE/'spm_multicore.cpp',HERE/'cache_qk_wrapper.cpp']
    watched+=list((ROOT.parent/'cacheflex_micro/experiments/end2end/src').glob('*.hpp'))
    watched += [ROOT.parent/'cacheflex_micro/experiments/end2end/src/cache_flash_attn.cpp',
        ROOT.parent/'cacheflex_micro/kernels/attention_kernels_spm/11_flash_attn_kv_spm_vl4.cpp']
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    manifest=out/'source_hashes.json'
    if resume: hashes=json.loads(manifest.read_text())
    else:
        hashes={str(p):digest(p) for p in watched};manifest.write_text(json.dumps(hashes,indent=2))
        shutil.copy2(OLD/'gem5.opt',out/'gem5.opt')
        (out/'simulator.sha256').write_text(digest(out/'gem5.opt'))
    def unchanged():
        for p,h in hashes.items(): assert digest(Path(p))==h,('modified dependency',p)
        assert digest(out/'gem5.opt')==(out/'simulator.sha256').read_text()
    unchanged()
    plan=dict(workload=dict(T=1024,d=64,Hq=8,Hkv=1,cores=8,VL=16,causal=False),
      hardware='Frozen default repaired Ruby, four eligible ways, no DMA/batch/multicast; no extra SRAM',
      stages=[
        'paired periodic + independent nonperiodic gates and archived-winner controls',
        'cache/K tiles Br16/32/64/128 x Bc128/256/512 x groups1/2/4/8/all (deduplicated)',
        'top two tiles per architecture: K/V/K+V x single/double x groups1/2/4/8/all; pinned K',
        'top two SPM candidates: set offsets0/128/256/512 subject to capacity',
        'top two per architecture: unroll1/2/4 x load-order0/1; QK rows8/16/24 x pipeline0/1 x rescale-fusion0/1',
        'top two per architecture: grouped phase order off/on; fusion disabled for phase-interleaved path',
        'winner neighborhood retile and regroup; retain every earlier candidate',
        'top two per architecture: full T1024 nonperiodic validation and three timing repetitions'],
      pending_implementation=[
        'One/two-way K layout: existing 2048-bit spm.ld1qd spans four ways; needs predicated loads and lane reassembly',
        'Direct softmax-to-packed-P fusion (distinct from existing rescale fusion)',
        'Wider PV register tiles (existing PV unroll is included)',
        'Query-block multicore scheduling alternatives to one whole head/core',
        'Multiple nonperiodic distributions/seeds (one rich distribution currently implemented)'],
      acceptance='Independent reference at each Bc; analytic gate; copies complete; full hardware audit; whole-operator ROI. Top candidates selected independently, not exhaustive Cartesian search.',
      numerical='2e-6+1% abs(reference); FP16 GEMMs/FP32 online state; per-Bc reference required',
      max_parallel_simulations=4)
    (out/'plan.json').write_text(json.dumps(plan,indent=2))
    rows=[];events=[];stage='starting';lock=threading.Lock();built={};refs={}
    baseline_cfg=read(OLD/'original_cache_t128_p0',16)[2]
    def save():
        tmp=out/'status.tmp'
        tmp.write_text(json.dumps(dict(stage=stage,pid=os.getpid(),updated=time.time(),
            accepted_runs=len(rows),events=events[-30:],pending_implementation=plan['pending_implementation']),indent=2))
        tmp.replace(out/'status.json')
        (out/'progress.json').write_text(json.dumps(rows,indent=2))
    def event(s):
        with lock:events.append(s);save();print(s,flush=True)
    def ref(t,bc):
        name=(t,bc)
        if name not in refs:
            p=out/'references'/f'rich_t{t}_bc{bc}.npy'
            if not p.exists():
                event(f'independent reference T{t} Bc{bc}')
                np.save(p,reference(t,rich=True,br=64,bc=bc))
            refs[name]=np.load(p)
        return refs[name]
    def simulate(c,t,br,bc,label,guest,rich=False,rep=0):
        ident=key(dict(c=c,t=t,br=br,bc=bc,label=label,rich=rich,rep=rep))
        p=out/'runs'/ident; done=p/'accepted.json'
        if done.exists():
            row=json.loads(done.read_text())
            with lock: rows.append(row);save()
            return row
        p.mkdir(exist_ok=True)
        spm=c['placement']!='cache'
        template=OLD/f'original_{"spm" if spm else "cache"}_t128_p0'
        args=json.loads((template/'command.json').read_text());args[0]=str(out/'gem5.opt')
        args=[f'--outdir={p}' if x.startswith('--outdir=') else x for x in args]
        args[args.index('--cmd')+1]=str(guest)
        args+=['-P',f"system.cpu[0].workload[0].cmd=['cacheflex-fa','{t}','64','{br}','{bc}']"]
        (p/'command.json').write_text(json.dumps(args,indent=2))
        with (p/'run.log').open('w') as log:
            subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=7200,env=env)
        output,counters,cfg=read(p,16);audit(baseline_cfg,cfg,0)
        for a,b in dict(T=t,Br=br,Bc=bc,d=64,Hq=8,Hkv=1,cores=8).items(): assert output[a]==str(b),(a,output)
        assert output['nonfinite']=='0' and (p/'output.fp16').stat().st_size==8*t*64*2
        if rich: validation=check(p/'output.fp16',t,expected=refs[t,bc])
        else:
            assert 'FA_ANALYTIC errors=0' in (p/'run.log').read_text()
            validation=dict(errors=0,analytic='identical V rows')
        groups=math.ceil(math.ceil(t/br)/(c['group'] or math.ceil(t/br)))
        if c['pinned']:groups=1
        copies=8*groups*((t//128)*64*4*(c['placement'] in ('k','kv'))+t*2*(c['placement'] in ('v','kv')))
        assert counters['m_spmcpRequestsIssued']==counters['m_spmcpRequestsCompleted']==copies,(c,copies,counters)
        row=dict(config=c,t=t,br=br,bc=bc,stage=label,rich=rich,rep=rep,path=str(p),
                 roi_us=counters['roi_us'],output=output,validation=validation,counters=counters)
        done.write_text(json.dumps(row,indent=2))
        with lock:rows.append(row);save();print(label,ident,c['placement'],br,bc,c['group'],row['roi_us'],'PASS',flush=True)
        return row
    def build(c,rich):
        config=dict(c,rich=int(rich));ident=key(config);p=out/'builds'/ident
        if not (p/'experiment.json').exists():
            unchanged()
            f=out/'builds'/(ident+'.json');f.write_text(json.dumps(config,indent=2))
            assert not p.exists(),('incomplete build; inspect before resuming',p)
            with (out/'builds'/(ident+'.log')).open('w') as log:
                subprocess.run(['python3',str(HERE/'build_correct_search.py'),str(p),str(f)],check=True,stdout=log,stderr=subprocess.STDOUT,timeout=600,env=env)
        return p/'guest'
    def prepare(c,bc):
        # Exercise two K/V panels even at Bc512, rather than a truncated smoke.
        ident=key(c),bc
        if ident not in built:
            t=max(256,2*bc);ref(t,bc)
            guest=build(c,False);rich=build(c,True)
            simulate(c,t,32,bc,'periodic_gate',guest)
            simulate(c,t,32,bc,'independent_gate',rich,True)
            built[ident]=guest
        return built[ident]
    def batch(jobs,label):
        nonlocal stage
        stage=label;event('begin '+label)
        unique={key(dict(c=c,br=br,bc=bc)):(c,br,bc) for c,br,bc in jobs if eligible(c,1024,br,bc)}
        # Build/gate serially; timing jobs use at most four workers.
        jobs=[(c,br,bc,prepare(c,bc)) for c,br,bc in unique.values()]
        unchanged()
        result=[]
        with cf.ThreadPoolExecutor(max_workers=4) as pool:
            futures=[pool.submit(simulate,c,1024,br,bc,label,guest) for c,br,bc,guest in jobs]
            for f in cf.as_completed(futures):result.append(f.result())
        event('complete '+label);return result
    def best(pool,spm,n=2):
        candidates=[r for r in pool if (r['config']['placement']!='cache')==spm]
        seen=set();answer=[]
        for r in sorted(candidates,key=lambda r:r['roi_us']):
            ident=key(dict(c=r['config'],br=r['br'],bc=r['bc']))
            if ident not in seen:answer.append(r);seen.add(ident)
            if len(answer)==n:break
        return answer
    try:
        pool=batch([(dict(DEFAULT,placement=p),64,128) for p in ('cache','k')],'controls')
        jobs=[]
        for p in ('cache','k'):
            for br in (16,32,64,128):
                for bc in (128,256,512):
                    for g in sorted(set(min(x,1024//br) for x in (1,2,4,8,1024//br))):
                        jobs.append((dict(DEFAULT,placement=p,group=g),br,bc))
        pool+=batch(jobs,'tiles_groups')
        jobs=[]
        seeds=best(pool,False)+best(pool,True)
        for seed in seeds:
            br,bc=seed['br'],seed['bc']
            for p in ('k','v','kv'):
                for overlap in (0,1):
                    for g in sorted(set(min(x,1024//br) for x in (1,2,4,8,1024//br))):
                        jobs.append((dict(DEFAULT,placement=p,group=g,overlap=overlap),br,bc))
            for g in (1,2,4,8,0):jobs.append((dict(DEFAULT,placement='k',group=g,pinned=1),br,bc))
        pool+=batch(jobs,'staging_buffering_pinning')
        jobs=[]
        for seed in best(pool,True):
            for offset in (0,128,256,512):jobs.append((dict(seed['config'],offset=offset),seed['br'],seed['bc']))
        pool+=batch(jobs,'set_offsets')
        jobs=[]
        for seed in best(pool,False)+best(pool,True):
            for u in (1,2,4):
                for order in ((0,1) if seed['config']['placement']!='cache' else (0,)):
                    jobs.append((dict(seed['config'],unroll=u,order=order),seed['br'],seed['bc']))
            for r in (8,16,24):
                for p in (0,1):
                    for f in (0,1):jobs.append((dict(seed['config'],rows=r,pipeline=p,fusion=f),seed['br'],seed['bc']))
        pool+=batch(jobs,'compute_scheduling_fusion')
        jobs=[]
        for seed in best(pool,False)+best(pool,True):
            for interleave in (0,1):jobs.append((dict(seed['config'],interleave=interleave,fusion=0 if interleave else seed['config']['fusion']),seed['br'],seed['bc']))
        pool+=batch(jobs,'phase_order')
        jobs=[]
        for seed in best(pool,False)+best(pool,True):
            for br in sorted(set(max(16,min(128,x)) for x in (seed['br']//2,seed['br'],seed['br']*2))):
                for bc in sorted(set(max(128,min(512,x)) for x in (seed['bc']//2,seed['bc'],seed['bc']*2))):
                    for g in sorted(set(max(1,min(1024//br,x)) for x in (seed['config']['group']//2,seed['config']['group'],seed['config']['group']*2))):
                        jobs.append((dict(seed['config'],group=g),br,bc))
        pool+=batch(jobs,'winner_neighborhood')
        stage='finalists';event('begin finalists')
        finals=[]
        for seed in best(pool,False)+best(pool,True):
            c=seed['config'];br=seed['br'];bc=seed['bc'];ref(1024,bc)
            guest=build(c,True)
            for rep in range(3):finals.append(simulate(c,1024,br,bc,stage,guest,True,rep))
        (out/'summary.json').write_text(json.dumps(dict(periodic_candidates=pool,rich_finalists=finals,pending_implementation=plan['pending_implementation']),indent=2))
        stage='implemented_stages_complete';event(stage)
    except BaseException as e:
        stage='failed';event(repr(e));raise

if __name__=='__main__':main()
