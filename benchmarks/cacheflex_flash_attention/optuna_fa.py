"""Persistent, correctness-gated Optuna FA tuning; no edits to active sweep.

Run with .venv-fa-optuna/bin/python optuna_fa.py OUT --after SWEEP.
An initial budget of 128 unique configurations per architecture is not a
convergence claim. Objective is full-size nonperiodic whole-operator latency.
"""
from pathlib import Path
import argparse, concurrent.futures as cf, fcntl, hashlib, json, math, os
import pickle, shutil, subprocess, sys, threading, time
import numpy as np
import optuna
from build_correct_search import build
from queue_correct_best import DEFAULT, eligible, key, OLD
from fa_independent_reference import check, reference
from run_causal import audit
from validate_attribution import read

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
LAYOUTS=dict(k_single=('k',0,0),k_double=('k',1,0),
             v_single=('v',0,0),v_double=('v',1,0),
             kv_single=('kv',0,0),kv_double=('kv',1,0),k_pinned=('k',1,1))

def suggest(trial, arch):
    br=trial.suggest_categorical('Br',[16,32,64,128])
    bc=trial.suggest_categorical('Bc',[128,256,512])
    group=trial.suggest_categorical('group',[1,2,4,8,16,32,64])
    c=dict(DEFAULT,group=min(group,1024//br))
    if arch=='spm':
        layout=trial.suggest_categorical('layout',list(LAYOUTS))
        c['placement'],c['overlap'],c['pinned']=LAYOUTS[layout]
        c['offset']=trial.suggest_categorical('offset',[0,128,256,512])
    kernel=trial.suggest_categorical('qk_kernel',['native','wide8','wide16','wide24'])
    c['rows']=0 if kernel=='native' else int(kernel[4:])
    if c['rows']: c['pipeline']=trial.suggest_categorical('pipeline',[0,1])
    elif arch=='spm' and c['placement'] in ('k','kv'):
        c['order']=trial.suggest_categorical('load_order',[0,1])
    c['unroll']=trial.suggest_categorical('unroll',[1,2,4])
    mode=trial.suggest_categorical('phase_mode',['plain','rescale_fused','interleaved'])
    c['fusion']=int(mode=='rescale_fused');c['interleave']=int(mode=='interleaved')
    return c,br,bc

class Evaluator:
    def __init__(self,out):
        self.out=out;self.lock=threading.Lock();self.refs={};self.build_locks={}
        self.refcfg=read(OLD/'original_cache_t128_p0',16)[2]
        self.env=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib',
                      OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        for n in ('runs','builds','references'):(out/n).mkdir(exist_ok=True)
    def getref(self,bc):
        with self.lock:
            if bc not in self.refs:
                p=self.out/'references'/f't1024_bc{bc}.npy'
                if not p.exists():np.save(p,reference(1024,rich=True,bc=bc))
                self.refs[bc]=np.load(p)
            return self.refs[bc]
    def guest(self,c,rich):
        cfg=dict(c,rich=int(rich));ident=key(cfg);p=self.out/'builds'/ident
        with self.lock:guard=self.build_locks.setdefault(ident,threading.Lock())
        with guard:
            if not (p/'experiment.json').exists():
                assert not p.exists(),('incomplete build',p)
                f=self.out/'builds'/(ident+'.json');f.write_text(json.dumps(cfg))
                with (self.out/'builds'/(ident+'.log')).open('w') as log:
                    subprocess.run([sys.executable,str(HERE/'build_correct_search.py'),str(p),str(f)],
                                   check=True,stdout=log,stderr=subprocess.STDOUT,timeout=600,env=self.env)
            return p/'guest'
    def run(self,c,br,bc,rich=True,rep=0):
        t=1024 if rich else max(256,2*bc)
        ident=key(dict(config=c,br=br,bc=bc,t=t,rich=rich,rep=rep))
        p=self.out/'runs'/ident
        if (p/'accepted.json').exists():return json.loads((p/'accepted.json').read_text())
        assert eligible(c,t,br,bc)
        guest=self.guest(c,rich);p.mkdir(exist_ok=True)
        template=OLD/f'original_{"cache" if c["placement"]=="cache" else "spm"}_t128_p0'
        args=json.loads((template/'command.json').read_text());args[0]=str(self.out/'gem5.opt')
        args=[f'--outdir={p}' if x.startswith('--outdir=') else x for x in args]
        args[args.index('--cmd')+1]=str(guest)
        args+=['-P',f"system.cpu[0].workload[0].cmd=['cacheflex-fa','{t}','64','{br}','{bc}']"]
        (p/'command.json').write_text(json.dumps(args,indent=2))
        with (p/'run.log').open('w') as log:
            subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=7200,env=self.env)
        output,counters,cfg=read(p,16);audit(self.refcfg,cfg,0)
        for name,value in dict(T=t,Br=br,Bc=bc,d=64,Hq=8,Hkv=1,cores=8).items():assert output[name]==str(value)
        assert output['nonfinite']=='0' and (p/'output.fp16').stat().st_size==8*t*64*2
        if rich:validation=check(p/'output.fp16',t,expected=self.getref(bc))
        else:
            assert 'FA_ANALYTIC errors=0' in (p/'run.log').read_text()
            validation=dict(errors=0,analytic='identical V rows')
        groups=1 if c['pinned'] else math.ceil(math.ceil(t/br)/(c['group'] or math.ceil(t/br)))
        copies=8*groups*((t//128)*256*(c['placement'] in ('k','kv'))+t*2*(c['placement'] in ('v','kv')))
        assert counters['m_spmcpRequestsIssued']==counters['m_spmcpRequestsCompleted']==copies
        row=dict(config=c,br=br,bc=bc,t=t,rich=rich,rep=rep,roi_us=counters['roi_us'],
                 path=str(p),output=output,validation=validation,counters=counters)
        (p/'accepted.json').write_text(json.dumps(row,indent=2));return row
    def evaluate(self,c,br,bc):
        self.run(c,br,bc,False)
        return self.run(c,br,bc,True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('out',type=Path)
    parser.add_argument('--after',type=Path);parser.add_argument('--trials',type=int,default=128)
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    assert args.trials>=1
    out=args.out.resolve();out.mkdir(exist_ok=args.resume)
    guard=(out/'study.lock').open('w');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def status(stage,**extra):
        p=out/'status.tmp';p.write_text(json.dumps(dict(stage=stage,pid=os.getpid(),updated=time.time(),**extra),indent=2));p.replace(out/'status.json')
    # Freeze hashes immediately, even while queued. Never mutate predecessor.
    watched=[Path(__file__),HERE/'build_correct_search.py',HERE/'queue_correct_best.py',
      HERE/'make_search_kernels.py',HERE/'spm_multicore.cpp',HERE/'cache_qk_wrapper.cpp',
      HERE/'correct_fa_conversions.hpp',HERE/'driver_vector_pack.hpp',HERE/'fa_independent_reference.py']
    watched += [HERE/'run_causal.py',HERE/'validate_attribution.py',HERE/'analyze_traffic.py']
    watched+=list((ROOT.parent/'cacheflex_micro/experiments/end2end/src').glob('*.hpp'))
    watched += [ROOT.parent/'cacheflex_micro/experiments/end2end/src/cache_flash_attn.cpp',
      ROOT.parent/'cacheflex_micro/kernels/attention_kernels_spm/11_flash_attn_kv_spm_vl4.cpp']
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    manifest=out/'hashes.json'
    if not args.resume:
        shutil.copy2(OLD/'gem5.opt',out/'gem5.opt')
        watched.append(out/'gem5.opt')
        manifest.write_text(json.dumps({str(p):digest(p) for p in watched},indent=2))
        with (out/'environment.freeze.txt').open('w') as log:
            subprocess.run([sys.executable,'-m','pip','freeze'],stdout=log,check=True)
    hashes=json.loads(manifest.read_text())
    def unchanged():
        for p,h in hashes.items():assert digest(Path(p))==h,('dependency changed',p)
    unchanged()
    plan=dict(optuna_version=optuna.__version__,budget_per_arch=args.trials,seed=137,
      sampler=dict(multivariate=True,group=True,constant_liar=True,n_startup_trials=20,n_ei_candidates=64),
      objective='T1024 nonperiodic full FA ROI us; independent Bc-specific reference every timed candidate',
      workload='8 cores, VL16, d64, Hq8/Hkv1, noncausal; fixed default hardware; scalar SPMCP',
      validation='Periodic gate then full-size rich validation. Top 3 each: 3 total rich measurements.',
      concurrency='Two candidate evaluations per architecture, four total; wait for predecessor lock release',
      search='Br16/32/64/128; Bc128/256/512; groups1/2/4/8/16/32/64; K/V/KV single/double or pinned K; offsets0/128/256/512; native/wide8/16/24 QK; pipeline; unroll1/2/4; load order; rescale fusion/phase interleaving',
      limits='No narrow-way kernels, direct P-pack fusion, wider PV or alternate multicore mapping; one rich input distribution. No global-optimum or convergence guarantee.',
      history='Fresh studies: no mixed periodic/small-T history imported. Preserve staged sweep as independent search.',
      after=str(args.after) if args.after else None)
    (out/'plan.json').write_text(json.dumps(plan,indent=2))
    studies={}
    for arch in ('cache','spm'):
        sampler_path=out/(arch+'.sampler.pkl')
        sampler=pickle.loads(sampler_path.read_bytes()) if args.resume and sampler_path.exists() else optuna.samplers.TPESampler(
            seed=137,multivariate=True,group=True,constant_liar=True,n_startup_trials=20,n_ei_candidates=64)
        study=optuna.create_study(study_name=arch,storage='sqlite:///'+str(out/'studies.sqlite3'),
            direction='minimize',sampler=sampler,load_if_exists=args.resume)
        # A lone coordinator owns this database; stale RUNNING trials are
        # retried from accepted run artifacts, never treated as measurements.
        for trial in study.get_trials(states=(optuna.trial.TrialState.RUNNING,)):
            study.tell(trial.number,state=optuna.trial.TrialState.FAIL)
            study.enqueue_trial(trial.params)
        if not args.resume:
            seed=dict(Br=64,Bc=128,group=4,qk_kernel='native',unroll=1,phase_mode='plain')
            if arch=='spm':seed.update(layout='k_double',offset=0,load_order=0)
            study.enqueue_trial(seed)
        studies[arch]=study
    try:
        if args.after:
            while True:
                predecessor=args.after/'queue.lock'
                with predecessor.open('r+') as lock:
                    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:
                        status('waiting_for_staged_sweep',after=str(args.after));time.sleep(30);continue
                assert (args.after/'summary.json').exists(),'Predecessor stopped without success; inspect its failure first'
                break
        unchanged();evaluator=Evaluator(out)
        for bc in (128,256,512):
            status('preparing_independent_reference',Bc=bc);evaluator.getref(bc)
        seen={a:set() for a in studies};complete={a:0 for a in studies}
        for a,s in studies.items():
            for t in s.get_trials(states=(optuna.trial.TrialState.COMPLETE,)):
                seen[a].add(t.user_attrs['candidate_key']);complete[a]+=1
        pending={}
        with cf.ThreadPoolExecutor(max_workers=4) as pool:
            while any(complete[a]<args.trials for a in studies) or pending:
                for arch,study in studies.items():
                    attempts=0
                    while complete[arch]+sum(a==arch for a,_,_ in pending.values())<args.trials and sum(a==arch for a,_,_ in pending.values())<2:
                        attempts+=1;assert attempts<2000,'Sampler exhausted legal unique suggestions'
                        trial=study.ask();c,br,bc=suggest(trial,arch);ident=key(dict(c=c,br=br,bc=bc))
                        if not eligible(c,1024,br,bc) or ident in seen[arch]:
                            trial.set_user_attr('rejection','illegal layout' if not eligible(c,1024,br,bc) else 'duplicate canonical config')
                            study.tell(trial,state=optuna.trial.TrialState.PRUNED);continue
                        unchanged();seen[arch].add(ident)
                        trial.set_user_attr('candidate_key',ident);trial.set_user_attr('config',c)
                        trial.set_user_attr('Br',br);trial.set_user_attr('Bc',bc)
                        future=pool.submit(evaluator.evaluate,c,br,bc);pending[future]=(arch,study,trial)
                    (out/(arch+'.sampler.pkl')).write_bytes(pickle.dumps(study.sampler))
                status('optimizing',completed=complete,running=len(pending),budget_per_arch=args.trials)
                done,_=cf.wait(pending,return_when=cf.FIRST_COMPLETED)
                for future in done:
                    arch,study,trial=pending.pop(future)
                    try:row=future.result()
                    except BaseException:
                        study.tell(trial,state=optuna.trial.TrialState.FAIL);raise
                    trial.set_user_attr('result_path',row['path']);study.tell(trial,row['roi_us']);complete[arch]+=1
                    print(arch,complete[arch],row['roi_us'],'PASS',flush=True)
                (out/'leaders.json').write_text(json.dumps({a:dict(value=s.best_value,params=s.best_params) for a,s in studies.items() if any(t.state==optuna.trial.TrialState.COMPLETE for t in s.trials)},indent=2))
        status('finalist_repeats');finals={}
        for arch,study in studies.items():
            top=sorted(study.get_trials(states=(optuna.trial.TrialState.COMPLETE,)),key=lambda t:t.value)[:3]
            finals[arch]=[]
            for trial in top:
                c=trial.user_attrs['config'];br=trial.user_attrs['Br'];bc=trial.user_attrs['Bc']
                results=[evaluator.run(c,br,bc,True,rep) for rep in range(3)]
                times=[r['roi_us'] for r in results]
                finals[arch].append(dict(trial=trial.number,config=c,br=br,bc=bc,times_us=times,
                    median_us=float(np.median(times)),spread_fraction=(max(times)-min(times))/float(np.median(times))))
        winners={a:min(v,key=lambda x:x['median_us']) for a,v in finals.items()}
        (out/'summary.json').write_text(json.dumps(dict(finalists=finals,winners=winners,
          speedup=winners['cache']['median_us']/winners['spm']['median_us'],
          claim='Best measured within fixed budget/search space; no global optimality claim'),indent=2))
        status('complete',completed=complete)
    except BaseException as e:status('failed',error=repr(e));raise

if __name__=='__main__':main()
