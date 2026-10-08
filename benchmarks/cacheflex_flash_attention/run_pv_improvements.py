"""Numerically gated fixed-hardware factorial of two software improvements."""
from pathlib import Path
import concurrent.futures as cf,json,os,shutil,subprocess,sys,hashlib
from optuna_fa import Evaluator,DEFAULT,OLD,key
from fa_export_patch import patch,verify_dump
from fa_independent_reference import reference,check
import numpy as np

HERE=Path(__file__).resolve().parent

def main():
    out=OLD.parent/'pv_improvements_20260913';out.mkdir()
    shutil.copy2(OLD.parent/'optuna_20260912/gem5.opt',out/'gem5.opt')
    evaluator=Evaluator(out);state=dict(stage='initializing',completed=[])
    def status(stage):
        state['stage']=stage;(out/'status.json').write_text(json.dumps(state,indent=2));print(stage,flush=True)
    files=[HERE/n for n in ['run_pv_improvements.py','build_pv_improvements.py','make_pv_fullwidth.py',
      'build_correct_search.py','spm_multicore.cpp','driver_vector_pack.hpp','correct_fa_conversions.hpp','fa_export_patch.py','fa_export_hook.S']]
    (out/'source_hashes.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))
    def guest(c,rich):
        config=dict(c,rich=int(rich));ident=key(config);p=out/'builds'/ident
        with evaluator.lock:guard=evaluator.build_locks.setdefault(ident,__import__('threading').Lock())
        with guard:
            if not (p/'variant_complete.json').exists():
                assert not p.exists(),('incomplete build',p)
                f=out/'builds'/(ident+'.json');f.write_text(json.dumps(config,indent=2))
                with (out/'builds'/(ident+'.log')).open('w') as log:
                    subprocess.run([sys.executable,str(HERE/'build_pv_improvements.py'),str(p),str(f)],
                      check=True,stdout=log,stderr=subprocess.STDOUT,env=evaluator.env,timeout=600)
                (p/'variant_complete.json').write_text(json.dumps(config))
            return patch(p/'guest')
    evaluator.guest=guest
    configs=[]
    for arch in ('cache','v'):
        for full in (0,1):
            for stride in ((0,1) if arch=='v' else (0,)):
                configs.append(dict(DEFAULT,placement=arch,group=16,overlap=0 if arch=='v' else 1,
                  rows=16,pipeline=1,offset=256 if arch=='v' else 0,stride_fix=stride,pv_fullwidth=full))
    (out/'plan.json').write_text(json.dumps(dict(configs=configs,T=1024,Br=16,Bc=512,
      smokes='T1024 periodic and nonperiodic for every variant; full-width additionally Br24 for 8-row tail',
      timing='Three rich repetitions per variant; unchanged default hardware; all ROI work included',
      limits='Full PV packs16 rows, doubles PV/pap scratch; these are software changes not isolated hardware interventions'),indent=2))
    results=[]
    def run(c,br,bc,rich,rep):return verify_dump(evaluator.run(c,br,bc,rich,rep))
    def batch(jobs,stage):
        status(stage)
        with cf.ThreadPoolExecutor(max_workers=4) as pool:
            fs=[pool.submit(run,*j) for j in jobs]
            for f in cf.as_completed(fs):
                r=f.result();results.append(r)
                state['completed'].append(dict(config=r['config'],br=r['br'],bc=r['bc'],rich=r['rich'],rep=r['rep'],roi_us=r['roi_us'],path=r['path']))
                (out/'progress.json').write_text(json.dumps(results,indent=2));status(stage)
    try:
        # Use small Bc128 smoke first to catch lane assembly errors quickly;
        # full T1024 rich evaluation still checks every output independently.
        for bc in (128,512):evaluator.getref(bc)
        batch([(c,16,128,False,0) for c in configs],'periodic_smokes')
        batch([(c,16,512,False,0) for c in configs],'target_periodic_gates')
        batch([(c,16,512,True,0) for c in configs],'target_rich_gates_and_timings')
        batch([(c,24,128,True,0) for c in configs if c['pv_fullwidth'] and not c['stride_fix']],'pv_tail_validation')
        batch([(c,16,512,True,rep) for c in configs for rep in (1,2)],'timing_repeats')
        summary=[]
        for c in configs:
            rs=[r for r in results if r['config']==c and r['br']==16 and r['bc']==512 and r['rich']]
            times=[r['roi_us'] for r in rs];assert len(times)==3
            summary.append(dict(config=c,times_us=times,median_us=float(np.median(times))))
        (out/'summary.json').write_text(json.dumps(summary,indent=2));status('complete')
    except BaseException as e:state['error']=repr(e);status('failed');raise

if __name__=='__main__':main()
