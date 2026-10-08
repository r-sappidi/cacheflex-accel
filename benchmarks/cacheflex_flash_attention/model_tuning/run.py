"""Correctness-gated, model-guided cache/SPM search; max two new simulations."""
import concurrent.futures as cf
import json,math,os,re,shutil,subprocess,sys,threading,time
from pathlib import Path
import numpy as np
from model import ROOT,B,HERE,FAMILIES,describe,candidates,observations,holdout,select,label,predict
sys.path[:0]=[str(ROOT/'scripts'),str(HERE.parent),str(HERE.parent/'kv_partition')]
from build_access_timing import sha,save
from run_access_timing import stats
from fa_headroom_profiles import leaves
from test_micro_timing import check_global_guard
from export_safe_elf import export_hash
from reference import reference,validate

BASE=B/'model_tuning_20260926';OUT=BASE/'experiment'
KV=B/'kv_partition_20260925/experiment_v3'
CONTROL=KV/'runs/target_t4096_n8_q_cache_br16_g4_l2512'
GUEST=BASE/'build/guest.export-safe'
ENV=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib',
         OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')

def command(c,path):
    kv=int(c['family'].startswith('kv_'));spm=int(c['family'].endswith('_spm'))
    args=list(map(str,(c['t'],8,c['br'],c['bc'],c['r']//c['br'],1024,kv,spm)))
    cmd=json.loads((CONTROL/'command.json').read_text())
    cmd[1]='--outdir='+str(path);cmd[2]=str(OUT/'configs/deprecated/example/micro_timing_se.py')
    cmd[cmd.index('--cmd')+1]=str(GUEST);cmd[cmd.index('--options')+1]=' '.join(args)
    for i,s in enumerate(cmd):
        if s.startswith('system.cpu[0].workload[0].cmd='):cmd[i]=f"system.cpu[0].workload[0].cmd={['kv-fa']+args!r}"
        if s.startswith('--l1d_num_spm_ways='):cmd[i]='--l1d_num_spm_ways='+str(6 if spm else 0)
    return cmd,args

def audit(path,args):
    a=json.loads((CONTROL/'config.json').read_text().replace(str(CONTROL),'<OUT>'))
    b=json.loads((path/'config.json').read_text().replace(str(path),'<OUT>'))
    old,new=dict(leaves(a)),dict(leaves(b))
    delta={k:[old.get(k),new.get(k)] for k in old.keys()|new.keys() if old.get(k)!=new.get(k)}
    allowed=r'system.cpu\[\d+\].workload\[0\].(cmd\[\d+\]|executable)|system.ruby.l1_cntrl\d+.cache.num_spm_ways'
    assert all(re.fullmatch(allowed,k) for k in delta),delta
    proc=b['system']['cpu'][0]['workload'][0]
    assert proc['executable']==str(GUEST) and proc['cmd'][1:]==args
    for i in range(8):
        assert b['system']['ruby'][f'l1_cntrl{i}']['cache']['num_spm_ways']==(6 if args[-1]=='1' else 0)
    return dict(all_fields_matched_except_guest_and_spm_eligibility=True,delta=delta)

def main():
    OUT.mkdir();(OUT/'runs').mkdir();(OUT/'references').mkdir()
    shutil.copytree(KV/'configs',OUT/'configs')
    cmd0=json.loads((CONTROL/'command.json').read_text())
    paths=[*HERE.glob('*.py'),HERE.parent/'kv_partition/reference.py',HERE.parent/'fa_independent_reference.py',
        HERE.parent/'export_safe_elf.py',Path(cmd0[0]),CONTROL/'config.json',CONTROL/'command.json',
        *list((OUT/'configs').rglob('*.py'))]
    frozen={str(p):sha(p) for p in paths};frozen.update(json.loads((BASE/'build/frozen.json').read_text()))
    save(OUT/'inputs.json',frozen)
    rows=observations();save(OUT/'historical_observations.json',rows)
    save(OUT/'historical_holdout.json',holdout(rows))
    for t in (4096,16384):
        good,bad=candidates(t);save(OUT/f'candidate_space_t{t}.json',dict(legal=good,rejected=bad))
    state=dict(pid=os.getpid(),stage='gates',active={},completed=[],observations=rows)
    lock=threading.RLock();halt=threading.Event();refs={}
    def status(stage=None):
        with lock:
            if stage:state['stage']=stage
            state['updated']=time.time();save(OUT/'status.json',state)
    def unchanged():
        for p,h in frozen.items():assert sha(p)==h,('frozen input changed',p)
    def getref(c):
        key=(c['t'],c['bc'],8 if c['family'].startswith('kv_') else 1)
        if key in refs:return refs[key]
        status('reference_t%d_bc%d_parts%d'%key)
        target=OUT/'references'/('t%d_bc%d_parts%d.npy'%key)
        choices=[KV/'references'/target.name]
        if key[2]==1:
            choices += [B/'cache_tuning_20260925'/exp/'references'/f't{key[0]}_bc{key[1]}.npy'
                        for exp in ('experiment_export_safe_20260926','experiment')]
        source=None
        for old in choices:
            meta=old.with_suffix('.json')
            if old.exists() and meta.exists():
                info=json.loads(meta.read_text())
                if info.get('sha256')==sha(old):source=old;break
        if source:
            shutil.copy2(source,target);a=np.load(target)
        else:
            a=reference(*key);np.save(target,a)
        assert a.shape==(key[0],64)
        save(target.with_suffix('.json'),dict(key=key,sha256=sha(target),source=str(source) if source else 'independent rounded reference'))
        refs[key]=a;return a
    def run(c):
        if halt.is_set():raise RuntimeError('cancelled before launch')
        name=label(c);path=OUT/'runs'/name;path.mkdir();proc=None;start=time.monotonic()
        try:
            unchanged();cmd,args=command(c,path);save(path/'command.json',cmd)
            with (path/'run.log').open('w') as log:
                proc=subprocess.Popen(cmd,cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT)
                with lock:state['active'][name]=dict(pid=proc.pid,case=c);status()
                while proc.poll() is None:
                    assert not halt.is_set(),'peer failure'
                    assert time.monotonic()-start<172800,'timeout'
                    assert shutil.disk_usage(OUT).free>50*2**30,'disk guard'
                    save(path/'progress.json',dict(pid=proc.pid,host_seconds=time.monotonic()-start))
                    try:proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:pass
            log=(path/'run.log').read_text()
            assert proc.returncode==0 and 'KV_ROI_END' in log and 'Simulated exit code not 0!' not in log,log[-2000:]
            cfg=audit(path,args);blocks=stats(path)
            for s in blocks:check_global_guard(s)
            s=blocks[0];export_hash(path/'output.fp16',log)
            mathcheck=validate(path/'output.fp16',refs[c['t'],c['bc'],8 if args[-2]=='1' else 1],c['t'],dense=c['t']==1024)
            # Gate cases can deliberately use Bc>local for PV-tail coverage;
            # search rejects those redundant choices, not the correct kernel.
            spm=args[-1]=='1';kv=args[-2]=='1'
            expected=(2*c['t'] if kv else 16*c['t']*math.ceil((c['t']//8)/c['r'])) if spm else 0
            for k in ('m_spmcpRequestsIssued','m_spmcpRequestsCompleted'):
                actual=sum(s[f'system.ruby.l0_cntrl{i}.sequencer.{k}'] for i in range(8))
                assert actual==expected,(k,actual,expected)
            released=s.get('system.ruby.L1Cache_Controller.X.SPM_release::total',0)
            # Query release covers configured Bc; gates ensure Bc<=T.
            expected_release=(2*c['t'] if kv else 16*c['bc']) if spm else 0
            assert released==expected_release,(released,expected_release)
            unchanged();row=dict(case=c,path=str(path),roi_us=s['simTicks']/1e6,
                copies=expected,arithmetic=mathcheck,config=cfg,output_sha256=sha(path/'output.fp16'),
                model=describe(c),host_seconds=time.monotonic()-start,packer='qpack')
            save(path/'validated.json',row)
            with lock:state['completed'].append(row)
            return row
        except BaseException as e:
            halt.set();save(path/'failed.json',dict(error=repr(e)))
            with lock:state.setdefault('error',f'{name}: {e!r}')
            raise
        finally:
            if proc and proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
            with lock:state['active'].pop(name,None);status()
    def batch(cs,stage):
        for c in cs:getref(c)
        status(stage)
        with cf.ThreadPoolExecutor(max_workers=2) as pool:return list(pool.map(run,cs))
    def record(new):
        for x in new:rows.append(dict(case=x['case'],roi_us=x['roi_us'],source=x['path'],packer='qpack'))
        state['observations']=rows;status()
    def planned(stage,t,neighbors=None):
        seen={label(x['case']) for x in state['completed']}
        plan=select(t,rows,seen,neighbors=neighbors);save(OUT/(stage+'_plan.json'),plan)
        out=batch([r['case'] for r in plan['selected']],stage)
        predictions={label(r['case']):r for r in plan['selected']}
        report=[dict(case=x['case'],predicted_us=predictions[label(x['case'])]['predicted_us'],
                     measured_us=x['roi_us'],relative_error=abs(x['roi_us']-predictions[label(x['case'])]['predicted_us'])/x['roi_us']) for x in out]
        save(OUT/(stage+'_prediction_check.json'),report);record(out)
        return out
    try:
        status()
        gates=[dict(t=1024,family=f,br=32,bc=128,r=192) for f in FAMILIES]
        gates += [dict(t=1024,family=f,br=64,bc=1024 if f.startswith('q_') else 128,r=256) for f in FAMILIES]
        out=batch(gates,'eight_numerical_gates')
        for i in (0,1,4,5):assert out[i]['output_sha256']==out[i+2]['output_sha256'],'cache/SPM arithmetic differs'
        save(OUT/'gates_passed.json',dict(cases=8,matched_output_bits=True,full_reference=True,dense_float64=True))
        planned('model_screen_t4096',4096)
        target=planned('direct_target_t16384',16384)
        winners={f:min((x for x in target if x['case']['family']==f),key=lambda x:x['roi_us'])['case'] for f in FAMILIES}
        planned('local_refinement_t16384',16384,neighbors=winners)
        results=[x for x in state['completed'] if x['case']['t']==16384]
        best={f:min((x for x in results if x['case']['family']==f),key=lambda x:x['roi_us']) for f in FAMILIES}
        cache=min((best[f] for f in FAMILIES if f.endswith('_cache')),key=lambda x:x['roi_us'])
        spm=min((best[f] for f in FAMILIES if f.endswith('_spm')),key=lambda x:x['roi_us'])
        save(OUT/'summary.json',dict(family_winners=best,cache=cache,spm=spm,speedup=cache['roi_us']/spm['roi_us'],
            caveat='Best measured in this bounded search; compare concurrent cache/llama suites before headline claim; edge winners require range expansion'))
        status('completed_bounded_search_requires_review')
    except BaseException as e:
        state.setdefault('error',repr(e));status('failed');raise

if __name__=='__main__':main()
