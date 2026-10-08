"""Gated LLC-retention experiment using unchanged full-width FA guests."""
from pathlib import Path
import concurrent.futures as cf
import copy,json,math,os,subprocess,time
import numpy as np
from optuna_fa import Evaluator
from validate_attribution import read
from run_causal import audit
from fa_independent_reference import check
from fa_export_patch import verify_dump

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/cacheflex_flash_attention/llc_retain_20260915'
LONG=ROOT/'results/cacheflex_flash_attention/long_context_pv_20260914'
PV=ROOT/'results/cacheflex_flash_attention/pv_improvements_20260913'

def main():
    ev=Evaluator(OUT);state=dict(stage='waiting_for_build',pid=os.getpid(),completed=[])
    def status(stage,**extra):
        state.update(stage=stage,**extra);tmp=OUT/'status.tmp'
        tmp.write_text(json.dumps(state,indent=2));tmp.replace(OUT/'status.json');print(stage,extra,flush=True)
    def run(arch,t,retain=False,small=False,old=False):
        name=f'{arch}_t{t}_retain{int(retain)}_small{int(small)}_old{int(old)}'
        p=OUT/'runs'/name;p.mkdir()
        args=json.loads((LONG/'runs'/f'{arch}_t16384'/'command.json').read_text())
        args[0]=str(LONG/'gem5.opt' if old else OUT/'build/gem5.opt')
        args=[f'--outdir={p}' if x.startswith('--outdir=') else x for x in args]
        for i,x in enumerate(args):
            if x.startswith('system.cpu[0].workload[0].cmd='):
                args[i]=f"system.cpu[0].workload[0].cmd=['cacheflex-fa','{t}','64','16','512']"
            if small and x.startswith('--l2_size='):args[i]='--l2_size=64kB'
        if not old:
            for i in range(8):args+=['-P',f'system.ruby.l2_cntrl{i}.spm_retain_llc={retain}']
        (p/'command.json').write_text(json.dumps(args,indent=2))
        with (p/'run.log').open('w') as log:
            subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True,env=ev.env,timeout=172800)
        output,counters,cfg=read(p,16)
        reference_cfg=copy.deepcopy(ev.refcfg)
        for i in range(8):
            section=f'system.ruby.l2_cntrl{i}'
            if not old:
                assert cfg[section]['spm_retain_llc']==str(retain).lower()
                del cfg[section]['spm_retain_llc']
            if small:reference_cfg[section+'.L2cache']['size']=str(64*1024)
        audit(reference_cfg,cfg,0)
        for k,v in dict(T=t,d=64,Hq=8,Hkv=1,cores=8,Br=16,Bc=512).items():assert output[k]==str(v)
        expected=np.load((LONG/'references'/f't{t}_bc512.npy') if t==16384 else
                         PV/'references/t1024_bc512.npy')
        validation=check(p/'output.fp16',t,expected=expected)
        copies=8*math.ceil(t/256)*t*2 if arch=='spm' else 0
        assert counters['m_spmcpRequestsIssued']==counters['m_spmcpRequestsCompleted']==copies
        row=dict(path=str(p),output=output,counters=counters,validation=validation)
        verify_dump(row)
        stats=(p/'stats.txt').read_text().split('End Simulation Statistics')[0]
        retained=0
        for line in stats.splitlines():
            f=line.split()
            if f and f[0]=='system.ruby.L2Cache_Controller.SPM_IS.Mem_Data_SPM_Retain::total':retained=int(f[1])
        row['retained_fills']=retained
        if small and retain:assert retained>0,'Test failed to exercise new path'
        if not retain:assert retained==0
        (p/'accepted.json').write_text(json.dumps(row,indent=2));return row
    def batch(jobs,stage):
        status(stage)
        with cf.ThreadPoolExecutor(max_workers=2) as pool:
            fs={pool.submit(run,*j):j for j in jobs};rows={}
            for f in cf.as_completed(fs):
                row=f.result();rows[fs[f]]=row;state['completed'].append(row);status(stage)
        return rows
    try:
        status('waiting_for_build');deadline=time.monotonic()+4*3600
        while not (OUT/'build/simulator.sha256').exists():
            log=OUT/'build/build.log'
            if log.exists():
                txt=log.read_text()[-12000:]
                assert 'building terminated because of errors' not in txt,txt[-3000:]
            assert time.monotonic()<deadline,'Build readiness timeout'
            time.sleep(10)
        controls=batch([(a,1024,False,False,old) for a in ('cache','spm') for old in (True,False)],'rebuild_controls')
        for a in ('cache','spm'):
            old,new=(controls[(a,1024,False,False,b)] for b in (True,False))
            assert old['output']==new['output']
            assert {k:v for k,v in old['counters'].items() if k!='path'}=={k:v for k,v in new['counters'].items() if k!='path'},'Rebuild changed control counters'
        batch([('spm',1024,b,True,False) for b in (False,True)],'forced_miss_correctness')
        batch([('spm',1024,True,False,False)],'stock_t1024_retention')
        batch([('cache',16384,False,False,False),('spm',16384,False,False,False),
               ('spm',16384,True,False,False)],'matched_t16384')
        status('complete')
    except BaseException as e:status('failed',error=repr(e));raise

if __name__=='__main__':main()
