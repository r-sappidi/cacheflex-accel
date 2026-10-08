"""Fixed-winner token scaling; preserve guests, hardware, and numerical gates."""
from pathlib import Path
import concurrent.futures as cf
import argparse,hashlib,json,math,os,shutil,subprocess,sys
import numpy as np
from optuna_fa import Evaluator
from fa_independent_reference import reference,check
from fa_export_patch import verify_dump,patch
from validate_attribution import read
from run_causal import audit

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results/cacheflex_flash_attention/pv_improvements_20260913'
OUT=ROOT/'results/cacheflex_flash_attention/large_t_pv_20260914'

def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--tokens',type=int,nargs='+',default=[2048,4096,8192,16384])
    parser.add_argument('--group',type=int,default=16)
    options=parser.parse_args();OUT=options.out.resolve()
    assert all(t>0 and t%512==0 for t in options.tokens)
    assert options.group>0
    OUT.mkdir()
    shutil.copy2(BASE/'gem5.opt',OUT/'gem5.opt')
    ev=Evaluator(OUT)
    seeds={a:json.loads((BASE/'runs'/r/'accepted.json').read_text()) for a,r in
           [('cache','3c716544869382b2'),('spm','550744dbae7b1832')]}
    commands={a:json.loads((Path(r['path'])/'command.json').read_text()) for a,r in seeds.items()}
    for a,args in commands.items():
        source=Path(args[args.index('--cmd')+1]);target=OUT/(a+'.guest')
        if options.group!=16:
            config=dict(seeds[a]['config'],group=options.group,rich=1)
            config_file=OUT/'builds'/(a+'.json');config_file.write_text(json.dumps(config,indent=2))
            build=OUT/'builds'/a
            with (OUT/'builds'/(a+'.log')).open('w') as log:
                subprocess.run([sys.executable,str(Path(__file__).with_name('build_pv_improvements.py')),
                                str(build),str(config_file)],check=True,stdout=log,
                               stderr=subprocess.STDOUT,env=ev.env,timeout=600)
            source=patch(build/'guest')
        shutil.copy2(source,target);args[args.index('--cmd')+1]=str(target)
    files=[Path(__file__),OUT/'gem5.opt',OUT/'cache.guest',OUT/'spm.guest']
    (OUT/'hashes.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))
    (OUT/'plan.json').write_text(json.dumps(dict(tokens=options.tokens,
      Br=16,Bc=512,group=options.group,validation='Full independent rich reference, finite output, exact export hash, copy counts, config audit',
      limits='One timing per size and architecture; T1024 winners with optional explicit group override, not exhaustive per-size tuning; initialized input state unchanged'),indent=2))
    state=dict(stage='starting',completed=[],pid=os.getpid())
    def status(stage,**kw):
        state.update(stage=stage,**kw)
        tmp=OUT/'status.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(OUT/'status.json')
        print(stage,kw,flush=True)
    def run(a,t):
        p=OUT/'runs'/f'{a}_t{t}';p.mkdir()
        args=list(commands[a]);args[0]=str(OUT/'gem5.opt')
        args=[f'--outdir={p}' if x.startswith('--outdir=') else x for x in args]
        matches=[i for i,x in enumerate(args) if x.startswith('system.cpu[0].workload[0].cmd=')]
        assert len(matches)==1,matches
        args[matches[0]]=f"system.cpu[0].workload[0].cmd=['cacheflex-fa','{t}','64','16','512']"
        (p/'command.json').write_text(json.dumps(args,indent=2))
        with (p/'run.log').open('w') as log:
            subprocess.run(args,check=True,stdout=log,stderr=subprocess.STDOUT,env=ev.env,timeout=172800)
        output,counters,cfg=read(p,16);audit(ev.refcfg,cfg,0)
        for k,v in dict(T=t,Br=16,Bc=512,d=64,Hq=8,Hkv=1,cores=8).items():assert output[k]==str(v)
        assert output['nonfinite']=='0' and (p/'output.fp16').stat().st_size==8*t*64*2
        copies=0 if a=='cache' else 8*math.ceil(math.ceil(t/16)/options.group)*t*2
        assert counters['m_spmcpRequestsIssued']==counters['m_spmcpRequestsCompleted']==copies
        row=dict(path=str(p),output=output,counters=counters,roi_us=counters['roi_us'],t=t,arch=a)
        verify_dump(row)
        return row
    try:
        for t in options.tokens:
            status('simulating_and_reference',T=t)
            with cf.ThreadPoolExecutor(max_workers=3) as pool:
                fs={a:pool.submit(run,a,t) for a in seeds}
                rf=pool.submit(reference,t,True,64,512)
                rows={a:f.result() for a,f in fs.items()}
                status('numerical_validation',T=t)
                ref=rf.result();np.save(OUT/'references'/f't{t}_bc512.npy',ref)
            for row in rows.values():
                row['validation']=check(Path(row['path'])/'output.fp16',t,expected=ref)
                (Path(row['path'])/'accepted.json').write_text(json.dumps(row,indent=2))
            assert rows['cache']['output']['hash']==rows['spm']['output']['hash']
            state['completed'].append(dict(T=t,speedup=rows['cache']['roi_us']/rows['spm']['roi_us'],results=rows))
            status('pair_complete',T=t)
        (OUT/'summary.json').write_text(json.dumps(state['completed'],indent=2));status('complete')
    except BaseException as e:
        status('failed',error=repr(e));raise

if __name__=='__main__':main()
