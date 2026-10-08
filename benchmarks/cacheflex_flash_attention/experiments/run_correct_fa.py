"""Numerically gated, canonical-argv corrected FA diagnostic comparisons."""
from _paths import FA_DIR
from pathlib import Path
import concurrent.futures as cf
import json,os,subprocess,sys
from validate_attribution import read
from run_causal import audit,parameters
from fa_independent_reference import check
HERE=FA_DIR;ROOT=HERE.parents[1]
BASE=ROOT/'results/cacheflex_flash_attention';OLD=BASE/'residual_retry_20260911'
out=Path(sys.argv[1]).resolve();out.mkdir()
rows=[];state={'stage':'initializing','completed':[]}
def status(stage):
    state['stage']=stage;(out/'status.json').write_text(json.dumps(state,indent=2));print(stage,flush=True)
def run(case):
    v,g,m,t,rich,hw=case
    name=f'{v}_g{g}_m{m}_r{int(rich)}';build=out/('build_'+name)
    if not build.exists():
        builder='build_driver_pinned.py' if m&64 else 'build_driver_aligned.py'
        with (out/(name+'.build.log')).open('w') as log:
            subprocess.run(['python3',str(HERE/builder),str(build),v,str(g),str(m),'--correct']+(['--rich'] if rich else []),stdout=log,stderr=subprocess.STDOUT,check=True)
    p=out/f'{name}_t{t}_h{hw}';p.mkdir()
    template=OLD/f'original_{"spm" if v=="spm" else "cache"}_t128_p0'
    args=json.loads((template/'command.json').read_text())
    args=[f'--outdir={p}' if x.startswith('--outdir=') else x for x in args]
    args[args.index('--cmd')+1]=str(build/'guest')
    args+=['-P',f"system.cpu[0].workload[0].cmd=['cacheflex-fa','{t}','64','64','128']"]
    for k,value in parameters(hw).items():args+=['-P',k+'='+value]
    (p/'command.json').write_text(json.dumps(args,indent=2))
    with (p/'run.log').open('w') as log:subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800,env=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib'))
    output,counters,cfg=read(p,16);audit(read(OLD/'original_cache_t128_p0',16)[2],cfg,hw)
    assert output['nonfinite']=='0'
    assert (p/'output.fp16').stat().st_size==8*t*64*2
    if rich:
        validation=check(p/'output.fp16',t)
    else:
        assert 'FA_ANALYTIC errors=0' in (p/'run.log').read_text()
        validation={'analytic':'All identical V rows imply output equals V exactly','errors':0}
    (p/'numerical_audit.json').write_text(json.dumps(validation,indent=2))
    copies=0 if v!='spm' else 8*(t//128)*64*4*(1 if m&64 else (t//64+g-1)//g)
    assert counters['m_spmcpRequestsIssued']==counters['m_spmcpRequestsCompleted']==copies
    result=dict(variant=v,group=g,mask=m,t=t,rich=rich,hardware_mask=hw,output=output,counters=counters,path=str(p),validation=validation)
    print(name,t,hw,counters['roi_us'],flush=True);return result
def batch(cases,stage):
    status(stage)
    with cf.ThreadPoolExecutor(max_workers=4) as pool:rows.extend(pool.map(run,cases))
    state['completed'].append(stage);(out/'progress.json').write_text(json.dumps(rows,indent=2))
try:
    batch([(v,4,3,128,False,0) for v in ['original','cache','spm']],'periodic_smokes')
    batch([(v,4,3,256,True,0) for v in ['cache','spm']],'independent_rich_validation')
    batch([(v,g,3,1024,False,0) for g in [4,1,2,8] for v in ['cache','spm']]+[('original',4,3,1024,False,0)],'corrected_group_search')
    (out/'summary.json').write_text(json.dumps(rows,indent=2));status('complete')
except BaseException as e:
    state['error']=repr(e);status('failed');raise
