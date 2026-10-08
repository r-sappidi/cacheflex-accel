"""Conditional residual study after alignment/packing/group controls.

Compare rolling staging against scalar per-core pinned K, then bounded existing
copy/load service and separate-backing diagnostics on the rolling winner.
"""
from _paths import FA_DIR
from pathlib import Path
import concurrent.futures as cf
import json,math,os,subprocess,sys,time
from validate_attribution import read
from run_causal import audit,parameters
HERE=FA_DIR;ROOT=HERE.parents[1];B=ROOT/'results/cacheflex_flash_attention'
old=B/'residual_retry_20260911';prior=B/'aligned_driver_causes_20260911'
out=Path(sys.argv[1]).resolve();out.mkdir(exist_ok='--resume-wait' in sys.argv)
state=dict(stage='waiting_for_aligned_study',completed=[],failed=None);rows=[]
def status(s):state['stage']=s;(out/'status.json').write_text(json.dumps(state,indent=2));print(s,flush=True)
refs={t:read(old/f'original_cache_t{t}_p0',16) for t in [128,1024]}
def run(case):
    g,m,t,hw=case;build=out/f'build_spm_g{g}_m{m}'
    if not m&64:build=prior/f'build_spm_g{g}_m{m}'
    if not build.exists():
        with (out/(build.name+'.log')).open('w') as log:subprocess.run(['python3',str(HERE/'build_driver_pinned.py'),str(build),'spm',str(g),str(m)],stdout=log,stderr=subprocess.STDOUT,check=True)
    p=out/f'spm_g{g}_m{m}_t{t}_h{hw}';p.mkdir()
    args=json.loads((old/f'original_spm_t{t}_p0/command.json').read_text())
    args=[f'--outdir={p}' if x.startswith('--outdir=') else x for x in args];args[args.index('--cmd')+1]=str(build/'guest')
    args+=['-P',f"system.cpu[0].workload[0].cmd=['cacheflex-fa','{t}','64','64','128']"]
    for k,v in parameters(hw).items():args+=['-P',k+'='+v]
    (p/'command.json').write_text(json.dumps(args,indent=2))
    with (p/'run.log').open('w') as log:subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1200,
        env=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib'))
    o,c,cfg=read(p,16);assert o['hash']==refs[t][0]['hash'];audit(refs[t][2],cfg,hw)
    copies=8*(t//128)*64*4*(1 if m&64 else math.ceil(math.ceil(t/64)/g))
    assert c['m_spmcpRequestsIssued']==c['m_spmcpRequestsCompleted']==copies
    return dict(group=g,mask=m,t=t,hardware_mask=hw,counters=c,output=o,path=str(p))
def batch(cases,s):
    status(s)
    with cf.ThreadPoolExecutor(max_workers=4) as pool:rows.extend(pool.map(run,cases))
    state['completed'].append(s);(out/'progress.json').write_text(json.dumps(rows,indent=2))
try:
    status('waiting_for_aligned_study')
    while not ((prior/'summary.json').exists() and (B/'pinned_rich_20260911/summary.json').exists()):time.sleep(10)
    candidates=json.loads((prior/'summary.json').read_text())
    best={v:min((r for r in candidates if r['t']==1024 and ((r['variant']=='spm')==(v=='spm'))),key=lambda r:r['counters']['roi_us']) for v in ['cache','spm']}
    (out/'starting_winners.json').write_text(json.dumps(best,indent=2))
    if best['spm']['counters']['roi_us']<=best['cache']['counters']['roi_us']:
        (out/'summary.json').write_text(json.dumps(dict(reason='Aligned software closes the SPM slowdown; no residual hardware probes required.',winners=best),indent=2));status('complete')
    else:
        m=best['spm']['mask'];g=best['spm']['group']
        batch([(pg,m|64,128,0) for pg in [1,2,4]],'pinned_smokes')
        batch([(pg,m|64,1024,0) for pg in [1,2,4]],'pinned_t1024')
        batch([(g,m,1024,h) for h in [0,1,2,8]],'rolling_service_capacity')
        pin=min((r for r in rows if r['t']==1024 and r['mask']&64),key=lambda r:r['counters']['roi_us'])
        batch([(pin['group'],m|64,1024,h) for h in [1,2,8]],'pinned_service_capacity')
        (out/'summary.json').write_text(json.dumps(dict(winners=best,rows=rows),indent=2));status('complete')
except BaseException as e:state['failed']=repr(e);status('failed');raise
