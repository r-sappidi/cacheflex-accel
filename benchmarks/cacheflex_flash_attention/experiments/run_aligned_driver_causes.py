"""Follow-up: control GEMM loop alignment, then isolate packing and grouping."""
import concurrent.futures as cf
import json,math,os,subprocess,sys,time
from _paths import FA_DIR
from pathlib import Path
from validate_attribution import read
from run_causal import audit
HERE=FA_DIR;ROOT=HERE.parents[1];B=ROOT/'results/cacheflex_flash_attention'
old=B/'residual_retry_20260911';out=Path(sys.argv[1]).resolve();out.mkdir(exist_ok='--resume-wait' in sys.argv)
state=dict(stage='waiting_for_preceding_studies',completed=[],failed=None);rows=[]
def status(s):state['stage']=s;(out/'status.json').write_text(json.dumps(state,indent=2));print(s,flush=True)
refs={t:read(old/f'original_cache_t{t}_p0',16) for t in [128,1024]}
def run(case):
    v,g,m,t=case;build=out/f'build_{v}_g{g}_m{m}'
    if not build.exists():
        with (out/(build.name+'.log')).open('w') as log:subprocess.run(['python3',str(HERE/'build_driver_aligned.py'),str(build),v,str(g),str(m)],stdout=log,stderr=subprocess.STDOUT,check=True)
    p=out/f'{v}_g{g}_m{m}_t{t}';p.mkdir()
    args=json.loads((old/f'original_{"spm" if v=="spm" else "cache"}_t{t}_p0/command.json').read_text())
    args=[f'--outdir={p}' if a.startswith('--outdir=') else a for a in args];args[args.index('--cmd')+1]=str(build/'guest')
    args+=['-P',f"system.cpu[0].workload[0].cmd=['cacheflex-fa','{t}','64','64','128']"]
    (p/'command.json').write_text(json.dumps(args,indent=2))
    with (p/'run.log').open('w') as log:subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1200,
        env=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib'))
    o,c,cfg=read(p,16);assert o['hash']==refs[t][0]['hash'];audit(refs[t][2],cfg,0)
    copies=8*(t//128)*64*4*math.ceil(math.ceil(t/64)/g) if v=='spm' else 0
    assert c['m_spmcpRequestsIssued']==c['m_spmcpRequestsCompleted']==copies
    return dict(variant=v,group=g,mask=m,t=t,counters=c,output=o,path=str(p))
def batch(cases,s):
    status(s)
    with cf.ThreadPoolExecutor(max_workers=4) as pool:rows.extend(pool.map(run,cases))
    state['completed'].append(s);(out/'progress.json').write_text(json.dumps(rows,indent=2))
try:
    status('waiting_for_preceding_studies')
    while not ((B/'driver_causes_20260911/STOPPED_FOR_ALIGNMENT.json').exists() and all((B/n/'summary.json').exists() for n in ['qk_alignment_20260911','pv_alignment_20260911'])):time.sleep(10)
    cases=[('original',1,0)]+[(v,g,m) for v,g in [('cache',1),('spm',4)] for m in [0,1,2,3,7]]
    batch([(*c,128) for c in cases],'packing_smokes')
    batch([(*c,1024) for c in cases],'packing_t1024')
    winners={v:min((r for r in rows if r['variant']==v and r['t']==1024),key=lambda r:r['counters']['roi_us'])['mask'] for v in ['cache','spm']}
    (out/'packing_winners.json').write_text(json.dumps(winners,indent=2))
    # Same packing alternatives on both architectures; retain all old points.
    cases=[(v,g,winners[v]) for v in ['cache','spm'] for g in [1,2,4,8] if g!=(1 if v=='cache' else 4)]
    batch([(*c,128) for c in cases],'group_smokes')
    batch([(*c,1024) for c in cases],'group_t1024')
    (out/'summary.json').write_text(json.dumps(rows,indent=2));status('complete')
except BaseException as e:state['failed']=repr(e);status('failed');raise
