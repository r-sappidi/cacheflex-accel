"""Same-executable counterfactuals after numerically corrected group search."""
from _paths import FA_DIR
from pathlib import Path
import concurrent.futures as cf
import json,os,subprocess,sys,time
from validate_attribution import read
from run_causal import audit,parameters
HERE=FA_DIR;ROOT=HERE.parents[1]
B=ROOT/'results/cacheflex_flash_attention';prior=B/'correct_fa_20260911'
out=Path(sys.argv[1]).resolve();out.mkdir();state={'stage':'waiting','completed':[]};rows=[]
def status(stage):
    state['stage']=stage;(out/'status.json').write_text(json.dumps(state,indent=2));print(stage,flush=True)
def run(case):
    row,mask=case;v=row['variant'];p=out/f'{v}_g{row["group"]}_h{mask}';p.mkdir()
    ref=Path(row['path']);args=json.loads((ref/'command.json').read_text())
    args=[f'--outdir={p}' if x.startswith('--outdir=') else x for x in args]
    for k,val in parameters(mask).items():args+=['-P',k+'='+val]
    (p/'command.json').write_text(json.dumps(args,indent=2))
    with (p/'run.log').open('w') as log:
        subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800,env=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib'))
    o,c,cfg=read(p,16);ro,rc,rf=read(ref,16);audit(rf,cfg,mask)
    assert o==ro and 'FA_ANALYTIC errors=0' in (p/'run.log').read_text()
    assert (p/'output.fp16').read_bytes()==(ref/'output.fp16').read_bytes()
    if not mask:
        assert all(c[k]==rc[k] for k in c.keys()-{'path'}),'Repeat is not exact'
    assert c['m_spmcpRequestsIssued']==c['m_spmcpRequestsCompleted']==rc['m_spmcpRequestsIssued']
    print(v,mask,c['roi_us'],flush=True)
    return dict(variant=v,group=row['group'],hardware_mask=mask,counters=c,output=o,path=str(p))
try:
    status('waiting')
    while not (prior/'summary.json').exists():
        if (prior/'status.json').exists():assert json.loads((prior/'status.json').read_text())['stage']!='failed'
        time.sleep(10)
    source=json.loads((prior/'summary.json').read_text())
    winners={v:min((r for r in source if r['variant']==v and r['t']==1024 and not r['rich']),key=lambda r:r['counters']['roi_us']) for v in ['cache','spm']}
    (out/'winners.json').write_text(json.dumps(winners,indent=2))
    status('spm_service_capacity')
    with cf.ThreadPoolExecutor(max_workers=4) as pool:rows.extend(pool.map(run,[(winners['spm'],h) for h in [0,1,2,8]]))
    (out/'progress.json').write_text(json.dumps(rows,indent=2))
    status('cache_repeat');rows.append(run((winners['cache'],0)))
    (out/'summary.json').write_text(json.dumps(rows,indent=2));status('complete')
except BaseException as e:state['error']=repr(e);status('failed');raise
