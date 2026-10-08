"""Controlled FA driver ablations; unchanged hardware and retained controls.

Usage: run_driver_causes.py NEW_OUT [--resume]
Bits: 1 vector P pack, 2 original Q pack, 4 vector K prepack,
8 remove redundant startup barrier, 16 restore Q copy, 32 direct cache QK.
"""
import concurrent.futures as cf
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from validate_attribution import read
from run_causal import audit
from analyze_boundaries import markers, analyze as boundaries

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OLD=ROOT/'results/cacheflex_flash_attention/residual_retry_20260911'
GROUP=ROOT/'results/cacheflex_flash_attention/query_groups_retry_20260911'
out=Path(sys.argv[1]).resolve();resume='--resume' in sys.argv
out.mkdir(exist_ok=resume)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
sources=[HERE/x for x in ('spm_multicore.cpp','driver_vector_pack.hpp',
    'build_matched_residual.sh','make_search_kernels.py','cache_qk_wrapper.cpp',
    'run_driver_causes.py','validate_attribution.py','run_causal.py','analyze_traffic.py')]
if not resume:
    shutil.copy2(OLD/'gem5.opt',out/'gem5.opt')
    (out/'sources').mkdir()
    for p in sources:shutil.copy2(p,out/'sources'/p.name)
    (out/'hashes.json').write_text(json.dumps({str(p):sha(p) for p in sources+[out/'gem5.opt']},indent=2))
hashes=json.loads((out/'hashes.json').read_text())
def unchanged():
    for p,h in hashes.items():assert sha(Path(p))==h,('source modified',p)
rows=[];state=dict(stage='starting',completed=[],failed=None)
def status(stage):
    state['stage']=stage;(out/'status.json').write_text(json.dumps(state,indent=2));print(stage,flush=True)
refs={t:read(OLD/f'original_cache_t{t}_p0',16) for t in [128,1024]}
def build(v,g,m):
    p=out/f'build_{v}_g{g}_m{m}'
    if p.exists():assert (p/'guest').exists();return p
    unchanged()
    with (out/(p.name+'.log')).open('w') as log:
        subprocess.run(['bash',str(HERE/'build_matched_residual.sh'),str(p),v,str(g),str(m)],stdout=log,stderr=subprocess.STDOUT,check=True)
    return p
def run(v,g,m,t,obs=False,original=False):
    unchanged()
    b=build(v,g,m) if not original else None
    guest=b/'guest' if b else OLD/'original_cache'
    name=f'{v}_g{g}_m{m}_t{t}_p{int(obs)}'+('_original' if original else '')
    p=out/name
    args=json.loads((OLD/f'original_{v}_t{t}_p0/command.json').read_text())
    args[0]=str(out/'gem5.opt');args=[f'--outdir={p}' if x.startswith('--outdir=') else x for x in args]
    args[args.index('--cmd')+1]=str(guest)
    if obs:
        args[1:1]=['--debug-flags=CacheFlexBoundary','--debug-file=boundaries.trace']
        pcs=repr(sorted(markers(b/'guest.debug')))
        for i in range(8):args+=['-P',f'system.cpu[{i}].cacheFlexBoundaryPcs={pcs}']
    if not p.exists():
        p.mkdir();(p/'command.json').write_text(json.dumps(args,indent=2))
        with (p/'run.log').open('w') as log:
            subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1200,
                env=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib'))
    else:assert json.loads((p/'command.json').read_text())==args
    output,counters,cfg=read(p,16)
    assert output['hash']==refs[t][0]['hash'],(name,'hash mismatch')
    copies=8*(t//128)*64*4*math.ceil(math.ceil(t/64)/(g or math.ceil(t/64))) if v=='spm' else 0
    assert counters['m_spmcpRequestsIssued']==counters['m_spmcpRequestsCompleted']==copies
    if obs:
        for i in range(8):cfg[f'system.cpu{i}']['cacheflexboundarypcs']=refs[t][2][f'system.cpu{i}']['cacheflexboundarypcs']
    audit(refs[t][2],cfg,0)
    row=dict(variant=v,group=g,mask=m,t=t,observer=obs,original=original,
             counters=counters,output=output,path=str(p),guest_sha256=sha(guest))
    if obs:row['boundaries']=boundaries(p,b/'guest.debug')
    return row
def batch(cases,stage):
    status(stage)
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        result=list(pool.map(lambda args:run(*args),cases))
    rows.extend(result);(out/'progress.json').write_text(json.dumps(rows,indent=2))
    state['completed'].append(stage);status(stage+'_complete')
    return result
try:
    # Every software intervention gets a small hash/copy/config gate first.
    masks=[0,1,2,4,8,16,32]
    for m in masks:
        batch([(v,g,m,128) for v,g in [('cache',1),('spm',4)]],f'smoke_m{m}')
        batch([(v,g,m,1024) for v,g in [('cache',1),('spm',4)]],f'timing_m{m}')
    batch([('cache',1,0,1024,False,True)],'original_control')
    # Complete factorial for packing interventions, not a sum of isolated deltas.
    for m in [3,5,6,7]:
        batch([(v,g,m,128) for v,g in [('cache',1),('spm',4)]],f'smoke_m{m}')
        batch([(v,g,m,1024) for v,g in [('cache',1),('spm',4)]],f'timing_m{m}')
    batch([(v,g,m,1024,True) for v,g in [('cache',1),('spm',4)] for m in [0,1,7]],'packing_phase_profiles')
    for r in rows:
        if not r['observer']:continue
        control=next(x for x in rows if (x['variant'],x['group'],x['mask'],x['t'],x['observer'])==
            (r['variant'],r['group'],r['mask'],r['t'],False))
        assert control['guest_sha256']==r['guest_sha256']
        assert control['output']==r['output']
        for k in r['counters'].keys()-{'path'}:assert r['counters'][k]==control['counters'][k],('observer perturbation',k)
    (out/'summary.json').write_text(json.dumps(rows,indent=2));status('complete')
except BaseException as e:
    state['failed']=repr(e);status('failed');raise
