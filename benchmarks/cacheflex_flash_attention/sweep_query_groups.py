"""Independent matched FA group-size sweep; two concurrent simulations.

Usage: sweep_query_groups.py NEW_OUTPUT [SIMULATOR]
Fixed T1024/d64/Br64/Bc128/VL16/eight cores and existing default hardware.
T128 regression per group, then three repetitions per T1024 cache/SPM pair.
"""
from pathlib import Path
import concurrent.futures as cf
import hashlib
import json
import math
import os
import shlex
import shutil
import statistics
import subprocess
import sys
from run_causal import audit
from validate_attribution import read

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OLD=ROOT/'results/cacheflex_flash_attention/residual_20260911'
out=Path(sys.argv[1]).resolve();out.mkdir()
shutil.copy2(Path(sys.argv[2]) if len(sys.argv)>2 else OLD/'gem5.opt',out/'gem5.opt')
sources=[HERE/n for n in ('sweep_query_groups.py','build_matched_residual.sh',
    'spm_multicore.cpp','cache_qk_wrapper.cpp','make_search_kernels.py',
    'validate_attribution.py','analyze_traffic.py','run_causal.py')]
hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources+[out/'gem5.opt']}
(out/'sources').mkdir()
for p in sources:shutil.copy2(p,out/'sources'/p.name)
(out/'hashes.json').write_text(json.dumps(hashes,indent=2))
(out/'experiment.json').write_text(json.dumps(dict(groups=[1,2,4,8,0],zero_means='all',
    t=1024,d=64,br=64,bc=128,cores=8,vl=16,repetitions=3,
    fixed='Matched shared-K packing, K-only double buffering, kernels, arithmetic, hardware and barriers.',
    changed='Query group size and its dependent scratch allocation/copy repetition count.',
    caveat='Finite hash agreement is functional regression, not independent full-FA numerical validation.'),indent=2))
state=dict(stage='starting',completed=[],failed=None);rows=[];reference={};refcfg={}
def status(stage):
    state['stage']=stage;(out/'status.json').write_text(json.dumps(state,indent=2));print(stage,flush=True)
def unchanged():
    for p,h in hashes.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,('source changed',p)
def build(v,g):
    b=out/f'build_{v}_g{g}';unchanged();status(b.name)
    with (out/(b.name+'.log')).open('w') as log:
        subprocess.run(['bash',str(HERE/'build_matched_residual.sh'),str(b),v,str(g)],stdout=log,stderr=subprocess.STDOUT,check=True)
    hashes[str(b/'guest')]=hashlib.sha256((b/'guest').read_bytes()).hexdigest()
    (out/'hashes.json').write_text(json.dumps(hashes,indent=2))
def run(v,g,t,rep):
    path=out/f'{v}_g{g}_t{t}_r{rep}';path.mkdir()
    original=OLD/f'original_{v}_t{t}_p0/run.log'
    line=next(x for x in original.read_text().splitlines() if x.startswith('command line: '))
    args=shlex.split(line.removeprefix('command line: '));args[0]=str(out/'gem5.opt')
    args=[f'--outdir={path}' if x.startswith('--outdir=') else x for x in args]
    args[args.index('--cmd')+1]=str(out/f'build_{v}_g{g}/guest')
    (path/'command.json').write_text(json.dumps(args,indent=2))
    with (path/'run.log').open('w') as log:
        subprocess.run(args,env=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib'),
            stdout=log,stderr=subprocess.STDOUT,check=True)
    output,counters,cfg=read(path,16)
    nq=math.ceil(t/64);actual_group=min(g,nq) if g else nq
    copies=8*(t//128)*64*4*math.ceil(nq/actual_group) if v=='spm' else 0
    assert counters['m_spmcpRequestsIssued']==counters['m_spmcpRequestsCompleted']==copies
    assert output['hash']==reference[t]
    audit(refcfg[t],cfg,0)
    return dict(variant=v,group=g,actual_group=actual_group,t=t,rep=rep,
                path=str(path),counters=counters,output=output),cfg
def pair(g,t,rep):
    unchanged();status(f'pair_g{g}_t{t}_r{rep}')
    with cf.ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(lambda v:run(v,g,t,rep),('cache','spm')))
    audit(results[0][1],results[1][1],0)
    rows.extend(r for r,_ in results);(out/'progress.json').write_text(json.dumps(rows,indent=2))
    state['completed'].append(state['stage']);status(state['stage']+'_complete')
try:
    for t in (128,1024):
        result,_,cfg=read(OLD/f'original_cache_t{t}_p0',16)
        reference[t]=result['hash'];refcfg[t]=cfg
    for g in (1,2,4,8,0):
        for v in ('cache','spm'):build(v,g)
        pair(g,128,0)
    # Repetition rounds rotate order to avoid tying a group to host scheduling.
    groups=[1,2,4,8,0]
    for rep in range(3):
        for g in groups[rep:]+groups[:rep]:pair(g,1024,rep)
    medians={}
    report=['# Matched query-group sweep','',
        'T1024, d64, Br64/Bc128, VL16, eight cores, shared-K preparation, K-only double buffering.',
        'Group size and its dependent scratch size/copy count are the only intended software changes.',
        'Three repetitions; no new hardware changes. Output hashes, completed copies and full configuration audit required.',
        'This is not independent full-FA numerical validation.','',
        '| Query blocks/group | Cache median us | SPM median us | Cache/SPM | SPM copies |',
        '|---:|---:|---:|---:|---:|']
    for g in groups:
        m={}
        for v in ('cache','spm'):
            subset=[r for r in rows if r['t']==1024 and r['group']==g and r['variant']==v]
            times=[r['counters']['roi_us'] for r in subset]
            m[v]=dict(median_us=statistics.median(times),range_us=max(times)-min(times),
                      copies=subset[0]['counters']['m_spmcpRequestsCompleted'])
        medians[g]=m
        report.append(f"| {g or 'all (16)'} | {m['cache']['median_us']:.4f} | {m['spm']['median_us']:.4f} | {m['cache']['median_us']/m['spm']['median_us']:.4f} | {int(m['spm']['copies'])} |")
    winners={v:min(groups,key=lambda g:medians[g][v]['median_us']) for v in ('cache','spm')}
    (out/'summary.json').write_text(json.dumps(dict(rows=rows,medians=medians,winners=winners),indent=2))
    (out/'REPORT.md').write_text('\n'.join(report)+'\n');status('complete')
except BaseException as error:
    state['failed']=repr(error);status('failed');raise
