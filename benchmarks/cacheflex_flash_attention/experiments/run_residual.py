"""Matched-driver, phase/imbalance, resident QK and targeted sensitivity queue."""
import concurrent.futures as cf
import configparser
import hashlib
import json
import os
from _paths import FA_DIR
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from analyze_boundaries import markers,analyze as boundaries
from analyze_traffic import analyze as traffic
from validate_attribution import read
from run_causal import audit,config_key

HERE=FA_DIR;ROOT=HERE.parents[1]
ARCH=ROOT/'results/cacheflex_flash_attention/causal_20260910_v3'
out=Path(sys.argv[1]).resolve();resume='--resume' in sys.argv[3:]
out.mkdir(exist_ok=resume)
sim=Path(sys.argv[2])
def freeze(source,target):
    if resume:
        assert hashlib.sha256(source.read_bytes()).digest()==hashlib.sha256(target.read_bytes()).digest(), ('frozen input changed',str(target))
    else:shutil.copy2(source,target)
freeze(sim,out/'gem5.opt')
if resume and (out/'status.json').exists():
    shutil.copy2(out/'status.json',out/'status.before_resume.json')
builds={v:ROOT/f'results/cacheflex_flash_attention/matched_{v}_build_20260911' for v in ('cache','spm')}
for v,b in builds.items():
    freeze(b/'guest',out/v);freeze(b/'guest.debug',out/(v+'.debug'))
    freeze(ARCH/v,out/('original_'+v))
    freeze(ROOT/f'results/cacheflex_flash_attention/resident_qk_build_20260911/{int(v=="spm")}',out/('resident_'+v))
rows=[];state=dict(stage='starting',completed=[],failed=None)
def status(stage):
    state['stage']=stage;(out/'status.json').write_text(json.dumps(state,indent=2));print(stage,flush=True)
def run(v,t,kind='matched',obs=False,params=None):
    params=params or {};name=f'{kind}_{v}_t{t}_p{int(obs)}';path=out/name
    reuse=resume and path.exists()
    if not reuse:path.mkdir()
    guest=out/(('original_' if kind=='original' else 'resident_' if kind=='resident' else '')+v)
    line=next(x for x in (ARCH/f'factorial_{v}_t1024_m00_r0/run.log').read_text().splitlines() if x.startswith('command line: '))
    args=shlex.split(line.removeprefix('command line: '));args[0]=str(out/'gem5.opt')
    args=[f'--outdir={path}' if x.startswith('--outdir=') else x for x in args]
    args[args.index('--cmd')+1]=str(guest);args[args.index('--options')+1]=f'{t} 64 64 128'
    if obs:
        pcs=sorted(markers(out/(v+'.debug')))
        args[1:1]=['--debug-flags=CacheFlexBoundary','--debug-file=boundaries.trace']
        for i in range(8):params[f'system.cpu[{i}].cacheFlexBoundaryPcs']=repr(pcs)
    for p,value in params.items():args+=['-P',p+'='+str(value)]
    if reuse:
        assert json.loads((path/'command.json').read_text())==args, ('resume command mismatch',name)
        assert 'exiting with last active thread context' in (path/'run.log').read_text(), ('incomplete archived run',name)
    else:
        (path/'command.json').write_text(json.dumps(args,indent=2))
        with (path/'run.log').open('w') as log:
            subprocess.run(args,env=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib'),stdout=log,stderr=subprocess.STDOUT,check=True)
    if kind=='resident':
        text=(path/'run.log').read_text();assert 'RESIDENT_RESULT errors=0' in text and 'exiting with last active thread context' in text
        output={'hash':'resident_analytic_pass'};counts=traffic(path)
        cfg=configparser.ConfigParser(interpolation=None,strict=False);cfg.read(path/'config.ini')
        expected=0
    else:
        output,counts,cfg=read(path,16);expected=8*(t//128)*64*4 if v=='spm' else 0
    assert counts['m_spmcpRequestsIssued']==counts['m_spmcpRequestsCompleted']==expected
    row=dict(kind=kind,variant=v,t=t,observer=obs,path=str(path),output=output,counters=counts)
    if obs:row['boundaries']=boundaries(path,out/(v+'.debug'))
    # Save exact configs; normalization is only for comparisons.
    for sec in cfg.sections():
        for key,val in list(cfg[sec].items()):cfg[sec][key]=val.replace(str(path),'<RESULT>')
    return row,cfg
def parameter_path(section):
    import re
    path=re.sub(r'(cpu|FUList|opList)(\d+)',lambda m:f'{m[1]}[{int(m[2])}]',section)
    if path.endswith('.opList'):path+='[0]'
    return path

def cfgcheck(a,b,params=None):
    ref=configparser.ConfigParser(interpolation=None,strict=False)
    ref.read_dict({s:dict(a[s]) for s in a.sections()})
    for p,value in (params or {}).items():
        sec,key=config_key(p)
        if sec not in b:
            matches=[s for s in b.sections() if parameter_path(s)==p.rsplit('.',1)[0]]
            assert len(matches)==1, ('parameter section ambiguity',p,matches)
            sec=matches[0]
        assert b[sec][key]==str(value), (p,b[sec][key],value)
        ref[sec][key]=b[sec][key]
        # ArmO3CPU derives CC capacity as Self.numPhysIntRegs * 5.
        # Validate that exact dependency, not an unrestricted audit exclusion.
        if key=='numphysintregs':
            assert int(a[sec]['numphysccregs'])==5*int(a[sec][key])
            assert int(b[sec]['numphysccregs'])==5*int(value)
            ref[sec]['numphysccregs']=str(5*int(value))
    for i in range(8):
        ref[f'system.cpu{i}']['cacheflexboundarypcs']=b[f'system.cpu{i}']['cacheflexboundarypcs']
    audit(ref,b,0)
def pair(t,kind='matched',obs=False,params=None):
    status(f'{kind}_t{t}_p{int(obs)}')
    with cf.ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(lambda v:run(v,t,kind,obs,dict(params or {})),('cache','spm')))
    a,ca=results[0];b,cb=results[1];assert a['output']['hash']==b['output']['hash'];cfgcheck(ca,cb)
    rows.extend(r for r,_ in results);(out/'summary.json').write_text(json.dumps(rows,indent=2))
    state['completed'].append(state['stage']);status(state['stage']+'_complete')
    return results
try:
    for t in (128,1024):
        original=pair(t,'original');off=pair(t);on=pair(t,obs=True)
        for i in range(2):
            assert original[i][0]['output']['hash']==off[i][0]['output']['hash']
            cfgcheck(original[i][1],off[i][1]);cfgcheck(off[i][1],on[i][1])
            assert off[i][0]['output']==on[i][0]['output']
            for k in off[i][0]['counters'].keys()-{'path'}:
                assert off[i][0]['counters'][k]==on[i][0]['counters'][k],('observer perturbation',k)
    pair(128,'resident')
    # These isolates address the broad phase gap: coherent operand service,
    # OoO overlap, controller issue capacity, and vector dependency latency.
    tests={}
    tests['regular_ports']={f'system.cpu[{i}].cacheLoadPorts':8 for i in range(8)}
    tests['window']={f'system.cpu[{i}].{key}':val for i in range(8) for key,val in
        [('numROBEntries',512),('numIQEntries',256),('LQEntries',128),('SQEntries',128),
         ('numPhysIntRegs',512),('numPhysFloatRegs',512),('numPhysVecRegs',512)]}
    tests['controllers']={f'system.ruby.{ctrl}{i}.transitions_per_cycle':128 for i in range(8)
        for ctrl in ('l0_cntrl','l1_cntrl','l2_cntrl','dir_cntrl')}
    tests['sequencer']={f'system.ruby.l0_cntrl{i}.sequencer.max_outstanding_requests':256 for i in range(8)}
    tests['simd_latency']={}
    cfg=off[0][1]
    for sec in cfg.sections():
        if sec.startswith('system.cpu') and cfg[sec].get('opclass','').startswith('Simd') and 'oplat' in cfg[sec]:
            # config section names use FUList5/opListN; -P uses vector indexing.
            p=parameter_path(sec)
            tests['simd_latency'][p+'.opLat']=1
    assert tests['simd_latency']
    (out/'sensitivity_parameters.json').write_text(json.dumps(tests,indent=2))
    for name,params in tests.items():
        result=pair(1024,name,params=params)
        for i in range(2):
            cfgcheck(off[i][1],result[i][1],params)
            assert off[i][0]['output']['hash']==result[i][0]['output']['hash']
    report=['# Matched FA residual study','',
        'Matched guests share packed-K preparation, loop order, tiling and barriers. The original pair is retained.',
        'Boundary intervals observe retirement, include intervening stalls/control, and are not causal instruction-cost bins.',
        'Observer off/on uses identical labeled guest bytes; empty-asm builds are not timing controls.',
        'Resident QK excludes staging, uses one warmup and 128 repetitions, and checks every output analytically.', '',
        'The window test also scales ARM condition-code registers from 640 to 2560 via the existing 5x integer-register default; this is a combined window/register-capacity intervention.', '',
        '| Case | T | Cache us | SPM us | SPM/cache |','|---|---:|---:|---:|---:|']
    for kind,t,obs in dict.fromkeys((r['kind'],r['t'],r['observer']) for r in rows):
        if obs:continue
        p={r['variant']:r['counters']['roi_us'] for r in rows if (r['kind'],r['t'],r['observer'])==(kind,t,obs)}
        report.append(f"| {kind} | {t} | {p['cache']:.4f} | {p['spm']:.4f} | {p['spm']/p['cache']:.4f} |")
    (out/'REPORT.md').write_text('\n'.join(report)+'\n');status('complete')
except BaseException as error:
    state['failed']=repr(error);status('failed');raise
