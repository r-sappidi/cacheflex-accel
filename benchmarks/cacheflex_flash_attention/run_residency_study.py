"""Baseline eviction/reuse and fixed-latency capacity sensitivity.

Usage: run_residency_study.py NEW_OUTPUT SIMULATOR DEBUG_ELF
No guest changes. Stop if observer invariance or paired config audits fail.
"""
from pathlib import Path
import concurrent.futures as cf
import hashlib
import json
import os
import shutil
import subprocess
import sys
from validate_attribution import read
from analyze_residency import analyze

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
out=Path(sys.argv[1]).resolve();out.mkdir()
shutil.copy2(sys.argv[2],out/'gem5.opt');shutil.copy2(sys.argv[3],out/'cache.debug')
old=ROOT/'results/cacheflex_flash_attention/clean_vl16_20260908'
shutil.copy2(old/'runs/guests/cache',out/'cache')
for name in ('analyze_residency.py','run_residency_study.py','run_8core.sh'):
    shutil.copy2(HERE/name,out/name)
env=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib')
helptext=subprocess.check_output([str(out/'gem5.opt'),'--debug-help'],env=env,text=True)
assert 'CacheFlexResidency' in helptext, 'stale simulator: residency observer absent'
for name in ('cache','cache.debug'):
    subprocess.run(['aarch64-linux-gnu-objcopy','--dump-section',f'.text={out/name}.text',str(out/name),str(out/(name+'.objcopy'))],check=True)
assert (out/'cache.text').read_bytes()==(out/'cache.debug.text').read_bytes()
(out/'hashes.json').write_text(json.dumps({n:hashlib.sha256((out/n).read_bytes()).hexdigest() for n in ('cache','cache.debug','gem5.opt')},indent=2))
state=dict(stage='starting',complete=[],failed=None);rows=[]
def status(name):
    state['stage']=name;(out/'status.json').write_text(json.dumps(state,indent=2));print(name,flush=True)
def run(t,kib,ways,trace):
    name=f'cache_t{t}_l2_{kib}k_w{ways}_trace{trace}';path=out/name
    e=dict(env,CACHEFLEX_FLASH_VL='16',CACHEFLEX_FLASH_MATCHED_LRU='1',
       CACHEFLEX_FLASH_SPM_WAYS='0',CACHEFLEX_FLASH_PROFILE='1',
       CACHEFLEX_FLASH_BIN=str(out/'cache'),CACHEFLEX_GEM5_BIN=str(out/'gem5.opt'),
       CACHEFLEX_FLASH_OUT=str(path),CACHEFLEX_FLASH_PRIVATE_L2_SIZE=f'{kib}kB',
       CACHEFLEX_FLASH_PRIVATE_L2_ASSOC=str(ways),
       CACHEFLEX_FLASH_DEBUG_FLAGS='CacheFlexResidency' if trace else '')
    with (out/(name+'.host.log')).open('w') as log:
        subprocess.run(['bash',str(HERE/'run_8core.sh'),'cache_pretouch',str(t),'64','128'],
                       env=e,stdout=log,stderr=subprocess.STDOUT,check=True)
    output,stats,cfg=read(path,16)
    assert stats['m_spmcpRequestsIssued']==0
    return dict(path=str(path),t=t,kib=kib,ways=ways,trace=trace,output=output,counters=stats),cfg
def config_equal(a,b,capacity=False):
    assert a.sections()==b.sections()
    for sec in a.sections():
        assert set(a[sec])==set(b[sec])
        for key,x in a[sec].items():
            y=b[sec][key]
            if x==y:continue
            assert capacity and sec.startswith('system.ruby.l1_cntrl') and sec.endswith('.cache') and key in ('size','assoc'),(sec,key,x,y)
try:
    for t in (128,1024):
        status(f't{t}_observer_control')
        with cf.ThreadPoolExecutor(max_workers=2) as pool:
            jobs=[pool.submit(run,t,512,8,trace) for trace in (0,1)]
            (a,ca),(b,cb)=[j.result() for j in jobs]
        assert a['output']==b['output'];config_equal(ca,cb)
        for k in a['counters'].keys()-{'path'}:
            assert a['counters'][k]==b['counters'][k],('observer changed counter',k)
        _,oldstats,_=read(old/f'runs/cache_t{t}_p1',16)
        b['archived_roi_us']=oldstats['roi_us']
        b['archived_timing_matches']=b['counters']['roi_us']==oldstats['roi_us']
        status(f't{t}_trace_analysis');analyze(Path(b['path']),out/'cache.debug',t)
        rows.extend((a,b));state['complete'].append(f't{t}_observer_control')
        (out/'summary.json').write_text(json.dumps(rows,indent=2))
    # Same number of sets, larger associativity; all modeled latencies fixed.
    # This isolates the net modeled benefit of extra L2 capacity/ways, not the
    # hypothetical saved cycles of any single buffer or a physical SRAM design.
    status('t1024_capacity_ablations')
    for kib,ways in ((1024,16),(2048,32)):
        row,cfg=run(1024,kib,ways,1);config_equal(cb,cfg,capacity=True)
        assert row['output']==b['output']
        analyze(Path(row['path']),out/'cache.debug',1024)
        row['speedup_vs_512k']=b['counters']['roi_us']/row['counters']['roi_us']
        rows.append(row);(out/'summary.json').write_text(json.dumps(rows,indent=2))
    report=['# Baseline FA eviction/reuse study','',
       'Native VL16, T1024, eight cores, identical production guest. Trace-on/off controls pass.',
       'Capacity ablations retain modeled latency and set count; not physically costed cache proposals.','',
       '| Private L2 | ROI us | Speedup vs 512 KiB |','|---|---:|---:|']
    for r in rows:
        if r['t']==1024 and r['trace']:
            report.append(f"| {r['kib']} KiB / {r['ways']} ways | {r['counters']['roi_us']:.4f} | {r.get('speedup_vs_512k',1):.4f} |")
    report+=['','Per-run evictions.csv provides removal tick, buffer, ROB-head phase, incoming buffer and next observed read.',
      'residency.json contains occupancy and reused-line request service diagnostics. These are not additive causal cycle costs.',
      'Changing capacity measures a net configuration effect, not exact per-buffer saved time. See retained limitations.',
      'SPM residency comparison remains pending: the current SPM debug-side build did not match the preserved production text.']
    (out/'REPORT.md').write_text('\n'.join(report)+'\n');status('complete')
except BaseException as error:
    state['failed']=repr(error);status('failed');raise
