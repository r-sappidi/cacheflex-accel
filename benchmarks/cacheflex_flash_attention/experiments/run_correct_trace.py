"""Nonintrusive same-guest residency validation after corrected ablations."""
from _paths import FA_DIR
from pathlib import Path
import concurrent.futures as cf
import json,os,subprocess,sys,time
from validate_attribution import read
from run_causal import audit
from analyze_correct_fa import analyze
HERE=FA_DIR;ROOT=HERE.parents[1];B=ROOT/'results/cacheflex_flash_attention'
prior=B/'correct_residual_20260911';out=Path(sys.argv[1]).resolve();out.mkdir()
state={'stage':'waiting'};rows=[]
def status(s):state['stage']=s;(out/'status.json').write_text(json.dumps(state,indent=2));print(s,flush=True)
def run(row):
    ref=Path(row['path']);p=out/ref.name;p.mkdir();args=json.loads((ref/'command.json').read_text())
    args=[f'--outdir={p}' if x.startswith('--outdir=') else x for x in args]
    args[1:1]=['--debug-flags=CacheFlexResidency,ProtocolTrace','--debug-file=debug.trace']
    (p/'command.json').write_text(json.dumps(args,indent=2))
    with (p/'run.log').open('w') as log:subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=3600,env=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib'))
    o,c,cfg=read(p,16);ro,rc,rf=read(ref,16);audit(rf,cfg,0)
    assert o==ro and all(c[k]==rc[k] for k in c.keys()-{'path'})
    assert (p/'output.fp16').read_bytes()==(ref/'output.fp16').read_bytes()
    result=dict(row,path=str(p),observer_gate='exact output, timing, extracted core/traffic counters')
    (p/'observer_gate.json').write_text(json.dumps(result,indent=2))
    analyze(p,Path(args[args.index('--cmd')+1]).with_name('guest.debug'))
    with (p/'analysis.log').open('w') as log:subprocess.run(['python3',str(HERE/'analyze_correct_residency.py'),str(p),str(row['group'])],stdout=log,stderr=subprocess.STDOUT,check=True)
    print(p.name,'analyzed',flush=True);return result
try:
    status('waiting')
    while not (prior/'summary.json').exists():
        if (prior/'status.json').exists():assert json.loads((prior/'status.json').read_text())['stage']!='failed'
        time.sleep(10)
    source=json.loads((prior/'summary.json').read_text())
    status('tracing_and_analyzing')
    with cf.ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(run,[r for r in source if r['hardware_mask'] in [0,8]]))
    (out/'summary.json').write_text(json.dumps(rows,indent=2));status('complete')
except BaseException as e:state['error']=repr(e);status('failed');raise
