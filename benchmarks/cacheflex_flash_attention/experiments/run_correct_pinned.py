"""Separate staging-repetition and occupied-capacity tradeoff, fixed group4."""
from _paths import FA_DIR
from pathlib import Path
import concurrent.futures as cf
import json,os,subprocess,sys,time
from validate_attribution import read
from run_causal import audit,parameters
from analyze_correct_fa import analyze
HERE=FA_DIR;ROOT=HERE.parents[1];B=ROOT/'results/cacheflex_flash_attention'
out=Path(sys.argv[1]).resolve();out.mkdir();state={'stage':'waiting'}
def status(s):state['stage']=s;(out/'status.json').write_text(json.dumps(state,indent=2));print(s,flush=True)
def run(hw):
    ref=B/'correct_fa_20260911/spm_g4_m3_r0_t1024_h0';p=out/f'pinned_g4_h{hw}';p.mkdir()
    args=json.loads((ref/'command.json').read_text());args=[f'--outdir={p}' if x.startswith('--outdir=') else x for x in args]
    build=B/'correct_pinned_20260911';args[args.index('--cmd')+1]=str(build/'guest')
    for k,v in parameters(hw).items():args+=['-P',k+'='+v]
    (p/'command.json').write_text(json.dumps(args,indent=2))
    with (p/'run.log').open('w') as log:subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800,env=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib'))
    o,c,cfg=read(p,16);ro,_,rf=read(ref,16);audit(rf,cfg,hw)
    assert o==ro and 'FA_ANALYTIC errors=0' in (p/'run.log').read_text()
    assert (p/'output.fp16').read_bytes()==(ref/'output.fp16').read_bytes()
    assert c['m_spmcpRequestsIssued']==c['m_spmcpRequestsCompleted']==16384
    # Pinned source line numbers differ; function-level-only phase matches
    # remain valid, unmatched driver PCs remain unresolved.
    analyze(p,build/'guest.debug');print(hw,c['roi_us'],flush=True)
    return dict(hardware_mask=hw,counters=c,output=o,path=str(p))
try:
    status('waiting')
    prior=B/'correct_software_20260911'
    while not (prior/'summary.json').exists():
        if (prior/'status.json').exists():assert json.loads((prior/'status.json').read_text())['stage']!='failed'
        time.sleep(10)
    status('pinned_capacity_pair')
    with cf.ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(run,[0,8]))
    (out/'summary.json').write_text(json.dumps(rows,indent=2));status('complete')
except BaseException as e:state['error']=repr(e);status('failed');raise
