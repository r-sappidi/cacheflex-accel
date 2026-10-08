"""Confirm packing and fixed-region frontend effects on correct FA arithmetic."""
from _paths import FA_DIR
from pathlib import Path
import concurrent.futures as cf
import json,os,subprocess,sys,time
from validate_attribution import read
from run_causal import audit
from analyze_correct_fa import analyze
HERE=FA_DIR;ROOT=HERE.parents[1];B=ROOT/'results/cacheflex_flash_attention'
prior=B/'correct_fa_20260911';wait=B/'correct_trace_20260911'
out=Path(sys.argv[1]).resolve();out.mkdir(exist_ok='--resume-wait' in sys.argv);state={'stage':'waiting'}
def status(s):state['stage']=s;(out/'status.json').write_text(json.dumps(state,indent=2));print(s,flush=True)
def run(case):
    name,v,build=case;ref=prior/f'{v}_g4_m3_r0_t1024_h0';p=out/name;p.mkdir()
    args=json.loads((ref/'command.json').read_text());args=[f'--outdir={p}' if x.startswith('--outdir=') else x for x in args]
    args[args.index('--cmd')+1]=str(build/'guest');(p/'command.json').write_text(json.dumps(args,indent=2))
    with (p/'run.log').open('w') as log:subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800,env=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib'))
    o,c,cfg=read(p,16);ro,rc,rf=read(ref,16);audit(rf,cfg,0)
    assert o==ro and 'FA_ANALYTIC errors=0' in (p/'run.log').read_text()
    assert (p/'output.fp16').read_bytes()==(ref/'output.fp16').read_bytes()
    assert c['m_spmcpRequestsIssued']==c['m_spmcpRequestsCompleted']==rc['m_spmcpRequestsIssued']
    analyze(p,build/'guest.debug');print(name,c['roi_us'],flush=True)
    return dict(name=name,variant=v,counters=c,output=o,path=str(p))
try:
    status('waiting')
    # All three observer gates are written only after their simulators exit.
    # Host-side trace analysis can then overlap these four timed simulations.
    while len(list(wait.glob('*/observer_gate.json')))!=3:
        if (wait/'status.json').exists():assert json.loads((wait/'status.json').read_text())['stage']!='failed'
        time.sleep(10)
    cases=[(f'qk_o{o}','spm',B/f'correct_qk_pair_v2_20260911/o{o}') for o in [0,16]]
    cases += [(f'pv_o{o}','cache',B/f'correct_pv_pair_v2_20260911/o{o}') for o in [8,24]]
    cases += [(f'legacy_pack_{v}',v,B/f'correct_pack_{v}_20260911') for v in ['cache','spm']]
    status('software_controls')
    with cf.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(run,cases))
    (out/'summary.json').write_text(json.dumps(rows,indent=2));status('complete')
except BaseException as e:state['error']=repr(e);status('failed');raise
