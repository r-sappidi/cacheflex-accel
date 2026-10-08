"""Full-size independent numerical and input-pattern sensitivity cross-check."""
from _paths import FA_DIR
from pathlib import Path
import concurrent.futures as cf
import json,os,subprocess,sys
from validate_attribution import read
from run_causal import audit,parameters
from fa_independent_reference import reference,check
HERE=FA_DIR;ROOT=HERE.parents[1];B=ROOT/'results/cacheflex_flash_attention'
out=Path(sys.argv[1]).resolve();out.mkdir();state={'stage':'building'}
def status(s):state['stage']=s;(out/'status.json').write_text(json.dumps(state,indent=2));print(s,flush=True)
def run(case):
    v,hw=case;p=out/f'{v}_h{hw}';p.mkdir()
    ref=B/'correct_fa_20260911'/f'{v}_g4_m3_r0_t1024_h0'
    build=out/'build_pinned' if v=='spm' else B/'correct_fa_20260911/build_cache_g4_m3_r1'
    args=json.loads((ref/'command.json').read_text());args=[f'--outdir={p}' if x.startswith('--outdir=') else x for x in args];args[args.index('--cmd')+1]=str(build/'guest')
    for k,val in parameters(hw).items():args+=['-P',k+'='+val]
    (p/'command.json').write_text(json.dumps(args,indent=2))
    with (p/'run.log').open('w') as log:subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800,env=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib'))
    o,c,cfg=read(p,16);audit(read(ref,16)[2],cfg,hw)
    assert o['nonfinite']=='0' and c['m_spmcpRequestsIssued']==c['m_spmcpRequestsCompleted']==(16384 if v=='spm' else 0)
    print(v,hw,c['roi_us'],flush=True)
    return dict(variant=v,layout='pinned' if v=='spm' else 'cache',hardware_mask=hw,counters=c,output=o,path=str(p))
try:
    status('building')
    with (out/'build.log').open('w') as log:subprocess.run(['python3',str(HERE/'build_driver_pinned.py'),str(out/'build_pinned'),'spm','4','67','--correct','--rich'],stdout=log,stderr=subprocess.STDOUT,check=True)
    status('simulations_and_independent_reference')
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        oracle=pool.submit(reference,1024)
        rows=list(pool.map(run,[('cache',0),('spm',0),('spm',8)]))
        expected=oracle.result()
    status('numerical_gates')
    for row in rows:
        p=Path(row['path']);row['validation']=check(p/'output.fp16',1024,expected=expected)
        (p/'numerical_audit.json').write_text(json.dumps(row['validation'],indent=2))
    assert len({row['output']['hash'] for row in rows})==1
    (out/'summary.json').write_text(json.dumps(rows,indent=2));status('complete')
except BaseException as e:state['error']=repr(e);status('failed');raise
