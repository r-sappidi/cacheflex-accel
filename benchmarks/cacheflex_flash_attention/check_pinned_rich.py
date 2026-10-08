"""Full-T1024 nonperiodic-input routing regression for scalar pinned K."""
from pathlib import Path
import concurrent.futures as cf
import json,os,subprocess,sys
from validate_attribution import read
from run_causal import audit
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];B=ROOT/'results/cacheflex_flash_attention'
out=Path(sys.argv[1]).resolve();out.mkdir();old=B/'residual_retry_20260911'
def run(v):
    b=out/f'build_{v}';script='build_driver_pinned.py' if v=='spm' else 'build_driver_aligned.py'
    with (out/f'build_{v}.log').open('w') as log:subprocess.run(['python3',str(HERE/script),str(b),v,'1',str(67 if v=='spm' else 3),'--rich'],stdout=log,stderr=subprocess.STDOUT,check=True)
    p=out/v;p.mkdir();args=json.loads((old/f'original_{v}_t1024_p0/command.json').read_text())
    args=[f'--outdir={p}' if x.startswith('--outdir=') else x for x in args];args[args.index('--cmd')+1]=str(b/'guest')
    args+=['-P',"system.cpu[0].workload[0].cmd=['cacheflex-fa','1024','64','64','128']"]
    (p/'command.json').write_text(json.dumps(args,indent=2))
    with (p/'run.log').open('w') as log:subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1200,
        env=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib'))
    o,c,cfg=read(p,16);return o,c,cfg
with cf.ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(run,['cache','spm']))
a,b=results;assert a[0]['hash']==b[0]['hash'];audit(a[2],b[2],0)
assert a[0]['hash']!='fdce7e0feced0383','rich input accidentally disabled'
assert a[1]['m_spmcpRequestsIssued']==a[1]['m_spmcpRequestsCompleted']==0
assert b[1]['m_spmcpRequestsIssued']==b[1]['m_spmcpRequestsCompleted']==16384
(out/'summary.json').write_text(json.dumps(dict(passed=True,output=a[0],counters=[a[1],b[1]],
    caveat='Nonperiodic coherent-vs-SPM routing regression, not independent FA numerical validation.'),indent=2))
print('PASS: nonperiodic full T1024 output hash and all 16384 pinned copies',flush=True)
