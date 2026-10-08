"""Check argv/startup sensitivity with frozen guests; no hardware changes."""
from pathlib import Path
import concurrent.futures as cf
import json,os,subprocess,sys
from validate_attribution import read
from run_causal import audit
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'results/cacheflex_flash_attention'
out=Path(sys.argv[1]).resolve();out.mkdir()
old=B/'residual_retry_20260911';group=B/'query_groups_retry_20260911';study=B/'driver_causes_20260911'
def run(case):
    name,guest,v,g=case;p=out/name;p.mkdir()
    args=json.loads((old/f'original_{v}_t1024_p0/command.json').read_text())
    args=[f'--outdir={p}' if x.startswith('--outdir=') else x for x in args]
    args[args.index('--cmd')+1]=str(guest)
    args+=['-P',"system.cpu[0].workload[0].cmd=['cacheflex-fa','1024','64','64','128']"]
    (p/'command.json').write_text(json.dumps(args,indent=2))
    with (p/'run.log').open('w') as log:
        subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1200,
            env=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib'))
    o,c,cfg=read(p,16);ro,rc,ref=read(old/'original_cache_t1024_p0',16)
    assert o['hash']==ro['hash'];audit(ref,cfg,0)
    assert cfg['system.cpu0.workload']['cmd']=='cacheflex-fa 1024 64 64 128'
    copies=0 if v=='cache' else 65536
    assert c['m_spmcpRequestsIssued']==c['m_spmcpRequestsCompleted']==copies
    return dict(name=name,counters=c,output=o)
cases=[('spm_archived',group/'build_spm_g4/guest','spm',4),
       ('spm_rebuilt',study/'build_spm_g4_m0/guest','spm',4),
       ('cache_original',old/'original_cache','cache',1),
       ('cache_matched',group/'build_cache_g1/guest','cache',1)]
rows=[]
for start in [0,2]:
    with cf.ThreadPoolExecutor(max_workers=2) as pool:rows+=list(pool.map(run,cases[start:start+2]))
    (out/'progress.json').write_text(json.dumps(rows,indent=2));print(rows,flush=True)
(out/'summary.json').write_text(json.dumps(rows,indent=2))
