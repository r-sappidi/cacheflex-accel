"""Gated, resumable-by-stage MLP study. Each stage requires a NEW directory.

Stage 1: first invocation plus three measured warmed invocations, both paths.
Stage 2: independent KC/mapping search using shared persistent packed weights.
"""
from pathlib import Path
import concurrent.futures as cf
import configparser
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
from audit_mlp_math import audit as math_audit

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]

def epochs(path,count):
    text=(path/'stats.txt').read_text()
    blocks=text.split('Begin Simulation Statistics')[1:]
    assert len(blocks)>=count
    rows=[]
    for block in blocks[:count]:
        stats={}
        for line in block.split('End Simulation Statistics')[0].splitlines():
            w=line.split()
            if len(w)>1:
                try:stats[w[0]]=float(w[1])
                except ValueError:pass
        def total(pattern):return sum(v for k,v in stats.items() if re.search(pattern,k))
        rows.append(dict(us=stats['simTicks']/1e6,
                         copies_issued=total(r'\.m_spmcpRequestsIssued$'),
                         copies_completed=total(r'\.m_spmcpRequestsCompleted$'),
                         l2_misses=total(r'l1_cntrl\d+.cache.m_demand_misses$'),
                         dram_read_bytes=total(r'\.dram.bytesRead::total$'),
                         packets=stats.get('system.ruby.network.packets_injected::total',0)))
    return rows

def audit(a,b):
    ca,cb=(configparser.ConfigParser(interpolation=None,strict=False) for _ in range(2))
    ca.read(Path(a['path'])/'config.ini');cb.read(Path(b['path'])/'config.ini')
    assert ca.sections()==cb.sections()
    for sec in ca.sections():
        assert set(ca[sec])==set(cb[sec])
        for key,x in ca[sec].items():
            y=cb[sec][key];x=x.replace(a['path'],'<RUN>');y=y.replace(b['path'],'<RUN>')
            if x==y:continue
            if sec=='system.cpu0.workload' and key in ('cmd','executable'):continue
            assert key=='num_spm_ways' and x=='0' and y==str(b['vl']//4),(sec,key,x,y)

def main():
    stage=int(sys.argv[1]);out=Path(sys.argv[2]).resolve();build=Path(sys.argv[3]).resolve()
    out.mkdir();rows=[]
    tile_rows=int((build/'rows.txt').read_text()) if (build/'rows.txt').exists() else 24
    acc32=int((build/'acc32.txt').read_text()) if (build/'acc32.txt').exists() else 0
    seed=int(os.environ.get('MLP_SEED','20260909'));xdiv=float(os.environ.get('MLP_XDIV','128'))
    cores=int(os.environ.get('MLP_CPUS','8'));vl=int(os.environ.get('MLP_VL','16'))
    llc_mib=int(os.environ.get('MLP_LLC_MIB','32'))
    for v in ('cache','spm'):shutil.copy2(build/v,out/v)
    old=ROOT/'results/ai_workloads/mlp_clean_20260909/runs'
    shutil.copy2(Path(os.environ.get('MLP_GEM5_BIN',str(old/'gem5.opt'))),out/'gem5.opt')
    for name in ('mlp_study.cpp','sweep_mlp_study.py','mlp_reference.py'):
        shutil.copy2(build/name if (build/name).exists() else HERE/name,out/name)
    hashes={v:hashlib.sha256((out/v).read_bytes()).hexdigest() for v in ('cache','spm','gem5.opt')}
    (out/'hashes.json').write_text(json.dumps(hashes,indent=2))

    def input_file(t,d,f,kc):
        p=out/f'input_{t}_{d}_{f}_kc{kc}.bin'
        if not p.exists():
            if kc==256 and not acc32 and seed==20260909 and xdiv==128 and (old/f'input_{t}_{d}_{f}.bin').exists():
                shutil.copy2(old/f'input_{t}_{d}_{f}.bin',p)
            else:
                subprocess.run([sys.executable,str(HERE/'mlp_reference.py'),str(t),str(d),str(f),str(p),str(kc),str(acc32),str(seed),str(xdiv)],check=True)
        return p

    def run(v,t,d,f,mp,kc,warm,repeats,prepack,inp):
        name=f'{v}_t{t}_d{d}_f{f}_mp{mp}_kc{kc}_w{warm}_r{repeats}_p{prepack}'
        path=out/name
        env=dict(PATH=os.environ['PATH'],HOME=os.environ['HOME'],
                 CACHEFLEX_FLASH_VL=str(vl),CACHEFLEX_FLASH_MATCHED_LRU='1',
                 CACHEFLEX_FLASH_SPM_WAYS=str(vl//4) if v=='spm' else '0',
                 CACHEFLEX_FLASH_OPTIONS=f'{t} {d} {f} {mp} {kc} {warm} {repeats} {prepack} {cores}',
                 CACHEFLEX_FLASH_READFILE=str(inp),CACHEFLEX_FLASH_BIN=str(out/v),
                 CACHEFLEX_FLASH_OUT=str(path),CACHEFLEX_GEM5_BIN=str(out/'gem5.opt'),
                 MLP_CPUS=str(cores),MLP_LLC_MIB=str(llc_mib))
        with (out/(name+'.host.log')).open('w') as log:
            runner=HERE/'run_mlp_config.sh' if stage==6 else HERE.parent/'cacheflex_flash_attention/run_8core.sh'
            subprocess.run(['bash',str(runner),
                            'cache_pretouch' if v=='cache' else 'spm_konly_overlap'],
                           env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        log=(path/'run.log').read_text()
        assert 'exiting with last active thread context' in log
        assert not re.search(r'panic:|fatal:| FAIL|Simulated exit code not 0',log)
        checks=[dict(re.findall(r'(\w+)=([^\s]+)',line)) for line in log.splitlines() if line.startswith('MLP_RESULT')]
        assert len(checks)==warm+repeats
        assert all(c['errors']==c['nonfinite']=='0' for c in checks)
        assert len({c['hash'] for c in checks})==1
        timings=epochs(path,repeats)
        expected=min(mp,(t+tile_rows-1)//tile_rows)*4*(2*d*(f//128)+f*(d//128)) if v=='spm' else 0
        assert all(r['copies_issued']==r['copies_completed']==expected for r in timings),(name,expected,timings)
        cfg=configparser.ConfigParser(interpolation=None,strict=False);cfg.read(path/'config.ini')
        for i in range(cores):
            assert int(cfg[f'system.cpu{i}.isa']['sve_vl_se'])==vl
            assert int(cfg[f'system.ruby.l1_cntrl{i}.cache']['num_spm_ways'])==(vl//4 if v=='spm' else 0)
        assert sum(int(cfg[f'system.ruby.l2_cntrl{i}.L2cache']['size']) for i in range(cores))==llc_mib*1024**2
        times=[r['us'] for r in timings]
        accuracy=math_audit(path,inp,t,d,f) if (path/'output.bin').exists() else None
        if stage==5:assert accuracy and accuracy['passed'],(name,accuracy)
        return dict(variant=v,t=t,d=d,f=f,mp=mp,kc=kc,warm=warm,repeats=repeats,tile_rows=tile_rows,
                    prepack=prepack,acc32=acc32,seed=seed,xdiv=xdiv,cores=cores,vl=vl,llc_mib=llc_mib,path=str(path),checks=checks,epochs=timings,
                    median_us=statistics.median(times),spread=(max(times)-min(times))/statistics.median(times),accuracy=accuracy)

    def batch(cases):
        inputs={case:input_file(case[0],case[1],case[2],case[4]) for case in cases}
        with cf.ThreadPoolExecutor(max_workers=4) as pool:
            jobs={(case,v):pool.submit(run,v,*case,inputs[case]) for case in cases for v in ('cache','spm')}
            for case in cases:
                a=jobs[case,'cache'].result();b=jobs[case,'spm'].result()
                audit(a,b);assert a['checks'][0]['hash']==b['checks'][0]['hash']
                rows.extend((a,b));(out/'progress.json').write_text(json.dumps(rows,indent=2))
                print('PASS',case,'cache',a['median_us'],'spm',b['median_us'],'speedup',a['median_us']/b['median_us'],flush=True)

    # t,d,f,mp,kc,warm,repeats,prepack
    if stage==1:
        batch([(24,128,256,2,256,1,2,1)])
        batch([(t,512,1536,2,256,warm,repeats,1) for t in (64,128) for warm,repeats in ((0,1),(2,3))])
    elif stage==11:
        batch([(128,512,1536,2,256,5,5,1)])
    elif stage in (2,3,4,5,6):
        batch([(24,128,256,min(2,cores),256,1,1,1)])
        if len(sys.argv)>4:
            batch([tuple(case) for case in json.loads(Path(sys.argv[4]).read_text())])
        else:
            for t in (64,128):
                batch([(t,512,1536,mp,kc,1,1,1) for kc in (128,256,512) for mp in (1,2,4,8)])
        finalists=set()
        for t,d,f in sorted({(r['t'],r['d'],r['f']) for r in rows if r['t']!=24}):
            for v in ('cache','spm'):
                winner=min((r for r in rows if (r['t'],r['d'],r['f'])==(t,d,f) and r['variant']==v),key=lambda r:r['median_us'])
                finalists.add((t,winner['d'],winner['f'],winner['mp'],winner['kc'],2,3,1))
        batch(sorted(finalists))
    else:raise ValueError(stage)
    (out/'summary.json').write_text(json.dumps(rows,indent=2))
    print('COMPLETE',flush=True)

if __name__=='__main__':main()
