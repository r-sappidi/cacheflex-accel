"""Matched FA-hardware full-MLP run and correctness/config audit."""
from pathlib import Path
import concurrent.futures as cf
import configparser
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'cacheflex_flash_attention'))
from analyze_traffic import analyze
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
out=Path(sys.argv[1]);build=Path(sys.argv[2]);out.mkdir()
for v in ('cache','spm'):shutil.copy2(build/v,out/v)
shutil.copy2(ROOT/'results/cacheflex_flash_attention/clean_vl16_20260908/runs/gem5.opt',out/'gem5.opt')
(out/'hashes.json').write_text(json.dumps({v:hashlib.sha256((out/v).read_bytes()).hexdigest() for v in ('cache','spm','gem5.opt')},indent=2))
rows=[]
def run(v,t,d,f,mp):
 name=f'{v}_t{t}_d{d}_f{f}_mp{mp}';path=out/name
 env=dict(PATH=os.environ['PATH'],HOME=os.environ['HOME'],CACHEFLEX_FLASH_VL='16',
          CACHEFLEX_FLASH_MATCHED_LRU='1',CACHEFLEX_FLASH_SPM_WAYS='4' if v=='spm' else '0',
          CACHEFLEX_FLASH_OPTIONS=f'{t} {d} {f} {mp}',CACHEFLEX_FLASH_READFILE=str(out/f'input_{t}_{d}_{f}.bin'),
          CACHEFLEX_FLASH_BIN=str(out/v),CACHEFLEX_FLASH_OUT=str(path),CACHEFLEX_GEM5_BIN=str(out/'gem5.opt'))
 with (out/(name+'.host.log')).open('w') as log:
  subprocess.run(['bash',str(ROOT/'benchmarks/cacheflex_flash_attention/run_8core.sh'),
                  'cache_pretouch' if v=='cache' else 'spm_konly_overlap'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 log=(path/'run.log').read_text()
 assert all(x in log for x in ('MLP_ROI begin','MLP_ROI end','exiting with last active thread context'))
 assert not re.search(r'panic:|fatal:| FAIL',log)
 record=next(l for l in log.splitlines() if l.startswith('MLP_RESULT'))
 result=dict(re.findall(r'(\w+)=([^\s]+)',record));assert result['errors']==result['nonfinite']=='0'
 stats=analyze(path)
 expected=min(mp,(t+23)//24)*4*(2*d*(f//128)+f*(d//128)) if v=='spm' else 0
 assert stats['m_spmcpRequestsIssued']==stats['m_spmcpRequestsCompleted']==expected,(name,expected,stats['m_spmcpRequestsCompleted'])
 c=configparser.ConfigParser(interpolation=None,strict=False);c.read(path/'config.ini')
 for i in range(8):
  assert c[f'system.cpu{i}.isa']['sve_vl_se']=='16'
  assert c[f'system.cpu{i}']['spmloadports']=='2'
  assert c[f'system.ruby.l1_cntrl{i}.cache.replacement_policy']['type']=='LRURP'
 return dict(variant=v,t=t,d=d,f=f,mp=mp,result=result,counters=stats,path=str(path)),c

def batch(t,d,f,mps):
 subprocess.run([sys.executable,str(HERE/'mlp_reference.py'),str(t),str(d),str(f),str(out/f'input_{t}_{d}_{f}.bin')],check=True)
 with cf.ThreadPoolExecutor(max_workers=4) as pool:
  jobs={(v,mp):pool.submit(run,v,t,d,f,mp) for mp in mps for v in ('cache','spm')}
  finished={k:fut.result() for k,fut in jobs.items()}
 for mp in mps:
  a,ca=finished['cache',mp];b,cb=finished['spm',mp]
  assert a['result']['hash']==b['result']['hash'],(t,mp,'pair hash mismatch')
  assert set(ca.sections())==set(cb.sections())
  for sec in ca.sections():
   assert set(ca[sec])==set(cb[sec])
   for key,x in ca[sec].items():
    y=cb[sec][key];x=x.replace(a['path'],'<RUN>');y=y.replace(b['path'],'<RUN>')
    if x==y or(sec=='system.cpu0.workload' and key in ('cmd','executable')):continue
    assert key=='num_spm_ways' and x=='0' and y=='4',(sec,key,x,y)
  rows.extend([a,b]);(out/'progress.json').write_text(json.dumps(rows,indent=2))
  print('PASS',t,mp,a['counters']['roi_us'],b['counters']['roi_us'],a['counters']['roi_us']/b['counters']['roi_us'],flush=True)
batch(24,128,256,[2])
for t in (64,128):batch(t,512,1536,[2,4])
summary={}
for t in (64,128):
 winners={v:min((r for r in rows if r['t']==t and r['variant']==v),key=lambda r:r['counters']['roi_us']) for v in ('cache','spm')}
 summary[t]=dict(winners=winners,speedup=winners['cache']['counters']['roi_us']/winners['spm']['counters']['roi_us'])
(out/'summary.json').write_text(json.dumps(summary,indent=2));print('COMPLETE',flush=True)
