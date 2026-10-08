"""Production ACL warmed contender at model-derived shapes. New output only."""
from pathlib import Path
import concurrent.futures as cf
import json
import os
import re
import shutil
import subprocess
import sys
from run_mlp_acl import config_audit
from sweep_mlp_study import epochs
from validate_mlp_acl import validate
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
out=Path(sys.argv[1]).resolve();build=Path(sys.argv[2]).resolve();cases=json.loads(Path(sys.argv[3]).read_text());out.mkdir()
shutil.copy2(build/'acl',out/'acl')
old=ROOT/'results/ai_workloads/mlp_clean_20260909/runs'
shutil.copy2(Path(os.environ.get('MLP_GEM5_BIN',str(old/'gem5.opt'))),out/'gem5.opt')
def run(case):
 t,d,f=case;name=f'acl_t{t}_d{d}_f{f}';path=out/name;inp=out/(name+'.bin')
 subprocess.run([sys.executable,str(HERE/'mlp_reference.py'),str(t),str(d),str(f),str(inp)],check=True)
 env=dict(PATH=os.environ['PATH'],HOME=os.environ['HOME'],CACHEFLEX_FLASH_VL='16',
          CACHEFLEX_FLASH_MATCHED_LRU='1',CACHEFLEX_FLASH_SPM_WAYS='0',
          CACHEFLEX_FLASH_OPTIONS=f'{t} {d} {f} warm 2 3',CACHEFLEX_FLASH_READFILE=str(inp),
          CACHEFLEX_FLASH_BIN=str(out/'acl'),CACHEFLEX_FLASH_OUT=str(path),CACHEFLEX_GEM5_BIN=str(out/'gem5.opt'))
 with (out/(name+'.log')).open('w') as log:
  subprocess.run(['bash',str(HERE.parent/'cacheflex_flash_attention/run_8core.sh'),'cache_pretouch'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 log=(path/'run.log').read_text()
 assert 'exiting with last active thread context' in log and log.count('MLP_ROI end')==3
 assert not re.search(r'panic:|fatal:|Simulated exit code not 0',log)
 config_audit(path,old/'cache_t64_d512_f1536_mp2')
 numeric=validate(path,inp,t,d,f);timings=epochs(path,3)
 assert all(e['copies_issued']==e['copies_completed']==0 for e in timings)
 return dict(t=t,d=d,f=f,epochs=timings,numerical=numeric,path=str(path))
rows=[]
with cf.ThreadPoolExecutor(max_workers=4) as pool:
 for row in pool.map(run,cases):
  rows.append(row);(out/'progress.json').write_text(json.dumps(rows,indent=2));print('PASS',row['t'],row['d'],row['epochs'],flush=True)
(out/'summary.json').write_text(json.dumps(rows,indent=2))
