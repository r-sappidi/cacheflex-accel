"""Compile one isolated object; link later under a separate memory reservation."""
import os,sys,subprocess,resource,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT/'results/parallel_overlap_20260930'));import broker
B=ROOT/'results/cacheflex_flash_attention/spm_isolation_20261004/build_v1';G=ROOT/'results/retained_protocol_restart_20261002/gem5';O=G/'build/ARM_MESI_Three_Level_SPM';ticket='fa40_compile'
size=4*2**30
with broker.locked() as b:
 assert ticket not in b['leases']
 assert broker.total(b)+size<=b['cap'],'Compile waits for reservation'
 b['leases'][ticket]=dict(bytes=size,owner=os.getpid(),family='SPM isolation build',acquired=time.time())
try:
 os.nice(10);budget=json.loads((ROOT/'results/parallel_budget_20260929/budget.json').read_text());os.sched_setaffinity(0,budget['cpu_set'][:1])
 def limit():resource.setrlimit(resource.RLIMIT_AS,(size-64*2**20,size-64*2**20))
 os.environ['CCACHE_DISABLE']='1'
 for name in ['lsq_unit']:
  cmd=['g++','-std=c++17','-O3','-pipe','-fno-strict-aliasing','-DTRACING_ON=1','-DNUMBER_BITS_PER_SET=64','-c',str(B/(name+'.cc')),'-o',str(B/(name+'.o'))]
  for p in [B,O,G/'src',G/'include',G/'ext/pybind11/include',G/'ext',G/'ext/nomali/include',G/'ext/libelf','/data2/rsappidi/conda-envs/cacheflex-gem5/include','/data2/rsappidi/conda-envs/cacheflex-gem5/include/python3.12']:cmd+=['-I'+str(p)]
  (B/(name+'.command.json')).write_text(json.dumps(cmd,indent=2))
  with (B/(name+'.compile.log')).open('w') as f:subprocess.run(cmd,cwd=G,stdout=f,stderr=subprocess.STDOUT,check=True,preexec_fn=limit)
 print('isolated pseudo_inst object compiled')
finally:broker.release(ticket)
