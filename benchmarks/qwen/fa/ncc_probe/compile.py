import os,sys,subprocess,resource,json,time,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[4];H=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'results/parallel_overlap_20260930'));import broker
B=R/'results/cacheflex_flash_attention/ncc_probe_20261005/build_v1';G=R/'results/retained_protocol_restart_20261002/gem5';O=G/'build/ARM_MESI_Three_Level_SPM';GIB=2**30
os.nice(10);budget=json.loads((R/'results/parallel_budget_20260929/budget.json').read_text());os.sched_setaffinity(0,budget['cpu_set'][12:14]);os.environ['CCACHE_DISABLE']='1'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def run(cmd,label,size,cwd):
 ticket='fa46_'+label
 while True:
  with broker.locked() as s:
   if time.time()-s['heartbeat']<20 and not s['hold_new'] and broker.total(s)+size<=s['cap']:
    s['leases'][ticket]=dict(bytes=size,owner=os.getpid(),family='FA-46 build');break
  time.sleep(5)
 try:
  def limit():resource.setrlimit(resource.RLIMIT_AS,(size-GIB//4,resource.RLIM_INFINITY))
  (B/(label+'_command.json')).write_text(json.dumps(cmd,indent=2))
  with (B/(label+'.log')).open('w') as f:subprocess.run(cmd,cwd=cwd,stdout=f,stderr=subprocess.STDOUT,check=True,preexec_fn=limit)
 finally:broker.release(ticket)
cmd=['g++','-std=c++17','-O3','-pipe','-fno-strict-aliasing','-DTRACING_ON=1','-DNUMBER_BITS_PER_SET=64','-c',str(B/'pseudo_inst.cc'),'-o',str(B/'pseudo_inst.o')]
for p in [B,O,G/'src',G/'include',G/'ext/pybind11/include',G/'ext',G/'ext/nomali/include',G/'ext/libelf','/data2/rsappidi/conda-envs/cacheflex-gem5/include','/data2/rsappidi/conda-envs/cacheflex-gem5/include/python3.12']:cmd+=['-I'+str(p)]
run(cmd,'compile',2*GIB,G)
link=json.loads((R/'results/retained_protocol_restart_20261002/link_manifest.json').read_text());cmd=list(link['argv']);frozen={}
for i,x in enumerate(cmd):
 if x=='sim/pseudo_inst.o':cmd[i]=str(B/'pseudo_inst.o')
 if x=='-Wl,--thread-count=8':cmd[i]='-Wl,--thread-count=1'
 if x.endswith(('.o','.a','.so')):
  p=Path(cmd[i]);p=p if p.is_absolute() else O/p;frozen[str(p)]=sha(p)
cmd+=['-Wl,--no-keep-files-mapped','-Wl,--no-map-whole-files'];cmd[cmd.index('-o')+1]=str(B/'gem5.opt')
(B/'link_inputs.json').write_text(json.dumps(frozen,indent=2));run(cmd,'link',24*GIB,O)
for p,h in frozen.items():assert sha(Path(p))==h
(B/'ready.json').write_text(json.dumps(dict(binary_sha256=sha(B/'gem5.opt'),completed=time.time())))
print('Isolated compile/link PASS')
