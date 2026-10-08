import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMBA_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
import sys,json,time,hashlib,resource,signal,subprocess,shutil
from pathlib import Path
R=Path(__file__).resolve().parents[4];H=Path(__file__).resolve().parent
sys.path[:0]=[str(R/'scripts'),str(R/'results/parallel_overlap_20260930')]
import broker
from parallel_budget_scheduler import owned
from run_access_timing import stats
from fa_headroom_profiles import leaves
GIB=2**30;BASE=R/'results/cacheflex_flash_attention/packet_diag_20261004';B=BASE/'build_v1';OUT=BASE/'experiment_v1'
G=R/'results/retained_protocol_restart_20261002/gem5';O=G/'build/ARM_MESI_Three_Level_SPM'
ENV={k:v for k,v in os.environ.items() if not k.startswith('CACHEFLEX_')};ENV['LD_LIBRARY_PATH']='/data2/rsappidi/conda-envs/cacheflex-gem5/lib'
def read(p):return json.loads(Path(p).read_text())
def save(p,v):
 p=Path(p);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
 return h.hexdigest()
def setup():
 os.sched_setaffinity(0,read(R/'results/parallel_budget_20260929/budget.json')['cpu_set'][:2]);os.nice(10)
 resource.setrlimit(resource.RLIMIT_AS,(2*GIB,resource.RLIM_INFINITY))
def pressure():
 m=read(R/'results/retained_protocol_restart_20261002/spm_only_resources.json')
 mem={l.split(':')[0]:int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines()}
 rss=m['user_rss_bytes']+sum(x['rss'] for x in owned().values());cap=m['ram_limit_bytes']
 return time.time()-m['updated']>30 or rss>cap-2*GIB or mem['MemAvailable']<64*GIB+max(0,cap-rss)
def acquire(ticket,gib,status):
 while True:
  with broker.locked() as b:
   if time.time()-b['heartbeat']<20 and not b['hold_new'] and not pressure() and broker.total(b)+gib*GIB<=b['cap']:
    assert ticket not in b['leases'];b['leases'][ticket]=dict(bytes=gib*GIB,owner=os.getpid(),family='FA-42',acquired=time.time());return
  status('waiting_for_reservation');time.sleep(5)
def execute(cmd,cwd,logpath,ticket,limit,status,env=ENV,tracepath=None):
 def limits():resource.setrlimit(resource.RLIMIT_AS,(limit,resource.RLIM_INFINITY))
 with Path(logpath).open('w') as log:proc=subprocess.Popen(cmd,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,preexec_fn=limits)
 start=time.monotonic();paused=False
 try:
  while proc.poll() is None:
   assert time.monotonic()-start<24*3600,'24h host limit'
   assert shutil.disk_usage(BASE).free>50*GIB,'disk reserve'
   if tracepath and Path(tracepath).exists():assert Path(tracepath).stat().st_size<2*GIB,'2GiB compressed trace cap'
   hold=pressure();info=owned().get(proc.pid)
   if info and info['vms']>limit-GIB:
    with broker.locked() as b:
     if not hold and broker.total(b)+4*GIB<=b['cap']:
      b['leases'][ticket]['bytes']+=4*GIB;limit+=4*GIB;resource.prlimit(proc.pid,resource.RLIMIT_AS,(limit,resource.RLIM_INFINITY))
     else:hold=True
   if hold!=paused:proc.send_signal(signal.SIGSTOP if hold else signal.SIGCONT);paused=hold
   status('running',pid=proc.pid,paused=paused,simulator_as=limit,host_seconds=time.monotonic()-start)
   try:proc.wait(timeout=5)
   except subprocess.TimeoutExpired:pass
  assert proc.returncode==0,(cmd[0],proc.returncode,str(logpath))
 finally:
  if proc.poll() is None:
   proc.terminate();proc.send_signal(signal.SIGCONT)
   try:proc.wait(timeout=10)
   except subprocess.TimeoutExpired:proc.kill();proc.wait()
 return time.monotonic()-start
