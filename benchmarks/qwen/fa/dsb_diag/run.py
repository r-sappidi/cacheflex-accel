"""Observer-only DSB replay: exact old stats/output plus full saved oracle checks."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMBA_NUM_THREADS='1',CCACHE_DISABLE='1')
import json,sys,time,hashlib,subprocess,resource,signal,traceback,fcntl
from pathlib import Path
R=Path(__file__).resolve().parents[4];H=Path(__file__).resolve().parent
sys.path[:0]=[str(R/'scripts'),str(R/'benchmarks/cacheflex_flash_attention'),str(R/'results/parallel_overlap_20260930')]
import broker,numpy as np
from run_access_timing import stats
from qwen_l3_reference import validate
from export_safe_elf import export_hash
from parallel_budget_scheduler import owned
BASE=R/'results/cacheflex_flash_attention/dsb_diag_20261003';B=BASE/'build_v1';OUT=BASE/'experiment_v1'
PRIOR=R/'results/cacheflex_flash_attention/v_only_ring_20261003/experiment_v1'
G=2**30
ENV=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib')
def read(p):return json.loads(Path(p).read_text())
def save(p,v):
 p=Path(p);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
 return h.hexdigest()
def main():
 OUT.mkdir(parents=True);(OUT/'runs').mkdir();lock=(OUT/'lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 cfg=read(R/'results/parallel_budget_20260929/budget.json');os.sched_setaffinity(0,cfg['cpu_set'][:2]);os.nice(10)
 resource.setrlimit(resource.RLIMIT_AS,(2*G,resource.RLIM_INFINITY))
 state=dict(pid=os.getpid(),namespace_pids=True,stage='preparing',active={},completed=[],failed=[],pending=[])
 def status(stage=None):
  if stage:state['stage']=stage
  state['updated']=time.time();save(OUT/'status.json',state)
 def pressure():
  ext=read(R/'results/retained_protocol_restart_20261002/spm_only_resources.json');rss=ext['user_rss_bytes']+sum(p['rss'] for p in owned().values())
  avail=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
  return rss>cfg['ram_limit']-2*G or avail<64*G+max(0,cfg['ram_limit']-rss)
 def acquire(ticket,size):
  while True:
   with broker.locked() as d:
    if time.time()-d['heartbeat']<20 and not d['hold_new'] and not pressure() and broker.total(d)+size<=d['cap']:
     assert ticket not in d['leases'];d['leases'][ticket]=dict(owner=os.getpid(),bytes=size,family='FA-37 DSB diagnostics',acquired=time.time());return
   status('waiting_reservation');time.sleep(5)
 frozen=read(PRIOR/'inputs.json');frozen.update(read(B/'source_manifest.json'))
 for p in list(H.glob('*'))+list(B.glob('*.o')):
  if p.is_file():frozen[str(p)]=sha(p)
 # Evidence used for numerical revalidation remains immutable.
 for p in PRIOR.rglob('*'):
  if p.is_file() and p.name in ('accepted.json','command.json','config.json','stats.txt','output.fp16') or p.is_file() and p.suffix=='.npy':frozen[str(p)]=sha(p)
 def unchanged():
  for p,h in frozen.items():assert sha(p)==h,('frozen input drift',p)
 status();save(OUT/'inputs.json',frozen)
 proc=None;ticket=None
 try:
  assert read(PRIOR/'status.json')['stage']=='complete'
  link=read(R/'results/retained_protocol_restart_20261002/link_manifest.json');cwd=R/'results/retained_protocol_restart_20261002/gem5/build/ARM_MESI_Three_Level_SPM';cmd=list(link['argv'])
  replacement={f'cpu/o3/{n}.o':B/(n+'.o') for n in ('commit','lsq','rename','mem_dep_unit')}
  replacement['sim/pseudo_inst.o']=R/'results/cacheflex_flash_attention/optimistic_ring_20261001/build_retained_v1/pseudo_inst.o'
  for i,x in enumerate(cmd):
   if x in replacement:cmd[i]=str(replacement[x])
   if x=='-Wl,--thread-count=8':cmd[i]='-Wl,--thread-count=1'
   if x.endswith(('.o','.a','.so')):
    p=Path(cmd[i]);p=p if p.is_absolute() else cwd/p;frozen[str(p)]=sha(p)
  cmd+=['-Wl,--no-keep-files-mapped','-Wl,--no-map-whole-files'];cmd[cmd.index('-o')+1]=str(B/'gem5.opt');save(B/'link_command.json',cmd);save(OUT/'inputs.json',frozen)
  ticket='dsb_link';acquire(ticket,24*G);status('linking');unchanged()
  with (B/'link.log').open('w') as f:subprocess.run(['prlimit',f'--as={23*G}:','--',*cmd],cwd=cwd,env=dict(link['environment'],CCACHE_DISABLE='1'),stdout=f,stderr=subprocess.STDOUT,check=True)
  broker.release(ticket);ticket=None;frozen[str(B/'gem5.opt')]=sha(B/'gem5.opt');save(OUT/'inputs.json',frozen)
  save(B/'status.json',dict(stage='built_requires_validation',sha256=frozen[str(B/'gem5.opt')]))
  plans=[('off_gate_t1024_m1','gate_t1024_m1',False),('off_gate_t1024_m17','gate_t1024_m17',False)]
  plans += [('on_'+r['name'],r['name'],True) for r in read(PRIOR/'status.json')['completed']]
  state['pending']=[n for n,_,_ in plans];save(OUT/'plan.json',plans)
  for name,source_name,enabled in plans:
   source=PRIOR/'runs'/source_name;old=read(source/'accepted.json');path=OUT/'runs'/name;path.mkdir()
   ticket='dsb_'+name;acquire(ticket,8*G);status(name);unchanged()
   cmd=read(source/'command.json');cmd[0]=str(B/'gem5.opt');cmd[1]='--outdir='+str(path);save(path/'command.json',cmd)
   env=dict(ENV);env.pop('CACHEFLEX_DSB_DIAG',None)
   if enabled:env['CACHEFLEX_DSB_DIAG']='1'
   if old['case']['mode']>=16:env['CACHEFLEX_OPTIMISTIC_RING']='1'
   cap=6*G
   with (path/'run.log').open('w') as f:
    proc=subprocess.Popen(['prlimit',f'--as={cap}:','--',*cmd],cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT)
    start=time.time();paused=False
    while proc.poll() is None:
     assert time.time()-start<96*3600
     hold=pressure();info=owned().get(proc.pid)
     if info and info['vms']>=cap-G:
      with broker.locked() as d:
       if broker.total(d)+4*G<=d['cap'] and not hold:
        d['leases'][ticket]['bytes']+=4*G;cap+=4*G;resource.prlimit(proc.pid,resource.RLIMIT_AS,(cap,resource.RLIM_INFINITY))
       else:hold=True
     if hold!=paused:proc.send_signal(signal.SIGSTOP if hold else signal.SIGCONT);paused=hold
     state['active']={name:dict(pid=proc.pid,path=str(path),paused=paused,simulator_as=cap)};status(name)
     try:proc.wait(timeout=5)
     except subprocess.TimeoutExpired:pass
   assert proc.returncode==0;log=(path/'run.log').read_text();assert 'because exiting with last active thread context' in log
   proc=None;state['active']={};status('validating_'+name)
   # Full equality of all pre-existing numeric non-host stats, every epoch.
   old_stats=stats(source);new_stats=stats(path);assert len(old_stats)==len(new_stats)
   for x,y in zip(old_stats,new_stats):
    clean={k:v for k,v in y.items() if '.dsbDiag.' not in k};assert x==clean,('observer changed execution',name,[(k,v,clean.get(k)) for k,v in x.items() if clean.get(k)!=v][:8])
    for core in range(8):
     prefix=f'system.cpu{core}.dsbDiag.'
     masks=sum(v for k,v in y.items() if k.startswith(prefix+'waitMask::') and not k.endswith('::total'))
     assert masks==y[prefix+'storeWaitSamples']
    if not enabled:assert all(v==0 for k,v in y.items() if '.dsbDiag.' in k)
   assert sha(path/'output.fp16')==sha(source/'output.fp16');export_hash(path/'output.fp16',log)
   def normalized(p):return json.loads((p/'config.json').read_text().replace(str(p),'<OUT>'))
   assert normalized(path)==normalized(source),'hardware/guest configuration drift'
   c=old['case'];ref=PRIOR/'references'/f't{c["t"]}_bc{c["bc"]}_q{c["hq"]}_{"ring" if c["mode"]>=16 else "ordinary"}.npy'
   arithmetic=validate(path/'output.fp16',np.load(ref),c['t'],c['hq'],2,True,dense=c['t']<=1024)
   ds={k:v for k,v in new_stats[0].items() if '.dsbDiag.' in k}
   if enabled:assert sum(v for k,v in ds.items() if k.endswith('.retired'))>0
   row=dict(name=name,case=c,path=str(path),roi_us=old['roi_us'],source=str(source),observer_enabled=enabled,exact_all_old_stats=True,arithmetic=arithmetic,output_sha256=sha(path/'output.fp16'),counters=ds)
   unchanged();save(path/'accepted.json',row);state['completed'].append(row);state['pending'].remove(name);status()
   broker.release(ticket);ticket=None
   save(OUT/'counters.json',[dict(name=r['name'],roi_us=r['roi_us'],counters=r['counters']) for r in state['completed']])
  status('complete');save(OUT/'complete.json',dict(cases=len(state['completed'])))
 except BaseException as e:
  if proc and proc.poll() is None:proc.kill();proc.wait()
  state['active']={};state['failed'].append(dict(error=repr(e),traceback=traceback.format_exc()));status('failed');raise
 finally:
  if ticket:broker.release(ticket)
if __name__=='__main__':main()
