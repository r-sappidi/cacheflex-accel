"""Isolated serial FA-39 queue using retained simulator and shared resource broker."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMBA_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
import ast,json,time,hashlib,subprocess,resource,signal,fcntl,traceback,re,sys,shutil
from pathlib import Path
R=Path(__file__).resolve().parents[4];H=Path(__file__).resolve().parent
sys.path[:0]=[str(R/'scripts'),str(R/'benchmarks/cacheflex_flash_attention'),str(R/'results/parallel_overlap_20260930')]
import broker,numpy as np
from plan import cases,parameters,work,argv,control_path,service
from run_access_timing import stats
from fa_headroom_profiles import leaves
from test_micro_timing import check_global_guard
from export_safe_elf import export_hash
from qwen_l3_reference import reference,validate
from parallel_budget_scheduler import owned
BASE=R/'results/cacheflex_flash_attention/bottleneck_20261004';G=R/'results/cacheflex_flash_attention/ncc_probe_20261005_v2/guest_v1';B=R/'results/cacheflex_flash_attention/ncc_probe_20261005/build_v1';OUT=G.parent/(os.environ['FA46_ARM']+('_target' if os.environ.get('FA46_TARGET')=='1' else ''));GIB=2**30
PARENT=R/'results/retained_protocol_restart_20261002/fa_matrix_spm_only/runs/t256_q14_legacy_v_br16_bc256_r256'
ENV=dict(os.environ,CACHEFLEX_NCC_PROBE='1',LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib')
def read(p):return json.loads(Path(p).read_text())
def save(p,v):
 p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def pressure():
 monitor=read(R/'results/retained_protocol_restart_20261002/spm_only_resources.json')
 mem={l.split(':')[0]:int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines()}
 # Host coordinator accounts all host processes; sandbox RSS is added conservatively.
 rss=monitor['user_rss_bytes']+sum(v['rss'] for v in owned().values());cap=monitor['ram_limit_bytes']
 return time.time()-monitor['updated']>30 or rss>cap-2*GIB or mem['MemAvailable']<64*GIB+max(0,cap-rss)
def config_check(path,c,cmd):
 old=dict(leaves(read(PARENT/'config.json')));new=dict(leaves(read(path/'config.json')))
 expected=dict(old)
 # Normalize only output-root redirects, then explicitly set expected guest/argv.
 for k,v in list(expected.items()):
  if isinstance(v,str):expected[k]=v.replace(str(PARENT),str(path))
 if c['kind']=='fa':
  expected.update(parameters(c['variant'],c['spm']))
  expected['system.cpu[0].workload[0].executable']=str(G/'guest.export-safe')
  args=['kv-fa',*map(str,argv(c))]
  for i,x in enumerate(args):expected[f'system.cpu[0].workload[0].cmd[{i}]']=x
 assert new==expected,{k:[expected.get(k),new.get(k)] for k in expected.keys()|new.keys() if expected.get(k)!=new.get(k)}
 assert cmd[cmd.index('--cmd')+1]==new['system.cpu[0].workload[0].executable']
 return {k:[old.get(k),new.get(k)] for k in old.keys()|new.keys() if old.get(k)!=new.get(k)}
def command(c,path):
 cmd=read(PARENT/'command.json');cmd[0]=str(B/'gem5.opt');cmd[1]='--outdir='+str(path)
 if c['kind']=='fa':
  cmd[cmd.index('--cmd')+1]=str(G/'guest.export-safe');cmd[cmd.index('--options')+1]=' '.join(map(str,argv(c)))
  cmd[cmd.index('--l1d_num_spm_ways=6')]='--l1d_num_spm_ways='+str(6 if c['spm'] else 0)
  for i,x in enumerate(cmd):
   if x.startswith('system.cpu[0].workload[0].cmd='):cmd[i]='system.cpu[0].workload[0].cmd='+repr(['kv-fa',*map(str,argv(c))])
  for k,v in parameters(c['variant'],c['spm']).items():
   found=[i for i,x in enumerate(cmd) if x.startswith(k+'=')]
   if found:
    for i in found:cmd[i]=k+'='+repr(v)
   else:cmd+=['-P',k+'='+repr(v)]
 return cmd
def main():
 def stop(signum,frame):raise KeyboardInterrupt('termination requested')
 signal.signal(signal.SIGTERM,stop)
 OUT.mkdir(parents=True,exist_ok=False);(OUT/'runs').mkdir();(OUT/'references').mkdir()
 lock=(OUT/'coordinator.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 budget=read(R/'results/parallel_budget_20260929/budget.json');os.sched_setaffinity(0,budget['cpu_set'][int(os.environ['FA46_CPU_OFFSET']):int(os.environ['FA46_CPU_OFFSET'])+2]);os.nice(10)
 resource.setrlimit(resource.RLIMIT_AS,(2*GIB,resource.RLIM_INFINITY))
 plan=cases();state=dict(pid=os.getpid(),stage='preparing',active={},pending=[c['name'] for c in plan],completed=[],failed=[])
 def status():state['updated']=time.time();save(OUT/'status.json',state)
 status();save(OUT/'plan.json',dict(cases=plan,max_simulators=1,attribution='paired fractional latency reduction; SPM gain minus cache gain; never subtract overlapping counters',scope='new guest both backends, same shapes/math; only named resource overrides'))
 frozen=read(G/'frozen.json');frozen.update(read(B/'link_inputs.json'));frozen[str(B/'gem5.opt')]=sha(B/'gem5.opt');frozen[str(B/'hook.hh')]=sha(B/'hook.hh');frozen.update(read(R/'results/retained_protocol_restart_20261002/build/frozen.json'))
 for f in [*H.glob('*.py'),*PARENT.glob('*.json'),Path(read(PARENT/'command.json')[0]),Path(read(PARENT/'command.json')[2])]:
  if f.is_file():frozen[str(f)]=sha(f)
 for mod in (broker,sys.modules['run_access_timing'],sys.modules['fa_headroom_profiles'],sys.modules['qwen_l3_reference'],sys.modules['fa_independent_reference'],sys.modules['export_safe_elf'],sys.modules['test_micro_timing'],sys.modules['parallel_budget_scheduler']):
  f=Path(mod.__file__);frozen[str(f)]=sha(f)
 frozen[str(PARENT/'output.fp16')]=sha(PARENT/'output.fp16');frozen[str(PARENT/'stats.txt')]=sha(PARENT/'stats.txt')
 for c in plan:
  control=control_path(c)
  if control is None:continue
  for name in ('accepted.json','output.fp16','stats.txt','command.json','config.json'):
   frozen[str(control/name)]=sha(control/name)
  assert sha(control/'output.fp16')==read(control/'accepted.json')['output_sha256']
 save(OUT/'inputs.json',frozen)
 def unchanged():
  for p,h in frozen.items():assert sha(p)==h,('frozen input drift',p)
 def acquire(ticket):
  while True:
   state['stage']='waiting_for_reservation';status()
   with broker.locked() as s:
    if time.time()-s['heartbeat']<20 and not s['hold_new'] and not pressure() and broker.total(s)+10*GIB<=s['cap']:
     assert ticket not in s['leases'];s['leases'][ticket]=dict(bytes=10*GIB,owner=os.getpid(),family='FA-46');return
   time.sleep(5)
 try:
  for c in plan:
   ticket='fa46_'+os.environ['FA46_ARM']+'_'+c['name'];acquire(ticket);path=OUT/'runs'/c['name'];path.mkdir();proc=None
   try:
    unchanged();cmd=command(c,path);save(path/'command.json',cmd);limit=8*GIB
    def child():resource.setrlimit(resource.RLIMIT_AS,(limit,resource.RLIM_INFINITY))
    replay=OUT/'disabled_replay'
    source=replay/'runs'/c['name']
    reused=source.is_dir()
    if reused:
     receipt=read(replay/'replay_verified.json')['raw_sha256']
     for rel,digest in receipt.items():
      if rel.startswith('runs/'+c['name']+'/'):assert sha(replay/rel)==digest,('replay drift',rel)
     assert read(source/'command.json')==cmd,'replay command changed'
     shutil.copytree(source,path,dirs_exist_ok=True)
     save(path/'replay_provenance.json',dict(source=str(source),verified_receipt=sha(replay/'replay_verified.json')))
     code=0
    else:
     with (path/'run.log').open('w') as log:proc=subprocess.Popen(cmd,cwd=R,env=ENV,stdout=log,stderr=subprocess.STDOUT,preexec_fn=child)
     state['stage']='running_'+c['name'];state['active']={c['name']:dict(pid=proc.pid,path=str(path),paused=False,simulator_as=limit)};status();start=time.monotonic();paused=False
     while proc.poll() is None:
      assert time.monotonic()-start<96*3600,'case timeout'
      hold=pressure();info=owned().get(proc.pid)
      if info and info['vms']>=limit-GIB:
       with broker.locked() as s:
        if not hold and broker.total(s)+4*GIB<=s['cap']:
         s['leases'][ticket]['bytes']+=4*GIB;limit+=4*GIB;resource.prlimit(proc.pid,resource.RLIMIT_AS,(limit,resource.RLIM_INFINITY))
        else:hold=True
      if hold!=paused:proc.send_signal(signal.SIGSTOP if hold else signal.SIGCONT);paused=hold
      state['active'][c['name']].update(paused=paused,simulator_as=limit);status()
      try:proc.wait(timeout=5)
      except subprocess.TimeoutExpired:pass
     code=proc.returncode
    log=(path/'run.log').read_text();assert code==0 and 'because exiting with last active thread context' in log,log[-3000:]
    assert not re.search(r'panic:|fatal:|\bFAIL\b',log),log[-3000:]
    delta=config_check(path,c,cmd);blocks=stats(path);s=blocks[0]
    for block in blocks:
     check_global_guard(block)
     for i in range(8):
      for domain in ('regular','spm'):
       for kind,cap in [('Load',32),('Store',48)]:assert block[f'system.cpu{i}.{domain}Lsq0.boundedMax{kind}Occupancy']<=cap
    if c['kind']=='exact':
     assert blocks==stats(PARENT),'default simulator statistics changed'
     assert sha(path/'output.fp16')==sha(PARENT/'output.fp16')
     row=dict(name=c['name'],case=c,path=str(path),outcome='exact_default_control',roi_us=s['simTicks']/1e6,config_delta=delta)
    else:
     ref=OUT/'references'/f't{c["t"]}.npy'
     if not ref.exists():np.save(ref,reference(c['t'],256,14,2,True))
     frozen[str(ref)]=sha(ref);save(OUT/'inputs.json',frozen)
     export_hash(path/'output.fp16',log);arithmetic=validate(path/'output.fp16',np.load(ref),c['t'],14,2,True,dense=c['t']<=256)
     tasks,updates,expected=work(c['t']);actual={int(i):(int(t),int(u)) for i,t,u in re.findall(r'GQA_WORK core=(\d+) tasks=(\d+) row_updates=(\d+)',log)}
     assert actual=={i:(tasks[i],updates[i]) for i in range(8)},actual
     copies=sum(v for k,v in s.items() if k.endswith('.m_spmcpRequestsIssued'))
     assert copies==(expected if c['spm'] else 0)
     assert copies==sum(v for k,v in s.items() if k.endswith('.m_spmcpRequestsCompleted'))
     assert sum(v for k,v in s.items() if k.endswith(('.reductionAccepted','.l3ReductionStarts')))==0
     assert f'BOTTLENECK stage_simple={int(c["variant"]=="stage")} packed_producer={int(c["variant"]=="packed")}' in log
     digest=sha(path/'output.fp16')
     control=control_path(c)
     if control is not None:assert digest==read(control/'accepted.json')['output_sha256'],'accepted control output differs'
     assert f'NCC_CONFIG mode={c["mode"]}' in log
     if c['mode']:
      counts={int(i):(int(pv),int(kp),int(pp),int(cy)) for i,pv,kp,pp,cy in re.findall(r'NCC_COUNTS core=(\d+) pv=(\d+) kpack=(\d+) ppack=(\d+) service_cycles=(\d+)',log)}
      expected_counts={i:(updates[i]//16 if 3<=c['mode']<=8 else 0,(2*c['t']//128+7-i)//8 if c['mode']==1 else 0,updates[i]//16 if c['mode']==2 else 0,(updates[i]//16)*service(c['mode'])) for i in range(8)}
      assert counts==expected_counts,(counts,expected_counts)
     for prev in state['completed']:
      if prev['case']['kind']=='fa' and prev['case']['t']==c['t']:assert prev['output_sha256']==digest,'matched output bits changed'
     row=dict(name=c['name'],case=c,path=str(path),outcome='full_fa_pass',roi_us=s['simTicks']/1e6,arithmetic=arithmetic,copies=copies,output_sha256=digest,config_delta=delta)
    unchanged();save(path/'accepted.json',row);state['completed'].append(row);state['pending'].remove(c['name']);state['active']={};status()
    save(OUT/'summary.json',state['completed'])
   finally:
    if proc is not None and proc.poll() is None:
     proc.terminate();proc.send_signal(signal.SIGCONT)
     try:proc.wait(timeout=10)
     except subprocess.TimeoutExpired:proc.kill();proc.wait()
    broker.release(ticket)
  state['stage']='complete';status()
 except BaseException as e:
  state['stage']='failed';state['active']={};state['failed'].append(dict(error=repr(e),traceback=traceback.format_exc()));status();raise
if __name__=='__main__':main()
