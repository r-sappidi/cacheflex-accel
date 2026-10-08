"""Claim pending case directories and parallelize without restarting active work.

The original runner creates each case directory exclusively before launching it.
Delegation uses that existing exclusion: after the current case is accepted, its
next mkdir stops the serial coordinator before it can launch a duplicate. The
expected FileExistsError is retained as a scheduling handoff, never a science fail.
"""
from worker import *
import uuid,fcntl,subprocess

def main():
 def stop(signum,frame):raise KeyboardInterrupt('coordinator termination requested')
 signal.signal(signal.SIGTERM,stop)
 setup();os.sched_setaffinity(0,read(R/'results/parallel_budget_20260929/budget.json')['cpu_set'][14:16])
 ROOT.mkdir(exist_ok=False);(ROOT/'workers').mkdir();lock=(ROOT/'coordinator.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 old=read(OUT/'status.json');plan=read(OUT/'plan.json')['cases']
 assert old['stage']=='running_early_cache_on',('unexpected handoff boundary',old['stage'])
 assert len(old['active'])==1 and 'early_cache_on' in old['active']
 assert time.time()-old['updated']<30,'stale original'
 names=['early_spm_off','early_spm_on','middle_cache_off','middle_cache_on','middle_spm_off','middle_spm_on']
 assert old['pending']==['early_cache_on',*names]
 assert all(not (OUT/'runs'/n).exists() for n in names),'already claimed or active'
 for n in ['exact_t256_off','exact_t256_on','early_cache_off']:assert (OUT/'runs'/n/'accepted.json').exists()
 for p,h in read(OUT/'inputs.json').items():assert sha(p)==h,('input drift',p)
 code=validation_source();save(ROOT/'validator.json',dict(source=str(original.H/'run.py'),ast_sha256=hashlib.sha256(ast.dump(ast.parse(code)).encode()).hexdigest(),policy='Original acceptance block verbatim; scheduling and off-peer wait only'))
 frozen={str(p):sha(p) for p in H.glob('*.py')};frozen[str(ROOT/'validator.json')]=sha(ROOT/'validator.json');save(ROOT/'inputs.json',frozen)
 save(ROOT/'before.json',old);token=uuid.uuid4().hex
 state=dict(pid=os.getpid(),stage='preparing_handoff',active={},pending=old['pending'],completed=old['completed'],failed=[],updated=time.time())
 def status(stage=None):
  if stage:state['stage']=stage
  state['updated']=time.time();save(ROOT/'status.json',state)
 ticket='fa42_parallel_coordinator';acquire(ticket,2,lambda x:status(x));children={};handles=[]
 try:
  # No signals or cross-namespace PID guesses. Directory claims are atomic.
  for n in names:
   p=OUT/'runs'/n;p.mkdir(exist_ok=False);save(p/'delegated.json',dict(owner='FA42-parallel-v1',token=token,case=n,original_active='early_cache_on',time=time.time()))
  save(ROOT/'handoff.json',dict(token=token,delegated=names,preserved_active=old['active'],serial_stop='exclusive mkdir at early_spm_off',limits='shared20percentCPUandRAM'))
  cpus=read(R/'results/parallel_budget_20260929/budget.json')['cpu_set']
  for i,n in enumerate(names):
   log=(ROOT/(n+'.worker.log')).open('w');handles.append(log)
   cmd=[sys.executable,str(H/'worker.py'),n,'--cpus',','.join(map(str,cpus[2+2*i:4+2*i])),'--token',token]
   children[n]=subprocess.Popen(cmd,cwd=R,env=ENV,stdout=log,stderr=subprocess.STDOUT)
  while True:
   previous=read(OUT/'status.json');rows=list(previous['completed']);active={};failed=[]
   if previous['stage'].startswith('running_'):
    assert set(previous['active'])<= {'early_cache_on'},'original launched delegated work'
    active.update(previous['active'])
   elif previous['stage']=='failed':
    err=previous['failed'][-1]
    assert err['case']['name']=='early_spm_off' and 'FileExistsError' in err['error'],('unexpected original failure',err)
    state['serial_handoff_complete']=True
   else:assert previous['stage'] in ('running_early_cache_on',),('unexpected original state',previous['stage'])
   for n,p in children.items():
    f=ROOT/'workers'/(n+'.json');w=read(f) if f.exists() else dict(stage='starting')
    marker=OUT/'runs'/n/'accepted.json'
    if marker.exists():rows.append(read(marker))
    elif w['stage']=='failed' or (p.poll() is not None and p.returncode!=0):failed.append(dict(name=n,error=w.get('error'),exitcode=p.returncode))
    else:active[n]=dict(path=str(OUT/'runs'/n),stage=w['stage'],**w.get('active',{}))
   state.update(completed=rows,active=active,failed=failed,pending=[c['name'] for c in plan if c['name'] not in {r['name'] for r in rows}])
   status('running_parallel' if not failed else 'parallel_with_failures')
   if all(p.poll() is not None for p in children.values()) and not previous['active']:break
   time.sleep(5)
  original.report(ROOT)
  status('completed_diagnostics_pending_interpretation' if not state['failed'] else 'failed')
 except BaseException as e:
  state['error']=repr(e);status('failed');raise
 finally:
  for p in children.values():
   if p.poll() is None:
    p.terminate()
    try:p.wait(timeout=15)
    except subprocess.TimeoutExpired:p.kill();p.wait()
  for f in handles:f.close()
  broker.release(ticket)
if __name__=='__main__':main()
