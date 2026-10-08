"""Run on the host: cancel only retained Qwen long queue; keep broker heartbeat.

The original parent kills peer queues on nonzero child exit. Stop that parent
before cancellation and replace only its monitoring role; never resume it.
Default is read-only preflight. --execute explicitly performs the handoff.
"""
import argparse,fcntl,json,os,signal,sys,time,hashlib,traceback
from pathlib import Path
R=Path('/data2/rsappidi/cacheflex_mc');B=R/'results/retained_protocol_restart_20261002';G=2**30
sys.path[:0]=[str(R/'scripts'),str(R/'results/parallel_overlap_20260930')]
import broker
from parallel_budget_scheduler import owned,memory
A=R/'results/cancel_qwen_long_20261004'
def read(p):return json.loads(p.read_text())
def save(p,d):
 q=p.with_name(p.name+'.'+str(os.getpid())+'.tmp');q.write_text(json.dumps(d,indent=2)+'\n');q.replace(p)
def command_matches(argv,cwd,needle):
 target=Path(needle).resolve();directory=target.is_dir()
 for arg in argv:
  value=arg.split('=',1)[1] if arg.startswith('--') and '=' in arg else arg
  if '/' not in value:continue
  candidate=Path(value)
  if not candidate.is_absolute():candidate=Path(cwd)/candidate
  candidate=candidate.resolve()
  if candidate==target or (directory and target in candidate.parents):return True
 return False

def process(pid,needle):
 p=Path('/proc')/str(pid)
 assert p.stat().st_uid==os.getuid(),('not owned',pid)
 argv=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x]
 cmd=' '.join(argv);cwd=(p/'cwd').resolve()
 assert command_matches(argv,cwd,needle),('identity mismatch',pid,cmd,needle,str(cwd))
 stat=(p/'stat').read_text().rsplit(')',1)[1].split()
 assert stat[0]!='Z',('already zombie',pid)
 fd=None
 if hasattr(os,'pidfd_open') and hasattr(signal,'pidfd_send_signal'):
  try:fd=os.pidfd_open(pid)
  except OSError:pass
 return dict(pid=pid,cmd=cmd,start=stat[19],fd=fd)
def sig(p,signum):
 if p['fd'] is not None:signal.pidfd_send_signal(p['fd'],signum);return
 # Older host Python: verify UID, start time and exact argv immediately before kill.
 root=Path('/proc')/str(p['pid'])
 assert root.stat().st_uid==os.getuid(),('changed owner',p['pid'])
 stat=(root/'stat').read_text().rsplit(')',1)[1].split()
 cmd=' '.join(x.decode() for x in (root/'cmdline').read_bytes().split(b'\0') if x)
 assert stat[19]==p['start'] and cmd==p['cmd'],('process identity changed',p['pid'])
 os.kill(p['pid'],signum)
def dead(p):
 try:
  s=(Path('/proc')/str(p['pid'])/'stat').read_text().rsplit(')',1)[1].split()
  return s[0]=='Z' or s[19]!=p['start']
 except FileNotFoundError:return True
def main(execute):
 parent_state=read(B/'spm_only_status.json');long_state=read(B/'long_context_spm_only/status.json')
 parent=process(parent_state['pid'],str(B/'run_spm_only.py'))
 worker=process(long_state['pid'],str(B/'run_long_spm_only.py'))
 assert parent_state['active']['long_spm']['pid']==worker['pid']
 cfg=read(R/'results/parallel_budget_20260929/budget.json');os.sched_setaffinity(0,cfg['cpu_set'])
 assert len(cfg['cpu_set'])<=int(os.cpu_count()*.2)
 for row in long_state.get('active',{}).values():
  checked=process(row['pid'],str(B/'long_context_spm_only/runs/'))
  if checked['fd'] is not None:os.close(checked['fd'])
 print('Verified host parent',parent['pid'],'long worker',worker['pid'],flush=True)
 if not execute:return
 A.mkdir(exist_ok=True);lock=(A/'lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (A/'cancelled.json').exists(),'already cancelled; do not repeat handoff'
 save(A/'parent_before.json',parent_state);save(A/'long_before.json',long_state)
 # Stop all producers of cancellation side effects before examining/killing jobs.
 sig(parent,signal.SIGSTOP);sig(worker,signal.SIGSTOP)
 deadline=time.monotonic()+10
 for stopped in (parent,worker):
  while (Path('/proc')/str(stopped['pid'])/'stat').read_text().rsplit(')',1)[1].split()[0] not in ('T','t'):
   assert time.monotonic()<deadline,'stop did not complete';time.sleep(.05)
 long_state=read(B/'long_context_spm_only/status.json')
 sims=[]
 for row in long_state.get('active',{}).values():
  sims.append(process(row['pid'],str(B/'long_context_spm_only/runs/')))
 accepted={}
 for row in long_state['completed']:
  p=Path(row['path'])/'accepted.json';accepted[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
 audit=dict(time=time.time(),parent={k:v for k,v in parent.items() if k!='fd'},worker={k:v for k,v in worker.items() if k!='fd'},
  simulations=[{k:v for k,v in p.items() if k!='fd'} for p in sims],accepted_sha256=accepted)
 save(A/'before_signals.json',audit)
 for p in sims:sig(p,signal.SIGKILL)
 sig(worker,signal.SIGKILL)
 deadline=time.monotonic()+15
 while not all(dead(p) for p in [worker,*sims]):
  assert time.monotonic()<deadline,'processes did not terminate; reservation retained';time.sleep(.2)
 long_state['cancelled_active']=long_state.pop('active',{});long_state['active']={}
 long_state['cancelled_pending']=long_state.pop('pending',[]);long_state['pending']=[]
 long_state.update(stage='cancelled_by_user',updated=time.time(),reason='User requested kill qwen long context; no NoC/direct32K continuation.')
 save(B/'long_context_spm_only/status.json',long_state)
 with broker.locked() as s:
  audit['released_lease']=s['leases'].pop('retained_long',None)
  s['owner']=os.getpid();s['heartbeat']=time.time()
 audit['replacement_monitor']=os.getpid();save(A/'cancelled.json',audit)
 for p,h in accepted.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
 with (R/'docs/experiments/LEDGER.md').open('a') as f:
  f.write('\n**User cancellation L3-10:** host script cancelled retained long queue and current32K simulation; accepted records and partials preserved. Original parent '+str(parent['pid'])+' intentionallySTOPPED, NEVERresume; monitor '+str(os.getpid())+' now owns shared heartbeat. Released retained_long lease only. Evidence results/cancel_qwen_long_20261004/cancelled.json. Other queues untouched.\n')
 print('Qwen long queue cancelled; reservation released. Monitoring remaining queues.',flush=True)
 # Preserve original host monitor policy exactly. Runs until explicitly retired.
 while True:
  rss=sum(x['rss'] for x in owned().values());m=memory()
  hold=rss>cfg['ram_limit']-4*G or m['MemAvailable']<64*G+max(0,cfg['ram_limit']-rss)
  with broker.locked() as s:
   s['heartbeat']=time.time();s['owner']=os.getpid();s['hold_new']=hold
  save(B/'spm_only_resources.json',dict(updated=time.time(),user_rss_bytes=rss,ram_limit_bytes=cfg['ram_limit'],pressure=hold,available_bytes=m['MemAvailable']))
  save(A/'status.json',dict(pid=os.getpid(),stage='monitoring_remaining_queues',updated=time.time(),old_parent_stopped=parent['pid']))
  time.sleep(5)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');args=parser.parse_args();main(args.execute)
