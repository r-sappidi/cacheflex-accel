"""Scheduling-only handoff of FA-35; preserve five running children and all inputs."""
import sys,os,signal,time,json,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_qwen_fa_matrix_v3 as m
old=m.BASE/'experiment_v2';journal=m.HANDOFF
journal.mkdir()
s=m.read(old/'status.json');pid=s['pid']
assert s['stage']=='placement_anchors_targets'
assert len(s['completed'])==12
assert b'run_qwen_fa_matrix_v2.py' in Path(f'/proc/{pid}/cmdline').read_bytes()
frozen=m.read(old/'inputs.json')
for path,h in frozen.items():assert m.sha(path)==h,path
os.kill(pid,signal.SIGSTOP)
started=None
try:
 for _ in range(100):
  if m.process(pid)['state']=='T':break
  time.sleep(.01)
 assert m.process(pid)['state']=='T'
 s=m.read(old/'status.json');assert s['stage']=='placement_anchors_targets'
 adopted={}
 for n,row in s['active'].items():
  ident=m.process(row['pid']);assert ident and ident['ppid']==pid
  assert n.encode() in Path(f'/proc/{row["pid"]}/cmdline').read_bytes()
  adopted[n]=dict(row,identity=ident)
 m.save(journal/'original_status.json',s);m.save(journal/'original_inputs.json',frozen);m.save(journal/'adopted.json',adopted)
 m.save(journal/'plan.json',dict(max_matrix=8,old_coordinator=pid,scope='scheduling only; no child restart or frozen input change',old_coordinator_policy='SIGSTOP until all adopted child exit codes validated, then retire; never resume concurrently'))
 with m.broker.locked() as b:
  assert all(b['leases'][v['lease']]['owner']==pid for v in adopted.values())
  with (journal/'launch.log').open('w') as log:
   started=subprocess.Popen(['python3',str(m.ROOT/'scripts/run_qwen_fa_matrix_v3.py')],cwd=m.ROOT,env=m.ENV,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  for v in adopted.values():b['leases'][v['lease']]['owner']=started.pid
 m.save(journal/'launch.json',dict(pid=started.pid,old_pid=pid,adopted=len(adopted),time=time.time()))
 print('new coordinator',started.pid,'adopted',len(adopted))
except BaseException:
 if started is None:os.kill(pid,signal.SIGCONT)
 raise
