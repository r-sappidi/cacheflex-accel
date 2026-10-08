"""Reserve link memory, produce isolated ELF, then keep foreground queue alive."""
import os,json,time,subprocess,resource
from pathlib import Path
import run
R=run.R;B=run.B;broker=run.broker;ticket='fa40_link'
budget=run.read(R/'results/parallel_budget_20260929/budget.json');os.sched_setaffinity(0,budget['cpu_set'][:2]);os.nice(10)
while True:
 with broker.locked() as b:
  if time.time()-b['heartbeat']<20 and not b['hold_new'] and not run.pressure() and broker.total(b)+24*run.GIB<=b['cap']:
   assert ticket not in b['leases'];b['leases'][ticket]=dict(bytes=24*run.GIB,owner=os.getpid(),family='FA-40 link');break
 run.save(B/'status.json',dict(stage='waiting_link_reservation',updated=time.time()));time.sleep(5)
try:
 link=run.read(R/'results/retained_protocol_restart_20261002/link_manifest.json')
 cwd=R/'results/retained_protocol_restart_20261002/gem5/build/ARM_MESI_Three_Level_SPM';cmd=list(link['argv']);inputs={}
 for i,x in enumerate(cmd):
  if x=='cpu/o3/lsq_unit.o':cmd[i]=str(B/'lsq_unit.o')
  if x=='-Wl,--thread-count=8':cmd[i]='-Wl,--thread-count=1'
  if x.endswith(('.o','.a','.so')):
   p=Path(cmd[i]);p=p if p.is_absolute() else cwd/p;inputs[str(p)]=run.sha(p)
 assert str(B/'lsq_unit.o') in cmd
 cmd[cmd.index('-o')+1]=str(B/'gem5.opt');cmd+=['-Wl,--no-keep-files-mapped','-Wl,--no-map-whole-files']
 run.save(B/'link_inputs.json',inputs);run.save(B/'link_command.json',cmd);run.save(B/'status.json',dict(stage='linking',updated=time.time()))
 def limits():resource.setrlimit(resource.RLIMIT_AS,(23*run.GIB,resource.RLIM_INFINITY))
 with (B/'link.log').open('w') as out:subprocess.run(cmd,cwd=cwd,env=dict(link['environment'],CCACHE_DISABLE='1'),stdout=out,stderr=subprocess.STDOUT,check=True,preexec_fn=limits)
 run.save(B/'status.json',dict(stage='built_runtime_pending',updated=time.time(),sha256=run.sha(B/'gem5.opt')))
finally:broker.release(ticket)
run.main()
