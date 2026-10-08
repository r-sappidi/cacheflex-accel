"""Gate and restart current FA/long-context queues on one retained-source build."""
import os,sys,json,time,subprocess,signal,hashlib,re,traceback,resource
from pathlib import Path
R=Path('/data2/rsappidi/cacheflex_mc');B=R/'results/retained_protocol_restart_20261002';G=2**30
sys.path[:0]=[str(R/'scripts'),str(R/'results/parallel_overlap_20260930')]
import broker
from parallel_budget_scheduler import owned,memory
from run_access_timing import stats
from fa_headroom_profiles import leaves
from test_micro_timing import check_global_guard

def read(p):return json.loads(p.read_text())
def save(p,d):
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(d,indent=2)+'\n');q.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 cfg=read(R/'results/parallel_budget_20260929/budget.json');os.sched_setaffinity(0,cfg['cpu_set']);children={};active=None
 state=dict(pid=os.getpid(),stage='waiting_for_build',completed=[],active={},updated=time.time())
 def status(stage=None):
  if stage:state['stage']=stage
  state['updated']=time.time();save(B/'status.json',state)
 def heartbeat():
  rss=sum(x['rss'] for x in owned().values());m=memory();hold=rss>cfg['ram_limit']-4*G or m['MemAvailable']<64*G+max(0,cfg['ram_limit']-rss)
  with broker.locked() as b:
   b['heartbeat']=time.time();b['owner']=os.getpid();b['hold_new']=hold or state['stage']=='waiting_for_build'
  save(B/'resources.json',dict(updated=time.time(),user_rss_bytes=rss,ram_limit_bytes=cfg['ram_limit'],pressure=hold,available_bytes=m['MemAvailable']))
  return hold
 status()
 try:
  while True:
   d=read(B/'build_status.json');assert d['stage']!='failed',d
   if d['stage']=='built_requires_runtime_validation':break
   heartbeat();time.sleep(5)
  formal=read(R/'results/rumur_refresh_20261001/verification_v6/status.json');assert formal['stage']=='completed' and not formal['failed'] and len(formal['accepted'])==15
  gate=B/'gates';gate.mkdir(exist_ok=True);(gate/'runs').mkdir(exist_ok=True);state['stage']='runtime_gates';status()
  roots=[(R/'results/cacheflex_flash_attention/spm_set_spread_20260927/experiment_retry_v2/runs',['spread_release_getx','spread_roundtrip','spread_roundtrip_raw','wide_0','wide_512']),
   (R/'results/cacheflex_l3_spm/initial_20260928/experiment_v7/runs',['smoke_normal','smoke_small_queues']),
   (R/'results/cacheflex_l3_spm/direct_link_20260928/experiment_v2/runs',['directed_fp32','directed_both','directed_init','directed_mixed','directed_one_entry_slow','directed_sync'])]
  if (gate/'complete.json').exists():
   assert read(gate/'complete.json')['cases']==13
   for path,digest in read(gate/'inputs.json').items():assert sha(Path(path))==digest,('accepted gate input changed',path)
   state['completed']=read(gate/'status.json')['completed'];roots=[]
  manifest=read(gate/'inputs.json') if (gate/'complete.json').exists() else read(B/'build/frozen.json');save(gate/'inputs.json',manifest)
  for base,names in roots:
   for name in names:
    source=base/name;out=gate/'runs'/name;out.mkdir()
    cmd=read(source/'command.json');cmd[0]=str(B/'build/gem5.opt');cmd[1]='--outdir='+str(out);cmd[2]=str(B/'build/configs/deprecated/example/l3_direct_reduction_se.py')
    old_fields=dict(leaves(read(source/'config.json')))
    for i in range(8):
     key=f'system.ruby.l2_cntrl{i}.l3ReductionQueue.buffer_size'
     cmd+=['-P',key+'='+str(old_fields.get(key,4))]
    for p in [source/'command.json',source/'config.json',(source/'accepted.json' if (source/'accepted.json').exists() else source/'validated.json'),source/'run.log',Path(cmd[cmd.index('--cmd')+1])]:manifest[str(p)]=sha(p)
    save(gate/'inputs.json',manifest);save(out/'command.json',cmd)
    with broker.locked() as b:
     assert broker.total(b)+12*G<=b['cap'];b['leases']['retained_gate']=dict(bytes=12*G,owner=os.getpid(),family='retained gates')
    with (out/'run.log').open('w') as f:
     active=subprocess.Popen(['prlimit','--as='+str(10*G)+':','--','nice','-n','10',*cmd],cwd=R,env=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib'),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
     state['active']={name:dict(pid=active.pid)};status();save(gate/'status.json',state);start=time.time()
     while active.poll() is None:
      assert time.time()-start<7200,'directed runtime timeout';hold=heartbeat();ps=owned().get(active.pid)
      if ps:
       if hold and ps['state']!='T':active.send_signal(signal.SIGSTOP)
       elif not hold and ps['state']=='T':active.send_signal(signal.SIGCONT)
      time.sleep(3)
    text=(out/'run.log').read_text();assert active.returncode==0 and 'because exiting with last active thread context' in text,text[-2000:]
    assert not re.search(r'panic:|fatal:|\bFAIL\b|Simulated exit code not 0',text),text[-2000:]
    markers=lambda s:[line for line in s.splitlines() if re.search(r'\bPASS\b',line)]
    expected=markers((source/'run.log').read_text());assert expected and markers(text)==expected,(expected,markers(text))
    fields=lambda p:dict(leaves(json.loads((p/'config.json').read_text().replace(str(p),'<OUT>'))))
    old,new=fields(source),fields(out);delta={k:[old.get(k),new.get(k)] for k in old.keys()|new.keys() if old.get(k)!=new.get(k)}
    # Older bridge gates predate optional L3 queues; preserve all original
    # fields and audit the newly required, finite, disabled engine structures.
    changed={k:v for k,v in delta.items() if k in old and k in new}
    removed={k:v for k,v in delta.items() if k not in new}
    assert not changed and not removed,(changed,removed)
    for i in range(8):
     prefix=f'system.ruby.l2_cntrl{i}.'
     if prefix+'l3_reduction_enabled' not in old:assert not new[prefix+'l3_reduction_enabled']
     if prefix+'l3_spm_enabled' not in old:assert not new[prefix+'l3_spm_enabled']
     assert new[prefix+'l3ReductionQueue.buffer_size']<=4

    for ep in stats(out):
     check_global_guard(ep)
     for k,v in ep.items():
      if k.endswith('boundedMaxLoadOccupancy'):assert v<=32,(k,v)
      if k.endswith('boundedMaxStoreOccupancy'):assert v<=48,(k,v)
    for p,h in manifest.items():assert sha(Path(p))==h,('frozen input changed',p)
    row=dict(name=name,source=str(source),pass_markers=expected,config_delta=delta,accepted=True);save(out/'accepted.json',row);state['completed'].append(row);state['active']={};status();save(gate/'status.json',state)
    with broker.locked() as b:b['leases'].pop('retained_gate',None)
    active=None
  save(gate/'complete.json',dict(stage='completed_validated',cases=len(state['completed'])));save(gate/'status.json',dict(state,stage='completed_validated'))
  for name,script in [('matrix','run_matrix.py'),('long','run_long.py')]:
   f=(B/(name+'_coordinator.log')).open('w');p=subprocess.Popen(['python3',str(B/script)],cwd=R,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'),stdout=f,stderr=subprocess.STDOUT,start_new_session=True);children[name]=p
  state['active']={name:dict(pid=p.pid) for name,p in children.items()};status('experiments_running')
  while children:
   heartbeat()
   for name,p in list(children.items()):
    if p.poll() is None:continue
    assert p.returncode==0,(name,p.returncode)
    if name=='long':
     with broker.locked() as b:b['leases'].pop('retained_long',None)
    del children[name];state['active'].pop(name,None);status()
   time.sleep(5)
  status('completed_validated')
 except BaseException as e:
  state['error']=repr(e);state['traceback']=traceback.format_exc();status('failed')
  for p in [active,*children.values()]:
   if p and p.poll() is None:
    os.killpg(p.pid,signal.SIGTERM);os.killpg(p.pid,signal.SIGCONT)
  with broker.locked() as b:b['hold_new']=True
  raise
if __name__=='__main__':main()
