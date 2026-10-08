"""Bounded verification with explicit outcomes and shared host admission."""
import os,sys,json,time,subprocess,resource,hashlib,re,signal,shutil,traceback
from pathlib import Path
ROOT=Path('/data2/rsappidi/cacheflex_mc');sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'results/parallel_overlap_20260930')]
import broker
from parallel_budget_scheduler import owned
BASE=ROOT/'results/rumur_refresh_20261001';OUT=BASE/'verification_v6';MODELS=ROOT/'verification/rumur/current';GIB=2**30
TOOL=BASE/'tools/rumur';TICKET='formal_retained_source'
def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(2**20),b''):h.update(x)
 return h.hexdigest()
def main():
 OUT.mkdir(exist_ok=False)
 budget=read(ROOT/'results/parallel_budget_20260929/budget.json');os.sched_setaffinity(0,budget['cpu_set'][:2])
 state=dict(pid=os.getpid(),stage='starting',accepted=[],failed=[],pending=[],active=None)
 def status():state['updated']=time.time();save(OUT/'status.json',state)
 files=list(MODELS.glob('*.m'))+[TOOL,Path(__file__)]
 files+=list((ROOT/'gem5/src/mem/ruby/protocol').glob('MESI_Three_Level_SPM*.sm'))
 files += [ROOT/p for p in ['gem5/src/cpu/o3/lsq_unit.cc','gem5/src/mem/ruby/structures/CacheMemory.cc','gem5/src/mem/ruby/network/garnet/NetworkInterface.cc','gem5/src/mem/ruby/network/garnet/OutputUnit.cc']]
 frozen={str(p):sha(p) for p in files};save(OUT/'inputs.json',frozen)
 (OUT/'models').mkdir()
 for p in MODELS.glob('*.m'):shutil.copy2(p,OUT/'models'/p.name)
 plans=[]
 def add(name,model,changes=None,expected=None,gb=1):plans.append(dict(name=name,model=model,changes=changes or {},expected=expected,gb=gb))
 add('private_n2_s2','private_spm',dict(NCORE=2,STEPS=2))
 add('network_fixed','network_progress')
 add('network_no_retry','network_progress',dict(RETRY=0),expected='liveness')
 add('network_no_partition','network_progress',dict(PARTITION=0),expected='liveness')
 add('l3_fixed','l3_ordering')
 add('l3_no_source_raw','l3_ordering',dict(SOURCE_RAW=0),expected='finalization overtook older reduction')
 add('l3_no_pending_release','l3_ordering',dict(PENDING_RELEASE=0),expected='no early release')
 add('l3_no_sink_order','l3_ordering',dict(SINK_ORDER=0),expected='update before initialization')
 add('private_n2_s3','private_spm',dict(NCORE=2,STEPS=3),gb=1)
 add('private_no_wb_invalidation','private_spm',dict(NCORE=2,STEPS=3,WB_INVALIDATE=0),expected='SWMR|fresh',gb=1)
 add('private_no_retention_n2_s3','private_spm',dict(NCORE=2,STEPS=3,RETAIN_LLC=0),gb=1)
 add('private_n3_s3','private_spm',dict(NCORE=3,STEPS=3),gb=4)
 add('private_n2_s4','private_spm',dict(NCORE=2,STEPS=4),gb=4)
 add('private_n2_s5','private_spm',dict(NCORE=2,STEPS=5),gb=12)
 add('private_old_release_n2_s5','private_spm',dict(NCORE=2,STEPS=5,RETAIN_SOURCE=0),expected='all cores can drain',gb=12)
 save(OUT/'plan.json',plans);state['pending']=[p['name'] for p in plans];status()
 def pressure():
  rss=sum(p['rss'] for p in owned().values());mem={x.split(':')[0]:int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith(('MemTotal:','MemAvailable:'))}
  return rss>budget['ram_limit']-GIB or mem['MemAvailable']<64*GIB+max(0,budget['ram_limit']-rss)
 def reserve(gb):
  while True:
   with broker.locked() as b:
    old=b['leases'].get(TICKET,{}).get('bytes',0)
    if not b['hold_new'] and time.time()-b['heartbeat']<20 and broker.total(b)-old+gb*GIB<=b['cap'] and not pressure():
     b['leases'][TICKET]=dict(bytes=gb*GIB,owner=os.getpid(),family='formal',acquired=time.time());return
   state['stage']='waiting_for_memory_reservation';status();time.sleep(5)
 def run(cmd,log,cap,timeout=3600):
  with log.open('w') as f:
   p=subprocess.Popen(['prlimit',f'--as={cap}:','--','nice','-n','10',*map(str,cmd)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
   state['active']=dict(pid=p.pid,command=list(map(str,cmd)),log=str(log));status();start=time.monotonic();paused=False
   try:
    while p.poll() is None:
     hold=pressure()
     if hold!=paused:p.send_signal(signal.SIGSTOP if hold else signal.SIGCONT);paused=hold
     state['active']['paused']=paused;status()
     if time.monotonic()-start>timeout:
      if paused:p.send_signal(signal.SIGCONT)
      p.terminate();p.wait(timeout=10);return 124
     time.sleep(.5)
    return p.returncode
   finally:
    if p.poll() is None:p.kill();p.wait()
    state['active']=None;status()
 try:
  for job in plans:
   reserve(job['gb']);state['stage']=job['name'];state['pending'].remove(job['name']);status()
   for p,h in frozen.items():assert sha(p)==h,('changed source',p)
   d=OUT/job['name'];d.mkdir();s=(OUT/'models'/(job['model']+'.m')).read_text()
   for k,v in job['changes'].items():
    s,n=re.subn(r'('+k+r':\s*)\d+;',lambda m:m[1]+str(v)+';',s);assert n==1,(k,n)
   model=d/'model.m';model.write_text(s);cap=job['gb']*GIB-256*2**20
   rc=run([TOOL,'--threads','2','--deadlock-detection','off','--output',d/'checker.c',model],d/'generate.log',cap)
   assert rc==0,('generate',job['name'],rc)
   rc=run(['cc','-std=c11','-O1','-pthread','-mcx16',d/'checker.c','-o',d/'checker','-latomic'],d/'compile.log',cap)
   assert rc==0,('compile',job['name'],rc)
   rc=run([d/'checker'],d/'run.log',cap,timeout=24*3600)
   text=(d/'run.log').read_text();expected=job['expected']
   ok=(rc==0 and 'No error found.' in text) if not expected else (rc!=0 and re.search(expected,text,re.I) is not None and ('The following is the error trace' in text or 'liveness property' in text and 'violated:' in text))
   result=dict(**job,returncode=rc,accepted=bool(ok),outcome='expected_counterexample' if expected and ok else ('bounded_pass' if ok else 'unaccepted'),states=re.findall(r'(\d+) states, (\d+) rules fired',text),path=str(d))
   save(d/'result.json',result)
   state['accepted' if ok else 'failed'].append(result);status()
   # Preserve and report unexpected findings; independent checks still run.
   with broker.locked() as b:b['leases'][TICKET]['bytes']=GIB
  state['stage']='completed' if not state['failed'] else 'completed_with_findings';status()
 except BaseException as e:
  state['stage']='failed';state['error']=repr(e);state['traceback']=traceback.format_exc();status();raise
 finally:
  with broker.locked() as b:b['leases'].pop(TICKET,None)
if __name__=='__main__':main()
