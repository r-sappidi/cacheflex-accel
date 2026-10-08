"""Isolated retained-source build; never overwrite frozen main build objects."""
import os,json,shutil,subprocess,time,hashlib,sys,resource
from pathlib import Path
R=Path('/data2/rsappidi/cacheflex_mc');B=R/'results/retained_protocol_restart_20261002';W=B/'gem5';G=2**30
sys.path[:0]=[str(R/'results/parallel_overlap_20260930'),str(R/'scripts')]
import broker
from parallel_budget_scheduler import owned,memory

def save(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 cfg=json.loads((R/'results/parallel_budget_20260929/budget.json').read_text());os.sched_setaffinity(0,cfg['cpu_set'][:8]);os.nice(10)
 def status(stage,**kw):save(B/'build_status.json',dict(stage=stage,pid=os.getpid(),updated=time.time(),**kw))
 with broker.locked() as b:
  assert broker.total(b)+32*G<=b['cap'];b['leases']['retained_build']=dict(bytes=32*G,owner=os.getpid(),family='retained build');b['heartbeat']=time.time();b['hold_new']=True
 try:
  status('copying_isolated_tree')
  shutil.copytree(R/'gem5',W,ignore=shutil.ignore_patterns('build','.git','__pycache__'),symlinks=True)
  (W/'build').mkdir();subprocess.run(['cp','-a','--reflink=auto',str(R/'gem5/build/ARM_MESI_Three_Level_SPM'),str(W/'build')],check=True)
  files=[p for p in (W/'src').rglob('*') if p.is_file()];save(B/'source_manifest.json',{str(p.relative_to(W)):sha(p) for p in files})
  env=dict(os.environ,LD_LIBRARY_PATH='/data2/rsappidi/conda-envs/cacheflex-gem5/lib',PYTHON_CONFIG='/data2/rsappidi/conda-envs/cacheflex-gem5/bin/python3-config',PROTOC='/usr/bin/protoc',CCFLAGS_EXTRA='-gdwarf-2',GEM5PY_CCFLAGS_EXTRA='-gdwarf-2',LINKFLAGS_EXTRA='-L/usr/lib64',CCACHE_DISABLE='1',CACHEFLEX_LINK_MANIFEST=str(B/'link_manifest.json'))
  cmd=['/data2/rsappidi/conda-envs/cacheflex-gem5/bin/scons','build/ARM_MESI_Three_Level_SPM/gem5.opt','-j8','--linker=gold']
  with (B/'build.log').open('w') as f:
   p=subprocess.Popen(cmd,cwd=W,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   status('building',child=p.pid);paused=False
   while p.poll() is None:
    m=memory();rss=sum(x['rss'] for x in owned().values());hold=rss>cfg['ram_limit']-4*G or m['MemAvailable']<64*G+max(0,cfg['ram_limit']-rss)
    if hold!=paused:
     import signal
     os.killpg(p.pid,signal.SIGSTOP if hold else signal.SIGCONT);paused=hold
    with broker.locked() as b:b['heartbeat']=time.time();b['hold_new']=True
    status('paused_pressure' if hold else 'building',child=p.pid,user_rss_bytes=rss);time.sleep(5)
   assert p.returncode==0,('build exit',p.returncode)
  out=B/'build';out.mkdir()
  link=json.loads((B/'link_manifest.json').read_text());argv=link['argv'];argv[argv.index('-o')+1]=str(out/'gem5.opt')
  argv=[x.replace('--thread-count=8','--thread-count=2') for x in argv]+['-Wl,--no-keep-files-mapped','-Wl,--no-map-whole-files']
  status('linking')
  with (B/'link.log').open('w') as f:subprocess.run(['prlimit',f'--as={24*G}:','--',*argv],cwd=W/'build/ARM_MESI_Three_Level_SPM',env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
  old=R/'results/cacheflex_l3_spm/direct_link_20260928/build_v4';shutil.copytree(old/'configs',out/'configs',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
  save(out/'frozen.json',{str(p):sha(p) for p in out.rglob('*') if p.is_file()})
  status('built_requires_runtime_validation',sha256=sha(out/'gem5.opt'))
 except BaseException as e:status('failed',error=repr(e));raise
 finally:
  with broker.locked() as b:b['leases'].pop('retained_build',None);b['heartbeat']=time.time()
if __name__=='__main__':main()
