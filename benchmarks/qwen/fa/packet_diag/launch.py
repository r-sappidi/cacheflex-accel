"""Build and run serial diagnostics under the shared host budget."""
from common import *
from prepare import prepare

def main():
 setup();prepare();OUT.mkdir();save(OUT/'status.json',dict(stage='building',active={},pending=['build','exact_gate','paired_windows'],completed=[],updated=time.time()))
 def st(stage,**kw):save(B/'status.json',dict(stage=stage,updated=time.time(),**kw))
 ticket='fa42_compile';acquire(ticket,6,st)
 try:
  cmd=['g++','-std=c++17','-O3','-pipe','-fno-strict-aliasing','-DTRACING_ON=1','-DNUMBER_BITS_PER_SET=64','-c',str(B/'NetworkInterface.cc'),'-o',str(B/'NetworkInterface.o')]
  for p in [B,O,G/'src',G/'include',G/'ext/pybind11/include',G/'ext',G/'ext/nomali/include',G/'ext/libelf','/data2/rsappidi/conda-envs/cacheflex-gem5/include','/data2/rsappidi/conda-envs/cacheflex-gem5/include/python3.12']:cmd+=['-I'+str(p)]
  save(B/'compile_command.json',cmd);execute(cmd,G,B/'compile.log',ticket,4*GIB,st,env=dict(ENV,CCACHE_DISABLE='1'))
 finally:broker.release(ticket)
 ticket='fa42_link';acquire(ticket,24,st)
 try:
  link=read(R/'results/retained_protocol_restart_20261002/link_manifest.json');cmd=list(link['argv']);inputs={}
  for i,x in enumerate(cmd):
   if x=='mem/ruby/network/garnet/NetworkInterface.o':cmd[i]=str(B/'NetworkInterface.o')
   if x=='-Wl,--thread-count=8':cmd[i]='-Wl,--thread-count=1'
   if x.endswith(('.o','.a','.so')):
    p=Path(cmd[i]);p=p if p.is_absolute() else O/p;inputs[str(p)]=sha(p)
  assert str(B/'NetworkInterface.o') in cmd
  cmd[cmd.index('-o')+1]=str(B/'gem5.opt');cmd+=['-Wl,--no-keep-files-mapped','-Wl,--no-map-whole-files']
  save(B/'link_inputs.json',inputs);save(B/'link_command.json',cmd)
  execute(cmd,O,B/'link.log',ticket,23*GIB,st,env=dict(link['environment'],CCACHE_DISABLE='1'))
  st('built_runtime_pending',sha256=sha(B/'gem5.opt'))
 finally:broker.release(ticket)
 import run
 run.main()
if __name__=='__main__':
 try:main()
 except BaseException as e:
  if OUT.exists():
   s=read(OUT/'status.json');s.update(stage='failed',error=repr(e),updated=time.time());save(OUT/'status.json',s)
  raise
