"""Run one delegated FA42 case; original scientific validator is reused verbatim."""
import sys,ast,argparse,hashlib,time,json
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]
sys.path.insert(0,str(H.parent/'packet_diag'))
import run as original
from common import *
H=Path(__file__).resolve().parent
ROOT=BASE/'parallel_v1'
def validation_source():
 s=(original.H/'run.py').read_text()
 start=s.index("    log=(path/'run.log').read_text();")
 end=s.index("    save(path/'accepted.json',row)",start)
 import textwrap
 code=textwrap.dedent(s[start:end]);ast.parse(code)
 return code

def main():
 ap=argparse.ArgumentParser();ap.add_argument('case');ap.add_argument('--cpus',required=True);ap.add_argument('--token',required=True);a=ap.parse_args()
 def stop(signum,frame):raise KeyboardInterrupt('worker termination requested')
 signal.signal(signal.SIGTERM,stop)
 setup();os.sched_setaffinity(0,[int(x) for x in a.cpus.split(',')])
 c=next(x for x in read(OUT/'plan.json')['cases'] if x['name']==a.case);assert not c['full']
 path=OUT/'runs'/a.case;source=Path(c['source']);ticket='fa42_parallel_'+a.case
 state=dict(name=a.case,path=str(path),stage='starting',pid=os.getpid(),active={},updated=time.time())
 def status(stage,**kw):
  state.update(stage=stage,updated=time.time());state['active']=kw;save(ROOT/'workers'/f'{a.case}.json',state)
 claim=read(path/'delegated.json');assert claim['token']==a.token and claim['owner']=='FA42-parallel-v1'
 frozen=read(OUT/'inputs.json');frozen.update(read(ROOT/'inputs.json'))
 def unchanged():
  for p,h in frozen.items():assert sha(p)==h,('frozen input drift',p)
 acquired=False
 try:
  acquire(ticket,10,status);acquired=True;unchanged()
  cmd=read(source/'command.json');cmd[0]=str(B/'gem5.opt');cmd[1]='--outdir='+str(path)
  if c['trace']:cmd[1:1]=['--debug-flags=CacheFlexTiming',f'--debug-start={c["start"]}',f'--debug-end={c["end"]}','--debug-file=trace.gz']
  cmd+=['--abs-max-tick='+str(c['stop'])];save(path/'command.json',cmd)
  wall=execute(cmd,R,path/'run.log',ticket,8*GIB,status,tracepath=path/'trace.gz' if c['trace'] else None)
  if c['trace']:
   off=OUT/'runs'/(c['name'].removesuffix('_on')+'_off');start=time.monotonic()
   while not (off/'accepted.json').exists():
    w=ROOT/'workers'/(off.name+'.json')
    if w.exists():assert read(w)['stage']!='failed','paired trace-off failed'
    assert time.monotonic()-start<24*3600,'paired validation wait timeout'
    status('waiting_for_paired_off_validation');time.sleep(5)
  status('validating');ctx=dict(original.__dict__)
  ctx.update(c=c,path=path,source=source,cmd=cmd,wall=wall,unchanged=unchanged,OUT=OUT)
  code=validation_source();assert hashlib.sha256(ast.dump(ast.parse(code)).encode()).hexdigest()==read(ROOT/'validator.json')['ast_sha256']
  exec(compile(code,str(original.H/'run.py')+':unchanged-validation','exec'),ctx)
  row=ctx['row'];save(path/'accepted.json',row);status('accepted')
 except BaseException as e:
  state['error']=repr(e);status('failed');raise
 finally:
  if acquired:broker.release(ticket)
if __name__=='__main__':main()
