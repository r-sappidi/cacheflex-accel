"""Passive, same-configuration replay windows and exact observer gates."""
from common import *
import fcntl,re
from analyze import analyze,report
PARENT=R/'results/retained_protocol_restart_20261002/fa_matrix_spm_only/runs/t256_q14_legacy_v_br16_bc256_r256'
PAIR=R/'results/cacheflex_flash_attention/bottleneck_20261004/targets_continuation_v1/runs'
def config_equal(source,path):
 old=dict(leaves(read(source/'config.json')));new=dict(leaves(read(path/'config.json')))
 expected={k:v.replace(str(source),str(path)) if isinstance(v,str) else v for k,v in old.items()}
 assert expected==new,{k:[expected.get(k),new.get(k)] for k in expected.keys()|new.keys() if expected.get(k)!=new.get(k)}
def plan():
 cases=[dict(name='exact_t256_off',source=str(PARENT),full=True,trace=False)]
 # Full replay with a whole-small-ROI trace window must also match every original stat/output.
 s=stats(PARENT)[0];start=int(s['finalTick']-s['simTicks'])
 cases.append(dict(name='exact_t256_on',source=str(PARENT),full=True,trace=True,start=start,end=int(s['finalTick'])))
 for window in ('early','middle'):
  for mode in ('cache','spm'):
   source=PAIR/f'target_{mode}_control';s=stats(source)[0];roi=int(s['finalTick']-s['simTicks']);duration=int(s['simTicks'])
   # Same normalized phase, NOT identical dynamic work. Analyze each separately.
   start=roi+(100_000_000 if window=='early' else duration*2//5)
   start=start//400*400;end=start+10_000_000;stop=end+4000
   for trace in (False,True):cases.append(dict(name=f'{window}_{mode}_{"on" if trace else "off"}',source=str(source),full=False,trace=trace,start=start,end=end,stop=stop,window=window,mode=mode))
 return cases
def main():
 (OUT/'runs').mkdir();lock=(OUT/'coordinator.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 cases=plan();state=dict(pid=os.getpid(),stage='freezing',active={},pending=[c['name'] for c in cases],completed=[],failed=[])
 def status(stage=None,**kw):
  if stage:state['stage']=stage
  state['updated']=time.time();save(OUT/'status.json',state)
 status();save(OUT/'plan.json',dict(cases=cases,max_simulators=1,trace_limit_bytes=2*GIB,diagnostic_only=True,prior_ids=['FA-13','FA-15','FA-16','FA-18','FA-21','FA-39','FA-40'],decision='Select a targeted NI/VC/router/ejection intervention only if repeatable complete-packet interval evidence identifies it. A hardware speedup requires paired full-ROI validation; prefixes cannot establish it.'))
 frozen=read(B/'source_manifest.json');frozen.update(read(B/'link_inputs.json'));frozen[str(B/'gem5.opt')]=sha(B/'gem5.opt')
 frozen.update(read(R/'results/retained_protocol_restart_20261002/build/frozen.json'))
 for p in H.glob('*.py'):frozen[str(p)]=sha(p)
 for source in {Path(c['source']) for c in cases}:
  marker=source/'accepted.json';assert marker.exists(),source
  cmd=read(source/'command.json')
  for p in [marker,source/'command.json',source/'config.json',source/'stats.txt',source/'output.fp16',Path(cmd[0]),Path(cmd[2]),Path(cmd[cmd.index('--cmd')+1])]:frozen[str(p)]=sha(p)
 # Freeze reused Python config tree too; no originals are edited.
 for p in (Path(read(PARENT/'command.json')[2]).parents[2]).rglob('*.py'):frozen[str(p)]=sha(p)
 save(OUT/'inputs.json',frozen)
 def unchanged():
  for p,h in frozen.items():assert sha(p)==h,('input drift',p)
 try:
  for c in cases:
   source=Path(c['source']);path=OUT/'runs'/c['name'];path.mkdir();ticket='fa42_'+c['name']
   acquire(ticket,10,lambda x:status(x))
   try:
    unchanged();cmd=read(source/'command.json');cmd[0]=str(B/'gem5.opt');cmd[1]='--outdir='+str(path)
    if c['trace']:cmd[1:1]=['--debug-flags=CacheFlexTiming',f'--debug-start={c["start"]}',f'--debug-end={c["end"]}','--debug-file=trace.gz']
    if not c['full']:cmd+=['--abs-max-tick='+str(c['stop'])]
    save(path/'command.json',cmd)
    def progress(stage,**kw):
     state['active']={c['name']:dict(path=str(path),**kw)};status('running_'+c['name'])
    wall=execute(cmd,R,path/'run.log',ticket,8*GIB,progress,tracepath=path/'trace.gz' if c['trace'] else None)
    log=(path/'run.log').read_text();assert not re.search(r'panic:|fatal:|\bFAIL\b',log),log[-3000:]
    config_equal(source,path);blocks=stats(path)
    if c['full']:
     assert 'because exiting with last active thread context' in log
     assert blocks==stats(source),'instrumented binary changed nonhost statistics'
     assert sha(path/'output.fp16')==sha(source/'output.fp16'),'output changed'
    else:
     assert 'simulate() limit reached' in log
     assert int(blocks[-1]['finalTick'])==c['stop']
     if c['trace']:
      off=OUT/'runs'/c['name'].removesuffix('_on');off=off.with_name(off.name+'_off')
      assert blocks==stats(off),'trace observation changed nonhost statistics'
    summary=analyze(path) if c['trace'] else None
    if c['trace'] and ('spm' in c['name'] or c['full']):assert summary['groups'].get('copy_DATA',{}).get('packets',0)>0,'no complete SPM DATA in window; expand window deliberately, no silent acceptance'
    unchanged()
    row=dict(name=c['name'],path=str(path),case=c,outcome='exact_full_replay_gate' if c['full'] else 'bounded_diagnostic',full_roi_performance=False,exact_config=True,observer_equivalence=bool(c['full'] or c['trace']),source_acceptance=str(source/'accepted.json'),host_seconds=wall,trace_summary=summary,artifacts={f.name:sha(f) for f in path.iterdir() if f.is_file() and f.name!='accepted.json'})
    save(path/'accepted.json',row);state['completed'].append(row);state['pending'].remove(c['name']);state['active']={};status();report(OUT)
    subprocess.run([sys.executable,str(R/'scripts/refresh_current_result_log.py')],cwd=R,env=ENV,check=True)
   finally:broker.release(ticket)
  status('completed_diagnostics_pending_interpretation')
 except BaseException as e:
  state['failed'].append(dict(error=repr(e),case=c));state['active']={};status('failed');raise
if __name__=='__main__':setup();main()
