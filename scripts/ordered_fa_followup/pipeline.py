"""Ordered, fail-closed continuation; no speculative baseline publication."""
import os,json,time,subprocess,sys,hashlib,fcntl
from pathlib import Path
R=Path('/data2/rsappidi/cacheflex_mc');D=R/'results/ordered_fa_followup_20261005';SW=R/'results/cacheflex_flash_attention/software_confirm_20261005'
ARMS=('cache','cache_stage','burst','stage','paced','combined')
def read(p):return json.loads(p.read_text())
def save(p,x):tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
state=dict(pid=os.getpid(),stage='starting',active={},completed=[],failed=[])
def status():state['updated']=time.time();save(D/'status.json',state)
def launch(script,label,env=None):
 log=(D/(label+'.log')).open('w');p=subprocess.Popen([sys.executable,str(script)],cwd=R,env=dict(os.environ,**(env or {})),stdout=log,stderr=subprocess.STDOUT);log.close();state['active'][label]=p.pid;status();return p
def wait(procs):
 while any(p.poll() is None for p in procs.values()):
  for label,p in procs.items():
   if p.poll() not in (None,0):raise RuntimeError((label,p.returncode))
  status();time.sleep(5)
 for label,p in procs.items():assert p.returncode==0,(label,p.returncode)
 state['active']={};status()
def refresh():subprocess.run([sys.executable,str(R/'scripts/refresh_current_result_log.py')],check=True,cwd=R)
def freeze():
 rows=[]
 for arm in ARMS:
  s=read(SW/arm/'status.json');assert s['stage']=='complete' and not s['failed']
  assert len(s['completed'])==3
  for a in s['completed']:
   assert read(Path(a['path'])/'accepted.json')==a
  a=next(x for x in s['completed'] if x['case']['t']==4096);rows.append(dict(arm=arm,**a))
 assert len({x['output_sha256'] for x in rows})==1
 cache=min((a for a in rows if not a['case']['spm']),key=lambda a:a['roi_us']);spm=min((a for a in rows if a['case']['spm']),key=lambda a:a['roi_us'])
 result=dict(scope='Best accepted same-ELF nominal-hardware software confirmation at Qwen T4096, Br16 Bc256 R256, V-only. Geometry grounded in FA-35; not global optimum or measured end-to-end model.',cache=cache,spm=spm,speedup=cache['roi_us']/spm['roi_us'],candidates=rows,prior=['FA-35','FA-39','FA-44','SYN-01'])
 save(D/'baseline_selection.json',result)
 with (R/'docs/experiments/LEDGER.md').open('a') as f:f.write('\n**FA-45 completed software confirmation:** 18 full-oracle cases accepted. Same-ELF bounded baseline cache '+cache['arm']+' '+str(cache['roi_us'])+' us; SPM '+spm['arm']+' '+str(spm['roi_us'])+' us; ratio '+str(result['speedup'])+'x. See results/ordered_fa_followup_20261005/baseline_selection.json. Nominal retained hardware, fixed screened geometry; no global optimum/end-to-end claim.\n')
 return result
lock=(D/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
try:
 state['stage']='fa40_recovery';status();wait({'fa40':launch(D/'fa40_run.py','fa40')})
 s=read(D/'fa40/status.json');assert s['stage']=='complete' and len(s['completed'])==38 and not s['failed'];refresh()
 data={a['name']:a.get('roi_us') for a in s['completed'] if a['name'].startswith('target_')};save(D/'diagnostic_summary.json',data)
 with (R/'docs/experiments/LEDGER.md').open('a') as f:f.write('\n**FA-40 recovery complete:** all38 accepted, original scientific validation unchanged. Target microseconds '+json.dumps(data)+'. Oracles remain diagnostic only, not candidate baselines. Authoritative continuation results/ordered_fa_followup_20261005/fa40/status.json.\n')
 state['completed'].append('fa40');state['stage']='software_confirmation';status()
 procs={arm:launch(R/'benchmarks/qwen/fa/software_confirm/run.py',arm,{'FA45_ARM':arm,'FA45_CPU_OFFSET':str(i*2)}) for i,arm in enumerate(ARMS)}
 wait(procs);refresh();state['completed'].append('software_confirmation');state['stage']='baseline_selection';status();freeze();state['completed'].append('baseline_selection')
 state['stage']='llama_continuation';status();wait({'llama':launch(D/'llama_run.py','llama')})
 s=read(D/'llama/status.json');assert s['stage']=='awaiting_final_review' and not s['failed'];refresh()
 state['completed'].append('llama_search');state['stage']='awaiting_final_scientific_review';status()
 # Preserve component policy distinctions; never combine incompatible ROI measurements.
 components=dict(projection_source='results/ai_workloads/qwen_projection_20260928/experiment_v2',ffn_source='results/ai_workloads/qwen_swiglu_20260928/experiment_v1',qwen_baseline='baseline_selection.json',llama_status='llama/status.json',limitations=['Historical component profile predates retained-source protocol; no implicit equivalence claim.','First calls include packing; second calls are not proven steady warm state.','No integrated layer/model timing; missing pipeline costs are not zero.','Review Llama local-neighbor closure or unresolved frontier before freezing.'])
 save(D/'component_consolidation.json',components)
except BaseException as e:
 state['failed'].append(repr(e));state['stage']='failed_requires_review';status()
 # Do not abandon active children after a failure.
 for pid in state['active'].values():
  try:os.kill(pid,15)
  except ProcessLookupError:pass
 raise
