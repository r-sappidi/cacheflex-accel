"""Append accepted non-reduction results; retain existing workbook run IDs."""
import ast, copy, datetime, hashlib, json, os, resource, sys
from pathlib import Path
ROOT=Path('/data2/rsappidi/cacheflex_mc');OUT=ROOT/'results/paper_tables_20260930';OUT.mkdir(exist_ok=True)
os.sched_setaffinity(0,{44});resource.setrlimit(resource.RLIMIT_AS,(512*1024**2,512*1024**2))
sys.path.insert(0,str(ROOT/'scripts'))
from run_access_timing import stats
# Reuse the original acceptance checks unchanged, without rerunning its old export.
source=ast.parse((ROOT/'scripts/prepare_paper_sheet.py').read_text())
funcs=ast.Module(body=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in ('read','add','h','best','tiling')],type_ignores=[])
exec(compile(funcs,'original_acceptance_helpers','exec'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
old=read(ROOT/'results/paper_tables_20260929/workbook.json');w=copy.deepcopy(old);tabs=w['tabs']
provenance=tabs['Provenance']['rows'];configs=tabs['Hardware']['rows'];runs=[];by_path={};hash_cache={}
config_groups={(c[1],*c[2:20]):c[0] for c in configs}
for raw,p in zip(tabs['Accepted runs']['rows'],provenance):
 assert raw[0]==p[0]
 c=dict(t=raw[2],arm=raw[3],**{k:v for k,v in zip(('br','bc','r','mp','kc','g'),raw[6:12]) if v is not None})
 r=dict(id=raw[0],workload=raw[1],case=c,arm=raw[3],first_us=raw[4],repeat_us=raw[5],phase=raw[12],config_id=raw[13],ledger=raw[14],path=str((ROOT/p[3]).parent))
 runs.append(r);by_path[r['path']]=r
old_count=len(runs)
roots={'fa':ROOT/'results/cacheflex_l3_spm/qwen_neighbors_20260929/experiment_v1',
 'ffn':ROOT/'results/ai_workloads/qwen_swiglu_20260928/experiment_v1',
 'proj':ROOT/'results/ai_workloads/qwen_projection_20260928/experiment_v2',
 'long':ROOT/'results/cacheflex_l3_spm/qwen_long_context_20260928/experiment_v1'}
states={k:read(v/'status.json') for k,v in roots.items()}
for k,s in states.items():(OUT/f'{k}_status_snapshot.json').write_text(json.dumps(s,indent=2))
fa=[r for r in states['fa']['completed'] if r['case']['t']>=1024 and r['case']['hq']==14 and r['case']['arm'] in ('cache','private')]
ff=[r for r in states['ffn']['completed'] if r['case']['t']>=1024]
pr=[r for r in states['proj']['completed'] if r['case']['t']>=1024]
for rows,ledger,workload,group in [(fa,'L3-11/12 (non-reduction arms)','Qwen FA','Qwen FA search'),(ff,'AI-10','Qwen SwiGLU','Qwen FFN/projection'),(pr,'AI-11','Qwen projection','Qwen FFN/projection')]:
 for r in rows:
  phase='Completed bounded target-size search' if r['case']['t']>=2048 else 'Completed 1K search candidate'
  if workload=='Qwen FA':phase='Completed axis-neighbor search; fixed Br16' if r['case']['t']==4096 else 'Completed bounded search candidate'
  if workload=='Qwen SwiGLU' and r['case']['t']==4096:phase='Accepted target; T4096 comparison incomplete'
  entry=add(r,ledger,workload,group,phase=phase);entry['phase']=phase
for r in states['long']['completed']:
 if r['name']=='t16384_cache':add(r,'L3-10 non-reduction control','Qwen FA','Qwen FA exact replay',phase='Accepted cache only; long-context pair incomplete')
# Preserve original exact-replay1K/2K references; replace only strengthened4K pair.
a,b=[by_path[best(fa,t=4096,arm=arm)['path']] for arm in ('cache','private')]
row=tabs['Qwen FA']['rows'][2];row[4:6]=[a['first_us'],b['first_us']];row[8:]=[tiling(a),tiling(b),a['id'],b['id'],'Target-size search and R/Bc axis-neighbors complete; Br16 fixed; best measured, not global optimum']
assert len([r for r in ff if r['case']['t']==2048])==4
for t in (2048,):
 a,b=[by_path[best(ff,t=t,arm=arm)['path']] for arm in ('cache','spm')];n=len(tabs['Qwen FFN']['rows'])+2
 tabs['Qwen FFN']['rows'].append([t,896,4864,a['first_us'],b['first_us'],f'=D{n}/E{n}',f'=1-E{n}/D{n}',tiling(a),tiling(b),a['repeat_us'],b['repeat_us'],a['id'],b['id'],'Target-size winner vs common control complete; first call includes packing/staging, 3 GEMMs and SiLU-product; second call diagnostic'])
assert states['proj']['stage']=='completed_validated'
for t in (2048,4096):
 for op,title,ncol in [('qkv','QKV + bias/head layout',1152),('out','Output + head merge',896)]:
  assert len([r for r in pr if r['case']['t']==t and r['case']['op']==op])==4
  a,b=[by_path[best(pr,t=t,op=op,arm=arm)['path']] for arm in ('cache','spm')];n=len(tabs['Qwen projections']['rows'])+2
  tabs['Qwen projections']['rows'].append([title,t,896,ncol,a['first_us'],b['first_us'],f'=E{n}/F{n}',f'=1-F{n}/E{n}',tiling(a),tiling(b),a['repeat_us'],b['repeat_us'],a['id'],b['id'],'Target-size winner vs common control complete; first call includes packing/staging and stated layout work; second call diagnostic'])
raw=[]
for r in runs:
 c=r['case'];raw.append([r['id'],r['workload'],c.get('t'),r['arm'],r['first_us'],r['repeat_us'],c.get('br'),c.get('bc'),c.get('r'),c.get('mp'),c.get('kc'),c.get('g'),r['phase'],r['config_id'],r['ledger']])
tabs['Accepted runs']['rows']=raw
pending=tabs['Pending and exclusions']['rows']
pending[0][2:]=['Completed; published in Qwen FFN','Four accepted target candidates; independent cache/SPM selection within tested budget']
a,b=[best(ff,t=4096,arm=arm) for arm in ('cache','spm')]
pending[1][2:]=['Two candidates still active',f"Accepted provisional pair cache {a['first_us']:.4f} us / SPM {b['first_us']:.4f} us = {a['first_us']/b['first_us']:.6f}x. Individual records included; final comparison withheld."]
for i in (2,3):pending[i][2:]=['Completed; published in Qwen projections','Both operators and four target candidates/operator accepted; all52 experiment cases complete']
pending[4][2:]=['Cache accepted; SPM comparison pending','T16384 cache154969.2524us accepted; private-SPM active. Host-allocation failures preserved separately; no paired speedup yet.']
pending[5][2:]=['Resource-queued','No accepted T32768 performance pair; preserved jobs await resource admission']
pending[6][3]='Remainder pipeline continuation pending after recovered targets; existing small gates retained. Integrated cache-state/precision/packing policy and full-model total not yet validated.'
snap=datetime.datetime.now(datetime.timezone.utc).isoformat()
for r in tabs['Read me']['rows']:
 if r[0]=='Snapshot UTC':r[1]=snap
 if r[0]=='Headline tables':r[1]='Qwen FA1K/2K/4K, FFN1K/2K, QKV/output projections1K/2K/4K. T4096FFN provisional pair appears only in Pending and exclusions.'
 if r[0]=='Selection':r[1]='FA4K target-size and R/Bc axis-neighbors complete with Br16fixed. FFN/projections four candidates/backend/operator at1K; target sizes compare selected winner and common control. No global-optimum claim.'
w.update(title='CacheFlex — validated non-reduction component results — 2026-09-30',snapshot=snap,run_count=len(runs),headline_pairs=11)
w['source_hashes'].update(hash_cache)
(OUT/'workbook.json').write_text(json.dumps(w,indent=2))
(OUT/'audit.json').write_text(json.dumps(dict(snapshot=snap,old_runs=old_count,new_runs=len(runs)-old_count,total_runs=len(runs),all_new_acceptance_checks_pass=True,original_helper_ast_reused=True,old_ids_preserved=True,non_reduction_scope_preserved=True),indent=2))
print(json.dumps(dict(new_runs=len(runs)-old_count,tabs={k:len(v['rows']) for k,v in tabs.items()},fa4k=row,ffn2k=tabs['Qwen FFN']['rows'][-1]),indent=2))
