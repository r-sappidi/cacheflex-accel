"""Add all accepted AI-12 evidence; publish comparisons only after equal-budget completion."""
import ast,copy,datetime,hashlib,json,os,resource,sys
from pathlib import Path
ROOT=Path('/data2/rsappidi/cacheflex_mc');OUT=ROOT/'results/paper_tables_20260930_remainder';OUT.mkdir(exist_ok=True)
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1');os.sched_setaffinity(0,{44});resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3))
sys.path.insert(0,str(ROOT/'scripts'))
from run_access_timing import stats
source=(ROOT/'scripts/prepare_paper_sheet.py').read_text()
# AI-12 names its full-oracle/bit-exact acceptance fields differently.
source=source.replace("ar.get('passed',False) or ar.get('errors')==0", "(ar.get('full_oracle') is True) or ar.get('passed',False) or ar.get('errors')==0")
tree=ast.parse(source);exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('read','add','h','best')],type_ignores=[]),'accepted_export_helpers','exec'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
w=read(ROOT/'results/paper_tables_20260930/workbook.json');old=copy.deepcopy(w);tabs=w['tabs'];provenance=tabs['Provenance']['rows'];configs=tabs['Hardware']['rows'];runs=[];by_path={};hash_cache={};config_groups={(c[1],*c[2:20]):c[0] for c in configs}
for raw,p in zip(tabs['Accepted runs']['rows'],provenance):
 c=dict(t=raw[2],arm=raw[3],**{k:v for k,v in zip(('br','bc','r','mp','kc','g'),raw[6:12]) if v is not None})
 r=dict(id=raw[0],workload=raw[1],case=c,arm=raw[3],first_us=raw[4],repeat_us=raw[5],phase=raw[12],config_id=raw[13],ledger=raw[14],path=str((ROOT/p[3]).parent));runs.append(r);by_path[r['path']]=r
state=read(ROOT/'results/ai_workloads/qwen_remainder_20260929/experiment_memory_v2/status.json');(OUT/'remainder_status_snapshot.json').write_text(json.dumps(state,indent=2))
labels=dict(rmsnorm='RMSNorm',residual='Residual add',rope='RoPE application',rope_tables='RoPE tables',embedding='Embedding',final_norm='Final RMSNorm',logits='Last-token logits')
for r in state['completed']:
 c=r['case'];assert r['math']['full_oracle'];assert r['math']['exact_oracle_bits'] or c['op'] not in ('residual','rope','embedding');assert len(r['checks'])==2 and all(x['nonfinite']=='0' for x in r['checks']);assert r['checks'][0]['hash']==r['checks'][1]['hash']
 assert all(e['copies_issued']==e['copies_completed']==0 for e in r['epochs'])
 gate=c['stage'] in ('tail_gates','shape_gates');phase=('Correctness gate only; not performance comparison' if gate else 'Accepted candidate; see Qwen remainder for complete comparisons')+f"; block{c['b']}; V{c['v']}; {c['stage']}; oracle={'exact bits' if r['math']['exact_oracle_bits'] else r['math']['bound']}; peer bits exact"
 add(r,'AI-12','Qwen '+labels[c['op']],'Qwen remainder',phase=phase)
# All exact-output peers must agree within operator/size/vocabulary.
peers={}
for r in state['completed']:
 c=r['case'];k=(c['op'],c['t'],c['v']);assert k not in peers or peers[k]==r['output_sha256'];peers[k]=r['output_sha256']
rows=[];incomplete=[]
for op,label in labels.items():
 candidates=[r for r in state['completed'] if r['case']['op']==op and r['case']['stage']=='screen']
 choices={1,2,4} if op=='logits' else {1,8,32}
 complete=all({r['case']['b'] for r in candidates if r['case']['arm']==a}==choices for a in ('cache','spm'))
 if not complete:incomplete.append(f'{label}: {len(candidates)}/6 screening cases accepted');continue
 a,b=[min((r for r in candidates if r['case']['arm']==arm),key=lambda r:r['first_us']) for arm in ('cache','spm')];n=len(rows)+2
 rows.append([label,a['case']['t'],a['first_us'],b['first_us'],f'=C{n}/D{n}',f'=1-D{n}/C{n}',a['case']['b'],b['case']['b'],by_path[a['path']]['id'],by_path[b['path']]['id'],'All6 screening candidates accepted. Coherent SVE on both platforms; zero SPM copies/reductions. First full call; second call diagnostic. Transfers/integration pending.'])
tabs['Qwen remainder']=dict(header=['Operator','Input tokens','Cache first us','SPM-platform first us','Speedup x','Latency reduction','Cache block','SPM block','Cache run','SPM run','Scope / timing policy'],rows=rows,widths={},formats={2:'0.0000',3:'0.0000',4:'0.000"x"',5:'0.00%'},filter=True)
for r in runs[len(old['tabs']['Accepted runs']['rows']):]:
 c=r['case'];tabs['Accepted runs']['rows'].append([r['id'],r['workload'],c['t'],r['arm'],r['first_us'],r['repeat_us'],None,None,None,None,None,None,r['phase'],r['config_id'],r['ledger']])
snap=datetime.datetime.now(datetime.timezone.utc).isoformat()
for r in tabs['Read me']['rows']:
 if r[0]=='Snapshot UTC':r[1]=snap
 if r[0]=='Headline tables':r[1]+=' Qwen remainder includes completed T1024 operator screens only; remaining screens/transfers pending.'
 if r[0]=='Supporting evidence':r[1]+=' AI-12 small correctness gates are labeled separately in Accepted runs; their timings are not headline performance evidence.'
 if r[0]=='Missing layer work':r[1]='Remainder evaluation is active. Completed operator screens appear in Qwen remainder; incomplete screens/transfers and integrated model validation remain outstanding. No full-model total.'
for r in tabs['Pending and exclusions']['rows']:
 if r[0]=='Layer/model total':r[3]='AI-12 active: '+ '; '.join(incomplete)+'. Integrated layer/model timing still unvalidated.'
 if r[0]=='Qwen FA' and r[1]==32768:r[2:]=['Cache/private running; reduction arms resource-queued','No accepted T32768 performance pair at snapshot']
w.update(snapshot=snap,run_count=len(runs),headline_pairs=old['headline_pairs']+len(rows));w['source_hashes'].update(hash_cache)
(OUT/'workbook.json').write_text(json.dumps(w,indent=2));(OUT/'audit.json').write_text(json.dumps(dict(snapshot=snap,accepted_remainder=len(state['completed']),gates=sum(r['case']['stage'] in ('tail_gates','shape_gates') for r in state['completed']),complete_screen_comparisons=len(rows),incomplete=incomplete,old_ids_preserved=True,full_oracle_peer_bits_output_hash_config_zero_reductions_verified=True),indent=2))
print(json.dumps(dict(new_records=len(state['completed']),total_records=len(runs),comparisons=rows,incomplete=incomplete),indent=2))
