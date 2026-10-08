"""Freeze validated non-reduction component tables for a native Google Sheet."""
import json, hashlib, datetime, sys
from pathlib import Path
ROOT=Path('/data2/rsappidi/cacheflex_mc');OUT=ROOT/'results/paper_tables_20260929'
sys.path.insert(0,str(ROOT/'scripts'))
from run_access_timing import stats
OUT.mkdir(exist_ok=True)
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
SNAP=datetime.datetime.now(datetime.timezone.utc).isoformat()
roots={
 'fa':ROOT/'results/cacheflex_l3_spm/qwen_cache_analytical_20260928/experiment_v1',
 'fa_controls':ROOT/'results/cacheflex_l3_spm/direct_link_20260928/experiment_v2',
 'ffn':ROOT/'results/ai_workloads/qwen_swiglu_20260928/experiment_v1',
 'proj':ROOT/'results/ai_workloads/qwen_projection_20260928/experiment_v2',
 'single':ROOT/'results/cacheflex_flash_attention/finite_model_20260926/analytical_experiment_v5/experiment',
 'spread':ROOT/'results/cacheflex_flash_attention/spm_set_spread_20260927/experiment_retry_v2',
 'oldffn':ROOT/'results/ai_workloads/swiglu_retry_transfer_20260927'}
states={k:read(p/'status.json') for k,p in roots.items()}
for k,s in states.items():(OUT/f'{k}_status_snapshot.json').write_text(json.dumps(s,indent=2))
runs=[];provenance=[];configs=[];by_path={};config_groups={};hash_cache={}
def h(p):
 p=str(p)
 if p not in hash_cache:hash_cache[p]=sha(p)
 return hash_cache[p]
def add(row,ledger,workload,group,case=None,path=None,phase='Accepted candidate'):
 p=Path(path or row['path'])
 if str(p) in by_path:return by_path[str(p)]
 marker=p/'accepted.json'
 if not marker.exists():marker=p/'validated.json'
 assert read(marker)==row,('marker/status mismatch',p)
 c=case or row.get('case',{});cfg=read(p/'config.json')['system'];rb=cfg['ruby']
 assert not any(rb[f'l2_cntrl{i}'].get('l3_reduction_enabled',False) for i in range(8)),p
 assert not any(rb[f'l0_cntrl{i}']['sequencer'].get('cacheflex_reduction',False) for i in range(8)),p
 epochs=stats(p);assert epochs
 assert all(sum(v for k,v in ep.items() if k.endswith('.reductionAccepted'))==0 for ep in epochs),p
 first=row.get('first_us',row.get('roi_us'));assert abs(first-epochs[0]['simTicks']/1e6)<1e-6,(p,first)
 ar=row.get('arithmetic',row.get('math',{}));assert ar and (ar.get('passed',False) or ar.get('errors')==0 or (ar.get('ordered_bits_exact') is True and ar.get('dense_errors')==0)),p
 output=p/'output.fp16'
 if not output.exists():
  candidates=[x for x in p.iterdir() if x.is_file() and x.suffix in ('.fp16','.bin')]
  expected=row.get('output_sha256');output=next((x for x in candidates if expected and h(x)==expected),None)
 assert output and output.exists(),('missing output',p)
 if 'output_sha256' in row:assert h(output)==row['output_sha256'],p
 cmd=read(p/'command.json');guest=Path(cmd[cmd.index('--cmd')+1]);binary=Path(cmd[0])
 rid=f'R{len(runs)+1:03d}'
 arm=c.get('arm',c.get('family',row.get('arch','')))
 run=dict(id=rid,ledger=ledger,workload=workload,group=group,case=c,path=str(p),first_us=first,
          repeat_us=row.get('repeat_us',row.get('warm_us')),arm=arm,phase=phase)
 runs.append(run);by_path[str(p)]=run
 provenance.append([rid,ledger,workload,str(marker.relative_to(ROOT)),h(marker),h(p/'config.json'),h(output),h(guest),h(binary),'Full arithmetic + saved configuration/resource checks; zero reduction commands'])
 # Per-run hardware, including differences that prevent pooling generations.
 cpu=cfg['cpu'][0];l2=rb['l1_cntrl0']['cache'];llc=rb['l2_cntrl0']['L2cache'];net=rb['network']
 hardware=[len(cfg['cpu']),1000/cfg['cpu_clk_domain']['clock'][0],1000/rb['clk_domain']['clock'][0],cpu['isa'][0]['sve_vl_se'],
 rb['l0_cntrl0']['Icache']['size']/1024,rb['l0_cntrl0']['Dcache']['size']/1024,l2['size']/1024,
 sum(rb[f'l2_cntrl{i}']['L2cache']['size'] for i in range(8))/2**20,l2['assoc'],l2['num_spm_ways'],
 l2.get('spm_set_stride',0),net.get('ni_flit_size'),net.get('vcs_per_vnet'),cpu['LQEntries'],cpu['SQEntries'],
 rb['l1_cntrl0'].get('spm_silent_retry',False),net.get('cacheflex_request_vc_partition',False)]
 key=(group,*hardware,h(binary))
 if key not in config_groups:
  cid=f'C{len(configs)+1:02d}';config_groups[key]=cid;configs.append([cid,group,*hardware,h(binary),rid])
 run['config_id']=config_groups[key]
 return run

def best(rows,**fields):return min((r for r in rows if all(r['case'].get(k)==v for k,v in fields.items())),key=lambda r:r.get('first_us',r.get('roi_us')))
def tiling(r):
 c=r['case']
 return f"Br{c['br']} / Bc{c['bc']} / R{c['r']}" if 'br' in c else f"MP{c['mp']} / KC{c['kc']}"+(f" / gate-up {'combined' if c['g'] else 'separate'}" if 'g' in c else '')
tabs={}
def tab(name,header,rows,widths=None,formats=None,filter=True):
 tabs[name]=dict(header=header,rows=rows,widths=widths or {},formats=formats or {},filter=filter)

fa_candidates=[r for r in states['fa']['completed'] if r['case']['t']>=1024 and r['case']['hq']==14 and r['case']['depth']==4 and r['case']['arm'] in ('cache','private')]
fa_runs=[add(r,'L3-06/07/08 (non-reduction arms)','Qwen FA','Qwen FA search',phase='Completed search candidate') for r in fa_candidates]
fa=[]
for t in (1024,2048,4096):
 selected={}
 for arm in ('cache','private'):
  r=next(r for r in states['fa_controls']['completed'] if r['name']==f't{t}_{arm}')
  assert r['roi_us']==best(fa_candidates,t=t,arm=arm)['roi_us']
  selected[arm]=add(r,'L3-09 disabled controls; selection L3-07/08','Qwen FA','Qwen FA exact replay',phase='Selected winner; exact replay')
 a,b=selected['cache'],selected['private'];n=len(fa)+2
 fa.append([t,14,2,64,a['first_us'],b['first_us'],f'=E{n}/F{n}',f'=1-F{n}/E{n}',tiling(a),tiling(b),a['id'],b['id'],'Independent bounded-search winners; causal prefill; single FA ROI'])
tab('Qwen FA',['Input tokens','Q heads','KV heads','Head dim','Cache us','Private SPM us','Speedup x','Latency reduction','Cache tile','SPM tile','Cache run','SPM run','Comparison scope'],fa,formats={4:'0.0000',5:'0.0000',6:'0.000"x"',7:'0.00%'})

ffrows=[r for r in states['ffn']['completed'] if r['case']['t']>=1024]
prrows=[r for r in states['proj']['completed'] if r['case']['t']>=1024]
for rows,ledger,workload in [(ffrows,'AI-10','Qwen SwiGLU'),(prrows,'AI-11','Qwen projection')]:
 for r in rows:add(r,ledger,workload,'Qwen FFN/projection',phase='Completed 1K search candidate' if r['case']['t']==1024 else 'Accepted target; size comparison incomplete')
assert len([r for r in ffrows if r['case']['t']==1024])==8
assert len([r for r in prrows if r['case']['t']==1024])==16
ff=[]
a,b=[by_path[best(ffrows,t=1024,arm=arm)['path']] for arm in ('cache','spm')]
ff.append([1024,896,4864,a['first_us'],b['first_us'],'=D2/E2','=1-E2/D2',tiling(a),tiling(b),a['repeat_us'],b['repeat_us'],a['id'],b['id'],'First call includes packing/staging, 3 GEMMs and SiLU-product; second call diagnostic only'])
tab('Qwen FFN',['Input tokens','Hidden D','Intermediate F','Cache first us','SPM first us','Speedup x','Latency reduction','Cache tuning','SPM tuning','Cache second us','SPM second us','Cache run','SPM run','Timing policy'],ff,formats={3:'0.0000',4:'0.0000',5:'0.000"x"',6:'0.00%',9:'0.0000',10:'0.0000'})
pr=[]
for op,title,ncol in [('qkv','QKV + bias/head layout',1152),('out','Output + head merge',896)]:
 a,b=[by_path[best(prrows,t=1024,op=op,arm=arm)['path']] for arm in ('cache','spm')];n=len(pr)+2
 pr.append([title,1024,896,ncol,a['first_us'],b['first_us'],f'=E{n}/F{n}',f'=1-F{n}/E{n}',tiling(a),tiling(b),a['repeat_us'],b['repeat_us'],a['id'],b['id'],'First call includes packing/staging and stated layout work; second call diagnostic only'])
tab('Qwen projections',['Operator','Input tokens','K','N','Cache first us','SPM first us','Speedup x','Latency reduction','Cache tuning','SPM tuning','Cache second us','SPM second us','Cache run','SPM run','Timing policy'],pr,formats={4:'0.0000',5:'0.0000',6:'0.000"x"',7:'0.00%',10:'0.0000',11:'0.0000'})

support=[]
def support_pair(label,t,a,b,metric,policy,note):
 n=len(support)+2
 av=a[metric] if metric in a else None;bv=b[metric] if metric in b else None
 assert av and bv
 support.append([label,t,av,bv,f'=C{n}/D{n}',f'=1-D{n}/C{n}',policy,a['id'],b['id'],note])
spread={r['name']:r for r in states['spread']['completed'] if r['name'].startswith(('t4096_','t16384_'))}
for t in (4096,16384):
 cache=spread[f't{t}_cache'];normal=spread[f't{t}_spm_control'];sp=spread[f't{t}_'+('spread' if t==4096 else 'spm_spread')]
 c=dict(t=t,hq=1,hkv=1,d=64,causal=False,br=16,bc=512,r=256 if t==4096 else 768)
 a=add(cache,'FA-31','Single-head FA','Repaired finite placement',case={**c,'arm':'cache'},path=roots['spread']/'runs'/cache['name'])
 b=add(normal,'FA-31','Single-head FA','Repaired finite placement',case={**c,'arm':'spm'},path=roots['spread']/'runs'/normal['name'])
 d=add(sp,'FA-31','Single-head FA','Repaired finite placement',case={**c,'arm':'spread-spm'},path=roots['spread']/'runs'/sp['name'])
 note='Matched cache tile, not independent cache optimum' if t==4096 else 'Transferred previously selected tile; no new exhaustive tuning'
 support_pair('Single-head FA: cache vs standard SPM',t,a,b,'first_us','Single full FA ROI',note)
 support_pair('Single-head FA: cache vs spread SPM',t,a,d,'first_us','Single full FA ROI',note)
 support_pair('Placement ablation: standard vs spread SPM',t,b,d,'first_us','Single full FA ROI','Same existing SRAM; stride change only; left column is standard SPM')
 # Original finite-model best-to-best result, separate hardware generation.
 cache_old=min((r for r in states['single']['completed'] if r['case']['t']==t and r['case']['family']=='q_cache'),key=lambda r:r['roi_us'])
 source=Path(normal['source']);old_spm=read(source/('accepted.json' if (source/'accepted.json').exists() else 'validated.json'))
 ca=add(cache_old,'FA-29','Single-head FA','Finite analytical before retry repair')
 cb=add(old_spm,'FA-29','Single-head FA','Finite analytical before retry repair',case={**c,'arm':'spm'},path=source)
 support_pair('Single-head FA: bounded-search winners',t,ca,cb,'first_us','Single full FA ROI','Older finite profile; keep separate from repaired placement and Qwen')
oldff=[]
for r in states['oldffn']['completed']:
 if r['case'][0]!=512:continue
 c=dict(t=512,d=2048,f=8192,mp=2,kc=128,g=0,arm=r['arch'])
 oldff.append(add(r,'AI-09','SwiGLU D2048/F8192','Finite retry transfer',case=c))
a,b=[next(r for r in oldff if r['arm']==arm) for arm in ('cache','spm')]
support_pair('SwiGLU finite-model transfer: first call',512,a,b,'first_us','First invocation including packing','Fixed inherited MP2/KC128; validation transfer, not tuned Qwen')
support_pair('SwiGLU finite-model transfer: warm median',512,a,b,'repeat_us','Median of last 3 of 5 invocations','Warm calls are not independent simulations; no confidence interval claimed')
tab('Supporting controls',['Comparison','Input tokens','Baseline us','SPM or variant us','Speedup x','Latency reduction','Timing policy','Baseline run','Variant run','Scope restriction'],support,formats={2:'0.0000',3:'0.0000',4:'0.000"x"',5:'0.00%'})

raw=[]
for r in runs:
 c=r['case'];raw.append([r['id'],r['workload'],c.get('t'),r['arm'],r['first_us'],r['repeat_us'],c.get('br'),c.get('bc'),c.get('r'),c.get('mp'),c.get('kc'),c.get('g'),r['phase'],r['config_id'],r['ledger']])
tab('Accepted runs',['Run ID','Workload','Input tokens','Backend','First or FA ROI us','Second or warm median us','Br','Bc','R','MP','KC','Combined gate-up','Selection status','Config ID','Ledger'],raw,formats={4:'0.0000',5:'0.0000'})
tab('Hardware',['Config ID','Profile','CPU cores','CPU GHz','Ruby GHz','SVE VL units','L1I KiB/core','L1D KiB/core','Private L2 KiB/core','Total LLC MiB','L2 ways','SPM eligible ways','SPM set stride','NoC flit bytes','VCs/vnet','Load entries','Store entries','Silent retry','Request VC partition','Simulator SHA256','Example run'],configs)
tab('Provenance',['Run ID','Ledger','Workload','Acceptance record (workspace relative)','Acceptance SHA256','Config SHA256','Output SHA256','Guest SHA256','Simulator SHA256','Validation scope'],provenance)

pending=[['Qwen FFN',2048,states['ffn']['stage'],'Comparison unfinished; target candidate results excluded from headline table'],['Qwen FFN',4096,'Queued','No accepted paired result at snapshot'],['Qwen projections',2048,states['proj']['stage'],'Accepted individual candidates are in Accepted runs; winner comparison incomplete'],['Qwen projections',4096,'Queued','No accepted paired result at snapshot'],['Qwen FA',16384,'Cache run active','No accepted long-context performance pair at snapshot'],['Qwen FA',32768,'Queued','No accepted long-context performance pair at snapshot'],['Layer/model total',None,'Not measured','RMSNorm, RoPE, residuals and integrated cache-state/precision/packing policy remain to validate']]
tab('Pending and exclusions',['Workload or evidence','Tokens','Status','Reason or boundary'],pending+[
 ['All arithmetic-offload mechanisms',None,'Excluded by request','L3 reduction, direct-reduction transport, near-SPM reduction, and their diagnostics'],
 ['Older FA through early Sep11',None,'Excluded','Known FP16 conversion defect; matched hashes did not establish correct attention'],
 ['Older FA and AI pilot generations',None,'Excluded from current paper tables','Superseded or materially different resource/cache/timing profiles; N-grade alone does not certify readiness'],
 ['Broad operator proxies / incomplete sweeps',None,'Excluded','Not tuned model-shaped component evidence; AI-01/04/05'],
 ['Historical GEMM / ACL reproduction',None,'Excluded from current paper tables','Not a complete W1-W7 suite; old profiles and precision/dispatch/correctness caveats; GEMM-01..05'],
 ['Original single-core paper numbers',None,'Reference only; not copied here','User-provided manuscript results are distinct from this multicore experiment archive'],
 ['Energy, area, RTL calibration',None,'Not established by these runs','Do not infer energy or area savings from runtime'],
 ['Partial stats / failed or cancelled targets',None,'Excluded','Only accepted full ROI and validated output evidence included']])

readme=[['Purpose','Validated non-reduction component performance tables for paper drafting.'],['Snapshot UTC',SNAP],['Primary scope','Qwen2.5-0.5B-Instruct-shaped causal prefill: batch1, 24-layer model shape, D896, F4864, Hq14/Hkv2/d64. Components measured independently.'],['Headline tables','Qwen FA: 1K, 2K, 4K. Qwen FFN and QKV/output projections: completed 1K searches.'],['Supporting evidence','Single-head finite-model analytical/placement controls and finite-model D2048/F8192 SwiGLU transfer are separately labeled; never pool profiles.'],['Paper-readiness meaning','Full numerical/configuration/resource acceptance supports scoped simulated component claims. Hardware timing/area calibration and broader paper evaluation remain separate requirements.'],['Excluded mechanism','No L3 or near-SPM arithmetic offload measurements. Every included run has zero reduction commands and disabled reduction engines.'],['Speedup definition','Baseline runtime / SPM runtime. >1 means faster. Latency reduction = 1 - SPM runtime / baseline runtime.'],['Units','All runtime inputs are microseconds (us); divide by1000 for milliseconds. No host wall time is used as simulated latency.'],['Token meaning','T is input/prefill sequence length, not generated output-token count. FA outputs T activation vectors.'],['FA timing','Complete kernel ROI includes its tile packing/staging, softmax normalization and drain; synthetic inputs.'],['FFN/projection timing','First-call ROI includes weight packing/staging and stated layout costs. Second-call figures are separate diagnostics, not assumed converged warm behavior.'],['Original-paper timing difference','Original single-core paper assumes load-time prepacked weights. A matched prepacked evaluation is needed for that deployment policy.'],['Precision','FA uses FP16 GEMMs with FP32 online state. Current FFN/projections use FP16 storage and sequential FP32 accumulation. Not BF16 checkpoint inference.'],['Selection','FA bounded analytical search; FFN/projections four candidates per backend per operator at1K. Independent winners within tested budget; no global-optimum claim.'],['Evidence','Each Run ID resolves to saved acceptance/config/output/guest/simulator hashes in Provenance. Workspace paths require access to the project server.'],['Supporting-control limitation','Matched placement controls and inherited FFN transfers support their stated ablations; they are not independently tuned Qwen headline comparisons.'],['Missing layer work','Two RMSNorms, two residual additions and RoPE; model extras and integrated correctness/cache-state policy remain unresolved. No full-model total provided.'],['Statistics','Deterministic accepted simulations; repeated guest calls are not independent simulation replicates. No confidence intervals invented.'],['Live vs snapshot','This workbook is a verified static snapshot. Existing simulation queues continue; new accepted results need an explicit refresh.'],['Formatting','Gray headers, frozen header rows, filters for run/provenance tables, formula-derived speedups.'],['Source ledger','docs/experiments/LEDGER.md; numerical grade N alone is not a blanket paper-readiness certification.']]
# Put overview first, then current performance, supporting, raw evidence and boundaries.
final={'Read me':dict(header=['Topic','Definition / scope'],rows=readme,widths={0:240,1:1000},formats={},filter=False)}
final.update(tabs)
result=dict(title='CacheFlex — validated non-reduction component results — 2026-09-29',snapshot=SNAP,tabs=final,run_count=len(runs),headline_pairs=len(fa)+len(ff)+len(pr),source_hashes=hash_cache)
(OUT/'workbook.json').write_text(json.dumps(result,indent=2))
(OUT/'audit.json').write_text(json.dumps(dict(snapshot=SNAP,runs=len(runs),headline_pairs=result['headline_pairs'],supporting_pairs=len(support),all_reduction_disabled=True,all_reduction_counts_zero=True,all_acceptance_records_match=True,all_output_hashes_verified=True),indent=2))
print(json.dumps(dict(runs=len(runs),tabs={k:len(v['rows']) for k,v in final.items()},headline_pairs=result['headline_pairs']),indent=2))
