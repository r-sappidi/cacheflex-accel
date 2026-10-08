"""Accepted-only Qwen component accounting. Missing components are never zero."""
import argparse,fcntl,json,os,time
from pathlib import Path
ROOT=Path('/data2/rsappidi/cacheflex_mc')
from build_access_timing import save,sha
SOURCES={
 'fa':ROOT/'results/cacheflex_l3_spm/direct_link_20260928/experiment_v2',
 'swiglu':ROOT/'results/ai_workloads/qwen_swiglu_20260928/experiment_v1',
 'projection':ROOT/'results/ai_workloads/qwen_projection_20260928/experiment_v2'}
OUT=ROOT/'results/ai_workloads/qwen_composition_20260928'
BACKENDS=('cache','private_spm','spm_l3_noc','spm_l3_direct')
COMPONENTS=('qkv_with_bias_and_head_layout','rope','attention','output_projection_with_head_merge','rmsnorm_attention','rmsnorm_ffn','residual_attention','residual_ffn','swiglu')

def accepted(root):
 s=json.loads((root/'status.json').read_text());rows=s.get('completed',[])
 for r in rows:
  p=Path(r['path'])/'accepted.json'
  assert json.loads(p.read_text())==r,('Unaccepted or changed record',p)
 return s,rows

def pick(rows,predicate):
 eligible=[r for r in rows if predicate(r)]
 return min(eligible,key=lambda r:r['first_us']) if eligible else None

def source_record(row,metric):
 if row is None:return None
 p=Path(row['path'])/'accepted.json'
 return dict(us=row[metric],source=str(p),sha256=sha(p),case=row['case'],metric=metric)

def collect(sources=SOURCES):
 states={};rows={}
 for k,p in sources.items():states[k],rows[k]=accepted(p)
 result=dict(updated=time.time(),policy='First invocation including component-internal packing/staging; no warm-call substitution.',
  model=dict(name='Qwen2.5-0.5B-Instruct-shaped',layers=24,batch=1,hidden=896,intermediate=4864,hq=14,hkv=2,head_dim=64),
  precision='FA: FP16 GEMMs/FP32 online state. Projections/FFN: FP16 storage and FP32 accumulation. Not BF16 checkpoint inference.',
  stages={k:v['stage'] for k,v in states.items()},layers=[],model_projection_ready=False,
  model_extras=dict(embedding=None,final_rmsnorm=None,last_position_logits=None),
  integration_validated=False,cache_state_policy_validated=False)
 for t in (1024,2048,4096):
  for backend in BACKENDS:
   arm='cache' if backend=='cache' else 'spm'
   fa_arm=dict(cache='cache',private_spm='private',spm_l3_noc='noc',spm_l3_direct='direct')[backend]
   fa=next((r for r in rows['fa'] if r['name']==f't{t}_{fa_arm}'),None)
   ff=pick(rows['swiglu'],lambda r:r['case']['t']==t and r['case']['arm']==arm and r['case']['stage'] in ('screen_t1024',f'target_t{t}'))
   projs={op:pick(rows['projection'],lambda r:r['case']['t']==t and r['case']['arm']==arm and r['case']['op']==op and r['case']['stage'] in ('screen_t1024',f'target_t{t}')) for op in ('qkv','out')}
   components=dict.fromkeys(COMPONENTS)
   components.update(attention=source_record(fa,'roi_us'),swiglu=source_record(ff,'first_us'),
    qkv_with_bias_and_head_layout=source_record(projs['qkv'],'first_us'),output_projection_with_head_merge=source_record(projs['out'],'first_us'))
   missing=[k for k,v in components.items() if v is None]
   result['layers'].append(dict(t=t,backend=backend,components=components,missing=missing,
    measured_component_subtotal_us=sum(v['us'] for v in components.values() if v is not None),
    layer_projection_us=None,model_projection_us=None,
    selection_status='best accepted so far; search may still be incomplete',
    non_attention_policy='SPM+L3 reuses private-SPM projection/FFN times; no extra L3 benefit assumed.'))
 return result

def write(result):
 OUT.mkdir(parents=True,exist_ok=True)
 save(OUT/'components.json',result)
 lines=['# Qwen component accounting — accepted results only','',result['policy'],'',
  'This is an incomplete accounting sheet, not an end-to-end speedup estimate. Missing entries are unknown, never zero. Layer/model totals remain withheld until remaining operators and integration/cache-state policy are validated.','',
  '| T | Backend | QKV + bias/layout us | FA us | Output + merge us | SwiGLU us | Missing components |',
  '|---|---|---:|---:|---:|---:|---:|']
 for r in result['layers']:
  c=r['components'];keys=('qkv_with_bias_and_head_layout','attention','output_projection_with_head_merge','swiglu')
  vals=['—' if c[k] is None else f"{c[k]['us']:.4f}" for k in keys]
  lines.append(f"| {r['t']} | {r['backend']} | "+' | '.join(vals)+f" | {len(r['missing'])} |")
 lines+=['','Missing layer operators: two RMSNorms, two residual additions, RoPE, plus any not-yet-accepted target component above. Model extras: embedding, final RMSNorm, last-position vocabulary projection.','',
  'QKV includes bias and conversion to Q14/K2/V2 head-major layout. Output projection includes FA head-major to token-major conversion. FA includes its own tile packing/staging/normalization/drain; SwiGLU includes SiLU-product and its three GEMMs. Do not add these costs twice.','',
  'Pending: common cache-state/weight-prepacking policy; integrated layer check; activation compatibility (FA inputs are currently independent synthetic patterns); hardware-build equivalence provenance. A sum of isolated first calls is a projection under that boundary, not measured layer execution.','',
  'No T16K/T32K model estimate is extrapolated from shorter FFN/projection cases. Best-so-far candidate selection is provisional until the relevant search completes. Per-record sources and SHA256 hashes are in components.json.']
 (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--watch',action='store_true');args=ap.parse_args()
 OUT.mkdir(parents=True,exist_ok=True)
 with (OUT/'collector.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:
   while True:
    result=collect();write(result)
    save(OUT/'watcher_status.json',dict(pid=os.getpid(),updated=time.time(),
      stage='watching' if args.watch else 'complete',sources=result['stages']))
    if not args.watch:break
    time.sleep(60)
  except BaseException as exc:
   save(OUT/'watcher_status.json',dict(pid=os.getpid(),updated=time.time(),stage='failed',error=repr(exc)))
   raise
