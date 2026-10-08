from pathlib import Path
R=Path(__file__).resolve().parents[4]
VARIANTS=('control','bridge96','bridge4096','copy_request2','copy_oracle','load_latency3','load_latency1','load_ports2','load_ports8','combined')
def parameters(variant,spm):
 p={f'system.ruby.l1_cntrl{i}.cache.num_spm_ways':6 if spm else 0 for i in range(8)}
 for i in range(8):
  if variant=='copy_request2':p[f'system.ruby.l0_cntrl{i}.spm_copy_request_latency']=2
  if variant in ('load_latency3','load_latency1','combined'):p[f'system.ruby.l1_cntrl{i}.spm_access_response_latency']=3 if variant=='load_latency3' else 1
  if variant in ('load_ports2','load_ports8','combined'):p[f'system.cpu[{i}].spmLoadPorts']=2 if variant=='load_ports2' else 8
 return p
def overrides(variant):
 return dict(CACHEFLEX_COPY_SHADOW_BUDGET='96' if variant=='bridge96' else '4096' if variant in ('bridge4096','combined') else '0',CACHEFLEX_ISOLATE_COPY_ORACLE='1' if variant in ('copy_oracle','combined') else '0')
def cases():
 rows=[dict(name='exact_private',kind='exact',t=256,spm=True,variant='control')]
 for variant in ('bridge96','bridge4096','copy_oracle','combined'):
  for gate in ('spread_roundtrip_raw','spread_release_getx'):
   rows.append(dict(name=variant+'_'+gate,kind='directed',variant=variant,source=str(R/'results/retained_protocol_restart_20261002/gates/runs'/gate)))
 for t,label in [(256,'gate'),(1024,'screen')]:
  rows += [dict(name=f'{label}_cache_{v}',kind='fa',t=t,spm=False,variant=v) for v in ('control','combined')]
  rows += [dict(name=f'{label}_spm_{v}',kind='fa',t=t,spm=True,variant=v) for v in VARIANTS]
 # Confirm the main staging headroom hypotheses at target size, without a full sweep.
 rows.append(dict(name='target_cache_control',kind='fa',t=4096,spm=False,variant='control'))
 rows += [dict(name=f'target_spm_{v}',kind='fa',t=4096,spm=True,variant=v) for v in ('control','bridge4096','copy_oracle','combined')]
 return rows
def work(t,hq=14):
 tasks=[0]*8;updates=[0]*8;copies=0;groups=t//256
 for task in range(hq*groups):
  head=task//groups;g=(task%groups+head)%groups
  tasks[task%8]+=1;updates[task%8]+=256*(g+1);copies+=512*(g+1)
 return tasks,updates,copies
def argv(c):return [c['t'],8,16,256,16,1024,0,int(c['spm']),1,1,14,2,1,int(c['spm'])]
