"""FA-39 paired attribution; changes to shared resources always have cache controls."""
import os
from pathlib import Path
R=Path(__file__).resolve().parents[4]
def parameters(variant,spm):
 return {f'system.ruby.l1_cntrl{i}.cache.num_spm_ways':6 if spm else 0 for i in range(8)}
def cases():
 arm=os.environ['FA45_ARM'];assert arm in ('cache','cache_stage','burst','stage','paced','combined')
 return [dict(name=f't{t}_{arm}',kind='fa',t=t,spm=arm not in ('cache','cache_stage'),variant='stage' if arm in ('stage','combined','cache_stage') else arm,schedule=int(arm in ('paced','combined'))) for t in (256,512,4096)]
def control_path(c):
 backend='spm' if c['spm'] else 'cache'
 if c['t']==256:return R/f'results/cacheflex_flash_attention/bottleneck_20261004/experiment_v1/runs/gate_{backend}_control'
 if c['t']==4096:return R/f'results/cacheflex_flash_attention/bottleneck_20261004/targets_continuation_v1/runs/target_{backend}_control'
 if c['t']==1024:return R/f'results/cacheflex_flash_attention/bottleneck_20261004/screening_t1024_v1/runs/screen_{backend}_control'
 return None
def work(t,hq=14):
 tasks=[0]*8;updates=[0]*8;copies=0;groups=t//256
 for h in range(hq):
  for group in range(groups):
   core=(h*groups+(group-h)%groups)%8;end=(group+1)*256
   tasks[core]+=1;updates[core]+=256*(end//256);copies+=2*end
 return tasks,updates,copies
def argv(c):
 return [c['t'],8,16,256,16,1024,0,int(c['spm']),1,1,14,2,1,int(c['spm']),int(c['variant']=='stage'),int(c['variant']=='packed'),c['schedule']]
