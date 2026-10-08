import os
from pathlib import Path
R=Path(__file__).resolve().parents[4]
# mode, async, intervention flags, passive observer
ARMS={'cache_off':(False,0),'spm_off':(True,0),'cache':(False,1),'spm':(True,1)}
def cases():
 arm=os.environ['FA49_ARM'];spm,observer=ARMS[arm]
 return [dict(name=f't{t}_{arm}',kind='fa',t=t,spm=spm,variant='control',mode=0,async_pv=0,flags=2,observer=observer,arm=arm) for t in (256,1024)]
def parameters(variant,spm):return {f'system.ruby.l1_cntrl{i}.cache.num_spm_ways':6 if spm else 0 for i in range(8)}
def argv(c):return [c['t'],8,16,256,16,1024,0,int(c['spm']),1,1,14,2,1,int(c['spm']),0,0,c['mode'],c['async_pv'],c['flags']]
def control_path(c):
 backend='spm' if c['spm'] else 'cache'
 if c['t']==256:return R/f'results/cacheflex_flash_attention/bottleneck_20261004/experiment_v1/runs/gate_{backend}_control'
 if c['t']==1024:return R/f'results/cacheflex_flash_attention/bottleneck_20261004/screening_t1024_v1/runs/screen_{backend}_control'
 return R/f'results/cacheflex_flash_attention/bottleneck_20261004/targets_continuation_v1/runs/target_{backend}_control'
def work(t,hq=14):
 tasks=[0]*8;updates=[0]*8;copies=0;groups=t//256
 for h in range(hq):
  for group in range(groups):
   core=(h*groups+(group-h)%groups)%8;end=(group+1)*256
   tasks[core]+=1;updates[core]+=256*(end//256);copies+=2*end
 return tasks,updates,copies

def service(mode,flags=0):
 if flags&2:return 0
 if mode<5 or mode>8:return 0
 lanes=64 if mode==5 else 256 if mode==7 else 128
 return 262144//lanes+32+8192//32+32768//32+2048//32+(43008//16+40 if mode==8 else 0)
