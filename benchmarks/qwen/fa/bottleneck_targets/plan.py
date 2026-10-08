"""FA-39 paired attribution; changes to shared resources always have cache controls."""
def parameters(variant,spm):
 p={f'system.ruby.l1_cntrl{i}.cache.num_spm_ways':6 if spm else 0 for i in range(8)}
 for i in range(8):
  if variant in ('l1','both'):p[f'system.ruby.l0_cntrl{i}.dataMissCredits.capacity']=16
  if variant in ('l2','both'):p[f'system.ruby.l1_cntrl{i}.missCredits.capacity']=40
 return p
def cases():
 rows=[]
 for t,label in [(4096,'target')]:
  for variant in ('control','stage','l1','l2','both'):
   for spm in (False,True):
    rows.append(dict(name=f'{label}_{"spm" if spm else "cache"}_{variant}',kind='fa',t=t,spm=spm,variant=variant))
 return rows
def work(t,hq=14):
 tasks=[0]*8;updates=[0]*8;copies=0;groups=t//256
 for h in range(hq):
  for group in range(groups):
   core=(h*groups+(group-h)%groups)%8;end=(group+1)*256
   tasks[core]+=1;updates[core]+=256*(end//256);copies+=2*end
 return tasks,updates,copies
def argv(c):
 return [c['t'],8,16,256,16,1024,0,int(c['spm']),1,1,14,2,1,int(c['spm']),int(c['variant']=='stage'),int(c['variant']=='packed')]
