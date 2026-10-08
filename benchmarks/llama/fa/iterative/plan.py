"""Llama FA: local, evidence-ordered geometry iteration, not a placement matrix."""
import math
LEVELS={'br':(16,32),'bc':(128,256,512),'r':(128,256,512)}
MAX_TARGETS=8  # per backend; reaching this is unresolved, never convergence
MIN_GAIN=.005

def case(spm,t=4096,br=16,bc=256,r=256):
 return dict(kind='fa',t=t,spm=spm,br=br,bc=bc,r=r,hq=32,hkv=8,causal=True)
def name(c):return f't{c["t"]}_{"spm" if c["spm"] else "cache"}_br{c["br"]}_bc{c["bc"]}_r{c["r"]}'
def argv(c):return [c['t'],8,c['br'],c['bc'],c['r']//c['br'],1024,0,int(c['spm']),1,1,32,8,1,int(c['spm'])]
def parameters(c):return {f'system.ruby.l1_cntrl{i}.cache.num_spm_ways':6 if c['spm'] else 0 for i in range(8)}
def describe(c):
 t,br,bc,r=c['t'],c['br'],c['bc'],c['r']
 assert c['hq']==32 and c['hkv']==8 and c['causal']
 assert br in LEVELS['br'] and bc in LEVELS['bc'] and r in LEVELS['r']
 assert t in (256,512,4096) and t%r==t%bc==r%br==0
 tasks=[0]*8;updates=[0]*8;copies=[0]*8;groups=t//r
 for h in range(32):
  for g in range(groups):
   core=(h*groups+(g-h)%groups)%8;end=math.ceil((g+1)*r/bc)*bc
   tasks[core]+=1;updates[core]+=r*(end//bc)
   if c['spm']:copies[core]+=2*end
 # Arrays allocated by legacy_worker: Qpacked,O,m/l,scores,prob,Ppacked,pv,alpha.
 scratch=128*r+256*r+8*r+6*br*bc+128*br+4*br
 return dict(tasks=tasks,updates=updates,copies=copies,spm_bytes=128*bc if c['spm'] else 0,
             task_scratch_bytes=scratch,computed_pairs=bc*sum(updates),useful_pairs=32*t*(t+1)//2,
             gemm_flops=256*bc*sum(updates),copy_payload_bytes=64*sum(copies),
             work_imbalance=max(updates)/(sum(updates)/8),
             note='exact structural work; scratch conservative allocated-array footprint, not live residency or latency')
def gate(c):return dict(c,t=max(256,c['r'],c['bc']))
def targets(rows,spm):return [x for x in rows if x['case']['spm']==spm and x['case']['t']==4096]
def winner(rows,spm):return min(targets(rows,spm),key=lambda x:x['roi_us'])
def choose(rows,spm):
 data=targets(rows,spm)
 if not data:return case(spm),dict(reason='Transferred d64 Qwen baseline; Llama working set/ownership needs measurement.',prior='FA-35/L3-11/12',status='initial')
 w=winner(rows,spm);c=w['case'];m=w['counters'];seen={name(x['case']) for x in data}
 pool=[]
 for axis,levels in LEVELS.items():
  idx=levels.index(c[axis])
  for j in (idx-1,idx+1):
   if j<0 or j>=len(levels):continue
   cand=dict(c,**{axis:levels[j]})
   if name(cand) in seen:continue
   old,new=describe(c),describe(cand)
   if axis=='r' and cand['r']>c['r']:
    reason='Amortize repeated V staging/task setup over more query rows; trade larger coherent Q/O state and causal padding.'
    priority=0 if spm and m['spm_sq_full']>0 else 3
    evidence=dict(spm_sq_full=m['spm_sq_full'],copies=old['copies'])
   elif axis=='r':
    reason='Reduce live Q/O footprint and padded causal work; accept more tasks and potentially more staging.'
    priority=1 if m['l1_credit_full']>0 else 4;evidence=dict(l1_credit_full=m['l1_credit_full'])
   elif axis=='bc' and cand['bc']<c['bc']:
    reason='Reduce K/score/probability panel footprint under coherent-load pressure; trade more softmax recurrence/packing steps.'
    priority=0 if m['l1_credit_full']>0 else 3;evidence=dict(l1_credit_full=m['l1_credit_full'],l1_misses=m['l1_misses'])
   elif axis=='bc':
    reason='Amortize softmax recurrence and panel setup; trade larger panels and possible extra causal arithmetic.'
    priority=4;evidence=dict(instructions=m['instructions'],l1_credit_full=m['l1_credit_full'])
   else:
    reason='Change producer/consumer batching while retaining 16-row GEMM microkernel; test instruction/layout overhead versus larger scratch.'
    priority=2;evidence=dict(instructions=m['instructions'],iq_full=m['iq_full'],icache_stall=m['icache_stall'])
   pool.append((priority,name(cand),cand,dict(axis=axis,reason=reason,observed=evidence,parent=w['name'],before=old,after=new,
       decision='Keep lowest accepted full ROI; <0.5% changes are marginal, not architectural evidence. Recenter around winner.',status='candidate')))
 if not pool:return None,dict(status='local_neighbors_measured',winner=w['name'],scope='Local Br16/32,Bc128/256/512,R128/256/512 only; placement and hardware fixed. Requires final human/agent review.')
 if len(data)>=MAX_TARGETS:return None,dict(status='budget_frontier_unresolved',winner=w['name'],untested=[x[1] for x in sorted(pool)])
 _,_,cand,why=min(pool,key=lambda x:(x[0],x[1]));return cand,why
