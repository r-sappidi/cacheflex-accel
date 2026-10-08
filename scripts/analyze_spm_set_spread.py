"""Bounded residency-window evidence, never an additive runtime attribution.

A qualified removal requires a retired coherent load before eviction, then
reallocation of the same physical line in that private cache, and a later
retired coherent load. Speculative accesses alone do not establish useful reuse.
Incomplete histories at either window boundary are conservatively excluded.
"""
from collections import Counter
import gzip
import json
from pathlib import Path
import re
import sys

LINE = re.compile(r'^(\d+): ([^:]+): CFRES (\w+) (.*)')
FIELDS = re.compile(r'(\w+)=([^\s]+)')
CORE = re.compile(r'system\.(?:cpu|ruby\.l[01]_cntrl)(\d+)')
PCS = {0x401980:'packed_K',0x4019e0:'packed_K',0x4019f0:'packed_Q',
       0x401d08:'O_rescale',0x40124c:'packer_stack',0x402300:'V_cache',
       0x401f04:'V_spm'}


def analyze(path):
    counts=Counter();removals=Counter();qualified=Counter();sets=Counter()
    translations={};last_read={};pending={};requests={};latency={}
    first=None;last=None
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt') as f:
        for line in f:
            match=LINE.match(line)
            if not match:continue
            tick=int(match[1]);name,kind=match[2],match[3]
            if first is None:first=tick
            last=tick;counts[kind]+=1
            d=dict(FIELDS.findall(match[4]))
            for k,v in list(d.items()):
                try:d[k]=int(v,0)
                except ValueError:pass
            core_match=CORE.match(name)
            if not core_match:continue
            core=int(core_match[1])
            if kind=='ISSUE':
                requests[core,d['id']]=(tick,d)
                if 'va' in d and 'addr' in d:
                    translations[core,d['va']//64]=(d['addr']//64,tick)
            elif kind=='DONE':
                old=requests.pop((core,d['id']),None)
                if old and old[1].get('addr')==d.get('addr') and old[1].get('pc')==d.get('pc'):
                    label=PCS.get(d['pc'],'other')+':'+str(d.get('type'))+':'+str(d.get('source'))
                    record=latency.setdefault(label,dict(count=0,total_ticks=0,max_ticks=0))
                    dt=tick-old[0];record['count']+=1;record['total_ticks']+=dt
                    record['max_ticks']=max(record['max_ticks'],dt)
            elif kind=='REMOVE' and re.search(r'ruby\.l1_cntrl\d+\.cache$',name):
                key=core,d['addr']//64;removals[str(d['set'])]+=1
                if key in last_read:
                    prior=last_read[key]
                    pending[key]=dict(tick=tick,set=d['set'],before_pc=prior[1],refilled=False)
            elif kind=='ALLOC' and re.search(r'ruby\.l1_cntrl\d+\.cache$',name):
                if d.get('spm')==1:sets[f"{core}:{d['set']}:{d['way']}"]+=1
                key=core,d['addr']//64
                if d.get('spm')==0 and key in pending:pending[key]['refilled']=True
            elif kind=='RETIRED' and not d.get('spm') and not d.get('store'):
                va=d['va'];size=d['bytes']
                for vline in range(va//64,(va+size-1)//64+1):
                    trans=translations.get((core,vline))
                    if not trans:continue
                    key=core,trans[0]
                    event=pending.pop(key,None)
                    if event and event['refilled'] and trans[1]>=event['tick']:
                        label=PCS.get(d['pc'],hex(d['pc']))
                        qualified[f"set{event['set']}:{label}"]+=1
                    last_read[key]=(tick,d['pc'])
    for r in latency.values():r['mean_cpu_cycles']=r['total_ticks']/r['count']/400
    return dict(window_first_tick=first,window_last_tick=last,events=dict(counts),
                private_l2_removals_by_set=dict(removals),
                qualified_removal_refill_reuse_by_set_pc=dict(qualified),
                observed_spm_allocations=dict(sets),request_lifetimes=latency,
                censored_requests=len(requests),censored_removals=len(pending),
                caveats=['Bounded window, not whole-run coverage or causal runtime percentage.',
                         'VA-to-PA mapping from observed fragments; no context switches in SE workload.',
                         'PC labels apply only to frozen model_tuning guest.export-safe.',
                         'Request lifetimes overlap; do not sum them into removable execution time.'])

if __name__=='__main__':print(json.dumps(analyze(Path(sys.argv[1])),indent=2))
