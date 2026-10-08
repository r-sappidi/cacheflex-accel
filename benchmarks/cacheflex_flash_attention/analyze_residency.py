"""Baseline native-VL16 FA residency/reuse trace. No causal cycle summation.

Usage: analyze_residency.py RUN DEBUG_ELF T
Trace begins before initialization to capture page mappings and initial tags.
"""
from pathlib import Path
from collections import Counter, defaultdict
import csv
import json
import re
import sys
from analyze_attribution import symbols

def events(path):
    with path.open() as stream:
        for line in stream:
            if 'CFRES ' not in line:
                continue
            match = re.match(r'\s*(\d+):\s*(.*?)CFRES (\w+)\s*(.*)', line)
            if match:
                tick, owner, kind, tail = match.groups()
                yield int(tick), owner.strip().rstrip(':'), kind, dict(re.findall(r'(\w+)=([^\s]+)', tail))

def number(value):
    return int(value, 0)

def phase(frames):
    names = ' '.join(n for n, _ in frames)
    if 'sve_mla_8x3VL' in names: return 'QK'
    if 'sve_cache_pv' in names or 'cache_pv_8x64_vl16' in names: return 'PV'
    if 'exp' in names or 'scale_f32' in names: return 'softmax_rescale'
    if 'acc_f16' in names: return 'PV_accumulate'
    if 'pack_A' in names: return 'pack_Q_or_P_shared'
    if 'transpose_K' in names or 'prepack_B' in names: return 'prepare_K'
    if 'barrier' in names or 'futex' in names: return 'synchronization'
    for _, loc in frames:
        m = re.search(r'cache_flash_attn.cpp:(\d+)', loc)
        if not m: continue
        n = int(m[1])
        if 640 <= n <= 677: return 'QK'
        if 678 <= n <= 709: return 'softmax_rescale'
        if 710 <= n <= 738: return 'PV_pack_accumulate'
        if 739 <= n <= 760: return 'final_normalize'
        if 622 <= n <= 639: return 'query_initialize'
    return 'unresolved_or_shared'

def buffers(log, t, d=64, br=64, bc=128):
    result=[]
    shared=dict(Q=8*t*d*2,K=t*d*2,V=t*d*2,O=8*t*d*2,KT=t*d*2,Kpack=((t+127)//128)*128*d*2)
    local=dict(qtile=br*d*2,spanel=br*bc*2,ptile=br*bc*2,pvpanel=8*d*2,
               qkap=br*d*2,pvap=8*bc*2,oacc=br*d*4,m=br*4,l=br*4)
    for line in log.splitlines():
        if not line.startswith(('FLASH_SHARED ', 'FLASH_ADDR ')): continue
        fields=dict(re.findall(r'(\w+)=(0x[0-9a-f]+|\d+)',line))
        sizes=local if 'tid' in fields else shared
        for name,size in sizes.items():
            if name in fields:
                start=number(fields[name]);label=name if 'tid' not in fields else f"core{fields['tid']}.{name}"
                result.append((start,start+size,label))
    assert len(result)==6+8*9, ('missing buffer ranges',len(result))
    return result

class TimeWeightedCounter(Counter):
    """Lazy, exact integer integration: update only the changed buffer key."""
    def __init__(self,area,start):
        self.area=area;self.now=start;self.since={}
        super().__init__()
    def __setitem__(self,key,value):
        self.area[key]+=self[key]*(self.now-self.since.get(key,self.now))
        self.since[key]=self.now
        super().__setitem__(key,value)
    def advance(self,tick):
        assert tick>=self.now
        self.now=tick
    def flush(self):
        for key,value in self.items():self[key]=value

def analyze(run, elf, t, fast_occupancy=False):
    trace=run/'debug.trace';pcs=set();bounds={}
    for tick,owner,kind,f in events(trace):
        if 'pc' in f: pcs.add(number(f['pc']))
        if kind in ('BEGIN','END'): bounds[kind]=tick
    assert bounds.get('END',0)>bounds.get('BEGIN',0)
    stacks=symbols(elf,sorted(pcs));phases={pc:phase(fr) for pc,fr in stacks.items()}
    (run/'residency_pc_map.json').write_text(json.dumps({hex(pc):dict(phase=phases[pc],frames=fr) for pc,fr in stacks.items()},indent=2))
    ranges=buffers((run/'run.log').read_text(),t)
    pages={};resident={};area=Counter();last_tick=bounds['BEGIN']
    occupancy=TimeWeightedCounter(area,last_tick) if fast_occupancy else Counter()
    entries={};permissions={};accessible_area=Counter()
    accessible=TimeWeightedCounter(accessible_area,last_tick) if fast_occupancy else Counter()
    heads={};probes={};removals=[];last_removed={};pending={};responses=Counter();reuse=Counter()
    def label(pa):
        page=pa&~4095
        if page not in pages:return 'unmapped'
        va=pages[page]+(pa&4095)
        return next((name for a,b,name in ranges if a<=va<b),'other')
    def core(owner):
        m=re.search(r'(?:cpu|cntrl)(\d+)',owner)
        return int(m[1]) if m else None
    def integrate(tick):
        nonlocal last_tick
        now=min(max(tick,bounds['BEGIN']),bounds['END'])
        delta=now-last_tick
        if delta>0:
            if fast_occupancy:
                occupancy.advance(now);accessible.advance(now)
            else:
                for key,count in occupancy.items():area[key]+=count*64*delta
                for key,count in accessible.items():accessible_area[key]+=count*64*delta
            last_tick=now
    for tick,owner,kind,f in events(trace):
        if tick>bounds['END']:break
        active=bounds['BEGIN']<=tick<=bounds['END'];c=core(owner)
        if kind=='MAP':
            va,pa,size=number(f['va']),number(f['pa']),number(f['bytes'])
            for offset in range(0,size,4096):pages[pa+offset]=va+offset
        elif kind=='HEAD':heads[c]=phases.get(number(f['pc']),'unresolved_or_shared')
        elif kind=='STATE' and f['entry'] in entries:
            integrate(tick);key=entries[f['entry']]
            if key in resident:
                old=permissions.get(key,'Invalid') in ('Read_Only','Read_Write')
                new=f['permission'] in ('Read_Only','Read_Write')
                accessible[(key[0],resident[key])]+=int(new)-int(old)
                permissions[key]=f['permission']
        elif kind in ('ALLOC','REMOVE'):
            integrate(tick);addr=number(f['addr']);key=(owner,addr)
            if kind=='ALLOC':
                if key in resident:occupancy[(owner,resident.pop(key))]-=1
                name='SPM' if f['spm']=='1' else label(addr)
                resident[key]=name;occupancy[(owner,name)]+=1
                if 'entry' in f:entries[f['entry']]=key
                permissions[key]='Invalid'
            else:
                name=resident.pop(key,label(addr));occupancy[(owner,name)]-=1
                if permissions.pop(key,'Invalid') in ('Read_Only','Read_Write'):
                    accessible[(owner,name)]-=1
                if active:
                    candidate=probes.pop(key,None)
                    row=dict(tick=tick,cache=owner,addr=hex(addr),buffer=name,
                             head_phase=heads.get(c,'unresolved_or_shared'),
                             replacement_candidate=bool(candidate),
                             incoming=hex(candidate[1]) if candidate else '',
                             incoming_buffer=label(candidate[1]) if candidate else '',
                             permission=f['permission'],next_read_tick='',reuse_phase='')
                    removals.append(row);last_removed[key]=row
        elif kind=='PROBE':probes[(owner,number(f['victim']))]=(tick,number(f['incoming']))
        elif kind=='ISSUE' and active:
            addr=number(f['addr']);typ=f['type'];pc=number(f['pc'])
            # CPU read requests only. Stores, SPM operations and instruction
            # fetches are not evidence that old buffer contents are useful.
            reads=typ in ('LD','Load_Linked')
            affected=[]
            if reads:
                first=addr&~63;last=(addr+number(f['bytes'])-1)&~63
                for line in range(first,last+1,64):
                    caches=[f'system.ruby.l0_cntrl{c}.Dcache',f'system.ruby.l1_cntrl{c}.cache']
                    caches += [f'system.ruby.l2_cntrl{i}.L2cache' for i in range(8)]
                    for cache in caches:
                        key=(cache,line)
                        if key in last_removed:
                            row=last_removed.pop(key);row['next_read_tick']=tick
                            row['reuse_phase']=phases.get(pc,'unresolved_or_shared')
                            reuse[(cache,row['buffer'])]+=1;affected.append(row)
            pending.setdefault((owner,f['id']),dict(tick=tick,buffer=label(addr),reads=reads,reused=bool(affected)))
        elif kind=='DONE' and active:
            req=pending.pop((owner,f['id']),None)
            if req and req['reads']:
                key=(req['buffer'],f['source'],str(req['reused']))
                responses[key+('count',)]+=1
                responses[key+('summed_service_ticks',)]+=tick-req['tick']
    integrate(bounds['END'])
    if fast_occupancy:
        occupancy.flush();accessible.flush()
        # Lazy counters integrate cache-line ticks, unlike the legacy byte-tick loop.
        for key in area:area[key]*=64
        for key in accessible_area:accessible_area[key]*=64
    assert min(occupancy.values(),default=0)>=0, 'unmatched cache removal'
    with (run/'evictions.csv').open('w') as stream:
        if removals:
            writer=csv.DictWriter(stream,fieldnames=list(removals[0]));writer.writeheader();writer.writerows(removals)
    counts=Counter((r['cache'],r['buffer'],r['head_phase'],str(r['replacement_candidate']),str(bool(r['next_read_tick']))) for r in removals)
    result=dict(roi_ticks=bounds['END']-bounds['BEGIN'],bounds=bounds,
        removals={'|'.join(k):v for k,v in counts.items()},
        average_allocated_bytes={'|'.join(k):v/(bounds['END']-bounds['BEGIN']) for k,v in area.items()},
        average_accessible_bytes={'|'.join(k):v/(bounds['END']-bounds['BEGIN']) for k,v in accessible_area.items()},
        read_responses={'|'.join(k):v for k,v in responses.items()},
        limitations=['Allocated tags include invalid/transient entries; not an exact valid-data occupancy measure.',
          'A PROBE is a candidate selection, not itself a completed eviction.',
          'Later reads establish reuse, not that eviction delayed the CPU: prefetch/refill can hide it.',
          'Permission does not distinguish clean-exclusive from dirty-modified lines.',
          'ROB-head phase is attribution, not an exact nonoverlapping algorithm phase.',
          'Read service times overlap and must not be summed into an elapsed-time penalty.',
          'Coalesced requests without ISSUE and reads after ROI are excluded.'])
    (run/'residency.json').write_text(json.dumps(result,indent=2));return result

if __name__=='__main__':analyze(Path(sys.argv[1]),Path(sys.argv[2]),int(sys.argv[3]))
