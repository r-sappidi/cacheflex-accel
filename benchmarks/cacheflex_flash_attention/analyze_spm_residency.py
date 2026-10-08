"""SPM phase/buffer labels and protocol/retired-read qualification of removals.

Uses a symbol-paired rebuilt SPM guest, never mismatched legacy DWARF.
Any overlapping retired store makes old-content reuse ambiguous: predication
and byte masks are deliberately not guessed. This is not causal stall timing.
"""
from pathlib import Path
import csv,json,re,sys
from collections import Counter,defaultdict
import analyze_residency as base

def spm_buffers(log,t,d=64,br=64,bc=128):
    result=[]
    shared=dict(Q=8*t*d*2,K=t*d*2,V=t*d*2,O=8*t*d*2,KT=t*d*2)
    local=dict(qap=t*d*2,S=br*bc*2,P=br*bc*2,PV=8*d*2,pap=8*bc*2,
               Oacc=t*d*4,m=t*4,l=t*4)
    for line in log.splitlines():
        if not line.startswith(('FLASH_SHARED ','FLASH_ADDR ')):continue
        fields=dict(re.findall(r'(\w+)=(0x[0-9a-f]+|\d+)',line));sizes=local if 'tid' in fields else shared
        for name,size in sizes.items():
            if name in fields:
                va=int(fields[name],0);label=f"core{fields['tid']}.{name}" if 'tid' in fields else name
                result.append((va,va+size,label))
    assert len(result)==5+8*8, len(result)
    return result

def spm_phase(frames):
    names=' '.join(f for f,_ in frames)
    if 'spm_kernel_1vl_vl16' in names:return 'QK'
    if 'cache_pv_8x64_vl16' in names:return 'PV'
    if 'exp' in names or 'scale_f32' in names:return 'softmax_rescale'
    if 'acc_f16' in names:return 'PV_accumulate'
    if 'pack_A' in names:return 'pack_Q_or_P_shared'
    if 'stage_kv' in names:return 'stage_K'
    if 'barrier' in names or 'futex' in names:return 'synchronization'
    for _,loc in frames:
        m=re.search(r'spm_multicore.cpp:(\d+)',loc)
        if not m:continue
        n=int(m[1])
        if 328<=n<=355:return 'QK'
        if 356<=n<=400:return 'softmax_rescale'
        if 401<=n<=428:return 'PV_pack_accumulate'
        if 294<=n<=327:return 'stage_K_or_panel_control'
        if n==452:return 'final_normalize'
    return 'unresolved_or_shared'

def qualify(run):
    # Numerical validation/printing reads output after the measured region.
    # Those reads must not turn an ROI eviction into purported useful reuse.
    end_tick=json.loads((run/'residency.json').read_text())['bounds']['END']
    with (run/'evictions.csv').open() as stream:
        rows=list(csv.DictReader(stream))
    byevent=defaultdict(list)
    for row in rows:
        row.update(protocol_event='',protocol_state='',retired_read_tick='',prior_store=False)
        byevent[(int(row['tick']),row['cache'],int(row['addr'],16))].append(row)
    pages={};pending={};outcomes=Counter()
    # ProtocolTrace is emitted after the actions, at the same tick/address.
    proto=re.compile(r'^\s*(\d+)\s+(\d+)\s+(L[012]Cache)\s*(\S+)\s+(\S+)>(\S+)\s+\[?(0x[0-9a-f]+)')
    event=re.compile(r'\s*(\d+):\s*(.*?)CFRES (\w+)\s*(.*)')
    with (run/'debug.trace').open() as stream:
        for line in stream:
            m=proto.match(line)
            if m:
                tick,c,level,reason,old,new,addr=m.groups();addr=int(addr,16)
                if int(tick)>end_tick:break
                names={'L0Cache':[f'system.ruby.l0_cntrl{c}.Dcache',f'system.ruby.l0_cntrl{c}.Icache'],
                       'L1Cache':[f'system.ruby.l1_cntrl{c}.cache'],
                       'L2Cache':[f'system.ruby.l2_cntrl{c}.L2cache']}[level]
                for cache in names:
                    for row in byevent.get((int(tick),cache,addr),[]):
                        row['protocol_event']=reason;row['protocol_state']=old+'>'+new
                continue
            m=event.match(line)
            if not m:continue
            tick,owner,kind,tail=m.groups();tick=int(tick);owner=owner.strip().rstrip(':')
            if tick>end_tick:break
            f=dict(re.findall(r'(\w+)=([^\s]+)',tail))
            if kind=='MAP':
                va,pa,size=(int(f[k],0) for k in ('va','pa','bytes'))
                for off in range(0,size,4096):pages[va+off]=pa+off
            elif kind=='REMOVE':
                key=(owner,int(f['addr'],0))
                for row in byevent.get((tick,*key),[]):pending[key]=row
            elif kind=='RETIRED' and f['spm']=='0':
                match=re.search(r'cpu(\d+)',owner)
                if not match:continue
                c=match[1];va,size=int(f['va'],0),int(f['bytes'],0)
                if not size:continue
                for address in range(va&~63,((va+size-1)&~63)+1,64):
                    if address&~4095 not in pages:continue
                    pa=pages[address&~4095]+(address&4095)
                    caches=[f'system.ruby.l0_cntrl{c}.Dcache',f'system.ruby.l1_cntrl{c}.cache']
                    caches += [f'system.ruby.l2_cntrl{i}.L2cache' for i in range(8)]
                    for cache in caches:
                        key=(cache,pa);row=pending.get(key)
                        if row is None:continue
                        if f['store']=='1':row['prior_store']=True
                        else:
                            row['retired_read_tick']=tick;pending.pop(key)
    for row in rows:
        if not row['retired_read_tick']:classification='no_observed_retired_read'
        elif row['prior_store']:classification='write_before_read_ambiguous'
        elif row['next_read_tick'] and int(row['next_read_tick'])<=int(row['retired_read_tick']):
            classification='post_removal_request_and_retired_read_without_store'
        else:classification='retired_read_without_post_removal_request'
        row['reuse_class']=classification
        outcomes[(row['cache'],row['buffer'],row['protocol_event'] or 'unresolved',classification)]+=1
    with (run/'qualified_removals.csv').open('w') as stream:
        if rows:
            w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    result=dict(counts={'|'.join(k):v for k,v in outcomes.items()},roi_end_tick=end_tick,
                unresolved_protocol=sum(not r['protocol_event'] for r in rows),total_removals=len(rows),
                limitations=['Retirement and issue are matched by line/time, not dynamic instruction ID.',
                  'Any store overlap is ambiguous; byte enables are not inferred.',
                  'A later retired read is not an additive causal stall penalty.'])
    (run/'qualified_removals.json').write_text(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    run=Path(sys.argv[1]);variant=sys.argv[4]
    if variant=='spm':base.buffers=spm_buffers;base.phase=spm_phase
    base.analyze(run,Path(sys.argv[2]),int(sys.argv[3]));qualify(run)
