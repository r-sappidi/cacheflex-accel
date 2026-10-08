"""Reconstruct actual pool occupancy/owners, not last-polled blocker durations."""
import argparse
import collections
import json
from pathlib import Path
from build_access_timing import save

CAPS={'system.ruby.l0_cntrl0.writeCredits':8,
      'system.ruby.l1_cntrl0.missCredits':20,
      'system.ruby.l1_cntrl0.writeCredits':8}


def coherent_queue_waits(events):
    types={(x['tick'],x['addr']):x['detail'] for x in events['protocol']
           if x['ctrl']=='Seq' and x['event']=='Begin'}
    service=collections.defaultdict(list)
    for x in events['protocol']:
        if x['ctrl']=='L0Cache' and x['event']=='Load' and not x['detail'].strip():
            service[x['tick'],x['addr']].append(x['state'])
    pending={};done=[]
    for x in events['queues']:
        if x['obj']!='system.ruby.l0_cntrl0.mandatoryQueue':continue
        ptr=x['msg'];a=int(x['addr'],16)
        if x['kind']=='QUEUE_ENQUEUE':
            pending[ptr]=dict(accepted=x['tick'],ready=int(x['ready']),addr=a,
                request_type=types.get((x['tick'],a&~63),'unknown'),
                evidence_line=x['evidence_line'],older_requests=len(pending),
                older_types=dict(collections.Counter(v['request_type'] for v in pending.values())),
                oldest_five_types=[v['request_type'] for v in list(pending.values())[:5]])
        elif x['kind']=='QUEUE_DEQUEUE' and ptr in pending:
            v=pending.pop(ptr)
            if v['request_type']!='LD':continue
            states=service.get((x['tick'],a&~63),[])
            if not any(s in ('M>M','S>S','E>E') for s in states):continue
            v.update(served=x['tick'],wait_cycles=(x['tick']-v['ready'])/400,
                     service_states=states)
            done.append(v)
    return done


def reconstruct(events, protocol, end):
    transitions=collections.defaultdict(list)
    for e in protocol:
        if not e['detail'].strip():
            transitions[e['tick'],e['ctrl'],e['addr']].append(e['event'])
        if e['ctrl']=='Seq' and e['event']=='Begin':
            transitions[e['tick'],'Seq',e['addr']].append(e['detail'].strip())
    output={}; allocation_rows=[]
    for pool,cap in CAPS.items():
        es=[e for e in events if e['obj']==pool]
        live=None; unknown=0; owners=collections.defaultdict(list);segments=[]
        prev=None; prev_snapshot=None
        def snapshot():
            rows=[dict(v,addr=a) for a,vs in owners.items() for v in vs]
            return dict(live=live,unknown=unknown,owners=rows)
        for e in es:
            t=e['tick'];reported=int(e['live']);kind=e['kind']
            if kind=='POOL_FULL':
                if live is None:
                    live=reported;unknown=live;prev=t;prev_snapshot=snapshot()
                assert reported==live,(pool,e,reported,live)
                continue
            if prev is not None and live==cap and t>prev:
                segments.append(dict(start=prev,end=t,**prev_snapshot))
            if live is None:
                live=reported-(1 if kind=='POOL_ALLOC' else -1 if kind=='POOL_RELEASE' else 0)
                unknown=live
            if kind=='POOL_ALLOC':
                a=int(e['addr'],16)
                ctrl='L1Cache' if '.l1_cntrl' in pool else 'L0Cache'
                kinds=transitions.get((t,ctrl,a),[]) or transitions.get((t,'Seq',a),[])
                role='|'.join(sorted(set(kinds))) if kinds else 'unclassified'
                item=dict(allocated=t,role=role,evidence_line=e['evidence_line'])
                if owners[a]:
                    item['ambiguous_same_address']=True
                    for prior in owners[a]:prior['ambiguous_same_address']=True
                owners[a].append(item);live+=1
                allocation_rows.append(dict(pool=pool,addr=a,**item))
            elif kind=='POOL_RELEASE':
                a=int(e['addr'],16)
                if owners[a]:
                    item=owners[a].pop(0)
                else:
                    assert unknown>0,(pool,e,'unmatched release')
                    unknown-=1
                live-=1
            assert reported==live,(pool,e,reported,live)
            assert live==unknown+sum(len(v) for v in owners.values())
            prev=t;prev_snapshot=snapshot()
        if prev is not None and live==cap and end>prev:
            segments.append(dict(start=prev,end=end,**prev_snapshot))
        output[pool]=segments
    return output,allocation_rows


def analyze(path):
    events=json.loads((path/'resource_events.json').read_text())
    panels=json.loads((path/'panels.json').read_text())
    end=json.loads((path/'validated_prefix.json').read_text())['final_tick']
    full,allocs=reconstruct(events['pools'],events['protocol'],end)
    coherent=coherent_queue_waits(events)
    results=[]
    for panel in panels['panels']:
        if not panel['complete'] or not all('forward' in c for c in panel['copies']):continue
        by_pool={};roles=collections.Counter();details=[]
        for pool,segments in full.items():
            ticks=unknown_ticks=0
            for c in panel['copies']:
                lo=c['accepted']+400;hi=c['forward']
                for seg in segments:
                    d=max(0,min(hi,seg['end'])-max(lo,seg['start']))
                    if not d:continue
                    ticks+=d
                    if seg['unknown']:unknown_ticks+=d
                    if pool.endswith('.missCredits'):
                        for o in seg['owners']:roles[o['role']]+=d
                        roles['boundary_unknown']+=seg['unknown']*d
            by_pool[pool]=dict(sum_copy_wait_full_cycles=ticks/400,
                               mean_copy_wait_full_cycles=ticks/400/256,
                               boundary_unknown_cycles=unknown_ticks/400)
        longest=max(panel['copies'],key=lambda c:c['forward']-c['accepted'])
        for pool,segs in full.items():
            for seg in segs:
                if seg['end']>longest['accepted']+400 and seg['start']<longest['forward']:
                    details.append(dict(pool=pool,**seg))
        total=sum(roles.values())
        examples=sorted((v for v in coherent if panel['first_accept']<=v['accepted']<=panel['last_ack']),
                        key=lambda v:-v['wait_cycles'])[:4]
        for ex in examples:
            bs=[b for b in events['blocks'] if b['obj']=='system.ruby.l0_cntrl0' and
                ex['ready']<=b['tick']<ex['served']]
            ex['intervening_rejection_counts']=dict(collections.Counter(b['resource'] for b in bs))
            ex['intervening_protocol_event_counts']=dict(collections.Counter(b['protocol_event'] for b in bs))
            ex['every_wait_cycle_observed']=sorted(set(b['tick'] for b in bs))==list(range(ex['ready'],ex['served'],400))
        results.append(dict(panel_index=panel['index'],first_src=panel['copies'][0]['src'],
                            by_pool=by_pool,
                            miss_credit_owner_weighted_share={k:v/total if total else 0 for k,v in roles.items()},
                            owner_share_note='Credit-time during sampled copy waiting; not fraction of kernel runtime',
                            longest_copy=longest,longest_copy_full_segments=details,
                            coherent_hit_wait_examples=examples))
    result=dict(panels=results,actual_pool_occupancy_consistency_passed=True,
                allocations=allocs, full_segments=full)
    save(path/'credit_analysis.json',result)
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('pair',type=Path);args=ap.parse_args()
    for w in (0,4):
        r=analyze(args.pair/f'w{w}')
        print('W',w)
        for p in r['panels']:
            print({k:v for k,v in p.items() if k not in ('longest_copy','longest_copy_full_segments')})
