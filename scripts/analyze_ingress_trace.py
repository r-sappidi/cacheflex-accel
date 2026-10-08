"""Join CPU copy identities, destination slots, protocol events and consumers.

All timestamps in gem5 ticks (400 ticks/core cycle). Boundary-censored data
are explicitly omitted. Aggregate blocker labels are not continuous occupancy.
"""
import argparse
import collections
import gzip
import json
from pathlib import Path
import re

from build_access_timing import save

PROTO=re.compile(r'^\s*(\d+)\s+(\d+)\s+(Seq|L0Cache|L1Cache|L2Cache)\s+(\S+)\s+(\S+)\s+\[0x[0-9a-f]+, line (0x[0-9a-f]+)\]\s*(.*)')
CPU=re.compile(r'^(\d+): system.cpu0\.(.*?): (.*)')
CFT=re.compile(r'^(\d+): (.*?): CFT (\w+) (.*)')
BOUNDARIES={'0x401e00':'stage', '0x401ed8':'QK', '0x402840':'softmax',
            '0x402abc':'PV', '0x402678':'normalize', '0x402764':'final_sync'}


def analyze(path):
    copies={}; live={}; loads={}; stages=[]; phases=[]; reserves={}; sent={}
    protocol_samples=[]; pool_events=[]; resource_blocks=[]; queue_events=[]
    with gzip.open(path/'trace.gz','rt') as stream:
        for line_no,line in enumerate(stream,1):
            m=CPU.match(line)
            if m:
                tick=int(m[1]); text=m[3]
                snm=re.search(r'SN=(\d+)',text)
                sn=int(snm[1]) if snm else None
                addr=re.search(r'Addr=(0x[0-9a-f]+|0)',text)
                if 'Reserve Shadow (addr)' in text:
                    reserves[sn]=int(addr[1],16)
                elif 'Store/Cmd Sent to Cache' in text and 'CP2SPM' in text:
                    sent[sn]=(tick,int(addr[1],16))
                elif 'Force Complete' in text and sn in sent and sn in reserves:
                    t,a=sent[sn]
                    if sn not in copies:
                        copies[sn]=dict(sn=sn,src=a,dst=reserves[sn],accepted=t,
                                        evidence_line=line_no)
                        live[a]=copies[sn]
                elif 'Resp Recv' in text and sn in copies:
                    copies[sn]['ack']=tick
                    copies[sn]['woken_loads']=int(re.search(r'Waking (\d+)',text)[1])
                elif 'Exec Load:' in text:
                    loads.setdefault(sn,dict(sn=sn,dst=int(addr[1],16),execute=tick))
                elif 'Load Issue to Cache' in text:
                    loads.setdefault(sn,dict(sn=sn,dst=int(addr[1],16)))['issued']=tick
                elif 'RAW block: load' in text:
                    load,copy=map(int,re.search(r'load SN=(\d+) on SPMCP SN=(\d+)',text).groups())
                    loads.setdefault(load,dict(sn=load)).setdefault('blockers',[]).append(copy)
                elif 'Committing instruction with PC' in text:
                    pc=re.search(r'PC \((0x[0-9a-f]+)',text)
                    if pc and pc[1] in BOUNDARIES:
                        phases.append(dict(tick=tick,phase=BOUNDARIES[pc[1]],
                                           sn=int(re.search(r'\[sn:(\d+)\]',text)[1])))
                continue
            m=PROTO.match(line)
            if m:
                t,core,machine,event,state,addr,detail=m.groups();t=int(t);a=int(addr,16)
                if int(core)!=0 or machine=='L2Cache':continue
                row=dict(tick=t,ctrl=machine,event=event,state=state,addr=a,detail=detail,
                         evidence_line=line_no)
                protocol_samples.append(row)
                if a not in live:continue
                c=live[a]
                if event=='SPMCP_fetch' and machine=='L0Cache':
                    c.setdefault('first_attempt',t)
                    if not detail.strip():c['forward']=t
                elif event=='SPMCP_fetch' and machine=='L1Cache' and not detail.strip():
                    c.setdefault('private_accept',t)
                elif machine=='L1Cache' and state.endswith('>X') and not detail.strip():
                    c['install']=t
                continue
            m=CFT.match(line)
            if m:
                t,obj,event,fields=m.groups();t=int(t)
                if not (obj.startswith('system.ruby.l0_cntrl0') or obj.startswith('system.ruby.l1_cntrl0')):continue
                values=dict(re.findall(r'(\w+)=(\S+)',fields))
                if 'event' in values:values['protocol_event']=values.pop('event')
                row=dict(tick=t,obj=obj,event=event,evidence_line=line_no,**values)
                row['kind']=event
                if event.startswith('POOL_'):pool_events.append(row)
                elif event=='CTRL_RESOURCE_BLOCK':resource_blocks.append(row)
                elif event.startswith('QUEUE_'):queue_events.append(row)
    cs=sorted(copies.values(),key=lambda c:c['sn'])
    panels=[]
    for c in cs:
        if c['dst']==0:panels.append(dict(index=len(panels),copies=[]))
        if panels:panels[-1]['copies'].append(c)
    for i,p in enumerate(panels):
        cc=p['copies'];p['complete']=len(cc)==256 and all('ack' in c for c in cc)
        p['first_sn']=cc[0]['sn'];p['first_accept']=cc[0]['accepted']
        p['last_accept']=cc[-1]['accepted']
        if p['complete']:
            assert {c['dst'] for c in cc}=={(r<<6)|(w<<16) for r in range(128) for w in (0,1)}
            p['last_ack']=max(c['ack'] for c in cc)
            if all('install' in c for c in cc):p['last_install']=max(c['install'] for c in cc)
        next_sn=panels[i+1]['copies'][0]['sn'] if i+1<len(panels) else 2**64
        ll=[v for sn,v in loads.items() if p['first_sn']<sn<next_sn and 'issued' in v]
        p['loads']=ll
        if ll:
            p['first_consumer_issue']=min(v['issued'] for v in ll)
            if p['complete']:p['all_copies_ready_before_first_consumer_cycles']=(p['first_consumer_issue']-p['last_ack'])/400
        p['phases']=[v for v in phases if p['first_sn']<v['sn']<next_sn]
    result=dict(panels=panels,copy_count=len(cs),loads=len(loads),
                raw_blocked_loads=sum(bool(v.get('blockers')) for v in loads.values()),
                phases=phases)
    save(path/'panels.json',result)
    if pool_events:
        save(path/'resource_events.json',dict(pools=pool_events,blocks=resource_blocks,
                                            queues=queue_events,protocol=protocol_samples))
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('pair',type=Path);args=ap.parse_args()
    for w in (0,4):
        r=analyze(args.pair/f'w{w}')
        print('W',w,'copies',r['copy_count'],'loads',r['loads'],'RAW-blocked',r['raw_blocked_loads'])
        for p in r['panels']:
            print({k:v for k,v in p.items() if k not in ('copies','loads','phases')})
